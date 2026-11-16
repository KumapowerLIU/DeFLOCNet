"""Paper TV reduction, output gradients, and narrowly scoped continuation."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from deflocnet import lpips_loss
from deflocnet.runtime import Config

ROOT = Path(__file__).resolve().parents[1]


def config(tv=False):
    values = json.loads((ROOT / "configs/sd-color-v2-l1-lpips-gan.json").read_text())
    if tv:
        values.update(loss_profile="l1_lpips_tv_gan_v1", tv_weight=0.1)
    return Config(**values)


class FakeLPIPS(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, a, b):
        return (a - b).square().mean()


def test_tv_reconstruction_matches_paper_pairs_and_gradients(monkeypatch):
    monkeypatch.setattr(lpips_loss, "FrozenLPIPS", FakeLPIPS)
    torch.manual_seed(3)
    fake = torch.randn(2, 3, 5, 6, requires_grad=True)
    target = torch.randn_like(fake)
    hole = torch.zeros(2, 1, 5, 6, dtype=torch.bool)
    hole[0, :, 1:4, 1:5] = True
    hole[1, :, 2, 2] = True  # Isolated hole pixel contributes no adjacent pairs.
    parts = lpips_loss.ThreeLossReconstruction(config(True))(fake, target, hole)
    old = lpips_loss.ThreeLossReconstruction(config())(fake, target, hole)
    assert set(old) == {"l1", "lpips"}
    assert set(parts) == {"l1", "lpips", "tv"}
    for key in old:
        torch.testing.assert_close(parts[key], old[key], rtol=0, atol=0)
    reference = fake.sum() * 0
    for b in range(2):
        for y in range(5):
            for x in range(6):
                for yy, xx in [(y + 1, x), (y, x + 1)]:
                    if yy < 5 and xx < 6 and hole[b, 0, y, x] and hole[b, 0, yy, xx]:
                        reference = reference + (fake[b, :, y, x] - fake[b, :, yy, xx]).abs().sum()
    reference = reference / fake.numel()
    torch.testing.assert_close(parts["tv"], reference)
    actual_grad = torch.autograd.grad(parts["tv"], fake, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference, fake)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[~hole.expand_as(fake)].count_nonzero() == 0
    assert actual_grad[0].count_nonzero() > 0
    assert actual_grad[1].count_nonzero() == 0
    assert lpips_loss.ThreeLossReconstruction(config(True))(fake, target, torch.zeros_like(hole))["tv"] == 0


def test_tv_profile_cannot_silently_change_legacy_objective():
    with pytest.raises(ValueError, match="disable"):
        Config(**dict(config().as_dict(), tv_weight=0.1))
    with pytest.raises(ValueError, match="positive TV"):
        Config(**dict(config(True).as_dict(), tv_weight=0))


def test_tv_migration_preserves_full_training_state_and_rejects_other_changes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("tv_migration", ROOT / "scripts/migrate_tv_checkpoint.py")
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    sources = {"runtime.py": "old", "train.py": "old", "lpips_loss.py": "old", "networks.py": "same"}
    new_sources = dict(sources, **{k: "new" for k in ["runtime.py", "train.py", "lpips_loss.py"]})
    parent = {"format": "deflocnet_modern_v1", "config": config().as_dict(), "source": sources,
              "step": 6469, "epoch": 1, "cursor": 128, "rng": {"torch": torch.get_rng_state()},
              "generator": {"w": torch.randn(3)}, "optimizer_g": {"state": {1: {"exp_avg": torch.randn(3)}}},
              "discriminator": {"w": torch.randn(3)}, "optimizer_d": {"state": {}}, "lineage": None}
    before = copy.deepcopy(parent)
    migrated, event = migration.prepare_migration(parent, config(True), new_sources, "old.pt", "sha")
    for key in parent.keys() - {"config", "source", "lineage"}:
        migration.assert_identical(parent[key], migrated[key])
    migration.assert_identical(parent, before)
    assert event["weight"] == 0.1 and len(migrated["lineage"]) == 1
    with pytest.raises(ValueError, match="other configuration"):
        migration.prepare_migration(parent, Config(**dict(config(True).as_dict(), batch_size=256)), new_sources, "old.pt", "sha")
    with pytest.raises(ValueError, match="source changes"):
        migration.prepare_migration(parent, config(True), dict(new_sources, **{"networks.py": "changed"}), "old.pt", "sha")
