import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from deflocnet.consistency import CompletionConsistency, boundary_gradient_loss
from deflocnet.runtime import Config

ROOT = Path(__file__).resolve().parents[1]


def test_real_edges_are_targets_and_do_not_get_smoothed():
    gt = torch.zeros(1, 3, 12, 12)
    gt[..., 6:] = 1
    hole = torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    hole[..., 4:9, 4:9] = True
    parts = CompletionConsistency()(gt.clone().requires_grad_(), gt, hole)
    assert all(x.item() == 0 for x in parts.values())
    # A flat prediction crossing this real edge must be penalized, not rewarded.
    flat = torch.zeros_like(gt)
    assert CompletionConsistency()(flat, gt, hole)["boundary"] > 0


def test_consistency_gradients_only_touch_generated_hole_and_target_is_detached():
    torch.manual_seed(3)
    fake = torch.randn(2, 3, 12, 12, requires_grad=True)
    target = torch.randn_like(fake, requires_grad=True)
    hole = torch.zeros(2, 1, 12, 12, dtype=torch.bool)
    hole[0, :, :8, 3:9] = True
    criterion = CompletionConsistency()
    parts = criterion(fake, target, hole)
    for value in parts.values():
        gradient = torch.autograd.grad(value, fake, retain_graph=True)[0]
        assert gradient[~hole.expand_as(fake)].count_nonzero() == 0
        assert gradient[hole.expand_as(fake)].count_nonzero() > 0
        assert torch.isfinite(gradient).all()
    sum(parts.values()).backward()
    assert target.grad is None
    poison = torch.where(hole, fake.detach(), torch.full_like(fake, 1e4))
    for k, v in criterion(poison, target, hole).items():
        torch.testing.assert_close(v, parts[k], rtol=0, atol=0)
    for v in criterion(fake, target, torch.zeros_like(hole)).values():
        assert v == 0


def test_low_frequency_detects_color_offset_but_not_fine_zero_mean_texture():
    criterion = CompletionConsistency()
    gt = torch.zeros(2, 3, 32, 32)
    hole = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    fake = gt.clone()
    fake[0] += 0.2
    fake[1] -= 0.4
    parts = criterion(fake, gt, hole)
    torch.testing.assert_close(parts["low_frequency"], torch.tensor(0.3))
    assert parts["boundary"] == 0  # No known/hole transition in a full-hole image.
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    checker = ((xx + yy) % 2 * 2 - 1).float().expand_as(gt) * 0.3
    assert criterion(checker, gt, hole)["low_frequency"] < 0.02


def test_boundary_matches_explicit_pixel_pairs():
    torch.manual_seed(4)
    x = torch.randn(1, 3, 5, 6, requires_grad=True)
    gt = torch.randn_like(x)
    hole = torch.zeros(1, 1, 5, 6, dtype=torch.bool)
    hole[..., :3, 2:5] = True
    completed = torch.where(hole, x, gt)
    errors = []
    def band(y, z):
        neighbors = hole[0, 0, max(0, y-1):min(5, y+2), max(0, z-1):min(6, z+2)]
        return bool(neighbors.any() and (~neighbors).any())
    for y in range(5):
        for z in range(6):
            for yy, zz in [(y+1, z), (y, z+1)]:
                if yy >= 5 or zz >= 6:
                    continue
                if (hole[0, 0, y, z] or hole[0, 0, yy, zz]) and (band(y, z) or band(yy, zz)):
                    residual = completed - gt
                    errors.append((residual[0, :, yy, zz] - residual[0, :, y, z]).abs())
    expected = torch.stack(errors).mean()
    actual = boundary_gradient_loss(completed, gt, hole, radius=1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0], torch.autograd.grad(expected, x)[0])


def test_profile_and_migration_keep_architecture_optimizers_rng_and_cursor(monkeypatch):
    config = Config(**json.loads((ROOT / "configs/sd-color-v2-l1-lpips-tv-gan.json").read_text()))
    with pytest.raises(ValueError, match="explicit"):
        Config(**dict(config.as_dict(), boundary_weight=0.1))
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("consistency_migration", ROOT / "scripts/migrate_consistency_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = {k: "old" for k in ["runtime.py", "lpips_loss.py", "train.py", "sd_networks.py"]}
    target = dict(source, **{k: "new" for k in ["runtime.py", "lpips_loss.py", "train.py", "consistency.py"]})
    parent = {"format": "deflocnet_modern_v1", "config": config.as_dict(), "source": source,
              "step": 123, "cursor": 128, "epoch": 1, "generator": {"p": torch.randn(2)},
              "discriminator": {"p": torch.randn(2)}, "optimizer_g": {"moment": torch.randn(2)},
              "optimizer_d": {"moment": torch.randn(2)}, "rng": torch.get_rng_state(), "lineage": None}
    before = copy.deepcopy(parent)
    for variant in ["baseline", "consistency"]:
        migrated, new_config, _ = module.prepare_migration(parent, target, "parent.pt", "sha", variant)
        for k in parent.keys() - {"source", "config", "lineage"}:
            module.assert_identical(parent[k], migrated[k])
        module.assert_identical(parent, before)
        expected = dict(config.as_dict())
        if variant == "consistency":
            expected.update(loss_profile="completion_consistency_v1", boundary_weight=0.1, low_frequency_weight=0.5)
        assert new_config.as_dict() == expected
    with pytest.raises(ValueError, match="Unrelated"):
        module.prepare_migration(parent, dict(target, **{"sd_networks.py": "modified"}), "parent.pt", "sha", "consistency")
    _, isolated, _ = module.prepare_migration(parent, target, "parent.pt", "sha", "consistency", 0.0)
    assert isolated.low_frequency_weight == 0 and isolated.boundary_weight == 0.1


def test_boundary_only_profile_does_not_add_low_frequency_gradients(monkeypatch):
    from deflocnet import lpips_loss
    from test_tv_continuation import FakeLPIPS
    monkeypatch.setattr(lpips_loss, "FrozenLPIPS", FakeLPIPS)
    base = json.loads((ROOT / "configs/sd-color-v2-l1-lpips-tv-gan.json").read_text())
    config = Config(**dict(base, loss_profile="completion_consistency_v1", boundary_weight=0.1,
                           low_frequency_weight=0))
    torch.manual_seed(2)
    x = torch.randn(1, 3, 16, 16, requires_grad=True)
    target = torch.zeros_like(x)
    hole = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    hole[..., 4:12, 4:12] = True
    criterion = lpips_loss.ThreeLossReconstruction(config)
    parts = criterion(x, target, hole)
    assert parts['low_frequency'] == 0
    expected = boundary_gradient_loss(torch.where(hole, x, target), target, hole)
    torch.testing.assert_close(parts['boundary'], expected)
    assert torch.count_nonzero(torch.autograd.grad(parts['low_frequency'], x)[0]) == 0
