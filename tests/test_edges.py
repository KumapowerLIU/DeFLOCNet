from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import label
import torch

from deflocnet.edges import LEGACY, THIN, MEDIUM, edge_nms, hed_sketch, postprocess_edges
from deflocnet.runtime import Config


def ridge(kind):
    yy, xx = np.mgrid[:65, :65]
    distance = {"vertical": xx - 32, "horizontal": yy - 32,
                "diagonal": (xx - yy) / np.sqrt(2)}[kind]
    return np.exp(-distance ** 2 / (2 * 3.0 ** 2)).astype(np.float32)


@pytest.mark.parametrize("kind", ["vertical", "horizontal", "diagonal"])
def test_nms_preserves_ridge_location_and_thins(kind):
    p = ridge(kind)
    lines = postprocess_edges(torch.from_numpy(p)[None, None], THIN)[0, 0].numpy()
    assert lines.sum() < (p > .5).sum() / 3
    assert label(lines, structure=np.ones((3, 3)))[1] == 1
    yy, xx = np.nonzero(lines)
    if kind == "vertical":
        assert np.max(np.abs(xx - 32)) <= 1 and len(set(yy)) >= 63
    elif kind == "horizontal":
        assert np.max(np.abs(yy - 32)) <= 1 and len(set(xx)) >= 63
    else:
        assert np.max(np.abs(xx - yy)) <= 1 and len(set(xx)) >= 60


def test_flat_maps_do_not_become_spurious_lines():
    for value in (0., .4, 1.):
        assert not postprocess_edges(torch.full((1, 1, 32, 32), value), THIN).any()


def test_nms_suppresses_shoulders_and_retains_peak():
    p = ridge("vertical")
    nms = edge_nms(p)
    assert nms[32, 32] == 1
    assert nms[32, 29] == 0 and nms[32, 35] == 0
    assert np.all(nms <= p)


def test_shared_guide_is_binary_deterministic_and_batch_independent():
    probabilities = torch.from_numpy(np.stack([ridge("vertical"), ridge("diagonal")]))[:, None]
    original = probabilities.clone()
    state = torch.get_rng_state()
    actual = postprocess_edges(probabilities, THIN)
    torch.testing.assert_close(actual, torch.cat([postprocess_edges(p[None], THIN) for p in probabilities]), atol=0, rtol=0)
    assert torch.equal(state, torch.get_rng_state()) and torch.equal(original, probabilities)
    assert actual.dtype == torch.float32 and set(actual.unique().tolist()) <= {0, 1}
    rgb = torch.rand(2, 3, 65, 65)
    assert torch.equal(hed_sketch(lambda _: probabilities, rgb, THIN), actual.expand_as(rgb))
    assert torch.equal(postprocess_edges(probabilities, LEGACY), (probabilities > .5).float())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities_rejected(value):
    p = torch.zeros(1, 1, 8, 8)
    p[0, 0, 4, 4] = value
    with pytest.raises(ValueError, match="finite"):
        postprocess_edges(p, THIN)


def config(**kwargs):
    paths = {k: "x" for k in ("train_manifest", "validation_manifest", "train_sha256", "validation_sha256", "vgg_weights", "hed_weights", "vgg_sha256", "hed_sha256", "rtv_cache")}
    return Config(**paths, **kwargs)


def migration_module():
    spec = importlib.util.spec_from_file_location("hed_migration", Path(__file__).resolve().parents[1] / "scripts/migrate_hed_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_checkpoint_defaults_and_migration_preserve_training_state():
    legacy = config().as_dict()
    legacy.pop("hed_postprocess")
    assert Config(**legacy).hed_postprocess == LEGACY
    parent = {"format": "deflocnet_modern_v1", "config": legacy, "step": 5208,
              "source": {"train.py": "old", "networks.py": "same", "losses.py": "same"},
              "generator": {"x": torch.randn(5)}, "optimizer_g": {"state": torch.randn(5)},
              "rng": {"numpy": np.arange(5)}, "cursor": 333312, "epoch": 0}
    snapshot = copy.deepcopy(parent)
    sources = dict(parent["source"], **{"train.py": "new", "edges.py": "added"})
    module = migration_module()
    migrated, event = module.prepare_migration(parent, config(hed_postprocess=THIN), sources, "/parent.pt", "parent-hash")
    module.assert_identical(parent, snapshot)
    for key in ("generator", "optimizer_g", "rng", "step", "epoch", "cursor"):
        module.assert_identical(parent[key], migrated[key])
    assert migrated["lineage"][-1]["parent_sha256"] == "parent-hash"
    with pytest.raises(ValueError, match="other configuration"):
        module.prepare_migration(parent, config(hed_postprocess=THIN, gan_weight=.2), sources, "x", "hash")
    with pytest.raises(ValueError, match="Unrelated source"):
        module.prepare_migration(parent, config(hed_postprocess=THIN), dict(sources, **{"networks.py": "changed"}), "x", "hash")


@pytest.mark.parametrize("kind", ["vertical", "horizontal", "diagonal"])
def test_medium_width_is_centered_and_within_one_pixel_of_thin(kind):
    p = torch.from_numpy(ridge(kind))[None, None]
    thin = postprocess_edges(p, THIN)[0, 0].bool()
    medium = postprocess_edges(p, MEDIUM)[0, 0].bool()
    yy, xx = torch.where(thin)
    expected = thin.clone()
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        y, x = yy + dy, xx + dx
        inside = (y >= 0) & (y < 65) & (x >= 0) & (x < 65)
        expected[y[inside], x[inside]] = True
    assert torch.equal(medium, expected)
    if kind == "vertical":
        assert torch.where(medium[32])[0].tolist() == [31, 32, 33]
    elif kind == "horizontal":
        assert torch.where(medium[:, 32])[0].tolist() == [31, 32, 33]


def test_medium_has_no_cross_batch_growth_and_preserves_rng_and_legacy():
    p = torch.stack([torch.zeros(65, 65), torch.from_numpy(ridge("vertical"))])[:, None]
    state = torch.get_rng_state()
    original = p.clone()
    result = postprocess_edges(p, MEDIUM)
    assert not result[0].any() and result[1].sum() > 65
    assert torch.equal(result[1:2], postprocess_edges(p[1:2], MEDIUM))
    assert torch.equal(state, torch.get_rng_state()) and torch.equal(p, original)
    assert result.dtype == torch.float32 and set(result.unique().tolist()) == {0, 1}
    assert torch.equal(postprocess_edges(p, LEGACY), (p > .5).float())
    assert torch.where(postprocess_edges(p[1:2], THIN)[0, 0, 32])[0].tolist() == [32]


def test_medium_migration_preserves_prior_lineage_and_rejects_other_edits():
    module = migration_module()
    parent = {"format": "deflocnet_modern_v1", "config": config(hed_postprocess=THIN).as_dict(),
              "step": 18000, "epoch": 0, "cursor": 1152000,
              "source": {"edges.py": "old", "runtime.py": "old", "train.py": "unchanged"},
              "lineage": [{"kind": "prior_migration", "parent_sha256": "original"}],
              "generator": {"x": torch.randn(5)}, "optimizer_g": {"state": torch.randn(5)},
              "rng": {"numpy": np.arange(5)}}
    original = copy.deepcopy(parent)
    sources = dict(parent["source"], **{"edges.py": "new", "runtime.py": "new"})
    migrated, event = module.prepare_migration(parent, config(hed_postprocess=MEDIUM), sources, "/parent.pt", "thin-hash")
    module.assert_identical(parent, original)
    assert migrated["lineage"][:-1] == parent["lineage"]
    assert (event["from"], event["to"]) == (THIN, MEDIUM)
    for key in ("generator", "optimizer_g", "rng", "step", "epoch", "cursor"):
        module.assert_identical(parent[key], migrated[key])
    with pytest.raises(ValueError, match="other configuration"):
        module.prepare_migration(parent, config(hed_postprocess=MEDIUM, batch_size=128), sources, "x", "h")
    with pytest.raises(ValueError, match="Unrelated source"):
        module.prepare_migration(parent, config(hed_postprocess=MEDIUM), dict(sources, **{"train.py": "changed"}), "x", "h")
    with pytest.raises(ValueError, match="Unsupported"):
        module.prepare_migration(parent, config(hed_postprocess=LEGACY), sources, "x", "h")
