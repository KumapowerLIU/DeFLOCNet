"""Audited, one-way guide migration; preserve full optimizer/RNG/data state.

Ordinary training resume remains strict. This command creates a NEW run from a
paused run, and records the changed preprocessing and both source maps.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from deflocnet.runtime import Config, atomic_json, sha256, source_hashes


def prepare_migration(parent, config, target_sources, parent_path, parent_sha):
    if parent.get("format") != "deflocnet_modern_v1":
        raise ValueError("Unsupported parent checkpoint")
    old_config = Config(**parent["config"]).as_dict()
    new_config = config.as_dict()
    transition = (old_config["hed_postprocess"], new_config["hed_postprocess"])
    allowed_changes = {
        ("threshold_v1", "hed_nms_thin_v1"): {"edges.py", "runtime.py", "train.py", "evaluate.py"},
        ("hed_nms_thin_v1", "hed_nms_medium_v1"): {"edges.py", "runtime.py"},
    }
    if transition not in allowed_changes:
        raise ValueError(f"Unsupported HED protocol migration: {transition}")
    expected = dict(old_config, hed_postprocess=new_config["hed_postprocess"])
    if new_config != expected:
        raise ValueError("HED migration must not change other configuration fields")
    changed = sorted(k for k in parent["source"].keys() | target_sources.keys()
                     if parent["source"].get(k) != target_sources.get(k))
    allowed = allowed_changes[transition]
    if set(changed) - allowed or parent["source"].keys() - target_sources.keys():
        raise ValueError(f"Unrelated source changes during HED migration: {changed}")
    event = {"kind": "hed_guide_postprocessing", "parent_checkpoint": str(parent_path),
             "parent_sha256": parent_sha, "step": parent["step"], "created_at": time.time(),
             "from": transition[0], "to": transition[1], "changed_modules": changed,
             "parent_source": parent["source"], "target_source": target_sources,
             "parent_config": parent["config"], "target_config": new_config}
    migrated = dict(parent)
    migrated.update(config=new_config, source=target_sources,
                    lineage=[*parent.get("lineage", []), event])
    return migrated, event


def assert_identical(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_identical(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_identical(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hed-postprocess", choices=("hed_nms_thin_v1", "hed_nms_medium_v1"),
                        default="hed_nms_thin_v1")
    args = parser.parse_args()
    parent_run = Path(args.parent_run).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Migration output must be a new run directory")
    with (parent_run / "worker.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((parent_run / "state.json").read_text())
        if state["status"] != "paused":
            raise ValueError("Parent run must be safely paused")
        checkpoint_path = (parent_run / "latest.pt").resolve()
        parent_sha = sha256(checkpoint_path)
        parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if parent["step"] != state["step"]:
            raise ValueError("Parent checkpoint is not the final paused step")
        frozen = parent_run / "source/deflocnet"
        actual_source = {p.name: sha256(p) for p in frozen.glob("*.py")}
        if actual_source != parent["source"]:
            raise ValueError("Parent source snapshot does not match checkpoint")
        target_config = Config(**dict(parent["config"], hed_postprocess=args.hed_postprocess))
        payload, event = prepare_migration(parent, target_config, source_hashes(), checkpoint_path, parent_sha)
        output.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[1]
        shutil.copytree(root / "src/deflocnet", output / "source/deflocnet",
                        ignore=shutil.ignore_patterns("__pycache__"))
        atomic_json(output / "config.json", target_config.as_dict())
        path = output / f"checkpoint-{parent['step']:08d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        with tmp.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        reloaded = torch.load(tmp, map_location="cpu", weights_only=False)
        preserved = sorted(set(parent) - {"config", "source", "lineage"})
        for key in preserved:
            assert_identical(parent[key], reloaded[key])
        assert_identical(payload, reloaded)
        if sha256(checkpoint_path) != parent_sha:
            raise ValueError("Parent changed during migration")
        tmp.replace(path)
        (output / "latest.pt").symlink_to(path.name)
        event.update(migrated_checkpoint=str(path), migrated_sha256=sha256(path),
                     preserved_fields_verified=preserved)
        atomic_json(output / "hed-migration.json", event)
        print(json.dumps({"output": str(output), "step": parent["step"],
                          "parent_sha256": parent_sha, "verified_preserved_fields": preserved}))


if __name__ == "__main__":
    main()
