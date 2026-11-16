"""Continue a paused LPIPS run with TV 0.1, preserving all training state."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import torch

from deflocnet.runtime import Config, atomic_json, sha256, source_hashes
from migrate_hed_checkpoint import assert_identical


def prepare_migration(parent, config, target_sources, parent_path, parent_sha):
    if parent.get("format") != "deflocnet_modern_v1":
        raise ValueError("Unsupported parent checkpoint")
    old = Config(**parent["config"]).as_dict()
    if old["loss_profile"] != "l1_lpips_gan_v1" or old["tv_weight"] != 0:
        raise ValueError("Parent must be the existing LPIPS run without TV")
    expected = dict(old, loss_profile="l1_lpips_tv_gan_v1", tv_weight=0.1)
    if config.as_dict() != expected:
        raise ValueError("TV migration must not change other configuration fields")
    changed = sorted(k for k in parent["source"].keys() | target_sources.keys()
                     if parent["source"].get(k) != target_sources.get(k))
    if parent["source"].keys() != target_sources.keys() or set(changed) != {
        "runtime.py", "train.py", "lpips_loss.py"
    }:
        raise ValueError(f"Unrelated or missing TV source changes: {changed}")
    event = {"kind": "add_paper_hole_tv", "weight": 0.1,
             "parent_checkpoint": str(parent_path), "parent_sha256": parent_sha,
             "step": parent["step"], "created_at": time.time(),
             "changed_modules": changed, "parent_source": parent["source"],
             "target_source": target_sources, "parent_config": old,
             "target_config": expected}
    migrated = dict(parent)
    migrated.update(config=expected, source=target_sources,
                    lineage=[*(parent.get("lineage") or []), event])
    return migrated, event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--output", required=True)
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
        parent_path = (parent_run / "latest.pt").resolve()
        parent_sha = sha256(parent_path)
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        if parent["step"] != state["step"]:
            raise ValueError("Parent checkpoint is not the final paused step")
        frozen = parent_run / "source/deflocnet"
        if {p.name: sha256(p) for p in frozen.glob("*.py")} != parent["source"]:
            raise ValueError("Parent frozen source does not match checkpoint")
        config = Config(**dict(parent["config"], loss_profile="l1_lpips_tv_gan_v1", tv_weight=0.1))
        payload, event = prepare_migration(parent, config, source_hashes(), parent_path, parent_sha)
        output.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[1]
        shutil.copytree(root / "src/deflocnet", output / "source/deflocnet",
                        ignore=shutil.ignore_patterns("__pycache__"))
        atomic_json(output / "config.json", config.as_dict())
        path = output / f"checkpoint-{parent['step']:08d}.pt"
        temp = path.with_suffix(".pt.tmp")
        with temp.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        restored = torch.load(temp, map_location="cpu", weights_only=False)
        preserved = sorted(set(parent) - {"config", "source", "lineage"})
        for key in preserved:
            assert_identical(parent[key], restored[key])
        assert_identical(payload, restored)
        if sha256(parent_path) != parent_sha:
            raise ValueError("Parent checkpoint changed during migration")
        temp.replace(path)
        (output / "latest.pt").symlink_to(path.name)
        event.update(migrated_checkpoint=str(path), migrated_sha256=sha256(path),
                     preserved_fields_verified=preserved)
        atomic_json(output / "tv-migration.json", event)
        print(json.dumps({"output": str(output), "step": parent["step"],
                          "verified_preserved_fields": preserved}))


if __name__ == "__main__":
    main()
