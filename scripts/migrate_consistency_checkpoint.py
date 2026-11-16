"""Create paired baseline/candidate continuations from one safely paused run."""
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


def prepare_migration(parent, target_sources, parent_path, parent_sha, variant, low_frequency_weight=0.5):
    if parent.get("format") != "deflocnet_modern_v1" or variant not in {"baseline", "consistency"}:
        raise ValueError("Unsupported checkpoint or variant")
    old = Config(**parent["config"]).as_dict()
    if old["loss_profile"] != "l1_lpips_tv_gan_v1" or old["boundary_weight"] or old["low_frequency_weight"]:
        raise ValueError("Parent must be the unmodified LPIPS/TV run")
    expected = dict(old)
    if variant == "consistency":
        if low_frequency_weight not in {0.0, 0.5}:
            raise ValueError("Use only the explicit paired or boundary-only ablation")
        expected.update(loss_profile="completion_consistency_v1", boundary_weight=0.1, low_frequency_weight=low_frequency_weight)
    config = Config(**expected)
    changed = sorted(k for k in parent["source"].keys() | target_sources.keys()
                     if parent["source"].get(k) != target_sources.get(k))
    if set(target_sources) != set(parent["source"]) | {"consistency.py"} or set(changed) != {
        "runtime.py", "train.py", "lpips_loss.py", "consistency.py"
    }:
        raise ValueError(f"Unrelated or missing consistency changes: {changed}")
    event = {"kind": "completion_consistency_controlled_continuation", "variant": variant,
             "parent_checkpoint": str(parent_path), "parent_sha256": parent_sha,
             "step": parent["step"], "created_at": time.time(), "changed_modules": changed,
             "parent_source": parent["source"], "target_source": target_sources,
             "parent_config": parent["config"], "target_config": expected}
    migrated = dict(parent)
    migrated.update(config=expected, source=target_sources,
                    lineage=[*(parent.get("lineage") or []), event])
    return migrated, config, event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", required=True, choices=("baseline", "consistency"))
    parser.add_argument("--low-frequency-weight", type=float, choices=(0.0, 0.5), default=0.5)
    args = parser.parse_args()
    parent_run = Path(args.parent_run).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output must be a new run directory")
    with (parent_run / "worker.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((parent_run / "state.json").read_text())
        if state["status"] != "paused":
            raise ValueError("Parent must be safely paused")
        parent_path = (parent_run / "latest.pt").resolve()
        parent_sha = sha256(parent_path)
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        if parent["step"] != state["step"]:
            raise ValueError("Parent checkpoint is not the final paused step")
        actual = {p.name: sha256(p) for p in (parent_run / "source/deflocnet").glob("*.py")}
        if actual != parent["source"]:
            raise ValueError("Parent source snapshot differs from checkpoint")
        payload, config, event = prepare_migration(parent, source_hashes(), parent_path, parent_sha, args.variant,
                                                 args.low_frequency_weight)
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
        event.update(migrated_checkpoint=str(path), migrated_sha256=sha256(path), preserved_fields_verified=preserved)
        atomic_json(output / "consistency-migration.json", event)
        print(json.dumps({"output": str(output), "step": parent["step"], "variant": args.variant,
                          "preserved_fields": preserved}))


if __name__ == "__main__":
    main()
