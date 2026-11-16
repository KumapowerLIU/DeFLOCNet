"""Run on the GPU server to launch a detached, source-pinned training worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", required=True, help="Physical GPU UUID")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if not args.gpu.startswith("GPU-"):
        parser.error("Use an explicit GPU UUID")
    root = Path(__file__).resolve().parents[1]
    run = Path(args.output).resolve()
    run.mkdir(parents=True, exist_ok=True)
    source = root / "src/deflocnet"
    frozen = run / "source/deflocnet"
    if frozen.exists():
        if {p.name: digest(p) for p in frozen.glob("*.py")} != {p.name: digest(p) for p in source.glob("*.py")}:
            raise ValueError("Source changed: preserve the old run and create a new one")
    else:
        shutil.copytree(source, frozen, ignore=shutil.ignore_patterns("__pycache__"))
    sys.path.insert(0, str(root / "src"))
    from deflocnet.runtime import Config
    config = Config.load(args.config).as_dict()
    target = run / "config.json"
    if target.exists():
        if json.loads(target.read_text()) != config:
            raise ValueError("Run configuration changed")
    else:
        target.write_text(json.dumps(config, indent=2) + "\n")
    command = [sys.executable, "-m", "deflocnet.train", "--config", str(target), "--output", str(run)]
    if args.resume:
        command += ["--resume", str(run / "latest.pt")]
    if args.stop_after is not None:
        command += ["--stop-after", str(args.stop_after)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(run / "source"), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (run / "console.log").open("ab", buffering=0) as log:
        worker = subprocess.Popen(command, cwd=run, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=log, start_new_session=True, close_fds=True)
    launch = {"pid": worker.pid, "command": command, "gpu_uuid": args.gpu, "output": str(run)}
    (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch))


if __name__ == "__main__":
    main()
