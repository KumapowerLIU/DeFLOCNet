"""Single-GPU, accumulated-batch training with independent D/G updates."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EpochMicrobatches
from .edges import hed_sketch
from .evaluate import evaluate, make_dataset
from .hed import HED
from .losses import ReconstructionLoss, hinge_discriminator, hinge_generator
from .networks import build_generator, predict_generator, Discriminators
from .runtime import Config, atomic_json, capture_rng, restore_rng, save_checkpoint, source_hashes


def worker_init(_):
    torch.set_num_threads(1)


def finite_gradients(module):
    return float(torch.nn.utils.clip_grad_norm_(module.parameters(), float("inf"), error_if_nonfinite=True))


def split_predictions(predictions, batch):
    fake = [[x[:batch] for x in scale] for scale in predictions]
    real = [[x[batch:] for x in scale] for scale in predictions]
    return fake, real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int, help="Pause after this committed global step (preflight).")
    args = parser.parse_args()
    config = Config.load(args.config)
    lpips_profile = config.loss_profile in {"l1_lpips_gan_v1", "l1_lpips_tv_gan_v1", "completion_consistency_v1"}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = output / "config.json"
    if config_path.exists() and Config.load(config_path) != config:
        raise ValueError("Run configuration differs from requested configuration")
    if (output / "latest.pt").exists() and not args.resume:
        raise ValueError("Run already has a checkpoint; use --resume")
    atomic_json(config_path, config.as_dict())
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Production training requires CUDA; unit tests cover CPU")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(config.memory_budget_gib * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    config.verify_weights()
    started = time.time()
    step, epoch, cursor = 0, 0, 0
    lineage = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def status(state, **extra):
        atomic_json(output / "state.json", {"status": state, "pid": os.getpid(), "step": step,
                    "epoch": epoch, "cursor": cursor, "updated_at": time.time(),
                    "gpu": torch.cuda.get_device_name(device), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "batch_size": config.batch_size, "microbatch_size": config.microbatch_size,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3, **extra})

    @contextmanager
    def amp():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.amp == "bf16"):
            yield

    try:
        status("initializing")
        train_data, val_data = make_dataset(config, True), make_dataset(config, False)
        atomic_json(output / "data-audit.json", {"train": train_data.audit(), "validation": val_data.audit()})
        generator = build_generator(config).to(device).train()
        discriminator = Discriminators(config.discriminator_width).to(device).train()
        if lpips_profile:
            from .lpips_loss import ThreeLossReconstruction
            reconstruction = ThreeLossReconstruction(config).to(device).eval()
            atomic_json(output / "lpips-provenance.json", reconstruction.lpips.provenance)
        else:
            reconstruction = ReconstructionLoss(config.vgg_weights).to(device).eval()
        hed = HED(config.hed_weights).to(device).eval()
        optim_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), foreach=False)
        optim_d = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), foreach=False)
        if args.resume:
            ck = torch.load(args.resume, map_location="cpu", weights_only=False)
            if ck["format"] != "deflocnet_modern_v1" or ck["config"] != config.as_dict() or ck["source"] != source_hashes():
                raise ValueError("Resume checkpoint config/source mismatch")
            generator.load_state_dict(ck["generator"], strict=True)
            discriminator.load_state_dict(ck["discriminator"], strict=True)
            optim_g.load_state_dict(ck["optimizer_g"])
            optim_d.load_state_dict(ck["optimizer_d"])
            step, epoch, cursor = ck["step"], ck["epoch"], ck["cursor"]
            lineage = ck.get("lineage")
            restore_rng(ck["rng"])
            del ck
        atomic_json(output / "source-hashes.json", source_hashes())
        accumulation = config.batch_size // config.microbatch_size
        terms = ("l1", "lpips", "gan") if lpips_profile else ("l1", "perceptual", "style", "tv", "gan")
        if config.loss_profile in {"l1_lpips_tv_gan_v1", "completion_consistency_v1"}:
            terms += ("tv",)
        if config.loss_profile == "completion_consistency_v1":
            terms += ("boundary", "low_frequency")
        weights = {name: getattr(config, name + "_weight") for name in terms}
        status("running", parameters_g=sum(p.numel() for p in generator.parameters()))
        print(json.dumps({"event": "ready", "pid": os.getpid(), "step": step, "source": source_hashes()}), flush=True)
        while step < config.max_steps and not stop_requested:
            sampler = EpochMicrobatches(len(train_data), config.batch_size, config.microbatch_size, epoch, config.seed, cursor)
            options = dict(num_workers=config.workers, pin_memory=True, worker_init_fn=worker_init,
                           generator=torch.Generator().manual_seed(config.seed + epoch))
            if config.workers:
                options.update(prefetch_factor=2)
            loader = DataLoader(train_data, batch_sampler=sampler, **options)
            iterator = iter(loader)
            while cursor < sampler.length and step < config.max_steps and not stop_requested:
                tick = time.time()
                batches, losses = [], {}
                # Prepare bounded microbatches, with matched controls/noise for D and G.
                for _ in range(accumulation):
                    batch = next(iterator)
                    rgb = batch["image"].to(device, non_blocking=True)
                    enabled = batch["sketch_enabled"].to(device)[:, None, None, None]
                    with torch.no_grad():
                        sketch = hed_sketch(hed, rgb, config.hed_postprocess) * enabled
                    batch["sketch"] = sketch.cpu()
                    batch["noise"] = torch.rand_like(rgb).cpu()
                    batch["crop"] = (random.randrange(config.image_size - 63), random.randrange(config.image_size - 63))
                    batches.append(batch)
                discriminator.requires_grad_(True).train()
                optim_d.zero_grad(set_to_none=True)
                for batch in batches:
                    image = batch["image"].to(device) * 2 - 1
                    hole = ~batch["valid"].to(device)
                    sketch, color, noise = (batch[k].to(device) for k in ("sketch", "color", "noise"))
                    with torch.no_grad(), amp():
                        fake = predict_generator(generator, image, hole, sketch, color, noise, batch["color_mask"].to(device))
                        if lpips_profile:
                            fake = torch.where(hole, fake, image)
                    with amp():
                        predictions = discriminator(torch.cat((fake, image)), torch.cat((hole, hole)), batch["crop"])
                        pf, pr = split_predictions(predictions, len(image))
                        d_loss = hinge_discriminator(pr, pf)
                    (d_loss / accumulation).backward()
                    losses["d"] = losses.get("d", 0.0) + float(d_loss.detach()) / accumulation
                losses["gradient_d"] = finite_gradients(discriminator)
                optim_d.step()
                optim_d.zero_grad(set_to_none=True)
                discriminator.requires_grad_(False).eval()  # freeze spectral buffers during G update
                optim_g.zero_grad(set_to_none=True)
                for batch in batches:
                    image = batch["image"].to(device) * 2 - 1
                    hole = ~batch["valid"].to(device)
                    sketch, color, noise = (batch[k].to(device) for k in ("sketch", "color", "noise"))
                    with amp():
                        fake = predict_generator(generator, image, hole, sketch, color, noise, batch["color_mask"].to(device))
                        adversarial_image = torch.where(hole, fake, image) if lpips_profile else fake
                        parts = {"gan": hinge_generator(discriminator(adversarial_image, hole, batch["crop"]))}
                    parts.update(reconstruction(fake, image, hole))
                    loss = sum(weights[k] * value for k, value in parts.items())
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite generator loss")
                    (loss / accumulation).backward()
                    for key, value in parts.items():
                        losses[key] = losses.get(key, 0.0) + float(value.detach()) / accumulation
                losses["gradient_g"] = finite_gradients(generator)
                optim_g.step()
                optim_g.zero_grad(set_to_none=True)
                step += 1
                cursor += config.batch_size
                record = {"step": step, "epoch": epoch, "cursor": cursor, "time": time.time(),
                          "seconds": time.time() - tick, "losses": losses,
                          "weighted": {k: losses[k] * weights[k] for k in weights}}
                noise_digest = hashlib.sha256()
                for batch in batches:
                    noise_digest.update(batch["noise"].contiguous().numpy().tobytes())
                record.update(indices=[int(i) for batch in batches for i in batch["index"]],
                              control_modes=[int(i) for batch in batches for i in batch["control_mode"]],
                              noise_sha256=noise_digest.hexdigest())
                with (output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record, allow_nan=False), flush=True)
                status("running", seconds_per_step=record["seconds"], losses=losses)
                stopping = stop_requested or (args.stop_after is not None and step >= args.stop_after) or step >= config.max_steps
                if step % config.save_every == 0 or stopping:
                    save_checkpoint(output, step, generator, discriminator, optim_g, optim_d, config, epoch, cursor, lineage)
                if step % config.evaluate_every == 0 or stopping:
                    rng = capture_rng()
                    evaluate(generator, hed, val_data, output / "evaluations" / f"step-{step:08d}", step,
                             hed_postprocess=config.hed_postprocess)
                    restore_rng(rng)
                if stopping:
                    stop_requested = True
            del iterator, loader
            if cursor >= sampler.length:
                epoch, cursor = epoch + 1, 0
        # Handles a signal received between steps, or before the first update.
        if not (output / f"checkpoint-{step:08d}.pt").exists():
            save_checkpoint(output, step, generator, discriminator, optim_g, optim_d, config, epoch, cursor, lineage)
        status("complete" if step >= config.max_steps else "paused", elapsed=time.time() - started)
    except Exception as exc:
        status("failed", error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
