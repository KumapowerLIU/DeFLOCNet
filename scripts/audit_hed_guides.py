"""Fixed-input guide visualization and unchanged-weight inference comparison."""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from deflocnet.edges import LEGACY, THIN, edge_nms, postprocess_edges
from deflocnet.evaluate import evaluate, make_dataset, pil
from deflocnet.hed import HED
from deflocnet.networks import Generator
from deflocnet.runtime import Config, atomic_json, sha256


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config(**ck["config"])
    config.verify_weights()
    hed = HED(config.hed_weights).to(args.device).eval()
    data = make_dataset(config, False)
    rows, records, probabilities = [], [], []
    for index in [0, 133, 533, 1599]:
        item = data[index]
        rgb = item["image"][None].to(args.device)
        probability = hed(rgb)
        old = postprocess_edges(probability, LEGACY)[0, 0].cpu().numpy()
        new = postprocess_edges(probability, THIN)[0, 0].cpu().numpy()
        p = probability[0, 0].cpu().numpy()
        nms = edge_nms(p)
        panels = [pil(rgb[0])] + [Image.fromarray(np.rint(a * 255).astype(np.uint8)).convert("RGB") for a in (p, old, nms, new)]
        row = Image.new("RGB", (1280, 280), "white")
        draw = ImageDraw.Draw(row)
        for i, (title, panel) in enumerate(zip(("RGB", "HED probability", "Old > 0.5", "Directional NMS", "New thin guide"), panels)):
            row.paste(panel, (256 * i, 24))
            draw.text((256 * i + 4, 6), f"{index}: {title}", fill="black")
        row.save(output / f"guides-{index:06d}.png")
        rows.append(row)
        hole = (~item["valid"])[0].numpy()
        records.append({"index": index, "old_pixels": int(old.sum()), "new_pixels": int(new.sum()),
                        "old_hole_pixels": int(old[hole].sum()), "new_hole_pixels": int(new[hole].sum())})
        probabilities.append(probability)
    grid = Image.new("RGB", (1280, 280 * len(rows)), "white")
    for i, row in enumerate(rows):
        grid.paste(row, (0, i * 280))
    grid.save(output / "guide-comparison.png")
    batch = torch.cat(probabilities * 4)
    postprocess_edges(batch, THIN)
    started = time.perf_counter()
    for _ in range(3):
        postprocess_edges(batch, THIN)
    if batch.is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / 3
    report = {"step": ck["step"], "parent_sha256": sha256(args.checkpoint), "records": records,
              "postprocess_batch16_seconds": elapsed,
              "note": "Identical full-image HED probabilities; only postprocessing differs. Hole masking happens afterwards."}
    atomic_json(output / "guide-audit.json", report)
    model = Generator(config.width).to(args.device).eval()
    model.load_state_dict(ck["generator"], strict=True)
    del ck
    evaluate(model, hed, data, output / "old-guides", report["step"], hed_postprocess=LEGACY)
    evaluate(model, hed, data, output / "thin-guides", report["step"], hed_postprocess=THIN)
    print(report)


if __name__ == "__main__":
    main()
