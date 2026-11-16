"""Gradio workspace for comparing two DeFLOCNet generators with shared controls."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
from secrets import token_hex
import time

import gradio as gr
import numpy as np
from PIL import Image, ImageOps


SIZE = (256, 256)
FACE = "HDTF portrait"
GENERAL = "General scenes"
MASK_FILL = (255, 255, 255)
PNG_URL = "data:image/png;base64,"
ALPHA_THRESHOLD = 32
FACE_ROWS = (454, 151, 302)
HELD_OUT = "Fixed HDTF sample"
PAINTED = "Painted edit"
BOTH = "Compare both"
AUTO_CROP = "Auto-frame portrait"
FULL_IMAGE = "Use full image"
START = "Upload an image or choose a sample below."

CSS = """
.gradio-container { max-width: 1390px !important; margin-inline: auto !important;
  padding: 24px clamp(12px, 3vw, 42px) 32px !important; }
#defloc-hero { padding: 24px 28px 22px; border-radius: 22px; background: #231937;
  color: #fbf6ff; border: 1px solid #47355e; margin-bottom: 10px; }
#defloc-hero .eyebrow { color: #ddc0ff; font-size: 11px; font-weight: 700; letter-spacing: .15em; }
#defloc-hero h1 { font-size: clamp(34px, 5vw, 57px); letter-spacing: -.065em; line-height: 1.04;
  margin: 6px 0 9px; font-weight: 720; color: #fff; }
#defloc-hero p { margin: 0; max-width: 690px; color: #e6dbf4; font-size: 14px; line-height: 1.55; }
#defloc-hero nav { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }
#defloc-hero nav a { color: #e5cfff; font-weight: 650; text-underline-offset: 3px; }
.workspace-card { border: 1px solid var(--border-color-primary); border-radius: 17px;
  padding: 14px; background: var(--block-background-fill); min-width: 0; }
.workspace-card h2 { font-size: 19px !important; letter-spacing: -.035em; margin: 0 0 2px; }
#run-comparison { font-weight: 720; min-height: 46px; border-radius: 11px; }
#defloc-draw .paint-toolbar, #defloc-draw .paint-options { display: flex; flex-wrap: wrap;
  gap: 7px; align-items: center; margin-bottom: 10px; }
#defloc-draw button { padding: 6px 11px; border: 1px solid var(--border-color-primary);
  border-radius: 8px; color: var(--body-text-color); background: var(--background-fill-secondary);
  font-size: 13px; font-weight: 650; }
#defloc-draw button[aria-pressed="true"] { background: var(--color-accent); color: white; }
#defloc-draw button:disabled { opacity: .48; cursor: default; }
#defloc-draw .paint-options { font-size: 13px; color: var(--body-text-color-subdued); }
#defloc-draw .paint-options label { display: inline-flex; gap: 7px; align-items: center; }
#defloc-draw .paint-options input[type="range"] { width: 90px; accent-color: var(--color-accent); }
#defloc-draw .paint-options [hidden] { display: none; }
#defloc-draw .paint-stage { position: relative; display: grid; place-items: center; overflow: hidden;
  background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);
  border-radius: 12px; }
#defloc-draw canvas { display: block; width: min(100%, 332px); height: auto; touch-action: none;
  cursor: crosshair; }
#defloc-draw [data-empty] { position: absolute; inset: 0; display: grid; place-items: center;
  padding: 16px; text-align: center; pointer-events: none; color: var(--body-text-color-subdued); }
#defloc-draw [data-empty][hidden] { display: none; }
.output-image img { object-fit: contain; }
#defloc-examples { width: 100%; max-width: 620px; margin: 12px auto 0; }
#defloc-gallery .grid-wrap { min-height: 0; max-height: none; overflow: visible; padding: 0; }
#defloc-gallery .grid-container {
  grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-rows: none; gap: 10px;
}
#defloc-gallery .thumbnail-item { width: 100%; height: auto; aspect-ratio: auto; }
#defloc-gallery .thumbnail-item img {
  display: block; width: 100%; height: auto; aspect-ratio: 1; object-fit: contain; border-radius: 0;
}
#defloc-gallery .caption-label { position: static; width: 100%; text-align: left; }
.quiet-note { font-size: 12px; color: var(--body-text-color-subdued); }
@media (max-width: 640px) {
  #defloc-hero { padding: 18px 16px; }
  .workspace-card { padding: 10px; }
  #defloc-gallery .grid-container { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""

HERO = """
<section id="defloc-hero">
  <div class="eyebrow">CVPR 2021</div>
  <h1>DeFLOCNet Studio</h1>
  <p>Restore a portrait or a scene. Paint a mask and add optional sketch and color hints.</p>
  <nav aria-label="Project resources">
    <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_DeFLOCNet_Deep_Image_Editing_via_Flexible_Low-Level_Controls_CVPR_2021_paper.pdf"
       target="_blank" rel="noopener noreferrer">Paper</a>
    <a href="https://github.com/KumapowerLIU/DeFLOCNet" target="_blank" rel="noopener noreferrer">Code</a>
    <a href="https://huggingface.co/KumaPower/DeFLOCNet" target="_blank" rel="noopener noreferrer">Weights</a>
  </nav>
</section>
"""

PAINT_TEMPLATE = """
<div class="paint-toolbar" role="group" aria-label="Drawing tools">
  <button type="button" data-tool="mask" aria-pressed="true" disabled>Mask</button>
  <button type="button" data-tool="sketch" aria-pressed="false" disabled>Sketch</button>
  <button type="button" data-tool="color" aria-pressed="false" disabled>Color</button>
  <button type="button" data-erase aria-pressed="false" disabled>Erase</button>
  <button type="button" data-undo disabled>Undo</button>
</div>
<div class="paint-options">
  <label>Brush size <input type="range" data-brush-size min="1" max="64" value="16" disabled>
    <output data-brush-label>16 px</output>
  </label>
  <label data-color-controls hidden>Hint color <input type="color" data-brush-color value="#dc8b77" disabled></label>
</div>
<div class="paint-stage">
  <canvas data-edit-canvas width="256" height="256" role="img" aria-label="Image editing canvas"></canvas>
  <span data-empty>Upload an image or choose a sample below.</span>
</div>
"""


def _square(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB").resize(SIZE, Image.Resampling.BICUBIC)


def _png_url(image: Image.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, "PNG")
        return PNG_URL + base64.b64encode(buffer.getvalue()).decode("ascii")


def _canvas_value(image: Image.Image, revision: str, hole: Image.Image | None = None) -> str:
    payload = {"background": _png_url(image), "revision": revision}
    if hole is not None:
        layer = Image.new("RGBA", SIZE, MASK_FILL)
        layer.putalpha(hole)
        payload["mask"] = _png_url(layer)
    return json.dumps(payload)


@dataclass(frozen=True)
class DemoSample:
    label: str
    image: Image.Image
    hole: Image.Image
    model: str
    seed: int = 2019


@dataclass(frozen=True)
class FaceSample:
    index: int
    image: Image.Image
    hole: Image.Image
    sketch: Image.Image
    rtv: np.ndarray
    color_mask: Image.Image

    @property
    def seed(self):
        return 20261101 + self.index


def load_face_samples(config_path: Path, device: str) -> dict[int, FaceSample]:
    import torch
    from .edges import hed_sketch
    from .evaluate import make_dataset, pil
    from .hed import HED
    from .runtime import Config, sha256

    config = Config.load(config_path)
    if sha256(config.hed_weights) != config.hed_sha256:
        raise ValueError("HED weights differ from the face run config")
    data = make_dataset(config, training=False)
    samples = {}
    try:
        if data.counts != {"image": len(data)} or len(data) <= max(FACE_ROWS) or not config.rtv_cache:
            raise ValueError("The face demo requires the HDTF holdout config and its RTV cache")
        hed = HED(config.hed_weights).to(device).eval()
        with torch.inference_mode():
            for index in FACE_ROWS:
                item = data[index]
                hole = ~item["valid"]
                edge = hed_sketch(hed, item["image"][None].to(device), config.hed_postprocess)[0, 0].cpu()
                samples[index] = FaceSample(index, pil(item["image"]), _white_mask(hole[0].numpy()),
                                            _white_mask((edge.bool() & hole[0]).numpy()),
                                            item["rtv"].permute(1, 2, 0).contiguous().numpy(),
                                            _white_mask(item["color_mask"][0].numpy()))
    finally:
        data.close()
    return samples


def _paint_layer(data: str | None) -> Image.Image:
    if data is None:
        return Image.new("RGBA", SIZE)
    if not isinstance(data, str) or not data.startswith(PNG_URL):
        raise ValueError("Invalid drawing layer. Clear the canvas and try again.")
    try:
        with Image.open(BytesIO(base64.b64decode(data[len(PNG_URL):], validate=True))) as layer:
            if layer.format == "PNG" and layer.size == SIZE:
                return layer.convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid drawing layer. Clear the canvas and try again.") from exc
    raise ValueError("Invalid drawing layer. Clear the canvas and try again.")


def _white_mask(selected):
    return Image.fromarray(selected.astype(np.uint8) * 255)


def prepare_controls(value: str, source: Image.Image, revision: str):
    if not isinstance(value, str) or len(value) > 2_000_000:
        raise ValueError("Invalid canvas data. Clear the canvas and try again.")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid canvas data. Clear the canvas and try again.") from exc
    if not isinstance(payload, dict) or payload.get("revision") != revision:
        raise ValueError("Wait for the current portrait to load before running.")
    mask = np.asarray(_paint_layer(payload.get("mask")).getchannel("A")) >= ALPHA_THRESHOLD
    if not mask.any() or mask.all():
        raise ValueError("Paint a mask that leaves some of the original image visible.")
    sketch_layer = _paint_layer(payload.get("sketch"))
    color_layer = _paint_layer(payload.get("color"))
    sketch = (np.asarray(sketch_layer.getchannel("A")) >= ALPHA_THRESHOLD) & mask
    supplied = (np.asarray(color_layer.getchannel("A")) >= ALPHA_THRESHOLD) & mask

    return source.convert("RGB"), _white_mask(mask), _white_mask(sketch), color_layer.convert("RGB"), _white_mask(supplied)


def build_demo(engine, samples: dict[int, FaceSample] | None = None, *, align_portrait=None, public_samples=()):
    metadata = engine.metadata()
    names = tuple(metadata)
    if len(names) != 2:
        raise ValueError("The comparison workspace requires exactly two models.")
    samples = samples or {}
    sample_rows = [(sample, None) for sample in public_samples]
    sample_rows.extend((DemoSample(f"HDTF {case.index}", case.image, case.hole, names[0], case.seed), case.index)
                       for case in samples.values())
    theme = gr.themes.Soft(primary_hue="violet", neutral_hue="slate", radius_size="lg",
                           font=["system-ui", "sans-serif"])

    def title(name):
        return f"{name} · {metadata[name]['step']:,} steps"

    with gr.Blocks(title="DeFLOCNet Studio", analytics_enabled=False) as demo:
        source = gr.State(None)
        revision = gr.State(None)
        case = gr.State(None)
        gr.HTML(HERO)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=325, elem_classes=["workspace-card"]):
                gr.Markdown("## Edit image")
                crop = gr.Radio([AUTO_CROP, FULL_IMAGE], value=AUTO_CROP if align_portrait else FULL_IMAGE,
                                visible=align_portrait is not None, label="Upload framing",
                                info="Use full image for scenes and portraits that are already cropped.")
                upload = gr.UploadButton("Upload image", file_types=["image"], type="filepath", size="sm")
                canvas = gr.HTML(
                    value="", elem_id="defloc-draw", html_template=PAINT_TEMPLATE, apply_default_css=False,
                    js_on_load=Path(__file__).with_name("draw_canvas.js").read_text(encoding="utf-8"),
                )
                gr.Markdown("Paint the area to fill in white. Add sketch lines and color hints inside the mask.",
                            elem_classes=["quiet-note"])
                with gr.Row():
                    clear = gr.Button("Clear strokes", size="sm", variant="secondary")
                with gr.Accordion("Mask and seed settings", open=False):
                    with gr.Row(visible=bool(samples)):
                        mode = gr.Radio([HELD_OUT, PAINTED], value=PAINTED, label="Mask source",
                                        info="Fixed mode uses the saved mask. Painted edit uses the canvas.")
                        guided = gr.Checkbox(label="Reference HED + RTV", value=False,
                                             info="Fixed samples only. Guides use the original image.")
                    seed = gr.Number(value=2019, precision=0,
                                     minimum=0, maximum=2**32 - 1, label="Shared seed",
                                     scale=1)

            with gr.Column(scale=7, min_width=335, elem_classes=["workspace-card"]):
                gr.Markdown("## Results")
                model_choice = gr.Radio([(title(name), name) for name in names] + [(BOTH, BOTH)],
                                        value=names[0], label="Model")
                run = gr.Button("Run inpainting", variant="primary", elem_id="run-comparison")
                status = gr.Markdown(START)
                with gr.Tabs():
                    with gr.Tab("Completion"):
                        with gr.Row():
                            out_a = gr.Image(label=title(names[0]), format="png", height=296, interactive=False,
                                             buttons=["download", "fullscreen"], elem_classes=["output-image"])
                            out_b = gr.Image(label=title(names[1]), format="png", height=296, interactive=False,
                                             visible=False, buttons=["download", "fullscreen"],
                                             elem_classes=["output-image"])
                    with gr.Tab("Wipe comparison"):
                        wipe = gr.ImageSlider(label=f"Masked input ↔ {names[0]}", type="pil", height=340,
                                              interactive=False)
                gr.Markdown("Each result copies the visible pixels from the 256 × 256 source. "
                            "Download the result as PNG.", elem_classes=["quiet-note"])
                with gr.Accordion("Checkpoint details", open=False):
                    for name, info in metadata.items():
                        gr.Markdown(f"**{title(name)}**  ·  `{info['architecture']}`  \nSHA-256 `{info['sha256']}`")

        if sample_rows:
            with gr.Column(elem_classes=["workspace-card"], elem_id="defloc-examples"):
                gr.Markdown("## Try a sample\nChoose an image to load its editable mask.")
                gallery = gr.Gallery(
                    [(Image.composite(Image.new("RGB", SIZE, MASK_FILL), sample.image, sample.hole),
                      sample.label) for sample, _ in sample_rows],
                    elem_id="defloc-gallery", columns=3, fit_columns=False, show_label=False,
                    object_fit="contain", interactive=False, allow_preview=False, buttons=[], format="png",
                )

        targets = [canvas, source, revision, case, mode, guided, seed, out_a, out_b, wipe, status]

        def reset_canvas(image, note, *, index=None, hole=None, shared_seed=2019):
            version = token_hex(8)
            value = _canvas_value(image, version, hole)
            return value, image, version, index, PAINTED, False, shared_seed, None, None, None, note

        def load_upload(path, framing):
            try:
                with Image.open(path) as raw:
                    if raw.width * raw.height > 24_000_000:
                        raise ValueError("Use an image under 24 megapixels.")
                    if framing == AUTO_CROP and align_portrait is not None:
                        image = align_portrait(raw)
                        note = "**Portrait framed** · HDTF-style square crop at 256 × 256. Paint a mask to begin."
                    else:
                        image = _square(raw)
                        note = "**Image loaded** · resized to 256 × 256. Paint a mask to begin."
            except (OSError, ValueError, RuntimeError, Image.DecompressionBombError) as exc:
                raise gr.Error(str(exc)) from exc
            return reset_canvas(image, note)

        def clear_strokes(image, shared_seed, index):
            if image is None:
                raise gr.Error(START)
            return reset_canvas(image, "Canvas cleared. Paint a new mask.", index=index, shared_seed=shared_seed)

        def sample_click(event: gr.SelectData):
            if type(event.index) is not int or not 0 <= event.index < len(sample_rows):
                raise gr.Error("Select a sample thumbnail.")
            sample, index = sample_rows[event.index]
            state = reset_canvas(sample.image, f"{sample.label} is ready. Its mask is editable.",
                                 index=index, hole=sample.hole, shared_seed=sample.seed)
            return (*state, sample.model)

        def show_results(choice, first=None, second=None, pair=None):
            comparing = choice == BOTH
            name = names[0] if comparing else choice
            wipe_label = f"{names[0]} ↔ {names[1]}" if comparing else f"Masked input ↔ {name}"
            return (gr.update(value=first, label=title(name)), gr.update(value=second, visible=comparing),
                    gr.update(value=pair, label=wipe_label))

        def switch_model(choice, source_image):
            note = ("Ready to run with the current image and mask." if source_image is not None else
                    START)
            return (*show_results(choice), note)

        def compare(canvas_value, shared_seed, source_image, source_revision, index, input_mode, use_guides, choice):
            started = time.perf_counter()
            try:
                if shared_seed is None or int(shared_seed) != shared_seed or not 0 <= shared_seed < 2**32:
                    raise ValueError("Enter an integer seed from 0 to 4,294,967,295.")
                if input_mode == HELD_OUT:
                    sample = samples.get(index)
                    if sample is None:
                        raise ValueError("Select an HDTF holdout sample or switch to Painted edit.")
                    image, hole = sample.image, sample.hole
                    lines, hues, hue_mask = (sample.sketch, sample.rtv, sample.color_mask) if use_guides else (None, None, None)
                    guide_label = "reference HED + RTV" if use_guides else "no guides"
                    input_label = f"HDTF {sample.index} · {guide_label}"
                else:
                    if source_image is None:
                        raise ValueError(START)
                    image, hole, lines, hues, hue_mask = prepare_controls(canvas_value, source_image, source_revision)
                    input_label = "painted edit"
                selected = names if choice == BOTH else (choice,)
                results = engine.predict(image, hole, sketch=lines, color=hues, color_mask=hue_mask,
                                         seed=int(shared_seed), model_names=selected)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            first = results[selected[0]]
            second = results[names[1]] if choice == BOTH else None
            masked = Image.composite(Image.new("RGB", SIZE, MASK_FILL), image, hole)
            pair = (first, second) if second is not None else (masked, first)
            coverage = np.asarray(hole).mean() / 255
            caption = "Both models" if choice == BOTH else choice
            note = (f"**{caption} completed** · {input_label} · {coverage:.1%} masked · "
                    f"seed `{int(shared_seed)}` · {time.perf_counter()-started:.2f} s")
            return (*show_results(choice, first, second, pair), note)

        upload.upload(load_upload, [upload, crop], targets, api_visibility="private")
        if sample_rows:
            gallery.select(sample_click, outputs=[*targets, model_choice], api_visibility="private", scroll_to_output=True)
        clear.click(clear_strokes, [source, seed, case], targets, api_visibility="private")
        run.click(compare, [canvas, seed, source, revision, case, mode, guided, model_choice],
                  [out_a, out_b, wipe, status],
                  api_visibility="private", concurrency_limit=1, concurrency_id="defloc-results")
        model_choice.change(switch_model, [model_choice, source], [out_a, out_b, wipe, status],
                            api_visibility="private", concurrency_id="defloc-results")
    return demo.queue(max_size=8), theme


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--label-a", default=FACE)
    parser.add_argument("--label-b", default=GENERAL)
    parser.add_argument("--sha256-a")
    parser.add_argument("--sha256-b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--face-config", type=Path, help="HDTF run config for private holdout samples")
    parser.add_argument("--face-detector", type=Path, help="Local YuNet ONNX model for automatic portrait framing")
    args = parser.parse_args()
    if not args.label_a.strip() or not args.label_b.strip() or args.label_a == args.label_b:
        parser.error("Provide two distinct, nonempty model labels.")
    import torch
    from .demo_inference import DemoInference

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    if torch.device(args.device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checkpoints = {args.label_a: args.checkpoint_a, args.label_b: args.checkpoint_b}
    hashes = {name: value for name, value in ((args.label_a, args.sha256_a), (args.label_b, args.sha256_b)) if value}
    engine = DemoInference(checkpoints, device=args.device, expected_sha256=hashes)
    samples = load_face_samples(args.face_config, args.device) if args.face_config else None
    align_portrait = None
    if args.face_detector is not None:
        from .portrait import PortraitCrop

        align_portrait = PortraitCrop(args.face_detector)
    demo, theme = build_demo(engine, samples, align_portrait=align_portrait)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False, theme=theme, css=CSS,
                max_file_size="16mb", show_error=False)


if __name__ == "__main__":
    main()
