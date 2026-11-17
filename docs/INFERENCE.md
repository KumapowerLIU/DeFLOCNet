# Local inference

## Interactive Studio

Install the [demo extras](../README.md#quick-start), download the two
[checkpoints](../README.md#checkpoints), and run:

```bash
python -m deflocnet.web_demo \
  --checkpoint-a checkpoints/deflocnet-hdtf-step10000.pt \
  --checkpoint-b checkpoints/deflocnet-general-step23500.pt \
  --label-a "HDTF portrait" --label-b "General scenes" \
  --device cuda:0
```

The local app opens at `http://127.0.0.1:7860`. Use `--device cpu` without CUDA,
or `--port` to choose another port. It binds to loopback with public sharing
disabled. Optional `--sha256-a` and `--sha256-b` verify the checkpoint files.

1. **Prepare the image.** Choose **Use full image** for a scene or an existing
   portrait crop. It resizes the RGB image to the 256 × 256 canvas used in
   training. For an uncropped portrait, choose **Auto-frame portrait** when
   available. Inspect the prepared image before drawing; the result is this canvas.
2. **Paint the hole.** White mask pixels mark the region to generate. Leave
   visible context around the edit.
3. **Add optional controls.** Sketch lines guide structure; sparse color marks
   suggest colors. You can enable either control independently or use both.
4. **Run a model or compare both.** Select **HDTF portrait**, **General scenes**,
   or **Compare both**. Comparison uses the same prepared image, mask, hints,
   and noise seed. Inspect the side-by-side results or wipe slider, then
   download the completed PNG.

### Optional portrait framing

The local app can use the OpenCV YuNet face detector to produce a square,
face-centered crop. Install the demo extras, download the detector, and add
`--face-detector weights/face_detection_yunet_2023mar.onnx` to the launch command:

```bash
mkdir -p weights
curl -fL https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -o weights/face_detection_yunet_2023mar.onnx
```

Use a photo with one clearly visible face. The app checks the detector file's
checksum and shows the crop before editing. This is an axis-aligned crop that
approximates HDTF framing; [HDTF's original crop windows](https://github.com/MRzzm/HDTF#processing-of-hdtf-dataset)
are defined per video clip. General-image editing does not require face detection.

The optional `--face-config` flag is for a locally available HDTF validation
manifest and RTV cache. It adds fixed validation samples and optional reference
guides to the editor. Uploaded images need no ground-truth guides or dataset.

## Command-line inference

Prepare aligned **256 × 256** inputs. The image and all control maps must refer
to the same crop and pixel coordinates. The CLI uses these supplied inputs
directly; use the Studio to prepare a portrait crop interactively.

```bash
python -m deflocnet.infer \
  --checkpoint checkpoints/deflocnet-general-step23500.pt \
  --image input.png --mask hole.png \
  --output results/example --device cuda:0 --seed 2019
```

For portraits, substitute `checkpoints/deflocnet-hdtf-step10000.pt`. Use
`--device cpu` to run on CPU. `--output` is a directory containing
`completed.png`, the uncomposited `raw.png`, and `inference.json` metadata.

| Input | Convention |
| --- | --- |
| `--image` | RGB image providing the visible context |
| `--mask` | White = hole to generate; black = known image |
| `--sketch` | Optional white sketch lines on a black background |
| `--color` | Optional RGB color hints |
| `--color-mask` | White = supplied color hint; black = absent hint |

Supply `--color` and `--color-mask` together. The separate validity mask allows
black and gray to be real hints. Controls are applied inside the editing hole.
For an edit with both types of guidance:

```bash
python -m deflocnet.infer \
  --checkpoint checkpoints/deflocnet-general-step23500.pt \
  --image input.png --mask hole.png --sketch sketch.png \
  --color color.png --color-mask color-mask.png \
  --output results/guided --device cuda:0 --seed 2019
```
