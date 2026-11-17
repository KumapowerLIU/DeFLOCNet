# Training DeFLOCNet

Install the package from the [quick start](../README.md#quick-start).

The JSON files in [configs/](../configs/) are editable training templates:

| Recipe | Config |
| --- | --- |
| HDTF portrait, 10,000-step schedule | [sd-color-v2-hdtf.json](../configs/sd-color-v2-hdtf.json) |
| General scenes, boundary-loss recipe | [sd-color-v2-boundary.json](../configs/sd-color-v2-boundary.json) |

Prepare separate training and validation JSONL manifests, local weight files,
and an RTV cache location; update the corresponding paths and SHA256 fields
before running. Training reads existing images and verifies the supplied hashes.
It does not download an image dataset.

A manifest can list ordinary image files, one absolute path per JSON record:

```json
{"format": "image", "path": "/absolute/path/to/image.jpg"}
```

Indexed TAR and TFRecord records are also supported. For video-derived portrait
frames, separate source videos when constructing training and validation sets.
Copy the selected template to your own config and edit its data and weight
locations:

```bash
cp configs/sd-color-v2-boundary.json configs/my-data.json
# Edit configs/my-data.json before training.
python -m deflocnet.train --config configs/my-data.json \
  --output runs/my-data --device cuda:0
```

Both released runs used effective batch 128, microbatch 32, BF16, activation
checkpointing, and Adam at `2e-4`. Adjust the microbatch to available GPU memory;
the effective batch must be divisible by it. Random rectangles, strokes, and
their unions cover 10%–40% of the training image.

Training samples four control modes: no controls, sketch only, color only,
and both. Sketch supervision uses versioned HED postprocessing with thin
2–3px strokes; color supervision uses sparse RTV-derived hints. These training
guides are derived from the target image. At inference, you supply your own
optional hints.

| Released models' training objective | Weight |
| --- | ---: |
| Hole-normalized RGB L1 | 0.1 |
| Calibrated LPIPS-VGG v0.1 | 1.0 |
| Hinge GAN with global/local Patch discriminators | 0.1 |
| Paper hole TV | 0.1 |
| Boundary-gradient reconstruction | 0.1 |
| Low-frequency reconstruction | 0 |

LPIPS and GAN operate on completed RGB. Independent VGG perceptual and style
terms are disabled in these profiles. LPIPS stays frozen while allowing
gradients to the generated image. See [control architecture](SD_CONTROLS.md)
and [completion-consistency losses](COMPLETION_CONSISTENCY.md) for the
precise reductions and earlier experimental variants.

To continue a training run, use its saved config and full checkpoint:

```bash
python -m deflocnet.train --config runs/my-data/config.json \
  --output runs/my-data --resume runs/my-data/latest.pt --device cuda:0
```

Resume checks source/config compatibility and restores optimizer state, RNG,
and the data cursor. Checkpoint retention is controlled by `keep_checkpoints`;
`latest.pt` points to the newest saved training state. Fixed evaluation previews
are generated separately. Use a new run for a new data or architecture profile.

For fixed validation previews from a full training checkpoint and its local
validation data:

```bash
python -m deflocnet.evaluate --checkpoint runs/my-data/latest.pt \
  --output evaluations/my-data
```

Guided and no-control results are reported separately. Reference HED/RTV guides
derived from the target contain information inside the hole, so guided metrics
must be interpreted with that input condition.

## Dataset sources

Datasets, private manifests, and held-out HDTF previews are not bundled. Obtain images from
their providers and follow their terms: [HDTF](https://github.com/MRzzm/HDTF),
[Places2](http://places2.csail.mit.edu/),
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and
[Unsplash](https://unsplash.com/). The exact mixed collection is not offered as
a public dataset download.

## Software checks

```bash
python -m pip install -e '.[test]'
pytest -q
```

For a shallow clone, run `git fetch origin legacy` to enable the two tests that
compare the PyTorch 2 reference profile against the original code.
