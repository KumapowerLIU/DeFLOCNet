# Boundary and color consistency experiment

The selected continuation enables **boundary 0.1 only**, with low-frequency
weight **0** (`configs/sd-color-v2-boundary.json`). The joint-loss variant below
is retained as an experiment, not the default. See the pilot outcome below.

This is an additive change to the current SD-color DeFLOCNet training objective,
not the original paper loss. Generator, normalization, gray-purple hole fill,
control sampling, data, masks and adversarial architecture remain unchanged.

The existing LPIPS 1, hole L1 0.1, hinge GAN 0.1 and paper hole TV 0.1 are retained.
The `completion_consistency_v1` profile adds:

- Boundary gradient reconstruction, weight 0.1. Match the generated completed
  image's horizontal and vertical differences to GT differences within a 4px
  band around the hole/known transition. A pair must touch at least one hole
  pixel. Normalize per image by the number of selected RGB pairs. This includes
  cross-boundary pairs and preserves real GT edges rather than pushing every
  edge toward zero.
- Low-frequency RGB reconstruction, weight 0.5. Gaussian-filter the completed
  RGB error at sigma 3 and 7 (15px and 31px separable kernels). Sum absolute
  filtered residuals and normalize by the corresponding filtered hole mass,
  averaging per image and scale. The residual is zero outside the hole. Smooth
  the ERROR signal, not the generated image, so this loss does not directly
  penalize texture present in GT.

Both are computed in FP32 on RGB [-1,1]. Targets are detached. Gradients cannot
flow into generated known pixels. No inference blending or color correction is
introduced; inference still copies known RGB exactly. No sky classifier or
special sky-only branch is used.

## Controlled validation

The parent is safely paused with a full checkpoint. A paired migration preserves
G, D, both Adam optimizers, RNG, step, epoch and cursor, and permits changes only
to runtime/train/LPIPS loss source plus the new consistency module. Baseline and
candidate use the same new source snapshot; the baseline disables both added
terms. The first 32 normal batch-128 updates are bounded; after mixed results the
paired probe is extended, with unchanged weights, to a fixed total of 64. Per-step image
indices, control modes and noise hashes allow matched-input verification.

Then evaluate both checkpoints on the same saved four nature input bundles and
16 fixed validation images. Retain unguided/guided results, hand-drawn edits and
red/blue color edits. Inspect boundaries and large flat-color regions as well as
overall reconstruction and edit response. The blue-sky diagnostic is fixed to
GT blue pixels in the top 45 rows of case 21820; it is a local diagnostic, not a
segmentation benchmark. These initial weights and the short pilot are proposals,
not evidence of a general solution; record measured results before continuation.

The PartialConv paper uses TV over a 1px-dilated hole region, including its
boundary. That supports examining boundary supervision; the GT-gradient and
Gaussian-error losses above are our own design, not copied paper formulas.
Reference: <https://arxiv.org/html/1804.07723v2#S3.SS3>.

```bash
PYTHONPATH=src python scripts/migrate_consistency_checkpoint.py \
  --parent-run runs/paused-tv-run --output runs/consistency-pilot \
  --variant consistency --low-frequency-weight 0
python scripts/launch.py --config runs/consistency-pilot/config.json \
  --output runs/consistency-pilot --gpu GPU-YOUR-UUID --resume \
  --stop-after PARENT_STEP_PLUS_32
```

## Pilot outcome

From the same full-state parent at step 21,361, retain joint boundary/low-frequency
pilots at +32 and +64 and a boundary-only ablation at +32. All compared updates
match image indices, control modes and noise hashes; checkpoint migration
verifies model, optimizer, RNG and cursor preservation. The local artifacts are
under `artifacts/deflocnet-consistency-fix-20261104` in the parent workspace.

The joint-loss variant reduced boundary error but had inconsistent color and
validation results, so it is paused. In the equal +32 comparison, the selected
boundary-only variant reduced the specified blue-sky neighbor jump from 2.00 to
1.27/255 and its RGB MAE from 2.70 to 1.87/255. Across 16 fixed validation images,
guided hole PSNR was 18.052 to 18.061 dB and unguided 16.420 to 16.504 dB. The four
nature guided mean PSNR decreased from 18.679 to 18.266 dB, and not every image
or metric improved. This is a short local diagnostic, not general validation or
proof that seams are eliminated. Hand-drawn edits and recoloring remain
inspectable alongside reconstruction outputs. Production continues with only
the boundary term added; all original loss weights, network and data stay fixed.
