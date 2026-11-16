# SD-style controlled RGB generator

## Color propagation revision (2026-11-02)

`sd_color_v2` starts a new experiment from scratch with the same RGB U-Net,
sketch path, dataset, medium-HED protocol, discriminator, Adam settings and
`0.1*hole_L1 + 1.0*LPIPS + 0.1*hinge_GAN`. Effective batch is 128 (32 x 4).
The earlier `sd_controls_v1` implementation and frozen run snapshots remain available.

The 1,750-step v1 audit found weak natural-image color control despite valid
inputs and gradients: across four fixed mountain cases, adding the original
color dots changed hole RGB by only 0.070/255 on average; swapping those dots
between red and blue changed it by 0.244/255. These are local diagnostics, not
benchmark results or proof of a single cause.

V2 changes only the color-conditioning mechanism:

- At each resolution, RGB is averaged over supplied color pixels, dividing by
  their valid count. Empty cells remain zero; an explicit presence channel
  distinguishes missing color from real gray. FP32 pooling prevents sparse
  guides from being diluted by empty pixels under BF16.
- The decoder carries learned color features through consecutive residual
  blocks and from coarse to fine scales. After sketch gating, learned features
  use FP32 GroupNorm and LeakyReLU, restoring the original SGB ordering of
  gating, normalization and propagation. This uses GroupNorm rather than the
  reference InstanceNorm; it is not an exact reproduction of original SGB.
- Color and sketch use separate affine heads. Missing color disables the whole
  color contribution, including learned biases. Supplied RGB itself is not
  normalized, and no colors are pasted into the completed image.

Checks include FP32/BF16 red-blue sensitivity, sparse/gray/missing-color
semantics, cross-scale gradients, checkpoint recomputation and hidden-GT
isolation. A small synthetic learning test holds context, mask, sketch and
noise fixed while swapping four red/blue hint pixels and checks that the
learned output follows the swap. It does not establish natural-image quality;
fixed mountain controls remain a checkpoint-level acceptance check.

Run the separate profile with `configs/sd-color-v2-l1-lpips-gan.json`.

## V1 backbone and objectives

This experiment replaces the released generator topology; it is not a numerical
refactor or a reproduction of the 2021 paper. The original implementation remains
available as `architecture=released_v1`, with `loss_profile=released_five_v1`.
Existing runs have independent frozen source snapshots.

## Backbone and control paths

`sd_controls_v1` is a one-pass RGB U-Net with five levels (256, 128, 64, 32, 16),
width multipliers (1, 2, 4, 8, 8), two residual blocks at each level, and a
residual/self-attention/residual bottleneck. GroupNorm statistics run in FP32 under
AMP. Decoder upsampling is nearest-neighbor resize followed by a 3x3 convolution.
No transposed convolutions, diffusion timesteps, text conditioning or SD pretrained
weights are used. Input channels are masked RGB, one validity channel and three
hole-restricted uniform-noise channels, retaining the original editor's noise.

Sketch (one channel plus hole) and color (RGB plus explicit hint-validity and hole)
have separate learned pyramids. Two convolutions precede the first downsampling,
and subsequent reductions use average pooling and learned convolutions to avoid
discarding sparse controls by nearest-sampling raw lines/dots. Invalid controls
are removed with `where` before processing; a valid middle-gray hint remains
distinguishable from an absent hint. The user-provided color mask is intersected
with the editing hole.

Every decoder residual block receives both pyramids. Sketch features gate color
features through a learned `1-sigmoid(gate)` factor; the concatenated features
produce spatial gamma/beta for `(1+gamma)*GroupNorm(h)+beta`. This retains the SGB
idea of structure-conditioned color propagation, but does not claim a hard
mathematical prohibition on colors crossing a drawn line.

Unconditioned residual and attention projections start small. Controlled residual
outputs use fan-scaled initialization with gain 0.1, and modulation heads use
standard deviation 0.02. Initializing both projections at 0.001 made the control
effect disappear under BF16 rounding in a diagnostic; the current initialization
is covered by an explicit BF16 sketch/color response regression test.

The design references residual blocks/GroupNorm/SiLU/resize-convolution from
[CompVis latent diffusion](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py)
and spatial scale/shift modulation from
[SPADE](https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py).

## Objective

`l1_lpips_gan_v1` uses exactly `0.1*hole_L1 + 1.0*LPIPS + 0.1*hinge_GAN`.

- L1 averages absolute RGB error over hole pixels separately for each image,
  then averages images. Images are represented in [-1,1]. An empty hole contributes zero.
- LPIPS uses the official calibrated VGG v0.1 implementation on completed RGB
  and GT in [-1,1]. It is evaluated in FP32, with frozen metric parameters and
  disabled calibration dropout. Two-image chunks and non-reentrant checkpointing
  bound activation memory. Backbone/calibration files are SHA256-checked; the
  fully loaded production metric has no randomly initialized feature/calibration weights.
- GAN retains the released global/local two-scale Patch discriminators and hinge
  reduction. Both G and D use the same completed RGB seen by the user. Local
  64px crops, controls, noise and data are shared across the paired D/G update.
- Independent VGG perceptual, Gram/style and TV losses must have zero weights;
  the config validator rejects accidental activation.

These coefficients are a starting experiment, not a guarantee that gradient
contributions are balanced or that the model obeys arbitrary hand-drawn controls.
Fixed mountain/ridge/color edits remain qualitative acceptance cases.

## Verification and resumption

Tests cover control response and parameter gradients throughout both pyramids,
BF16 control visibility, no hidden-GT/invalid-guide leakage, gray hint semantics,
checkpointed output/gradient equality, exact next-update Adam/RNG restoration,
per-image hole-L1 reduction, strict color-mask/config contracts, and LPIPS output
and gradient agreement with its official module. Synthetic unit-test VGG weights
are explicitly fixtures; server training uses the existing verified ImageNet weights.

Training keeps the original data manifest, deterministic masks/control modes,
2-3px `hed_nms_medium_v1` guides, Adam 2e-4 and effective batch 64. Checkpoints
contain model/discriminator/optimizer/RNG/cursor/config/source hashes. LPIPS
weight paths and checksums are pinned in config and verified again on resume.
The prior step-23003 run is preserved, and the new architecture starts at step zero.
