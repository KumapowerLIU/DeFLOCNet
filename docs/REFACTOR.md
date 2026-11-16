# Reference alignment and repaired training behavior

The modern implementation follows the released network at commit
`e8f2c3e` (full original commit is recorded by `git rev-parse HEAD`). This is a
new experiment on the CSA mixed data, not a reproduction of the published
Places2/CelebA-HQ scores.

## Network retained

- Six stride-2 encoder layers, width 64, four dilation-2 bottleneck residuals.
- Nine encoder channels: masked RGB, three repeated valid-mask channels, RGB
  uniform noise restricted to the hole. No target RGB enters the encoder hole.
- SGBs at 128, 64, 32, 16, 8, 4 pixels, with 6, 5, 4, 3, 2, 1 cells.
- Each SGB retains sketch refinement, sketch-gated color propagation, and
  feature fusion, including the original residual addition order.
- Modulation is `(1 + sigmoid(gamma_logits)) * InstanceNorm(x) + beta`, as in
  public code. This is more specific than the paper's general affine equation.
- Original transposed-convolution structure decoder and parallel texture
  decoder, with five feature additions. Removed only the unused texture RGB
  head and six unused sketch-supervision heads.
- Two-scale global and two-scale local discriminators. Global input has RGB
  plus three repeated hole channels (6); local input is a random 64x64 RGB crop
  (3). Spectral parametrizations are used for the same intermediate conv layers.

Tests load identical parameters into old/new networks and compare full-width
outputs exactly in CPU FP32. Separate SGB tests compare all used gradients.
Activation checkpointing is also checked for exact output/gradient equivalence.
These establish refactor equivalence, not checkpoint quality.

## Loss profile: `csa-mixed-hinge`

Weights follow paper Eq.11: L1 1, VGG perceptual 0.05, style 250, TV 0.1,
GAN 0.1. The loss implementations retain the public code's raw/completed-image
supervision and hinge GAN, with the correctness fixes below:

| Term | Actual reduction |
| --- | --- |
| L1 | mean over B*3*H*W of `abs(raw-GT) * (1+hole)` |
| Perceptual | sum of VGG feature L1 at relu1_1/2_1/3_1/4_1/5_1, raw and completed |
| Style | normalized Gram L1 at relu2_2/3_3/4_3/5_2, raw and completed |
| TV | adjacent horizontal/vertical pairs both inside the hole, divided by B*3*H*W |
| GAN | hinge; average scales within a D, sum global/local D |

The paper text calls its GAN "relativistic average LS", but Eq.9 uses a
sigmoid/log form and the public code uses hinge. This profile explicitly uses
the public hinge implementation; it must not be described as exact paper GAN.
The paper says style uses the perceptual layers; the public code uses the four
layers listed above. That public layer selection is retained, with relu5_2
fixed. No LPIPS, feature-matching, diversity, or Real-ESRGAN is added.

VGG receives RGB [-1,1] mapped to [0,1] then ImageNet mean/std normalization.
All VGG reductions, Gram matrices and TV run in FP32. VGG parameters are frozen;
the output path remains differentiable.

## Defects repaired

1. Target/image aliasing: `set_input` assigned the same tensor to GT and input,
   then filled input in place. The new generator uses `torch.where` without
   modifying the target; poisoning hole GT while holding controls fixed must
   leave the output unchanged.
2. Dataset pairing: the old loader used color/GT index 0 and inconsistent path
   lists. One manifest record now supplies aligned RGB, RTV, and HED controls.
3. Discriminator input channels: the old single config constructed both D with
   6 channels but passed only 3 channels to local D. They are explicit now.
4. TV: fixed inverted hole convention and the vertical first-row/last-row
   subtraction. Tests compare against a brute-force neighbor sum.
5. VGG: fixed relu5_2/5_3 repeat calls to relu5_1, corrected RGB normalization,
   and shared one frozen feature extractor instead of two full VGG models.
6. Loss configuration: `lambda_Style` was referenced but only `lambda_S` was
   declared; decimal weights used `type=int`; parentheses weighted only the
   completed perceptual term. Explicit float weights apply to whole terms.
7. Removed hard-coded `.cuda()`, fixed batch-size-dependent masking, removed
   retained D graphs and D gradients during G updates. Spectral buffers freeze
   during G updates, and real/fake D evaluation shares one batch.

## Data and controls

The manifest and SHA256 match the existing CSA mixed experiment. Indexed image,
TAR and TFRecord records are decoded directly; no dataset is downloaded or
unpacked. All RGB is EXIF-corrected, resized to 256 square with PIL bicubic,
then optionally flipped. Masks use the CSA rectangles/strokes/mixed sampler,
with 10%-40% missing area.

RTV uses the MATLAB equations and parameters in this repo:
lambda=0.015, sigma=3, sharpness=0.001, iterations=3. The SciPy direct sparse
solve is reused from the audited MEDFE implementation. It is not a blur proxy.
Existing content-checked MEDFE RTV entries are read without alteration; new
entries go into DeFLOCNet's own cache. Like MEDFE, RTV runs after square resize,
whereas original offline preprocessing could run before resizing.

HED uses Simon Niklaus's PyTorch implementation/BSDS500 weights with correct
RGB-to-BGR conversion. The initial `threshold_v1` profile directly thresholded
probabilities at 0.5, producing broad white bands/blobs. The continuation uses
`hed_nms_thin_v1`: Gaussian smoothing (sigma 1) for Hessian ridge-normal
estimation, bilinear NMS at +/-1 pixel on the original probabilities (1% response
tolerance), the same >0.5 threshold, then Zhang-Suen skeletonization. A constant
map has no ridge. The later `hed_nms_medium_v1` profile keeps those exact
centerlines and adds one symmetric radius-1 cross dilation per image, yielding
approximately 2-3px strokes. This matches the reviewed intermediate-width
candidate, and changes the actual training and evaluation guides. Processing
uses the full image before masking the hole, so the mask boundary cannot create
new edges. Training and evaluation call the same helper. This explicit
SciPy/skimage implementation is not a bitwise reproduction of HED's MATLAB
evaluation toolbox. The checkpoint config selects the protocol; old configs
default to their original threshold-only behavior. Uniform
choices of no controls, sketch only, color only, and both prevent requiring
GT-derived guides at inference. Color controls use 1-12 random RTV dots of
radius 2-6 pixels, restricted to the hole. This is a documented modern sampling
choice because the released training entrypoint/protocol was missing.
RTV is used on all mixed images, including faces; the original face protocol
used segmentation-derived median colors. Thus this is not an exact original
CelebA-HQ control-data reproduction.

Evaluation always separates GT-guided editing from zero-control inpainting.
Guided results have target-derived hole information and must not be compared
as if they were unguided inpainting results.

## Training and resumption

BF16 convolution, FP32 master weights/Adam, non-reentrant SGB activation
checkpointing. Effective batch 64 is accumulated from microbatches; per-image
InstanceNorm avoids microbatch batch-statistic changes. D spectral power
iteration occurs per D forward; accumulated training is not claimed bitwise
equivalent to a single full-batch spectral update.

Each G step follows one completed accumulated D step. Data/noise/control/crop
are shared between those two passes. Both gradients must be finite before
their optimizer step. An unsuccessful G step never writes a checkpoint with a
partially advanced D state. SIGTERM/SIGINT finishes a full step and saves.
Checkpoints include G/D/Adam/RNG/epoch/cursor/config/source hashes, are atomically
replaced, and retain two numbered checkpoints plus an atomic `latest.pt` symlink.
No best-checkpoint claim is made. Preview generation preserves training RNG.
