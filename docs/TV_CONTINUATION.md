# TV continuation

The `l1_lpips_tv_gan_v1` profile adds only paper Eq.10 TV, with weight 0.1,
to the existing `l1_lpips_gan_v1` training objective. LPIPS remains 1, hole L1
0.1 and hinge GAN 0.1. All architecture, control generation, data, mask,
batch, optimizer and evaluation settings are preserved.

TV sums absolute horizontal and vertical RGB differences only for adjacent
pairs whose two pixels are inside the hole, then divides by B*3*H*W. It is
computed in FP32 on completed RGB in [-1,1]. There is no boundary-pair penalty,
and no gradient into generated pixels outside the hole. Perceptual/style
losses remain disabled. The earlier three-loss profile still rejects nonzero TV,
so old experiments cannot silently change their objective.

To continue, safely pause the parent, then create a separate run:

```bash
PYTHONPATH=src python scripts/migrate_tv_checkpoint.py \
  --parent-run runs/parent-without-tv --output runs/continuation-with-tv
python scripts/launch.py --config runs/continuation-with-tv/config.json \
  --output runs/continuation-with-tv --gpu GPU-YOUR-UUID --resume
```

The migration accepts only the loss-profile/TV changes and the corresponding
runtime/train/LPIPS-loss source updates. It preserves and verifies serialized
generator, discriminator, both Adam optimizers, RNG, step, epoch and data cursor.
It records parent checkpoint SHA256, both source maps and config maps in
`tv-migration.json`. Ordinary resume retains strict source/config checks.

Tests cover TV values and gradients against explicit adjacent pairs, empty and
isolated hole pixels, unchanged original reconstruction terms, profile validation,
and rejection of unrelated configuration/source changes during migration.
