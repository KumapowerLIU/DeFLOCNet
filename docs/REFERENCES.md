# Sources and attribution

- [DeFLOCNet paper, CVPR 2021](https://arxiv.org/abs/2103.12723), especially
  Figures 2-3 and Section 3.3.
- [Original DeFLOCNet code](https://github.com/KumapowerLIU/DeFLOCNet), local
  original files retained under `model`, `config`, `data`, and `demo.py`.
- HED: Saining Xie and Zhuowen Tu, *Holistically-Nested Edge Detection*, ICCV
  2015. [Original HED](https://github.com/s9xie/hed).
- PyTorch HED implementation: Simon Niklaus,
  [sniklaus/pytorch-hed](https://github.com/sniklaus/pytorch-hed/tree/d36b4506f0c0cb5c628f43db2697d9e0421d50bd).
  `src/deflocnet/hed.py` is adapted from its `run.py`, with explicit weight
  loading, RGB input conversion, and no global autograd/device side effects.
  Modified November 1, 2026. Its GPL-3.0 license is included in
  [HED-LICENSE.txt](HED-LICENSE.txt); the original project MIT license does not
  replace the upstream license for this component.
- HED weights: `https://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch`,
  SHA256 `58a858782f5fa3e0ca3dc92e7a1a609add93987d77be3dfa54f8f8419d881a94`.
- Edge postprocessing: HED's official README specifies NMS and thinning for
  boundary evaluation. Our versioned implementation uses Hessian ridge normals
  and bilinear directional suppression, followed by scikit-image's
  [Zhang-Suen skeletonize](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize).
  It is not copied from, or numerically identical to, the MATLAB/C++ toolbox.
- VGG16: torchvision/ImageNet `vgg16-397923af.pth`, SHA256
  `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`.
- RTV: Li Xu et al., *Structure Extraction from Texture via Relative Total
  Variation*, SIGGRAPH Asia 2012. The original MATLAB source/license remains
  in `data/matlab/code/tsmooth.m`; the Python sparse-solver implementation was
  reused from the local modern MEDFE project with identical parameters.
- Random masks: local modern CSA sampler, reused without a distribution change.

Datasets remain subject to their original licenses. This repo contains neither
the training image collection nor pretrained model binaries.
