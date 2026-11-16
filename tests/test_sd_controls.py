from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from deflocnet.lpips_loss import FrozenLPIPS, hole_l1
from deflocnet.networks import build_generator, predict_generator
from deflocnet.runtime import Config, restore_rng, save_checkpoint, sha256
from deflocnet.sd_networks import FP32GroupNorm, SDControlGenerator

torch.set_num_threads(1)


def inputs(size=64):
    image = torch.rand(2, 3, size, size) * 2 - 1
    hole = torch.zeros(2, 1, size, size, dtype=torch.bool)
    hole[:, :, size//4:3*size//4, size//4:3*size//4] = True
    sketch = torch.zeros_like(image)
    sketch[:, :, size//2:size//2+2, size//4:3*size//4] = 1
    color_mask = torch.zeros_like(hole)
    color_mask[:, :, size//3:size//3+4, size//3:size//3+4] = True
    color = (torch.rand_like(image) * 2 - 1) * color_mask
    return image, hole, sketch, color, torch.rand_like(image), color_mask


def call(model, args):
    return model(*args[:5], color_mask=args[5])


def test_sd_controls_reach_output_and_both_pyramids_have_gradients():
    torch.manual_seed(123)
    model = SDControlGenerator(width=4, activation_checkpoint=False)
    args = inputs()
    out = call(model, args)
    assert out.shape == args[0].shape and torch.isfinite(out).all()
    assert out.abs().max() <= 1
    sketchless = list(args)
    sketchless[2] = torch.zeros_like(args[2])
    colorless = list(args)
    colorless[5] = torch.zeros_like(args[5])
    assert (out - call(model, sketchless)).abs().max() > 1e-8
    assert (out - call(model, colorless)).abs().max() > 1e-8
    (out * torch.randn_like(out)).mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert not any(isinstance(m, (nn.ConvTranspose2d, nn.InstanceNorm2d, nn.BatchNorm2d)) for m in model.modules())


def test_sd_ignores_hidden_target_and_invalid_control_values():
    torch.manual_seed(41)
    model = SDControlGenerator(4, False).eval()
    args = inputs()
    before = [a.clone() for a in args]
    corrupted = list(args)
    corrupted[0] = torch.where(args[1], torch.full_like(args[0], 1000), args[0])
    corrupted[2] = torch.where(args[1], args[2], torch.full_like(args[2], float('nan')))
    corrupted[3] = torch.where(args[5], args[3], torch.full_like(args[3], float('nan')))
    corrupted[4] = torch.where(args[1], args[4], torch.full_like(args[4], float('nan')))
    with torch.no_grad():
        torch.testing.assert_close(call(model, args), call(model, corrupted), atol=0, rtol=0)
    for a, b in zip(args, before):
        assert torch.equal(a, b)


def test_gray_color_hint_is_distinct_from_no_color_hint():
    torch.manual_seed(12)
    model = SDControlGenerator(4, False).eval()
    args = list(inputs())
    args[3].zero_()  # Zero in [-1,1] is real middle gray, not missing data.
    with torch.no_grad():
        gray = call(model, args)
        args[5].zero_()
        absent = call(model, args)
    assert (gray - absent).abs().max() > 1e-8


def test_bfloat16_does_not_round_away_initial_sketch_or_color_influence():
    torch.manual_seed(123)
    model = SDControlGenerator(4, False).eval()
    args = inputs()
    sketchless, colorless = list(args), list(args)
    sketchless[2] = torch.zeros_like(args[2])
    colorless[5] = torch.zeros_like(args[5])
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        full = call(model, args)
        assert torch.count_nonzero(full - call(model, sketchless)) > 0
        assert torch.count_nonzero(full - call(model, colorless)) > 0


def test_sd_checkpointing_preserves_output_and_parameter_gradients():
    torch.manual_seed(20)
    a = SDControlGenerator(4, False).train()
    b = copy.deepcopy(a)
    b.activation_checkpoint = True
    args = inputs()
    ya, yb = call(a, args), call(b, args)
    torch.testing.assert_close(ya, yb, atol=0, rtol=0)
    ya.square().mean().backward()
    yb.square().mean().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_groupnorm_bfloat16_uses_fp32_statistics():
    norm = FP32GroupNorm(8)
    x = (torch.randn(2, 8, 16, 16) * 10).bfloat16()
    expected = nn.functional.group_norm(x.float(), norm.num_groups, norm.weight, norm.bias, norm.eps).bfloat16()
    torch.testing.assert_close(norm(x), expected, atol=0, rtol=0)


def test_hole_l1_is_per_image_normalized_and_has_no_known_region_gradient():
    prediction = torch.zeros(2, 3, 8, 8, requires_grad=True)
    target = torch.ones_like(prediction)
    target[1] *= 3
    hole = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    hole[0, :, 0, 0] = True
    hole[1, :, :6, :6] = True
    loss = hole_l1(prediction, target, hole)
    assert loss == 2
    loss.backward()
    assert torch.count_nonzero(prediction.grad.masked_select(~hole.expand_as(prediction))) == 0
    assert hole_l1(prediction, target, torch.zeros_like(hole)) == 0


def test_three_loss_config_and_color_mask_contract():
    common = dict(train_manifest='', validation_manifest='', train_sha256='', validation_sha256='',
                  vgg_weights='', hed_weights='', vgg_sha256='', hed_sha256='', rtv_cache='')
    with pytest.raises(ValueError, match='disable'):
        Config(**common, loss_profile='l1_lpips_gan_v1')
    config = Config(**common, architecture='sd_controls_v1', width=4,
                    loss_profile='l1_lpips_gan_v1', l1_weight=.1, lpips_weight=1,
                    perceptual_weight=0, style_weight=0, tv_weight=0,
                    lpips_weights='explicit-file', lpips_sha256='a'*64)
    model = build_generator(config)
    args = inputs()
    with pytest.raises(ValueError, match='color_mask'):
        predict_generator(model, *args[:5])
    assert predict_generator(model, *args).shape == args[0].shape


def test_sd_checkpoint_restores_optimizer_rng_and_next_update(tmp_path):
    torch.manual_seed(2)
    config = Config('', '', '', '', '', '', '', '', '', architecture='sd_controls_v1', width=4)
    a = build_generator(config)
    a.activation_checkpoint = False
    opt = torch.optim.Adam(a.parameters(), lr=2e-4)
    d = nn.Linear(1, 1)
    od = torch.optim.Adam(d.parameters())
    args = inputs()
    call(a, args).square().mean().backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    path = save_checkpoint(tmp_path, 1, a, d, opt, od, config, 0, 64)
    state = torch.load(path, map_location='cpu', weights_only=False)
    target = torch.rand_like(args[0])
    (call(a, args) - target).square().mean().backward()
    opt.step()
    b = build_generator(config)
    b.activation_checkpoint = False
    b.load_state_dict(state['generator'])
    ob = torch.optim.Adam(b.parameters(), lr=2e-4)
    ob.load_state_dict(state['optimizer_g'])
    restore_rng(state['rng'])
    restored_target = torch.rand_like(args[0])
    assert torch.equal(target, restored_target)
    (call(b, args) - restored_target).square().mean().backward()
    ob.step()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_lpips_offline_load_matches_official_value_and_gradient(tmp_path):
    import lpips
    from torchvision.models.vgg import make_layers, cfgs
    torch.manual_seed(28)
    features = make_layers(cfgs['D'])[:30].eval()
    # Deliberately synthetic fixture backbone: the test checks exact loading
    # and the official LPIPS calculation, not perceptual quality/pretraining.
    weights = tmp_path / 'fixture-vgg.pth'
    torch.save({'features.'+k: v for k, v in features.state_dict().items()}, weights)
    calibration = Path(lpips.__file__).parent / 'weights/v0.1/vgg.pth'
    loss = FrozenLPIPS(weights, calibration, sha256(weights), sha256(calibration))
    loss.train(True)
    assert not loss.training and not loss.metric.training
    assert all(not p.requires_grad for p in loss.parameters())
    x = (torch.rand(3,3,32,32)*2-1).requires_grad_()
    y = (torch.rand_like(x)*2-1).requires_grad_()
    for key, value in loss.metric.net.state_dict().items():
        torch.testing.assert_close(value, features.state_dict()[key.split('.',1)[1]], atol=0, rtol=0)
    a = loss(x, y)
    grad = torch.autograd.grad(a, x)[0]
    b = loss.metric(x, y.detach(), normalize=False).mean()
    expected_grad = torch.autograd.grad(b, x)[0]
    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(grad, expected_grad, atol=1e-7, rtol=1e-5)
    assert y.grad is None and grad.abs().sum() > 0
    with pytest.raises(ValueError, match='checksum'):
        FrozenLPIPS(weights, calibration, '0'*64, sha256(calibration))
