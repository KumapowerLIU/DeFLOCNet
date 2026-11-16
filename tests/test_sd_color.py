from __future__ import annotations

import copy

import pytest
import torch

from deflocnet.sd_color_networks import SDColorGenerator, SparseColorPyramid, pooled_color_guides
from deflocnet.runtime import Config
from deflocnet.networks import build_generator
from deflocnet.lpips_loss import hole_l1
from test_sd_controls import inputs, call


def test_sparse_pooling_preserves_rgb_and_gray_presence():
    color = torch.zeros(1, 3, 64, 64)
    valid = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    hole = torch.ones_like(valid)
    valid[:, :, 9, 11] = True
    color[:, :, 9, 11] = torch.tensor([.75, -.25, .5])
    for size in [64, 32, 16, 8, 4]:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            pooled = pooled_color_guides(color, valid, hole, (size, size))
        assert pooled.dtype == torch.float32
        present = pooled[:, 3:4] > 0
        actual = pooled[:, :3].masked_select(present.expand(-1, 3, -1, -1))
        torch.testing.assert_close(actual, torch.tensor([.75, -.25, .5]), atol=0, rtol=0)
    gray = pooled_color_guides(color * 0, valid, hole, (4, 4))
    absent = pooled_color_guides(color * 0, valid & False, hole, (4, 4))
    assert torch.equal(gray[:, :3], absent[:, :3])
    assert not torch.equal(gray[:, 3], absent[:, 3])


def test_no_color_is_exactly_zero_even_after_biases_change():
    module = SparseColorPyramid([4, 4])
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith('bias'):
                param.fill_(.7)
    color = torch.full((2, 3, 32, 32), float('nan'))
    valid = torch.zeros(2, 1, 32, 32, dtype=torch.bool)
    outputs = module(color, valid, ~valid)
    assert all(torch.isfinite(x).all() and torch.count_nonzero(x) == 0 for x in outputs)


@pytest.mark.parametrize('amp', [False, True])
def test_red_blue_changes_output_and_reaches_all_color_stages(amp):
    torch.manual_seed(33)
    model = SDColorGenerator(4, False)
    args = list(inputs())
    red = torch.tensor([1., -1., -1.]).view(1, 3, 1, 1).expand_as(args[3])
    blue = torch.tensor([-1., -1., 1.]).view(1, 3, 1, 1).expand_as(args[3])
    args[3] = red
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
        a = call(model, args)
        args[3] = blue
        b = call(model, args)
        delta = (a.float() - b.float()).abs()[args[1].expand_as(a)].mean()
        assert delta > 1 / 255  # Visible-scale response, beyond nonzero-gradient checks.
        (a.float().square().mean() + b.float().square().mean()).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if name.startswith(('color_pyramid.', 'color_up.')) or '.control.color' in name:
            assert p.grad.abs().sum() > 0, name


def test_recurrent_color_checkpointing_preserves_outputs_and_gradients():
    torch.manual_seed(3)
    a = SDColorGenerator(4, False).train()
    b = copy.deepcopy(a); b.activation_checkpoint = True
    args = inputs()
    ya, yb = call(a, args), call(b, args)
    torch.testing.assert_close(ya, yb, atol=0, rtol=0)
    ya.square().mean().backward(); yb.square().mean().backward()
    for (name, p), q in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)


def test_hidden_gt_invalid_guides_and_missing_color_do_not_leak():
    torch.manual_seed(9)
    model = SDColorGenerator(4, False).eval()
    args = inputs()
    corrupted = list(args)
    corrupted[0] = torch.where(args[1], 999., args[0])
    corrupted[2] = torch.where(args[1], args[2], float('nan'))
    corrupted[3] = torch.where(args[5], args[3], float('nan'))
    corrupted[4] = torch.where(args[1], args[4], float('nan'))
    with torch.no_grad():
        torch.testing.assert_close(call(model, args), call(model, corrupted), atol=0, rtol=0)
    empty = list(args); empty[5] = torch.zeros_like(args[5])
    with torch.no_grad():
        a = call(model, empty)
        empty[3] = torch.full_like(args[3], float('nan'))
        torch.testing.assert_close(a, call(model, empty), atol=0, rtol=0)


def test_v2_factory_is_explicit_and_v1_remains_separate():
    common = dict(train_manifest='', validation_manifest='', train_sha256='', validation_sha256='',
                  vgg_weights='', hed_weights='', vgg_sha256='', hed_sha256='', rtv_cache='', width=4)
    assert build_generator(Config(**common, architecture='sd_color_v2')).architecture == 'sd_color_v2'
    assert build_generator(Config(**common, architecture='sd_controls_v1')).architecture == 'sd_controls_v1'


def test_sparse_color_can_learn_and_swap_red_blue_with_identical_context():
    # A tiny learning regression, not a natural-image quality benchmark. Context,
    # mask, sketch and noise are identical; only four supplied RGB pixels differ.
    torch.manual_seed(6102)
    model = SDColorGenerator(4, False).train()
    hole = torch.zeros(2, 1, 32, 32, dtype=torch.bool)
    hole[:, :, 8:24, 8:24] = True
    hints = torch.zeros_like(hole); hints[:, :, 15:17, 15:17] = True
    palette = torch.tensor([[.7, -.7, -.7], [-.7, -.7, .7]])[:, :, None, None]
    target = torch.where(hole, palette, torch.zeros(2, 3, 32, 32))
    colors = palette.expand_as(target) * hints
    sketch = torch.zeros_like(target); sketch[:, :, 10:12, 8:24] = 1
    noise = torch.rand(1, 3, 32, 32).expand(2, -1, -1, -1).clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        output = model(target, hole, sketch, colors, noise, color_mask=hints)
        hole_l1(output, target, hole).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(target, hole, sketch, colors, noise, color_mask=hints)
        swapped = model(target, hole, sketch, colors.flip(0), noise, color_mask=hints)
        assert hole_l1(output, target, hole) < .06
        assert hole_l1(swapped, target.flip(0), hole) < .06
        means = (output * hole).sum((2, 3)) / hole.sum((2, 3))
        assert means[0, 0] - means[1, 0] > 1
        assert means[1, 2] - means[0, 2] > 1
