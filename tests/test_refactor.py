from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import pickle
import random
import struct

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from deflocnet.controls import sample_controls, RTVCache
from deflocnet.data import IndexedImages, EpochMicrobatches, decode_tfrecord
from deflocnet.losses import hole_tv, gram, hinge_discriminator, hinge_generator
from deflocnet.masks import random_mask
from deflocnet.networks import Generator, StructureGenerationBlock, Discriminators
from deflocnet.runtime import Config, capture_rng, restore_rng, save_checkpoint

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def legacy(name):
    spec = importlib.util.spec_from_file_location("reference_" + name, ROOT / "model/network" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs(batch=1):
    image = torch.rand(batch, 3, 256, 256) * 2 - 1
    hole = torch.zeros(batch, 1, 256, 256, dtype=torch.bool)
    hole[:, :, 63:191, 91:215] = True
    return image, hole, torch.rand_like(image) * hole, torch.rand_like(image) * hole, torch.rand_like(image)


def test_full_generator_matches_released_network():
    """Full-width, all 21 SGB cells and both decoders, identical state/input."""
    torch.manual_seed(27)
    modern = Generator(checkpoint_sgb=False).eval()
    en = legacy("encoder").Encoder(input_nc=9, norm_layer=nn.InstanceNorm2d).eval()
    de = legacy("decoder").Decoder(norm_layer=nn.InstanceNorm2d).eval()
    edit = legacy("structure_generation_block").Edit().eval()
    for i, block in enumerate(modern.encoder.blocks, 1):
        getattr(en, f"Encoder_{i}").model.load_state_dict(block.state_dict())
    en.middle.load_state_dict(modern.encoder.middle.state_dict())
    for i, block in enumerate(modern.decoder.structure, 1):
        getattr(de, f"Decoder_{i}").model.load_state_dict(block.state_dict())
    de.Decoder_6.model.load_state_dict(modern.decoder.output.state_dict())
    for i, block in enumerate(modern.decoder.texture, 1):
        getattr(de, f"Decoder_{i}_2").model.load_state_dict(block.state_dict())
    for scale, block in zip((128, 64, 32, 16, 8, 4), modern.sgb):
        getattr(edit, f"e_{scale}").load_state_dict(block.state_dict(), strict=True)
    image, hole, sketch, color, noise = inputs()
    with torch.no_grad():
        masked = torch.where(hole, modern.fill, image)
        encoded = en(torch.cat((masked, (~hole).expand_as(image).float(), noise * hole), 1))
        expected = de(edit(encoded, sketch, color, noise * hole, hole.float()))
        actual = modern(image, hole, sketch, color, noise)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_sgb_backward_matches_reference():
    torch.manual_seed(3)
    modern = StructureGenerationBlock(8, 3)
    old = legacy("structure_generation_block").Edit_block(3, 8, 8)
    old.load_state_dict(modern.state_dict())
    x = torch.randn(2, 8, 16, 16)
    guides = [torch.rand(2, 3, 32, 32) for _ in range(3)] + [torch.ones(2, 1, 32, 32)]
    modern(x, *guides).square().mean().backward()
    old(x, *guides)[1].square().mean().backward()
    for (name, p), (old_name, q) in zip(sorted(modern.named_parameters()), sorted(old.named_parameters())):
        assert name == old_name
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_masking_does_not_modify_target_or_leak_hole_rgb():
    model = Generator(width=4).eval()
    args = inputs()
    original = args[0].clone()
    poisoned = torch.where(args[1], torch.full_like(original, 999), original)
    with torch.no_grad():
        first = model(*args)
        second = model(poisoned, *args[1:])
    assert torch.equal(args[0], original)
    assert torch.equal(first, second)
    complete = torch.where(args[1], first, original)
    assert torch.equal(complete.masked_select(~args[1]), original.masked_select(~args[1]))


def test_activation_checkpoint_matches_gradients():
    a = Generator(width=4, checkpoint_sgb=False).train()
    b = copy.deepcopy(a)
    b.checkpoint_sgb = True
    args = inputs()
    ya, yb = a(*args), b(*args)
    torch.testing.assert_close(ya, yb, atol=0, rtol=0)
    ya.square().mean().backward()
    yb.square().mean().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_controls_reach_output():
    model = Generator(width=4).eval()
    image, hole, sketch, color, noise = inputs()
    with torch.no_grad():
        base = model(image, hole, sketch * 0, color * 0, noise)
        by_sketch = model(image, hole, sketch, color * 0, noise)
        by_color = model(image, hole, sketch * 0, color, noise)
    assert (base - by_sketch).abs().max() > 1e-5
    assert (base - by_color).abs().max() > 1e-5


def test_discriminators_and_gradient_isolation():
    model = Discriminators(width=8).eval().requires_grad_(False)
    image, hole, *_ = inputs(2)
    image.requires_grad_()
    outputs = model(image, hole, (50, 70))
    assert len(outputs) == 2 and all(len(x) == 2 for x in outputs)
    loss = hinge_generator(outputs)
    loss.backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_hinge_reduction():
    fake = [[torch.tensor([2., 3.]), torch.tensor([1.])], [torch.tensor([-2.])]]
    real = [[torch.tensor([2., 3.]), torch.tensor([1.])], [torch.tensor([2.])]]
    assert float(hinge_generator(fake)) == 0.25
    assert float(hinge_discriminator(real, fake)) == 2.75


@pytest.mark.parametrize("height,width", [(4, 7), (7, 4), (5, 5)])
def test_tv_matches_explicit_adjacent_pairs(height, width):
    image = torch.randn(2, 3, height, width, requires_grad=True)
    hole = torch.rand(2, 1, height, width) > .4
    expected = 0
    for b in range(2):
        for y in range(height):
            for x in range(width):
                if not hole[b, 0, y, x]:
                    continue
                if x + 1 < width and hole[b, 0, y, x + 1]:
                    expected = expected + (image[b, :, y, x + 1] - image[b, :, y, x]).abs().sum()
                if y + 1 < height and hole[b, 0, y + 1, x]:
                    expected = expected + (image[b, :, y + 1, x] - image[b, :, y, x]).abs().sum()
    torch.testing.assert_close(hole_tv(image, hole), expected / image.numel())


def test_gram_is_per_image_and_normalized():
    x = torch.ones(2, 3, 4, 5)
    x[1] *= 2
    g = gram(x)
    torch.testing.assert_close(g[0], torch.full((3, 3), 1 / 3))
    torch.testing.assert_close(g[1], torch.full((3, 3), 4 / 3))


def test_random_mask_bounds_and_variety():
    samples = [random_mask(256, 256, random.Random(i)) for i in range(30)]
    assert all(.1 <= float(x.float().mean()) <= .4 for x in samples)
    assert len({x.numpy().tobytes() for x in samples}) == len(samples)


def test_controls_are_sparse_aligned_and_include_no_control():
    hole = torch.zeros(1, 64, 64, dtype=torch.bool)
    hole[:, 10:54, 10:54] = True
    rtv = torch.rand(3, 64, 64)
    samples = [sample_controls(hole, rtv, random.Random(i)) for i in range(50)]
    assert {s["control_mode"] for s in samples} == {0, 1, 2, 3}
    for s in samples:
        assert not s["color_mask"].masked_select(~hole).any()
        assert not s["color"].masked_select(~s["color_mask"]).any()


def test_sampler_resume_is_exact():
    whole = list(EpochMicrobatches(103, 16, 4, 3, 2019))
    resumed = list(EpochMicrobatches(103, 16, 4, 3, 2019, start=32))
    assert resumed == whole[8:]
    flat = [i for chunk in whole for i, epoch in chunk]
    assert len(flat) == len(set(flat)) == 96


def test_dataset_reads_matching_images_and_flip(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (100, 80), (i * 90, 15, 20)).save(path)
        paths.append({"format": "image", "path": str(path)})
    manifest = tmp_path / "data.jsonl"
    manifest.write_text("\n".join(json.dumps(p) for p in paths))
    data = IndexedImages(str(manifest))
    assert [round(float(data[i]["image"][0].mean()) * 255) for i in range(3)] == [0, 90, 180]
    assert torch.equal(data[(1, 4)]["valid"], data[(1, 4)]["valid"])
    assert not torch.equal(data[(1, 4)]["valid"], data[(1, 5)]["valid"])


def test_rtv_cache_roundtrip_and_integrity(tmp_path):
    cache = RTVCache(str(tmp_path))
    rgb = np.ones((12, 12, 3), dtype=np.float32) * .3
    result = cache(rgb)
    np.testing.assert_allclose(result, rgb, atol=1e-7)
    assert np.array_equal(result, cache(rgb))
    next(tmp_path.rglob("*.npy")).write_bytes(b"bad")
    with pytest.raises(ValueError, match="mismatch"):
        cache(rgb)


def test_rng_capture_resume():
    state = capture_rng()
    expected = torch.randn(5), random.random(), np.random.rand()
    restore_rng(state)
    assert torch.equal(torch.randn(5), expected[0])
    assert random.random() == expected[1] and np.random.rand() == expected[2]


def test_config_rejects_invalid_batch():
    kwargs = {k: "x" for k in ("train_manifest", "validation_manifest", "train_sha256", "validation_sha256", "vgg_weights", "hed_weights", "vgg_sha256", "hed_sha256", "rtv_cache")}
    with pytest.raises(ValueError):
        Config(**kwargs, batch_size=7, microbatch_size=2)


def test_checkpoint_roundtrip(tmp_path):
    kwargs = {k: "x" for k in ("train_manifest", "validation_manifest", "train_sha256", "validation_sha256", "vgg_weights", "hed_weights", "vgg_sha256", "hed_sha256", "rtv_cache")}
    config = Config(**kwargs)
    g, d = nn.Linear(3, 2), nn.Linear(3, 2)
    og, od = torch.optim.Adam(g.parameters()), torch.optim.Adam(d.parameters())
    g(torch.ones(1, 3)).sum().backward()
    og.step()
    for step in (1, 2, 3):
        save_checkpoint(tmp_path, step, g, d, og, od, config, 2, 64 * step)
    assert len(list(tmp_path.glob("checkpoint-*.pt"))) == 2
    ck = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert ck["step"] == 3 and ck["cursor"] == 192
    assert torch.equal(ck["generator"]["weight"], g.weight)
    assert ck["optimizer_g"]["state"]
