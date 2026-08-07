import math

import numpy as np
import pytest

from model.metrics import psnr_int8, ssim_int8


def test_psnr_identical_is_inf():
    a = np.arange(-128, 127, dtype=np.int64).reshape(1, 5, 51)
    assert psnr_int8(a, a) == float("inf")


def test_psnr_constant_offset_closed_form():
    # decoded difference is exactly 16/255 everywhere
    a = np.full((3, 8, 8), -128, dtype=np.int64)
    b = np.full((3, 8, 8), -112, dtype=np.int64)
    want = 20.0 * math.log10(255.0 / 16.0)
    assert psnr_int8(a, b) == pytest.approx(want, rel=1e-9)
    assert psnr_int8(b, a) == pytest.approx(want, rel=1e-9)  # symmetric


def test_ssim_self_is_one_and_noise_degrades():
    rng = np.random.default_rng(1058)
    a = rng.integers(-128, 128, size=(3, 32, 32)).astype(np.int64)
    assert ssim_int8(a, a) == pytest.approx(1.0, abs=1e-9)
    noise = rng.integers(-40, 41, size=a.shape)
    b = np.clip(a + noise, -128, 127)
    s = ssim_int8(a, b)
    assert s < 0.99
    assert s == pytest.approx(ssim_int8(b, a), abs=1e-12)


def test_ssim_smooth_images_score_higher_than_noisy():
    # structural similarity: a small constant shift barely hurts, noise hurts a lot
    yy = np.linspace(-100, 100, 32)
    a = np.clip(np.round(np.tile(yy, (32, 1))), -128, 127).astype(np.int64)[None]
    shifted = np.clip(a + 4, -128, 127)
    rng = np.random.default_rng(7)
    noisy = np.clip(a + rng.integers(-30, 31, size=a.shape), -128, 127)
    assert ssim_int8(a, shifted) > ssim_int8(a, noisy)
