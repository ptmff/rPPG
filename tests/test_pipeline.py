"""Offline checks on synthetic data (no webcam needed):  python -m pytest tests -q"""
import os

import numpy as np
import pytest

from rppg import signal as sig
from rppg.methods.pos import POS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = [os.path.join(ROOT, "weights", f) for f in
           ("MMPD_intra_RhythmMamba.pth", "PURE_cross_RhythmMamba.pth", "UBFC_cross_RhythmMamba.pth")]
# Relative pulse amplitude per channel (green strongest), roughly as in real skin
PULSE_RGB = np.array([0.33, 0.77, 0.53])


def jittery_timestamps(seconds, fps=30.0, seed=0):
    """Webcam-like clock: nominal fps with jitter and a few dropped frames."""
    rng = np.random.default_rng(seed)
    dt = 1 / fps + rng.normal(0, 0.004, int(seconds * fps))
    dt[rng.random(len(dt)) < 0.03] += 1 / fps
    return np.cumsum(np.clip(dt, 0.01, None))


def pulse(t, bpm):
    ph = 2 * np.pi * bpm / 60 * t
    return np.sin(ph) + 0.35 * np.sin(2 * ph - 0.8)  # fundamental + dicrotic harmonic


def test_resample_uniform_recovers_sine():
    t = jittery_timestamps(10)
    tu, y = sig.resample_uniform(t, np.sin(2 * np.pi * 1.2 * t), 30.0)
    assert np.allclose(np.diff(tu), 1 / 30)
    assert np.max(np.abs(y - np.sin(2 * np.pi * 1.2 * tu))) < 0.05


@pytest.mark.parametrize("bpm", [52, 72, 95, 130])
def test_pos_on_synthetic_skin_trace(bpm):
    rng = np.random.default_rng(bpm)
    t = jittery_timestamps(10, seed=bpm)
    base = np.array([170.0, 120.0, 100.0])
    light = 1 + 0.02 * np.sin(2 * np.pi * 0.2 * t) + 0.01 * rng.standard_normal(1).item() * t / t[-1]
    rgb = base * light[:, None] * (1 + 0.004 * PULSE_RGB * pulse(t, bpm)[:, None])
    rgb += rng.normal(0, 0.15, rgb.shape)
    fs, bvp = POS().bvp(t, rgb)
    hr, snr, _, _ = sig.estimate_hr(bvp, fs)
    assert abs(hr - bpm) < 2.0, (hr, snr)
    assert snr > 0


def test_estimate_hr_interpolates_between_bins():
    fs = 30.0
    t = np.arange(300) / fs
    hr, snr, _, _ = sig.estimate_hr(np.sin(2 * np.pi * 1.2345 * t), fs)
    assert abs(hr - 60 * 1.2345) < 0.3
    assert snr > 10


def synthetic_face_clip(t, bpm, seed=0, size=128):
    """Skin-coloured ellipse with texture on a dark background; skin pulses with bpm."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    face = (((xx - size / 2) / (size * 0.33)) ** 2 + ((yy - size / 2) / (size * 0.42)) ** 2) < 1
    shade = 1 - 0.25 * ((xx - size / 2) / size) ** 2
    base = np.where(face[..., None], np.array([185, 135, 115]) * shade[..., None], 35.0)
    base = base + rng.normal(0, 3, base.shape)
    mod = 1 + 0.006 * PULSE_RGB[None, :] * pulse(t, bpm)[:, None]  # (N, 3)
    frames = base[None] * np.where(face[None, ..., None], mod[:, None, None, :], 1.0)
    frames += rng.normal(0, 1.0, frames.shape)
    return np.clip(frames, 0, 255).astype(np.uint8)
