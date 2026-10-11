"""Multi-region run with denoising (rppg/multiregion.py), skin pixel selection (rppg/face.py)
and HR tracking (rppg/tracking.py). No webcam, no torch."""
import numpy as np
import pytest

from bench_signal import features, mean_skin
from rppg import multiregion
from rppg import signal as sig
from rppg import tracking
from rppg.face import F_BG, skin_pixels
from rppg.methods.chrom import CHROM
from rppg.methods.pos import POS
from tests.test_pipeline import jittery_timestamps


@pytest.mark.parametrize("bpm", [55, 80, 125])
@pytest.mark.parametrize("method", [POS, CHROM])
def test_denoising_recovers_pulse_under_camera_jitter_and_motion(method, bpm):
    t = jittery_timestamps(10, seed=bpm)
    feat = features(t, lambda tt: bpm, "motion", seed=bpm)
    fs, b = method().bvp(t, feat)
    hr = sig.estimate_hr(b, fs)[0]
    assert abs(hr - bpm) < 3, hr


def test_denoising_beats_single_trace_on_hard_conditions():
    err_old, err_new = [], []
    for bpm in range(50, 151, 10):
        t = jittery_timestamps(10, seed=bpm + 7)
        feat = features(t, lambda tt: bpm, "motion", seed=bpm + 7)
        fs, b = POS().trace(t, mean_skin(feat))
        err_old.append(abs(sig.estimate_hr(b, fs)[0] - bpm))
        fs, b = POS().bvp(t, feat)
        err_new.append(abs(sig.estimate_hr(b, fs)[0] - bpm))
    assert np.mean(err_new) < 0.5 * np.mean(err_old), (np.mean(err_new), np.mean(err_old))


def test_region_weights_ignore_a_region_without_pulse():
    bpm = 90
    t = jittery_timestamps(10, seed=3)
    feat = features(t, lambda tt: bpm, "clean", seed=3)
    rng = np.random.default_rng(0)
    feat[:, 2] = feat[:, 2].mean(axis=0) + rng.normal(0, 0.3, (len(t), 3))  # pure noise
    fs, b, info = multiregion.bvp(POS().trace, 30.0, t, feat)
    assert abs(sig.estimate_hr(b, fs)[0] - bpm) < 2
    assert info[2]["weight"] < 0.2 * max(info[0]["weight"], info[1]["weight"])


def test_missing_region_and_background_are_tolerated():
    bpm = 72
    t = jittery_timestamps(10, seed=4)
    feat = features(t, lambda tt: bpm, "clean", seed=4)
    feat[:, 1] = np.nan          # cheek never visible
    feat[::3, F_BG] = np.nan     # background a third of the time: dropped as a reference
    fs, b, info = multiregion.bvp(POS().trace, 30.0, t, feat)
    assert info[1]["weight"] == 0.0
    assert abs(sig.estimate_hr(b, fs)[0] - bpm) < 2


def test_skin_pixels_drop_brows_and_highlights():
    rng = np.random.default_rng(0)
    skin = np.array([110, 135, 185]) + rng.normal(0, 3, (40, 40, 3))   # BGR
    skin[:8] = (40, 45, 50)        # brow
    skin[-4:, :10] = (245, 245, 245)  # highlight (not clipped)
    px = skin_pixels(np.clip(skin, 0, 255).astype(np.uint8))
    assert len(px) == pytest.approx(40 * 40 - 8 * 40 - 4 * 10, rel=0.03)
    assert np.allclose(px.mean(axis=0), (110, 135, 185), atol=1.5)


def _spectrum_with_peaks(peaks, fs=30.0, n=300, seed=0):
    rng = np.random.default_rng(seed)
    tt = np.arange(n) / fs
    x = sum(a * np.sin(2 * np.pi * bpm / 60 * tt) for bpm, a in peaks) + rng.normal(0, 0.3, n)
    f, p = sig.spectrum(x, fs)
    return f, p


def test_tracker_ignores_a_single_wrong_window():
    trk = tracking.HRTracker()
    for i in range(10):
        trk.update(float(i), f=_spectrum_with_peaks([(80, 1.0), (110, 0.4)], seed=i)[0],
                   p=_spectrum_with_peaks([(80, 1.0), (110, 0.4)], seed=i)[1])
    f, p = _spectrum_with_peaks([(80, 0.7), (110, 1.0)], seed=99)  # the wrong peak wins once
    assert abs(trk.update(10.0, f=f, p=p) - 80) < 1.5


def test_tracker_follows_a_falling_pulse():
    trk = tracking.HRTracker()
    for i in range(40):
        bpm = 130 - 1.5 * i
        f, p = _spectrum_with_peaks([(bpm, 1.0), (75, 0.5)], seed=i)
        hr = trk.update(float(i), f=f, p=p)
    assert abs(hr - (130 - 1.5 * 39)) < 2


def test_tracked_smoother_drops_stale_track():
    sm = tracking.TrackedSmoother()
    assert sm.push(80.0, 5.0, 0.0)
    assert abs(sm.value - 80) < 1
    assert not sm.push(80.0, -20.0, 10.0)   # rejected and the last update is 10 s old
    assert np.isnan(sm.value)
