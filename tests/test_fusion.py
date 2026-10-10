"""CHROM and the octave-aware vote (no torch needed)."""
import numpy as np
import pytest

from rppg import fusion
from rppg import signal as sig
from rppg.methods.chrom import CHROM
from tests.test_pipeline import PULSE_RGB, jittery_timestamps, pulse


# Below ~48 bpm CHROM's 1.6 s sub-windows hold barely one beat and it may jump to the harmonic
@pytest.mark.parametrize("bpm", [52, 72, 95, 130])
def test_chrom_on_synthetic_skin_trace(bpm):
    rng = np.random.default_rng(bpm)
    t = jittery_timestamps(10, seed=bpm)
    light = 1 + 0.02 * np.sin(2 * np.pi * 0.2 * t)
    rgb = np.array([170.0, 120.0, 100.0]) * light[:, None] * (1 + 0.004 * PULSE_RGB * pulse(t, bpm)[:, None])
    rgb += rng.normal(0, 0.15, rgb.shape)
    fs, bvp = CHROM().bvp(t, rgb)
    hr, snr, _, _ = sig.estimate_hr(bvp, fs)
    assert abs(hr - bpm) < 2.0, (hr, snr)


def test_octave_vote_prefers_fundamental_over_confident_harmonic():
    # RhythmMamba at x1.0 and x1.25 locked onto the 2nd harmonic of a 45 bpm pulse
    hr, _, support = fusion.octave_vote([(44.8, 6.3, "learned"), (89.5, 15.5, "learned"), (89.7, 16.3, "learned")])
    assert abs(hr - 44.8) < 1 and support == 3


def test_octave_vote_keeps_agreeing_estimates():
    hr, _, support = fusion.octave_vote([(99.9, 18.0, "learned"), (100.2, 17.2, "learned"), (56.0, 3.0, "learned")])
    assert abs(hr - 100) < 1 and support == 2


def test_octave_vote_network_scales_cannot_outvote_classic():
    # Weak signal: the network sits at ~75 bpm at every scale with a high SNR, POS and CHROM
    # agree on the true 130 bpm with a negative SNR
    hr, _, _ = fusion.octave_vote([(130.6, -2.9, "classic"), (131.6, -3.7, "classic"),
                                   (74.0, 9.0, "learned"), (73.1, 9.3, "learned"), (75.5, 8.0, "learned")])
    assert abs(hr - 131) < 1.5


def test_octave_vote_empty():
    hr, _, support = fusion.octave_vote([(float("nan"), 0.0, "classic")])
    assert np.isnan(hr) and support == 0


class _ScaleEcho:
    """Stands in for a learned method: reports fs*scale-dependent HR like RhythmMamba does."""
    fs = 30.0

    def bvp(self, t, x):
        tt = np.arange(300) / self.fs
        return self.fs, np.sin(2 * np.pi * 1.0 * tt * 30.0 / self.fs)


def test_multiscale_restores_fs():
    m = _ScaleEcho()
    out = fusion.multiscale(m, None, None)
    assert m.fs == 30.0
    assert [s for s, _, _ in out] == list(fusion.SCALES)
