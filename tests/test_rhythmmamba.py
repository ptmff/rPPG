"""RhythmMamba checkpoint + pure-PyTorch Mamba on synthetic data (needs torch)."""
import os

import pytest

from rppg import signal as sig
from tests.test_pipeline import WEIGHTS, jittery_timestamps, synthetic_face_clip

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("weights", [w for w in WEIGHTS if os.path.exists(w)])
def test_rhythmmamba_checkpoint_loads_and_runs(weights):
    from rppg.methods.rhythmmamba import load_model

    net = load_model(weights)
    n_params = sum(p.numel() for p in net.parameters())
    # The paper reports 1.07 M; the released checkpoints (depth 24, dim 96) have ~4.9 M.
    assert 4.5e6 < n_params < 5.5e6
    with torch.no_grad():
        out = net(torch.randn(1, 160, 3, 128, 128))
    assert out.shape == (1, 160)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("bpm", [60, 80, 110])
@pytest.mark.skipif(not os.path.exists(WEIGHTS[0]), reason="weights not downloaded")
def test_rhythmmamba_finds_pulse_on_synthetic_face(bpm):
    from rppg.methods.rhythmmamba import RhythmMambaMethod

    t = jittery_timestamps(10, seed=bpm)
    m = RhythmMambaMethod(WEIGHTS[0])
    fs, bvp = m.bvp(t, synthetic_face_clip(t, bpm, seed=bpm))
    hr, snr, _, _ = sig.estimate_hr(bvp, fs)
    assert abs(hr - bpm) < 3.0, (hr, snr)
