"""RhythmMamba (Zou et al., AAAI 2025) with the authors' pretrained checkpoints.

Preprocessing mirrors the rPPG-Toolbox configs the checkpoints were trained with:
RGB face crop, large box x1.5, 128x128, "Standardized" (zero mean / unit std over the clip),
30 fps. Post-processing mirrors the toolbox too: detrend + band-pass on the output."""
import numpy as np
import torch

from .. import signal as sig
from ..models.rhythmmamba import RhythmMamba as _Net


def load_model(weights_path, device="cpu"):
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    net = _Net()
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


class RhythmMambaMethod:
    name = "RhythmMamba"
    needs = "crop"

    def __init__(self, weights_path, fs=30.0, window_s=10.0, device="cpu"):
        self.fs = fs
        self.window_s = window_s
        self.device = device
        self.net = load_model(weights_path, device)

    @torch.no_grad()
    def raw_output(self, frames):
        """frames: (D, H, W, 3) float RGB at self.fs. Returns the network output (D,)."""
        x = frames.astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        d = len(x) - len(x) % 2  # the temporal stem has stride 2
        x = torch.from_numpy(np.ascontiguousarray(x[-d:].transpose(0, 3, 1, 2)))[None]
        return self.net(x.to(self.device))[0].float().cpu().numpy()

    def bvp(self, t, crops):
        """t: (N,) timestamps, crops: (N, 128, 128, 3) uint8 RGB. Returns uniform-fs BVP."""
        _, frames = sig.resample_uniform(t, crops.astype(np.float32), self.fs)
        y = self.raw_output(frames).astype(np.float64)
        y = sig.detrend(y, 100)
        return self.fs, sig.bandpass(y, self.fs)
