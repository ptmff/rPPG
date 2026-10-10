"""POS: Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 2017.
No training; works on the mean skin colour over time."""
import numpy as np

from .. import signal as sig


class POS:
    name = "POS"
    needs = "rgb"

    def __init__(self, fs=30.0, window_s=10.0):
        self.fs = fs
        self.window_s = window_s

    def bvp(self, t, rgb):
        """t: (N,) timestamps in s, rgb: (N, 3) mean skin colour. Returns uniform-fs BVP."""
        _, c = sig.resample_uniform(t, rgb, self.fs)
        n = len(c)
        l = int(np.ceil(1.6 * self.fs))  # 1.6 s sub-windows, overlap-added
        h = np.zeros(n)
        proj = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
        for m in range(l, n + 1):
            cw = c[m - l:m].T
            cn = cw / (cw.mean(axis=1, keepdims=True) + 1e-9)
            s = proj @ cn
            hw = s[0] + (s[0].std() / (s[1].std() + 1e-9)) * s[1]
            h[m - l:m] += hw - hw.mean()
        h = sig.detrend(h, 100)
        return self.fs, sig.bandpass(h, self.fs)
