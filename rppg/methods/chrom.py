"""CHROM: de Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG", IEEE TBME 2013.
No training; works on the mean skin colour over time, like POS."""
import numpy as np

from .. import signal as sig


class CHROM:
    name = "CHROM"
    needs = "rgb"

    def __init__(self, fs=30.0, window_s=10.0, sub_s=1.6):
        self.fs = fs
        self.window_s = window_s
        self.sub_s = sub_s

    def bvp(self, t, rgb):
        """t: (N,) timestamps in s, rgb: (N, 3) mean skin colour. Returns uniform-fs BVP."""
        _, c = sig.resample_uniform(t, rgb, self.fs)
        n = len(c)
        l = int(np.ceil(self.sub_s * self.fs))  # sub-windows, Hann-weighted and overlap-added
        # Normalise by the local mean, then band-pass X and Y over the whole window: filtering
        # each short piece separately cuts off pulses below ~60 bpm
        cn = c / (np.apply_along_axis(lambda v: np.convolve(v, np.ones(l) / l, "same"), 0, c) + 1e-9)
        x = sig.bandpass(3 * cn[:, 0] - 2 * cn[:, 1], self.fs)
        y = sig.bandpass(1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2], self.fs)
        hop = max(l // 2, 1)
        win = np.hanning(l)
        h = np.zeros(n)
        for s in range(0, n - l + 1, hop):
            xs, ys = x[s:s + l], y[s:s + l]
            hw = xs - (xs.std() / (ys.std() + 1e-9)) * ys
            h[s:s + l] += win * (hw - hw.mean())
        h = sig.detrend(h, 100)
        return self.fs, sig.bandpass(h, self.fs)
