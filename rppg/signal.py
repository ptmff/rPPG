"""Signal processing shared by all methods: resampling onto a uniform clock,
band-pass filtering, spectral heart-rate estimate and signal quality (SNR)."""
import numpy as np
from scipy import signal as sps

# Pulse band: 42-180 bpm
F_LO, F_HI = 0.7, 3.0


def resample_uniform(t, values, fs):
    """Linearly resamples samples taken at irregular times t onto a uniform fs grid
    ending at t[-1]. values: (N, ...) array. Returns (t_uniform, resampled)."""
    t = np.asarray(t, dtype=np.float64)
    n = int(np.floor((t[-1] - t[0]) * fs)) + 1
    tu = t[-1] - np.arange(n)[::-1] / fs
    idx = np.clip(np.searchsorted(t, tu, side="right") - 1, 0, len(t) - 2)
    w = ((tu - t[idx]) / np.maximum(t[idx + 1] - t[idx], 1e-9)).clip(0, 1)
    values = np.asarray(values)
    shape = (-1,) + (1,) * (values.ndim - 1)
    out = values[idx] * (1 - w).reshape(shape) + values[idx + 1] * w.reshape(shape)
    return tu, out


def detrend(x, lam=100):
    """Smoothness-priors detrending (Tarvainen et al.), as used in rPPG-Toolbox."""
    from scipy.sparse import diags, identity
    from scipy.sparse.linalg import spsolve

    n = len(x)
    if n < 3:
        return x - np.mean(x)
    d2 = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    trend = spsolve((identity(n) + lam ** 2 * (d2.T @ d2)).tocsc(), x)
    return x - trend


def bandpass(x, fs, lo=F_LO, hi=F_HI, order=2):
    b, a = sps.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="bandpass")
    return sps.filtfilt(b, a, x)


def spectrum(x, fs, nfft_min=2048):
    """Hann-windowed periodogram, zero-padded for a fine frequency grid."""
    nfft = max(nfft_min, 1 << int(np.ceil(np.log2(len(x)))))
    f, p = sps.periodogram(x, fs=fs, window="hann", nfft=nfft, detrend=False)
    return f, p


def estimate_hr(bvp, fs, lo=F_LO, hi=F_HI):
    """Returns (hr_bpm, snr_db, freqs, power) of a blood-volume-pulse signal.

    SNR follows de Haan & Jeanne (2013) / rPPG-Toolbox: power around the peak and its
    first harmonic versus all other power in the pulse band."""
    f, p = spectrum(bvp, fs)
    band = (f >= lo) & (f <= hi)
    if not band.any() or p[band].sum() <= 0:
        return float("nan"), float("-inf"), f, p
    k = np.flatnonzero(band)[np.argmax(p[band])]
    # Parabolic interpolation between FFT bins
    if 0 < k < len(p) - 1:
        a, b, c = np.log(p[k - 1:k + 2] + 1e-20)
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        shift = 0.0
    f_hr = f[k] + shift * (f[1] - f[0])
    width = 0.1  # Hz, i.e. +-6 bpm
    near = (np.abs(f - f_hr) <= width) | (np.abs(f - 2 * f_hr) <= width)
    wide = (f >= lo) & (f <= max(hi, 2 * f_hr + width))
    sig = p[near & wide].sum()
    noise = p[wide & ~near].sum()
    snr = 10 * np.log10(sig / noise) if noise > 0 else float("inf")
    return 60.0 * f_hr, snr, f, p


class HRSmoother:
    """Median of recent good estimates, so the display does not jump. Estimates older than
    max_age seconds are dropped: better show nothing than a stale number."""

    def __init__(self, size=5, min_snr_db=-3.0, max_age=5.0):
        self.size = size
        self.min_snr = min_snr_db
        self.max_age = max_age
        self.values = []  # (t, hr)

    def push(self, hr, snr, t):
        self.values = [(tv, v) for tv, v in self.values if t - tv <= self.max_age]
        if np.isfinite(hr) and snr >= self.min_snr:
            self.values = (self.values + [(t, hr)])[-self.size:]
            return True
        return False

    def reset(self):
        self.values = []

    @property
    def value(self):
        return float(np.median([v for _, v in self.values])) if self.values else float("nan")
