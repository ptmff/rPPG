"""Heart-rate tracking over time instead of one spectral peak per window.

HRTracker is a forward filter of a hidden Markov model on a grid of heart rates (the online
form of Viterbi tracking over a spectrogram). Between updates the belief spreads by a
Gaussian whose width grows with the elapsed time (the pulse cannot change by more than a
few bpm per second), plus a small uniform part so the tracker can recover from a wrong
lock. Each window's spectrum, sharpened and normalised, is the likelihood. The output is the
most probable heart rate.

Optionally harmonics of the breathing rate are attenuated in the spectrum (resp_harmonics).
Off by default: on the synthetic benchmark (bench_signal.py) it did not help, with or
without denoising, because a harmonic of breathing often falls on the true pulse; to be
re-checked on real recordings.

Against the median of recent estimates (signal.HRSmoother) this
- uses the whole spectrum, so a second-strongest peak near the track wins over a jump;
- follows a changing pulse without the extra lag of the median;
- weighs noisy windows less by itself: a flat spectrum is a flat likelihood.
"""
import numpy as np

from . import signal as sig


class HRTracker:
    def __init__(self, lo_bpm=40.0, hi_bpm=180.0, step_bpm=0.5, max_rate=3.0, floor=0.02,
                 sharpness=2.0, point_sigma=3.0, resp_harmonics=(), resp_atten=0.5,
                 max_age=5.0, min_conf=0.0):
        """max_rate: s.d. of the pulse change, bpm per second; floor: share of the belief
        spread uniformly at each step; sharpness: power applied to the normalised spectrum;
        point_sigma: width (bpm) of the likelihood built from a single estimate;
        max_age: s without updates after which the track is dropped; min_conf: below this
        posterior mass around the peak, value is NaN."""
        self.grid = np.arange(lo_bpm, hi_bpm + step_bpm / 2, step_bpm)
        self.step = step_bpm
        self.max_rate = max_rate
        self.floor = floor
        self.sharpness = sharpness
        self.point_sigma = point_sigma
        self.resp_harmonics = resp_harmonics
        self.resp_atten = resp_atten
        self.max_age = max_age
        self.min_conf = min_conf
        self.reset()

    def reset(self):
        self.belief = None
        self.t = None
        self.conf = 0.0

    def _predict(self, t):
        if self.belief is None or self.t is None or t - self.t > self.max_age:
            return np.full(len(self.grid), 1.0 / len(self.grid))
        dt = max(t - self.t, 1e-3)
        sigma = max(self.max_rate * dt / self.step, 0.5)
        k = np.arange(-int(4 * sigma) - 1, int(4 * sigma) + 2)
        kern = np.exp(-0.5 * (k / sigma) ** 2)
        b = np.convolve(self.belief, kern / kern.sum(), "same")
        b = (1 - self.floor) * b / b.sum() + self.floor / len(b)
        return b

    def spectrum_likelihood(self, f, p, resp_bpm=None):
        """Likelihood on the grid from a power spectrum (f in Hz)."""
        lik = np.interp(self.grid / 60.0, f, p, left=0.0, right=0.0)
        if resp_bpm is not None and np.isfinite(resp_bpm) and resp_bpm > 0:
            for k in self.resp_harmonics:
                lik *= 1 - (1 - self.resp_atten) * np.exp(-0.5 * ((self.grid - k * resp_bpm) / 2.0) ** 2)
        m = lik.max()
        if not np.isfinite(m) or m <= 0:
            return np.ones(len(self.grid))
        return (lik / m) ** self.sharpness + 1e-3

    def point_likelihood(self, hr):
        if not np.isfinite(hr):
            return np.ones(len(self.grid))
        return 0.2 + np.exp(-0.5 * ((self.grid - hr) / self.point_sigma) ** 2)

    def update(self, t, hr=None, f=None, p=None, resp_bpm=None):
        """New window ending at time t: either its spectrum (f, p) or a single estimate hr.
        Returns the tracked heart rate."""
        lik = self.spectrum_likelihood(f, p, resp_bpm) if f is not None else self.point_likelihood(hr)
        b = self._predict(t) * lik
        self.belief = b / b.sum()
        self.t = t
        return self.value

    @property
    def value(self):
        if self.belief is None:
            return float("nan")
        k = int(np.argmax(self.belief))
        near = np.abs(self.grid - self.grid[k]) <= 3.0
        self.conf = float(self.belief[near].sum())
        if self.conf < self.min_conf:
            return float("nan")
        if 0 < k < len(self.grid) - 1:  # parabolic refinement between grid points
            a, c, e = np.log(self.belief[k - 1:k + 2] + 1e-30)
            d = a - 2 * c + e
            return float(self.grid[k] + (0.5 * (a - e) / d if d != 0 else 0.0) * self.step)
        return float(self.grid[k])


class TrackedSmoother:
    """Drop-in for signal.HRSmoother in live.py: push() takes the window's spectrum when there
    is one, else the single estimate. Estimates older than max_age are not shown."""

    def __init__(self, min_snr_db=-10.0, **kw):
        self.tracker = HRTracker(**kw)
        self.min_snr = min_snr_db
        self.t_last = None

    def push(self, hr, snr, t, f=None, p=None, resp_bpm=None):
        if not np.isfinite(hr) or snr < self.min_snr:
            if self.t_last is not None and t - self.t_last > self.tracker.max_age:
                self.reset()  # better show nothing than a stale number
            return False
        self.tracker.update(t, hr=hr, f=f, p=p, resp_bpm=resp_bpm)
        self.t_last = t
        return True

    def reset(self):
        self.tracker.reset()
        self.t_last = None

    def value_at(self, t):
        if self.t_last is None or t - self.t_last > self.tracker.max_age:
            return float("nan")
        return self.tracker.value

    @property
    def value(self):
        return self.tracker.value if self.t_last is not None else float("nan")


def breathing_rate(t, y, fs=30.0, lo_hz=0.1, hi_hz=0.7):
    """Breathing rate (breaths/min) from a slow signal such as the vertical head position."""
    _, y = sig.resample_uniform(t, np.asarray(y, dtype=np.float64), fs)
    if len(y) < 3 * fs:
        return float("nan")
    y = sig.detrend(y - y.mean(), 300)
    f, p = sig.spectrum(y, fs)
    band = (f >= lo_hz) & (f <= hi_hz)
    return 60 * f[band][np.argmax(p[band])] if band.any() and p[band].sum() > 0 else float("nan")
