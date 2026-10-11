"""Classic methods (POS, CHROM) on several skin regions with reference-channel denoising.

Input per frame is FaceTracker.features(): mean RGB of the forehead and both cheeks, mean
RGB of the background beside the face, and the face box (centre x, centre y, width).

1. Denoising. Changes of exposure, white balance and room light reach the background as well
   as the skin, head motion shows in the face box; neither carries the pulse. Within the
   window each region's normalised colour is regressed on these reference channels (fit on
   the pulse band, where the harm is done) and the fitted part is subtracted.
2. Regions. The method runs on each region separately and the BVPs are summed with weights
   from their SNR, so a shadow or a highlight on one cheek does not spoil the whole signal.
   A region missing in more than a fifth of the frames is skipped.
"""
import numpy as np

from . import signal as sig
from .face import F_BG, F_POSE, REGIONS

MAX_MISSING = 0.2


def _fill_nan(x):
    """Linear interpolation over NaN runs of each column; None if too many are missing."""
    x = np.array(x, dtype=np.float64)
    bad = ~np.isfinite(x).all(axis=1)
    if bad.mean() > MAX_MISSING or bad.all():
        return None
    if bad.any():
        idx = np.arange(len(x))
        for j in range(x.shape[1]):
            x[bad, j] = np.interp(idx[bad], idx[~bad], x[~bad, j])
    return x


def references(feat, fs):
    """Reference channels (n, k) from uniform-clock features: background colour normalised by
    its mean, face position and size relative to the face width. Missing ones are left out."""
    refs = []
    bg = _fill_nan(feat[:, F_BG])
    if bg is not None and (bg.mean(axis=0) > 1).all():
        refs.append(bg / bg.mean(axis=0) - 1)
    pose = _fill_nan(feat[:, F_POSE])
    if pose is not None:
        width = np.median(pose[:, 2])
        if width > 0:
            refs.append((pose - pose.mean(axis=0)) / width)
    return np.concatenate(refs, axis=1) if refs else None


def regress_out(rgb, refs, fs, ridge=1e-3):
    """Removes from each colour channel the part explained by the reference channels.
    rgb: (n, 3) uniform-clock colour, refs: (n, k). Coefficients are fitted on the band-passed
    signals (0.5-4 Hz) and the fitted part is subtracted from the full signals."""
    mean = rgb.mean(axis=0)
    c = rgb / mean - 1
    lo, hi = 0.5, min(4.0, 0.45 * fs)
    cb = np.stack([sig.bandpass(c[:, j], fs, lo, hi) for j in range(3)], 1)
    rb = np.stack([sig.bandpass(refs[:, j], fs, lo, hi) for j in range(refs.shape[1])], 1)
    scale = rb.std(axis=0) + 1e-12
    a = rb / scale
    beta = np.linalg.solve(a.T @ a + ridge * len(a) * np.eye(a.shape[1]), a.T @ cb)
    return (1 + c - ((refs - refs.mean(axis=0)) / scale) @ beta) * mean


def bvp(core, fs, t, feat, denoise=True):
    """Runs core(t, rgb) -> (fs, bvp) on every region of feat (N, N_FEATURES, 3) and combines.
    Returns (fs, bvp, info) with info = per-region dicts (snr, weight)."""
    tu, f = sig.resample_uniform(t, feat, fs)
    refs = references(f, fs) if denoise else None
    bvps, info = [], []
    for i, name in enumerate(REGIONS):
        rgb = _fill_nan(f[:, i])
        if rgb is None:
            info.append(dict(region=name, snr=float("nan"), weight=0.0))
            continue
        if refs is not None:
            rgb = regress_out(rgb, refs, fs)
        fs_out, b = core(tu, rgb)
        snr = sig.estimate_hr(b, fs_out)[1]
        bvps.append((b / (b.std() + 1e-12), snr))
        info.append(dict(region=name, snr=snr, weight=None))
    if not bvps:
        return fs, np.zeros(len(tu)), info
    # Weight = linear signal-to-noise ratio of the region; regions far below the best one
    # contribute almost nothing
    snrs = np.array([s if np.isfinite(s) else -30.0 for _, s in bvps])
    w = 10 ** ((snrs - snrs.max()) / 10)
    w /= w.sum()
    it = iter(w)
    for r in info:
        if r["weight"] is None:
            r["weight"] = float(next(it))
    out = sum(wi * b for wi, (b, _) in zip(w, bvps))
    return fs, out, info
