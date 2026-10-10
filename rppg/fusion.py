"""Final heart-rate estimate from several methods and time scales.

Multi-scale run (WORKLOG 6.4): a learned model is fed the window resampled at fs*scale
while it still assumes fs, so it "sees" the pulse at hr/scale; the HR is then read on the
true clock. On synthetic faces the RhythmMamba checkpoints lock onto the 2nd harmonic
when the pulse they see is below ~55 bpm, with a high SNR, so picking the scale with the
best SNR picks the wrong answer. They never report half the pulse.

Hence an octave-aware vote: an estimate of a learned method that sits at twice another one
counts as support for the lower one. Votes are counted per family of methods ("classic":
POS, CHROM; "learned": the network at all scales), so the scales of one network cannot
outvote the classic methods: on weak synthetic signals the network collapses onto its
training range (70-90 bpm) at every scale, again with a high SNR. SNR is not comparable
between families and only breaks ties."""
import numpy as np

from . import signal as sig

SCALES = (0.75, 1.0, 1.25)


def multiscale(method, t, x, scales=SCALES):
    """Runs a learned method at several time scales. Returns [(scale, hr, snr, fs, bvp), ...]."""
    base, out = method.fs, []
    try:
        for s in scales:
            method.fs = base * s
            fs, bvp = method.bvp(t, x)
            hr, snr = sig.estimate_hr(bvp, fs)[:2]
            out.append((s, hr, snr, fs, bvp))
    finally:
        method.fs = base
    return out


def octave_vote(estimates, tol=0.04):
    """estimates: [(hr, snr, family), ...], family "classic" or "learned".
    Returns (hr, snr, support) with support = number of agreeing estimates.

    A candidate is supported by estimates within tol (relative) of it and by learned
    estimates at twice its value. Candidates are ranked by the number of families behind
    them, then by classic support, then by the number of estimates, then by summed SNR.
    The result is the mean of the supporting fundamentals and their best SNR."""
    est = [e for e in estimates if np.isfinite(e[0])]
    if not est:
        return float("nan"), float("-inf"), 0
    best = None
    for hc, _, _ in est:
        fund, snrs, fams = [], [], []
        for h, s, fam in est:
            if abs(h - hc) <= tol * hc:
                fund.append(h)
            elif fam == "learned" and abs(h - 2 * hc) <= 2 * tol * hc:
                fund.append(h / 2)
            else:
                continue
            snrs.append(s if np.isfinite(s) else 0.0)
            fams.append(fam)
        key = (len(set(fams)), fams.count("classic"), len(fund), sum(snrs))
        if best is None or key > best[0]:
            best = (key, float(np.mean(fund)), max(snrs), len(fund))
    _, hr, snr, support = best
    return hr, snr, support


def family(method):
    return "classic" if method.needs == "rgb" else "learned"


def run_window(methods, t, rgb, crop, scales=SCALES):
    """Runs every method on one window; learned ones at all scales (if more than one).

    Returns (per_method, fused): per_method[name] = dict(hr, snr, fs, bvp, f, p), where a
    learned method's hr/snr is its octave vote over the scales and fs/bvp/f/p are from
    scale 1.0; fused = (hr, snr, support) over all methods and scales (NaN with < 2 votes)."""
    per, votes = {}, []
    for m in methods:
        x = rgb if m.needs == "rgb" else crop
        if family(m) == "learned" and len(scales) > 1:
            runs = multiscale(m, t, x, scales)
            votes += [(h, s, "learned") for _, h, s, _, _ in runs]
            hr, snr, _ = octave_vote([(h, s, "learned") for _, h, s, _, _ in runs])
            _, _, _, fs, bvp = min(runs, key=lambda r: abs(r[0] - 1.0))
            _, _, f, p = sig.estimate_hr(bvp, fs)
        else:
            fs, bvp = m.bvp(t, x)
            hr, snr, f, p = sig.estimate_hr(bvp, fs)
            votes.append((hr, snr, family(m)))
        per[m.name] = dict(hr=hr, snr=snr, fs=fs, bvp=bvp, f=f, p=p)
    fused = octave_vote(votes) if len(votes) > 1 else (float("nan"), float("-inf"), 0)
    return per, fused
