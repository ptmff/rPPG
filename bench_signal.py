"""Synthetic check of the signal-level improvements: multi-region run with background and
head-motion denoising (rppg/multiregion.py) and heart-rate tracking (rppg/tracking.py).

    python bench_signal.py                 # both parts
    python bench_signal.py --part windows  # single 10 s windows, pulse 45..170
    python bench_signal.py --part tracks   # 2 min recordings with a changing pulse

Per-frame features are generated directly (no images): mean colour of the forehead and both
cheeks, of the background, and the face box, as FaceTracker.features() returns them.
Conditions:
  clean  - steady light, little noise
  hard   - as bench_synthetic.py: weaker pulse, sensor noise, per-frame colour jitter of the
           whole image (auto exposure / white balance), light drift, 0.25 Hz brightness change
  motion - hard plus head sway at the breathing rate with harmonics (talking, breathing after
           exercise): the skin brightness changes with the head position, differently per
           region, and one cheek is in shadow (weaker pulse, more noise)
"Before" is the previous pipeline: one mean colour over all skin, one POS / CHROM trace,
median of the last 5 estimates. "After" is the multi-region run with denoising and the
tracker. RhythmMamba is not involved: it gets the face crop, not these features."""
import argparse

import numpy as np

from rppg import fusion
from rppg import signal as sig
from rppg import tracking
from rppg.face import F_BG, F_POSE, N_FEATURES, REGIONS
from rppg.methods.chrom import CHROM
from rppg.methods.pos import POS
from tests.test_pipeline import PULSE_RGB, jittery_timestamps

SKIN = np.array([[175.0, 128, 110], [185, 135, 115], [180, 132, 112]])  # forehead, cheeks
NPIX = np.array([2500.0, 1500, 1500])


def pulse_wave(phase):
    return np.sin(phase) + 0.35 * np.sin(2 * phase - 0.8)


def features(t, bpm_of_t, cond, seed, resp_bpm=16.0):
    """(N, N_FEATURES, 3) features for timestamps t and a pulse rate bpm_of_t(t)."""
    rng = np.random.default_rng(seed)
    n = len(t)
    hard = cond != "clean"
    bpm = np.broadcast_to(bpm_of_t(t), t.shape)
    phase = 2 * np.pi * np.cumsum(np.r_[0, np.diff(t)] * bpm / 60)
    pls = pulse_wave(phase)
    amp = np.full(3, 0.002 if hard else 0.006)
    noise = np.full(3, 6.0 if hard else 1.0)
    light = np.ones(n)
    if hard:
        ph = 2 * np.pi * 0.25 * t
        light = 1 + 0.03 * t / t[-1] + 0.006 * np.sin(ph) + 0.004 * np.sin(2 * ph + 1.0)
    sway = np.zeros(n)
    if cond == "motion":
        amp[2] *= 0.4          # image-right cheek in shadow
        noise[2] *= 1.5
        rph = 2 * np.pi * resp_bpm / 60 * t + 0.3 * np.sin(2 * np.pi * 0.05 * t)
        sway = np.sin(rph) + 0.6 * np.sin(2 * rph + 0.5) + 0.45 * np.sin(3 * rph + 1.3) \
            + 0.3 * np.sin(4 * rph + 2.1) + 0.3 * np.convolve(rng.normal(0, 1, n), np.ones(9) / 9, "same")
    f = np.full((n, N_FEATURES, 3), np.nan)
    jitter = rng.normal(0, 0.2, (n, 3)) if hard else np.zeros((n, 3))
    shade_gain = np.array([0.002, 0.006, -0.005])  # brightness change per unit sway, per region
    for i in range(len(REGIONS)):
        mod = 1 + amp[i] * PULSE_RGB[None, :] * pls[:, None]
        f[:, i] = SKIN[i] * light[:, None] * mod * (1 + shade_gain[i] * sway)[:, None] + jitter \
            + rng.normal(0, noise[i] / np.sqrt(NPIX[i]), (n, 3))
    f[:, F_BG] = 60.0 * light[:, None] + jitter + rng.normal(0, noise[0] / np.sqrt(4000), (n, 3))
    w = 220.0
    f[:, F_POSE] = np.stack([320 + 4 * sway, 240 + 6 * sway, w + 0.5 * sway], 1) \
        + rng.normal(0, 0.3, (n, 3))
    return f


def mean_skin(feat):
    """The old input: pixel-weighted mean colour over all skin."""
    return (feat[:, :len(REGIONS)] * NPIX[None, :, None]).sum(axis=1) / NPIX.sum()


def part_windows(conditions, bpms):
    names = ["POS before", "POS after", "CHROM before", "CHROM after"]
    print("\nSingle 10 s windows: MAE, bpm / share of windows within 5 bpm")
    print("| Condition | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for cond in conditions:
        err = {n: [] for n in names}
        for bpm in bpms:
            seed = int(bpm * 10) + {"clean": 0, "hard": 1000, "motion": 2000}[cond]
            t = jittery_timestamps(10, seed=seed)
            feat = features(t, lambda tt: bpm, cond, seed)
            for m, label in ((POS(), "POS"), (CHROM(), "CHROM")):
                fs, b = m.trace(t, mean_skin(feat))
                err[label + " before"].append(sig.estimate_hr(b, fs)[0] - bpm)
                fs, b = m.bvp(t, feat)
                err[label + " after"].append(sig.estimate_hr(b, fs)[0] - bpm)
        cells = [f"{np.abs(e).mean():.1f} / {100 * np.mean(np.abs(e) <= 5):.0f}%" for e in map(np.array, err.values())]
        print(f"| {cond} | " + " | ".join(cells) + " |")


PROFILES = {
    # after exercise: 135 bpm falling to 85 with a time constant of 30 s
    "recovery": lambda t: 85 + 50 * np.exp(-t / 30.0),
    # rest with slow variation
    "rest": lambda t: 64 + 4 * np.sin(2 * np.pi * t / 40.0),
    # climbing: 75 to 125 over the recording
    "rise": lambda t: 75 + 50 * np.clip(t / 100.0, 0, 1),
}


def run_track(t, feat, profile, window=10.0, step=1.0, min_window=6.0):
    """Windows every step s like live.py. Returns rows of (t_end, true, {name: hr})."""
    methods = [POS(), CHROM()]
    med = {k: sig.HRSmoother() for k in ("POS", "Fused")}
    trk = {k: tracking.TrackedSmoother() for k in ("POS", "Fused")}
    rows = []
    for t_end in np.arange(t[0] + min_window, t[-1] + 1e-9, step):
        a, b = np.searchsorted(t, t_end - window), np.searchsorted(t, t_end, "right")
        tw, fw = t[a:b], feat[a:b]
        out = {}
        # before: mean colour, single trace, vote, median
        votes = []
        for m in methods:
            fs, bv = m.trace(tw, mean_skin(fw))
            hr, snr = sig.estimate_hr(bv, fs)[:2]
            votes.append((hr, snr, "classic"))
            if m.name == "POS":
                med["POS"].push(hr, snr, t_end)
        fhr, fsnr, _ = fusion.octave_vote(votes)
        med["Fused"].push(fhr, fsnr, t_end)
        out["POS before"], out["Fused before"] = med["POS"].value, med["Fused"].value
        # after: regions + denoising, vote, tracker on the spectrum
        resp = tracking.breathing_rate(tw, fw[:, F_POSE, 1])
        per, (fhr, fsnr, _) = fusion.run_window(methods, tw, fw, None)
        r = per["POS"]
        trk["POS"].push(r["hr"], r["snr"], t_end, f=r["f"], p=r["p"], resp_bpm=resp)
        trk["Fused"].push(fhr, fsnr, t_end)  # the vote is a single number, as in live.py
        out["POS after"], out["Fused after"] = trk["POS"].value, trk["Fused"].value
        rows.append((t_end, float(profile(t_end)), out))
    return rows


def part_tracks(conditions, window=10.0, seconds=120.0, seeds=(1, 2, 3)):
    names = ["POS before", "POS after", "Fused before", "Fused after"]
    print(f"\nRecordings of {seconds:.0f} s, estimate every 1 s from the last {window:.0f} s, error vs the pulse "
          "at the end of the window (what the user sees now): MAE, bpm / share within 5 bpm")
    print("| Condition | Pulse | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for cond in conditions:
        for pname, profile in PROFILES.items():
            err = {n: [] for n in names}
            for s in seeds:
                seed = s * 100 + {"clean": 0, "hard": 1000, "motion": 2000}[cond] + len(pname)
                t = jittery_timestamps(seconds, seed=seed)
                feat = features(t, profile, cond, seed, resp_bpm=14 + 2 * s)
                for t_end, true, out in run_track(t, feat, profile, window=window, min_window=min(6.0, window)):
                    for n in names:
                        err[n].append(out[n] - true if np.isfinite(out[n]) else np.nan)
            cells = []
            for n in names:
                e = np.array(err[n])
                shown = np.isfinite(e)
                cells.append(f"{np.abs(e[shown]).mean():.1f} / {100 * np.mean(np.abs(e[shown]) <= 5):.0f}%"
                             + ("" if shown.all() else f" ({100 * (1 - shown.mean()):.0f}% ---)"))
            print(f"| {cond} | {pname} | " + " | ".join(cells) + " |", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["windows", "tracks", "all"], default="all")
    p.add_argument("--conditions", nargs="*", default=["clean", "hard", "motion"])
    p.add_argument("--bpm", nargs="*", type=float, default=list(range(45, 171, 5)))
    p.add_argument("--window", type=float, default=10.0, help="analysis window of the tracks part, s")
    args = p.parse_args()
    if args.part in ("windows", "all"):
        part_windows(args.conditions, args.bpm)
    if args.part in ("tracks", "all"):
        part_tracks(args.conditions, args.window)


if __name__ == "__main__":
    main()
