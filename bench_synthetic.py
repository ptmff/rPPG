"""Compares the HR methods and the fusion on synthetic faces with a known pulse.

    python bench_synthetic.py                       # 45..170 bpm, both conditions
    python bench_synthetic.py --bpm 50 64 90 --weights weights/PURE_cross_RhythmMamba.pth

Each clip is a skin-coloured ellipse pulsing at the given rate (tests/test_pipeline.py);
POS and CHROM get the mean colour of the skin pixels of the same frames, RhythmMamba gets
the frames. Conditions:
  clean - as in the unit tests
  hard  - weaker pulse, more sensor noise, per-frame colour jitter of the whole image,
          illumination drift and a 0.25 Hz brightness modulation (breathing / head motion)
          with its 2nd harmonic
Methods: POS, CHROM, RhythmMamba at x1.0, RhythmMamba multi-scale (octave vote over the
scales), fusion (octave vote over POS, CHROM and all scales)."""
import argparse
import csv
import os

import numpy as np

from rppg import fusion
from rppg import signal as sig
from rppg.methods.chrom import CHROM
from rppg.methods.pos import POS
from tests.test_pipeline import PULSE_RGB, jittery_timestamps, pulse

HERE = os.path.dirname(os.path.abspath(__file__))


def clip(t, bpm, hard, seed, size=128):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    face = (((xx - size / 2) / (size * 0.33)) ** 2 + ((yy - size / 2) / (size * 0.42)) ** 2) < 1
    shade = 1 - 0.25 * ((xx - size / 2) / size) ** 2
    base = np.where(face[..., None], np.array([185, 135, 115]) * shade[..., None], 35.0)
    base = base + rng.normal(0, 3, base.shape)
    amp, noise = (0.002, 6.0) if hard else (0.006, 1.0)
    mod = 1 + amp * PULSE_RGB[None, :] * pulse(t, bpm)[:, None]
    light = np.ones_like(t)
    if hard:
        ph = 2 * np.pi * 0.25 * t
        light = 1 + 0.03 * t / t[-1] + 0.006 * np.sin(ph) + 0.004 * np.sin(2 * ph + 1.0)
    frames = base[None] * np.where(face[None, ..., None], mod[:, None, None, :], 1.0) * light[:, None, None, None]
    frames += rng.normal(0, noise, frames.shape)
    if hard:  # per-frame colour jitter common to the whole image (camera AE/WB, compression):
        frames += rng.normal(0, 0.2, (len(t), 1, 1, 3))  # does not average out over the skin
    frames = np.clip(frames, 0, 255)
    rgb = frames[:, face].mean(axis=1)
    return frames.astype(np.uint8), rgb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bpm", nargs="*", type=float, default=list(range(45, 171, 5)))
    p.add_argument("--conditions", nargs="*", default=["clean", "hard"])
    p.add_argument("--weights", default=os.path.join(HERE, "weights", "MMPD_intra_RhythmMamba.pth"))
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--out", default=os.path.join(HERE, "results", "synthetic.csv"))
    args = p.parse_args()

    from rppg.methods.rhythmmamba import RhythmMambaMethod
    rm = RhythmMambaMethod(args.weights)
    names = ["POS", "CHROM", "RM", "RM-multiscale", "Fusion"]
    rows = []
    for cond in args.conditions:
        for bpm in args.bpm:
            seed = int(bpm * 10) + (1000 if cond == "hard" else 0)
            t = jittery_timestamps(args.seconds, seed=seed)
            frames, rgb = clip(t, bpm, cond == "hard", seed)
            r = dict(condition=cond, bpm=bpm)
            for name, m in (("POS", POS()), ("CHROM", CHROM())):
                fs, bvp = m.bvp(t, rgb)
                r[name], r[name + "_snr"] = sig.estimate_hr(bvp, fs)[:2]
            ms = fusion.multiscale(rm, t, frames)
            r["RM"], r["RM_snr"] = next((h, s) for sc, h, s, *_ in ms if sc == 1.0)
            r["RM-multiscale"], r["RM-multiscale_snr"], _ = fusion.octave_vote([(h, s, "learned") for _, h, s, *_ in ms])
            r["Fusion"], r["Fusion_snr"], _ = fusion.octave_vote(
                [(r["POS"], r["POS_snr"], "classic"), (r["CHROM"], r["CHROM_snr"], "classic")]
                + [(h, s, "learned") for _, h, s, *_ in ms])
            r["scales"] = " ".join(f"x{sc}:{h:.1f}" for sc, h, *_ in ms)
            rows.append(r)
            print(f"{cond:5s} {bpm:5.0f} | " + "  ".join(f"{n} {r[n]:6.1f}" for n in names)
                  + f"  [{r['scales']}]", flush=True)

    print("\n| Condition | HR range | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for cond in args.conditions:
        for label, keep in (("all", lambda b: True), ("<70", lambda b: b < 70), (">=70", lambda b: b >= 70)):
            sub = [r for r in rows if r["condition"] == cond and keep(r["bpm"])]
            if not sub:
                continue
            cells = []
            for n in names:
                err = np.abs([r[n] - r["bpm"] for r in sub])
                cells.append(f"{err.mean():.1f} / {100 * np.mean(err <= 5):.0f}%")
            print(f"| {cond} | {label} ({len(sub)}) | " + " | ".join(cells) + " |")
    print("\ncells: MAE, bpm / share of clips within 5 bpm")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"rows: {args.out}")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
