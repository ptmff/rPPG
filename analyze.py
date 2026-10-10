"""Offline diagnosis of a recording made with `live.py --record rec.npz`.

    python analyze.py rec.npz
    python analyze.py rec.npz --ref 121,110,91,98,91 --ref-start 5 --ref-step 10

For every 1 s step it re-runs the methods on the recorded frames and prints the estimate,
the three strongest spectral peaks, frame rate, head motion and breathing rate (from the
head's up/down motion). With reference values it also tells, per method:
  MAE          - error of the reported HR
  ref in top-3 - how often the true HR is at least one of the three strongest peaks:
                 if high, better peak selection (fusion / tracking) can fix the errors;
                 if low, the signal is not there and only a better model / data helps.
"""
import argparse
import os

import numpy as np

from rppg import signal as sig
from rppg.methods.pos import POS

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording")
    p.add_argument("--ref", help="reference bpm values, comma separated (instead of those typed in live.py)")
    p.add_argument("--ref-start", type=float, default=0.0, help="time of the first --ref value, s from recording start")
    p.add_argument("--ref-step", type=float, default=10.0, help="interval between --ref values, s")
    p.add_argument("--weights", nargs="*", default=[os.path.join(HERE, "weights", "MMPD_intra_RhythmMamba.pth")],
                   help="RhythmMamba checkpoints to evaluate (none = POS only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--window", type=float, default=10.0)
    p.add_argument("--min-window", type=float, default=6.0)
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--csv", help="also write the per-window table to this CSV")
    return p.parse_args()


def top_peaks(f, p, n=3, lo=sig.F_LO, hi=sig.F_HI, min_sep_hz=0.1):
    band = (f >= lo) & (f <= hi)
    fb, pb = f[band], p[band]
    local_max = np.r_[False, (pb[1:-1] > pb[:-2]) & (pb[1:-1] >= pb[2:]), False]
    peaks = []
    for i in np.flatnonzero(local_max)[np.argsort(pb[local_max])[::-1]]:
        if all(abs(fb[i] - q) > min_sep_hz for q in peaks):
            peaks.append(fb[i])
        if len(peaks) == n:
            break
    return [60 * q for q in peaks]


def breathing(t, face, fs=30.0):
    """Breathing rate (breaths/min) from the vertical head motion, 6-42 /min."""
    y = face[:, 1] + face[:, 3] / 2
    _, y = sig.resample_uniform(t, y, fs)
    y = sig.detrend(y - y.mean(), 300)
    f, p = sig.spectrum(y, fs)
    band = (f >= 0.1) & (f <= 0.7)
    return 60 * f[band][np.argmax(p[band])] if band.any() and p[band].sum() > 0 else float("nan")


def load_refs(args, data, t0):
    if args.ref:
        hr = np.array([float(v) for v in args.ref.split(",")])
        return args.ref_start + args.ref_step * np.arange(len(hr)), hr
    if "ref_t" in data and len(data["ref_t"]):
        return data["ref_t"] - t0, data["ref_hr"].astype(float)
    return np.array([]), np.array([])


def ref_at(rt, rh, t, tol=5.0):
    """Linear interpolation between reference readings; nothing more than tol s outside them."""
    if len(rt) == 0 or t < rt[0] - tol or t > rt[-1] + tol:
        return float("nan")
    return float(np.interp(t, rt, rh))


def main():
    args = parse_args()
    data = np.load(args.recording)
    t_all = data["t"]
    t0 = t_all[0]
    t_all = t_all - t0
    rgb_all, crop_all = data["rgb"], data["crop"]
    face_all = data["face"] if "face" in data else None
    rt, rh = load_refs(args, data, t0)

    methods = [POS(window_s=args.window)]
    if args.weights:
        import torch
        from rppg.methods.rhythmmamba import RhythmMambaMethod
        dev = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
        for w in args.weights:
            m = RhythmMambaMethod(w, window_s=args.window, device=dev)
            m.name = "RM:" + os.path.basename(w).split("_RhythmMamba")[0]
            methods.append(m)

    dt = np.diff(t_all)
    print(f"{len(t_all)} frames, {t_all[-1]:.1f} s, median fps {1 / np.median(dt):.1f}, "
          f"frame gaps > 0.2 s: {(dt > 0.2).sum()}, face lost (gaps > 0.5 s): {(dt > 0.5).sum()}")
    if len(rt):
        print("reference: " + ", ".join(f"{h:.0f}@{t:.0f}s" for t, h in zip(rt, rh)))

    # Continuous segments (the recording has holes where the face was lost)
    cuts = np.r_[0, np.flatnonzero(dt > 0.5) + 1, len(t_all)]
    rows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        t_seg = t_all[a:b]
        if len(t_seg) < 2 or t_seg[-1] - t_seg[0] < args.min_window:
            continue
        for t_end in np.arange(t_seg[0] + args.min_window, t_seg[-1] + 1e-9, args.step):
            sel = slice(a + np.searchsorted(t_seg, t_end - args.window), a + np.searchsorted(t_seg, t_end, "right"))
            t = t_all[sel]
            row = dict(t=t_end, fps=(len(t) - 1) / (t[-1] - t[0]), ref=ref_at(rt, rh, t_end))
            if face_all is not None:
                face = face_all[sel]
                c = face[:, :2] + face[:, 2:4] / 2
                row["motion"] = np.ptp(c, axis=0).max() / np.median(face[:, 2])
                row["breath"] = breathing(t, face)
            for m in methods:
                fs, bvp = m.bvp(t, rgb_all[sel] if m.needs == "rgb" else crop_all[sel])
                hr, snr, f, p = sig.estimate_hr(bvp, fs)
                row[m.name] = (hr, snr, top_peaks(f, p))
            rows.append(row)

    names = [m.name for m in methods]
    head = f"{'t':>6} {'fps':>5} {'mot':>5} {'brth':>5} {'ref':>5} | " + " | ".join(
        f"{n[:22]:22s} {'SNR':>5} {'top peaks':>14}" for n in names)
    print(head)
    print("-" * len(head))
    for r in rows:
        cells = []
        for n in names:
            hr, snr, pk = r[n]
            flag = ""
            if np.isfinite(r["ref"]):
                flag = "*" if abs(hr - r["ref"]) <= 5 else ("~" if any(abs(q - r["ref"]) <= 5 for q in pk) else "x")
            cells.append(f"{hr:6.1f}{flag:1s}{'':15s} {snr:5.1f} {'/'.join(f'{q:.0f}' for q in pk):>14}")
        print(f"{r['t']:6.1f} {r['fps']:5.1f} {r.get('motion', float('nan')):5.2f} "
              f"{r.get('breath', float('nan')):5.1f} {r['ref']:5.0f} | " + " | ".join(cells))
    print("flags: * within 5 bpm of reference, ~ reference is among the top-3 peaks, x missed; "
          "mot = head motion / face width; brth = breaths/min from head motion")

    have_ref = [r for r in rows if np.isfinite(r["ref"])]
    if have_ref:
        print(f"\nwith reference: {len(have_ref)} windows")
        for n in names:
            err = np.array([r[n][0] - r["ref"] for r in have_ref])
            in_top = np.mean([any(abs(q - r["ref"]) <= 5 for q in r[n][2]) for r in have_ref])
            print(f"  {n:24s} MAE {np.abs(err).mean():5.1f}  bias {err.mean():+5.1f}  "
                  f"within 5 bpm {np.mean(np.abs(err) <= 5):4.0%}  ref in top-3 {in_top:4.0%}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "fps", "motion", "breath", "ref"] + [f"{n}_{k}" for n in names for k in ("hr", "snr", "peaks")])
            for r in rows:
                w.writerow([f"{r['t']:.1f}", f"{r['fps']:.1f}", f"{r.get('motion', float('nan')):.3f}",
                            f"{r.get('breath', float('nan')):.1f}", f"{r['ref']:.1f}"]
                           + [x for n in names for x in (f"{r[n][0]:.1f}", f"{r[n][1]:.1f}",
                                                         "/".join(f"{q:.0f}" for q in r[n][2]))])


if __name__ == "__main__":
    main()
