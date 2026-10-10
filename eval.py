"""Benchmark on UBFC-rPPG (DATASET_2): POS and RhythmMamba against the contact PPG.

    python eval.py                                   # all subjects in data/UBFC-rPPG
    python eval.py --subjects subject1 subject3      # a few
    python eval.py --weights weights/PURE_cross_RhythmMamba.pth

Expected layout (as on the dataset's Google Drive):  data/UBFC-rPPG/subjectN/{vid.avi, ground_truth.txt}
ground_truth.txt: line 1 - PPG waveform, line 2 - oximeter HR, line 3 - timestamps, s.

Each video goes through the same pipeline as live.py: YuNet tracking frame by frame,
skin-ROI mean RGB for POS, 128x128 face crop for RhythmMamba. The per-frame data is cached
(data/UBFC-rPPG/cache/*.npz), so re-running with other weights skips the video decoding.

Two protocols:
  window - 10 s windows every 1 s (what the user sees in live.py)
  video  - one estimate per whole video (the rPPG-Toolbox / paper protocol; RhythmMamba is
           run on 160-frame chunks as in the toolbox and the outputs are concatenated)
Reference HR = spectral peak of the contact PPG over the same span, through the same
detrend / band-pass / spectrum as the estimates. Subjects whose mean reference HR < --low-hr
are reported separately.

UBFC_cross weights are trained on UBFC-rPPG, so they are not evaluated here by default.
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

from rppg import signal as sig
from rppg.face import FaceTracker
from rppg.methods.pos import POS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = [os.path.join(HERE, "weights", f)
                   for f in ("PURE_cross_RhythmMamba.pth", "MMPD_intra_RhythmMamba.pth")]
CHUNK = 160  # RhythmMamba clip length in rPPG-Toolbox


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(HERE, "data", "UBFC-rPPG"))
    p.add_argument("--subjects", nargs="*", help="subject folder names (default: all with vid.avi)")
    p.add_argument("--weights", nargs="*", default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="auto")
    p.add_argument("--fs", type=float, default=30.0)
    p.add_argument("--window", type=float, default=10.0)
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--low-hr", type=float, default=70.0)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--tag", default="ubfc", help="prefix of the output files")
    return p.parse_args()


# ---------------------------------------------------------------- data

def load_ground_truth(path):
    with open(path) as fh:
        rows = [np.array(line.split(), dtype=np.float64) for line in fh if line.strip()]
    ppg, hr, t = rows[0], rows[1], rows[2]
    return t - t[0], ppg, hr


def video_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield fps, frame
    finally:
        cap.release()


def extract(video_path, cache_path):
    """Per-frame skin RGB and face crop via FaceTracker (as live.py). Frames without a face
    repeat the last good values; their count is reported."""
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["idx"], d["rgb"], d["crop"], float(d["fps"]), int(d["missed"])
    tracker = FaceTracker()
    idx, rgb, crop, missed, fps = [], [], [], 0, 30.0
    for i, (fps, frame) in enumerate(video_frames(video_path)):
        ok = tracker.update(frame)
        c, k = (tracker.mean_rgb(frame), tracker.crop(frame)) if ok else (None, None)
        if c is None or k is None:
            missed += 1
            if not rgb:
                continue  # nothing to repeat yet: the first frames are dropped
            c, k = rgb[-1], crop[-1]
        idx.append(i), rgb.append(c), crop.append(k)
    idx, rgb, crop = np.array(idx), np.array(rgb), np.array(crop, dtype=np.uint8)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, idx=idx, rgb=rgb, crop=crop, fps=fps, missed=missed, n_frames=i + 1)
    return idx, rgb, crop, fps, missed


# ---------------------------------------------------------------- estimates

def ref_hr(t_gt, ppg, t0, t1, fs):
    sel = (t_gt >= t0) & (t_gt <= t1)
    _, y = sig.resample_uniform(t_gt[sel], ppg[sel], fs)
    y = sig.bandpass(sig.detrend(y, 100), fs)
    return sig.estimate_hr(y, fs)[0]


def rm_whole_video(m, t, crops, fs):
    """rPPG-Toolbox protocol: standardise and run each 160-frame chunk, concatenate."""
    _, frames = sig.resample_uniform(t, crops.astype(np.float32), fs)
    n = len(frames) // CHUNK * CHUNK
    y = np.concatenate([m.raw_output(frames[i:i + CHUNK]) for i in range(0, n, CHUNK)])
    y = sig.detrend(y.astype(np.float64), 100)
    return sig.bandpass(y, fs)


def metrics(est, ref, snr):
    est, ref, snr = map(np.asarray, (est, ref, snr))
    ok = np.isfinite(est) & np.isfinite(ref)
    e, r = est[ok], ref[ok]
    if len(e) == 0:
        return dict(n=0)
    err = e - r
    pearson = float(np.corrcoef(e, r)[0, 1]) if len(e) > 2 and e.std() > 0 and r.std() > 0 else float("nan")
    s = snr[ok]
    return dict(n=len(e), MAE=np.abs(err).mean(), RMSE=np.sqrt((err ** 2).mean()),
                MAPE=100 * np.mean(np.abs(err) / r), Pearson=pearson,
                SNR=float(np.mean(s[np.isfinite(s)])) if np.isfinite(s).any() else float("nan"),
                within5=100 * np.mean(np.abs(err) <= 5))


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    subjects = args.subjects or sorted(
        (s for s in os.listdir(args.data) if os.path.exists(os.path.join(args.data, s, "vid.avi"))),
        key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))
    if not subjects:
        raise SystemExit(f"no subjects with vid.avi in {args.data}")

    methods = [POS(fs=args.fs, window_s=args.window)]
    if args.weights:
        import torch
        from rppg.methods.rhythmmamba import RhythmMambaMethod
        dev = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
        for w in args.weights:
            m = RhythmMambaMethod(w, fs=args.fs, window_s=args.window, device=dev)
            m.name = "RM:" + os.path.basename(w).split("_RhythmMamba")[0]
            methods.append(m)
    names = [m.name for m in methods]

    win_rows, vid_rows = [], []
    for s in subjects:
        t_start = time.perf_counter()
        d = os.path.join(args.data, s)
        t_gt, ppg, ox = load_ground_truth(os.path.join(d, "ground_truth.txt"))
        idx, rgb, crop, fps, missed = extract(os.path.join(d, "vid.avi"),
                                              os.path.join(args.data, "cache", s + ".npz"))
        # Frame times: the ground-truth clock when it has a sample for every frame, else the file's fps
        t = t_gt[idx] if idx[-1] < len(t_gt) else idx / fps
        dur = min(t[-1], t_gt[-1])

        # whole video
        t0 = t[0]
        vr = dict(subject=s, frames=len(rgb), missed=missed, duration=dur - t0,
                  ref=ref_hr(t_gt, ppg, t0, dur, args.fs), ox=float(np.mean(ox[ox > 0])))
        for m in methods:
            if m.needs == "rgb":
                fs, bvp = m.bvp(t, rgb)
            else:
                fs, bvp = args.fs, rm_whole_video(m, t, crop, args.fs)
            vr[m.name], vr[m.name + "_snr"] = sig.estimate_hr(bvp, fs)[:2]
        vid_rows.append(vr)

        # 10 s windows, 1 s step
        for t_end in np.arange(t[0] + args.window, dur + 1e-9, args.step):
            sel = slice(np.searchsorted(t, t_end - args.window), np.searchsorted(t, t_end, "right"))
            wr = dict(subject=s, t=t_end, ref=ref_hr(t_gt, ppg, t_end - args.window, t_end, args.fs),
                      ox=float(np.mean(ox[(t_gt > t_end - args.window) & (t_gt <= t_end)])))
            for m in methods:
                fs, bvp = m.bvp(t[sel], rgb[sel] if m.needs == "rgb" else crop[sel])
                wr[m.name], wr[m.name + "_snr"] = sig.estimate_hr(bvp, fs)[:2]
            win_rows.append(wr)
        print(f"{s:10s} {len(rgb):5d} frames (missed {missed}), ref {vr['ref']:5.1f} ox {vr['ox']:5.1f} | "
              + "  ".join(f"{n} {vr[n]:5.1f}" for n in names) + f"  [{time.perf_counter() - t_start:.0f} s]",
              flush=True)

    # mean reference HR over the video decides "low HR"
    low = {r["subject"] for r in vid_rows if r["ref"] < args.low_hr}
    summary = []
    for proto, rows in (("video", vid_rows), ("window", win_rows)):
        for subset, keep in (("all", lambda r: True), (f"HR<{args.low_hr:.0f}", lambda r: r["subject"] in low),
                             (f"HR>={args.low_hr:.0f}", lambda r: r["subject"] not in low)):
            sub = [r for r in rows if keep(r)]
            for n in names:
                mt = metrics([r[n] for r in sub], [r["ref"] for r in sub], [r[n + "_snr"] for r in sub])
                summary.append(dict(protocol=proto, subset=subset, method=n, videos=len({r["subject"] for r in sub}), **mt))

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, f"{args.tag}_summary.csv"), summary)
    write_csv(os.path.join(args.out, f"{args.tag}_videos.csv"), vid_rows)
    write_csv(os.path.join(args.out, f"{args.tag}_windows.csv"), win_rows)
    md = markdown(args, summary, vid_rows, names, low)
    with open(os.path.join(args.out, f"{args.tag}_summary.md"), "w") as fh:
        fh.write(md)
    print("\n" + md)


def fmt(v, nd=2):
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}" if np.isfinite(v) else "—"
    return str(v)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v, 3) for k, v in r.items()})


def markdown(args, summary, vid_rows, names, low):
    out = [f"# UBFC-rPPG (DATASET_2): {len(vid_rows)} videos, {len(low)} with mean HR < {args.low_hr:.0f}\n",
           f"Window {args.window:.0f} s, step {args.step:.0f} s; reference = spectral peak of the contact PPG "
           "over the same span. Errors in bpm, MAPE in %, SNR in dB.\n"]
    for proto in ("video", "window"):
        out.append(f"\n## Protocol: {proto}\n")
        out.append("| Subset | Method | Videos | N | MAE | RMSE | MAPE | Pearson | SNR | ≤5 bpm, % |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in summary:
            if r["protocol"] == proto and r.get("n"):
                out.append(f"| {r['subset']} | {r['method']} | {r['videos']} | {r['n']} | {fmt(r['MAE'])} | "
                           f"{fmt(r['RMSE'])} | {fmt(r['MAPE'])} | {fmt(r['Pearson'])} | {fmt(r['SNR'])} | "
                           f"{fmt(r['within5'], 0)} |")
    out.append("\n## Per video (whole-video estimate)\n")
    out.append("| Subject | Ref | Oximeter | " + " | ".join(names) + " | Missed frames |")
    out.append("|---|---|---|" + "---|" * len(names) + "---|")
    for r in vid_rows:
        mark = " **(low)**" if r["subject"] in low else ""
        out.append(f"| {r['subject']}{mark} | {fmt(r['ref'], 1)} | {fmt(r['ox'], 1)} | "
                   + " | ".join(fmt(r[n], 1) for n in names) + f" | {r['missed']} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
