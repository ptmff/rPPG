"""Live heart rate from a webcam (or a video file) with POS and/or RhythmMamba.

    python live.py                          # webcam 0, both methods
    python live.py --method pos             # classical baseline only
    python live.py --video clip.mp4 --headless --log out.csv

Keys in the window: q / Esc - quit, r - pause/resume recording (--record path).
Type a reference heart rate (from a pulse oximeter / watch) and press Enter: it is logged
with the current timestamp to the CSV (method "REF") and to the recording.
"""
import argparse
import csv
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from rppg import signal as sig
from rppg.face import FaceTracker
from rppg.methods.pos import POS
from rppg.source import Camera, VideoFile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "MMPD_intra_RhythmMamba.pth")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", help="process a video file instead of the webcam")
    p.add_argument("--method", choices=["pos", "rm", "both"], default="both")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="RhythmMamba checkpoint (.pth)")
    p.add_argument("--device", default="auto", help="torch device for RhythmMamba: auto, cpu or mps")
    p.add_argument("--window", type=float, default=10.0, help="analysis window, s")
    p.add_argument("--min-window", type=float, default=6.0, help="first estimate after this many s")
    p.add_argument("--update", type=float, default=1.0, help="re-estimate every N s")
    p.add_argument("--fs", type=float, default=30.0, help="internal resampling rate, Hz")
    p.add_argument("--no-lock", action="store_true", help="do not try to lock exposure / WB")
    p.add_argument("--headless", action="store_true", help="no window, print estimates")
    p.add_argument("--video-fps", type=float, help="override the FPS the video file reports")
    p.add_argument("--realtime", action="store_true", help="play --video at its real speed")
    p.add_argument("--log", help="append estimates to this CSV")
    p.add_argument("--record", help="save timestamps, ROI colour and face crops to this .npz")
    return p.parse_args()


class Buffer:
    """Last `span` seconds of per-frame data; cleared when the face is lost so windows
    never contain a gap."""

    def __init__(self, span):
        self.span = span
        self.t, self.rgb, self.crop = deque(), deque(), deque()

    def add(self, t, rgb, crop):
        self.t.append(t), self.rgb.append(rgb), self.crop.append(crop)
        while self.t and self.t[-1] - self.t[0] > self.span:
            self.t.popleft(), self.rgb.popleft(), self.crop.popleft()

    def clear(self):
        self.t.clear(), self.rgb.clear(), self.crop.clear()

    @property
    def duration(self):
        return self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0

    def snapshot(self):
        return np.array(self.t), np.array(self.rgb), np.array(self.crop)


class Estimator:
    """Runs the methods off the UI thread; the newest result per method is in .results."""

    def __init__(self, methods):
        self.methods = methods
        self.smoothers = {m.name: sig.HRSmoother() for m in methods}
        self.results = {}
        self.busy = False
        self._lock = threading.Lock()

    def submit(self, snapshot, t_now):
        if self.busy:
            return False
        self.busy = True
        threading.Thread(target=self._run, args=(snapshot, t_now), daemon=True).start()
        return True

    def _run(self, snapshot, t_now):
        t, rgb, crop = snapshot
        try:
            for m in self.methods:
                t0 = time.perf_counter()
                fs, bvp = m.bvp(t, rgb if m.needs == "rgb" else crop)
                hr, snr, _, _ = sig.estimate_hr(bvp, fs)
                sm = self.smoothers[m.name]
                accepted = sm.push(hr, snr, t_now)
                with self._lock:
                    self.results[m.name] = dict(
                        t=t_now, hr=hr, snr=snr, hr_smooth=sm.value, accepted=accepted,
                        bvp=bvp[-int(5 * fs):], ms=1000 * (time.perf_counter() - t0))
        except Exception as e:  # keep the UI alive; show the error instead
            with self._lock:
                self.results["error"] = repr(e)
        finally:
            self.busy = False

    def reset(self):
        for s in self.smoothers.values():
            s.reset()
        with self._lock:
            self.results = {}

    def get(self):
        with self._lock:
            return dict(self.results)


def snr_color(snr):
    if not np.isfinite(snr) or snr < 0:
        return (60, 60, 230)      # red: probably noise
    if snr < 3:
        return (40, 200, 230)     # yellow: usable
    return (80, 200, 80)          # green: clean


def draw(frame, tracker, buf, results, fps, args, exposure_locked, recording, motion):
    h, w = frame.shape[:2]
    for x0, y0, x1, y1 in tracker.skin_rois():
        cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 200, 80), 1)
    r = tracker.crop_rect()
    if r:
        cv2.rectangle(frame, r[:2], r[2:], (40, 200, 230), 1)

    names = [n for n in results if n != "error"]
    panel = np.full((max(2, len(names)) * 52 + 80, w, 3), 24, np.uint8)
    y = 32
    for name in names:
        res = results[name]
        hr = res["hr_smooth"]
        txt = f"{name}: {hr:5.1f} bpm" if np.isfinite(hr) else f"{name}: --- bpm"
        cv2.putText(panel, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, snr_color(res["snr"]), 2)
        cv2.putText(panel, f"raw {res['hr']:5.1f}   SNR {res['snr']:5.1f} dB   {res['ms']:4.0f} ms",
                    (14, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
        y += 52
    if not names:
        msg = "no face" if tracker.face is None else \
            f"collecting {buf.duration:4.1f}/{args.min_window:.0f} s - keep still"
        cv2.putText(panel, msg, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if "error" in results:
        cv2.putText(panel, results["error"][:80], (12, panel.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 230), 1)

    # Waveform of the first method's BVP (last 5 s)
    if names:
        bvp = results[names[0]]["bvp"]
        if len(bvp) > 2:
            x0, x1, yc, amp = 12, w - 12, panel.shape[0] - 40, 24
            v = bvp / (np.abs(bvp).max() + 1e-9)
            pts = np.stack([np.linspace(x0, x1, len(v)), yc - amp * v], 1).astype(np.int32)
            cv2.polylines(panel, [pts], False, (230, 230, 230), 1, cv2.LINE_AA)

    status = f"{fps:4.1f} fps | window {min(buf.duration, args.window):4.1f}/{args.window:.0f} s"
    if exposure_locked is False:
        status += " | AE/WB not locked"
    if motion:
        status += " | MOTION"
    if recording:
        status += " | REC"
    (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (0, 0), (tw + 16, th + 14), (24, 24, 24), -1)
    cv2.putText(frame, status, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (80, 80, 240) if motion else (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([frame, panel])


def main():
    args = parse_args()

    methods = []
    if args.method in ("pos", "both"):
        methods.append(POS(fs=args.fs, window_s=args.window))
    if args.method in ("rm", "both"):
        if not os.path.exists(args.weights):
            raise SystemExit(f"RhythmMamba weights not found: {args.weights} (use --method pos)")
        import torch
        from rppg.methods.rhythmmamba import RhythmMambaMethod
        if args.device == "auto":
            args.device = "mps" if torch.backends.mps.is_available() else "cpu"
        methods.append(RhythmMambaMethod(args.weights, fs=args.fs, window_s=args.window,
                                         device=args.device))

    src = VideoFile(args.video, realtime=args.realtime, fps=args.video_fps) if args.video else \
        Camera(args.camera, lock_exposure=not args.no_lock)
    tracker = FaceTracker()
    buf = Buffer(args.window)
    est = Estimator(methods)
    log = open(args.log, "a", newline="") if args.log else None
    logw = csv.writer(log) if log else None
    if log and log.tell() == 0:
        logw.writerow(["t", "method", "hr_raw", "snr_db", "hr_smooth", "accepted"])
    rec = {"t": [], "rgb": [], "crop": [], "face": [], "ref_t": [], "ref_hr": []}
    recording = bool(args.record)
    typed = ""  # reference HR being typed in the window
    last_logged = {}

    seq, last_submit, frame_times = -1, -1e9, deque(maxlen=60)
    centers = deque(maxlen=int(args.fs * 2))
    try:
        while True:
            got = src.read(seq)
            if got is None:
                if args.video:
                    break
                continue
            seq, t, frame = got
            frame_times.append(t)

            if tracker.update(frame):
                rgb, crop = tracker.mean_rgb(frame), tracker.crop(frame)
                if rgb is not None and crop is not None:
                    buf.add(t, rgb, crop)
                    if recording:
                        rec["t"].append(t), rec["rgb"].append(rgb), rec["crop"].append(crop)
                        rec["face"].append(tracker.face.copy())
                    centers.append(tracker.face[:2] + tracker.face[2:4] / 2)
            else:
                buf.clear()
                est.reset()
                centers.clear()

            # Head motion over the last 2 s, relative to face size
            motion = False
            if len(centers) > 5 and tracker.face is not None:
                c = np.array(centers)
                motion = np.ptp(c, axis=0).max() > 0.15 * tracker.face[2]

            if buf.duration >= args.min_window and t - last_submit >= args.update:
                if est.submit(buf.snapshot(), t):
                    last_submit = t
                if args.video and not args.realtime:
                    while est.busy:  # offline: process every window, do not skip
                        time.sleep(0.001)

            results = est.get()
            for name, r in results.items():
                if name == "error" or last_logged.get(name) == r["t"]:
                    continue
                last_logged[name] = r["t"]
                if logw:
                    logw.writerow([f"{r['t']:.3f}", name, f"{r['hr']:.2f}", f"{r['snr']:.2f}",
                                   f"{r['hr_smooth']:.2f}", int(r["accepted"])])
                if args.headless:
                    print(f"t={r['t']:7.2f}s  {name:12s} HR {r['hr']:6.1f}  smooth {r['hr_smooth']:6.1f}"
                          f"  SNR {r['snr']:5.1f} dB  ({r['ms']:.0f} ms)", flush=True)
            if "error" in results and args.headless:
                raise SystemExit(results["error"])

            if not args.headless:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) \
                    if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0
                view = draw(frame.copy(), tracker, buf, results, fps, args,
                            src.exposure_locked, recording, motion)
                ref_txt = f"reference: {typed}_" if typed else \
                    (f"reference {rec['ref_hr'][-1]:.0f} logged" if rec["ref_hr"] else "type reference bpm + Enter")
                cv2.putText(view, ref_txt, (view.shape[1] - 250, view.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
                cv2.imshow("rPPG", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r") and args.record:
                    recording = not recording
                elif ord("0") <= key <= ord("9") and len(typed) < 3:
                    typed += chr(key)
                elif key in (8, 127):
                    typed = typed[:-1]
                elif key in (10, 13) and typed:
                    rec["ref_t"].append(t), rec["ref_hr"].append(float(typed))
                    if logw:
                        logw.writerow([f"{t:.3f}", "REF", typed, "", "", ""])
                        log.flush()
                    typed = ""
    finally:
        src.release()
        if log:
            log.close()
        if not args.headless:
            cv2.destroyAllWindows()
        if args.record and rec["t"]:
            np.savez_compressed(args.record, t=np.array(rec["t"]), rgb=np.array(rec["rgb"]),
                                crop=np.array(rec["crop"], dtype=np.uint8),
                                face=np.array(rec["face"]), ref_t=np.array(rec["ref_t"]),
                                ref_hr=np.array(rec["ref_hr"]))
            print(f"saved {len(rec['t'])} frames to {args.record}")


if __name__ == "__main__":
    main()
