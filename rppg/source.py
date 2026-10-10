"""Frame sources: webcam (threaded, timestamped) or video file."""
import threading
import time

import cv2


class Camera:
    """Grabs frames in a background thread so slow processing never drops the clock.

    Every frame gets its own monotonic timestamp: webcams do not deliver a
    constant frame rate, so the signal code resamples on these timestamps.
    """

    def __init__(self, index=0, width=640, height=480, fps=30, lock_exposure=True):
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {index}. On macOS allow camera access for your "
                "terminal/VS Code in System Settings > Privacy & Security > Camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.exposure_locked = False
        if lock_exposure:
            self.exposure_locked = self._lock_exposure()
        self._frame = None
        self._t = None
        self._seq = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _lock_exposure(self):
        # Auto exposure / white balance change skin colour far more than the pulse does.
        # Many drivers (incl. most AVFoundation webcams) ignore these; we report the result.
        ok = False
        for prop, value in (
            (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),  # V4L2 "manual" mode
            (cv2.CAP_PROP_AUTO_EXPOSURE, 1),
            (cv2.CAP_PROP_AUTO_WB, 0),
        ):
            ok = self.cap.set(prop, value) or ok
        return ok

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame, self._t = frame, t
                self._seq += 1

    def read(self, last_seq=-1, timeout=1.0):
        """Returns (seq, timestamp, frame_bgr) for the newest frame newer than last_seq."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    return self._seq, self._t, self._frame
            time.sleep(0.002)
        return None

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


class VideoFile:
    """Video file played frame by frame; timestamps come from the file's FPS."""

    def __init__(self, path, realtime=False, fps=None):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {path}")
        # Image sequences report a made-up FPS (often 25), so allow an override
        self.fps = fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime = realtime
        self.exposure_locked = None
        self._seq = 0
        self._start = time.monotonic()

    def read(self, last_seq=-1, timeout=1.0):
        ok, frame = self.cap.read()
        if not ok:
            return None
        t = self._seq / self.fps
        if self.realtime:
            delay = self._start + t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._seq += 1
        return self._seq, t, frame

    def release(self):
        self.cap.release()
