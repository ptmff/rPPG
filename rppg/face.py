"""Face tracking (OpenCV YuNet) and the two inputs the methods need:
skin-ROI mean RGB for POS, and a 128x128 face crop for RhythmMamba."""
import os

import cv2
import numpy as np

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "face_detection_yunet_2023mar.onnx"
)


class FaceTracker:
    def __init__(self, model_path=DEFAULT_MODEL, det_width=320, score_threshold=0.7,
                 box_alpha=0.15, deadband=0.03, large_box_coef=1.5, crop_size=128):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        self.det = cv2.FaceDetectorYN.create(model_path, "", (det_width, det_width),
                                             score_threshold, 0.3, 5000)
        self.det_width = det_width
        self.box_alpha = box_alpha
        self.deadband = deadband
        self.coef = large_box_coef
        self.crop_size = crop_size
        self.face = None       # smoothed [x, y, w, h, 5 landmarks (x, y)...] in full-res pixels
        self.crop_box = None   # smoothed square crop box (cx, cy, side)
        self.missed = 0

    def update(self, frame_bgr):
        """Detects the face and updates the smoothed state. Returns True if a face is tracked."""
        h, w = frame_bgr.shape[:2]
        scale = self.det_width / w
        small = cv2.resize(frame_bgr, (self.det_width, int(round(h * scale))))
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None or len(faces) == 0:
            self.missed += 1
            if self.missed > 15:  # ~0.5 s without a face: drop the track
                self.face, self.crop_box = None, None
            return self.face is not None
        self.missed = 0
        best = faces[np.argmax(faces[:, 2] * faces[:, 3])]  # largest face
        raw = best[:14].astype(np.float64) / scale
        if self.face is None:
            self.face = raw
        else:
            self.face += self.box_alpha * (raw - self.face)
        self._update_crop_box(raw)
        return True

    def _update_crop_box(self, raw):
        x, y, bw, bh = raw[:4]
        side = max(bw, bh) * self.coef
        target = np.array([x + bw / 2, y + bh / 2, side])
        if self.crop_box is None:
            self.crop_box = target
            return
        # The crop must stay still: any jitter of the box is added straight into the signal.
        # Only follow the face once it moved noticeably, and then slowly.
        if np.max(np.abs(target - self.crop_box)) > self.deadband * self.crop_box[2]:
            self.crop_box += self.box_alpha * (target - self.crop_box)

    def _landmarks(self):
        return self.face[4:14].reshape(5, 2)  # right eye, left eye, nose, mouth right, mouth left

    def skin_rois(self):
        """Forehead and both cheeks as (x0, y0, x1, y1): the areas with the strongest pulse
        and no eyes, brows, mouth or background."""
        if self.face is None:
            return []
        eyes = self._landmarks()[:2]
        mouth = self._landmarks()[3:5]
        eye_l, eye_r = eyes[np.argmin(eyes[:, 0])], eyes[np.argmax(eyes[:, 0])]
        d = max(np.linalg.norm(eye_r - eye_l), 1.0)
        eye_y = eyes[:, 1].mean()
        mouth_y = mouth[:, 1].mean()
        rois = [
            (eye_l[0], eye_y - 0.95 * d, eye_r[0], eye_y - 0.45 * d),                    # forehead
            (eye_l[0] - 0.35 * d, eye_y + 0.35 * d, eye_l[0] + 0.15 * d, mouth_y - 0.10 * d),  # cheek
            (eye_r[0] - 0.15 * d, eye_y + 0.35 * d, eye_r[0] + 0.35 * d, mouth_y - 0.10 * d),  # cheek
        ]
        return [tuple(int(round(v)) for v in r) for r in rois if r[2] > r[0] and r[3] > r[1]]

    def mean_rgb(self, frame_bgr):
        """Pixel-count-weighted mean RGB over the skin ROIs, or None."""
        h, w = frame_bgr.shape[:2]
        total, count = np.zeros(3), 0
        for x0, y0, x1, y1 in self.skin_rois():
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            patch = frame_bgr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
            patch = patch[(patch.max(axis=1) < 250) & (patch.min(axis=1) > 5)]  # drop clipped pixels
            total += patch.sum(axis=0)
            count += len(patch)
        if count < 50:
            return None
        return (total / count)[::-1]  # BGR -> RGB

    def crop(self, frame_bgr):
        """RGB face crop, crop_size x crop_size uint8, as in rPPG-Toolbox (large box x1.5,
        clipped to the frame, INTER_AREA resize)."""
        if self.crop_box is None:
            return None
        h, w = frame_bgr.shape[:2]
        cx, cy, side = self.crop_box
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        face = cv2.resize(frame_bgr[y0:y1, x0:x1], (self.crop_size, self.crop_size),
                          interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    def crop_rect(self):
        if self.crop_box is None:
            return None
        cx, cy, side = self.crop_box
        return int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)
