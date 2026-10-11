"""Face tracking (OpenCV YuNet, optionally MediaPipe FaceMesh) and the inputs the methods need:
per-region skin colour plus background colour and head position for POS / CHROM
(`features`), and a 128x128 face crop for RhythmMamba (`crop`).

Skin regions are the forehead and both cheeks. With FaceMesh (468 landmarks) they are
polygons that follow the face; without it, rectangles placed from YuNet's 5 landmarks.
Inside a region only skin-coloured pixels are averaged (`skin_pixels`): brows, hair,
shadows, highlights and clipped pixels are dropped."""
import os

import cv2
import numpy as np

MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_MODEL = os.path.join(MODELS, "face_detection_yunet_2023mar.onnx")
MESH_MODEL = os.path.join(MODELS, "face_landmarker.task")

REGIONS = ("forehead", "cheek_l", "cheek_r")
# Rows of FaceTracker.features(): mean RGB of each skin region, mean RGB of the background
# beside the face, and the raw face box (centre x, centre y, width) in pixels
F_BG, F_POSE = len(REGIONS), len(REGIONS) + 1
N_FEATURES = len(REGIONS) + 2

# FaceMesh landmarks whose convex hull is the region (image-left cheek first)
MESH_REGIONS = (
    (109, 10, 338, 297, 299, 296, 336, 9, 107, 66, 69, 67),             # forehead
    (116, 117, 118, 119, 100, 142, 203, 206, 205, 187, 123, 50),        # cheek, image left
    (345, 346, 347, 348, 329, 371, 423, 426, 425, 411, 352, 280),       # cheek, image right
)


def skin_pixels(patch_bgr, mask=None, chroma_tol=10.0):
    """Skin pixels of a patch as (M, 3) float BGR. Keeps unclipped pixels whose chroma (Cr, Cb)
    is near the patch median and whose brightness is not far from it: drops brows, hair,
    eyes, shadows and highlights without a fixed skin-colour model."""
    px = patch_bgr.reshape(-1, 3)
    if mask is not None:
        px = px[mask.reshape(-1)]
    px = px[(px.max(axis=1) < 250) & (px.min(axis=1) > 5)]
    if len(px) < 20:
        return px.astype(np.float64)
    ycc = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    ycc = ycc.astype(np.float64)
    med = np.median(ycc, axis=0)
    keep = (np.abs(ycc[:, 1] - med[1]) < chroma_tol) & (np.abs(ycc[:, 2] - med[2]) < chroma_tol) \
        & (ycc[:, 0] > 0.6 * med[0]) & (ycc[:, 0] < 1.4 * med[0])
    return px[keep].astype(np.float64)


class _Mesh:
    """MediaPipe FaceLandmarker in video mode; returns the 468 landmarks in pixels or None."""

    def __init__(self, model_path=MESH_MODEL):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        self.lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))
        self.ts = 0

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(frame_bgr[..., ::-1]))
        self.ts += 33  # must only increase; the real clock is not needed
        res = self.lm.detect_for_video(img, self.ts)
        if not res.face_landmarks:
            return None
        return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])


def mesh_available(model_path=MESH_MODEL):
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(model_path)


class FaceTracker:
    def __init__(self, model_path=DEFAULT_MODEL, det_width=320, score_threshold=0.7,
                 box_alpha=0.15, deadband=0.03, large_box_coef=1.5, crop_size=128,
                 mesh=None, mesh_alpha=0.3):
        """mesh: True / False / None (= use FaceMesh if mediapipe and its model are present)."""
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
        self.raw = None        # last raw detection (before smoothing)
        if mesh is None:
            mesh = mesh_available()
        self.mesh = _Mesh() if mesh else None
        self.mesh_alpha = mesh_alpha
        self.landmarks = None  # smoothed FaceMesh landmarks, (468, 2) pixels

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
                self.face, self.crop_box, self.landmarks = None, None, None
            return self.face is not None
        self.missed = 0
        best = faces[np.argmax(faces[:, 2] * faces[:, 3])]  # largest face
        raw = best[:14].astype(np.float64) / scale
        self.raw = raw
        if self.face is None:
            self.face = raw.copy()
        else:
            self.face += self.box_alpha * (raw - self.face)
        self._update_crop_box(raw)
        if self.mesh is not None:
            pts = self.mesh(frame_bgr)
            if pts is None:
                self.landmarks = None
            elif self.landmarks is None or self.landmarks.shape != pts.shape:
                self.landmarks = pts
            else:  # landmarks jitter by a pixel or two; every jitter of a region is noise
                self.landmarks += self.mesh_alpha * (pts - self.landmarks)
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

    def skin_polygons(self):
        """Forehead, image-left cheek, image-right cheek as (K, 2) int point arrays: FaceMesh
        hulls when landmarks are available, else the YuNet rectangles of skin_rois()."""
        if self.face is None:
            return []
        if self.landmarks is not None:
            return [cv2.convexHull(self.landmarks[list(idx)].astype(np.float32)).reshape(-1, 2)
                    .round().astype(np.int32) for idx in MESH_REGIONS]
        return [np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], np.int32)
                for x0, y0, x1, y1 in self.skin_rois()]

    def region_pixels(self, frame_bgr):
        """Skin pixels (M, 3) BGR of each region in skin_polygons() order."""
        h, w = frame_bgr.shape[:2]
        out = []
        for poly in self.skin_polygons():
            x0, y0 = np.maximum(poly.min(axis=0), 0)
            x1, y1 = np.minimum(poly.max(axis=0) + 1, (w, h))
            if x1 - x0 < 3 or y1 - y0 < 3:
                out.append(np.zeros((0, 3)))
                continue
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(mask, poly - (x0, y0), 1)
            out.append(skin_pixels(frame_bgr[y0:y1, x0:x1], mask.astype(bool)))
        return out

    def mean_rgb(self, frame_bgr):
        """Pixel-count-weighted mean RGB over all skin regions, or None."""
        px = [p for p in self.region_pixels(frame_bgr) if len(p)]
        px = np.concatenate(px) if px else np.zeros((0, 3))
        if len(px) < 50:
            return None
        return px.mean(axis=0)[::-1]  # BGR -> RGB

    def background_rgb(self, frame_bgr):
        """Mean RGB of two strips left and right of the face crop, or NaNs. The background
        sees the same exposure, white balance and light changes as the skin, but no pulse,
        so it serves as a reference channel to subtract (rppg/multiregion.py)."""
        if self.crop_box is None:
            return np.full(3, np.nan)
        h, w = frame_bgr.shape[:2]
        cx, cy, side = self.crop_box
        y0, y1 = int(max(0, cy - side / 3)), int(min(h, cy + side / 3))
        strips = []
        for xa, xb in ((cx - side / 2 - 0.3 * side, cx - side / 2), (cx + side / 2, cx + side / 2 + 0.3 * side)):
            xa, xb = int(max(0, xa)), int(min(w, xb))
            if xb - xa >= 4 and y1 - y0 >= 4:
                p = frame_bgr[y0:y1, xa:xb].reshape(-1, 3)
                strips.append(p[(p.max(axis=1) < 250) & (p.min(axis=1) > 5)])
        px = np.concatenate(strips) if strips else np.zeros((0, 3))
        if len(px) < 200:
            return np.full(3, np.nan)
        return px.astype(np.float64).mean(axis=0)[::-1]

    def features(self, frame_bgr):
        """(N_FEATURES, 3) per-frame input of POS / CHROM (see rppg/multiregion.py): mean RGB of
        each skin region (NaN if too few skin pixels), of the background, and the raw face
        box centre and width."""
        out = np.full((N_FEATURES, 3), np.nan)
        for i, px in enumerate(self.region_pixels(frame_bgr)[:len(REGIONS)]):
            if len(px) >= 30:
                out[i] = px.mean(axis=0)[::-1]
        out[F_BG] = self.background_rgb(frame_bgr)
        if self.raw is not None:
            x, y, bw, bh = self.raw[:4]
            out[F_POSE] = (x + bw / 2, y + bh / 2, bw)
        return out

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
