"""Hand tracking on a phone clip with MediaPipe: 21 keypoints per frame and a pinch signal."""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

from .config import HAND_MODEL, VIDEO_FPS


def make_landmarker():
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)
    return vision.HandLandmarker.create_from_options(opts)


def track_clip(path):
    """Returns pts (T, 21, 2) in pixels (NaN when no hand), det (T,) bool, and pinch (T,):
    thumb-tip to index-tip distance divided by hand size (wrist to middle-finger base), so it does not depend on
    how far the hand is from the camera."""
    cap = cv2.VideoCapture(path)
    lm = make_landmarker()
    pts, det, pinch, shape, i = [], [], [], None, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        shape = bgr.shape
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(img, int(i * 1000 / VIDEO_FPS))
        if res.hand_landmarks:
            p = np.array([(q.x * w, q.y * h) for q in res.hand_landmarks[0]])
            pts.append(p)
            det.append(True)
            pinch.append(np.linalg.norm(p[4] - p[8]) / np.linalg.norm(p[0] - p[9]))
        else:
            pts.append(np.full((21, 2), np.nan))
            det.append(False)
            pinch.append(np.nan)
        i += 1
    return dict(pts=np.array(pts), det=np.array(det), pinch=np.array(pinch), shape=shape)
