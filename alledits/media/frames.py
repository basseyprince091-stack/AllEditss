"""Frame sampling. All heavy analysis runs on downscaled frames from the proxy,
never on the original (Principle 11 / cost control)."""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


class FrameReader:
    """Iterate frames at a target analysis rate and a target small width."""

    def __init__(self, path: Path | str, analysis_fps: float = 8.0, width: int = 256):
        self.path = str(path)
        self.analysis_fps = analysis_fps
        self.width = width
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {path}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w <= self.width:
            return frame
        nh = max(2, int(round(h * self.width / w)))
        return cv2.resize(frame, (self.width, nh), interpolation=cv2.INTER_AREA)

    def iter_frames(self, start: float = 0.0, end: float | None = None):
        """Yield (timestamp, bgr_frame_small)."""
        step = max(1, int(round(self.src_fps / self.analysis_fps)))
        if start > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        i = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if end is not None and t > end:
                break
            if i % step == 0:
                yield t, self._resize(frame)
            i += 1

    def frame_at(self, t: float):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = self.cap.read()
        return self._resize(frame) if ok else None

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()
