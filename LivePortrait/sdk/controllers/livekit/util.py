from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from sdk.controllers.speech2video.video import _IDLE_FRAME_PATH

logger = logging.getLogger(__name__)

WIDTH, HEIGHT = 512, 512


def _load_idle_frame() -> bytes | None:
    if not os.path.exists(_IDLE_FRAME_PATH):
        return None
    bgr = cv2.imread(_IDLE_FRAME_PATH)
    if bgr is None:
        return None
    if bgr.shape[:2] != (HEIGHT, WIDTH):
        bgr = cv2.resize(bgr, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    logger.info("[agent] idle frame loaded from %s", _IDLE_FRAME_PATH)
    return bytes(rgba.tobytes())


idle_frame: bytes | None = _load_idle_frame()
