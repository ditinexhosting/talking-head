from __future__ import annotations

import os
import pickle
import tempfile

import cv2
from fastapi.responses import Response

from src.config.argument_config import ArgumentConfig
from sdk.controllers.liveportrait import _get_pipeline
from sdk.controllers.speech2video.motion import neutral_keyframes, build_template, add_blinks

_DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "uploads", "my_photo.png"
)

_FPS = 25
_NEUTRAL_SECONDS = 2


def run_liveportrait() -> Response:
    keyframes = neutral_keyframes(seconds=_NEUTRAL_SECONDS, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

    tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    try:
        pickle.dump(template, tpl_file)
        tpl_file.close()

        args = ArgumentConfig(
            source=_DEFAULT_SOURCE,
            driving=tpl_file.name,
            output_dir=tempfile.mkdtemp(),
        )
        frame_gen, _ = _get_pipeline().execute_streaming(args)
        frame = next(frame_gen)  # RGB uint8 HxWx3
    finally:
        os.unlink(tpl_file.name)

    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return Response(content=buf.tobytes(), media_type="image/jpeg")
