from __future__ import annotations

import os
import pickle
import shutil
import tempfile

from starlette.background import BackgroundTask
from fastapi.responses import FileResponse

from src.config.argument_config import ArgumentConfig
from sdk.controllers.liveportrait import _get_pipeline
from sdk.controllers.speech2video.motion import neutral_keyframes, build_template, add_blinks

_DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "uploads", "my_photo.png"
)

_FPS = 25
_NEUTRAL_SECONDS = 2


def run_liveportrait() -> FileResponse:
    keyframes = neutral_keyframes(seconds=_NEUTRAL_SECONDS, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

    out_dir = tempfile.mkdtemp()
    tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    try:
        pickle.dump(template, tpl_file)
        tpl_file.close()

        args = ArgumentConfig(
            source=_DEFAULT_SOURCE,
            driving=tpl_file.name,
            output_dir=out_dir,
        )
        wfp, _ = _get_pipeline().execute(args)
    finally:
        os.unlink(tpl_file.name)

    return FileResponse(
        wfp,
        media_type="video/mp4",
        filename="animated.mp4",
        background=BackgroundTask(shutil.rmtree, out_dir, ignore_errors=True),
    )
