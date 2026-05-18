from __future__ import annotations

import time

import torch

from sdk.controllers.speech2video.pipeline import LivePortraitPipeline
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.rprint import rlog as log

_pipeline: LivePortraitPipeline | None = None


def _warmup(pipeline: LivePortraitPipeline) -> None:
    from sdk.controllers.speech2video.video import warmup_stream  # lazy: avoids import cycle

    log("[WARMUP] Driving a 4-frame stream through the real pipeline to specialize torch.compile graphs...")
    t0 = time.time()
    warmup_stream(pipeline)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log(f"[WARMUP] Done in {time.time() - t0:.1f}s")


def _get_pipeline() -> LivePortraitPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LivePortraitPipeline(
            inference_cfg=InferenceConfig(
                flag_do_torch_compile=True,
                flag_use_half_precision=True,
                flag_stitching=False,
                flag_eye_retargeting=False,
                flag_lip_retargeting=False,
                flag_normalize_lip=False,
            ),
            crop_cfg=CropConfig(),
        )
        _warmup(_pipeline)
    return _pipeline
