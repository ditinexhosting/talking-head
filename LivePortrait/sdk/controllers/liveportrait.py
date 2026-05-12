from __future__ import annotations

from sdk.controllers.speech2video.pipeline import LivePortraitPipeline
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

_pipeline: LivePortraitPipeline | None = None


def _get_pipeline() -> LivePortraitPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LivePortraitPipeline(
            inference_cfg=InferenceConfig(),
            crop_cfg=CropConfig(),
        )
    return _pipeline
