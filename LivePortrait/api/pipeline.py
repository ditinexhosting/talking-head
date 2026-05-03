import os.path as osp
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

_pipeline: LivePortraitPipeline | None = None


def get_pipeline() -> LivePortraitPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LivePortraitPipeline(
            inference_cfg=InferenceConfig(),
            crop_cfg=CropConfig(),
        )
    return _pipeline
