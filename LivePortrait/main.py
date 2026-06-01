import gc
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from sdk.controllers.tts import _get_kokoro
from sdk.controllers.liveportrait import _get_pipeline
from sdk.controllers.viseme import _get_align_model
import sdk.controllers.tts as _tts_ctrl
import sdk.controllers.liveportrait as _lp_ctrl
import sdk.controllers.viseme as _vis_ctrl
from sdk.routes.rest.router import router as rest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_kokoro()
    _get_pipeline()
    _get_align_model()
    try:
        yield
    finally:
        _tts_ctrl._kokoro = None
        _lp_ctrl._pipeline = None
        _vis_ctrl._align_model = None
        _vis_ctrl._align_metadata = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="Talking Head SDK", lifespan=lifespan)

app.include_router(rest_router)
