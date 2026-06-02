import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from sdk.controllers.tts import _get_emotion_classifier, _get_pipeline as _get_tts_pipeline
from sdk.routes.rest.router import router as rest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_tts_pipeline, "a")
    await loop.run_in_executor(None, _get_emotion_classifier)
    yield


app = FastAPI(title="Talking Head SDK", lifespan=lifespan)

app.include_router(rest_router)
