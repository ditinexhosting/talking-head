import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from sdk.controllers.tts import _get_kokoro
from sdk.controllers.liveportrait import _get_pipeline
from sdk.routes.rest.router import router as rest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_kokoro()
    _get_pipeline()
    yield


app = FastAPI(title="Talking Head SDK", lifespan=lifespan)

app.include_router(rest_router)
