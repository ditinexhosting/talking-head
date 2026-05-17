from contextlib import asynccontextmanager

from fastapi import FastAPI

from sdk.controllers.tts import _get_kokoro
from sdk.controllers.liveportrait import _get_pipeline
from sdk.routes.rest.router import router as rest_router
from sdk.routes.ws.router import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_kokoro()
    _get_pipeline()
    yield


app = FastAPI(title="Talking Head SDK", lifespan=lifespan)

app.include_router(rest_router)
app.include_router(ws_router)
