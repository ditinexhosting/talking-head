from contextlib import asynccontextmanager

from fastapi import FastAPI

from controllers.tts import _get_kokoro
from routes.rest.router import router as rest_router
from routes.ws.router import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_kokoro()
    yield


app = FastAPI(title="Talking Head SDK", lifespan=lifespan)

app.include_router(rest_router)
app.include_router(ws_router)
