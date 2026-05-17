import logging
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from gpu_service import notify_gpu_join
from livekit_service import create_room, generate_ui_token, LIVEKIT_URL

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

# VPS public base URL — used to build the callback URL sent to the GPU server.
# Override in .env when the VPS sits behind a reverse proxy.
VPS_BASE_URL = os.getenv("VPS_BASE_URL", "https://cargo.ditinex.com")

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VPS orchestrator starting up — LiveKit: %s", LIVEKIT_URL)
    yield
    logger.info("VPS orchestrator shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Talking-Head VPS Orchestrator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SessionRequest(BaseModel):
    user_id: str


class SessionResponse(BaseModel):
    room_id: str
    token: str
    livekit_url: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/session", response_model=SessionResponse)
async def create_session(body: SessionRequest, request: Request):
    room_id = "session-" + uuid.uuid4().hex[:8]
    log = logger.getChild("create_session")
    log.info("Creating session room_id=%s user_id=%s", room_id, body.user_id)

    # 1. Create the LiveKit room
    try:
        await create_room(room_id)
    except Exception as exc:
        log.error("Failed to create LiveKit room room_id=%s: %s", room_id, exc)
        raise HTTPException(status_code=502, detail="Failed to create LiveKit room") from exc

    # 2. Generate a UI access token (subscribe-only)
    token = generate_ui_token(room_id=room_id, user_id=body.user_id)
    log.info("Generated UI token room_id=%s", room_id)

    # 3. Notify the GPU agent to join the room
    await notify_gpu_join(room_id=room_id, vps_callback_base=VPS_BASE_URL)

    log.info("Session ready room_id=%s", room_id)
    return SessionResponse(room_id=room_id, token=token, livekit_url=LIVEKIT_URL)


@app.post("/session-ended/{room_id}")
async def session_ended(room_id: str):
    logger.info("Session ended room_id=%s", room_id)
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}
