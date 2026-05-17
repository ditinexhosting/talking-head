import asyncio

from fastapi import APIRouter
from pydantic import BaseModel

from sdk.controllers.rest import get_root, get_health
from sdk.controllers.speech2video.video import run_liveportrait, run_liveportrait_video
from sdk.controllers.webrtc import webrtc_offer, OfferPayload, AnswerPayload
from sdk.controllers.livekit_agent import run_agent

router = APIRouter(prefix="/api", tags=["rest"])


@router.get("/")
def root():
    return get_root()


@router.get("/health")
def health():
    return get_health()


@router.post("/animate")
def animate():
    return run_liveportrait()


@router.post("/webrtc/offer", response_model=AnswerPayload)
async def webrtc_offer_route(payload: OfferPayload):
    return await webrtc_offer(payload)


class JoinBody(BaseModel):
    room_id: str
    callback_url: str | None = None


@router.post("/join")
async def join_room(body: JoinBody):
    asyncio.create_task(run_agent(body.room_id, body.callback_url))
    return {"status": "ok"}
