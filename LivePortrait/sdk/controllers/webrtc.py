from __future__ import annotations

import asyncio
import os
import uuid

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import HTTPException
from pydantic import BaseModel

from sdk.controllers.speech2video.webrtc_track import LivePortraitTrack
from sdk.controllers.webrtc_config import apply_aiortc_patches, rewrite_sdp_public_ip

apply_aiortc_patches()

_DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "speech2video", "uploads", "my_photo.png",
)

_DEMO_VISEMES = [
    {"start": 0.0,  "end": 0.01, "duration": 0.01, "viseme": "X"},
    {"start": 0.01, "end": 0.12, "duration": 0.11, "viseme": "B"},
    {"start": 0.12, "end": 0.19, "duration": 0.07, "viseme": "C"},
    {"start": 0.19, "end": 0.32, "duration": 0.13, "viseme": "A"},
    {"start": 0.32, "end": 0.59, "duration": 0.27, "viseme": "B"},
    {"start": 0.59, "end": 0.66, "duration": 0.07, "viseme": "G"},
    {"start": 0.66, "end": 0.8,  "duration": 0.14, "viseme": "E"},
    {"start": 0.8,  "end": 0.87, "duration": 0.07, "viseme": "F"},
    {"start": 0.87, "end": 0.94, "duration": 0.07, "viseme": "B"},
    {"start": 0.94, "end": 1.08, "duration": 0.14, "viseme": "C"},
    {"start": 1.08, "end": 1.22, "duration": 0.14, "viseme": "E"},
    {"start": 1.22, "end": 1.36, "duration": 0.14, "viseme": "B"},
    {"start": 1.36, "end": 1.44, "duration": 0.08, "viseme": "A"},
    {"start": 1.44, "end": 1.61, "duration": 0.17, "viseme": "E"},
    {"start": 1.61, "end": 1.69, "duration": 0.08, "viseme": "A"},
    {"start": 1.69, "end": 1.86, "duration": 0.17, "viseme": "C"},
    {"start": 1.86, "end": 1.93, "duration": 0.07, "viseme": "B"},
    {"start": 1.93, "end": 2.04, "duration": 0.11, "viseme": "A"},
    {"start": 2.04, "end": 2.1,  "duration": 0.06, "viseme": "B"},
    {"start": 2.1,  "end": 2.16, "duration": 0.06, "viseme": "G"},
    {"start": 2.16, "end": 2.23, "duration": 0.07, "viseme": "C"},
    {"start": 2.23, "end": 2.3,  "duration": 0.07, "viseme": "B"},
    {"start": 2.3,  "end": 2.38, "duration": 0.08, "viseme": "A"},
    {"start": 2.38, "end": 2.46, "duration": 0.08, "viseme": "E"},
    {"start": 2.46, "end": 2.53, "duration": 0.07, "viseme": "A"},
    {"start": 2.53, "end": 2.72, "duration": 0.19, "viseme": "F"},
    {"start": 2.72, "end": 2.79, "duration": 0.07, "viseme": "E"},
    {"start": 2.79, "end": 2.93, "duration": 0.14, "viseme": "C"},
    {"start": 2.93, "end": 3.0,  "duration": 0.07, "viseme": "B"},
    {"start": 3.0,  "end": 3.36, "duration": 0.36, "viseme": "X"},
    {"start": 3.36, "end": 3.62, "duration": 0.26, "viseme": "B"},
    {"start": 3.62, "end": 3.83, "duration": 0.21, "viseme": "C"},
    {"start": 3.83, "end": 3.9,  "duration": 0.07, "viseme": "B"},
    {"start": 3.9,  "end": 4.11, "duration": 0.21, "viseme": "C"},
    {"start": 4.11, "end": 4.18, "duration": 0.07, "viseme": "E"},
    {"start": 4.18, "end": 4.25, "duration": 0.07, "viseme": "C"},
    {"start": 4.25, "end": 4.39, "duration": 0.14, "viseme": "B"},
    {"start": 4.39, "end": 4.51, "duration": 0.12, "viseme": "A"},
    {"start": 4.51, "end": 4.58, "duration": 0.07, "viseme": "C"},
    {"start": 4.58, "end": 4.64, "duration": 0.06, "viseme": "E"},
    {"start": 4.64, "end": 4.78, "duration": 0.14, "viseme": "F"},
    {"start": 4.78, "end": 4.85, "duration": 0.07, "viseme": "E"},
    {"start": 4.85, "end": 4.92, "duration": 0.07, "viseme": "F"},
    {"start": 4.92, "end": 4.99, "duration": 0.07, "viseme": "H"},
    {"start": 4.99, "end": 5.09, "duration": 0.1,  "viseme": "D"},
    {"start": 5.09, "end": 5.13, "duration": 0.04, "viseme": "C"},
    {"start": 5.13, "end": 5.2,  "duration": 0.07, "viseme": "A"},
    {"start": 5.2,  "end": 5.27, "duration": 0.07, "viseme": "F"},
    {"start": 5.27, "end": 5.33, "duration": 0.06, "viseme": "A"},
    {"start": 5.33, "end": 5.41, "duration": 0.08, "viseme": "E"},
    {"start": 5.41, "end": 5.55, "duration": 0.14, "viseme": "D"},
    {"start": 5.55, "end": 5.62, "duration": 0.07, "viseme": "C"},
    {"start": 5.62, "end": 5.7,  "duration": 0.08, "viseme": "A"},
    {"start": 5.7,  "end": 6.01, "duration": 0.31, "viseme": "B"},
    {"start": 6.01, "end": 6.09, "duration": 0.08, "viseme": "A"},
    {"start": 6.09, "end": 6.45, "duration": 0.36, "viseme": "B"},
    {"start": 6.45, "end": 6.48, "duration": 0.03, "viseme": "X"},
]

_peers: dict[str, RTCPeerConnection] = {}


class OfferPayload(BaseModel):
    sdp: str
    type: str


class AnswerPayload(BaseModel):
    session_id: str
    sdp: str
    type: str


async def webrtc_offer(payload: OfferPayload) -> AnswerPayload:
    if payload.type != "offer":
        raise HTTPException(status_code=400, detail="payload.type must be 'offer'")

    pc = RTCPeerConnection()
    session_id = str(uuid.uuid4())
    _peers[session_id] = pc

    track = LivePortraitTrack(visemes=_DEMO_VISEMES, source_path=_DEFAULT_SOURCE)
    pc.addTrack(track)

    @pc.on("connectionstatechange")
    async def on_state_change():
        print(f"[webrtc] {session_id} state={pc.connectionState}", flush=True)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            track.stop()
            await pc.close()
            _peers.pop(session_id, None)

    @pc.on("iceconnectionstatechange")
    async def on_ice_change():
        print(f"[webrtc] {session_id} ice={pc.iceConnectionState}", flush=True)

    @pc.on("icegatheringstatechange")
    def on_gather_change():
        print(f"[webrtc] {session_id} gather={pc.iceGatheringState}", flush=True)

    offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return AnswerPayload(
        session_id=session_id,
        sdp=rewrite_sdp_public_ip(pc.localDescription.sdp),
        type=pc.localDescription.type,
    )


async def shutdown_all_peers():
    coros = [pc.close() for pc in _peers.values()]
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)
    _peers.clear()
