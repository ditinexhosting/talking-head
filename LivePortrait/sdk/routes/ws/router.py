from fastapi import APIRouter, WebSocket
from sdk.controllers.ws import WebSocketConnection

router = APIRouter(tags=["websocket"])


@router.websocket("/audio")
async def echo(ws: WebSocket):
    conn = WebSocketConnection(ws)
    await conn.run_tts()

@router.websocket("/video")
async def echo(ws: WebSocket):
    conn = WebSocketConnection(ws)
    await conn.run_viseme2video()
