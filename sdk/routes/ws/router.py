from fastapi import APIRouter, WebSocket
from controllers.ws_controller import WebSocketConnection

router = APIRouter(tags=["websocket"])


@router.websocket("/audio")
async def echo(ws: WebSocket):
    conn = WebSocketConnection(ws)
    await conn.run_tts()
