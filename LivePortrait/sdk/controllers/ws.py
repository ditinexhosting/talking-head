import json
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from sdk.controllers.tts import text_to_speech_kokoro, text_to_speech_kokoro_sample
from sdk.controllers.streamFrame import stream_frame


class ConnectionManager:
    def __init__(self):
        self._connections: list["WebSocketConnection"] = []

    def add(self, conn: "WebSocketConnection"):
        self._connections.append(conn)

    def remove(self, conn: "WebSocketConnection"):
        self._connections.remove(conn)

    async def broadcast(self, message: str):
        for conn in self._connections:
            await conn.send(message)


manager = ConnectionManager()


class WebSocketConnection:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.session_id = str(uuid.uuid4())

    async def accept(self):
        await self.ws.accept()
        manager.add(self)

    async def close(self):
        manager.remove(self)

    async def send(self, message: str):
        await self.ws.send_text(message)

    async def receive(self) -> str:
        return await self.ws.receive_text()

    async def send_json(self, payload: dict):
        await self.ws.send_text(json.dumps(payload))

    async def send_bytes(self, data: bytes):
        await self.ws.send_bytes(data)

    async def run_tts(self):
        await self.accept()
        await self.send_json({"session_id": self.session_id, "event": "connected"})
        try:
            while True:
                raw = await self.receive()
                try:
                    data = json.loads(raw)
                    text = data.get("text", "")
                except (json.JSONDecodeError, AttributeError):
                    await self.send_json({"session_id": self.session_id, "error": "invalid JSON"})
                    continue

                if not text:
                    await self.send_json({"session_id": self.session_id, "error": "missing 'text' field"})
                    continue

                #await text_to_speech_kokoro(self, text)
                await text_to_speech_kokoro_sample(self)
                await self.send_json({"session_id": self.session_id, "event": "done"})
        except WebSocketDisconnect:
            await self.close()

    async def run_viseme2video(self):
        await self.accept()
        await self.send_json({"session_id": self.session_id, "event": "connected"})
        try:
            while True:
                raw = await self.receive()
                try:
                    data = json.loads(raw)
                    text = data.get("visemes", None)
                except (json.JSONDecodeError, AttributeError):
                    await self.send_json({"session_id": self.session_id, "error": "invalid JSON"})
                    continue

                if text is None:
                    await self.send_json({"session_id": self.session_id, "error": "missing 'visemes' field"})
                    continue

                await stream_frame(self)
                await self.send_json({"session_id": self.session_id, "event": "done"})
        except WebSocketDisconnect:
            await self.close()
