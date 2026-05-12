from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from sdk.controllers.liveportrait import _get_pipeline

if TYPE_CHECKING:
    from sdk.controllers.ws import WebSocketConnection


async def stream_frame(conn: WebSocketConnection):
    """Dummy video stream — sends placeholder chunk events until the client disconnects."""
    try:
        for frame_index in range(25):
            await conn.send_json({
                "session_id": conn.session_id,
                "event": "frame",
                "frame_index": frame_index,
                "data": None,  # replace with real frame bytes (base64 or binary send)
            })
            await asyncio.sleep(1 / 25)  # 25 fps pacing
    except Exception:
        pass
