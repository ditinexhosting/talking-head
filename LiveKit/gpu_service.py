import logging
import os

import httpx
from fastapi import HTTPException

GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "http://182.224.239.168:54582/api")
GPU_REQUEST_TIMEOUT = 5.0  # seconds

logger = logging.getLogger(__name__)


async def notify_gpu_join(room_id: str, vps_callback_base: str) -> None:
    """Tell the GPU agent to join a LiveKit room.

    Raises HTTP 503 if the GPU server is unreachable or returns non-200.
    """
    callback_url = f"{vps_callback_base}/session-ended/{room_id}"
    payload = {"room_id": room_id, "callback_url": callback_url}

    # LOAD BALANCER HOOK: replace GPU_SERVER_URL with a service-discovery lookup
    # or a pool selector when multiple GPU nodes are available.
    target_url = f"{GPU_SERVER_URL}/join"

    try:
        async with httpx.AsyncClient(timeout=GPU_REQUEST_TIMEOUT) as client:
            response = await client.post(target_url, json=payload)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        logger.error(
            "GPU server unreachable at %s for room %s: %s",
            target_url,
            room_id,
            exc,
        )
        raise HTTPException(status_code=503, detail="GPU server unavailable") from exc

    if response.status_code != 200:
        logger.error(
            "GPU server returned %s for room %s: %s",
            response.status_code,
            room_id,
            response.text,
        )
        raise HTTPException(status_code=503, detail="GPU server unavailable")

    logger.info("GPU server accepted join request for room %s", room_id)
