from __future__ import annotations

import asyncio
import json
import urllib.request

_KEYFRAMES_URL = "http://localhost:3000/animate/keyframes"


def _post_keyframes(visemes: list[dict]) -> bytes:
    body = json.dumps({"visemes": visemes}).encode()
    req = urllib.request.Request(
        _KEYFRAMES_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=520) as resp:
        return resp.read()


async def speech_to_video(
    visemes: list[dict],
    source: str | None = None,
) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _post_keyframes, visemes)
