import asyncio
import subprocess
import threading

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

from sdk.controllers.rest import get_root, get_health
from sdk.controllers.livekit_agent import run_agent

router = APIRouter(prefix="/api", tags=["rest"])


@router.get("/")
def root():
    return get_root()


@router.get("/health")
def health():
    return get_health()


class JoinBody(BaseModel):
    room_id: str
    callback_url: str | None = None


@router.post("/join")
async def join_room(body: JoinBody):
    asyncio.create_task(run_agent(body.room_id, body.callback_url))
    return {"status": "ok"}


_STREAM_CHUNK_FRAMES = 25


@router.post("/animate")
def animate():
    from sdk.controllers.speech2video.video import liveportrait_frame_gen, _FPS

    frame_gen = liveportrait_frame_gen()
    first_frame = next(frame_gen)
    height, width = first_frame.shape[:2]

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(_FPS),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-g", str(_STREAM_CHUNK_FRAMES),
            "-keyint_min", str(_STREAM_CHUNK_FRAMES),
            "-sc_threshold", "0",
            "-flush_packets", "1",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    stats = {"frames_in": 0}

    def _feed_ffmpeg():
        try:
            proc.stdin.write(first_frame.tobytes())
            stats["frames_in"] = 1
            for frame in frame_gen:
                proc.stdin.write(frame.tobytes())
                stats["frames_in"] += 1
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    feeder = threading.Thread(target=_feed_ffmpeg, daemon=True)
    feeder.start()

    video_bytes = proc.stdout.read()
    proc.wait()
    feeder.join()

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "Content-Disposition": "inline; filename=animation.mp4",
            "Content-Length": str(len(video_bytes)),
        },
    )
