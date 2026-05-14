from __future__ import annotations

import io
import itertools
import os
import pickle
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pillow_avif  # registers AVIF format with Pillow
from PIL import Image
from fastapi.responses import FileResponse, Response

from src.config.argument_config import ArgumentConfig
from sdk.controllers.liveportrait import _get_pipeline
from sdk.controllers.speech2video.motion import neutral_keyframes, build_template, add_blinks, add_talks

_encoder_pool = ThreadPoolExecutor(max_workers=4)


def _encode_webp(frame_np, quality=80):
    buf = io.BytesIO()
    Image.fromarray(frame_np).save(buf, format="WEBP", quality=quality, method=4)
    return buf.getvalue()


def _encode_fmp4(frames, fps=25):
    # Fragmented MP4: empty_moov writes the init segment (ftyp+moov) inline,
    # frag_keyframe starts a new moof+mdat fragment on every keyframe. The
    # resulting bytes are a single self-contained, playable fMP4 file.
    # Encoder: NVIDIA NVENC (h264_nvenc) — dedicated GPU silicon, ~1000+ fps
    # at 256x256, runs in parallel with model inference on CUDA cores.
    height, width = frames[0].shape[:2]
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "h264_nvenc",
            "-preset", "p1",
            "-tune", "ull",
            "-rc", "cbr",
            "-b:v", "2M",
            "-pix_fmt", "yuv420p",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4",
            "pipe:1",
        ],
        input=b"".join(f.tobytes() for f in frames),
        capture_output=True,
        check=True,
    )
    return proc.stdout


_DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "uploads", "my_photo.png"
)

_FPS = 25
_STREAM_CHUNK_SECONDS = 2
_STREAM_CHUNK_FRAMES = max(1, round(_FPS * _STREAM_CHUNK_SECONDS))
_NEUTRAL_SECONDS = 2

_VISEME_SEQUENCE = {
    "visemes": [
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
}


def warmup_stream(pipeline) -> None:
    keyframes = neutral_keyframes(seconds=2 / _FPS, fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

    tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    try:
        pickle.dump(template, tpl_file)
        tpl_file.close()

        args = ArgumentConfig(
            source=_DEFAULT_SOURCE,
            driving=tpl_file.name,
            output_dir=tempfile.mkdtemp(),
            flag_pasteback=False,
        )
        frame_gen, _ = pipeline.execute_streaming(args)
        for _ in frame_gen:
            pass
    finally:
        os.unlink(tpl_file.name)


def run_liveportrait() -> Response:
    t_req_start = time.perf_counter()

    _total_frames = round(_VISEME_SEQUENCE["visemes"][-1]["end"] * _FPS) + 15
    _total_seconds = _total_frames / _FPS
    keyframes = neutral_keyframes(seconds=_total_seconds, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    keyframes = add_talks(keyframes, _VISEME_SEQUENCE["visemes"], fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

    tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    try:
        pickle.dump(template, tpl_file)
        tpl_file.close()

        args = ArgumentConfig(
            source=_DEFAULT_SOURCE,
            driving=tpl_file.name,
            output_dir=tempfile.mkdtemp(),
            flag_pasteback=False,
        )
        frame_gen, _ = _get_pipeline().execute_streaming(args)

        # TEMPORARY: cap at the first 25 frames; encode and return the first 8
        # as a single fMP4 chunk. Remaining frames within the 25 are discarded
        # — full-stream handling comes later.
        frames = []
        t_gen_start = time.perf_counter()
        for frame in itertools.islice(frame_gen, 25):
            frames.append(frame)
            if len(frames) >= 8:
                break
        gen_ms = (time.perf_counter() - t_gen_start) * 1000.0
    finally:
        os.unlink(tpl_file.name)

    t_enc_start = time.perf_counter()
    chunk = _encode_fmp4(frames, fps=_FPS)
    enc_ms = (time.perf_counter() - t_enc_start) * 1000.0

    total_ms = (time.perf_counter() - t_req_start) * 1000.0
    print(
        f"[liveportrait] frames={len(frames)} gen={gen_ms:7.2f}ms "
        f"encode={enc_ms:7.2f}ms total={total_ms:7.2f}ms "
        f"chunk={len(chunk)}B",
        flush=True,
    )

    return Response(content=chunk, media_type="video/mp4")


def run_liveportrait_stream():
    t_start = time.perf_counter()

    _total_frames = round(_VISEME_SEQUENCE["visemes"][-1]["end"] * _FPS) + 15
    _total_seconds = _total_frames / _FPS
    keyframes = neutral_keyframes(seconds=_total_seconds, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    keyframes = add_talks(keyframes, _VISEME_SEQUENCE["visemes"], fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

    tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    proc = None
    try:
        pickle.dump(template, tpl_file)
        tpl_file.close()

        args = ArgumentConfig(
            source=_DEFAULT_SOURCE,
            driving=tpl_file.name,
            output_dir=tempfile.mkdtemp(),
            flag_pasteback=False,
        )
        frame_gen, _ = _get_pipeline().execute_streaming(args)

        try:
            first_frame = next(frame_gen)
        except StopIteration:
            return
        height, width = first_frame.shape[:2]

        # One ffmpeg process for the entire stream. Init segment (ftyp+moov)
        # is emitted once at the start; moof+mdat fragments follow every
        # _STREAM_CHUNK_FRAMES frames (2 seconds at the current FPS). Bytes are
        # forwarded byte-for-byte to the WebSocket — MSE on the client side
        # parses fragments incrementally.
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

        writer = threading.Thread(target=_feed_ffmpeg, daemon=True)
        writer.start()

        chunk_count = 0
        bytes_out = 0
        first_chunk_ms = None
        last_t = t_start
        while True:
            t_read_start = time.perf_counter()
            data = proc.stdout.read(65536)
            if not data:
                break
            now = time.perf_counter()
            chunk_count += 1
            bytes_out += len(data)
            elapsed_ms = (now - t_start) * 1000.0
            gap_ms = (now - last_t) * 1000.0
            last_t = now
            if first_chunk_ms is None:
                first_chunk_ms = elapsed_ms
                print(
                    f"[stream] FIRST chunk at {elapsed_ms:.2f}ms "
                    f"(frames_in={stats['frames_in']}, size={len(data)}B)",
                    flush=True,
                )
            print(
                f"[stream] chunk={chunk_count:04d} size={len(data):6d}B "
                f"gap={gap_ms:7.2f}ms elapsed={elapsed_ms:8.2f}ms "
                f"frames_in={stats['frames_in']}",
                flush=True,
            )
            yield data

        writer.join(timeout=10)
        proc.wait(timeout=5)

        total_s = time.perf_counter() - t_start
        fps = stats["frames_in"] / total_s if total_s > 0 else 0.0
        print(
            f"[stream] DONE frames={stats['frames_in']} chunks={chunk_count} "
            f"bytes={bytes_out} first_chunk={first_chunk_ms:.2f}ms "
            f"total={total_s * 1000.0:.2f}ms ({total_s:.2f}s) "
            f"→ {fps:.1f} fps",
            flush=True,
        )
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
        os.unlink(tpl_file.name)


def run_liveportrait_video() -> FileResponse:
    _total_frames = round(_VISEME_SEQUENCE["visemes"][-1]["end"] * _FPS) + 15
    _total_seconds = _total_frames / _FPS
    keyframes = neutral_keyframes(seconds=_total_seconds, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    keyframes = add_talks(keyframes, _VISEME_SEQUENCE["visemes"], fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

    tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    try:
        pickle.dump(template, tpl_file)
        tpl_file.close()

        args = ArgumentConfig(
            source=_DEFAULT_SOURCE,
            driving=tpl_file.name,
            output_dir=tempfile.mkdtemp(),
        )
        wfp, _ = _get_pipeline().execute(args)
    finally:
        os.unlink(tpl_file.name)

    return FileResponse(wfp, media_type="video/mp4", filename="output.mp4")
