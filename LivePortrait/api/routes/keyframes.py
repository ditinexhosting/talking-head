import os
import os.path as osp
import pickle
import shutil
import subprocess
import tempfile
import threading
from flask import Blueprint, Response, request, jsonify, stream_with_context
from api.pipeline import get_pipeline
from api.utils.motion import add_blinks, add_talks, build_template, neutral_keyframes
from src.config.argument_config import ArgumentConfig

keyframes_bp = Blueprint("keyframes", __name__)

_UPLOADS = osp.abspath(osp.join(osp.dirname(__file__), "..", "uploads"))
_SOURCE = osp.join(_UPLOADS, "my_photo.png")

_FPS = 25
_PIPELINE_LOCK = threading.Lock()

_VISEME_SEQUENCE = [
        {
            "start": 0.0,
            "end": 0.01,
            "duration": 0.01,
            "viseme": "X",
            "description": "Silence / rest"
        },
        {
            "start": 0.01,
            "end": 0.05,
            "duration": 0.04,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 0.05,
            "end": 0.09,
            "duration": 0.04,
            "viseme": "G",
            "description": "Relaxed open mouth  (a, i)"
        },
        {
            "start": 0.09,
            "end": 0.16,
            "duration": 0.07,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 0.16,
            "end": 0.51,
            "duration": 0.35,
            "viseme": "E",
            "description": "Tongue up  (l)"
        },
        {
            "start": 0.51,
            "end": 0.58,
            "duration": 0.07,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 0.58,
            "end": 0.65,
            "duration": 0.07,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 0.65,
            "end": 0.72,
            "duration": 0.07,
            "viseme": "F",
            "description": "Rounded / puckered lips  (w, q)"
        },
        {
            "start": 0.72,
            "end": 0.79,
            "duration": 0.07,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 0.79,
            "end": 0.93,
            "duration": 0.14,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 0.93,
            "end": 1.01,
            "duration": 0.08,
            "viseme": "A",
            "description": "Closed lips  (m, b, p)"
        },
        {
            "start": 1.01,
            "end": 1.38,
            "duration": 0.37,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 1.38,
            "end": 1.45,
            "duration": 0.07,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 1.45,
            "end": 1.73,
            "duration": 0.28,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 1.73,
            "end": 1.94,
            "duration": 0.21,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 1.94,
            "end": 2.08,
            "duration": 0.14,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 2.08,
            "end": 2.34,
            "duration": 0.26,
            "viseme": "X",
            "description": "Silence / rest"
        },
        {
            "start": 2.34,
            "end": 2.53,
            "duration": 0.19,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 2.53,
            "end": 2.61,
            "duration": 0.08,
            "viseme": "A",
            "description": "Closed lips  (m, b, p)"
        },
        {
            "start": 2.61,
            "end": 2.83,
            "duration": 0.22,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 2.83,
            "end": 3.11,
            "duration": 0.28,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 3.11,
            "end": 3.19,
            "duration": 0.08,
            "viseme": "A",
            "description": "Closed lips  (m, b, p)"
        },
        {
            "start": 3.19,
            "end": 3.61,
            "duration": 0.42,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 3.61,
            "end": 3.66,
            "duration": 0.05,
            "viseme": "X",
            "description": "Silence / rest"
        }
    ]


@keyframes_bp.post("/animate/keyframes")
def animate_keyframes():
    body = request.get_json(silent=True) or {}
    viseme_sequence = body.get("visemes") or _VISEME_SEQUENCE

    if not isinstance(viseme_sequence, list) or len(viseme_sequence) == 0:
        return jsonify({"error": "visemes must be a non-empty array"}), 400

    # 1 frame lead-in + 3 frames tail + 3 frames pause buffer = 7 extra; round() on sequence end
    _TOTAL_FRAMES = round(viseme_sequence[-1]["end"] * _FPS) + 15
    _TOTAL_SECONDS = _TOTAL_FRAMES / _FPS
    # Generate blank keyframe sequence
    keyframes = neutral_keyframes(seconds=_TOTAL_SECONDS, fps=_FPS)
    # Add blink animation
    keyframes = add_blinks(keyframes, fps=_FPS)
    # Add lips animation
    keyframes = add_talks(keyframes, viseme_sequence, fps=_FPS)
    # Build the template
    template = build_template([kf.to_dict() for kf in keyframes], _FPS)

    # json_path = osp.join(_UPLOADS, "motion.json")
    # with open(json_path, "w") as f:
    #     json.dump([kf.to_dict() for kf in keyframes], f, indent=2)

    # template_json_path = osp.join(_UPLOADS, "template.json")
    # with open(template_json_path, "w") as f:
    #     json.dump(template_to_json(template), f, indent=2)

    tmp_dir = tempfile.mkdtemp()
    proc = None
    proc_ready = threading.Event()
    setup_error = [None]

    def feed_frames():
        nonlocal proc
        try:
            tmp_pkl = osp.join(tmp_dir, "motion.pkl")
            output_dir = osp.join(tmp_dir, "outputs")
            os.makedirs(output_dir)
            with open(tmp_pkl, "wb") as f:
                pickle.dump(template, f)

            args = ArgumentConfig(source=_SOURCE, driving=tmp_pkl, output_dir=output_dir)
            pipeline = get_pipeline()

            with _PIPELINE_LOCK:
                frames, fps = pipeline.execute_streaming(args)
                first_frame = next(frames)
                h, w = first_frame.shape[:2]

                proc = subprocess.Popen(
                    [
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{w}x{h}", "-r", str(fps),
                        "-i", "pipe:0",
                        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "frag_keyframe+empty_moov",
                        "-f", "mp4", "pipe:1",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                proc_ready.set()

                proc.stdin.write(first_frame.tobytes())
                for frame in frames:
                    proc.stdin.write(frame.tobytes())
        except Exception as e:
            setup_error[0] = e
            proc_ready.set()
        finally:
            if proc and proc.stdin:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

    feeder = threading.Thread(target=feed_frames, daemon=True)
    feeder.start()
    proc_ready.wait()

    if setup_error[0]:
        feeder.join()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"error": str(setup_error[0])}), 500

    def stream_output():
        try:
            while chunk := proc.stdout.read(8192):
                yield chunk
        finally:
            feeder.join()
            proc.wait()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return Response(
        stream_with_context(stream_output()),
        mimetype="video/mp4",
        headers={
            "Content-Disposition": "inline; filename=animation.mp4",
            "X-Accel-Buffering": "no",
        },
    )
