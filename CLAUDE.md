# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python lives in `/venv/main/bin/python`. There is no `python` or `python3` on PATH — always use the full venv path or activate the venv first.

```bash
source /venv/main/bin/activate
# or invoke directly:
/venv/main/bin/python inference.py ...
```

FFmpeg must be installed system-wide (`ffmpeg` and `ffprobe` on PATH). The pipeline raises `ImportError` if it is missing.

## Key Commands

```bash
# Human portrait animation (CLI)
python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4

# Use a pre-computed motion template (much faster, no GPU needed for driving)
python inference.py -s assets/my_photo.png -d assets/examples/driving/talking.pkl

# Animals mode (requires XPose CUDA op to be built first)
cd src/utils/dependencies/XPose/models/UniPose/ops && python setup.py build install && cd -
python inference_animals.py -s assets/examples/source/s39.jpg -d assets/examples/driving/wink.pkl

# Gradio web UI
python app.py                        # humans
python app_animals.py                # animals
python app.py --flag_do_torch_compile  # 20-30% faster after first-run warmup (Linux only)

# macOS Apple Silicon — prefix every command with:
PYTORCH_ENABLE_MPS_FALLBACK=1 python inference.py ...

# Speed benchmark
python speed.py

# Flask API (custom, in this repo)
python run_api.py                    # starts on 0.0.0.0:5000
python run_api.py --preload          # load models at startup
```

All outputs go to `animations/` by default (or whatever `--output_dir` / `-o` is set to).

## Architecture

### Inference entry points

| File | Purpose |
|---|---|
| `inference.py` | CLI for humans — parses args via `tyro`, builds configs, runs pipeline |
| `inference_animals.py` | Same for animals |
| `app.py` / `app_animals.py` | Gradio web UIs |
| `run_api.py` | Flask REST API entry point |

### Config dataclasses (`src/config/`)

Three dataclasses are split from what was once one config. `inference.py` uses `partial_fields()` to fan out a single `ArgumentConfig` into all three:

- **`ArgumentConfig`** — user-facing CLI arguments (source, driving, output_dir, flags, crop params). Parsed by `tyro.cli()`.
- **`InferenceConfig`** — model checkpoint paths, FP16 flag, device, output FPS/format. All paths are resolved to absolute via `make_abs_path(__file__)`, so they are correct regardless of working directory.
- **`CropConfig`** — InsightFace/XPose roots, crop scale/ratio, face detection threshold.

When constructing these programmatically (e.g., in the Flask API), instantiate them directly as dataclasses — no `tyro` needed.

### Pipeline (`src/live_portrait_pipeline.py`)

`LivePortraitPipeline.__init__` loads two stateful objects:
- `self.live_portrait_wrapper` — `LivePortraitWrapper` (holds the 5 PyTorch models)
- `self.cropper` — `Cropper` (InsightFace face detector + landmark runner)

`pipeline.execute(args: ArgumentConfig)` runs the full inference loop: load source → detect/crop face → load/extract driving motion → per-frame warp-and-decode → paste-back → encode MP4.

### Model wrapper (`src/live_portrait_wrapper.py`)

`LivePortraitWrapper.__init__` loads **5 models** from `pretrained_weights/`:
- **F** — Appearance Feature Extractor (3D feature volume)
- **M** — Motion Extractor (keypoints, scale, rotation, expression)
- **W** — Warping Module
- **G** — SPADE Generator / Decoder
- **S** — Stitching & Retargeting Module (3 sub-networks: stitching, eye, lip)

This is the expensive one-time cost. Keep the pipeline object alive across requests.

### Driving inputs: videos vs templates

`.pkl` motion templates are pre-extracted keypoint trajectories. They are **much faster** than `.mp4` driving videos (no per-frame motion extraction needed) and protect the driver's identity. Named templates in `assets/examples/driving/`: `talking`, `laugh`, `shake_face`, `wink`, `shy`, `aggrieved`, `open_lip`.

Generate your own template from any driving video:
```bash
python inference.py -s <source> -d <driving.mp4>
# the pipeline auto-saves a .pkl alongside the output
```

### Flask API (`api/`)

`api/app.py` wraps the pipeline in Flask. The singleton `LivePortraitPipeline` is created once (double-checked locking) and serialized across requests with `_infer_lock` to avoid GPU OOM. Emotion presets are defined in `api/emotions.py` as `EmotionPreset` dataclasses mapping names → driving templates + `ArgumentConfig` overrides.

Endpoints:
- `GET /api/health` — liveness + model-loaded flag
- `GET /api/emotions` — list all preset names and descriptions
- `POST /api/warmup` — pre-load models without inference
- `POST /api/animate` — run animation; accepts multipart `source` file upload, `source_base64`, or `source_path`; query/body param `emotion`; returns `video/mp4`

### Pretrained weights

All weights live under `pretrained_weights/` (gitignored). Download via:
```bash
huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

Expected layout: `pretrained_weights/liveportrait/base_models/`, `pretrained_weights/liveportrait/retargeting_models/`, `pretrained_weights/insightface/`.

### Key flags

| Flag | Effect |
|---|---|
| `--flag_use_half_precision` (default True) | FP16 inference; set False if black boxes appear |
| `--flag_pasteback` (default True) | Composite cropped result back onto original image |
| `--flag_stitching` (default True) | Smooth seam between face crop and original; set False for large head movements |
| `--flag_relative_motion` (default True) | Drive relative to source pose rather than absolute |
| `--animation_region` | `all` / `exp` / `pose` / `lip` / `eyes` — restrict what is animated |
| `--driving_option` | `expression-friendly` (scales expression) or `pose-friendly` |
| `--driving_multiplier` | Scale driving motion magnitude |
| `--flag_do_torch_compile` | Compile models with `torch.compile` for ~20-30% speedup (Linux only, slow first run) |
