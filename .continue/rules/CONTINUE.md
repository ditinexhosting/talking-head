# LivePortrait Project Guide

## 1. Project Overview

**LivePortrait** is a Python/PyTorch project for portrait animation: it animates a source portrait (or, for humans, a source video) using motion from a driving video, image, or precomputed motion template. The repository includes:

- a CLI workflow for humans and animals
- Gradio web UIs for interactive use
- a lightweight Flask API wrapper for preset-based animation
- utilities for motion-template generation, cropping, paste-back, and speed benchmarking

### Key technologies

- **Python 3.10** (recommended in `readme.md`)
- **PyTorch** for model inference
- **Gradio** for the web UI (`app.py`, `app_animals.py`)
- **Flask** for the custom API (`api/app.py`, `run_api.py`)
- **OpenCV / FFmpeg** for image and video processing
- **ONNX Runtime / InsightFace** for detection and landmarks
- **XPose** for animal landmark detection (Linux/Windows + NVIDIA only)
- **tyro** for CLI/dataclass-based argument parsing

### High-level architecture

The project is organized around a small set of entrypoints that all fan into shared pipeline code:

- **CLI**: `inference.py`, `inference_animals.py`
- **Web UI**: `app.py`, `app_animals.py`
- **API**: `run_api.py` -> `api/app.py`
- **Core pipelines**: `src/live_portrait_pipeline.py`, `src/live_portrait_pipeline_animal.py`
- **Model wrappers**: `src/live_portrait_wrapper.py`
- **Config dataclasses**: `src/config/`
- **Low-level utilities and model modules**: `src/utils/`, `src/modules/`

At runtime, the flow is roughly:

1. Parse user-facing arguments into config dataclasses
2. Load pretrained models and landmark/cropping helpers
3. Read and crop the source input
4. Read the driving input or load a `.pkl` motion template
5. Extract motion / keypoint data
6. Warp and decode frames with the neural pipeline
7. Optionally paste the generated crop back into the original image space
8. Encode the final image/video output

---

## 2. Getting Started

### Prerequisites

From `readme.md` and the codebase:

- **Python 3.10**
- **git**
- **FFmpeg** (`ffmpeg` and `ffprobe` must be on `PATH`)
- **conda** is recommended in the upstream docs
- **NVIDIA GPU** strongly recommended for normal use
- **Apple Silicon macOS** is supported for human mode only via MPS fallback
- **XPose build step** is required for animal mode on Linux/Windows

### Installation

1. Clone the repository.
2. Create and activate an environment.
3. Install dependencies.
4. Download pretrained weights into `pretrained_weights/`.

Recommended setup from the repo docs:

- `pip install -r requirements.txt` for Linux/Windows
- `pip install -r requirements_macOS.txt` for Apple Silicon macOS

Important dependency split:

- `requirements.txt` includes `requirements_base.txt` plus GPU-oriented extras like `onnxruntime-gpu` and `flask`
- `requirements_macOS.txt` uses CPU/MPS-friendly Torch wheels and `onnxruntime-silicon`

### Pretrained weights

The code expects model assets under `pretrained_weights/`, especially:

- `pretrained_weights/liveportrait/...`
- `pretrained_weights/liveportrait_animals/...`
- `pretrained_weights/insightface/...`

The expected structure is documented in `assets/docs/directory-structure.md`.

### Basic usage examples

#### Human CLI inference

- Run defaults: `python inference.py`
- Custom source/driving: `python inference.py -s <source> -d <driving>`

Examples of supported human inputs:

- source image + driving video
- source video + driving video
- source image/video + driving `.pkl` template
- source image + driving image

#### Animal CLI inference

- `python inference_animals.py -s <animal_source> -d <driving>`

Animal mode is more constrained:

- tested on Linux/Windows with NVIDIA GPU
- requires XPose setup
- source is image-based

#### Gradio apps

- `python app.py` for humans
- `python app_animals.py` for animals

#### API

- `python run_api.py`
- optional preload: `python run_api.py --preload`

API endpoints found in `api/app.py`:

- `GET /api/health`
- `GET /api/emotions`
- `POST /api/warmup`
- `POST /api/animate`

### Running tests

**Needs verification:** no automated test suite or `tests/` directory was found in the repository snapshot, and no `pytest.ini`/`tox.ini` was present.

Current validation appears to be mostly manual:

- run example inferences from `readme.md`
- launch Gradio and verify sample examples
- call the Flask API endpoints
- use `python speed.py` for module-level performance checks

---

## 3. Project Structure

### Top-level directories and files

- `src/` - core implementation
- `api/` - Flask API wrapper and emotion presets
- `assets/` - examples, Gradio markdown snippets, docs, and sample media
- `pretrained_weights/` - model weights and InsightFace assets
- `app.py` - human Gradio entrypoint
- `app_animals.py` - animal Gradio entrypoint
- `inference.py` - human CLI entrypoint
- `inference_animals.py` - animal CLI entrypoint
- `run_api.py` - Flask API entrypoint
- `speed.py` - benchmarking script
- `requirements*.txt` - dependency definitions
- `readme.md` - primary upstream documentation

### `src/` breakdown

- `src/config/`
  - `argument_config.py` - user-facing CLI/runtime options
  - `inference_config.py` - model paths and inference settings
  - `crop_config.py` - detector/crop/landmark settings
  - `models.yaml` - model architecture parameters
- `src/live_portrait_pipeline.py` - main human animation pipeline
- `src/live_portrait_pipeline_animal.py` - animal animation pipeline
- `src/live_portrait_wrapper.py` - loads and runs the neural modules
- `src/gradio_pipeline.py` - UI-specific orchestration and retargeting helpers
- `src/modules/` - neural network building blocks
- `src/utils/` - IO, video processing, cropping, landmarking, filtering, helpers
- `src/utils/dependencies/` - vendored third-party components such as InsightFace and XPose

### `api/` breakdown

- `api/app.py` - Flask app with lazy singleton pipeline loading and serialized inference
- `api/emotions.py` - emotion preset definitions mapped to driving templates and settings
- `api/nod_patterns.py` - additional API-related motion pattern helpers

### Important configuration files

- `requirements.txt` - default dependency entrypoint
- `requirements_base.txt` - shared core dependencies
- `requirements_macOS.txt` - Apple Silicon/macOS dependency variant
- `src/config/models.yaml` - model topology parameters
- `src/config/*.py` - runtime dataclass configs
- `CLAUDE.md` - local repo guidance for AI/dev tooling; useful as supplemental context

### Important runtime assets

- `assets/examples/source/` - sample source portraits and videos
- `assets/examples/driving/` - sample driving videos, images, and `.pkl` templates
- `src/utils/resources/` - internal resources such as `mask_template.png` and `lip_array.pkl`
- `assets/my_photo.png` - default source used by the custom Flask API

---

## 4. Development Workflow

### Coding conventions observed in the codebase

The repo does not expose a formal linter/formatter config in the inspected files, but the implementation follows a few strong patterns:

- keep **entrypoints thin** and push logic into `src/`
- use **dataclasses** for configuration (`ArgumentConfig`, `InferenceConfig`, `CropConfig`)
- use **tyro** for CLI parsing
- centralize heavy model handling inside **wrapper/pipeline classes**
- separate **human** and **animal** flows while sharing utility layers where possible
- use utility modules for IO/cropping/video concerns instead of duplicating logic

**Needs verification:** no explicit Black/Ruff/isort/flake8 config was found.

### Testing approach

Since no formal test suite was found, the practical workflow appears to be:

1. install dependencies and weights
2. run a known-good example from `assets/examples/`
3. verify CLI output in `animations/`
4. verify UI behavior in Gradio
5. verify API responses if touching `api/`
6. optionally benchmark with `speed.py`

If you add new features, prefer creating reproducible example inputs or templates so others can validate behavior quickly.

### Build and deployment process

There is no Dockerfile, Makefile, or CI workflow visible in the inspected files.

Current "build" steps are effectively:

1. install Python dependencies
2. download pretrained weights
3. ensure FFmpeg is installed
4. for animal mode, build the XPose CUDA op
5. run one of the entrypoints (`inference.py`, `app.py`, `run_api.py`, etc.)

For API deployment, `api/app.py` already follows a helpful pattern:

- lazy one-time model initialization
- cached precomputed source image
- serialized inference via lock to reduce GPU OOM risk

**Needs verification:** production deployment likely benefits from running Flask behind a real WSGI/ASGI-serving layer or process supervisor, but that is not documented in the repo.

### Contribution guidelines

No `CONTRIBUTING.md` was found, so the following are project-informed recommendations:

- keep new entrypoints minimal; prefer extending existing pipelines/wrappers
- add new runtime options through the config dataclasses instead of ad hoc globals
- reuse `assets/examples/` for demos and regression checks
- avoid reloading models per request/process if working on server mode
- treat `.pkl` motion templates as first-class artifacts for reusable motions and privacy-preserving workflows
- document platform constraints clearly, especially for animal mode and CUDA-dependent code

---

## 5. Key Concepts

### Domain terms

- **Source**: the portrait image or video to animate
- **Driving input**: the video/image/template providing motion
- **Motion template (`.pkl`)**: precomputed driving motion that avoids reprocessing a raw video
- **Paste-back**: compositing the generated face crop back onto the original source frame
- **Stitching**: learned correction that improves consistency between source and driven keypoints
- **Retargeting**: explicitly controlling eyes/lips/pose/expression rather than only replaying driving motion

### Core model abstractions

`src/live_portrait_wrapper.py` exposes the key model components described in the paper and comments:

- **F** - appearance feature extractor
- **M** - motion extractor
- **W** - warping module
- **G** - SPADE generator/decoder
- **S** - stitching and retargeting module

These are loaded once into `LivePortraitWrapper` / `LivePortraitWrapperAnimal` and then used by the pipeline classes.

### Design patterns used

- **Wrapper pattern** for model loading/inference (`LivePortraitWrapper`)
- **Pipeline/orchestrator pattern** for end-to-end animation (`LivePortraitPipeline`)
- **Config-as-dataclass pattern** for typed runtime settings
- **Lazy singleton initialization** in the Flask API (`get_pipeline()`)
- **Lock-based serialization** in the API to avoid concurrent GPU overcommit
- **Template caching** via `.pkl` motion files for repeatable faster runs
- **Source precomputation cache** in `LivePortraitPipeline.precompute_source()` for repeated API calls

### Important runtime switches

Some of the most important flags in `ArgumentConfig` / `InferenceConfig` are:

- `flag_use_half_precision` - faster/lower-memory inference; can cause artifacts on some hardware
- `flag_stitching` - improves stability for small movements; may be undesirable for large movement cases
- `flag_pasteback` - returns result in original image space
- `flag_relative_motion` - drive relative to the source state
- `animation_region` - limit animation to `exp`, `pose`, `lip`, `eyes`, or `all`
- `driving_option` - `expression-friendly` vs `pose-friendly`
- `driving_multiplier` - scales motion strength
- `flag_crop_driving_video` - auto-crops driving video when needed
- `flag_do_torch_compile` - optional compile-time optimization for supported environments

---

## 6. Common Tasks

### Run a quick human animation

1. Confirm FFmpeg is installed.
2. Confirm pretrained weights exist under `pretrained_weights/`.
3. Run `python inference.py`.
4. Check `animations/` for the generated result.

### Animate a custom portrait with a custom driving video

1. Choose a source image or video.
2. Choose a driving video.
3. Run `python inference.py -s <source> -d <driving>`.
4. If the driving video is not tightly framed or square, enable cropping.
5. Adjust crop parameters (`scale`, `vx_ratio`, `vy_ratio`) if the face is not aligned well.

### Reuse a motion template for speed/privacy

1. Run a normal inference with a raw driving video.
2. Let the pipeline auto-save a `.pkl` motion template.
3. Re-run future animations using the `.pkl` as the driving input.

This is one of the most important productivity patterns in the repo.

### Launch the human Gradio app

1. Run `python app.py`.
2. Open the local Gradio URL.
3. Try the provided examples from `assets/examples/`.
4. Use the UI sliders for pose/expression/lip/eye retargeting.

### Launch the animal Gradio app

1. Make sure animal dependencies and XPose build steps are complete.
2. Run `python app_animals.py`.
3. Use a sample animal image and a driving template/video.

### Serve the preset API

1. Ensure `assets/my_photo.png` is the desired source image.
2. Run `python run_api.py --preload` for lower first-request latency.
3. Query `GET /api/emotions` to discover supported presets.
4. Call `POST /api/animate` with an emotion name and optional multiplier.

### Add a new API emotion preset

1. Open `api/emotions.py`.
2. Add a new `EmotionPreset` entry to `EMOTIONS`.
3. Point it to a valid driving template/video under `assets/examples/driving/`.
4. Set `animation_region`, `driving_option`, `driving_multiplier`, and `flag_stitching` appropriately.
5. Test through `/api/emotions` and `/api/animate`.

### Benchmark performance

1. Make sure the required weights and GPU environment are available.
2. Run `python speed.py`.
3. Use the numbers to compare environment changes or `torch.compile` experiments.

---

## 7. Troubleshooting

### FFmpeg errors on startup

Symptoms:

- `ImportError` complaining that FFmpeg is not installed

Fix:

- install FFmpeg and ensure both `ffmpeg` and `ffprobe` are on `PATH`
- note that entrypoints explicitly check for this before running

### Missing weights / model load failures

Symptoms:

- errors loading `.pth` or `.onnx` files
- failures during pipeline or cropper initialization

Fix:

- verify the `pretrained_weights/` structure matches `assets/docs/directory-structure.md`
- confirm both `liveportrait` and `insightface` assets exist
- for animal mode, also verify `liveportrait_animals` and `xpose.pth`

### No face detected

Symptoms:

- exceptions such as `No face detected in the source image!`
- poor crop quality or failed inference

Fix:

- try a more frontal, higher-resolution source image
- adjust crop parameters (`scale`, `vx_ratio`, `vy_ratio`)
- lower/adjust detection threshold if needed
- keep the face prominent and unobstructed

### Black boxes or precision-related artifacts

Symptoms:

- corrupted output regions during inference

Fix:

- set `flag_use_half_precision=False`
- this is explicitly suggested by comments in `ArgumentConfig`

### Animal mode does not work on macOS

This is expected based on the docs/code:

- XPose is not supported on macOS in this project
- human mode can use Apple Silicon with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- animal mode is intended for Linux/Windows with NVIDIA GPU

### Non-square or poorly framed driving video gives bad results

Fix:

- use `flag_crop_driving_video`
- manually crop to a square head-focused driving clip when possible
- prefer a neutral/frontal first frame
- minimize shoulder/body motion in the driving video

### API first request is slow

This is expected because models load lazily.

Fix:

- start the API with `--preload`
- keep the process warm instead of restarting frequently
- avoid concurrent GPU-heavy requests; the API intentionally serializes inference

### Out-of-memory during server use

Fix:

- keep a single shared pipeline instance
- serialize inference requests
- reduce concurrency at the process/server level
- try smaller sources or disable costly options if needed

### No automated tests found

If you are changing behavior, create a manual validation checklist:

- one human image -> video run
- one human video -> video run
- one `.pkl` template run
- one Gradio smoke test
- one API smoke test
- one animal-mode run if touching shared cropping/wrapper logic

---

## 8. References

### Repository docs

- `readme.md` - primary setup and usage guide
- `readme_zh_cn.md` - Chinese version of the main README
- `assets/docs/directory-structure.md` - required weights layout
- `assets/docs/how-to-install-ffmpeg.md` - FFmpeg installation help
- `assets/docs/speed.md` - benchmark reference results
- `assets/docs/changelog/` - feature updates and behavior changes

### External references already linked by the project

- Paper: https://arxiv.org/pdf/2407.03168
- Project homepage: https://liveportrait.github.io
- Hugging Face model/demo: https://huggingface.co/spaces/KlingTeam/LivePortrait
- Upstream repository: https://github.com/KlingTeam/LivePortrait
- PyTorch install matrix: https://pytorch.org/get-started/previous-versions
- FFmpeg: https://ffmpeg.org/download.html
- InsightFace: https://github.com/deepinsight/insightface
- XPose: https://github.com/IDEA-Research/X-Pose

### Notes needing verification

- formal coding-style/linting policy
- official contribution workflow
- production deployment recommendations for the Flask API
- existence of any private/internal test or release process not present in the repo snapshot
