# LivePortrait

Efficient portrait animation system for real-time facial reenactment. Supports human portraits and animals (cats, dogs). Built on a deep learning pipeline with 5 neural network components. Originally from Kuaishou Technology / USTC / Fudan University.

## Entry Points

```bash
# CLI — human portraits
python inference.py -s source.jpg -d driving.mp4

# CLI — animals
python inference_animals.py -s source.jpg -d driving.pkl

# Gradio web UI
python app.py           # human mode
python app_animals.py   # animal mode

# Benchmarking
python speed.py
```

## Project Structure

```
src/
  config/               # Argument, crop, inference configs + models.yaml
  modules/              # Neural network modules (F, M, W, G, S, R)
  utils/                # Landmark runners, cropper, video I/O, filters
  live_portrait_wrapper.py     # Loads all 5 models, handles device/precision
  live_portrait_pipeline.py    # Main animation pipeline (human)
  live_portrait_pipeline_animal.py
  gradio_pipeline.py           # Web UI orchestration
pretrained_weights/
  liveportrait/         # Human models (base_models/ + retargeting_models/)
  liveportrait_animals/ # Animal models (v1.0 and v1.1)
assets/examples/        # Sample source images and driving videos
```

## Model Architecture

Five components defined in `src/config/models.yaml`:

| Symbol | Module | File | Role |
|--------|--------|------|------|
| F | Appearance Feature Extractor | `modules/appearance_feature_extractor.py` | Source image → 32×16×64×64 3D feature volume |
| M | Motion Extractor | `modules/motion_extractor.py` | Frame → 21 keypoints + head pose (pitch/yaw/roll) + expression |
| W | Dense Motion / Warping | `modules/dense_motion.py` + `modules/warping_network.py` | Sparse keypoints → dense deformation field + occlusion map |
| G | SPADE Generator | `modules/spade_generator.py` | Warped features → 256×256 (or 512×512) RGB frame |
| S/R | Stitching & Retargeting | `modules/stitching_retargeting_network.py` | Paste-back alignment; eye/lip shape correction |

Landmark detection runs via ONNX (`src/utils/human_landmark_runner.py`, 203 keypoints at 224×224 input).

## Inference Pipeline (data flow)

1. Load source image/video → resize to max 1280px → crop 256×256 face region
2. `F(source_crop)` → 3D appearance features
3. Per driving frame: ONNX landmark detection → `M(frame)` → keypoints/pose/expression
4. `W(features, kp_source, kp_driving)` → deformation field
5. Bilinear warp of 3D features
6. `G(warped)` → RGB frame
7. Stitching/retargeting MLPs (S, R_eyes, R_lip) → corrected frame
8. Inverse crop transform → paste back into original space
9. FFmpeg assembly → MP4/GIF with audio preserved

## Configuration

Key inference flags (`src/config/inference_config.py`):

- `--flag_use_half_precision` — float16 for speed (default on GPU)
- `--flag_torch_compile` — torch.compile graph optimization (~20–30% faster, NVIDIA only)
- `--flag_crop_driving_video` — auto-crop driving input
- `--driving_option` — region control: `expression`, `pose`, `lip`, `eye`, or `all`
- `--output_fps` — output frame rate (default 25)
- `--source_max_dim` — max source resolution (default 1280)

## Motion Templates

Driving videos can be pre-processed into `.pkl` motion template files (keypoints + pose only, no raw video). Faster on repeated use and privacy-preserving. Animals mode uses `.pkl` by default.

## Dependencies

Install via:
```bash
pip install -r requirements.txt       # Linux/Windows
pip install -r requirements_macOS.txt # macOS
```

Key packages: PyTorch 2.x + CUDA, OpenCV, ONNX Runtime GPU, Gradio 5.x, FFmpeg bindings, PyKalman (motion smoothing), Tyro (CLI parsing).

Animal mode requires compiling X-Pose CUDA ops (`src/utils/dependencies/XPose/`) — Linux/Windows with CUDA only.

## Performance Notes

- Run `python speed.py` for per-module timing breakdown
- `torch.compile` requires a warm-up pass before reaching full speed
- ONNX Runtime handles landmark detection separately for lightweight execution
- Half-precision (float16) has minimal quality impact and is recommended on GPU
