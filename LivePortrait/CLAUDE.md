# CLAUDE.md — LivePortrait

LivePortrait inference server with FastAPI SDK.

## Environment

The virtual environment lives at `/workspace/talking-head/LivePortrait/venv`. Python 3.12 is used.

```bash
# Activate
source /workspace/talking-head/LivePortrait/venv/bin/activate

# Install all dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 2000
```

## PyTorch (CUDA)

The project requires PyTorch with GPU support. The environment has CUDA 12.8; use the cu124 wheels (CUDA is backward compatible).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Installed versions:
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`

## Dependencies

| File | Purpose |
|---|---|
| `requirements.txt` | Top-level deps — includes `requirements_base.txt` and PyTorch |
| `requirements_base.txt` | Core inference deps (numpy 2.4.4, albumentations 1.4.24, etc.) |

## Key Notes

- `numpy` is pinned to `2.4.4`; `albumentations` must be `>=1.4.11` to be compatible.
- Do **not** use the `sdk/venv/` environment for the main server — they are separate.
- PyTorch wheels are pulled from `https://download.pytorch.org/whl/cu124`.

## Architecture

| Path | Purpose |
|---|---|
| `main.py` | FastAPI app entry point |
| `sdk/` | Standalone FastAPI SDK server (separate venv, port 2000) |
| `src/` | LivePortrait model code |
| `pretrained_weights/` | Model checkpoints |
| `inference.py` | CLI inference script |
