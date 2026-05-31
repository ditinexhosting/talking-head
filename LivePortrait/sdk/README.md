# Talking Head SDK

FastAPI server with REST and WebSocket support.

## Setup

**1. Download Rhubarb Lip Sync into `assets/`:**

Get the latest release from https://github.com/DanielSWolf/rhubarb-lip-sync/releases/tag/v1.14.0

```bash
cd sdk/assets

# Download (replace version number if newer is available)
wget https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/rhubarb-lip-sync-1.14.0-linux.zip

# Unzip
unzip rhubarb-lip-sync-1.14.0-linux.zip

# Make executable
chmod +x rhubarb-lip-sync-1.14.0/rhubarb

# Test it
./rhubarb-lip-sync-1.14.0/rhubarb --version
```

The binary must be at `sdk/assets/rhubarb-1.14.0/rhubarb` (rename the extracted folder if needed).

**2. Download Kokoro ONNX model files into `assets/`:**

Reference: https://github.com/thewh1teagle/kokoro-onnx/tree/main

Download `kokoro-v1.0.onnx` and `voices-v1.0.bin` and place them in `sdk/assets/`:

```bash
cd sdk/assets

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

**3. Create the virtual environment inside the `sdk/` directory:**

```bash
cd sdk
python3 -m venv venv
```

**4. Activate it:**

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**5. Install dependencies:**

```bash
pip install -r requirements.txt
```

## Troubleshooting

### Kokoro TTS not using GPU (`CUDAExecutionProvider` not available)

**Symptom:** Warning on startup:
```
UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```

**Cause:** Both `onnxruntime` (CPU-only) and `onnxruntime-gpu` are installed simultaneously. Python imports the CPU package because it is newer, ignoring CUDA. Additionally, if `onnxruntime` is uninstalled naively, pip removes shared `__init__.py` files that `onnxruntime-gpu` also registered, leaving it broken.

**Fix:** Remove the CPU package and force-reinstall the GPU one:

```bash
pip uninstall onnxruntime -y
pip install --force-reinstall onnxruntime-gpu==1.20.1
```

Verify CUDA is now active:

```bash
python -c "import onnxruntime as rt; print(rt.get_available_providers())"
# Expected: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

> `onnxruntime-gpu` is pinned to `1.20.1` — versions 1.21+ on PyPI ship a broken namespace package (`CUDAExecutionProvider` not available). Do not upgrade until a fixed wheel is confirmed.

### `libcudnn_adv.so.9: cannot open shared object file`

**Symptom:**
```
[E:onnxruntime] Failed to load library libonnxruntime_providers_cuda.so
  with error: libcudnn_adv.so.9: cannot open shared object file: No such file or directory
```

**Cause:** `nvidia-cudnn-cu12` installs its `.so` files under `site-packages/nvidia/cudnn/lib/`, which is not on `LD_LIBRARY_PATH`. The onnxruntime CUDA provider can't find them at load time.

**Fix (already applied in `controllers/tts.py`):** `_preload_cudnn()` uses `ctypes.CDLL(RTLD_GLOBAL)` to load the cuDNN libraries before onnxruntime is imported, making them visible to all subsequent `dlopen()` calls. No manual `LD_LIBRARY_PATH` changes needed.

If you ever need to apply the same fix elsewhere, the pattern is:

```python
import ctypes, site
from pathlib import Path
for sp in site.getsitepackages():
    cudnn_lib = Path(sp) / "nvidia" / "cudnn" / "lib"
    if cudnn_lib.exists():
        for name in ["libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_adv.so.9"]:
            p = cudnn_lib / name
            if p.exists():
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        break
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 2000
```

## Endpoints

| Type      | Path             | Description                              |
|-----------|------------------|------------------------------------------|
| GET       | `/api/`          | Hello World                              |
| GET       | `/api/health`    | Health check                             |
| WebSocket | `/ws/audio`      | TTS with viseme data                     |

Swagger UI is available at `http://localhost:2000/docs`.


ref links :
https://github.com/warmshao/FasterLivePortrait/tree/master -> FasterLivePortal

https://github.com/GVCLab/PersonaLive --> Personal live stream
