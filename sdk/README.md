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
