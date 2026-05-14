# CLAUDE.md — sdk/

This directory is a standalone FastAPI server (the "Talking Head SDK") with its own Python environment.

## Environment

The virtual environment lives at `sdk/venv/`. Python 3.13 is used.

```bash
# Activate (run from the sdk/ directory or use the full path)
source /home/ubuntu/talking-head/sdk/venv/bin/activate

# Install / update dependencies
pip install -r requirements.txt
```

Never use the parent repo's `/venv/main/` environment for this project — they are separate.

## Key Commands

```bash
# Start the dev server (from sdk/)
uvicorn main:app --reload --host 0.0.0.0 --port 2000

# Install a new package and save it to requirements.txt
pip install <package> && pip freeze > requirements.txt
```

## Architecture

| Path | Purpose |
|---|---|
| `main.py` | FastAPI app — mounts REST and WebSocket routers |
| `routes/rest/router.py` | REST route definitions |
| `routes/ws/router.py` | WebSocket route definitions |
| `controllers/rest_controller.py` | REST request handlers |
| `controllers/ws_controller.py` | WebSocket connection/session management |
| `controllers/tts.py` | Text-to-speech helpers (sentence splitting, Kokoro TTS) |
| `requirements.txt` | Pinned dependencies for this venv |

## Endpoints

| Type | Path | Description |
|---|---|---|
| GET | `/api/` | Hello World |
| GET | `/api/health` | Health check |
| WebSocket | `/ws/echo` | Echoes each message back to sender |
| WebSocket | `/ws/broadcast` | Broadcasts each message to all clients |

Swagger UI: `http://localhost:2000/docs`
