# LiveKit VPS Orchestrator

FastAPI service that acts as the session coordinator between the browser UI, LiveKit SFU, and the GPU inference server in the talking-head streaming pipeline.

## Architecture

```
Browser UI
    │
    │  POST /session
    ▼
VPS Orchestrator  ──── LiveKit API ────► LiveKit SFU
    │                                        ▲
    │  POST /join                            │
    ▼                                        │
GPU Server ──────────── agent joins room ───┘
```

**Flow for a new session:**

1. UI calls `POST /session` with a `user_id`.
2. Orchestrator creates a LiveKit room (auto-deletes after 5 min empty, max 2 participants).
3. Orchestrator mints a subscribe-only JWT for the UI.
4. Orchestrator tells the GPU server to join the room.
5. UI receives `{ room_id, token, livekit_url }` and connects directly to LiveKit.
6. When the GPU agent finishes, it calls `POST /session-ended/{room_id}` back to the VPS.

## Project structure

```
LiveKit/
├── main.py              # FastAPI app, CORS, lifespan, endpoint wiring
├── livekit_service.py   # Room creation + JWT token generation
├── gpu_service.py       # HTTP call to GPU server /join endpoint
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── CLAUDE.md
```

## Setup

### 1. Python environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `LIVEKIT_URL` | WebSocket URL of your LiveKit server |
| `LIVEKIT_API_KEY` | LiveKit API key |
| `LIVEKIT_API_SECRET` | LiveKit API secret |
| `GPU_SERVER_URL` | Base HTTP URL of the GPU inference server |
| `VPS_BASE_URL` | Public base URL of this VPS (used to build callback URLs) |
| `ALLOWED_ORIGINS` | Comma-separated allowed CORS origins (currently set to `*`) |

### 3. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 2000
```

For production with auto-reload off and multiple workers:

```bash
uvicorn main:app --host 0.0.0.0 --port 2000 --workers 4
```

## API reference

### `POST /session`

Start a new talking-head session.

**Request**
```json
{ "user_id": "alice" }
```

**Response `200`**
```json
{
  "room_id": "session-a1b2c3d4",
  "token": "<livekit-jwt>",
  "livekit_url": "wss://livekit.ditinex.com"
}
```

**Error responses**
- `502` — LiveKit room creation failed
- `503` — GPU server unreachable or returned non-200

---

### `POST /session-ended/{room_id}`

Webhook called by the GPU server when a session ends.

**Response `200`**
```json
{ "status": "ok" }
```

---

### `GET /health`

Liveness probe.

**Response `200`**
```json
{ "status": "ok" }
```

## Token grants

The JWT issued to the UI is **subscribe-only**:

| Grant | Value |
|---|---|
| `room_join` | `true` |
| `can_subscribe` | `true` |
| `can_publish` | `false` |
| `can_publish_data` | `false` |
| TTL | 1 hour |

The GPU agent mints its own token internally — this server never publishes one on its behalf.

## Load balancer hooks

Both `livekit_service.py` and `gpu_service.py` contain `# LOAD BALANCER HOOK` comments marking the exact lines to swap in service-discovery or pool-selector logic when scaling to multiple LiveKit nodes or GPU workers.
