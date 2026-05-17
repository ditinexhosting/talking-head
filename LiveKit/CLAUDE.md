# CLAUDE.md — LiveKit VPS Orchestrator

## What this service does

Stateless FastAPI session coordinator. It sits between the browser UI, LiveKit SFU, and the GPU inference server. It never handles media — it only creates rooms, mints tokens, and forwards join signals.

## Running locally

```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## File responsibilities

| File | Owns |
|---|---|
| `main.py` | App factory, CORS, lifespan, endpoint routing |
| `livekit_service.py` | LiveKit room creation, JWT minting |
| `gpu_service.py` | HTTP POST to GPU `/join`, error mapping to 503 |

**Do not add LiveKit or GPU logic to `main.py`** — keep it as thin wiring only.

## Key design decisions

- **Stateless** — no in-memory session tracking. Each request is self-contained. If session state is needed later, use Redis, not a Python dict.
- **Subscribe-only UI tokens** — `can_publish=False`. The GPU agent publishes; the UI only receives. Do not change this without understanding the LiveKit room permission model.
- **5-second GPU timeout** — `gpu_service.py` uses a hard 5s timeout on the `/join` call. If the GPU is slow to respond, raise the timeout there, not by adding retries in `main.py`.
- **Room limits** — `empty_timeout=300`, `max_participants=2`. Raising `max_participants` beyond 2 requires verifying the GPU agent handles multiple subscribers.

## Environment variables

All config is in `.env` (copy from `.env.example`). Never hardcode credentials. The defaults in each service file are for local dev only.

`VPS_BASE_URL` must be set to the public-facing URL of this server so the GPU callback URL is reachable. Default is `http://localhost:8000` which only works for local testing.

## Load balancer hooks

Look for `# LOAD BALANCER HOOK` comments in `livekit_service.py` and `gpu_service.py`. These mark the exact lines to replace when adding multi-node LiveKit or GPU pool routing.

## Adding new endpoints

1. Define the Pydantic schema in `main.py`.
2. Put any LiveKit calls in `livekit_service.py`, any GPU calls in `gpu_service.py`.
3. Keep endpoints `async` — all I/O here is async (httpx, livekit-api).

## Dependencies

Pinned in `requirements.txt`. Do not upgrade `livekit-api` without checking the import surface — the SDK has had breaking changes (e.g. `RoomServiceClient` was removed; use `LiveKitAPI().room` instead).
