# Talking Head SDK

FastAPI server with REST and WebSocket support.

## Setup

**1. Create the virtual environment inside the `sdk/` directory:**

```bash
cd sdk
python3 -m venv venv
```

**2. Activate it:**

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies:**

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
| WebSocket | `/ws/echo`       | Echoes each message back to the sender   |
| WebSocket | `/ws/broadcast`  | Broadcasts each message to all clients   |

Swagger UI is available at `http://localhost:8000/docs`.
