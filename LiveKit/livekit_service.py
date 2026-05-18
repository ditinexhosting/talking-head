import os
from datetime import timedelta
from livekit import api
from livekit.api import AccessToken, VideoGrants, CreateRoomRequest

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://livekit.ditinex.com")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "devsecret123changeme")

TOKEN_TTL = timedelta(hours=1)


async def create_room(room_id: str) -> None:
    """Create a LiveKit room via the server API."""
    # LOAD BALANCER HOOK: swap LIVEKIT_URL here when routing
    # through a load balancer in front of multiple LiveKit nodes.
    livekit_api = api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )
    try:
        await livekit_api.room.create_room(
            CreateRoomRequest(
                name=room_id,
                empty_timeout=300,   # auto-delete 5 min after last participant leaves
                max_participants=2,  # one UI client + one GPU agent
            )
        )
    finally:
        await livekit_api.aclose()


def generate_ui_token(room_id: str, user_id: str) -> str:
    """Generate a signed JWT for the UI client.

    The UI only receives video/audio — it never publishes.
    """
    grants = VideoGrants(
        room_join=True,
        room=room_id,
        can_subscribe=True,
        can_publish=True,
        can_publish_data=True,
    )
    token = (
        AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity(user_id)
        .with_name(user_id)
        .with_grants(grants)
        .with_ttl(TOKEN_TTL)
        .to_jwt()
    )
    return token
