from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def _handle_chat(participant: str, text: str) -> None:
    logger.info("[agent] chat from %s: %s", participant, text)


def _handle_command(participant: str, text: str) -> None:
    try:
        payload = json.loads(text)
        logger.info("[agent] command from %s: %s", participant, payload)
    except json.JSONDecodeError:
        logger.warning("[agent] invalid command JSON from %s", participant)


TOPIC_HANDLERS = {
    "chat": _handle_chat,
    "command": _handle_command,
}
