import json
import logging
from typing import Any


PROFILE_EVENT_PREFIX = "OF3_PROFILE_EVENT"


def emit_profile_event(
    logger: logging.Logger,
    *,
    stage: str,
    event: str,
    **metadata: Any,
) -> None:
    payload = {
        "stage": stage,
        "event": event,
        **metadata,
    }
    line = f"{PROFILE_EVENT_PREFIX} {json.dumps(payload, sort_keys=True)}"

    # Keep events visible even when INFO logs are filtered by runtime logging config.
    logger.warning(line)
    print(line, flush=True)
