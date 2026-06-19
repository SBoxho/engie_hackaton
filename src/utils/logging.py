from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


_CONFIGURED = False


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Write one structured JSON log line.

    The app runs in environments where stdout is the most reliable log sink, so
    this keeps request IDs, run IDs, and error categories machine-readable
    without requiring a separate collector.
    """

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=_json_default))
