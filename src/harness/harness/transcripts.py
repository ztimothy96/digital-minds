"""Save steered-chat transcripts to timestamped JSON log files."""

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save(
    session_id: str,
    model_id: str,
    trait_name: str,
    turns: list[dict],
) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{session_id}_{model_id.replace('/', '__')}.json"
    payload = {
        "session_id": session_id,
        "model_id": model_id,
        "trait_name": trait_name,
        "turns": turns,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
