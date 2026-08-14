"""Persist trained steering vectors to disk so they survive app restarts
and don't need retraining every session.

Each saved vector is two files sharing a base name: a JSON sidecar with
inspectable metadata, and a `.pkl` with the raw pickled `ControlVector`
bytes returned by the Modal training call.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

VECTORS_DIR = Path(__file__).resolve().parent.parent / "vectors"

_META_FIELDS = (
    "model_id",
    "trait_name",
    "layer_ids",
    "dataset_size",
    "mid_layer",
    "activation_norm",
)


def _safe(model_id: str) -> str:
    return model_id.replace("/", "__")


def save(trained: dict) -> str:
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{timestamp}_{_safe(trained['model_id'])}_{trained['trait_name']}"

    meta = {field: trained[field] for field in _META_FIELDS}
    meta["timestamp"] = timestamp
    meta["id"] = base

    (VECTORS_DIR / f"{base}.meta.json").write_text(json.dumps(meta, indent=2))
    (VECTORS_DIR / f"{base}.pkl").write_bytes(trained["vector_bytes"])
    return base


def list_saved() -> list[dict]:
    if not VECTORS_DIR.exists():
        return []
    metas = [
        json.loads(p.read_text()) for p in VECTORS_DIR.glob("*.meta.json")
    ]
    return sorted(metas, key=lambda m: m["timestamp"], reverse=True)


def load(vector_id: str) -> dict:
    meta = json.loads((VECTORS_DIR / f"{vector_id}.meta.json").read_text())
    vector_bytes = (VECTORS_DIR / f"{vector_id}.pkl").read_bytes()
    return {**{field: meta[field] for field in _META_FIELDS}, "vector_bytes": vector_bytes}


def label(meta: dict) -> str:
    return (
        f"{meta['trait_name']} — {meta['model_id']} "
        f"({meta['dataset_size']} pairs, {meta['timestamp']})"
    )
