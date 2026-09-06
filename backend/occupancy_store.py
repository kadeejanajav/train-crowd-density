"""
The "single source of truth" for current occupancy per train.

Kept deliberately simple for a prototype: a JSON file on disk, guarded
by a lock. This is NOT how you'd do it in production (you'd use Redis
or a real database so multiple processes/webcams can update it safely
at scale) — but for a demo with one simulated feed and one API server,
a JSON file is honest, debuggable, and good enough. Swap in a real DB
later without changing the interface below.
"""
import json
import threading
from pathlib import Path
from datetime import datetime

from src.config import COACH_CAPACITY
from src.density.classifier import classify_density

STORE_PATH = Path(__file__).resolve().parent / "occupancy_store.json"
_lock = threading.Lock()


def _read_all() -> dict:
    if not STORE_PATH.exists():
        return {}
    with open(STORE_PATH, "r") as f:
        return json.load(f)


def _write_all(data: dict):
    with open(STORE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def set_count(train_number: str, count: int):
    """Called by the webcam simulator when it has a fresh count."""
    with _lock:
        data = _read_all()
        data[train_number] = {
            "count": count,
            "capacity": COACH_CAPACITY,
            "last_updated": datetime.now().isoformat(),
        }
        _write_all(data)


def increment_count(train_number: str, by: int = 1):
    """Called when a new ticket is booked - one more expected passenger."""
    with _lock:
        data = _read_all()
        current = data.get(train_number, {"count": 0, "capacity": COACH_CAPACITY})
        current["count"] += by
        current["last_updated"] = datetime.now().isoformat()
        data[train_number] = current
        _write_all(data)


def get_status(train_number: str) -> dict:
    data = _read_all()
    entry = data.get(train_number, {"count": 0, "capacity": COACH_CAPACITY})
    density = classify_density(entry["count"], capacity=entry["capacity"])
    return {
        "train_number": train_number,
        "count": entry["count"],
        "capacity": entry["capacity"],
        "occupancy_ratio": density.occupancy_ratio,
        "level": density.level,
        "last_updated": entry.get("last_updated"),
    }
