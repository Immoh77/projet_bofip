from __future__ import annotations
import json, os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from rag.config import BASE_DIR  # déjà défini dans config.py

LOG_DIR = BASE_DIR / "logs"
HISTORY_PATH = LOG_DIR / "search_history.jsonl"

def _ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.touch()

def append_history(entry: Dict[str, Any]) -> None:
    _ensure_dirs()
    entry = {**entry, "timestamp": datetime.now().isoformat(timespec="seconds")}
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_history(max_items: int = 200) -> List[Dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    lines: List[str]
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    items = [json.loads(line) for line in lines if line.strip()]
    return items[-max_items:]

def clear_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink(missing_ok=True)