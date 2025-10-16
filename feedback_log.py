# feedback_log.py
import json
from datetime import datetime
from pathlib import Path

FB_PATH = Path("logs/feedback_log.jsonl")
FB_PATH.parent.mkdir(parents=True, exist_ok=True)

def append_feedback(d: dict):
    d = {**d, "ts": datetime.now().isoformat(timespec="seconds")}
    with open(FB_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
