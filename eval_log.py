# eval_log.py
import json
from datetime import datetime
from pathlib import Path

EVAL_PATH = Path("logs/eval_judge.jsonl")
EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)

def append_eval(d: dict):
    d = {**d, "ts": datetime.now().isoformat(timespec="seconds")}
    with open(EVAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")