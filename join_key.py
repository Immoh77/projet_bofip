# join_key.py
import hashlib, textwrap

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    return " ".join(s.split())

def make_sig(question: str, answer: str, width: int = 400) -> str:
    a_head = textwrap.shorten(answer or "", width=width, placeholder="")
    raw = f"{_norm(question)}||{_norm(a_head)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
