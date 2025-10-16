# rag/retrieval/decompose.py
from __future__ import annotations
from typing import List
from openai import OpenAI
from rag.config import OPENAI_CHAT_MODEL

client = OpenAI()

_PROMPT_RAG = (
    "Tu es un assistant de recherche documentaire. "
    "À partir de la question suivante, génère 3 à 6 sous-questions précises, complémentaires et auto-suffisantes. "
    "Objectif : maximiser les chances de trouver des passages pertinents dans une base documentaire "
    "juridique/fiscale/comptable/assurance (RAG). "
    "Évite les reformulations vagues ; chaque sous-question doit être directement interrogeable. "
    "Couvre, si pertinent pour la question, différents angles (définition/base/conditions/procédure/exceptions/délais/sources), "
    "sans forcer artificiellement des catégories. "
    "Réponds UNIQUEMENT par une liste JSON de chaînes (ex: [\"...\",\"...\"])."
)

def generate_subquestions(question: str) -> List[str]:
    """
    Retourne une liste de 3–6 sous-questions RAG-oriented.
    Robuste : si le JSON est invalide, on tente un fallback simple.
    """
    try:
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": _PROMPT_RAG},
                {"role": "user",   "content": question}
            ],
            temperature=0
        )
        content = (resp.choices[0].message.content or "").strip()

        # 1) Tentative de JSON direct
        import json
        arr = json.loads(content)
        if isinstance(arr, list):
            out = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
            return out[:6] if out else [question]

    except Exception:
        pass

    # 2) Fallback : parser une liste en puces ou texte brut
    lines = []
    for raw in content.splitlines() if 'content' in locals() else []:
        s = raw.strip(" -*•\t").strip()
        if len(s) >= 3:
            lines.append(s)
    lines = [l for l in lines if l] or [question]
    return lines[:6]