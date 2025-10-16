from __future__ import annotations
import json, time
from typing import Any, Dict, List
from openai import OpenAI

client = OpenAI()

# ---- PROMPT avec échelle 1–3 et explications brèves ----
INSTR = """Tu es l’évaluateur d’un système RAG.

But : 

1. Produire des scores de 1 à 3 :

   - "score_context" : adéquation des extraits par rapport à la question
      1 = extraits peu ou pas liés à la question
      2 = extraits partiellement liés mais avec manques
      3 = extraits adaptés et globalement suffisants

   - "score_coherence" : cohérence d’utilisation des extraits dans la réponse
      1 = réponse largement déconnectée des extraits
      2 = réponse partiellement cohérente, avec approximations
      3 = réponse cohérente avec les extraits, sans contradictions majeures

   - "score_answer" : pertinence de la réponse vis-à-vis de la question
      1 = réponse hors sujet ou très incomplète
      2 = réponse partiellement pertinente, couvre certains points mais pas tous
      3 = réponse pertinente et couvre les points essentiels attendus

2. Déterminer un verdict d’hallucination :

   - "flags.hallucination" = 1 si :
       * la réponse contient des affirmations non étayées par les extraits,
       * ou si les extraits sont vides, insuffisants ou contradictoires.
   - "flags.hallucination" = 0 sinon.

3. Donner une explication brève, par critère, en indiquant aussi pourquoi la note n’est pas maximale (si < 3) :
   - En 1 phrase max par critère.
   - Si la note = 3, indiquer "RAS" ou "aucun manque notable".

Contraintes :

- Utiliser UNIQUEMENT les extraits.
- Sortie = JSON STRICTEMENT VALIDE uniquement (pas de texte hors JSON).

Format de sortie EXACT (JSON strict) :
{
  "scores": { "score_context": 1..3, "score_coherence": 1..3, "score_answer": 1..3 },
  "flags": { "hallucination": 0|1 },
  "explanation": "Contexte: … | Cohérence: … | Réponse: …",
  "explanations": {
    "context":   {"reason":"…", "why_not_3":"… ou 'RAS'"},
    "coherence": {"reason":"…", "why_not_3":"… ou 'RAS'"},
    "answer":    {"reason":"…", "why_not_3":"… ou 'RAS'"}
  }
}
"""

# ---- JSON Schema (1–3 au lieu de 1–5) ----
JUDGE_JSON_SCHEMA = {
    "name": "judge_schema",
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    "score_context": {"type": "integer", "minimum": 1, "maximum": 3},
                    "score_coherence": {"type": "integer", "minimum": 1, "maximum": 3},
                    "score_answer": {"type": "integer", "minimum": 1, "maximum": 3}
                },
                "required": ["score_context", "score_coherence", "score_answer"],
                "additionalProperties": False
            },
            "flags": {
                "type": "object",
                "properties": {
                    "hallucination": {"type": "integer", "enum": [0, 1]}
                },
                "required": ["hallucination"],
                "additionalProperties": False
            },
            "explanation": {"type": "string", "minLength": 1, "maxLength": 400},
            "explanations": {
                "type": "object",
                "properties": {
                    "context":   {"type":"object","properties":{"reason":{"type":"string"},"why_not_3":{"type":"string"}},"required":["reason","why_not_3"],"additionalProperties":False},
                    "coherence": {"type":"object","properties":{"reason":{"type":"string"},"why_not_3":{"type":"string"}},"required":["reason","why_not_3"],"additionalProperties":False},
                    "answer":    {"type":"object","properties":{"reason":{"type":"string"},"why_not_3":{"type":"string"}},"required":["reason","why_not_3"],"additionalProperties":False}
                },
                "required": ["context", "coherence", "answer"],
                "additionalProperties": False
            }
        },
        "required": ["scores", "flags", "explanation"],
        "additionalProperties": False
    }
}

# ---- Fallback function-calling ----
TOOLS = [{
    "type": "function",
    "function": {
        "name": "return_judge_scores",
        "description": "Retourne les scores (1–3), le flag d'hallucination, et des explications brèves par critère",
        "parameters": JUDGE_JSON_SCHEMA["schema"]
    }
}]


def _parse_fc(resp) -> Dict[str, Any]:
    tcalls = resp.choices[0].message.tool_calls or []
    if not tcalls:
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    args = tcalls[0].function.arguments
    return json.loads(args)


def run_judge(
        question: str,
        contexts: List[Dict[str, Any]] | List[str],
        answer: str,
        model: str = "gpt-4o",
) -> Dict[str, Any]:
    payload = {"question": question, "contexts": contexts, "answer": answer}
    messages = [
        {"role": "system", "content": INSTR},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    t0 = time.perf_counter()
    out: Dict[str, Any]

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": JUDGE_JSON_SCHEMA},
            messages=messages,
        )
        out = json.loads(resp.choices[0].message.content)
    except Exception as e:
        msg = str(e)
        if "json_schema" in msg and ("not supported" in msg or "Invalid parameter" in msg):
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "return_judge_scores"}},
                messages=messages,
            )
            out = _parse_fc(resp)
        else:
            raise

    dur_ms = int((time.perf_counter() - t0) * 1000)

    scores: Dict[str, Any] = out.get("scores", {}) or {}
    flags: Dict[str, Any] = out.get("flags", {}) or {}
    scores.setdefault("score_context", 1)
    scores.setdefault("score_coherence", 1)
    scores.setdefault("score_answer", 1)
    flags.setdefault("hallucination", 1)

    try:
        halluc = int(flags.get("hallucination", 1))
    except Exception:
        halluc = 1
    scores["groundedness"] = 0.0 if halluc == 1 else 1.0

    explanation = str(out.get("explanation", "") or "").strip()[:400]
    explanations = out.get("explanations") or {}
    if not isinstance(explanations, dict):
        explanations = {}
    def _norm_exp(d):
        if not isinstance(d, dict): return {"reason": "", "why_not_3": ""}
        r = str(d.get("reason", "") or "")[:200].strip()
        w = str(d.get("why_not_3", "") or "")[:120].strip()
        return {"reason": r, "why_not_3": w or ("RAS" if r else "")}
    explanations_norm = {
        "context": _norm_exp(explanations.get("context", {})),
        "coherence": _norm_exp(explanations.get("coherence", {})),
        "answer": _norm_exp(explanations.get("answer", {})),
    }

    return {
        "scores": scores,
        "flags": flags,
        "explanation": explanation,
        "explanations": explanations_norm,
        "_duration_ms": dur_ms
    }