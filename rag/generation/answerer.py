import os
from openai import OpenAI
from rag.config import PROMPT_ANSWER, OPENAI_CHAT_MODEL
from rag.retrieval.qdrant_retriever import QdrantRetriever


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# 🔹 1. Utilitaires de formatage
# -------------------------------

def _norm_text_from_chunk(ch: dict) -> str:
    """Récupère le texte du chunk (supporte small/big et variantes)."""
    if ch.get("text"):
        return ch["text"]
    if ch.get("contenu"):
        return ch["contenu"]
    meta = ch.get("metadata", {}) or {}
    return meta.get("text", "")


def _build_context_from_big_chunks(big_chunks: list) -> str:
    """
    Construit le contexte à injecter au LLM à partir des big chunks.
    Pas de troncature — les big chunks sont supposés cohérents.
    """
    context_lines = []
    seen = set()

    for ch in big_chunks:
        cid = ch.get("chunk_id") or ch.get("metadata", {}).get("chunk_id")
        if cid in seen:
            continue
        seen.add(cid)

        meta = ch.get("metadata", {}) or {}
        titre = meta.get("titre_document", "Sans titre")
        base = (meta.get("source", "source inconnue") or "").upper()
        url = meta.get("permalien", "Source inconnue")

        texte = _norm_text_from_chunk(ch).strip()
        context_lines.append(
            f"Titre : {titre}\nBase : {base}\nSource : {url}\n\n{texte}\n"
        )

    return "\n".join(context_lines)


def append_sources(answer: str, chunks: list) -> str:
    """Ajoute les sources uniques à la fin de la réponse."""
    seen, sources = set(), []
    for ch in chunks:
        meta = ch.get("metadata", {}) or {}
        titre = meta.get("titre_document", "Sans titre")
        url = meta.get("permalien", None)
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append(f"- [{titre}]({url})")
    if not sources:
        return answer
    return f"{answer}\n\n---\n\n📎 **Sources utilisées :**\n" + "\n".join(sources)


# -------------------------------
# 🔹 2. Génération principale
# -------------------------------

def generate_answer(query: str, chunks: list, include_sources: bool = True) -> str:
    """
    Génère la réponse finale à partir des small chunks (issus du retriever).
    Le module remonte automatiquement les big chunks pour le contexte.
    """
    # 1️⃣ Remonter aux big chunks associés
    try:
        retriever = QdrantRetriever()
        big_chunks = retriever.get_big_chunks_from_small(chunks)
    except Exception:
        big_chunks = chunks

    # 2️⃣ Construire le contexte complet
    context = _build_context_from_big_chunks(big_chunks)

    # 3️⃣ Prompt structuré — étapes de raisonnement
    user_prompt = f"""
Question posée :
{query}

Contexte documentaire :
{context}

---

**1ère étape :** Sélectionne parmi ces extraits juridiques ceux qui répondent directement ou partiellement à la question posée.
> Format attendu : ne doit pas apparaître dans la réponse.

**2ème étape :** Vérifie la cohérence des extraits sélectionnés.  
> Format attendu : ne doit pas apparaître dans la réponse.  
> Si tu n’as pas assez d’informations pour répondre, dis uniquement : "Je n’ai pas assez d’éléments en ma possession pour répondre" et arrête-toi.

**3ème étape :** Résume les textes applicables en citant les sources exactes.  
> Format attendu :  
> - **Textes juridiques applicables** (titre de section en gras et plus grand)  
> - Résumé clair de l’article ou des extraits pertinents  
> - Indique la **source** (ex. BOFiP, Code général des impôts, etc.)

**4ème étape :** Explique comment ces textes s’appliquent concrètement à la question posée, sans extrapolation ni ajout externe.  
> Format attendu :  
> - **Application au cas d’espèce** (titre en gras et plus grand)
"""

    # 4️⃣ Appel au modèle OpenAI
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": PROMPT_ANSWER},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.0,
    )

    answer = response.choices[0].message.content.strip()

    # 5️⃣ Ajouter les sources si demandé
    if include_sources:
        answer = append_sources(answer, big_chunks)

    return answer
