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

def generate_answer(
    question: str,
    small_chunks: list,
    include_sources: bool = True,
    llm_model: str = None,
):
    """
    Génère la réponse finale en s'appuyant sur les BIG chunks liés aux small_chunks.
    """
    from rag.config import OPENAI_CHAT_MODEL, PROMPT_ANSWER
    from rag.retrieval.qdrant_retriever import QdrantRetriever
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    retriever = QdrantRetriever()

    # ✅ 1. Aller chercher les BigChunks comme avant (corrigé)
    try:
        big_chunks = retriever.get_big_chunks_from_small(small_chunks)
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération des BigChunks : {e}")
        big_chunks = small_chunks  # fallback

    # ✅ 2. Construire le contexte depuis les BigChunks
    context = "\n\n---\n\n".join(
        ch.get("text")
        or ch.get("content")
        or ch.get("metadata", {}).get("text")
        or ""
        for ch in big_chunks
        if ch
    )

    # ✅ 3. Construire le prompt complet
    user_prompt = f"""Contexte :
{context}

Question :
{question}
"""

    # ✅ 4. Utiliser le modèle choisi dans app.py (si fourni)
    model_name = llm_model

    # ✅ 5. Appel au modèle
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": PROMPT_ANSWER},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.0,
    )

    answer = response.choices[0].message.content.strip()

    # ✅ 6. Ajouter les sources si demandé
    if include_sources:
        sources = []
        seen = set()
        for ch in big_chunks:
            md = ch.get("metadata", {}) or {}
            cid = ch.get("chunk_id") or md.get("chunk_id")
            if cid in seen:
                continue
            seen.add(cid)
            titre = md.get("titre_document") or md.get("title") or "Source"
            page = md.get("page") or md.get("page_number") or ""
            url = md.get("url") or md.get("lien") or ""
            line = f"- {titre}"
            if page:
                line += f" (p.{page})"
            if url:
                line += f" — {url}"
            sources.append(line)
        if sources:
            answer += "\n\n---\n**Sources :**\n" + "\n".join(sources)

    return answer
