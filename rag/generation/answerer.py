# -*- coding: utf-8 -*-
"""
answerer.py — Génération finale de la réponse (prompt multi-étape enrichi)
---------------------------------------------------------------------------
Ce module :
1. Prend la question utilisateur et les résultats du retriever (QdrantRetriever)
2. Construit un prompt complet et contextuel à partir du RAG
3. Génère une réponse cohérente et sourcée via OpenAI
4. Peut être exécuté manuellement pour test
"""

import logging
from openai import OpenAI
from rag.retrieval.qdrant_retriever import QdrantRetriever
from rag.config import (
    OPENAI_API_KEY,
    DEFAULT_CHAT_MODEL,
    PROMPT_ANSWER_HEADER,
    PROMPT_USER_ANSWER,
)

# === Configuration logging (épurée) ===
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def append_sources_to_context(chunks):
    """
    Construit un texte structuré avec les métadonnées et le contenu de chaque chunk.
    Chaque bloc affiche :
    - Titre du document
    - Source / Base
    - URL (permalien)
    - Contenu complet du chunk (big_chunk)
    """

    formatted = []
    for idx, ch in enumerate(chunks, start=1):
        meta = ch.get("metadata", {}) or {}

        titre = meta.get("titre_document") or meta.get("titre_bloc") or "Sans titre"
        base = (meta.get("base") or meta.get("source") or "Source inconnue").capitalize()
        url = meta.get("permalien") or meta.get("url") or "Non précisée"
        contenu = ch.get("text") or ch.get("contenu") or ""

        bloc = (
            f"=== BIG CHUNK {idx} ===\n"
            f"Titre : {titre}\n"
            f"Base : {base}\n"
            f"Source : {url}\n"
            f"Contenu :\n{contenu.strip()}\n"
        )

        formatted.append(bloc)

    return "\n\n".join(formatted).strip()

# ==========================================================
# Fonction principale de génération de réponse
# ==========================================================
def generate_answer(question, retriever_output, include_sources=True, llm_model=None):
    """
    Génère la réponse finale à partir :
    - de la question originale
    - du résultat complet du retriever (question clarifiée, sous-questions, big_chunks)
    """
    logger.info("🚀 Début de generate_answer()")

    # --- Vérification / fallback du modèle ---
    if not llm_model:
        llm_model = DEFAULT_CHAT_MODEL
        logger.info(f"🧠 Utilisation du modèle par défaut : {llm_model}")

    # --- Initialisation du client OpenAI ---
    client = OpenAI(api_key=OPENAI_API_KEY)

    # --- Extraction des données RAG ---
    big_chunks = retriever_output.get("big_chunks_associes", [])
    clarified = retriever_output.get("question_clarifiee", "Non disponible")
    subqs = retriever_output.get("sous_questions", [])

    # --- Construction du contexte enrichi avec métadonnées ---
    context = append_sources_to_context(big_chunks)
    logger.info(f"📚 Contexte enrichi construit ({len(context)} caractères).")

    # --- Formatage des sous-questions ---
    subqs_text = "\n".join([f"- {s}" for s in subqs]) or "Aucune sous-question générée."

    # --- Construction du prompt complet depuis la config ---
    user_prompt = PROMPT_USER_ANSWER.format(
        question_originale=question,
        question_clarifiee=clarified,
        subquestions=subqs_text,
        context=context,
    )
    logger.info("🧩 Prompt multi-étape (issu de config) construit avec succès.")

    # --- Appel au modèle ---

    # --- Log du prompt envoyé au modèle ---
    logger.info("\n" + "=" * 80)
    logger.info("🧠 PROMPT ENVOYÉ AU LLM")
    logger.info("-" * 80)
    logger.info(user_prompt)
    logger.info("=" * 80 + "\n")

    try:
        logger.info("📨 Envoi du prompt au modèle OpenAI...")
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system",
                 "content": "Tu es un expert-comptable et fiscaliste. Réponds de manière claire, concise et sourcée aux extraits suivants."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
        )
        answer = response.choices[0].message.content.strip()
        logger.info("✅ Réponse générée avec succès.")
    except Exception as e:
        logger.exception(f"💥 Erreur lors de l’appel au modèle : {e}")
        raise

    logger.info("🏁 Fin de generate_answer()")
    return answer


# ==========================================================
# Exécution manuelle (test en console)
# ==========================================================
if __name__ == "__main__":
    logger.info("🧠 Test manuel de génération de réponse.")
    retriever = QdrantRetriever()
    q = input("❓ Question : ")
    retrieved = retriever.retrieve_with_subquery_rerank(q)
    final_answer = generate_answer(q, retrieved)
    print("\n=== 🧾 Réponse finale ===\n")
    print(final_answer)
