import logging
from openai import OpenAI
from rag.retrieval.qdrant_retriever import QdrantRetriever
from rag.config import PROMPT_ANSWER

# Configuration de base du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_answer(question, small_chunks, include_sources=True, llm_model=None):
    logging.info("🚀 Début de generate_answer()")

    # --- Vérification du modèle ---
    if not llm_model:
        logging.error("❌ Aucun modèle LLM n’a été spécifié. Vérifie l’UI Streamlit.")
        raise ValueError("Aucun modèle LLM n’a été spécifié.")

    logging.info(f"🧠 Modèle sélectionné : {llm_model}")

    client = OpenAI()
    retriever = QdrantRetriever()

    # --- Étape 1 : Récupération des big_chunks ---
    try:
        logging.info("📥 Récupération des BigChunks depuis Qdrant...")
        big_chunks = retriever.get_big_chunks_from_small(small_chunks)
        logging.info(f"✅ {len(big_chunks)} BigChunks récupérés.")
    except Exception as e:
        logging.error(f"⚠️ Erreur Qdrant : {e}")
        big_chunks = small_chunks  # fallback minimal
        logging.info("➡️ Utilisation des small_chunks comme fallback.")

    # --- Étape 2 : Construction du contexte ---
    context = "\n\n---\n\n".join(
        ch.get("text") or ch.get("content") or ch.get("metadata", {}).get("text") or ""
        for ch in big_chunks
        if ch
    )
    logging.info(f"📚 Contexte construit ({len(context)} caractères).")

    # --- Étape 3 : Construction du prompt ---
    user_prompt = (
        f"Voici la question : \n\n{question}\n\n"
        f"Voici des extraits juridiques avec leur source issus de bases documentaires :\n\n{context}\n\n"
        f"** 1ère étape ** : En fonction de la question posée et des extraits juridiques, "
        f"sélectionne ceux qui répondent totalement ou partiellement à la question.\n"
        f"** 2ème étape ** : Vérifie que les réponses sélectionnées sont cohérentes entre elles.\n"
        f"Si tu n’as pas assez d’éléments, réponds : "
        f"\"je n’ai pas assez d’éléments à ma disposition pour répondre\".\n"
        f"** 3ème étape ** : Formule un résumé complet des textes en citant les sources.\n"
        f"** 4ème étape ** : Explique comment ces textes et uniquement ces textes "
        f"s’appliquent concrètement à la question."
    )
    logging.info("🧩 Prompt construit avec succès.")

    # --- Étape 4 : Appel au modèle ---
    try:
        logging.info("📨 Envoi du prompt au modèle OpenAI...")
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": PROMPT_ANSWER},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.0,
        )
        logging.info("✅ Réponse reçue du modèle.")
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logging.exception(f"💥 Erreur lors de l’appel au modèle : {e}")
        raise  # On laisse remonter l’erreur pour la voir dans Streamlit

    # --- Étape 5 : Ajout des sources ---
    if include_sources:
        try:
            seen, sources = set(), []
            for ch in big_chunks:
                md = ch.get("metadata", {}) or {}
                url = md.get("url") or md.get("permalien")
                if not url or url in seen:
                    continue
                seen.add(url)
                titre = md.get("titre_document") or "Source inconnue"
                sources.append(f"- [{titre}]({url})")
            if sources:
                answer += "\n\n---\n📎 **Sources utilisées :**\n" + "\n".join(sources)
            logging.info(f"📎 {len(sources)} sources ajoutées.")
        except Exception as e:
            logging.warning(f"⚠️ Échec lors de l’ajout des sources : {e}")

    logging.info("🏁 Fin de generate_answer()")
    return answer
