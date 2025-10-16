import os
from openai import OpenAI
from dotenv import load_dotenv
from rag.config import OPENAI_API_KEY
from rag.config import PROMPT_ANSWER
from rag.config import OPENAI_CHAT_MODEL
from rag.retrieval.retriever import get_big_chunks_from_small

# === CHARGEMENT CLE API ===

client = OpenAI(api_key=OPENAI_API_KEY)

# === AJOUT DES SOURCES EN ANNEXE===

def append_sources(answer, chunks):

    sources = set()  # Utilise un set pour éviter les doublons de sources
    for chunk in chunks:
        meta = chunk.get("metadata", {})  # Récupère les métadonnées associées au chunk
        titre = meta.get("titre_document", "Sans titre")  # Titre du document (fallback si manquant)
        base = meta.get("source", "source inconnue").upper()  # Nom de la base (ex: BOFiP), en majuscules
        url = meta.get("permalien", None)  # Lien vers la source juridique

        if url:  # Si un permalien est disponible, on ajoute une entrée formatée
            sources.add(f"- [{titre} — {base}]({url})")  # Format Markdown : lien cliquable

    if sources:  # S'il y a au moins une source, on les ajoute à la réponse
        answer += "\n\n📎 **Sources utilisées :**\n" + "\n".join(sorted(sources))

    return answer

# === ENVOI DU CONTEXTE AU LLM ET FORMULATION DE LA REPONSE===

def generate_answer(query, chunks, include_sources=True):
    print("\n🧠 Étape : Génération de la réponse finale")

    context = "\n\n---\n\n".join(
        f"{chunk.get('metadata', {}).get('titre_document', '')}\n"
        f"{chunk.get('metadata', {}).get('titre_bloc', '')}\n"
        f"Source : {chunk.get('metadata', {}).get('permalien', 'Source inconnue')}\n\n"
        f"{chunk.get('contenu', '')}"
        for chunk in chunks
    )

    user_prompt = (
        f"Voici la question : \n\n{query}\n\n"
        f"Voici des extraits juridiques avec leur source issus de bases documentaires :\n\n{context}\n\n"
        f"** 1ère étape ** : En fonction de la question posée et des extraits juridiques, sélectionne ceux qui répondent totalement ou partiellement à la question.\n"
        f"Format attendu : ne doit pas apparaitre dans la réponse.\n"
        f"** 2ème étape ** : Vérifie que les réponses sélectionnées sont cohérentes entre elles.\n"
        f"Format attendu : ne doit pas apparraitre dans la réponse. \n"
        f"Si tu n'as pas assez d'éléments, réponds : \"je n'ai pas assez d'éléments à ma disposition pour répondre\" et ne passe pas aux étapes suivantes.\n"
        f"** 3ème étape ** : Formule un résumé complète des textes en citant les sources.\n"
        f"Format attendu : Sous-partie : Textes juridiques applicables (en gras et avec une police supérieure) - Résumé de l'article - Source\n"
        f"** 4ème étape ** : Explique comment ces textes et uniquement ces texte s’appliquent concrètement à la question. Tu ne dois pas extrapoler\n"
        f"Format attendu : Sous-partie : Application au cas d’espèce (en gras et avec une police supérieure)"
    )

    print("\n📨 Prompt envoyé au LLM :\n")
    print(user_prompt)

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": PROMPT_ANSWER},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()