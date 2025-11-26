import os
from dotenv import load_dotenv
from pathlib import Path

# Chargement des variables d’environnement
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# === OpenAI ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = "chatgpt-4o-latest"  # "gpt-4o" ou "gpt-3.5-turbo" "chatgpt-4o-latest"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# === Qdrant ===
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bofip_hybrid")
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))

# === Fichiers ===
SOURCE_FILE = BASE_DIR / "data" / "raw" / "fiscale" / "bofip" / "bofip-vigueur.json"
OUTPUT_BIG_CHUNKS = "data/processed/bofip_chunks_bs.json"
OUTPUT_SMALL_CHUNKS = "data/processed/bofip_small_chunks.json"
SMALL_CHUNKS_JSON_PATH = BASE_DIR / "data" / "processed" / "all_small_chunks.json"
CHARTE_IA_PATH = BASE_DIR / "Charte IA" / "Charte IA.pdf"

DOCUMENT_SOURCES = {
    "code_assurances": {
        "PDF_PATH": BASE_DIR / "data" / "raw" / "juridique" / "Code_des_assurances.pdf",
        "OUTPUT_BIG_CHUNKS": BASE_DIR / "data" / "processed" / "code_assurances_chunks.json",
        "OUTPUT_SMALL_CHUNKS": BASE_DIR / "data" / "processed" / "code_assurances_small_chunks.json",
        "CHUNK_SIZE": 800,
        "CHUNK_OVERLAP": 0,
    },
}

# === Paramètres de découpage ===
CHUNK_SIZE = 800
CHUNK_OVERLAP = 0

# === Séries autorisées ===
ALLOWED_SERIES = {
    "IR", "RSA", "RPPM", "BIC", "IS", "TVA", "TCA",
    "CVAE", "TPS", "TFP", "ENR", "TCAS", "AIS", "RES"
}
EXCLUDED_DOCUMENT_PREFIXES = ["ACTU"]

# === Indexation / vectorisation ===
CHROMA_PATH = BASE_DIR / "chroma_bofip"
COLLECTION_NAME = "bofip_chunks"
EMBEDDINGS_PATH = "embeddings.pkl"
BATCH_SIZE = 75
TOP_K = 20
LEXICAL_WEIGHT = 0.7
MIN_SIMILARITY = 0.35

# === Prompts ===

# === Prompts RAG / Retriever ===
PROMPT_CLARIFY_QUESTION = (
    "Voici la question :\n{question}\n\n"
    "Ton rôle est de la reformuler pour qu'elle puisse être utilisée dans un système RAG. "
    "Tu ne dois rien faire d'autre."
)

PROMPT_SUBQUESTIONS = (
    "Voici la question d’un utilisateur :\n{question}\n\n"
    "Ton rôle est de décomposer cette question en 2 à 4 sous-questions courtes et claires, "
    "chacune sur une seule ligne, sans explications ni puces inutiles.\n\n"
    "⚖️ Règles :\n"
    "- Si la question concerne un thème spécifique (ex. taxe sur les salaires, TVA, IS, impôt sur le revenu, etc.), "
    "chaque sous-question doit explicitement rappeler ce thème.\n"
    "- Reformule les sous-questions pour qu’elles soient autonomes et informatives, sans dépendre du contexte implicite.\n"
    "- Si la question est déjà simple, renvoie-la telle quelle.\n"
    "- Ne fournis aucune introduction ni commentaire, uniquement les sous-questions séparées par des retours à la ligne.\n\n"
    "Sous-questions :\n"
    "Le déficit provenant d’une activité professionnelle X est-il imputable sur le bénéfice d’une autre activité Y au sein de la même entreprise pour l’impôt sur le revenu ?\n"
    "Quelles sont les règles d’imputation des déficits professionnels entre différentes activités exercées dans la même structure soumise à l’impôt sur le revenu ?\n"
    "Le changement ou l’ajout d’une activité (passage de l’activité X à Y) a-t-il une incidence sur la possibilité de reporter un déficit antérieur pour l’impôt sur le revenu ?\n"
    "Existe-t-il des restrictions à l’imputation des déficits lorsque les activités X et Y relèvent de catégories fiscales différentes (BIC, BNC, BA) ?"
)


PROMPT_RERANK_LOCAL = (
    "Question : {question}\n\nTexte : {text}\n\n"
    "Note la pertinence de ce texte pour répondre à la question sur une échelle de 0 (inutile) à 5 (très pertinent). "
    "Répond uniquement par un nombre."
)

PROMPT_RERANK = (
    "Tu es un expert-comptable chargé de trouver les extraits les plus utiles pour répondre à une question. "
    "Ignore les extraits : hors sujet, trop vagues, ou qui abordent des dispositifs non évoqués dans la question. "
    "Ta réponse doit contenir uniquement les numéros des extraits pertinents, triés par ordre décroissant de pertinence (ex. : 4, 2, 1). "
    "Tu ne dois rien expliquer, ne rien reformuler, ne rien justifier."
)

PROMPT_ANSWER_HEADER = (
    "Tu es un expert-comptable chargé de répondre à une question en fonction d'éléments communiqués"
)

PROMPT_USER_ANSWER = (
    "Voici la question initiale posée par l’utilisateur :\n {question_originale}\n\n"
    "Voici la reformulation de cette question par le système :\n {question_clarifiee}\n\n"
    "Voici les sous-questions générées pour structurer la recherche :\n{subquestions}\n\n"
    "Voici maintenant les extraits de textes issus de la recherche RAG "
    ":\n\n{context}\n\n"
    "=== Consignes multi-étapes ===\n"
    f"** 1ère étape ** : En fonction de la question posée et des extraits juridiques, sélectionne ceux qui répondent totalement ou partiellement à la question.\n"
    f"Format attendu : ne doit pas apparaitre dans la réponse.\n"
    f"** 2ème étape ** : Vérifie que les réponses sélectionnées sont cohérentes entre elles.\n"
    f"Format attendu : ne doit pas apparraitre dans la réponse. \n"
    f"Si tu n'as pas assez d'éléments, réponds : \"je n'ai pas assez d'éléments à ma disposition pour répondre\" et ne passe pas aux étapes suivantes.\n"
    f"** 3ème étape ** : Formule un résumé complet des textes en citant les sources.\n"
    f"Format attendu : Sous-partie : Commence la section par un titre Markdown H2 : '## 📜 **TEXTES JURIDIQUES APPLICABLES**'"
    f"Titre de l'article (en gras) \n Résumé de l'article \n Source\n"
    f"** 4ème étape ** : Explique comment ces textes et uniquement ces texte s’appliquent concrètement à la question. Tu ne dois pas extrapoler\n"
    f"Format attendu : Sous-partie : Commence la section par un titre Markdown H2 : '## 🔍 **APPLICATION AU CAS D'ESPÈCE**'"
)

PROMPT_ANSWER = (
    "Tu es un expert-comptable chargé de répondre à une question en fonction d'éléments communiqués."
)

# === Filtre de métadonnées (FILTER_TREE) ===
FILTER_TREE = {
    "fiscal": {
        "bofip": {
            "BIC": [
                "AMT", "BASE", "CESS", "CHAMP", "CHG", "DECLA", "DEF", "PDSTK",
                "PROCD", "PROV", "PTP", "PVMV", "RICI"
            ],
            "CVAE": ["BASE", "CHAMP", "DECLA", "LIEU", "LIQ", "PROCD"],
            "ENR": ["AVS", "DG", "DMTG", "DMTOI", "JOMI", "PTG", "TIM"],
            "IR": ["BASE", "CESS", "CHAMP", "DECLA", "DOMIC", "LIQ", "PROCD", "RICI"],
            "IS": [
                "BASE", "CESS", "CHAMP", "DECLA", "DEF", "FUS", "GEO", "GPE",
                "LIQ", "PROCD", "RICI"
            ],
            "RPPM": ["PVBMC", "PVBMI", "RCM"],
            "RSA": ["BASE", "CHAMP", "ES", "GEO", "GER", "PENS"],
            "TCA": [
                "AHJ", "AUTO", "BEU", "CAEA", "CAR", "CDP", "CPD", "CSR", "EHR", "EOL",
                "FIN", "FTPV", "IMP", "INPES", "MEDIC", "OCE", "PCT", "PJP", "PPA",
                "PRT", "PTV", "RPE", "RSAB", "RSD", "RSP", "SECUR", "SIPV", "TAB",
                "THA", "TPA", "TPC", "VLV"
            ],
            "TCAS": ["ASSUR", "AUT"],
            "TFP": [
                "AIFER", "ASSUR", "CAP", "GUF", "IFER", "MINES", "PYL",
                "RSB", "TASC", "TEM", "TSC", "TVS"
            ],
            "TPS": ["FPC", "PEEC", "TA", "TS"],
            "TVA": ["BASE", "CHAMP", "DECLA", "DED", "GEO", "IMM", "LIQ", "PROCD", "SECT"],
            "AIS": ["MOB", "CCN"],
            "RES": [""]
        }
    },
    "Juridique": {
        "Code des assurances": {
            "Le contrat": [""],
            "Assurances obligatoires": [""],
            "Les entreprises.": [""],
            "Organisations et régimes particuliers d'assurance": [""],
            "Distributeurs d'assurances": [""]
        }
    },
    "Sociale": {
        "Convention collective": {
            "": [""]
        }
    },
    "Générale": {
        "": {
            "": [""]
        }
    }
}

# ==========================================================
# === AJOUTS NÉCESSAIRES POUR APP.PY ===
# ==========================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", OPENAI_CHAT_MODEL)

# === Paramètres du pipeline RAG / Retriever ===
TOP_K_SUBQUESTION = 5        # Nombre de résultats par sous-question
TOP_K_FINAL = 15             # Nombre total après fusion
MAX_SUBQUERIES = 4           # Nombre max de sous-questions
PREFETCH_K = 10              # Préchargement Qdrant (dense + sparse)
BIG_CHUNKS_JSON_PATH = os.getenv(
    "BIG_CHUNKS_JSON_PATH",
    str(BASE_DIR / "data" / "processed" / "bofip_chunks_bs.json")
)

# === Logs ===
LOG_DIR = BASE_DIR / "logs"