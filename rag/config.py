import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # Charge les variables d’environnement depuis .env

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = Path("data/rag/chroma_bofip")
CHROMA_COLLECTION = "bofip_code_assurances_v2"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Embedding dense local (FR/multi recommandé) ---
SENTENCE_TRANSFORMERS_MODEL = "intfloat/multilingual-e5-base"

# Modèles OpenAI utilisés
OPENAI_CHAT_MODEL = "chatgpt-4o-latest"     #"gpt-4o"  # gpt-3.5-turbo #chatgpt-4o-latest

# Base vectorielle
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bofip_hybrid")
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))

# Chemins fichiers
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
# Paramètres découpage
CHUNK_SIZE = 800
CHUNK_OVERLAP = 0

# Séries autorisées
ALLOWED_SERIES = {
    "IR", "RSA", "RPPM", "BIC", "IS", "TVA", "TCA",
    "CVAE", "TPS", "TFP", "ENR", "TCAS", "AIS", "RES"
}
EXCLUDED_DOCUMENT_PREFIXES = ["ACTU"]

# Indexation / vectorisation
CHROMA_PATH = BASE_DIR / "chroma_bofip"
COLLECTION_NAME = "bofip_chunks"
EMBEDDINGS_PATH = "embeddings.pkl"
BATCH_SIZE = 75
TOP_K = 20
LEXICAL_WEIGHT = 0.7
MIN_SIMILARITY = 0.35

PROMPT_REWRITE_QUERY = (
    "Voici la question :\n{question}\n\n"
    "Ton role est de la reformuler pour qu'elle puisse être utilisée dans un système RAG ?. Tu ne dois rien faire d'autre"
)

PROMPT_RERANK = (
    "Tu es un expert-comptable chargé de trouver les extraits les plus utiles pour répondre à une question."
    "Ignore les extraits : hors sujet, trop vagues, qui abordent des dispositifs non évoqués dans la question."
    "Ta réponse doit contenir uniquement les numéros des extraits pertinents, triés par ordre décroissant de pertinence (ex. : 4, 2, 1)."
    "Tu ne dois rien expliquer, ne rien reformuler, ne rien justifier."
)

PROMPT_ANSWER = (
    "Tu es un expert-comptable chargé de répondre à une question en fonction d'éléments communiqués"
)

# Filtre métadonnées

FILTER_TREE = {
    "Fiscale": {
        "bofip": {
            "BIC": [
                "AMT", "BASE", "CESS", "CHAMP", "CHG", "DECLA", "DEF", "PDSTK",
                "PROCD", "PROV", "PTP", "PVMV", "RICI"
            ],
            "CVAE": [
                "BASE", "CHAMP", "DECLA", "LIEU", "LIQ", "PROCD"
            ],
            "ENR": [
                "AVS", "DG", "DMTG", "DMTOI", "JOMI", "PTG", "TIM"
            ],
            "IR": [
                "BASE", "CESS", "CHAMP", "DECLA", "DOMIC", "LIQ", "PROCD", "RICI"
            ],
            "IS": [
                "BASE", "CESS", "CHAMP", "DECLA", "DEF", "FUS", "GEO", "GPE", "LIQ", "PROCD", "RICI"
            ],
            "RPPM": [
                "PVBMC", "PVBMI", "RCM"
            ],
            "RSA": [
                "BASE", "CHAMP", "ES", "GEO", "GER", "PENS"
            ],
            "TCA": [
                "AHJ", "AUTO", "BEU", "CAEA", "CAR", "CDP", "CPD", "CSR", "EHR", "EOL", "FIN",
                "FTPV", "IMP", "INPES", "MEDIC", "OCE", "PCT", "PJP", "PPA", "PRT", "PTV", "RPE",
                "RSAB", "RSD", "RSP", "SECUR", "SIPV", "TAB", "THA", "TPA", "TPC", "VLV"
            ],
            "TCAS": [
                "ASSUR", "AUT"
            ],
            "TFP": [
                "AIFER", "ASSUR", "CAP", "GUF", "IFER", "MINES", "PYL", "RSB", "TASC", "TEM",
                "TSC", "TVS"
            ],
            "TPS": [
                "FPC", "PEEC", "TA", "TS"
            ],
            "TVA": [
                "BASE", "CHAMP", "DECLA", "DED", "GEO", "IMM", "LIQ", "PROCD", "SECT"
            ],
            "AIS": [
                "MOB", "CCN"
            ],
            "RES": [
                ""
            ]
        }
    },
    "Juridique": {
        "Code des assurances": {
            "Le contrat": [
                ""
            ],
            "Assurances obligatoires": [
                ""
            ],
            "Les entreprises.": [
                ""
            ],
            " Organisations et régimes particuliers d'assurance": [
                ""
            ],
            "Distributeurs d'assurances": [
                ""
            ]
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

