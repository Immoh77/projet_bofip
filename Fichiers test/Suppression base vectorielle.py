import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# === OPTION 1 : DÉTECTION DU CHEMIN PROJET ===
CURRENT_DIR = Path(__file__).resolve()
for parent in CURRENT_DIR.parents:
    if (parent / "rag" / "config.py").exists():
        sys.path.append(str(parent))
        break
else:
    raise RuntimeError("❌ Impossible de trouver le dossier contenant 'rag/config.py'")

# === CONFIGURATION ===
load_dotenv()

from rag.config import CHROMA_PATH, COLLECTION_NAME

# === CONNEXION À CHROMA ===
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# === SUPPRESSION DES VECTEURS BOFIP ===
print("🔍 Recherche des documents à supprimer...")

collection.delete(
    where={"source": "Code des assurances"}
)

print("🧹 Suppression terminée.")
print(f"📊 Nombre de documents restants dans la base : {collection.count()}")

