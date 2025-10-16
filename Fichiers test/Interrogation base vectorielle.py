import os
import sys
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# === OPTION 1 : AJOUT MANUEL DU CHEMIN PROJET ===
# On cherche la racine du projet (là où se trouve le dossier "rag")
CURRENT_DIR = Path(__file__).resolve()
for parent in CURRENT_DIR.parents:
    if (parent / "rag" / "config.py").exists():
        sys.path.append(str(parent))
        break
else:
    raise RuntimeError("❌ Impossible de trouver le dossier contenant 'rag/config.py'")

# === CHARGEMENT DES VARIABLES ===
load_dotenv()

from rag.config import CHROMA_PATH, COLLECTION_NAME

# === CONNEXION À CHROMA ===
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# === INSPECTION ===
print("📥 Récupération des métadonnées...")
data = collection.get(include=["metadatas"], limit=200_000)

metadatas = data["metadatas"]
total = len(metadatas)
print(f"\n📊 Nombre total de vecteurs dans la collection : {total}")

# Répartition par base
counter_base = Counter(meta.get("base", "INCONNU") for meta in metadatas)
print("\n📁 Répartition par base :")
for base, count in counter_base.items():
    print(f"  - {base}: {count}")

# Répartition par source
counter_source = Counter(meta.get("source", "INCONNU") for meta in metadatas)
print("\n📁 Répartition par source :")
for source, count in counter_source.items():
    print(f"  - {source}: {count}")