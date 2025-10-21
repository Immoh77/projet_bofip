"""
test_query_qdrant.py
====================

Test complet pour une base vectorielle hybride (dense + sparse) sous Qdrant.
Affiche séparément les résultats lexicaux, sémantiques et hybrides.
Compatible avec Qdrant 1.15+.
"""

# ------------------------------
# 📦 IMPORTS
# ------------------------------
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models as qm
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# ------------------------------
# ⚙️ CONFIGURATION
# ------------------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "bofip_hybrid")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise ValueError("🚨 Clé API OpenAI manquante ! Vérifie ton fichier .env")

# ------------------------------
# 🔌 INITIALISATION
# ------------------------------
print("🔗 Connexion à Qdrant et OpenAI...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
hv = HashingVectorizer(n_features=2**16, alternate_sign=False, norm=None)

# ------------------------------
# 💬 SAISIE REQUÊTE
# ------------------------------
if len(sys.argv) > 1:
    query_text = " ".join(sys.argv[1:])
else:
    query_text = input("Entrez votre requête : ")

print(f"\n🧠 Requête : {query_text}\n")

# ------------------------------
# 🧮 GÉNÉRATION DES VECTEURS
# ------------------------------
print("⚙️ Génération du vecteur dense (OpenAI)...")
dense_vector = openai_client.embeddings.create(
    model=OPENAI_EMBED_MODEL,
    input=query_text
).data[0].embedding

print("⚙️ Génération du vecteur sparse (HashingVectorizer)...")
X = hv.transform([query_text])
idx = X[0].indices.tolist()
val = X[0].data.astype(np.float32).tolist()
if not idx:
    idx, val = [0], [1e-9]
sparse_vector = qm.SparseVector(indices=idx, values=val)

# ------------------------------
# 🔍 RECHERCHE SÉPARÉE : SPARSE
# ------------------------------
print("\n🔎 Recherche lexicale (sparse)...")
sparse_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=sparse_vector,
    using="sparse",
    limit=5,
    with_payload=True,
).points

# ------------------------------
# 🔍 RECHERCHE SÉPARÉE : DENSE
# ------------------------------
print("\n🔎 Recherche sémantique (dense)...")
dense_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=dense_vector,
    using="dense",
    limit=5,
    with_payload=True,
).points

# ------------------------------
# 🤝 RECHERCHE HYBRIDE (RRF)
# ------------------------------
print("\n⚡ Recherche hybride (fusion RRF)...")
hybrid_results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        qm.Prefetch(query=sparse_vector, using="sparse", limit=20),
        qm.Prefetch(query=dense_vector, using="dense", limit=20),
    ],
    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
    limit=10,
    with_payload=True,
).points

# ------------------------------
# 📜 AFFICHAGE DES RÉSULTATS
# ------------------------------
def print_results(title, results, label):
    print(f"\n=== {title} ===\n")
    if not results:
        print(f"(Aucun résultat trouvé pour {label})\n")
        return
    for i, r in enumerate(results, start=1):
        score = round(r.score, 4)
        text = r.payload.get("text", "").replace("\n", " ")[:250]
        print(f"{i}. [{label}] (score={score}) {text}...\n")

# Affiche les trois catégories de résultats
print_results("🔹 Résultats lexicaux (sparse)", sparse_results, "lexical")
print_results("🔹 Résultats sémantiques (dense)", dense_results, "semantic")
print_results("🔹 Résultats hybrides (fusion dense + sparse)", hybrid_results, "hybrid")

print("\n✅ Test terminé avec succès.")
