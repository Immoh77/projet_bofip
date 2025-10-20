import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models as qm
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# --------------------------
# ⚙️ CONFIGURATION
# --------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("🚨 Clé API OpenAI manquante dans .env")

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bofip_hybrid"
VECTOR_SIZE = 1536  # modèle text-embedding-3-small

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_qdrant = QdrantClient(url=QDRANT_URL)
hv = HashingVectorizer(n_features=2**16, alternate_sign=False, norm=None)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# --------------------------
# 🧠 PHRASE DE TEST
# --------------------------
QUERY = "Quels sont les régimes fiscaux applicables aux entreprises innovantes ?"

# --------------------------
# 🔹 RECHERCHE SÉMANTIQUE
# --------------------------
def semantic_search(query_text):
    logging.info("🔹 Recherche sémantique (via embeddings OpenAI)...")

    # Génération embedding
    emb = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query_text]
    ).data[0].embedding

    # Requête dans Qdrant
    results = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=emb,
        using="dense",  # vecteur dense
        limit=6,
        with_payload=True
    )
    return results.points

# --------------------------
# 🔸 RECHERCHE LEXICALE
# --------------------------
def lexical_search(query_text):
    logging.info("🔸 Recherche lexicale (via HashingVectorizer)...")

    X = hv.transform([query_text])
    indices = X[0].indices.tolist()
    values = X[0].data.tolist()

    sparse_vector = qm.SparseVector(indices=indices, values=values)

    results = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vector,
        using="sparse",  # vecteur lexical
        limit=6,
        with_payload=True
    )
    return results.points

# --------------------------
# 🚀 EXÉCUTION DU TEST
# --------------------------
if __name__ == "__main__":
    logging.info(f"🧠 Requête : {QUERY}\n")

    dense_results = semantic_search(QUERY)
    sparse_results = lexical_search(QUERY)

    print("\n=== 🔹 Résultats SÉMANTIQUES (OpenAI) ===")
    for i, r in enumerate(dense_results, start=1):
        text = r.payload.get("text", "").replace("\n", " ")
        print(f"{i}. [Score={r.score:.4f}] {text[:200]}...")

    print("\n=== 🔸 Résultats LEXICAUX (HashingVectorizer) ===")
    for i, r in enumerate(sparse_results, start=1):
        text = r.payload.get("text", "").replace("\n", " ")
        print(f"{i}. [Score={r.score:.4f}] {text[:200]}...")
