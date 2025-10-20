from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI
from qdrant_client import QdrantClient, models as qm

# === CONFIGURATION ===
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "bofip_hybrid")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === CLIENTS ===
client = QdrantClient(url=QDRANT_URL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# === QUESTION ===
query_text = "Quelles sont les conditions d’exonération de TVA ?"
print(f"\n🧠 Question : {query_text}")

# === EMBEDDING OPENAI ===
embedding = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=[query_text]
).data[0].embedding

# === RECHERCHE DANS QDRANT (SÉMANTIQUE SEULEMENT) ===
print("\n=== 🔵 Recherche SÉMANTIQUE (vecteur dense) ===")

results_sem = client.query_points(
    collection_name=COLLECTION_NAME,
    query=embedding,              # vecteur dense
    using="dense",                # nom du vecteur dans ta collection
    limit=5,
    with_payload=True
)

# === AFFICHAGE DES RÉSULTATS ===
if results.points:
    for i, hit in enumerate(results.points, 1):
        score = round(hit.score, 3)
        text = hit.payload.get("text", "").replace("\n", " ")[:350]
        print(f"\n{i}. [score={score}] {text}...")
else:
    print("⚠️ Aucun résultat trouvé.")

print("\n✅ Recherche sémantique terminée.")
