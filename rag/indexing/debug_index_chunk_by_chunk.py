import json
import uuid
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models as qm
from sklearn.feature_extraction.text import HashingVectorizer
import time

# -------------------------
# ⚙️ CONFIGURATION
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("🚨 Clé API OpenAI manquante ! Vérifie ton fichier .env")

COLLECTION_NAME = "bofip_hybrid"
CHUNKS_FILE = "C:/Users/GAUTH/OneDrive/Documents/Code Python/Projets/projet_bofip/data/processed/bofip_small_chunks.json"
QDRANT_URL = "http://localhost:6333"
VECTOR_SIZE = 1536  # text-embedding-3-small

# -------------------------
# 🪵 LOGGING
# -------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/chunks_errors.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# -------------------------
# 🔌 CLIENTS
# -------------------------
client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_qdrant = QdrantClient(url=QDRANT_URL)
hv = HashingVectorizer(n_features=2**16, alternate_sign=False, norm=None)

# -------------------------
# 📂 CHARGEMENT DES CHUNKS
# -------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"📄 {len(chunks):,} chunks à tester...")

# -------------------------
# 🧱 COLLECTION
# -------------------------
collections = [c.name for c in client_qdrant.get_collections().collections]
if COLLECTION_NAME not in collections:
    print(f"🧩 Création de la collection {COLLECTION_NAME}")
    client_qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"sparse": qm.SparseVectorParams()},
    )

# -------------------------
# 🧠 TEST UNITAIRE
# -------------------------
for i, chunk in enumerate(chunks):
    text = chunk.get("contenu", "").strip()
    if not text:
        log.info(f"[VIDE] Chunk {i} - id={chunk.get('id')}")
        continue

    try:
        # Embedding dense
        emb = client_openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        ).data[0].embedding

        # Sparse
        X = hv.transform([text])
        indices = X[0].indices.tolist()
        values = X[0].data.tolist()

        # Point
        point = qm.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": emb,
                "sparse": qm.SparseVector(indices=indices, values=values),
            },
            payload={"text": text, **chunk.get("metadata", {})},
        )

        # Test insertion
        client_qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])

        print(f"✅ {i+1}/{len(chunks)} — OK ({chunk.get('id')})")
        time.sleep(0.05)  # léger délai pour éviter de saturer Qdrant

    except Exception as e:
        err_msg = f"❌ Chunk {i} — id={chunk.get('id')} — Erreur: {e}"
        print(err_msg)
        log.error(err_msg)

print("🎯 Test terminé — consulte 'logs/chunks_errors.log' pour les chunks problématiques.")
