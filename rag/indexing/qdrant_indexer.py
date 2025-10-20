import json
import uuid
import logging
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models as qm
from sklearn.feature_extraction.text import HashingVectorizer

# -------------------------
# ⚙️ CONFIGURATION
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("🚨 Clé API OpenAI manquante ! Vérifie ton fichier .env")

COLLECTION_NAME = "bofip_hybrid"
CHUNKS_FILE = "data/processed/bofip_small_chunks.json"
QDRANT_URL = "http://localhost:6333"
BATCH_SIZE = 500
VECTOR_SIZE = 1536  # text-embedding-3-small
LOG_FILE = "logs/indexation_openai.log"

# -------------------------
# 🪵 LOGGING
# -------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# -------------------------
# 🔌 INITIALISATION DES CLIENTS
# -------------------------
client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_qdrant = QdrantClient(url=QDRANT_URL)
hv = HashingVectorizer(n_features=2**16, alternate_sign=False, norm=None)

# -------------------------
# 🧪 TEST DE CONNEXION OPENAI
# -------------------------
try:
    test = client_openai.embeddings.create(model="text-embedding-3-small", input=["test"])
    log.info("✅ Connexion OpenAI réussie — modèle accessible.")
except Exception as e:
    log.error(f"🚨 Erreur lors de la connexion à OpenAI : {e}")
    raise SystemExit(1)

# -------------------------
# 🧱 COLLECTION QDRANT
# -------------------------
collections = [c.name for c in client_qdrant.get_collections().collections]
if COLLECTION_NAME not in collections:
    log.info(f"Création de la collection {COLLECTION_NAME}...")
    client_qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"sparse": qm.SparseVectorParams()},
    )
else:
    log.info(f"Collection existante trouvée : {COLLECTION_NAME}")

# -------------------------
# 📄 LECTURE DES CHUNKS
# -------------------------
if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(f"❌ Fichier introuvable : {CHUNKS_FILE}")

log.info(f"📄 Lecture des chunks depuis {CHUNKS_FILE}...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk.get("contenu", "") for chunk in chunks]
log.info(f"📊 {len(texts):,} chunks détectés — début de l’indexation")

# -------------------------
# 🧠 EMBEDDINGS OPENAI
# -------------------------
def embed_openai(batch_texts):
    """Crée des embeddings via l’API OpenAI (text-embedding-3-small)."""
    response = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=batch_texts
    )
    return [d.embedding for d in response.data]

# -------------------------
# 🪶 ENCODAGE SPARSE
# -------------------------
log.info("🪶 Génération des embeddings sparse (HashingVectorizer)...")
X_sparse = hv.transform(texts)
indices_list = [X_sparse[i].indices.tolist() for i in range(X_sparse.shape[0])]
values_list = [X_sparse[i].data.tolist() for i in range(X_sparse.shape[0])]

# -------------------------
# 📤 INDEXATION DANS QDRANT
# -------------------------
log.info("🚀 Début de l’insertion dans Qdrant...")

for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Indexation Qdrant"):
    end = min(start + BATCH_SIZE, len(texts))
    batch_texts = texts[start:end]

    # Génération des embeddings OpenAI par lot
    dense_vectors = embed_openai(batch_texts)

    # Construction des points à insérer
    points = [
        qm.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vectors[i],
                "sparse": qm.SparseVector(
                    indices=indices_list[start + i],
                    values=values_list[start + i],
                ),
            },
            payload={"text": batch_texts[i], **chunks[start + i].get("metadata", {})},
        )
        for i in range(len(batch_texts))
    ]

    # Insertion dans Qdrant
    client_qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    log.info(f"📦 Batch {start//BATCH_SIZE + 1}: {len(points)} points insérés ({end}/{len(texts)})")

log.info(f"✅ Indexation terminée — {len(texts):,} points insérés dans {COLLECTION_NAME}")

