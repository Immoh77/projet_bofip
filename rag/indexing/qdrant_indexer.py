import os
import uuid
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer

# === CONFIGURATION ===
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "bofip_hybrid")
VECTOR_SIZE = 768  # modèle "intfloat/multilingual-e5-base"

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("qdrant_index.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# === INITIALISATION ===
dense_model = SentenceTransformer("intfloat/multilingual-e5-base")
sparse_vec = HashingVectorizer(ngram_range=(1, 2), n_features=50000, alternate_sign=False, norm=None)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# -----------------------------------------------------------------------------
# 🧩 UTILITAIRES
# -----------------------------------------------------------------------------
def ensure_collection():
    """Crée la collection hybride si elle n'existe pas."""
    if client.collection_exists(COLLECTION):
        logger.info(f"✅ Collection '{COLLECTION}' déjà existante.")
        return

    logger.info(f"🛠️  Création de la collection hybride '{COLLECTION}' ...")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": qm.SparseVectorParams(index={"on_disk": False}),
        },
    )

    logger.info(f"✅ Collection '{COLLECTION}' créée avec succès (dense + sparse).")


def _encode_sparse(texts):
    """Encode les textes en vecteurs sparses (hashing)."""
    X = sparse_vec.transform(texts)
    dicts = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        dicts.append({str(k): float(v) for k, v in zip(row.indices, row.data)})
    return dicts

# -----------------------------------------------------------------------------
# 🚀 INDEXATION
# -----------------------------------------------------------------------------
def index_documents(docs):
    """
    Indexe une liste de documents dans Qdrant (compatible client 1.15.x).
    On stocke les vecteurs denses dans Qdrant et garde les sparses en payload.
    """
    ensure_collection()
    logger.info(f"📥 Indexation de {len(docs)} documents dans '{COLLECTION}'...")

    dense = dense_model.encode(
        [d["text"] for d in docs],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    sparse = _encode_sparse([d["text"] for d in docs])

    points = []
    for i, d in enumerate(docs):
        payload = {
            "text": d["text"],
            "source": d.get("source", ""),
            "metadata": d.get("metadata", {}),
            "sparse_vector": sparse[i],  # ✅ sauvegardé ici (pas dans "vector")
        }
        points.append(
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=dense[i].tolist(),  # ✅ uniquement le vecteur dense
                payload=payload,
            )
        )

    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    logger.info(f"✅ {len(points)} vecteurs denses indexés (sparse conservé dans le payload).")

# -----------------------------------------------------------------------------
# 🧠 EXECUTION DIRECTE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Lancement du module d'indexation Qdrant")

    # Exemple de documents à indexer
    docs = [
        {"text": "L'impôt sur le revenu est calculé selon un barème progressif.", "source": "bofip_impot.html"},
        {"text": "Les contrats d’assurance peuvent être résiliés après un an conformément à la loi Hamon.", "source": "bofip_assurance.html"},
        {"text": "Le taux de TVA normal en France est de 20%.", "source": "bofip_tva.html"},
    ]

    try:
        index_documents(docs)
        logger.info("✅ Indexation terminée avec succès.")
    except Exception as e:
        logger.error(f"❌ Erreur pendant l’indexation : {e}")
