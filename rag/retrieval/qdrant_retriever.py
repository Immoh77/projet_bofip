from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
import os
import uuid
from dotenv import load_dotenv
import logging

load_dotenv()

# === CONFIGURATION DU LOGGING ===
logging.basicConfig(
    level=logging.INFO,  # Niveau de log : DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("qdrant_index.log", mode="a", encoding="utf-8"),  # écrit dans un fichier
        logging.StreamHandler(),  # affiche aussi dans le terminal
    ]
)
logger = logging.getLogger(__name__)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "bofip_hybrid")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
dense_model = SentenceTransformer("intfloat/multilingual-e5-base")
sparse_vec = HashingVectorizer(ngram_range=(1, 2), n_features=50000, alternate_sign=False, norm=None)

def ensure_collection():
    """Crée la collection hybride si elle n’existe pas."""
    if client.collection_exists(COLLECTION):
        logger.info(f"✅ Collection '{COLLECTION}' déjà existante.")
        return

    logger.info(f"🛠️  Création de la collection '{COLLECTION}' ...")

    vectors_config = {
        "dense": qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE)
    }
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=vectors_config,
        sparse_vectors_config={"sparse": qm.SparseVectorParams(index={"on_disk": False})},
    )

    logger.info(f"✅ Collection '{COLLECTION}' créée avec succès.")

def _encode_sparse(texts):
    X = sparse_vec.transform(texts)
    dicts = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        dicts.append({int(k): float(v) for k, v in zip(row.indices, row.data)})
    return dicts


def index_documents(docs):
    logger.info(f"📥 Début d’indexation de {len(docs)} documents dans Qdrant '{COLLECTION}'...")

    try:
        ensure_collection()
        dense = dense_model.encode([d["text"] for d in docs], convert_to_numpy=True, normalize_embeddings=True)
        sparse = _encode_sparse([d["text"] for d in docs])
        logger.info("🧠 Encodage dense et sparse terminé.")

        points = []
        for i, d in enumerate(docs):
            payload = {"text": d["text"], "source": d.get("source")}
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": dense[i].tolist(), "sparse": sparse[i]},
                    payload=payload,
                )
            )

        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        logger.info(f"✅ {len(points)} documents indexés avec succès dans Qdrant.")
    except Exception as e:
        logger.error(f"❌ Erreur pendant l’indexation : {e}")
        raise

def hybrid_search(query: str, top_k=5, alpha=0.5):
    dense_q = dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    sparse_q = _encode_sparse([query])[0]
    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            qm.Prefetch(query=qm.QueryVector(name="dense", vector=dense_q, weight=alpha)),
            qm.Prefetch(query=qm.QueryVector(name="sparse", vector=sparse_q, weight=(1-alpha))),
        ],
        limit=top_k,
        with_payload=True,
    )
    return [{"text": r.payload.get("text"), "score": r.score} for r in results.points]

