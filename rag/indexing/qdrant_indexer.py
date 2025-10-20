import os
import json
import uuid
import math
import logging
import traceback
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from qdrant_client import QdrantClient, models as qm
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

# -------------------------
# ⚙️ CONFIG
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("🚨 Clé API OpenAI manquante ! Vérifie ton fichier .env")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "bofip_hybrid")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

# Modèle OpenAI utilisé
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# 1536 pour text-embedding-3-small, 3072 pour large, etc.
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))

# Données à indexer
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "data/processed/bofip_small_chunks.json")

# Batching
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))

# Logs
os.makedirs("logs", exist_ok=True)
LOG_FILE = os.path.abspath(os.getenv("LOG_FILE", "logs/indexation_openai.log"))

# -------------------------
# 🪵 LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("qdrant_indexer")
log.info(f"🧭 Fichier indexeur: {os.path.abspath(__file__)}")
log.info(f"🧾 Fichier de logs: {LOG_FILE}")

# Rendre verbeux les requêtes HTTP du client Qdrant si besoin :
logging.getLogger("qdrant_client").setLevel(logging.INFO)

# -------------------------
# 🔌 Clients
# -------------------------
client_openai  = OpenAI(api_key=OPENAI_API_KEY)
client_qdrant  = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
hv = HashingVectorizer(n_features=2**16, alternate_sign=False, norm=None)

# -------------------------
# ✅ Sanity checks
# -------------------------
def assert_finite(vec: List[float], where: str):
    arr = np.asarray(vec, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Vecteur non-fini (NaN/Inf) détecté dans {where}")

def embed_openai(batch_texts: List[str]) -> List[List[float]]:
    resp = client_openai.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch_texts)
    dense = [d.embedding for d in resp.data]
    for i, v in enumerate(dense):
        if len(v) != VECTOR_SIZE:
            raise ValueError(f"Dimension embedding inattendue: {len(v)} ≠ {VECTOR_SIZE} (idx batch {i})")
        assert_finite(v, f"embedding dense idx {i}")
    return dense

def to_sparse_lists(X):
    # Retourne deux listes parallèles [indices], [values] pour chaque ligne
    indices_list = []
    values_list = []
    for i in range(X.shape[0]):
        row = X[i]
        idx = row.indices.tolist()
        val = row.data.astype(np.float32).tolist()
        # Filtre préventif : éviter des points totalement vides (Qdrant n’aime pas trop)
        if len(idx) == 0:
            # on met un "token" quasi nul pour éviter les crashs
            idx, val = [0], [1e-9]
        indices_list.append(idx)
        values_list.append(val)
    return indices_list, values_list

def recreate_collection_if_needed():
    exists = client_qdrant.get_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client_qdrant.get_collections().collections] else None
    need_recreate = False
    if exists:
        # Vérifie présence des named vectors attendus
        config = exists.config
        has_dense = "dense" in (config.params.vectors or {})
        has_sparse = bool(config.params.sparse_vectors)
        if not has_dense or not has_sparse:
            log.warning("⚠️ Schéma de collection incompatible. Recréation nécessaire.")
            need_recreate = True

    if need_recreate:
        client_qdrant.delete_collection(COLLECTION_NAME)
        log.info("🧹 Collection supprimée.")

    if (not exists) or need_recreate:
        log.info(f"🛠️ Création de la collection '{COLLECTION_NAME}' (dense {VECTOR_SIZE}/cosine + sparse).")
        client_qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )
        log.info("✅ Collection prête.")

def main():
    log.info("🚀 Démarrage de l’indexation Qdrant.")
    # Test OpenAI
    try:
        _ = client_openai.embeddings.create(model=OPENAI_EMBED_MODEL, input=["ping"])
        log.info(f"✅ OpenAI OK — modèle: {OPENAI_EMBED_MODEL} (dim {VECTOR_SIZE})")
    except Exception as e:
        log.exception(f"❌ OpenAI KO: {e}")
        raise

    # Collection prête
    recreate_collection_if_needed()

    # Lit les chunks
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"❌ Fichier introuvable: {os.path.abspath(CHUNKS_FILE)}")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c.get("contenu", "") for c in chunks]
    metas = [c.get("metadata", {}) for c in chunks]
    log.info(f"📦 {len(texts):,} chunks à indexer depuis {os.path.abspath(CHUNKS_FILE)}")

    # Sparse global (pour éviter de re-vectoriser dans les batchs)
    log.info("🪶 Génération sparse (HashingVectorizer)…")
    Xs = hv.transform(texts)
    all_idx, all_val = to_sparse_lists(Xs)

    # Indexation par batch
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Indexation"):
        end = min(start + BATCH_SIZE, len(texts))
        subtexts = texts[start:end]

        try:
            dense = embed_openai(subtexts)
            points = []
            for i in range(len(subtexts)):
                sp = qm.SparseVector(indices=all_idx[start + i], values=all_val[start + i])
                pts = qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": dense[i], "sparse": sp},
                    payload={"text": subtexts[i], **(metas[start + i] or {})},
                )
                points.append(pts)

            client_qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
            log.info(f"✅ Batch {start//BATCH_SIZE + 1} inséré ({end}/{len(texts)})")
        except Exception as e:
            # Log hyper complet + chemin
            log.error("❌ Erreur pendant l’upsert batch.")
            log.error(f"Fichier: {os.path.abspath(__file__)}")
            log.error(f"Batch range: {start}-{end}")
            log.exception(e)
            # On continue ou on stoppe selon ton choix:
            raise  # si tu préfères continuer: remplace par `continue`

    log.info(f"🎉 Fini: {len(texts):,} points insérés dans '{COLLECTION_NAME}'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Dernière barrière de logs (chemin inclus)
        logging.error("💥 Exception non gérée.")
        logging.error(f"Fichier: {os.path.abspath(__file__)}")
        logging.error(traceback.format_exc())
        raise
