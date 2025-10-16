# rag/indexing/test_index.py
import json
import time
import logging
from pathlib import Path

from rag.retrieval.qdrant_backend import upsert_chunks, ensure_collection
from rag.config import SMALL_CHUNKS_JSON_PATH, DOCUMENT_SOURCES

# --- LOGGING lisible ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("index_test")

# --- Chargement des chunks (bofip + code des assurances) ---
def load_chunks():
    paths = [SMALL_CHUNKS_JSON_PATH, DOCUMENT_SOURCES["code_assurances"]["OUTPUT_SMALL_CHUNKS"]]
    all_chunks = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            log.warning(f"Fichier introuvable: {p}")
            continue
        log.info(f"Lecture: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            all_chunks.extend(data)
    log.info(f"Total chunks chargés: {len(all_chunks)}")
    return all_chunks

def main():
    ensure_collection()  # s’assure que la collection existe
    chunks = load_chunks()

    # Limite initiale pour valider (tu peux monter à 20000 ensuite)
    MAX_POINTS = 5000
    BATCH = 256

    log.info(f"Début indexation: max_points={MAX_POINTS}, batch={BATCH}")
    t0 = time.perf_counter()
    upsert_chunks(chunks, batch=BATCH, max_points=MAX_POINTS)
    dt = time.perf_counter() - t0
    log.info(f"✅ Fini en {dt:.1f}s")

if __name__ == "__main__":
    main()
