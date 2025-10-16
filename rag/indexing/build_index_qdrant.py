# rag/indexing/build_index_qdrant.py
# --- bootstrapping pour exécution directe ---
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------------------------

import json
from rag.config import SMALL_CHUNKS_JSON_PATH, DOCUMENT_SOURCES
from rag.retrieval.qdrant_backend import upsert_chunks, ensure_collection

def _load_small_chunks():
    with open(SMALL_CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        bofip = json.load(f)
    code_sm = DOCUMENT_SOURCES["code_assurances"]["OUTPUT_SMALL_CHUNKS"]
    with open(code_sm, "r", encoding="utf-8") as f:
        code = json.load(f)
    return bofip + code

def main():
    ensure_collection()
    chunks = _load_small_chunks()
    print(f"📦 Total small chunks à indexer : {len(chunks)}")
    upsert_chunks(chunks, batch=256)
    print("✅ Ingestion Qdrant terminée.")

if __name__ == "__main__":
    main()
