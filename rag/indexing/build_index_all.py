# rag/indexing/build_index_all.py
from __future__ import annotations
import json
from pathlib import Path

# 1) on réutilise le getter de retriever pour éviter toute divergence
from rag.retrieval.retriever import _get_chroma_collection
from rag.config import (
    SMALL_CHUNKS_JSON_PATH,
    DOCUMENT_SOURCES,
)

def _load_chunks() -> list[dict]:
    # smalls BOFiP
    with open(SMALL_CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        bofip = json.load(f)
    # smalls Code des assurances
    code_sm_path = DOCUMENT_SOURCES["code_assurances"]["OUTPUT_SMALL_CHUNKS"]
    with open(code_sm_path, "r", encoding="utf-8") as f:
        code = json.load(f)
    return bofip + code

def main():
    coll = _get_chroma_collection()

    # ⚠️ si tu veux repartir de zéro à chaque build, décommente la ligne suivante :
    # coll.delete(where={})  # supprime tout le contenu de la collection

    chunks = _load_chunks()
    print(f"📦 Total chunks à indexer : {len(chunks)}")

    # Ingestion par batch
    BATCH = 100
    ids, docs, metas = [], [], []
    for i, ch in enumerate(chunks, 1):
        md = ch.get("metadata", {})
        cid = md.get("chunk_id") or md.get("id") or f"auto-{i}"
        ids.append(str(cid))
        docs.append(ch.get("contenu") or "")
        # on garde toutes les métadonnées utiles aux filtres/affichage
        metas.append(md)

        # push batch
        if len(ids) >= BATCH:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)
            print(f"✅ upsert {i:>6} / {len(chunks)}")
            ids, docs, metas = [], [], []

    # reste
    if ids:
        coll.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"✅ upsert final : {len(ids)}")

    print("🎉 Index v2 (hybride dense+sparse) prêt.")

if __name__ == "__main__":
    main()