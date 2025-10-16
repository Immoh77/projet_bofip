# tests/test_qdrant.py

import pytest
from rag.retrieval.qdrant_backend import ensure_collection, upsert_chunks, hybrid_search
from rag.data.loader import load_all_chunks  # si tu as un loader

def test_index_and_query():
    # 1. indexer quelques chunks
    chunks = load_all_chunks()[:50]  # juste les 50 premiers pour un test rapide
    ensure_collection()
    upsert_chunks(chunks, batch=16)

    # 2. faire une recherche
    question = "Quelle est la condition de résiliation d’un contrat d’assurance ?"
    results = hybrid_search(question, top_k=5)

    print("Résultats de test Qdrant :")
    for r in results:
        print(f"- id={r.get('chunk_id')} score={r.get('score_hybrid'):.4f}")

    assert isinstance(results, list)
    assert len(results) > 0
    # au moins un résultat avec score > 0.0
    assert any(r.get("score_hybrid", 0) > 0 for r in results)
