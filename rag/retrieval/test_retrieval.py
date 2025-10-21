# test_retrieval_pipeline.py
from rag.retrieval.qdrant_retriever import QdrantRetriever

if __name__ == "__main__":
    retriever = QdrantRetriever()
    question = "Comment fonctionne la TVTS ?"

    print("\n🧠 Question d'origine :", question)
    result = retriever.retrieve_with_subquery_rerank(question)

    print("\n=== 🧹 Question clarifiée ===")
    print(result["question_clarifiee"])

    print("\n=== 🔍 Sous-questions générées ===")
    for s in result["sous_questions"]:
        print(" -", s)

    print("\n=== 📄 Re-ranking par sous-question ===")
    for sq, hits in result["reranked_par_sous_question"].items():
        print(f"\n--- {sq} ---")
        for i, h in enumerate(hits[:3], start=1):
            print(f"{i}. score={h.get('rerank_score', 0):.2f} → {h.get('text', '')[:150]}...")

    print("\n=== 🧩 Fusion finale des résultats ===")
    for i, f in enumerate(result["fusion_finale"][:10], start=1):
        print(f"{i}. chunk_id={f['chunk_id']} | score_final={f['score_final']:.2f}")
