from rag.retrieval.qdrant_retriever import QdrantRetriever
from rag.generation.answerer import generate_answer

def test_full_answer_generation():
    """
    Test complet : question -> retrieval -> génération -> affichage
    """
    retriever = QdrantRetriever()

    # 🧠 Exemple de question
    question = "Comment fonctionne la taxe sur les véhicules de tourisme des sociétés (TVTS) ?"

    print(f"\n🔎 Question : {question}\n")

    # 1️⃣ Étape de retrieval (clarification, sous-questions, re-ranking)
    results = retriever.retrieve_with_subquery_rerank(question)

    print("✅ Retrieval terminé.")
    print(f"Nombre de sous-questions : {len(results['sous_questions'])}")

    # 2️⃣ Concatène les small chunks re-rankés pour chaque sous-question
    small_chunks = []
    for sq, hits in results["reranked_par_sous_question"].items():
        print(f"\nSous-question : {sq}")
        for h in hits[:5]:  # top 5 par sous-question
            print(f" - {h.get('metadata', {}).get('titre_document', 'Sans titre')} (score={h.get('rerank_score', 0)})")
        small_chunks.extend(hits[:5])

    # 3️⃣ Génération finale
    print("\n🧩 Génération de la réponse complète...")
    answer = generate_answer(results["question_clarifiee"], small_chunks, include_sources=True)

    print("\n=== ✨ RÉPONSE FINALE ✨ ===\n")
    print(answer)


if __name__ == "__main__":
    test_full_answer_generation()
