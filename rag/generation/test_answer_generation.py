from dotenv import load_dotenv
load_dotenv()

from rag.retrieval.qdrant_retriever import QdrantRetriever
from rag.generation.answerer import generate_answer
import rag.config as config  # ✅ utilisé pour spécifier le modèle

def test_full_answer_generation():
    """
    Test complet : question -> retrieval -> affichage (small + big chunks) -> génération -> réponse
    """
    retriever = QdrantRetriever()

    # 🧠 Exemple de question
    question = "Comment fonctionne la taxe sur les véhicules de tourisme des sociétés (TVTS) ?"

    print(f"\n🔎 Question : {question}\n")

    # 1️⃣ Étape de retrieval
    results = retriever.retrieve_with_subquery_rerank(question)

    print("✅ Retrieval terminé.")
    print(f"Nombre de sous-questions : {len(results['sous_questions'])}")

    # 2️⃣ Affichage des small chunks
    small_chunks = []
    for sq, hits in results["reranked_par_sous_question"].items():
        print(f"\nSous-question : {sq}")
        for h in hits[:5]:
            titre = h.get("metadata", {}).get("titre_document")
            if not titre:
                titre = h.get("page_content", "")[:80].replace("\n", " ") + "..."
            print(f" - {titre} (score={h.get('rerank_score', 0)})")
        small_chunks.extend(hits[:5])

    # 3️⃣ Affichage des big chunks (si présents)
    if "big_chunks" in results:
        print("\n=== 🧩 BIG CHUNKS ===")
        for i, chunk in enumerate(results["big_chunks"], 1):
            titre = chunk.get("metadata", {}).get("titre_document", "Sans titre")
            extrait = chunk.get("page_content", "")[:150].replace("\n", " ")
            print(f"{i}. {titre} → {extrait}...")
    else:
        print("\n⚠️ Aucun big chunk trouvé dans les résultats.")

    # ✅ 4️⃣ Spécifie le modèle à utiliser (comme le ferait app.py)
    config.MODEL_NAME = "gpt-3.5-turbo"

    # 5️⃣ Génération finale (appel du vrai generate_answer)
    print("\n🧠 Génération de la réponse complète...")
    answer = generate_answer(results["question_clarifiee"], small_chunks, include_sources=True)

    print("\n=== ✨ RÉPONSE FINALE ✨ ===\n")
    print(answer)


if __name__ == "__main__":
    test_full_answer_generation()
