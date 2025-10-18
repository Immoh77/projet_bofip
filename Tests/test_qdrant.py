import pytest
from qdrant_client import QdrantClient

COLLECTION_NAME = "bofip_hybrid"

@pytest.fixture(scope="module")
def qdrant_client():
    """Initialise un client Qdrant local."""
    return QdrantClient(url="http://localhost:6333")

def test_connection(qdrant_client):
    """Vérifie que la connexion à Qdrant fonctionne."""
    try:
        qdrant_client.get_collections()
        assert True
    except Exception as e:
        pytest.fail(f"Qdrant n'est pas joignable : {e}")

def test_collection_exists(qdrant_client):
    """Vérifie que la collection 'bofip_hybrid' existe."""
    collections = qdrant_client.get_collections()
    names = [c.name for c in collections.collections]
    assert COLLECTION_NAME in names, f"La collection {COLLECTION_NAME} est introuvable."

def test_search_vector(qdrant_client):
    """Effectue une requête simple de similarité vectorielle."""
    import random
    query_vector = [random.random() for _ in range(384)]  # ✅ bonne dimension

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",  # vecteur nommé 'dense' dans ta collection
        limit=3,
    )

    assert results is not None, "La recherche n'a rien retourné."
    assert len(results.points) > 0, "Aucun résultat trouvé dans Qdrant."

    for point in results.points:
        print(f"ID={point.id}, Score={point.score:.4f}, Payload={point.payload}")

