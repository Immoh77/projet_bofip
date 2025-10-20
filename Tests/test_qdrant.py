from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
import random

COLLECTION_NAME = "bofip_hybrid"


def test_connection():
    """Vérifie la connexion à Qdrant."""
    client = QdrantClient(url="http://localhost:6333")
    resp = client.get_collections()
    assert resp is not None
    assert any(c.name == COLLECTION_NAME for c in resp.collections)
    print("✅ Connexion et collection détectées")


def test_collection_exists():
    """Vérifie que la collection existe."""
    client = QdrantClient(url="http://localhost:6333")
    collections = [c.name for c in client.get_collections().collections]
    assert COLLECTION_NAME in collections
    print(f"✅ Collection {COLLECTION_NAME} présente")


def test_search_vector():
    """Recherche hybride (dense + sparse)"""
    client = QdrantClient(url="http://localhost:6333")

    # Vecteur dense (384 dims)
    query_dense = [random.random() for _ in range(384)]

    # Vecteur sparse fictif
    query_sparse = SparseVector(indices=[10, 200, 555], values=[0.3, 0.5, 0.2])

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query={"dense": query_dense, "sparse": query_sparse},
        limit=3,
    )

    assert results is not None
    assert len(results.points) > 0, "Aucun résultat renvoyé"
    print("✅ Recherche hybride réussie")

