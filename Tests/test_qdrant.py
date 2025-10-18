from qdrant_client import QdrantClient

def test_qdrant_connection():
    client = QdrantClient(url="http://localhost:6333")
    info = client.get_collection("bofip_hybrid")
    assert info.status.value == "green"
    assert info.points_count > 0