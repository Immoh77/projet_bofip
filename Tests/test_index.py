from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
info = client.get_collection("bofip_hybrid")
print(info.config)