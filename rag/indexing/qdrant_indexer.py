"""
=== QDRANT INDEXER (compatible client Qdrant <1.10) ===
"""

import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer


def index_documents(
    texts: List[str],
    collection_name: str = "bofip_hybrid",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    client = QdrantClient(url="http://localhost:6333")

    print(f"🔧 Préparation de la collection '{collection_name}'...")

    vector_size = 384 if "MiniLM" in model_name else 768

    if client.collection_exists(collection_name):
        print(f"⚠️ Collection '{collection_name}' déjà existante — suppression...")
        client.delete_collection(collection_name=collection_name)

    # ✅ Création compatible (dictionnaires pour dense et sparse)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": qm.SparseVectorParams(),
        },
    )

    print(f"✅ Collection '{collection_name}' initialisée (hybride dense + sparse).")

    print("🔢 Génération des embeddings denses...")
    dense_model = SentenceTransformer(model_name)
    dense_vectors = dense_model.encode(texts, normalize_embeddings=True)

    print("🔠 Génération des embeddings sparse (HashingVectorizer)...")
    hv = HashingVectorizer(n_features=2**16, alternate_sign=False, norm=None)
    X = hv.transform(texts)

    print("📦 Préparation des points à indexer...")
    points = []
    for i, txt in enumerate(texts):
        row = X[i]
        idx = row.indices.tolist()
        val = row.data.tolist()

        points.append(
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vectors[i].tolist(),
                    "sparse": qm.SparseVector(indices=idx, values=val),
                },
                payload={"text": txt},
            )
        )

        if (i + 1) % 100 == 0:
            print(f"→ {i + 1} documents préparés...")

    print("🚀 Insertion des vecteurs dans Qdrant...")
    client.upsert(collection_name=collection_name, points=points)

    print(f"✅ {len(points)} documents indexés avec succès dans '{collection_name}'.")


if __name__ == "__main__":
    texts = [
        "Définition de l'impôt sur le revenu.",
        "Régime fiscal des sociétés.",
        "TVA applicable sur les ventes intracommunautaires.",
    ]
    index_documents(texts)
