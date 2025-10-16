import os
import json
import pickle
from math import ceil
from time import sleep
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from config import SMALL_CHUNKS_JSON_PATH, CHROMA_PATH, EMBEDDINGS_PATH, BATCH_SIZE



load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === OUTILS ===

def batch_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def clean_metadata(metadata):
    return {k: (v if v is not None else "") for k, v in metadata.items()}

def add_to_chroma_in_batches(collection, documents, embeddings, metadatas, ids, batch_size=5000):
    total = len(documents)
    for i in range(0, total, batch_size):
        print(f"📥 Ajout batch {i} → {min(i+batch_size, total)} / {total}")
        collection.add(
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

# === CHARGEMENT DES CHUNKS ===

print("🔄 Chargement des small chunks...")
with open(SMALL_CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

documents = [chunk["contenu"] for chunk in chunks]
ids = [f"{chunk['parent_chunk_id']}_{chunk['small_index']}" for chunk in chunks]
metadatas = [clean_metadata(chunk["metadata"]) for chunk in chunks]

# === GESTION DU CACHE EMBEDDINGS ===

if os.path.exists(EMBEDDINGS_PATH):
    print("📂 Chargement des embeddings depuis le cache...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings = pickle.load(f)
else:
    print("🔤 Génération des embeddings en batch...")
    embeddings = []
    total_batches = ceil(len(documents) / BATCH_SIZE)

    for i, doc_batch in enumerate(batch_list(documents, BATCH_SIZE)):
        print(f"→ Batch {i+1}/{total_batches} ({len(doc_batch)} docs)")
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=doc_batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            if len(batch_embeddings) != len(doc_batch):
                print(
                    f"⚠️ Mismatch embeddings: {len(batch_embeddings)} vs {len(doc_batch)} — complétion avec vecteurs nuls")
                missing = len(doc_batch) - len(batch_embeddings)
                batch_embeddings.extend([[0.0] * 1536] * missing)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"❌ Erreur batch {i+1} : {e}")
            embeddings.extend([[0.0] * 1536 for _ in doc_batch])
        sleep(0.5)  # anti-rate-limit

    print("💾 Sauvegarde des embeddings dans le cache...")
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

# === INDEXATION DANS CHROMA ===

print("📦 Création de l’index vectoriel (ChromaDB)...")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = chroma_client.get_or_create_collection(name="bofip_chunks")

count = len(collection.get()['ids'])
print(f"📊 Collection existante : {count} documents")

if count == 0:
    print("📥 Ajout des documents dans l’index...")
    add_to_chroma_in_batches(collection, documents, embeddings, metadatas, ids, batch_size=5000)
    print("✅ Indexation ajoutée.")
else:
    print("✅ Index déjà existant, rien ajouté.")

print("✅ Indexation terminée et persistée dans :", CHROMA_PATH)

print(f"\n📚 Total de documents dans la base : {collection.count()}")