import os
import subprocess
import time

# --- ⚙️ 1. Configuration des chemins et des ports ---
QDRANT_NAME = "qdrant"
QDRANT_PORT = 6333

# 💾 Chemin de stockage persistant sur ton disque local
# (tu peux adapter selon ton organisation)
STORAGE_PATH = r"C:\Users\GAUTH\OneDrive\Documents\Code Python\Projets\qdrant_storage"

# --- 📁 2. Création du dossier local si besoin ---
os.makedirs(STORAGE_PATH, exist_ok=True)
print(f"📂 Dossier de stockage vérifié : {STORAGE_PATH}")

# --- 🐳 3. Vérifie si Qdrant tourne déjà ---
print("🔍 Vérification du conteneur Qdrant existant...")
containers = subprocess.getoutput("docker ps -a --format '{{.Names}}'")

if QDRANT_NAME in containers:
    print(f"⚠️  Un conteneur '{QDRANT_NAME}' existe déjà. On le redémarre...")
    subprocess.run(["docker", "start", QDRANT_NAME])
else:
    print("🚀 Lancement d’un nouveau conteneur Qdrant avec stockage persistant...")
    subprocess.run([
        "docker", "run", "-d",
        "--name", QDRANT_NAME,
        "-p", f"{QDRANT_PORT}:6333",
        "-v", f"{STORAGE_PATH}:/qdrant/storage",
        "qdrant/qdrant:latest"
    ])

# --- ⏳ 4. Attendre un peu que Qdrant démarre ---
print("⏳ Démarrage du service Qdrant...")
time.sleep(5)

# --- 🔎 5. Vérification de l’accès Qdrant ---
from qdrant_client import QdrantClient
try:
    client = QdrantClient(url=f"http://localhost:{QDRANT_PORT}")
    info = client.get_collections()
    print("✅ Qdrant est en ligne ! Collections existantes :")
    for c in info.collections:
        print(f"   - {c.name}")
except Exception as e:
    print("❌ Erreur lors de la connexion à Qdrant :", e)
