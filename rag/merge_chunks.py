import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# === Chemins vers les SMALL chunks ===
bofip_small_path = BASE_DIR / "data" / "processed" / "bofip_small_chunks.json"
code_ass_small_path = BASE_DIR / "data" / "processed" / "code_assurances_small_chunks.json"
all_small_path = BASE_DIR / "data" / "processed" / "all_small_chunks.json"

# === Chemins vers les BIG chunks ===
bofip_big_path = BASE_DIR / "data" / "processed" / "bofip_chunks_bs.json"
code_ass_big_path = BASE_DIR / "data" / "processed" / "code_assurances_chunks.json"
all_big_path = BASE_DIR / "data" / "processed" / "all_big_chunks.json"

# === Chargement des fichiers SMALL ===
with open(bofip_small_path, "r", encoding="utf-8") as f_bofip:
    bofip_small = json.load(f_bofip)

with open(code_ass_small_path, "r", encoding="utf-8") as f_code:
    code_small = json.load(f_code)

# === Fusion SMALL chunks ===
merged_small = bofip_small + code_small

with open(all_small_path, "w", encoding="utf-8") as f_out:
    json.dump(merged_small, f_out, indent=2, ensure_ascii=False)

print(f"✅ [SMALL] Fichier fusionné écrit dans : {all_small_path}")
print(f"📄 Total de small chunks : {len(merged_small)}")

# === Chargement des fichiers BIG ===
with open(bofip_big_path, "r", encoding="utf-8") as f_bofip_big:
    bofip_big = json.load(f_bofip_big)

with open(code_ass_big_path, "r", encoding="utf-8") as f_code_big:
    code_big = json.load(f_code_big)

# === Fusion BIG chunks avec contrôle des doublons par chunk_id ===
seen_ids = set()
merged_big = []
for chunk in bofip_big + code_big:
    cid = chunk.get("chunk_id")
    if cid and cid not in seen_ids:
        merged_big.append(chunk)
        seen_ids.add(cid)

with open(all_big_path, "w", encoding="utf-8") as f_out:
    json.dump(merged_big, f_out, indent=2, ensure_ascii=False)

print(f"\n✅ [BIG] Fichier fusionné écrit dans : {all_big_path}")
print(f"📄 Total de big chunks : {len(merged_big)}")

# === Affichage rapide des sources incluses (vérification) ===
sources = set(c.get("metadata", {}).get("source", "N/A") for c in merged_big)
print(f"\n📚 Sources présentes dans les big chunks : {sources}")