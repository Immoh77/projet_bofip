import os
import json
import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# import robuste pour la sparse EF selon les versions de Chroma
try:
    from chromadb.utils.embedding_functions import DefaultSparseEmbeddingFunction
except Exception:
    try:
        # ancien emplacement possible
        from chromadb.utils.sparse_embedding_functions import DefaultSparseEmbeddingFunction
    except Exception:
        DefaultSparseEmbeddingFunction = None  # on gèrera un message clair au runtime

from dotenv import load_dotenv
from openai import OpenAI
import json as _json
from collections import defaultdict
from .decompose import generate_subquestions
from rag.config import (
    OPENAI_CHAT_MODEL,
    PROMPT_REWRITE_QUERY,
    PROMPT_RERANK,
    SMALL_CHUNKS_JSON_PATH,
    BIG_CHUNKS_JSON_PATH,
    DOCUMENT_SOURCES,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    SENTENCE_TRANSFORMERS_MODEL
)

# === CHARGEMENT CLE API ===

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Singleton Chroma hybride (dense + sparse) ---
_chroma_client = None
_chroma_coll = None
_dense_fn = None
_sparse_fn = None

def _get_chroma_collection():
    """
    Récupère une collection Chroma avec embeddings denses + sparse (hybride natif).
    """
    global _chroma_client, _chroma_coll, _dense_fn, _sparse_fn
    if _chroma_coll is not None:
        return _chroma_coll

    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _dense_fn = SentenceTransformerEmbeddingFunction(model_name=SENTENCE_TRANSFORMERS_MODEL)
    if DefaultSparseEmbeddingFunction is None:
        raise ImportError(
            "Votre version de Chroma ne fournit pas DefaultSparseEmbeddingFunction. "
            "Mettez à jour avec: pip install -U 'chromadb>=0.5.7' scikit-learn"
        )
    _sparse_fn = DefaultSparseEmbeddingFunction()

    _chroma_coll = _chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_dense_fn,
        sparse_embedding_function=_sparse_fn,
    )
    return _chroma_coll

# === CHARGEMENT DES BIG ET SMALLS CHUNKS ===

with open(BIG_CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    BIG_CHUNKS = json.load(f)

with open(SMALL_CHUNKS_JSON_PATH, "r", encoding="utf-8") as f_bofip:
    bofip_chunks = json.load(f_bofip)

# Chargement chunks Code des assurances
code_path = DOCUMENT_SOURCES["code_assurances"]["OUTPUT_SMALL_CHUNKS"]
with open(code_path, "r", encoding="utf-8") as f_code:
    code_chunks = json.load(f_code)

# Fusion
ALL_CHUNKS = bofip_chunks + code_chunks

# === REFORMULATION DE LA QUESTION ===

def rewrite_query(original_question):
    print(f"📝 Question originale : {original_question}")

    # Interpolation ici
    prompt = PROMPT_REWRITE_QUERY.format(question=original_question)

    try:
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()
        print("🔁 Reformulation :", content)
        return [content] if content else [original_question]

    except Exception as e:
        print(f"❌ Erreur lors de la reformulation : {e}")
        return [original_question]


# === TOKENIZATION DU TEXTE ===

def tokenize(text):
    if not isinstance(text, str):
        print(f"⚠️ Mauvais format de texte dans tokenize : {type(text)} → {text}")
        text = " ".join(text) if isinstance(text, list) else str(text)
    return re.findall(r"\b\w+\b", text.lower())

# === FILTRES METADATAS ===

def match_metadata(metadata, filters):
    return all(
        not filters[k] or metadata.get(k) in filters[k]
        for k in filters
    )


# === COMBINE / FUSION / SUPPRESSION DOUBLONS ===

def normalize_scores(results, key="distance", reverse=False):
    # ✅ si aucun résultat, on renvoie tel quel
    if not results:
        return []

    values = [r.get(key) for r in results if r.get(key) is not None]
    if not values:  # sécurité si la clé manque
        for r in results:
            r["score_norm"] = 0.0
        return results

    min_val, max_val = min(values), max(values)
    if max_val - min_val < 1e-6:
        for r in results:
            r["score_norm"] = 1.0
        return results

    for r in results:
        raw = r.get(key, min_val)
        norm = (raw - min_val) / (max_val - min_val + 1e-6)
        r["score_norm"] = 1 - norm if reverse else norm
    return results


def hybrid_search(question: str, top_k: int = 20, filters=None, lexical_weight: float = 0.5, min_similarity: float | None = None):
    coll = _get_chroma_collection()

    # Traduction simple des filtres en 'where'
    where = {}
    if isinstance(filters, dict):
        if filters.get("base"):     where["base"] = {"$in": filters["base"]}
        if filters.get("source"):   where["source"] = {"$in": filters["source"]}
        if filters.get("serie"):    where["serie"] = {"$in": filters["serie"]}
        if filters.get("division"): where["division"] = {"$in": filters["division"]}
    if not where:
        where = None

    # Chroma hybride: dense (embedding_function) + sparse (lexical) pour la requête
    results = coll.query(
        query_texts=[question],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
        query_sparse_embedding=_sparse_fn(question),
    )

    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    fusion = []
    for i, doc in enumerate(docs):
        md = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None
        sim = None
        if dist is not None:
            try:
                sim = max(0.0, 1.0 - float(dist))  # distance -> similarité
            except Exception:
                sim = None

        fusion.append({
            "chunk_id": md.get("chunk_id") or md.get("id") or md.get("parent_chunk_id"),
            "contenu": doc,
            "metadata": md,
            "score_hybrid": sim if sim is not None else 0.0,
            "similarity_raw": sim,
            "provenance": "chroma_hybrid",
        })

    if min_similarity is not None:
        fusion = [x for x in fusion if (x.get("similarity_raw") or 0.0) >= float(min_similarity)]

    return {"fusion": fusion}

# === RERANKING AVEC GPT ===

def rerank_chunks_with_gpt(query, chunks, force=False):
    if not chunks:
        return []

    # 🔍 Analyser les scores hybrides
    scores = [c.get("score_hybrid", 0.0) for c in chunks[:5]]
    print(f"📚 Nombre de passages à reclasser : {len(chunks)}")

    numbered_passages = "\n".join([f"[{i + 1}] {chunk['contenu']}" for i, chunk in enumerate(chunks)])

    instruction = (
        "Tu es un assistant juridique. On te donne des extraits numérotés et une question.\n"
        "Rends UNIQUEMENT un JSON valide (pas d'explication) au format :\n"
        '[{"index": <numero_extrait_commençant_à_1>, "relevance": <score_0_a_1>}, ...]\n'
        "Inclue seulement les extraits pertinents. Le champ 'relevance' est un score de pertinence "
        "contextuelle par rapport à l'intention de la question (0 = hors sujet, 1 = totalement pertinent).\n\n"
        f"Extraits :\n{numbered_passages}\n\n"
        f"Question : {query}\n"
    )

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": PROMPT_RERANK},
            {"role": "user", "content": instruction}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    print("📥 JSON de reclassement :", content)

    try:
        parsed = _json.loads(content)
        # attendu: liste d'objets {"index": int>=1, "relevance": float in [0,1]}
        indices = []
        for item in parsed:
            idx = int(item.get("index", 0)) - 1
            rel = float(item.get("relevance", 0.0))
            if 0 <= idx < len(chunks) and 0.0 <= rel <= 1.0:
                indices.append((idx, rel))

        # trier par pertinence décroissante (au cas où)
        indices.sort(key=lambda x: x[1], reverse=True)

        selected_chunks = []
        for idx, rel in indices:
            c = chunks[idx]
            c["context_score"] = rel  # 🔎 score de pertinence contextuelle
            # s'assurer que similarity_score existe (sinon 0)
            c["similarity_score"] = float(c.get("similarity_score", c.get("score_hybrid", 0.0)))
            selected_chunks.append(c)

        print("\n🧾 Chunks sélectionnés par le LLM :")
        for c in selected_chunks:
            meta = c.get("metadata", {})
            print(
                f"✅ chunk_id={meta.get('chunk_id', 'N/A')} | context={c['context_score']:.2f} | sim={c.get('similarity_score', 0):.2f}")

        # exemple
        if selected_chunks:
            print("🔍 Exemple chunk sélectionné :", selected_chunks[0])

        return selected_chunks

    except Exception as e:
        print("⚠️ Erreur parsing JSON pertinence, fallback ordre simple:", e)
        # fallback: si pas de JSON, on garde l'ordre initial et on génère des scores dégressifs
        n = len(chunks)
        for i, c in enumerate(chunks):
            c["context_score"] = max(0.0, (n - i) / max(1.0, n))  # 1.0 → ~0.0
            c["similarity_score"] = float(c.get("similarity_score", c.get("score_hybrid", 0.0)))
        return chunks



# === RECONSTRUCTION DES BIG CHUNKS ===

def get_big_chunks_from_small(selected_small_chunks):
    parent_ids = {chunk.get("parent_chunk_id") for chunk in selected_small_chunks if chunk.get("parent_chunk_id")}
    return [chunk for chunk in BIG_CHUNKS if chunk.get("chunk_id") in parent_ids]

def aggregate_big_chunks_from_small(selected_small_chunks, top_k=None, w_sim=0.4, w_ctx=0.6):
    """
    Agrège des scores au niveau du big chunk (parent_chunk_id).
    - big_similarity = max des similarity_raw des smalls
    - big_context    = moyenne pondérée des context_score par similarity_raw
    - big_score      = w_ctx*big_context + w_sim*big_similarity
    - big_support    = nb de smalls retenus
    """
    if not selected_small_chunks:
        return []

    # Index des BIG_CHUNKS déjà chargés dans ce module
    big_index = {b.get("chunk_id"): b for b in BIG_CHUNKS}

    def safe_sim_raw(s):
        # priorité au champ explicite, sinon fallback
        if "similarity_raw" in s:
            try:
                return max(0.0, min(1.0, float(s["similarity_raw"])))
            except Exception:
                pass
        d = s.get("distance", None)
        try:
            return max(0.0, min(1.0, 1.0 - float(d))) if d is not None else float(s.get("score_norm", 0.0))
        except Exception:
            return 0.0

    groups = defaultdict(list)
    for s in selected_small_chunks:
        pid = s.get("parent_chunk_id") or s.get("metadata", {}).get("parent_chunk_id")
        if pid:
            groups[pid].append(s)

    scored_big = []
    for parent_id, smalls in groups.items():
        if not smalls:
            continue

        sims = [safe_sim_raw(s) for s in smalls]
        ctxs = [float(s.get("context_score", 0.0)) for s in smalls]

        sim_max = max(sims) if sims else 0.0
        w = sum(sims)

        # ⚠️ éviter division par zéro
        if w <= 1e-12:
            ctx_wavg = 0.0
        else:
            ctx_wavg = sum(c * s for c, s in zip(ctxs, sims)) / w

        support = len(smalls)
        big_score = (w_ctx * ctx_wavg) + (w_sim * sim_max)

        big = big_index.get(parent_id)
        if not big:
            continue

        big_copy = dict(big)
        big_copy["big_similarity"] = sim_max
        big_copy["big_context"] = ctx_wavg
        big_copy["big_support"] = support
        big_copy["big_score"] = big_score
        scored_big.append(big_copy)

    scored_big.sort(key=lambda x: -x["big_score"])
    return scored_big[:top_k] if (top_k and top_k > 0) else scored_big

def hybrid_search_multi(question: str, top_k: int = 20, filters=None,
                        lexical_weight: float = 0.5, min_similarity: float | None = None,
                        per_sub: int | None = None):
    subs = (generate_subquestions(question) or [question])[:6]
    if per_sub is None:
        per_sub = max(4, min(6, top_k // max(1, len(subs))))

    pool = {}
    for sq in subs:
        res = hybrid_search(sq, top_k=per_sub, filters=filters,
                            lexical_weight=lexical_weight, min_similarity=min_similarity) or {}
        for r in (res.get("fusion") or []):
            uid = r.get("chunk_id") or r.get("metadata", {}).get("chunk_id")
            if not uid:
                continue
            cur = pool.get(uid)
            if cur is None or (r.get("score_hybrid", 0) > cur.get("score_hybrid", 0)):
                pool[uid] = r

    fused = list(pool.values())
    fused.sort(key=lambda x: x.get("score_hybrid", 0), reverse=True)
    return {"fusion": fused[:top_k], "subquestions": subs}