# rag/retrieval/qdrant_backend.py
from __future__ import annotations
from typing import Dict, Any, List, Iterable, Optional

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer  # <- dense local e5

# === Config Qdrant (local docker) ===
QDRANT_URL = "http://localhost:6333"
COLLECTION = "bofip_code_assurances"

# === Espaces vectoriels ===
DENSE_NAME = "dense"
SPARSE_NAME = "sparse"

# === Modèles ===
# Dense : on EMBED localement avec sentence-transformers (multilingue FR OK)
DENSE_ST_MODEL = "intfloat/multilingual-e5-base"
DENSE_DIM = 768  # e5-base = 768

# Sparse : SPLADE géré par Qdrant FastEmbed (OK pour le signal lexical)
SPARSE_FASTEMBED_MODEL = "prithivida/Splade_PP_en_v1"

_client: Optional[QdrantClient] = None
_st_model: Optional[SentenceTransformer] = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client

def get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(DENSE_ST_MODEL)
    return _st_model

def ensure_collection():
    """
    Crée la collection avec 2 espaces :
      - dense (cosine, 768 dims)
      - sparse (SPLADE)
    """
    client = get_client()

    # Existe déjà ?
    try:
        cols = client.get_collections().collections
        if any(c.name == COLLECTION for c in cols):
            return
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE_NAME: models.VectorParams(
                size=DENSE_DIM, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_NAME: models.SparseVectorParams()
        },
    )

def _build_filter(filters: Dict[str, list] | None) -> Optional[models.Filter]:
    if not filters:
        return None
    must = []
    for key in ("base", "source", "serie", "division"):
        vals = filters.get(key)
        if vals:
            must.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=vals))
            )
    return models.Filter(must=must) if must else None

def upsert_chunks(chunks: Iterable[Dict[str, Any]], batch: int = 256):
    """
    Ingestion des *small chunks* :
      - IDs Qdrant : entiers séquentiels (Qdrant exige int ou UUID strict)
      - DENSE : vecteur (list[float]) généré localement avec SentenceTransformer
      - SPARSE : SPLADE via FastEmbed côté serveur
    """
    ensure_collection()
    client = get_client()
    st = get_st_model()

    buf_dense: List[List[float]] = []
    buf_payloads: List[dict] = []
    buf_ids: List[int] = []
    next_id = 1  # ← IDs entiers séquentiels

    def _flush():
        nonlocal buf_dense, buf_payloads, buf_ids
        if not buf_dense:
            return
        client.upload_collection(
            collection_name=COLLECTION,
            vectors=[
                {
                    DENSE_NAME: d,  # dense vector list[float]
                    SPARSE_NAME: models.Document(
                        text=p["text"],
                        model=SPARSE_FASTEMBED_MODEL
                    ),
                }
                for d, p in zip(buf_dense, buf_payloads)
            ],
            payload=buf_payloads,
            ids=buf_ids,       # ← entiers
            parallel=1,        # ← si tu vois encore des warnings, garde 1
        )
        buf_dense, buf_payloads, buf_ids = [], [], []

    texts: List[str] = []
    metas: List[dict] = []
    ids_tmp: List[int] = []

    for ch in chunks:
        txt = ch.get("contenu") or ""
        md = ch.get("metadata", {}) or {}

        # on garde l'id d'origine EN PAYLOAD, mais l'ID Qdrant = entier
        payload = {
            "text": txt,
            "base": md.get("base"),
            "source": md.get("source"),
            "serie": md.get("serie"),
            "division": md.get("division"),
            "chunk_id": md.get("chunk_id") or md.get("id"),  # id d'origine préservé ici
            "parent_chunk_id": md.get("parent_chunk_id"),
            "titre_document": md.get("titre_document"),
            "titre_bloc": md.get("titre_bloc"),
            "permalien": md.get("permalien") or md.get("url") or md.get("lien"),
        }

        texts.append(txt)
        metas.append(payload)
        ids_tmp.append(next_id)
        next_id += 1

        if len(texts) >= batch:
            vecs = st.encode(texts, normalize_embeddings=True).tolist()
            buf_dense.extend(vecs)
            buf_payloads.extend(metas)
            buf_ids.extend(ids_tmp)
            texts, metas, ids_tmp = [], [], []
            _flush()

    if texts:
        vecs = st.encode(texts, normalize_embeddings=True).tolist()
        buf_dense.extend(vecs)
        buf_payloads.extend(metas)
        buf_ids.extend(ids_tmp)
        _flush()


    def _flush():
        nonlocal buf_dense, buf_sparse_docs, buf_payloads, buf_ids
        if not buf_dense:
            return
        # On upload les deux espaces en parallèle :
        client.upload_collection(
            collection_name=COLLECTION,
            vectors=[
                {
                    DENSE_NAME: d,                                # vecteur dense (list[float])
                    SPARSE_NAME: models.Document(                 # doc sparse à embed par Qdrant
                        text=p["text"],
                        model=SPARSE_FASTEMBED_MODEL
                    ),
                }
                for d, p in zip(buf_dense, buf_payloads)
            ],
            payload=buf_payloads,
            ids=buf_ids,
            parallel=2,
        )
        buf_dense, buf_sparse_docs, buf_payloads, buf_ids = [], [], [], []

    texts: List[str] = []
    metas: List[dict] = []
    ids: List[Any] = []

    for ch in chunks:
        txt = ch.get("contenu") or ""
        md = ch.get("metadata", {}) or {}
        cid = md.get("chunk_id") or md.get("id") or (count + 1)

        payload = {
            "text": txt,
            "base": md.get("base"),
            "source": md.get("source"),
            "serie": md.get("serie"),
            "division": md.get("division"),
            "chunk_id": cid,
            "parent_chunk_id": md.get("parent_chunk_id"),
            "titre_document": md.get("titre_document"),
            "titre_bloc": md.get("titre_bloc"),
            "permalien": md.get("permalien") or md.get("url") or md.get("lien"),
        }

        texts.append(txt)
        metas.append(payload)
        ids.append(cid)
        count += 1

        # batch encode dense
        if len(texts) >= batch:
            vecs = st.encode(texts, normalize_embeddings=True).tolist()
            buf_dense.extend(vecs)
            buf_payloads.extend(metas)
            buf_ids.extend(ids)
            texts, metas, ids = [], [], []
            _flush()

    if texts:
        vecs = st.encode(texts, normalize_embeddings=True).tolist()
        buf_dense.extend(vecs)
        buf_payloads.extend(metas)
        buf_ids.extend(ids)
        _flush()

def qdrant_hybrid_search(
    question: str,
    top_k: int = 20,
    filters: Dict[str, list] | None = None,
    min_similarity: float | None = None,
) -> Dict[str, Any]:
    """
    Recherche hybride (sparse SPLADE + dense e5) via Prefetch + Fusion RRF.
    Renvoie le même format que ta recherche actuelle: {"fusion":[...]}.
    """
    ensure_collection()
    client = get_client()

    where = _build_filter(filters)

    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            # Sparse (lexical)
            models.Prefetch(
                query=models.Document(text=question, model=SPARSE_FASTEMBED_MODEL),
                using=SPARSE_NAME,
                limit=top_k,
            ),
            # Dense (sémantique) : Qdrant accepte aussi Document(text, model) côté requête,
            # mais comme nos vecteurs d'index sont e5-base, utiliser le même modèle est cohérent.
            models.Prefetch(
                query=models.Document(text=question, model=DENSE_ST_MODEL),
                using=DENSE_NAME,
                limit=top_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),  # ou DBSF
        limit=top_k,
        query_filter=where,
        with_payload=True,
    )

    fusion = []
    for p in (res.points or []):
        md = p.payload or {}
        score = float(p.score) if p.score is not None else 0.0
        fusion.append({
            "chunk_id": md.get("chunk_id") or md.get("parent_chunk_id"),
            "parent_chunk_id": md.get("parent_chunk_id"),
            "contenu": md.get("text", ""),
            "metadata": md,
            "score_hybrid": score,
            "similarity_raw": score,
            "provenance": "qdrant_hybrid",
        })

    if min_similarity is not None:
        fusion = [x for x in fusion if (x.get("similarity_raw") or 0.0) >= float(min_similarity)]

    return {"fusion": fusion}

