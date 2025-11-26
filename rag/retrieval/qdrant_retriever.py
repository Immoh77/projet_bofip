# -*- coding: utf-8 -*-
"""
QdrantRetriever — Version finale avec filtrage par métadonnées
--------------------------------------------------------------
Pipeline :
1. Clarification de la question principale
2. Décomposition en sous-questions logiques
3. Recherche hybride (dense + sparse) dans Qdrant (+ filtres)
4. Re-ranking par sous-question
5. Fusion pondérée des résultats
6. Récupération des big chunks associés
7. Logging propre et chronologique
"""

import os
import json
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models as qm
from openai import OpenAI
from sklearn.feature_extraction.text import HashingVectorizer
from rag.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    OPENAI_CHAT_MODEL,
    BIG_CHUNKS_JSON_PATH,
    TOP_K_SUBQUESTION,
    TOP_K_FINAL,
    MAX_SUBQUERIES,
    PREFETCH_K,
    LOG_DIR,
    PROMPT_CLARIFY_QUESTION,
    PROMPT_SUBQUESTIONS,
    PROMPT_RERANK_LOCAL
)

load_dotenv()

# === CONFIG LOGGING ===
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"retriever_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Désactivation des logs HTTP parasites ===
for noisy_logger in ["httpx", "urllib3", "openai"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# === Initialisation clients ===
_openai = OpenAI(api_key=OPENAI_API_KEY)
_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
_hv = HashingVectorizer(n_features=2**16, alternate_sign=False, norm=None)


# ==========================================================
# Classe principale QdrantRetriever
# ==========================================================

class QdrantRetriever:
    def __init__(self):
        self.client = _qdrant
        self.openai = _openai
        self.vectorizer = _hv
        self.collection = QDRANT_COLLECTION
        self.embed_model = OPENAI_EMBED_MODEL
        self.chat_model = OPENAI_CHAT_MODEL

    # ======================================================
    # 🔸 Construction dynamique du filtre Qdrant
    # ======================================================
    def _build_qdrant_filter(self, filters: dict | None):
        """
        Construit un filtre Qdrant à partir d'un dictionnaire simple.
        Exemple :
            {"base": ["fiscal"], "division": ["AUT", "TVA"]}
        """
        if not filters:
            return None

        must_conditions = []
        for key, values in filters.items():
            if not values:
                continue
            if isinstance(values, list):
                if len(values) == 1:
                    must_conditions.append(qm.FieldCondition(
                        key=key,
                        match=qm.MatchValue(value=values[0])
                    ))
                else:
                    must_conditions.append(qm.FieldCondition(
                        key=key,
                        match=qm.MatchAny(any=values)
                    ))
            else:
                must_conditions.append(qm.FieldCondition(
                    key=key,
                    match=qm.MatchValue(value=values)
                ))

        return qm.Filter(must=must_conditions) if must_conditions else None

    # ======================================================
    # 🔸 Étape 1 : Clarification de la question principale
    # ======================================================
    def clarify_question(self, question: str) -> str:
        logger.info("Étape 1 — Clarification de la question...")
        prompt = PROMPT_CLARIFY_QUESTION.format(question=question)

        resp = self.openai.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        clarified = resp.choices[0].message.content.strip()
        logger.info(f"✅ Question clarifiée : {clarified}")
        return clarified

    # ======================================================
    # 🔸 Étape 2 : Génération des sous-questions
    # ======================================================
    def generate_subquestions(self, clarified_question: str) -> List[str]:
        logger.info("Étape 2 — Génération des sous-questions...")
        prompt = PROMPT_SUBQUESTIONS.format(question=clarified_question)
        resp = self.openai.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        lines = [l.strip("-• \t") for l in resp.choices[0].message.content.splitlines()]
        subqs = [l for l in lines if l][:MAX_SUBQUERIES] or [clarified_question]
        logger.info(f"✅ {len(subqs)} sous-questions générées.")
        return subqs

    # ======================================================
    # 🔸 Étape 3 : Embeddings
    # ======================================================
    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        return [d.embedding for d in self.openai.embeddings.create(model=self.embed_model, input=texts).data]

    def embed_sparse(self, texts: List[str]) -> List[qm.SparseVector]:
        X = self.vectorizer.transform(texts)
        vectors = []
        for i in range(X.shape[0]):
            idx = X[i].indices.tolist()
            val = X[i].data.astype(np.float32).tolist()
            if not idx:
                idx, val = [0], [1e-9]
            vectors.append(qm.SparseVector(indices=idx, values=val))
        return vectors

    # ======================================================
    # 🔸 Étape 4 : Recherche hybride (avec filtres)
    # ======================================================
    def query_hybrid(
        self,
        dense_vec,
        sparse_vec,
        limit=TOP_K_SUBQUESTION,
        prefetch_k=PREFETCH_K,
        filters: dict | None = None
    ):
        qdrant_filter = self._build_qdrant_filter(filters)
        if qdrant_filter:
            logger.info(f"🎛️ Application du filtre Qdrant : {json.dumps(filters, ensure_ascii=False)}")

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(query=sparse_vec, using="sparse", limit=prefetch_k),
                qm.Prefetch(query=dense_vec, using="dense", limit=prefetch_k),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
            query_filter=qdrant_filter,
        )

        out = []
        logger.info(f"🔍 {len(results.points)} small chunks récupérés (hybride)")
        for i, p in enumerate(results.points, start=1):
            payload = p.payload or {}
            payload["score_hybrid"] = float(p.score)
            text_preview = (payload.get("text") or payload.get("contenu") or "").strip().replace("\n", " ")
            logger.info(f"   {i:02d}. score={p.score:.3f} | chunk_id={payload.get('chunk_id')} | extrait : {text_preview[:150]}...")
            if "chunk_id" not in payload:
                payload["chunk_id"] = payload.get("metadata", {}).get("chunk_id")
            out.append(payload)
        return out

    # ======================================================
    # 🔸 Étape 5 : Re-ranking des résultats
    # ======================================================
    def rerank_results(self, chunks: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        logger.info("⚖️ Re-ranking des small chunks...")
        reranked = []
        for i, ch in enumerate(chunks, start=1):
            text = ch.get("text") or ch.get("contenu", "")
            prompt = PROMPT_RERANK_LOCAL.format(question=question, text=text[:1000])
            resp = self.openai.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            note = resp.choices[0].message.content.strip()
            try:
                ch["rerank_score"] = float(note)
            except ValueError:
                ch["rerank_score"] = 0.0
            reranked.append(ch)
            logger.info(f"   {i:02d}. chunk_id={ch.get('chunk_id')} | rerank_score={ch['rerank_score']:.2f}")

        reranked = [ch for ch in reranked if ch.get("rerank_score", 0.0) > 1.0]
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        logger.info(f"✅ {len(reranked)} chunks conservés après filtrage (note > 1).")
        return reranked

    # ======================================================
    # 🔸 Étape 6 : Fusion multi-sous-questions
    # ======================================================
    def fuse_reranked_results(self, all_reranked: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        pooled = {}
        meta_ref = {}
        for sq, results in all_reranked.items():
            for r in results:
                cid = r.get("chunk_id")
                score = r.get("rerank_score", 0.0)
                pooled.setdefault(cid, []).append(score)
                if cid not in meta_ref:
                    meta_ref[cid] = {
                        "metadata": r.get("metadata", {}),
                        "parent_chunk_id": r.get("parent_chunk_id"),
                        "text": r.get("text"),
                    }
        fused = []
        for cid, scores in pooled.items():
            item = {"chunk_id": cid, "score_final": float(np.mean(scores))}
            if cid in meta_ref:
                item.update(meta_ref[cid])
            fused.append(item)
        fused.sort(key=lambda x: x["score_final"], reverse=True)
        return fused

    # ======================================================
    # 🔸 Étape 7 : Récupération des big chunks
    # ======================================================
    def get_big_chunks(self, fused_results: list) -> list:
        associated_big_chunks = []
        try:
            if not os.path.exists(BIG_CHUNKS_JSON_PATH):
                logger.warning(f"⚠️ Fichier des big chunks introuvable : {BIG_CHUNKS_JSON_PATH}")
                return []
            with open(BIG_CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                big_chunks_data = json.load(f)
            big_chunks_map = {b["chunk_id"]: b for b in big_chunks_data}
            logger.info(f"🧩 Chargé {len(big_chunks_map)} big chunks depuis {BIG_CHUNKS_JSON_PATH}")
            logger.info("🔍 Association des small chunks via parent_chunk_id...")
            parent_ids_seen = set()
            for item in fused_results:
                meta = item.get("metadata", {})
                parent_id = meta.get("parent_chunk_id") or item.get("parent_chunk_id")
                if parent_id:
                    parent_ids_seen.add(parent_id)
                    if parent_id in big_chunks_map:
                        item["big_chunk"] = big_chunks_map[parent_id]
                        associated_big_chunks.append(big_chunks_map[parent_id])
                    else:
                        item["big_chunk"] = None
                else:
                    item["big_chunk"] = None
            unique_big_chunks = {b["chunk_id"]: b for b in associated_big_chunks}.values()
            logger.info(f"✅ Big chunks associés : {len(unique_big_chunks)}")
        except Exception as e:
            logger.exception(f"Erreur lors de la récupération des big chunks : {e}")
            unique_big_chunks = []
        return list(unique_big_chunks)

    # ======================================================
    # 🔸 Étape 8 : Pipeline complet avec filtres
    # ======================================================
    def retrieve_with_subquery_rerank(self, question: str, filters: dict | None = None) -> Dict[str, Any]:
        logger.info(f"🧠 Question d'origine : {question}")
        if filters:
            logger.info(f"🎛️ Filtres actifs : {json.dumps(filters, ensure_ascii=False)}")

        clarified = self.clarify_question(question)
        subqs = self.generate_subquestions(clarified)
        all_reranked = {}

        for sq in subqs:
            logger.info(f"🔹 Sous-question : {sq}")
            dense_vec = self.embed_dense([sq])[0]
            sparse_vec = self.embed_sparse([sq])[0]
            hits = self.query_hybrid(dense_vec, sparse_vec, limit=TOP_K_SUBQUESTION, filters=filters)
            reranked = self.rerank_results(hits, question=sq)
            all_reranked[sq] = reranked

        fused_all = self.fuse_reranked_results(all_reranked)
        fused = fused_all[:TOP_K_FINAL]
        big_chunks = self.get_big_chunks(fused)
        return {
            "question_originale": question,
            "question_clarifiee": clarified,
            "sous_questions": subqs,
            "reranked_par_sous_question": all_reranked,
            "fusion_finale": fused,
            "big_chunks_associes": big_chunks,
        }


# === Exécution manuelle ===
if __name__ == "__main__":
    retriever = QdrantRetriever()

    # 🧠 Exemple d'entrée utilisateur
    question = input("🧠 Question : ")
    filters = {"base": ["fiscal"]}  # 🔍 Exemple de filtre test

    # 🔧 Appel du pipeline complet
    res = retriever.retrieve_with_subquery_rerank(question, filters=filters)

    logger.info("\n=== 🧹 Question clarifiée ===")
    logger.info(res["question_clarifiee"])

    logger.info("\n=== 🔍 Sous-questions générées ===")
    for s in res["sous_questions"]:
        logger.info(f" - {s}")

    logger.info("\n=== 🧩 Fusion finale des résultats ===")
    for i, f in enumerate(res["fusion_finale"][:10], start=1):
        logger.info(f"{i}. chunk_id={f['chunk_id']} | score_final={f['score_final']:.2f}")

    logger.info("\n=== 🧱 Big Chunks associés ===")
    for i, bc in enumerate(res.get("big_chunks_associes", []), start=1):
        text = bc.get('text') or bc.get('contenu') or ''
        logger.info(f"{i}. chunk_id={bc.get('chunk_id')} | extrait : {text[:200].strip()}...")