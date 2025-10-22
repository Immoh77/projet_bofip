# -*- coding: utf-8 -*-
"""
QdrantRetriever — Version Option A (Re-ranking par sous-question)

Pipeline :
1. Clarification de la question principale (développe sigles, abréviations)
2. Décomposition en sous-questions logiques
3. Recherche hybride (dense + sparse) dans Qdrant
4. Re-ranking par sous-question (LLM)
5. Fusion pondérée des résultats
6. (Optionnel) récupération des big chunks associés
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models as qm
from openai import OpenAI
from sklearn.feature_extraction.text import HashingVectorizer

# --- Chargement des variables d'environnement ---
load_dotenv()

# === CONFIG ===
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bofip_hybrid")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

BIG_CHUNKS_JSON_PATH = os.getenv("BIG_CHUNKS_JSON_PATH", "data/processed/bofip_chunks_bs.json")

# Nombre de résultats par sous-question
TOP_K_SUBQUESTION = 5

# Nombre total de chunks après fusion finale
TOP_K_FINAL = 15

# Nombre de sous-questions max
MAX_SUBQUERIES = 4

# Nombre d’éléments préchargés par Qdrant avant fusion dense/sparse
PREFETCH_K = 10

# --- Initialisation clients ---
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

    # --- Étape 1 : Clarification de la question principale ---
    def clarify_question(self, question: str) -> str:
        prompt = (
            "Clarifie et reformule la question suivante pour qu’elle soit parfaitement compréhensible "
            "par un moteur de recherche ou un modèle de langage. "
            "Développe toutes les abréviations et acronymes, corrige les formulations ambiguës "
            "et ajoute le contexte fiscal ou juridique si nécessaire, sans changer le sens.\n\n"
            f"Question : {question}\n\nQuestion clarifiée :"
        )
        resp = self.openai.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        clarified = resp.choices[0].message.content.strip()
        return clarified

    # --- Étape 2 : Génération des sous-questions logiques ---
    def generate_subquestions(self, clarified_question: str) -> List[str]:
        prompt = (
            "Divise la question suivante en plusieurs sous-questions logiques permettant d’y répondre étape par étape. "
            "Chaque sous-question doit correspondre à un aspect clé du problème (définition, conditions, calcul, etc.). "
            "Ne reformule pas simplement : découpe le raisonnement.\n\n"
            f"Question : {clarified_question}\n\nSous-questions :"
        )
        resp = self.openai.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        lines = [l.strip("-• \t") for l in resp.choices[0].message.content.splitlines()]
        return [l for l in lines if l][:MAX_SUBQUERIES] or [clarified_question]

    # --- Étape 3 : Embeddings dense et sparse ---
    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        resp = self.openai.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]

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

    # --- Étape 4 : Recherche hybride Qdrant ---
    def query_hybrid(self, dense_vec, sparse_vec, limit=None, prefetch_k=None):
        limit = limit or TOP_K_SUBQUESTION
        prefetch_k = prefetch_k or PREFETCH_K
        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(query=sparse_vec, using="sparse", limit=prefetch_k),
                qm.Prefetch(query=dense_vec, using="dense", limit=prefetch_k),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        out = []
        for p in results.points:
            pl = p.payload or {}
            pl["score_hybrid"] = float(p.score)
            pl["chunk_id"] = pl.get("chunk_id") or pl.get("metadata", {}).get("chunk_id")
            out.append(pl)
        return out

    # --- Étape 5 : Re-ranking local (par sous-question) ---
    def rerank_results(self, chunks: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        reranked = []
        for ch in chunks:
            text = ch.get("text") or ch.get("contenu", "")
            prompt = (
                f"Question : {question}\n\nTexte : {text[:1000]}\n\n"
                "Note la pertinence de ce texte pour répondre à la question sur une échelle de 0 (inutile) à 5 (très pertinent). "
                "Répond uniquement par un nombre."
            )
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
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked

    # --- Étape 6 : Fusion pondérée multi-sous-questions ---
    def fuse_reranked_results(self, all_reranked: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        pooled = {}
        for sq, results in all_reranked.items():
            for r in results:
                cid = r["chunk_id"]
                pooled.setdefault(cid, []).append(r["rerank_score"])
        fused = []
        for cid, scores in pooled.items():
            fused.append({"chunk_id": cid, "score_final": float(np.mean(scores))})
        fused.sort(key=lambda x: x["score_final"], reverse=True)
        return fused

    # --- Étape 7 : Pipeline complet ---
    def retrieve_with_subquery_rerank(self, question: str) -> Dict[str, Any]:
        clarified = self.clarify_question(question)
        subqs = self.generate_subquestions(clarified)
        all_reranked = {}

        for sq in subqs:
            dense_vec = self.embed_dense([sq])[0]
            sparse_vec = self.embed_sparse([sq])[0]
            hits = self.query_hybrid(dense_vec, sparse_vec, limit=TOP_K_SUBQUESTION)
            reranked = self.rerank_results(hits, question=sq)
            all_reranked[sq] = reranked

        # --- Fusion finale (correction du bug) ---
        fused_all = [
            item
            for sublist in all_reranked.values()
            for item in sublist
        ]
        fused_all = sorted(fused_all, key=lambda x: x.get("rerank_score", 0), reverse=True)
        fused = fused_all[:TOP_K_FINAL]  # limite à 15 chunks maximum

        return {
            "question_originale": question,
            "question_clarifiee": clarified,
            "sous_questions": subqs,
            "reranked_par_sous_question": all_reranked,
            "fusion_finale": fused,
        }

        # --- Étape 8 : Remontée small → big chunks (depuis JSON local) ---

    def get_big_chunks_from_small(self, small_chunks: list):
        """
        Récupère les big chunks uniques associés aux small chunks.
        Si les big chunks ne sont pas dans Qdrant, ils sont chargés depuis le JSON local.
        """
        seen = set()
        big_chunks = []

        # Charger le fichier JSON des big chunks
        try:
            with open(BIG_CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                all_big = json.load(f)
                all_big_by_id = {b["chunk_id"]: b for b in all_big}
            print(f"📂 Fichier des big chunks chargé ({len(all_big_by_id)} éléments)")
        except Exception as e:
            print(f"⚠️ Impossible de charger le fichier des big chunks : {e}")
            all_big_by_id = {}

        for ch in small_chunks:
            meta = ch.get("metadata", {}) or {}
            parent_id = meta.get("parent_chunk_id") or ch.get("parent_chunk_id")

            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)

            # Cas 1 : big chunk déjà dans le payload (optimisation)
            if meta.get("big_chunk"):
                big_chunks.append(meta["big_chunk"])
                continue

            # Cas 2 : big chunk présent dans le JSON local
            big_local = all_big_by_id.get(parent_id)
            if big_local:
                big_chunks.append(big_local)
                continue

            # Cas 3 : fallback Qdrant (rare)
            try:
                results = self.client.query_points(
                    collection_name=self.collection,
                    query=parent_id,
                    with_payload=True,
                    limit=1,
                )
                if results.points:
                    pl = results.points[0].payload or {}
                    pl["chunk_id"] = pl.get("chunk_id") or pl.get("metadata", {}).get("chunk_id")
                    big_chunks.append(pl)
            except Exception as e:
                print(f"⚠️ Impossible de récupérer le big chunk {parent_id}: {e}")

        print(f"✅ {len(big_chunks)} big chunks récupérés (via JSON ou Qdrant).")
        return big_chunks