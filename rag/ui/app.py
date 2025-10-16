# === bootstrapping sys.path pour trouver le package 'rag' ===
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# === fin bootstrapping ===
from join_key import make_sig
from eval_log import append_eval
from feedback_log import append_feedback
import base64
import time
import threading
import streamlit as st

from rag.config import TOP_K, FILTER_TREE, CHARTE_IA_PATH
from rag.retrieval.retriever import (
    rewrite_query,
    hybrid_search,
    rerank_chunks_with_gpt,
    get_big_chunks_from_small,
    aggregate_big_chunks_from_small,
    hybrid_search_multi
)
from rag.generation.answerer import generate_answer
from rag.utils.history import load_history, append_history
from rag.evaluation.judge import run_judge

# --- JUGE ASYNCHRONE ---
def _launch_judge_async(question: str, contexts: list, answer: str):
    """Lance le juge en arrière-plan et enregistre le résultat dans st.session_state."""
    def _task():
        try:
            verdict = run_judge(question, contexts, answer)
            st.session_state["last_judge"] = verdict
            # (option) log minimal d'évaluation
            try:
                append_eval({"ts": time.time(), "question": question, "scores": verdict.get("scores", {}), "flags": verdict.get("flags", {})})
            except Exception:
                pass
        finally:
            st.session_state["judge_running"] = False

    st.session_state["judge_running"] = True
    threading.Thread(target=_task, daemon=True).start()
# --- fin juge async ---



# --- État initial ---
if "run_search" not in st.session_state:
    st.session_state["run_search"] = False
if "question" not in st.session_state:
    st.session_state["question"] = ""
if "answer" not in st.session_state:
    st.session_state["answer"] = ""
if "history" not in st.session_state:
    st.session_state["history"] = load_history()
if "citations" not in st.session_state:
    st.session_state["citations"] = []
if "contexts_for_judge" not in st.session_state:
    st.session_state["contexts_for_judge"] = []
if "last_sig" not in st.session_state:
    st.session_state["last_sig"] = None
if "last_user_latency_ms" not in st.session_state:
    st.session_state["last_user_latency_ms"] = None
if "last_judge_scores" not in st.session_state:
    st.session_state["last_judge_scores"] = None
if "judge_running" not in st.session_state:
    st.session_state["judge_running"] = False



# --- Configuration de la page ---
st.set_page_config(
    page_title="Assistant de recherche juridique et réglementaire en courtage d’assurance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Fonction pour le lien PDF ===
def get_pdf_download_link(path, link_text="charte de l'IA"):
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="charte-ia.pdf" style="color:#e07a1e; font-weight: bold;">{link_text}</a>'
    return href

# --- STYLES ---
st.markdown("""
<style>
/* Fond et typographie générale */
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background-color: #f9f6f0 !important;
    font-family: 'Georgia', serif;
    color: #333333;
}
/* En-tête Streamlit */
header { background-color: #f9f6f0 !important; box-shadow: none !important; }
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f0ede4;
    color: #333333;
    font-family: 'Georgia', serif;
}
/* Dropdowns et filtres */
[data-baseweb="select"] {
    background-color: #e9e3d5 !important;
    color: #333 !important;
    font-family: 'Georgia', serif;
}
/* Titres principaux */
h1 { font-weight: 600; font-size: 2.5rem; margin-bottom: 0; font-family: 'Georgia', serif; }
h2 { margin-top: 0; font-weight: 400; color: #555555; }
/* Labels */
label, .stSelectSlider label, .stTextArea label {
    color: #666666 !important; font-weight: normal; font-family: 'Georgia', serif;
}
/* Champ input */
div[data-baseweb="input"] > div { background-color: #e9e3d5 !important; border-radius: 8px 0 0 8px !important; }
/* Placeholder */
div[data-baseweb="input"] input::placeholder { color: #666 !important; font-style: italic; }


div[data-baseweb="input"] input {
  color: #333 !important;         /* texte saisi */
  caret-color: #333 !important;   /* curseur */
}

.stTextInput input,
.stTextArea textarea {
  color: #333 !important;
  background-color: #e9e3d5 !important; /* cohérent avec votre fond */
}

/* Sélecteurs (multiselect/select) : texte et items */
[data-baseweb="select"] * {
  color: #333 !important;
}

/* Boutons */
.stButton > button {
    background-color: #e07a1e; color: white; font-weight: bold;
    border-radius: 0 8px 8px 0; padding: 0.5rem 1.2rem;
    border: 1px solid #999; border-left: none; margin-left: -1px;
}
.stButton > button:hover { background-color: #cc6919; color: white; }
/* Zone info */
details {
    background-color: #fff4d1; border-left: 5px solid #e07a1e;
    border-radius: 8px; padding: 12px 20px; font-size: 1.1rem;
    font-family: 'Georgia', serif; margin-bottom: 20px;
}
summary { font-weight: bold; font-size: 1.3rem; color: #333; cursor: pointer; }
/* Masquer footer Streamlit */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- UI helpers (badges & citations) ---
def format_badge(p: float) -> str:
    pct = int(round(max(0.0, min(1.0, float(p))) * 100))
    return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #bbb;font-size:12px;margin-right:6px'>{pct}%</span>"

def render_citation_block(chunk) -> str:
    meta = chunk.get("metadata", {})
    url = meta.get("url") or meta.get("lien") or meta.get("source_url")
    ident = meta.get("id_juridique") or meta.get("boi_id") or meta.get("identifiant") or "—"

    # ⬇️ prioriser le score agrégé des BIGs
    sim = float(
        chunk.get("big_similarity",
        chunk.get("similarity_raw", chunk.get("similarity_score", chunk.get("score_hybrid", 0.0))))
    )
    ctx = float(
        chunk.get("big_context",
        chunk.get("context_score", 0.0))
    )

    extrait = chunk.get("contenu", "")
    if len(extrait) > 600:
        extrait = extrait[:600] + "…"

    badges = (
        f"<strong>Similarité</strong> {format_badge(sim)}"
        f"<strong>Pertinence contextuelle</strong> {format_badge(ctx)}"
        f"<span style='margin-left:8px;font-size:12px;color:#666'>support: {int(chunk.get('big_support', 1))}</span>"
    )
    link = f"<a href='{url}' target='_blank'>Ouvrir la source</a>" if url else ""
    id_line = f"Identifiant juridique : <code>{ident}</code>"

    return f"""
    <div style="border:1px solid #ddd;padding:12px;border-radius:12px;margin:10px 0;background:#faf9f6">
      <div style="margin-bottom:6px">{badges}</div>
      <div style="font-size:13px;color:#555;margin-bottom:8px">{id_line}</div>
      <div style="font-size:13px;margin-bottom:8px">{link}</div>
      <div style="white-space:pre-wrap">{extrait}</div>
    </div>
    """

# === SIDEBAR POUR CONFIGURATION ===
st.sidebar.header("Configuration")

# Paramètres recherche
st.sidebar.markdown("**Nombre de résultats (Top K)**", unsafe_allow_html=True)
top_k = st.sidebar.slider("", min_value=5, max_value=30, value=TOP_K, step=1)

st.sidebar.markdown("**Poids de la recherche par mot-clé**", unsafe_allow_html=True)
lexical_weight = st.sidebar.slider("", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

st.sidebar.markdown("**Seuil de similarité minimum**", unsafe_allow_html=True)
min_similarity = st.sidebar.slider(
    "", min_value=0.0, max_value=0.9, value=0.35, step=0.05,
    help="Filtre les passages dont la similarité est inférieure à ce seuil (basé sur la distance cosinus pour le vectoriel et un score normalisé pour BM25)."
)

st.sidebar.markdown("**Filtres par métadonnées**", unsafe_allow_html=True)

def get_all_values(level: str):
    values = set()
    for base, sources in FILTER_TREE.items():
        if level == "base":
            values.add(base)
        for source, series in sources.items():
            if level == "source":
                values.add(source)
            for serie, divisions in series.items():
                if level == "serie":
                    values.add(serie)
                if level == "division":
                    values.update(divisions)
    return sorted(values)

# Filtres hiérarchiques
selected_base = st.sidebar.multiselect("Base", options=get_all_values("base"))

def get_sources():
    values = set()
    bases = selected_base or FILTER_TREE
    for base in bases:
        for source in FILTER_TREE.get(base, {}):
            values.add(source)
    return sorted(values)

selected_source = st.sidebar.multiselect("Source", options=get_sources())

def get_series():
    values = set()
    bases = selected_base or FILTER_TREE
    for base in bases:
        sources = selected_source or FILTER_TREE.get(base, {})
        for source in sources:
            series = FILTER_TREE.get(base, {}).get(source, {})
            values.update(series)
    return sorted(values)

selected_serie = st.sidebar.multiselect("Série", options=get_series())

def get_divisions():
    values = set()
    bases = selected_base or FILTER_TREE
    for base in bases:
        sources = selected_source or FILTER_TREE.get(base, {})
        for source in sources:
            series = selected_serie or FILTER_TREE.get(base, {}).get(source, {})
            for serie in series:
                divisions = FILTER_TREE.get(base, {}).get(source, {}).get(serie, [])
                values.update(divisions)
    return sorted(values)

selected_division = st.sidebar.multiselect("Division", options=get_divisions())

# Dictionnaire des filtres actifs
filters = {k: v for k, v in {
    "base": selected_base,
    "source": selected_source,
    "serie": selected_serie,
    "division": selected_division
}.items() if v}


# === STRUCTURE EN 2 COLONNES ===
col_main, col_droite = st.columns([4, 1])

with col_main:
    st.markdown("""
    <div style='text-align: center;'>
        <h1>Assistant comptable spécialisé en courtage d’assurance</h1>
        <p>Cet assistant vous aide à rechercher rapidement les informations pertinentes dans une documentation juridique spécialisée en courtage d’assurance.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚠️ Informations importantes", expanded=True):
        st.markdown("""
            <div class="information-box">
            • Ne saisissez jamais d’informations confidentielles dans ce système.<br>
            • L’assistant peut se tromper, chaque réponse doit être analysée avec esprit critique.<br><br>
            </div>  <!-- ⬅️ fermeture -->
        """, unsafe_allow_html=True)

        if CHARTE_IA_PATH.exists():
            lien = get_pdf_download_link(CHARTE_IA_PATH)
            st.markdown(f"• En cas de doute, vous pouvez consulter la {lien}.", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Le fichier de la charte de l’IA est introuvable.")

    col1, col2 = st.columns([6, 1])
    with col1:
        question = st.text_input("", placeholder="Exemple : Quelles sont les conditions de résiliation d’un contrat collectif santé ?", label_visibility="collapsed")
    with col2:
        if st.button("Interroger", use_container_width=True) and question.strip():
            st.session_state["run_search"] = True
            st.session_state["question"] = question.strip()

    if st.session_state["run_search"]:
        with st.spinner("Recherche et génération de la réponse..."):
            try:
                t0 = time.perf_counter()  # chrono global start
                rewrites = rewrite_query(st.session_state["question"])
                rewrite = rewrites[0] if rewrites else st.session_state["question"]
                multi = hybrid_search_multi(rewrite, top_k=top_k, filters=filters,
                                            lexical_weight=lexical_weight, min_similarity=min_similarity)
                chunks = multi["fusion"]
                reranked = rerank_chunks_with_gpt(st.session_state["question"], chunks)
                passages = reranked if reranked else chunks

                # Agrégation au niveau big + tri par score agrégé
                big_scored = aggregate_big_chunks_from_small(passages, top_k=top_k)
                if not big_scored:
                    # Fallback : reconstruire les BIGs “bruts” (sans scores agrégés) depuis les smalls
                    big_scored = get_big_chunks_from_small(passages[:top_k])

                # On affiche désormais les BIGs (avec scores agrégés)
                st.session_state["citations"] = big_scored

                # Construire la liste des contexts à envoyer au juge
                contexts = []
                for ch in big_scored or []:
                    md = ch.get("metadata", {})
                    contexts.append({
                        "chunk_id": ch.get("chunk_id") or md.get("chunk_id") or md.get("parent_chunk_id"),
                        "source": md.get("permalien") or md.get("url") or md.get("lien"),
                        "titre": md.get("titre_document") or md.get("titre_bloc"),
                        "extrait": (ch.get("contenu") or "")[:1500]
                    })

                # Et on génère la réponse à partir des BIGs sélectionnés
                answer = generate_answer(rewrite, big_scored, include_sources=True)

                st.session_state["answer"] = answer

                # === Historique : trace compacte ===
                def _chunk_summary(cs):
                    md = cs.get("metadata", {})
                    return {
                        "chunk_id": cs.get("chunk_id") or md.get("chunk_id"),
                        "parent_chunk_id": cs.get("parent_chunk_id") or md.get("parent_chunk_id"),
                        "titre_document": md.get("titre_document"),
                        "titre_bloc": md.get("titre_bloc"),
                        "permalien": md.get("permalien"),
                        "score_hybrid": cs.get("score_hybrid"),
                        "provenance": cs.get("provenance"),
                    }


                vj = st.session_state.get("last_judge") or {}
                history_entry = {
                    "ts": time.time(),
                    "question": st.session_state["question"],
                    "answer": st.session_state["answer"],
                    "judge_scores": vj.get("scores", {}),
                    "judge_flags": vj.get("flags", {}),
                    "judge_explanation": vj.get("explanation", "")
                }

                append_history(history_entry)
                st.session_state["history"].append(history_entry)

            except Exception as e:
                st.session_state["answer"] = f"<div style='color: black; background-color: #ffe6e6; border-left: 5px solid #cc0000; padding: 1rem; border-radius: 6px;'>❌ Une erreur est survenue : {e}</div>"
            finally:
                st.session_state["run_search"] = False

    if st.session_state["answer"]:
        if "Une erreur est survenue" in st.session_state["answer"]:
            st.markdown(st.session_state["answer"], unsafe_allow_html=True)
        else:
            st.markdown("### ✅ Réponse")
            st.write(st.session_state["answer"])

            # ---- Lancer le juge APRÈS affichage de la réponse ----
            # Construit les contexts à partir des citations affichées
            contexts = []
            for ch in (st.session_state.get("citations") or []):
                md = ch.get("metadata", {})
                contexts.append({
                    "chunk_id": ch.get("chunk_id") or md.get("chunk_id") or md.get("parent_chunk_id"),
                    "source": md.get("permalien") or md.get("url") or md.get("lien"),
                    "titre": md.get("titre_document") or md.get("titre_bloc"),
                    "extrait": (ch.get("contenu") or "")[:1500]
                })

            # Eviter de relancer plusieurs fois pour la même réponse
            sig = f"{st.session_state.get('question', '')}::{st.session_state.get('answer', '')[:200]}"
            if st.session_state.get("last_sig") != sig and contexts:
                _launch_judge_async(st.session_state.get("question"), contexts, st.session_state["answer"])
                st.session_state["last_sig"] = sig

            # Message d'état
            if st.session_state.get("judge_running"):
                st.info("⏳ Évaluation automatique en cours…")
            elif st.session_state.get("last_judge"):
                s = st.session_state["last_judge"].get("scores", {})
                st.success(
                    f"Évaluation : contexte={s.get('score_context')}/5 · cohérence={s.get('score_coherence')}/5 · réponse={s.get('score_answer')}/5")

            # --- Sources + indicateurs de confiance ---
            if st.session_state.get("citations"):
                st.markdown("### Sources et indicateurs")
                for ch in st.session_state["citations"]:
                    st.markdown(render_citation_block(ch), unsafe_allow_html=True)
            else:
                st.caption("Aucune source pertinente n'a été identifiée.")

        st.markdown("### Votre avis sur la réponse")

        # Initialisation des variables dans le state
        if "user_feedback" not in st.session_state:
            st.session_state["user_feedback"] = None
        if "feedback_sent" not in st.session_state:
            st.session_state["feedback_sent"] = False

        if not st.session_state["feedback_sent"]:  # 🔒 feedback non encore envoyé
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", use_container_width=True):
                    st.session_state["user_feedback"] = 1
            with col2:
                if st.button("👎", use_container_width=True):
                    st.session_state["user_feedback"] = -1

            reason = None
            comment = None

            # Si l'utilisateur a cliqué sur 👎 → proposer un menu déroulant
            if st.session_state["user_feedback"] == -1:
                reason = st.selectbox(
                    "Pourquoi la réponse ne vous satisfait pas ?",
                    [
                        "Réponse incomplète",
                        "Hors sujet",
                        "Ambiguë / mal formulée",
                        "Sources insuffisantes",
                        "Ton inadapté",
                        "Latence excessive",
                        "Autre"
                    ],
                    index=None,
                    placeholder="Choisir une raison"
                )

                if reason == "Autre":
                    comment = st.text_area("Précisez (facultatif)", max_chars=200)

            # Bouton pour valider l'avis
            if st.session_state["user_feedback"] is not None:
                if st.button("✅ Envoyer mon avis"):
                    append_feedback({
                        "ts": time.time(),
                        "question": st.session_state["question"],
                        "answer": st.session_state["answer"],
                        "rating": st.session_state["user_feedback"],  # 👍 = 1, 👎 = -1
                        "reason_user": reason,
                        "comment": comment
                    })
                    st.success("Merci pour votre retour 🙏")
                    st.session_state["feedback_sent"] = True  # 🔒 verrouillage
        else:
            st.info("✅ Votre avis a déjà été enregistré. Merci !")

with col_droite:
    st.markdown("""
        <div style="background-color: #f0ede4; padding: 1rem; border-radius: 8px; overflow-y: auto; max-height: 90vh;">
            <h3 style="text-align:center;">📜 Historique</h3>
    """, unsafe_allow_html=True)

    if st.session_state["history"]:
        for idx, h in enumerate(reversed(st.session_state["history"]), 1):
            question = h.get("question", "")
            answer = h.get("answer", "")
            ts = h.get("timestamp") or ""
            date_str = (ts[:10] if isinstance(ts, str) and len(ts) >= 10 else "—")
            with st.expander(f"❓ {question}", expanded=False):
                st.markdown(answer, unsafe_allow_html=True)
    else:
        st.caption("Aucune recherche pour le moment.")

    st.markdown("</div>", unsafe_allow_html=True)


