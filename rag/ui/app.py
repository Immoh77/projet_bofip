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
import logging

# --- IMPORTS RAG / QDRANT ---
from rag.config import TOP_K, FILTER_TREE, CHARTE_IA_PATH, LOG_LEVEL, DEFAULT_CHAT_MODEL
from rag.retrieval.qdrant_retriever import QdrantRetriever
from rag.generation.answerer import generate_answer
from rag.utils.history import load_history, append_history
from rag.evaluation.judge import run_judge

# ==========================================================
# CONFIGURATION DU LOGGER
# ==========================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("🔧 Application RAG lancée (niveau de log : %s)", LOG_LEVEL)

# --- Initialisation Qdrant ---
try:
    retriever = QdrantRetriever()
    logging.info("✅ QdrantRetriever initialisé.")
except Exception as e:
    st.error(f"❌ Erreur lors de l'initialisation de Qdrant : {e}")
    logging.exception("💥 Échec d'initialisation Qdrant : %s", e)
    retriever = None

# --- JUGE ASYNCHRONE ---
def _launch_judge_async(question: str, contexts: list, answer: str):
    """Lance le juge en arrière-plan et enregistre le résultat dans st.session_state."""
    def _task():
        try:
            verdict = run_judge(question, contexts, answer)
            st.session_state["last_judge"] = verdict
            try:
                append_eval({
                    "ts": time.time(),
                    "question": question,
                    "scores": verdict.get("scores", {}),
                    "flags": verdict.get("flags", {})
                })
            except Exception:
                pass
        finally:
            st.session_state["judge_running"] = False

    st.session_state["judge_running"] = True
    threading.Thread(target=_task, daemon=True).start()

# --- ÉTAT INITIAL ---
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
if "judge_running" not in st.session_state:
    st.session_state["judge_running"] = False

# --- CONFIGURATION PAGE ---
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


# --- STYLES ORIGINAUX ---
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background-color: #f9f6f0 !important;
    font-family: 'Georgia', serif;
    color: #333333;
}

/* --- HEADER --- */
header { background-color: #f9f6f0 !important; box-shadow: none !important; }

/* --- SIDEBAR --- */
[data-testid="stSidebar"] {
    background-color: #f0ede4;
    color: #333333;
    font-family: 'Georgia', serif;
}

/* --- SELECTS GÉNÉRAUX (métadonnées, menus) --- */
[data-baseweb="select"] {
    background-color: #e9e3d5 !important;
    color: #333 !important;
    font-family: 'Georgia', serif;
}

/* --- TITRES --- */
h1 { font-weight: 600; font-size: 2.5rem; margin-bottom: 0; font-family: 'Georgia', serif; }
/* H2 plus gros et bien en gras pour les sections TEXTES/APPLICATION */
h2 { 
    margin-top: 1.25rem; 
    margin-bottom: 0.5rem;
    font-weight: 700;         /* gras marqué */
    font-size: 1.8rem;        /* plus grand */
    color: #333333; 
    font-family: 'Georgia', serif;
}

/* Optionnel : rendre le gras plus lisible partout */
strong, b { color: #222; }

/* --- LABELS ET TEXTES --- */
label, .stSelectSlider label, .stTextArea label {
    color: #666666 !important;
    font-weight: normal;
    font-family: 'Georgia', serif;
}

/* --- INPUTS --- */
div[data-baseweb="input"] > div {
    background-color: #e9e3d5 !important;
    border-radius: 8px 0 0 8px !important;
}
div[data-baseweb="input"] input::placeholder {
    color: #666 !important;
    font-style: italic;
}
div[data-baseweb="input"] input {
    color: #333 !important;
    caret-color: #333 !important;
}

/* --- TEXTAREA --- */
.stTextInput input,
.stTextArea textarea {
    color: #333 !important;
    background-color: #e9e3d5 !important;
}

/* --- TEXT GÉNÉRAL DANS SELECTS --- */
[data-baseweb="select"] * { color: #333 !important; }

/* --- BOUTONS --- */
.stButton > button {
    background-color: #e07a1e;
    color: white;
    font-weight: bold;
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 1.2rem;
    border: 1px solid #999;
    border-left: none;
    margin-left: -1px;
}
.stButton > button:hover {
    background-color: #cc6919;
    color: white;
}

/* === STYLE DU SELECTEUR DE MODELE DE LANGAGE (radio) === */

/* Conteneur */
div[data-testid="stRadio"] {
    margin-top: 0.5rem;
}

/* Style général des boutons radio */
div[data-testid="stRadio"] label {
    background-color: #e9e3d5 !important;
    border: 2px solid #e07a1e !important;
    border-radius: 8px !important;
    color: #333 !important;  /* couleur par défaut noire */
    padding: 6px 10px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
}

/* Survol */
div[data-testid="stRadio"] label:hover {
    border-color: #cc6919 !important;
}

/* Bouton radio sélectionné */
div[data-testid="stRadio"] label div[role="radio"][aria-checked="true"] {
    background-color: #e07a1e !important;
    border-color: #e07a1e !important;
}

/* Texte du modèle sélectionné — rouge foncé lisible */
div[data-testid="stRadio"] label div[role="radio"][aria-checked="true"] + p {
    color: #a22c00 !important;  /* rouge foncé */
    font-weight: bold;
}

/* Texte des modèles non sélectionnés — noir */
div[data-testid="stRadio"] label p {
    margin-left: 8px;
    font-weight: bold;
    font-family: 'Georgia', serif;
    color: #333 !important;
}

/* === TITRES DE SECTION (même style pour Métadonnées et Modèle de langage) === */
.sidebar-content h3, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h2 {
    font-size: 1.1rem !important;
    font-weight: bold !important;
    color: #333 !important;
    text-transform: uppercase;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

/* --- EXPANDERS --- */
details {
    background-color: #fff4d1;
    border-left: 5px solid #e07a1e;
    border-radius: 8px;
    padding: 12px 20px;
    font-size: 1.1rem;
    font-family: 'Georgia', serif;
    margin-bottom: 20px;
}
summary {
    font-weight: bold;
    font-size: 1.3rem;
    color: #333;
    cursor: pointer;
}

/* --- FOOTER --- */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# === SIDEBAR ===
st.sidebar.header("Configuration")

# --- TOP K ---
st.sidebar.markdown("### 🎯 NOMBRE DE RESULTATS (TOP K)", unsafe_allow_html=True)
top_k = st.sidebar.slider("", min_value=5, max_value=30, value=TOP_K, step=1)

# --- MODELE DE LANGAGE ---
model_choices = ["chatgpt-4o-latest", "gpt-4o", "gpt-3.5-turbo"]

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = model_choices[0]

selected_model = st.sidebar.radio(
    "🧠 Choisissez le modèle :",
    options=model_choices,
    index=model_choices.index(st.session_state["selected_model"]),
    key="llm_model_select",
)
st.session_state["selected_model"] = selected_model
logging.info("🧠 Modèle sélectionné dans l'UI : %s", selected_model)

# --- FILTRES PAR METADONNEES ---
st.sidebar.markdown("### 🔎 FILTRES PAR METADONNEES", unsafe_allow_html=True)


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

selected_base = st.sidebar.multiselect("Base", options=get_all_values("base"))
selected_source = st.sidebar.multiselect("Source", options=get_all_values("source"))
selected_serie = st.sidebar.multiselect("Série", options=get_all_values("serie"))
selected_division = st.sidebar.multiselect("Division", options=get_all_values("division"))

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
            </div>
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

    # --- Recherche Qdrant ---
    if st.session_state["run_search"]:
        with st.spinner("Recherche et génération de la réponse..."):
            try:
                if retriever is None:
                    raise RuntimeError("Qdrant non initialisé.")

                docs = retriever.retrieve_with_subquery_rerank(
                    st.session_state["question"],
                    filters=filters
                )
                fused_docs = docs.get("fusion_finale", [])

                st.session_state["citations"] = fused_docs
                answer = generate_answer(
                    st.session_state["question"],
                    docs,
                    include_sources=True,
                    llm_model=st.session_state["selected_model"],
                )
                st.session_state["answer"] = answer

                history_entry = {
                    "ts": time.time(),
                    "question": st.session_state["question"],
                    "answer": answer
                }
                append_history(history_entry)
                st.session_state["history"].append(history_entry)

            except Exception as e:
                st.session_state["answer"] = f"<div style='color: black; background-color: #ffe6e6; border-left: 5px solid #cc0000; padding: 1rem; border-radius: 6px;'>❌ Une erreur est survenue : {e}</div>"
            finally:
                st.session_state["run_search"] = False

    # --- Affichage des résultats ---
    if st.session_state["answer"]:
        if "Une erreur est survenue" in st.session_state["answer"]:
            st.markdown(st.session_state["answer"], unsafe_allow_html=True)
        else:
            st.markdown(st.session_state["answer"], unsafe_allow_html=False)

            contexts = [{"extrait": (c.get("contenu") or "")[:1500]} for c in (st.session_state.get("citations") or [])]
            sig = f"{st.session_state.get('question', '')}::{st.session_state.get('answer', '')[:200]}"
            if st.session_state.get("last_sig") != sig and contexts:
                _launch_judge_async(st.session_state.get("question"), contexts, st.session_state["answer"])
                st.session_state["last_sig"] = sig

            if st.session_state.get("judge_running"):
                st.info("⏳ Évaluation automatique en cours…")
            elif st.session_state.get("last_judge"):
                s = st.session_state["last_judge"].get("scores", {})
                st.success(
                    f"Évaluation : contexte={s.get('score_context')}/5 · cohérence={s.get('score_coherence')}/5 · réponse={s.get('score_answer')}/5"
                )

        st.markdown("### Votre avis sur la réponse")
        if "user_feedback" not in st.session_state:
            st.session_state["user_feedback"] = None
        if "feedback_sent" not in st.session_state:
            st.session_state["feedback_sent"] = False

        if not st.session_state["feedback_sent"]:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", use_container_width=True):
                    st.session_state["user_feedback"] = 1
            with col2:
                if st.button("👎", use_container_width=True):
                    st.session_state["user_feedback"] = -1

            reason, comment = None, None
            if st.session_state["user_feedback"] == -1:
                reason = st.selectbox(
                    "Pourquoi la réponse ne vous satisfait pas ?",
                    ["Réponse incomplète", "Hors sujet", "Sources insuffisantes", "Autre"],
                    index=None,
                    placeholder="Choisir une raison"
                )
                if reason == "Autre":
                    comment = st.text_area("Précisez (facultatif)", max_chars=200)

            if st.session_state["user_feedback"] is not None and st.button("✅ Envoyer mon avis"):
                append_feedback({
                    "ts": time.time(),
                    "question": st.session_state["question"],
                    "answer": st.session_state["answer"],
                    "rating": st.session_state["user_feedback"],
                    "reason_user": reason,
                    "comment": comment
                })
                st.success("Merci pour votre retour 🙏")
                st.session_state["feedback_sent"] = True
        else:
            st.info("✅ Votre avis a déjà été enregistré. Merci !")


# --- Historique (colonne droite) ---
with col_droite:
    st.markdown("""
        <div style="background-color: #f0ede4; padding: 1rem; border-radius: 8px; overflow-y: auto; max-height: 90vh;">
            <h3 style="text-align:center;">📜 Historique</h3>
    """, unsafe_allow_html=True)

    if st.session_state["history"]:
        for idx, h in enumerate(reversed(st.session_state["history"]), 1):
            question = h.get("question", "")
            answer = h.get("answer", "")
            with st.expander(f"❓ {question}", expanded=False):
                st.markdown(answer, unsafe_allow_html=True)
    else:
        st.caption("Aucune recherche pour le moment.")

    st.markdown("</div>", unsafe_allow_html=True)
