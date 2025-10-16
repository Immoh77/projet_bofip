# dashboard.py
# ──────────────────────────────────────────────────────────────
# Dashboard Streamlit : 2 onglets "Dashboard" et "Historique".
# Ajouts :
#  - extraction d'un flag hallucination binaire (0/1) depuis flags.hallucination
#  - ajout d'une colonne date_requête (YYYY-MM-DD)
#  - KPI hallucinations (%) + KPI avis moyen utilisateur
#  - gestion robuste de latence (None-safe)
#  - graphiques quotidiens (satisfaction & scores) sans affichage des heures
# ──────────────────────────────────────────────────────────────

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# Constantes / fichiers logs
# ============================================================================
LOG_DIR  = Path("logs")
EVAL_PATH = LOG_DIR / "eval_judge.jsonl"      # écrit via append_eval(...)
FB_PATH   = LOG_DIR / "user_feedback.jsonl"   # 👍/👎 + commentaires (si dispo)

CATEGORIES = [
    "Pas de problème",
    "Hallucination",
    "Réponse incomplète",
    "Réponse imprécise",
    "Mauvais format",
    "Autre",
]

# ============================================================================
# Helpers
# ============================================================================
def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def to_df(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def _note_to01(v):
    """Normalise la note utilisateur (👍/👎, bool, 1/0, texte) en 0/1/NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower()
    # positifs
    if s in {"👍","up","oui","true","1","like","thumbs up","+","positif","positive","yes","y"}:
        return 1
    # négatifs
    if s in {"👎","down","non","false","0","dislike","thumbs down","-","negatif","négatif","no","n"}:
        return 0
    try:
        f = float(s)
        if np.isnan(f):
            return np.nan
        return 1 if f > 0 else 0
    except Exception:
        return np.nan

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ============================================================================
# Chargement
# ============================================================================
rows_eval = read_jsonl(EVAL_PATH)
rows_fb   = read_jsonl(FB_PATH)

df_eval = to_df(rows_eval)
df_fb   = to_df(rows_fb)

# ============================================================================
# Prétraitements df_eval
# ============================================================================
if not df_eval.empty:
    # Date d'évaluation
    dt_col = "ts" if "ts" in df_eval.columns else _pick_col(df_eval, ["timestamp","created_at","date"])
    df_eval["dt_eval"] = pd.to_datetime(df_eval.get(dt_col), errors="coerce")

    # Aplatir scores -> colonnes
    if "scores" in df_eval.columns:
        sc = df_eval["scores"].apply(lambda d: d if isinstance(d, dict) else {})
        df_eval["score_context"]   = sc.apply(lambda d: d.get("score_context"))
        df_eval["score_coherence"] = sc.apply(lambda d: d.get("score_coherence"))
        df_eval["score_reponse"]   = sc.apply(lambda d: d.get("score_answer"))
        if "groundedness" not in df_eval.columns:
            df_eval["groundedness"] = sc.apply(lambda d: d.get("groundedness"))

    # Latence -> secondes (None-safe)
    if "latency" in df_eval.columns:
        df_eval["lat_total_s"] = df_eval["latency"].apply(
            lambda d: (float(d.get("lat_total_ms"))/1000.0)
            if isinstance(d, dict) and d.get("lat_total_ms") is not None
            else np.nan
        )

    # Hallucination binaire (0/1) à partir de flags.hallucination
    if "flags" in df_eval.columns:
        def _to01_flag(x):
            if not isinstance(x, dict):
                return np.nan
            v = x.get("hallucination", None)
            if v is None:
                return np.nan
            s = str(v).strip().lower()
            if s in {"1","true","yes","y"}: return 1
            if s in {"0","false","no","n"}: return 0
            try:
                iv = int(float(s))
                return 1 if iv == 1 else 0 if iv == 0 else np.nan
            except Exception:
                return np.nan
        df_eval["hallucination"] = df_eval["flags"].apply(_to01_flag)

# ============================================================================
# Prétraitements df_fb (avis utilisateur) — robustes
# ============================================================================
if not df_fb.empty:
    # 1) Date feedback -> dt_fb (auto-détection)
    date_col = _pick_col(df_fb, ["dt_fb", "ts", "timestamp", "created_at", "date"])
    df_fb["dt_fb"] = pd.to_datetime(df_fb[date_col], errors="coerce") if date_col else pd.NaT

    # 2) Note -> note01 (auto-détection & normalisation 0/1)
    if "note01" not in df_fb.columns:
        note_col = _pick_col(
            df_fb,
            ["note_utilisateur","note","rating","score","satisfaction",
             "feedback","emoji","thumb","thumbs","like","is_positive"]
        )
        df_fb["note01"] = df_fb[note_col].apply(_note_to01) if note_col else np.nan

    # KPI avis moyen
    user_rate_all = 100 * df_fb["note01"].mean() if df_fb["note01"].notna().any() else np.nan

    # Dernier avis par question (si souhaité)
    df_fb_last = (
        df_fb.sort_values("dt_fb")
             .drop_duplicates(subset=["question"], keep="last")
             .copy()
    )
    if "note01" not in df_fb_last.columns and "note_utilisateur" in df_fb_last.columns:
        df_fb_last["note01"] = df_fb_last["note_utilisateur"].apply(_note_to01)
    user_rate_last = 100 * df_fb_last["note01"].mean() if "note01" in df_fb_last and df_fb_last["note01"].notna().any() else np.nan

    keep_fb_cols = ["question","note_utilisateur","note01","comment","grille_interne","raison_si_mal_notee","dt_fb"]
    df_fb_last = df_fb_last[[c for c in keep_fb_cols if c in df_fb_last.columns]].copy()
else:
    df_fb_last = pd.DataFrame(columns=["question","note_utilisateur","note01","comment","grille_interne","raison_si_mal_notee","dt_fb"])
    user_rate_all = np.nan
    user_rate_last = np.nan

# ============================================================================
# Fusion eval + fb pour l'historique  (doit être AVANT la partie UI)
# ============================================================================
if not df_eval.empty:
    keep_cols = ["question","lat_total_s","score_context","score_coherence","score_reponse","dt_eval"]
    if "hallucination" in df_eval.columns: keep_cols.append("hallucination")
    if "groundedness"  in df_eval.columns: keep_cols.append("groundedness")
    if "explanation" in df_eval.columns: keep_cols.append("explanation")
    df_eval_last = (
        df_eval.sort_values("dt_eval")
               .drop_duplicates(subset=["question"], keep="last")
               [keep_cols]
               .copy()
    )
else:
    df_eval_last = pd.DataFrame(
        columns=["question","lat_total_s","score_context","score_coherence","score_reponse","dt_eval","hallucination","groundedness"]
    )

df_full = df_fb_last.merge(df_eval_last, on="question", how="outer")

# Ajout de la date de requête (priorité à dt_eval, sinon dt_fb)
df_full["date_requête"] = (
    df_full.get("dt_eval", pd.NaT)
           .fillna(df_full.get("dt_fb", pd.NaT))
           .apply(lambda x: pd.to_datetime(x, errors="coerce"))
           .dt.date
)

# ============================================================================
# UI Streamlit
# ============================================================================
st.set_page_config(page_title="Dashboard RAG", layout="wide")
page = st.sidebar.radio("Navigation", ["Dashboard","Historique"])

# ============================================================================
# Onglet Dashboard
# ============================================================================
if page == "Dashboard":
    st.title("📊 Dashboard")

    # KPI hallucinations (sur df_eval brut)
    if not df_eval.empty and "hallucination" in df_eval.columns and df_eval["hallucination"].notna().any():
        halluc_rate = 100 * df_eval["hallucination"].mean()
    else:
        halluc_rate = np.nan

    # Moyennes des 3 scores
    avg_ctx = df_eval["score_context"].dropna().astype(float).mean() if "score_context" in df_eval else np.nan
    avg_coh = df_eval["score_coherence"].dropna().astype(float).mean() if "score_coherence" in df_eval else np.nan
    avg_ans = df_eval["score_reponse"].dropna().astype(float).mean() if "score_reponse" in df_eval else np.nan

    # KPI : 5 colonnes dont avis utilisateur (tous les avis)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hallucinations", f"{halluc_rate:.0f}%" if np.isfinite(halluc_rate) else "—")
    c2.metric("Score contexte (moy.)", f"{avg_ctx:.2f}" if np.isfinite(avg_ctx) else "—")
    c3.metric("Score cohérence (moy.)", f"{avg_coh:.2f}" if np.isfinite(avg_coh) else "—")
    c4.metric("Score réponse (moy.)", f"{avg_ans:.2f}" if np.isfinite(avg_ans) else "—")
    c5.metric("Avis utilisateur (👍)", f"{user_rate_all:.0f}%" if np.isfinite(user_rate_all) else "—")
    # Variante : dernier avis par question
    # c5.metric("Avis utilisateur (dernier)", f"{user_rate_last:.0f}%" if np.isfinite(user_rate_last) else "—")

    # ─────────────────────────────────────────────────────────
    # ÉVOLUTIONS DANS LE TEMPS
    # ─────────────────────────────────────────────────────────
    st.subheader("📈 Évolutions dans le temps")

    # === Satisfaction utilisateur (quotidien + MM7) =========
    with st.expander("Satisfaction utilisateur (quotidien + MM7)", expanded=True):
        if df_fb.empty:
            st.caption("Aucun feedback utilisateur disponible.")
        else:
            fb_clean = df_fb.dropna(subset=["dt_fb", "note01"]).copy()
            if fb_clean.empty:
                st.caption("Aucune donnée exploitable (dates ou notes manquantes).")
            else:
                sat = (
                    fb_clean.assign(date=lambda d: pd.to_datetime(d["dt_fb"]).dt.date)  # jour uniquement
                           .groupby("date", dropna=True)["note01"]
                           .mean()
                           .mul(100)  # en %
                           .rename("Satisfaction (%)")
                           .to_frame()
                )
                if sat.empty:
                    st.caption("Aucune agrégation quotidienne disponible.")
                else:
                    sat["MM7"] = sat["Satisfaction (%)"].rolling(7, min_periods=1).mean()
                    sat.index = sat.index.astype(str)  # pas d'heure affichée
                    st.subheader("📈 Satisfaction utilisateur (moyenne quotidienne)")
                    st.line_chart(sat)

    # === Scores du LLM Judge (quotidien + MM7) ==============
    with st.expander("Scores du LLM Judge (quotidien + MM7)", expanded=True):
        needed = {"dt_eval","score_context","score_coherence","score_reponse"}
        if not df_eval.empty and needed.issubset(df_eval.columns):
            sc_daily = (
                df_eval.dropna(subset=["dt_eval"])
                       .assign(date=lambda d: pd.to_datetime(d["dt_eval"]).dt.date)
                       .groupby("date", dropna=True)[["score_context","score_coherence","score_reponse"]]
                       .mean()
                       .rename(columns={
                           "score_context":   "Contexte (1–5)",
                           "score_coherence": "Cohérence (1–5)",
                           "score_reponse":   "Réponse (1–5)",
                       })
            )
            if sc_daily.empty:
                st.caption("Pas encore de données d’évaluation quotidiennes.")
            else:
                sc_daily.index = sc_daily.index.astype(str)  # pas d'heure affichée
                st.subheader("📈 Scores du LLM Judge (moyenne quotidienne)")
                st.line_chart(sc_daily)

                sc_roll = sc_daily.rolling(7, min_periods=1).mean()
                sc_roll.index = sc_roll.index.astype(str)
                st.caption("Moyenne mobile 7 jours (lissage)")
                st.line_chart(sc_roll)
        else:
            st.caption("Données d’évaluation insuffisantes pour tracer les scores.")

# ============================================================================
# Onglet Historique
# ============================================================================
if page == "Historique":
    st.title("📜 Historique")

    final_cols = [
        "date_requête",
        "question",
        "answer",
        "lat_total_s",
        "score_context","score_coherence","score_reponse",
        "groundedness",
        "hallucination",
        "explanation",
        "note_utilisateur","comment","grille_interne","raison_si_mal_notee",
    ]
    # garde anti-NameError (au cas où)
    if 'df_full' not in locals():
        df_full = pd.DataFrame()
    df_view = df_full[[c for c in final_cols if c in df_full.columns]].copy()

    st.data_editor(
        df_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date_requête":    st.column_config.DateColumn("Date"),
            "question":         st.column_config.TextColumn("Question posée", disabled=True),
            "answer":           st.column_config.TextColumn("Réponse donnée", disabled=True),
            "lat_total_s":      st.column_config.NumberColumn("Latence (s)", format="%.2f", disabled=True),
            "score_context":    st.column_config.NumberColumn("Score LLM — pertinence du contexte", step=1, disabled=True),
            "score_coherence":  st.column_config.NumberColumn("Score LLM — cohérence utilisation passages", step=1, disabled=True),
            "score_reponse":    st.column_config.NumberColumn("Score LLM — pertinence de la réponse", step=1, disabled=True),
            "groundedness":     st.column_config.NumberColumn("Fiabilité (0/1)", format="%.0f", help="1 = aucune hallucination ; 0 = hallucination", disabled=True),
            "hallucination":    st.column_config.SelectboxColumn("Hallucination", options=[0,1], help="0 = non ; 1 = oui", disabled=True),
            "note_utilisateur": st.column_config.TextColumn("Note utilisateur (👍/👎)", disabled=True),
            "comment":          st.column_config.TextColumn("Commentaire", disabled=True),
            "explanation": st.column_config.TextColumn("Explication du juge (brève)", help="Justification 1–2 phrases", disabled=True),
            "grille_interne":   st.column_config.SelectboxColumn(
                "Catégorie interne (éditable)",
                options=CATEGORIES,
                required=False,
                help="Sélectionne la cause selon la grille interne."
            ),
            "raison_si_mal_notee": st.column_config.TextColumn("Raison (si 👎)", disabled=True),
        },
    )