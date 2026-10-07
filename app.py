import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import joblib
from sklearn.metrics import auc, confusion_matrix, roc_curve

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#1f3a2a", "grid.linestyle": "--", "grid.linewidth": 0.6,
    "text.color": "#d7e6dc",
})

try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))

def art(filename):
    return os.path.join(ARTIFACTS_DIR, filename)

KINGDOM_LABELS = {
    "bct": "Bacteria", "vrl": "Virus",    "pln": "Plant",
    "vrt": "Vertebrate","inv": "Invertebrate","mam": "Mammal",
    "phg": "Phage",     "rod": "Rodent",   "pri": "Primate",
    "arc": "Archaea",   "plm": "Plasmid",
}

KINGDOM_COLORS = {
    "bct": "#4E9AF1", "vrl": "#E05C5C", "pln": "#4ade80",
    "vrt": "#F5A623", "inv": "#9B59B6", "mam": "#E67E22",
    "phg": "#1ABC9C", "rod": "#F39C12", "pri": "#D35400",
    "arc": "#7F8C8D", "plm": "#BDC3C7",
}

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    X_train = pd.read_csv(art("X_train.csv"))
    X_test  = pd.read_csv(art("X_test.csv"))
    y_train = pd.Series(pd.read_csv(art("y_train.csv")).iloc[:, -1].astype(str).values)
    y_test  = pd.Series(pd.read_csv(art("y_test.csv")).iloc[:, -1].astype(str).values)
    X_all   = pd.concat([X_train, X_test], ignore_index=True)
    y_all   = pd.concat([y_train, y_test], ignore_index=True)
    return X_train, X_test, y_train, y_test, X_all, y_all

@st.cache_resource
def load_models():
    pca_model  = joblib.load(art("pca_model.pkl"))
    pca_scaler = joblib.load(art("pca_scaler.pkl"))
    svm_model  = joblib.load(art("svm_model.pkl"))
    custom_rf  = joblib.load(art("custom_rf_model.pkl"))
    return pca_model, pca_scaler, svm_model, custom_rf

@st.cache_data
def load_coords():
    train = pd.read_csv(art("pca_train_coords.csv")).drop(columns=["Kingdom"], errors="ignore")
    test  = pd.read_csv(art("pca_test_coords.csv")).drop(columns=["Kingdom"],  errors="ignore")
    return train, test

@st.cache_data
def load_som():
    return pd.read_csv(art("som_bmu_coords.csv"))

@st.cache_data
def load_feature_importance():
    df = pd.read_csv(art("rf_feature_importance.csv"))
    df.columns = [c.strip() for c in df.columns]
    cols = [c for c in df.columns if not c.lower().startswith("unnamed")]
    df = df[cols]
    rename = {}
    if len(cols) >= 1: rename[cols[0]] = "feature"
    if len(cols) >= 2: rename[cols[1]] = "importance"
    return df.rename(columns=rename)[["feature","importance"]]

@st.cache_data
def load_classifier_comparison():
    return pd.read_csv(art("classifier_comparison_table.csv")).drop(columns=["Unnamed: 0"], errors="ignore")

@st.cache_data
def load_vrl_misclassified():
    return pd.read_csv(art("vrl_misclassified_proba.csv")).drop(columns=["Unnamed: 0"], errors="ignore")

@st.cache_data
def load_custom_rf_results():
    comparison   = pd.read_csv(art("custom_rf_comparison.csv"))
    perclass     = pd.read_csv(art("custom_rf_perclass_f1.csv"))
    host_mimicry = pd.read_csv(art("host_mimicry_scores.csv"))
    return comparison, perclass, host_mimicry

@st.cache_data
def load_evaluation_artifacts():
    ablation     = pd.read_csv(art("ablation_feature_sets.csv"))
    weight_just  = pd.read_csv(art("weight_justification.csv"))
    return ablation, weight_just

@st.cache_data(show_spinner="Calculating CodonPrint test-set predictions...")
def evaluate_custom_rf(_model, X_test, y_test):
    predictions = _model.predict(X_test)
    probabilities = _model.predict_proba(X_test)
    classes = np.asarray(_model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=classes)

    roc_curves = []
    actual = np.asarray(y_test)
    for index, kingdom in enumerate(classes):
        binary_labels = actual == kingdom
        if binary_labels.all() or not binary_labels.any():
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_labels, probabilities[:, index]
        )
        roc_curves.append((
            str(kingdom),
            false_positive_rate,
            true_positive_rate,
            auc(false_positive_rate, true_positive_rate),
        ))
    return classes, matrix, roc_curves

@st.cache_data
def load_species_names():
    return pd.read_csv(art("species_names.csv"), index_col=0)

# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_data():
    np.random.seed(42)
    kingdoms = ["bct","vrl","pln","vrt","inv","mam","phg","rod","pri","arc","plm"]
    counts   = [2919,2831,2523,2077,1345,572,220,215,180,126,18]
    n = sum(counts)

    codon_cols = [f"codon_{i:02d}" for i in range(64)]
    rscu_cols  = [f"rscu_{i:02d}" for i in range(61)]
    X = pd.DataFrame(np.random.rand(n, 126), columns=codon_cols + ["GC3"] + rscu_cols)
    y = pd.Series(np.repeat(kingdoms, counts), name="Kingdom")
    split = int(0.8 * n)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pca_train = pd.DataFrame(np.random.randn(len(X_train), 50),
                              columns=[f"PC{i+1}" for i in range(50)])
    pca_test  = pd.DataFrame(np.random.randn(len(X_test),  50),
                              columns=[f"PC{i+1}" for i in range(50)])
    for k, kdf in pca_train.assign(k=y_train.values).groupby("k"):
        offset = hash(k) % 7
        pca_train.loc[kdf.index, "PC1"] += offset * 0.8
        pca_train.loc[kdf.index, "PC2"] += (offset % 3) * 1.2
    for k, kdf in pca_test.assign(k=y_test.values).groupby("k"):
        offset = hash(k) % 7
        pca_test.loc[kdf.index, "PC1"] += offset * 0.8
        pca_test.loc[kdf.index, "PC2"] += (offset % 3) * 1.2

    bmu_coords = pd.DataFrame({
        "bmu_row": np.random.randint(0, 20, n),
        "bmu_col": np.random.randint(0, 20, n),
    })
    fi = pd.DataFrame({
        "feature":    [f"PC{i+1}" for i in range(50)],
        "importance": np.sort(np.random.exponential(0.02, 50))[::-1],
    })
    comp = pd.DataFrame({
        "model":         ["LogisticRegression","RandomForest","SVM_RBF","CodonPrint"],
        "bal_acc_mean":  [0.7037, 0.6085, 0.7981, 0.7045],
        "bal_acc_std":   [0.0250, 0.0169, 0.0127, 0.0236],
        "f1_macro_mean": [0.5491, 0.6652, 0.7874, 0.7463],
        "f1_macro_std":  [0.0079, 0.0148, 0.0099, 0.0227],
        "f1_wtd_mean":   [0.7108, 0.8474, 0.8991, 0.8934],
        "f1_wtd_std":    [0.0037, 0.0111, 0.0060, 0.0070],
        "test_bal_acc":  [0.7037, 0.6070, 0.8256, 0.7242],
        "test_f1_macro": [0.5491, 0.6676, 0.8090, 0.7682],
        "test_f1_wtd":   [0.7108, 0.8602, 0.9104, None],
    })
    vrl_mis = pd.DataFrame({
        "species":      ["Tomato spotted wilt virus","Cricket paralysis virus",
                         "Tobacco necrosis virus D","Peanut yellow spot virus","Kyzylagach virus"],
        "true_kingdom": ["vrl"]*5,
        "pred_kingdom": ["pln","inv","pln","pln","inv"],
        "confidence":   [0.949, 0.919, 0.852, 0.762, 0.727],
    })
    comp_rf = pd.DataFrame({
        "Model":             ["Baseline RF","CodonPrint (ours)"],
        "Balanced_Accuracy": [0.6070, 0.7242],
        "F1_Macro":          [0.6676, 0.7682],
    })
    perclass_rf = pd.DataFrame({
        "Kingdom":     ["arc","bct","inv","mam","phg","plm","pln","pri","rod","vrl","vrt"],
        "Baseline_F1": [0.6842,0.9222,0.7862,0.7358,0.6154,0.0000,0.9135,0.4400,0.4828,0.8809,0.8822],
        "Custom_F1":   [0.7907,0.9425,0.8206,0.8638,0.7500,0.0000,0.9361,0.7302,0.7568,0.9238,0.9351],
        "Change":      [0.1065,0.0204,0.0344,0.1281,0.1346,0.0000,0.0226,0.2902,0.2740,0.0429,0.0528],
    })
    host_mimicry = pd.DataFrame({
        "true_label":          ["vrl"]*10,
        "predicted":           ["vrl","pln","vrl","vrl","vrl","pln","vrl","vrl","vrl","vrl"],
        "host_mimicry_score":  [0.636,0.573,0.555,0.552,0.542,0.549,0.542,0.539,0.538,0.536],
        "top_euk_host":        ["vrt","pln","pln","inv","inv","pln","pln","pln","pln","pln"],
        "correctly_classified":[True,False,True,True,True,False,True,True,True,True],
    })
    ablation = pd.DataFrame({
        "Feature Set":   ["Raw 64 codons only","Raw + GC3","RSCU only","RSCU + GC3","Raw + RSCU + GC3 (ours)"],
        "Balanced Acc":  [0.7206, 0.7167, 0.6682, 0.6772, 0.7242],
        "F1 Macro":      [0.7610, 0.7598, 0.7203, 0.7292, 0.7682],
    })
    weight_just = pd.DataFrame({
        "Kingdom":          ["vrl","arc","plm","phg","mam/rod/pri","bct/pln/vrt/inv"],
        "Multiplier":       [2.0, 1.8, 3.0, 1.8, 1.3, 1.0],
        "Biological Reason":["Viruses mimic host codon usage — hardest boundary",
                             "Archaea/Bacteria share common ancestor — fuzzy boundary",
                             "Only 14 training samples — maximum attention needed",
                             "Phages co-evolve with bacterial hosts",
                             "Mammalian sub-groups are rare and biologically similar",
                             "Abundant and well-separated — no boost needed"],
    })
    return (X_train, X_test, y_train, y_test, X, y,
            pca_train, pca_test, bmu_coords, fi, comp, vrl_mis,
            comp_rf, perclass_rf, host_mimicry, ablation, weight_just)

def in_demo_mode():
    return not os.path.exists(art("X_train.csv"))

# ─────────────────────────────────────────────────────────────────────────────
# ICONS (inline SVG, Lucide-style — used instead of emojis)
# ─────────────────────────────────────────────────────────────────────────────

_ICONS = {
    "dna": '<path d="m10 16 1.5 1.5"/><path d="m14 8-1.5-1.5"/><path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="m16.5 10.5 1 1"/><path d="m17 6-2.891-2.891"/><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="m20 9 .891.891"/><path d="M3.109 14.109 4 15"/><path d="m6.5 12.5 1 1"/><path d="m7 18 2.891 2.891"/><path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993"/>',
    "check-circle": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "x-circle": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "bug": '<path d="m8 2 1.88 1.88"/><path d="M14.12 3.88 16 2"/><path d="M9 7.13v-1a3.003 3.003 0 1 1 6 0v1"/><path d="M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v3c0 3.3-2.7 6-6 6"/><path d="M12 20v-9"/><path d="M6.53 9C4.6 8.8 3 7.1 3 5"/><path d="M6 13H2"/><path d="M3 21c0-2.1 1.7-3.9 3.8-4"/><path d="M20.97 5c0 2.1-1.6 3.8-3.5 4"/><path d="M22 13h-4"/><path d="M17.2 17c2.1.1 3.8 1.9 3.8 4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "zap": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
}

def icon(name, size=18, color="currentColor", stroke=1.8):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="flex-shrink:0">{_ICONS[name]}</svg>')

def metric_card(ic, label, value, sub="", color=None, small=False):
    vstyle = f"color:{color};" if color else ""
    cls = "val sm" if small else "val"
    sub_html = f"<div class='lbl'>{sub}</div>" if sub else ""
    return (f"<div class='metric-card'><div class='mc-head'>{icon(ic, 16)}<span>{label}</span></div>"
            f"<div class='{cls}' style='{vstyle}'>{value}</div>{sub_html}</div>")

def heading(ic, text):
    return f"<div class='h-icon'>{icon(ic, 18)}<span>{text}</span></div>"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Codon Usage Fingerprint",
    page_icon=":material/genetics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Unbounded:wght@300;400;500&display=swap');
  :root { --accent:#4ade80; --accent-2:#86efac; --muted:#7a9a86; --text:#e6f0e9; --border:rgba(255,255,255,.08);
          --glass:linear-gradient(160deg,rgba(74,222,128,.10),rgba(74,222,128,.02)); }
  html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif; color:var(--text); }
  .stApp { background:radial-gradient(900px 520px at 0% 0%,rgba(13,148,136,.30),transparent 60%),
                      radial-gradient(700px 420px at 100% 0%,rgba(34,197,94,.10),transparent 60%), #09100c; }
  [data-testid="stHeader"] { background:transparent; }
  .block-container { padding-top:2rem; max-width:1400px; }
  hr { border-color:var(--border) !important; }

  /* sidebar */
  [data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1a12 0%,#08120d 100%); border-right:1px solid var(--border); }
  .brand { display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .4rem; }
  .brand .logo { width:34px; height:34px; border-radius:9px; background:linear-gradient(135deg,#4ade80,#22c55e); color:#06210f;
                 display:flex; align-items:center; justify-content:center; box-shadow:0 0 18px rgba(74,222,128,.35); }
  .brand span { font-weight:500; font-size:1.02rem; color:var(--text); }
  .side-label { font-size:.72rem; color:var(--muted); letter-spacing:.06em; text-transform:uppercase; display:flex; align-items:center; gap:.4rem; margin:.4rem 0 .2rem; }

  /* header */
  .app-header { display:flex; align-items:center; gap:.9rem; background:var(--glass); border:1px solid var(--border);
                border-radius:16px; padding:1rem 1.4rem; margin-bottom:1.5rem; backdrop-filter:blur(8px); }
  .app-header .logo { width:38px; height:38px; border-radius:10px; background:linear-gradient(135deg,#4ade80,#22c55e); color:#06210f;
                      display:flex; align-items:center; justify-content:center; box-shadow:0 0 18px rgba(74,222,128,.35); }
  .app-header h1 { font-family:'Inter',sans-serif; font-weight:500; font-size:1.25rem; color:var(--text); margin:0; padding:0; letter-spacing:-.01em; }
  .app-header h1 .crumb { color:var(--muted); font-weight:400; }
  .app-header h1 .sep { color:var(--muted); margin:0 .5rem; }
  .app-header p { font-size:.78rem; color:var(--muted); margin:.15rem 0 0; font-weight:300; }

  /* cards */
  .metric-card { background:var(--glass); border:1px solid var(--border); border-radius:14px; padding:1rem 1.2rem; min-width:140px; flex:1; backdrop-filter:blur(8px); }
  .metric-card .mc-head { display:flex; align-items:center; gap:.55rem; color:var(--muted); font-size:.8rem; margin-bottom:.7rem; }
  .metric-card .val { font-family:'Unbounded',sans-serif; font-weight:400; font-size:1.9rem; color:#f1f8f3; line-height:1.1; letter-spacing:-.02em; }
  .metric-card .val.sm { font-size:1.1rem; padding:.35rem 0; }
  .metric-card .lbl { font-size:.72rem; color:var(--muted); margin-top:.45rem; }
  [data-testid="stMetric"] { background:var(--glass); border:1px solid var(--border); border-radius:14px; padding:.9rem 1rem; }
  [data-testid="stMetricLabel"] { color:var(--muted); font-size:.74rem; }
  [data-testid="stMetricValue"] { font-family:'Unbounded',sans-serif; font-weight:400; font-size:1.05rem; color:#f1f8f3; }
  [data-testid="stMetricDelta"] { color:var(--accent) !important; font-size:.72rem; }
  [data-testid="stMetricDelta"] svg { display:none; }

  .section-label { font-size:.7rem; color:var(--accent); letter-spacing:.1em; text-transform:uppercase; font-weight:500; margin-bottom:.6rem; }
  .h-icon { display:flex; align-items:center; gap:.55rem; font-weight:600; font-size:1.05rem; margin:.4rem 0 .8rem; color:var(--text); }
  .h-icon svg { color:var(--accent); }

  /* info bars */
  .demo-banner, .contribution-banner { display:flex; align-items:flex-start; gap:.75rem; background:rgba(255,255,255,.03);
        border:1px solid var(--border); border-radius:12px; padding:.7rem 1rem; font-size:.84rem; margin-bottom:1.2rem; color:#cfe0d5; }
  .ib { width:28px; height:28px; border-radius:8px; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
  .demo-banner .ib { background:rgba(224,92,92,.18); color:#f87171; }
  .contribution-banner .ib { background:rgba(74,222,128,.18); color:var(--accent); }

  h2 { font-size:1.05rem !important; font-weight:600 !important; }
  h3 { font-size:.92rem !important; font-weight:500 !important; color:var(--muted) !important; }
  code { background:rgba(74,222,128,.10) !important; color:var(--accent-2) !important; border-radius:6px; }

  /* tabs — active = green with underline, like the 1D/1W/1M selector */
  [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--border); }
  button[data-baseweb="tab"] { background:transparent; color:var(--muted); font-size:.86rem; padding:.6rem .9rem; }
  button[data-baseweb="tab"][aria-selected="true"] { color:var(--accent); }
  button[data-baseweb="tab"]:hover { color:var(--accent-2); }
  [data-baseweb="tab-highlight"] { background:var(--accent) !important; height:2px; }
  [data-baseweb="tab-border"] { background:transparent !important; }

  /* inputs / tables */
  [data-testid="stDataFrame"], .stDataFrame { border:1px solid var(--border) !important; border-radius:12px; overflow:hidden; }
  div[data-baseweb="select"] > div { background:rgba(255,255,255,.03) !important; border:1px solid var(--border) !important; border-radius:10px !important; font-size:.82rem !important; }
  [data-testid="stSelectbox"] span { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:100%; }
  .stButton > button { background:linear-gradient(180deg,#86efac,#22c55e); color:#06210f; border:0; border-radius:10px; font-weight:600; }

  /* badges */
  .badge { display:inline-block; background:rgba(255,255,255,.06); color:var(--accent-2); font-size:.7rem; padding:.18rem .6rem; border-radius:999px; margin-left:.4rem; border:1px solid var(--border); }
  .badge-green { display:inline-block; background:#4ade80; color:#06210f; font-weight:600; font-size:.7rem; padding:.18rem .6rem; border-radius:999px; margin-left:.4rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

DEMO = in_demo_mode()

if DEMO:
    (X_train, X_test, y_train, y_test, X_all, y_all,
     pca_train_coords, pca_test_coords, bmu_coords,
     feat_imp, comp_table, vrl_mis,
     comp_rf, perclass_rf, host_mimicry,
     ablation_df, weight_just_df) = build_demo_data()
    pca_model = pca_scaler = svm_model = custom_rf = None
    # Demo species names — generic placeholders
    kingdoms = ["bct","vrl","pln","vrt","inv","mam","phg","rod","pri","arc","plm"]
    counts   = [2919,2831,2523,2077,1345,572,220,215,180,126,18]
    demo_kings = []
    for k, c in zip(kingdoms, counts):
        demo_kings.extend([k]*c)
    species_df = pd.DataFrame({
        "SpeciesName": [f"Demo organism {i}" for i in range(sum(counts))],
        "Kingdom":     demo_kings,
    })
else:
    X_train, X_test, y_train, y_test, X_all, y_all = load_data()
    pca_model, pca_scaler, svm_model, custom_rf = load_models()
    pca_train_coords, pca_test_coords = load_coords()
    bmu_coords   = load_som()
    feat_imp     = load_feature_importance()
    comp_table   = load_classifier_comparison()
    vrl_mis      = load_vrl_misclassified()
    comp_rf, perclass_rf, host_mimicry = load_custom_rf_results()
    ablation_df, weight_just_df = load_evaluation_artifacts()
    species_df   = load_species_names()

pca_all_coords = pd.concat([pca_train_coords, pca_test_coords], ignore_index=True)
y_all_reset = pd.Series(list(y_all.astype(str)), name="Kingdom")

if len(y_all_reset) == 0:
    st.error("y_all is empty — check CSV files.")
    st.stop()

def model_metrics(model_name):
    aliases = {
        "CodonPrint": {"CodonPrint", "CustomRF_BiologyWeighted"},
    }
    accepted_names = aliases.get(model_name, {model_name})
    rows = comp_table.loc[comp_table["model"].isin(accepted_names)]
    if rows.empty:
        raise ValueError(f"Missing evaluation metrics for {model_name}.")
    return rows.iloc[0]

baseline_rf_metrics = model_metrics("RandomForest")
codonprint_metrics = model_metrics("CodonPrint")
svm_metrics = model_metrics("SVM_RBF")
improved_kingdom_count = int((perclass_rf["Change"] > 0).sum())
correct_hms = host_mimicry.loc[
    host_mimicry["correctly_classified"], "host_mimicry_score"
]
misclassified_hms = host_mimicry.loc[
    ~host_mimicry["correctly_classified"], "host_mimicry_score"
]
hms_ratio = (
    misclassified_hms.mean() / correct_hms.mean()
    if not misclassified_hms.empty and not correct_hms.empty and correct_hms.mean() > 0
    else None
)
if custom_rf is not None:
    _, _, runtime_roc_curves = evaluate_custom_rf(custom_rf, X_test, y_test)
    mean_roc_auc = float(np.mean([curve[3] for curve in runtime_roc_curves]))
else:
    mean_roc_auc = None

rscu_cols = [c for c in X_all.columns if c.lower().startswith("rscu")]
raw_cols  = [c for c in X_all.columns if c.lower().startswith("codon") or
             (len(c)==3 and c[0] in "ACGU" and c[1] in "ACGU" and c[2] in "ACGU")]
if not raw_cols:
    raw_cols = [c for c in X_all.columns if c not in rscu_cols and c != "GC3"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown(f"<div class='brand'><div class='logo'>{icon('dna', 20, stroke=2.2)}</div><span>Codon Usage Fingerprint</span></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown(f"<div class='side-label'>{icon('filter', 13)} Filters</div>", unsafe_allow_html=True)

all_kingdoms_sorted = sorted(y_all_reset.unique())
kingdom_filter = st.sidebar.selectbox(
    "Filter by kingdom",
    ["All"] + [f"{k}  ({KINGDOM_LABELS.get(k,k)})" for k in all_kingdoms_sorted],
)
sel_kingdom_code = None if kingdom_filter == "All" else kingdom_filter.split()[0]

available_indices = [int(i) for i in range(len(y_all_reset))]
if sel_kingdom_code:
    available_indices = [int(i) for i in range(len(y_all_reset))
                         if y_all_reset.iloc[i] == sel_kingdom_code]
if not available_indices:
    available_indices = list(range(len(y_all_reset)))

def _fmt_organism(i):
    i = int(i)
    name = species_df["SpeciesName"].iloc[i] if i < len(species_df) else f"#{i}"
    king = y_all_reset.iloc[i]
    return f"{name}  [{king}]"

organism_idx = int(st.sidebar.selectbox(
    "Select organism",
    available_indices,
    format_func=_fmt_organism,
))
st.sidebar.markdown("---")
st.sidebar.markdown("<span style='font-size:0.72rem;color:#7a9a86'>CSE4889 · The Outliers · Oct 2026</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <div class="logo">""" + icon("dna", 22, stroke=2.2) + """</div>
  <div><h1><span class="crumb">Dashboard</span><span class="sep">/</span>Codon Usage Fingerprint</h1></div>
</div>
""", unsafe_allow_html=True)

if DEMO:
    st.markdown("""<div class="demo-banner"><div class="ib">""" + icon("info", 16) + """</div><div><strong>Demo mode</strong> — running on synthetic data. Drop real artifact files into the same folder as <code>app.py</code> and restart.</div></div>""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    ":material/biotech: Organism Explorer",
    ":material/scatter_plot: PCA Space",
    ":material/hub: SOM Map",
    ":material/leaderboard: Classifier Comparison",
    ":material/coronavirus: Virus / Host Mimicry",
    ":material/star: CodonPrint (Our Model)",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — ORGANISM EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    org_row   = X_all.reset_index(drop=True).iloc[organism_idx]
    true_king = str(y_all_reset.iloc[organism_idx])
    true_label = KINGDOM_LABELS.get(true_king, true_king)

    if not DEMO and svm_model is not None and pca_model is not None:
        rscu_gc3_vals = org_row[rscu_cols + ["GC3"]].values.reshape(1, -1)
        scaled        = pca_scaler.transform(rscu_gc3_vals)
        pca_vec       = pca_model.transform(scaled)
        proba         = svm_model.predict_proba(pca_vec)[0]
        classes       = svm_model.classes_
        pred_idx      = np.argmax(proba)
        pred_king     = classes[pred_idx]
        confidence    = proba[pred_idx]
    else:
        np.random.seed(int(organism_idx))
        fake_proba = np.random.dirichlet(np.ones(11) * 2)
        fake_proba[list(KINGDOM_LABELS.keys()).index(true_king)] *= 3
        fake_proba /= fake_proba.sum()
        pred_idx   = np.argmax(fake_proba)
        pred_king  = list(KINGDOM_LABELS.keys())[pred_idx]
        confidence = fake_proba[pred_idx]
        classes    = list(KINGDOM_LABELS.keys())
        proba      = fake_proba

    pred_label = KINGDOM_LABELS.get(pred_king, pred_king)
    correct    = pred_king == true_king

    species_name = species_df["SpeciesName"].iloc[organism_idx] if organism_idx < len(species_df) else f"#{organism_idx}"
    st.markdown(f"<div class='section-label'>{species_name} · TRUE KINGDOM: {true_king.upper()} ({true_label})</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("target", "SVM prediction", pred_label, small=True), unsafe_allow_html=True)
    with c2:
        color   = "#4ade80" if correct else "#E05C5C"
        verdict = "Correct" if correct else "Misclassified"
        st.markdown(metric_card("check-circle" if correct else "x-circle", "Verdict", verdict,
                                f"vs true: {true_label}", color=color, small=True), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("gauge", "Model confidence", f"{confidence:.1%}"), unsafe_allow_html=True)
    with c4:
        gc3 = org_row["GC3"] if "GC3" in org_row.index else 0.0
        st.markdown(metric_card("dna", "GC3 content", f"{gc3:.3f}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("**RSCU Codon Fingerprint**")
        st.caption("Relative Synonymous Codon Usage — values > 1 mean this codon is used more than expected by chance")
        if rscu_cols:
            rscu_vals = org_row[rscu_cols]
            fig, ax = plt.subplots(figsize=(9, 3), facecolor="none")
            ax.set_facecolor("none")
            ax.bar(range(len(rscu_vals)), rscu_vals.values,
                   color=KINGDOM_COLORS.get(true_king, "#4E9AF1"), width=0.7, alpha=0.85)
            ax.axhline(1.0, color="#7a9a86", linewidth=0.8, linestyle="--", label="Expected (RSCU=1)")
            ax.set_xticks(range(0, len(rscu_vals), 5))
            ax.set_xticklabels([rscu_cols[i].replace("rscu_","").replace("RSCU_","")
                                 for i in range(0, len(rscu_vals), 5)], color="#7a9a86", fontsize=7)
            ax.tick_params(axis="y", colors="#7a9a86", labelsize=8)
            ax.set_ylabel("RSCU", color="#7a9a86", fontsize=8)
            ax.spines[:].set_color("#1f3a2a")
            ax.legend(fontsize=7, labelcolor="#7a9a86", framealpha=0)
            fig.tight_layout(pad=0.5)
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("No RSCU columns found.")

    with col_r:
        st.markdown("**Kingdom Probabilities**")
        st.caption("Predicted probability across all 11 kingdoms")
        prob_df = pd.DataFrame({"Kingdom":[KINGDOM_LABELS.get(c,c) for c in classes],"P":proba}).sort_values("P", ascending=True)
        fig2, ax2 = plt.subplots(figsize=(4.5, 4), facecolor="none")
        ax2.set_facecolor("none")
        bar_colors = ["#4ade80" if KINGDOM_LABELS.get(true_king,"") == k else
                      "#E05C5C" if k == pred_label and not correct else "#2f7a4d"
                      for k in prob_df["Kingdom"]]
        ax2.barh(prob_df["Kingdom"], prob_df["P"], color=bar_colors, height=0.6)
        ax2.set_xlabel("Probability", color="#7a9a86", fontsize=8)
        ax2.tick_params(colors="#7a9a86", labelsize=8)
        ax2.spines[:].set_color("#1f3a2a")
        ax2.set_xlim(0, 1)
        fig2.tight_layout(pad=0.5)
        st.pyplot(fig2); plt.close(fig2)

    st.markdown("---")
    st.markdown("**Top Feature Importances (RF baseline)**")
    top_fi = feat_imp.sort_values("importance", ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(8, 2.2), facecolor="none")
    ax3.set_facecolor("none")
    ax3.bar(top_fi["feature"], top_fi["importance"], color="#2f7a4d", alpha=0.9)
    ax3.tick_params(colors="#7a9a86", labelsize=8)
    ax3.set_ylabel("Importance", color="#7a9a86", fontsize=8)
    ax3.spines[:].set_color("#1f3a2a")
    fig3.tight_layout(pad=0.5)
    st.pyplot(fig3); plt.close(fig3)
    st.caption("PC2 and PC3 carry more discriminative signal than PC1 (which captures GC-content variance, not kingdom boundaries).")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PCA SPACE
# ═════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("**Kingdom Structure in PCA Space**")
    st.caption("62-feature RSCU+GC3 matrix → StandardScaler → PCA. PC1 = 45.1% var, PC2 = 17.3% var (62.4% combined).")

    pc_options = [c for c in pca_all_coords.columns if c.startswith("PC") and c != "Kingdom"]
    col_x, col_y = st.columns(2)
    pc_x = col_x.selectbox("X axis", pc_options, index=0)
    pc_y = col_y.selectbox("Y axis", pc_options, index=1)

    fig4, ax4 = plt.subplots(figsize=(9, 6), facecolor="none")
    ax4.set_facecolor("none")
    for king in sorted(y_all_reset.unique()):
        kmask    = (y_all_reset == king).values
        coords_k = pca_all_coords.reset_index(drop=True)[kmask]
        if pc_x in coords_k.columns and pc_y in coords_k.columns:
            ax4.scatter(coords_k[pc_x], coords_k[pc_y],
                        c=KINGDOM_COLORS.get(king, "#888"),
                        label=f"{king} ({KINGDOM_LABELS.get(king,king)})",
                        alpha=0.35, s=8, linewidths=0)
    sel_coords = pca_all_coords.reset_index(drop=True).iloc[int(organism_idx)]
    if pc_x in sel_coords.index and pc_y in sel_coords.index:
        ax4.scatter(sel_coords[pc_x], sel_coords[pc_y],
                    c="white", s=120, zorder=10, marker="*",
                    edgecolors=KINGDOM_COLORS.get(true_king, "white"), linewidths=1.5,
                    label=f"Selected #{organism_idx} ({true_king})")
    ax4.set_xlabel(pc_x, color="#7a9a86")
    ax4.set_ylabel(pc_y, color="#7a9a86")
    ax4.tick_params(colors="#7a9a86", labelsize=8)
    ax4.spines[:].set_color("#1f3a2a")
    ax4.legend(fontsize=7, labelcolor="#d7e6dc", framealpha=0.15,
               facecolor="#10231a", ncol=2, markerscale=1)
    fig4.tight_layout(pad=0.6)
    st.pyplot(fig4); plt.close(fig4)

    st.markdown("""
**Key findings:**
- **Vertebrates (vrt)** form a visually distinct cluster elevated on PC2 — clean separation across all methods.
- **Bacteria (bct) and Viruses (vrl)** overlap heavily in the high-PC1 region — first visual confirmation of the host-mimicry hypothesis.
- Kingdom is *not* the natural clustering structure (ARI = 0.15); GC-content and host environment dominate the geometry.
""")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SOM MAP
# ═════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("**Self-Organizing Map — Kingdom Distribution**")
    st.caption("20×20 MiniSOM grid trained on PCA-20 space (10,000 iterations). Each cell coloured by majority kingdom of organisms that map there.")

    grid_size    = 20
    kingdom_list = sorted(y_all_reset.unique())
    bmu_reset    = bmu_coords.reset_index(drop=True)
    grid_kingdom = np.full((grid_size, grid_size), "", dtype=object)
    counts_grid  = np.zeros((grid_size, grid_size), dtype=int)

    _num_cols = [c for c in bmu_reset.columns if bmu_reset[c].dtype in ['int64','float64','Int64','Float64']]
    if not _num_cols:
        _num_cols = [c for c in bmu_reset.columns if c != 'Kingdom']
    bmu_col_r = next((c for c in ["BMU_y","bmu_row","row"] if c in bmu_reset.columns), _num_cols[0] if _num_cols else bmu_reset.columns[0])
    bmu_col_c = next((c for c in ["BMU_x","bmu_col","col"] if c in bmu_reset.columns), _num_cols[1] if len(_num_cols)>1 else bmu_reset.columns[1])

    for idx, row in bmu_reset.iterrows():
        try:
            r, c = int(float(row["BMU_y"])), int(float(row["BMU_x"]))
        except (ValueError, TypeError, KeyError):
            try:
                r, c = int(float(row[bmu_col_r])), int(float(row[bmu_col_c]))
            except Exception:
                continue
        if 0 <= r < grid_size and 0 <= c < grid_size:
            king = y_all_reset.iloc[idx] if idx < len(y_all_reset) else "bct"
            counts_grid[r, c] += 1
            grid_kingdom[r, c] = king

    color_mat = np.zeros((grid_size, grid_size, 4))
    for r in range(grid_size):
        for c in range(grid_size):
            k = grid_kingdom[r, c]
            if k and k in KINGDOM_COLORS:
                rgba = list(mcolors.to_rgba(KINGDOM_COLORS[k]))
                rgba[3] = min(0.9, 0.2 + counts_grid[r, c] * 0.04)
                color_mat[r, c] = rgba
            else:
                color_mat[r, c] = mcolors.to_rgba("#10231a")

    fig5, ax5 = plt.subplots(figsize=(8, 7), facecolor="none")
    ax5.set_facecolor("none")
    ax5.imshow(color_mat, origin="lower", aspect="auto")

    if int(organism_idx) < len(bmu_reset):
        try:
            sel_r = int(float(bmu_reset.iloc[int(organism_idx)]["BMU_y"]))
            sel_c = int(float(bmu_reset.iloc[int(organism_idx)]["BMU_x"]))
        except (KeyError, ValueError):
            sel_r = int(float(bmu_reset.iloc[int(organism_idx)][bmu_col_r]))
            sel_c = int(float(bmu_reset.iloc[int(organism_idx)][bmu_col_c]))
        ax5.scatter(sel_c, sel_r, s=200, c="white", marker="*", zorder=10,
                    edgecolors=KINGDOM_COLORS.get(true_king, "white"), linewidths=1.5)
        ax5.annotate(f" #{organism_idx}", (sel_c, sel_r), color="white", fontsize=7, va="center")

    ax5.set_xlabel("SOM column", color="#7a9a86")
    ax5.set_ylabel("SOM row",    color="#7a9a86")
    ax5.tick_params(colors="#7a9a86", labelsize=8)
    ax5.spines[:].set_color("#1f3a2a")
    patches = [mpatches.Patch(color=KINGDOM_COLORS[k], label=f"{k} ({KINGDOM_LABELS.get(k,k)})") for k in kingdom_list]
    ax5.legend(handles=patches, fontsize=7, labelcolor="#d7e6dc", framealpha=0.15,
               facecolor="#10231a", ncol=2, loc="lower right")
    fig5.tight_layout(pad=0.5)
    st.pyplot(fig5); plt.close(fig5)

    st.markdown("""
**Interpretation:**
- bct and vrl scatter across nearly the entire grid — no localized SOM territory for either.
- vrt maps to relatively few nodes (consistent with its tight PCA cluster).
- All three unsupervised methods (PCA, hierarchical, SOM) tell the same story: **kingdom is not the natural axis** of this data.
""")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLASSIFIER COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("**Model Comparison — 5-Fold Stratified CV + Test Set**")
    st.caption("Per-class F1 and balanced accuracy used throughout. Raw accuracy suppressed due to severe class imbalance (~160:1).")

    col_map = {
        'model':'Model',
        'bal_acc_mean':'Bal Acc (CV)','bal_acc_std':'± Std',
        'f1_macro_mean':'F1 Macro (CV)','f1_macro_std':'± Std.1',
        'f1_wtd_mean':'F1 Wtd (CV)','f1_wtd_std':'± Std.2',
        'test_bal_acc':'Bal Acc (Test)',
        'test_f1_macro':'F1 Macro (Test)',
        'test_f1_wtd':'F1 Wtd (Test)',
    }
    ct = comp_table.rename(columns=col_map)

    def style_comp(df):
        best_cols = ["Bal Acc (Test)","F1 Macro (Test)"]
        num_cols  = df.select_dtypes("float").columns
        styled    = df.style.format({c:"{:.4f}" for c in num_cols}, na_rep="—")
        for col in best_cols:
            if col in df.columns:
                max_val = df[col].max()
                styled  = styled.apply(
                    lambda s, col=col, mv=max_val: [
                        "background-color:rgba(74,222,128,.16);color:#86efac;" if v == mv else ""
                        for v in s], subset=[col])
        return styled

    if not ct.empty:
        st.dataframe(style_comp(ct), use_container_width=True, hide_index=True)

    custom_gain = codonprint_metrics["test_bal_acc"] - baseline_rf_metrics["test_bal_acc"]
    st.markdown(
        f"""
<span class='badge'>Best overall: SVM (RBF)</span> Test Bal Acc <b>{svm_metrics['test_bal_acc']:.4f}</b> · F1 Macro <b>{svm_metrics['test_f1_macro']:.4f}</b><br>
<span class='badge-green'>Our model: CodonPrint</span> CV Bal Acc <b>{codonprint_metrics['bal_acc_mean']:.4f} ± {codonprint_metrics['bal_acc_std']:.4f}</b> · Test Bal Acc <b>{codonprint_metrics['test_bal_acc']:.4f}</b> · <b>{custom_gain:+.1%}</b> over Baseline RF
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Why SVM wins on raw numbers:**")
        st.markdown("- RBF kernel carves non-linear boundaries in PCA space\n- SVM is a powerful general-purpose discriminator\n- Our CodonPrint was not designed to beat SVM — it was designed to be interpretable and biologically justified")
    with col_b:
        st.markdown("**What our CodonPrint proves:**")
        improved_classes = int((perclass_rf["Change"] > 0).sum())
        st.markdown(
            f"- {custom_gain:+.1%} test balanced-accuracy change over Baseline RF\n"
            f"- CV balanced-accuracy std ±{codonprint_metrics['bal_acc_std']:.4f}\n"
            f"- Per-class F1 improved for {improved_classes} of {len(perclass_rf)} kingdoms\n"
            "- Outputs Host-Mimicry Score — SVM does not provide this project diagnostic"
        )

    st.markdown("---")
    st.markdown("**Feature Importance (RF baseline) — Top 20 PCA Components**")
    st.caption("PC2 and PC3 ranked highest — discriminative signal lives in secondary variance directions, not the dominant GC-content axis (PC1).")
    top20 = feat_imp.sort_values("importance", ascending=False).head(20)
    fig6, ax6 = plt.subplots(figsize=(9, 2.8), facecolor="none")
    ax6.set_facecolor("none")
    ax6.bar(top20["feature"], top20["importance"], color="#2f7a4d", alpha=0.9)
    ax6.tick_params(colors="#7a9a86", labelsize=8, axis="x", rotation=45)
    ax6.tick_params(colors="#7a9a86", labelsize=8, axis="y")
    ax6.set_ylabel("Importance", color="#7a9a86", fontsize=8)
    ax6.spines[:].set_color("#1f3a2a")
    fig6.tight_layout(pad=0.5)
    st.pyplot(fig6); plt.close(fig6)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — VIRUS / HOST MIMICRY
# ═════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("**Headline Finding: Virus / Eukaryote Host-Mimicry**")
    virus_misclassified = vrl_mis.rename(columns={
        "y_pred_svm": "pred_kingdom",
        "confidence_wrong": "confidence",
        "SpeciesName": "species",
    })
    eukaryotic_kingdoms = {"inv", "mam", "pln", "pri", "rod", "vrt"}
    euk_errors = virus_misclassified[
        virus_misclassified["pred_kingdom"].isin(eukaryotic_kingdoms)
    ]
    wrong_count = len(euk_errors)
    highest_wrong_confidence = (
        float(euk_errors["confidence"].max()) if wrong_count else 0.0
    )
    leading_eukaryote_errors = euk_errors["pred_kingdom"].value_counts().head(2)
    leading_eukaryote_labels = " · ".join(leading_eukaryote_errors.index.tolist()) or "None"
    st.caption(
        f"{wrong_count} viruses in the held-out set were misclassified as eukaryotes "
        "by SVM. This pattern motivates the host-adaptation analysis; it does not "
        "by itself establish the biological cause of each error."
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(metric_card("bug", "Viruses misclassified as eukaryotes", str(wrong_count)), unsafe_allow_html=True)
    mc2.markdown(metric_card("alert", "Highest wrong-prediction confidence", f"{highest_wrong_confidence:.1%}"), unsafe_allow_html=True)
    mc3.markdown(metric_card("layers", "Top predicted eukaryote classes", leading_eukaryote_labels, small=True), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    vrl_mis = virus_misclassified
    display_cols = ["species","pred_kingdom","confidence"]
    avail_cols   = [c for c in display_cols if c in vrl_mis.columns]
    if avail_cols:
        vrl_display = vrl_mis[avail_cols].copy()
        if "pred_kingdom" in vrl_display.columns:
            vrl_display["pred_kingdom"] = vrl_display["pred_kingdom"].map(lambda x: f"{x} ({KINGDOM_LABELS.get(x,x)})")
        if "confidence" in vrl_display.columns:
            vrl_display["confidence"] = vrl_display["confidence"].apply(lambda x: f"{x:.1%}")
        vrl_display.columns = [c.replace("_"," ").title() for c in vrl_display.columns]
        st.dataframe(vrl_display, use_container_width=True, hide_index=True)

    if "species" in vrl_mis.columns and "confidence" in vrl_mis.columns:
        vrl_chart = vrl_mis.sort_values("confidence", ascending=False).head(5)
        st.markdown("**Top 5 Misclassified Viruses by Confidence**")
        fig7, ax7 = plt.subplots(figsize=(8, 2.8), facecolor="none")
        ax7.set_facecolor("none")
        conf_vals      = vrl_chart["confidence"].values
        species_labels = vrl_chart["species"].values
        bar_cols = [KINGDOM_COLORS.get(vrl_chart["pred_kingdom"].iloc[i], "#E05C5C")
                    if "pred_kingdom" in vrl_chart.columns else "#E05C5C"
                    for i in range(len(conf_vals))]
        ax7.barh(range(len(conf_vals)), conf_vals, color=bar_cols, height=0.55)
        ax7.set_yticks(range(len(species_labels)))
        ax7.set_yticklabels([s[:40] for s in species_labels], color="#d7e6dc", fontsize=8)
        ax7.set_xlabel("SVM confidence (wrong prediction)", color="#7a9a86", fontsize=8)
        ax7.set_xlim(0, 1.05)
        ax7.axvline(0.5, color="#7a9a86", linewidth=0.8, linestyle="--")
        ax7.tick_params(colors="#7a9a86", labelsize=8)
        ax7.spines[:].set_color("#1f3a2a")
        for i, v in enumerate(conf_vals):
            ax7.text(v + 0.01, i, f"{v:.1%}", va="center", color="#d7e6dc", fontsize=8)
        fig7.tight_layout(pad=0.5)
        st.pyplot(fig7); plt.close(fig7)

    st.markdown("---")
    st.markdown("""
**Biological interpretation:**
The misclassification pattern follows known virology:
- **Plant-infecting viruses** (Tomato spotted wilt, Tobacco necrosis, Peanut yellow spot) adapt their codon usage to match plant host tRNA pools.
- **Insect viruses** (Cricket paralysis virus, Kyzylagach virus) adapt to invertebrate translation machinery.

The model is not making errors — it is measuring host adaptation. Where it is confident and wrong, the biology agrees with the model.
""")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — CUSTOM RF (OUR CONTRIBUTION)
# ═════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("""<div class="contribution-banner"><div class="ib">""" + icon("star", 16) + """</div><div><strong>Our Contribution</strong> — CodonPrint — Biologically-Weighted Random Forest, designed and trained by us. Biology-informed class weights + richer feature space + novel Host-Mimicry Score output. Validated with 5-fold CV, confusion matrix, ROC curves, and ablation study.</div></div>""", unsafe_allow_html=True)

    # ── Quick Stats Banner ────────────────────────────────────────────────
    st.markdown(heading("zap", "At a Glance"), unsafe_allow_html=True)
    qs1, qs2, qs3, qs4, qs5, qs6 = st.columns(6)
    custom_gain = codonprint_metrics["test_bal_acc"] - baseline_rf_metrics["test_bal_acc"]
    custom_f1_gain = codonprint_metrics["test_f1_macro"] - baseline_rf_metrics["test_f1_macro"]
    qs1.metric("Balanced Acc", f"{codonprint_metrics['test_bal_acc']:.4f}", f"{custom_gain:+.1%} vs baseline")
    qs2.metric("F1 Macro", f"{codonprint_metrics['test_f1_macro']:.4f}", f"{custom_f1_gain:+.1%} vs baseline")
    qs3.metric("5-Fold CV", f"{codonprint_metrics['bal_acc_mean']:.4f}", f"±{codonprint_metrics['bal_acc_std']:.3f}")
    qs4.metric("Mean AUC", f"{mean_roc_auc:.3f}" if mean_roc_auc is not None else "n/a", "one-vs-rest")
    qs5.metric("Host-Mimicry", f"{hms_ratio:.2f}×" if hms_ratio is not None else "n/a", "misclassified vs correct")
    qs6.metric("Kingdoms Improved", f"{improved_kingdom_count}/{len(perclass_rf)}", "per-class F1")
    st.markdown("---")

    st.markdown("## CodonPrint — Biologically-Weighted Random Forest")
    st.caption("Base: scikit-learn RandomForestClassifier (300 trees, min_samples_leaf=2) · Custom: biology-informed weights + raw+RSCU+GC3 features (126 dim)")

    # ── Sub-tabs inside Tab 6 ──────────────────────────────────────────────
    sub = st.tabs([":material/tune: Design & Weights", ":material/bar_chart: Performance", ":material/science: Ablation Study", ":material/grid_view: Confusion Matrix", ":material/coronavirus: Host-Mimicry Score"])

    # ── Sub-tab A: Design & Weights ────────────────────────────────────────
    with sub[0]:
        st.markdown("### What We Customized")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""**1. Biology-Informed Class Weights**

Instead of equal weights, multipliers based on known biological difficulty:
- `vrl` ×2.0 — host-mimicry makes viruses hardest
- `arc` ×1.8 — archaea/bacteria boundary is fuzzy
- `plm` ×3.0 — only 14 training samples
- `phg` ×1.8 — phages mimic bacterial codon usage""")
        with c2:
            st.markdown("""**2. Richer Feature Space**

Baseline RF: 50 PCA components (compressed).

Our RF: all 126 raw features:
- 64 raw codon frequencies
- 1 GC3 content
- 61 RSCU values

Preserves biological meaning instead of abstracting it away.""")
        with c3:
            st.markdown("""**3. Host-Mimicry Score Output**

Novel metric we defined:

`HMS = P(inv)+P(mam)+P(pln)+P(pri)+P(rod)+P(vrt)`

For each virus: how much probability mass lands on eukaryote kingdoms? Higher = stronger host adaptation signal.""")

        st.markdown("---")
        st.markdown("### Model Architecture")
        st.caption("How data flows through CodonPrint — from raw input to three outputs. Compare with Baseline RF on the right.")

        def draw_arch():
            BG      = "none"
            CARD    = "#10231a"
            BLUE    = "#2f7a4d"
            LBLUE   = "#86efac"
            GREEN   = "#4ade80"
            DGREEN  = "#0D2B1A"
            ORANGE  = "#F5A623"
            MUTED   = "#7a9a86"
            DIM     = "#d7e6dc"
            GREY    = "#4a5568"
            LGREY   = "#7a9a86"

            fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
            ax.set_facecolor(BG)
            ax.set_xlim(0, 14); ax.set_ylim(0, 9)
            ax.axis("off")

            def box(x, y, w, h, fc, ec, lw=1.5):
                ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                    boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=lw, zorder=2))

            def txt(x, y, s, c, fs=8, fw="normal", ha="center"):
                ax.text(x, y, s, color=c, fontsize=fs, fontweight=fw,
                        ha=ha, va="center", zorder=3, fontfamily="monospace" if fw=="bold" else "sans-serif")

            def arrow(x1, y1, x2, y2, c=MUTED):
                ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4), zorder=3)

            # ── COLUMN TITLES ─────────────────────────────────────────────
            txt(3.1, 8.7, "CodonPrint  (Our Model)", GREEN, 12, "bold")
            txt(10.9, 8.7, "Baseline RF  (Before Us)", MUTED, 12, "bold")
            # divider
            ax.plot([7.0, 7.0], [0.2, 8.5], color="#1f3a2a", lw=2, ls="--", zorder=1)
            txt(7.0, 0.05, "vs", MUTED, 9)

            # ── LEFT: CODON RF ────────────────────────────────────────────

            # Box 1 — Input
            box(1.0, 7.0, 4.2, 1.2, CARD, BLUE, 1.8)
            txt(3.1, 7.75, "INPUT", LBLUE, 10, "bold")
            txt(3.1, 7.38, "126 features per organism", DIM, 8)
            txt(3.1, 7.08, "64 raw codons  +  61 RSCU values  +  GC3", MUTED, 7.5)

            arrow(3.1, 7.0, 3.1, 6.35, MUTED)

            # Box 2 — Bio Weights (GREEN highlight — our contribution)
            box(1.0, 5.15, 4.2, 1.1, DGREEN, GREEN, 2.2)
            txt(3.1, 5.9, "Biology-Informed Weights", GREEN, 10, "bold")
            txt(3.1, 5.6, "vrl ×2.0   arc ×1.8   plm ×3.0   phg ×1.8", DIM, 8)
            txt(3.1, 5.28, "Backed by 6 published biology papers", MUTED, 7.5)

            arrow(3.1, 5.15, 3.1, 4.5, MUTED)

            # Box 3 — 300 Trees
            box(1.0, 3.3, 4.2, 1.1, CARD, BLUE, 1.8)
            txt(3.1, 4.05, "300 Decision Trees", LBLUE, 10, "bold")
            txt(3.1, 3.75, "Each tree trained on random feature subset", DIM, 8)
            txt(3.1, 3.45, "min_samples_leaf=2  ·  random_state=42", MUTED, 7.5)

            arrow(3.1, 3.3, 3.1, 2.65, MUTED)

            # Box 4 — Vote
            box(1.0, 1.65, 4.2, 0.9, CARD, BLUE, 1.8)
            txt(3.1, 2.2, "Weighted Vote Aggregation", LBLUE, 10, "bold")
            txt(3.1, 1.85, "300 predictions combined → probability per kingdom", MUTED, 7.5)

            # split arrow into 3 outputs
            ax.plot([3.1, 3.1], [1.65, 1.35], color=MUTED, lw=1.4, zorder=3)
            ax.plot([1.4, 4.8], [1.35, 1.35], color=MUTED, lw=1.2, zorder=3)
            arrow(1.4, 1.35, 1.4, 1.05, GREEN)
            arrow(3.1, 1.35, 3.1, 1.05, BLUE)
            arrow(4.8, 1.35, 4.8, 1.05, ORANGE)

            # Output 1 — Kingdom
            box(0.55, 0.1, 1.75, 0.9, DGREEN, GREEN, 1.5)
            txt(1.43, 0.72, "Predicted Kingdom", GREEN, 8, "bold")
            txt(1.43, 0.48, "Hard label output", DIM, 7.5)
            txt(1.43, 0.27, '"vrl", "pln", "bct"', MUTED, 7)

            # Output 2 — Probabilities
            box(2.22, 0.1, 1.75, 0.9, "#0f2f20", BLUE, 1.5)
            txt(3.1, 0.72, "Probabilities", LBLUE, 8, "bold")
            txt(3.1, 0.48, "P(kingdom) for all 11", DIM, 7.5)
            txt(3.1, 0.27, "predict_proba()", MUTED, 7)

            # Output 3 — HMS (orange — novel)
            box(3.9, 0.1, 1.75, 0.9, "#2B1A00", ORANGE, 2.0)
            txt(4.78, 0.72, "Host-Mimicry Score", ORANGE, 8, "bold")
            txt(4.78, 0.48, "HMS = Σ P(euk kingdoms)", DIM, 7.5)
            txt(4.78, 0.27, "NOVEL OUTPUT", ORANGE, 7, "bold")

            # ── MIDDLE CALLOUTS ───────────────────────────────────────────
            for (cy, label1, label2) in [
                (7.55, "+76 features", "126 vs 50"),
                (5.68, "Bio weights", "vs equal"),
                (3.83, "300 trees", "vs 100"),
                (0.57, "3 outputs", "vs 1 output"),
            ]:
                box(6.35, cy-0.32, 1.28, 0.64, DGREEN, GREEN, 1.0)
                txt(7.0, cy+0.1,  label1, GREEN, 7.5, "bold")
                txt(7.0, cy-0.15, label2, GREEN, 7)

            # ── RIGHT: BASELINE RF ────────────────────────────────────────

            # Box 1 — Input
            box(7.8, 7.0, 4.2, 1.2, CARD, "#1f3a2a", 1.5)
            txt(9.9, 7.75, "INPUT", LGREY, 10, "bold")
            txt(9.9, 7.38, "50 PCA components per organism", GREY, 8)
            txt(9.9, 7.08, "Compressed — biological meaning lost", GREY, 7.5)

            arrow(9.9, 7.0, 9.9, 6.35, GREY)

            # Box 2 — Equal Weights
            box(7.8, 5.15, 4.2, 1.1, CARD, "#1f3a2a", 1.5)
            txt(9.9, 5.9, "Equal / Default Weights", LGREY, 10, "bold")
            txt(9.9, 5.6, "class_weight = balanced_subsample", GREY, 8)
            txt(9.9, 5.28, "Inverse-frequency class balancing", GREY, 7.5)

            arrow(9.9, 5.15, 9.9, 4.5, GREY)

            # Box 3 — 500 Trees
            box(7.8, 3.3, 4.2, 1.1, CARD, "#1f3a2a", 1.5)
            txt(9.9, 4.05, "500 Decision Trees", LGREY, 10, "bold")
            txt(9.9, 3.75, "n_estimators=500 · max_features=sqrt", GREY, 8)
            txt(9.9, 3.45, "random_state=42", GREY, 7.5)

            arrow(9.9, 3.3, 9.9, 2.65, GREY)

            # Box 4 — Simple Vote
            box(7.8, 1.65, 4.2, 0.9, CARD, "#1f3a2a", 1.5)
            txt(9.9, 2.2, "Simple Majority Vote", LGREY, 10, "bold")
            txt(9.9, 1.85, "500 predictions → most common label wins", GREY, 7.5)

            arrow(9.9, 1.65, 9.9, 1.05, GREY)

            # Output — label only
            box(8.4, 0.1, 3.0, 0.9, CARD, "#1f3a2a", 1.5)
            txt(9.9, 0.72, "Predicted Kingdom Only", LGREY, 8, "bold")
            txt(9.9, 0.48, "Single hard label output", GREY, 7.5)
            txt(9.9, 0.27, "No probabilities · No HMS · No biology", GREY, 7)

            fig.tight_layout(pad=0.3)
            return fig

        arch_fig = draw_arch()
        st.pyplot(arch_fig)
        plt.close(arch_fig)
        st.markdown("---")
        st.markdown("### Weight Justification Table")
        st.caption("Every multiplier is grounded in published biology — not arbitrary tuning.")
        st.dataframe(weight_just_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Class Weight Visualization (log scale)")
        class_counts = y_train.value_counts()
        weight_data = pd.DataFrame({
            "Kingdom": class_counts.index,
            "Weight": [
                len(y_train) / (len(class_counts) * class_counts[kingdom])
                * {"vrl": 2.0, "arc": 1.8, "plm": 3.0, "phg": 1.8,
                   "mam": 1.3, "rod": 1.3, "pri": 1.3}.get(kingdom, 1.0)
                for kingdom in class_counts.index
            ],
        })
        fig_w, ax_w = plt.subplots(figsize=(10, 3), facecolor="none")
        ax_w.set_facecolor("none")
        ax_w.bar(weight_data["Kingdom"], weight_data["Weight"],
                 color=[KINGDOM_COLORS.get(k,"#888") for k in weight_data["Kingdom"]], alpha=0.85)
        ax_w.set_ylabel("Class Weight (log scale)", color="#7a9a86", fontsize=9)
        ax_w.set_yscale("log")
        ax_w.tick_params(colors="#d7e6dc", labelsize=9)
        ax_w.yaxis.set_tick_params(colors="#7a9a86")
        ax_w.spines[:].set_color("#1f3a2a")
        fig_w.tight_layout(pad=0.5)
        st.pyplot(fig_w); plt.close(fig_w)
        plasmid_weight = weight_data.loc[weight_data["Kingdom"] == "plm", "Weight"]
        if not plasmid_weight.empty:
            st.caption(
                f"plm weight = {plasmid_weight.iloc[0]:.2f} — log scale keeps the smaller weights visible."
            )

    # ── Sub-tab B: Performance ─────────────────────────────────────────────
    with sub[1]:
        st.markdown("### Overall Performance: Baseline RF vs CodonPrint")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Baseline Bal Acc", f"{baseline_rf_metrics['test_bal_acc']:.4f}")
        m2.metric("CodonPrint Bal Acc", f"{codonprint_metrics['test_bal_acc']:.4f}", delta=f"{custom_gain:+.1%}")
        m3.metric("Baseline F1 Macro", f"{baseline_rf_metrics['test_f1_macro']:.4f}")
        m4.metric("CodonPrint F1 Macro", f"{codonprint_metrics['test_f1_macro']:.4f}", delta=f"{custom_f1_gain:+.1%}")
        m5.metric("CV Bal Acc (5-fold)", f"{codonprint_metrics['bal_acc_mean']:.4f} ± {codonprint_metrics['bal_acc_std']:.3f}")
        m6.metric("CV F1 Macro (5-fold)", f"{codonprint_metrics['f1_macro_mean']:.4f} ± {codonprint_metrics['f1_macro_std']:.3f}")

        st.caption("Cross-validation results summarize performance across the stratified folds.")

        st.markdown("---")
        st.markdown("### Per-Kingdom F1: Baseline RF vs CodonPrint")
        improved_perclass = perclass_rf.sort_values("Change", ascending=False)
        best_gains = ", ".join(
            f"{row.Kingdom} {row.Change:+.2f}"
            for row in improved_perclass.head(3).itertuples()
            if row.Change > 0
        ) or "no positive per-class F1 changes"
        st.caption(f"F1 improved in {(perclass_rf['Change'] > 0).sum()} of {len(perclass_rf)} kingdoms. Largest gains: {best_gains}.")

        fig_rf, ax_rf = plt.subplots(figsize=(11, 4), facecolor="none")
        ax_rf.set_facecolor("none")
        x     = np.arange(len(perclass_rf))
        width = 0.35
        ax_rf.bar(x - width/2, perclass_rf["Baseline_F1"], width, label="Baseline RF", color="#2f7a4d", alpha=0.85)
        ax_rf.bar(x + width/2, perclass_rf["Custom_F1"],   width, label="CodonPrint (ours)", color="#4ade80", alpha=0.85)
        for i, row in perclass_rf.iterrows():
            if abs(row["Change"]) > 0.01:
                ax_rf.text(i + width/2, row["Custom_F1"] + 0.01,
                           f"+{row['Change']:.2f}", ha="center", color="#4ade80", fontsize=7)
        ax_rf.set_xticks(x)
        ax_rf.set_xticklabels(perclass_rf["Kingdom"], color="#d7e6dc", fontsize=9)
        ax_rf.set_ylabel("F1 Score", color="#7a9a86", fontsize=9)
        ax_rf.set_ylim(0, 1.1)
        ax_rf.tick_params(colors="#7a9a86")
        ax_rf.spines[:].set_color("#1f3a2a")
        ax_rf.legend(facecolor="#10231a", labelcolor="#d7e6dc", fontsize=9)
        fig_rf.tight_layout(pad=0.5)
        st.pyplot(fig_rf); plt.close(fig_rf)

    # ── Sub-tab C: Ablation Study ──────────────────────────────────────────
    with sub[2]:
        st.markdown("### Ablation Study: Which Features Actually Help?")
        st.caption("We trained our CodonPrint (same weights, same hyperparameters) on 5 different feature sets to isolate the contribution of each feature engineering choice.")

        # Highlight best row
        def highlight_best_ablation(df):
            return df.style.apply(
                lambda row: ["background-color:rgba(74,222,128,.14);color:#4ade80;font-weight:bold" if row.name == df["Balanced Acc"].idxmax() else "" for _ in row],
                axis=1
            ).format({"Balanced Acc": "{:.4f}", "F1 Macro": "{:.4f}"})

        st.dataframe(highlight_best_ablation(ablation_df), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Bar chart
        fig_abl, ax_abl = plt.subplots(figsize=(9, 3.5), facecolor="none")
        ax_abl.set_facecolor("none")
        x_abl = np.arange(len(ablation_df))
        best_ablation_index = ablation_df["Balanced Acc"].idxmax()
        colors_abl = [
            "#4ade80" if i == best_ablation_index else "#2f7a4d"
            for i in ablation_df.index
        ]
        ax_abl.bar(x_abl, ablation_df["Balanced Acc"], color=colors_abl, alpha=0.85, width=0.6)
        ax_abl.set_xticks(x_abl)
        ax_abl.set_xticklabels(ablation_df["Feature Set"], color="#d7e6dc", fontsize=8, rotation=12)
        ax_abl.set_ylabel("Balanced Accuracy", color="#7a9a86", fontsize=9)
        ax_abl.set_ylim(
            max(0, float(ablation_df["Balanced Acc"].min()) - 0.04),
            min(1, float(ablation_df["Balanced Acc"].max()) + 0.04),
        )
        ax_abl.tick_params(colors="#7a9a86")
        ax_abl.spines[:].set_color("#1f3a2a")
        for i, v in enumerate(ablation_df["Balanced Acc"]):
            ax_abl.text(i, v + 0.001, f"{v:.4f}", ha="center", color="#d7e6dc", fontsize=8)
        fig_abl.tight_layout(pad=0.5)
        st.pyplot(fig_abl); plt.close(fig_abl)

        raw_score = ablation_df.loc[
            ablation_df["Feature Set"] == "Raw 64 codons only", "Balanced Acc"
        ].iloc[0]
        rscu_score = ablation_df.loc[
            ablation_df["Feature Set"] == "RSCU only", "Balanced Acc"
        ].iloc[0]
        full_score = ablation_df.loc[
            ablation_df["Feature Set"] == "Raw + RSCU + GC3 (ours)", "Balanced Acc"
        ].iloc[0]
        best_feature_set = ablation_df.loc[
            ablation_df["Balanced Acc"].idxmax(), "Feature Set"
        ]
        st.markdown(f"""
**What this tells us:**
- Raw codons alone achieved {raw_score:.4f} balanced accuracy.
- RSCU alone achieved {rscu_score:.4f}; compare feature sets rather than assuming normalization improves results.
- The complete feature set achieved {full_score:.4f}; the best balanced accuracy in this run came from **{best_feature_set}**.

These values are recalculated when `generate_artifacts.py` is run.
""")

    # ── Sub-tab D: Confusion Matrix ────────────────────────────────────────
    with sub[3]:
        st.markdown("### Confusion Matrix — CodonPrint (Test Set)")
        st.caption("Rows = true kingdom, Columns = predicted kingdom. Diagonal = correct predictions.")

        if custom_rf is not None:
            classes, matrix, roc_curves = evaluate_custom_rf(
                custom_rf, X_test, y_test
            )
            class_labels = [
                f"{kingdom} ({KINGDOM_LABELS.get(str(kingdom), kingdom)})"
                for kingdom in classes
            ]
            fig_cm, ax_cm = plt.subplots(
                figsize=(10, 8), facecolor="none"
            )
            ax_cm.set_facecolor("none")
            image = ax_cm.imshow(matrix, cmap="Greens")
            fig_cm.colorbar(image, ax=ax_cm, fraction=0.046, pad=0.04)
            ax_cm.set_xticks(np.arange(len(classes)), labels=class_labels, rotation=45, ha="right")
            ax_cm.set_yticks(np.arange(len(classes)), labels=class_labels)
            ax_cm.set_xlabel("Predicted kingdom", color="#7a9a86")
            ax_cm.set_ylabel("True kingdom", color="#7a9a86")
            ax_cm.tick_params(colors="#d7e6dc", labelsize=8)
            ax_cm.spines[:].set_color("#1f3a2a")
            threshold = matrix.max() / 2 if matrix.size else 0
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    ax_cm.text(
                        column, row, f"{matrix[row, column]:,}",
                        ha="center", va="center",
                        color="white" if matrix[row, column] > threshold else "#10231a",
                        fontsize=8,
                    )
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            st.markdown("### One-vs-Rest ROC Curves — CodonPrint")
            st.caption("Curves are calculated from the saved model's test-set class probabilities.")
            fig_roc, ax_roc = plt.subplots(figsize=(9, 6), facecolor="none")
            ax_roc.set_facecolor("none")
            for kingdom, false_positive_rate, true_positive_rate, score in roc_curves:
                ax_roc.plot(
                    false_positive_rate,
                    true_positive_rate,
                    color=KINGDOM_COLORS.get(kingdom, "#86efac"),
                    linewidth=1.7,
                    label=f"{kingdom} ({score:.3f})",
                )
            ax_roc.plot([0, 1], [0, 1], "--", color="#7a9a86", linewidth=1)
            ax_roc.set(
                xlim=(0, 1), ylim=(0, 1.02),
                xlabel="False positive rate", ylabel="True positive rate",
            )
            ax_roc.set_title("One-vs-Rest ROC by Kingdom", color="#d7e6dc")
            ax_roc.tick_params(colors="#7a9a86")
            ax_roc.spines[:].set_color("#1f3a2a")
            ax_roc.legend(
                loc="lower right", fontsize=8, ncol=2,
                facecolor="#10231a", labelcolor="#d7e6dc",
            )
            fig_roc.tight_layout()
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        else:
            st.info("Evaluation plots require the trained CodonPrint model and test data. They are not available in synthetic demo mode.")

        if custom_rf is not None:
            error_rows = []
            for actual_index, kingdom in enumerate(classes):
                for predicted_index, predicted_kingdom in enumerate(classes):
                    if actual_index != predicted_index and matrix[actual_index, predicted_index] > 0:
                        error_rows.append({
                            "True kingdom": kingdom,
                            "Predicted kingdom": predicted_kingdom,
                            "Count": int(matrix[actual_index, predicted_index]),
                        })
            frequent_errors = pd.DataFrame(error_rows)
            if not frequent_errors.empty:
                frequent_errors = frequent_errors.nlargest(8, "Count")
                st.markdown("**Most frequent test-set confusions**")
                st.dataframe(frequent_errors, use_container_width=True, hide_index=True)
            else:
                st.success("No off-diagonal errors in this test set.")

    # ── Sub-tab E: Host-Mimicry Score ──────────────────────────────────────
    with sub[4]:
        st.markdown("### Host-Mimicry Score — Novel Output")
        st.caption("For each virus in the test set: sum of probability mass assigned to all eukaryote kingdoms. No prior paper on this dataset computed this metric.")

        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("Virus samples tested", f"{len(host_mimicry)}")
        hm2.metric("Correctly classified", f"{int(host_mimicry['correctly_classified'].sum())}")
        hm3.metric(
            "Mean HMS (correct)",
            f"{correct_hms.mean():.4f}" if not correct_hms.empty else "n/a",
        )
        hm4.metric(
            "Mean HMS (misclassified)",
            f"{misclassified_hms.mean():.4f}" if not misclassified_hms.empty else "n/a",
            delta=f"{misclassified_hms.mean() - correct_hms.mean():+.4f}"
            if not misclassified_hms.empty and not correct_hms.empty else None,
            delta_color="inverse",
        )

        st.markdown("""
**Interpretation:** Compare the mean Host-Mimicry Score for correctly and incorrectly classified viruses above.

The score summarizes probability mass assigned to eukaryotic kingdoms. It is a model diagnostic, not confirmed host information or proof of evolutionary causation.
""")

        st.markdown("**Top 10 Strongest Host-Mimics in Test Set**")
        top_mimics = host_mimicry.nlargest(10, "host_mimicry_score").reset_index(drop=True)
        display_hm = top_mimics.copy()
        display_hm["host_mimicry_score"] = display_hm["host_mimicry_score"].round(4)
        display_hm["predicted"]          = display_hm["predicted"].map(lambda x: f"{x} ({KINGDOM_LABELS.get(x,x)})")
        display_hm["top_euk_host"]       = display_hm["top_euk_host"].map(lambda x: f"{x} ({KINGDOM_LABELS.get(x,x)})")
        display_hm["correctly_classified"]= display_hm["correctly_classified"].map(lambda x: "Yes" if x else "No (host-mimic)")
        display_hm.columns = ["True Label","Predicted","Host-Mimicry Score","Top Euk Host","Classified Correctly"]
        st.dataframe(display_hm, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Honest Limitation")
        plasmid_rows = perclass_rf.loc[perclass_rf["Kingdom"] == "plm", "Custom_F1"]
        plasmid_f1 = float(plasmid_rows.iloc[0]) if not plasmid_rows.empty else None
        plasmid_count = int((y_train == "plm").sum())
        plasmid_weight = (
            len(y_train) / (len(y_train.unique()) * plasmid_count) * 3.0
            if plasmid_count else None
        )
        plasmid_f1_text = f"{plasmid_f1:.4f}" if plasmid_f1 is not None else "not available"
        plasmid_weight_text = f"{plasmid_weight:.2f}" if plasmid_weight is not None else "not available"
        st.markdown(f"""
**Plasmid test F1:** {plasmid_f1_text}  
**Plasmid training records:** {plasmid_count}  
**Configured class weight:** {plasmid_weight_text}

The plasmid class is extremely small, so its score is sensitive to a few test examples. Class weighting cannot replace representative training data. Treat this class-specific result cautiously and consider collecting more plasmid records before drawing strong conclusions.
""")
