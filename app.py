"""
Activity recognition from wearable sensors - Streamlit demo
Models: 1D-CNN (+augmentation), CNN-LSTM, LSTM trained on PAMAP2 (subject-wise split)

Files needed next to this script (created by the save cell in the notebook):
    meta.json, demo_windows.npz, cnn1d_aug.keras, cnn_lstm.keras, lstm.keras
Optional: put config.toml in a folder named .streamlit for the matching theme.
Run:  streamlit run app.py
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import decimate

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="Activity recognition from wearable sensors",
                   page_icon=None, layout="wide", initial_sidebar_state="collapsed")

FS = 50              # model sampling rate (Hz)
WIN, STEP = 128, 64  # 2.56 s windows, 50% overlap

MODEL_FILES = {
    "1D-CNN": "cnn1d_aug.keras",
    "CNN-LSTM": "cnn_lstm.keras",
    "LSTM": "lstm.keras",
}

# Palette: one colour per body location, used everywhere that location appears
INK, MUTED, HAIR, PAPER = "#1B2A38", "#5E6E7C", "#CBD4D0", "#F2F4F3"
HAND, CHEST, ANKLE = "#2B59C3", "#0F8B7D", "#B8325F"
GOOD, BAD = "#0F7A6B", "#B8325F"

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&display=swap');
html, body, .stApp {{ font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; }}
/* keep Streamlit's icon font intact */
.stApp :where(p, span, div, label, li, a, button, input, textarea, summary, th, td, h1, h2, h3, h4):not([data-testid="stIconMaterial"]) {{ font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif; }}
.stApp code, .stApp pre, .stApp pre * {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
header[data-testid="stHeader"], #MainMenu, footer {{ display: none; }}
.block-container {{ max-width: 1060px; padding-top: 2.6rem; padding-bottom: 4rem; }}

.page-title {{ font-stretch: 125%; font-weight: 700; font-size: 2.5rem; line-height: 1.08;
               letter-spacing: -0.015em; margin: 0 0 .7rem; color: {INK}; }}
.page-lede {{ max-width: 38rem; font-size: 1.05rem; line-height: 1.5; color: {MUTED}; margin: 0 0 1.6rem; }}

button[data-baseweb="tab"] {{ font-stretch: 112%; font-weight: 600; font-size: 1rem; }}

.readout {{ padding: .2rem 0 1.1rem; }}
.readout .small {{ font-size: .88rem; color: {MUTED}; margin-bottom: .15rem; }}
.readout .value {{ font-stretch: 125%; font-weight: 700; font-size: 2.35rem; line-height: 1.05;
                   letter-spacing: -0.01em; margin-bottom: .55rem; }}
.readout .note {{ font-size: .98rem; line-height: 1.45; }}
.ok {{ color: {GOOD}; font-weight: 600; }}
.off {{ color: {BAD}; font-weight: 600; }}

.p {{ display: grid; grid-template-columns: 8.6rem 1fr 2.8rem; align-items: center;
      gap: .6rem; padding: .22rem 0; font-size: .93rem; }}
.p .bar {{ height: 6px; background: #DDE4E0; }}
.p .bar i {{ display: block; height: 100%; background: #8FA0AD; }}
.p.first .bar i {{ background: {INK}; }}
.p.first .l {{ font-weight: 600; }}
.p .v {{ text-align: right; font-variant-numeric: tabular-nums; color: {MUTED}; }}

table.rows {{ width: 100%; border-collapse: collapse; margin: .4rem 0 1.4rem; }}
table.rows th {{ font-weight: 500; font-size: .86rem; color: {MUTED}; text-align: left;
                 padding: .35rem .8rem .35rem 0; border-bottom: 1.5px solid {INK}; }}
table.rows td {{ padding: .6rem .8rem .6rem 0; border-bottom: 1px solid {HAIR}; font-size: .98rem; }}
table.rows .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.rows .best td {{ font-weight: 700; }}
.meter {{ display: inline-block; height: 6px; background: {INK}; vertical-align: middle; margin-right: .5rem; }}

.section {{ font-stretch: 112%; font-weight: 650; font-size: 1.15rem; margin: 1.8rem 0 .3rem; }}
.hint {{ font-size: .93rem; color: {MUTED}; line-height: 1.5; max-width: 40rem; }}
.colophon {{ margin-top: 3rem; padding-top: .9rem; border-top: 1px solid {HAIR};
             font-size: .85rem; color: {MUTED}; max-width: 44rem; line-height: 1.5; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": HAIR,
                     "text.color": INK, "axes.labelcolor": MUTED})


# ----------------------------------------------------------------------------
# Custom layer used during training (must match the notebook definition)
# ----------------------------------------------------------------------------
class RandomScale(keras.layers.Layer):
    """Random per-channel amplitude scaling, active only during training."""

    def __init__(self, sigma=0.15, **kw):
        super().__init__(**kw)
        self.sigma = sigma

    def call(self, x, training=None):
        if training:
            s = tf.random.normal((tf.shape(x)[0], 1, tf.shape(x)[2]), 1.0, self.sigma)
            return x * s
        return x

    def get_config(self):
        return {**super().get_config(), "sigma": self.sigma}


# ----------------------------------------------------------------------------
# Loading and inference
# ----------------------------------------------------------------------------
@st.cache_resource
def load_meta():
    with open("meta.json") as f:
        m = json.load(f)
    m["mean"] = np.array(m["mean"], dtype=np.float32)
    m["scale"] = np.array(m["scale"], dtype=np.float32)
    return m


@st.cache_resource
def load_models():
    out = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            out[name] = keras.models.load_model(
                path, custom_objects={"RandomScale": RandomScale}, compile=False)
    return out


@st.cache_data
def load_demo():
    d = np.load("demo_windows.npz")
    return d["X"], d["y"]


def predict_proba(model, W, meta):
    """W: raw (unscaled) windows (n, 128, 18) -> class probabilities (n, 12)"""
    Ws = ((W - meta["mean"]) / meta["scale"]).astype("float32")
    return model.predict(Ws, verbose=0)


def summary_text(model):
    lines = []
    model.summary(print_fn=lambda s, **k: lines.append(s))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#E6ECE9", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=MUTED, length=0)


def plot_window(w):
    """Acceleration magnitude at each body location, labelled directly."""
    t = np.arange(w.shape[0]) / FS
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 4.3), sharex=True)
    for ax, (name, sl, col) in zip(axes, [("Hand", slice(0, 3), HAND),
                                          ("Chest", slice(6, 9), CHEST),
                                          ("Ankle", slice(12, 15), ANKLE)]):
        ax.plot(t, np.linalg.norm(w[:, sl], axis=1), color=col, lw=1.7)
        style_axis(ax)
        ax.set_ylabel(name, color=col, rotation=0, ha="right", va="center",
                      fontsize=11, fontweight="bold", labelpad=8)
        ax.margins(x=0)
    axes[-1].set_xlabel("seconds", fontsize=9)
    axes[0].set_title("Acceleration magnitude at each sensor, m/s²", loc="left",
                      fontsize=9.5, color=MUTED, pad=8)
    fig.patch.set_alpha(0)
    fig.tight_layout(h_pad=0.6)
    return fig


def plot_timeline(times, pred_ids, conf, classes, total_s):
    shown = sorted(set(pred_ids.tolist()))
    row = {c: i for i, c in enumerate(shown)}
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(shown) + 1.2))
    off = 0.5 * (WIN - STEP) / FS
    for ts, pid, c in zip(times, pred_ids, conf):
        ax.hlines(row[pid], ts + off, ts + off + STEP / FS, color=INK,
                  lw=11, alpha=0.2 + 0.8 * float(c), capstyle="butt")
    style_axis(ax)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([classes[i] for i in shown], fontsize=10, color=INK)
    ax.set_ylim(-0.6, len(shown) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, total_s)
    ax.set_xlabel("seconds into the recording", fontsize=9)
    fig.patch.set_alpha(0)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Small HTML helpers
# ----------------------------------------------------------------------------
def html(s):
    st.markdown(s.replace("\n", ""), unsafe_allow_html=True)


def table(headers, rows, num_cols=(), best_row=None):
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(headers))
    body = ""
    for r_i, r in enumerate(rows):
        cls = ' class="best"' if r_i == best_row else ""
        tds = "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r))
        body += f"<tr{cls}>{tds}</tr>"
    html(f'<table class="rows"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>')


# ----------------------------------------------------------------------------
# Start-up checks
# ----------------------------------------------------------------------------
try:
    meta = load_meta()
    Xdemo, ydemo = load_demo()
except FileNotFoundError:
    st.error("meta.json or demo_windows.npz is missing. Run the save cell in the notebook "
             "and copy the downloaded files into the same folder as app.py.")
    st.stop()

with st.spinner("Loading models"):
    models = load_models()
if not models:
    st.error("No model files found. Copy " + ", ".join(MODEL_FILES.values())
             + " into the same folder as app.py.")
    st.stop()

classes, channels = meta["classes"], meta["channels"]
names = list(models.keys())

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
html('<div class="page-title">Activity recognition from wearable sensors</div>')
html('<div class="page-lede">Reads 2.56 seconds of motion from sensors on the hand, chest and ankle '
     'and names the activity. The sample windows come from a person the models never saw during training.</div>')

tab1, tab2, tab3 = st.tabs(["Sample windows", "Your recording", "Results"])

# ----------------------------- Sample windows ------------------------------
with tab1:
    c1, c2, c3 = st.columns([2, 2, 2])
    true_name = c1.selectbox("Activity in the recording", classes, index=classes.index("walking")
                             if "walking" in classes else 0)
    true_id = classes.index(true_name)
    idxs = np.where(ydemo == true_id)[0]
    k = c2.slider("Sample", 1, max(len(idxs), 2), 1) if len(idxs) > 1 else 1
    primary = c3.selectbox("Model", names)
    w = Xdemo[idxs[k - 1]]

    probas = {nm: predict_proba(m, w[None], meta)[0] for nm, m in models.items()}
    p = probas[primary]
    pred = int(p.argmax())

    left, right = st.columns([5, 4], gap="large")
    with left:
        st.pyplot(plot_window(w), width="stretch")
    with right:
        if pred == true_id:
            verdict = '<span class="ok">Matches the recording.</span>'
        else:
            verdict = f'<span class="off">Differs from the recording, which was {true_name}.</span>'
        html(f'<div class="readout"><div class="small">{primary} says</div>'
             f'<div class="value">{classes[pred].capitalize()}</div>'
             f'<div class="note">{p[pred]:.0%} confident. {verdict}</div></div>')
        top = np.argsort(p)[::-1][:5]
        html("".join(
            f'<div class="p{" first" if j == 0 else ""}"><span class="l">{classes[i]}</span>'
            f'<span class="bar"><i style="width:{p[i] * 100:.0f}%"></i></span>'
            f'<span class="v">{p[i]:.0%}</span></div>' for j, i in enumerate(top)))

    html('<div class="section">All models on this window</div>')
    rows = []
    for nm, pp in probas.items():
        pi = int(pp.argmax())
        status = f'<span class="ok">Matches</span>' if pi == true_id else '<span class="off">Differs</span>'
        rows.append([nm, classes[pi], f'<span class="meter" style="width:{pp.max() * 70:.0f}px"></span>{pp.max():.0%}', status])
    table(["Model", "Predicted activity", "Confidence", "Against the recording"], rows)

# ----------------------------- Your recording ------------------------------
with tab2:
    html('<div class="hint">Upload a CSV with 18 columns: hand, chest and ankle, each with accelerometer '
         'X, Y, Z then gyroscope X, Y, Z. Accelerometer in m/s², gyroscope in rad/s. '
         'If the column names do not match, the first 18 numeric columns are used in that order.</div>')
    with st.expander("Column names the app expects"):
        st.code(", ".join(channels))

    pick = [classes.index(c) for c in ["sitting", "walking", "running", "lying"] if c in classes]
    chunks = [Xdemo[i] for cid in pick for i in np.where(ydemo == cid)[0][:2]]
    sample_df = pd.DataFrame(np.concatenate(chunks), columns=channels)
    st.download_button("Download an example recording (sitting, walking, running, lying)",
                       sample_df.to_csv(index=False), "example_recording.csv", "text/csv")

    c1, c2 = st.columns(2)
    rate = c1.radio("Sampling rate of your file", [50, 100], horizontal=True,
                    help="The models run at 50 Hz. A 100 Hz file is downsampled automatically.")
    use_model = c2.selectbox("Model for this recording", names)
    up = st.file_uploader("Recording (CSV)", type="csv")

    if up is None:
        html('<div class="hint">Choose a file to see which activity the model reads in each 2.56 second window.</div>')
    else:
        raw = pd.read_csv(up)
        if all(c in raw.columns for c in channels):
            data = raw[channels].to_numpy(dtype=float)
        else:
            num = raw.select_dtypes("number")
            if num.shape[1] < 18:
                st.error(f"The file has {num.shape[1]} numeric columns. It needs 18.")
                st.stop()
            data = num.iloc[:, :18].to_numpy(dtype=float)
        data = data[~np.isnan(data).any(axis=1)]
        if rate == 100:
            data = decimate(data, 2, axis=0, zero_phase=True)
        if len(data) < WIN:
            st.error(f"The recording is too short. It needs at least {WIN} samples at 50 Hz "
                     f"({WIN / FS:.2f} s) and has {len(data)}.")
            st.stop()

        starts = list(range(0, len(data) - WIN + 1, STEP))
        W = np.stack([data[s:s + WIN] for s in starts])
        P = predict_proba(models[use_model], W, meta)
        pred_ids, conf = P.argmax(1), P.max(1)
        times = np.array(starts) / FS
        vote = np.bincount(pred_ids, minlength=len(classes)).argmax()

        html(f'<div class="readout"><div class="small">Most common activity in {len(data) / FS:.0f} seconds of data</div>'
             f'<div class="value">{classes[vote].capitalize()}</div>'
             f'<div class="note">{len(starts)} windows read by {use_model}. '
             f'Darker bars in the chart mean higher confidence.</div></div>')
        st.pyplot(plot_timeline(times, pred_ids, conf, classes, len(data) / FS), width="stretch")

        share = pd.Series([classes[i] for i in pred_ids]).value_counts(normalize=True)
        table(["Activity", "Share of windows"], [[a, f"{v:.0%}"] for a, v in share.items()], num_cols=(1,))

# ----------------------------- Results -------------------------------------
with tab3:
    html('<div class="section" style="margin-top:.6rem">Test on one unseen person</div>')
    html('<div class="hint">Subject 5 was held out completely. Accuracy is the share of windows labelled '
         'correctly. Macro-F1 averages the per-activity F1 score, so rare activities count as much as common ones.</div>')
    mt = meta.get("metrics", {})
    if mt:
        order = sorted(mt.items(), key=lambda kv: -kv[1]["acc"])
        table(["Model", "Accuracy", "Macro-F1"],
              [[k, f'{v["acc"]:.3f}', f'{v["f1"]:.3f}'] for k, v in order], num_cols=(1, 2), best_row=0)

    loso = meta.get("loso")
    if loso:
        html('<div class="section">Tested on each of the eight people in turn</div>')
        html('<div class="hint">Each person is held out once and the models train on the other seven. '
             'A large spread means the model works well for some people and poorly for others.</div>')
        table(["Model", "Mean accuracy", "Spread (std)"],
              [[k, f'{v.get("mean acc", float("nan")):.3f}', f'{v.get("std", float("nan")):.3f}'] for k, v in loso.items()],
              num_cols=(1, 2))

    html('<div class="section">What the numbers say</div>')
    html('<div class="hint">With only six or seven people to learn from, the networks partly memorise how '
         'individuals move and wear the sensors, so accuracy falls on a new person. Random scaling and noise '
         'during training reduced this for the 1D-CNN. The Random Forest, which works from summary statistics '
         'of each window, held up best across people. Sitting and standing, and vacuum cleaning and ironing, '
         'are the activities every model confuses.</div>')

    html('<div class="section">How each network is built</div>')
    for nm, mdl in models.items():
        with st.expander(f"{nm}, {mdl.count_params():,} parameters"):
            st.code(summary_text(mdl))

    html('<div class="section">What goes in and what comes out</div>')
    html(f'<div class="hint">Input is one window of 128 readings at 50 Hz from 18 channels. Output is one of '
         f'{len(classes)} activities: {", ".join(classes)}.</div>')

html('<div class="colophon">Trained on PAMAP2, where the sensors sit on the dominant wrist, the chest and the '
     'ankle. The models will not carry over to a phone in a pocket or any other placement, and the app is not '
     'for medical use.</div>')