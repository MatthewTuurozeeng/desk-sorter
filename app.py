"""
Desk Sorter — CS111 Assignment 2, bonus part.

A small Streamlit app that loads the image model I trained in Google
Teachable Machine (Pen / Phone / Water bottle) and predicts what's in a
new photo.

Model files go in the  model/  folder next to this file:
    model/keras_model.h5
    model/labels.txt
(Teachable Machine -> Export Model -> Tensorflow -> Keras -> Download my model)

Run with:   streamlit run app.py
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"


def find_model_folder() -> Path:
    """Use model/ itself, or a folder inside it (unzipping usually makes
    model/converted_keras/), whichever has keras_model.h5."""
    if (MODEL_DIR / "keras_model.h5").exists():
        return MODEL_DIR
    for sub in sorted(MODEL_DIR.glob("*/keras_model.h5")):
        return sub.parent
    return MODEL_DIR


MODEL_FOLDER = find_model_folder()
MODEL_PATH = MODEL_FOLDER / "keras_model.h5"
LABELS_PATH = MODEL_FOLDER / "labels.txt"

IMG_SIZE = (224, 224)  # what Teachable Machine image models expect

# One muted colour per class, like coloured pencils on a desk.
# Any extra classes fall back to the later colours.
CLASS_COLOURS = ["#2E4A6B", "#B4532A", "#4F7A5A", "#8A6A2F", "#6B4C6E"]


# ----------------------------------------------------------------------------
# Page setup + styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Desk Sorter · CS111", page_icon="✏️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, li {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
h1, h2, h3, .serif { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.01em; }

.block-container { padding-top: 3.6rem; max-width: 1150px; }
[data-testid="stMarkdownContainer"] h4, [data-testid="stMarkdownContainer"] h5 {
    font-family: 'Fraunces', Georgia, serif !important; }
[data-testid="stCode"] pre, [data-testid="stCodeBlock"] pre, .stCode pre {
    background: #FBF8F2 !important; border: 1px solid #DDD3C0; border-radius: 6px; }

/* faint ruled-notebook lines on the page */
[data-testid="stAppViewContainer"] {
    background-image: repeating-linear-gradient(
        to bottom, transparent 0, transparent 31px, rgba(46,74,107,0.06) 32px);
}

.kicker {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.12em; color: #8A7F6E;
}
.title { font-family: 'Fraunces', Georgia, serif; font-size: 2.9rem; font-weight: 700;
         line-height: 1.05; color: #23272A; margin: 0.2rem 0 0.4rem 0; }
.title span { color: #B4532A; }
.lede { color: #4A4F52; font-size: 1.05rem; max-width: 46rem; }

.card {
    background: #FBF8F2; border: 1px solid #DDD3C0; border-radius: 6px;
    padding: 1.3rem 1.4rem; box-shadow: 3px 3px 0 #E2D8C4;
}
.verdict-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
                 text-transform: uppercase; letter-spacing: 0.1em; color: #8A7F6E; }
.verdict { font-family: 'Fraunces', Georgia, serif; font-size: 2.4rem; font-weight: 700;
           line-height: 1.1; margin: 0.15rem 0 0.2rem 0; }
.verdict-conf { font-family: 'IBM Plex Mono', monospace; font-size: 1rem; color: #4A4F52; }

.unsure {
    margin-top: 0.9rem; padding: 0.6rem 0.8rem; border-left: 3px solid #C99A2E;
    background: #F6EBD0; color: #5B4614; font-size: 0.92rem;
}

.bar-row { margin: 0.55rem 0; }
.bar-head { display: flex; justify-content: space-between; font-size: 0.92rem; margin-bottom: 3px; }
.bar-head .pct { font-family: 'IBM Plex Mono', monospace; }
.bar-track { height: 12px; background: #E9E1D2; border-radius: 2px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 2px; }

.empty {
    border: 1.5px dashed #C9BDA6; border-radius: 6px; padding: 2.2rem 1.2rem;
    text-align: center; color: #8A7F6E;
}

.stat { font-family: 'Fraunces', Georgia, serif; font-size: 2.6rem; font-weight: 700; line-height: 1; }
.stat-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #8A7F6E; }

code, pre { font-family: 'IBM Plex Mono', monospace !important; }
p code, li code { color: #9A4522 !important; background: #EDE3D1 !important; }
[data-testid="stDecoration"] { display: none; }

[data-testid="stSidebar"] { border-right: 1px solid #DDD3C0; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def read_labels(path: Path) -> list[str]:
    """labels.txt from Teachable Machine looks like '0 Pen', '1 Phone', ..."""
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        first, _, rest = line.partition(" ")
        labels.append(rest.strip() if first.isdigit() and rest else line)
    return labels


@st.cache_resource(show_spinner="Loading the model…")
def load_model(model_path: str):
    # Teachable Machine saves its .h5 with an older Keras. tf_keras (Keras 2)
    # loads it cleanly if installed; otherwise use whatever TF ships with.
    try:
        import tf_keras as keras  # type: ignore
    except ImportError:
        from tensorflow import keras  # type: ignore

    # Newer Keras versions choke on a 'groups' argument stored in the file.
    class DepthwiseConv2DCompat(keras.layers.DepthwiseConv2D):
        def __init__(self, *args, groups=None, **kwargs):
            super().__init__(*args, **kwargs)

    return keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"DepthwiseConv2D": DepthwiseConv2DCompat},
    )


def prepare(image: Image.Image) -> np.ndarray:
    """Same preprocessing as Teachable Machine's own Keras sample code."""
    image = ImageOps.exif_transpose(image).convert("RGB")  # fix phone rotation
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # scale pixels to [-1, 1]
    return arr[np.newaxis, ...]


def predict(model, image: Image.Image) -> np.ndarray:
    return model.predict(prepare(image), verbose=0)[0]


# ----------------------------------------------------------------------------
# Little UI helpers
# ----------------------------------------------------------------------------
def colour_for(i: int) -> str:
    return CLASS_COLOURS[i % len(CLASS_COLOURS)]


def render_result(probs: np.ndarray, labels: list[str], threshold: float) -> None:
    top = int(np.argmax(probs))
    conf = float(probs[top])
    order = np.argsort(probs)[::-1]

    bars = ""
    for i in order:
        pct = float(probs[i]) * 100
        bars += f"""
        <div class="bar-row">
          <div class="bar-head"><span>{html.escape(labels[i])}</span>
               <span class="pct">{pct:5.1f}%</span></div>
          <div class="bar-track"><div class="bar-fill"
               style="width:{pct:.1f}%; background:{colour_for(i)};
                      opacity:{1 if i == top else 0.45};"></div></div>
        </div>"""

    unsure = ""
    if conf < threshold:
        unsure = (
            f'<div class="unsure">Only {conf*100:.0f}% sure — below your '
            f"{threshold*100:.0f}% cut-off. In a real system this one should "
            "go to a person to check.</div>"
        )

    st.markdown(
        f"""
        <div class="card">
          <div class="verdict-label">Looks like</div>
          <div class="verdict" style="color:{colour_for(top)}">{html.escape(labels[top])}</div>
          <div class="verdict-conf">confidence {conf*100:.1f}%</div>
          <div style="margin-top:1.1rem">{bars}</div>
          {unsure}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_setup_help(problem: str) -> None:
    st.markdown(
        f"""
        <div class="card">
          <div class="verdict-label">Model not loaded</div>
          <div class="verdict" style="font-size:1.6rem; color:#B4532A">{html.escape(problem)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Where the model goes")
    st.markdown(
        "1. In Teachable Machine, click **Export Model**.\n"
        "2. Open the **Tensorflow** tab, pick **Keras**, then **Download my model**.\n"
        "3. Unzip `converted_keras.zip`. Copy both files into the `model` folder:"
    )
    st.code(
        f"""{APP_DIR.name}/
├── app.py
├── requirements.txt
└── model/
    ├── keras_model.h5   <- from the zip
    └── labels.txt       <- from the zip""",
        language=None,
    )
    st.caption(f"Full path: {MODEL_DIR}")
    st.markdown("4. Refresh this page.")


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="kicker">CS111 · Assignment 2 · Bonus</div>
    <div class="title">Desk <span>Sorter</span></div>
    <p class="lede">Show it a photo of something on your desk and the model I trained
    in Teachable Machine guesses what it is: a pen, a phone or a water bottle.</p>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ----------------------------------------------------------------------------
# Sidebar: model status + settings
# ----------------------------------------------------------------------------
model, labels, problem = None, [], None

if not MODEL_PATH.exists():
    problem = "keras_model.h5 is missing"
elif not LABELS_PATH.exists():
    problem = "labels.txt is missing"
else:
    labels = read_labels(LABELS_PATH)
    try:
        model = load_model(str(MODEL_PATH))
        n_out = int(model.output_shape[-1])
        if n_out != len(labels):
            problem = f"Model has {n_out} outputs but labels.txt has {len(labels)} names"
            model = None
    except Exception as exc:  # show the error instead of a blank page
        problem = "Couldn't open keras_model.h5"
        st.sidebar.error(f"{type(exc).__name__}: {exc}")

with st.sidebar:
    st.markdown("### The model")
    if model is not None:
        st.markdown(f"Loaded from `{MODEL_PATH.relative_to(APP_DIR)}`")
        for i, name in enumerate(labels):
            st.markdown(
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'background:{colour_for(i)};border-radius:2px;margin-right:8px"></span>{html.escape(name)}',
                unsafe_allow_html=True,
            )
    else:
        st.markdown("Not loaded yet. See the steps on the right.")

    st.markdown("---")
    st.markdown("### Settings")
    threshold = (
        st.slider(
            "Flag a guess as unsure below",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            format="%d%%",
            help="Low-confidence guesses get a warning. This is the 'have a human check it' idea from my reflection.",
        )
        / 100
    )

    st.markdown("---")
    st.caption(
        "Built with Streamlit + TensorFlow. Images are resized to 224×224 "
        "and scaled the same way Teachable Machine does."
    )

if model is None:
    show_setup_help(problem or "Something went wrong")
    st.stop()

if "log" not in st.session_state:
    st.session_state.log = []


def log_prediction(source: str, name: str, probs: np.ndarray) -> None:
    top = int(np.argmax(probs))
    key = (source, name, round(float(probs[top]), 4))
    # Streamlit reruns the script a lot; don't log the same photo twice in a row.
    if st.session_state.get("last_logged") == key:
        return
    st.session_state.last_logged = key
    st.session_state.log.append(
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Input": name,
            "From": source,
            "Prediction": labels[top],
            "Confidence": f"{probs[top]*100:.1f}%",
            "Correct?": None,
        }
    )


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_one, tab_test, tab_log = st.tabs(["Try a photo", "Run my test set", "Prediction log"])

# --- 1. single photo --------------------------------------------------------
with tab_one:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        source = st.radio("Input", ["Upload a photo", "Use the webcam"], horizontal=True,
                          label_visibility="collapsed")
        if source == "Upload a photo":
            file = st.file_uploader("Drop a photo here", type=["jpg", "jpeg", "png", "webp"])
            src_name = "upload"
        else:
            file = st.camera_input("Hold the object up to the camera")
            src_name = "webcam"

        image = None
        if file is not None:
            try:
                image = Image.open(file)
                image.load()
            except Exception:
                st.error("That file doesn't look like an image I can open.")
                image = None
        if image is not None and src_name == "upload":
            st.image(ImageOps.exif_transpose(image), use_container_width=True)

    with right:
        if image is None:
            st.markdown(
                '<div class="empty"><div class="serif" style="font-size:1.3rem;color:#4A4F52">'
                "Nothing to look at yet</div>Upload a photo or take one with the webcam."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            probs = predict(model, image)
            render_result(probs, labels, threshold)
            name = getattr(file, "name", "webcam photo") if src_name == "upload" else "webcam photo"
            log_prediction(src_name, name, probs)

# --- 2. whole test set ------------------------------------------------------
with tab_test:
    st.markdown(
        "Upload my test photos, one class at a time, so the app knows the right answer. "
        "It runs them all and works out accuracy the same way as my observation log: "
        "**correct ÷ total × 100**."
    )
    uploads: dict[str, list] = {}
    cols = st.columns(len(labels))
    for i, (col, name) in enumerate(zip(cols, labels)):
        with col:
            st.markdown(
                f'<div style="border-top:4px solid {colour_for(i)};padding-top:6px;'
                f'font-weight:600">{html.escape(name)}</div>',
                unsafe_allow_html=True,
            )
            uploads[name] = st.file_uploader(
                f"Test photos of {name}",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key=f"test_{i}",
                label_visibility="collapsed",
            ) or []

    total = sum(len(v) for v in uploads.values())
    if total == 0:
        st.markdown(
            '<div class="empty">No test photos yet. Drop each class\'s photos in its own box.</div>',
            unsafe_allow_html=True,
        )
    elif st.button(f"Run {total} test photos", type="primary"):
        rows = []
        progress = st.progress(0.0, text="Testing…")
        done = 0
        for true_name, files in uploads.items():
            for f in files:
                try:
                    probs = predict(model, Image.open(f))
                    pred = labels[int(np.argmax(probs))]
                    conf = float(np.max(probs))
                except Exception:
                    pred, conf = "(couldn't read)", 0.0
                rows.append({"File": f.name, "Actual": true_name, "Predicted": pred,
                             "Confidence": conf, "Right": pred == true_name})
                done += 1
                progress.progress(done / total, text=f"Testing… {done}/{total}")
        progress.empty()
        st.session_state.test_results = pd.DataFrame(rows)

    results = st.session_state.get("test_results")
    if results is not None and total > 0:
        correct = int(results["Right"].sum())
        n = len(results)
        acc = correct / n * 100

        st.write("")
        a, b, c = st.columns(3)
        a.markdown(f'<div class="card"><div class="stat">{correct}/{n}</div>'
                   f'<div class="stat-sub">correct</div></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="card"><div class="stat" style="color:#B4532A">{acc:.0f}%</div>'
                   f'<div class="stat-sub">accuracy</div></div>', unsafe_allow_html=True)
        wrong = results[~results["Right"]]
        if len(wrong):
            pair = (wrong["Actual"] + " → " + wrong["Predicted"]).value_counts()
            top_pair, times = pair.index[0], int(pair.iloc[0])
        else:
            top_pair, times = "none", 0
        c.markdown(f'<div class="card"><div class="stat" style="font-size:1.4rem;line-height:1.3">'
                   f'{html.escape(top_pair)}</div><div class="stat-sub">most common mistake'
                   f'{f" ({times}×)" if times else ""}</div></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown("##### Where it got confused")
        st.caption("Rows are what the photo really was, columns are what the model said.")
        confusion = pd.crosstab(results["Actual"], results["Predicted"]).reindex(
            index=labels, columns=labels, fill_value=0)
        st.dataframe(confusion, use_container_width=True)

        st.markdown("##### Every photo")
        show = results.copy()
        show["Confidence"] = (show["Confidence"] * 100).round(1).astype(str) + "%"
        show["Right"] = show["Right"].map({True: "✓", False: "✗"})
        st.dataframe(show, use_container_width=True, hide_index=True)

# --- 3. log -----------------------------------------------------------------
with tab_log:
    st.markdown(
        "Everything I've tried in the **Try a photo** tab this session. Tick whether "
        "each guess was right. The assignment asks for at least 3 inputs, including one it gets wrong."
    )
    if not st.session_state.log:
        st.markdown('<div class="empty">No predictions yet.</div>', unsafe_allow_html=True)
    else:
        edited = st.data_editor(
            pd.DataFrame(st.session_state.log),
            use_container_width=True,
            hide_index=True,
            disabled=["Time", "Input", "From", "Prediction", "Confidence"],
            column_config={"Correct?": st.column_config.CheckboxColumn("Correct?")},
            key="log_editor",
        )
        for row, val in zip(st.session_state.log, edited["Correct?"].tolist()):
            row["Correct?"] = val
        if st.button("Clear log"):
            st.session_state.log = []
            st.session_state.pop("last_logged", None)
            st.rerun()
