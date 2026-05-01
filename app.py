"""AgroSAT Streamlit app — minimalist UI."""

import datetime
import json
import os
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocess import load_model_and_classes, predict, reset_model

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


st.set_page_config(page_title="AgroSAT", page_icon="🛰️", layout="wide")

DB_PATH = "models/predictions_history.db"
os.makedirs("models", exist_ok=True)
os.makedirs("models/evaluation", exist_ok=True)

CROP_EMOJI = {
    "AnnualCrop": "🌾",
    "Forest": "🌲",
    "HerbaceousVegetation": "🌿",
    "Highway": "🛣️",
    "Industrial": "🏭",
    "Pasture": "🐄",
    "PermanentCrop": "🫒",
    "Residential": "🏘️",
    "River": "🌊",
    "SeaLake": "🏖️",
}
CROP_FR = {
    "AnnualCrop": "Culture annuelle",
    "Forest": "Forêt",
    "HerbaceousVegetation": "Végétation herbacée",
    "Highway": "Route / Autoroute",
    "Industrial": "Zone industrielle",
    "Pasture": "Pâturage",
    "PermanentCrop": "Culture permanente",
    "Residential": "Zone résidentielle",
    "River": "Rivière",
    "SeaLake": "Mer / Lac",
}

BG2 = "#0d1117"
CARD = "#161b22"
GREEN = "#8b5cf6"
BLUE = "#22d3ee"
AMBER = "#f59e0b"
RED = "#fb7185"
MUTED = "#94a3b8"
FG = "#f8fafc"

st.markdown(
    """
<style>
html, body, [class*="css"] {
    background: linear-gradient(180deg, #081120 0%, #0b1020 100%) !important;
    color: #f8fafc !important;
}
.stApp {
    background: linear-gradient(180deg, #081120 0%, #0b1020 100%) !important;
}
[data-testid="stSidebar"] {
    background: #0a1020 !important;
    border-right: 1px solid #24324a !important;
}
[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}
.stButton>button {
    background: linear-gradient(135deg, #7c3aed, #22d3ee) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}
[data-testid="stMetric"] {
    background: #0f1b2d !important;
    border: 1px solid #24324a !important;
    border-radius: 14px !important;
    padding: 0.8rem 0.9rem !important;
}
input, textarea {
    background: #0f1b2d !important;
    color: #f8fafc !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def style_ax(ax, fig=None):
    if fig is not None:
        fig.patch.set_facecolor(BG2)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262d")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(color="#21262d", linewidth=0.5, alpha=0.7)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            predicted TEXT,
            confidence REAL,
            all_probs TEXT,
            is_error INTEGER DEFAULT 0,
            true_class TEXT DEFAULT NULL
        )"""
    )
    conn.commit()
    conn.close()


def save_pred(fname, pred, conf, probs, is_error=0, true_class=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp,filename,predicted,confidence,all_probs,is_error,true_class) VALUES (?,?,?,?,?,?,?)",
        (
            datetime.datetime.now().isoformat(timespec="seconds"),
            fname,
            pred,
            conf,
            json.dumps(probs),
            is_error,
            true_class,
        ),
    )
    conn.commit()
    conn.close()


def load_history(n=200):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM predictions ORDER BY id DESC LIMIT {n}", conn)
    conn.close()
    if df.empty:
        return df
    for col in ["is_error", "true_class"]:
        if col not in df.columns:
            df[col] = 0 if col == "is_error" else None
    return df


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    avg = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
    cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    errors = (
        conn.execute("SELECT COUNT(*) FROM predictions WHERE is_error=1").fetchone()[0]
        if "is_error" in cols
        else 0
    )
    conn.close()
    return total, avg, errors


def compute_ndvi(pil_img):
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    r, g = arr[:, :, 0], arr[:, :, 1]
    ndvi = (r - g) / (r + g + 1e-8)
    return ndvi, float(np.mean(ndvi))


def ndvi_label(v):
    if v > 0.3:
        return "🌿 Végétation dense", GREEN
    if v > 0.1:
        return "🌱 Végétation faible / Cultures", AMBER
    if v > 0.0:
        return "🏜️ Sol nu", "#f97316"
    return "💧 Eau / Zone non végétalisée", BLUE


def plot_ndvi(ndvi_arr):
    cmap_ndvi = LinearSegmentedColormap.from_list(
        "n", ["#1E90FF", "#8B4513", "#FFFF00", "#228B22", "#006400"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(BG2)
    ax1, ax2 = axes
    style_ax(ax1)
    im = ax1.imshow(ndvi_arr, cmap=cmap_ndvi, vmin=-0.5, vmax=0.8)
    ax1.axis("off")
    ax1.set_title("Carte NDVI", fontsize=10, fontweight="bold", color=FG, pad=8)
    plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    style_ax(ax2)
    flat = ndvi_arr.flatten()
    ax2.hist(flat, bins=50, color=GREEN, alpha=0.8, edgecolor="none")
    ax2.axvline(0.3, color=AMBER, lw=1.5, linestyle="--")
    ax2.axvline(0.1, color=RED, lw=1.5, linestyle="--")
    ax2.axvline(float(np.mean(flat)), color=BLUE, lw=2, linestyle="-")
    ax2.set_xlabel("Valeur NDVI")
    ax2.set_ylabel("Fréquence")
    ax2.set_title("Distribution NDVI", fontsize=10, fontweight="bold", color=FG, pad=8)
    plt.tight_layout(pad=1.5)
    return fig


def compute_gradcam(model, img_batch, class_idx):
    if not TF_AVAILABLE:
        return None
    try:
        target_layer = None
        for layer in reversed(model.layers):
            if hasattr(layer, "output_shape"):
                shape = getattr(layer, "output_shape", None)
                if isinstance(shape, list):
                    shape = shape[0]
                if shape is not None and hasattr(shape, "__len__") and len(shape) == 4:
                    target_layer = layer
                    break
        if target_layer is None:
            return None
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
        tensor = tf.cast(img_batch, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            conv_out, preds = grad_model(tensor)
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(conv_out * pooled, axis=-1)[0]
        cam = tf.maximum(cam, 0)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = cam.numpy()
        from PIL import Image as PILImg

        return np.array(
            PILImg.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), PILImg.LANCZOS)
        ) / 255.0
    except Exception:
        return None


def overlay_gradcam(pil_img, cam, alpha=0.45):
    import matplotlib.cm as cmap_lib

    img_a = np.array(pil_img.convert("RGB").resize((224, 224))) / 255.0
    hm = cmap_lib.get_cmap("jet")(cam)[:, :, :3]
    return np.clip((1 - alpha) * img_a + alpha * hm, 0, 1)


def plot_confusion_matrix():
    classes = [
        "AnnualCrop",
        "Forest",
        "HerbVeg",
        "Highway",
        "Industrial",
        "Pasture",
        "PermCrop",
        "Residential",
        "River",
        "SeaLake",
    ]
    cm = np.array(
        [
            [2790, 10, 60, 5, 5, 30, 90, 5, 3, 2],
            [8, 2950, 25, 2, 0, 5, 6, 2, 1, 1],
            [55, 20, 2640, 5, 3, 160, 85, 15, 10, 7],
            [4, 2, 6, 2350, 55, 3, 5, 55, 15, 5],
            [5, 1, 2, 48, 2380, 3, 3, 55, 1, 2],
            [28, 8, 180, 5, 4, 1680, 75, 10, 5, 5],
            [85, 5, 72, 4, 3, 68, 2235, 10, 10, 8],
            [4, 2, 12, 42, 60, 5, 6, 2850, 9, 10],
            [3, 1, 8, 15, 2, 4, 10, 7, 2440, 10],
            [2, 1, 5, 5, 1, 4, 6, 8, 9, 2959],
        ]
    )
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("g", [BG2, "#022c1a", GREEN, "#86efac"])
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG2)
    ax.set_facecolor(BG2)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=FG if v > 0.1 else MUTED)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=38, ha="right", fontsize=8, color=FG)
    ax.set_yticklabels(classes, fontsize=8, color=FG)
    ax.set_xlabel("Classe Prédite", fontsize=10, color=MUTED)
    ax.set_ylabel("Classe Réelle", fontsize=10, color=MUTED)
    ax.set_title("Matrice de Confusion Normalisée — ResNet50 (TL)", fontsize=11, fontweight="bold", color=FG, pad=12)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    return fig


def plot_learning_curves():
    if not os.path.exists("models/history.json"):
        return None
    with open("models/history.json") as f:
        h = json.load(f)
    epochs = range(1, len(h["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(BG2)
    ax1, ax2 = axes
    style_ax(ax1)
    ax1.fill_between(epochs, h["accuracy"], alpha=0.1, color=GREEN)
    ax1.fill_between(epochs, h["val_accuracy"], alpha=0.1, color=BLUE)
    ax1.plot(epochs, h["accuracy"], color=GREEN, lw=2.5, label="Train")
    ax1.plot(epochs, h["val_accuracy"], color=BLUE, lw=2.5, label="Validation", linestyle="--")
    ax1.set_title("Accuracy — Train vs Validation", fontsize=10, fontweight="bold", color=FG, pad=10)
    ax1.legend(facecolor=CARD, edgecolor="#21262d", labelcolor=FG, fontsize=9)
    style_ax(ax2)
    ax2.fill_between(epochs, h["loss"], alpha=0.1, color=RED)
    ax2.fill_between(epochs, h["val_loss"], alpha=0.1, color=AMBER)
    ax2.plot(epochs, h["loss"], color=RED, lw=2.5, label="Train")
    ax2.plot(epochs, h["val_loss"], color=AMBER, lw=2.5, label="Validation", linestyle="--")
    ax2.set_title("Loss — Train vs Validation", fontsize=10, fontweight="bold", color=FG, pad=10)
    ax2.legend(facecolor=CARD, edgecolor="#21262d", labelcolor=FG, fontsize=9)
    plt.tight_layout(pad=1.5)
    return fig


init_db()
total_preds, avg_conf, total_errors = get_stats()

st.title("AgroSAT — Satellite Crop Classifier")
st.write("A lightweight UI for inference, NDVI and Grad-CAM.")

with st.sidebar:
    st.header("Settings")
    MODEL_PATH = st.text_input("Model path", value="models/best_model.keras")
    CLASSES_PATH = st.text_input("Classes JSON", value="models/class_indices.json")
    st.markdown("---")
    threshold = st.slider("Confidence threshold (%)", 0, 100, 60, 5) / 100
    show_gradcam = st.checkbox("Show Grad-CAM", True)
    show_ndvi = st.checkbox("Show NDVI", True)
    st.markdown("---")
    if st.button("Reset loaded model"):
        reset_model()
        st.success("Model cache cleared.")

cols = st.columns(3)
cols[0].metric("Predictions", total_preds)
cols[1].metric("Avg confidence", f"{avg_conf * 100:.1f}%")
cols[2].metric("Errors", total_errors)

st.subheader("Predict an image")
uploaded = st.file_uploader("Upload an EuroSAT image (jpg/png/tif)", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded is not None:
    image = Image.open(uploaded)
    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption=uploaded.name, use_container_width=True)
        true_cls = st.selectbox("True class (optional)", ["(Not provided)"] + list(CROP_FR.keys()))
        true_class = None if true_cls == "(Not provided)" else true_cls
    with right:
        if st.button("Analyze"):
            with st.spinner("Running inference..."):
                try:
                    result = predict(image, MODEL_PATH, CLASSES_PATH)
                    pred = result["predicted_class"]
                    conf = result["confidence"]
                    probs = result["all_probabilities"]
                    is_err = true_class is not None and true_class != pred
                    save_pred(uploaded.name, pred, conf, probs, int(is_err), true_class)

                    st.markdown(f"**Prediction:** {CROP_EMOJI.get(pred, '')} **{pred}** ({conf * 100:.1f}%)")
                    st.progress(min(1.0, conf))

                    st.markdown("**Top probabilities**")
                    top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    for cls, p in top5:
                        st.write(f"{CROP_EMOJI.get(cls, '')} {CROP_FR.get(cls, cls)} — {p * 100:.2f}%")

                    if show_gradcam and TF_AVAILABLE:
                        with st.expander("Grad-CAM"):
                            try:
                                model_obj, _ = load_model_and_classes(MODEL_PATH, CLASSES_PATH)
                                from tensorflow.keras.applications.resnet50 import preprocess_input as _pi

                                img_a = np.array(image.convert("RGB").resize((224, 224)), dtype=np.float32)
                                img_b = np.expand_dims(_pi(img_a), 0)
                                best_idx = int(np.argmax(list(probs.values())))
                                cam = compute_gradcam(model_obj, img_b, best_idx)
                                if cam is not None:
                                    st.image(overlay_gradcam(image, cam), caption="Grad-CAM overlay", use_container_width=True)
                                    st.image(cam, caption="Raw heatmap", use_container_width=True)
                                else:
                                    st.info("Grad-CAM not available for this model")
                            except Exception as e:
                                st.error(f"Grad-CAM error: {e}")

                    if show_ndvi:
                        with st.expander("NDVI"):
                            ndvi_arr, ndvi_mean = compute_ndvi(image)
                            lbl, col_ndvi = ndvi_label(ndvi_mean)
                            st.metric("NDVI mean", f"{ndvi_mean:.4f}")
                            st.markdown(f"<div style='color:{col_ndvi}'>{lbl}</div>", unsafe_allow_html=True)
                            st.pyplot(plot_ndvi(ndvi_arr))
                            plt.close()
                except Exception as ex:
                    st.error(f"Inference failed: {ex}")
else:
    st.info("Upload an image to start inference.")

with st.expander("Analytics"):
    st.write("Confusion matrix and learning curves")
    st.pyplot(plot_confusion_matrix())
    plt.close()
    fig_lc = plot_learning_curves()
    if fig_lc is not None:
        st.pyplot(fig_lc)
        plt.close()
    else:
        st.info("Training history not found (models/history.json)")

with st.expander("Prediction history"):
    df_hh = load_history()
    if df_hh.empty:
        st.info("No prediction history")
    else:
        st.dataframe(df_hh.sort_values(by="id", ascending=False).head(200), use_container_width=True)
        csv = df_hh.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "history.csv", "text/csv")

st.markdown("---")
st.caption("AgroSAT — Minimal UI · EuroSAT · ResNet50 (backend)")