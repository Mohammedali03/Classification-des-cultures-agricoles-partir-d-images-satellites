"""
evaluate.py — Évaluation complète du modèle EuroSAT
PFE Licence d'excellence en IA

✅ FIXES:
  - preprocess_input appliqué une seule fois (cohérent avec train.py)
  - Utilise tf.data pipeline (pas ImageDataGenerator)
  - Sauvegarde automatique de tous les graphiques et métriques
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras.applications.resnet50 import preprocess_input

# ───────────────── CONFIG ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH   = os.path.join(BASE_DIR, "models", "best_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")
RESULTS_DIR  = os.path.join(BASE_DIR, "models", "evaluation")
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "EuroSAT", "2750")
DATA_DIR = os.environ.get("EUROSAT_DATA_DIR", DEFAULT_DATA_DIR)

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)


# ───────────────── MAIN ───────────────────────────────────────────────────────
def main():
    print("\n[INFO] Starting evaluation...\n")

    # ─── 1. Load model ──────────────────────────────────────────────────────
    print(f"MODEL PATH : {MODEL_PATH}")
    print(f"EXISTS     : {os.path.exists(MODEL_PATH)}\n")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[ERROR] Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully ✅\n")

    # ─── 2. Load classes ────────────────────────────────────────────────────
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"[ERROR] Classes file not found: {CLASSES_PATH}")

    with open(CLASSES_PATH) as f:
        class_indices = json.load(f)

    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"[INFO] Classes: {class_names}\n")

    # ─── 3. Load & preprocess dataset ───────────────────────────────────────
    # ✅ FIX: preprocess_input appliqué UNE SEULE FOIS ici,
    #         identique au pipeline de train.py (preprocess_only).
    #         Pas d'augmentation en évaluation.

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    total = sum(y.shape[0] for _, y in val_ds)
    print(f"[INFO] Total samples: {total}\n")

    # ─── 4. Predict ─────────────────────────────────────────────────────────
    print("[INFO] Running predictions...\n")
    preds  = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)

    # ─── 5. Global metrics ──────────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print("=" * 50 + "\n")

    # ─── 6. Classification report ───────────────────────────────────────────
    report_str  = classification_report(y_true, y_pred, target_names=class_names)
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    print("[REPORT]\n")
    print(report_str)

    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)
    print(f"[INFO] Report saved → {report_path}")

    # ─── 7. Confusion matrix ────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {cm_path}")

    # ─── 8. Metrics per class (bar chart) ───────────────────────────────────
    per_class = {
        cls: {
            "precision": report_dict[cls]["precision"],
            "recall":    report_dict[cls]["recall"],
            "f1-score":  report_dict[cls]["f1-score"],
        }
        for cls in class_names
    }

    x = np.arange(len(class_names))
    w = 0.25

    plt.figure(figsize=(13, 5))
    plt.bar(x - w, [per_class[c]["precision"] for c in class_names], w, label="Precision", color="#4C9BE8")
    plt.bar(x,     [per_class[c]["recall"]    for c in class_names], w, label="Recall",    color="#F4A261")
    plt.bar(x + w, [per_class[c]["f1-score"]  for c in class_names], w, label="F1-score",  color="#2A9D8F")

    plt.xticks(x, class_names, rotation=35, ha="right", fontsize=9)
    plt.ylim(0, 1.15)
    plt.title("Metrics per Class", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    metrics_img = os.path.join(RESULTS_DIR, "metrics_per_class.png")
    plt.savefig(metrics_img, dpi=150)
    plt.close()
    print(f"[INFO] Metrics per class saved → {metrics_img}")

    # ─── 9. Prediction distribution (detect class collapse) ─────────────────
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_dist = dict(zip([class_names[i] for i in unique], counts.tolist()))

    print("\n[INFO] Prediction distribution:")
    for cls, cnt in pred_dist.items():
        bar = "█" * int(cnt / total * 40)
        print(f"  {cls:<25} {cnt:>5}  {bar}")

    plt.figure(figsize=(10, 4))
    all_counts = [pred_dist.get(c, 0) for c in class_names]
    colors = ["#E63946" if c == max(all_counts) else "#457B9D" for c in all_counts]
    plt.bar(class_names, all_counts, color=colors)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.title("Prediction Distribution (red = dominant class)", fontsize=12)
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    dist_path = os.path.join(RESULTS_DIR, "prediction_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"\n[INFO] Distribution chart saved → {dist_path}")

    # ─── 10. Save JSON summary ──────────────────────────────────────────────
    summary = {
        "global": {
            "accuracy":  float(acc),
            "precision": float(prec),
            "recall":    float(rec),
            "f1":        float(f1),
        },
        "per_class":             per_class,
        "prediction_distribution": pred_dist,
    }

    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Metrics JSON saved → {json_path}")

    print("\n✅ [DONE] Evaluation complete!\n")


# ───────────────── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    main()