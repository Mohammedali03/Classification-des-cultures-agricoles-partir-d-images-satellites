"""
utils/preprocess.py — Prétraitement & prédiction
PFE Licence d'excellence en IA

✅ FIXES:
  - preprocess_input يُطبَّق مرة واحدة فقط (كان يُطبَّق مرتين من قبل)
  - لا يوجد data augmentation أثناء التنبؤ
  - singleton pattern محسَّن مع إمكانية reset
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
import os

IMG_SIZE = (224, 224)

# ─── Singleton ───────────────────────────────────────────────────────────────

_model       = None
_class_names = None
_loaded_path = None   # لتتبع أي موديل محمول حالياً


def reset_model():
    """إعادة تعيين الـ singleton (مفيد عند تغيير الموديل)"""
    global _model, _class_names, _loaded_path
    _model       = None
    _class_names = None
    _loaded_path = None


def load_model_and_classes(
    model_path:   str = "models/best_model.keras",
    classes_path: str = "models/class_indices.json"
):
    global _model, _class_names, _loaded_path

    # إعادة تحميل إذا تغيّر مسار الموديل
    if _model is None or _loaded_path != model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}\n"
                "Lancez d'abord : python src/train.py"
            )
        print(f"[INFO] Loading model: {model_path}")
        _model       = tf.keras.models.load_model(model_path)
        _loaded_path = model_path
        print("[INFO] Model loaded ✅")

    if _class_names is None:
        if os.path.exists(classes_path):
            with open(classes_path) as f:
                idx_to_class = {v: k for k, v in json.load(f).items()}
            _class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        else:
            # Fallback — ordre alphabétique EuroSAT
            _class_names = [
                "AnnualCrop", "Forest", "HerbaceousVegetation",
                "Highway", "Industrial", "Pasture",
                "PermanentCrop", "Residential", "River", "SeaLake",
            ]
        print(f"[INFO] Classes: {_class_names}")

    return _model, _class_names


# ─── Prétraitement ────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prépare une image PIL pour l'inférence ResNet50.

    Pipeline:
        1. Conversion RGB
        2. Redimensionnement → 224×224 (LANCZOS)
        3. preprocess_input ResNet50 → pixels normalisés         ← ✅ UNE SEULE FOIS
        4. Ajout dimension batch → (1, 224, 224, 3)

    ⚠️  NE PAS appeler preprocess_input en dehors de cette fonction,
        sinon les pixels seront normalisés deux fois et le modèle
        donnera des prédictions incorrectes (même classe pour tout).
    """
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)          # [0, 255]
    arr = preprocess_input(arr)                    # [-1, 1]  ← une seule fois ✅
    return np.expand_dims(arr, axis=0)             # (1, 224, 224, 3)


# ─── Prédiction ───────────────────────────────────────────────────────────────

def predict(
    image:        Image.Image,
    model_path:   str = "models/best_model.keras",
    classes_path: str = "models/class_indices.json"
) -> dict:
    """
    Prédit la classe d'une image satellite EuroSAT.

    Retourne:
        {
          "predicted_class":   str,
          "confidence":        float,   # 0–1
          "all_probabilities": {class: prob, ...}
        }
    """
    model, class_names = load_model_and_classes(model_path, classes_path)

    # ✅ preprocess_input est appelé UNE SEULE FOIS ici
    tensor = preprocess_image(image)                       # (1, 224, 224, 3)
    probs  = model.predict(tensor, verbose=0)[0]           # (num_classes,)

    best_idx   = int(np.argmax(probs))
    confidence = float(probs[best_idx])

    all_probs = {name: float(probs[i]) for i, name in enumerate(class_names)}

    return {
        "predicted_class":   class_names[best_idx],
        "confidence":        confidence,
        "all_probabilities": all_probs,
    }


# ─── Évaluation ───────────────────────────────────────────────────────────────

def evaluate_model(
    model_path:   str,
    data_dir:     str,
    classes_path: str = "models/class_indices.json"
) -> dict:
    """
    Calcule Accuracy, Precision, Recall, F1 sur le set de validation.

    ✅ Utilise tf.data + preprocess_input (cohérent avec train.py)
    """
    from sklearn.metrics import classification_report, confusion_matrix

    model, class_names = load_model_and_classes(model_path, classes_path)

    IMG_SIZE_TF = (224, 224)
    BATCH_SIZE  = 32

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE_TF,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ✅ même preprocessing que train.py — une seule fois, sans augmentation
    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    preds  = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "report":           report,
        "confusion_matrix": cm.tolist(),
        "classes":          class_names
    }