import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "EuroSAT", "2750")
DATA_DIR         = os.environ.get("EUROSAT_DATA_DIR", DEFAULT_DATA_DIR)
IMG_SIZE         = (224, 224)
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", "4"))
EPOCHS           = 15
FINE_TUNE_EPOCHS = 10

os.makedirs("models", exist_ok=True)

tf.get_logger().setLevel("ERROR")

# ─── GPU ────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("🚀 Mixed precision ON")
else:
    print("❌ No GPU — running on CPU")

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_DIR}\n"
        "Set EUROSAT_DATA_DIR or place the dataset under EuroSAT/2750."
    )

# ─── LOAD DATASET ───────────────────────
print(f"[DATA] Loading dataset from: {DATA_DIR}")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("[DATA] Building validation split")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

CLASS_NAMES   = train_ds.class_names
class_indices = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES   = len(CLASS_NAMES)

print(f"[INFO] Classes ({NUM_CLASSES}): {CLASS_NAMES}")

with open("models/class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)

# ─── DATA AUGMENTATION (خارج الموديل!) ──
# ✅ FIX: الـ augmentation والـ preprocessing يُطبَّقان على الـ dataset
#         وليس داخل الموديل، لأنه إذا كانا داخل الموديل يُطبَّقان أيضاً
#         أثناء التنبؤ مما يسبب نتائج خاطئة.

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
MAP_PARALLEL_CALLS = int(os.environ.get("MAP_PARALLEL_CALLS", "1"))
PREFETCH_BUFFER = int(os.environ.get("PREFETCH_BUFFER", "1"))

print("[DATA] Preparing augmentation and preprocessing pipeline")

def augment_and_preprocess(x, y):
    x = data_augmentation(x, training=True)
    x = preprocess_input(x)   # ResNet50 normalization
    return x, y

def preprocess_only(x, y):
    x = preprocess_input(x)   # بدون augmentation للـ validation
    return x, y

train_ds = (train_ds
            .shuffle(1000)
                        .map(augment_and_preprocess, num_parallel_calls=MAP_PARALLEL_CALLS)
                        .prefetch(PREFETCH_BUFFER))

val_ds = (val_ds
                    .map(preprocess_only, num_parallel_calls=MAP_PARALLEL_CALLS)
                    .prefetch(PREFETCH_BUFFER))

print("[DATA] Dataset pipeline ready")

# ─── MODEL ──────────────────────────────
# ✅ FIX: الموديل يستقبل الصور مباشرة بدون أي preprocessing داخله

base_model = ResNet50(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))

# لا augmentation ولا preprocess_input هنا!
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = models.Model(inputs, outputs)
model.summary()

# ─── COMPILE ────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── CALLBACKS ──────────────────────────
cbs = [
    callbacks.ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        verbose=1
    )
]

# ─── PHASE 1: Feature extraction ────────
print("\n🚀 Phase 1 — Feature Extraction (base frozen)\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=1
)

# ─── PHASE 2: Fine-tuning ────────────────
print("\n🔥 Phase 2 — Fine-Tuning (last 30 layers)\n")

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers[-40:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=cbs,
    verbose=1
)

# ─── SAVE ───────────────────────────────
model.save("models/final_model.keras")
print("\n✅ final_model.keras saved!")

history = {key: history1.history[key] + history2.history[key] for key in history1.history.keys()}
with open("models/history.json", "w") as f:
    json.dump(history, f, indent=2)
print("📝 history.json saved!")

# ─── TRAINING CURVES ────────────────────
def merge(h1, h2, key):
    return h1.history[key] + h2.history[key]

acc      = merge(history1, history2, "accuracy")
val_acc  = merge(history1, history2, "val_accuracy")
loss     = merge(history1, history2, "loss")
val_loss = merge(history1, history2, "val_loss")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(acc,     label="Train")
axes[0].plot(val_acc, label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(loss,     label="Train")
axes[1].plot(val_loss, label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("models/curves.png", dpi=150)
plt.close()
print("📈 curves.png saved!\n")