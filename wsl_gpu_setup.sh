#!/bin/bash
# Complete GPU setup and training script for WSL2

set -e

PROJECT_DIR="/mnt/c/Users/elidr/Downloads/Classification-des-cultures-agricoles-partir-d-images-satellites-"
TRAINING_ONLY="${TRAINING_ONLY:-0}"
cd "$PROJECT_DIR"

echo "========== WSL2 GPU Setup & Training =========="
echo ""

# Activate venv
echo "[1/5] Activating Python virtual environment..."
source wsl_venv/bin/activate
export LD_LIBRARY_PATH="${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/cublas/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/nvvm/lib64:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/cufft/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/curand/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/cusolver/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/cusparse/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/nccl/lib:${PROJECT_DIR}/wsl_venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
echo "✓ venv activated"

if [ "$TRAINING_ONLY" = "1" ]; then
    echo ""
    echo "[setup] TRAINING_ONLY=1, skipping TensorFlow verification and dependency install"
    echo "[setup] Proceeding directly to training"
fi

if [ "$TRAINING_ONLY" != "1" ]; then
    # Verify TensorFlow and GPU
    echo ""
    echo "[2/5] Verifying TensorFlow installation..."
    python3 -u << 'PYEOF'
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")
if gpus:
    for gpu in gpus:
        print(f"  - {gpu}")
    print("✓ GPU successfully detected!")
else:
    print("✗ WARNING: No GPU detected - training will be on CPU")
PYEOF

    # Install remaining requirements
    echo ""
    echo "[3/5] Installing remaining project dependencies..."
    pip install -r requirements.txt
    echo "✓ All dependencies installed"
fi

# Run training
echo ""
echo "[4/5] Starting ResNet50 training on GPU..."
echo "This will run two phases:"
echo "  - Phase 1: Feature Extraction (20 epochs)"
echo "  - Phase 2: Fine-tuning (10 epochs)"
echo ""
python -u src/train.py

# Evaluate
echo ""
echo "[5/5] Running model evaluation..."
python -u src/evaluate.py

echo ""
echo "========== Setup Complete =========="
echo "✓ Training finished"
echo "✓ Model saved to: models/best_model.keras"
echo "✓ Metrics saved to: models/evaluation/"
echo ""
echo "To launch Streamlit app:"
echo "  streamlit run app.py"
