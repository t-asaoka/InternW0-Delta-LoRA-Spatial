#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f ".venv/bin/activate" ]]; then
    . .venv/bin/activate
fi

export WAM_CHECKPOINT_ROOT="${WAM_CHECKPOINT_ROOT:-$PWD/checkpoints}"

export WAM_VLM_PATH="${WAM_VLM_PATH:-$WAM_CHECKPOINT_ROOT/RynnBrain1.1-2B}"

WAN_SUBDIR="DiffSynth-Studio/Wan-Series-Converted-Safetensors"

WAN_DIR="$WAM_CHECKPOINT_ROOT/$WAN_SUBDIR"

mkdir -p "$WAM_CHECKPOINT_ROOT"

# ========================================
# 1. libero.pt
# ========================================

echo "[1/5] Download libero.pt"

if [[ -s "$WAM_CHECKPOINT_ROOT/libero.pt" ]]; then
    echo "[SKIP] libero.pt already exists"
else
    hf download InternRobotics/InternW0-Delta-Libero \
        libero.pt \
        --local-dir "$WAM_CHECKPOINT_ROOT"
fi

# ========================================
# 2. Wan2.2 VAE
# ========================================

echo "[2/5] Download Wan2.2 VAE"

if [[ -s "$WAN_DIR/Wan2.2_VAE.safetensors" ]]; then
    echo "[SKIP] Wan2.2_VAE.safetensors already exists"
else
    hf download SereneC/wan-series-checkpoint \
        "$WAN_SUBDIR/Wan2.2_VAE.safetensors" \
        --local-dir "$WAM_CHECKPOINT_ROOT"
fi

# ========================================
# 3. UMT5-XXL BF16
# ========================================

echo "[3/5] Download UMT5-XXL BF16"

if [[ -s "$WAN_DIR/models_t5_umt5-xxl-enc-bf16.safetensors" ]]; then
    echo "[SKIP] UMT5 weight already exists"
else
    hf download SereneC/wan-series-checkpoint \
        "$WAN_SUBDIR/models_t5_umt5-xxl-enc-bf16.safetensors" \
        --local-dir "$WAM_CHECKPOINT_ROOT"
fi

# ========================================
# 4. UMT5 tokenizer
# ========================================

echo "[4/5] Download UMT5 tokenizer"

hf download Wan-AI/Wan2.1-T2V-1.3B \
    --include 'google/umt5-xxl/*' \
    --local-dir "$WAM_CHECKPOINT_ROOT/Wan-AI/Wan2.1-T2V-1.3B"

# ========================================
# 5. RynnBrain1.1-2B
# ========================================

echo "[5/5] Download RynnBrain1.1-2B"

hf download Alibaba-DAMO-Academy/RynnBrain1.1-2B \
    --local-dir "$WAM_VLM_PATH"

echo ""
echo "[DOWNLOAD COMPLETE]"

du -sh "$WAM_CHECKPOINT_ROOT"/*

echo ""
echo "Checkpoint root: $WAM_CHECKPOINT_ROOT"
echo "RynnBrain path: $WAM_VLM_PATH"