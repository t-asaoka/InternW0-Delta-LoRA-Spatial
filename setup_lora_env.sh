#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$PWD/.venv}"

echo "[1/7] Check environment"

command -v "$PYTHON_BIN"
command -v ffmpeg
command -v nvidia-smi

nvidia-smi

echo "[2/7] Create virtual environment"

"$PYTHON_BIN" -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

echo "[3/7] Upgrade pip"

python -m pip install --upgrade \
    pip setuptools wheel ninja packaging

echo "[4/7] Install PyTorch CUDA 12.8"

python -m pip install \
    torch==2.10.0 \
    torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "[5/7] Install InternW0-Delta"

python -m pip install -e '.[train,modelscope]'

echo "[6/7] Check dependencies"

python -m pip check

echo "[7/7] Check CUDA and libraries"

python - <<'PY'
import torch
import torchvision
import transformers
import accelerate
import deepspeed
import torchcodec
import av
import hydra
import wam

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("DeepSpeed:", deepspeed.__version__)
print("TorchCodec:", torchcodec.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16 is not supported")

print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))

props = torch.cuda.get_device_properties(0)
print("VRAM:", props.total_memory / 1024**3, "GiB")

x = torch.randn(
    1024, 1024,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)

y = (x @ x).float().square().mean()
y.backward()

print("BF16 Forward/Backward: PASS")
print("[SETUP COMPLETE]")
PY

echo ""
echo "Activate environment:"
echo "source $VENV_DIR/bin/activate"
