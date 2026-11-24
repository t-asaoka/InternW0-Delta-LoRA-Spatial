#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$PWD/data/libero}"
export WAM_CHECKPOINT_ROOT="${WAM_CHECKPOINT_ROOT:-$PWD/checkpoints}"
export WAM_PRETRAIN_CHECKPOINT="${WAM_PRETRAIN_CHECKPOINT:-$WAM_CHECKPOINT_ROOT/libero.pt}"
export WAM_VLM_PATH="${WAM_VLM_PATH:-$WAM_CHECKPOINT_ROOT/RynnBrain1.1-2B}"
if [[ ${DRY_RUN:-0} != 1 ]]; then
 python tools/check_spatial_dataset.py
 [[ -f "$WAM_PRETRAIN_CHECKPOINT" ]] || { echo "Missing checkpoint: $WAM_PRETRAIN_CHECKPOINT" >&2; exit 1; }
 if [[ ${GENERATE_TEXT_CACHE:-0} == 1 ]]; then
  python tools/text_cache.py task=libero
 fi
fi
exec bash run_lora.sh "$@"
