#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export WAM_LORA_ENABLED=1
export WAM_LORA_SCOPE=${WAM_LORA_SCOPE:-action}
export WAM_LORA_RANK=${WAM_LORA_RANK:-16}
export WAM_LORA_ALPHA=${WAM_LORA_ALPHA:-32}
export WAM_LORA_DROPOUT=${WAM_LORA_DROPOUT:-0}
export WAM_LORA_FFN=${WAM_LORA_FFN:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export WAM_PRETRAIN_CHECKPOINT=${WAM_PRETRAIN_CHECKPOINT:-checkpoints/libero.pt}
args=()
if [[ "$WAM_LORA_SCOPE" == action ]]; then
 args+=(model.loss.lambda_video=0.0 model.future_delta.loss_weight=0.0 model.future_delta.semantic_alignment.loss_weight=0.0)
fi
export WAM_OUTPUT_ROOT=${WAM_OUTPUT_ROOT:-runs/spatial}
exec bash run.sh libero batch_size=1 gradient_accumulation_steps=16 learning_rate=3e-5 weight_decay=0.0 num_epochs=3 num_workers=4 save_every=200 eval_every=200 "${args[@]}" "$@"
