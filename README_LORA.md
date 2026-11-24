# LIBERO LoRA 継続学習

公式ZIPをベースにした追加実装。モデル重み・データは同梱していません。既存READMEの環境構築と外部重みの配置を済ませ、量子化前の libero.pt を使用してください。

```bash
export LIBERO_DATA_ROOT=/data/libero
export WAM_PRETRAIN_CHECKPOINT=/data/model_weights/libero.pt
python tools/text_cache.py task=libero
# 10 optimizer steps
bash run_lora.sh gradient_accumulation_steps=1 max_steps=10 save_every=10 log_every=1 eval_every=1000
# 本学習
bash run_lora.sh
# Video + Action (公式補助lossを維持)
WAM_LORA_SCOPE=both bash run_lora.sh learning_rate=5e-5
```

標準: action attention q/k/v/o, rank16, alpha32, dropout0, batch1, accumulation16, LR1e-4, 3 epochs. WAM_LORA_FFN=1 でFFN追加。WAM_LORA_RANK, WAM_LORA_ALPHA, WAM_LORA_DROPOUT, WAM_LORA_SCOPE (action/video/both) で変更可能。通常の run.sh は LoRA 無効。新規データは configs/data/libero.yaml の train.dataset_dirs を編集し、同一の LeRobot schema と正規化を維持してください。Text Cache を再生成してください。混合率は自動調整されません。

保存は公式の全重みチェックポイント方式。weights_manifest.json を確認して対象slotを選ぶ。

```bash
python tools/merge_lora_checkpoint.py runs/libero/RUN_ID/checkpoints/weights/slot_0.pt checkpoints/libero_lora_merged.pt
```

マージ後は公式の非LoRA推論で使用する。FP8化は別工程。マージ済み checkpoint からの再LoRA学習は可能だが、未マージLoRA checkpointからの厳密な学習再開は未対応。GPU・DeepSpeed・実データでの動作は未検証。Action-onlyでも補助Lossの計算自体は残る場合がある。
