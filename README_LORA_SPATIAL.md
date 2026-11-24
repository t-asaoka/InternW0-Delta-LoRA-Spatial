# LIBERO-Spatial 専用 LoRA

元の LoRA ZIP をベースに、configs/data/libero.yaml の train.dataset_dirs を Spatial 1種類に限定。val は同じリストを参照。公式の val_set_proportion=0.0 を維持するため、独立した検証セットではありません。正規化・画像変換・モデル構造は変更なし。Text Cache の既定パスは .cache/internw0/text_embed/libero_spatial。学習率は 3e-5 に変更、その他の LoRA 設定は維持。

公式 README と docs/models.md に従って環境と backbone を準備。量子化前の libero.pt を checkpoints/libero.pt に配置。必要に応じて環境変数で既存ファイルを指定。

```bash
pip install -U huggingface_hub
bash download_spatial.sh
hf download InternRobotics/InternW0-Delta-Libero libero.pt --local-dir checkpoints
GENERATE_TEXT_CACHE=1 bash run_lora_spatial.sh gradient_accumulation_steps=1 max_steps=10 save_every=10 log_every=1 eval_every=1000
bash run_lora_spatial.sh
```

既存データの場合 download_spatial.sh は不要。LIBERO_DATA_ROOT はデータセットの親ディレクトリ。ダウンロード候補の実データ互換性は未確認。メタデータチェックは必須キーと state/action 次元のみ検証し、意味・画像解像度・実ファイルは保証しない。

出力は runs/spatial/libero/RUN_ID/checkpoints/weights/。weights_manifest.json で対象 slot を選択。

```bash
python tools/merge_lora_checkpoint.py runs/spatial/libero/RUN_ID/checkpoints/weights/slot_0.pt checkpoints/libero_spatial_lora_merged.pt
```

重み・データは同梱なし。GPU 学習、ダウンロード、実データ検証は未実施。元の README_LORA.md も参照（学習率・出力先は本書が優先）。
