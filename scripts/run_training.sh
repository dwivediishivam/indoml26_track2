#!/usr/bin/env bash
# Full training recipe on one GPU (see README for data download). Usage: bash scripts/run_training.sh DATA_DIR
set -euo pipefail
cd "$(dirname "$0")/.."
D=${1:?data folder with validation/, esc50/, fleurs/, vaani/}
python checkpoints/download_checkpoints.py --all                     # DNS64 init (+ released checkpoints)
python src/prep_data.py --val_dir $D/validation --out $D/bank --split configs/split.json \
  --esc50_dir $D/esc50/ESC-50-master/audio --parquet_glob "$D/vaani/data/*.parquet" --floor_thr -28 --max_vaani_hours 8 \
  --extra_clean_glob "$D/fleurs/parquet-data/*/*.parquet" --max_extra_hours 25
python src/make_pseudotest.py --val_dir $D/validation --split configs/split.json --out $D/pt_testlike --extra 0.5 --gain_db 1.5
A="--bank $D/bank --val_dir $D/validation --asr_dir models/sravaani"
python train/train.py --config configs/base_dns64.yaml $A --out runs/base_dns64
python train/train.py --config configs/base_bsrnn.yaml $A --out runs/base_bsrnn
cp runs/base_dns64/best.pt checkpoints/b200_dns64.pt; cp runs/base_bsrnn/best.pt checkpoints/colabA_bsrnn_feat112.pt
python train/train.py --config configs/ft_dns64.yaml $A --out runs/ft_dns64 --pseudotest $D/pt_testlike
python train/train.py --config configs/ft_bsrnn.yaml $A --out runs/ft_bsrnn --pseudotest $D/pt_testlike
cp runs/ft_dns64/best.pt checkpoints/ft_dns64.pt; cp runs/ft_bsrnn/best.pt checkpoints/ft_bsrnn.pt
