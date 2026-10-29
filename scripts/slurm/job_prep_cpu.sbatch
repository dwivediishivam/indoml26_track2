#!/usr/bin/env bash
#SBATCH --job-name=t2-prep
#SBATCH --account=sanand
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# CPU only: build training banks, then a full PREFLIGHT dry run of the GPU pipeline.
set -eo pipefail
source "$HOME/indoml-t2/code/omni_env.sh"
set -u
cd "$ROOT/code"
python -c "import torch, transformers, jiwer, sentencepiece, julius, scipy, pyarrow; print('imports ok', torch.__version__)"
[ -s "$ROOT/bank/stats.json" ] && [ -s "$ROOT/bank/clean_bank.npy" ] && echo "bank exists, skipping prep" || python prep_data.py --val_dir "$ROOT/data/validation" --out "$ROOT/bank" --workers 10 \
  --esc50_dir "$ROOT/data/esc50/ESC-50-master/audio" \
  --parquet_glob "$HOME/indoml-offline/dataset/data/*.parquet" --floor_thr -28 --max_vaani_hours 8 \
  --extra_clean_glob "$ROOT/data/fleurs/parquet-data/*/*.parquet" --max_extra_hours 25
bash make_preflight_data.sh
PREFLIGHT=1 bash run_pipeline.sh
echo "PREP AND PREFLIGHT PASSED"
