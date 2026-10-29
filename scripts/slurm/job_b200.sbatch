#!/usr/bin/env bash
#SBATCH --job-name=t2_b200
#SBATCH --account=sanand
#SBATCH --partition=medium
#SBATCH --qos=medium
#SBATCH --nodelist=B2001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00              # TRAIN_MIN (210) + ~60 min eval/test/ASR + margin
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# GPU job. Only submit after job_prep_cpu.sbatch printed "PREP AND PREFLIGHT PASSED".
set -eo pipefail
source "$HOME/indoml-t2/code/omni_env.sh"
export TRAIN_MIN=${TRAIN_MIN:-210}
nvidia-smi
bash "$ROOT/code/run_pipeline.sh"
