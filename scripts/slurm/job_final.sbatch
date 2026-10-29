#!/usr/bin/env bash
#SBATCH --job-name=t2_final
#SBATCH --account=sanand
#SBATCH --partition=medium
#SBATCH --qos=medium
#SBATCH --nodelist=B2001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
set -eo pipefail
source "$HOME/indoml-t2/code/omni_env.sh"
export OMP_NUM_THREADS=2 CANDS="$(cat $ROOT/final_cands.txt)"
bash $ROOT/code/final_select.sh
