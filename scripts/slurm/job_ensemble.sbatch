#!/usr/bin/env bash
#SBATCH --job-name=t2_ens
#SBATCH --account=sanand
#SBATCH --partition=medium
#SBATCH --qos=medium
#SBATCH --nodelist=B2001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# Final ensemble scoring + packaging. Needs: ens_cands.txt passing the CPU dry run (job_ens_cpu.sbatch).
set -eo pipefail
source "$HOME/indoml-t2/code/omni_env.sh"
export OMP_NUM_THREADS=4
export CANDS="$(cat "$ROOT/ens_cands.txt")" TOP_N=${TOP_N:-2}
nvidia-smi -L
bash "$ROOT/code/run_ensemble.sh"
