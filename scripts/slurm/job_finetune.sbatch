#!/usr/bin/env bash
#SBATCH --job-name=t2_ft
#SBATCH --account=sanand
#SBATCH --partition=medium
#SBATCH --qos=medium
#SBATCH --nodelist=B2001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# Fine-tune the two best models on test-like mixtures (long crops, denser + louder events), select checkpoints
# on the pseudo-test (holdout-only), rank candidates on the pseudo-test, package the best for the real test.
set -eo pipefail
source "$HOME/indoml-t2/code/omni_env.sh"
R=$ROOT; PT=$R/data/pt_testlike; FT=$R/runs_ft; mkdir -p $FT
FT_MIN=${FT_MIN:-50}
cd $R/code
nvidia-smi -L
COMMON="--bank $R/bank --val_dir $R/data/validation --eval_dir $PT --eval_split $PT/split.json --minutes $FT_MIN \
  --ev_mult 1 2 --db_shift 0 3 --eval_every_min 10 --asr_dir $R/assets/sravaani --asr_w 0.1 --asr_every 2"
( python train.py --arch demucs --denoiser_root $R/assets/denoiser --init $R/runs/demucs/best.pt --out $FT/demucs \
    --batch 16 --seconds 10 --lr 5e-5 --workers 6 $COMMON && touch $FT/demucs/TRAINING_DONE ) > $R/logs/ft_demucs.log 2>&1 &
( python train.py --arch bsrnn --feat 112 --init $R/colab_ckpts/colabA_bsrnn.pt --out $FT/bsrnn --no_vaani \
    --batch 8 --seconds 10 --lr 2e-4 --workers 6 $COMMON && touch $FT/bsrnn/TRAINING_DONE ) > $R/logs/ft_bsrnn.log 2>&1 &
wait || true
grep -h -E "init holdout|EVAL|FINAL" $R/logs/ft_demucs.log $R/logs/ft_bsrnn.log
DM=$R/runs/demucs/best.pt; BA=$R/colab_ckpts/colabA_bsrnn.pt; DB=$R/colab_ckpts/colabB_demucs.pt
DMF=$FT/demucs/best.pt; BAF=$FT/bsrnn/best.pt
C="ens_dm_bsA|0.2|$DM $BA"
[ -f $DMF ] && C="$C
dm_ft|1.0|$DMF"
[ -f $BAF ] && C="$C
bsA_ft|0.2|$BAF"
[ -f $DMF ] && [ -f $BAF ] && C="$C
ens_ft|0.2|$DMF $BAF
ens_ft_dmB|0.2|$DMF $DB $BAF"
echo "$C"
CANDS="$C" VAL_DIR=$PT SPLIT=$PT/split.json NOISY_TX=$R/out/pt_testlike_noisy_tx.jsonl OUT_DIR=$R/ens_ft TOP_N=1 bash run_ensemble.sh
echo "FINETUNE JOB DONE"
