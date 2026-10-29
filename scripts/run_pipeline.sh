#!/usr/bin/env bash
# Full GPU job: gpu check -> (noisy-val ASR || train bsrnn || train demucs) -> candidate eval on
# validation -> pick best by estimated combined score -> enhance test -> ASR -> validated ZIPs.
# Every stage is resumable: rerunning skips finished outputs.
set -euo pipefail
: "${ROOT:?set ROOT}"
TRAIN_MIN=${TRAIN_MIN:-220}
CPUS=${SLURM_CPUS_PER_TASK:-16}
CODE=$ROOT/code
A=$ROOT/assets
VAL=$ROOT/data/validation
TEST=$ROOT/data/test
BANK=$ROOT/bank
RUNS=$ROOT/runs
OUT=$ROOT/out
PREFLIGHT=${PREFLIGHT:-0}
EXTRA_TRAIN=""; EXTRA_SCORE=""; BATCH=16; SECONDS_CROP=${SECONDS_CROP:-6}
TRAIN_BSRNN=${TRAIN_BSRNN:-1}; TRAIN_DEMUCS=${TRAIN_DEMUCS:-1}   # per-machine model selection
BS_ARGS=${BS_ARGS:-}; DM_ARGS=${DM_ARGS:-}                       # per-machine variant flags
if [ "$PREFLIGHT" = 1 ]; then
  # CPU dry run of this exact script on a tiny subset: catches bash/Python bugs before any GPU time.
  P=$ROOT/preflight; VAL=$P/validation; TEST=$P/test; RUNS=$P/runs; OUT=$P/out
  EXTRA_TRAIN="--max_steps 2 --eval_files 2"; EXTRA_SCORE="--allow_missing"; BATCH=2; CPUS=4; TRAIN_MIN=5
fi
mkdir -p "$RUNS" "$OUT" "$ROOT/logs"
cd "$CODE"
PY=python
COMMON_DNS="--denoiser_root $A/denoiser --dns64_ckpt $A/dns64-a7761ff99a7d5bb6.th"
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ "$PREFLIGHT" = 1 ]; then stamp "PREFLIGHT: skipping GPU check"; else
stamp "GPU check"
$PY gpu_check.py --asr_dir "$A/sravaani" $COMMON_DNS --wav_dir "$VAL/naturalNoisyAudio"
fi

# ---------------------------------------------------------------- stage 1: train (+ noisy ASR)
pids=()
if [ ! -s "$OUT/val_noisy_tx.done" ]; then
  ( mkdir -p "$OUT/val_noisy" && ln -sf "$VAL"/naturalNoisyAudio/*.wav "$VAL"/syntheticNoiseAudio/*.wav "$OUT/val_noisy/" \
    && $PY asr.py --asr_dir "$A/sravaani" --wav_dir "$OUT/val_noisy" --out "$OUT/val_noisy_tx.jsonl" \
    && touch "$OUT/val_noisy_tx.done" ) > "$ROOT/logs/asr_noisy.log" 2>&1 &
  pids+=($!); [ "$PREFLIGHT" = 1 ] && wait $! || true
fi
W1=$(( CPUS / 2 > 2 ? CPUS / 2 - 1 : 2 )); W2=$(( CPUS / 4 > 2 ? CPUS / 4 : 2 ))
if [ "$TRAIN_BSRNN" = 1 ] && [ ! -f "$RUNS/bsrnn/TRAINING_DONE" ]; then
  ( $PY train.py --arch bsrnn --bank "$BANK" --val_dir "$VAL" --out "$RUNS/bsrnn" --minutes "$TRAIN_MIN" \
      --batch $BATCH --seconds $SECONDS_CROP --lr 1e-3 --workers $W1 $EXTRA_TRAIN --asr_dir "$A/sravaani" --asr_w 0.1 --asr_every 2 $BS_ARGS \
    && touch "$RUNS/bsrnn/TRAINING_DONE" ) > "$ROOT/logs/train_bsrnn.log" 2>&1 &
  pids+=($!); [ "$PREFLIGHT" = 1 ] && wait $! || true
fi
if [ "$TRAIN_DEMUCS" = 1 ] && [ ! -f "$RUNS/demucs/TRAINING_DONE" ]; then
  ( $PY train.py --arch demucs $COMMON_DNS --bank "$BANK" --val_dir "$VAL" --out "$RUNS/demucs" --minutes "$TRAIN_MIN" \
      --batch $BATCH --seconds $SECONDS_CROP --lr 1e-4 --stft_w 0.5 --workers $W2 $EXTRA_TRAIN --asr_dir "$A/sravaani" --asr_w 0.1 --asr_every 2 $DM_ARGS \
    && touch "$RUNS/demucs/TRAINING_DONE" ) > "$ROOT/logs/train_demucs.log" 2>&1 &
  pids+=($!); [ "$PREFLIGHT" = 1 ] && wait $! || true
fi
stamp "stage 1 running: pids ${pids[*]:-none} (logs in $ROOT/logs)"
fail=0
for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || fail=1; }; done
stamp "stage 1 finished"
for m in $([ "$TRAIN_BSRNN" = 1 ] && echo bsrnn) $([ "$TRAIN_DEMUCS" = 1 ] && echo demucs); do
  [ -f "$RUNS/$m/TRAINING_DONE" ] || { stamp "WARNING: $m training did not finish (see $ROOT/logs/train_$m.log)"; [ "$PREFLIGHT" = 1 ] && exit 1; }
done
[ -f "$OUT/val_noisy_tx.done" ] || { stamp "noisy ASR failed"; exit 1; }

# ---------------------------------------------------------------- stage 2: candidates on validation
declare -A CK FL
[ -f "$RUNS/bsrnn/best.pt" ] && { CK[bs_f02]="$RUNS/bsrnn/best.pt"; FL[bs_f02]=0.2
                                  CK[bs_f05]="$RUNS/bsrnn/best.pt"; FL[bs_f05]=0.5
                                  CK[bs_raw]="$RUNS/bsrnn/best.pt"; FL[bs_raw]=1.0; }
[ -f "$RUNS/demucs/best.pt" ] && { CK[dm]="$RUNS/demucs/best.pt"; FL[dm]=1.0; }
[ -f "$RUNS/bsrnn/best.pt" ] && [ -f "$RUNS/demucs/best.pt" ] && { CK[ens_f02]="$RUNS/bsrnn/best.pt $RUNS/demucs/best.pt"; FL[ens_f02]=0.2; }
[ ${#CK[@]} -gt 0 ] || { stamp "no trained checkpoints"; exit 1; }
pids=()
for c in "${!CK[@]}"; do
  [ -s "$OUT/val_$c.json" ] && continue
  ( $PY enhance.py --ckpts ${CK[$c]} --denoiser_root "$A/denoiser" --in_dirs "$VAL/naturalNoisyAudio" "$VAL/syntheticNoiseAudio" \
        --out_dir "$OUT/val_$c" --floor "${FL[$c]}" \
    && $PY asr.py --asr_dir "$A/sravaani" --wav_dir "$OUT/val_$c" --out "$OUT/val_${c}_tx.jsonl" \
    && $PY score_val.py --val_dir "$VAL" --split "$BANK/split.json" --enh_dir "$OUT/val_$c" \
        --noisy_tx "$OUT/val_noisy_tx.jsonl" --enh_tx "$OUT/val_${c}_tx.jsonl" --out "$OUT/val_$c.json" $EXTRA_SCORE ) \
    > "$ROOT/logs/val_$c.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || true; }; done
stamp "validation results:"
$PY - "$OUT" <<'EOF'
import glob, json, os, sys
rows = []
for p in glob.glob(os.path.join(sys.argv[1], "val_*.json")):
    r = json.load(open(p)); rows.append((r.get("combined_est", -1e9), os.path.basename(p)[4:-5], r))
rows.sort(reverse=True)
for s, n, r in rows:
    print(f"{n:10s} combined {s:7.3f}  sisdr_holdout {r['sisdr_holdout']:7.3f}  dwer_unseen {100 * r.get('dwer_unseen', 0):+.2f} pts  dwer_all {100 * r.get('dwer_all', 0):+.2f}")
open(os.path.join(sys.argv[1], "ranking.txt"), "w").write("".join(n + "\n" for _, n, _ in rows))
EOF

# ---------------------------------------------------------------- stage 3: test submissions (top 2)
n=0
while read -r c || [ -n "$c" ]; do
  n=$((n + 1)); [ $n -gt 2 ] && break
  [ -f "$OUT/submission_$c.zip" ] && continue
  stamp "test submission for $c"
  $PY enhance.py --ckpts ${CK[$c]} --denoiser_root "$A/denoiser" --in_dirs "$TEST" --out_dir "$OUT/test_$c" --floor "${FL[$c]}"
  $PY asr.py --asr_dir "$A/sravaani" --wav_dir "$OUT/test_$c" --out "$OUT/test_${c}_tx.jsonl"
  $PY package.py --input_dir "$TEST" --enh_dir "$OUT/test_$c" --tx "$OUT/test_${c}_tx.jsonl" --zip "$OUT/submission_$c.zip"
done < "$OUT/ranking.txt"
stamp "ALL DONE"; ls -la "$OUT"/submission_*.zip
