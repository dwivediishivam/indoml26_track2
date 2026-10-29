#!/usr/bin/env bash
# Final stage: score ensemble candidates on validation (official SI-SDR on the holdout + mandated-ASR dWER),
# rank by estimated combined score, then enhance the test set with the top N and write validated ZIPs.
# CANDS: one candidate per line, "name|gate_floor|ckpt1 ckpt2 ..." (floor applies to BSRNN members only).
set -euo pipefail
: "${ROOT:?set ROOT}"; : "${CANDS:?set CANDS}"
TOP_N=${TOP_N:-2}
PREFLIGHT=${PREFLIGHT:-0}
A=$ROOT/assets; VAL=${VAL_DIR:-$ROOT/data/validation}; TEST=$ROOT/data/test; SPLIT=${SPLIT:-$ROOT/bank/split.json}
OUT=${OUT_DIR:-$ROOT/ens_out}; EXTRA_SCORE=""; SKIP_TEST=${SKIP_TEST:-0}
if [ "$PREFLIGHT" = 1 ]; then VAL=$ROOT/preflight/validation; TEST=$ROOT/preflight/test; OUT=$ROOT/preflight/ens_out; EXTRA_SCORE="--allow_missing"; fi
NOISY_TX=${NOISY_TX:-$ROOT/out/val_noisy_tx.jsonl}
[ "$PREFLIGHT" = 1 ] && NOISY_TX=$ROOT/preflight/out/val_noisy_tx.jsonl
mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT/code"
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
[ -s "$NOISY_TX" ] || { stamp "missing noisy-validation transcripts $NOISY_TX"; exit 1; }

declare -A CK FL
declare -A TT
while IFS='|' read -r name floor ckpts tta; do
  [ -z "$name" ] && continue
  for c in $ckpts; do [ -f "$c" ] || { stamp "missing checkpoint $c for $name"; exit 1; }; done
  CK[$name]="$ckpts"; FL[$name]="$floor"; TT[$name]="${tta:-1}"
done <<< "$CANDS"
stamp "candidates: ${!CK[*]}"

pids=()
for c in "${!CK[@]}"; do
  [ -s "$OUT/val_$c.json" ] && continue
  ( python enhance.py --ckpts ${CK[$c]} --denoiser_root "$A/denoiser" --in_dirs "$VAL/naturalNoisyAudio" "$VAL/syntheticNoiseAudio" \
        --out_dir "$OUT/val_$c" --floor "${FL[$c]}" --tta "${TT[$c]}" \
    && python asr.py --asr_dir "$A/sravaani" --wav_dir "$OUT/val_$c" --out "$OUT/val_${c}_tx.jsonl" \
    && python score_val.py --val_dir "$VAL" --split "$SPLIT" --enh_dir "$OUT/val_$c" \
        --noisy_tx "$NOISY_TX" --enh_tx "$OUT/val_${c}_tx.jsonl" --out "$OUT/val_$c.json" $EXTRA_SCORE ) \
    > "$ROOT/logs/ens_val_$c.log" 2>&1 &
  pids+=($!)
  [ "$PREFLIGHT" = 1 ] && { wait $! || true; }
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || true; }; done
stamp "validation results:"
python - "$OUT" <<'EOF'
import glob, json, os, sys
rows = []
for p in glob.glob(os.path.join(sys.argv[1], "val_*.json")):
    r = json.load(open(p)); rows.append((r.get("combined_est", -1e9), os.path.basename(p)[4:-5], r))
rows.sort(reverse=True)
for s, n, r in rows:
    print(f"{n:14s} combined {s:7.3f}  sisdr_holdout {r['sisdr_holdout']:7.3f}  dwer_unseen {100 * r.get('dwer_unseen', 0):+.2f} pts  dwer_all {100 * r.get('dwer_all', 0):+.2f}")
open(os.path.join(sys.argv[1], "ranking.txt"), "w").write("".join(n + "\n" for _, n, _ in rows))
EOF

[ "$SKIP_TEST" = 1 ] && { stamp "SKIP_TEST=1: no test packaging"; exit 0; }
n=0
while read -r c || [ -n "$c" ]; do
  n=$((n + 1)); [ $n -gt "$TOP_N" ] && break
  [ -f "$OUT/submission_$c.zip" ] && continue
  stamp "test submission for $c"
  python enhance.py --ckpts ${CK[$c]} --denoiser_root "$A/denoiser" --in_dirs "$TEST" --out_dir "$OUT/test_$c" --floor "${FL[$c]}" --tta "${TT[$c]}"
  python asr.py --asr_dir "$A/sravaani" --wav_dir "$OUT/test_$c" --out "$OUT/test_${c}_tx.jsonl"
  python package.py --input_dir "$TEST" --enh_dir "$OUT/test_$c" --tx "$OUT/test_${c}_tx.jsonl" --zip "$OUT/submission_$c.zip"
done < "$OUT/ranking.txt"
stamp "ENSEMBLE DONE"; ls -la "$OUT"/submission_*.zip
