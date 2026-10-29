#!/usr/bin/env bash
# For each candidate: pseudo-test (SI-SDR + ASR dWER) and natural-validation dWER, then an estimated REAL
# leaderboard score; package the best for the real test. CANDS lines: "name|floor|ckpt1 ckpt2 ...|tta"
set -euo pipefail
: "${ROOT:?}"; : "${CANDS:?}"
R=$ROOT; PT=$R/data/pt_testlike; V=$R/data/validation; OUT=${OUT_DIR:-$R/final}; mkdir -p $OUT
cd $R/code
declare -A CK FL TT
while IFS='|' read -r name floor ckpts tta; do
  [ -z "$name" ] && continue
  for c in $ckpts; do [ -f "$c" ] || { echo "missing $c"; exit 1; }; done
  CK[$name]="$ckpts"; FL[$name]="$floor"; TT[$name]="${tta:-1}"
done <<< "$CANDS"
pids=()
for c in "${!CK[@]}"; do
  [ -s $OUT/$c.done ] && continue
  ( python enhance.py --ckpts ${CK[$c]} --denoiser_root $R/assets/denoiser --in_dirs $PT/syntheticNoiseAudio --out_dir $OUT/pt_$c --floor ${FL[$c]} --tta ${TT[$c]} \
    && python asr.py --asr_dir $R/assets/sravaani --wav_dir $OUT/pt_$c --out $OUT/pt_${c}_tx.jsonl \
    && python score_val.py --val_dir $PT --split $PT/split.json --enh_dir $OUT/pt_$c --noisy_tx $R/out/pt_testlike_noisy_tx.jsonl --enh_tx $OUT/pt_${c}_tx.jsonl --out $OUT/pt_$c.json \
    && python enhance.py --ckpts ${CK[$c]} --denoiser_root $R/assets/denoiser --in_dirs $R/data/val_hold_syn --out_dir $OUT/hold_$c --floor ${FL[$c]} --tta ${TT[$c]} \
    && python asr.py --asr_dir $R/assets/sravaani --wav_dir $OUT/hold_$c --out $OUT/hold_${c}_tx.jsonl \
    && python enhance.py --ckpts ${CK[$c]} --denoiser_root $R/assets/denoiser --in_dirs $V/naturalNoisyAudio --out_dir $OUT/nat_$c --floor ${FL[$c]} --tta ${TT[$c]} \
    && python asr.py --asr_dir $R/assets/sravaani --wav_dir $OUT/nat_$c --out $OUT/nat_${c}_tx.jsonl && touch $OUT/$c.done ) > $R/logs/final_$c.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait $p || true; }; done
python - "$OUT" "$V" "$R/out/val_noisy_tx.jsonl" <<'P'
import glob, json, os, sys, jiwer
from score_val import normalize, read_jsonl
out, V, ntx = sys.argv[1:]
m = json.load(open(V + "/validationMetadata.json")); gt = {x["segmentFileName"][:-4]: x["transcript"] for x in m if not x.get("syntheticData")}
noisy = read_jsonl(ntx); ids = [c for c in gt if normalize(gt[c])]; R = [normalize(gt[c]) for c in ids]
def wer(h): return jiwer.wer(R, [normalize(h.get(c, "")) or "@" for c in ids])
wn = wer(noisy); rows = []
# synthetic dWER: held-out validation clips scored against ASR(clean) transcripts (matches how the real test
# synthetic subset behaves); calibrated so the submitted ensemble (+6.60 here) maps to its real +5.205
cl = read_jsonl(os.environ["ROOT"] + "/out/clean_hold_tx.jsonl"); hid = [c for c in cl if normalize(cl[c])]; HR = [normalize(cl[c]) for c in hid]
hw = lambda h: jiwer.wer(HR, [normalize(h.get(c, "")) or "@" for c in hid])
hwn = hw(noisy); K = 5.205 / 6.60
for d in glob.glob(out + "/*.done"):
    c = os.path.basename(d)[:-5]; p = json.load(open(f"{out}/pt_{c}.json"))
    nat = 100 * (wn - wer(read_jsonl(f"{out}/nat_{c}_tx.jsonl")))
    hold_d = 100 * (hwn - hw(read_jsonl(f"{out}/hold_{c}_tx.jsonl")))
    syn_real = K * hold_d
    p["dwer_unseen"] = hold_d / 100
    est = (p["sisdr_holdout"] - 0.70) + 0.32 * syn_real + 0.68 * nat
    rows.append((est, c, p["sisdr_holdout"], 100 * p["dwer_unseen"], nat))
rows.sort(reverse=True)
print("estimated REAL leaderboard (submitted ensemble scored 14.46 for reference):")
for est, c, s, pd, nat in rows:
    print(f"  {c:16s} est {est:6.2f} | pseudo SI-SDR {s:6.2f} | holdout dWER vs ASR(clean) {pd:+5.2f} | natural dWER {nat:+5.2f}")
open(out + "/ranking.txt", "w").write("".join(r[1] + "\n" for r in rows))
P
[ "${PACKAGE:-1}" = 1 ] || exit 0
best=$(head -1 $OUT/ranking.txt)
[ -f $OUT/submission_$best.zip ] || {
  python enhance.py --ckpts ${CK[$best]} --denoiser_root $R/assets/denoiser --in_dirs $R/data/test --out_dir $OUT/test_$best --floor ${FL[$best]} --tta ${TT[$best]}
  python asr.py --asr_dir $R/assets/sravaani --wav_dir $OUT/test_$best --out $OUT/test_${best}_tx.jsonl
  python package.py --input_dir $R/data/test --enh_dir $OUT/test_$best --tx $OUT/test_${best}_tx.jsonl --zip $OUT/submission_$best.zip; }
echo "FINAL SELECT DONE: $best"
