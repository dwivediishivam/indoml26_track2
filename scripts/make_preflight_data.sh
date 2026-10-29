#!/usr/bin/env bash
# Builds $ROOT/preflight/{validation,test}: a tiny real subset for the PREFLIGHT=1 dry run.
set -euo pipefail
: "${ROOT:?set ROOT}"
P=$ROOT/preflight; V=$ROOT/data/validation
rm -rf "$P"; mkdir -p "$P/validation/naturalNoisyAudio" "$P/validation/syntheticNoiseAudio" "$P/validation/syntheticCleanRefAudio" "$P/test"
cp "$V/validationMetadata.json" "$P/validation/"
nat=("$V"/naturalNoisyAudio/*.wav)
cp "${nat[@]:0:3}" "$P/validation/naturalNoisyAudio/"
mapfile -t hold < <(python -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['holdout'][:3]))" "$ROOT/bank/split.json")
for f in "${hold[@]}"; do
  cp "$V/syntheticNoiseAudio/$f" "$P/validation/syntheticNoiseAudio/"
  cp "$V/syntheticCleanRefAudio/$f" "$P/validation/syntheticCleanRefAudio/"
done
tst=("$ROOT"/data/test/*.wav)
cp "${tst[@]:0:3}" "$P/test/"
echo "preflight data: ${#nat[@]} nat available, ${#hold[@]} syn holdout copied, ${#tst[@]} test clips available"
