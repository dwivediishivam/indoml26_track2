#!/usr/bin/env bash
# Download the external training data into DATA (default ./data): ESC-50 and Google FLEURS Indic train shards.
# The IndoML validation set comes from the competition page; Vaani from ARTPARK-IISc/Vaani-Noise-Event-Dataset.
# Usage: HF_TOKEN=... bash scripts/fetch_data.sh [DATA]
set -euo pipefail
D=${1:-data}; mkdir -p "$D/esc50" "$D/fleurs" "$D/vaani"
if [ ! -d "$D/esc50/ESC-50-master/audio" ]; then
  curl -L -o "$D/esc50/esc50.zip" https://github.com/karolpiczak/ESC-50/archive/master.zip
  unzip -q -o "$D/esc50/esc50.zip" -d "$D/esc50" && rm "$D/esc50/esc50.zip"
fi
python - "$D" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download
D = sys.argv[1]; tok = os.environ.get("HF_TOKEN")
for l in ["hi_in", "bn_in", "te_in", "ta_in", "kn_in", "ml_in", "mr_in", "or_in", "as_in", "gu_in", "pa_in", "ne_np"]:
    hf_hub_download("google/fleurs", f"parquet-data/{l}/train-00000-of-00001.parquet", repo_type="dataset",
                    local_dir=os.path.join(D, "fleurs"), token=tok)
    print("FLEURS", l, flush=True)
snapshot_download("ARTPARK-IISc/Vaani-Noise-Event-Dataset", repo_type="dataset", local_dir=os.path.join(D, "vaani"),
                  allow_patterns=["data/*.parquet"], token=tok)
snapshot_download("ARTPARK-IISc/SraVaani-1.0", local_dir="models/sravaani", token=tok)
print("done")
PY
