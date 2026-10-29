#!/usr/bin/env bash
# Usage: bash scripts/run_inference.sh INPUT_DIR OUTPUT_DIR [submission.zip]
# Requires HF_TOKEN for the gated mandated ASR (ARTPARK-IISc/SraVaani-1.0) on first run.
set -euo pipefail
cd "$(dirname "$0")/.."
python checkpoints/download_checkpoints.py
if [ $# -ge 3 ]; then python inference/infer.py "$1" "$2" --zip "$3"; else python inference/infer.py "$1" "$2"; fi
