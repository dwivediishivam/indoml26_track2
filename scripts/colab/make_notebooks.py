"""Generates colab_A.ipynb / colab_B.ipynb (same steps, different model variant)."""
import json
import sys

HF_TOKEN = sys.argv[1]
TEST_GDRIVE_ID = "12aKe2XmOFIZsfY4yuKjoyBCD60fT_ryG"  # organizer Drive: input_data_track-2_phase-2.zip

VARIANTS = {
    "A": dict(desc="BSRNN, wider (feat 112), seed 1, 6 s crops", TRAIN_BSRNN="1", TRAIN_DEMUCS="0",
              BS_ARGS="--feat 112 --seed 1", DM_ARGS="", SECONDS_CROP="6", TRAIN_MIN="180",
              FLEURS="hi_in ml_in mr_in or_in as_in gu_in pa_in"),
    "B": dict(desc="DNS64 fine-tune, 8 s crops, stronger ASR loss, seed 1", TRAIN_BSRNN="0", TRAIN_DEMUCS="1",
              BS_ARGS="", DM_ARGS="--seed 1 --asr_w 0.2 --lr 2e-4", SECONDS_CROP="8", TRAIN_MIN="150",
              FLEURS="hi_in bn_in te_in ta_in kn_in ne_np"),
}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def notebook(v, cfg):
    cells = [
        md(f"# Track 2 Phase 2 — Colab {v}: {cfg['desc']}\n\n"
           "Runtime → Change runtime type → **A100 GPU** (High-RAM if offered). Then run cells top to bottom.\n"
           "Before running: upload `t2_colab_pack.zip` to **My Drive / indoml_t2/**.\n\n"
           "Checkpoints and logs are written to Drive, so a disconnect loses nothing: re-run all cells and "
           "training resumes from `last.pt`.\n\n**Do not share this notebook: it contains an HF token.**"),
        code(f"""# 1. Config
import os
VARIANT = "{v}"
os.environ.update(dict(
    HF_TOKEN="{HF_TOKEN}",
    TEST_GDRIVE_ID="{TEST_GDRIVE_ID}",
    ROOT="/content/t2",
    DRIVE_DIR="/content/drive/MyDrive/indoml_t2",
    VARIANT=VARIANT,
    TRAIN_BSRNN="{cfg['TRAIN_BSRNN']}", TRAIN_DEMUCS="{cfg['TRAIN_DEMUCS']}",
    BS_ARGS="{cfg['BS_ARGS']}", DM_ARGS="{cfg['DM_ARGS']}",
    SECONDS_CROP="{cfg['SECONDS_CROP']}",
    TRAIN_MIN="{cfg['TRAIN_MIN']}",          # training wall-clock minutes (eval + test + ASR adds ~45 min)
    FLEURS_LANGS="{cfg['FLEURS']}",
    HF_HUB_ENABLE_HF_TRANSFER="0",
))
print({{k: os.environ[k] for k in ["VARIANT", "TRAIN_BSRNN", "TRAIN_DEMUCS", "BS_ARGS", "DM_ARGS", "SECONDS_CROP", "TRAIN_MIN"]}})
"""),
        code("""# 2. GPU + Drive
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
import os
assert os.path.exists(os.environ["DRIVE_DIR"] + "/t2_colab_pack.zip"), "upload t2_colab_pack.zip to MyDrive/indoml_t2/ first"
print("pack found")
"""),
        code("""%%bash
# 3. Setup: packages + unpack code/validation/denoiser. runs/ and logs/ live on Drive.
set -euo pipefail
pip -q install "transformers>=4.44,<5" "huggingface_hub>=0.25,<1" "jiwer>=3,<5" sentencepiece julius gdown soundfile pyarrow 2>&1 | tail -2
mkdir -p $ROOT/assets $ROOT/data/test $ROOT/data/fleurs $ROOT/data/esc50
cd $ROOT && unzip -q -o $DRIVE_DIR/t2_colab_pack.zip
unzip -q -o pack/validation.zip -x '__MACOSX/*' -d data/
cp -r pack/code $ROOT/ && cp -r pack/denoiser $ROOT/assets/ && cp pack/dns64-a7761ff99a7d5bb6.th $ROOT/assets/
mkdir -p $DRIVE_DIR/run_$VARIANT/runs $DRIVE_DIR/run_$VARIANT/logs
ln -sfn $DRIVE_DIR/run_$VARIANT/runs $ROOT/runs
ln -sfn $DRIVE_DIR/run_$VARIANT/logs $ROOT/logs
python -c "import torch, transformers, jiwer, sentencepiece, julius; print('ok torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
echo "val: $(ls $ROOT/data/validation/naturalNoisyAudio | wc -l) nat, $(ls $ROOT/data/validation/syntheticNoiseAudio | wc -l) syn"
"""),
        code("""%%bash
# 4. Downloads (parallel): mandated ASR, Phase 2 test, ESC-50, FLEURS clean speech
set -uo pipefail
cd $ROOT
( python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download("ARTPARK-IISc/SraVaani-1.0", local_dir=os.environ["ROOT"] + "/assets/sravaani", token=os.environ["HF_TOKEN"])
print("ASR OK")
EOF
) > dl_asr.log 2>&1 &
( gdown -q $TEST_GDRIVE_ID -O test.zip && unzip -q -o test.zip -d test_raw && mv test_raw/audio/*.wav data/test/ && rm -rf test_raw test.zip && echo "TEST OK" ) > dl_test.log 2>&1 &
( curl -sL -o esc.zip https://github.com/karolpiczak/ESC-50/archive/master.zip && unzip -q -o esc.zip -d data/esc50 && rm esc.zip && echo "ESC OK" ) > dl_esc.log 2>&1 &
( python - <<'EOF'
import os
from huggingface_hub import hf_hub_download
for l in os.environ["FLEURS_LANGS"].split():
    try:
        p = hf_hub_download("google/fleurs", f"parquet-data/{l}/train-00000-of-00001.parquet", repo_type="dataset",
                            local_dir=os.environ["ROOT"] + "/data/fleurs", token=os.environ["HF_TOKEN"])
        print("OK", p, flush=True)
    except Exception as e:
        print("FAIL", l, str(e)[:100], flush=True)
EOF
) > dl_fleurs.log 2>&1 &
wait
tail -n 2 dl_asr.log dl_test.log dl_esc.log dl_fleurs.log
echo "test clips: $(ls data/test | wc -l) (expect 3178)"
python - <<'EOF'
import hashlib, os
h = hashlib.md5(open(os.environ["ROOT"] + "/assets/sravaani/model-asr.fp16.ts", "rb").read()).hexdigest()
print("ASR md5", h, "OK (identical to audited copy)" if h == "c1b642165a7489a066c18c5dd21147d7" else "MISMATCH!")
EOF
"""),
        code("""%%bash
# 5. Build training banks (same holdout split as the B200 run)
set -euo pipefail
cd $ROOT/code
python prep_data.py --val_dir $ROOT/data/validation --out $ROOT/bank --split $ROOT/pack/split.json --workers $(nproc) \\
  --esc50_dir $ROOT/data/esc50/ESC-50-master/audio \\
  --extra_clean_glob "$ROOT/data/fleurs/parquet-data/*/*.parquet" --max_extra_hours 12 2>&1 | grep -v "parquet ok"
"""),
        code("""%%bash
# 6. Dry run of the exact pipeline on a tiny subset (~3-5 min). Must end with ALL DONE.
set -euo pipefail
cd $ROOT/code
export SLURM_CPUS_PER_TASK=$(nproc)
bash make_preflight_data.sh
PREFLIGHT=1 bash run_pipeline.sh 2>&1 | grep -v -i -E "warn|detach|forward_call" | grep -E "^\\[|combined|PACKAGE|rror|ALL DONE"
"""),
        code("""# 7. Launch the real run in the background (survives cell interrupts; resumes if re-run)
import subprocess, os
os.environ["SLURM_CPUS_PER_TASK"] = str(os.cpu_count())
log = open(os.environ["ROOT"] + "/logs/pipeline.log", "a")
p = subprocess.Popen(["bash", os.environ["ROOT"] + "/code/run_pipeline.sh"], stdout=log, stderr=subprocess.STDOUT,
                     cwd=os.environ["ROOT"] + "/code", start_new_session=True)
print("pipeline pid", p.pid)
"""),
        code("""%%bash
# 8. Monitor (re-run any time)
cd $ROOT/logs
grep -v -i -E "warn|detach|forward_call" pipeline.log | tail -n 8
for f in train_*.log; do echo "== $f"; grep -E "EVAL|init|FINAL" $f | tail -n 3; grep "^step" $f | tail -n 1; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
"""),
        code("""%%bash
# 9. When pipeline.log shows ALL DONE: copy results to Drive (named by variant)
set -euo pipefail
mkdir -p $DRIVE_DIR/results_$VARIANT
cp $ROOT/out/ranking.txt $ROOT/out/val_*.json $DRIVE_DIR/results_$VARIANT/
for z in $ROOT/out/submission_*.zip; do cp "$z" "$DRIVE_DIR/results_$VARIANT/${VARIANT}_$(basename $z)"; done
ls -la $DRIVE_DIR/results_$VARIANT $DRIVE_DIR/run_$VARIANT/runs/*/best.pt
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 0}


for v, cfg in VARIANTS.items():
    json.dump(notebook(v, cfg), open(f"colab_{v}.ipynb", "w"), indent=1)
    print("wrote", f"colab_{v}.ipynb")
