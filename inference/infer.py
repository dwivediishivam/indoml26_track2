"""Track 2 inference: noisy WAVs -> enhanced WAVs + transcripts.jsonl (Phase 2 Codabench format).

    python inference/infer.py INPUT_DIR OUTPUT_DIR [--zip submission.zip]

INPUT_DIR holds the 16 kHz mono test WAVs (an `audio/` sub-folder, as in the Codabench input_data ZIP, is also
accepted). OUTPUT_DIR receives one PCM-16 WAV per input clip with the same name and length, plus
`transcripts.jsonl` produced by the unmodified mandated ASR ARTPARK-IISc/SraVaani-1.0 on those exact WAVs.
Checkpoints are downloaded from Hugging Face if they are not already in checkpoints/.
"""
import argparse
import glob
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def resolve_input(d):
    if glob.glob(os.path.join(d, "*.wav")):
        return d
    sub = os.path.join(d, "audio")
    if glob.glob(os.path.join(sub, "*.wav")):
        return sub
    sys.exit(f"no .wav files found in {d} or {sub}")


def ensure_checkpoints(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.path.insert(0, os.path.join(ROOT, "checkpoints"))
        from download_checkpoints import download
        download([os.path.basename(p) for p in missing])


def ensure_asr(asr_dir, repo):
    if asr_dir and os.path.isdir(asr_dir):
        return asr_dir
    from huggingface_hub import snapshot_download
    print(f"downloading {repo} (gated: accept its terms on Hugging Face and set HF_TOKEN)", flush=True)
    return snapshot_download(repo, local_dir=asr_dir or os.path.join(ROOT, "models", "sravaani"),
                             token=os.environ.get("HF_TOKEN"))


def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.call(cmd)
    if r:
        sys.exit(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "inference.yaml"))
    ap.add_argument("--asr_dir", default=os.path.join(ROOT, "models", "sravaani"),
                    help="local SraVaani-1.0 folder (downloaded here if absent)")
    ap.add_argument("--zip", default="", help="also write a validated Codabench submission ZIP here")
    ap.add_argument("--skip_asr", action="store_true", help="only write enhanced WAVs")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    inp = resolve_input(os.path.abspath(a.input_dir))
    out = os.path.abspath(a.output_dir)
    ckpts = [os.path.join(ROOT, m) for m in cfg["members"]]
    ensure_checkpoints(ckpts)
    py = sys.executable
    run([py, os.path.join(SRC, "enhance.py"), "--ckpts", *ckpts, "--in_dirs", inp, "--out_dir", out,
         "--floor", str(cfg["gate_floor"]), "--tta", str(cfg.get("tta", 1)),
         "--denoiser_root", os.path.join(ROOT, "third_party", "denoiser")])
    if a.skip_asr:
        return
    asr_dir = ensure_asr(a.asr_dir, cfg["asr_model"])
    tx = os.path.join(out, "transcripts.jsonl")
    run([py, os.path.join(SRC, "asr.py"), "--asr_dir", asr_dir, "--wav_dir", out, "--out", tx,
         "--batch", str(cfg.get("asr_batch", 4))])
    if a.zip:
        run([py, os.path.join(SRC, "package.py"), "--input_dir", inp, "--enh_dir", out, "--tx", tx, "--zip", a.zip])


if __name__ == "__main__":
    main()
