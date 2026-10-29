"""Train one stage of the Track 2 enhancer from a YAML config.

    python train/train.py --config configs/base_bsrnn.yaml --bank data/bank --val_dir data/validation \
        --out runs/base_bsrnn --asr_dir models/sravaani

Stage-2 configs (ft_*.yaml) start from the checkpoint named by `init` and select checkpoints on the pseudo-test
(build it with src/make_pseudotest.py; pass --pseudotest data/pt_testlike).
"""
import argparse
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS = ["arch", "minutes", "seconds", "batch", "lr", "workers", "p_ext", "p_valref", "stft_w", "act_w", "gate_floor",
        "asr_w", "asr_every", "feat", "layers", "eval_every_min", "seed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bank", required=True, help="output folder of src/prep_data.py")
    ap.add_argument("--val_dir", required=True, help="extracted validation/ folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr_dir", required=True, help="local copy of ARTPARK-IISc/SraVaani-1.0")
    ap.add_argument("--dns64_ckpt", default=os.path.join(ROOT, "checkpoints", "dns64-a7761ff99a7d5bb6.th"))
    ap.add_argument("--pseudotest", default="", help="pseudo-test folder (required by configs with eval_on_pseudotest)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cmd = [sys.executable, os.path.join(ROOT, "src", "train.py"), "--bank", a.bank, "--val_dir", a.val_dir,
           "--out", a.out, "--asr_dir", a.asr_dir]
    for k in PASS:
        if k in cfg:
            cmd += [f"--{k}", str(cfg[k])]
    for k in ("ev_mult", "db_shift"):
        if k in cfg:
            cmd += [f"--{k}"] + [str(v) for v in cfg[k]]
    if cfg.get("no_vaani"):
        cmd.append("--no_vaani")
    if cfg["arch"] == "demucs":
        cmd += ["--denoiser_root", os.path.join(ROOT, "third_party", "denoiser")]
        if cfg.get("init_pretrained_dns64"):
            cmd += ["--dns64_ckpt", a.dns64_ckpt]
    if cfg.get("init"):
        cmd += ["--init", os.path.join(ROOT, cfg["init"]) if not os.path.isabs(cfg["init"]) else cfg["init"]]
    if cfg.get("eval_on_pseudotest"):
        if not a.pseudotest:
            sys.exit("this config selects checkpoints on the pseudo-test: pass --pseudotest <folder>")
        cmd += ["--eval_dir", a.pseudotest, "--eval_split", os.path.join(a.pseudotest, "split.json")]
    print(" ".join(cmd), flush=True)
    if not a.dry_run:
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
