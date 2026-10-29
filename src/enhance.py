"""Enhance every WAV in --in_dirs with one or more checkpoints (outputs averaged), write PCM_16.

Output length always equals input length. Long clips are processed in overlapping chunks.
"""
import argparse
import glob
import os

import numpy as np
import torch

from common import SR, read_wav, write_wav_int16
from models import apply_gate, build_model


def load(ckpt, dev, denoiser_root):
    ck = torch.load(ckpt, map_location="cpu")
    a = ck["args"]
    m = build_model(a["arch"], denoiser_root or a.get("denoiser_root", ""), "", a.get("feat", 96), a.get("layers", 6))
    m.load_state_dict(ck["model"])
    return m.to(dev).eval()


@torch.no_grad()
def run_one(model, x, floor, chunk_s=20.0, hop_s=18.0):
    """x: (L,) tensor on device -> (L,) tensor."""
    L = x.shape[0]
    C, H = int(chunk_s * SR), int(hop_s * SR)
    if L <= C:
        enh, p = model(x[None])
        out = enh if (p is None or floor >= 1.0) else apply_gate(x[None], enh, p, floor)
        return out[0].float()
    out = torch.zeros(L, device=x.device)
    wsum = torch.zeros(L, device=x.device)
    ov = C - H
    for s in range(0, L, H):
        seg = x[s:s + C]
        y = run_one(model, seg, floor, chunk_s=1e9)
        w = torch.ones_like(y)
        if s > 0:
            w[:ov] = torch.linspace(0, 1, ov, device=x.device)[: len(w)]
        if s + C < L:
            w[-ov:] = torch.minimum(w[-ov:], torch.linspace(1, 0, ov, device=x.device))
        out[s:s + len(y)] += w * y
        wsum[s:s + len(y)] += w
        if s + C >= L:
            break
    return out / wsum.clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--in_dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--floor", type=float, default=0.2, help=">=1 disables the event gate")
    ap.add_argument("--blend", type=float, default=1.0, help="out = blend*enh + (1-blend)*mix")
    ap.add_argument("--denoiser_root", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tta", type=int, default=1, help="number of shifted passes per model (1 = off)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load(c, dev, a.denoiser_root) for c in a.ckpts]
    os.makedirs(a.out_dir, exist_ok=True)
    files = []
    for d in a.in_dirs:
        files += sorted(glob.glob(os.path.join(d, "*.wav")))
    names = [os.path.basename(f) for f in files]
    assert len(set(names)) == len(names), "duplicate clip names across input dirs"
    if a.limit:
        files = files[: a.limit]
    for i, f in enumerate(files):
        y = read_wav(f)
        x = torch.from_numpy(y).to(dev)
        outs = []
        for m in models:
            for sh in range(a.tta):  # shift-TTA: delay input by sh*37 samples, undo, average
                d = sh * 37
                xs = torch.nn.functional.pad(x, (d, 0)) if d else x
                outs.append(run_one(m, xs, a.floor)[d:d + len(x)])
        e = torch.stack(outs).mean(0)
        e = a.blend * e + (1 - a.blend) * x
        e = e.cpu().numpy().astype(np.float64)
        # restore input loudness (SI-SDR is scale-invariant; ASR prefers natural levels)
        rin, rout = np.sqrt((y.astype(np.float64) ** 2).mean()), np.sqrt((e ** 2).mean()) + 1e-12
        e = e * (rin / rout) if rin > 1e-6 else e
        assert len(e) == len(y)
        write_wav_int16(os.path.join(a.out_dir, os.path.basename(f)), e)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(files)}", flush=True)
    print(f"ENHANCE DONE {len(files)} files -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
