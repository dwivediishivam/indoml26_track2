"""First thing the GPU job runs (~1-2 min). Exits non-zero on any problem so the job dies
before burning hours. Also prints throughput so the training budget can be sanity-checked."""
import argparse
import glob
import os
import sys
import time

import torch

from models import ASRFeatLoss, apply_gate, build_model


def bench(name, model, amp, B=16, L=64000, steps=5):
    dev = "cuda"
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    x = torch.randn(B, L, device=dev) * 0.05
    for i in range(steps + 1):
        if i == 1:
            torch.cuda.synchronize()
            t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            enh, p = model(x)
        out = enh.float() if p is None else apply_gate(x, enh.float(), p, 0.2)
        loss = (out - x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / steps
    print(f"[ok] {name}: {dt * 1000:.0f} ms/step at batch {B}x{L / 16000:.0f}s, "
          f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr_dir", required=True)
    ap.add_argument("--denoiser_root", required=True)
    ap.add_argument("--dns64_ckpt", required=True)
    ap.add_argument("--wav_dir", required=True, help="a folder with a few real WAVs")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("[FAIL] CUDA not available")
    cap = torch.cuda.get_device_capability()
    print(f"[ok] {torch.cuda.get_device_name()} cc{cap} torch {torch.__version__} cuda {torch.version.cuda} "
          f"archs {torch.cuda.get_arch_list()}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bench("bsrnn", build_model("bsrnn").cuda(), amp=True)
    bench("demucs", build_model("demucs", a.denoiser_root, a.dns64_ckpt).cuda(), amp=False)
    asr = ASRFeatLoss(a.asr_dir).cuda()
    m = build_model("bsrnn").cuda()
    x = torch.randn(4, 64000, device="cuda") * 0.05
    enh, _ = m(x)
    l = asr(enh.float(), x)
    l.backward()
    g = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
    if not (g > 0 and torch.isfinite(l)):
        sys.exit("[FAIL] ASR feature loss gave no gradient on GPU")
    print(f"[ok] ASR feature loss on GPU: {float(l):.3f}, grad {g:.1f}", flush=True)
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.asr_dir, trust_remote_code=True).to("cuda").eval()
    wavs = sorted(glob.glob(os.path.join(a.wav_dir, "*.wav")))[:4]
    hyps = model.transcribe(wavs, batch_size=4, return_hypotheses=True)
    if not any(h.text.strip() for h in hyps):
        sys.exit("[FAIL] ASR produced only empty text on GPU")
    print(f"[ok] ASR on GPU: {hyps[0].text[:80]}", flush=True)
    print("GPU CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
