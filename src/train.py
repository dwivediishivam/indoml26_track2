"""Train an in-domain enhancer with on-the-fly event mixing.

The LR schedule is driven by wall-clock time (--minutes), so the run always ends inside the
Slurm allocation, saving last.pt / best.pt. It resumes from last.pt if present.
Holdout evaluation uses the official SI-SDR on full-length synthetic validation clips.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import SR, load_json, read_wav, save_json, si_sdr
from models import HOP, ASRFeatLoss, apply_gate, build_model


# ----------------------------------------------------------------------------- data
class Banks:
    def __init__(self, d):
        self.noise = np.load(os.path.join(d, "noise_bank.npy"), mmap_mode="r")
        self.clean = np.load(os.path.join(d, "clean_bank.npy"), mmap_mode="r")
        self.nidx = load_json(os.path.join(d, "noise_index.json"))
        self.cidx = load_json(os.path.join(d, "clean_index.json"))
        st = load_json(os.path.join(d, "stats.json"))
        self.ev_db = np.array(st["event_db"], np.float32)
        self.ev_dur = np.array(st["event_dur"], np.float32)
        self.n_ev = np.array(st["n_events"], np.int64)
        self.n_by = {s: [i for i, x in enumerate(self.nidx) if x[2] == s] for s in {x[2] for x in self.nidx}}
        self.c_by = {s: [i for i, x in enumerate(self.cidx) if x[2] == s] for s in {x[2] for x in self.cidx}}


class MixDataset(torch.utils.data.IterableDataset):
    def __init__(self, bank_dir, seconds, p_ext, p_valref, seed, ev_mult=(1.0, 1.0), db_shift=(0.0, 0.0), no_vaani=False):
        self.bank_dir, self.L, self.p_ext, self.p_valref, self.seed = bank_dir, int(seconds * SR), p_ext, p_valref, seed
        self.ev_mult, self.db_shift, self.no_vaani = ev_mult, db_shift, no_vaani
        self.b = None

    def _clean(self, rng):
        b = self.b
        others = [k for k in ("vaani", "extra") if k in b.c_by and not (self.no_vaani and k == "vaani")]
        src = "valref" if (not others or rng.random() < self.p_valref) else rng.choice(others)
        out = np.zeros(self.L, np.float32)
        pos = rng.randrange(int(0.3 * SR))  # leading silence
        while pos < self.L:  # join utterances (rescaled to a common level) until the crop is full
            o, n, _ = b.cidx[rng.choice(b.c_by[src])]
            x = np.asarray(b.clean[o:o + n], np.float32)
            if pos == 0 and n > self.L:
                s = rng.randrange(n - self.L + 1)
                return x[s:s + self.L].copy()
            x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 10 ** (rng.uniform(-3, 3) / 20)
            m = min(n, self.L - pos)
            out[pos:pos + m] = x[:m]
            pos += m + rng.randrange(int(0.05 * SR), int(0.6 * SR))
        return out

    def _noise(self, rng):
        b = self.b
        src = "esc50" if ("esc50" in b.n_by and rng.random() < self.p_ext) else "valres"
        o, n, _ = b.nidx[rng.choice(b.n_by[src])]
        x = np.asarray(b.noise[o:o + n], np.float32)
        if src == "esc50":  # crop to an empirical event duration, fade edges
            d = int(max(0.15, float(rng.choice(b.ev_dur))) * SR)
            if n > d:
                s = rng.randrange(n - d + 1)
                x = x[s:s + d]
            f = min(80, len(x) // 4)
            if f > 0:
                ramp = np.linspace(0, 1, f, dtype=np.float32)
                x[:f] *= ramp
                x[-f:] *= ramp[::-1]
        if rng.random() < 0.5:
            x = -x
        return x[: self.L]

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 7919 + int(time.time()))
        if self.b is None:
            self.b = Banks(self.bank_dir)
        b = self.b
        frames = self.L // HOP + 1
        while True:
            c = self._clean(rng)
            crms = math.sqrt(float((c ** 2).mean())) + 1e-9
            if crms < 1e-4:
                continue
            g = 10 ** (rng.uniform(-30, -15) / 20) / crms  # random level
            c *= g
            crms *= g
            noise = np.zeros(self.L, np.float32)
            act = np.zeros(frames, np.float32)
            # validation clips are ~4.5 s; test clips are ~12 s: scale event count with crop length
            k = max(1, int(round(int(rng.choice(b.n_ev)) * rng.uniform(1.0, max(1.0, self.L / (4.5 * SR)))
                                 * rng.uniform(*self.ev_mult))))
            shift = rng.uniform(*self.db_shift)
            for _ in range(k):
                x = self._noise(rng)
                xr = math.sqrt(float((x ** 2).mean())) + 1e-9
                db = float(rng.choice(b.ev_db)) + rng.gauss(0, 2.0) + shift
                x = x * (crms * 10 ** (db / 20) / xr)
                s = rng.randrange(max(1, self.L - len(x) + 1))
                noise[s:s + len(x)] += x
                act[s // HOP: (s + len(x)) // HOP + 1] = 1.0
            mix = c + noise
            pk = np.abs(mix).max()
            if pk > 0.99:
                mix, c = mix * (0.99 / pk), c * (0.99 / pk)
            yield torch.from_numpy(mix), torch.from_numpy(c), torch.from_numpy(act)


# ----------------------------------------------------------------------------- losses
def neg_si_sdr(est, ref, eps=1e-8):
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    t = a * ref
    e = est - t
    return -(10 * torch.log10(t.pow(2).sum(-1) / (e.pow(2).sum(-1) + eps) + eps)).mean()


def mrstft(est, ref):
    loss = 0.0
    s = ref.pow(2).mean(-1, keepdim=True).sqrt() + 1e-5
    est, ref = est / s, ref / s
    for n, h in ((256, 64), (512, 128), (1024, 256)):
        w = torch.hann_window(n, device=est.device)
        E = torch.stft(est, n, h, window=w, return_complex=True).abs()
        R = torch.stft(ref, n, h, window=w, return_complex=True).abs()
        loss = loss + (torch.log(E + 1e-5) - torch.log(R + 1e-5)).abs().mean()
    return loss / 3


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, val_dir, files, dev, floors, max_files=0):
    model.eval()
    res = {f"floor{fl}": [] for fl in floors}
    res["ungated"] = []
    for f in files[: max_files or None]:
        y = read_wav(os.path.join(val_dir, "syntheticNoiseAudio", f))
        c = read_wav(os.path.join(val_dir, "syntheticCleanRefAudio", f))
        x = torch.from_numpy(y).to(dev)[None]
        enh, p = model(x)
        res["ungated"].append(si_sdr(c, enh[0].float().cpu().numpy()))
        for fl in floors:
            out = enh if p is None else apply_gate(x, enh, p, fl)
            res[f"floor{fl}"].append(si_sdr(c, out[0].float().cpu().numpy()))
    model.train()
    return {k: float(np.mean(v)) for k, v in res.items()}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["bsrnn", "demucs"], required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, required=True, help="training wall-clock budget")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--p_ext", type=float, default=0.3, help="prob. a noise event comes from ESC-50")
    ap.add_argument("--p_valref", type=float, default=0.5, help="prob. clean speech is a validation ref")
    ap.add_argument("--stft_w", type=float, default=0.5)
    ap.add_argument("--act_w", type=float, default=1.0)
    ap.add_argument("--gate_floor", type=float, default=0.2)
    ap.add_argument("--asr_dir", default="")
    ap.add_argument("--asr_w", type=float, default=0.0)
    ap.add_argument("--asr_every", type=int, default=2)
    ap.add_argument("--denoiser_root", default="")
    ap.add_argument("--dns64_ckpt", default="")
    ap.add_argument("--feat", type=int, default=96)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--eval_every_min", type=float, default=20)
    ap.add_argument("--eval_files", type=int, default=0, help="0 = whole holdout")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ev_mult", type=float, nargs=2, default=[1.0, 1.0], help="uniform range multiplying events/clip")
    ap.add_argument("--db_shift", type=float, nargs=2, default=[0.0, 0.0], help="uniform range added to event loudness")
    ap.add_argument("--no_vaani", action="store_true", help="exclude Vaani segments from clean speech")
    ap.add_argument("--init", default="", help="initialise model weights from this checkpoint (fresh optimizer/schedule)")
    ap.add_argument("--eval_dir", default="", help="evaluate on this dir (validation layout) instead of --val_dir")
    ap.add_argument("--eval_split", default="", help="split.json whose holdout list is evaluated")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    amp = dev.type == "cuda" and a.arch == "bsrnn"

    model = build_model(a.arch, a.denoiser_root, a.dns64_ckpt, a.feat, a.layers).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    asr = ASRFeatLoss(a.asr_dir).to(dev) if a.asr_w > 0 else None

    hold = load_json(a.eval_split or os.path.join(a.bank, "split.json"))["holdout"]
    eval_dir = a.eval_dir or a.val_dir
    if a.init and not os.path.exists(os.path.join(a.out, "last.pt")):
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
        print(f"initialised from {a.init}", flush=True)
    floors = [a.gate_floor, 0.5]
    state = {"step": 0, "elapsed": 0.0, "best": -1e9, "log": []}
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        print(f"resumed step {state['step']} elapsed {state['elapsed'] / 60:.1f} min", flush=True)
    else:
        r0 = evaluate(model, eval_dir, hold, dev, floors, a.eval_files)
        print(f"init holdout {json.dumps(r0)}", flush=True)
        state["log"].append({"step": 0, **r0})

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "args": vars(a)}, path + ".tmp")
        os.replace(path + ".tmp", path)

    dl = torch.utils.data.DataLoader(MixDataset(a.bank, a.seconds, a.p_ext, a.p_valref, a.seed, tuple(a.ev_mult),
                                                tuple(a.db_shift), a.no_vaani),
                                     batch_size=a.batch, num_workers=a.workers, pin_memory=dev.type == "cuda",
                                     persistent_workers=a.workers > 0)
    budget = a.minutes * 60
    t0 = time.time() - state["elapsed"]
    last_eval = last_save = time.time()
    run = {"loss": 0.0, "sisdr": 0.0, "n": 0}
    model.train()
    for mix, clean, act in dl:
        el = time.time() - t0
        if el >= budget or (a.max_steps and state["step"] >= a.max_steps):
            break
        frac = el / budget
        lr = a.lr * min(1.0, (state["step"] + 1) / 500) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        mix, clean, act = mix.to(dev, non_blocking=True), clean.to(dev, non_blocking=True), act.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            enh, p = model(mix)
        enh = enh.float()
        loss = neg_si_sdr(enh, clean) + a.stft_w * mrstft(enh, clean)
        if p is not None:
            T = min(p.shape[1], act.shape[1])
            gated = apply_gate(mix, enh, p, a.gate_floor)
            loss = 0.5 * loss + 0.5 * (neg_si_sdr(gated, clean) + a.stft_w * mrstft(gated, clean))
            loss = loss + a.act_w * F.binary_cross_entropy(p[:, :T].clamp(1e-4, 1 - 1e-4), act[:, :T])
        if asr is not None and state["step"] % a.asr_every == 0:
            loss = loss + a.asr_w * asr(gated if p is not None else enh, clean)
        if not torch.isfinite(loss):
            print(f"non-finite loss at step {state['step']}, skipping batch", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        state["step"] += 1
        run["loss"] += loss.item()
        run["sisdr"] += -float(neg_si_sdr(enh.detach(), clean))
        run["n"] += 1
        if state["step"] % 100 == 0:
            print(f"step {state['step']} {el / 60:.1f}min lr {lr:.2e} loss {run['loss'] / run['n']:.3f} "
                  f"train-sisdr {run['sisdr'] / run['n']:.2f}", flush=True)
            run = {"loss": 0.0, "sisdr": 0.0, "n": 0}
        state["elapsed"] = time.time() - t0
        if time.time() - last_save > 600:
            save(last)
            last_save = time.time()
        if time.time() - last_eval > a.eval_every_min * 60:
            r = evaluate(model, eval_dir, hold, dev, floors, a.eval_files)
            score = max(r.values())
            state["log"].append({"step": state["step"], **r})
            print(f"EVAL step {state['step']} {json.dumps(r)}", flush=True)
            if score > state["best"]:
                state["best"] = score
                save(os.path.join(a.out, "best.pt"))
            last_eval = time.time()
    state["elapsed"] = time.time() - t0
    r = evaluate(model, eval_dir, hold, dev, floors, a.eval_files)
    state["log"].append({"step": state["step"], **r})
    print(f"FINAL step {state['step']} {json.dumps(r)}", flush=True)
    if max(r.values()) > state["best"]:
        state["best"] = max(r.values())
        save(os.path.join(a.out, "best.pt"))
    save(last)
    save_json(state["log"], os.path.join(a.out, "eval_log.json"))
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
