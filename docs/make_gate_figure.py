"""Fig. 3 of the technical report: final ensemble on one held-out validation clip.

    python docs/make_gate_figure.py VALIDATION_PARENT CLIP.wav docs/fig_gate_example.png

VALIDATION_PARENT holds validation/{validationMetadata.json,syntheticNoiseAudio,syntheticCleanRefAudio}.
Run from the repository root with checkpoints/ft_dns64.pt and checkpoints/ft_bsrnn.pt present (needs matplotlib).
"""
import sys, os, re, json
import numpy as np, torch
sys.path.insert(0, "src")
from common import read_wav, SR
from enhance import load, run_one
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D, F, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
x = read_wav(f"{D}/validation/syntheticNoiseAudio/{F}"); s = read_wav(f"{D}/validation/syntheticCleanRefAudio/{F}")
meta = json.load(open(f"{D}/validation/validationMetadata.json"))
rec = [m for m in meta if m["segmentFileName"] == F][0]
ev = [(t, float(a), float(b)) for t, a, b in re.findall(r"'tag': '([^']+)', 'start': '([\d.]+)', 'end': '([\d.]+)'", str(rec["NoiseSubCategoryTimeStamp"]))]
dev = torch.device("cpu")
dm = load("checkpoints/ft_dns64.pt", dev, "third_party/denoiser"); bs = load("checkpoints/ft_bsrnn.pt", dev, "third_party/denoiser")
xt = torch.from_numpy(x)
with torch.no_grad():
    _, p = bs(xt[None]); p = p[0].numpy()
    e = torch.stack([run_one(dm, xt, 0.2), run_one(bs, xt, 0.2)]).mean(0).numpy().astype(np.float64)
e *= np.sqrt((x.astype(np.float64)**2).mean()) / (np.sqrt((e**2).mean()) + 1e-12)
def sisdr(a, r):
    a = a - a.mean(); r = r - r.mean(); al = (a @ r) / (r @ r); t = al * r
    return 10 * np.log10((t @ t) / ((a - t) @ (a - t)))
si_in, si_out = sisdr(x.astype(np.float64), s.astype(np.float64)), sisdr(e, s.astype(np.float64))
print(json.dumps({"dur": len(x)/SR, "events": ev, "sisdr_in": si_in, "sisdr_out": si_out}))
def spec(y):
    S = np.abs(torch.stft(torch.from_numpy(np.asarray(y, np.float32)), 512, 128, window=torch.hann_window(512), return_complex=True).numpy())
    return 20 * np.log10(S + 1e-5)
T = len(x) / SR
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig, ax = plt.subplots(4, 1, figsize=(3.45, 3.25), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.62]})
vmax = spec(x).max()
for a_, y, ttl in zip(ax[:3], [x, e, s], [f"noisy input (SI-SDR {si_in:.1f} dB)", f"enhanced, final ensemble (SI-SDR {si_out:.1f} dB)", "clean reference"]):
    a_.imshow(spec(y), origin="lower", aspect="auto", extent=[0, T, 0, 8], vmin=vmax - 80, vmax=vmax, cmap="magma")
    a_.set_ylabel("kHz"); a_.set_title(ttl, fontsize=7, pad=2)
    for _, b0, b1 in ev:
        a_.axvline(b0, color="w", lw=0.5, ls=":"); a_.axvline(b1, color="w", lw=0.5, ls=":")
t = np.arange(len(p)) * 128 / SR
mask = np.zeros_like(t)
for _, b0, b1 in ev: mask[(t >= b0) & (t <= b1)] = 1
ax[3].fill_between(t, 0, mask, color="0.85", step="mid", label="labelled event spans")
ax[3].plot(t, p, color="C1", lw=0.9, label="BSRNN gate $p(t)$")
ax[3].set_ylim(0, 1.05); ax[3].set_xlabel("time (s)"); ax[3].set_ylabel("$p(t)$")
ax[3].legend(loc="upper center", ncol=2, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.72))
fig.tight_layout(h_pad=0.25)
fig.savefig(OUT, dpi=300)
