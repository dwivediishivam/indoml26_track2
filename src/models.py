"""Enhancement models.

BSRNN  : band-split RNN predicting a complex mask, plus a frame-level noise-event activity head.
         Output gating: y = mix + g * (enh - mix), g = floor + (1 - floor) * p_event.
Demucs : DNS64 (facebookresearch/denoiser) initialised from its pretrained checkpoint.
ASRFeatLoss : L1 between frozen SraVaani encoder outputs on enhanced vs clean audio.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FFT, HOP = 512, 128
BAND_WIDTHS = [4] * 16 + [8] * 16 + [16] * 4 + [1]  # 257 bins
assert sum(BAND_WIDTHS) == N_FFT // 2 + 1


def stft(x):
    w = torch.hann_window(N_FFT, device=x.device)
    return torch.stft(x.float(), N_FFT, HOP, window=w, return_complex=True)  # (B, F, T)


def istft(X, length):
    w = torch.hann_window(N_FFT, device=X.device)
    return torch.istft(X, N_FFT, HOP, window=w, length=length)


def apply_gate(mix, enh, p, floor):
    """p: (B, T_frames) event probability -> sample-level gate."""
    g = floor + (1.0 - floor) * p
    g = F.interpolate(g.unsqueeze(1), size=mix.shape[-1], mode="linear", align_corners=False).squeeze(1)
    return mix + g * (enh - mix)


class BSRNN(nn.Module):
    def __init__(self, feat=96, layers=6):
        super().__init__()
        self.widths = BAND_WIDTHS
        K = len(self.widths)
        self.norms = nn.ModuleList(nn.GroupNorm(1, 2 * w) for w in self.widths)
        self.fcs = nn.ModuleList(nn.Linear(2 * w, feat) for w in self.widths)
        self.t_norm = nn.ModuleList(nn.GroupNorm(1, feat) for _ in range(layers))
        self.t_rnn = nn.ModuleList(nn.LSTM(feat, 2 * feat, batch_first=True, bidirectional=True) for _ in range(layers))
        self.t_fc = nn.ModuleList(nn.Linear(4 * feat, feat) for _ in range(layers))
        self.k_norm = nn.ModuleList(nn.GroupNorm(1, feat) for _ in range(layers))
        self.k_rnn = nn.ModuleList(nn.LSTM(feat, 2 * feat, batch_first=True, bidirectional=True) for _ in range(layers))
        self.k_fc = nn.ModuleList(nn.Linear(4 * feat, feat) for _ in range(layers))
        self.mask = nn.ModuleList(
            nn.Sequential(nn.GroupNorm(1, feat), nn.Conv1d(feat, 4 * feat, 1), nn.Tanh(),
                          nn.Conv1d(4 * feat, 2 * w * 2, 1), nn.GLU(dim=1))
            for w in self.widths)
        self.gate = nn.Sequential(nn.Linear(feat * K, 256), nn.GELU(), nn.Linear(256, 1))
        self.K, self.feat = K, feat

    def forward(self, mix):
        """mix (B, L) -> enh (B, L), p_event (B, T)."""
        B, L = mix.shape
        scale = mix.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-5)
        X = stft(mix / scale)  # (B, F, T) complex
        T = X.shape[-1]
        Xr = torch.view_as_real(X)  # (B, F, T, 2)
        feats, f0 = [], 0
        for k, w in enumerate(self.widths):
            sub = Xr[:, f0:f0 + w].permute(0, 3, 1, 2).reshape(B, 2 * w, T)  # (B, 2w, T)
            sub = self.norms[k](sub).transpose(1, 2)  # (B, T, 2w)
            feats.append(self.fcs[k](sub))
            f0 += w
        z = torch.stack(feats, 1)  # (B, K, T, N)
        N = self.feat
        for i in range(len(self.t_rnn)):
            h = self.t_norm[i](z.reshape(B * self.K, T, N).transpose(1, 2)).transpose(1, 2)
            h = self.t_fc[i](self.t_rnn[i](h)[0])
            z = z + h.reshape(B, self.K, T, N)
            zt = z.transpose(1, 2).reshape(B * T, self.K, N)
            h = self.k_norm[i](zt.transpose(1, 2)).transpose(1, 2)
            h = self.k_fc[i](self.k_rnn[i](h)[0])
            z = (zt + h).reshape(B, T, self.K, N).transpose(1, 2)
        p = torch.sigmoid(self.gate(z.permute(0, 2, 1, 3).reshape(B, T, self.K * N)).squeeze(-1).float())
        masks, f0 = [], 0
        for k, w in enumerate(self.widths):
            m = self.mask[k](z[:, k].transpose(1, 2)).reshape(B, 2, w, T).float()  # re/im mask
            masks.append(torch.complex(m[:, 0], m[:, 1]))
            f0 += w
        M = torch.cat(masks, 1)  # (B, F, T)
        enh = istft(X * M, L) * scale
        return enh, p


class DemucsDNS64(nn.Module):
    def __init__(self, denoiser_root, ckpt):
        super().__init__()
        if denoiser_root not in sys.path:
            sys.path.insert(0, denoiser_root)
        from denoiser.demucs import Demucs
        self.net = Demucs(hidden=64, sample_rate=16000)
        if ckpt:
            self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))

    def forward(self, mix):
        return self.net(mix.unsqueeze(1)).squeeze(1)[..., : mix.shape[-1]], None


class ASRFeatLoss(nn.Module):
    """Frozen SraVaani encoder; front end copied from the official wrapper (without no_grad)."""

    def __init__(self, asr_dir):
        super().__init__()
        mod = torch.jit.load(os.path.join(asr_dir, "model-asr.fp16.ts"), map_location="cpu").eval()
        self.enc = mod.encoder.float()
        for prm in self.enc.parameters():
            prm.requires_grad_(False)
        pp = torch.load(os.path.join(asr_dir, "preproc.pt"), map_location="cpu")
        self.p = pp["params"]
        self.register_buffer("win", pp["window"].float(), persistent=False)
        self.register_buffer("fb", pp["fb"].float(), persistent=False)

    def feats(self, wav):
        p = self.p
        B, L = wav.shape
        seq_len = torch.full((B,), L // p["hop_length"], dtype=torch.long, device=wav.device)
        x = torch.cat((wav[:, :1], wav[:, 1:] - p["preemph"] * wav[:, :-1]), dim=1)
        spec = torch.stft(x, n_fft=p["n_fft"], hop_length=p["hop_length"], win_length=p["win_length"],
                          window=self.win, center=True, pad_mode="constant", return_complex=True)
        x = torch.view_as_real(spec)
        x = torch.sqrt(x.pow(2).sum(-1) + 1e-12).pow(p["mag_power"])
        x = torch.log(torch.matmul(self.fb, x) + p["log_zero_guard_value"])
        mean = x.mean(2, keepdim=True)
        std = x.std(2, keepdim=True) + p["CONSTANT"]
        return (x - mean) / std, seq_len

    def encode(self, wav):
        f, n = self.feats(wav.float())
        out = self.enc(f, n)
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, enh, clean):
        with torch.no_grad():
            tgt = self.encode(clean)
        e = self.encode(enh)
        return (e - tgt).abs().mean() / (tgt.abs().mean() + 1e-6)


def build_model(arch, denoiser_root="", dns64_ckpt="", feat=96, layers=6):
    if arch == "bsrnn":
        return BSRNN(feat=feat, layers=layers)
    if arch == "demucs":
        return DemucsDNS64(denoiser_root, dns64_ckpt)
    raise ValueError(arch)
