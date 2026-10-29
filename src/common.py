"""Shared helpers: audio IO, the official SI-SDR, span masks, noise-floor estimate."""
import io
import json
import os

import numpy as np
import soundfile as sf

SR = 16000


def read_wav(path_or_bytes):
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    x, sr = sf.read(src, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        raise ValueError(f"expected {SR} Hz, got {sr}")
    return x


def write_wav_int16(path, x):
    """Write PCM_16 mono at 16 kHz. Peak-normalise only if the signal would clip."""
    x = np.asarray(x, dtype=np.float64)
    peak = np.abs(x).max() if x.size else 0.0
    if peak > 0.999:
        x = x * (0.999 / peak)
    sf.write(path, x.astype(np.float32), SR, subtype="PCM_16")


def si_sdr(reference, enhanced):
    """Verbatim copy of the official scorer snippet."""
    ref = np.asarray(reference, dtype=np.float64)
    enh = np.asarray(enhanced, dtype=np.float64)
    n = min(len(ref), len(enh))
    ref, enh = ref[:n], enh[:n]
    scale = np.dot(enh, ref) / np.dot(ref, ref)
    s_target = scale * ref
    e_noise = enh - s_target
    value = 10.0 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise))
    return float(np.clip(value, -100.0, 100.0))


def parse_events(evs):
    """List of (start_s, end_s) from a NoiseSubCategoryTimeStamp list; skips malformed entries."""
    out = []
    for e in evs or []:
        try:
            s, t = float(e["start"]), float(e["end"])
        except (TypeError, ValueError, KeyError):
            continue
        if t > s >= 0:
            out.append((s, t))
    return out


def noise_floor_db(x, frame=320):
    """10th-percentile frame energy relative to clip energy (dB). Lower = cleaner background."""
    n = len(x) // frame * frame
    if n < frame * 5:
        return 0.0
    f = x[:n].reshape(-1, frame)
    e = 10 * np.log10((f ** 2).mean(1) + 1e-12)
    r = 10 * np.log10((x ** 2).mean() + 1e-12)
    return float(np.percentile(e, 10) - r)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
