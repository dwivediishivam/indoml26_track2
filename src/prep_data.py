"""Build the training banks for on-the-fly mixing.

Outputs (in --out):
  split.json          synthetic validation clips -> {"train": [...], "holdout": [...]}
  noise_bank.npy      float16 concat of noise events; noise_index.json [[offset, length, source], ...]
  clean_bank.npy      float16 concat of clean speech; clean_index.json [[offset, length, source], ...]
  stats.json          empirical event loudness (dB re clean-clip RMS), events/clip, event durations

Noise events come from (noisy - clean) of the TRAIN split of the synthetic validation pairs only,
plus optional ESC-50 clips. Holdout clips never enter any bank.
"""
import argparse
import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import SR, load_json, noise_floor_db, parse_events, read_wav, save_json


class Bank:
    def __init__(self):
        self.chunks, self.index, self.off = [], [], 0

    def add(self, x, source):
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all() or len(x) == 0:
            return
        self.chunks.append(x.astype(np.float16))
        self.index.append([self.off, len(x), source])
        self.off += len(x)

    def save(self, out, name):
        arr = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float16)
        np.save(os.path.join(out, f"{name}_bank.npy"), arr)
        save_json(self.index, os.path.join(out, f"{name}_index.json"))
        return len(self.index), self.off / SR / 3600


def vaani_shard(args):
    """Clean (low noise floor) non-event regions from one parquet shard."""
    path, floor_thr, pad_s, min_seg_s = args
    import pyarrow.parquet as pq
    out = []
    try:
        t = pq.read_table(path, columns=["audio", "annotationQuality", "NoiseSubCategoryTimeStamp"])
    except Exception as e:  # corrupt shard: skip, don't kill the job
        return path, [], f"read error {e}"
    for row in t.to_pylist():
        if row["annotationQuality"] not in ("verified_timestamps", "unverified_timestamps"):
            continue
        evs = parse_events(row["NoiseSubCategoryTimeStamp"])
        if not evs:
            continue
        try:
            x = read_wav(row["audio"]["bytes"])
        except Exception:
            continue
        if noise_floor_db(x) > floor_thr:
            continue
        keep = np.ones(len(x), bool)
        for s, e in evs:
            keep[max(0, int((s - pad_s) * SR)):int((e + pad_s) * SR)] = False
        # contiguous runs of keep
        d = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if b - a >= min_seg_s * SR:
                seg = x[a:b]
                if np.sqrt((seg ** 2).mean()) > 1e-3:  # not silence
                    out.append(seg.astype(np.float16))
    return path, out, "ok"


def extra_shard(args):
    """Clean speech from a generic HF audio parquet (e.g. IndicVoices-R); any rate, resampled to 16 kHz."""
    path, max_s, max_items = args
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    out = []
    try:
        pf = pq.ParquetFile(path)
        col = next(f.name for f in pf.schema_arrow if str(f.type).startswith("struct") and "bytes" in str(f.type))
        rows = (r for b in pf.iter_batches(batch_size=64, columns=[col]) for r in b.column(col).to_pylist())
    except Exception as e:
        return path, [], f"read error {e}"
    for row in rows:
        if len(out) >= max_items:
            break
        try:
            x, sr = sf.read(io.BytesIO(row["bytes"]), dtype="float32", always_2d=False)
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            from math import gcd
            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        x = x[: int(max_s * SR)]
        if len(x) >= 1.5 * SR and np.sqrt((x ** 2).mean()) > 1e-3 and noise_floor_db(x) < -20:
            out.append(x.astype(np.float16))
    return path, out, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True, help="extracted validation/ folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout_n", type=int, default=135)
    ap.add_argument("--split", default="", help="reuse this split.json instead of drawing one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parquet_glob", default="", help="Vaani train parquet shards (optional)")
    ap.add_argument("--max_shards", type=int, default=0, help="0 = all matched shards")
    ap.add_argument("--floor_thr", type=float, default=-28.0)
    ap.add_argument("--max_vaani_hours", type=float, default=25.0)
    ap.add_argument("--esc50_dir", default="", help="ESC-50 root with audio/*.wav (optional)")
    ap.add_argument("--extra_clean_glob", default="", help="generic clean-speech parquet shards (optional)")
    ap.add_argument("--max_extra_hours", type=float, default=20.0)
    ap.add_argument("--max_extra_per_shard", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta = load_json(os.path.join(a.val_dir, "validationMetadata.json"))
    syn = sorted(x["segmentFileName"] for x in meta if x.get("syntheticData"))
    if a.split:  # reuse an existing split so every machine scores the same holdout
        split = load_json(a.split)
        hold = set(split["holdout"])
    else:
        rng = random.Random(a.seed)
        hold = set(rng.sample(syn, a.holdout_n))
        split = {"train": [f for f in syn if f not in hold], "holdout": sorted(hold)}
    save_json(split, os.path.join(a.out, "split.json"))
    by_name = {x["segmentFileName"]: x for x in meta}

    noise, clean = Bank(), Bank()
    ev_db, ev_dur, n_ev = [], [], []
    for f in split["train"]:
        y = read_wav(os.path.join(a.val_dir, "syntheticNoiseAudio", f))
        c = read_wav(os.path.join(a.val_dir, "syntheticCleanRefAudio", f))
        n = min(len(y), len(c))
        y, c = y[:n], c[:n]
        r = y - c
        crms = np.sqrt((c ** 2).mean()) + 1e-9
        clean.add(c, "valref")
        evs = parse_events(by_name[f]["NoiseSubCategoryTimeStamp"])
        n_ev.append(len(evs))
        for s, e in evs:
            a0, b0 = max(0, int((s - 0.05) * SR)), min(n, int((e + 0.05) * SR))
            seg = r[a0:b0]
            if len(seg) < 0.05 * SR or np.sqrt((seg ** 2).mean()) < 1e-5:
                continue
            noise.add(seg, "valres")
            ev_db.append(float(20 * np.log10(np.sqrt((seg ** 2).mean()) / crms)))
            ev_dur.append(len(seg) / SR)
    print(f"val: {len(split['train'])} train pairs, {len(noise.index)} noise events, holdout {len(hold)}")

    if a.esc50_dir:
        from scipy.signal import resample_poly
        import soundfile as sf
        files = sorted(glob.glob(os.path.join(a.esc50_dir, "**", "*.wav"), recursive=True))
        k = 0
        for p in files:
            x, sr = sf.read(p, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(1)
            if sr != SR:
                x = resample_poly(x, SR, sr).astype(np.float32)
            fr = x[: len(x) // 320 * 320].reshape(-1, 320)
            if len(fr) == 0:
                continue
            e = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
            act = np.where(e > e.max() - 30)[0]
            if len(act) == 0:
                continue
            seg = x[act[0] * 320:(act[-1] + 1) * 320]
            if len(seg) >= 0.1 * SR:
                noise.add(seg, "esc50")
                k += 1
        print(f"esc50: {k} clips")

    if a.parquet_glob:
        shards = sorted(glob.glob(a.parquet_glob))
        if a.max_shards:
            shards = shards[: a.max_shards]
        print(f"vaani: scanning {len(shards)} shards")
        hours = 0.0
        jobs = [(p, a.floor_thr, 0.25, 1.5) for p in shards]
        with ProcessPoolExecutor(a.workers) as ex:
            for p, segs, status in ex.map(vaani_shard, jobs):
                for s in segs:
                    clean.add(s.astype(np.float32), "vaani")
                    hours += len(s) / SR / 3600
                print(f"  {os.path.basename(p)} {status} +{len(segs)} segs, total vaani {hours:.2f} h", flush=True)
                if hours >= a.max_vaani_hours:
                    ex.shutdown(cancel_futures=True)
                    break

    if a.extra_clean_glob:
        shards = sorted(glob.glob(a.extra_clean_glob))
        print(f"extra clean: scanning {len(shards)} shards")
        hours = 0.0
        with ProcessPoolExecutor(a.workers) as ex:
            for p, segs, status in ex.map(extra_shard, [(p, 12.0, a.max_extra_per_shard) for p in shards]):
                for s_ in segs:
                    clean.add(s_.astype(np.float32), "extra")
                    hours += len(s_) / SR / 3600
                print(f"  {os.path.basename(p)} {status} +{len(segs)} segs, total extra {hours:.2f} h", flush=True)
                if hours >= a.max_extra_hours:
                    ex.shutdown(cancel_futures=True)
                    break

    nn, nh = noise.save(a.out, "noise")
    cn, ch = clean.save(a.out, "clean")
    save_json({"event_db": ev_db, "event_dur": ev_dur, "n_events": n_ev}, os.path.join(a.out, "stats.json"))
    print(f"DONE noise {nn} items {nh:.2f} h | clean {cn} items {ch:.2f} h")


if __name__ == "__main__":
    main()
