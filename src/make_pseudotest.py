"""Build a test-like evaluation set from the HOLDOUT synthetic clips only (never trained on).

The Phase 2 test clips are ~12 s (validation ~4.5 s) and noisier. Each pseudo-test clip joins 2-3 holdout clean
references, keeps their own noise events at the original offsets, then adds extra holdout noise events
(density x --extra, gain +--gain_db). Output mirrors the validation layout so score_val.py works unchanged:
  OUT/validationMetadata.json, OUT/syntheticNoiseAudio/, OUT/syntheticCleanRefAudio/, OUT/naturalNoisyAudio/ (empty),
  OUT/split.json (all clips are 'holdout').
"""
import argparse
import json
import os
import random

import numpy as np

from common import SR, load_json, parse_events, read_wav, save_json, write_wav_int16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groupings", type=int, default=2, help="independent random groupings of the holdout")
    ap.add_argument("--extra", type=float, default=1.0, help="extra events per original event")
    ap.add_argument("--gain_db", type=float, default=3.0, help="mean gain added to every noise event")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    meta = {x["segmentFileName"]: x for x in load_json(os.path.join(a.val_dir, "validationMetadata.json"))}
    hold = load_json(a.split)["holdout"]
    clips, events = {}, []
    for f in hold:
        y = read_wav(os.path.join(a.val_dir, "syntheticNoiseAudio", f))
        c = read_wav(os.path.join(a.val_dir, "syntheticCleanRefAudio", f))
        n = min(len(y), len(c))
        clips[f] = (c[:n], y[:n] - c[:n])
        for s, e in parse_events(meta[f]["NoiseSubCategoryTimeStamp"]):
            seg = (y[:n] - c[:n])[max(0, int((s - 0.05) * SR)):int((e + 0.05) * SR)]
            if len(seg) > 0.05 * SR:
                events.append(seg)
    for d in ("syntheticNoiseAudio", "syntheticCleanRefAudio", "naturalNoisyAudio"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    out_meta, names = [], []
    for g in range(a.groupings):
        order = hold[:]
        rng.shuffle(order)
        i = 0
        while i < len(order):
            k = rng.choice([2, 3])
            grp = order[i:i + k]
            i += k
            gap = np.zeros(int(0.3 * SR), np.float32)
            clean = np.concatenate([np.concatenate([clips[f][0], gap]) for f in grp])
            noise = np.concatenate([np.concatenate([clips[f][1], gap]) for f in grp])
            g_lin = 10 ** (a.gain_db / 20)
            noise = noise * g_lin
            n_orig = sum(len(parse_events(meta[f]["NoiseSubCategoryTimeStamp"])) for f in grp)
            crms = np.sqrt((clean ** 2).mean()) + 1e-9
            for _ in range(int(round(n_orig * a.extra))):
                ev = events[rng.randrange(len(events))]
                ev = ev * g_lin * 10 ** (rng.uniform(-3, 3) / 20)
                s = rng.randrange(max(1, len(clean) - len(ev)))
                noise[s:s + len(ev)] += ev[: len(clean) - s]
            mix = clean + noise
            pk = np.abs(mix).max()
            if pk > 0.99:
                mix, clean = mix * 0.99 / pk, clean * 0.99 / pk
            name = f"pseudo_g{g}_{len(names):04d}.wav"
            names.append(name)
            write_wav_int16(os.path.join(a.out, "syntheticNoiseAudio", name), mix)
            write_wav_int16(os.path.join(a.out, "syntheticCleanRefAudio", name), clean)
            out_meta.append({"segmentFileName": name, "duration": len(mix) / SR, "syntheticData": True,
                             "transcript": " ".join(meta[f]["transcript"] for f in grp)})
    save_json(out_meta, os.path.join(a.out, "validationMetadata.json"))
    save_json({"train": [], "holdout": names}, os.path.join(a.out, "split.json"))
    d = np.array([m["duration"] for m in out_meta])
    print(f"pseudo-test: {len(names)} clips, {d.sum() / 3600:.2f} h, duration median {np.median(d):.1f}s")


if __name__ == "__main__":
    main()
