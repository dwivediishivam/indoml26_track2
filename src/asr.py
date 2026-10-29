"""Transcribe the exact WAV files in --wav_dir with the unmodified mandated SraVaani-1.0 wrapper.

Same call path as the audited Phase 1 submission: AutoModel(trust_remote_code), default fp32,
model.transcribe(paths, batch_size=4). Failed batches are retried per clip; a clip that still
fails is reported loudly (never silently written as empty).
"""
import argparse
import glob
import json
import os

import torch
from transformers import AutoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr_dir", required=True, help="local SraVaani-1.0 repo directory")
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out", required=True, help="transcripts.jsonl")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(a.asr_dir, trust_remote_code=True).to(dev).eval()
    paths = sorted(glob.glob(os.path.join(a.wav_dir, "*.wav")))
    if a.limit:
        paths = paths[: a.limit]
    done = {}
    if os.path.exists(a.out):  # resume
        for line in open(a.out, encoding="utf-8"):
            r = json.loads(line)
            done[r["clip_id"]] = r["text"]
    todo = [p for p in paths if os.path.basename(p)[:-4] not in done]
    failed = []
    with open(a.out, "a", encoding="utf-8") as fo:
        for s in range(0, len(todo), a.batch):
            batch = todo[s:s + a.batch]
            try:
                texts = [h.text for h in model.transcribe(batch, batch_size=a.batch, return_hypotheses=True)]
            except Exception as e:
                print(f"batch failed ({e}); retrying singly", flush=True)
                texts = []
                for p in batch:
                    try:
                        texts.append(model.transcribe([p], return_hypotheses=True)[0].text)
                    except Exception as e2:
                        print(f"FAILED {p}: {e2}", flush=True)
                        texts.append(None)
            for p, t in zip(batch, texts):
                if t is None:
                    failed.append(p)
                    continue
                fo.write(json.dumps({"clip_id": os.path.basename(p)[:-4], "text": t}, ensure_ascii=False) + "\n")
            fo.flush()
            if (s // a.batch) % 100 == 0:
                print(f"{s + len(batch)}/{len(todo)}", flush=True)
    n = sum(1 for _ in open(a.out, encoding="utf-8"))
    print(f"ASR DONE {n}/{len(paths)} transcribed, {len(failed)} failed", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} clips failed ASR: rerun this script to retry them")


if __name__ == "__main__":
    main()
