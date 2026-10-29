"""Validate and zip a Codabench submission: every input clip has a same-length 16 kHz mono PCM_16 WAV
and exactly one non-empty transcript line. Refuses to write the ZIP if any check fails."""
import argparse
import glob
import json
import os
import zipfile

import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="original test WAVs")
    ap.add_argument("--enh_dir", required=True)
    ap.add_argument("--tx", required=True)
    ap.add_argument("--zip", required=True)
    a = ap.parse_args()
    ins = {os.path.basename(p): p for p in glob.glob(os.path.join(a.input_dir, "*.wav"))}
    errs = []
    for n, p in ins.items():
        q = os.path.join(a.enh_dir, n)
        if not os.path.exists(q):
            errs.append(f"missing {n}")
            continue
        i, o = sf.info(p), sf.info(q)
        if o.samplerate != 16000 or o.channels != 1 or o.subtype != "PCM_16" or o.frames != i.frames:
            errs.append(f"bad format {n}: {o.samplerate} {o.channels} {o.subtype} {o.frames} vs {i.frames}")
    tx = {}
    for line in open(a.tx, encoding="utf-8"):
        r = json.loads(line)
        if r["clip_id"] in tx:
            errs.append(f"duplicate transcript {r['clip_id']}")
        tx[r["clip_id"]] = r["text"]
    want = {n[:-4] for n in ins}
    if set(tx) != want:
        errs.append(f"transcript ids mismatch: missing {len(want - set(tx))}, extra {len(set(tx) - want)}")
    empty = [k for k, v in tx.items() if not (v or "").strip()]
    print(f"{len(ins)} clips, {len(tx)} transcripts, {len(empty)} empty transcripts")
    if errs:
        print("\n".join(errs[:20]))
        raise SystemExit(f"{len(errs)} problems: ZIP not written")
    with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
        with open(os.path.join(os.path.dirname(os.path.abspath(a.zip)), "transcripts.jsonl"), "w", encoding="utf-8") as f:
            for k in sorted(tx):
                f.write(json.dumps({"clip_id": k, "text": tx[k]}, ensure_ascii=False) + "\n")
        z.write(os.path.join(os.path.dirname(os.path.abspath(a.zip)), "transcripts.jsonl"), "transcripts.jsonl")
        for n in sorted(ins):
            z.write(os.path.join(a.enh_dir, n), n)
    print(f"PACKAGE OK -> {a.zip}")


if __name__ == "__main__":
    main()
