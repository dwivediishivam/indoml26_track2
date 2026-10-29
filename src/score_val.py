"""Score an enhanced validation folder with the official metrics.

SI-SDR : holdout synthetic clips only (the train split is in-sample).
dWER   : pooled over all clips, and over clips not used for training (natural + holdout).
Combined = SI-SDR + 100 * dWER.
"""
import argparse
import json
import os
import re
import unicodedata

from jiwer import wer

from common import load_json, read_wav, si_sdr

TAG_RE = re.compile(r"</?[^<>]*>|\[[^\[\]]*\]")


def normalize(text):
    s = TAG_RE.sub(" ", text or "")
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.lower().split())


def pooled_wer(gt, hyp, ids):
    refs, hs = [], []
    for c in ids:
        g = normalize(gt[c])
        if not g:
            continue
        refs.append(g)
        hs.append(normalize(hyp.get(c, "")) or "@")
    return wer(refs, hs)


def read_jsonl(p):
    return {r["clip_id"]: r["text"] for r in map(json.loads, open(p, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--split", required=True, help="bank/split.json")
    ap.add_argument("--enh_dir", required=True)
    ap.add_argument("--noisy_tx", required=True)
    ap.add_argument("--enh_tx", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--allow_missing", action="store_true", help="preflight only: skip absent holdout files")
    a = ap.parse_args()
    meta = load_json(os.path.join(a.val_dir, "validationMetadata.json"))
    split = load_json(a.split)
    hold = split["holdout"]
    if a.allow_missing:
        hold = [f for f in hold if os.path.exists(os.path.join(a.enh_dir, f))]
    s = [si_sdr(read_wav(os.path.join(a.val_dir, "syntheticCleanRefAudio", f)),
                read_wav(os.path.join(a.enh_dir, f))) for f in hold]
    res = {"sisdr_holdout": sum(s) / len(s), "n_holdout": len(s)}
    if a.enh_tx:
        gt = {x["segmentFileName"][:-4]: x["transcript"] for x in meta}
        noisy, enh = read_jsonl(a.noisy_tx), read_jsonl(a.enh_tx)
        train = {f[:-4] for f in split["train"]}
        for name, ids in (("all", list(gt)), ("unseen", [c for c in gt if c not in train])):
            wn, we = pooled_wer(gt, noisy, ids), pooled_wer(gt, enh, ids)
            res[f"wer_noisy_{name}"], res[f"wer_enh_{name}"], res[f"dwer_{name}"] = wn, we, wn - we
        res["combined_est"] = res["sisdr_holdout"] + 100 * res["dwer_unseen"]
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
