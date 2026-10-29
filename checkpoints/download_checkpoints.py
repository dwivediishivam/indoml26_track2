"""Download the Track 2 checkpoints from Hugging Face into this folder and verify their SHA-256.

    python checkpoints/download_checkpoints.py            # the two members of the final system
    python checkpoints/download_checkpoints.py --all      # also the stage-1 checkpoints and the DNS64 init
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "dwivediishivam/indoml26_track2"
SHA256 = {
    "ft_dns64.pt": "071f5c47dce8baa8426112291b96b1122790b89a0378cbf3d006d6d40f8cfd63",
    "ft_bsrnn.pt": "e3ba7fbafb005c71183f4022b8ca0e404490c2dc1b343c7d4a1983eacc9989a5",
    "b200_dns64.pt": "e34970ba13827bf58bc75468bacb6389c5603405fc78aff312b6ceca839380c9",
    "colabA_bsrnn_feat112.pt": "ef401e9b4e55011f6a83d05a5f0196f68b995e20337298c8e8afafa19c058cf4",
    "colabB_dns64_ft.pt": "2102ee0d542eae5c075e25f44228408fec706501cb2e89d380c8dc1d18ad09c1",
}
FINAL = ["ft_dns64.pt", "ft_bsrnn.pt"]
DNS64_URL = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(names):
    from huggingface_hub import hf_hub_download
    for n in names:
        dst = os.path.join(HERE, n)
        if not os.path.exists(dst):
            p = hf_hub_download(REPO, n, local_dir=HERE)
            if os.path.abspath(p) != dst:
                shutil.copy(p, dst)
        got = sha256(dst)
        if got != SHA256[n]:
            sys.exit(f"SHA-256 mismatch for {n}: {got}")
        print(f"ok {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    download(list(SHA256) if a.all else FINAL)
    if a.all:
        dst = os.path.join(HERE, os.path.basename(DNS64_URL))
        if not os.path.exists(dst):
            urllib.request.urlretrieve(DNS64_URL, dst)
        print(f"ok {os.path.basename(dst)} (pretrained DNS64, used to initialise stage-1 DNS64 training)")


if __name__ == "__main__":
    main()
