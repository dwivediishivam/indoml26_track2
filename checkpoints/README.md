# Checkpoints

GitHub limits files to 100 MB, so the trained weights are hosted on Hugging Face:
https://huggingface.co/dwivediishivam/indoml26_track2

```bash
python checkpoints/download_checkpoints.py          # final system: ft_dns64.pt + ft_bsrnn.pt
python checkpoints/download_checkpoints.py --all    # plus stage-1 checkpoints and the DNS64 initialisation
```

| File | Role | Size | SHA-256 |
|---|---|---|---|
| `ft_dns64.pt` | **final system member 1**: DNS64, stage-2 test-condition fine-tune (`configs/ft_dns64.yaml`) | 402 MB | `071f5c47dce8baa8426112291b96b1122790b89a0378cbf3d006d6d40f8cfd63` |
| `ft_bsrnn.pt` | **final system member 2**: BSRNN + event gate, stage-2 fine-tune (`configs/ft_bsrnn.yaml`) | 137 MB | `e3ba7fbafb005c71183f4022b8ca0e404490c2dc1b343c7d4a1983eacc9989a5` |
| `b200_dns64.pt` | stage-1 DNS64 (`configs/base_dns64.yaml`); initialisation of `ft_dns64.pt` | 402 MB | `e34970ba13827bf58bc75468bacb6389c5603405fc78aff312b6ceca839380c9` |
| `colabA_bsrnn_feat112.pt` | stage-1 BSRNN (`configs/base_bsrnn.yaml`); initialisation of `ft_bsrnn.pt` | 137 MB | `ef401e9b4e55011f6a83d05a5f0196f68b995e20337298c8e8afafa19c058cf4` |
| `colabB_dns64_ft.pt` | stage-1 DNS64, second run (`configs/base_dns64_b.yaml`); member of earlier ensembles | 402 MB | `2102ee0d542eae5c075e25f44228408fec706501cb2e89d380c8dc1d18ad09c1` |

Each `.pt` file is a dict with `model` (state dict), `args` (the exact training arguments) and `state`
(step, elapsed time, evaluation log).
