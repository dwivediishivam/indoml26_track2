# Event-Gated Ensemble Enhancement for Indic Noise Event Removal

**Datathon@IndoML 2026, Track 2: Noise Event Removal** · Team **TDS** (IIIT Delhi):
Shivam Dwivedi, Animish Yadav, Shubhi Jain

| | |
|---|---|
| Final Codabench submission | **995085** (`SUB2_ens_ft.zip`), Phase 2 leaderboard rank **13** |
| Phase 2 test score | SI-SDR **12.78 dB** · ΔWER **+1.87** (synthetic +5.41, natural +0.21) · combined **14.65** · ASR audit **pass** |
| Model weights | https://huggingface.co/dwivediishivam/indoml26_track2 |
| Technical report | [`docs/technical_report.pdf`](docs/technical_report.pdf) (LaTeX source alongside) |

## 1. Task

Given noisy Indic speech recordings from the Vaani corpus, suppress non-speech noise events (horns, animals,
breathing, children, alarms, machines, music) while preserving speech. Each test clip is a 16 kHz mono WAV. The
score is `SI-SDR(synthetic clips) + 100 × ΔWER(all clips)`, where ΔWER is the pooled word-error-rate reduction of
the frozen mandated ASR `ARTPARK-IISc/SraVaani-1.0` measured on our enhanced audio.

## 2. Approach in brief

One learned enhancer is applied identically to every clip, with no routing by clip type and no noise lookup. The
final system averages two models:

1. **DNS64 (Demucs)**, initialised from the public DNS64 checkpoint and fine-tuned on in-domain event mixtures.
2. **Event-gated band-split RNN (BSRNN)**, trained from scratch. A complex spectral mask comes with a frame-level
   noise-event activity head *p(t)*. The output is `y = x + g(t)·(ŷ − x)`, where `g = 0.2 + 0.8·p(t)`. The model thus
   learns where events are, and leaves noise-free regions close to the input while still processing every frame.

Both are trained in two stages:
- **Stage 1** uses on-the-fly mixtures whose event count, loudness and duration follow the empirical statistics of
  the released validation synthetic set.
- **Stage 2** is a test-condition fine-tune on longer (10 s), denser (1–2× events) and louder (+0–3 dB) mixtures.
  Its checkpoints are selected on a **pseudo-test** built only from held-out validation clips.

Training adds an **ASR-consistency loss**: an L1 distance between the frozen SraVaani encoder features of the
enhanced and clean signals. This keeps the output transcription-friendly.

## 3. Training data and preprocessing

| Data | Use | Licence / source |
|---|---|---|
| **Vaani Noise Event Dataset** (`ARTPARK-IISc/Vaani-Noise-Event-Dataset`, 90,637 segments, 154.6 h; Gold/Silver/Bronze annotation tiers) | Clean-speech pool: non-event regions (≥1.5 s, at least 0.25 s away from any labelled event) of Gold/Silver clips whose 10th-percentile frame energy is < −28 dB below clip RMS. Capped at 8 h | CC BY 4.0 |
| **IndoML validation set** (2,501 clips: 1,806 natural, 695 synthetic with clean references) | 695 synthetic pairs split **560 train / 135 holdout** (`configs/split.json`). From the 560 we take the clean references (0.81 h) and the **1,783 noise events** cut from `noisy − clean` at the labelled spans (±50 ms) | organizers |
| **ESC-50** (2,000 environmental clips, resampled to 16 kHz, trimmed to the region within 30 dB of the peak) | 30% of the noise events (`p_ext = 0.3`), for generalisation beyond the validation noise | CC BY-NC 3.0 |
| **Google FLEURS** (Indic languages hi, bn, te, ta, kn, ml, mr, or, as, gu, pa, ne; first train shard, ≤600 utterances per language, ≤12 s each, 16 kHz) | Additional clean read speech | CC BY 4.0 |

`src/prep_data.py` writes these into memory-mapped float16 banks: noise 3,783 items (2.72 h) and clean speech
17.6 h for the B200 runs.

**Augmentation pipeline** (`src/train.py`, `MixDataset`, generated on the fly by the data-loader workers; every
batch is new):
1. Clean speech: draw a validation reference with p = 0.5, otherwise a Vaani or FLEURS segment. Join utterances
   (each rescaled within ±3 dB, separated by 50–600 ms of silence, with up to 0.3 s of leading silence) until the
   crop (6–10 s) is full. Scale to −30…−15 dBFS.
2. Event count: `k = n · U(1, L/4.5 s) · U(ev_mult)`, where *n* is drawn from the per-clip event counts of the
   validation synthetic set. Stage 2 uses `ev_mult = (1, 2)`.
3. Each event is a validation noise event (p = 0.7) or an ESC-50 crop (p = 0.3) with an empirical event duration and
   5 ms fades. It gets random polarity and is placed at a uniformly random offset. Its loudness relative to the clean
   RMS is drawn from the empirical validation distribution (median +4.3 dB), plus N(0, 2 dB), plus `U(db_shift)`
   (stage 2: 0–3 dB).
4. The frame-level event mask (hop 128) becomes the target of the activity head. The mixture and target are
   peak-normalised together if they would clip.

The simulated mixtures match the validation statistics: median input SI-SDR −2.5 dB vs −2.6 dB on real
validation, with event coverage 43% vs 45%.

## 4. Models, losses and training

**DNS64** (`third_party/denoiser`, Demucs, hidden 64, 33.5 M parameters). Waveform-domain U-Net encoder/decoder
with an LSTM bottleneck, initialised from `dns64-a7761ff99a7d5bb6.th`.

**Event-gated BSRNN** (`src/models.py`, 11.3 M parameters at feature width 112):
- STFT: 512-point, hop 128. The 257 bins are split into 37 bands (16×4, 16×8, 4×16, 1).
- Per band: GroupNorm, then a linear projection to 112 features.
- Six blocks, each a residual BLSTM over time followed by a residual BLSTM over bands.
- Per-band GLU MLPs predict a complex mask.
- A gate head predicts the event activity *p(t)* from all band features.

**Loss** for a mixture *x* with clean target *s*:
- BSRNN: `L = ½·[−SI-SDR(ŷ, s) + 0.5·MRSTFT(ŷ, s)] + ½·[−SI-SDR(y_g, s) + 0.5·MRSTFT(y_g, s)] + BCE(p, m) + 0.1·L_ASR`,
  where *y_g* is the gated output (floor 0.2) and *m* the frame event mask.
- DNS64: the same, without the gated and BCE terms.
- MRSTFT: L1 distance between log-magnitudes at FFT sizes 256, 512 and 1024.
- `L_ASR`: L1 distance between the frozen SraVaani-1.0 encoder outputs on ŷ and *s*, normalised by the target's
  mean magnitude, applied every second step.

**Optimisation:** AdamW (weight decay 1e-2), 500-step warm-up, then a cosine schedule driven by wall-clock time
down to 5% of the peak learning rate. Gradient clipping at 5; bf16 autocast for BSRNN. The checkpoint with the best
evaluation SI-SDR is kept.

| Stage / config | Model | Initialisation | Crops | LR | Time / hardware | Selection SI-SDR |
|---|---|---|---|---|---|---|
| 1a `base_dns64.yaml` | DNS64 | public DNS64 | 6 s | 1e-4 | 210 min, B200 | 19.04 dB (holdout) |
| 1b `base_bsrnn.yaml` | BSRNN f112 | scratch | 6 s | 1e-3 | 180 min, A100 | 19.04 dB (holdout, gated) |
| 1c `base_dns64_b.yaml` | DNS64 | public DNS64 | 8 s | 2e-4 | 150 min, A100 | 18.75 dB (holdout) |
| 2a `ft_dns64.yaml` | DNS64 | 1a | 10 s, events ×1–2, +0–3 dB | 5e-5 | 50 min, B200 | 12.93 dB (pseudo-test) |
| 2b `ft_bsrnn.yaml` | BSRNN f112 | 1b | same; no Vaani clean | 2e-4 | 50 min, B200 | 12.85 dB (pseudo-test) |

**Pseudo-test** (`src/make_pseudotest.py`): 106 clips of 2–3 joined holdout clean references (median 13.5 s),
carrying their own noise plus 0.5× extra holdout events at +1.5 dB. Its noisy-input median SI-SDR is −4.5 dB, which
matches the noise level we estimated for the real synthetic test clips (−4.48 dB). The real per-clip score
distribution of our first submission is reproduced closely (median 12.79 vs 12.27 dB).

**Final system** (`configs/inference.yaml`): mean of the 2a and 2b outputs, BSRNN gate floor 0.2, output RMS matched
to the input, PCM-16. Long clips are processed in 20 s chunks with a 2 s cross-fade. Transcripts come from the
unmodified SraVaani-1.0 (fp32, greedy TDT decoding, batch 4) on the written WAVs.

## 5. Results

| System | Validation holdout SI-SDR | Pseudo-test SI-SDR | Phase 2 test: SI-SDR / ΔWER / combined |
|---|---|---|---|
| Noisy input | −0.76 dB | −3.75 dB | — |
| DNS64 1a | 19.04 | 12.81 | — |
| BSRNN 1b (gated) | 19.04 | 12.67 | — |
| 1a + 1b | 19.78 | 13.36 | 12.66 / +1.80 / **14.46** (994924) |
| 1a + 1b + 1c | 19.71 | 13.36 | — / — / **14.41** (995155) |
| **2a + 2b (final)** | — | **13.52** | **12.78 / +1.87 / 14.65 (995085)** |

On the real test, ΔWER is +5.41 on synthetic clips and +0.21 on natural clips; the hidden ASR audit passed with
0% mismatch. Findings:
- **Mixing architectures helps.** DNS64 + BSRNN gains about 0.75 dB over either model alone and turns ΔWER
  positive. Averaging two DNS64 models gives nothing.
- **Noise level, not clip length, limits test performance.** Per-clip test SI-SDR correlates 0.82 with the
  estimated input SNR and 0.09 with duration. Clips below −6 dB input SNR (39%) average 8.5 dB; clips above 0 dB
  average 20.7 dB.
- **The holdout-built pseudo-test predicted the leaderboard well.** It estimated 14.52 for 995085, which scored
  14.65. The plain validation holdout overestimates by about 5 dB, because test clips are longer and noisier.

## 6. Environment and hardware

| | Training | Inference |
|---|---|---|
| Python | 3.10 (cluster), 3.13 (Colab) | 3.10–3.11 tested |
| PyTorch / CUDA | 2.9.1+cu128 / CUDA 12.8, driver 570.195.03 (cluster); 2.11.0+cu130 (Colab) | 2.9.1 (CPU or cu128) |
| GPU | 1× NVIDIA B200, 183,359 MiB (stages 1a, 2a, 2b; 16 CPU cores, 128 GB RAM, SLURM); 1× NVIDIA A100-SXM4 80 GB (stages 1b, 1c; Google Colab Pro) | any CUDA GPU with ≥8 GB, or CPU |
| Other | `transformers` 4.57.6, `huggingface_hub` 0.36.2, `sentencepiece` 0.2 | same |

System dependencies: `libsndfile` (needed by `soundfile`; bundled in the PyPI wheels on Linux and macOS).

**Pretrained dependencies:**
- **DNS64**, `dns64-a7761ff99a7d5bb6.th` from `facebookresearch/denoiser`
  (https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th). Initialises the DNS64 members.
- **SraVaani-1.0**, `ARTPARK-IISc/SraVaani-1.0` (https://huggingface.co/ARTPARK-IISc/SraVaani-1.0). The mandated ASR,
  frozen: its encoder provides the ASR-consistency loss, and the full model produces the transcripts. The model is
  gated, so accept its terms on Hugging Face and set `HF_TOKEN`.

## 7. Setup

```bash
git clone https://github.com/dwivediishivam/indoml26_track2.git && cd indoml26_track2
python -m venv .venv && source .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # or the CPU wheel
pip install -r requirements_inference.txt          # add requirements_train.txt for training
export HF_TOKEN=<your Hugging Face token>            # gated ASR download
python checkpoints/download_checkpoints.py           # final weights, SHA-256 verified
```

## 8. Inference

```bash
python main.py INPUT_DIR OUTPUT_DIR                  # same as: python inference/infer.py INPUT_DIR OUTPUT_DIR
python inference/infer.py INPUT_DIR OUTPUT_DIR --zip submission.zip   # also build a validated Codabench ZIP
bash scripts/run_inference.sh INPUT_DIR OUTPUT_DIR submission.zip
```

- **Input:** a folder of 16 kHz mono WAVs. An `audio/` sub-folder, as in the Codabench `input_data` ZIP, is detected
  automatically.
- **Output:** `OUTPUT_DIR/<clip>.wav` for every input clip (PCM-16, 16 kHz, mono, same number of samples) and
  `OUTPUT_DIR/transcripts.jsonl`, one line per clip: `{"clip_id": "<clip>", "text": "<SraVaani transcript>"}`.
- **Submission ZIP:** with `--zip`, `src/package.py` validates the format, lengths and transcript coverage, then
  writes all files at the ZIP root.

On one B200 GPU the 3,178-clip Phase 2 test set (10.9 h) takes about 15 minutes in total (enhancement, ASR and
packaging). Re-running the released checkpoints reproduces the submitted transcripts exactly. The audio matches the
submitted WAVs to 72–79 dB SI-SDR on CPU; small differences come from CPU vs GPU floating-point arithmetic.

## 9. Training

```bash
pip install -r requirements_train.txt
HF_TOKEN=... bash scripts/fetch_data.sh DATA   # ESC-50, FLEURS Indic shards, Vaani parquet shards, SraVaani-1.0
# unzip the IndoML validation set (validation.zip from the competition page) to DATA/validation
bash scripts/run_training.sh DATA
```

`run_training.sh` runs every step in order:
1. Build the banks (`src/prep_data.py`).
2. Build the pseudo-test (`src/make_pseudotest.py`).
3. Train stage 1 (`configs/base_*.yaml`).
4. Train stage 2 (`configs/ft_*.yaml`).

Any single stage can also be run alone:
```bash
python train/train.py --config configs/base_bsrnn.yaml --bank DATA/bank --val_dir DATA/validation \
    --out runs/base_bsrnn --asr_dir models/sravaani
```

The SLURM job files (single B200 GPU) and Colab A100 notebooks used for the actual runs are in `scripts/slurm/`
and `scripts/colab/`. They use the cluster's folder layout (`~/indoml-t2/{code,assets,data,bank,runs}`); insert
your own Hugging Face token into the notebooks. `scripts/run_ensemble.sh` and
`scripts/final_select.sh` score candidate ensembles on validation, the pseudo-test and natural clips, then
package the best.

## 10. Repository layout

```
main.py                     organizer entry point: python main.py INPUT OUTPUT
inference/infer.py          enhancement + mandated ASR + optional ZIP
train/train.py              YAML-driven training of one stage
src/                        models.py (BSRNN, DNS64 wrapper, ASR-consistency loss), train.py, prep_data.py,
                            enhance.py, asr.py, package.py, score_val.py, make_pseudotest.py, common.py
configs/                    stage configs, inference.yaml, split.json (560/135 validation split)
checkpoints/                download_checkpoints.py, checkpoint table (weights on Hugging Face)
scripts/                    run_inference.sh, run_training.sh, fetch_data.sh, ensemble selection, slurm/, colab/
third_party/denoiser/       facebookresearch/denoiser (CC BY-NC 4.0)
docs/                       technical report (PDF + LaTeX)
```

## 11. References

1. Vaani: S. Pulikodan et al., "VAANI: Capturing the language landscape for an inclusive digital India," arXiv:2603.28714, 2026.
2. A. Défossez, G. Synnaeve, Y. Adi, "Real Time Speech Enhancement in the Waveform Domain," Interspeech 2020 (DNS64, `facebookresearch/denoiser`).
3. Y. Luo, J. Yu, "Music Source Separation with Band-Split RNN," IEEE/ACM TASLP, 2023.
4. J. Le Roux et al., "SDR – half-baked or well done?," ICASSP 2019 (SI-SDR).
5. R. Yamamoto et al., "Parallel WaveGAN," ICASSP 2020 (multi-resolution STFT loss).
6. K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," ACM MM 2015.
7. A. Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech," IEEE SLT 2022.
8. ARTPARK-IISc, SraVaani-1.0 ASR, https://huggingface.co/ARTPARK-IISc/SraVaani-1.0.
9. H. Xu et al., "Efficient Sequence Transduction by Jointly Predicting Tokens and Durations" (TDT), ICML 2023.
10. I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019.
