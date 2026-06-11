# AIC 2026 Track 4 — Pipeline (AIO Paper + 4 Upgrades)

Re-implementation of [Hybrid, Unified and Iterative: A Novel Framework for Text-based Person Anomaly Retrieval](https://dl.acm.org/doi/10.1145/3701716.3717850) (WWW 2025, R@1=89.23%) with 4 upgrades + k-reciprocal rerank.

## Target hardware

**Colab A100 80GB high-RAM** throughout. All notebooks are tuned for this configuration: BF16, channels_last, torch.compile, Flash-SDPA, batch sizes increased to 256/128 for image encoders.

## Run order

| # | Notebook | Stage | ETA | Hardware |
|---|----------|-------|-----|----------|
| 00 | `00_manifest_qc.ipynb` | Build train/gallery/query manifests + val_split + val_zeroshot_scene | ~30 min | CPU |
| 01a | `01a_pe_g14_features.ipynb` | PE-Core-G14-448 encode (train + gallery + queries + val) | ~5-6h | A100-80GB |
| 01b | `01b_vitpose_features.ipynb` | ViTPose++ keypoints (for LHP local crop in 03) | ~3h | A100-80GB |
| 02 | `02_uit_train.ipynb` | **UIT training** (Swin-B + BERT, 22 epoch, ITC+ITM+MLM+0.1356·MIM) | ~8h | A100-80GB |
| 03 | `03_lhp_peg14_train.ipynb` | **LHP with PE-G14-448 + LoRA** (Upgrade 3) — freeze backbone, LoRA r=16 on Q/K/V | ~2-3h | A100-80GB |
| 04 | `04_uit_inference.ipynb` | **LHP-guided Feature Selection + ITM Rerank** (paper Algorithm 1) — top-K from LHP scores, cross-encoder UIT + itm_head rerank (Round 1) | ~45-60 min | A100-80GB |
| 05 | `05_blip2_inference.ipynb` | BLIP-2 ITC + ITM rerank top-1024 (Round 2) | ~1.5-2h | A100-80GB |
| 06 | `06_clip_inference.ipynb` | OpenAI CLIP ViT-L/14@336 (Round 3) | ~30 min | A100-80GB |
| 07 | `07_pe_g14_scores.ipynb` | PE-G14 score matrix from 01a embeddings (Round 4) — **Upgrade 1** | ~5 min | T4/A100 |
| 08 | `08_kreciprocal_rerank.ipynb` | k-reciprocal Jaccard rerank for each model (k1=20, k2=6, λ=0.3) | ~40 min | A100-80GB |
| 09 | `09_adaptive_ensemble_submit.ipynb` | **Adaptive 4-way fusion** (Upgrades 2 + 4) + val gate + submission | ~15 min | A100 |

**Total wall-clock:** ~22-24h (can be split across 2-3 Colab sessions with resume from Drive).

## 4 Upgrades vs paper

1. **Upgrade 1 — Add PE-Core-G14-448** as Round 4 model. Embeddings from `01a` are already available; only build the score matrix in `07`.
2. **Upgrade 2 — Extend iterative loop to Round 4**: UIT → BLIP-2 → CLIP → **PE-G14** with w_4=0.85.
3. **Upgrade 3 — Replace BEiT-3 with PE-G14-448** in LHP. Freeze 100% backbone, LoRA r=16 on Q/K/V of MHA. ~8-12M trainable params (~0.5% of 1.8B), lr=3e-4, 3 epoch.
4. **Upgrade 4 — Adaptive Weighting** replaces fixed weights. Per-query, per-model confidence (top-1, margin-z, entropy, agreement) → softmax with floor η=0.05 → single-pass fusion. Auto-fallback if val mAP drops ≥0.5%.

**Extra upgrade:** k-reciprocal rerank in `08` for each model before fusion.

## Inputs expected on Drive

Mount: `/content/drive/MyDrive/aic2026_data/`

- `raw/` — Kaggle PAB dataset (annotation + train webp + test set)
- `manifests/` — parquet manifests (after running 00)
- `output/` — features, checkpoints, scores (auto-synced from local SSD each chunk)
- `aio_repo/` (optional) — clone of paper repo if pre-cached, otherwise rsync from workspace

## Outputs

Final submission at `output/submission/answer.zip` containing `answer.txt` — 1978 lines × 10 gallery IDs space-separated.

Debug + validation:

- `output/validation/adaptive_weights_meta.json` — fusion winner decision
- `output/validation/ablation_table.parquet` — mAP/R@k for each config on val + val_zeroshot_scene
- `output/submission/debug_topk.parquet` — per-query top-10 with per-model weight breakdown

## Resume / safety

- Every chunk NPZ atomic save + auto async sync to Drive (`aic_colab_utils.sync_chunk_to_drive`)
- UIT training checkpoint saved every epoch + restore latest from Drive on resume
- LHP-LoRA saves adapter (~40MB) after each epoch
- k-reciprocal output uses the same filename with suffix `_rr` — safe to re-run
- Pipeline has **fallback** at 03 (LHP-LoRA if worse than zero-shot) and 09 (adaptive if worse than fixed-w)

## Validation strategy

- **val_split** (5% stratified by label_type×scene): in-distribution val for tuning hyperparams
- **val_zeroshot_scene** (1 random scene class held out): proxy for OOD test gallery
- Adaptive enabled only when it beats fixed-w on **both** splits
- Sim2Real caveat: val is also synthetic → absolute mAP does not match test; use as a *relative* signal
