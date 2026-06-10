# AIC 2026 Track 4 — Pipeline (AIO Paper + 4 Upgrades)

Re-implementation of [Hybrid, Unified and Iterative: A Novel Framework for Text-based Person Anomaly Retrieval](https://dl.acm.org/doi/10.1145/3701716.3717850) (WWW 2025, R@1=89.23%) với 4 upgrades + k-reciprocal rerank.

## Target hardware

**Colab A100 80GB high-RAM** xuyên suốt. Tất cả notebook tune cho cấu hình này: BF16, channels_last, torch.compile, Flash-SDPA, batch tăng lên 256/128 cho image encoders.

## Run order

| # | Notebook | Stage | ETA | Hardware |
|---|----------|-------|-----|----------|
| 00 | `00_manifest_qc.ipynb` | Build train/gallery/query manifests + val_split + val_zeroshot_scene | ~30 min | CPU |
| 01a | `01a_pe_g14_features.ipynb` | PE-Core-G14-448 encode (train + gallery + queries + val) | ~5-6h | A100-80GB |
| 01b | `01b_vitpose_features.ipynb` | ViTPose++ keypoints (cho LHP local crop trong 03) | ~3h | A100-80GB |
| 02 | `02_uit_train.ipynb` | **UIT training** (Swin-B + BERT, 22 epoch, ITC+ITM+MLM+0.1356·MIM) | ~8h | A100-80GB |
| 03 | `03_lhp_peg14_train.ipynb` | **LHP với PE-G14-448 + LoRA** (Upgrade 3) — freeze backbone, LoRA r=16 trên Q/K/V | ~2-3h | A100-80GB |
| 04 | `04_uit_inference.ipynb` | **LHP-guided Feature Selection + ITM Rerank** (paper Algorithm 1) — top-K từ LHP scores, cross-encoder UIT + itm_head rerank (Round 1) | ~45-60 min | A100-80GB |
| 05 | `05_blip2_inference.ipynb` | BLIP-2 ITC + ITM rerank top-1024 (Round 2) | ~1.5-2h | A100-80GB |
| 06 | `06_clip_inference.ipynb` | OpenAI CLIP ViT-L/14@336 (Round 3) | ~30 min | A100-80GB |
| 07 | `07_pe_g14_scores.ipynb` | PE-G14 score matrix từ 01a embeddings (Round 4) — **Upgrade 1** | ~5 min | T4/A100 |
| 08 | `08_kreciprocal_rerank.ipynb` | k-reciprocal Jaccard rerank cho mỗi model (k1=20, k2=6, λ=0.3) | ~40 min | A100-80GB |
| 09 | `09_adaptive_ensemble_submit.ipynb` | **Adaptive 4-way fusion** (Upgrades 2 + 4) + val gate + submission | ~15 min | A100 |

**Total wall-clock:** ~22-24h (có thể split 2-3 Colab session với resume từ Drive).

## 4 Upgrades vs paper

1. **Upgrade 1 — Add PE-Core-G14-448** as Round 4 model. Đã có sẵn embeddings từ `01a`; chỉ cần build score matrix ở `07`.
2. **Upgrade 2 — Extend iterative loop to Round 4**: UIT → BLIP-2 → CLIP → **PE-G14** với w_4=0.85.
3. **Upgrade 3 — Replace BEiT-3 với PE-G14-448** trong LHP. Freeze 100% backbone, LoRA r=16 trên Q/K/V của MHA. ~8-12M trainable params (~0.5% của 1.8B), lr=3e-4, 3 epoch.
4. **Upgrade 4 — Adaptive Weighting** thay weights cố định. Per-query, per-model confidence (top-1, margin-z, entropy, agreement) → softmax với floor η=0.05 → single-pass fusion. Auto-fallback nếu val mAP drop ≥0.5%.

**Extra upgrade:** k-reciprocal rerank ở `08` cho mỗi model trước fusion.

## Inputs expected on Drive

Mount: `/content/drive/MyDrive/aic2026_data/`

- `raw/` — Kaggle PAB dataset (annotation + train webp + test set)
- `manifests/` — parquet manifests (sau khi chạy 00)
- `output/` — features, checkpoints, scores (auto-synced từ local SSD mỗi chunk)
- `aio_repo/` (optional) — clone của paper repo nếu pre-cached, nếu không sẽ rsync từ workspace

## Outputs

Final submission tại `output/submission/answer.zip` chứa `answer.txt` — 1978 dòng × 10 gallery IDs space-separated.

Debug + validation:

- `output/validation/adaptive_weights_meta.json` — quyết định fusion winner
- `output/validation/ablation_table.parquet` — mAP/R@k cho từng config trên val + val_zeroshot_scene
- `output/submission/debug_topk.parquet` — per-query top-10 với per-model weight breakdown

## Resume / safety

- Mọi chunk NPZ atomic save + auto async sync sang Drive (`aic_colab_utils.sync_chunk_to_drive`)
- UIT training checkpoint save mỗi epoch + restore latest từ Drive khi resume
- LHP-LoRA save adapter (~40MB) sau mỗi epoch
- k-reciprocal output cùng filename với suffix `_rr` — re-run an toàn
- Pipeline có **fallback** ở 03 (LHP-LoRA nếu kém zero-shot) và 09 (adaptive nếu kém fixed-w)

## Validation strategy

- **val_split** (5% stratified theo label_type×scene): in-distribution val cho tune hyperparams
- **val_zeroshot_scene** (1 random scene class hold-out): proxy cho OOD test gallery
- Adaptive enable chỉ khi beat fixed-w trên **cả** 2 splits
- Sim2Real caveat: val cũng là synthetic → mAP tuyệt đối không khớp test, dùng làm tín hiệu *tương đối*
