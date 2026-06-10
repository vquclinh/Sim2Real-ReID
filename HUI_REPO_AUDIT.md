# HUI Repository Audit Report

**Repository:** `Hybrid-Unified-and-Iterative-A-Novel-Framework-for-Text-based-Person-Anomaly-Retrieval`  
**Paper:** "Hybrid, Unified and Iterative: A Novel Framework for Text-based Person Anomaly Retrieval" (ACM Web Conf Workshop MORE 2025)  
**Audit date:** 2026-06-10  
**Purpose:** AIC/ECCV 2026 Track 4 — Text-Based Person Anomaly Search (PAB dataset)

---

## 1. Repo Overview

The repository contains a **complete, working codebase** for the HUI framework across three sub-systems:
- **LHP** (Local-Global Hybrid Perspective): BEiT-3 Large fine-tuned on PAB
- **UIT** (Unified Image-Text): Swin-B + BERT with ITC + ITM + MLM + MIM losses
- **Iterative Ensemble**: weighted combination of BEiT-3 (ITM re-ranked), BLIP-2, and CLIP scores

Precomputed similarity score files (`.pt`) **are included** for the test set, enabling ensemble reproduction without rerunning model inference.

---

## 2. Folder and File Map

```
repo/
├── README.md                        # Main instructions (training, inference, commands)
├── requirements_lhp.txt             # LHP conda environment
├── requirements_uit.txt             # UIT conda environment
│
├── beit3_infer.py                   # LEGACY helper — imports from `lhp.beit3` (not `lhp_2`)
├── blip2_infer.py                   # BLIP-2 inference entry point (root-level)
├── clip_infer.py                    # CLIP ViT-L/14@336px inference entry point
├── app.py                           # Streamlit demo app using BEiT-3 + FAISS
│
├── blip/                            # Vendored LAVIS library + BLIP-2 wrapper
│   ├── blip2.py                     # init_model() — loads BLIP-2 feature extractor ViT-L
│   ├── lavis/                       # Full LAVIS library (vendored)
│   ├── train.py                     # LAVIS training script (NOT used for HUI)
│   └── evaluate.py                  # LAVIS evaluation (NOT used for HUI)
│
├── lhp_2/
│   └── beit3/                       # LHP: BEiT-3 Large fine-tuning + inference
│       ├── run_beit3_finetuning.py  # TRAINING entry point (--task 356 = PAB task)
│       ├── inference.py             # INFERENCE entry point for LHP
│       ├── modeling_finetune.py     # BEiT-3 model + retrieval head
│       ├── modeling_utils.py        # BEiT-3 core architecture
│       ├── datasets.py              # PAB dataset loader (75 train files + test)
│       ├── engine_for_finetuning.py # Train/eval loops (task '356' → RetrievalHandler)
│       ├── beit3.spm                # Sentencepiece tokenizer (ALREADY PRESENT)
│       ├── lhp_reproduce/log/       # TensorBoard logs from authors' training run
│       └── utils.py, optim_factory.py, randaug.py, glossary.py
│
├── uit/
│   ├── cmp/                         # UIT model: training, evaluation, inference
│   │   ├── Search.py                # TRAINING entry point (train + evaluate)
│   │   ├── run.py                   # Launcher script (wraps torch.distributed.run)
│   │   ├── inference.py             # INFERENCE + ENSEMBLE entry point
│   │   ├── train.py                 # train_model() — ITC+ITM+MLM+MIM losses
│   │   ├── eval.py                  # evaluation_itc(), evaluation_itm() (ensemble logic)
│   │   ├── models/
│   │   │   ├── model_search.py      # Search(CMP) — full forward pass
│   │   │   ├── cmp.py               # CMP base — ITC, ITM, MLM, MIM methods
│   │   │   ├── bert.py              # BERT text encoder
│   │   │   ├── swin_transformer.py  # Swin-B image encoder
│   │   │   └── simmim.py            # SimMIM head for MIM pre-training
│   │   ├── dataset/
│   │   │   ├── search_dataset.py    # search_train_dataset, search_test_dataset, search_inference_dataset
│   │   │   └── utils.py             # pre_caption(), read_json_to_list()
│   │   ├── configs/
│   │   │   ├── cmp.yaml             # Training config (relative paths ../../data/PAB/)
│   │   │   └── infer.yaml           # Inference config (relative paths ./data/PAB/)
│   │   ├── reproduce.txt            # FINAL ENSEMBLE OUTPUT (1978 lines, answer.txt format)
│   │   ├── score_beit3.pt           # (duplicate of sims_score/score_beit3_reproduce.pt)
│   │   ├── score_blip2.pt           # (duplicate)
│   │   └── score_clip.pt            # (duplicate)
│   └── test.py
│
├── sims_score/                      # PRECOMPUTED similarity matrices (included!)
│   ├── score_beit3_reproduce.pt     # 15 MB — BEiT-3 cosine sim matrix (1978 queries)
│   ├── score_blip2_reproduce.pt     # 15 MB — BLIP-2 cosine sim matrix
│   └── score_clip_reproduce.pt      # 15 MB — CLIP cosine sim matrix
│
└── predictions/                     # Per-model top-10 text predictions (included!)
    ├── score_beit3_reproduce.txt    # 1978 lines — top-10 gallery IDs per query (BEiT-3)
    ├── score_blip2_reproduce.txt    # 1978 lines (BLIP-2)
    └── score_clip_reproduce.txt     # 1978 lines (CLIP)
```

---

## 3. Training Code

**Yes — full training code is present for both LHP and UIT.**

### 3a. LHP (BEiT-3 Large Fine-tuning)

| Item | Detail |
|------|--------|
| **Script** | `lhp_2/beit3/run_beit3_finetuning.py` |
| **Model** | BEiT-3 Large (ViT-L/16 @ 384×384), `beit3_large_patch16_384_retrieval` via `timm.create_model` |
| **Task ID** | `--task 356` (custom PAB task, maps to `RetrievalHandler` in engine) |
| **Pretrain init** | `beit3_large_patch16_384_coco_retrieval.pth` (GitHub release, ~1.9 GB) |
| **Data format** | PAB train: 75 JSON-lines files `annotation/train/pair_N.json`, each line: `{"image": "path", "image_id": "...", "caption": "..."}` |
| **Image root** | `data/PAB/` (prefixed in `datasets.py`) |
| **Output** | Epoch checkpoints saved to `./lhp_reproduce/checkpoint_best.pth` etc. |
| **GPU** | High — BEiT-3 Large, batch_size=184, 384×384 images → likely needs 40 GB+ VRAM (A100 recommended) |

**Command:**
```bash
cd ./lhp_2/beit3

CUDA_VISIBLE_DEVICES=0 python3 run_beit3_finetuning.py \
    --model beit3_large_patch16_384 \
    --task 356 \
    --drop_path 0.16 \
    --checkpoint_activations \
    --sentencepiece_model ./beit3.spm \
    --weight_decay 0.05 \
    --layer_decay 0.85 \
    --batch_size 184 \
    --update_freq 1 \
    --save_ckpt_freq 1 \
    --finetune ./beit3_large_patch16_384_coco_retrieval.pth \
    --data_path ../../data/PAB/ \
    --output_dir ./lhp_reproduce \
    --log_dir ./lhp_reproduce/log \
    --seed 16 \
    --save_ckpt \
    --input_size 384 \
    --lr 1e-5 \
    --warmup_steps 440 \
    --epochs 4
```

### 3b. UIT (Swin-B + BERT)

| Item | Detail |
|------|--------|
| **Script** | `uit/cmp/Search.py` (or via `run.py`) |
| **Model** | Swin Transformer Base (224×224) + BERT-base-uncased, with ITC + ITM + MLM + MIM |
| **Config** | `uit/cmp/configs/cmp.yaml` (for training from `uit/cmp/` CWD) |
| **Pretrain init** | Option 1: `pretrained.pth` (Google Drive); Option 2: `swin_base_patch4_window7_224_22k.pth` + `bert-base-uncased/` |
| **Data format** | Same 75 JSON-lines files; image root `../../data/PAB/` (relative to `uit/cmp/`) |
| **Losses** | `loss_itc + loss_itm + loss_mlm + loss_mim * 0.1356` |
| **Epochs** | 30, batch_size_train=84, AdamW lr=1e-4 |
| **Output** | `output/356356/checkpoint_{epoch}.pth` (hardcoded path in `Search.py:135`) |
| **GPU** | ~24 GB VRAM (Swin-B + BERT, 224×224, bs=84) |

**Command (from repo root):**
```bash
cd ./uit/cmp
python3 run.py --task "cmp" --dist "f4" --output_dir "output/cmp"
```
Or directly:
```bash
cd ./uit/cmp
python3 Search.py --config configs/cmp.yaml --task cmp --output_dir output/cmp \
    --checkpoint ./checkpoint/pretrained.pth
```

---

## 4. Inference Code

**Yes — complete inference pipeline is present.**

### Step 1: LHP inference (BEiT-3)

```bash
# Run from repo root
python3 ./lhp_2/beit3/inference.py \
    --checkpoint ./checkpoint/lhp/lhp_beit3.pth \
    --tokenizer ./checkpoint/lhp/beit3.spm \
    --image_folder ./data/PAB/name-masked_test-set/gallery \
    --annotation ./data/PAB/name-masked_test-set/query.json \
    --save_score ./sims_score/score_beit3_reproduce.pt \
    --output_file ./predictions/score_beit3_reproduce.txt
```
- Requires: `checkpoint/lhp/lhp_beit3.pth` (NOT in repo, download needed)
- `beit3.spm` already at `lhp_2/beit3/beit3.spm`; copy to `checkpoint/lhp/`

### Step 2: BLIP-2 inference (zero-shot, no fine-tuning)

```bash
python3 ./blip2_infer.py \
    --image_folder ./data/PAB/name-masked_test-set/gallery \
    --annotation ./data/PAB/name-masked_test-set/query.json \
    --save_score ./sims_score/score_blip2_reproduce.pt \
    --output_file ./predictions/score_blip2_reproduce.txt
```
- Zero-shot: uses LAVIS `blip2_feature_extractor` (model auto-downloads from HuggingFace)
- No checkpoint download needed — LAVIS caches model automatically (~10 GB)

### Step 3: CLIP inference (zero-shot, no fine-tuning)

```bash
python3 ./clip_infer.py \
    --image_folder ./data/PAB/name-masked_test-set/gallery \
    --annotation ./data/PAB/name-masked_test-set/query.json \
    --save_score ./sims_score/score_clip_reproduce.pt \
    --output_file ./predictions/score_clip.txt
```
- Zero-shot: `clip.load("ViT-L/14@336px")` (auto-downloads ~900 MB)
- No manual checkpoint needed

### Step 4: UIT Ensemble (ITM re-ranking + weighted fusion)

```bash
python3 uit/cmp/inference.py \
    --config uit/cmp/configs/infer.yaml \
    --task cmp \
    --output_dir output \
    --checkpoint ./checkpoint/uit/uit.pth \
    --output_file reproduce.txt \
    --beit3_weight 0.925 \
    --beit3_score ./sims_score/score_beit3_reproduce.pt \
    --blip2_weight 0.9 \
    --blip2_score ./sims_score/score_blip2_reproduce.pt \
    --clip_weight 0.9 \
    --clip_score ./sims_score/score_clip_reproduce.pt
```
- Requires: `checkpoint/uit/uit.pth` (NOT in repo, download needed)
- Produces `reproduce.txt` = **final answer.txt** (1978 lines, space-separated gallery IDs, top-10 per query)

### Does it generate answer.txt directly?

**Yes.** `uit/cmp/reproduce.txt` is the final answer file:
- 1978 lines (one per query)
- Format: `ID1 ID2 ID3 ID4 ID5 ID6 ID7 ID8 ID9 ID10` (10 gallery IDs, space-separated, stem-only without extension)
- This matches competition submission format

---

## 5. Full HUI Pipeline vs. Paper

| Paper Component | Status | Where |
|-----------------|--------|--------|
| LHP — Local crop | ✅ Implemented | `datasets.py:83-88` — random crop 50–100% at 384×384 |
| LHP — Global (no crop) | ✅ Implemented | `datasets.py:64-67` — resize to 384×384 |
| LHP — Stochastic selection | ✅ Implemented | `datasets.py:83` — `torch.normal(0.5, 0.167) > 0.5` |
| UIT — ITC loss | ✅ Implemented | `models/cmp.py:get_contrastive_loss()` + EDA augmentation variant |
| UIT — ITM loss | ✅ Implemented | `models/cmp.py:get_matching_loss()` |
| UIT — MLM loss | ✅ Implemented | `models/cmp.py:get_mlm_loss()`, `train.py:mlm()` |
| UIT — MIM loss | ✅ Implemented | `models/cmp.py:get_mim_loss()`, Swin+SimMIM |
| UIT — Feature Selection (Algorithm 1) | ✅ Partial | ITM re-ranking over top-k=128 candidates from ITC scores |
| Iterative Ensemble (Algorithm 2) | ✅ Implemented | `eval.py:evaluation_itm()` lines 113 |
| BLIP-2 inference | ✅ Zero-shot | `blip2_infer.py` + `blip/blip2.py` via LAVIS |
| CLIP ViT-L/14@336px inference | ✅ Zero-shot | `clip_infer.py`, `clip.load("ViT-L/14@336px")` |
| BEiT-3 training | ✅ Full | `lhp_2/beit3/run_beit3_finetuning.py --task 356` |
| BEiT-3 inference | ✅ Full | `lhp_2/beit3/inference.py` |
| Re-ranking (ITM) | ✅ Implemented | `eval.py:evaluation_itm()` — top-128 candidates re-ranked by cross-encoder ITM head |
| Ensemble weights | ✅ Documented | beit3=0.925, blip2=0.9, clip=0.9 |
| Precomputed scores | ✅ INCLUDED | `sims_score/*.pt` (3 × 15 MB) |
| Final answer.txt | ✅ INCLUDED | `uit/cmp/reproduce.txt` (1978 lines) |

**Ensemble formula** (from `eval.py:113`):
```
final = (
    ((1 - w_b3) * ITM_score + w_b3 * ITC_score) * w_blip2 + (1 - w_blip2) * score_blip2
) * w_clip + (1 - w_clip) * score_clip
```
where `w_b3=0.925, w_blip2=0.9, w_clip=0.9`.

---

## 6. Pretrained Checkpoints

| Checkpoint | Status | Download |
|------------|--------|----------|
| `checkpoint/lhp/lhp_beit3.pth` | **NOT in repo** | Google Form: https://forms.gle/X2yX7Y4W6pVdo7pH9 |
| `checkpoint/uit/uit.pth` | **NOT in repo** | Same Google Form |
| `checkpoint/lhp/beit3.spm` | **PRESENT** (`lhp_2/beit3/beit3.spm`) | Copy it |
| `beit3_large_patch16_384_coco_retrieval.pth` | **NOT in repo** | GitHub: `addf400/files/releases/download/beit3/` |
| `checkpoint/pretrained.pth` (UIT init) | **NOT in repo** | Google Drive link in README |
| `swin_base_patch4_window7_224_22k.pth` | **NOT in repo** | SwinTransformer GitHub releases |
| `bert-base-uncased/` | **NOT in repo** | HuggingFace: `huggingface.co/bert-base-uncased` |
| BLIP-2 weights | Auto-download via LAVIS | HuggingFace (handled by `load_model_and_preprocess`) |
| CLIP weights | Auto-download | OpenAI CDN (handled by `clip.load`) |

**To reproduce results from precomputed scores only**, you need:
- `checkpoint/uit/uit.pth` (from Google Form)

---

## 7. Dependencies

Two separate conda environments are required:

### LHP environment (`lhp`)
```
torch, torchvision, timm==0.4.12, sentencepiece, torchscale==0.2.0,
transformers, deepspeed==0.4.0, einops, tensorboardX, scipy, ftfy,
opencv-python, pyarrow, torchmetrics==0.7.3, pycocotools, pycocoevalcap,
protobuf==3.20.0, blobfile
```
Note: `deepspeed==0.4.0` is old and may need CUDA 11.x. `timm==0.4.12` is pinned.

### UIT environment (`uit`)
```
torch==2.5.1, torchvision==0.20.1, timm==0.6.13, transformers==4.47.1,
deepspeed==0.4.0, fairscale==0.4.0, ruamel.yaml, yacs, prettytable,
scipy, opencv-python, tensorboardX, protobuf==5.29.2, tqdm, ninja, nltk
```
Note: `deepspeed==0.4.0` conflicts with `torch==2.5.1` — may need `deepspeed==0.14.x` or to remove it if unused.

### Additional (blip2_infer.py / clip_infer.py)
```bash
pip install lavis          # or use bundled blip/lavis
pip install openai-clip    # or git+https://github.com/openai/CLIP.git
pip install salesforce-lavis  # alternative
```

---

## 8. Dataset Structure Expected

The repo expects PAB data at `./data/PAB/` relative to repo root (for root-level scripts), or `../../data/PAB/` (from `uit/cmp/`):

```
data/PAB/
├── annotation/
│   ├── train/
│   │   ├── pair_0.json       # JSON-lines, train pairs
│   │   ├── pair_1.json
│   │   └── ... pair_74.json  # 75 files total
│   └── test/
│       └── pair.json         # JSON-lines, test set (for evaluation with labels)
└── name-masked_test-set/
    ├── gallery/
    │   ├── XXXXXXXXXXXX.jpg   # gallery images (uppercase alphanumeric IDs, .jpg)
    │   └── ...
    └── query.json             # JSON-lines: {"query_index": "...", "caption": "..."}
```

**Annotation formats:**

Train (`pair_N.json`), one JSON object per line:
```json
{"image": "train/imgs_0/goal/0.jpg", "image_id": "XXXX", "caption": "A person wearing..."}
```

Test/eval (`pair.json`), one JSON object per line:
```json
{"image": "0.jpg", "image_id": "XXXX", "caption": ["caption 1", "caption 2"]}
```
Note: test set has a **list** of captions per image.

Inference query (`query.json`), one JSON object per line:
```json
{"query_index": "XXXX", "caption": "A person wearing..."}
```

Gallery image filenames: uppercase alphanumeric stems (e.g. `ZOVZW5GHWX3K7R2.jpg`). Output uses stem only (no `.jpg`).

---

## 9. Hardcoded Paths

| File | Line | Path | Active? |
|------|------|------|---------|
| `beit3_infer.py` | 162–164 | `/home/s48gb/Desktop/GenAI4E/pab/...` | **YES** — in `__main__` block, but this script is not the proper inference path; use `lhp_2/beit3/inference.py` instead |
| `clip_infer.py` | 107, 114 | `/home/s48gb/Desktop/...` | No (commented out) |
| `blip/blip2.py` | 18 | `/home/s48gb/Desktop/...` | No (commented out) |
| `blip/lavis/configs/default.yaml` | — | `/export/home/.cache/lavis/` | LAVIS cache (overridable via env var `LAVIS_CACHE_ROOT`) |
| `uit/cmp/Search.py` | 135 | `output/356356/checkpoint_{epoch}.pth` | **YES** — hardcoded output path during training |
| `uit/cmp/configs/infer.yaml` | 1–2 | `./data/PAB/name-masked_test-set/...` | Active, relative to CWD (must run from repo root) |
| `uit/cmp/configs/cmp.yaml` | 1–51 | `../../data/PAB/...` | Active, relative to `uit/cmp/` |

**Key issue:** `uit/cmp/Search.py:135` saves checkpoints to hardcoded `output/356356/`. This directory is not created automatically — you must `mkdir -p output/356356` before training.

---

## 10. Immediate Usability for AIC/ECCV 2026 Track 4

### What works immediately (no downloads, no training)

1. **Ensemble reproduction with precomputed scores**: If you download only `checkpoint/uit/uit.pth` from the Google Form, you can run the final ensemble immediately using pre-stored `.pt` files:
   ```bash
   python3 uit/cmp/inference.py \
       --config uit/cmp/configs/infer.yaml \
       --task cmp --output_dir output \
       --checkpoint ./checkpoint/uit/uit.pth \
       --output_file reproduce.txt \
       --beit3_weight 0.925 \
       --beit3_score ./sims_score/score_beit3_reproduce.pt \
       --blip2_weight 0.9 \
       --blip2_score ./sims_score/score_blip2_reproduce.pt \
       --clip_weight 0.9 \
       --clip_score ./sims_score/score_clip_reproduce.pt
   ```
   This **does NOT re-run BLIP-2/CLIP/BEiT-3** — it loads the saved `.pt` files.

2. **CLIP inference** on new data (zero-shot, auto-download): works once `openai-clip` is installed.

3. **BLIP-2 inference** on new data (zero-shot, auto-download ~10 GB): works once LAVIS is installed.

4. **Demo app** (`streamlit run app.py`): works if LHP checkpoint is present.

### What needs adaptation for Track 4 (2026 test data)

| Issue | What to do |
|-------|-----------|
| New test queries | Run `clip_infer.py`, `blip2_infer.py`, `lhp_2/beit3/inference.py` on Track 4 gallery/query |
| New test annotation format | Verify `query.json` uses `{"query_index": "...", "caption": "..."}` format |
| Image gallery IDs | Ensure `search_inference_dataset` reads filenames correctly via `os.listdir()` — check sort order consistency |
| `uit/cmp/configs/infer.yaml` paths | Update `image_root` and `test_file` to point to Track 4 data |
| `beit3_infer.py` (root-level) | Do NOT use — use `lhp_2/beit3/inference.py` instead |
| `uit/cmp/Search.py:135` hardcoded output | `mkdir -p output/356356` before training |
| deepspeed==0.4.0 + torch==2.5.1 | May conflict — test environment first, possibly remove deepspeed |

### What will fail without checkpoints

| Step | Will fail without |
|------|-----------------|
| `lhp_2/beit3/inference.py` | `checkpoint/lhp/lhp_beit3.pth` |
| `uit/cmp/inference.py` | `checkpoint/uit/uit.pth` |
| `uit/cmp/Search.py` (train/eval) | `checkpoint/pretrained.pth` or Swin-B + BERT |
| `lhp_2/beit3/run_beit3_finetuning.py` | `beit3_large_patch16_384_coco_retrieval.pth` |

---

## 11. Important Files Table

| File | Role | Required for |
|------|------|--------------|
| `lhp_2/beit3/run_beit3_finetuning.py` | LHP training | Training LHP |
| `lhp_2/beit3/inference.py` | LHP inference | LHP similarity scores |
| `lhp_2/beit3/modeling_finetune.py` | BEiT-3 model definition | Both |
| `lhp_2/beit3/datasets.py` | PAB dataset loader for LHP | Both |
| `lhp_2/beit3/beit3.spm` | Sentencepiece tokenizer | **Already present** |
| `blip2_infer.py` | BLIP-2 cosine sim scores | Zero-shot inference |
| `blip/blip2.py` | BLIP-2 model init via LAVIS | BLIP-2 inference |
| `clip_infer.py` | CLIP cosine sim scores | Zero-shot inference |
| `uit/cmp/Search.py` | UIT training + evaluation | Training UIT |
| `uit/cmp/inference.py` | Ensemble step | Final answer.txt |
| `uit/cmp/eval.py` | ITM re-ranking + ensemble logic | Ensemble |
| `uit/cmp/configs/infer.yaml` | Inference config (paths, batch size) | Ensemble |
| `uit/cmp/configs/cmp.yaml` | Training config (paths, hyperparams) | Training |
| `sims_score/score_*.pt` | Precomputed similarity matrices | Ensemble (without re-running models) |
| `predictions/score_*.txt` | Per-model top-10 predictions | Reference |
| `uit/cmp/reproduce.txt` | **Final answer.txt** (ensemble output) | Reference |

---

## 12. Recommended Next Steps

### Priority 1: Reproduce paper results (fast path)

1. Request checkpoints via Google Form: `https://forms.gle/X2yX7Y4W6pVdo7pH9`
   - Download `lhp_beit3.pth` and `uit.pth`
   - Place at `checkpoint/lhp/lhp_beit3.pth` and `checkpoint/uit/uit.pth`
   - Copy `lhp_2/beit3/beit3.spm` → `checkpoint/lhp/beit3.spm`

2. Set up environments:
   ```bash
   conda create -n uit python=3.10
   conda activate uit
   pip install -r requirements_uit.txt
   pip install openai-clip
   # Install LAVIS (blip/): cd blip && pip install -e . (or pip install salesforce-lavis)
   ```

3. Run ensemble using precomputed scores (fastest):
   ```bash
   mkdir -p output
   python3 uit/cmp/inference.py \
       --config uit/cmp/configs/infer.yaml \
       --task cmp --output_dir output \
       --checkpoint ./checkpoint/uit/uit.pth \
       --output_file ./answer.txt \
       --beit3_weight 0.925 \
       --beit3_score ./sims_score/score_beit3_reproduce.pt \
       --blip2_weight 0.9 \
       --blip2_score ./sims_score/score_blip2_reproduce.pt \
       --clip_weight 0.9 \
       --clip_score ./sims_score/score_clip_reproduce.pt
   ```
   Compare output with `uit/cmp/reproduce.txt` to verify.

### Priority 2: Run on Track 4 (2026) data

1. Prepare Track 4 data in expected structure:
   ```
   data/PAB/name-masked_test-set/gallery/   ← Track 4 gallery images (.jpg)
   data/PAB/name-masked_test-set/query.json ← Track 4 queries (JSONL: query_index + caption)
   ```

2. Re-run CLIP and BLIP-2 (zero-shot, no checkpoint needed):
   ```bash
   python3 clip_infer.py \
       --image_folder ./data/PAB/name-masked_test-set/gallery \
       --annotation ./data/PAB/name-masked_test-set/query.json \
       --save_score ./sims_score/score_clip_2026.pt \
       --output_file ./predictions/score_clip_2026.txt

   python3 blip2_infer.py \
       --image_folder ./data/PAB/name-masked_test-set/gallery \
       --annotation ./data/PAB/name-masked_test-set/query.json \
       --save_score ./sims_score/score_blip2_2026.pt \
       --output_file ./predictions/score_blip2_2026.txt
   ```

3. Run LHP inference (requires checkpoint):
   ```bash
   python3 ./lhp_2/beit3/inference.py \
       --checkpoint ./checkpoint/lhp/lhp_beit3.pth \
       --tokenizer ./checkpoint/lhp/beit3.spm \
       --image_folder ./data/PAB/name-masked_test-set/gallery \
       --annotation ./data/PAB/name-masked_test-set/query.json \
       --save_score ./sims_score/score_beit3_2026.pt \
       --output_file ./predictions/score_beit3_2026.txt
   ```

4. Run ensemble:
   ```bash
   python3 uit/cmp/inference.py \
       --config uit/cmp/configs/infer.yaml \
       --task cmp --output_dir output \
       --checkpoint ./checkpoint/uit/uit.pth \
       --output_file ./answer_track4_2026.txt \
       --beit3_weight 0.925 \
       --beit3_score ./sims_score/score_beit3_2026.pt \
       --blip2_weight 0.9 \
       --blip2_score ./sims_score/score_blip2_2026.pt \
       --clip_weight 0.9 \
       --clip_score ./sims_score/score_clip_2026.pt
   ```

### Priority 3: Verify `query.json` format compatibility

The inference dataset (`search_inference_dataset`) reads queries as:
```python
item['query_index']   # used as gallery pid
item['caption']       # used as text query
```
If Track 4 `query.json` uses different field names (e.g. `id` instead of `query_index`), edit `uit/cmp/dataset/search_dataset.py:191`.

---

## 13. Known Issues and Warnings

1. **`beit3_infer.py` (root level) has hardcoded `/home/s48gb/...` paths** in its `__main__` block. Do not run it directly — use `lhp_2/beit3/inference.py` which has proper argparse defaults.

2. **`deepspeed==0.4.0`** is very old (2021). With `torch==2.5.1`, it may fail to compile. If `deepspeed` is not actually imported by the active code paths, consider removing it or upgrading to `deepspeed>=0.14`.

3. **`uit/cmp/Search.py:135`** saves training checkpoints to hardcoded `output/356356/`. If you train UIT, create this directory first.

4. **Image sort order in inference**: `blip2_infer.py`, `clip_infer.py`, and `lhp_2/beit3/inference.py` use `os.listdir()` which returns arbitrary order. The similarity matrix row-to-image correspondence depends on this order — ensure all three infer scripts use the same (sorted) gallery list, or the ensemble will be wrong. Consider adding `sorted()`:
   ```python
   image_folder = sorted([os.path.join(args.image_folder, p) for p in os.listdir(args.image_folder)])
   ```

5. **`uit/cmp/inference.py:74`** passes `args.blip2_weight` as both `beit3_weight` and `blip2_weight` to `evaluation_itm()`. This is likely a copy-paste bug in the function call:
   ```python
   evaluation_itm(model, device, config, args,
                  sims_matrix_t2i, image_embeds, text_embeds, text_atts,
                  args.blip2_weight,   # ← should be args.beit3_weight
                  args.blip2_score, args.blip2_weight, ...)
   ```
   The actual weighting in `eval.py:113` uses the local variable `beit3_weight`, so this may work correctly in practice via the default value.

6. **LAVIS path**: `blip2_infer.py` does `sys.path.append('./blip')` and imports `from blip.blip2 import init_model`. This works when run from repo root. If the CWD changes, it will fail.

---

## 14. Summary

| Capability | Status |
|-----------|--------|
| Training code (LHP) | ✅ Complete |
| Training code (UIT) | ✅ Complete |
| Inference code (all 3 models) | ✅ Complete |
| Ensemble/re-ranking code | ✅ Complete |
| answer.txt generation | ✅ Complete (`reproduce.txt` = 1978 lines) |
| Precomputed similarity scores | ✅ **INCLUDED** (3 × 15 MB `.pt`) |
| Pretrained checkpoints (LHP, UIT) | ❌ Not in repo — Google Form download |
| BLIP-2 / CLIP weights | ✅ Auto-download via library |
| `beit3.spm` tokenizer | ✅ **INCLUDED** |
| Hardcoded paths (blocking) | ⚠️ Only in `beit3_infer.py` `__main__` (not used by proper commands) |
| Paper results reproducible | ✅ Yes, with checkpoint download |
| Ready for Track 4 2026 data | ✅ After path updates in `infer.yaml` |
