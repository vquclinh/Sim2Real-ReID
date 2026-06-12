# AIC26 Author Training Reproduction Plan

> **Status:** Read-only audit. Nothing here has been run.  
> **Scope:** All training and inference code under `lhp/`, `lhp_2/`, `uit/cmp/`, `blip/`,  
> `clip_infer.py`, `blip2_infer.py`, `beit3_infer.py`, `sims_score/`, `predictions/`.  
> **Original code is never modified.** All new artifacts live under `aic26/` only.

---

## 1. Executive Summary

The author pipeline is a three-tower cross-modal retrieval ensemble:

```
CLIP (ViT-L/14@336px)  ──┐
BLIP-2 (LAVIS)          ──┼──► score matrices (.pt) ──► UIT/CMP ITM reranking ──► answer.txt
BEiT-3 (LHP fine-tuned) ──┘
```

The BEiT-3 score matrix **overwrites** the UIT/CMP ITC result before ITM reranking runs.
Effective pipeline order at inference time: CLIP + BLIP-2 + BEiT-3 → UIT ITM.

Three pre-computed score matrices are committed to the repo (`sims_score/`), making a
**zero-training reproduction possible today** for the ensemble reranking step.  
Full reproduction (retraining UIT/CMP from scratch) requires external checkpoints that
are not present in the repo.

**Current zero-shot baseline (PE-Core-G14-448, local eval on PAB `attr.json`):**

| Metric | Value |
|--------|-------|
| mAP    | 0.8829 |
| R@1    | 0.7932 |
| R@5    | 0.9838 |
| R@10   | 0.9944 |

This is the reference point for measuring whether any reproduction step improves or
regresses retrieval quality on the local evaluation split.

---

## 2. Current Local Baseline

| Field | Value |
|-------|-------|
| Run ID | `local_001_pe_g14_attr_zero_shot` |
| Model | PE-Core-G14-448 |
| Training | None (zero-shot) |
| Fine-tuning | None |
| Ensemble | None |
| Eval split | PAB `attr.json` (single caption per row) |
| mAP | 0.8829 |
| R@1 | 0.7932 |
| R@5 | 0.9838 |
| R@10 | 0.9944 |
| Median rank | 1 |
| Mean rank | 1.8 |

Source: `aic26/docs/local_eval/local_001_pe_g14_attr_zero_shot/run_info.md`

This baseline is produced by `aic26/pipelines/local_eval/pab_original/pe_g14_attr_local_eval.ipynb`
and does **not** use any author training code. It is used throughout this document as the
comparison point.

---

## 3. Author Training Code Inventory

### 3.1 Top-level inference scripts

| File | Purpose | Key dependency |
|------|---------|---------------|
| `clip_infer.py` | CLIP ViT-L/14@336px gallery+query embed → `.pt` score matrix | `clip` (auto-download) |
| `blip2_infer.py` | BLIP-2 embed → `.pt` score matrix | `blip/` local, HuggingFace download |
| `beit3_infer.py` | BEiT-3 fine-tuned embed → `.pt` score matrix | `lhp_2/beit3/`, LHP checkpoint |

### 3.2 LHP / BEiT-3 training

| Path | Role |
|------|------|
| `lhp/` | Original BEiT-3 training scaffolding (older version) |
| `lhp_2/` | Author's fine-tuned BEiT-3 code (active version used in inference) |
| `lhp_2/beit3/run_beit3_retrieval.py` | Training entrypoint for BEiT-3 fine-tuning |
| `lhp_2/beit3/datasets.py` | Dataset loader (`BaseDataset`): 75 `pair_*.json` files |
| `lhp_2/beit3/inference.py` | Standalone BEiT-3 inference (separate from `beit3_infer.py`) |
| `lhp_2/beit3/beit3.spm` | Sentencepiece tokenizer — **PRESENT in repo** |

Checkpoint expected at: `checkpoint/lhp/lhp_beit3.pth` — **NOT present**.

### 3.3 UIT / CMP training and inference

| Path | Role |
|------|------|
| `uit/cmp/Search.py` | UIT/CMP training entrypoint |
| `uit/cmp/inference.py` | Ensemble reranking entrypoint (production) |
| `uit/cmp/dataset/search_dataset.py` | Three dataset classes (train / test / inference) |
| `uit/cmp/configs/cmp.yaml` | Full training config (75 pair files, 30 epochs) |
| `uit/cmp/configs/baseline.yaml` | Reduced training config (8 pair files) |
| `uit/cmp/configs/infer.yaml` | Inference config (hidden test set) |
| `uit/cmp/output/cmp/config.yaml` | Saved config from a past training run (read-only artifact) |

Checkpoint expected at: `uit/cmp/checkpoint/pretrained.pth` — **NOT present**.  
Training writes to: `output/356356/checkpoint_{epoch}.pth` (hard-coded in `Search.py:135`).

### 3.4 BLIP-2

| Path | Role |
|------|------|
| `blip/` | BLIP-2 model code (local, non-HuggingFace) |
| `blip/blip2.py` | `init_model()` entry used by `blip2_infer.py` |

BLIP-2 weights auto-download from HuggingFace at first use. No local checkpoint committed.

### 3.5 Pre-computed score matrices (committed)

| File | Content |
|------|---------|
| `sims_score/score_beit3_reproduce.pt` | BEiT-3 text→image similarity matrix |
| `sims_score/score_blip2_reproduce.pt` | BLIP-2 text→image similarity matrix |
| `sims_score/score_clip_reproduce.pt` | CLIP text→image similarity matrix |
| `predictions/score_beit3_reproduce.txt` | BEiT-3 top-10 predictions (text) |
| `predictions/score_blip2_reproduce.txt` | BLIP-2 top-10 predictions (text) |
| `predictions/score_clip_reproduce.txt` | CLIP top-10 predictions (text) |

These matrices were computed against the **hidden test set** (`name-masked_test-set/`),
not the local `attr.json` split. They cannot be directly used for local metric computation.

---

## 4. Pipeline-by-Pipeline Findings

### 4.1 LHP / BEiT-3

**Architecture:** `beit3_large_patch16_384_retrieval` (746M parameters, XLM-Roberta
tokenizer). Input: 384×384 images. Task identifier: `356`.

**Training data:** 75 files `annotation/train/pair_{0..74}.json` via `BaseDataset`.  
Each record has: `image` (relative path), `caption` (single string), `image_id` (overwritten
with a sequential counter — the JSON value is ignored at training time).

**Key finding — `beit3_infer.py` import path bug:**
```python
# beit3_infer.py (top-level, used in pipeline)
from lhp.beit3 import ...   # BUG: should be lhp_2.beit3
```
`lhp/` is an older scaffold; the active model code is in `lhp_2/`. This import fails at
runtime. The standalone `lhp_2/beit3/inference.py` uses correct relative imports.

**Key finding — hard-coded researcher path in `beit3_infer.py`:**
```python
annotation_path = '/home/s48gb/Desktop/GenAI4E/pab/data/PAB/name-masked_test-set/gallery/query.json'
```
This path must be overridden by the `--annotation` argument. It is a developer left-over
and does not affect the inference path when the argument is provided.

**Tokenizer:** `beit3.spm` is present at `lhp_2/beit3/beit3.spm`. The standalone
`lhp_2/beit3/inference.py` accepts `--tokenizer` pointing to this file.

**Blocker:** `checkpoint/lhp/lhp_beit3.pth` is absent. BEiT-3 inference cannot run
until this checkpoint is obtained from the original team.

---

### 4.2 UIT / CMP

**Architecture:** SwinB image encoder + `bert-base-uncased` text encoder.  
Four training objectives: ITC + ITM + MLM + MIM. EDA augmentation on captions.

**Full training config (`cmp.yaml`):**

| Parameter | Value |
|-----------|-------|
| Pair files | 75 (`pair_0.json` … `pair_74.json`) |
| Batch size | 84 |
| Epochs | 30 |
| k_test | 128 |
| Image root | `../../data/PAB/` |
| Text encoder | `checkpoint/bert-base-uncased` |

**Baseline config (`baseline.yaml`):** Only 8 pair files (`pair_0` … `pair_7`). Suitable
for smoke-testing the training loop without the full dataset.

**Key finding — hard-coded checkpoint output path:**
```python
# Search.py:135
torch.save(save_obj, os.path.join('output/356356', f'checkpoint_{epoch+1}.pth'))
```
The output directory `output/356356` is not configurable. Training must be run from a
working directory where this relative path is writable.

**Key finding — `bert-base-uncased` checkpoint path:**
The text encoder is loaded from `checkpoint/bert-base-uncased` (relative path).
This directory is absent from the repo. `bert-base-uncased` must be downloaded
separately (HuggingFace or manual copy) before training can start.

**Key finding — pretrained SwinB + BEiT-3 init:**
`cmp.yaml` references `load_pretrained: True`. The pretrained checkpoint path is
`uit/cmp/checkpoint/pretrained.pth`, which is absent. Without it, the model trains
from random init — results will likely be much worse.

**Key finding — non-deterministic `os.listdir()` in `search_inference_dataset`:**
```python
# dataset/search_dataset.py — search_inference_dataset.__init__
gallery = [f for f in os.listdir(image_dir)]
```
Gallery ordering is filesystem-dependent. The score matrix produced by `clip_infer.py`
and `blip2_infer.py` uses the same `os.listdir()` ordering, so the columns of the
three score matrices are consistent *within a single run* on the same machine — but
not reproducible across machines or filesystems.

**Inference flow (`uit/cmp/inference.py`):**
1. Load `search_inference_dataset` (hidden test: `query.json` + `os.listdir()` gallery).
2. Run UIT/CMP `evaluation_itc()` → get image/text embeddings and ITC score matrix.
3. **Overwrite ITC with BEiT-3:** `sims_matrix_t2i = torch.load(args.beit3_score)`.
4. Run `evaluation_itm()` fusing BEiT-3 + BLIP-2 + CLIP matrices with weights:
   `beit3=0.925`, `blip2=0.9`, `clip=0.9`.
5. Write top-10 answer.txt using `test_dataset.g_pids` (gallery IDs from
   `filename.split('.')[0]` — strips extension).

**Implication:** The UIT/CMP ITC result is never used in the final ensemble. Training
UIT/CMP is needed only for the ITM reranking head (`evaluation_itm()`), not for the
initial retrieval scoring.

---

### 4.3 CLIP / PE / BLIP-2

**CLIP (`clip_infer.py`):**
- Model: `ViT-L/14@336px`, auto-downloaded by the `clip` library.
- Gallery IDs: `filename[:-4]` (strips 4 characters = `.ext`).
- Non-deterministic `os.listdir()` bug (same as UIT inference).
- No local checkpoint needed — CLIP weights download automatically.

**PE-Core-G14-448 (PE-G14, zero-shot):**
- Used in `aic26/pipelines/official_submission/` and `aic26/pipelines/local_eval/`.
- Not part of the author ensemble. Serves as the zero-shot baseline.
- No training required.

**BLIP-2 (`blip2_infer.py`):**
- Uses `blip/blip2.py::init_model()` which downloads weights from HuggingFace.
- Same `os.listdir()` bug.
- `sys.path.append('./blip')` must be run from the repo root.

---

### 4.4 Ensemble and Reranking

**Committed matrices (`sims_score/*.pt`) are the only runnable artifact today.**  
They feed directly into `uit/cmp/inference.py` via `--beit3_score`, `--blip2_score`,
`--clip_score` arguments.

**Blocker for running `inference.py`:** Requires a trained UIT/CMP model checkpoint
at `uit/cmp/checkpoint/pretrained.pth` (or `output/356356/checkpoint_30.pth`
from a completed training run) to load the ITM head.

If only the score matrices are available (no UIT checkpoint), a simplified reranking
script could fuse the three matrices directly without the ITM head. This is not
currently in the codebase but would be straightforward to add under `aic26/`.

---

## 5. Required External Assets

The following assets are **absent from the repo** and must be obtained before each
pipeline stage can run:

| Asset | Path | Size (est.) | Stage |
|-------|------|-------------|-------|
| BEiT-3 fine-tuned checkpoint | `checkpoint/lhp/lhp_beit3.pth` | ~3 GB | BEiT-3 inference |
| UIT/CMP pretrained init | `uit/cmp/checkpoint/pretrained.pth` | ~300 MB | UIT training |
| `bert-base-uncased` weights | `checkpoint/bert-base-uncased/` | ~440 MB | UIT training + inference |
| CLIP ViT-L/14@336px | auto-download (`~/.cache/clip/`) | ~890 MB | CLIP inference |
| BLIP-2 weights | auto-download (HuggingFace cache) | ~5 GB | BLIP-2 inference |
| PAB full dataset | `../../data/PAB/` (relative to `uit/cmp/`) | ~10 GB (est.) | Any training |
| Hidden test set | `data/PAB/name-masked_test-set/` | unknown | Official inference |

Assets that **are** present:
- `lhp_2/beit3/beit3.spm` (sentencepiece tokenizer for BEiT-3) — PRESENT
- `sims_score/score_{beit3,blip2,clip}_reproduce.pt` — PRESENT (hidden test set only)
- `predictions/score_{beit3,blip2,clip}_reproduce.txt` — PRESENT

---

## 6. Dataset Format Requirements

### PAB annotation formats

The codebase uses three distinct annotation formats. These must not be mixed.

**Training format (`pair_*.json`, JSONL):**
```json
{"image": "images/person_001.jpg", "caption": "A person in a red jacket.", "image_id": "001"}
```
- `caption`: single string (not a list)
- `image_id`: present in JSON but **overwritten** by BEiT-3 `BaseDataset` with a sequential counter
- `image`: relative path from the dataset root

**Local test format (`annotation/test/pair.json`, JSON array):**
```json
[{"image_id": "001", "caption": ["A person in a red jacket.", "Wearing red outerwear."]}]
```
- `caption`: **must be a list** (read by `search_test_dataset`)
- Used by `uit/cmp/` evaluation when `cmp.yaml` or `baseline.yaml` is active

**Hidden inference format (`name-masked_test-set/query.json`, JSON array):**
```json
[{"query_index": "0", "caption": "A person in a red jacket."}]
```
- `caption`: single string
- `query_index`: string key used for g_pids/q_pids
- Used by `search_inference_dataset` and all top-level infer scripts

**Local eval format (`annotation/test/attr.json`, JSONL):**
```json
{"image_id": "img_001.jpg", "caption": "A person in a red jacket."}
```
- `caption`: single string
- Used by `aic26/pipelines/local_eval/pab_original/pe_g14_attr_local_eval.ipynb`
- This is the format compatible with the PE-G14 local evaluation pipeline

---

## 7. Compatibility With Current PAB Data

| Component | Requires | Local PAB has | Compatible? |
|-----------|----------|---------------|-------------|
| UIT/CMP training | `annotation/train/pair_{0..74}.json` | Unknown (data not in repo) | Cannot verify without data |
| UIT/CMP local eval | `annotation/test/pair.json` (caption as LIST) | Unknown | Cannot verify without data |
| BEiT-3 training | `annotation/train/pair_{0..74}.json` | Unknown | Cannot verify without data |
| BEiT-3 inference (standalone) | `name-masked_test-set/query.json` | Unknown | Cannot verify without data |
| CLIP/BLIP-2 inference | `name-masked_test-set/query.json` + gallery dir | Unknown | Cannot verify without data |
| PE-G14 local eval | `annotation/test/attr.json` | Confirmed present (run completed) | **YES** |
| Ensemble reranking | `sims_score/*.pt` + UIT checkpoint | Score matrices present, no checkpoint | Partial |

The only confirmed working pipeline end-to-end is the PE-G14 local eval against `attr.json`.

---

## 8. Data Leakage Risks

### 8.1 Sequential `image_id` overwrite in BEiT-3

`lhp_2/beit3/datasets.py::BaseDataset` overwrites `item["image_id"]` with `ann_id`
(a sequential counter over the 75 training files). If the test set happens to contain
images with the same sequential IDs, these IDs would collide silently. This is a
correctness issue rather than a leakage risk.

### 8.2 `os.listdir()` ordering in inference scripts

The non-deterministic gallery ordering in `clip_infer.py`, `blip2_infer.py`, and
`search_inference_dataset` means that the same query against the same gallery can
produce different score matrix column orderings on different machines. The committed
`sims_score/*.pt` matrices reflect a specific ordering captured on the author's
machine. Running inference on a different machine will produce a matrix whose columns
do not align — silently corrupting the ensemble without any error.

**Mitigation:** Before running any inference script, add explicit `sorted()` around
the `os.listdir()` call. Do not modify original files; instead, create patched
inference wrappers under `aic26/`.

### 8.3 Implicit train/test overlap risk

The PAB dataset has 75 training annotation files and a test split. If the training
images include any images from the test gallery, the model has seen positive-image
pairs during training. The codebase contains no explicit check for this. This should
be verified before reporting training-based results as a fair comparison.

### 8.4 Hard-coded researcher path in `beit3_infer.py`

```python
annotation_path = '/home/s48gb/Desktop/...'
```
This path is used as a fallback only if `--annotation` is not provided. Do not run
`beit3_infer.py` without explicitly supplying `--annotation`.

---

## 9. Recommended Reproduction Roadmap

Stages are ordered by risk and dependency. Each stage is a prerequisite for the next.
**Do not skip stages.** Do not train until Stage 3 is confirmed passing.

### Stage 0 — Environment and Asset Verification (no GPU required)

- [ ] Confirm `timm==0.4.12` environment for LHP/BEiT-3
- [ ] Confirm `timm==0.6.13` environment for UIT/CMP
- [ ] Verify `lhp_2/beit3/beit3.spm` loads with `sentencepiece`
- [ ] Confirm all 75 `annotation/train/pair_*.json` files are present and valid JSONL
- [ ] Confirm `annotation/test/pair.json` has `caption` as a list (not string)
- [ ] Confirm `annotation/test/attr.json` is present and valid
- [ ] Verify score matrices load: `torch.load('sims_score/score_beit3_reproduce.pt')`
- [ ] Verify matrix shapes match gallery × query dimensions for the hidden test set

**Blocker exit:** All checks pass. No GPU used. No code modified.

---

### Stage 1 — CLIP Inference (low-risk, auto-download)

- [ ] Obtain PAB test gallery directory (hidden test set)
- [ ] Obtain `data/PAB/name-masked_test-set/query.json`
- [ ] Create `aic26/inference/clip_infer_patched.py`:
  - Wrap `clip_infer.py` logic
  - Replace `os.listdir()` with `sorted(os.listdir())` for deterministic gallery order
  - Do not modify `clip_infer.py`
- [ ] Run CLIP inference; compare output `.pt` shape and row/column count against
  `sims_score/score_clip_reproduce.pt`
- [ ] If shapes match, compute cosine similarity between columns to check gallery ordering

**Expected blocker:** None (CLIP auto-downloads, no private checkpoint).

---

### Stage 2 — BLIP-2 Inference (medium-risk, HuggingFace download)

- [ ] Confirm HuggingFace access and BLIP-2 model download (~5 GB)
- [ ] Create `aic26/inference/blip2_infer_patched.py` with sorted gallery
- [ ] Run BLIP-2 inference; compare against `sims_score/score_blip2_reproduce.pt`

**Expected blocker:** HuggingFace download, VRAM (BLIP-2 is memory-intensive).

---

### Stage 3 — BEiT-3 Inference (high-risk, private checkpoint required)

- [ ] Obtain `checkpoint/lhp/lhp_beit3.pth` from original team
- [ ] Fix `beit3_infer.py` import: `from lhp_2.beit3 import ...` (or create patched
  wrapper under `aic26/inference/`)
- [ ] Confirm `beit3.spm` loads correctly
- [ ] Run `lhp_2/beit3/inference.py` (standalone, no import bug) with correct paths
- [ ] Compare output against `sims_score/score_beit3_reproduce.pt`

**Expected blocker:** Private checkpoint unavailable without contacting original team.

---

### Stage 4 — UIT/CMP Training (highest risk, all dependencies required)

- [ ] Download `bert-base-uncased` to `checkpoint/bert-base-uncased/`
- [ ] Obtain `uit/cmp/checkpoint/pretrained.pth` (SwinB init) from original team
- [ ] Create `output/356356/` directory (hard-coded output path)
- [ ] Run `baseline.yaml` smoke test first (8 pair files, reduced epoch count)
- [ ] Confirm training completes and writes `output/356356/checkpoint_1.pth`
- [ ] Verify eval metrics on `annotation/test/pair.json` match the expected format
- [ ] Full training run with `cmp.yaml` (75 files, 30 epochs)

**Expected blockers:** `pretrained.pth` absent; `bert-base-uncased` not downloaded;
`output/356356` path not pre-created; high VRAM requirement for batch size 84.

---

### Stage 5 — Full Ensemble (requires Stages 1–4 complete)

- [ ] Confirm all three new score matrices have matching gallery ordering
  (use `sorted()` fix from patched wrappers)
- [ ] Create `aic26/inference/run_ensemble.py` wrapping `uit/cmp/inference.py` logic
  with path arguments pointing to new matrices and Stage 4 checkpoint
- [ ] Run ensemble; verify `answer.txt` format matches `aic26/docs/submissions/`
- [ ] Submit to leaderboard only after local validation confirms no regression
  vs. PE-G14 baseline (mAP ≥ 0.8829)

---

## 10. First Safe Implementation Target

Given the current asset inventory, the **safest non-GPU-training step** to implement
is a score-matrix fusion script that uses the **committed score matrices** without
the UIT/CMP ITM head:

```
sims_score/score_beit3_reproduce.pt   (weight 0.925)
sims_score/score_blip2_reproduce.pt   (weight 0.900)
sims_score/score_clip_reproduce.pt    (weight 0.900)
        └──► weighted sum ──► top-10 ──► answer.txt
```

This requires:
- No training
- No inference
- No checkpoint downloads
- Only the hidden test gallery order (to reconstruct `g_pids`)

If this replicates the committed `predictions/score_*_reproduce.txt`, it confirms that
the gallery ordering in the committed matrices is recoverable and that a Stage 5
replication is feasible without re-running Stages 1–3.

Proposed path: `aic26/inference/fuse_score_matrices.py`

---

## 11. Open Questions Before Training

1. **Gallery ordering in committed `.pt` files:** What is the exact gallery ID order
   embedded in `sims_score/*.pt`? The `.pt` files contain raw tensors with no column
   labels. The ordering was produced by a specific `os.listdir()` call on the author's
   machine. Without the corresponding ID list, the matrices cannot be correctly
   mapped to gallery images. Is there a companion `.json` or `.txt` file that records
   this ordering?

2. **`checkpoint/lhp/lhp_beit3.pth` availability:** Is this checkpoint stored in a
   shared drive or cloud bucket? Contact: original team (s48gb author per path in
   `beit3_infer.py`). Without this, Stage 3 is blocked.

3. **`uit/cmp/checkpoint/pretrained.pth` content:** Is this the SwinB ImageNet
   pretrained weight, or a BEiT-3 cross-initialized weight? The `cmp.yaml` field
   `load_params_vision: False` (in `infer.yaml`) suggests the vision branch may not
   be loaded from it at inference time. Clarify what this checkpoint actually
   initialises.

4. **Data leakage — train/test overlap:** Were the 75 training pair files constructed
   to exclude all images in the test gallery? If `attr.json` and `pair.json` test images
   appear in training pairs, the training mAP is not a fair generalization measurement.

5. **`infer.yaml` load_pretrained vs. load_params_vision:** `load_pretrained: True`
   combined with `load_params_vision: False` is unusual. Does `inference.py` actually
   load any weights at all, or does it skip the vision init and rely entirely on the
   overwritten BEiT-3 score matrix?

6. **`search_test_dataset` caption format:** `cmp.yaml` points to `pair.json` where
   `caption` is a list. `attr.json` has a single string. Are there any code paths in
   `uit/cmp/` that support single-string captions during local evaluation, or would
   running `Search.py` with `attr.json` crash?

7. **Timm conflict resolution:** The LHP environment needs `timm==0.4.12` and the UIT
   environment needs `timm==0.6.13`. Are conda environments or Docker images available
   from the original team, or must they be reconstructed from requirements?

---

*Generated by static code audit — no training was run, no models were loaded,
no datasets were read, no code was modified outside `aic26/`.*
