# FORK_REPO_AUDIT.md

**Repository:** `Sim2Real-ReID` (forked from HUI: Hybrid, Unified and Iterative)  
**Paper:** "Hybrid, Unified and Iterative: A Novel Framework for Text-based Person Anomaly Retrieval" (ACM Web Conf Workshop MORE 2025)  
**Target competition:** AIC/ECCV 2026 Track 4 — Text-Based Person Anomaly Search (PAB dataset)  
**Audit date:** 2026-06-10  
**Purpose:** Establish a clean baseline repo before merging teammate contributions

---

## 1. Repo Overview

This fork contains the full codebase for the HUI (Hybrid, Unified, Iterative) framework:

| Sub-system | Role |
|---|---|
| **LHP** (Local-Global Hybrid Perspective) | BEiT-3 Large fine-tuned on PAB for retrieval |
| **UIT / CMP** (Unified Image-Text) | Swin-B + BERT with ITC + ITM + MLM + MIM |
| **Iterative Ensemble** | Weighted combination of BEiT-3 ITM, BEiT-3 ITC, BLIP-2, and CLIP scores |

Precomputed similarity score tensors (`.pt`) for the **1978-image real-world test set** are already present. The ensemble can be reproduced from these files without rerunning any model.

The README at the repository root says only `"Comming soon!"` — all actual documentation is in subdirectory READMEs.

---

## 2. Current Folder Structure

```
Sim2Real-ReID/
│
├── README.md                        # Empty ("Comming soon!")
├── HUI_REPO_AUDIT.md                # Previous partial audit (incomplete)
├── FORK_REPO_AUDIT.md               # THIS FILE
├── requirements_lhp.txt             # LHP/BEiT-3 Python environment
├── requirements_uit.txt             # UIT/CMP Python environment
│
├── app.py                           # Streamlit demo (BEiT-3 + FAISS image retrieval)
├── beit3_infer.py                   # LEGACY root-level BEiT-3 helper (BROKEN: imports from empty lhp/)
├── blip2_infer.py                   # BLIP-2 inference entry point
├── clip_infer.py                    # CLIP ViT-L/14@336px inference entry point
│
├── blip/                            # Vendored LAVIS library + BLIP-2 wrapper
│   ├── blip2.py                     # init_model() — loads BLIP-2 feature extractor ViT-L
│   ├── lavis/                       # Full LAVIS library (Salesforce vendored copy)
│   ├── train.py, evaluate.py        # LAVIS generic scripts (NOT used by HUI)
│   └── app/, examples/, docs/       # LAVIS demo/doc files (NOT used by HUI)
│
├── lhp/                             # EMPTY DIRECTORY — lhp v1 placeholder (no files)
│
├── lhp_2/                           # Clone of Microsoft unilm repo
│   └── beit3/                       # LHP: ACTUAL BEiT-3 fine-tuning + inference
│       ├── run_beit3_finetuning.py  # TRAINING entry point (--task 356 = PAB task)
│       ├── inference.py             # INFERENCE entry point for LHP
│       ├── modeling_finetune.py     # BEiT-3 retrieval model definition
│       ├── modeling_utils.py        # Magneto architecture core
│       ├── datasets.py              # PAB dataset loader
│       ├── engine_for_finetuning.py # Train/eval loops (handles task '356')
│       ├── beit3.spm                # Sentencepiece tokenizer (PRESENT IN REPO)
│       ├── lhp_reproduce/log/       # TensorBoard tfevents from authors' training run
│       └── utils.py, optim_factory.py, randaug.py, glossary.py, requirements.txt
│   │
│   ├── beit/, beit2/                # BEiT v1/v2 image-only models (NOT used)
│   ├── adalm, beats, bitnet, ...    # >20 unrelated Microsoft research projects
│   └── README.md, LICENSE, ...      # Microsoft unilm repo metadata
│
├── uit/
│   ├── README.md                    # UIT/CMP README
│   ├── test.py                      # (uncertain — not inspected)
│   └── cmp/                         # UIT/CMP main code
│       ├── Search.py                # TRAINING entry point (DDP launcher)
│       ├── inference.py             # INFERENCE + ENSEMBLE entry point
│       ├── run.py                   # Python wrapper to launch distributed Search.py
│       ├── train.py                 # train_model() function
│       ├── eval.py                  # evaluation_itc(), evaluation_itm(), mAP()
│       ├── reproduce.txt            # Precomputed ensemble output (1978 lines)
│       ├── score_beit3.pt           # Copy of beit3 score tensor (15MB)
│       ├── score_blip2.pt           # Copy of blip2 score tensor (15MB)
│       ├── score_clip.pt            # Copy of clip score tensor (15MB)
│       ├── configs/
│       │   ├── cmp.yaml             # Config for CMP training (relative paths ../../data/PAB)
│       │   ├── infer.yaml           # Config for inference (relative paths ./data/PAB)
│       │   ├── baseline.yaml        # Baseline config
│       │   ├── config_bert.json, config_bertL.json
│       │   ├── config_swinB.json, config_swinL.json
│       │   └── config_simmim.yaml, config_swinBmim.yaml
│       ├── dataset/
│       │   ├── search_dataset.py    # Dataset classes: train / test / inference
│       │   ├── utils.py             # pre_caption(), read_json_to_list()
│       │   └── eda.py, randaugment.py, random_erasing.py
│       ├── models/
│       │   ├── model_search.py      # Search model (ITC+ITM+MLM+MIM)
│       │   ├── bert.py              # BERT text encoder
│       │   ├── swin_transformer.py  # Swin-B vision encoder
│       │   ├── simmim.py            # MIM auxiliary head
│       │   └── cmp.py, config.py
│       └── requirements.txt
│
├── sims_score/
│   ├── score_beit3_reproduce.pt     # BEiT-3 score matrix [1978×1978 float32] (15MB)
│   ├── score_blip2_reproduce.pt     # BLIP-2 score matrix [1978×1978 float32] (15MB)
│   └── score_clip_reproduce.pt      # CLIP score matrix [1978×1978 float32] (15MB)
│
├── predictions/
│   ├── score_beit3_reproduce.txt    # BEiT-3 top-10 predictions (1978 lines)
│   ├── score_blip2_reproduce.txt    # BLIP-2 top-10 predictions (1978 lines)
│   └── score_clip_reproduce.txt     # CLIP top-10 predictions (1978 lines)
│
└── document/
    └── AIO_paper.pdf                # Reference paper PDF
```

---

## 3. Important Files Table

| File | Role | Status |
|---|---|---|
| `lhp_2/beit3/inference.py` | **LHP inference** — produces `score_beit3_reproduce.pt` + `predictions/score_beit3.txt` | Active, usable with args |
| `lhp_2/beit3/run_beit3_finetuning.py` | **LHP training** — fine-tunes BEiT-3 Large on PAB | Active, needs large checkpoint |
| `blip2_infer.py` | **BLIP-2 inference** — produces `score_blip2_reproduce.pt` + predictions | Active, needs LAVIS auto-download |
| `clip_infer.py` | **CLIP inference** — produces `score_clip_reproduce.pt` + predictions | Active, needs CLIP auto-download |
| `uit/cmp/inference.py` | **UIT inference + final ensemble** — combines all 3 scores | Active, needs UIT checkpoint |
| `uit/cmp/Search.py` | **UIT training** | Active, needs pretrained checkpoint |
| `uit/cmp/run.py` | Python launcher for distributed training | Active |
| `uit/cmp/reproduce.txt` | Final ensemble prediction output (1978 queries) | Reference output, not code |
| `sims_score/score_*.pt` | Precomputed score tensors for reproduction | Data files (15MB each) |
| `predictions/score_*.txt` | Per-model top-10 text predictions | Data files |
| `beit3_infer.py` | Legacy helper — imports from `lhp.beit3` which is empty | **BROKEN** |
| `app.py` | Streamlit demo app — image/text query against FAISS index | Demo only |
| `lhp_2/beit3/beit3.spm` | Sentencepiece tokenizer for BEiT-3 | Present in repo |
| `lhp_2/beit3/lhp_reproduce/log/` | TensorBoard logs from authors' run | Reference only |
| `document/AIO_paper.pdf` | HUI paper PDF | Reference only |

### Files irrelevant to AIC26 Track 4

| File / Folder | Reason |
|---|---|
| `lhp/` (empty directory) | Legacy placeholder, remove or ignore |
| `lhp_2/beit/`, `lhp_2/beit2/` | BEiT v1/v2 image-only models, not used |
| `lhp_2/adalm/`, `lhp_2/beats/`, `lhp_2/bitnet/` | Unrelated Microsoft research projects |
| `lhp_2/deltalm/`, `lhp_2/dit/`, `lhp_2/e5/`, `lhp_2/edgelm/` | Unrelated |
| `lhp_2/glan/`, `lhp_2/infoxlm/`, `lhp_2/kosmos-1/`, `lhp_2/kosmos-2/` | Unrelated |
| `lhp_2/kosmos-2.5/`, `lhp_2/LatentLM/`, `lhp_2/layoutlm*/` | Unrelated |
| `lhp_2/longnet/`, `lhp_2/longvit/`, `lhp_2/markuplm/`, `lhp_2/mathscale/` | Unrelated |
| `lhp_2/metalm/`, `lhp_2/minilm/`, `lhp_2/retnet/`, `lhp_2/s2s-ft/` | Unrelated |
| `lhp_2/simlm/`, `lhp_2/speechlm/`, `lhp_2/speecht5/`, `lhp_2/textdiffuser*` | Unrelated |
| `lhp_2/trocr/`, `lhp_2/unilm/`, `lhp_2/unilm-v1/`, `lhp_2/Diff-Transformer/` | Unrelated |
| `lhp_2/decoding/`, `lhp_2/deepnet/`, `lhp_2/vlmo/`, `lhp_2/vl-beit/` | Unrelated |
| `lhp_2/wavlm/`, `lhp_2/YOCO/`, `lhp_2/xdoc/`, `lhp_2/xlmt/`, `lhp_2/xmoe/` | Unrelated |
| `lhp_2/xtune/`, `lhp_2/unimim/`, `lhp_2/layoutreader/` | Unrelated |
| `lhp_2/storage/` | BERT config/vocab files for unilm, not used by BEiT-3 |
| `blip/app/`, `blip/examples/`, `blip/docs/`, `blip/run_scripts/` | LAVIS demo/doc, not used |
| `blip/train.py`, `blip/evaluate.py` | LAVIS generic entry points, not used by HUI |
| `beit3_infer.py` (root) | Broken legacy file (imports empty module) |
| `uit/cmp/out/` | Left-over intermediate output folder |

---

## 4. Train / Inference / Ensemble Entry Points

### 4.1 LHP Training — `lhp_2/beit3/run_beit3_finetuning.py`

| Property | Value |
|---|---|
| Purpose | Fine-tune BEiT-3 Large on PAB for image-text retrieval |
| Action | Training |
| Expected inputs | PAB data at `<data_path>`, BEiT-3 pretrained checkpoint, beit3.spm |
| Expected outputs | Checkpoint `.pth` saved to `--output_dir` |
| Requires checkpoint | Yes — BEiT-3 Large pretrained (NOT included, ~2.4GB) |
| Key args | `--task 356`, `--sentencepiece_model`, `--finetune`, `--data_path` |
| AIC26 ready | Needs adaptation: task name `356`, dataset hardcoded to 75 `pair_N.json` files |

Training is launched as distributed, e.g.:
```bash
python -m torch.distributed.launch --nproc_per_node=4 run_beit3_finetuning.py \
    --task 356 --model beit3_large_patch16_384_retrieval \
    --sentencepiece_model ./beit3.spm \
    --finetune /path/to/beit3_large_patch16_384_coco_retrieval.pth \
    --data_path /data/PAB --output_dir ./output/lhp
```

### 4.2 LHP Inference — `lhp_2/beit3/inference.py`

| Property | Value |
|---|---|
| Purpose | Extract BEiT-3 embeddings, compute text→image similarity matrix |
| Action | Inference |
| Expected inputs | Gallery folder, `query.json`, trained checkpoint, `beit3.spm` tokenizer |
| Expected outputs | `score_beit3.pt` (similarity tensor), `predictions/score_beit3.txt` (top-10 per query) |
| Requires checkpoint | Yes — `checkpoint/lhp/lhp_beit3.pth` (NOT included) |
| CWD requirement | Must be run from repo root (paths like `./checkpoint/lhp/`, `./sims_score/`, `./data/PAB/`) |
| AIC26 ready | Gallery ordering bug (see Section 6). Needs AIC26 gallery path and query.json |

Run as:
```bash
cd Sim2Real-ReID
python lhp_2/beit3/inference.py \
    --checkpoint ./checkpoint/lhp/lhp_beit3.pth \
    --tokenizer ./lhp_2/beit3/beit3.spm \
    --image_folder ./data/PAB/name-masked_test-set/gallery \
    --annotation ./data/PAB/name-masked_test-set/query.json \
    --save_score ./sims_score/score_beit3_reproduce.pt \
    --output_file ./predictions/score_beit3.txt
```

### 4.3 BLIP-2 Inference — `blip2_infer.py`

| Property | Value |
|---|---|
| Purpose | Extract BLIP-2 embeddings, compute text→image similarity matrix |
| Action | Inference |
| Expected inputs | Gallery folder, `query.json` (JSONL with `caption` field) |
| Expected outputs | `sims_score/score_blip2_reproduce.pt`, `predictions/score_blip2.txt` |
| Requires checkpoint | Auto-download via LAVIS (`blip2_feature_extractor`, `pretrain_vitL`) |
| Gallery order | `os.listdir()` WITHOUT `sorted()` — CRITICAL RISK |
| AIC26 ready | Gallery ordering fix required; query.json `caption` field must exist |

### 4.4 CLIP Inference — `clip_infer.py`

| Property | Value |
|---|---|
| Purpose | Extract CLIP ViT-L/14@336px embeddings, compute similarity |
| Action | Inference |
| Expected inputs | Gallery folder, `query.json` (JSONL with `caption` field) |
| Expected outputs | `sims_score/score_clip_reproduce.pt`, `predictions/score_clip.txt` |
| Requires checkpoint | Auto-download via `clip.load("ViT-L/14@336px")` |
| Gallery order | `os.listdir()` WITHOUT `sorted()` — CRITICAL RISK |
| AIC26 ready | Gallery ordering fix required |

### 4.5 UIT Training — `uit/cmp/Search.py` (launched via `run.py`)

| Property | Value |
|---|---|
| Purpose | Train CMP (Swin-B + BERT + ITC + ITM + MLM + MIM) on PAB |
| Action | Training |
| Expected inputs | PAB annotation JSON files, pretrained checkpoint (Swin-B + BERT or `pretrained.pth`) |
| Config | `uit/cmp/configs/cmp.yaml` or `baseline.yaml` |
| Expected outputs | Checkpoint saved to `output/356356/checkpoint_N.pth` (hardcoded subfolder name) |
| Requires checkpoint | Yes — `checkpoint/pretrained.pth` or separate Swin-B + BERT-base |
| AIC26 ready | Needs dataset path update; 75 training JSON files hardcoded in config |

### 4.6 UIT Inference + Final Ensemble — `uit/cmp/inference.py`

| Property | Value |
|---|---|
| Purpose | Run UIT ITC + ITM, load external BEiT-3/BLIP-2/CLIP scores, compute weighted ensemble, output final answer |
| Action | Inference + Ensemble |
| Expected inputs | Gallery + query.json, UIT checkpoint, beit3_score.pt, blip2_score.pt, clip_score.pt |
| Config | `uit/cmp/configs/infer.yaml` |
| Expected outputs | Final ranking text file (one line per query, space-separated image stems) |
| Requires checkpoint | Yes — UIT/CMP trained checkpoint (NOT included) |
| Gallery order | Reads gallery via `os.listdir()` in `search_inference_dataset` — CRITICAL RISK |
| AIC26 ready | Gallery ordering fix required; query.json must have `caption` and `query_index` fields |

Default ensemble weights (from `inference.py` args):
- `beit3_weight = 0.925` (ITC vs ITM blend within BEiT-3)
- `blip2_weight = 0.9`
- `clip_weight = 0.9`

Final formula (from `eval.py:113`):
```
final = ((1-beit3_w)*ITM + beit3_w*ITC) * blip2_w + (1-blip2_w)*blip2) * clip_w + (1-clip_w)*clip
```

---

## 5. Dataset Expected Structure

### 5.1 Data root

Scripts assume `data/PAB/` relative to CWD (run from repo root) or `../../data/PAB/` when run from `uit/cmp/`. No absolute path is enforced — the paths are passed as CLI args or set in YAML configs.

### 5.2 Expected training annotation structure (PAB)

```
data/PAB/
├── annotation/
│   ├── train/
│   │   ├── pair_0.json     # JSONL files
│   │   ├── pair_1.json
│   │   └── ... pair_74.json   (75 files total)
│   ├── test/
│   │   └── pair.json           # JSONL test annotations
│   └── source_caption.json
├── train/
│   ├── imgs_0/
│   │   ├── goal/
│   │   │   └── 0.jpg, 1.jpg, ...
│   │   ├── full/
│   │   └── wentrong/
│   ├── imgs_1/
│   └── ...
└── test/
    └── 0.jpg, 1.jpg, ...
```

### 5.3 Expected test/inference structure

```
data/PAB/name-masked_test-set/
├── gallery/
│   └── IMAGEID.jpg    (1978 images, filenames are person IDs)
└── query.json         (JSONL file, 1978 lines for inference)
```

### 5.4 Training annotation JSON format (each line)

```json
{
  "image": "train/imgs_0/goal/0.jpg",
  "caption": "The image shows a band performing...",
  "image_id": "0_0",
  "hard_i": "imgs_0/full/0.jpg",
  "hard_c": "...",
  "hard_i_id": "0_8954",
  "source_id": "1_0",
  "source_caption": "...",
  "normal": "Performing",
  "scene": "outdoor concert"
}
```

### 5.5 Test annotation JSON format (each line, for evaluation with ground truth)

```json
{
  "image": "test/0.jpg",
  "caption": ["description1", "description2", ...],
  "image_id": "some_id"
}
```

Note: `search_test_dataset` expects `ann['caption']` to be a **list** (multiple captions per image).

### 5.6 Inference query.json format (each line, for AIC26-style blind test)

Required fields (from `search_inference_dataset`):
```json
{
  "caption": "Natural language description...",
  "query_index": "some_unique_query_id"
}
```

`query_index` is used as the query ID in the output. `caption` provides the text query.

The `clip_infer.py` and `blip2_infer.py` only use the `caption` field (see `read_json_to_list`):
```python
data_list.append(item['caption'])
```

The `lhp_2/beit3/inference.py` also only reads `item['caption']`.

Only `search_inference_dataset` reads `ann['query_index']` — this is the field that controls query ordering in the UIT inference output.

### 5.7 Image filename format

Gallery images are referenced by filename stem only. Output lines are space-separated stems (`file.split('/')[-1][:-4]` removes path and `.jpg` extension). Output does NOT include file extensions.

### 5.8 Query ordering

- For CLIP, BLIP-2, and LHP: query order is controlled by the **line order in query.json** (JSONL is read sequentially)
- For UIT inference: query order is controlled by `query_index` field in query.json
- There is NO `query_index.txt` file — order depends entirely on JSONL iteration order
- **If query.json is re-sorted or re-written between models, row order in score matrices will mismatch**

---

## 6. Hardcoded Paths and Fragile Assumptions

### 6.1 Active broken hardcoded paths

| File | Line | Path | Type | Impact |
|---|---|---|---|---|
| `beit3_infer.py` | 162 | `/home/s48gb/Desktop/.../gallery` | **Active code** | Will break on any machine |
| `beit3_infer.py` | 164 | `/home/s48gb/Desktop/.../query.json` | **Active code** | Will break on any machine |
| `beit3_infer.py` | 140 | `torch.save(sims_tensor,'score_beit.pt')` | **Active code** | Saves to wrong location (CWD, not `./sims_score/`) |
| `beit3_infer.py` | 53 | `'unilm/beit3/beit3_large_patch16_384_coco_retrieval.pth'` | Default arg | Wrong path (expects COCO checkpoint, not LHP checkpoint) |

### 6.2 Commented-out hardcoded paths (inactive, informational)

| File | Line | Path | Note |
|---|---|---|---|
| `clip_infer.py` | 107, 114 | `/home/s48gb/Desktop/...` | Commented out |
| `blip/blip2.py` | 17 | `/home/s48gb/Desktop/...` | Commented out |

### 6.3 Relative paths (fragile — require specific CWD)

| File | Path Reference | Requires CWD |
|---|---|---|
| `lhp_2/beit3/inference.py` | `./checkpoint/lhp/lhp_beit3.pth` | Repo root |
| `lhp_2/beit3/inference.py` | `./sims_score/score_beit3_reproduce.pt` | Repo root |
| `lhp_2/beit3/inference.py` | `./data/PAB/name-masked_test-set/gallery` | Repo root |
| `clip_infer.py` | `./data/PAB/name-masked_test-set/gallery` | Repo root |
| `blip2_infer.py` | `./data/PAB/name-masked_test-set/gallery` | Repo root |
| `uit/cmp/configs/cmp.yaml` | `../../data/PAB/annotation/train/pair_N.json` | From `uit/cmp/` |
| `uit/cmp/configs/infer.yaml` | `./data/PAB/name-masked_test-set/gallery` | Repo root |
| `uit/cmp/configs/infer.yaml` | `uit/cmp/checkpoint/bert-base-uncased` | Repo root |
| `uit/cmp/run.py` | `./checkpoint/pretrained.pth` | From `uit/cmp/` |
| `uit/cmp/Search.py` | `output/356356/checkpoint_N.pth` | From `uit/cmp/` |

### 6.4 os.listdir() and glob

| File | Line | Pattern | Note |
|---|---|---|---|
| `clip_infer.py` | 108 | `os.listdir(args.image_folder)` | Active — see Section 7 |
| `blip2_infer.py` | 112 | `os.listdir(args.image_folder)` | Active — see Section 7 |
| `lhp_2/beit3/inference.py` | 145 | `os.listdir(args.image_folder)` | Active — see Section 7 |
| `uit/cmp/dataset/search_dataset.py` | 186 | `os.listdir(self.image_root)` | Active — see Section 7 |
| `lhp_2/beit3/datasets.py` | 12 | `import glob` | Imported but not used for PAB loading |

---

## 7. Gallery Ordering Risks

### 7.1 Summary

`os.listdir()` returns directory entries in **arbitrary filesystem order**, which differs between:
- Linux ext4 vs tmpfs vs NFS
- Different machines
- Before and after adding/removing files
- Different Python versions on some filesystems

All four inference scripts use `os.listdir()` **without** `sorted()`. This is a **critical alignment risk**: if the gallery order differs between two scripts, the score matrix columns map to different images, and the final ensemble will produce incorrect results.

### 7.2 Exact locations

**`clip_infer.py:108`** — CRITICAL
```python
image_folder = [os.path.join(args.image_folder, image_path) for image_path in os.listdir(args.image_folder)]
```

**`blip2_infer.py:112`** — CRITICAL
```python
image_folder = [os.path.join(args.image_folder, image_path) for image_path in os.listdir(args.image_folder)]
```

**`lhp_2/beit3/inference.py:145`** — CRITICAL
```python
image_folder = [os.path.join(args.image_folder, image_path) for image_path in os.listdir(args.image_folder)]
```

**`uit/cmp/dataset/search_dataset.py:186`** — CRITICAL (in `search_inference_dataset.__init__`)
```python
for img_id, file in enumerate(os.listdir(self.image_root)):
    self.g_pids.append(file.split('.')[0])
    self.image.append(file)
```

### 7.3 Why this breaks ensemble

The ensemble in `eval.py` computes:
```python
score_matrix_t2i = ... * beit3_itm + ... * beit3_itc + ... * blip2 + ... * clip
```

All four matrices must have the same column ordering (gallery order). If `clip_infer.py` runs on one machine and `blip2_infer.py` on another (or even in different Python processes), the column-to-gallery-image mapping may differ, silently corrupting the ensemble.

### 7.4 Fix required (do not apply yet — report only)

Replace `os.listdir(folder)` with `sorted(os.listdir(folder))` in all four locations above.

---

## 8. Score Matrix and Prediction Alignment

### 8.1 Score matrix shape

Based on file sizes (~15MB each) and 1978 query lines in prediction files:

```
sims_score/score_*.pt → torch.Tensor, shape [1978, 1978], dtype=float32
```

The matrix is text→image: `matrix[i, j]` = similarity of query `i` to gallery image `j`.

### 8.2 Gallery ID association

Column `j` in the score matrix corresponds to `image_folder[j]`, where `image_folder` is built by `os.listdir()`. The gallery image stem (filename without extension) is used as the answer ID. Each script writes:
```python
string = ' '.join([image_folder[id].split('/')[-1][:-4] for id in top10])
```

This strips the `.jpg` extension. Output answers use **stems only**, not full filenames.

### 8.3 Query ID association

Row `i` in the score matrix corresponds to line `i` of `query.json` (JSONL read sequentially). There is no explicit query ID tracked in the CLIP/BLIP-2/LHP scripts — the output file has one line per query in the same order as query.json.

For UIT inference, the dataset class tracks `ann['query_index']` as `q_pids`, but does NOT print it — the output file line order still maps to the JSONL iteration order.

### 8.4 Ensemble assumption

The ensemble in `uit/cmp/eval.py` assumes:
- `score_matrix_t2i` (UIT ITC), `score_matrix_t2i` (UIT ITM), `score_blip2`, and `score_clip` are all **exactly the same shape and same gallery order**
- No explicit alignment check is performed
- If any score tensor was computed with a different gallery ordering, results will be wrong

### 8.5 UIT inference re-uses BEiT-3 scores

`uit/cmp/inference.py:71`:
```python
sims_matrix_t2i = torch.load(args.beit3_score)
```
This **overwrites** the ITC features computed just before by `evaluation_itc()`. The loaded BEiT-3 score tensor is used as-is. This means the BEiT-3 tensor must already be computed before running UIT inference.

---

## 9. Checkpoint Requirements

| Checkpoint | Path expected | In repo? | Download link | Required for | Notes |
|---|---|---|---|---|---|
| LHP BEiT-3 fine-tuned | `checkpoint/lhp/lhp_beit3.pth` | NO | Not provided publicly (authors' trained weight) | LHP inference | **Must obtain from team or retrain** |
| BEiT-3 Large pretrained (COCO retrieval) | flexible | NO | [GitHub release](https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_coco_retrieval.pth) | LHP training init | ~2.4GB |
| BEiT-3 tokenizer | `lhp_2/beit3/beit3.spm` | **YES** | N/A | LHP training + inference | Already in repo |
| UIT/CMP checkpoint | `checkpoint/pretrained.pth` or `checkpoint/cmp.pth` | NO | [Google Drive](https://drive.google.com/file/d/1KffesfZD45kOQH2E4G31Sd3rbj9djD3d/view) (pretrained.pth from IRRA) | UIT inference | ~900MB |
| Swin-B ImageNet-22K | `checkpoint/swin_base_patch4_window7_224_22k.pth` | NO | [SwinTransformer GitHub](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) | UIT training init (alternative) | ~390MB |
| BERT-base-uncased | `checkpoint/bert-base-uncased/` | NO | [HuggingFace](https://huggingface.co/bert-base-uncased) | UIT training + inference | ~440MB, auto-download possible |
| BLIP-2 ViT-L pretrain | auto | NO | Auto-downloaded by LAVIS to `~/.cache/torch/hub/` | BLIP-2 inference | ~3.8GB |
| CLIP ViT-L/14@336px | auto | NO | Auto-downloaded by OpenAI CLIP | CLIP inference | ~900MB |
| BEiT-3 Large init | flexible | NO | [GitHub release](https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_224.pth) | Optional LHP pretrain init | ~2.9GB |

**Note on `beit3_infer.py` (root-level):** This file references `unilm/beit3/beit3_large_patch16_384_coco_retrieval.pth` as default — this is the ORIGINAL BEiT-3 COCO checkpoint path, not the LHP fine-tuned path. This file is broken and should not be used.

---

## 10. Dependency Notes

### 10.1 Two separate environments required

| Env | Python | PyTorch | Key packages | For |
|---|---|---|---|---|
| **LHP / BEiT-3** | Not specified | Not pinned | timm==0.4.12, torchscale==0.2.0, deepspeed==0.4.0, transformers, sentencepiece, protobuf==3.20.0 | `lhp_2/beit3/` scripts |
| **UIT / CMP** | Python 3.10 | 2.2.0 (README) / 2.5.1 (requirements_uit.txt) | timm==0.6.13, torchscale==0.3.0, transformers==4.47.1, ruamel.yaml, prettytable, yacs, nltk | `uit/cmp/` scripts |

### 10.2 Conflicts

- `timm==0.4.12` (LHP) vs `timm==0.6.13` (UIT) — **CONFLICT** — cannot share one conda env
- `torchscale==0.2.0` (LHP) vs `torchscale==0.3.0` (UIT) — **CONFLICT**
- `protobuf==3.20.0` (LHP) vs `protobuf==5.29.2` (UIT) — **CONFLICT**
- `deepspeed==0.4.0` (both) — same version but very old; may not compile on newer CUDA
- LAVIS (`blip/`) installs its own dependencies via `setup.py` — must be installed with `pip install -e blip/`

### 10.3 Colab A100 compatibility

- Uncertain — `deepspeed==0.4.0` is very old and may fail to compile CUDA kernels on A100 (SM80)
- BLIP-2 inference via LAVIS should work on A100 (LAVIS supports modern PyTorch)
- CLIP should work on any CUDA GPU
- LHP inference should work if timm==0.4.12 and torchscale==0.2.0 install correctly
- For Colab, recommend pinning PyTorch to a stable version before installing requirements

### 10.4 First-run steps

For LHP environment:
```bash
conda create -n lhp python=3.9
conda activate lhp
pip install torch torchvision timm==0.4.12 transformers sentencepiece torchscale==0.2.0
pip install -r requirements_lhp.txt
pip install -e blip/          # needed for blip2_infer.py
```

For UIT environment:
```bash
conda create -n uit python=3.10
conda activate uit
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_uit.txt
python -c "import nltk; nltk.download('wordnet')"
```

---

## 11. What This Fork Needs Before Becoming the Clean AIC26 Repo

### 11.1 Immediate blockers

1. **Gallery sort fix** (4 files, 4 lines) — All `os.listdir()` calls must be wrapped in `sorted()` before any new scores are generated
2. **Fix `beit3_infer.py`** — This file is broken (imports from empty `lhp/`). Either delete it or redirect imports to `lhp_2/beit3/`
3. **Resolve checkpoint locations** — Establish a standard `checkpoint/` layout at repo root and document it
4. **AIC26 query.json adapter** — AIC26 Track 4 likely uses a different annotation format; a conversion script is needed

### 11.2 Uncertain items that must be checked

- What is the exact AIC26 Track 4 query.json format? Does it have `caption`? Does it have `query_index`?
- Does AIC26 gallery use the same filename format as PAB (uppercase alphanumeric stems)?
- Does the LHP fine-tuned checkpoint (`lhp_beit3.pth`) still exist and produce valid scores?
- Is the existing `sims_score/score_*.pt` (1978×1978) computed on PAB test or AIC26 data?
- Is `uit/cmp/reproduce.txt` the final submission answer, or an intermediate? (Appears to be final ensemble output — 1978 lines, space-separated stems)
- Does the UIT checkpoint exist? Where is it stored?

---

## 12. Proposed Clean `aic26/` Folder Structure

Do not create these files yet. This is a proposed layout for the new AIC26-specific folder:

```
aic26/
│
├── README.md                    # AIC26 Track 4 specific guide
│
├── docs/
│   ├── setup_env.md             # Step-by-step environment setup
│   ├── data_prep.md             # AIC26 data download and layout instructions
│   ├── inference_guide.md       # End-to-end inference walkthrough
│   └── ensemble_notes.md        # Notes on score alignment and ensemble
│
├── configs/
│   ├── lhp_infer.yaml           # LHP inference config (AIC26 paths)
│   ├── uit_infer.yaml           # UIT inference config (AIC26 paths)
│   └── ensemble.yaml            # Ensemble weights config
│
├── scripts/
│   ├── prepare_data.sh          # Data layout and symlink setup
│   ├── run_lhp_infer.sh         # LHP inference wrapper
│   ├── run_blip2_infer.sh       # BLIP-2 inference wrapper
│   ├── run_clip_infer.sh        # CLIP inference wrapper
│   ├── run_uit_infer.sh         # UIT inference + ensemble wrapper
│   └── run_full_pipeline.sh     # End-to-end pipeline
│
├── utils/
│   ├── convert_query.py         # Convert AIC26 query format → PAB-compatible query.json
│   ├── validate_answer.py       # Validate output format before submission
│   ├── check_gallery_order.py   # Verify gallery order consistency across score files
│   └── merge_scores.py          # Standalone ensemble script (no UIT model required)
│
├── baselines/
│   └── pe_g14_baseline.py       # PE-G14 baseline implementation
│
└── submissions/
    └── .gitkeep                 # Placeholder for submission files (gitignored)
```

---

## 13. Recommended Next Inspection / Copy Steps

### Step 1 — Verify precomputed scores

Before copying anything from teammate repos:
```bash
python3 -c "
import torch
t = torch.load('sims_score/score_beit3_reproduce.pt', map_location='cpu')
print('beit3 shape:', t.shape, 'dtype:', t.dtype)
t = torch.load('sims_score/score_blip2_reproduce.pt', map_location='cpu')
print('blip2 shape:', t.shape, 'dtype:', t.dtype)
t = torch.load('sims_score/score_clip_reproduce.pt', map_location='cpu')
print('clip shape:', t.shape, 'dtype:', t.dtype)
"
```
Expected output: `[1978, 1978]` for all three.

### Step 2 — Confirm reproduce.txt is the ensemble output

The `uit/cmp/reproduce.txt` (309KB) contains 1978 lines. Confirm these match the format expected by AIC26 evaluation. If the AIC26 evaluator expects query IDs on each line, this file is missing them.

### Step 3 — Check AIC26 query format

Obtain the AIC26 Track 4 test query file. Compare its JSON fields against what `read_json_to_list()` expects (`caption` key, JSONL format). Write `aic26/utils/convert_query.py` if needed.

### Step 4 — Fix gallery sort before touching any inference code

Apply the `sorted()` fix to all four `os.listdir()` calls. This is a one-line change per file and must be done before any teammate code is merged.

### Step 5 — Establish checkpoint folder

Create `checkpoint/lhp/`, `checkpoint/bert-base-uncased/` layout. Copy or symlink required checkpoint files. Add `checkpoint/` to `.gitignore`.

### Step 6 — Copy teammate code

Once Steps 1–5 are done, copy teammate scripts/models into new folders under `aic26/` without modifying the original `lhp_2/`, `uit/`, or root-level files.

---

## Appendix A — Key Code Locations Quick Reference

| Concern | File:Line |
|---|---|
| Gallery sort bug (CLIP) | `clip_infer.py:108` |
| Gallery sort bug (BLIP-2) | `blip2_infer.py:112` |
| Gallery sort bug (LHP) | `lhp_2/beit3/inference.py:145` |
| Gallery sort bug (UIT) | `uit/cmp/dataset/search_dataset.py:186` |
| Hardcoded `/home/s48gb` (active) | `beit3_infer.py:162-164` |
| Wrong save path for beit3 score | `beit3_infer.py:140` |
| Ensemble formula | `uit/cmp/eval.py:113` |
| Ensemble weight defaults | `uit/cmp/inference.py:103-107` |
| BEiT-3 score overwrite | `uit/cmp/inference.py:71` |
| Query reading (caption field) | `clip_infer.py:54`, `blip2_infer.py:54` |
| Query reading (query_index field) | `uit/cmp/dataset/search_dataset.py:191` |
| Output uses stems not filenames | `clip_infer.py:93`, `lhp_2/beit3/inference.py:139` |
| UIT checkpoint save path | `uit/cmp/Search.py:135` (`output/356356/`) |
| Training data: 75 JSON files | `lhp_2/beit3/datasets.py:35` (`self.num_files = 75`) |
| Training data: 75 JSON files | `uit/cmp/configs/cmp.yaml:5-52` |
| BERT path in infer config | `uit/cmp/configs/infer.yaml:62` |
