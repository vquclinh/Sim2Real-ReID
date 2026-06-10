# TEAMMATE_REPO_COPY_AUDIT.md

> **AIC/ECCV 2026 — Track 4: Text-Based Person Anomaly Search**
> Independent audit of teammate repo for safe cherry-pick into forked HUI repo.
> Generated: 2026-06-10. Based on file inspection + git log. Do not copy until all
> RISK items are resolved.

---

## Table of Contents

1. [Repo Overview](#1-repo-overview)
2. [SECURITY WARNING — Live Credentials](#2-security-warning--live-credentials)
3. [Current Submitted Pipeline](#3-current-submitted-pipeline)
4. [Execution Status of Every Notebook](#4-execution-status-of-every-notebook)
5. [PE-G14 Baseline Code Walkthrough](#5-pe-g14-baseline-code-walkthrough)
6. [AIC26 Utility Scripts](#6-aic26-utility-scripts)
7. [Notebooks Upgrade Pipeline](#7-notebooks-upgrade-pipeline)
8. [Notebooks AIO Pipeline](#8-notebooks-aio-pipeline)
9. [Hardcoded Paths](#9-hardcoded-paths)
10. [Dependencies Between Files](#10-dependencies-between-files)
11. [Important Files Table](#11-important-files-table)
12. [Notebooks Status Table](#12-notebooks-status-table)
13. [Copy Candidates Table](#13-copy-candidates-table)
14. [Do-Not-Copy List](#14-do-not-copy-list)
15. [Recommended Safe Copy Order](#15-recommended-safe-copy-order)

---

## 1. Repo Overview

```
ACC26/
├── ARCHITECTURE.md              ← Master technical doc (~1250 lines, Vietnamese)
│                                   Complete math, per-notebook deep-dive, 7 documented bugs
├── REPO_HANDOFF_SUMMARY.md      ← Teammate's own handoff doc (34 KB) — well written, accurate
├── zero_shot/                   ← ONLY EXECUTED PIPELINE
│   ├── zero_shot_pe_g14.ipynb   ← Generated Document/answer.txt (submitted result)
│   └── aic_colab_utils.py       ← Infrastructure utilities
├── notebooks_upgrade/           ← FULL PIPELINE — WRITTEN, NEVER RUN
│   ├── 00_manifest_qc.ipynb
│   ├── 01a_pe_g14_features.ipynb
│   ├── 01b_vitpose_features.ipynb
│   ├── 02_uit_train.ipynb
│   ├── 03_lhp_peg14_train.ipynb
│   ├── 04_uit_inference.ipynb
│   ├── 05_blip2_inference.ipynb
│   ├── 06_clip_inference.ipynb
│   ├── 07_pe_g14_scores.ipynb
│   ├── 08_kreciprocal_rerank.ipynb
│   ├── 09_adaptive_ensemble_submit.ipynb
│   ├── README.md
│   └── aic_colab_utils.py       ← Identical to zero_shot version
├── notebooks_AIO/               ← PAPER-FAITHFUL BASELINE — WRITTEN, NEVER RUN
│   ├── 01_lhp_beit3_train.ipynb
│   ├── 02_uit_train.ipynb
│   ├── 03_lhp_beit3_inference.ipynb
│   ├── 04_blip2_inference.ipynb
│   ├── 05_clip_inference.ipynb
│   ├── 06_uit_ensemble_submit.ipynb
│   ├── README.md
│   ├── aic_colab_utils.py       ← Identical to zero_shot version
│   ├── aio_paper_utils.py       ← Paper-specific helpers (different from aic_colab_utils.py)
│   └── __pycache__/
│       └── aic_colab_utils.cpython-314.pyc   ← Bytecode cache — do not copy
├── Document/
│   ├── answer.txt               ← SUBMITTED ANSWER (1978 lines, verified correct format)
│   ├── AIO_paper.pdf
│   ├── AIC 2026 Track 4_...pdf
│   ├── BPAD_E_v2.1_Pipeline.docx
│   ├── Track4_BPAD_IdeaReport.docx
│   └── Track4_Synthesis_Report.docx
├── Dataset_Structure/
│   └── image_dataset_structure_on_drive.png
├── Hybrid-Unified-and-Iterative-.../  ← EMPTY DIRECTORY (placeholder — not a submodule)
├── rclone.txt                   ← 🚨 LIVE Google Drive OAuth token — REVOKE NOW
├── token_kaggle.txt             ← 🚨 LIVE Kaggle API token — REVOKE NOW
└── .gitignore                   ← Only ignores Dataset_Structure/aic2026_data and
                                    ui-streamlit.ipynb — does NOT protect credentials
```

**Git history** (3 commits):
| Commit | Message | Key files added |
|--------|---------|-----------------|
| `5783ad7f` | upload pipeline | notebooks_upgrade/, notebooks_AIO/, ARCHITECTURE.md |
| `3bdc8a75` | build baseline | Cleanup of AIC2025 artifacts, removed annotation JSONs |
| `8318cf7c` | complete build zero-shot | **Document/answer.txt**, zero_shot/, updated aic_colab_utils.py |

---

## 2. SECURITY WARNING — Live Credentials

**Both credential files contain LIVE, UNREVOCED tokens committed to git history.**
They are accessible to anyone with repo access and cannot be fully removed by `.gitignore` alone because they are already in git history.

### rclone.txt

**Type**: Google Drive OAuth2 token
**Content structure**:
```
[gdrive]
type: drive
token: {"access_token":"ya29.a0AQ...", "refresh_token":"1//0gRQ_P5MkdMt_...", "expiry":"..."}
```
**Risk**: The `refresh_token` is permanent until revoked. Anyone with this file can mount your Google Drive with full read/write access.

**Action required**: Go to **Google Account → Security → Third-party apps with account access** → find rclone → remove access.

### token_kaggle.txt

**Type**: Kaggle API access token
**Content**: `KGAT_f560a9fca6f3bb106c990cc457f33fa1`
**Risk**: Allows anyone to download datasets, submit kernels, and use Kaggle API resources under this account.

**Action required**: Go to **kaggle.com → Account → API → Expire API Token**.

### Git history note

Even after revoking both tokens, their VALUES remain in git history at commit `5783ad7f` (rclone) and `8318cf7c` (token_kaggle). If this repo is ever pushed to a public remote, those commits will be visible. Use `git filter-branch` or BFG Repo-Cleaner to scrub history if public exposure is a risk.

---

## 3. Current Submitted Pipeline

**Leaderboard result** (confirmed from commit `8318cf7c` which added both the notebook and the answer.txt simultaneously):

| Metric | Score |
|--------|-------|
| mAP | **67.9311** |
| R@1 | 56.5217 |
| R@5 | 82.8109 |
| R@10 | 88.3721 |

**Pipeline**: `zero_shot/zero_shot_pe_g14.ipynb`

**Type**: Zero-shot cosine retrieval. No fine-tuning. No ensemble. No reranking.

### Exact code path that produced answer.txt

```
Cell 1: Bootstrap
  - Calls setup_aic2026_environment() → returns PATHS dict
  - Calls select_a100_device() → A100 GPU
  - Default Drive: /content/drive/MyDrive/aic2026_data/

Cell 2: Locate query files
  - QUERY_JSON = TEST_DIR / 'query_text.json'   (JSONL: query_index + caption)
  - QUERY_INDEX_TXT = TEST_DIR / 'query_index.txt'  (1978 lines, DEFINES ORDER)
  - submission_qids = [ln.strip() for ln in QUERY_INDEX_TXT.read_text().splitlines() if ln.strip()]
  - qid_to_caption = dict from JSONL

Cell 3: Gallery listing
  - _ls_dir(GALLERY_DIR) — uses shell `ls` (not Python iterdir) to avoid Drive FUSE Errno 5
  - gallery_paths = sorted([...])  ← lexicographic sort, deterministic
  - gallery_ids = [p.stem for p in gallery_paths]  ← filename stems only, NO extension

Cell 4: Install perception_models repo
  - git clone facebookresearch/perception_models

Cell 5: Load model
  - pe_model = pe.CLIP.from_config('PE-Core-G14-448', pretrained=True)
  - BF16, channels_last memory format

Cell 6: Encode gallery images
  - IMG_BATCH = 96 (A100-40GB) or 256 (A100-80GB)
  - DataLoader with 16 workers + prefetch_factor=4
  - Embeddings: BF16 → F.normalize(dim=-1) → fp16 → NPZ cache chunks
  - Gallery shape: (36773, 1280) float16, L2-normalized

Cell 7: Encode query texts
  - ORDER: query_index.txt order (CRITICAL for submission correctness)
  - captions = [qid_to_caption[qid] for qid in submission_qids]
  - TEXT_BATCH = 256
  - Query shape: (1978, 1280) float16, L2-normalized

Cell 8: Similarity + top-10
  - G_t = gallery embeddings → float32 on GPU
  - Q_t = query embeddings → float32 on GPU
  - Chunked: for s in range(0, Q_t.size(0), Q_CHUNK=256):
      sims = Q_chunk @ G_t.T   ← cosine (vectors are L2-normalized)
      top10_indices = torch.topk(sims, k=10, dim=1).indices

Cell 9: Sanity check
  - Verifies Milvus L2 argsort ≡ cosine argsort for L2-normalized vectors

Cell 10: Write output
  - ANSWER_TXT = LOCAL_ROOT / 'submission_zero_shot' / 'answer.txt'
  - 1978 lines × 10 space-separated gallery stems

Cell 11: Format validation
  - assert 1978 lines
  - assert 10 unique gallery IDs per line
  - assert all IDs exist in gallery set
  - Optional comparison vs /home/bao/Documents/AIC2026/Document/answer.txt
    (SAFE: guarded by if sample.exists() — silently skipped on new machine)
```

**Execution note**: The notebook has `execution_count=None` for all cells (outputs were cleared before committing — standard practice). Evidence that it was run: `Document/answer.txt` was committed in the same commit as the notebook.

---

## 4. Execution Status of Every Notebook

**Methodology**: checked `execution_count` field and `outputs` for all code cells via JSON parsing.

| Notebook | Location | Code Cells | exec_count set | Has Outputs | **Verdict** |
|----------|----------|-----------|---------------|-------------|------------|
| zero_shot_pe_g14.ipynb | zero_shot/ | 11 | None (cleared) | No | **WAS RUN** (Document/answer.txt is proof) |
| 00_manifest_qc.ipynb | notebooks_upgrade/ | 8 | 0 | No | NEVER RUN |
| 01a_pe_g14_features.ipynb | notebooks_upgrade/ | 6 | 0 | No | NEVER RUN |
| 01b_vitpose_features.ipynb | notebooks_upgrade/ | 6 | 0 | No | NEVER RUN |
| 02_uit_train.ipynb | notebooks_upgrade/ | 8 | 0 | No | NEVER RUN |
| 03_lhp_peg14_train.ipynb | notebooks_upgrade/ | 9 | 0 | No | NEVER RUN |
| 04_uit_inference.ipynb | notebooks_upgrade/ | 10 | 0 | No | NEVER RUN |
| 05_blip2_inference.ipynb | notebooks_upgrade/ | 6 | 0 | No | NEVER RUN |
| 06_clip_inference.ipynb | notebooks_upgrade/ | 5 | 0 | No | NEVER RUN |
| 07_pe_g14_scores.ipynb | notebooks_upgrade/ | 1 | 0 | No | NEVER RUN |
| 08_kreciprocal_rerank.ipynb | notebooks_upgrade/ | 3 | 0 | No | NEVER RUN |
| 09_adaptive_ensemble_submit.ipynb | notebooks_upgrade/ | 9 | 0 | No | NEVER RUN |
| 01_lhp_beit3_train.ipynb | notebooks_AIO/ | 10 | 0 | No | NEVER RUN |
| 02_uit_train.ipynb | notebooks_AIO/ | 10 | 0 | No | NEVER RUN |
| 03_lhp_beit3_inference.ipynb | notebooks_AIO/ | 9 | 0 | No | NEVER RUN |
| 04_blip2_inference.ipynb | notebooks_AIO/ | 7 | 0 | No | NEVER RUN |
| 05_clip_inference.ipynb | notebooks_AIO/ | 6 | 0 | No | NEVER RUN |
| 06_uit_ensemble_submit.ipynb | notebooks_AIO/ | 10 | 0 | No | NEVER RUN |

**Generated file search**: no `.npz`, `.pt`, `.pth`, `.parquet`, `.log` files exist anywhere in the repo (only PDFs and answer.txt). All training artifacts, embeddings, and checkpoints are on the teammate's Google Drive only — not in the repo.

---

## 5. PE-G14 Baseline Code Walkthrough

### Files involved

| File | Lines | Role |
|------|-------|------|
| `zero_shot/zero_shot_pe_g14.ipynb` | 11 code cells | Main pipeline |
| `zero_shot/aic_colab_utils.py` | 1099 lines | All infrastructure |

### Required external dependencies

| Dependency | How obtained | Notes |
|-----------|--------------|-------|
| `facebookresearch/perception_models` | `git clone --depth 1` in cell 4 | PE-Core-G14-448 model |
| `timm`, `ftfy`, `regex`, `tokenizers`, `einops` | `pip install` in cell 4 | PE-G14 deps |
| `torch>=2.3` | Pre-installed on Colab A100 | BF16, channels_last |
| PAB test set | From Drive via `setup_aic2026_environment()` | gallery/, query_text.json, query_index.txt |

### Dataset path assumptions

The notebook calls `setup_aic2026_environment()` with defaults:
- `drive_root = "/content/drive/MyDrive/aic2026_data"` — **HARDCODED DEFAULT, must be overridden if your Drive path differs**
- `local_root = "/content/aic_local"` — fine for Colab
- Expects: `<drive_root>/raw/name-masked_test-set/gallery/gallery/*.jpg` (nested) OR `.../gallery/*.jpg` (flat). Auto-detected.
- Expects: `<drive_root>/raw/name-masked_test-set/query_text.json`
- Expects: `<drive_root>/raw/name-masked_test-set/query_index.txt`

### Output paths

- Local: `<local_root>/submission_zero_shot/answer.txt` + `answer.zip`
- Drive: `<drive_root>/output/submission_zero_shot/answer.txt`

### Cache/embedding paths

- Gallery embeddings cached in NPZ chunks: `<local_root>/output/features/pe_g14/chunk_*.npz`
- Each chunk: `{"embeddings": (chunk_size, 1280), "ids": [...]}` — resume-safe
- Drive sync: async background thread via `sync_chunk_to_drive()`

### Can it be moved cleanly?

**Yes**, with two changes:
1. Override `drive_root` parameter if your Drive layout is different
2. The optional comparison in cell 11 (`/home/bao/Documents/AIC2026/Document/answer.txt`) silently skips on new machines — no change needed

### What needs converting from notebook to script

The pipeline is currently 11 notebook cells. To convert to a `.py` script:
- Cells 1-3: argument parsing + setup
- Cells 4-7: model loading + encoding (can be parameterized)
- Cells 8-11: similarity + output + validation

This conversion is **optional** — the notebook runs end-to-end as-is on Colab.

---

## 6. AIC26 Utility Scripts

### aic_colab_utils.py — three identical copies

**Confirmed**: the three copies at `zero_shot/aic_colab_utils.py`, `notebooks_upgrade/aic_colab_utils.py`, and `notebooks_AIO/aic_colab_utils.py` are **byte-for-byte identical** (verified with `diff`).

**Use the `zero_shot/` version** as the canonical source (it's the one that was actually tested by the executed pipeline). All three are safe to copy.

| Function | Purpose | Colab-specific? | Path-sensitive? | Safe to commit? |
|---------|---------|----------------|-----------------|-----------------|
| `setup_aic2026_environment()` | Mount Drive, symlink/rsync raw+manifests, return paths dict | YES (Drive mount) | YES (drive_root default) | YES (no secrets embedded) |
| `select_a100_device()` | Pick GPU, enable TF32/BF16/Flash-SDPA | No | No | YES |
| `save_npz_atomic()` | tmp → rename atomic write | No | No | YES |
| `sync_chunk_to_drive()` | Background copy local → Drive | YES (Drive path) | YES | YES |
| `wait_for_pending_syncs()` | Join all sync threads | No | No | YES |
| `find_existing_chunks()` | Union of local + Drive chunk basenames | No | No | YES |
| `chunk_done()` | True if chunk exists locally or on Drive | No | YES | YES |
| `l2_normalize_np()` | L2-norm for numpy arrays | No | No | YES |
| `mirror_raw_as_tar_split()` | Archive raw data as 4.5GB parts | YES (Drive write) | YES | YES |
| `restore_raw_from_tar_split()` | Restore from tar parts | YES | YES | YES |
| `stage_test_set_local()` | Copy test set from Drive to local SSD | YES | YES | YES |
| `_ensure_rclone()` | Install rclone + write config | YES (Colab install) | YES (rclone.txt path) | YES — but note: uses rclone.txt internally via auto-search. Do NOT commit rclone.txt alongside. |

**Credentials inside the utility**: The `_ensure_rclone()` function auto-searches for `rclone.txt` at several paths including `~/Documents/AIC2026/rclone.txt`. It does NOT embed credentials — it reads them from a separate file. Safe to commit the utility itself.

**Hardcoded path in utility**: `_ensure_rclone()` line ~294-297 has a fallback candidate `Path.home() / "Documents/AIC2026/rclone.txt"`. This is a local dev fallback — harmless on Colab.

### aio_paper_utils.py — only in notebooks_AIO/

This file is distinct from `aic_colab_utils.py` and provides paper-specific helpers:

| Function | Purpose | Required for |
|---------|---------|--------------|
| `stage_paper_layout()` | Build symlink farm mapping raw PAB → paper expected `data/PAB/...` layout | All AIO notebooks + notebooks_upgrade/02 |
| `clone_aio_repo()` | rsync or `git clone` paper repo to local | notebooks_upgrade/02, all AIO notebooks |
| `ensure_lhp_assets()` | Download BEiT-3 weights from GitHub release | AIO/01 |
| `ensure_uit_assets()` | Verify/fetch Swin-B + BERT for UIT | AIO/02, upgrade/02 |
| `generate_pair_jsonl()` | Convert parquet manifest → paper-format pair_N.json JSONL | AIO/01, AIO/02 |
| `get_sorted_gallery_paths()` | Single sorted gallery order for score-matrix alignment | AIO/03, 04, 05, 06 |
| `drive_sync_thread()` | Background daemon copying checkpoints to Drive | AIO/01, AIO/02 |
| `latest_checkpoint()` | Find highest-epoch checkpoint | AIO/01, AIO/02 |

**Hardcoded URLs** in aio_paper_utils.py:
- `AIO_REPO_GIT_URL`: `https://github.com/AIVIETNAM-Hub/Hybrid-Unified-and-Iterative-...` (line 146) — this is the paper's public GitHub repo. If your HUI fork is at a different URL, update this constant.
- `_BEIT3_SPM_URL`, `_BEIT3_CKPT_URL`: GitHub release assets for BEiT-3 — stable
- `_SWIN_B_URL`: GitHub release for Swin-B — stable

**Credentials**: None embedded. Safe to commit.

---

## 7. Notebooks Upgrade Pipeline

All 11 notebooks share a common structure: (1) bootstrap + config cell, (2-N) code cells, all written but never executed.

| # | Notebook | Purpose | Corresponds to HUI component | Inputs needed | Outputs produced | Run status |
|---|----------|---------|------------------------------|--------------|-----------------|------------|
| 00 | manifest_qc | Build train/gallery/query parquets + val splits | New (not in paper) | raw/ on Drive | manifests/*.parquet | NEVER RUN |
| 01a | pe_g14_features | PE-G14 encode gallery+queries+val (chunked) | Upgrade 1 (new model) | gallery/, query_text.json, manifests | features/pe_g14/{gallery,queries,val,val_text}.npz | NEVER RUN |
| 01b | vitpose_features | ViTPose++ keypoints for anatomical crop | Upgrade (optional) | gallery/ | features/vitpose/keypoints.npz | NEVER RUN |
| 02 | uit_train | UIT Swin-B+BERT training 22 epoch, 4 losses | Paper UIT (modified) | aio_repo/, manifests, train/ | checkpoints/uit/checkpoint_best.pth | NEVER RUN |
| 03 | lhp_peg14_train | LHP with PE-G14-448 + LoRA r=16, 3 epoch | Upgrade 3 (PE-G14 replaces BEiT-3) | train/, manifests, PE-G14 | checkpoints/lhp_peg14/lora_best/, features/lhp_peg14/scores_lhp.pt | NEVER RUN |
| 04 | uit_inference | Algorithm 1: LHP top-256 → ITM rerank | Paper Algorithm 1 (corrected) | scores_lhp.pt, uit checkpoint, gallery | features/uit/scores_uit.pt | NEVER RUN |
| 05 | blip2_inference | BLIP-2 ITC + ITM rerank top-1024 | Paper BLIP-2 (Round 2) | gallery, queries | features/blip2/scores_blip2.pt | NEVER RUN |
| 06 | clip_inference | OpenAI CLIP ViT-L/14@336 cosine | Paper CLIP (Round 3) | gallery, queries | features/clip/scores_clip.pt | NEVER RUN |
| 07 | pe_g14_scores | Build score matrix from 01a embeddings | Upgrade 1 (Round 4) | features/pe_g14/{gallery,queries}.npz | features/pe_g14/scores_pe.pt | NEVER RUN |
| 08 | kreciprocal_rerank | k-reciprocal Jaccard rerank k1=20,k2=6,λ=0.3 | New (not in paper) | scores_{uit,blip2,clip,pe}.pt | scores_*_rr.pt | NEVER RUN |
| 09 | adaptive_ensemble_submit | Adaptive 4-way fusion + val gate + submission | Upgrade 2+4 | all *_rr.pt (or raw scores), val manifests | answer.zip | NEVER RUN |

### Bug status in upgrade notebooks

Seven bugs are documented and marked as fixed in ARCHITECTURE.md §9. Two are in the upgrade notebooks:

| Bug | Notebook | Status in code |
|-----|----------|----------------|
| Bug 1: `proj_head` orphan (LHP LoRA train/infer) | `03_lhp_peg14_train.ipynb` | **CONFIRMED FIXED** — cell 8 has explicit `# CRITICAL FIX` comment and loads `proj_head.pt` at inference |
| Bug 2: Algorithm 1 bypass (ITC only, no ITM) | `04_uit_inference.ipynb` | Uncertain — notebook was never run; fix described in ARCHITECTURE.md §9.2 |

---

## 8. Notebooks AIO Pipeline

Paper-faithful reproduction of the original HUI paper (BEiT-3-large backbone, not PE-G14).

| # | Notebook | Purpose | Original/Adapted | Inputs | Outputs | Run status |
|---|----------|---------|-----------------|--------|---------|------------|
| 01 | lhp_beit3_train | LHP with BEiT-3-large, full fine-tune, 4 epochs | Original (paper-faithful) | aio_repo/, train/ | output/lhp/checkpoint-best.pth | NEVER RUN |
| 02 | uit_train | UIT Swin-B+BERT, 30 epoch, 4 losses | Original (paper-faithful) | aio_repo/, train/ | output/uit/checkpoint_29.pth | NEVER RUN |
| 03 | lhp_beit3_inference | BEiT-3 score matrix (36773×1978) | Original | checkpoint-best.pth, gallery | sims_score/score_beit3.pt | NEVER RUN |
| 04 | blip2_inference | BLIP-2 inference via LAVIS | Original | gallery, queries | sims_score/score_blip2.pt | NEVER RUN |
| 05 | clip_inference | CLIP ViT-L/14@336 cosine | Original | gallery, queries | sims_score/score_clip.pt | NEVER RUN |
| 06 | uit_ensemble_submit | Algorithm 1 + Algorithm 2 + answer.zip | Original | all score matrices, uit checkpoint | submission/answer.zip | NEVER RUN |

**Relationship to HUI repo**: The AIO notebooks wrap the paper's original code. They call into `aio_repo/` (cloned via `clone_aio_repo()`) for training loops and model architectures. The AIO notebooks themselves are thin orchestration wrappers, NOT standalone implementations.

**If you already have a fork of HUI repo**: The AIO notebooks are still useful as reference, but `aio_paper_utils.py` (especially `stage_paper_layout()`, `get_sorted_gallery_paths()`, `generate_pair_jsonl()`) remains valuable infrastructure regardless.

---

## 9. Hardcoded Paths

All hardcoded absolute paths found in the repo:

| File | Location | Path | Risk | Action needed |
|------|----------|------|------|---------------|
| `zero_shot/zero_shot_pe_g14.ipynb` | Cell 11 (format validation) | `/home/bao/Documents/AIC2026/Document/answer.txt` | **SAFE** — guarded by `if sample.exists()`. Silently skipped on any other machine. | No change needed |
| `notebooks_upgrade/02_uit_train.ipynb` | Cell 2, line 6 | `/home/bao/Documents/AIC2026/Hybrid-Unified-and-Iterative-.../` | **RISK** — fallback path for `WORKSPACE_AIO_REPO`. Silently skipped on other machines; if Drive also missing, `git clone` runs from GitHub. Not catastrophic but confusing. | Change to your HUI fork path or remove fallback |
| All notebooks | Cell 1 bootstrap | `/content/drive/MyDrive/aic2026_data` | Soft default — pass `drive_root=...` to override. | Override when calling `setup_aic2026_environment()` |
| All notebooks | Cell 1 bootstrap | `/content/aic_local` | Soft default for Colab local SSD | Fine for Colab; override for Kaggle |
| `aic_colab_utils.py` | `_ensure_rclone()` line ~294 | `~/Documents/AIC2026/rclone.txt` | Dev fallback — harmless if no rclone config file exists there | No action needed |
| `aio_paper_utils.py` | `AIO_REPO_GIT_URL` (line 146) | `https://github.com/AIVIETNAM-Hub/...` | Hardcoded GitHub URL for paper repo | **Update to your HUI fork URL** before using |

---

## 10. Dependencies Between Files

### If copying `zero_shot/zero_shot_pe_g14.ipynb`

- **Requires**: `zero_shot/aic_colab_utils.py` (same directory, `import aic_colab_utils`)
- **Requires**: PAB test set on Google Drive at `drive_root/raw/name-masked_test-set/`
- **Requires**: Colab A100 (or A100-equivalent) — `select_a100_device()` will warn on lesser GPU
- **Does NOT require**: manifests, training data, AIO repo, checkpoints

### If copying `notebooks_upgrade/aic_colab_utils.py`

- **Self-contained** — no imports from repo other than stdlib + numpy
- **Requires at runtime**: `google.colab` (Drive mount), `kagglehub` (optional, only if `use_kaggle=True`), `torch` (only in `select_a100_device()`)

### If copying `notebooks_AIO/aio_paper_utils.py`

- **Requires**: `pandas`, `huggingface_hub` (for BERT download in `ensure_uit_assets()`), `subprocess` (for rsync/git)
- **Requires at runtime**: The AIO paper repo (will `git clone` if not found)
- **Does NOT import**: `aic_colab_utils` — they are complementary, not nested

### If copying `notebooks_upgrade/02_uit_train.ipynb`

- **Requires**: `aic_colab_utils.py` in same folder
- **Requires**: `aio_paper_utils.py` OR the paper repo already cloned (UIT training runs inside `aio_repo/`)
- **Requires**: `manifests/train_manifest.parquet` (from notebook 00)
- **Hardcoded fallback**: `/home/bao/Documents/...` — update to your path

### If copying `notebooks_upgrade/03_lhp_peg14_train.ipynb`

- **Requires**: `aic_colab_utils.py` + `manifests/` + gallery images
- **Requires**: PE-G14 perception_models (cloned in notebook)
- **Bug fix in place**: `proj_head` is correctly applied at train AND inference (cell 8 explicitly documents this)
- **Verify before running**: Load logic at cell 8 — ensure `proj_head.load_state_dict(...)` is reached correctly

### If copying any `notebooks_upgrade/` notebook from 04 onward

- Notebooks 04–09 consume score matrices produced by earlier notebooks
- **Gallery ID alignment is critical**: all score matrices must use the same gallery sort order. Both the `aic_colab_utils.py` gallery loader and `aio_paper_utils.get_sorted_gallery_paths()` do lexicographic sort — they will agree if gallery paths are identical.
- **09 requires val split** (from 00) for the adaptive weight gate to function correctly

### If copying `ARCHITECTURE.md`

- Self-contained reference document — no executable dependencies
- References old local paths (`/home/bao/...`) in some code snippets but these are documentation only

---

## 11. Important Files Table

| Path in teammate repo | Purpose | Category | Reason | Risks / Required edits |
|----------------------|---------|----------|--------|------------------------|
| `zero_shot/zero_shot_pe_g14.ipynb` | Submitted pipeline — reproduces ~67.9 mAP | **MUST COPY** | Only executed pipeline; baseline to verify your setup works | Cell 11 optional path harmless. Drive root may need override. |
| `zero_shot/aic_colab_utils.py` | Infrastructure — use this canonical version | **MUST COPY** | Tested by the submitted pipeline; 3 copies are identical | No edits needed; do not commit rclone.txt alongside |
| `notebooks_upgrade/README.md` | Run order, ETA, upgrade descriptions | **MUST COPY** | Essential operational reference | None |
| `notebooks_upgrade/00_manifest_qc.ipynb` | Build train/gallery/query parquets + val splits | **MUST COPY** | Required before any training notebook | None |
| `notebooks_upgrade/01a_pe_g14_features.ipynb` | PE-G14 encode all splits | **MUST COPY** | Required for 03, 07, and backup zero-shot | None |
| `notebooks_upgrade/02_uit_train.ipynb` | UIT training (core model) | **MUST COPY** | Biggest expected mAP jump | Fix cell 2 line 6: change `/home/bao/...` fallback to your path |
| `notebooks_upgrade/03_lhp_peg14_train.ipynb` | LHP-LoRA fine-tune PE-G14 | **MUST COPY** | Guides Algorithm 1 candidate selection | proj_head bug confirmed fixed. Verify cell 8 before running. |
| `notebooks_upgrade/04_uit_inference.ipynb` | Algorithm 1 + ITM rerank | **MUST COPY** | Critical — activates ITM signal from UIT training | Bug 2 fix: verify Algorithm 1 is fully implemented |
| `notebooks_upgrade/05_blip2_inference.ipynb` | BLIP-2 Round 2 | **MUST COPY** | Ensemble component | None |
| `notebooks_upgrade/06_clip_inference.ipynb` | CLIP Round 3 | **MUST COPY** | Ensemble component | None |
| `notebooks_upgrade/07_pe_g14_scores.ipynb` | PE-G14 score matrix from 01a | **MUST COPY** | Fast (5 min), uses cached embeddings | None |
| `notebooks_upgrade/08_kreciprocal_rerank.ipynb` | k-reciprocal Jaccard rerank | **MUST COPY** | +2-3 mAP expected | Needs 5.4GB GPU for GG matrix |
| `notebooks_upgrade/09_adaptive_ensemble_submit.ipynb` | Adaptive 4-way fusion + submission | **MUST COPY** | Final output generator | Score scale mismatch (BLIP-2 logits vs cosine) handled by z-norm — verify it's present |
| `notebooks_upgrade/01b_vitpose_features.ipynb` | ViTPose++ keypoints | **SHOULD COPY** | Optional anatomical crop enhancement | Can skip on first run |
| `notebooks_AIO/aio_paper_utils.py` | Paper-specific helpers: symlink farm, repo clone, JSONL gen | **MUST COPY** | Required by 02_uit_train; `get_sorted_gallery_paths()` critical for score alignment | Update `AIO_REPO_GIT_URL` (line 146) to your HUI fork URL |
| `notebooks_AIO/README.md` | Paper-faithful baseline notes | **SHOULD COPY** | Reference for comparison | None |
| `notebooks_AIO/*.ipynb` | Paper-faithful 6-notebook baseline | **SHOULD COPY** | Diagnostic ablation; upper-bound reference | These wrap `aio_repo/` — need paper repo accessible |
| `ARCHITECTURE.md` | Complete technical reference (Vietnamese) | **SHOULD COPY** | 7 documented bugs, math, hyperparameters | Mentions `/home/bao/` in code snippets (docs only, not executable) |
| `Document/answer.txt` | Submitted answer (1978 lines, verified format) | **SHOULD COPY** | Baseline to compare any new submission against | None |
| `Document/AIO_paper.pdf` | Foundation paper | **SHOULD COPY** | Essential reading | None |
| `Document/AIC 2026 Track 4_...pdf` | Strategic roadmap | **OPTIONAL** | Background context | None |
| `Document/*.docx` | Internal team documents | **OPTIONAL** | Background only | None |
| `Dataset_Structure/image_dataset_structure_on_drive.png` | Drive layout diagram | **OPTIONAL** | Reference | None |
| `REPO_HANDOFF_SUMMARY.md` | Teammate's handoff doc | **OPTIONAL** | Already superseded by this audit | None |
| `rclone.txt` | Google Drive OAuth token | **DO NOT COPY** | Live credentials — revoke immediately | N/A |
| `token_kaggle.txt` | Kaggle API token | **DO NOT COPY** | Live credentials — revoke immediately | N/A |
| `notebooks_AIO/__pycache__/` | Python bytecode cache | **DO NOT COPY** | Generated, CPython 3.14 specific | N/A |
| `Hybrid-Unified-and-Iterative-.../` | Empty directory placeholder | **DO NOT COPY** | Contains nothing | N/A |
| `.git/` | Git history with committed credentials | **DO NOT COPY** | Start fresh git history in your fork | N/A |
| `.gitignore` | Minimal ignore rules | **DO NOT COPY** | Insufficient — does not protect credentials or generated files | Write a new one |

---

## 12. Notebooks Status Table

### notebooks_upgrade/

| Notebook | Paper component | Original/Adapted | Run? | Depends on |
|----------|----------------|-----------------|------|------------|
| 00_manifest_qc | New (not in paper) | New | No | raw/ annotation JSONs |
| 01a_pe_g14_features | Upgrade 1 (new model) | New | No | PE-G14 repo, gallery/, manifests |
| 01b_vitpose_features | Upgrade (optional) | New | No | ViTPose++ repo, gallery/ |
| 02_uit_train | Paper UIT (Swin-B+BERT) | Adapted | No | aio_repo/, manifests, train/ |
| 03_lhp_peg14_train | Upgrade 3 (PE-G14 LoRA) | New | No | PE-G14, train/, manifests |
| 04_uit_inference | Paper Algorithm 1 (corrected) | Adapted | No | scores_lhp.pt, uit checkpoint, gallery |
| 05_blip2_inference | Paper BLIP-2 Round 2 | Adapted | No | gallery, queries |
| 06_clip_inference | Paper CLIP Round 3 | Adapted | No | gallery, queries |
| 07_pe_g14_scores | Upgrade 1 Round 4 | New | No | 01a gallery+queries .npz |
| 08_kreciprocal_rerank | New (not in paper) | New | No | all 4 score matrices |
| 09_adaptive_ensemble_submit | Upgrade 2+4 | New | No | all *_rr.pt, val manifests |

### notebooks_AIO/

| Notebook | Paper component | Original/Adapted | Run? | Depends on |
|----------|----------------|-----------------|------|------------|
| 01_lhp_beit3_train | Paper LHP (BEiT-3) | Paper-faithful | No | aio_repo/, train/ |
| 02_uit_train | Paper UIT | Paper-faithful | No | aio_repo/, train/ |
| 03_lhp_beit3_inference | Paper LHP inference | Paper-faithful | No | checkpoint-best.pth |
| 04_blip2_inference | Paper BLIP-2 | Paper-faithful | No | gallery, queries |
| 05_clip_inference | Paper CLIP | Paper-faithful | No | gallery, queries |
| 06_uit_ensemble_submit | Paper Algorithm 1+2 | Paper-faithful | No | all score matrices, uit checkpoint |

---

## 13. Copy Candidates Table

### Group A: PE-G14 Baseline (reproduce current score)

| Source path | Proposed destination in HUI fork | Category | Notes |
|------------|----------------------------------|----------|-------|
| `zero_shot/zero_shot_pe_g14.ipynb` | `notebooks/zero_shot/zero_shot_pe_g14.ipynb` | **MUST COPY** | Reproduce baseline first |
| `zero_shot/aic_colab_utils.py` | `notebooks/aic_colab_utils.py` | **MUST COPY** | Single canonical copy |

### Group B: AIC26 Utility Scripts

| Source path | Proposed destination in HUI fork | Category | Notes |
|------------|----------------------------------|----------|-------|
| `zero_shot/aic_colab_utils.py` | `notebooks/aic_colab_utils.py` | **MUST COPY** | Canonical — use zero_shot version |
| `notebooks_AIO/aio_paper_utils.py` | `notebooks/aio_paper_utils.py` | **MUST COPY** | Update AIO_REPO_GIT_URL to your fork |

### Group C: Answer Validation Logic

| Source path | Proposed destination in HUI fork | Category | Notes |
|------------|----------------------------------|----------|-------|
| Cell 11 of `zero_shot_pe_g14.ipynb` | Extract to `scripts/validate_answer.py` | **SHOULD COPY** | 1978 rows × 10 unique gallery IDs check |

### Group D: Data Preparation / Query Conversion Logic

| Source path | Proposed destination in HUI fork | Category | Notes |
|------------|----------------------------------|----------|-------|
| `notebooks_upgrade/00_manifest_qc.ipynb` | `notebooks/upgrade/00_manifest_qc.ipynb` | **MUST COPY** | Required before any training |
| `aio_paper_utils.py::stage_paper_layout()` | (already in aio_paper_utils.py) | **MUST COPY** | Symlink farm for paper repo layout |
| `aio_paper_utils.py::generate_pair_jsonl()` | (already in aio_paper_utils.py) | **MUST COPY** | Parquet → JSONL for training |

### Group E: Notebooks Worth Preserving

| Source path | Proposed destination in HUI fork | Category | Notes |
|------------|----------------------------------|----------|-------|
| `notebooks_upgrade/01a_pe_g14_features.ipynb` | `notebooks/upgrade/01a_pe_g14_features.ipynb` | **MUST COPY** | |
| `notebooks_upgrade/02_uit_train.ipynb` | `notebooks/upgrade/02_uit_train.ipynb` | **MUST COPY** | Fix `/home/bao` fallback path |
| `notebooks_upgrade/03_lhp_peg14_train.ipynb` | `notebooks/upgrade/03_lhp_peg14_train.ipynb` | **MUST COPY** | Verify cell 8 proj_head before run |
| `notebooks_upgrade/04_uit_inference.ipynb` | `notebooks/upgrade/04_uit_inference.ipynb` | **MUST COPY** | |
| `notebooks_upgrade/05_blip2_inference.ipynb` | `notebooks/upgrade/05_blip2_inference.ipynb` | **MUST COPY** | |
| `notebooks_upgrade/06_clip_inference.ipynb` | `notebooks/upgrade/06_clip_inference.ipynb` | **MUST COPY** | |
| `notebooks_upgrade/07_pe_g14_scores.ipynb` | `notebooks/upgrade/07_pe_g14_scores.ipynb` | **MUST COPY** | |
| `notebooks_upgrade/08_kreciprocal_rerank.ipynb` | `notebooks/upgrade/08_kreciprocal_rerank.ipynb` | **MUST COPY** | |
| `notebooks_upgrade/09_adaptive_ensemble_submit.ipynb` | `notebooks/upgrade/09_adaptive_ensemble_submit.ipynb` | **MUST COPY** | |
| `notebooks_upgrade/01b_vitpose_features.ipynb` | `notebooks/upgrade/01b_vitpose_features.ipynb` | **SHOULD COPY** | Optional |
| `notebooks_upgrade/README.md` | `notebooks/upgrade/README.md` | **MUST COPY** | |
| `notebooks_AIO/01_lhp_beit3_train.ipynb` | `notebooks/aio/01_lhp_beit3_train.ipynb` | **SHOULD COPY** | Ablation baseline |
| `notebooks_AIO/02_uit_train.ipynb` | `notebooks/aio/02_uit_train.ipynb` | **SHOULD COPY** | |
| `notebooks_AIO/03_lhp_beit3_inference.ipynb` | `notebooks/aio/03_lhp_beit3_inference.ipynb` | **SHOULD COPY** | |
| `notebooks_AIO/04_blip2_inference.ipynb` | `notebooks/aio/04_blip2_inference.ipynb` | **SHOULD COPY** | |
| `notebooks_AIO/05_clip_inference.ipynb` | `notebooks/aio/05_clip_inference.ipynb` | **SHOULD COPY** | |
| `notebooks_AIO/06_uit_ensemble_submit.ipynb` | `notebooks/aio/06_uit_ensemble_submit.ipynb` | **SHOULD COPY** | |
| `notebooks_AIO/README.md` | `notebooks/aio/README.md` | **SHOULD COPY** | |

### Group F: Documentation

| Source path | Proposed destination in HUI fork | Category | Notes |
|------------|----------------------------------|----------|-------|
| `ARCHITECTURE.md` | `docs/ARCHITECTURE_teammate.md` | **SHOULD COPY** | Rename to avoid clash with HUI's own docs |
| `Document/AIO_paper.pdf` | `docs/AIO_paper.pdf` | **SHOULD COPY** | Keep offline copy |
| `Document/AIC 2026 Track 4_...pdf` | `docs/strategic_roadmap.pdf` | **OPTIONAL** | |
| `Document/*.docx` | `docs/` | **OPTIONAL** | |

### Group G: Baseline answer.txt

| Source path | Proposed destination in HUI fork | Category | Notes |
|------------|----------------------------------|----------|-------|
| `Document/answer.txt` | `baselines/answer_zeroshot_pe_g14.txt` | **SHOULD COPY** | Reference for improvement comparison |

### Group H: Generated files — DO NOT COPY

None exist in the repo (no .npz, .pt, .pth, .parquet). They live only on teammate's Drive.

### Group I: Credentials — DO NOT COPY

| File | Reason |
|------|--------|
| `rclone.txt` | Live Google Drive OAuth refresh token. Must be revoked. Never commit credentials. |
| `token_kaggle.txt` | Live Kaggle API token. Must be revoked. Never commit. |

---

## 14. Do-Not-Copy List

| File/Folder | Why not to copy |
|------------|----------------|
| `rclone.txt` | **Live Google OAuth refresh_token** — permanent until revoked. Copying propagates the exposure. Revoke first, then generate a new token if needed. |
| `token_kaggle.txt` | **Live Kaggle API token** — allows API abuse under teammate's account. Revoke at kaggle.com. |
| `notebooks_AIO/__pycache__/aic_colab_utils.cpython-314.pyc` | Python bytecode cache — CPython 3.14 specific, machine-generated, not portable. Will be regenerated on import. |
| `Hybrid-Unified-and-Iterative-.../` | Empty directory. Contains nothing. |
| `.git/` | Git history contains committed credentials in plaintext at commits `5783ad7f` and `8318cf7c`. Starting fresh git history in your fork is safer. |
| `.gitignore` | Inadequate — only ignores 2 paths. Write a new comprehensive one for your fork. |
| `REPO_HANDOFF_SUMMARY.md` | Superseded by this audit. Teammate's self-assessment rather than independent verification. |
| `Dataset_Structure/image_dataset_structure_on_drive.png` | Useful only if you're using the same Drive layout as the teammate. |

### Recommended .gitignore for new repo

```gitignore
# Credentials — NEVER commit
rclone.txt
rclone.conf
token_kaggle.txt
kaggle.json
*.key
*.secret
.env

# Generated files — live on Drive only
*.npz
*.pt
*.pth
*.parquet
*.pyc
__pycache__/
*.log
*.tmp

# Large binary outputs
submission_*/
features/
checkpoints/
sims_score/
output/

# Dataset (should be on Drive or Kaggle)
Dataset_Structure/aic2026_data
data/PAB/
raw/
```

---

## 15. Recommended Safe Copy Order

### Phase 1: Safety first (do before anything else)

1. **REVOKE `rclone.txt` token** — Google Account → Security → Third-party apps
2. **REVOKE `token_kaggle.txt` token** — kaggle.com → Account → API → Expire
3. Generate new credentials for yourself separately, store outside git

### Phase 2: Copy core infrastructure (no path conflicts with HUI repo)

4. Copy `zero_shot/aic_colab_utils.py` → `notebooks/aic_colab_utils.py`
5. Copy `notebooks_AIO/aio_paper_utils.py` → `notebooks/aio_paper_utils.py`
   - **Edit**: change `AIO_REPO_GIT_URL` (line 146) to your HUI fork URL
6. Create new `.gitignore` using the template above

### Phase 3: Copy and verify the working baseline

7. Copy `zero_shot/zero_shot_pe_g14.ipynb` → `notebooks/zero_shot/zero_shot_pe_g14.ipynb`
8. Copy `Document/answer.txt` → `baselines/answer_zeroshot_pe_g14.txt`
9. **Test**: run `zero_shot_pe_g14.ipynb` end-to-end on Colab A100 → verify output matches baseline answer.txt

### Phase 4: Copy upgrade pipeline (10 notebooks)

10. Copy all `notebooks_upgrade/` notebooks to `notebooks/upgrade/`
11. **Edit `02_uit_train.ipynb` cell 2**: replace `/home/bao/Documents/AIC2026/Hybrid-Unified-.../` with your HUI fork path
12. Copy `notebooks_upgrade/README.md`

### Phase 5: Copy AIO baseline (6 notebooks)

13. Copy all `notebooks_AIO/*.ipynb` to `notebooks/aio/`
14. Copy `notebooks_AIO/README.md`

### Phase 6: Copy documentation

15. Copy `ARCHITECTURE.md` → `docs/ARCHITECTURE_teammate.md`
16. Copy `Document/AIO_paper.pdf` → `docs/AIO_paper.pdf`

### Phase 7: Before first run of any upgrade notebook

17. Verify `03_lhp_peg14_train.ipynb` cell 8 loads `proj_head.pt` at inference (confirmed present, but read it again after copying to new context)
18. Verify `04_uit_inference.ipynb` implements full Algorithm 1 with top-256 + cross-encoder ITM rerank (not just ITC cosine)
19. Verify `09_adaptive_ensemble_submit.ipynb` applies per-query z-normalization before score fusion

### Phase 8: Before any training run

20. Run `00_manifest_qc.ipynb` first to generate parquet manifests
21. Run `01a_pe_g14_features.ipynb` to encode gallery+queries
22. Then proceed sequentially: 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09

---

*End of TEAMMATE_REPO_COPY_AUDIT.md*
*Files inspected: 3 commits, 17 notebooks, 3 Python utilities, 2 READMEs, 1 ARCHITECTURE.md*
*Credentials found: 2 live tokens in rclone.txt and token_kaggle.txt — revoke immediately*
