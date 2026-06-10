# AIC 2026 — Track 4 Adaptation Code

This folder contains all AIC/ECCV 2026 Track 4 competition-specific adaptation code.
It is kept **separate** from the original HUI paper code so that the upstream baseline
remains intact and traceable.

---

## What lives where

### This repo — original HUI code (do not modify)

| Path | Description |
|---|---|
| `lhp_2/` | LHP-2 backbone and submodule code from the HUI paper |
| `uit/` | UIT (Unified Image-Text) model code from the HUI paper |
| `blip/` | BLIP-2 inference submodule |
| `clip_infer.py` | CLIP inference script |
| `blip2_infer.py` | BLIP-2 inference script |
| `beit3_infer.py` | BEiT-3 inference script |
| `sims_score/` | Precomputed similarity matrices from HUI reproduce run |
| `predictions/` | Precomputed prediction files from HUI reproduce run |

### `aic26/` — competition adaptation (this folder)

```
aic26/
├── pe_g14/             # Verified PE-Core-G14-448 zero-shot baseline
├── utils/              # Colab/Drive/Kaggle helper utilities
├── notebooks_upgrade/  # Experimental notebooks copied from teammate repo
└── docs/
    ├── references/     # Long-term reference docs, paper, baseline answer
    └── audits/         # Setup/copy/audit reports for traceability
```

---

## Quick start

The verified baseline is in [`pe_g14/zero_shot_pe_g14.ipynb`](pe_g14/zero_shot_pe_g14.ipynb).

Run it from `aic26/pe_g14/` on a Colab A100 instance. It will:
1. Mount Google Drive and set up the dataset.
2. Load `PE-Core-G14-448` (1.8B, zero-shot, no training).
3. Encode 36,773 gallery images and 1,978 query captions.
4. Produce `answer.txt` (1978 rows × 10 gallery IDs).

The shared infrastructure is in [`utils/aic_colab_utils.py`](utils/aic_colab_utils.py).
The notebook imports it via `../utils` — do not move the utility file.

---

## `notebooks_upgrade/` — experimental, not yet executed

These notebooks implement the upgrade pipeline (PE-G14 features, UIT training, LHP training,
k-reciprocal re-ranking, adaptive ensemble). They were copied from the teammate repo for
reference. **Do not treat them as runnable without verification.**
Each notebook must be reviewed before use.

---

## What NOT to commit

```
data/               # datasets
checkpoint/         # training checkpoints
checkpoints/        # same
output/ outputs/    # inference outputs
*.pt *.pth          # model weights
*.npz               # score matrices
*.parquet           # manifest/feature files
*.zip               # submission archives
*.log               # training logs
rclone.txt          # rclone config (contains GDrive credentials)
token_kaggle.txt    # Kaggle token
kaggle.json         # Kaggle credentials
.env                # any environment secrets
```

These patterns are already covered by `.gitignore` at the repo root.

---

## Docs

### References (`docs/references/`)

| File | Description |
|---|---|
| `docs/references/ARCHITECTURE.md` | Full pipeline architecture for the 10-notebook upgrade system |
| `docs/references/baseline_answer_pe_g14.txt` | Verified PE-G14 zero-shot baseline submission (1978 rows) |
| `docs/references/AIO_paper.pdf` | AIO (WWW 2025) paper — foundation of the HUI pipeline |
| `docs/references/notebooks_AIO_README.md` | Notes on the AIO-faithful notebook pipeline |
| `docs/references/REPO_HANDOFF_SUMMARY.md` | Handoff summary from teammate repo |

### Audits (`docs/audits/`)

| File | Description |
|---|---|
| `docs/audits/TEAMMATE_REPO_COPY_AUDIT.md` | Pre-copy security and content audit of teammate repo |
| `docs/audits/FORK_REPO_AUDIT.md` | Audit of this fork after initial setup |
| `docs/audits/AIC26_COPY_VERIFICATION.md` | Verification report after copying files from teammate repo |
| `docs/audits/AIC26_FIX_REPORT.md` | Report of fixes applied after copy verification |
