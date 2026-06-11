# notebooks_AIO — paper-faithful baseline for AIC 2026 Track 4

6 notebooks that reproduce the **exact logic** of the AIO paper (Hybrid, Unified and Iterative — WWW 2025, Tien-Huy Nguyen et al., R@1 89.23 on PAB) on Colab A100 40GB, dataset mounted via Drive `aic2026_data/`.

> This baseline runs in parallel with `notebooks_upgrade/` (10 NBs containing 4 upgrades: PE-G14, LoRA, adaptive ensemble, reversed iterative order). Purpose: (1) reproduce the paper numbers for comparison, (2) isolate bugs from upgrades.

## Run order

| # | Notebook | Output | ETA A100 40GB |
|---|---|---|---|
| 01 | `01_lhp_beit3_train.ipynb` | `<drive>/output/lhp/checkpoint-best.pth` | ~12h (4 epoch) |
| 02 | `02_uit_train.ipynb` | `<drive>/output/uit/checkpoint_29.pth` | ~110h (30 epoch, multiple sessions) |
| 03 | `03_lhp_beit3_inference.ipynb` | `<drive>/sims_score/score_beit3.pt` | ~30 min |
| 04 | `04_blip2_inference.ipynb` | `<drive>/sims_score/score_blip2.pt` | ~1.5h |
| 05 | `05_clip_inference.ipynb` | `<drive>/sims_score/score_clip.pt` | ~25 min |
| 06 | `06_uit_ensemble_submit.ipynb` | `<drive>/submission/answer.zip` | ~45 min |

Notebooks 01-02 (training) run independently and can be parallelized across 2 Colab sessions. Notebooks 03-05 (inference) are also independent of each other. Notebook 06 waits for all of them.

## Assets to upload to Drive **once**

1. **Paper repo** — upload the folder `Hybrid-Unified-and-Iterative-A-Novel-Framework-for-Text-based-Person-Anomaly-Retrieval/` (~150MB code) as `<drive>/aic2026_data/aio_repo/`. Notebooks auto-rsync to local on each session.
2. **UIT init weights** (1 of 2 options):
   - **Option 1 (paper-recommended)**: download `pretrained.pth` from the Google Drive form in the paper README (`https://drive.google.com/file/d/1KffesfZD45kOQH2E4G31Sd3rbj9djD3d/view`) and upload to `<drive>/aic2026_data/aio_repo/checkpoint/pretrained.pth`.
   - **Option 2 (auto fallback)**: skip this step; notebook 02 will auto-download `swin_base_patch4_window7_224_22k.pth` + `bert-base-uncased`.
3. **BEiT-3 weights** — notebook 01 downloads automatically from the GitHub release (`addf400/files`).

## Hyperparameters (paper-faithful)

| Stage | Param | Value |
|---|---|---|
| LHP (BEiT-3-large) | epochs / batch / lr | 4 / 184 / 1e-5 |
| LHP | drop_path / wd / layer_decay | 0.16 / 0.05 / 0.85 |
| UIT (Swin-B + BERT) | epochs / batch / lr | 30 / 84 / 1e-4 |
| UIT loss | combine | `itc + itm + mlm + 0.1356·mim` |
| Algorithm 1 | k_test | 128 |
| Algorithm 2 weights | beit3 / blip2 / clip | 0.925 / 0.9 / 0.9 |

A100 40GB does not have enough VRAM for LHP batch 184 as-is — the notebook automatically falls back to `batch=92, update_freq=2` to maintain an effective batch size of 184. UIT 84 still fits with BF16 + gradient checkpointing.

## Output layout on Drive

```
<drive>/aic2026_data/
├── aio_repo/                         # paper repo
├── output/
│   ├── lhp/checkpoint-best.pth       # NB 01
│   └── uit/checkpoint_29.pth         # NB 02
├── sims_score/
│   ├── score_beit3.pt                # NB 03 — shape (1978, 36773)
│   ├── score_blip2.pt                # NB 04
│   └── score_clip.pt                 # NB 05
└── submission/
    ├── reproduce.txt                 # NB 06 — 1978 lines × 10 gallery IDs
    └── answer.zip
```

## Comparison with `notebooks_upgrade/`

| Component | upgrade | AIO baseline |
|---|---|---|
| LHP backbone | PE-G14 + LoRA r=16 | BEiT-3-large full FT |
| UIT epochs / lr | 22 / 1e-5 | 30 / 1e-4 |
| Ensemble | 4-model adaptive + reversed iterative order | 3-model fixed weights |
| k-reciprocal | yes | no |

See `Document/AIO_paper.pdf` + `ARCHITECTURE.md` at the repo root for more details.

## Shared files

- `aic_colab_utils.py` — copied directly from `notebooks_upgrade/`. Drive mount, Kaggle creds, manifest restore, BF16/Flash SDPA, async chunk sync, robust mkdir.
- `aio_paper_utils.py` — paper-specific helpers: `stage_paper_layout()` symlink farm, `clone_aio_repo()`, `ensure_lhp_assets()`, `ensure_uit_assets()`, `generate_pair_jsonl()`, `get_sorted_gallery_paths()`, `drive_sync_thread()`, `latest_checkpoint()`.
