# notebooks_AIO — paper-faithful baseline cho AIC 2026 Track 4

6 notebook reproduce **đúng logic** paper AIO (Hybrid, Unified and Iterative — WWW 2025, Tien-Huy Nguyen et al., R@1 89.23 trên PAB) trên Colab A100 40GB, dataset mount qua Drive `aic2026_data/`.

> Baseline này song song với `notebooks_upgrade/` (10 NB chứa 4 upgrade: PE-G14, LoRA, adaptive ensemble, đảo iterative order). Mục đích: (1) reproduce con số paper để đối chứng, (2) tách bug từ upgrade.

## Run order

| # | Notebook | Output | ETA A100 40GB |
|---|---|---|---|
| 01 | `01_lhp_beit3_train.ipynb` | `<drive>/output/lhp/checkpoint-best.pth` | ~12h (4 epoch) |
| 02 | `02_uit_train.ipynb` | `<drive>/output/uit/checkpoint_29.pth` | ~110h (30 epoch, nhiều session) |
| 03 | `03_lhp_beit3_inference.ipynb` | `<drive>/sims_score/score_beit3.pt` | ~30 min |
| 04 | `04_blip2_inference.ipynb` | `<drive>/sims_score/score_blip2.pt` | ~1.5h |
| 05 | `05_clip_inference.ipynb` | `<drive>/sims_score/score_clip.pt` | ~25 min |
| 06 | `06_uit_ensemble_submit.ipynb` | `<drive>/submission/answer.zip` | ~45 min |

Notebook 01-02 (training) chạy độc lập, có thể song song trên 2 Colab session. Notebook 03-05 (inference) cũng độc lập với nhau. Notebook 06 chờ toàn bộ.

## Assets cần upload Drive **một lần**

1. **Paper repo** — upload thư mục `Hybrid-Unified-and-Iterative-A-Novel-Framework-for-Text-based-Person-Anomaly-Retrieval/` (~150MB code) thành `<drive>/aic2026_data/aio_repo/`. Notebook tự rsync về local mỗi session.
2. **UIT init weights** (1 trong 2 option):
   - **Option 1 (paper-recommended)**: tải `pretrained.pth` từ form Google Drive trong README paper (`https://drive.google.com/file/d/1KffesfZD45kOQH2E4G31Sd3rbj9djD3d/view`) lên `<drive>/aic2026_data/aio_repo/checkpoint/pretrained.pth`.
   - **Option 2 (auto fallback)**: bỏ qua, notebook 02 sẽ auto-DL `swin_base_patch4_window7_224_22k.pth` + `bert-base-uncased`.
3. **BEiT-3 weights** — notebook 01 tự download từ GitHub release (`addf400/files`).

## Hyperparameters (paper-faithful)

| Stage | Param | Value |
|---|---|---|
| LHP (BEiT-3-large) | epochs / batch / lr | 4 / 184 / 1e-5 |
| LHP | drop_path / wd / layer_decay | 0.16 / 0.05 / 0.85 |
| UIT (Swin-B + BERT) | epochs / batch / lr | 30 / 84 / 1e-4 |
| UIT loss | combine | `itc + itm + mlm + 0.1356·mim` |
| Algorithm 1 | k_test | 128 |
| Algorithm 2 weights | beit3 / blip2 / clip | 0.925 / 0.9 / 0.9 |

A100 40GB không đủ VRAM cho batch 184 LHP nguyên si → notebook tự fallback `batch=92, update_freq=2` để giữ effective batch 184. UIT 84 vẫn fit được với BF16 + gradient checkpointing.

## Layout output trên Drive

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
    ├── reproduce.txt                 # NB 06 — 1978 dòng × 10 gallery IDs
    └── answer.zip
```

## So với `notebooks_upgrade/`

| Bộ phận | upgrade | AIO baseline |
|---|---|---|
| LHP backbone | PE-G14 + LoRA r=16 | BEiT-3-large full FT |
| UIT epochs / lr | 22 / 1e-5 | 30 / 1e-4 |
| Ensemble | 4-model adaptive + iterative đảo order | 3-model fixed weights |
| k-reciprocal | có | không |

Xem `Document/AIO_paper.pdf` + `ARCHITECTURE.md` ở repo root để biết thêm chi tiết.

## File chia sẻ

- `aic_colab_utils.py` — copy nguyên từ `notebooks_upgrade/`. Drive mount, Kaggle creds, manifest restore, BF16/Flash SDPA, async chunk sync, robust mkdir.
- `aio_paper_utils.py` — helper paper-specific: `stage_paper_layout()` symlink farm, `clone_aio_repo()`, `ensure_lhp_assets()`, `ensure_uit_assets()`, `generate_pair_jsonl()`, `get_sorted_gallery_paths()`, `drive_sync_thread()`, `latest_checkpoint()`.
