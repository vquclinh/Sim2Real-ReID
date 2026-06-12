# AIC26 BEiT-3 + LHP Smoke Training Prep (Stage 1)

> **Status:** Launcher ready; the smoke run itself must be executed in Colab with a GPU
> after the BEiT-3 init checkpoint is in place. Nothing was trained or downloaded here.
> **References:** `AIO_paper.pdf` §3.1, `AIC26_AUTHOR_ALIGNED_TRAINING_READY_CHECK.md`,
> `AIC26_PAB_AUTHOR_LAYOUT_PREP_REPORT.md` (prepared layout verified: 225/225 sampled
> paths resolve, annotations converted to `.webp`).

---

## Exact Author Entrypoint

`lhp_2/beit3/run_beit3_finetuning.py` — must run with `lhp_2/beit3` as the working
directory (it imports sibling modules `utils`, `datasets`, `engine_for_finetuning`).

- Model is **hard-coded** to `beit3_large_patch16_384_retrieval` at
  `run_beit3_finetuning.py:259` — the `--model` argument is ignored for task `356`.
- Task `356` wiring: `datasets.py:740` maps `"356"` to `BaseDataset`;
  `datasets.py:879` builds only a train loader — `create_downstream_dataset` returns
  `(train_loader, None)`, so **no validation data is needed** and the per-epoch eval
  block is skipped (`data_loader_val is None`).
- `engine_for_finetuning.py:452`: task `356` uses `RetrievalHandler`.
- LHP transform is internal to `BaseDataset._get_image` (`datasets.py:83`):
  `torch.normal(mean=0.5, std=0.166667) > 0.5` → random-resized crop (local) else
  full-image resize (global). The `build_transform(...)` object passed in is ignored.
- Single process with no `RANK`/`WORLD_SIZE` env → `utils.init_distributed_mode`
  prints "Not using distributed mode" and runs single-GPU (`utils.py:306-309`).

## Exact CLI Arguments Found (from `run_beit3_finetuning.py` argparse)

| Purpose | Argument | Default | Smoke value |
|---|---|---|---|
| Task id | `--task` (required, choices include `'356'`) | — | `356` |
| Image size | `--input_size` | 224 | `384` |
| Init checkpoint | `--finetune` | `''` (random init if empty) | `$BEIT3_CKPT` |
| Checkpoint key match | `--model_key` | `model\|module` | default |
| Tokenizer | `--sentencepiece_model` (**required**) | — | `$BEIT3_SPM` |
| Data path | `--data_path` | imagenet path | `$SMOKE_ROOT` |
| Output dir | `--output_dir` | `''` | `$OUT_DIR` |
| Log dir | `--log_dir` | None | `$OUT_DIR/log` |
| Batch size | `--batch_size` | 64 | `2` |
| Epochs | `--epochs` | 20 | `1` |
| Grad accumulation | `--update_freq` | 1 | default |
| Optimizer | `--opt` | `adamw` | `adamw` |
| LR | `--lr` | 5e-4 | `1e-5` (paper) |
| Min LR | `--min_lr` | 1e-6 | default |
| Warmup | `--warmup_epochs` / `--warmup_steps` | 5 / -1 | **`--warmup_epochs 0`** (see Risks) |
| Save checkpoints | `--save_ckpt` / `--save_ckpt_freq` | True / 5 | `--save_ckpt --save_ckpt_freq 1` |
| Resume | `--auto_resume` / `--no_auto_resume` | True | `--no_auto_resume` |
| Workers | `--num_workers` | 10 | `2` |
| Seed | `--seed` | 0 | `0` |
| Max BPE tokens | `--num_max_bpe_tokens` | 64 | default |

**There is no max-steps / debug-steps option.** The argparse has nothing like
`--max_steps`; steps per epoch = `len(dataset) // (batch_size * update_freq * world_size)`
(`run_beit3_finetuning.py:291`). Smoke duration is therefore controlled by a tiny
**data subset**, not by a step limit — that is what `make_pab_smoke_subset.py` is for.

The scheduler is cosine with warmup, built unconditionally
(`run_beit3_finetuning.py:338` → `utils.cosine_scheduler`, `utils.py:420`).
Checkpoints are written as `checkpoint-{epoch}.pth` under `--output_dir`
(`utils.py:448`).

## Required Checkpoint / Tokenizer Files

| Asset | Path (default) | Status |
|---|---|---|
| BEiT-3 public init | `$BEIT3_CKPT` = `/content/drive/MyDrive/aic2026_data/checkpoints/beit3_large_patch16_384_coco_retrieval.pth` | **must be placed manually** — the launcher fails with a clear message; it never downloads. (Open question from the readiness audit: the author may have used `beit3_large_itc_patch16_224.pth` instead, per the official retrieval recipe — either public file works mechanically via `--finetune`.) |
| Sentencepiece tokenizer | `$BEIT3_SPM`, defaults to the **repo copy** `lhp_2/beit3/beit3.spm` | present in repo |
| Prepared PAB layout | `$PAB_AUTHOR_ROOT` = `/content/aic_local/pab_author_layout` | verified (Stage 0B: 225/225 resolved, `.webp`-converted annotations) |
| Python env | `requirements_lhp.txt`: `timm==0.4.12`, `torchscale==0.2.0`, `sentencepiece`, `deepspeed==0.4.0` (not needed — deepspeed only imports with `--enable_deepspeed`), `protobuf==3.20.0`, etc. | install in Colab before running |

## Smoke Command

One call (the launcher auto-creates the smoke subset on first run):

```bash
PAB_AUTHOR_ROOT=/content/aic_local/pab_author_layout \
BEIT3_CKPT=/content/drive/MyDrive/aic2026_data/checkpoints/beit3_large_patch16_384_coco_retrieval.pth \
OUT_DIR=/content/drive/MyDrive/aic2026_data/output/beit3_lhp_smoke \
bash aic26/scripts/run_beit3_lhp_smoke_train.sh
```

Which executes, from `lhp_2/beit3/`:

```bash
python run_beit3_finetuning.py \
    --task 356 --input_size 384 \
    --batch_size 2 --epochs 1 --warmup_epochs 0 \
    --lr 1e-5 --min_lr 1e-6 --opt adamw \
    --sentencepiece_model "$BEIT3_SPM" \
    --finetune "$BEIT3_CKPT" \
    --data_path "$SMOKE_ROOT" \
    --output_dir "$OUT_DIR" --log_dir "$OUT_DIR/log" \
    --num_workers 2 --save_ckpt --save_ckpt_freq 1 --no_auto_resume --seed 0
```

The smoke subset (`aic26/tools/make_pab_smoke_subset.py`) keeps the mandatory
75-file `pair_{0..74}.json` pattern: `pair_0`/`pair_1` carry 2 records each, the
other 73 are 0-record files (the loader accepts them — it just logs "Load 0
image-text pairs"), and `train` is a single symlink to the prepared image tree.
4 records / batch 2 → 2 optimizer steps. `DRY_RUN=1` prints the command without
running anything.

## Expected Colab Cells

1. Mount Drive; clone repo branch `clean-adaption`.
2. `pip install -r requirements_lhp.txt` (or minimally: `torch torchvision
   timm==0.4.12 torchscale==0.2.0 sentencepiece tensorboardX einops ftfy
   opencv-python pyarrow protobuf==3.20.0 scipy`).
3. Run `aic26/tools/prepare_pab_author_layout.py` (Stage 0B) if
   `/content/aic_local/pab_author_layout` is not present in this runtime.
4. Place/verify the BEiT-3 init checkpoint at `$BEIT3_CKPT` (one-time manual
   download to Drive; outside this task's scope).
5. `bash aic26/scripts/run_beit3_lhp_smoke_train.sh` (optionally `DRY_RUN=1` first).
6. Inspect: finite `loss` in the log, `checkpoint-0.pth` in `$OUT_DIR`, and that
   the checkpoint loads back (`torch.load(..., map_location='cpu')['model']`).

## Known Risks

1. **Warmup assert (handled):** `utils.cosine_scheduler` asserts
   `len(schedule) == epochs * niter_per_ep`; with the default `--warmup_epochs 5`
   and a tiny subset, warmup steps exceed total steps and the assert fails.
   The launcher pins `--warmup_epochs 0`.
2. **Steps-per-epoch zero:** if the subset had fewer records than
   `batch_size * update_freq`, `num_training_steps_per_epoch` is 0 and the
   scheduler gets an empty schedule. Defaults (4 records, batch 2) avoid this;
   don't raise `BATCH_SIZE` above 2 without adding records
   (`--records-per-file` / `--files-with-data`).
3. **`.webp` decoding:** images are loaded by torchvision's `default_loader`
   (PIL); Colab's Pillow decodes `.webp` natively. If PIL lacks webp support in
   a custom env, the loader raises on the first batch — that is a real
   environment failure, not a layout problem.
4. **Random init silently degrading the test:** `--finetune ''` would run from
   random weights; the launcher refuses to start without `$BEIT3_CKPT` instead.
5. **746M model at 384² on small GPUs:** batch 2 fits on a T4/L4-class GPU for a
   forward/backward, but is close on T4 (~15 GB); if OOM, set `BATCH_SIZE=1`.
6. **Old-checkpoint resume:** `--auto_resume` defaults on and would silently
   resume from a previous smoke checkpoint in `$OUT_DIR`; the launcher passes
   `--no_auto_resume`.
7. **`timm==0.4.12` pin:** newer timm breaks the author code
   (`RandomResizedCropAndInterpolation` import path); keep the LHP environment
   separate from any UIT (timm 0.6.13) environment.
8. **No eval during smoke:** task 356 has no val loader by design; success is
   "loss finite + checkpoint written + loads back", not a metric.

## Next Step

After the smoke run passes in Colab: Stage 2 — full author-aligned fine-tuning
(3 epochs, effective batch 184 via `--batch_size`/`--update_freq`, LR 1e-5,
cosine with default warmup restored, full 75-file prepared layout as
`--data_path`), per `AIC26_AUTHOR_ALIGNED_TRAINING_READY_CHECK.md` Stage 2.
