# AIC26 BEiT-3 + LHP Full Fine-Tuning Prep (Stage 2)

> **Status:** Launcher ready (`aic26/scripts/run_beit3_lhp_full_train.sh`); the run
> itself must be started deliberately in Colab — it is a many-hour job.
> **Prerequisites met:** Stage 1 smoke PASS (`AIC26_BEIT3_LHP_SMOKE_TRAIN_RESULT.md`)
> on the same layout, checkpoint, tokenizer, and code path.

---

## Exact Command

```bash
PAB_AUTHOR_ROOT=/content/aic_local/pab_author_layout \
BEIT3_CKPT=/content/drive/MyDrive/aic2026_data/checkpoints/beit3_large_patch16_384_coco_retrieval.pth \
OUT_DIR=/content/drive/MyDrive/aic2026_data/output/beit3_lhp_full \
bash aic26/scripts/run_beit3_lhp_full_train.sh
```

Which executes, from `lhp_2/beit3/`:

```bash
python run_beit3_finetuning.py \
    --task 356 --input_size 384 \
    --batch_size 1 --update_freq 184 --epochs 3 \
    --warmup_steps 500 \
    --lr 1e-5 --min_lr 1e-6 --opt adamw \
    --sentencepiece_model "$BEIT3_SPM" \
    --finetune "$BEIT3_CKPT" \
    --data_path "$PAB_AUTHOR_ROOT" \
    --output_dir "$OUT_DIR" --log_dir "$OUT_DIR/log" \
    --num_workers 2 --save_ckpt --save_ckpt_freq 1 --no_auto_resume --seed 0
```

Probe first without committing to the run:

```bash
DRY_RUN=1      bash aic26/scripts/run_beit3_lhp_full_train.sh   # print command only
SANITY_ONLY=1  bash aic26/scripts/run_beit3_lhp_full_train.sh   # env+data probe, no training
```

## Paper-Aligned vs Practical Settings

| Setting | Paper says (§3.1) | Launcher default | Note |
|---|---|---|---|
| Task / model | BEiT-3 + LHP | `--task 356`, model hard-coded `beit3_large_patch16_384_retrieval` | author code unmodified |
| Image size | 384×384 | 384 | match |
| Epochs | 3 | 3 | match |
| Batch size | 184 | **1 × 184 accumulation = 184 effective** | real batch 184 is impossible on Colab GPUs; gradient accumulation approximates it. Not bit-identical (BN-free ViT + contrastive loss makes accumulation a close approximation, but the in-batch negative pool per contrastive step is the *micro*-batch — see Risks #6) |
| Optimizer | AdamW | `--opt adamw` | match |
| Scheduler | cosine annealing | built-in `utils.cosine_scheduler` | match |
| LR | 1e-5 | `--lr 1e-5` | match |
| Warmup | *not specified in paper* | `--warmup_steps 500` | **our chosen value, documented here.** The author default `--warmup_epochs 5` crashes the scheduler's length assert when `epochs < 5` (`utils.py:441`), so an explicit value is mandatory |
| Checkpointing | n/a | every epoch (`--save_ckpt_freq 1`) → `checkpoint-0/1/2.pth` | |
| Seed | not specified | 0 (author CLI default) | |

Steps per optimizer update epoch: `1,013,606 // 184 ≈ 5,509`; total ≈ 16,527
updates (~3.04M forward/backward passes at batch 1).

## Estimated Risks

1. **Runtime length.** ~1.01M forward+backward passes per epoch at 384² on a
   746M-param model. Rough single-GPU throughput: T4 ~1–2 img/s (≥6 days/epoch —
   **not viable**), L4 ~3–5 img/s (~2.5–4 days/epoch — marginal), A100-40GB with
   `BATCH_SIZE=8..16, UPDATE_FREQ=23..12` ~20–40 img/s (~8–14 h/epoch — the
   realistic target). Plan GPU class accordingly; Colab sessions rarely survive
   a multi-day run.
2. **Drive output speed.** Each checkpoint (fp32 model + AdamW moments + scaler)
   is roughly 8–9 GB; 3 epochs ≈ 25 GB+. Writing to Drive can take many minutes
   per checkpoint and stalls training; consider `OUT_DIR` on local disk
   (`/content/aic_local/...`) with a manual copy of the final checkpoint to Drive.
3. **Checkpoint size / quota.** Verify ≥30 GB free on the destination before
   starting (`SANITY_ONLY=1` prints the init checkpoint size as a reference).
4. **OOM.** Smoke peaked at ~13.1 GB with batch 2 — Stage 2 defaults to batch 1
   (~half the activations) for T4/L4 headroom. On A100, raise `BATCH_SIZE` and
   lower `UPDATE_FREQ` keeping the product at 184.
5. **Session disconnect.** Likely on a job this long — see Resume Strategy.
6. **Contrastive batch semantics (accumulation caveat).** The in-batch negatives
   for the contrastive loss come from the *micro*-batch (size `BATCH_SIZE`), not
   the effective batch — with `BATCH_SIZE=1` the contrastive signal per step is
   degenerate. **This is the strongest argument for running Stage 2 on a GPU that
   permits `BATCH_SIZE` ≥ 8–16**, accepting accumulation only as a partial bridge.
   Document the chosen micro-batch with the run.
7. **Dataloader stalls from Drive-backed images.** Images stream through the
   `train/imgs_*` symlinks into Drive; with `NUM_WORKERS=2` expect I/O-bound
   epochs. Raising workers helps until Drive rate-limits.

## Resume Strategy

- The launcher **refuses to start** if `OUT_DIR` already contains
  `checkpoint-*.pth` — no silent resume, no accidental overwrite.
- To continue an interrupted run: `RESUME=1 bash aic26/scripts/run_beit3_lhp_full_train.sh`
  → passes `--auto_resume`, and the author's `utils.auto_load_model` picks the
  latest `checkpoint-*.pth` (model + optimizer + scaler + epoch) and continues
  from the next epoch.
- Resume granularity is **per epoch** — work within an unfinished epoch is lost.
  With multi-day epochs on slow GPUs, that is another reason to use an A100-class
  runtime where an epoch fits inside one session.

## What to Inspect After Epoch 1

1. `checkpoint-0.pth` exists in `OUT_DIR` and loads:
   `torch.load(..., map_location="cpu")["model"]` has no missing/unexpected keys
   against `beit3_large_patch16_384_retrieval`.
2. Loss trend in `OUT_DIR/log` (TensorBoard) — contrastive loss should decline
   clearly within the first few hundred updates and keep a downward trend; a flat
   curve at the random-init level suggests the `--finetune` load silently failed.
3. Wall-clock per epoch — extrapolate: if epoch 2–3 won't fit the remaining
   session budget, stop now and restart on a faster GPU with `RESUME=1`.
4. (Optional, cheap quality probe) Run `lhp_2/beit3/inference.py` with
   `checkpoint-0.pth` on the local PAB `attr.json` split and compare against the
   PE-G14 zero-shot baseline (mAP 0.8829) — a 1-epoch checkpoint already being in
   a sane range (R@1 well above random) confirms training is real.

## When to Stop

- **Loss becomes NaN/inf** (visible in the per-step log): stop immediately; do not
  resume on top of a NaN checkpoint. Restart from the last finite checkpoint with
  a lower LR (e.g. 5e-6) and report the deviation.
- **Repeated dataloader exceptions** (missing/corrupt `.webp`, Drive I/O errors):
  stop and re-run Stage 0B verification (`prepare_pab_author_layout.py`) plus
  `SANITY_ONLY=1`; do not let the run skip-and-continue silently.
- **OOM loops**: stop, halve `BATCH_SIZE` (or move to a larger GPU); do not enable
  gradient checkpointing flags without recording the change.
- **Epoch ETA exceeds session budget** (see inspection #3): stop after the epoch
  checkpoint, switch runtime, `RESUME=1`.

## Next Step After Stage 2

Stage 3 — deterministic BEiT-3 score generation with the final checkpoint
(sorted gallery + `query_ids.json`/`gallery_ids.json`), then local evaluation
against `attr.json` to compare with the PE-G14 zero-shot baseline (mAP 0.8829),
per `AIC26_AUTHOR_ALIGNED_TRAINING_READY_CHECK.md` Stage 3.
