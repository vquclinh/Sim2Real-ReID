# AIC26 BEiT-3 + LHP Smoke Training Result (Stage 1)

> **Status: PASS — Stage 1 fully closed.** Executed in Colab on 2026-06-12 (reported
> by the user from the live run). The checkpoint was written **and verified to load
> back** (see Checkpoint Load Verification below). Stage 1's purpose was to validate
> the author dataloader and one tiny train pass on the prepared competition layout —
> not model quality.
> **Prep audit:** `AIC26_BEIT3_LHP_SMOKE_TRAIN_PREP.md`.

---

## Exact Command Used

Launcher:

```bash
PAB_AUTHOR_ROOT=/content/aic_local/pab_author_layout \
BEIT3_CKPT=/content/drive/MyDrive/aic2026_data/checkpoints/beit3_large_patch16_384_coco_retrieval.pth \
OUT_DIR=/content/drive/MyDrive/aic2026_data/output/beit3_lhp_smoke \
bash aic26/scripts/run_beit3_lhp_smoke_train.sh
```

Which executed, from `lhp_2/beit3/`:

```bash
python run_beit3_finetuning.py \
    --task 356 --input_size 384 \
    --batch_size 2 --epochs 1 --warmup_epochs 0 \
    --lr 1e-5 --min_lr 1e-6 --opt adamw \
    --sentencepiece_model /content/Sim2Real-ReID/lhp_2/beit3/beit3.spm \
    --finetune /content/drive/MyDrive/aic2026_data/checkpoints/beit3_large_patch16_384_coco_retrieval.pth \
    --data_path /content/aic_local/pab_author_layout_smoke_subset \
    --output_dir /content/drive/MyDrive/aic2026_data/output/beit3_lhp_smoke \
    --log_dir /content/drive/MyDrive/aic2026_data/output/beit3_lhp_smoke/log \
    --num_workers 2 --save_ckpt --save_ckpt_freq 1 --no_auto_resume --seed 0
```

## Paths

| Item | Path |
|---|---|
| Prepared root (Stage 0B, `.webp`-converted) | `/content/aic_local/pab_author_layout` |
| Smoke subset (75-file pattern, 4 records) | `/content/aic_local/pab_author_layout_smoke_subset` |
| BEiT-3 init checkpoint | `/content/drive/MyDrive/aic2026_data/checkpoints/beit3_large_patch16_384_coco_retrieval.pth` |
| Tokenizer | `/content/Sim2Real-ReID/lhp_2/beit3/beit3.spm` (repo copy) |
| Output checkpoint | `/content/drive/MyDrive/aic2026_data/output/beit3_lhp_smoke/checkpoint-0.pth` |

## Environment Versions

Not captured during the run. To record them next session, run in the same Colab env:

```python
import torch, torchvision, timm, torchscale, sentencepiece, PIL, numpy, google.protobuf
for m in (torch, torchvision, timm, torchscale, PIL, numpy, google.protobuf):
    print(m.__name__, getattr(m, "__version__", "?"))
print("sentencepiece", sentencepiece.__version__)
print("cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0))
```

Expected pins from `requirements_lhp.txt`: `timm==0.4.12`, `torchscale==0.2.0`,
`torchmetrics==0.7.3`, `protobuf==3.20.0`.

## Run Facts

| Fact | Value |
|---|---|
| Records in subset | 4 (pair_0 + pair_1, 2 each; pair_2..74 empty) |
| Optimizer steps | 2 (batch 2, 1 epoch, update_freq 1) |
| Logged loss | finite (contrastive loss; absolute value not meaningful — see Warnings) |
| Dataloader errors | none |
| `.webp` decode errors | none (PIL decoded the converted-annotation paths correctly) |
| Checkpoint written | yes — `checkpoint-0.pth` |
| Checkpoint loads back | yes — verified in Colab (see below) |
| Peak GPU memory | ~13.1 GB (batch 2 @ 384², 746M-param model) |

## Checkpoint Load Verification

Verified in Colab against
`/content/drive/MyDrive/aic2026_data/output/beit3_lhp_smoke/checkpoint-0.pth`:

```text
exists: True
size GB: 7.547
top-level keys: ['model', 'optimizer', 'epoch', 'scaler', 'args']
has model: True
model keys: 974
epoch: 0
OK: checkpoint can be loaded
```

The structure matches what `utils.save_model` writes (`utils.py:448-456`:
model + optimizer + epoch + scaler + args), and the 7.5 GB size confirms the
Stage 2 estimate that each full checkpoint (fp32 model + AdamW moments + scaler)
is in the 8–9 GB class.

**PyTorch load note:** Colab ships a newer PyTorch whose `torch.load` defaults to
`weights_only=True`, which rejects the pickled `args` Namespace inside the
checkpoint. Loading therefore required:

```python
torch.load(path, map_location="cpu", weights_only=False)
```

This is acceptable here because the checkpoint was produced by **our own** smoke
training run — `weights_only=False` should only ever be used on checkpoints whose
provenance is trusted. The same flag will be needed when loading Stage 2
checkpoints and any author/unilm `.pth` files under new PyTorch versions.

## Success Criteria vs Result

| Criterion (from Stage 1 plan) | Result |
|---|---|
| Author dataloader yields LHP-transformed batches from the prepared layout | PASS |
| Forward/backward completes with finite loss | PASS |
| Checkpoint writes to `--output_dir` | PASS (`checkpoint-0.pth`) |
| Checkpoint loads back without key errors | PASS (974 model keys, epoch 0) |
| No metric target | n/a by design |

**Stage 1 is fully closed** — all exit criteria met, including the load-back check.

This closes the Stage 1 exit criteria from
`AIC26_AUTHOR_ALIGNED_TRAINING_READY_CHECK.md`: the `.webp` adapter layout, the
75-file annotation pattern, the LHP transform path, the tokenizer, and the public
COCO-retrieval init all work together end to end with **unmodified author code**.

## Warnings

- **The loss value is not meaningful.** 2 optimizer steps over 4 records validates
  mechanics only; it says nothing about convergence or retrieval quality.
- **This does not validate model quality.** No evaluation ran (task 356 has no val
  loader by design); Stage 2 + Stage 3 score generation are where quality appears.
- **Dependency pins can fight Colab.** `protobuf==3.20.0` and the pinned
  `transformers`/`torchmetrics` from `requirements_lhp.txt` may downgrade packages
  other preinstalled Colab libraries expect — keep the LHP environment in its own
  runtime (and separate from any UIT/timm-0.6.13 environment), and re-install after
  runtime resets.
- **Peak memory headroom is thin on T4 (~15 GB) at batch 2.** Stage 2 defaults drop
  to batch 1 + gradient accumulation for this reason.

## Next Step

Stage 2 — full author-aligned fine-tuning: 3 epochs over all 1,013,606 records,
effective batch 184 via gradient accumulation, LR 1e-5, cosine schedule. Launcher:
`aic26/scripts/run_beit3_lhp_full_train.sh`; prep notes:
`AIC26_BEIT3_LHP_FULL_TRAIN_PREP.md`.
