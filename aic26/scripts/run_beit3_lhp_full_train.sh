#!/usr/bin/env bash
# Stage 2 — BEiT-3 + LHP full author-aligned fine-tuning launcher (author code unmodified).
#
# Paper-aligned settings (AIO_paper.pdf §3.1): task 356, image 384, 3 epochs,
# AdamW, cosine schedule, LR 1e-5, effective batch 184. On Colab GPUs a real
# batch of 184 is impossible, so the effective batch is approximated with
# gradient accumulation: BATCH_SIZE * UPDATE_FREQ = 184.
#
# !! THIS IS A LONG JOB. One epoch is ~1.01M images at 384^2 through a 746M-param
# !! model — many hours to days depending on GPU (see the prep doc). Each saved
# !! checkpoint (model + AdamW state + scaler) is roughly 8-9 GB; 3 epochs can
# !! write ~25 GB+ to OUT_DIR. Make sure Drive has space and expect slow writes.
#
# Environment variables (all overridable):
#   PAB_AUTHOR_ROOT  full prepared layout (default /content/aic_local/pab_author_layout)
#   BEIT3_CKPT       public BEiT-3 init (REQUIRED; never downloaded here)
#   BEIT3_SPM        sentencepiece tokenizer (default repo copy)
#   OUT_DIR          checkpoint/log output (default .../output/beit3_lhp_full)
#   BATCH_SIZE       per-step batch (default 1 — conservative single-GPU)
#   UPDATE_FREQ      gradient accumulation (default 184; BATCH_SIZE*UPDATE_FREQ should be 184)
#   EPOCHS           default 3 (paper)
#   WARMUP_STEPS     default 500 — REQUIRED deviation: warmup is not specified in the
#                    paper, and the author default (--warmup_epochs 5) crashes
#                    utils.cosine_scheduler's length assert when epochs < 5.
#   NUM_WORKERS      default 2
#   SEED             default 0
#   RESUME=1         allow resuming from existing checkpoint-*.pth in OUT_DIR
#                    (without it, existing checkpoints make the launcher fail)
#   DRY_RUN=1        print the exact command and exit
#   SANITY_ONLY=1    run preflight + a no-training python sanity probe, then exit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PAB_AUTHOR_ROOT="${PAB_AUTHOR_ROOT:-/content/aic_local/pab_author_layout}"
BEIT3_CKPT="${BEIT3_CKPT:-/content/drive/MyDrive/aic2026_data/checkpoints/beit3_large_patch16_384_coco_retrieval.pth}"
BEIT3_SPM="${BEIT3_SPM:-$REPO_ROOT/lhp_2/beit3/beit3.spm}"
OUT_DIR="${OUT_DIR:-/content/drive/MyDrive/aic2026_data/output/beit3_lhp_full}"
BATCH_SIZE="${BATCH_SIZE:-1}"
UPDATE_FREQ="${UPDATE_FREQ:-184}"
EPOCHS="${EPOCHS:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-0}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# ---- preflight: required assets (nothing is downloaded here) ----------------
[ -d "$PAB_AUTHOR_ROOT" ] || fail "PAB_AUTHOR_ROOT missing: $PAB_AUTHOR_ROOT
Run aic26/tools/prepare_pab_author_layout.py first (Stage 0B)."
[ -f "$PAB_AUTHOR_ROOT/annotation/train/pair_0.json" ] || \
    fail "PAB_AUTHOR_ROOT has no annotation/train/pair_0.json: $PAB_AUTHOR_ROOT"
[ -f "$PAB_AUTHOR_ROOT/annotation/train/pair_74.json" ] || \
    fail "PAB_AUTHOR_ROOT is incomplete (pair_74.json missing) — the author loader needs all 75 files."
[ -f "$BEIT3_CKPT" ] || fail "BEIT3_CKPT missing: $BEIT3_CKPT
This script does not download checkpoints."
[ -f "$BEIT3_SPM" ] || fail "BEIT3_SPM missing: $BEIT3_SPM"

EFFECTIVE_BATCH=$(( BATCH_SIZE * UPDATE_FREQ ))
if [ "$EFFECTIVE_BATCH" -ne 184 ]; then
    echo "WARNING: BATCH_SIZE*UPDATE_FREQ = $EFFECTIVE_BATCH != 184 (paper batch)." >&2
    echo "         Document this deviation if you proceed." >&2
fi

# ---- resume policy -----------------------------------------------------------
RESUME_FLAG="--no_auto_resume"
if compgen -G "$OUT_DIR/checkpoint-*.pth" > /dev/null 2>&1; then
    if [ "${RESUME:-0}" = "1" ]; then
        RESUME_FLAG="--auto_resume"
        echo "RESUME=1 — will auto-resume from the latest checkpoint in $OUT_DIR:"
        ls -la "$OUT_DIR"/checkpoint-*.pth
    else
        fail "OUT_DIR already contains checkpoints:
$(ls "$OUT_DIR"/checkpoint-*.pth)
Set RESUME=1 to resume from the latest one, or point OUT_DIR elsewhere.
Refusing to start a fresh run on top of existing checkpoints."
    fi
fi

mkdir -p "$OUT_DIR"

# ---- the full-training command ------------------------------------------------
# Working dir must be lhp_2/beit3 (sibling imports). Model is hard-coded to
# beit3_large_patch16_384_retrieval (run_beit3_finetuning.py:259). Single
# process => author code runs non-distributed.
CMD=(python run_beit3_finetuning.py
    --task 356
    --input_size 384
    --batch_size "$BATCH_SIZE"
    --update_freq "$UPDATE_FREQ"
    --epochs "$EPOCHS"
    --warmup_steps "$WARMUP_STEPS"
    --lr 1e-5
    --min_lr 1e-6
    --opt adamw
    --sentencepiece_model "$BEIT3_SPM"
    --finetune "$BEIT3_CKPT"
    --data_path "$PAB_AUTHOR_ROOT"
    --output_dir "$OUT_DIR"
    --log_dir "$OUT_DIR/log"
    --num_workers "$NUM_WORKERS"
    --save_ckpt
    --save_ckpt_freq 1
    "$RESUME_FLAG"
    --seed "$SEED"
)

echo "Data root      : $PAB_AUTHOR_ROOT"
echo "Init checkpoint: $BEIT3_CKPT"
echo "Tokenizer      : $BEIT3_SPM"
echo "Output dir     : $OUT_DIR"
echo "Effective batch: $EFFECTIVE_BATCH (= $BATCH_SIZE x $UPDATE_FREQ), epochs $EPOCHS, warmup $WARMUP_STEPS steps"
echo "Command        : (cd lhp_2/beit3 && ${CMD[*]})"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 — not executing."
    exit 0
fi

if [ "${SANITY_ONLY:-0}" = "1" ]; then
    echo "SANITY_ONLY=1 — probing environment and data, NOT training."
    python - "$PAB_AUTHOR_ROOT" "$BEIT3_CKPT" <<'PY'
import json, sys
from pathlib import Path
root, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
rec = json.loads(open(root / "annotation/train/pair_0.json", encoding="utf-8").readline())
img = root / rec["image"]
print("first record image:", rec["image"], "->", "OK" if img.is_file() else "MISSING")
n74 = sum(1 for ln in open(root / "annotation/train/pair_74.json", encoding="utf-8") if ln.strip())
print("pair_74.json records:", n74)
print("checkpoint size (GB): %.2f" % (ckpt.stat().st_size / 1e9))
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu:", p.name, "| vram (GB): %.1f" % (p.total_memory / 1e9))
except ImportError:
    print("torch NOT importable — install requirements_lhp.txt first")
PY
    echo "SANITY_ONLY done — re-run without SANITY_ONLY to start full training."
    exit 0
fi

echo
echo "Starting FULL fine-tuning — this will run for many hours. Ctrl-C to abort."
cd "$REPO_ROOT/lhp_2/beit3"
"${CMD[@]}"

echo
echo "Full training finished. Checkpoints in $OUT_DIR:"
ls -la "$OUT_DIR"/checkpoint-*.pth 2>/dev/null || echo "(no checkpoint written?)"
