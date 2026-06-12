#!/usr/bin/env bash
# Stage 1 — BEiT-3 + LHP tiny smoke training launcher (author code unmodified).
#
# Tests the author dataloader + one tiny train pass on the prepared PAB layout.
# This is NOT full training: tiny batch, 1 epoch over a ~4-record subset.
#
# Environment variables (all overridable):
#   PAB_AUTHOR_ROOT  prepared author layout (default /content/aic_local/pab_author_layout)
#   SMOKE_ROOT       smoke subset root      (default /content/aic_local/pab_author_layout_smoke_subset)
#                    auto-created from PAB_AUTHOR_ROOT via make_pab_smoke_subset.py if absent
#   BEIT3_CKPT       public BEiT-3 init checkpoint (REQUIRED, not downloaded by this script)
#                    default /content/drive/MyDrive/aic2026_data/checkpoints/beit3_large_patch16_384_coco_retrieval.pth
#   BEIT3_SPM        sentencepiece tokenizer (default: the copy committed in this repo)
#   OUT_DIR          output dir for checkpoint-*.pth and logs
#                    default /content/drive/MyDrive/aic2026_data/output/beit3_lhp_smoke
#   BATCH_SIZE       default 2
#   EPOCHS           default 1
#   NUM_WORKERS      default 2
#   DRY_RUN=1        print the exact training command and exit without running it
#
# Author-aligned fixed settings (AIO_paper.pdf §3.1): task 356, input 384,
# AdamW, cosine schedule (built into run_beit3_finetuning.py), LR 1e-5.
# Full training later uses: 3 epochs, batch 184 — NOT this script.
#
# Smoke-specific required deviations (see AIC26_BEIT3_LHP_SMOKE_TRAIN_PREP.md):
#   --warmup_epochs 0   (default 5 fails utils.cosine_scheduler's length assert
#                        when warmup steps exceed the tiny total step count)
#   --no_auto_resume    (don't silently resume from old smoke checkpoints)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PAB_AUTHOR_ROOT="${PAB_AUTHOR_ROOT:-/content/aic_local/pab_author_layout}"
SMOKE_ROOT="${SMOKE_ROOT:-/content/aic_local/pab_author_layout_smoke_subset}"
BEIT3_CKPT="${BEIT3_CKPT:-/content/drive/MyDrive/aic2026_data/checkpoints/beit3_large_patch16_384_coco_retrieval.pth}"
BEIT3_SPM="${BEIT3_SPM:-$REPO_ROOT/lhp_2/beit3/beit3.spm}"
OUT_DIR="${OUT_DIR:-/content/drive/MyDrive/aic2026_data/output/beit3_lhp_smoke}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# ---- preflight: required assets (nothing is downloaded here) ----------------
[ -d "$PAB_AUTHOR_ROOT" ] || fail "PAB_AUTHOR_ROOT missing: $PAB_AUTHOR_ROOT
Run aic26/tools/prepare_pab_author_layout.py first (Stage 0B)."
[ -f "$PAB_AUTHOR_ROOT/annotation/train/pair_0.json" ] || \
    fail "PAB_AUTHOR_ROOT has no annotation/train/pair_0.json: $PAB_AUTHOR_ROOT"

[ -f "$BEIT3_CKPT" ] || fail "BEIT3_CKPT missing: $BEIT3_CKPT
This script does not download checkpoints. Place the public BEiT-3 init there
(e.g. beit3_large_patch16_384_coco_retrieval.pth or beit3_large_itc_patch16_224.pth
from the unilm/beit3 release), or set BEIT3_CKPT to its location."

[ -f "$BEIT3_SPM" ] || fail "BEIT3_SPM missing: $BEIT3_SPM
The repo copy lives at lhp_2/beit3/beit3.spm; set BEIT3_SPM if you moved it."

# ---- smoke subset (auto-created, local disk only) ---------------------------
if [ ! -f "$SMOKE_ROOT/annotation/train/pair_74.json" ]; then
    echo "Smoke subset not found — creating it from $PAB_AUTHOR_ROOT ..."
    python "$REPO_ROOT/aic26/tools/make_pab_smoke_subset.py" \
        --source-root "$PAB_AUTHOR_ROOT" \
        --out-root "$SMOKE_ROOT"
fi

mkdir -p "$OUT_DIR"

# ---- the smoke command -------------------------------------------------------
# run_beit3_finetuning.py imports sibling modules (utils, datasets, ...), so it
# must run with lhp_2/beit3 as the working directory. Model is hard-coded to
# beit3_large_patch16_384_retrieval at run_beit3_finetuning.py:259 (--model is
# ignored for task 356). Single process => author code runs non-distributed.
CMD=(python run_beit3_finetuning.py
    --task 356
    --input_size 384
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --warmup_epochs 0
    --lr 1e-5
    --min_lr 1e-6
    --opt adamw
    --sentencepiece_model "$BEIT3_SPM"
    --finetune "$BEIT3_CKPT"
    --data_path "$SMOKE_ROOT"
    --output_dir "$OUT_DIR"
    --log_dir "$OUT_DIR/log"
    --num_workers "$NUM_WORKERS"
    --save_ckpt
    --save_ckpt_freq 1
    --no_auto_resume
    --seed 0
)

echo "Smoke root : $SMOKE_ROOT"
echo "Checkpoint : $BEIT3_CKPT"
echo "Tokenizer  : $BEIT3_SPM"
echo "Output dir : $OUT_DIR"
echo "Command    : (cd lhp_2/beit3 && ${CMD[*]})"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 — not executing."
    exit 0
fi

cd "$REPO_ROOT/lhp_2/beit3"
"${CMD[@]}"

echo
echo "Smoke training finished. Checkpoints in $OUT_DIR:"
ls -la "$OUT_DIR"/checkpoint-*.pth 2>/dev/null || echo "(no checkpoint written?)"
echo "SUCCESS CRITERION: loss logged as finite, checkpoint-0.pth (or -$((EPOCHS-1))) present."
