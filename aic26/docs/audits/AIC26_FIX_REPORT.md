# AIC26 Fix Report

**Date:** 2026-06-10
**Branch:** `clean-adaption`
**Based on:** `AIC26_COPY_VERIFICATION.md`

---

## Changes Made

### 1. Created `aic26/README.md`

**Why:** The `aic26/` folder had no entry-point documentation. Without a README, a new
contributor has no way to know what the folder is for, which files are safe to run, and
what must never be committed.

**What it covers:**
- Explains that `aic26/` contains AIC/ECCV 2026 Track 4 adaptation code, separate from
  the original HUI paper code.
- Lists original HUI paths (`lhp_2/`, `uit/`, `blip/`, `clip_infer.py`, etc.) and notes
  they should not be modified.
- Documents the sub-folder layout: `pe_g14/`, `utils/`, `docs/`, `notebooks_upgrade/`.
- Notes that `notebooks_upgrade/` is experimental and was copied for reference only.
- Lists all file types that must not be committed (datasets, weights, credentials, etc.).
- Links to `docs/ARCHITECTURE.md` and `docs/TEAMMATE_REPO_COPY_AUDIT.md`.

---

### 2. Fixed import path in `aic26/pe_g14/zero_shot_pe_g14.ipynb` (cell 1)

**Why:** The notebook's bootstrap cell added only the current working directory (`.`)
to `sys.path` and then did a bare `from aic_colab_utils import ...`. When the notebook
is run from `aic26/pe_g14/`, Python looks for `aic_colab_utils.py` in `aic26/pe_g14/`
— where it does not exist. The utility was placed at `aic26/utils/aic_colab_utils.py`.
This would have caused an `ImportError` at the very first execution.

**Change (cell 1 bootstrap block):**

Before:
```python
NOTEBOOK_DIR = Path('.').resolve()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))
```

After:
```python
NOTEBOOK_DIR = Path('.').resolve()
UTILS_DIR = NOTEBOOK_DIR.parent / 'utils'  # aic26/utils/
for _p in (str(UTILS_DIR), str(NOTEBOOK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

**How it works:**
- `NOTEBOOK_DIR` resolves to wherever `.` is when the kernel starts — normally `aic26/pe_g14/`.
- `UTILS_DIR` is `aic26/pe_g14/../utils` = `aic26/utils/`, which is where `aic_colab_utils.py` lives.
- Both paths are inserted (utilities first, notebook dir second) so that `aic_colab_utils`
  from `utils/` takes priority over any accidental same-folder file.
- The fix is robust to Colab's working directory conventions: `Path('.').resolve()` is the
  standard way to get the kernel working directory inside a Jupyter cell (notebooks do not
  define `__file__`).

**Will the import work now?** Yes — assuming the notebook is run from `aic26/pe_g14/`
(the standard Colab behavior when opening a notebook). `UTILS_DIR` will resolve to
`aic26/utils/` and `aic_colab_utils.py` will be importable.

---

### 3. Added `checkpoints/` to `.gitignore`

**Why:** The audit identified that `.gitignore` covered `checkpoint/` (singular) but not
`checkpoints/` (plural). Training scripts commonly write to either directory name.
Leaving `checkpoints/` uncovered would allow accidentally staging large model weight files.

**Change:** Added `checkpoints/` on the line immediately after `checkpoint/`.

---

## Remaining Missing Files

These were explicitly deferred per task instructions — do not fix yet.

| File | Status | Next action |
|---|---|---|
| `aic26/docs/AIO_paper.pdf` | Missing — `document/AIO_paper.pdf` was staged then deleted | Re-copy from teammate repo or local backup |
| `aic26/docs/REPO_HANDOFF_SUMMARY.md` | Missing | Confirm with teammate whether it was renamed to `TEAMMATE_REPO_COPY_AUDIT.md` or is a separate document |

---

## PE-G14 Notebook Import — Final Verdict

**The import should now work correctly when the notebook is run from `aic26/pe_g14/`.**

| Item | Status |
|---|---|
| `aic_colab_utils.py` present at `aic26/utils/` | ✅ |
| `sys.path` now includes `aic26/utils/` before import | ✅ |
| Bare `from aic_colab_utils import ...` will resolve | ✅ |
| Hardcoded `/home/bao` path in cell 11 | ⚠️ Non-blocking — guarded by `if sample.exists():` |
| Outputs cleared | ✅ |

---

## Git Status After Fixes

```
A  FORK_REPO_AUDIT.md          ← staged, new file
 D HUI_REPO_AUDIT.md           ← unstaged deletion
AD document/AIO_paper.pdf      ← staged-then-deleted (net: file gone, not in aic26/docs/ yet)
?? .gitignore                  ← UNTRACKED — must be staged before next commit
?? AIC26_COPY_VERIFICATION.md  ← UNTRACKED — audit report
?? aic26/                      ← UNTRACKED — entire aic26/ folder
```

The `diff --stat` shows only the pre-existing staged changes (`HUI_REPO_AUDIT.md` deletion,
`document/AIO_paper.pdf` deletion). All new files (`aic26/`, `.gitignore`, audit reports) are
untracked and do not appear in the diff yet — they will appear after `git add`.

---

## Recommended Next Commit Sequence

```bash
# 1. Stage .gitignore first (so it protects everything staged after)
git add .gitignore

# 2. Stage aic26/ and the audit/fix reports
git add aic26/ AIC26_COPY_VERIFICATION.md AIC26_FIX_REPORT.md

# 3. Stage FORK_REPO_AUDIT.md (already staged) and remove HUI_REPO_AUDIT.md
git add FORK_REPO_AUDIT.md
git rm HUI_REPO_AUDIT.md

# 4. Review what will be committed
git diff --staged --stat

# 5. Commit
git commit -m "aic26: initial competition adaptation folder with PE-G14 baseline"
```

Do not stage `document/AIO_paper.pdf` — it is deleted from disk and should be cleaned
from the index with `git restore --staged document/AIO_paper.pdf` if desired.
