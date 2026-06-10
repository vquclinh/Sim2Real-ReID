# AIC26 Copy Verification Report

**Date:** 2026-06-10
**Branch:** `clean-adaption`
**Auditor:** Claude Code (automated, read-only inspection)

---

## Summary Verdict: PARTIAL PASS

The `aic26/` folder structure is mostly correct and no forbidden secrets were found committed.
However, **3 required files are missing**, there is a **critical import path mismatch** in the PE-G14
notebook, and `.gitignore` has one gap. Recommended actions are listed at the end.

| Check | Status |
|---|---|
| Expected files present | ⚠️ 5/8 found — 3 missing |
| Misplaced / duplicate files | ✅ None |
| Forbidden secrets committed | ✅ CLEAR |
| Generated / large files committed | ⚠️ `.pt` files in `sims_score/` and `uit/cmp/` (HUI originals — see §5) |
| `.gitignore` coverage | ⚠️ Missing `checkpoints/` entry |
| `aic26/README.md` exists | ❌ Missing |
| PE-G14 notebook import path | ❌ Will break at runtime — `aic_colab_utils` not on path |
| Baseline answer format | ✅ 1978 rows × 10 IDs, no extensions |
| `notebooks_upgrade/` contents | ✅ All 10 notebooks + README present |
| `aio_paper_utils.py` — repo URL | ⚠️ Points to original upstream, not your fork (probably correct — see §11) |

---

## Section 1 — Current `aic26/` Tree

```
aic26/
├── docs/
│   ├── ARCHITECTURE.md                ✅
│   ├── baseline_answer_pe_g14.txt     ✅
│   ├── notebooks_AIO_README.md        ✅ (extra, not in spec — acceptable)
│   └── TEAMMATE_REPO_COPY_AUDIT.md    ✅
│   ✗ AIO_paper.pdf                    ❌ MISSING
│   ✗ REPO_HANDOFF_SUMMARY.md         ❌ MISSING
├── notebooks_upgrade/
│   ├── 00_manifest_qc.ipynb           ✅
│   ├── 01a_pe_g14_features.ipynb      ✅
│   ├── 01b_vitpose_features.ipynb     ✅ (extra — see §12)
│   ├── 02_uit_train.ipynb             ✅
│   ├── 03_lhp_peg14_train.ipynb       ✅
│   ├── 04_uit_inference.ipynb         ✅
│   ├── 05_blip2_inference.ipynb       ✅
│   ├── 06_clip_inference.ipynb        ✅
│   ├── 07_pe_g14_scores.ipynb         ✅
│   ├── 08_kreciprocal_rerank.ipynb    ✅
│   ├── 09_adaptive_ensemble_submit.ipynb ✅
│   └── README.md                      ✅
├── pe_g14/
│   └── zero_shot_pe_g14.ipynb         ✅
├── utils/
│   ├── aic_colab_utils.py             ✅
│   └── aio_paper_utils.py             ✅ (optional — present)
✗ README.md                            ❌ MISSING
```

Root-level files outside `aic26/` that are relevant:

```
./sims_score/score_beit3_reproduce.pt    (original HUI)
./sims_score/score_blip2_reproduce.pt    (original HUI)
./sims_score/score_clip_reproduce.pt     (original HUI)
./uit/cmp/score_beit3.pt                 (original HUI)
./uit/cmp/score_blip2.pt                 (original HUI)
./uit/cmp/score_clip.pt                  (original HUI)
./FORK_REPO_AUDIT.md                     (staged — new)
./document/AIO_paper.pdf                 (DELETED — AD in git status)
```

---

## Section 2 — Expected Files Checklist

| File | Status | Notes |
|---|---|---|
| `aic26/README.md` | ❌ MISSING | Must be created |
| `aic26/pe_g14/zero_shot_pe_g14.ipynb` | ✅ Found | |
| `aic26/utils/aic_colab_utils.py` | ✅ Found | |
| `aic26/docs/ARCHITECTURE.md` | ✅ Found | |
| `aic26/docs/REPO_HANDOFF_SUMMARY.md` | ❌ MISSING | Was it renamed or not copied? |
| `aic26/docs/TEAMMATE_REPO_COPY_AUDIT.md` | ✅ Found | |
| `aic26/docs/baseline_answer_pe_g14.txt` | ✅ Found | |
| `aic26/docs/AIO_paper.pdf` | ❌ MISSING | `document/AIO_paper.pdf` was staged (A) then deleted (D) — never landed in `aic26/docs/` |

---

## Section 3 — Misplaced / Duplicate Files

**Result: No duplicates or misplaced files found.**

Each searched filename appears in exactly one location:

| Filename | Path Found | Expected? | Risk |
|---|---|---|---|
| `zero_shot_pe_g14.ipynb` | `aic26/pe_g14/` only | ✅ Correct location | None |
| `aic_colab_utils.py` | `aic26/utils/` only | ✅ Correct location | See §8 for import issue |
| `ARCHITECTURE.md` | `aic26/docs/` only | ✅ Correct location | None |
| `TEAMMATE_REPO_COPY_AUDIT.md` | `aic26/docs/` only | ✅ Correct location | None |
| `AIO_paper.pdf` | Not found | ❌ Should be in `aic26/docs/` | Was deleted from `document/` |
| `aio_paper_utils.py` | `aic26/utils/` only | ✅ Optional, correct | None |
| `REPO_HANDOFF_SUMMARY.md` | Not found anywhere | ❌ Missing | Check teammate repo |
| `answer.txt` | Not found (note: `baseline_answer_pe_g14.txt` is the renamed version) | ✅ Renamed correctly | None |

---

## Section 4 — Forbidden Secrets Check

**Result: CLEAR — no forbidden files found anywhere in the repository.**

| File | Status |
|---|---|
| `rclone.txt` | ✅ NOT found |
| `token_kaggle.txt` | ✅ NOT found |
| `kaggle.json` | ✅ NOT found |
| `.env` | ✅ NOT found |

Note: `aic_colab_utils.py` references `rclone.txt` and `kaggle.json` **as runtime search paths**
(e.g., `drive_root/rclone.txt`, `~/.kaggle/kaggle.json`). These are path strings in code logic,
not embedded credentials. The actual files are not present. Safe to commit.

---

## Section 5 — Generated / Large Files Check

### Files requiring attention

| Path | Classification | Action |
|---|---|---|
| `sims_score/score_beit3_reproduce.pt` | Original HUI reproduce file | Covered by `*.pt` in `.gitignore` — but already tracked by git if committed previously; check `git ls-files` |
| `sims_score/score_blip2_reproduce.pt` | Original HUI reproduce file | Same |
| `sims_score/score_clip_reproduce.pt` | Original HUI reproduce file | Same |
| `uit/cmp/score_beit3.pt` | Original HUI reproduce file | Same |
| `uit/cmp/score_blip2.pt` | Original HUI reproduce file | Same |
| `uit/cmp/score_clip.pt` | Original HUI reproduce file | Same |
| `lhp_2/speechlm/.../speechlmp_base_cfg.pt` | Original HUI submodule config | Uncertain — could be a model weight bundled with the repo; verify |
| `lhp_2/beit3/__pycache__/*.pyc` | Original HUI generated bytecache | Not from teammate repo |
| `uit/cmp/__pycache__/*.pyc` | Original HUI generated bytecache | Not from teammate repo |
| `lhp_2/*/data/` directories | Original HUI Python package dirs (not datasets) | These are Python module paths, not data directories; do not confuse with actual dataset folders |
| `lhp_2/decoding/GAD/data` | Original HUI submodule data dir | Uncertain — inspect if actual files are inside |
| `uit/cmp/output` | Original HUI empty output dir | Empty directory; no concern |

**Key finding:** All `.pt` files are in original HUI code directories (`sims_score/`, `uit/cmp/`),
not in `aic26/`. None appear to be copied from the teammate repo by mistake.

---

## Section 6 — `.gitignore` Check

`.gitignore` was found at repo root (currently **untracked** — must be staged with `git add .gitignore`).

### Coverage audit

| Pattern | Present? |
|---|---|
| `data/` | ✅ |
| `checkpoint/` | ✅ |
| `checkpoints/` | ❌ **MISSING** |
| `output/` | ✅ |
| `outputs/` | ✅ |
| `*.pt` | ✅ |
| `*.pth` | ✅ |
| `*.npz` | ✅ |
| `*.parquet` | ✅ |
| `*.zip` | ✅ |
| `*.log` | ✅ |
| `__pycache__/` | ✅ |
| `*.pyc` | ✅ |
| `rclone.txt` | ✅ |
| `token_kaggle.txt` | ✅ |
| `kaggle.json` | ✅ |
| `.env` | ✅ |

### Missing `.gitignore` entries

```
checkpoints/
```

Add this line to `.gitignore` — do not edit yet as per audit-only constraint.

**Important:** `.gitignore` itself is currently untracked (`??` in git status).
It must be staged (`git add .gitignore`) before the next commit.

---

## Section 7 — `aic26/README.md` Check

**Result: FILE MISSING.**

`aic26/README.md` does not exist. This is the top-level entry point for anyone entering the
`aic26/` folder. Without it, the folder structure is not self-explanatory.

### Suggested sections to add (do not create yet — audit only)

```markdown
# AIC 2026 — Track 4 Adaptation Code

What this folder is:
- All AIC/ECCV 2026 Track 4 competition code lives here.
- The rest of this repo is the original HUI (Hybrid-Unified-and-Iterative) paper code.

What NOT to commit:
- *.pt, *.pth, *.npz, *.parquet, *.zip — model weights and score matrices
- rclone.txt, token_kaggle.txt, kaggle.json, .env — credentials

Structure:
- pe_g14/   — PE-G14 zero-shot baseline notebook (run this first)
- utils/    — shared Colab infrastructure (aic_colab_utils.py, aio_paper_utils.py)
- docs/     — architecture doc, teammate audit, baseline submission file
- notebooks_upgrade/ — experimental upgrade pipeline (NOT yet executed)
```

---

## Section 8 — PE-G14 Notebook Import/Path Check

**File:** `aic26/pe_g14/zero_shot_pe_g14.ipynb`
**Cells:** 12 total (11 code cells)
**Outputs cleared:** ✅ Yes — all code cells have empty outputs

### Import analysis

Cell 1 bootstrap:

```python
NOTEBOOK_DIR = Path('.').resolve()
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from aic_colab_utils import (
    setup_aic2026_environment, select_a100_device, ...
)
```

**Problem:** The notebook prepends the **current working directory** (`.`) to `sys.path` and then
does a bare `from aic_colab_utils import ...`. This means it looks for `aic_colab_utils.py` in
whatever directory the notebook is **run from**.

- If run from `aic26/pe_g14/`: looks for `aic26/pe_g14/aic_colab_utils.py` — **does not exist** → ImportError
- If run from `aic26/utils/`: would find it, but that is an unusual run location for a notebook
- `aic_colab_utils.py` is currently at `aic26/utils/aic_colab_utils.py`

**Verdict: CRITICAL — import will fail when notebook is run from its own folder.**

### Hardcoded paths

| Cell | Path | Severity |
|---|---|---|
| Cell 1 | `/content/drive/MyDrive` (inside `setup_aic2026_environment()`) | LOW — inside utility function, Colab-expected default |
| Cell 11 | `/home/bao/Documents/AIC2026/Document/answer.txt` | LOW — guarded by `if sample.exists():`, fails silently on non-dev machines |

The `/home/bao` path is inside a **graceful comparison block** — it is a dev convenience check
that prints a diff vs. a local reference answer. It will simply be skipped when the path does
not exist. Not blocking, but flags that this cell was written on the teammate's machine.

### Resolution options (do not apply yet)

Option A: Place a thin wrapper `aic26/pe_g14/aic_colab_utils.py` that re-exports from
`../utils/aic_colab_utils.py`.

Option B: Change cell 1 to:
```python
sys.path.insert(0, str(Path('.').resolve().parent / 'utils'))
```

Option C: Run the notebook from `aic26/` root and adjust the `NOTEBOOK_DIR` path.

---

## Section 9 — `aic26/utils/aic_colab_utils.py` Check

**File:** `aic26/utils/aic_colab_utils.py` (1098 lines)

| Check | Status | Detail |
|---|---|---|
| File exists | ✅ Yes | |
| Embedded credential strings | ✅ CLEAR | No hardcoded tokens, passwords, or API keys |
| `rclone.txt` references | ✅ Safe | Runtime path search only (e.g. `drive_root/rclone.txt`) — file itself not present |
| Hardcoded Colab paths | ⚠️ Expected | `/content/drive/MyDrive` appears as default argument — standard for Colab utilities |
| `kaggle.json` references | ✅ Safe | Runtime path lookups only (reads from `~/.kaggle/kaggle.json` at runtime) |
| Safe to commit | ✅ Yes | No secrets, standard Colab infrastructure code |

---

## Section 10 — `aic26/docs/baseline_answer_pe_g14.txt` Check

| Check | Status |
|---|---|
| Non-empty lines | ✅ 1978 |
| First 3 lines: 10 space-separated IDs each | ✅ Confirmed |
| IDs include file extensions | ✅ None (IDs are alphanumeric strings only) |
| Sample IDs | `ZOVZW5GHWX3K7R2`, `OT6C4QNNMQFAF0J`, `LLWPRL2M86UM0TN` … |

Format is correct and matches the 1978-row × 10-ID submission format.

---

## Section 11 — `aic26/utils/aio_paper_utils.py` — Repo URL Check

**File:** `aic26/utils/aio_paper_utils.py` (412 lines)

```python
AIO_REPO_GIT_URL = (
    'https://github.com/AIVIETNAM-Hub/'
    'Hybrid-Unified-and-Iterative-A-Novel-Framework-for-Text-based-Person-Anomaly-Retrieval.git'
)
```

**This URL points to the original upstream AIO paper repo (AIVIETNAM-Hub org), NOT your fork.**

Your fork remote is: `https://github.com/vquclinh/Sim2Real-ReID`

**Verdict: Probably intentional and correct.** The function `clone_aio_repo()` clones the
paper repo for running paper-faithful experiments. Pointing to the canonical upstream means
you always get the reference implementation. If you intend to run modified paper code from
your fork instead, you would need to update this URL — but do not do so unless that is the
explicit intent.

Mark as: **uncertain — verify with teammate whether this should point to a fork or the original.**

---

## Section 12 — `aic26/notebooks_upgrade/` Check

**Status: Present. Marked as EXPERIMENTAL / COPIED FOR REFERENCE — not yet executed.**

| Expected Notebook | Found |
|---|---|
| `README.md` | ✅ |
| `00_manifest_qc.ipynb` | ✅ |
| `01a_pe_g14_features.ipynb` | ✅ |
| `02_uit_train.ipynb` | ✅ |
| `03_lhp_peg14_train.ipynb` | ✅ |
| `04_uit_inference.ipynb` | ✅ |
| `05_blip2_inference.ipynb` | ✅ |
| `06_clip_inference.ipynb` | ✅ |
| `07_pe_g14_scores.ipynb` | ✅ |
| `08_kreciprocal_rerank.ipynb` | ✅ |
| `09_adaptive_ensemble_submit.ipynb` | ✅ |

**Extra file (not in expected spec):**

| File | Notes |
|---|---|
| `01b_vitpose_features.ipynb` | Extra notebook copied from teammate repo — acceptable, mark as experimental |

All 10 expected notebooks are present. The extra `01b_vitpose_features.ipynb` is not
confusing as it follows the naming convention and is clearly experimental.

---

## Section 13 — Git Status

```
A  FORK_REPO_AUDIT.md          ← staged, new file
 D HUI_REPO_AUDIT.md           ← unstaged deletion
AD document/AIO_paper.pdf      ← staged addition + unstaged deletion (net: never committed)
?? .gitignore                  ← UNTRACKED — must be staged
?? aic26/                      ← UNTRACKED — entire aic26/ folder must be staged
```

**Key finding:** `document/AIO_paper.pdf` shows `AD` — it was added to staging (`A`) but then
deleted from disk (`D`). The file never made it into `aic26/docs/AIO_paper.pdf` and is now gone.
This explains why `aic26/docs/AIO_paper.pdf` is missing.

**`.gitignore` is untracked.** It will not protect any files until it is committed. Run
`git add .gitignore` before the next commit.

---

## Section 14 — Recommended Next Actions

Ordered by priority. Do not run training or inference.

### CRITICAL (must fix before committing `aic26/`)

1. **Fix notebook import path** — `zero_shot_pe_g14.ipynb` cell 1 does a bare
   `from aic_colab_utils import ...` using the current working directory. Since
   `aic_colab_utils.py` is in `aic26/utils/`, not `aic26/pe_g14/`, the import will fail.
   Choose one of the three options listed in §8.

2. **Stage `.gitignore`** — `git add .gitignore`. Without this, all the ignore rules are
   inoperative. Do this before staging `aic26/`.

### HIGH (missing required files)

3. **Restore `aic26/docs/AIO_paper.pdf`** — The PDF was deleted from `document/AIO_paper.pdf`
   and was never placed in `aic26/docs/`. Re-copy it from the teammate's repo or your local
   backup.

4. **Create `aic26/README.md`** — Folder has no entry-point documentation. Use the suggested
   sections from §7.

5. **Locate `aic26/docs/REPO_HANDOFF_SUMMARY.md`** — Check whether this file was renamed to
   `TEAMMATE_REPO_COPY_AUDIT.md` (already present) or is a separate document that was not
   copied. Confirm with teammate.

### MEDIUM (before next commit)

6. **Add `checkpoints/` to `.gitignore`** — Currently missing from `.gitignore` (see §6).

7. **Clarify `AIO_REPO_GIT_URL`** in `aio_paper_utils.py` — Verify with teammate whether
   this should remain pointing to `AIVIETNAM-Hub` upstream or be changed to your fork (§11).

8. **Unstage `document/AIO_paper.pdf` deletion** — Run `git restore --staged document/AIO_paper.pdf`
   if you want to clean up the staging area, or leave it if `document/` is being removed.

### LOW (housekeeping, not blocking)

9. **`/home/bao` path in notebook cell 11** — Non-blocking (guarded by `if sample.exists():`),
   but should eventually be updated to use `aic26/docs/baseline_answer_pe_g14.txt` for
   cross-developer compatibility.

10. **Verify `.pt` files in `sims_score/` and `uit/cmp/`** — Run `git ls-files | grep '\.pt'`
    to check if these were previously committed. If they are already tracked, add them to a
    `.gitignore`-then-untrack cycle (`git rm --cached`). Do not do this yet.

---

*End of report. Nothing was modified.*
