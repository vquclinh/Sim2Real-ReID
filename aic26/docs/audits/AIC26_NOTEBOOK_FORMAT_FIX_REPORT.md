# AIC26 Notebook Format Fix Report

**Date:** 2026-06-11
**Branch:** `clean-adaption`
**Scope:** All notebooks under `aic26/` — format corruption fix + residual Vietnamese translation

---

## Summary Verdict: PASS

All 11 corrupted notebooks were fixed. Source lists are now valid line-list format.
All 12 notebooks load as valid JSON. Zero Vietnamese characters remain in any notebook.
First markdown cell of every notebook renders as a clean English heading.

---

## Background: What Was Corrupted

The Vietnamese→English translation workflow iterated over notebook `source` fields
character-by-character instead of line-by-line. This produced two corruption types:

### Type B (10 notebooks)
Each character became its own list item:

```json
"source": ["#", " ", "0", "1", " ", "—", " ", "M", "a", "n", "i", "f", "e", "s", "t", ...]
```

`''.join(src)` returns the correct string, so rendering appears correct in some editors
but VS Code's Jupyter renderer inserts spaces between items → `M a n i f e s t`.

**Fix:** `''.join(src)` → `splitlines(keepends=True)` → proper line list.

### Type A (1 notebook: `00_manifest_qc.ipynb` cells 0 and 7)
Each character got `\n` appended:

```json
"source": ["#\n", " \n", "0\n", "1\n", " \n", "—\n", " \n", "M\n", "a\n", "n\n", ...]
```

`''.join(src)` gives `"# \n0\n1\n \n—\n \nM\na\nn\n..."` — single `\n` per char,
so `splitlines(keepends=True)` returns the same corrupted list.

**Fix:** `''.join(s[0] for s in src)` (strip the appended `\n`) → `splitlines(keepends=True)`.

---

## Files Checked

| Notebook | Had Format Corruption | Had Residual Vietnamese | Action |
|---|---|---|---|
| `pe_g14/zero_shot_pe_g14.ipynb` | ❌ None | ✅ Yes (3 phrases) | Translation only |
| `notebooks_upgrade/00_manifest_qc.ipynb` | ✅ Yes — Type A (cells 0, 7) | ✅ Yes (2 phrases) | Format fix + translation |
| `notebooks_upgrade/01a_pe_g14_features.ipynb` | ✅ Yes — Type B (cells 0, 1, 4, 6) | ✅ Yes (9 phrases) | Format fix + translation |
| `notebooks_upgrade/01b_vitpose_features.ipynb` | ✅ Yes — Type B (cells 0, 1) | ✅ Yes (2 phrases) | Format fix + translation |
| `notebooks_upgrade/02_uit_train.ipynb` | ✅ Yes — Type B (9 cells) | ✅ Yes (4 phrases) | Format fix + translation |
| `notebooks_upgrade/03_lhp_peg14_train.ipynb` | ✅ Yes — Type B (10 cells) | ✅ Yes (4 phrases) | Format fix + translation |
| `notebooks_upgrade/04_uit_inference.ipynb` | ✅ Yes — Type B (11 cells) | ✅ Yes (9 phrases) | Format fix + translation |
| `notebooks_upgrade/05_blip2_inference.ipynb` | ✅ Yes — Type B (7 cells) | ✅ Yes (5 phrases) | Format fix + translation |
| `notebooks_upgrade/06_clip_inference.ipynb` | ✅ Yes — Type B (6 cells) | ✅ Yes (1 phrase) | Format fix + translation |
| `notebooks_upgrade/07_pe_g14_scores.ipynb` | ✅ Yes — Type B (2 cells) | ✅ Yes (6 phrases) | Format fix + translation |
| `notebooks_upgrade/08_kreciprocal_rerank.ipynb` | ✅ Yes — Type B (4 cells) | ✅ Yes (5 phrases) | Format fix + translation |
| `notebooks_upgrade/09_adaptive_ensemble_submit.ipynb` | ✅ Yes — Type B (10 cells) | ✅ Yes (8 phrases) | Format fix + translation |

**Total cells fixed (format):** 69 cells across 11 notebooks.
**Total Vietnamese phrases fixed (in notebooks):** 58 phrases across 12 notebooks.

---

## Before / After Examples

### Type A: `00_manifest_qc.ipynb` cell 0 (markdown)

**Before (837 list items, first 8 shown):**
```json
["#\n", " \n", "0\n", "0\n", " \n", "—\n", " \n", "M\n", ...]
```
**Rendered in VS Code:** `# 0 0 — M a n i f e s t Q C + V a l i d a t i o n S p l i t s`

**After (15 list items):**
```json
["# 00 — Manifest QC + Validation Splits\n", "\n", "**Run first.**  ...]
```
**Rendered:** `# 00 — Manifest QC + Validation Splits`

---

### Type A: `00_manifest_qc.ipynb` cell 7 (code)

**Before:** 3251 list items (one per character + `\n`)
**After:** 77 list items (one per source line, ending with `\n` except last)

---

### Type B: `01a_pe_g14_features.ipynb` cell 0 (markdown)

**Before (67 list items, first 6 shown):**
```json
["#", " ", "0", "1", "a", " ", "—", " ", "P", "E", "-", "C", "o", "r", "e", ...]
```
**Rendered in VS Code:** `# 0 1 a — P E - C o r e - G 1 4 - 4 4 8 F e a t u r e ...`

**After (14 list items):**
```json
["# 01a — PE-Core-G14-448 Feature Extraction (Colab A100 80GB high-RAM)\n", "\n", ...]
```
**Rendered:** `# 01a — PE-Core-G14-448 Feature Extraction (Colab A100 80GB high-RAM)`

---

## Vietnamese Translation Examples (Notebook-specific)

| Original | Translated |
|---|---|
| `6. Ghi \`answer.txt\`` | `6. Write \`answer.txt\`` |
| `proxy cho OOD test gallery` | `proxy for OOD test gallery` |
| `high-RAM cho phép 12 workers` | `high-RAM allows 12 workers` |
| `polymorphic theo USE_COMPILE` | `polymorphic based on USE_COMPILE` |
| `chọn strategy theo MIRROR_RAW_STRATEGY` | `select strategy based on MIRROR_RAW_STRATEGY` |
| `trong final score matrix bằng ITM probability` | `in the final score matrix with ITM probability` |
| `Run ITM rerank cho mọi query` | `Run ITM rerank for all queries` |
| `Top-200 candidates theo similarity` | `Top-200 candidates by similarity` |
| `bằng mutual top-k1 (q ∈ topk(g) AND g ∈ topk(q))` | `via mutual top-k1 (q ∈ topk(g) AND g ∈ topk(q))` |
| `Bootstrap qua \`setup_aic2026_environment()\`` | `Bootstrap via \`setup_aic2026_environment()\`` |
| `(updated trong cell-train)` | `(updated in the training cell)` |
| `val score files cho gate` | `val score files for the gate` |
| `(theo MODEL_SLOT order: ...)` | `(following MODEL_SLOT order: ...)` |

---

## Validation Results

### JSON validity (all 12 notebooks)

| Notebook | `json.load` result |
|---|---|
| `pe_g14/zero_shot_pe_g14.ipynb` | ✅ PASS |
| `notebooks_upgrade/00_manifest_qc.ipynb` | ✅ PASS |
| `notebooks_upgrade/01a_pe_g14_features.ipynb` | ✅ PASS |
| `notebooks_upgrade/01b_vitpose_features.ipynb` | ✅ PASS |
| `notebooks_upgrade/02_uit_train.ipynb` | ✅ PASS |
| `notebooks_upgrade/03_lhp_peg14_train.ipynb` | ✅ PASS |
| `notebooks_upgrade/04_uit_inference.ipynb` | ✅ PASS |
| `notebooks_upgrade/05_blip2_inference.ipynb` | ✅ PASS |
| `notebooks_upgrade/06_clip_inference.ipynb` | ✅ PASS |
| `notebooks_upgrade/07_pe_g14_scores.ipynb` | ✅ PASS |
| `notebooks_upgrade/08_kreciprocal_rerank.ipynb` | ✅ PASS |
| `notebooks_upgrade/09_adaptive_ensemble_submit.ipynb` | ✅ PASS |

### Source format check (all cells across all notebooks)

No character-split patterns detected (average item length > 2 for all non-trivial cells).

Maximum source list size per notebook (after fix):

| Notebook | Max items (any cell) |
|---|---|
| `zero_shot_pe_g14.ipynb` | 85 |
| `00_manifest_qc.ipynb` | 78 (was 3251) |
| `01a_pe_g14_features.ipynb` | 148 |
| `01b_vitpose_features.ipynb` | 209 |
| `02_uit_train.ipynb` | 75 |
| `03_lhp_peg14_train.ipynb` | 133 |
| `04_uit_inference.ipynb` | 73 |
| `05_blip2_inference.ipynb` | 59 |
| `06_clip_inference.ipynb` | 39 |
| `07_pe_g14_scores.ipynb` | 50 |
| `08_kreciprocal_rerank.ipynb` | 69 |
| `09_adaptive_ensemble_submit.ipynb` | 165 |

### First markdown cell (rendering verification)

| Notebook | First markdown heading |
|---|---|
| `zero_shot_pe_g14.ipynb` | `# Zero-shot retrieval — PE-Core-G14-448 only` |
| `00_manifest_qc.ipynb` | `# 00 — Manifest QC + Validation Splits` |
| `01a_pe_g14_features.ipynb` | `# 01a — PE-Core-G14-448 Feature Extraction (Colab A100 80GB high-RAM)` |
| `01b_vitpose_features.ipynb` | `# 01b — ViTPose++ Feature Extraction (Colab A100 edition)` |
| `02_uit_train.ipynb` | `# 02 — UIT Training (Paper-faithful, Colab A100 80GB high-RAM)` |
| `03_lhp_peg14_train.ipynb` | `# 03 — LHP with PE-Core-G14-448 (LoRA fine-tune, A100 80GB)` |
| `04_uit_inference.ipynb` | `# 04 — UIT Inference (LHP-guided Feature Selection + ITM Rerank)` |
| `05_blip2_inference.ipynb` | `# 05 — BLIP-2 ITM Inference (Round 2 of the iterative ensemble)` |
| `06_clip_inference.ipynb` | `# 06 — OpenAI CLIP ViT-L/14@336 Inference (Round 3 of the iterative ensemble)` |
| `07_pe_g14_scores.ipynb` | `# 07 — PE-G14 Score Matrix (Upgrade 1: Round 4 of the iterative ensemble)` |
| `08_kreciprocal_rerank.ipynb` | `# 08 — k-Reciprocal Re-ranking (Extra Upgrade)` |
| `09_adaptive_ensemble_submit.ipynb` | `# 09 — Adaptive Ensemble + Submission (Upgrades 2 + 4)` |

### Vietnamese residual check

`grep` for full Vietnamese diacritical set + plain-Latin Vietnamese words (`cho`, `theo`, `trong`, `bằng`, `qua`, `Ghi`, `từ`, `nếu`, `chọn`, `lỗi`) across all 12 notebooks: **zero matches**.

---

## What Was Left Unchanged

- Execution counts and cell outputs: **not modified**
- Code logic, variable names, function names, file paths: **not modified**
- Notebook metadata (`kernelspec`, `language_info`): **not modified**
- Cells not affected by corruption or Vietnamese: **not touched**

---

## Note on Diff Size

Notebooks re-serialized with `json.dump(indent=1)` during the fix pass. If the original
file used a different indentation (e.g., 2 spaces), the diff will be large despite only
a handful of content lines changing. This is cosmetic only — content, outputs, and
execution counts are identical.

---

## Final Verdict: PASS

- 11 notebooks fixed (character-split source lists → proper line-list format)
- 1 notebook had no format corruption (`zero_shot_pe_g14.ipynb`)
- 58 Vietnamese phrases translated across all 12 notebooks
- All 12 notebooks load as valid JSON
- No character-split patterns remain (max source list: 209 items, normal for a 200-line cell)
- All first markdown headings render as clean English text
- Execution counts and outputs preserved throughout
