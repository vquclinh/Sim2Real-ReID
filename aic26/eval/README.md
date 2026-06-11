# `aic26/eval/` — Shared Evaluation Utilities

---

## Overview

`pab_metrics.py` is the shared evaluator for all PAB local retrieval experiments.
Import it from any notebook or script that runs local evaluation on PAB data.

---

## Local evaluation vs. official submission

| | Local eval | Official submission |
|---|---|---|
| Ground truth | Public (`attr.json`) | Hidden (leaderboard server) |
| Outputs | `metrics.json`, `metrics.md`, `rankings_top10.jsonl`, `positive_ranks.jsonl`, `run_info.md` | `answer.txt`, `answer.zip` |
| Purpose | Measure model quality before submitting | Spend a leaderboard attempt |
| Notebook | `aic26/pipelines/local_eval/` | `aic26/pipelines/official_submission/` |

Always run local evaluation first. Only submit officially when local metrics are satisfactory.

---

## `pab_metrics.py` — API reference

### `compute_single_positive_metrics(positive_ranks, k_values=(1, 5, 10))`

Compute retrieval metrics for queries with exactly one positive image each.

**Metric definitions:**

| Metric | Definition |
|---|---|
| AP | `1 / rank` for found queries; `0` if positive not found |
| mAP | mean AP over all queries |
| R@K | fraction of queries where `positive_rank <= K` |
| mean_rank | arithmetic mean of positive ranks (found queries only) |
| median_rank | median of positive ranks (found queries only) |

**Inputs:**

- `positive_ranks` — list of 1-based ranks, `None` if positive not found

**Output keys:** `queries`, `found_any`, `mAP`, `R@1`, `R@5`, `R@10`, `mean_rank`, `median_rank`

---

### `find_positive_ranks(gallery_ids, ranked_indices, positive_images)`

Map each query's positive image to its 1-based rank in the sorted gallery.

- `ranked_indices` may be the full `(Q, G)` ranking or a top-K truncated array.
- Returns `None` for a query if the positive is not present in the ranked slice.

---

### `build_topk_records(query_ids, query_captions, positive_images, gallery_ids, ranked_indices, positive_ranks, topk=10)`

Build per-query inspection records for `rankings_top10.jsonl`.

Each record contains: `query_id`, `caption`, `positive_image`, `positive_rank`, `top10`.

---

### `write_json(path, obj)`

Write a JSON-serialisable object to an indented `.json` file.

---

### `write_jsonl(path, rows)`

Write a list of dicts to a `.jsonl` file (one JSON object per line).

---

### `write_metrics_markdown(path, run_id, metrics, extra=None)`

Write a markdown table summarising the metrics dict.
`extra` is an optional `{label: value}` dict for additional rows
(e.g. `{"Model": "PE-Core-G14-448", "Dataset": "PAB attr test"}`).

---

## Output files per run

Every local eval run saves the following files to a dated run folder under Google Drive:

```
/content/drive/MyDrive/aic2026_data/pab_original/runs/<run_id>/
    metrics.json            # machine-readable metrics dict
    metrics.md              # human-readable markdown table
    rankings_top10.jsonl    # per-query top-10 results + positive rank
    positive_ranks.jsonl    # per-query positive rank only (lightweight)
    run_info.md             # run metadata (date, model, method, results)
```

Example run ID: `local_001_pe_g14_attr_zero_shot`
