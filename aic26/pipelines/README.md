# `aic26/pipelines/` — Notebook Pipelines

This folder organises all runnable notebooks into two categories so their
purpose is unambiguous.

---

## `official_submission/`

Notebooks that produce **official leaderboard submission files** for hidden test
sets whose ground truth is not publicly available.

- Output format: `answer.txt` (1978 rows × 10 gallery IDs) + `answer.zip`
- These notebooks consume the competition's hidden query files and should only
  be run when you intend to use a leaderboard submission attempt.
- Do not run these notebooks casually — each run costs a leaderboard slot.

| Notebook | Description |
|---|---|
| [`official_submission/pe_g14_zero_shot/pe_g14_zero_shot_official_submission.ipynb`](official_submission/pe_g14_zero_shot/pe_g14_zero_shot_official_submission.ipynb) | PE-Core-G14-448 zero-shot on AIC 2026 Track 4 hidden test set |

---

## `local_eval/`

Notebooks that evaluate models **locally** using datasets where public ground
truth is available. These produce quality metrics, not submission files.

- Output format: `metrics.json`, `metrics.md`, `rankings_top10.jsonl`,
  `run_info.md`
- Run these notebooks before spending official leaderboard submissions to
  measure model quality on the known test split.
- No submission files are created.

| Notebook | Dataset | Description |
|---|---|---|
| [`local_eval/pab_original/pe_g14_attr_local_eval.ipynb`](local_eval/pab_original/pe_g14_attr_local_eval.ipynb) | PAB original `attr.json` test split | PE-Core-G14-448 zero-shot — computes mAP, R@1, R@5, R@10 |

---

## Workflow

```
local_eval/  →  decide if model is good enough  →  official_submission/
```

Always evaluate locally first. The local eval pipeline uses the original PAB
`attr.json` ground truth which gives mAP and recall metrics. Only submit
officially once you are satisfied with local metrics.
