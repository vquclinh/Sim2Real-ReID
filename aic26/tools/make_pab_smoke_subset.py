#!/usr/bin/env python3
"""Stage 1 — build a tiny PAB smoke-training subset from the prepared author layout.

The author dataloader (lhp_2/beit3/datasets.py::BaseDataset) hard-codes reading
ALL 75 files `annotation/train/pair_{0..74}.json`, and run_beit3_finetuning.py
has no max-steps/debug option. So a tiny smoke run needs a subset root that
keeps the full 75-file pattern but contains only a handful of records:

    <out-root>/
    ├── annotation/train/pair_0.json    (first N records, copied from source)
    ├── annotation/train/pair_1.json    (first N records)
    ├── annotation/train/pair_2.json    (empty — 0 records, loader prints "Load 0")
    ├── ...                              (all 75 files exist)
    └── train -> <source-root>/train    (single symlink, no per-image links)

Reads ONLY from --source-root (the local prepared layout — Drive is never
touched). Standard library only.

Usage:

    python aic26/tools/make_pab_smoke_subset.py \
        --source-root /content/aic_local/pab_author_layout \
        --out-root /content/aic_local/pab_author_layout_smoke_subset
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import sys
from pathlib import Path

EXPECTED_INDICES = list(range(75))  # author loader hard-codes pair_0..pair_74
REPORT_NAME = "AIC26_PAB_SMOKE_SUBSET_REPORT.md"


def take_records(src: Path, n: int):
    """Stream the first n non-empty JSONL lines from src."""
    lines = []
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        for line in itertools.islice((ln for ln in f if ln.strip()), n):
            lines.append(line.strip())
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Create a tiny 75-file PAB smoke subset from the prepared "
                    "author layout (reads source only; never touches Drive).")
    parser.add_argument("--source-root", default="/content/aic_local/pab_author_layout",
                        help="Prepared author layout root (default: %(default)s)")
    parser.add_argument("--out-root",
                        default="/content/aic_local/pab_author_layout_smoke_subset",
                        help="Smoke subset root to create (default: %(default)s)")
    parser.add_argument("--files-with-data", type=int, default=2,
                        help="How many pair_i.json get records; the rest are empty "
                             "(default: %(default)s)")
    parser.add_argument("--records-per-file", type=int, default=2,
                        help="Records copied into each non-empty file "
                             "(default: %(default)s)")
    parser.add_argument("--report-dir", default=None,
                        help="Report output dir (default: aic26/docs/audits next to "
                             "this script)")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    src_ann = source_root / "annotation" / "train"
    src_train = source_root / "train"

    if not src_ann.is_dir() or not src_train.is_dir():
        print(f"ERROR: source root is not a prepared author layout: {source_root}")
        print("Expected annotation/train/ and train/ — run "
              "aic26/tools/prepare_pab_author_layout.py first.")
        return 1
    missing_src = [i for i in EXPECTED_INDICES
                   if not (src_ann / f"pair_{i}.json").is_file()]
    if missing_src:
        print(f"ERROR: source is missing pair_*.json indices: {missing_src[:10]}")
        return 1

    report_dir = (Path(args.report_dir) if args.report_dir
                  else Path(__file__).resolve().parents[1] / "docs" / "audits")
    report_dir.mkdir(parents=True, exist_ok=True)

    out_ann = out_root / "annotation" / "train"
    out_ann.mkdir(parents=True, exist_ok=True)

    # Single symlink to the source image tree — no per-image symlinks.
    train_link = out_root / "train"
    target = src_train.resolve()
    if train_link.is_symlink():
        if os.readlink(train_link) != str(target):
            train_link.unlink()
            train_link.symlink_to(target)
    elif train_link.exists():
        print(f"ERROR: {train_link} exists and is not a symlink. Remove it manually.")
        return 1
    else:
        train_link.symlink_to(target)

    total_records = 0
    per_file = []
    for i in EXPECTED_INDICES:
        dst = out_ann / f"pair_{i}.json"
        if i < args.files_with_data:
            lines = take_records(src_ann / f"pair_{i}.json", args.records_per_file)
            dst.write_text("\n".join(lines) + ("\n" if lines else ""),
                           encoding="utf-8")
            total_records += len(lines)
            per_file.append((f"pair_{i}.json", len(lines)))
        else:
            dst.write_text("", encoding="utf-8")  # 0-record file keeps the pattern
    print(f"Wrote 75 pair_*.json ({args.files_with_data} with data, "
          f"{total_records} records total)")

    # Verify every copied record's image resolves under the subset root.
    sampled = resolved = 0
    unresolved = []
    for name, _count in per_file:
        for line in (out_ann / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            image = rec.get("image")
            sampled += 1
            if isinstance(image, str) and (out_root / image.lstrip("./")).is_file():
                resolved += 1
            else:
                unresolved.append({"file": name, "image": image})
    status = "PASS" if sampled and resolved == sampled else "FAIL"
    print(f"Verification: {resolved}/{sampled} image paths resolve -> {status}")

    lines = [
        "# AIC26 PAB Smoke Subset Report",
        "",
        f"> Generated: {datetime.datetime.now().isoformat(timespec='seconds')}  ",
        "> Tool: `aic26/tools/make_pab_smoke_subset.py` (reads prepared layout only; "
        "Drive untouched)  ",
        f"> Status: **{status}**",
        "",
        "## Layout",
        "",
        f"- Source root: `{source_root}`",
        f"- Subset root: `{out_root}`",
        f"- `annotation/train/pair_{{0..74}}.json`: all 75 present "
        f"({args.files_with_data} with {args.records_per_file} records each, "
        f"rest empty — the author loader accepts 0-record files).",
        f"- `train` -> `{target}` (single directory symlink).",
        f"- Total records: {total_records} (smoke steps = records // batch size).",
        "",
        "## Verification",
        "",
        f"- {resolved}/{sampled} copied record image paths resolve under the subset root.",
    ]
    if unresolved:
        lines.append("")
        lines.append("Unresolved:")
        lines.append("```")
        lines.extend(f"{u['file']}: {u['image']}" for u in unresolved[:10])
        lines.append("```")
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append("Run `aic26/scripts/run_beit3_lhp_smoke_train.sh` with "
                 f"`SMOKE_ROOT={out_root}`.")
    lines.append("")
    (report_dir / REPORT_NAME).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_dir / REPORT_NAME}")
    print(f"OVERALL: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
