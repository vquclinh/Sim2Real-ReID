#!/usr/bin/env python3
"""Stage 0B — prepare an author-compatible PAB layout from the competition Drive layout.

The competition dataset ships train annotations as `imgs_{0..74}.json` and images
nested as `raw/Part K/imgs_N/imgs_N/{full,goal,wentwrong}/...`, while the author
training code (lhp_2/beit3/datasets.py, uit/cmp/configs/*.yaml) hard-codes
`annotation/train/pair_{0..74}.json` with image paths like `train/imgs_N/goal/0.jpg`.

This tool bridges the two WITHOUT modifying author code and WITHOUT copying data:
it builds a symlink farm under --out-root (local disk) pointing back into the
read-only Drive dataset, then verifies that the author-side relative paths resolve.

The competition images are stored as `.webp` while the annotations reference
`.jpg`. When that mismatch is detected (--rewrite-image-ext auto, the default),
the annotation files are not symlinked but CONVERTED: each `imgs_i.json` is
streamed line by line into a local `pair_i.json` with `record["image"]`
rewritten `.jpg` -> `.webp` (all other fields preserved exactly, Drive source
untouched, no per-image symlinks).

Read-only with respect to the Drive dataset. Creates only symlinks and tiny
reports under --out-root / the report directory. Never deletes anything outside
--out-root. Standard library only (no torch / PIL / transformers / GPU).

Usage:

    python aic26/tools/prepare_pab_author_layout.py \
        --data-root /content/drive/MyDrive/aic2026_data \
        --out-root /content/aic_local/pab_author_layout \
        --max-sample-per-file 3

Reports (default: aic26/docs/audits next to this script, override --report-dir):

    AIC26_PAB_AUTHOR_LAYOUT_PREP_REPORT.md
    AIC26_PAB_AUTHOR_LAYOUT_PREP_SUMMARY.json
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import re
import sys
from pathlib import Path

EXPECTED_INDICES = list(range(75))  # author code hard-codes pair_0..pair_74
IMG_SUBDIRS = ("goal", "full", "wentwrong")
MAX_EXAMPLES = 10

REPORT_NAME = "AIC26_PAB_AUTHOR_LAYOUT_PREP_REPORT.md"
SUMMARY_NAME = "AIC26_PAB_AUTHOR_LAYOUT_PREP_SUMMARY.json"


class AdapterError(Exception):
    """Fatal layout-preparation error with a user-actionable message."""


# ---------------------------------------------------------------- input validation

def validate_input(data_root: Path):
    ann_dir = data_root / "raw" / "annotation" / "train"
    info = {"annotation_dir": str(ann_dir), "annotation_dir_exists": ann_dir.is_dir(),
            "annotation_missing_indices": [], "parts_found": []}
    if info["annotation_dir_exists"]:
        info["annotation_missing_indices"] = [
            i for i in EXPECTED_INDICES if not (ann_dir / f"imgs_{i}.json").is_file()]
    else:
        info["annotation_missing_indices"] = list(EXPECTED_INDICES)
    for k in range(1, 11):
        if (data_root / "raw" / f"Part {k}").is_dir():
            info["parts_found"].append(f"Part {k}")
    return info


# ---------------------------------------------------------------- image mapping

def discover_image_mapping(data_root: Path):
    """Map imgs_N -> real source dir by inspecting Part 1..10 children.

    Prefers `Part K/imgs_N/imgs_N` (the doubled-component competition nesting);
    falls back to `Part K/imgs_N` when it directly contains goal/full/wentwrong.
    """
    rx = re.compile(r"^imgs_(\d+)$")
    mapping = {}
    duplicates = []
    skipped = []
    for k in range(1, 11):
        part = data_root / "raw" / f"Part {k}"
        if not part.is_dir():
            continue
        try:
            children = sorted(os.listdir(part))
        except OSError:
            continue
        for name in children:
            m = rx.match(name)
            if not m:
                continue
            child = part / name
            if not child.is_dir():
                continue
            nested = child / name
            if nested.is_dir():
                source = nested
            elif any((child / s).is_dir() for s in IMG_SUBDIRS):
                source = child
            else:
                skipped.append(f"{child} (no nested {name}/ and no "
                               f"{'/'.join(IMG_SUBDIRS)} subdirs)")
                continue
            if name in mapping:
                duplicates.append(f"{name}: kept {mapping[name]}, also found {source}")
            else:
                mapping[name] = source
    missing = [i for i in EXPECTED_INDICES if f"imgs_{i}" not in mapping]
    return mapping, missing, duplicates, skipped


# ---------------------------------------------------------------- symlink farm

def ensure_symlink(link: Path, target: Path, overwrite: bool) -> str:
    """Create `link` -> `target`. Returns 'kept' | 'created' | 'replaced'."""
    if link.is_symlink():
        if os.readlink(link) == str(target):
            return "kept"
        if not overwrite:
            raise AdapterError(
                f"{link} is a symlink to {os.readlink(link)!r}, expected "
                f"{str(target)!r}. Re-run with --overwrite to recreate it.")
        link.unlink()
        link.symlink_to(target)
        return "replaced"
    if link.exists():
        if link.is_dir():
            raise AdapterError(
                f"{link} exists and is a real directory (not a symlink). "
                "Refusing to remove a real directory — clean it up manually.")
        if not overwrite:
            raise AdapterError(
                f"{link} exists and is a regular file, expected a symlink to "
                f"{target}. Re-run with --overwrite to replace it.")
        link.unlink()
        link.symlink_to(target)
        return "replaced"
    link.symlink_to(target)
    return "created"


def build_image_links(out_root: Path, mapping: dict, overwrite: bool):
    train_out = out_root / "train"
    train_out.mkdir(parents=True, exist_ok=True)
    img_actions = {"kept": 0, "created": 0, "replaced": 0}
    for i in EXPECTED_INDICES:
        name = f"imgs_{i}"
        action = ensure_symlink(train_out / name, mapping[name].resolve(), overwrite)
        img_actions[action] += 1
    return img_actions


def build_annotation_links(data_root: Path, out_root: Path, overwrite: bool):
    """Symlink mode: pair_i.json -> imgs_i.json on Drive (no conversion)."""
    ann_src = data_root / "raw" / "annotation" / "train"
    ann_out = out_root / "annotation" / "train"
    ann_out.mkdir(parents=True, exist_ok=True)
    ann_actions = {"kept": 0, "created": 0, "replaced": 0}
    for i in EXPECTED_INDICES:
        action = ensure_symlink(ann_out / f"pair_{i}.json",
                                (ann_src / f"imgs_{i}.json").resolve(), overwrite)
        ann_actions[action] += 1
    return ann_actions


# ---------------------------------------------------------------- extension rewrite

def detect_rewrite_ext(data_root: Path, out_root: Path, max_sample: int,
                       files_to_probe: int = 5):
    """Auto mode: probe sampled records against the prepared image symlinks.

    Returns ('.webp', detail) when sampled `.jpg` paths are missing but the same
    stem with `.webp` exists; otherwise ('none', detail).
    """
    ann_src = data_root / "raw" / "annotation" / "train"
    probed = jpg_hits = webp_hits = 0
    for i in EXPECTED_INDICES[:files_to_probe]:
        src = ann_src / f"imgs_{i}.json"
        try:
            with open(src, "r", encoding="utf-8", errors="replace") as f:
                for line in itertools.islice(
                        (ln for ln in f if ln.strip()), max(max_sample, 3)):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    image = rec.get("image")
                    if not isinstance(image, str):
                        continue
                    probed += 1
                    candidate = out_root / image.lstrip("./")
                    if candidate.is_file():
                        jpg_hits += 1
                    elif image.lower().endswith(".jpg") and \
                            candidate.with_suffix(".webp").is_file():
                        webp_hits += 1
        except OSError:
            continue
    detail = {"probed": probed, "as_annotated": jpg_hits, "webp_stem": webp_hits}
    if webp_hits and not jpg_hits:
        return ".webp", detail
    return "none", detail


def convert_annotation_file(src: Path, dst: Path, overwrite: bool):
    """Stream src JSONL to a local dst, rewriting image '.jpg' -> '.webp'.

    Returns (action, records_written, image_path_rewrites, examples).
    Atomic: writes to a .tmp sibling then os.replace(). Never touches src.
    """
    if dst.is_symlink():
        if not overwrite:
            raise AdapterError(
                f"{dst} is a symlink (from a previous symlink-mode run) but "
                "converted-annotation mode needs a regular file. "
                "Re-run with --overwrite to replace it.")
        dst.unlink()
        action = "replaced"
    elif dst.exists():
        if dst.is_dir():
            raise AdapterError(
                f"{dst} exists and is a real directory (not a file). "
                "Refusing to remove a real directory — clean it up manually.")
        if not overwrite:
            # Assume a previous converted run produced it; --overwrite regenerates.
            return "kept", 0, 0, []
        action = "replaced"
    else:
        action = "created"

    tmp = dst.with_name(dst.name + ".tmp")
    records = rewrites = 0
    examples = []
    with open(src, "r", encoding="utf-8", errors="replace") as fin, \
            open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                fout.write(stripped + "\n")  # pass through unparseable lines as-is
                records += 1
                continue
            image = rec.get("image")
            if isinstance(image, str) and image.lower().endswith(".jpg"):
                new_image = image[:-4] + ".webp"
                rec["image"] = new_image
                rewrites += 1
                if len(examples) < 3:
                    examples.append({"before": image, "after": new_image})
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records += 1
    os.replace(tmp, dst)
    return action, records, rewrites, examples


def build_converted_annotations(data_root: Path, out_root: Path, overwrite: bool):
    ann_src = data_root / "raw" / "annotation" / "train"
    ann_out = out_root / "annotation" / "train"
    ann_out.mkdir(parents=True, exist_ok=True)
    actions = {"kept": 0, "created": 0, "replaced": 0}
    total_records = total_rewrites = 0
    examples = []
    for i in EXPECTED_INDICES:
        action, n_rec, n_rw, ex = convert_annotation_file(
            ann_src / f"imgs_{i}.json", ann_out / f"pair_{i}.json", overwrite)
        actions[action] += 1
        total_records += n_rec
        total_rewrites += n_rw
        if len(examples) < 3:
            examples.extend(ex[:3 - len(examples)])
        if i % 15 == 0 or i == EXPECTED_INDICES[-1]:
            print(f"  pair_{i}.json: {action} ({n_rec} records)", flush=True)
    return actions, total_records, total_rewrites, examples


# ---------------------------------------------------------------- verification

def verify_author_paths(out_root: Path, max_sample: int):
    """Sample records from each pair_i.json and resolve image paths under out_root."""
    ann_out = out_root / "annotation" / "train"
    per_file = []
    sampled = resolved = 0
    unresolved_examples = []
    for i in EXPECTED_INDICES:
        fpath = ann_out / f"pair_{i}.json"
        entry = {"file": f"pair_{i}.json", "sampled": 0, "resolved": 0,
                 "status": "FAIL", "error": None}
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in itertools.islice(
                        (ln for ln in f if ln.strip()), max_sample):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        entry["error"] = "JSON parse error in sampled line"
                        continue
                    image = rec.get("image")
                    if not isinstance(image, str):
                        entry["error"] = f"record has no string 'image' field: {sorted(rec)}"
                        continue
                    entry["sampled"] += 1
                    sampled += 1
                    candidate = out_root / image.lstrip("./")
                    if candidate.is_file():
                        entry["resolved"] += 1
                        resolved += 1
                    elif len(unresolved_examples) < MAX_EXAMPLES:
                        unresolved_examples.append(
                            {"annotation": entry["file"], "image": image,
                             "checked": str(candidate)})
        except OSError as exc:
            entry["error"] = f"cannot read: {exc}"
        if entry["sampled"] and entry["resolved"] == entry["sampled"]:
            entry["status"] = "PASS"
        elif entry["resolved"] > 0:
            entry["status"] = "WARN"
        per_file.append(entry)
    return {"per_file": per_file, "sampled": sampled, "resolved": resolved,
            "unresolved_examples": unresolved_examples,
            "rate": (resolved / sampled) if sampled else 0.0}


def check_uit_local_eval(data_root: Path):
    candidates = [
        data_root / "pab_original" / "annotation" / "test" / "pair.json",
        data_root / "raw" / "annotation" / "test" / "pair.json",
    ]
    found = [str(p) for p in candidates if p.is_file()]
    return {"checked": [str(p) for p in candidates], "found": found,
            "exists": bool(found)}


# ---------------------------------------------------------------- reporting

def write_markdown(path: Path, ctx):
    v = ctx["verify"]
    L = []
    L.append("# AIC26 PAB Author Layout Prep Report\n")
    L.append(f"> Generated: {ctx['timestamp']}  ")
    L.append("> Tool: `aic26/tools/prepare_pab_author_layout.py` "
             "(symlink adapter; Drive dataset untouched)  ")
    L.append(f"> Overall status: **{ctx['status']}**\n")

    mode = ctx.get("annotation_mode", "symlink")
    L.append("## Executive Summary\n")
    L.append(f"- Overall: **{ctx['status']}**.")
    L.append(f"- Annotation files `pair_{{0..74}}.json`: "
             f"{ctx['ann_total']}/75 in place — mode **{mode}** "
             f"(created {ctx['ann_actions']['created']}, kept {ctx['ann_actions']['kept']}, "
             f"replaced {ctx['ann_actions']['replaced']}).")
    L.append(f"- Image extension rewrite mode: `{ctx.get('rewrite_image_ext', 'none')}`"
             + (f" — {ctx.get('annotation_records_written', 0)} records written, "
                f"{ctx.get('image_path_rewrites', 0)} image paths rewritten "
                f"`.jpg` -> `.webp`." if mode == "converted" else "."))
    L.append(f"- Image directory symlinks `train/imgs_{{0..74}}`: "
             f"{ctx['img_total']}/75 in place "
             f"(created {ctx['img_actions']['created']}, kept {ctx['img_actions']['kept']}, "
             f"replaced {ctx['img_actions']['replaced']}).")
    L.append(f"- Author-path verification: {v['resolved']}/{v['sampled']} sampled "
             f"image paths resolve under the prepared root ({v['rate']:.0%}).")
    L.append(f"- BEiT-3/LHP ready: **{'yes' if ctx['beit3_ready'] else 'NO'}**; "
             f"UIT/CMP ready: **{'yes' if ctx['uit_ready'] else 'NO'}** "
             "(see compatibility sections).\n")

    L.append("## Input Data Root\n")
    L.append(f"```\n{ctx['data_root']}\n```")
    inp = ctx["input"]
    L.append(f"- Annotation dir: `{inp['annotation_dir']}` "
             f"({'exists' if inp['annotation_dir_exists'] else 'MISSING'})")
    L.append(f"- Missing `imgs_*.json` indices: {inp['annotation_missing_indices'] or 'none'}")
    L.append(f"- Parts found: {', '.join(inp['parts_found']) or 'none'}\n")

    L.append("## Output Prepared Root\n")
    ann_arrow = ("local converted copy of raw/annotation/train/imgs_{i}.json"
                 if mode == "converted" else
                 "-> raw/annotation/train/imgs_{i}.json")
    L.append(f"```\n{ctx['out_root']}\n├── annotation/train/pair_{{0..74}}.json  "
             f"{ann_arrow}\n└── train/imgs_{{0..74}}           "
             f"     -> raw/Part K/imgs_N[/imgs_N]\n```\n")

    L.append("## Annotation Symlinks\n")
    if mode == "converted":
        L.append("- Mode: **converted** — local JSONL files written under the "
                 "prepared root (Drive source annotations untouched; no per-image "
                 "symlinks created).")
        L.append(f"- Image extension rewrite: `{ctx.get('rewrite_image_ext')}` — "
                 f"{ctx.get('annotation_records_written', 0)} records written, "
                 f"{ctx.get('image_path_rewrites', 0)} image paths rewritten.")
    else:
        L.append("- Mode: **symlink** — `pair_i.json` symlinks to the Drive source "
                 "`imgs_i.json` (no conversion needed).")
    if ctx.get("detect_detail") is not None:
        d = ctx["detect_detail"]
        L.append(f"- Auto-detection probe: {d['probed']} sampled records — "
                 f"{d['as_annotated']} resolved as annotated, "
                 f"{d['webp_stem']} resolved only as `.webp` stem.")
    L.append("")
    L.append("| Action | Count |")
    L.append("|---|---:|")
    for k in ("created", "kept", "replaced"):
        L.append(f"| {k} | {ctx['ann_actions'][k]} |")
    L.append(f"| **total** | **{ctx['ann_total']}** |")
    L.append("")
    if ctx.get("rewrite_examples"):
        L.append("Sample image-path rewrites (before -> after):\n")
        L.append("```")
        for ex in ctx["rewrite_examples"]:
            L.append(f"{ex['before']} -> {ex['after']}")
        L.append("```")
        L.append("")

    L.append("## Image Directory Mapping\n")
    L.append("Mapping was discovered dynamically from `Part 1..10` children "
             "(no hard-coded Part->imgs ranges). Preference: "
             "`Part K/imgs_N/imgs_N` if present, else `Part K/imgs_N` when it "
             "contains goal/full/wentwrong.\n")
    L.append("| imgs_N | Source directory |")
    L.append("|---|---|")
    for name in sorted(ctx["mapping"], key=lambda n: int(n.split("_")[1])):
        L.append(f"| `{name}` | `{ctx['mapping'][name]}` |")
    L.append("")
    if ctx["map_duplicates"]:
        L.append("Duplicate candidates (first occurrence kept):\n")
        L.extend(f"- {d}" for d in ctx["map_duplicates"])
        L.append("")
    if ctx["map_skipped"]:
        L.append("Skipped candidate dirs (unrecognized internal layout):\n")
        L.extend(f"- {s}" for s in ctx["map_skipped"][:MAX_EXAMPLES])
        L.append("")

    L.append("## Author Path Verification\n")
    L.append(f"Sampled up to {ctx['max_sample']} records per `pair_i.json`; each "
             f"record's `image` value was resolved against `{ctx['out_root']}`.\n")
    L.append(f"- Sampled: {v['sampled']}; resolved: {v['resolved']} ({v['rate']:.0%}).")
    fails = [e for e in v["per_file"] if e["status"] == "FAIL"]
    warns = [e for e in v["per_file"] if e["status"] == "WARN"]
    passes = sum(1 for e in v["per_file"] if e["status"] == "PASS")
    L.append(f"- Per-file: {passes} PASS, {len(warns)} WARN, {len(fails)} FAIL.\n")
    if warns or fails:
        L.append("| File | Sampled | Resolved | Status | Error |")
        L.append("|---|---:|---:|---|---|")
        for e in warns + fails:
            L.append(f"| `{e['file']}` | {e['sampled']} | {e['resolved']} "
                     f"| {e['status']} | {e['error'] or '-'} |")
        L.append("")
    if v["unresolved_examples"]:
        L.append("Unresolved examples:\n")
        L.append("```")
        for ex in v["unresolved_examples"]:
            L.append(f"{ex['annotation']}: {ex['image']} -> checked {ex['checked']}")
        L.append("```\n")

    L.append("## Compatibility With BEiT-3/LHP\n")
    if ctx["beit3_ready"]:
        L.append("**Ready.** `lhp_2/beit3/datasets.py::BaseDataset` reads "
                 "`<data_path>/annotation/train/pair_{0..74}.json` and resolves each "
                 "record's `image` relative to `data_path`. Both now work with "
                 f"`--data_path {ctx['out_root']}` — author code unmodified. "
                 "BEiT-3/LHP smoke training (Stage 1) can proceed with this prepared "
                 "layout after this verification.")
    else:
        L.append("**Not ready.** Author-side paths did not fully resolve under the "
                 "prepared root — see Blocking Issues.")
    L.append("")

    L.append("## Compatibility With UIT/CMP\n")
    uit = ctx["uit_check"]
    L.append("- Training annotations: same `pair_*.json` symlinks work via an "
             "`aic26/`-side config copy pointing `train_file`/`image_root` at the "
             "prepared root (author yaml uses `../../data/PAB/`).")
    L.append(f"- Local evaluation: `annotation/test/pair.json` (caption as LIST, "
             f"required by `search_test_dataset`) — "
             f"{'found: ' + ', '.join(f'`{p}`' for p in uit['found']) if uit['exists'] else '**still missing**'} "
             f"(checked: {', '.join(f'`{p}`' for p in uit['checked'])}).")
    L.append("- **This script does not create validation data.** UIT/CMP training "
             "may still need `annotation/test/pair.json` or a config adjustment "
             "before its eval step can run.")
    L.append("")

    L.append("## Blocking Issues\n")
    L.extend([f"- {b}" for b in ctx["blocking"]] or ["- None."])
    L.append("")

    L.append("## Warnings\n")
    L.extend([f"- {w}" for w in ctx["warnings"]] or ["- None."])
    L.append("")

    L.append("## Next Step\n")
    if ctx["status"] == "FAIL":
        L.append("Resolve the blocking issues (image-dir mapping / path resolution) "
                 "and re-run with `--overwrite`.")
    else:
        L.append("Proceed to Stage 1 — BEiT-3 + LHP tiny smoke training using "
                 f"`--data_path {ctx['out_root']}` per "
                 "`AIC26_AUTHOR_ALIGNED_TRAINING_READY_CHECK.md`. For UIT/CMP, first "
                 "source or derive `annotation/test/pair.json` (local validation).")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_summary(path: Path, ctx):
    v = ctx["verify"]
    summary = {
        "overall_status": ctx["status"],
        "data_root": str(ctx["data_root"]),
        "out_root": str(ctx["out_root"]),
        "annotation_links_created": ctx["ann_total"],
        "image_links_created": ctx["img_total"],
        "image_dirs_mapped": len(ctx["mapping"]),
        "annotation_mode": ctx.get("annotation_mode", "symlink"),
        "rewrite_image_ext": ctx.get("rewrite_image_ext", "none"),
        "annotation_records_written": ctx.get("annotation_records_written", 0),
        "image_path_rewrites": ctx.get("image_path_rewrites", 0),
        "sampled_paths": v["sampled"],
        "sampled_paths_resolved": v["resolved"],
        "unresolved_examples": v["unresolved_examples"],
        "beit3_lhp_ready": ctx["beit3_ready"],
        "uit_cmp_ready": ctx["uit_ready"],
        "blocking_issues": ctx["blocking"],
        "warnings": ctx["warnings"],
        "annotation_link_actions": ctx["ann_actions"],
        "image_link_actions": ctx["img_actions"],
        "generated": ctx["timestamp"],
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="Prepare an author-compatible PAB layout (pair_*.json + train/imgs_N "
                    "symlinks) from the competition Drive layout. Drive dataset is "
                    "never modified; symlinks are created under --out-root only.")
    parser.add_argument("--data-root", default="/content/drive/MyDrive/aic2026_data",
                        help="Competition dataset root on mounted Drive (default: %(default)s)")
    parser.add_argument("--out-root", default="/content/aic_local/pab_author_layout",
                        help="Local prepared-layout root to create (default: %(default)s)")
    parser.add_argument("--max-sample-per-file", type=int, default=3,
                        help="Records sampled per annotation file for verification "
                             "(default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recreate existing symlinks/files inside --out-root that "
                             "point to wrong targets")
    parser.add_argument("--rewrite-image-ext", choices=["auto", "none", ".webp"],
                        default="auto",
                        help="Annotation image-extension handling: 'none' symlinks "
                             "annotations as-is; '.webp' writes converted local "
                             "pair_*.json with image paths rewritten .jpg -> .webp; "
                             "'auto' (default) probes sampled records and picks "
                             "'.webp' when annotated .jpg files are missing but the "
                             "same stem exists as .webp")
    parser.add_argument("--report-dir", default=None,
                        help="Where to write the report and JSON summary "
                             "(default: aic26/docs/audits next to this script)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    if not data_root.is_dir():
        print(f"ERROR: data root does not exist or is not a directory: {data_root}")
        print("Run this in an environment where Google Drive is mounted "
              "(e.g. Colab after drive.mount). No layout or report was written.")
        return 1

    report_dir = (Path(args.report_dir) if args.report_dir
                  else Path(__file__).resolve().parents[1] / "docs" / "audits")
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {data_root}")
    print(f"Out root : {out_root}")

    print("Validating input layout ...")
    inp = validate_input(data_root)
    if not inp["annotation_dir_exists"]:
        print(f"ERROR: annotation folder missing: {inp['annotation_dir']}")
        return 1
    if len(inp["annotation_missing_indices"]) > 5:
        print(f"ERROR: {len(inp['annotation_missing_indices'])} of 75 imgs_*.json "
              f"files missing (first: {inp['annotation_missing_indices'][:10]}). "
              "Refusing to build a partial layout.")
        return 1

    print("Discovering image-folder mapping from Part 1..10 ...")
    mapping, map_missing, duplicates, skipped = discover_image_mapping(data_root)
    print(f"  mapped {len(mapping)} imgs_N directories")

    blocking, warnings = [], []
    if inp["annotation_missing_indices"]:
        warnings.append(f"Missing imgs_*.json indices: {inp['annotation_missing_indices']}")
    if duplicates:
        warnings.append(f"{len(duplicates)} imgs_N found in multiple Parts "
                        "(first occurrence kept) — see report.")
    if map_missing:
        print(f"ERROR: cannot map imgs_N for indices: {map_missing}")
        blocking.append(f"Unmapped imgs_N indices (no source dir found in any Part): "
                        f"{map_missing}")
        ctx_fail = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "data_root": data_root, "out_root": out_root, "input": inp,
            "mapping": {k: str(v) for k, v in mapping.items()},
            "map_duplicates": duplicates, "map_skipped": skipped,
            "ann_actions": {"kept": 0, "created": 0, "replaced": 0},
            "img_actions": {"kept": 0, "created": 0, "replaced": 0},
            "ann_total": 0, "img_total": 0,
            "verify": {"per_file": [], "sampled": 0, "resolved": 0,
                       "unresolved_examples": [], "rate": 0.0},
            "uit_check": check_uit_local_eval(data_root),
            "beit3_ready": False, "uit_ready": False,
            "status": "FAIL", "blocking": blocking, "warnings": warnings,
            "max_sample": args.max_sample_per_file,
        }
        write_markdown(report_dir / REPORT_NAME, ctx_fail)
        write_summary(report_dir / SUMMARY_NAME, ctx_fail)
        print(f"Report : {report_dir / REPORT_NAME}")
        print("OVERALL: FAIL")
        return 1

    print("Building image directory symlinks ...")
    try:
        img_actions = build_image_links(out_root, mapping, args.overwrite)
    except AdapterError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"  image dir links : {img_actions}")

    rewrite_mode = args.rewrite_image_ext
    detect_detail = None
    if rewrite_mode == "auto":
        rewrite_mode, detect_detail = detect_rewrite_ext(
            data_root, out_root, args.max_sample_per_file)
        print(f"  auto-detected rewrite mode: {rewrite_mode} ({detect_detail})")

    annotation_mode = "converted" if rewrite_mode == ".webp" else "symlink"
    print(f"Building annotation files (mode: {annotation_mode}) ...")
    try:
        if annotation_mode == "converted":
            ann_actions, records_written, path_rewrites, rewrite_examples = \
                build_converted_annotations(data_root, out_root, args.overwrite)
        else:
            ann_actions = build_annotation_links(data_root, out_root, args.overwrite)
            records_written = path_rewrites = 0
            rewrite_examples = []
    except AdapterError as exc:
        print(f"ERROR: {exc}")
        return 1
    ann_total = sum(ann_actions.values())
    img_total = sum(img_actions.values())
    print(f"  annotation files: {ann_actions}")
    if annotation_mode == "converted":
        print(f"  records written : {records_written}; "
              f"image paths rewritten: {path_rewrites}")
        if ann_actions["kept"]:
            warnings_note = (f"{ann_actions['kept']} existing converted pair_*.json "
                             "kept without content re-check — use --overwrite to "
                             "force regeneration.")
        else:
            warnings_note = None
    else:
        warnings_note = None

    print("Verifying author-side image paths ...")
    verify = verify_author_paths(out_root, args.max_sample_per_file)
    print(f"  resolved {verify['resolved']}/{verify['sampled']} sampled paths")

    uit_check = check_uit_local_eval(data_root)

    beit3_ready = (ann_total == 75 and img_total == 75
                   and verify["sampled"] > 0
                   and verify["resolved"] == verify["sampled"])
    uit_ready = uit_check["exists"]

    if not beit3_ready:
        if verify["sampled"] == 0:
            blocking.append("No image paths could be sampled from pair_*.json.")
        elif verify["resolved"] < verify["sampled"]:
            blocking.append(
                f"{verify['sampled'] - verify['resolved']} of {verify['sampled']} "
                "sampled author-side image paths do not resolve under the prepared "
                "root — image mapping is incomplete or path convention differs.")
    if not uit_ready:
        warnings.append("UIT/CMP local evaluation data `annotation/test/pair.json` "
                        "is still missing — UIT/CMP training eval step blocked until "
                        "it is sourced or the config is adjusted. (BEiT-3/LHP is "
                        "unaffected.) This script does not create validation data.")
    if warnings_note:
        warnings.append(warnings_note)

    if blocking:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    ctx = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_root": data_root, "out_root": out_root, "input": inp,
        "mapping": {k: str(v) for k, v in mapping.items()},
        "map_duplicates": duplicates, "map_skipped": skipped,
        "ann_actions": ann_actions, "img_actions": img_actions,
        "ann_total": ann_total, "img_total": img_total,
        "annotation_mode": annotation_mode,
        "rewrite_image_ext": rewrite_mode if rewrite_mode == ".webp" else "none",
        "annotation_records_written": records_written,
        "image_path_rewrites": path_rewrites,
        "rewrite_examples": rewrite_examples,
        "detect_detail": detect_detail,
        "verify": verify, "uit_check": uit_check,
        "beit3_ready": beit3_ready, "uit_ready": uit_ready,
        "status": status, "blocking": blocking, "warnings": warnings,
        "max_sample": args.max_sample_per_file,
    }
    write_markdown(report_dir / REPORT_NAME, ctx)
    write_summary(report_dir / SUMMARY_NAME, ctx)

    print()
    print(f"Report : {report_dir / REPORT_NAME}")
    print(f"Summary: {report_dir / SUMMARY_NAME}")
    print(f"BEiT-3/LHP ready: {beit3_ready}; UIT/CMP ready: {uit_ready}")
    for b in blocking:
        print(f"BLOCKING: {b}")
    for w in warnings:
        print(f"WARNING : {w}")
    print(f"OVERALL: {status}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
