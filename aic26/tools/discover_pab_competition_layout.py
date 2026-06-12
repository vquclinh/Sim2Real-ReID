#!/usr/bin/env python3
"""Stage 0A — discover and verify the PAB competition dataset layout on Google Drive.

Read-only. Standard library only (no torch / transformers / PIL / GPU).

Inspects the competition Drive layout (which does NOT exactly match the author
code layout), detects both `pair_*.json` (author) and `imgs_*.json` (competition)
training-annotation patterns, samples records, resolves sampled image paths
against candidate roots, inspects test/official assets, and reports what adapter
is needed before author-aligned training (see
aic26/docs/audits/AIC26_AUTHOR_ALIGNED_TRAINING_READY_CHECK.md).

Usage:

    python aic26/tools/discover_pab_competition_layout.py \
        --data-root /content/drive/MyDrive/aic2026_data \
        --max-sample-per-file 3

Outputs (under aic26/docs/audits/ by default, override with --output-dir):

    AIC26_PAB_COMPETITION_LAYOUT_DISCOVERY_REPORT.md
    AIC26_PAB_COMPETITION_LAYOUT_DISCOVERY_SUMMARY.json
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
MAX_ARRAY_LOAD_BYTES = 256 * 1024 * 1024  # refuse to json.load() bigger arrays
MAX_EXAMPLES = 10

REPORT_NAME = "AIC26_PAB_COMPETITION_LAYOUT_DISCOVERY_REPORT.md"
SUMMARY_NAME = "AIC26_PAB_COMPETITION_LAYOUT_DISCOVERY_SUMMARY.json"


# ---------------------------------------------------------------- helpers

def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def detect_format(path: Path) -> str:
    """Return 'jsonl', 'json-array', 'json-object', or 'unknown'."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(8192)
    except OSError:
        return "unreadable"
    stripped = head.lstrip()
    if not stripped:
        return "empty"
    if stripped.startswith("["):
        return "json-array"
    if stripped.startswith("{"):
        first_line = stripped.splitlines()[0].strip()
        try:
            json.loads(first_line)
            return "jsonl"
        except json.JSONDecodeError:
            return "json-object"
    return "unknown"


def sample_and_count(path: Path, fmt: str, max_sample: int):
    """Return (records_sample, record_count or None). Streams JSONL."""
    if fmt == "jsonl":
        samples, count = [], 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                if len(samples) < max_sample:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        samples.append({"__parse_error__": line[:200]})
        return samples, count
    if fmt == "json-array":
        if path.stat().st_size > MAX_ARRAY_LOAD_BYTES:
            return [], None
        try:
            data = json.load(open(path, "r", encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return [], None
        if isinstance(data, list):
            return data[:max_sample], len(data)
        return [data], 1
    if fmt == "json-object":
        try:
            data = json.load(open(path, "r", encoding="utf-8", errors="replace"))
            return [data], 1
        except (json.JSONDecodeError, OSError):
            return [], None
    return [], None


def first_text_lines(path: Path, n=3):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n")[:200] for line in itertools.islice(f, n)]
    except OSError:
        return []


def list_dir_entries(path: Path, cap=50):
    """List up to `cap` entry names without scanning the whole directory."""
    names, capped = [], False
    try:
        with os.scandir(path) as it:
            for entry in it:
                names.append(entry.name)
                if len(names) >= cap:
                    capped = True
                    break
    except OSError:
        return [], False
    return sorted(names), capped


# ---------------------------------------------------------------- inspection

def check_top_level(root: Path):
    sub = ["raw", "pab_original", "archives", "output", "runs"]
    return {name: (root / name).is_dir() for name in sub}


def inspect_raw(root: Path):
    raw = root / "raw"
    info = {"exists": raw.is_dir(), "dirs": [], "dirs_capped": False,
            "parts_found": [], "annotation_exists": False,
            "test_set_exists": False}
    if not info["exists"]:
        return info
    names, capped = list_dir_entries(raw, cap=100)
    info["dirs"] = [n for n in names if (raw / n).is_dir()]
    info["dirs_capped"] = capped
    for i in range(1, 11):
        if (raw / f"Part {i}").is_dir():
            info["parts_found"].append(f"Part {i}")
    info["annotation_exists"] = (raw / "annotation").is_dir()
    info["test_set_exists"] = (raw / "name-masked_test-set").is_dir()
    return info


def inspect_annotation_file(path: Path, max_sample: int):
    fmt = detect_format(path)
    samples, count = sample_and_count(path, fmt, max_sample)
    keys = sorted({k for rec in samples if isinstance(rec, dict) for k in rec})
    rec0 = samples[0] if samples and isinstance(samples[0], dict) else {}
    caption = rec0.get("caption")
    image = rec0.get("image")
    detail = {
        "file": path.name,
        "size": path.stat().st_size,
        "format": fmt,
        "record_count": count,
        "sample_keys": keys,
        "samples": samples,
        "has_image": "image" in keys,
        "has_caption": "caption" in keys,
        "has_image_id": "image_id" in keys,
        "caption_type": type(caption).__name__ if caption is not None else "absent",
        "image_is_str": isinstance(image, str),
    }
    # Author train loaders need image(str) + caption(str); UIT train also needs image_id.
    if not detail["has_image"] or not detail["has_caption"] or fmt in ("unreadable", "empty", "unknown"):
        detail["status"] = "FAIL"
    elif not detail["image_is_str"] or detail["caption_type"] != "str" or not detail["has_image_id"]:
        detail["status"] = "WARN"
    else:
        detail["status"] = "PASS"
    return detail


def discover_train_annotations(root: Path, max_sample: int):
    candidates = [
        root / "raw" / "annotation" / "train",
        root / "pab_original" / "annotation" / "train",
        root / "raw" / "annotation",
        root / "pab_original" / "annotation",
    ]
    rx = re.compile(r"^(pair|imgs)_(\d+)\.json$")
    result = {"search_dirs": [], "dir_used": None, "pattern": "none",
              "files": [], "indices_found": [], "indices_missing": [],
              "total_records": 0, "counts_complete": True}
    for cand in candidates:
        entry = {"dir": str(cand), "exists": cand.is_dir(), "pair": 0, "imgs": 0}
        matches = []
        if entry["exists"]:
            names, _ = list_dir_entries(cand, cap=500)
            for name in names:
                m = rx.match(name)
                if m and (cand / name).is_file():
                    matches.append((m.group(1), int(m.group(2)), cand / name))
            entry["pair"] = sum(1 for p, _, _ in matches if p == "pair")
            entry["imgs"] = sum(1 for p, _, _ in matches if p == "imgs")
        result["search_dirs"].append(entry)
        if matches and result["dir_used"] is None:
            result["dir_used"] = str(cand)
            prefixes = {p for p, _, _ in matches}
            result["pattern"] = "mixed" if len(prefixes) > 1 else prefixes.pop()
            matches.sort(key=lambda t: t[1])
            indices = []
            for _, idx, fpath in matches:
                print(f"  inspecting {fpath.name} ...", flush=True)
                detail = inspect_annotation_file(fpath, max_sample)
                detail["index"] = idx
                result["files"].append(detail)
                indices.append(idx)
                if detail["record_count"] is None:
                    result["counts_complete"] = False
                else:
                    result["total_records"] += detail["record_count"]
            result["indices_found"] = sorted(set(indices))
            result["indices_missing"] = [i for i in EXPECTED_INDICES
                                         if i not in set(indices)]
    return result


def candidate_image_roots(root: Path):
    roots = [root / "raw"]
    roots += [root / "raw" / f"Part {i}" for i in range(1, 11)]
    roots += [root / "raw" / "train", root / "raw" / "train_webp",
              root / "pab_original", root / "pab_original" / "train",
              root / "pab_original" / "train_webp"]
    return [r for r in roots if r.is_dir()]


def resolve_images(root: Path, train_files: list, max_sample: int):
    roots = candidate_image_roots(root)
    samples = []
    for detail in train_files:
        for rec in detail["samples"][:max_sample]:
            if isinstance(rec, dict) and isinstance(rec.get("image"), str):
                samples.append((detail["file"], rec["image"]))
    result = {"candidate_roots": [str(r) for r in roots],
              "sampled": len(samples), "resolved": 0,
              "resolved_roots": [], "resolved_examples": [],
              "unresolved_examples": []}
    hit_roots = {}
    for src_file, rel in samples:
        rel_clean = rel.lstrip("./")
        found = None
        for r in roots:
            direct = r / rel_clean
            if direct.is_file():
                found = (str(r), str(direct))
                break
            # competition nesting doubles the first component:
            # raw/Part 1/imgs_7/imgs_7/full/...  for image "imgs_7/full/..."
            first = rel_clean.split("/", 1)[0]
            doubled = r / first / rel_clean
            if doubled.is_file():
                found = (f"{r}/{first} (doubled component)", str(doubled))
                break
        if found:
            result["resolved"] += 1
            hit_roots[found[0]] = hit_roots.get(found[0], 0) + 1
            if len(result["resolved_examples"]) < MAX_EXAMPLES:
                result["resolved_examples"].append(
                    {"annotation": src_file, "image": rel, "resolved_to": found[1]})
        elif len(result["unresolved_examples"]) < MAX_EXAMPLES:
            result["unresolved_examples"].append(
                {"annotation": src_file, "image": rel})
    result["resolved_roots"] = [
        {"root": k, "hits": v} for k, v in sorted(hit_roots.items())]
    result["rate"] = (result["resolved"] / result["sampled"]) if samples else 0.0
    return result


TEST_PATHS = [
    "raw/name-masked_test-set",
    "raw/name-masked_test-set/query_text.json",
    "raw/name-masked_test-set/query_index.txt",
    "raw/name-masked_test-set/gallery.zip",
    "raw/name-masked_test-set/gallery",
    "pab_original/annotation/test",
    "pab_original/archives/test.zip",
    "pab_original/archives/ucc.zip",
    "pab_original/annotation/test/attr.json",
    "pab_original/annotation/test/pair.json",
    "pab_original/annotation/test/ucc.json",
]

# usefulness: training / local validation / official submission / distractor-eval only
TEST_USEFULNESS = {
    "raw/name-masked_test-set": "official submission (hidden test container)",
    "raw/name-masked_test-set/query_text.json": "official submission (queries, no ground truth expected)",
    "raw/name-masked_test-set/query_index.txt": "official submission (query order for answer.txt)",
    "raw/name-masked_test-set/gallery.zip": "official submission (gallery archive)",
    "raw/name-masked_test-set/gallery": "official submission (gallery images)",
    "pab_original/annotation/test": "local validation (public test annotations)",
    "pab_original/archives/test.zip": "local validation (archived test split)",
    "pab_original/archives/ucc.zip": "distractor/eval only (UCC archive)",
    "pab_original/annotation/test/attr.json": "local validation (single-caption ground truth; used by PE-G14 local eval)",
    "pab_original/annotation/test/pair.json": "local validation (caption-list ground truth; required by UIT search_test_dataset)",
    "pab_original/annotation/test/ucc.json": "distractor/eval only",
}


def discover_test_assets(root: Path, max_sample: int):
    out = []
    for rel in TEST_PATHS:
        p = root / rel
        entry = {"path": rel, "exists": p.exists(),
                 "kind": "dir" if p.is_dir() else ("file" if p.is_file() else "missing"),
                 "usefulness": TEST_USEFULNESS[rel]}
        if p.is_file():
            entry["size"] = p.stat().st_size
            if p.suffix == ".json":
                fmt = detect_format(p)
                samples, count = sample_and_count(p, fmt, max_sample)
                entry.update({"format": fmt, "record_count": count,
                              "samples": samples[:max_sample]})
                keys = sorted({k for r in samples if isinstance(r, dict) for k in r})
                entry["sample_keys"] = keys
                if rel.startswith("raw/name-masked_test-set"):
                    entry["has_ground_truth"] = any(
                        k in keys for k in ("image", "image_id", "positive", "gt"))
            elif p.suffix == ".txt":
                entry["first_lines"] = first_text_lines(p, 3)
        elif p.is_dir():
            names, capped = list_dir_entries(p, cap=20)
            entry["entries_preview"] = names
            entry["entries_capped"] = capped
        out.append(entry)
    return out


# ---------------------------------------------------------------- decisions

def build_compatibility(train, images, tests):
    t = {e["path"]: e for e in tests}
    pattern = train["pattern"]
    n_found = len(train["indices_found"])
    pair_ok = pattern == "pair" and not train["indices_missing"]
    imgs_like = pattern in ("imgs", "mixed") and n_found > 0
    ann_found = (f"{n_found}/75 `{pattern}_*.json` in `{train['dir_used']}`"
                 if train["dir_used"] else "no train annotation files found")
    img_found = (f"{images['resolved']}/{images['sampled']} sampled images resolved"
                 if images["sampled"] else "no image samples to test")
    gallery = t["raw/name-masked_test-set/gallery"]["exists"] or \
        t["raw/name-masked_test-set/gallery.zip"]["exists"]
    query = t["raw/name-masked_test-set/query_text.json"]["exists"]
    pair_json = t["pab_original/annotation/test/pair.json"]["exists"]

    def row(component, expects, found, compatible, adapter):
        return {"component": component, "expects": expects, "found": found,
                "compatible": compatible, "adapter": adapter}

    ann_adapter = ("none" if pair_ok else
                   "rename/convert imgs_*.json -> pair_*.json in a prepared dir"
                   if imgs_like else "BLOCKED: train annotations missing")
    img_adapter = ("none" if images["rate"] >= 0.9 and images["sampled"] else
                   "image-root symlink layout (see Adapter Recommendation)"
                   if images["resolved"] else
                   "BLOCKED: sampled images did not resolve")
    rows = [
        row("BEiT-3/LHP training",
            "`<data_path>/annotation/train/pair_{0..74}.json` (datasets.py:35-40) + images under data_path",
            f"{ann_found}; {img_found}",
            "YES" if pair_ok and images["rate"] >= 0.9 else ("NO" if not imgs_like else "WITH ADAPTER"),
            f"{ann_adapter}; {img_adapter}"),
        row("BEiT-3/LHP inference",
            "gallery folder + query json (lhp_2/beit3/inference.py args)",
            f"gallery={'yes' if gallery else 'no'}, query_text.json={'yes' if query else 'no'}",
            "YES" if gallery and query else "NO",
            "none (CLI paths)" if gallery and query else "obtain hidden test assets"),
        row("UIT/CMP training",
            "`../../data/PAB/annotation/train/pair_*.json` (cmp.yaml) + image_root + bert-base-uncased + pretrained.pth",
            ann_found,
            "WITH ADAPTER" if (pair_ok or imgs_like) else "NO",
            f"{ann_adapter}; plus checkpoints (out of scope here)"),
        row("UIT/CMP local evaluation",
            "`annotation/test/pair.json` with caption as LIST (search_test_dataset)",
            "pair.json " + ("present" if pair_json else "missing"),
            "YES" if pair_json else "NO",
            "none" if pair_json else "obtain/derive pair.json"),
        row("CLIP/BLIP-2 official score generation",
            "gallery dir + query.json (clip_infer.py/blip2_infer.py args)",
            f"gallery={'yes' if gallery else 'no (zip only?)'}, query={'yes' if query else 'no'}",
            "YES" if gallery and query else "WITH ADAPTER",
            "unzip gallery.zip locally if dir absent; sorted-listdir wrapper regardless"),
        row("official answer generation",
            "query order + gallery ids (uit/cmp/inference.py g_pids convention)",
            "query_index.txt " + ("present" if t["raw/name-masked_test-set/query_index.txt"]["exists"] else "missing"),
            "YES" if query else "NO",
            "persist query_ids.json/gallery_ids.json in score wrappers"),
    ]
    return rows


def decide_adapter(train, images):
    if train["pattern"] == "pair" and not train["indices_missing"]:
        return False, ("No annotation adapter needed: author-expected pair_*.json "
                       "layout already present. Only image-root wiring may be needed.")
    if not train["dir_used"]:
        return True, ("BLOCKED: no pair_*.json or imgs_*.json found in any candidate "
                      "annotation dir. Locate the train annotations before choosing an adapter.")
    fields_ok = train["files"] and all(
        d["status"] in ("PASS", "WARN") and d["has_image"] and d["has_caption"]
        for d in train["files"])
    if fields_ok:
        return True, (
            "Option A (recommended): build a lightweight prepared directory on local "
            "Colab disk, e.g. /content/aic_local/pab_author_layout/annotation/train/"
            "pair_{i}.json, each symlinked (or copied/converted) from imgs_{i}.json — "
            "record fields already match the author loaders (image/caption[/image_id]), "
            "so a pure rename layer suffices; add image-root symlinks matching the "
            "resolved roots so author code runs unmodified. Option B (an aic26/ dataset "
            "wrapper reading imgs_*.json directly) is not needed when a rename suffices.")
    return True, (
        "Option B leaning: sampled records do NOT directly match the author loader "
        "fields, so a symlink rename is insufficient — either convert records while "
        "writing pair_{i}.json (Option A with conversion) or wrap dataset loading "
        "under aic26/ (Option B). Re-inspect sample records before deciding.")


def decide_status(train, images, tests):
    blocking, warnings = [], []
    if not train["dir_used"]:
        blocking.append("No training annotation files (pair_*.json or imgs_*.json) found.")
    else:
        fails = [d["file"] for d in train["files"] if d["status"] == "FAIL"]
        if fails:
            blocking.append(f"Train annotation files unreadable/missing required fields: {fails[:5]}")
        if train["indices_missing"]:
            msg = (f"{len(train['indices_missing'])} of 75 expected indices missing: "
                   f"{train['indices_missing'][:10]}{'...' if len(train['indices_missing']) > 10 else ''}")
            (blocking if len(train["indices_missing"]) > 25 else warnings).append(msg)
        if train["pattern"] != "pair":
            warnings.append(f"Annotation filename pattern is `{train['pattern']}_*` — "
                            "author code expects `pair_*`; adapter required.")
        warn_fields = [d["file"] for d in train["files"] if d["status"] == "WARN"]
        if warn_fields:
            warnings.append(f"Files with field deviations (caption not str / image_id missing): "
                            f"{warn_fields[:5]}{'...' if len(warn_fields) > 5 else ''}")
    if images["sampled"]:
        if images["resolved"] == 0:
            blocking.append("0 sampled image paths resolved against any candidate root.")
        elif images["rate"] < 0.9:
            warnings.append(f"Image resolution rate {images['rate']:.0%} (<90%).")
    elif train["dir_used"]:
        warnings.append("No image paths could be sampled from annotations.")
    t = {e["path"]: e for e in tests}
    if not t["raw/name-masked_test-set"]["exists"]:
        warnings.append("Hidden test set (name-masked_test-set) not found — official "
                        "score generation blocked, training unaffected.")
    if not t["pab_original/annotation/test/pair.json"]["exists"]:
        warnings.append("annotation/test/pair.json not found — UIT local evaluation "
                        "(caption-list format) unavailable as-is.")
    if blocking:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"
    return status, blocking, warnings


# ---------------------------------------------------------------- reporting

def fmt_record(rec, maxlen=300):
    s = json.dumps(rec, ensure_ascii=False)
    return s if len(s) <= maxlen else s[:maxlen] + "...(truncated)"


def write_markdown(path: Path, ctx):
    train, images, tests = ctx["train"], ctx["images"], ctx["tests"]
    L = []
    L.append("# AIC26 PAB Competition Layout Discovery Report\n")
    L.append(f"> Generated: {ctx['timestamp']}  ")
    L.append(f"> Tool: `aic26/tools/discover_pab_competition_layout.py` (read-only)  ")
    L.append(f"> Overall status: **{ctx['status']}**\n")

    L.append("## Executive Summary\n")
    pat = train["pattern"]
    L.append(f"- Overall: **{ctx['status']}** — "
             + ("data usable for author-aligned smoke training with only a path/name adapter."
                if ctx["status"] == "PASS" else
                "data mostly usable; adapter/conversion needed (see Warnings)."
                if ctx["status"] == "WARN" else
                "required train annotations or image roots are not accessible (see Blocking Issues)."))
    L.append(f"- Train annotation pattern detected: `{pat}` "
             f"({len(train['indices_found'])}/75 expected indices"
             + (f", dir `{train['dir_used']}`" if train["dir_used"] else "") + ").")
    L.append(f"- Total train records (streamed count): "
             f"{train['total_records']}{'' if train['counts_complete'] else ' (some files not counted)'} "
             f"— paper reports 1,013,605 synthesized pairs (AIO_paper.pdf §3.2).")
    L.append(f"- Sampled image resolution: {images['resolved']}/{images['sampled']} "
             f"({images['rate']:.0%}).")
    L.append(f"- Adapter needed: **{'yes' if ctx['adapter_needed'] else 'no'}**.\n")

    L.append("## Checked Data Root\n")
    L.append(f"```\n{ctx['data_root']}\n```\n")

    L.append("## Top-Level Layout\n")
    L.append("| Entry | Exists |")
    L.append("|---|---|")
    for k, v in ctx["top_level"].items():
        L.append(f"| `{k}/` | {'yes' if v else 'NO'} |")
    L.append("")

    L.append("## Raw Dataset Layout\n")
    raw = ctx["raw"]
    L.append(f"- `raw/` exists: **{'yes' if raw['exists'] else 'NO'}**")
    if raw["exists"]:
        L.append(f"- Directories under `raw/`"
                 f"{' (first 100)' if raw['dirs_capped'] else ''}: "
                 + (", ".join(f"`{d}`" for d in raw["dirs"]) or "(none)"))
        L.append(f"- `Part 1..10` found: {len(raw['parts_found'])}/10 "
                 f"({', '.join(raw['parts_found']) or 'none'})")
        L.append(f"- `annotation/` exists: {'yes' if raw['annotation_exists'] else 'NO'}")
        L.append(f"- `name-masked_test-set/` exists: {'yes' if raw['test_set_exists'] else 'NO'}")
    L.append("")

    L.append("## Training Annotation Discovery\n")
    L.append("Searched directories:\n")
    L.append("| Directory | Exists | pair_*.json | imgs_*.json |")
    L.append("|---|---|---:|---:|")
    for e in train["search_dirs"]:
        L.append(f"| `{e['dir']}` | {'yes' if e['exists'] else 'no'} | {e['pair']} | {e['imgs']} |")
    L.append("")
    if train["dir_used"]:
        L.append(f"Using `{train['dir_used']}` — pattern `{train['pattern']}_*.json`, "
                 f"indices missing vs 0..74: {train['indices_missing'] or 'none'}\n")
        L.append("| File | Index | Size | Format | Records | image | caption | image_id | caption type | Status |")
        L.append("|---|---:|---:|---|---:|---|---|---|---|---|")
        for d in train["files"]:
            L.append(f"| `{d['file']}` | {d['index']} | {human_size(d['size'])} | {d['format']} "
                     f"| {d['record_count'] if d['record_count'] is not None else '?'} "
                     f"| {'yes' if d['has_image'] else 'NO'} | {'yes' if d['has_caption'] else 'NO'} "
                     f"| {'yes' if d['has_image_id'] else 'no'} | {d['caption_type']} | {d['status']} |")
        L.append("")
        shown = 0
        for d in train["files"]:
            if d["samples"] and shown < 3:
                L.append(f"Sample records from `{d['file']}` (keys: {d['sample_keys']}):\n")
                L.append("```json")
                for rec in d["samples"]:
                    L.append(fmt_record(rec))
                L.append("```\n")
                shown += 1
    else:
        L.append("**No train annotation files found in any searched directory.**\n")

    L.append("## Image Path Resolution\n")
    L.append(f"- Candidate roots checked (existing only): "
             + (", ".join(f"`{r}`" for r in images["candidate_roots"]) or "(none exist)"))
    L.append(f"- Sampled paths: {images['sampled']}; resolved: {images['resolved']} "
             f"({images['rate']:.0%}). Doubled-component nesting "
             f"(`root/imgs_N/imgs_N/...`) is tried automatically.")
    if images["resolved_roots"]:
        L.append("\n| Resolving root | Hits |")
        L.append("|---|---:|")
        for r in images["resolved_roots"]:
            L.append(f"| `{r['root']}` | {r['hits']} |")
    if images["resolved_examples"]:
        L.append("\nResolved examples:\n")
        L.append("```")
        for ex in images["resolved_examples"]:
            L.append(f"{ex['annotation']}: {ex['image']} -> {ex['resolved_to']}")
        L.append("```")
    if images["unresolved_examples"]:
        L.append("\nUnresolved examples:\n")
        L.append("```")
        for ex in images["unresolved_examples"]:
            L.append(f"{ex['annotation']}: {ex['image']}")
        L.append("```")
    L.append("")

    L.append("## Test and Official Annotation Discovery\n")
    L.append("| Path | Exists | Kind | Size | Format | Records | Useful for |")
    L.append("|---|---|---|---:|---|---:|---|")
    for e in tests:
        L.append(f"| `{e['path']}` | {'yes' if e['exists'] else 'NO'} | {e['kind']} "
                 f"| {human_size(e['size']) if 'size' in e else '-'} "
                 f"| {e.get('format', '-')} | {e.get('record_count', '-')} "
                 f"| {e['usefulness']} |")
    L.append("")
    for e in tests:
        if e.get("samples"):
            gt = e.get("has_ground_truth")
            gt_note = ("" if gt is None else
                       " — **contains ground-truth-like keys**" if gt else
                       " — queries only, no ground truth keys")
            L.append(f"First records of `{e['path']}` (keys: {e.get('sample_keys')}){gt_note}:\n")
            L.append("```json")
            for rec in e["samples"]:
                L.append(fmt_record(rec))
            L.append("```\n")
        elif e.get("first_lines"):
            L.append(f"First lines of `{e['path']}`:\n")
            L.append("```")
            L.extend(e["first_lines"])
            L.append("```\n")
        elif e.get("entries_preview"):
            cap = " (capped at 20)" if e.get("entries_capped") else ""
            L.append(f"Entries under `{e['path']}`{cap}: "
                     + ", ".join(f"`{n}`" for n in e["entries_preview"][:20]) + "\n")

    L.append("## Author-Code Compatibility\n")
    L.append("| Component | Author code expects | Found in current dataset | Compatible now? | Adapter needed |")
    L.append("|---|---|---|---:|---|")
    for r in ctx["compat"]:
        L.append(f"| {r['component']} | {r['expects']} | {r['found']} "
                 f"| {r['compatible']} | {r['adapter']} |")
    L.append("")

    L.append("## Adapter Recommendation\n")
    L.append(ctx["adapter_recommendation"] + "\n")
    L.append("The adapter is NOT implemented by this tool — recommendation only.\n")

    L.append("## Blocking Issues\n")
    L.extend([f"- {b}" for b in ctx["blocking"]] or ["- None."])
    L.append("")

    L.append("## Warnings\n")
    L.extend([f"- {w}" for w in ctx["warnings"]] or ["- None."])
    L.append("")

    L.append("## Next Step\n")
    if ctx["status"] == "FAIL":
        L.append("Resolve the blocking issues above (locate train annotations / image "
                 "roots on Drive), then re-run this discovery tool.")
    else:
        L.append("Implement the recommended adapter (prepared author-layout directory "
                 "on local Colab disk), then run Stage 1 — BEiT-3 + LHP tiny smoke "
                 "training per `AIC26_AUTHOR_ALIGNED_TRAINING_READY_CHECK.md`.")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_summary(path: Path, ctx):
    train, images, tests = ctx["train"], ctx["images"], ctx["tests"]
    summary = {
        "overall_status": ctx["status"],
        "data_root": str(ctx["data_root"]),
        "raw_exists": ctx["raw"]["exists"],
        "parts_found": ctx["raw"]["parts_found"],
        "train_annotation_pattern": train["pattern"],
        "train_annotation_files_found": len(train["files"]),
        "train_annotation_indices_found": train["indices_found"],
        "train_annotation_indices_missing": train["indices_missing"],
        "train_total_records_estimate": train["total_records"],
        "sample_image_resolution_rate": round(images["rate"], 4),
        "resolved_roots": [r["root"] for r in images["resolved_roots"]],
        "official_test_exists": next(e["exists"] for e in tests
                                     if e["path"] == "raw/name-masked_test-set"),
        "adapter_needed": ctx["adapter_needed"],
        "recommended_adapter": ctx["adapter_recommendation"],
        "blocking_issues": ctx["blocking"],
        "warnings": ctx["warnings"],
        "train_annotation_dir": train["dir_used"],
        "generated": ctx["timestamp"],
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="Discover the PAB competition dataset layout on Google Drive and "
                    "report how it maps to the author training code (read-only, stdlib only).")
    parser.add_argument("--data-root", default="/content/drive/MyDrive/aic2026_data",
                        help="Dataset root on mounted Drive (default: %(default)s)")
    parser.add_argument("--max-sample-per-file", type=int, default=3,
                        help="Records to sample from each annotation file (default: %(default)s)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write the report and JSON summary "
                             "(default: aic26/docs/audits next to this script)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"ERROR: data root does not exist or is not a directory: {data_root}")
        print("Run this in an environment where Google Drive is mounted "
              "(e.g. Colab after drive.mount). No report was written.")
        return 1

    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(__file__).resolve().parents[1] / "docs" / "audits")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {data_root}")
    print("Checking top-level layout ...")
    top_level = check_top_level(data_root)
    print("Inspecting raw/ ...")
    raw = inspect_raw(data_root)
    print("Discovering training annotations ...")
    train = discover_train_annotations(data_root, args.max_sample_per_file)
    print("Resolving sampled image paths ...")
    images = resolve_images(data_root, train["files"], args.max_sample_per_file)
    print("Inspecting test/official assets ...")
    tests = discover_test_assets(data_root, args.max_sample_per_file)

    compat = build_compatibility(train, images, tests)
    adapter_needed, adapter_recommendation = decide_adapter(train, images)
    status, blocking, warnings = decide_status(train, images, tests)

    ctx = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_root": data_root, "top_level": top_level, "raw": raw,
        "train": train, "images": images, "tests": tests, "compat": compat,
        "adapter_needed": adapter_needed,
        "adapter_recommendation": adapter_recommendation,
        "status": status, "blocking": blocking, "warnings": warnings,
    }

    report_path = out_dir / REPORT_NAME
    summary_path = out_dir / SUMMARY_NAME
    write_markdown(report_path, ctx)
    write_summary(summary_path, ctx)

    print()
    print(f"Report : {report_path}")
    print(f"Summary: {summary_path}")
    print(f"Train annotations: pattern={train['pattern']}, "
          f"files={len(train['files'])}, records={train['total_records']}")
    print(f"Image resolution : {images['resolved']}/{images['sampled']} ({images['rate']:.0%})")
    for b in blocking:
        print(f"BLOCKING: {b}")
    for w in warnings:
        print(f"WARNING : {w}")
    print(f"OVERALL: {status}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
