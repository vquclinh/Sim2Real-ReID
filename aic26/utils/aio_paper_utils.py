"""
Helpers for the AIO paper-faithful notebooks in `notebooks_AIO/`.

These complement `aic_colab_utils.py` (Drive/Colab infrastructure shared with the
upgrade notebooks). The functions here are paper-specific:

  * `stage_paper_layout()` — build a symlink farm so the AIC 2026 raw layout
    matches what the paper repo expects (`data/PAB/...`).
  * `clone_aio_repo()` — pull the paper repo from Drive (or fall back to a known
    workspace path on the dev box) into local SSD for fast subprocess launches.
  * `ensure_lhp_assets()` / `ensure_uit_assets()` — download or verify the
    pretrained weights & tokenizers the paper scripts need.
  * `generate_pair_jsonl()` — turn the `train_manifest_trainable.parquet` into
    paper-format `pair_<N>.json` JSONL files (one per source `imgs_<N>.json`).
  * `get_sorted_gallery_paths()` — pinned gallery ordering so the score matrices
    saved by NB 03/04/05 align column-for-column at ensemble time.
  * `drive_sync_thread()` — background daemon used by the training notebooks to
    copy fresh checkpoints from local SSD to Drive every minute.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Symlink farm: raw → paper-expected layout
# ---------------------------------------------------------------------------

def _safe_symlink(src: Path, link: Path) -> None:
    """Create `link → src` if `link` does not already exist (idempotent)."""
    if link.is_symlink() or link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(src)


def stage_paper_layout(paths: dict) -> Path:
    """Build symlink farm `<local_root>/data/PAB/...` that mirrors the paper layout.

    Layout sources (from `paths['input_root']` which is `<local>/raw/`):
      raw/Part X/imgs_N/imgs_N/{goal,full,wentwrong}/*           (75 imgs_N)
      raw/annotation/train/imgs_<N>.json                          (75 JSONL)
      raw/name-masked_test-set/gallery/gallery/*.jpg              (36,773 nested)
      raw/name-masked_test-set/query_text.json                    (1,978)

    Layout produced:
      <local>/data/PAB/train/imgs_<N>/{goal,full,wentwrong}/*    (links to inner imgs_N)
      <local>/data/PAB/annotation/train/pair_<N>.json             (renamed link)
      <local>/data/PAB/annotation/test/pair.json                  (stub: → query_text.json)
      <local>/data/PAB/name-masked_test-set/gallery               (flat link)
      <local>/data/PAB/name-masked_test-set/query.json            (link → query_text.json)

    Idempotent: re-running on a fully staged layout is a no-op (~50 ms).
    Returns the resolved `<local>/data/PAB/` path.
    """
    input_root = Path(paths['input_root'])
    pab = Path(paths['local_root']) / 'data' / 'PAB'
    pab.mkdir(parents=True, exist_ok=True)

    # 1) train/imgs_<N> → raw/Part X/imgs_<N>/imgs_<N>/
    train_root = pab / 'train'
    train_root.mkdir(exist_ok=True)
    linked = 0
    for part_dir in sorted(input_root.glob('Part *')):
        for outer in sorted(part_dir.glob('imgs_*')):
            inner = outer / outer.name
            if not inner.is_dir():
                continue
            link = train_root / outer.name
            if not (link.is_symlink() or link.exists()):
                link.symlink_to(inner)
                linked += 1

    # Fallback: some sessions extract directly to raw/train/imgs_<N>/...
    raw_train = input_root / 'train'
    if raw_train.is_dir():
        for child in sorted(raw_train.iterdir()):
            if child.name.startswith('imgs_'):
                link = train_root / child.name
                if not (link.is_symlink() or link.exists()):
                    link.symlink_to(child)
                    linked += 1

    # 2) annotation/train/imgs_<N>.json → annotation/train/pair_<N>.json
    src_ann = input_root / 'annotation' / 'train'
    dst_ann = pab / 'annotation' / 'train'
    dst_ann.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_ann.glob('imgs_*.json')):
        m = re.match(r'imgs_(\d+)\.json', src.name)
        if not m:
            continue
        idx = m.group(1)
        _safe_symlink(src, dst_ann / f'pair_{idx}.json')

    # 3) annotation/test/pair.json — stub for code paths that init the train-time
    #    eval dataset. AIC test set has no GT, so we just point it at the query
    #    file; nothing actually reads this for AIC submission.
    test_ann_dir = pab / 'annotation' / 'test'
    test_ann_dir.mkdir(parents=True, exist_ok=True)
    _safe_symlink(
        input_root / 'name-masked_test-set' / 'query_text.json',
        test_ann_dir / 'pair.json',
    )

    # 4) name-masked_test-set/gallery (flat) + query.json
    test_root = pab / 'name-masked_test-set'
    test_root.mkdir(exist_ok=True)
    nested_gallery = input_root / 'name-masked_test-set' / 'gallery' / 'gallery'
    flat_gallery = input_root / 'name-masked_test-set' / 'gallery'
    if nested_gallery.is_dir():
        _safe_symlink(nested_gallery, test_root / 'gallery')
    elif flat_gallery.is_dir():
        _safe_symlink(flat_gallery, test_root / 'gallery')
    _safe_symlink(
        input_root / 'name-masked_test-set' / 'query_text.json',
        test_root / 'query.json',
    )

    return pab


def get_sorted_gallery_paths(pab_root: Path) -> list[Path]:
    """Single source of truth for gallery ordering across NB 03/04/05/06.

    Paper's `search_inference_dataset` does `os.listdir(image_root)` which has
    arbitrary order; sorting is the only way to keep score-matrix columns
    aligned between models. Gallery IDs are the filename stems.
    """
    gallery = Path(pab_root) / 'name-masked_test-set' / 'gallery'
    return sorted(p for p in gallery.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})


# ---------------------------------------------------------------------------
# AIO repo clone
# ---------------------------------------------------------------------------

AIO_REPO_GIT_URL = (
    'https://github.com/AIVIETNAM-Hub/'
    'Hybrid-Unified-and-Iterative-A-Novel-Framework-for-Text-based-Person-Anomaly-Retrieval.git'
)


def clone_aio_repo(local_root: Path, drive_root: Path | None = None,
                   workspace_fallback: Path | None = None,
                   subdir: str = 'aio_repo',
                   git_url: str = AIO_REPO_GIT_URL,
                   git_ref: str | None = None) -> Path:
    """Make the paper repo available at `<local_root>/<subdir>/`.

    Resolution order (first match wins):
      1. Already present at target — no-op.
      2. Drive cache at `<drive_root>/aio_repo/` — rsync into local SSD.
      3. Workspace path (dev box) — rsync.
      4. `git clone` from `git_url` (default: AIVIETNAM-Hub upstream).

    Once cloned via git, also mirror to `<drive_root>/aio_repo/` (if drive
    available) so subsequent sessions can use the faster rsync path.

    Returns the local repo root. Will raise if all four strategies fail.
    """
    local_root = Path(local_root)
    target = local_root / subdir
    if target.exists() and (target / 'uit' / 'cmp' / 'Search.py').exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

    # 2/3) try rsync from local copies first (fastest)
    candidates: list[Path] = []
    if drive_root is not None:
        candidates.append(Path(drive_root) / 'aio_repo')
    if workspace_fallback is not None:
        candidates.append(Path(workspace_fallback))
    src = next(
        (c for c in candidates
         if c.exists() and (c / 'uit' / 'cmp' / 'Search.py').exists()),
        None,
    )
    if src is not None:
        target.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(['rsync', '-a', f'{src}/', f'{target}/'])
        return target

    # 4) fall back to git clone
    print(f'[aio_repo] cloning {git_url} → {target}')
    if target.exists():
        # might be a partial dir from a failed previous attempt
        shutil.rmtree(target)
    cmd = ['git', 'clone', '--depth', '1']
    if git_ref:
        cmd += ['--branch', git_ref]
    cmd += [git_url, str(target)]
    subprocess.check_call(cmd)
    assert (target / 'uit' / 'cmp' / 'Search.py').exists(), (
        f'git clone succeeded but {target}/uit/cmp/Search.py missing — '
        'check repo layout has changed upstream.'
    )

    # Mirror to Drive for next session's rsync path (best-effort)
    if drive_root is not None:
        drive_target = Path(drive_root) / 'aio_repo'
        try:
            drive_target.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(['rsync', '-a', '--exclude', '.git',
                                   f'{target}/', f'{drive_target}/'])
            print(f'[aio_repo] mirrored to Drive at {drive_target}')
        except Exception as exc:
            print(f'[aio_repo] WARNING: Drive mirror failed ({exc}). '
                  'Next session will re-clone from GitHub.')

    return target


# ---------------------------------------------------------------------------
# Asset download (BEiT-3 weights, BERT, Swin-B)
# ---------------------------------------------------------------------------

_BEIT3_SPM_URL = 'https://github.com/addf400/files/releases/download/beit3/beit3.spm'
_BEIT3_CKPT_URL = (
    'https://github.com/addf400/files/releases/download/beit3/'
    'beit3_large_patch16_384_coco_retrieval.pth'
)
_SWIN_B_URL = (
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/'
    'swin_base_patch4_window7_224_22k.pth'
)


def _download(url: str, dst: Path, force: bool = False) -> Path:
    dst = Path(dst)
    if dst.exists() and not force and dst.stat().st_size > 1024:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    print(f'[download] {url} → {dst} ({dst.stat().st_size if dst.exists() else 0} bytes existing)')
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def ensure_lhp_assets(repo_root: Path) -> dict:
    """Download `beit3.spm` + `beit3_large_patch16_384_coco_retrieval.pth` into
    `<repo>/checkpoint/lhp/` (paper-expected layout). Returns paths dict.
    """
    repo_root = Path(repo_root)
    lhp_dir = repo_root / 'checkpoint' / 'lhp'
    spm = _download(_BEIT3_SPM_URL, lhp_dir / 'beit3.spm')
    ckpt = _download(_BEIT3_CKPT_URL, lhp_dir / 'beit3_large_patch16_384_coco_retrieval.pth')
    return {'spm': spm, 'finetune_ckpt': ckpt, 'lhp_dir': lhp_dir}


def ensure_uit_assets(repo_root: Path, drive_root: Path | None = None) -> dict:
    """Verify / fetch UIT init weights into `<repo>/checkpoint/`.

    The paper offers two init paths:
      Option 1 — `pretrained.pth` (paper Google Drive). Requires OAuth, so user
                 must upload manually to `<drive>/aio_repo/checkpoint/pretrained.pth`.
                 This function only checks for it and copies into local repo.
      Option 2 — initialise Swin-B + bert-base-uncased separately. Downloaded
                 automatically here.

    Returns paths dict with optional `pretrained_pth` (None if missing).
    """
    repo_root = Path(repo_root)
    uit_ckpt_dir = repo_root / 'checkpoint'
    uit_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Option 1: copy paper's pretrained.pth from Drive if present
    paper_pth_local = uit_ckpt_dir / 'pretrained.pth'
    paper_pth_drive = None
    if drive_root is not None:
        drive_candidates = [
            Path(drive_root) / 'aio_repo' / 'checkpoint' / 'pretrained.pth',
            Path(drive_root) / 'pretrained' / 'uit_pretrained.pth',
        ]
        paper_pth_drive = next((p for p in drive_candidates if p.exists()), None)
    if not paper_pth_local.exists() and paper_pth_drive is not None:
        shutil.copy2(paper_pth_drive, paper_pth_local)
        print(f'[uit] copied {paper_pth_drive} → {paper_pth_local}')

    # Option 2: Swin-B + BERT
    swin_pth = _download(_SWIN_B_URL, uit_ckpt_dir / 'swin_base_patch4_window7_224_22k.pth')

    bert_dir = uit_ckpt_dir / 'bert-base-uncased'
    if not (bert_dir / 'pytorch_model.bin').exists() and not (bert_dir / 'model.safetensors').exists():
        try:
            from huggingface_hub import snapshot_download  # type: ignore
            snapshot_download(repo_id='bert-base-uncased', local_dir=str(bert_dir),
                              local_dir_use_symlinks=False)
            print(f'[uit] BERT downloaded → {bert_dir}')
        except Exception as exc:
            print(f'[uit] WARNING: BERT download failed ({exc}). Either install '
                  'huggingface_hub or place bert-base-uncased manually at '
                  f'{bert_dir}.')

    return {
        'pretrained_pth': paper_pth_local if paper_pth_local.exists() else None,
        'swin_pth': swin_pth,
        'bert_dir': bert_dir,
    }


# ---------------------------------------------------------------------------
# JSONL generation from manifest
# ---------------------------------------------------------------------------

def generate_pair_jsonl(manifest_parquet: Path, out_dir: Path,
                        group_col: str = 'annotation_file') -> list[Path]:
    """Read `train_manifest_trainable.parquet` and write paper-format JSONL files.

    Output: `out_dir/pair_<N>.json`, one per source `imgs_<N>.json`. Schema per
    line matches what `lhp_2/beit3/datasets.py` and `uit/cmp/dataset/...` read:
        {"image": "<rel path under data/PAB>", "caption": "<text>", "image_id": <int>,
         "scene": "<str>", "normal"|"anomaly": "<action>"}

    The `image_id` is taken straight from the manifest (string like "0_0" works
    for both loaders because BEiT-3 overwrites it with a counter and UIT uses
    it as a string ID for the contrastive loss bucketing).
    """
    import pandas as pd  # local import — heavy

    df = pd.read_parquet(manifest_parquet)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in df.itertuples(index=False):
        item = {
            'image': getattr(row, 'annotation_image'),
            'caption': getattr(row, 'caption'),
            'image_id': getattr(row, 'image_id'),
            'scene': getattr(row, 'scene', ''),
        }
        label_type = getattr(row, 'label_type', 'normal')
        action = getattr(row, 'action', '')
        item['anomaly' if label_type == 'anomaly' else 'normal'] = action
        groups[getattr(row, group_col)].append(item)

    written: list[Path] = []
    for src_name in sorted(groups.keys(), key=lambda s: int(re.search(r'(\d+)', s).group(1))):
        m = re.search(r'(\d+)', src_name)
        idx = m.group(1)
        out = out_dir / f'pair_{idx}.json'
        with open(out, 'w', encoding='utf-8') as f:
            for item in groups[src_name]:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        written.append(out)
    print(f'[jsonl] wrote {len(written)} pair_*.json → {out_dir}')
    return written


# ---------------------------------------------------------------------------
# Background checkpoint sync
# ---------------------------------------------------------------------------

def drive_sync_thread(stop_event: threading.Event, src_dir: Path, dst_dir: Path,
                      patterns: Iterable[str] = ('checkpoint*.pth', 'checkpoint-*.pth',
                                                 'checkpoint_*.pth', 'log.txt'),
                      poll_sec: float = 60.0) -> threading.Thread:
    """Spawn a daemon that copies new checkpoint files from `src_dir` to
    `dst_dir` every `poll_sec` seconds. Returns the started thread handle —
    caller should `stop_event.set(); thread.join()` at the end of training.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    def _loop() -> None:
        while not stop_event.is_set():
            for pattern in patterns:
                for p in src_dir.glob(pattern):
                    key = f'{p.name}:{p.stat().st_mtime_ns}' if p.exists() else p.name
                    if key in seen:
                        continue
                    try:
                        shutil.copy2(p, dst_dir / p.name)
                        seen.add(key)
                        print(f'[drive-sync] {p.name} → {dst_dir}')
                    except Exception as exc:
                        print(f'[drive-sync] {p.name} failed: {exc}')
            stop_event.wait(poll_sec)

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    return th


def latest_checkpoint(dir_path: Path, patterns: Iterable[str] = ('checkpoint-*.pth',
                                                                  'checkpoint_*.pth')) -> Path | None:
    """Return the checkpoint with the highest epoch number, or None."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return None
    best: tuple[int, Path] | None = None
    for pattern in patterns:
        for p in dir_path.glob(pattern):
            m = re.search(r'(\d+)', p.stem)
            if not m:
                continue
            ep = int(m.group(1))
            if best is None or ep > best[0]:
                best = (ep, p)
    return best[1] if best else None
