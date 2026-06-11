"""Shared utilities for AIC2026 Track 4 feature extraction on Colab A100.

Both notebooks `01a_pe_g14_features.ipynb` and `01b_vitpose_features.ipynb`
share this module to:
  1. Bootstrap the environment: mount Drive, cache Kaggle dataset, rsync to
     the Colab local SSD.
  2. Pick A100 device, set TF32 / cuDNN benchmark / Flash SDPA.
  3. Save NPZ atomically + async sync to Drive in chunks.
  4. List existing chunks (Drive + local) for resume logic.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

_DEFAULT_KAGGLE_DATASET = "vnhtbo/pab-eccv26-track4-annotation-testset"
_DEFAULT_KAGGLE_MANIFEST = "vnhtbo/manifests"
# _DEFAULT_KAGGLE_TRAIN_IMAGES: tuple[str, ...] = (
#     "vnhtbo/pab-eccv26-track4-train-webp-part-01-05",
#     "vnhtbo/pab-eccv26-track4-train-webp-part-06-10",
# )


def _on_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def _materialize_kaggle_token(
    text: str,
    target: Path,
    fallback_username: str = "anonymous",
) -> bool:
    """Write kaggle.json from `text` (JSON or raw access token).

    Returns True on success. Supports:
      - JSON form `{"username":..., "key":...}` (standard kaggle.json).
      - Raw access token `KGAT_xxx` or plain string (1 line) — will be
        wrapped into `{"username": KAGGLE_USERNAME or "anonymous", "key": <token>}`.
    """
    text = text.strip()
    if not text:
        return False
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if "key" in obj and obj["key"]:
                obj.setdefault("username", fallback_username)
                target.write_text(json.dumps(obj))
                os.environ["KAGGLE_USERNAME"] = obj["username"]
                os.environ["KAGGLE_KEY"] = obj["key"]
                return True
        except json.JSONDecodeError:
            return False
        return False

    # Raw token (KGAT_... or plain alphanum). Wrap into JSON.
    username = os.environ.get("KAGGLE_USERNAME") or fallback_username
    target.write_text(json.dumps({"username": username, "key": text}))
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = text
    return True


def _ensure_kaggle_credentials(
    drive_root: Path,
    kaggle_token_path: str | None,
) -> None:
    """Set KAGGLE_CONFIG_DIR and ensure kaggle.json exists.

    Token lookup order (each candidate may be JSON or raw `KGAT_*`):
      1. Arg `kaggle_token_path` (supports ~ expand).
      2. `drive_root/.kaggle/kaggle.json`.
      3. `~/.kaggle/kaggle.json`.
      4. `~/.kaggle/access_token` (new Kaggle access token format).
      5. Env var KAGGLE_USERNAME + KAGGLE_KEY (already set externally).
    """
    cfg_dir = Path.home() / ".kaggle"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    target = cfg_dir / "kaggle.json"

    candidates: list[Path] = []
    if kaggle_token_path:
        candidates.append(Path(os.path.expanduser(kaggle_token_path)))
    candidates.extend([
        drive_root / ".kaggle" / "kaggle.json",
        drive_root / ".kaggle" / "access_token",
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".kaggle" / "access_token",
    ])

    for src in candidates:
        if not src.is_file():
            continue
        try:
            text = src.read_text()
        except OSError:
            continue
        if _materialize_kaggle_token(text, target):
            print(f"[setup] Kaggle credentials loaded from {src}")
            break

    if target.exists():
        os.chmod(target, 0o600)
        os.environ["KAGGLE_CONFIG_DIR"] = str(cfg_dir)
        return

    if os.environ.get("KAGGLE_KEY"):
        return

    raise FileNotFoundError(
        "Kaggle credentials not found. Two options:\n"
        f"  1) Upload kaggle.json to {drive_root}/.kaggle/kaggle.json\n"
        "  2) Create a raw access token: \n"
        "     !mkdir -p ~/.kaggle && echo <KGAT_...> > ~/.kaggle/access_token\n"
        "     && chmod 600 ~/.kaggle/access_token\n"
        "Or pass kaggle_token_path='~/.kaggle/access_token'."
    )


def _run(cmd: list[str]) -> None:
    print("[setup]", " ".join(cmd))
    subprocess.check_call(cmd)


def _pip_install_quiet(*packages: str) -> None:
    _run([sys.executable, "-m", "pip", "install", "-q", *packages])


def _ensure_kagglehub() -> None:
    try:
        import kagglehub  # noqa: F401
    except ImportError:
        _pip_install_quiet("kagglehub>=0.3.0")


def _rsync(src: Path, dst: Path) -> None:
    """rsync src/ -> dst/ — copy only missing/changed files.

    Falls back to `shutil.copytree(..., dirs_exist_ok=True)` if the system
    does not have `rsync` (rare on Colab).
    """
    src.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        subprocess.check_call(
            ["rsync", "-a", "--info=progress2", f"{src}/", f"{dst}/"]
        )
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _marker_for(slug: str, dest: Path) -> Path:
    safe = slug.replace("/", "__")
    return dest / f".synced_{safe}"


def _download_kaggle_dataset(slug: str, dest: Path, force: bool = False) -> Path:
    """Download 1 Kaggle dataset -> `dest` and merge structure.

    Multiple datasets can be merged into the same `dest` (e.g. annotation +
    train-webp-part-01-05 + train-webp-part-06-10 share the `raw/` directory).
    Each slug has its own `.synced_<slug>` marker file to skip when already synced.
    """
    _ensure_kagglehub()
    import kagglehub

    dest.mkdir(parents=True, exist_ok=True)
    marker = _marker_for(slug, dest)
    if marker.exists() and not force:
        print(f"[setup] {slug} already synced previously — skip")
        return dest

    print(f"[setup] Downloading {slug} …")
    cached = Path(kagglehub.dataset_download(slug))
    print(f"[setup] kagglehub cache: {cached}")
    _rsync(cached, dest)
    marker.write_text(f"synced from {cached}\n")
    print(f"[setup] Synced {slug} → {dest}")
    return dest


def _robust_drive_mkdir(
    path: Path,
    retries: int = 3,
    sleep_base: float = 1.0,
) -> bool:
    """mkdir on Drive FUSE with multiple strategies + retry.

    Drive FUSE Errno 5 is usually caused by orphan folders in trash or FUSE state
    corruption — pure retry does NOT fix it. This helper tries 3 strategies
    in order (each attempt in the loop):
      1. Python `pathlib.mkdir(parents=True, exist_ok=True)`
      2. Shell `mkdir -p` (via subprocess — sometimes bypasses FUSE quirks)
      3. Check `path.exists()` after mkdir error (FUSE sometimes succeeds
         but still raises — the folder was actually created)

    Returns:
        True if path ultimately exists.
        False if all retries fail — caller should skip the Drive op.
    """
    # Fast path: already exists
    try:
        if path.exists():
            return True
    except OSError:
        pass

    for attempt in range(retries):
        last_err = None

        # Strategy 1: Python mkdir
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            last_err = e

        # Strategy 2: shell mkdir -p (different syscall path)
        try:
            subprocess.check_call(
                ["mkdir", "-p", str(path)],
                stderr=subprocess.DEVNULL,
            )
            if path.exists():
                return True
        except (subprocess.CalledProcessError, OSError) as e:
            last_err = e

        # Strategy 3: re-check existence after both strategies
        try:
            if path.exists():
                return True
        except OSError:
            pass

        if attempt == retries - 1:
            print(f"[drive] mkdir {path} failed after {retries} retries (strategies 1+2+3): {last_err}")
            return False

        wait = sleep_base * (2 ** attempt)
        print(f"[drive] mkdir {path} attempt {attempt+1}/{retries} → {last_err}; sleep {wait:.0f}s")
        time.sleep(wait)

    return False


def _ensure_rclone(
    config_source: str | os.PathLike | None = None,
    drive_root: str | os.PathLike | None = None,
) -> Path:
    """Install rclone (if missing) and write config to `~/.config/rclone/rclone.conf`.

    `config_source` can be:
      - Path to an rclone.conf / rclone.txt file (local or Drive)
      - Raw INI text (containing `[remote_name]`)
      - None -> auto-search `<drive_root>/rclone.txt|conf`, `/content/rclone.txt|conf`
              or `~/Documents/AIC2026/rclone.txt` (dev fallback)

    Returns path to the written rclone.conf.
    """
    # 1) Install rclone if needed
    if not shutil.which("rclone"):
        print("[rclone] installing via curl https://rclone.org/install.sh ...")
        subprocess.check_call(
            "curl -fsSL https://rclone.org/install.sh | sudo bash",
            shell=True,
        )

    # 2) Resolve config source
    config_text: str | None = None
    if config_source is None:
        candidates = []
        if drive_root is not None:
            for name in ("rclone.conf", "rclone.txt"):
                candidates.append(Path(drive_root) / name)
        candidates += [
            Path("/content/rclone.conf"),
            Path("/content/rclone.txt"),
            Path.home() / "rclone.conf",
            Path.home() / "Documents/AIC2026/rclone.txt",
        ]
        for c in candidates:
            if c.exists():
                config_source = c
                print(f"[rclone] auto-found config: {c}")
                break
        if config_source is None:
            raise FileNotFoundError(
                "rclone config not found. Upload rclone.txt to Drive "
                f"({drive_root}/rclone.txt) or pass config_source='...'."
            )

    if isinstance(config_source, (str, os.PathLike)) and Path(config_source).exists():
        config_text = Path(config_source).read_text()
    elif isinstance(config_source, str) and config_source.lstrip().startswith("["):
        config_text = config_source
    else:
        raise ValueError(f"Invalid rclone config_source: {config_source!r}")

    # 3) Write to standard rclone config location
    cfg_dir = Path.home() / ".config" / "rclone"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "rclone.conf"
    cfg_path.write_text(config_text)
    cfg_path.chmod(0o600)

    # 4) Verify
    out = subprocess.check_output(["rclone", "listremotes"], text=True).strip()
    print(f"[rclone] configured remotes: {out or '(none — config invalid)'}")
    return cfg_path


def _fuse_path_to_rclone(
    fuse_path: Path,
    remote: str = "gdrive",
    drive_mount: Path = Path("/content/drive/MyDrive"),
) -> str:
    """Convert a Drive FUSE path (or symlink to one) → `gdrive:relative/path`.

    Resolves symlinks (e.g. `/content/aic_local/raw → /content/drive/MyDrive/.../raw`).
    """
    p = Path(fuse_path).resolve()
    drive_mount = Path(drive_mount).resolve()
    if not str(p).startswith(str(drive_mount)):
        raise ValueError(
            f"Path {p} not under Drive mount {drive_mount} — can't map to rclone remote."
        )
    rel = p.relative_to(drive_mount)
    return f"{remote}:{rel.as_posix()}"


def _shell_count_dir(path: Path, retries: int = 5, sleep_base: float = 2.0) -> int:
    """Count entries in a directory using shell `ls` (more robust on Drive FUSE
    than Python's iterdir which often raises Errno 5 on shortcut folders).
    Returns -1 if all retries fail.
    """
    cmd = ["bash", "-c", f'ls -1 "{path}" 2>/dev/null | wc -l']
    for attempt in range(retries):
        try:
            out = subprocess.check_output(cmd, timeout=300, text=True)
            n = int(out.strip())
            if n > 0:
                return n
        except Exception as exc:
            print(f"[shell-count] attempt {attempt+1}/{retries} failed: {exc}")
        time.sleep(sleep_base * (2 ** attempt))
    return -1


def stage_test_set_local(
    paths: dict,
    retries: int = 3,
    sleep_base: float = 5.0,
    use_tar_stream: bool = True,
    use_rclone: bool = False,
    rclone_remote: str = "gdrive",
    rclone_config: str | os.PathLike | None = None,
    rclone_transfers: int = 32,
    rclone_checkers: int = 16,
    drive_mount: str | os.PathLike = "/content/drive/MyDrive",
) -> dict:
    """Copy `<input_root>/name-masked_test-set/` from Drive to local SSD.

    Drive FUSE raises `OSError: [Errno 5]` when iterating folders with many files.
    This helper avoids the issue by:

    1) Counting files via shell `ls | wc -l` (more fault-tolerant than Python iterdir).
    2) Stream copying with `tar | tar` over a pipe (1 metadata read + sequential
       file reads) — faster than rsync (rsync stats each file = 36k × 50ms over
       FUSE = 30+ minutes just to enumerate before transfer).
    3) Copying small query JSON files separately via `shutil.copy2`.

    Idempotent — re-running does not re-copy if local count already matches.
    """
    input_root = Path(paths["input_root"])
    local_root = Path(paths["local_root"])
    drive_test = input_root / "name-masked_test-set"

    local_test = local_root / "test_set_local"
    local_gallery = local_test / "gallery"
    local_gallery.mkdir(parents=True, exist_ok=True)

    drive_gallery_nested = drive_test / "gallery" / "gallery"
    drive_gallery_flat = drive_test / "gallery"
    drive_gallery = drive_gallery_nested if drive_gallery_nested.exists() else drive_gallery_flat

    # Count via shell (Drive FUSE-friendly)
    print(f"[stage-test] counting files in {drive_gallery} (shell ls)...")
    t0 = time.time()
    target_count = _shell_count_dir(drive_gallery, retries=retries, sleep_base=sleep_base)
    print(f"[stage-test] Drive gallery count: {target_count} ({time.time()-t0:.1f}s)")

    have_count = _shell_count_dir(local_gallery, retries=2, sleep_base=1.0)
    if have_count < 0:
        have_count = 0

    if target_count > 0 and have_count >= target_count:
        print(f"[stage-test] gallery local already full ({have_count}/{target_count}) — skip copy")
    else:
        print(f"[stage-test] copy {drive_gallery} → {local_gallery} "
              f"(have={have_count}, target={target_count if target_count > 0 else '?'})")

        # Pick transport: rclone (parallel, ~5-10× faster) > tar-stream > rsync.
        if use_rclone:
            _ensure_rclone(config_source=rclone_config, drive_root=paths.get("drive_root"))
            remote_path = _fuse_path_to_rclone(
                drive_gallery, remote=rclone_remote, drive_mount=Path(drive_mount),
            )
            cmd_list = [
                "rclone", "copy", remote_path, str(local_gallery),
                "--transfers", str(rclone_transfers),
                "--checkers",  str(rclone_checkers),
                "--drive-chunk-size", "64M",
                "--fast-list",
                "--stats-one-line",       # avoid TTY progress bar that breaks in Jupyter
                "--stats", "5s",
                "--verbose",              # log each file as it transfers
            ]
            transport_desc = (
                f"rclone {remote_path} → {local_gallery} "
                f"(transfers={rclone_transfers}, checkers={rclone_checkers})"
            )
        elif use_tar_stream:
            # tar | tar streams: 1 readdir on Drive side + sequential file reads;
            # no per-file stat. ~10× faster than rsync but still bottlenecked by
            # Drive FUSE per-file open latency. rclone is much faster.
            cmd_list = None
            cmd_shell = (
                f'tar -cf - -C "{drive_gallery}" . | '
                f'tar -xf - -C "{local_gallery}"'
            )
            transport_desc = f"tar-stream: {cmd_shell}"
        else:
            cmd_list = None
            cmd_shell = None
            transport_desc = f"rsync {drive_gallery} → {local_gallery}"

        def _run_streaming(args, shell: bool = False) -> int:
            """Stream subprocess stdout+stderr line-by-line to Python stdout
            (unbuffered) so Jupyter/Colab show realtime progress.
            """
            proc = subprocess.Popen(
                args, shell=shell,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
            proc.wait()
            return proc.returncode

        last_rc = None
        for attempt in range(retries):
            print(f"[stage-test] attempt {attempt+1}/{retries}: {transport_desc}", flush=True)
            if use_rclone:
                rc = _run_streaming(cmd_list)
            elif use_tar_stream:
                rc = _run_streaming(cmd_shell, shell=True)
            else:
                try:
                    _rsync(drive_gallery, local_gallery)
                    rc = 0
                except subprocess.CalledProcessError as exc:
                    rc = exc.returncode
            if rc == 0:
                last_rc = 0
                break
            last_rc = rc
            wait = sleep_base * (2 ** attempt)
            print(f"[stage-test] copy attempt {attempt+1}/{retries} rc={rc}, "
                  f"retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
        if last_rc != 0:
            raise RuntimeError(f"[stage-test] copy failed after {retries} attempts")

        # Verify count after copy
        have_after = _shell_count_dir(local_gallery, retries=2, sleep_base=1.0)
        print(f"[stage-test] copy done. Local count: {have_after}")

    # Mirror small files (query_text.json, query_index.txt) — quick best-effort
    for fname in ("query_text.json", "query_index.txt", "query.json"):
        src = drive_test / fname
        dst = local_test / fname
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(src, dst)
                print(f"[stage-test] copied {fname}")
            except Exception as exc:
                print(f"[stage-test] copy {fname} failed: {exc} — keep Drive path")

    # Update paths so callers see local versions
    paths = dict(paths)
    paths["test_dir"] = local_test
    paths["gallery_dir"] = local_gallery
    print(f"[stage-test] test_dir    → {local_test}")
    print(f"[stage-test] gallery_dir → {local_gallery}")
    return paths


def setup_aic2026_environment(
    drive_root: str | os.PathLike = "/content/drive/MyDrive/aic2026_data",
    local_root: str | os.PathLike = "/content/aic_local",
    force_redownload: bool = False,
    mount_drive: bool = True,
    sync_existing_output: bool = True,
    use_kaggle: bool = False,
    kaggle_dataset: str = _DEFAULT_KAGGLE_DATASET,
    kaggle_manifest_dataset: str = _DEFAULT_KAGGLE_MANIFEST,
    kaggle_token_path: str | None = None,
    copy_raw_to_local: bool = False,
) -> dict[str, Path]:
    """Full bootstrap for a single Colab session.

    Returns paths dict containing:
      drive_root, local_root, input_root, manifest_dir, output_root,
      drive_output_root, annotation_train_dir, test_dir, gallery_dir.

    `output_root` always points to local SSD (fast writes).
    `drive_output_root` is where chunks are synced to for persistence.

    **Default flow (no Kaggle)**: dataset is already present at `<drive_root>/raw/` +
    `<drive_root>/manifest/` (or `manifests/`). This function only symlinks those
    folders into `<local_root>/raw/` and `<local_root>/manifests/` so the rest of
    the code (reading via `paths['input_root']`, `paths['manifest_dir']`) does not
    need to change.

    **Kaggle fallback**: set `use_kaggle=True` to re-enable the old pipeline
    (download via kagglehub). Requires kaggle.json or access_token.

    **Local SSD speed**: for training notebooks (reading 1M+ images many times),
    set `copy_raw_to_local=True` to rsync raw from Drive to local SSD —
    slow (30-60 min) on first run but training is ~5-10× faster.
    """
    drive_root = Path(drive_root)
    local_root = Path(local_root)

    # Step 1: Mount Drive ----------------------------------------------------
    if mount_drive and _on_colab():
        try:
            from google.colab import drive  # type: ignore

            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")
            else:
                print("[setup] Drive already mounted.")
        except ImportError:
            warnings.warn("google.colab.drive not available — skipping mount.")

    drive_root.mkdir(parents=True, exist_ok=True)
    local_root.mkdir(parents=True, exist_ok=True)

    # Step 2: Kaggle credentials (only when use_kaggle=True) ------------------
    if use_kaggle:
        _ensure_kaggle_credentials(drive_root, kaggle_token_path)

    # Step 3: Restore raw + manifests from Drive -> local SSD ------------------
    drive_raw = drive_root / "raw"
    # Drive folder may be named 'manifest' (singular, per user screenshot) or
    # 'manifests' (legacy). Pick whichever exists.
    drive_manifests = next(
        (p for p in (drive_root / "manifest", drive_root / "manifests") if p.exists()),
        drive_root / "manifest",
    )
    local_raw = local_root / "raw"
    local_manifests = local_root / "manifests"

    if force_redownload:
        for p in (local_raw, local_manifests):
            if p.is_symlink():
                p.unlink()
            elif p.exists():
                shutil.rmtree(p)

    raw_paths = {"drive_root": drive_root, "local_root": local_root}

    # Step 3a: Restore raw
    if local_raw.exists() and (local_raw.is_symlink() or any(local_raw.iterdir())):
        print(f"[setup] raw already ready: {local_raw}")
    elif (drive_root / "raw_tar_parts" / ".tar_complete").exists():
        # FAST PATH (legacy session with tar-split mirror)
        restore_raw_from_tar_split(raw_paths, force=force_redownload)
    elif drive_raw.exists() and any(drive_raw.iterdir()):
        if copy_raw_to_local:
            print(f"[setup] rsync raw from Drive → {local_raw} (may take 30-60 min)")
            _rsync(drive_raw, local_raw)
        else:
            # Symlink Drive -> local. Instant, reads directly via Drive FUSE.
            # OK for inference / zero-shot; training should use copy_raw_to_local=True.
            if local_raw.exists():
                local_raw.rmdir() if not any(local_raw.iterdir()) else shutil.rmtree(local_raw)
            local_raw.symlink_to(drive_raw)
            print(f"[setup] symlink {local_raw} → {drive_raw}")
    elif use_kaggle:
        pass  # will be filled by Step 4 Kaggle download
    else:
        raise FileNotFoundError(
            f"Raw dataset not found.\n"
            f"  - Drive: {drive_raw} (does not exist or is empty)\n"
            f"  - Local: {local_raw} (does not exist or is empty)\n"
            f"Please upload the dataset to {drive_raw}, or set use_kaggle=True to "
            f"download via Kaggle."
        )

    # Step 3b: Restore manifests
    if local_manifests.exists() and (local_manifests.is_symlink() or any(local_manifests.iterdir())):
        print(f"[setup] manifests already ready: {local_manifests}")
    elif drive_manifests.exists() and any(drive_manifests.iterdir()):
        # Manifests are lightweight (~100MB parquets) — symlink or rsync both OK
        if local_manifests.exists():
            local_manifests.rmdir() if not any(local_manifests.iterdir()) else shutil.rmtree(local_manifests)
        local_manifests.symlink_to(drive_manifests)
        print(f"[setup] symlink {local_manifests} → {drive_manifests}")
    elif use_kaggle:
        pass  # will be filled by Step 4 Kaggle download
    else:
        print(f"[setup] WARNING: No manifest at {drive_manifests} or {local_manifests} "
              f"— pipelines depending on manifests will fail. Zero-shot pipeline does not need it.")

    # Step 4: Kaggle download fallback (only when use_kaggle=True) ------------
    if use_kaggle:
        _download_kaggle_dataset(kaggle_dataset, local_raw, force=force_redownload)
        _download_kaggle_dataset(kaggle_manifest_dataset, local_manifests, force=force_redownload)

    # Step 5: resume existing output on Drive --------------------------------
    # Drive FUSE sometimes throws I/O error (Errno 5) on mkdir of new subpaths,
    # especially on new paths before Drive sync has had time to index them.
    # Retry a few times with backoff, then fall back: still create local_output,
    # skip Drive sync (call mirror_* later).
    drive_output = drive_root / "output"
    local_output = local_root / "output"
    local_output.mkdir(parents=True, exist_ok=True)

    drive_output_ok = _robust_drive_mkdir(drive_output, retries=5, sleep_base=2.0)
    if drive_output_ok and sync_existing_output and any(drive_output.rglob("chunk_*.npz")):
        print(f"[setup] resume: rsync existing chunks {drive_output} → {local_output}")
        _rsync(drive_output, local_output)
    elif not drive_output_ok:
        print(f"[setup] WARNING: Drive output mkdir failed after retries — pipeline will still run "
              f"on local. Call mirror_dataset_to_drive() in the last cell once Drive "
              f"is stable, or retry setup_aic2026_environment().")

    # Step 6: resolve concrete subdirs --------------------------------------
    def _first_existing(*candidates: Path) -> Path | None:
        for c in candidates:
            if c.exists():
                return c
        return None

    annotation_train_dir = _first_existing(
        local_raw / "annotation" / "train",
        local_raw / "PAB" / "annotation" / "train",
    )
    test_dir = _first_existing(
        local_raw / "name-masked_test-set",
        local_raw / "PAB" / "name-masked_test-set",
    )
    gallery_dir = None
    if test_dir is not None:
        gallery_dir = _first_existing(
            test_dir / "gallery" / "gallery",
            test_dir / "gallery",
        )

    paths = {
        "drive_root": drive_root,
        "local_root": local_root,
        "input_root": local_raw,
        "manifest_dir": local_manifests,
        "output_root": local_output,
        "drive_output_root": drive_output,
        "annotation_train_dir": annotation_train_dir,
        "test_dir": test_dir,
        "gallery_dir": gallery_dir,
    }

    print("[setup] Done. Paths:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    if use_kaggle:
        print(
            "[setup] Kaggle mode: call mirror_dataset_to_drive(PATHS) in the last "
            "cell to cache raw/manifests to Drive for future sessions."
        )
    return paths


def mirror_raw_as_tar_split(
    paths: dict,
    part_size: str = "4500M",
    force: bool = False,
) -> Path | None:
    """Tar local raw -> split into multiple .tar.part_* (<5GB each part) on Drive.

    Reason for splitting: Drive FUSE only accepts uploads <5GB/file. 100GB raw -> ~22 parts.
    Total time ~30-40 min for 100GB (Drive write ~50 MB/s).

    After this step, a new session can call `restore_raw_from_tar_split()` instead
    of `kagglehub.dataset_download()` — restore ~30-40 min instead of 2h re-DL.

    Args:
        paths: dict returned from setup_aic2026_environment().
        part_size: size of each part (4500M = 4.5GB, below 5GB FUSE limit).
        force: if True, delete old parts on Drive and re-tar.

    Returns:
        Path to the directory containing parts on Drive, or None if skipped.
    """
    local_raw = Path(paths["local_root"]) / "raw"
    drive_root = Path(paths["drive_root"])
    parts_dir = drive_root / "raw_tar_parts"

    if not local_raw.exists() or not any(local_raw.iterdir()):
        print("[mirror-tar] local_raw is empty — skip")
        return None

    parts_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(parts_dir.glob("raw.tar.part_*"))
    marker = parts_dir / ".tar_complete"

    if marker.exists() and existing and not force:
        size_gb = sum(p.stat().st_size for p in existing) / 2**30
        print(f"[mirror-tar] already have {len(existing)} parts ({size_gb:.1f} GB) — skip "
              f"(force=True to overwrite)")
        return parts_dir

    if force:
        for p in existing:
            p.unlink()
        if marker.exists():
            marker.unlink()

    # rglob count size — follow_symlinks by stat() directly (auto-dereferences).
    # Important if local_raw contains symlinks to kagglehub cache.
    def _real_size(path: Path) -> int:
        try:
            return path.stat().st_size  # stat() follow symlinks by default
        except (OSError, FileNotFoundError):
            return 0
    raw_size_gb = sum(_real_size(f) for f in local_raw.rglob("*")) / 2**30
    print(f"[mirror-tar] tar {raw_size_gb:.1f} GB → split {part_size} parts → {parts_dir}")
    print(f"[mirror-tar] ETA: ~{int(raw_size_gb / 50 * 60)} min (Drive write ~50 MB/s)")

    # tar -chf: -h dereferences symlinks -> archives actual content, not the link.
    # Important when local_raw is a symlink farm to kagglehub cache:
    #   raw/train/imgs_0 -> /root/.cache/kagglehub/.../train/imgs_0/
    # Without -h -> tar archive would contain broken symlinks (empty when restored).
    cmd = (
        f'tar -chf - -C {local_raw.parent} {local_raw.name} | '
        f'split -b {part_size} - "{parts_dir}/raw.tar.part_"'
    )
    subprocess.check_call(cmd, shell=True)

    # Verify + write marker
    new_parts = sorted(parts_dir.glob("raw.tar.part_*"))
    total_size_gb = sum(p.stat().st_size for p in new_parts) / 2**30
    marker.write_text(
        json.dumps({
            "n_parts": len(new_parts),
            "total_size_gb": round(total_size_gb, 2),
            "part_size": part_size,
            "local_raw_size_gb": round(raw_size_gb, 2),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2),
    )
    print(f"[mirror-tar] done: {len(new_parts)} parts × ~{part_size} = {total_size_gb:.1f} GB")
    return parts_dir


def restore_raw_from_tar_split(paths: dict, force: bool = False) -> bool:
    """Untar Drive raw.tar.part_* -> local SSD (~30-40 min for 100GB).

    Faster than kagglehub re-download (~2h for 100GB).

    Args:
        paths: dict returned from setup_aic2026_environment().
        force: if True, delete old local_raw and untar again.

    Returns:
        True if restore succeeded (data is on local), False if failed/skipped.
    """
    local_root = Path(paths["local_root"])
    drive_root = Path(paths["drive_root"])
    local_raw = local_root / "raw"
    parts_dir = drive_root / "raw_tar_parts"
    marker = parts_dir / ".tar_complete"

    if not (parts_dir.exists() and marker.exists()):
        return False

    parts = sorted(parts_dir.glob("raw.tar.part_*"))
    if not parts:
        return False

    if local_raw.exists() and any(local_raw.iterdir()) and not force:
        # Heuristic: if local_raw already has >50GB -> assume already restored
        local_size_gb = sum(f.stat().st_size for f in local_raw.rglob("*") if f.is_file()) / 2**30
        if local_size_gb > 50:
            print(f"[restore-tar] {local_raw} already has {local_size_gb:.1f} GB — skip")
            return True

    if force and local_raw.exists():
        shutil.rmtree(local_raw)

    local_raw.parent.mkdir(parents=True, exist_ok=True)
    total_size_gb = sum(p.stat().st_size for p in parts) / 2**30
    print(f"[restore-tar] cat {len(parts)} parts ({total_size_gb:.1f} GB) | untar → {local_root}")
    print(f"[restore-tar] ETA: ~{int(total_size_gb / 50 * 60)} min (Drive read ~50 MB/s)")

    # cat parts | tar -xf - (stream untar, no temp file needed)
    cat_cmd = " ".join(f'"{p}"' for p in parts)
    cmd = f'cat {cat_cmd} | tar -xf - -C "{local_root}"'
    subprocess.check_call(cmd, shell=True)

    # Touch synced marker so legacy Kaggle download path skips re-download
    m = _marker_for(_DEFAULT_KAGGLE_DATASET, local_raw)
    m.write_text(f"restored from tar split at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"[restore-tar] done — local_raw ready")
    return True


def mirror_dataset_to_drive(
    paths: dict,
    include_raw: bool = True,
    include_manifests: bool = True,
    include_output: bool = False,
) -> None:
    """Push local SSD raw/manifests to Drive for future sessions to use as cache.

    Call in the last notebook cell after encoding is complete — separated from setup()
    so the first session starts quickly (no Drive write overhead before
    encoding begins).

    Args:
        paths: dict returned from `setup_aic2026_environment()`.
        include_raw: copy `raw/` (annotation + train webp + test set) to Drive.
        include_manifests: copy `manifests/` (parquet files) to Drive.
        include_output: copy `output/` to Drive (usually NOT needed — chunks
            are already synced per-chunk via `sync_chunk_to_drive`).
    """
    local_root = Path(paths["local_root"])
    drive_root = Path(paths["drive_root"])

    targets: list[tuple[Path, Path, str]] = []
    if include_raw:
        targets.append((local_root / "raw", drive_root / "raw", "raw"))
    if include_manifests:
        targets.append(
            (local_root / "manifests", drive_root / "manifests", "manifests"),
        )
    if include_output:
        targets.append(
            (local_root / "output", drive_root / "output", "output"),
        )

    for src, dst, label in targets:
        if not src.exists() or not any(src.iterdir()):
            print(f"[mirror] {label}: local is empty — skip")
            continue
        print(f"[mirror] rsync local → Drive ({label})")
        _rsync(src, dst)
        print(f"[mirror] {label}: synced → {dst}")

    print("[mirror] Done. Future sessions will auto-restore from Drive in setup().")


# ---------------------------------------------------------------------------
# Device selection (A100)
# ---------------------------------------------------------------------------


def select_a100_device(prefer_a100: bool = True, verbose: bool = True):
    """Select the most powerful GPU (A100 if available) and tune CUDA backends.

    Returns torch.device. Warns if not A100/H100.
    """
    import torch  # local import so module does not require torch at import time

    if not torch.cuda.is_available():
        if verbose:
            print("[device] CUDA not available — using CPU.")
        return torch.device("cpu")

    candidates = []
    for idx in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(idx)
        with torch.cuda.device(idx):
            free, total = torch.cuda.mem_get_info()
        candidates.append((free, total, idx, name))
        if verbose:
            print(
                f"[device] cuda:{idx} {name} — free={free/2**30:.1f}GB / "
                f"total={total/2**30:.1f}GB"
            )

    if prefer_a100:
        a100s = [c for c in candidates if "A100" in c[3] or "H100" in c[3]]
        chosen = max(a100s)[2] if a100s else max(candidates)[2]
    else:
        chosen = max(candidates)[2]

    device = torch.device(f"cuda:{chosen}")
    name = torch.cuda.get_device_name(chosen)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

    if verbose:
        print(f"[device] Selected {device} ({name})")
        if "A100" not in name and "H100" not in name:
            print(
                "[device] WARNING: Not A100/H100 — the default batch_size "
                "in the notebook may OOM. Reduce IMAGE_BATCH_SIZE if needed."
            )

    return device


# ---------------------------------------------------------------------------
# Output helpers — atomic save + async Drive sync
# ---------------------------------------------------------------------------


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype("float32", copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def save_npz_atomic(path: str | os.PathLike, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return path


def chunk_file(out_dir: Path, start: int, end: int) -> Path:
    return Path(out_dir) / f"chunk_{start:07d}_{end:07d}.npz"


def maybe_clear_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


# Background sync uses a single worker thread to avoid hammering Drive I/O
# with many concurrent copies.
_SYNC_LOCK = threading.Lock()
_SYNC_THREADS: list[threading.Thread] = []


def _copy_with_retry(src: Path, dst: Path, retries: int = 3) -> None:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            return
        except Exception as exc:  # broad: Drive I/O can throw varied errors
            last_err = exc
            time.sleep(2 ** attempt)
    print(f"[sync] FAILED after {retries} attempts: {src} → {dst}: {last_err}")


def sync_chunk_to_drive(
    local_path: str | os.PathLike,
    local_root: str | os.PathLike,
    drive_output_root: str | os.PathLike,
    background: bool = True,
) -> threading.Thread | None:
    """Copy 1 chunk from local SSD -> Drive, preserving the relative path.

    When `background=True`, the copy runs in a daemon thread — does not block training.
    Returns the thread handle (or None if sync is synchronous).
    """
    local_path = Path(local_path)
    local_root = Path(local_root)
    drive_output_root = Path(drive_output_root)
    rel = local_path.relative_to(local_root)
    dst = drive_output_root.parent / rel  # local_root/output/... → drive_root/output/...
    # local_root convention: local_root/output/<rel>. drive_output_root already points
    # to drive_root/output. Compute more simply:
    try:
        rel = local_path.relative_to(local_root / "output")
        dst = drive_output_root / rel
    except ValueError:
        # local_path is not under local_root/output — fallback to filename only
        dst = drive_output_root / local_path.name

    def _job():
        with _SYNC_LOCK:
            _copy_with_retry(local_path, dst)

    if background:
        t = threading.Thread(target=_job, daemon=True)
        t.start()
        _SYNC_THREADS.append(t)
        # Remove dead threads from the list (avoid leak)
        _SYNC_THREADS[:] = [x for x in _SYNC_THREADS if x.is_alive()]
        return t
    _job()
    return None


def wait_for_pending_syncs(timeout_per_thread: float = 600.0) -> None:
    """Block until all background syncs have completed.

    Call at the end of the notebook before the session terminates.
    """
    for t in list(_SYNC_THREADS):
        if t.is_alive():
            t.join(timeout=timeout_per_thread)
    _SYNC_THREADS[:] = [x for x in _SYNC_THREADS if x.is_alive()]
    if _SYNC_THREADS:
        print(f"[sync] WARNING: {len(_SYNC_THREADS)} thread(s) still running after timeout.")
    else:
        print("[sync] All chunks have been synced to Drive.")


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------


def find_existing_chunks(
    *output_dirs: str | os.PathLike,
    pattern: str = "chunk_*.npz",
) -> set[str]:
    """Collect basenames of chunks already present on both local and Drive.

    Pass multiple dirs to union them (e.g. local + Drive with same subpath).
    """
    found: set[str] = set()
    for d in output_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in d.glob(pattern):
            found.add(p.name)
    return found


def chunk_done(
    chunk_path: Path,
    drive_output_root: Path | None,
    local_root: Path | None,
) -> bool:
    """True if the chunk already exists locally or has been synced to Drive.

    Checks Drive as a fallback to avoid wasting compute re-encoding when
    local SSD is lost (session reset) but Drive already has the chunk.
    """
    if chunk_path.exists():
        return True
    if drive_output_root and local_root:
        try:
            rel = chunk_path.relative_to(local_root / "output")
            drive_path = drive_output_root / rel
            if drive_path.exists():
                # Restore to local for continued use
                drive_path.parent.mkdir(parents=True, exist_ok=True)
                _copy_with_retry(drive_path, chunk_path)
                return True
        except ValueError:
            pass
    return False
