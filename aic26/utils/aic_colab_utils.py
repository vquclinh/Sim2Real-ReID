"""Shared utilities for AIC2026 Track 4 feature extraction on Colab A100.

Hai notebook `01a_pe_g14_features.ipynb` và `01b_vitpose_features.ipynb`
share module này để:
  1. Bootstrap môi trường: mount Drive, cache Kaggle dataset, rsync sang
     local SSD của Colab.
  2. Pick A100 device, set TF32 / cuDNN benchmark / Flash SDPA.
  3. Save NPZ atomically + async sync sang Drive theo chunk.
  4. Liệt kê chunks đã có (Drive + local) cho resume logic.
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
    """Ghi kaggle.json từ `text` (JSON hoặc raw access token).

    Trả về True nếu thành công. Hỗ trợ:
      - JSON form `{"username":..., "key":...}` (kaggle.json chuẩn).
      - Raw access token `KGAT_xxx` hoặc plain string (1 dòng) — sẽ
        wrap thành `{"username": KAGGLE_USERNAME or "anonymous", "key": <token>}`.
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

    # Raw token (KGAT_... hoặc plain alphanum). Wrap vào JSON.
    username = os.environ.get("KAGGLE_USERNAME") or fallback_username
    target.write_text(json.dumps({"username": username, "key": text}))
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = text
    return True


def _ensure_kaggle_credentials(
    drive_root: Path,
    kaggle_token_path: str | None,
) -> None:
    """Đặt KAGGLE_CONFIG_DIR và đảm bảo kaggle.json tồn tại.

    Thứ tự tìm token (mỗi candidate có thể là JSON hoặc raw `KGAT_*`):
      1. Arg `kaggle_token_path` (hỗ trợ ~ expand).
      2. `drive_root/.kaggle/kaggle.json`.
      3. `~/.kaggle/kaggle.json`.
      4. `~/.kaggle/access_token` (Kaggle access token format mới).
      5. Env var KAGGLE_USERNAME + KAGGLE_KEY (đã set bên ngoài).
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
        "Không tìm thấy kaggle credentials. Có 2 cách:\n"
        f"  1) Upload kaggle.json vào {drive_root}/.kaggle/kaggle.json\n"
        "  2) Tạo access token raw: \n"
        "     !mkdir -p ~/.kaggle && echo <KGAT_...> > ~/.kaggle/access_token\n"
        "     && chmod 600 ~/.kaggle/access_token\n"
        "Hoặc truyền kaggle_token_path='~/.kaggle/access_token'."
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
    """rsync src/ → dst/ — chỉ copy file thiếu/đã thay đổi.

    Nếu hệ thống không có `rsync` (hiếm trên Colab) fallback sang
    `shutil.copytree(..., dirs_exist_ok=True)`.
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
    """Download 1 Kaggle dataset → `dest` và merge structure.

    Nhiều dataset có thể merge vào cùng `dest` (ví dụ annotation +
    train-webp-part-01-05 + train-webp-part-06-10 share thư mục `raw/`).
    Mỗi slug có file marker `.synced_<slug>` riêng để skip khi đã sync.
    """
    _ensure_kagglehub()
    import kagglehub

    dest.mkdir(parents=True, exist_ok=True)
    marker = _marker_for(slug, dest)
    if marker.exists() and not force:
        print(f"[setup] {slug} đã sync trước đó — skip")
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
    """mkdir trên Drive FUSE với multiple strategies + retry.

    Drive FUSE Errno 5 thường do orphan folder trong trash hoặc FUSE state
    corruption — retry thuần KHÔNG giải quyết. Helper này thử 3 strategies
    theo thứ tự (mỗi attempt qua loop):
      1. Python `pathlib.mkdir(parents=True, exist_ok=True)`
      2. Shell `mkdir -p` (qua subprocess — đôi khi bypass FUSE quirks)
      3. Check `path.exists()` sau khi mkdir error (FUSE đôi khi thành công
         nhưng vẫn raise — folder vẫn tạo được)

    Returns:
        True nếu cuối cùng path tồn tại.
        False nếu fail sau hết retries — caller nên skip Drive op.
    """
    # Fast path: đã tồn tại
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

        # Strategy 3: re-check existence sau cả 2 strategies
        try:
            if path.exists():
                return True
        except OSError:
            pass

        if attempt == retries - 1:
            print(f"[drive] mkdir {path} fail sau {retries} retries (strategies 1+2+3): {last_err}")
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

    `config_source` có thể là:
      - Path đến file rclone.conf / rclone.txt (local hoặc Drive)
      - Raw INI text (chứa `[remote_name]`)
      - None → auto-search `<drive_root>/rclone.txt|conf`, `/content/rclone.txt|conf`
              hoặc `~/Documents/AIC2026/rclone.txt` (dev fallback)

    Returns path đến rclone.conf đã ghi.
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
                "Không tìm thấy rclone config. Upload rclone.txt lên Drive "
                f"({drive_root}/rclone.txt) hoặc truyền config_source='...'."
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
    """Copy `<input_root>/name-masked_test-set/` từ Drive sang local SSD.

    Drive FUSE raise `OSError: [Errno 5]` khi iter folder nhiều file. Helper
    này tránh issue bằng cách:

    1) Đếm file qua shell `ls | wc -l` (chịu lỗi tốt hơn Python iterdir).
    2) Stream copy bằng `tar | tar` qua pipe (1 metadata read + sequential
       file reads) — nhanh hơn rsync (rsync stat từng file = 36k × 50ms qua
       FUSE = 30+ phút chỉ để liệt kê chưa transfer).
    3) Sao chép file query JSON nhỏ riêng qua `shutil.copy2`.

    Idempotent — chạy lại không re-copy nếu local count đã match.
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
        print(f"[stage-test] gallery local đã đầy ({have_count}/{target_count}) — skip copy")
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
    """Bootstrap đầy đủ cho 1 session Colab.

    Returns paths dict gồm:
      drive_root, local_root, input_root, manifest_dir, output_root,
      drive_output_root, annotation_train_dir, test_dir, gallery_dir.

    `output_root` luôn trỏ local SSD (write nhanh).
    `drive_output_root` là nơi sync chunks về để persistent.

    **Default flow (no Kaggle)**: dataset đã có sẵn ở `<drive_root>/raw/` +
    `<drive_root>/manifest/` (hoặc `manifests/`). Hàm này chỉ symlink các
    folder đó vào `<local_root>/raw/` và `<local_root>/manifests/` để code
    còn lại (đọc qua `paths['input_root']`, `paths['manifest_dir']`) không
    cần thay đổi.

    **Kaggle fallback**: set `use_kaggle=True` để bật lại pipeline cũ
    (download qua kagglehub). Yêu cầu có kaggle.json hoặc access_token.

    **Local SSD speed**: với training notebook (đọc 1M+ ảnh nhiều lần),
    set `copy_raw_to_local=True` để rsync raw từ Drive về local SSD —
    chậm 30-60 min lần đầu nhưng training nhanh hơn ~5-10×.
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
                print("[setup] Drive đã mount sẵn.")
        except ImportError:
            warnings.warn("google.colab.drive không khả dụng — bỏ qua mount.")

    drive_root.mkdir(parents=True, exist_ok=True)
    local_root.mkdir(parents=True, exist_ok=True)

    # Step 2: Kaggle credentials (chỉ khi use_kaggle=True) ------------------
    if use_kaggle:
        _ensure_kaggle_credentials(drive_root, kaggle_token_path)

    # Step 3: Restore raw + manifests từ Drive → local SSD ------------------
    drive_raw = drive_root / "raw"
    # Drive folder có thể tên 'manifest' (singular, theo screenshot user) hoặc
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
        print(f"[setup] raw đã ready: {local_raw}")
    elif (drive_root / "raw_tar_parts" / ".tar_complete").exists():
        # FAST PATH (legacy session với tar-split mirror)
        restore_raw_from_tar_split(raw_paths, force=force_redownload)
    elif drive_raw.exists() and any(drive_raw.iterdir()):
        if copy_raw_to_local:
            print(f"[setup] rsync raw từ Drive → {local_raw} (có thể mất 30-60 min)")
            _rsync(drive_raw, local_raw)
        else:
            # Symlink Drive → local. Instant, đọc trực tiếp qua Drive FUSE.
            # OK cho inference / zero-shot; training nên dùng copy_raw_to_local=True.
            if local_raw.exists():
                local_raw.rmdir() if not any(local_raw.iterdir()) else shutil.rmtree(local_raw)
            local_raw.symlink_to(drive_raw)
            print(f"[setup] symlink {local_raw} → {drive_raw}")
    elif use_kaggle:
        pass  # will be filled by Step 4 Kaggle download
    else:
        raise FileNotFoundError(
            f"Không tìm thấy raw dataset.\n"
            f"  - Drive: {drive_raw} (không tồn tại hoặc rỗng)\n"
            f"  - Local: {local_raw} (không tồn tại hoặc rỗng)\n"
            f"Hãy upload dataset lên {drive_raw}, hoặc set use_kaggle=True nếu "
            f"muốn download qua Kaggle."
        )

    # Step 3b: Restore manifests
    if local_manifests.exists() and (local_manifests.is_symlink() or any(local_manifests.iterdir())):
        print(f"[setup] manifests đã ready: {local_manifests}")
    elif drive_manifests.exists() and any(drive_manifests.iterdir()):
        # Manifests nhẹ (~100MB parquets) — symlink hoặc rsync đều OK
        if local_manifests.exists():
            local_manifests.rmdir() if not any(local_manifests.iterdir()) else shutil.rmtree(local_manifests)
        local_manifests.symlink_to(drive_manifests)
        print(f"[setup] symlink {local_manifests} → {drive_manifests}")
    elif use_kaggle:
        pass  # will be filled by Step 4 Kaggle download
    else:
        print(f"[setup] ⚠️  Không có manifest ở {drive_manifests} hoặc {local_manifests} "
              f"— pipeline phụ thuộc manifest sẽ lỗi. Pipeline zero-shot không cần.")

    # Step 4: Kaggle download fallback (chỉ khi use_kaggle=True) ------------
    if use_kaggle:
        _download_kaggle_dataset(kaggle_dataset, local_raw, force=force_redownload)
        _download_kaggle_dataset(kaggle_manifest_dataset, local_manifests, force=force_redownload)

    # Step 5: resume output đã có trên Drive --------------------------------
    # Drive FUSE đôi khi throw I/O error (Errno 5) khi mkdir subpath, đặc biệt
    # path mới + Drive sync chưa kịp index. Retry vài lần với backoff, sau đó
    # fall back: vẫn tạo local_output, skip Drive sync (gọi mirror_* later).
    drive_output = drive_root / "output"
    local_output = local_root / "output"
    local_output.mkdir(parents=True, exist_ok=True)

    drive_output_ok = _robust_drive_mkdir(drive_output, retries=5, sleep_base=2.0)
    if drive_output_ok and sync_existing_output and any(drive_output.rglob("chunk_*.npz")):
        print(f"[setup] resume: rsync existing chunks {drive_output} → {local_output}")
        _rsync(drive_output, local_output)
    elif not drive_output_ok:
        print(f"[setup] ⚠️  Drive output mkdir failed sau retries — pipeline vẫn chạy "
              f"trên local. Gọi mirror_dataset_to_drive() ở cell cuối khi Drive "
              f"đã ổn định, hoặc retry lại setup_aic2026_environment().")

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
            "[setup] Kaggle mode: gọi mirror_dataset_to_drive(PATHS) ở cell "
            "cuối để cache raw/manifests lên Drive cho session sau."
        )
    return paths


def mirror_raw_as_tar_split(
    paths: dict,
    part_size: str = "4500M",
    force: bool = False,
) -> Path | None:
    """Tar local raw → split thành nhiều .tar.part_* (<5GB mỗi part) trên Drive.

    Lý do tách: Drive FUSE chỉ accept upload <5GB/file. 100GB raw → ~22 parts.
    Total time ~30-40 min cho 100GB (Drive write ~50 MB/s).

    Sau bước này, session mới có thể gọi `restore_raw_from_tar_split()` thay
    cho `kagglehub.dataset_download()` — restore ~30-40 min thay vì 2h re-DL.

    Args:
        paths: dict trả về từ setup_aic2026_environment().
        part_size: kích thước mỗi part (4500M = 4.5GB, dưới 5GB FUSE limit).
        force: nếu True, xóa parts cũ trên Drive và tar lại.

    Returns:
        Path tới thư mục chứa parts trên Drive, hoặc None nếu skip.
    """
    local_raw = Path(paths["local_root"]) / "raw"
    drive_root = Path(paths["drive_root"])
    parts_dir = drive_root / "raw_tar_parts"

    if not local_raw.exists() or not any(local_raw.iterdir()):
        print("[mirror-tar] local_raw trống — skip")
        return None

    parts_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(parts_dir.glob("raw.tar.part_*"))
    marker = parts_dir / ".tar_complete"

    if marker.exists() and existing and not force:
        size_gb = sum(p.stat().st_size for p in existing) / 2**30
        print(f"[mirror-tar] đã có {len(existing)} parts ({size_gb:.1f} GB) — skip "
              f"(force=True để overwrite)")
        return parts_dir

    if force:
        for p in existing:
            p.unlink()
        if marker.exists():
            marker.unlink()

    # rglob đếm size — follow_symlinks bằng cách stat() trực tiếp (tự dereference).
    # Đây quan trọng nếu local_raw chứa symlinks tới kagglehub cache.
    def _real_size(path: Path) -> int:
        try:
            return path.stat().st_size  # stat() follow symlinks by default
        except (OSError, FileNotFoundError):
            return 0
    raw_size_gb = sum(_real_size(f) for f in local_raw.rglob("*")) / 2**30
    print(f"[mirror-tar] tar {raw_size_gb:.1f} GB → split {part_size} parts → {parts_dir}")
    print(f"[mirror-tar] ETA: ~{int(raw_size_gb / 50 * 60)} phút (Drive write ~50 MB/s)")

    # tar -chf: -h dereference symlinks → archive nội dung thật, không phải link.
    # Quan trọng khi local_raw là symlink farm tới kagglehub cache:
    #   raw/train/imgs_0 → /root/.cache/kagglehub/.../train/imgs_0/
    # Không có -h → tar archive sẽ chứa broken symlinks (rỗng khi restore).
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
    """Untar Drive raw.tar.part_* → local SSD (~30-40 min cho 100GB).

    Faster than kagglehub re-download (~2h cho 100GB).

    Args:
        paths: dict trả về từ setup_aic2026_environment().
        force: nếu True, xóa local_raw cũ và untar lại.

    Returns:
        True nếu restore thành công (data đã ở local), False nếu fail/skip.
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
        # Heuristic: nếu local_raw đã có >50GB → assume đã restore
        local_size_gb = sum(f.stat().st_size for f in local_raw.rglob("*") if f.is_file()) / 2**30
        if local_size_gb > 50:
            print(f"[restore-tar] {local_raw} đã có {local_size_gb:.1f} GB — skip")
            return True

    if force and local_raw.exists():
        shutil.rmtree(local_raw)

    local_raw.parent.mkdir(parents=True, exist_ok=True)
    total_size_gb = sum(p.stat().st_size for p in parts) / 2**30
    print(f"[restore-tar] cat {len(parts)} parts ({total_size_gb:.1f} GB) | untar → {local_root}")
    print(f"[restore-tar] ETA: ~{int(total_size_gb / 50 * 60)} phút (Drive read ~50 MB/s)")

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
    """Push local SSD raw/manifests lên Drive để session sau dùng cache.

    Gọi ở cell cuối notebook sau khi đã encode xong — tách khỏi setup()
    để session đầu khởi động nhanh (không pay Drive write 1GB trước
    khi bắt đầu encoding).

    Args:
        paths: dict trả về từ `setup_aic2026_environment()`.
        include_raw: copy `raw/` (annotation + train webp + test set) lên Drive.
        include_manifests: copy `manifests/` (parquet files) lên Drive.
        include_output: copy `output/` lên Drive (thường KHÔNG cần — chunks
            đã được sync per-chunk qua `sync_chunk_to_drive`).
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
            print(f"[mirror] {label}: local trống — skip")
            continue
        print(f"[mirror] rsync local → Drive ({label})")
        _rsync(src, dst)
        print(f"[mirror] {label}: synced → {dst}")

    print("[mirror] Done. Session sau sẽ tự restore từ Drive trong setup().")


# ---------------------------------------------------------------------------
# Device selection (A100)
# ---------------------------------------------------------------------------


def select_a100_device(prefer_a100: bool = True, verbose: bool = True):
    """Chọn GPU mạnh nhất (A100 nếu có) và tune CUDA backends.

    Returns torch.device. Warn nếu không phải A100/H100.
    """
    import torch  # local import để module không cần torch khi import

    if not torch.cuda.is_available():
        if verbose:
            print("[device] CUDA không khả dụng — dùng CPU.")
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
                "[device] ⚠️ Không phải A100/H100 — batch_size mặc định "
                "trong notebook có thể OOM. Giảm IMAGE_BATCH_SIZE nếu cần."
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
    """Copy 1 chunk từ local SSD → Drive, giữ nguyên relative path.

    Khi `background=True`, copy chạy trong thread daemon — không block training.
    Trả về thread handle (hoặc None nếu sync sync).
    """
    local_path = Path(local_path)
    local_root = Path(local_root)
    drive_output_root = Path(drive_output_root)
    rel = local_path.relative_to(local_root)
    dst = drive_output_root.parent / rel  # local_root/output/... → drive_root/output/...
    # local_root convention: local_root/output/<rel>. drive_output_root đã trỏ
    # tới drive_root/output. Tính lại đơn giản hơn:
    try:
        rel = local_path.relative_to(local_root / "output")
        dst = drive_output_root / rel
    except ValueError:
        # local_path không nằm trong local_root/output — fallback dùng filename
        dst = drive_output_root / local_path.name

    def _job():
        with _SYNC_LOCK:
            _copy_with_retry(local_path, dst)

    if background:
        t = threading.Thread(target=_job, daemon=True)
        t.start()
        _SYNC_THREADS.append(t)
        # Dọn các thread đã chết khỏi list (tránh leak)
        _SYNC_THREADS[:] = [x for x in _SYNC_THREADS if x.is_alive()]
        return t
    _job()
    return None


def wait_for_pending_syncs(timeout_per_thread: float = 600.0) -> None:
    """Block tới khi tất cả background sync hoàn tất.

    Gọi cuối notebook trước khi kết thúc session.
    """
    for t in list(_SYNC_THREADS):
        if t.is_alive():
            t.join(timeout=timeout_per_thread)
    _SYNC_THREADS[:] = [x for x in _SYNC_THREADS if x.is_alive()]
    if _SYNC_THREADS:
        print(f"[sync] ⚠️ {len(_SYNC_THREADS)} thread vẫn chạy sau timeout.")
    else:
        print("[sync] Tất cả chunks đã sync xong lên Drive.")


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------


def find_existing_chunks(
    *output_dirs: str | os.PathLike,
    pattern: str = "chunk_*.npz",
) -> set[str]:
    """Tập hợp basename của chunks đã có trên cả local lẫn Drive.

    Truyền nhiều dir để union (ví dụ local + Drive cùng subpath).
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
    """True nếu chunk đã tồn tại ở local hoặc đã sync xong lên Drive.

    Kiểm tra Drive đối chiếu để không tốn compute re-encode khi local SSD
    bị mất (session reset) trong khi Drive đã có.
    """
    if chunk_path.exists():
        return True
    if drive_output_root and local_root:
        try:
            rel = chunk_path.relative_to(local_root / "output")
            drive_path = drive_output_root / rel
            if drive_path.exists():
                # Khôi phục về local để dùng tiếp
                drive_path.parent.mkdir(parents=True, exist_ok=True)
                _copy_with_retry(drive_path, chunk_path)
                return True
        except ValueError:
            pass
    return False
