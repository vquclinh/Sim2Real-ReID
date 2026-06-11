# PE-G14 Attr Local Eval — Cell Reference

Copy each cell below into a new Colab notebook (code cells only, no markdown cells).

Notebook path: `aic26/pipelines/local_eval/pab_original/pe_g14_attr_local_eval.ipynb`

## Cell 1

```python
# Cell 1 — Mount Drive + runtime preflight

from google.colab import drive
drive.mount("/content/drive", force_remount=True, timeout_ms=300000)

!nvidia-smi
!free -h

import torch, psutil

assert torch.cuda.is_available(), "CUDA GPU is required."

gpu_name = torch.cuda.get_device_name(0)
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
ram_gb = psutil.virtual_memory().total / 1e9

print("GPU:", gpu_name)
print("GPU memory GB:", gpu_mem_gb)
print("System RAM GB:", ram_gb)

if ("A100" in gpu_name) or gpu_mem_gb >= 35:
    print("PASS: A100/high-VRAM runtime detected.")
    print("This should be fast and stable for PE-Core-G14-448.")
elif ("L4" in gpu_name) or gpu_mem_gb >= 22:
    print("PASS: L4/24GB-class runtime detected.")
    print("This should work for PE-Core-G14-448, but it may be slower than A100.")
elif ("T4" in gpu_name) or gpu_mem_gb < 20:
    print("WARNING: T4/low-VRAM runtime detected.")
    print("PE-Core-G14-448 may be slow or may run out of memory.")
    print("Continue only if you accept the risk.")
else:
    print("WARNING: Unknown GPU runtime.")
    print("The notebook will continue, but monitor for OOM errors.")

if ram_gb < 35:
    print("WARNING: System RAM is below 35GB. High-RAM runtime is recommended.")
else:
    print("PASS: system RAM is enough.")

print("PASS: runtime preflight completed.")
```

## Cell 2

```python
# Cell 2 — Prepare original PAB attr data

from pathlib import Path
import shutil, zipfile, subprocess, time

DRIVE_PAB     = Path("/content/drive/MyDrive/aic2026_data/pab_original")
ATTR_JSON_SRC = DRIVE_PAB / "annotation/test/attr.json"
TEST_ZIP_SRC  = DRIVE_PAB / "archives/test.zip"

RAW_DIR = Path("/content/aic_local/pab_original/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ATTR_JSON_LOCAL = RAW_DIR / "attr.json"
TEST_ZIP_LOCAL  = RAW_DIR / "test.zip"

assert ATTR_JSON_SRC.exists(), f"Missing Drive file: {ATTR_JSON_SRC}"
assert TEST_ZIP_SRC.exists(),  f"Missing Drive file: {TEST_ZIP_SRC}"

for src, dst in [(ATTR_JSON_SRC, ATTR_JSON_LOCAL), (TEST_ZIP_SRC, TEST_ZIP_LOCAL)]:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print("SKIP existing:", dst)
    else:
        print("Copying:", src, "->", dst)
        shutil.copy2(src, dst)


def count_images_recursive(folder: Path) -> int:
    if not folder.exists():
        return 0
    cmd = (
        'find "' + str(folder) + '" -type f '
        r'\( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l'
    )
    return int(subprocess.check_output(["bash", "-lc", cmd], text=True).strip())


def find_test_image_root(base: Path) -> Path:
    for candidate in [base / "test", base]:
        if candidate.exists() and count_images_recursive(candidate) > 0:
            return candidate
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and count_images_recursive(sub) > 0:
            return sub
    return base


n_images_before = count_images_recursive(RAW_DIR)
if n_images_before == 0:
    print("Extracting test.zip...")
    t0 = time.time()
    with zipfile.ZipFile(TEST_ZIP_LOCAL, "r") as zf:
        zf.extractall(RAW_DIR)
    print(f"Extraction done in {time.time() - t0:.1f}s")
else:
    print(f"Images already extracted ({n_images_before} found). Skipping extraction.")

TEST_IMAGE_ROOT = find_test_image_root(RAW_DIR)
n_images = count_images_recursive(TEST_IMAGE_ROOT)

assert n_images > 0, f"No images found under {RAW_DIR}"

print("RAW_DIR:", RAW_DIR)
print("TEST_IMAGE_ROOT:", TEST_IMAGE_ROOT)
print("Image count:", n_images)
print("PASS: PAB original test data is ready.")
```

## Cell 3

```python
# Cell 3 — Install PE-G14 dependencies

from pathlib import Path
import sys

PM_REPO = Path("/content/perception_models")
if not PM_REPO.exists():
    !git clone --depth 1 https://github.com/facebookresearch/perception_models.git "{PM_REPO}"

!pip install -q timm ftfy regex tokenizers einops iopath

if str(PM_REPO) not in sys.path:
    sys.path.insert(0, str(PM_REPO))

print("perception_models repo:", PM_REPO)
print("PASS: PE-G14 dependencies ready.")

# Clone repo to access aic26.eval utilities
REPO_DIR = Path("/content/Sim2Real-ReID")
if not REPO_DIR.exists():
    !git clone -b clean-adaption https://github.com/vquclinh/Sim2Real-ReID.git "{REPO_DIR}"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

print("Repo path:", REPO_DIR)
print("PASS: repo utilities ready.")

```

## Cell 4

```python
# Cell 4 — Build attr local eval manifest

from pathlib import Path
import json, subprocess

RAW_DIR         = Path("/content/aic_local/pab_original/raw")
ATTR_JSON_LOCAL = RAW_DIR / "attr.json"


def count_images_recursive(folder: Path) -> int:
    if not folder.exists():
        return 0
    cmd = (
        'find "' + str(folder) + '" -type f '
        r'\( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l'
    )
    return int(subprocess.check_output(["bash", "-lc", cmd], text=True).strip())


rows = []
with open(ATTR_JSON_LOCAL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

query_ids       = []
query_captions  = []
positive_images = []

for row in rows:
    caption = row["caption"]
    if isinstance(caption, list):
        caption = caption[0]
    query_ids.append(str(row["image_id"]))
    query_captions.append(caption)
    positive_images.append(row["image"])

# Gallery: sorted unique image paths, e.g. "test/0.jpg"
gallery_ids = sorted(set(row["image"] for row in rows))

# Absolute paths for each gallery image
gallery_paths = [RAW_DIR / rel for rel in gallery_ids]

# Validate all gallery images exist
missing_gallery = [rel for rel in gallery_ids if not (RAW_DIR / rel).exists()]
assert len(missing_gallery) == 0, f"Missing gallery images: {missing_gallery[:5]}"

# Validate all positives are in gallery
gallery_id_set = set(gallery_ids)
missing_positives = [img for img in positive_images if img not in gallery_id_set]
assert len(missing_positives) == 0, f"Missing positive images: {missing_positives[:5]}"

assert len(query_ids) > 0,   "No queries parsed from attr.json"
assert len(gallery_ids) > 0, "No gallery images found"

print("Queries:", len(query_ids))
print("Gallery images:", len(gallery_ids))
print("Sample query_id:", query_ids[0])
print("Sample caption:", query_captions[0][:120])
print("Sample positive:", positive_images[0])
print("Sample gallery ID:", gallery_ids[0])

MANIFEST_DIR = Path("/content/drive/MyDrive/aic2026_data/pab_original/manifests")
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = MANIFEST_DIR / "attr_eval.jsonl"

with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
    for qid, cap, pos in zip(query_ids, query_captions, positive_images):
        f.write(json.dumps({"query_id": qid, "caption": cap, "positive_image": pos}, ensure_ascii=False) + "\n")

print("Manifest saved to Drive:", MANIFEST_OUT)
print("PASS: manifest built.")
```

## Cell 5

```python
# Cell 5 — Load PE-G14 model

from pathlib import Path
from PIL import Image
import torch
from core.vision_encoder import pe
from core.vision_encoder import transforms as pe_transforms

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

print("GPU:", gpu_name)
print("GPU memory GB:", total_gb)

major, minor = torch.cuda.get_device_capability(0)
AUTOCAST_DTYPE = torch.bfloat16 if major >= 8 else torch.float16
print("autocast dtype:", AUTOCAST_DTYPE)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

PE_MODEL_NAME = "PE-Core-G14-448"
print("Loading", PE_MODEL_NAME)

pe_model = pe.CLIP.from_config(PE_MODEL_NAME, pretrained=True).to(device).eval()
pe_model = pe_model.to(memory_format=torch.channels_last)

pe_preprocess = pe_transforms.get_image_transform(pe_model.image_size)
pe_tokenizer  = pe_transforms.get_text_tokenizer(pe_model.context_length)

fallback_size = pe_model.image_size if isinstance(pe_model.image_size, int) else pe_model.image_size[0]
pe_fallback   = pe_preprocess(Image.new("RGB", (fallback_size, fallback_size))).clone()

print("image_size:", pe_model.image_size)
print("context_length:", pe_model.context_length)
print("PASS: PE-G14 model loaded.")
```

## Cell 6

```python
# Cell 6 — Encode gallery and queries

from pathlib import Path
import shutil, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

RAW_DIR   = Path("/content/aic_local/pab_original/raw")
CACHE_DIR = Path("/content/drive/MyDrive/aic2026_data/pab_original/cache/pe_g14_attr_zero_shot")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GALLERY_CACHE = CACHE_DIR / "gallery_emb.npz"
QUERY_CACHE   = CACHE_DIR / "query_emb.npz"

# gallery_ids and query_captions are defined in Cell 4
gallery_abs_paths = [RAW_DIR / rel for rel in gallery_ids]

# Batch size only affects speed and VRAM usage, not retrieval accuracy.
# L4 worked before, so keep L4 around 96. A100 can use a larger batch.
if total_gb >= 60:
    IMG_BATCH = 256
elif total_gb >= 22:
    IMG_BATCH = 96
else:
    IMG_BATCH = 32

NUM_WORKERS     = 4
PREFETCH_FACTOR = 2
TEXT_BATCH      = 256

print("Gallery images:", len(gallery_ids))
print("Query captions:", len(query_captions))
print("GPU memory GB:", total_gb)
print("IMG_BATCH:", IMG_BATCH)
print("TEXT_BATCH:", TEXT_BATCH)


class GalleryDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return pe_preprocess(img), idx, True
        except Exception:
            return pe_fallback.clone(), idx, False


@torch.inference_mode()
def encode_gallery():
    if GALLERY_CACHE.exists():
        cached = np.load(GALLERY_CACHE, allow_pickle=False)
        if list(cached["ids"]) == gallery_ids and cached["emb"].shape[0] == len(gallery_ids):
            print("Reusing gallery cache:", cached["emb"].shape, cached["emb"].dtype)
            return cached["emb"]

    ds = GalleryDataset(gallery_abs_paths)
    dl = DataLoader(
        ds,
        batch_size=IMG_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True,
    )

    chunks, ok_flags = [], np.zeros(len(gallery_abs_paths), dtype=bool)
    t0, seen = time.time(), 0

    for tensors, idxs, oks in dl:
        tensors = tensors.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
            feats = pe_model.encode_image(tensors)
            feats = F.normalize(feats.float(), dim=-1).half().cpu().numpy()

        chunks.append(feats)
        ok_flags[idxs.numpy()] = oks.numpy().astype(bool)

        seen += tensors.size(0)
        if seen % (IMG_BATCH * 10) == 0 or seen == len(gallery_abs_paths):
            elapsed = time.time() - t0
            rate    = seen / elapsed if elapsed > 0 else 0
            eta     = (len(gallery_abs_paths) - seen) / rate if rate > 0 else 0
            print(f"encoded {seen}/{len(gallery_abs_paths)} | {rate:.1f} img/s | ETA {eta:.0f}s")

    gallery_emb = np.concatenate(chunks, axis=0).astype(np.float16)
    print("Gallery encode done:", gallery_emb.shape, gallery_emb.dtype)
    print("failed images:", int((~ok_flags).sum()))

    np.savez_compressed(GALLERY_CACHE, ids=np.array(gallery_ids), emb=gallery_emb, ok=ok_flags)
    print("Saved gallery cache:", GALLERY_CACHE)
    return gallery_emb


@torch.inference_mode()
def encode_queries():
    if QUERY_CACHE.exists():
        cached = np.load(QUERY_CACHE, allow_pickle=False)
        if list(cached["ids"]) == query_ids and cached["emb"].shape[0] == len(query_ids):
            print("Reusing query cache:", cached["emb"].shape, cached["emb"].dtype)
            return cached["emb"]

    chunks, t0 = [], time.time()

    for start in range(0, len(query_captions), TEXT_BATCH):
        end   = min(start + TEXT_BATCH, len(query_captions))
        texts = query_captions[start:end]
        tokens = pe_tokenizer(texts).to(device)

        with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE):
            feats = pe_model.encode_text(tokens)
            feats = F.normalize(feats.float(), dim=-1).half().cpu().numpy()

        chunks.append(feats)
        print(f"encoded queries {end}/{len(query_captions)}")

    query_emb = np.concatenate(chunks, axis=0).astype(np.float16)
    np.savez_compressed(QUERY_CACHE, ids=np.array(query_ids), emb=query_emb)

    print("Saved query cache:", QUERY_CACHE)
    print(f"Query encode done in {time.time() - t0:.1f}s")
    return query_emb


gallery_emb_np = encode_gallery()
query_emb_np   = encode_queries()

print("gallery_emb_np:", gallery_emb_np.shape, gallery_emb_np.dtype)
print("query_emb_np:", query_emb_np.shape, query_emb_np.dtype)
```

## Cell 7

```python
# Cell 7 — Evaluate and save local metrics

from pathlib import Path
import json, datetime
import numpy as np
import torch

from aic26.eval.pab_metrics import (
    compute_single_positive_metrics,
    find_positive_ranks,
    build_topk_records,
    write_json,
    write_jsonl,
    write_metrics_markdown,
)

RUN_ID    = "local_001_pe_g14_attr_zero_shot"
LOCAL_RUN = Path("/content/aic_local/pab_original/runs") / RUN_ID
DRIVE_RUN = Path("/content/drive/MyDrive/aic2026_data/pab_original/runs") / RUN_ID

LOCAL_RUN.mkdir(parents=True, exist_ok=True)
DRIVE_RUN.mkdir(parents=True, exist_ok=True)

# Compute full similarity matrix and ranked indices.
# gallery_emb_np, query_emb_np from Cell 6; device from Cell 5.
G_t = torch.from_numpy(gallery_emb_np).to(device, dtype=torch.float32)
Q_t = torch.from_numpy(query_emb_np).to(device, dtype=torch.float32)

print("Computing similarity matrix...")
with torch.inference_mode():
    sims_np = (Q_t @ G_t.T).cpu().numpy()  # shape: (n_queries, n_gallery)

ranked_indices = np.argsort(-sims_np, axis=1)  # full ranking per query

# gallery_ids, query_ids, query_captions, positive_images from Cell 4.
positive_ranks = find_positive_ranks(gallery_ids, ranked_indices, positive_images)

metrics = compute_single_positive_metrics(positive_ranks)
metrics.update({
    "run_id":    RUN_ID,
    "model":     "PE-Core-G14-448",
    "dataset":   "PAB original attr test",
    "n_queries": len(query_ids),
    "n_gallery": len(gallery_ids),
})

top10_records = build_topk_records(
    query_ids=query_ids,
    query_captions=query_captions,
    positive_images=positive_images,
    gallery_ids=gallery_ids,
    ranked_indices=ranked_indices,
    positive_ranks=positive_ranks,
    topk=10,
)

positive_rank_records = [
    {"query_id": qid, "positive_image": pos, "positive_rank": rank}
    for qid, pos, rank in zip(query_ids, positive_images, positive_ranks)
]

now = datetime.datetime.now().isoformat(timespec="seconds")
extra = {
    "Model": "PE-Core-G14-448",
    "Dataset": "PAB original attr test",
    "Date": now,
}

run_info_lines = [
    "# " + RUN_ID,
    "",
    "**Date:** " + now,
    "",
    "## Purpose",
    "",
    "Local evaluation of PE-Core-G14-448 zero-shot on the original PAB attr test split.",
    "",
    "## Method",
    "",
    "Model: PE-Core-G14-448  ",
    "Training: none  ",
    "Fine-tuning: none  ",
    "Ensemble: none  ",
    "",
    "## Results",
    "",
    "mAP: " + f"{metrics['mAP']:.4f}",
    "R@1: " + f"{metrics['R@1']:.4f}",
    "R@5: " + f"{metrics['R@5']:.4f}",
    "R@10: " + f"{metrics['R@10']:.4f}",
    "Median rank: " + f"{metrics['median_rank']:.0f}",
    "Mean rank: " + f"{metrics['mean_rank']:.1f}",
]

run_info_md = "\n".join(run_info_lines) + "\n"

for run_dir in [LOCAL_RUN, DRIVE_RUN]:
    write_json(run_dir / "metrics.json", metrics)
    write_metrics_markdown(run_dir / "metrics.md", RUN_ID, metrics, extra=extra)
    write_jsonl(run_dir / "rankings_top10.jsonl", top10_records)
    write_jsonl(run_dir / "positive_ranks.jsonl", positive_rank_records)
    (run_dir / "run_info.md").write_text(run_info_md, encoding="utf-8")

print("=== LOCAL EVAL RESULTS ===")
print(f"Queries:     {metrics['queries']}")
print(f"Found any:   {metrics['found_any']}")
print(f"mAP:         {metrics['mAP']:.4f}")
print(f"R@1:         {metrics['R@1']:.4f}")
print(f"R@5:         {metrics['R@5']:.4f}")
print(f"R@10:        {metrics['R@10']:.4f}")
print(f"Median rank: {metrics['median_rank']:.0f}")
print(f"Mean rank:   {metrics['mean_rank']:.1f}")
print()
print("Saved to:", LOCAL_RUN)
print("Saved to:", DRIVE_RUN)

```
