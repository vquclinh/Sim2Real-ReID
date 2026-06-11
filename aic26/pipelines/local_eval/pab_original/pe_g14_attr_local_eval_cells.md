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

assert ("A100" in gpu_name) or gpu_mem_gb >= 35, "PE-Core-G14-448 needs A100-class GPU. Do not run on T4."
assert ram_gb >= 35, "High-RAM runtime is recommended."

print("PASS: runtime is suitable.")
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

# Gallery: sorted unique image paths (relative, e.g. "test/0.jpg")
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

IMG_BATCH       = 96 if total_gb < 60 else 256
NUM_WORKERS     = 4
PREFETCH_FACTOR = 2
TEXT_BATCH      = 256

print("Gallery images:", len(gallery_ids))
print("Query captions:", len(query_captions))
print("IMG_BATCH:", IMG_BATCH)


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
import json, datetime, subprocess
import numpy as np
import torch

RUN_ID    = "local_001_pe_g14_attr_zero_shot"
LOCAL_RUN = Path("/content/aic_local/pab_original/runs") / RUN_ID
DRIVE_RUN = Path("/content/drive/MyDrive/aic2026_data/pab_original/runs") / RUN_ID

LOCAL_RUN.mkdir(parents=True, exist_ok=True)
DRIVE_RUN.mkdir(parents=True, exist_ok=True)

# gallery_ids, query_ids, query_captions, positive_images from Cell 4
# gallery_emb_np, query_emb_np from Cell 6
# device, AUTOCAST_DTYPE, total_gb from Cell 5

gallery_id_to_idx = {rel: i for i, rel in enumerate(gallery_ids)}

G_t = torch.from_numpy(gallery_emb_np).to(device, dtype=torch.float32)
Q_t = torch.from_numpy(query_emb_np).to(device, dtype=torch.float32)

TOPK    = 10
Q_CHUNK = 256

rankings_top10 = []
aps, r_at_1, r_at_5, r_at_10, ranks = [], [], [], [], []

for start in range(0, Q_t.size(0), Q_CHUNK):
    end  = min(start + Q_CHUNK, Q_t.size(0))
    sims = Q_t[start:end] @ G_t.T
    sorted_idx = torch.argsort(sims, dim=1, descending=True).cpu().numpy()

    for local_i, global_i in enumerate(range(start, end)):
        qid     = query_ids[global_i]
        cap     = query_captions[global_i]
        pos_img = positive_images[global_i]
        pos_idx = gallery_id_to_idx.get(pos_img)

        if pos_idx is None:
            rank = len(gallery_ids) + 1
        else:
            rank_arr = np.where(sorted_idx[local_i] == pos_idx)[0]
            rank = int(rank_arr[0]) + 1 if len(rank_arr) > 0 else len(gallery_ids) + 1

        aps.append(1.0 / rank)
        r_at_1.append(1 if rank <= 1 else 0)
        r_at_5.append(1 if rank <= 5 else 0)
        r_at_10.append(1 if rank <= 10 else 0)
        ranks.append(rank)

        top10_ids = [gallery_ids[j] for j in sorted_idx[local_i][:TOPK]]
        rankings_top10.append({
            "query_id":       qid,
            "caption":        cap,
            "positive_image": pos_img,
            "positive_rank":  rank,
            "top10":          top10_ids,
        })

mAP         = float(np.mean(aps))
R1          = float(np.mean(r_at_1))
R5          = float(np.mean(r_at_5))
R10         = float(np.mean(r_at_10))
median_rank = float(np.median(ranks))
mean_rank   = float(np.mean(ranks))

metrics = {
    "run_id":       RUN_ID,
    "model":        "PE-Core-G14-448",
    "dataset":      "PAB original attr test",
    "n_queries":    len(query_ids),
    "n_gallery":    len(gallery_ids),
    "mAP":          round(mAP, 6),
    "R@1":          round(R1, 6),
    "R@5":          round(R5, 6),
    "R@10":         round(R10, 6),
    "median_rank":  median_rank,
    "mean_rank":    round(mean_rank, 2),
}

now = datetime.datetime.now().isoformat(timespec='seconds')

metrics_md = (
    "# Local Eval: " + RUN_ID + "\n\n"
    "**Date:** " + now + "\n\n"
    "| Metric | Value |\n"
    "|---|---|\n"
    "| Model | PE-Core-G14-448 |\n"
    "| Dataset | PAB original attr test |\n"
    "| Queries | " + str(metrics["n_queries"]) + " |\n"
    "| Gallery | " + str(metrics["n_gallery"]) + " |\n"
    "| mAP | " + f"{mAP:.4f}" + " |\n"
    "| R@1 | " + f"{R1:.4f}" + " |\n"
    "| R@5 | " + f"{R5:.4f}" + " |\n"
    "| R@10 | " + f"{R10:.4f}" + " |\n"
    "| Median rank | " + f"{median_rank:.0f}" + " |\n"
    "| Mean rank | " + f"{mean_rank:.1f}" + " |\n"
)

run_info_md = (
    "# " + RUN_ID + "\n\n"
    "**Date:** " + now + "\n\n"
    "## Purpose\n\n"
    "Local evaluation of PE-Core-G14-448 zero-shot on the original PAB attr test split.\n\n"
    "## Method\n\n"
    "Model: PE-Core-G14-448  \nTraining: none  \nFine-tuning: none  \nEnsemble: none  \n\n"
    "## Results\n\n"
    "mAP: " + f"{mAP:.4f}" + "  \n"
    "R@1: " + f"{R1:.4f}"  + "  \n"
    "R@5: " + f"{R5:.4f}"  + "  \n"
    "R@10: " + f"{R10:.4f}" + "  \n"
    "Median rank: " + f"{median_rank:.0f}" + "  \n"
    "Mean rank: " + f"{mean_rank:.1f}" + "  \n"
)

for run_dir in [LOCAL_RUN, DRIVE_RUN]:
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "metrics.md").write_text(metrics_md, encoding="utf-8")
    with open(run_dir / "rankings_top10.jsonl", "w", encoding="utf-8") as f:
        for row in rankings_top10:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "run_info.md").write_text(run_info_md, encoding="utf-8")

print("=== LOCAL EVAL RESULTS ===")
print(f"mAP:         {mAP:.4f}")
print(f"R@1:         {R1:.4f}")
print(f"R@5:         {R5:.4f}")
print(f"R@10:        {R10:.4f}")
print(f"Median rank: {median_rank:.0f}")
print(f"Mean rank:   {mean_rank:.1f}")
print()
print("Saved to:", LOCAL_RUN)
print("Saved to:", DRIVE_RUN)

```
