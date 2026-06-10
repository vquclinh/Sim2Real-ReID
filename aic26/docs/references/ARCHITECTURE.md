# AIC 2026 Track 4 — Pipeline Architecture

Tài liệu kiến trúc đầy đủ của pipeline cho **AI City Challenge 2026 — Track 4: Text-Based Person Anomaly Search**.

> Tài liệu này được viết để người mới vào dự án có thể hiểu được: (1) bài toán đang giải, (2) thuật toán nền tảng từ paper AIO (WWW 2025), (3) 4 upgrades chúng ta áp dụng, (4) chi tiết từng notebook trong pipeline, (5) infrastructure Colab/Drive/Kaggle, và (6) các bug đã fix.

---

## Mục lục

1. [Bài toán và metric](#1-bài-toán-và-metric)
2. [Paper foundation — AIO (WWW 2025)](#2-paper-foundation--aio-www-2025)
3. [4 Upgrades vs paper](#3-4-upgrades-vs-paper)
4. [Pipeline overview — 10 notebooks](#4-pipeline-overview--10-notebooks)
5. [Per-notebook deep dive](#5-per-notebook-deep-dive)
6. [Adaptive Ensemble — Math chi tiết](#6-adaptive-ensemble--math-chi-tiết)
7. [Infrastructure (`aic_colab_utils.py`)](#7-infrastructure-aic_colab_utilspy)
8. [Validation strategy](#8-validation-strategy)
9. [Critical bugs đã fix](#9-critical-bugs-đã-fix)
10. [Storage & disk strategy](#10-storage--disk-strategy)
11. [Run order & checklist](#11-run-order--checklist)

---

## 1. Bài toán và metric

### Task

**Text-Based Person Anomaly Search**: cho một query text mô tả hành vi của một người (có thể bình thường hoặc bất thường), tìm và xếp hạng top-10 ảnh từ gallery 36,773 ảnh real-world chứa người khớp với mô tả đó.

### Dataset — PAB (Pedestrian Anomaly Behavior) ECCV 2026

| Split | Số lượng | Đặc điểm |
|-------|---------|----------|
| Train | 1,013,606 cặp (ảnh, caption) | **Synthetic** (diffusion-generated) → Sim2Real gap |
| Gallery (test) | 36,773 ảnh | **Real-world** |
| Query (test) | 1,978 captions | Real-world descriptions |

**Khó khăn cốt lõi (Sim2Real gap):** model phải học từ ảnh synthetic nhưng infer trên ảnh thật → distribution shift đáng kể.

### Metric

- **Primary**: mAP (mean Average Precision) trên 1,978 queries × 36,773 gallery
- **Secondary**: R@1, R@5, R@10
- **Submission format**: `answer.txt` với 1,978 dòng, mỗi dòng 10 gallery IDs space-separated
- **Single-positive setting**: mỗi query có **1 GT image** trong gallery → AP = `1/(rank+1)` nếu rank<10 else 0

### Deadline

- Test data release: 18/5/2026
- Submission deadline: **10/7/2026 AoE**
- Paper deadline: 24/7/2026 (4-8 pages)
- ECCV workshop: 8-9/9/2026

---

## 2. Paper foundation — AIO (WWW 2025)

Reference: **"Hybrid, Unified and Iterative: A Novel Framework for Text-based Person Anomaly Retrieval"** (Tien-Huy Nguyen et al., ACM WWW 2025) — đạt R@1 = 89.23% trên PAB.

Pipeline AIO có **3 thành phần chính**:

### 2.1 LHP — Local-global Hybrid Perspective

**Mục đích**: model học cả fine-grained details (lúc local view) và holistic context (lúc global view) thay vì chỉ một perspective.

**Probabilistic sampling**: mỗi training sample chọn ngẫu nhiên local hoặc global view:

```
r ~ N(0.5, 1 + δ),   δ ∈ [0, 6]
if r > 0.5:  apply local transform (random crop fine-grained)
else:        apply global transform (full image holistic)
```

**Backbone**: BEiT-3-large @ 384×384, 3 epoch, batch 384.

**Loss**: ITC (Image-Text Contrastive) only:

$$\mathcal{L}_{cl} = -\mathbb{E}\left[\log \frac{\exp(s(f_i, f_t)/\tau)}{\sum_n \exp(s(\tilde{f}_i, f_t)/\tau)}\right]$$

### 2.2 UIT — Unified Image-Text Modeling

**Mục đích**: học joint image-text representation thông qua 4 task đồng thời (multi-task learning).

**Architecture**:
- Image encoder: Swin-B (`swin_base_patch4_window7_224_22k`)
- Text encoder: BERT-base-uncased
- Cross encoder: BERT fusion head (Q from text, K/V from image)
- ITM head: 2-layer MLP → 2-class softmax

**4 losses (combined)**:

$$\mathcal{L} = \mathcal{L}_{itc} + \mathcal{L}_{itm} + \mathcal{L}_{mlm} + 0.1356 \cdot \mathcal{L}_{mim}$$

| Loss | Mục đích |
|------|----------|
| $\mathcal{L}_{itc}$ — Image-Text Contrastive | global retrieval signal (cosine sim) |
| $\mathcal{L}_{itm}$ — Image-Text Matching | cross-encoder binary match/not-match |
| $\mathcal{L}_{mlm}$ — Masked Language Modeling | text token prediction conditioned on image |
| $\mathcal{L}_{mim}$ — Masked Image Modeling | image patch reconstruction (SimMIM) |

**Training**: AdamW lr=1e-5, wd=0.05, cosine LR + warmup 5%, batch 84, 224×224, 22 epoch.

### 2.3 Algorithm 1 — Feature Selection (LHP → UIT)

**Mục đích**: kết nối LHP và UIT — LHP làm "người dẫn đường" chọn candidates cho UIT cross-encoder.

```
Input:  query text q, gallery G
Output: top-K ranked candidates by ITM score

1. Compute LHP similarity: S_lhp[q, g] = f_lhp(q) · f_lhp(g)
2. Select top-K = 256 candidates: K_q = topk(S_lhp[q, ·], K=256)
3. For each (q, g ∈ K_q):
     image_embeds, image_atts = UIT.get_vision_embeds(g)
     text_embeds = UIT.get_text_embeds(q)
     cross = UIT.get_cross_embeds(image_embeds, image_atts, text_embeds, text_atts)
     p_match = softmax(UIT.itm_head(cross[:, 0]))[1]
4. Fuse: S_final[q, g] = α · S_itc + (1-α) · p_match  for g ∈ K_q
        = S_itc                                       for g ∉ K_q
```

→ Đây là **điểm chết người nếu bỏ ITM**: chỉ dùng cosine ITC sẽ vứt 1/4 training signal.

### 2.4 Algorithm 2 — Iterative Ensemble

**Mục đích**: kết hợp 3 models tuần tự, mỗi round "kéo nhẹ" ranking về phía model mới.

**Paper-original (3 round)**:

```
S^(0) = scores_UIT                                   (Round 1, base model)
S^(1) = w_2 · S^(0) + (1 - w_2) · scores_BLIP2       (Round 2, w_2 = 0.9125)
S^(2) = w_3 · S^(1) + (1 - w_3) · scores_CLIP        (Round 3, w_3 = 0.925)
```

**Final ranking**: `pred = topk(S^(2), k=10, dim=1)`.

**Effective contribution**:
- UIT (base): $w_2 \cdot w_3 = 0.9125 \cdot 0.925 = 0.844$ (~84%)
- BLIP-2: $(1-w_2) \cdot w_3 = 0.0875 \cdot 0.925 = 0.081$ (~8%)
- CLIP: $(1-w_3) = 0.075$ (~7.5%)

Weights cố định, tune thủ công bằng grid search.

### 2.5 Kết quả paper

| Method | Train data | R@1 | R@5 | R@10 |
|--------|-----------|-----|-----|------|
| APTM (baseline) | 1M | 69.92 | 95.60 | 97.32 |
| CMP (prior SOTA) | 1M | 79.33 | 97.93 | 98.84 |
| **AIO (paper)** | **1M** | **89.23** | **99.49** | **99.85** |

---

## 3. 4 Upgrades vs paper

| # | Upgrade | Lý do |
|---|---------|-------|
| **1** | **Add PE-Core-G14-448** (Facebook Perception Encoder G14) as 4th model trong ensemble | PE-G14 1.8B params, state-of-the-art zero-shot CLIP-style, mạnh nhất standalone |
| **2** | **Extend iterative loop → 4 rounds** với PE-G14 base (đảo ngược order: strongest first) | Order paper (UIT base) khiến UIT chiếm ~72% effective contribution dù không phải model mạnh nhất |
| **3** | **Replace BEiT-3 → PE-G14-448 trong LHP**, fine-tune bằng **LoRA r=16** | PE-G14 (1.8B) full fine-tune sẽ catastrophic-forget kiến thức zero-shot trên 1M synthetic. LoRA freeze 100% backbone, chỉ train ~8-12M params (~0.5%) trên Q/K/V projection của MHA |
| **4** | **Adaptive Weighting** thay weights cố định | Paper weights $w \in \{0.9125, 0.925, 0.85\}$ tune thủ công, không phù hợp per-query confidence variability |

**Extra upgrade**: k-reciprocal Jaccard re-ranking trên mỗi model trước fusion (+2-3 mAP từ literature).

---

## 4. Pipeline overview — 10 notebooks

```
┌──────────────────────────────────────────────────────────────────────┐
│   00 Manifest QC  →  build train/gallery/query parquets + val splits │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        ▼                         ▼                          ▼
  01a PE-G14 features      01b ViTPose (optional)      (raw data ready)
  - gallery, queries        - keypoints cho LHP
  - val, val_zs              local crop
        │                         │
        │                         ▼
        │                  ┌───────────────────────┐
        │                  │  02 UIT train         │
        │                  │  Swin-B + BERT        │
        │                  │  22 epoch, 4 losses   │
        │                  └──────────┬────────────┘
        │                             │
        ▼                             │
  ┌───────────────────────────────────┴───┐
  │  03 LHP-PE-G14 LoRA train             │
  │  freeze backbone, LoRA Q/K/V          │
  │  ITC loss only, 3 epoch               │
  │  → scores_lhp.pt                      │
  └──────────────────────┬────────────────┘
                         │
        ┌────────────────┼─────────────────┬────────────────┐
        ▼                ▼                 ▼                ▼
  04 UIT inf       05 BLIP-2          06 CLIP          07 PE-G14
  (LHP-guided)     (ITC + ITM         (ViT-L/14        scores from
  + ITM rerank     rerank top-1024)   @ 336)           01a embeds
  Algorithm 1                                          (Upgrade 1)
        │                │                 │                │
        └────────────────┴───────┬─────────┴────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  08 k-reciprocal       │
                    │  rerank (k1=20,k2=6,   │
                    │  λ=0.3) per model      │
                    └───────────┬────────────┘
                                ▼
                    ┌─────────────────────────────┐
                    │  09 Adaptive Ensemble       │
                    │  - per-query z-norm         │
                    │  - 4-signal confidence      │
                    │  - softmax weights          │
                    │  - iterative or single-pass │
                    │  - val gate                 │
                    │  → answer.zip               │
                    └─────────────────────────────┘
```

| # | Notebook | Stage | ETA (A100-80GB) |
|---|----------|-------|-----------------|
| 00 | `00_manifest_qc.ipynb` | Build manifests + val splits (chạy trên Kaggle hoặc Colab) | ~30 min |
| 01a | `01a_pe_g14_features.ipynb` | PE-G14 encode (gallery + queries + val) | ~25-75 min |
| 01b | `01b_vitpose_features.ipynb` | ViTPose keypoints (**OPTIONAL** — chỉ cho LHP anatomical crop) | ~3h |
| 02 | `02_uit_train.ipynb` | UIT training (Swin-B + BERT, 22 epoch) | ~8h |
| 03 | `03_lhp_peg14_train.ipynb` | LHP với PE-G14 + LoRA | ~2-3h |
| 04 | `04_uit_inference.ipynb` | LHP-guided + ITM rerank (Algorithm 1) | ~45-60 min |
| 05 | `05_blip2_inference.ipynb` | BLIP-2 ITC retrieval + ITM rerank | ~1.5-2h |
| 06 | `06_clip_inference.ipynb` | OpenAI CLIP ViT-L/14@336 cosine | ~30 min |
| 07 | `07_pe_g14_scores.ipynb` | Build Q×G score matrix từ 01a embeddings | ~5 min |
| 08 | `08_kreciprocal_rerank.ipynb` | k-reciprocal Jaccard rerank | ~40 min |
| 09 | `09_adaptive_ensemble_submit.ipynb` | Adaptive 4-way fusion + submission | ~15 min |

**Total wall-clock** (skip 01b ViTPose, dùng 01a partial): ~14-16h trên A100 80GB. Có thể split nhiều Colab session với resume từ Drive.

---

## 5. Per-notebook deep dive

### 5.1 `00_manifest_qc.ipynb` — Manifests + Validation Splits

**Mục đích**: parse PAB JSONL annotations + index image files → build canonical parquet manifests + tạo val splits cho gating.

**Algorithm**:

```python
# Step 1: discover dataset structure
annotation_train_dir = find_annotation_train_dir(INPUT_ROOT)   # auto-detect
test_dir              = find_test_dir(INPUT_ROOT)
gallery_dir           = find_gallery_dir(test_dir)

# Step 2: build {(shard, action, stem) → path} index for train images
train_image_index = build_train_image_index(INPUT_ROOT)
# Walks INPUT_ROOT/train/imgs_*/{goal,full,wentwrong}/*.{jpg,webp,png}

# Step 3: parse 75 JSONL files → train_df
for ann_path in annotation_train_dir.glob("imgs_*.json"):
    for line in ann_path:
        obj = json.loads(line)
        row = {
            "row_id": ..., "image_id": obj["image_id"],
            "annotation_image": obj["image"],   # relative path
            "image_path": str(train_image_index.get(key)),
            "caption": obj["caption"], "scene": obj["scene"],
            "label_type": "anomaly" if "anomaly" in obj else "normal",
            "action": obj.get(label_type, ""),
            "missing_image": image_path is None,
        }

# Step 4: gallery + queries (1,978 each)
gallery_df = pd.DataFrame({...})    # 36,773 rows
query_df = pd.DataFrame({...})      # 1,978 queries with q_index, caption, change

# Step 5: STRICT ASSERTIONS
assert len(train_df) == 1_013_606
assert len(gallery_df) == 36_773
assert len(query_df) == 1_978

# Step 6: validation splits (seed=20260514, deterministic)
val_zs_df = resolved[resolved.scene == random_scene]   # 1 scene class hold-out
val_df = stratified_sample(resolved \ val_zs, frac=0.05, by=[label_type, scene])
train_trainable_df = train \ (val_zs ∪ val)
```

**Outputs**:
- `train_manifest.parquet` (1,013,606 rows)
- `train_manifest_trainable.parquet` (~95% = ~962K rows, dùng để train UIT + LHP)
- `gallery_manifest.parquet` (36,773)
- `query_manifest.parquet` (1,978)
- `val_split.parquet` (~5% in-distribution)
- `val_zeroshot_scene.parquet` (OOD proxy)
- `manifest_summary.json` (counts + metadata)

**Note**: notebook hoạt động trên cả Kaggle (`/kaggle/input/` → `/kaggle/working/`) và Colab (qua `aic_colab_utils`).

---

### 5.2 `01a_pe_g14_features.ipynb` — PE-Core-G14-448 Encoding

**Model**: PE-Core-G14-448 from Meta's `facebookresearch/perception_models` repo. 1.8B params (~1.6B vision + 200M text), output dim 1280, image size 448.

**Configuration**:

```python
PE_MODEL_NAME = 'PE-Core-G14-448'
IMAGE_BATCH_SIZE = 256        # A100-80GB tuned
TEXT_BATCH_SIZE = 512
NUM_WORKERS = 12
USE_BF16 = True               # A100 native
CHANNELS_LAST = True
USE_COMPILE = False           # ← DISABLED do CUDA Graphs bug với PE-G14 RoPE cache

# Run flags (default tối ưu cho pipeline)
RUN_TRAIN_IMAGE_EMBEDDINGS = False   # ← train embeds không ai consume, tiết kiệm 5h
RUN_TRAIN_TEXT_EMBEDDINGS = False
RUN_GALLERY_IMAGE_EMBEDDINGS = True  # ← required cho 07 (Round 4 ensemble)
RUN_QUERY_TEXT_EMBEDDINGS = True
RUN_VAL_IMAGE_EMBEDDINGS = True      # ← required cho 03 fallback compare
RUN_VAL_TEXT_EMBEDDINGS = True
```

**Encoding logic**:

```python
# Load model
pe_model = pe.CLIP.from_config(PE_MODEL_NAME, pretrained=True).eval()
pe_model = pe_model.to(device, memory_format=torch.channels_last)

@torch.inference_mode()
def encode_image(x):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        feats = pe_model.encode_image(x)
    return F.normalize(feats.float(), dim=-1)   # L2 norm, fp16 save

@torch.inference_mode()
def encode_text(tokens):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        feats = pe_model.encode_text(tokens)
    return F.normalize(feats.float(), dim=-1)
```

**Outputs** (NPZ format, chunked):
- `features/pe_g14/gallery.npz`: `{ids, paths, embeddings (36773, 1280) fp16, ok}`
- `features/pe_g14/queries.npz`: `{ids, embeddings (1978, 1280) fp16}`
- `features/pe_g14/val.npz`, `val_text.npz`
- `features/pe_g14/val_zs.npz`, `val_zs_text.npz`

**Resumability**: `chunk_done()` check local SSD + Drive trước khi encode. Async sync mỗi chunk → Drive.

---

### 5.3 `01b_vitpose_features.ipynb` — ViTPose Keypoints (OPTIONAL)

**Model**: RTDetr (`PekingU/rtdetr_r50vd_coco_o365`) + ViTPose++ (`usyd-community/vitpose-plus-huge`).

**Pipeline**:
1. RTDetr detect persons trên ảnh (top-1 per image, score × area ranking)
2. Crop person bbox → ViTPose++ infer 17 COCO keypoints (x, y, confidence)
3. Normalize keypoints về [0,1] image space

**Output**: `pose/vitposepp/train/chunk_*.npz` + `pose/vitposepp/gallery.npz` với keypoints (17, 3) per image.

**Vai trò trong pipeline**: chỉ dùng cho **LHP anatomical crop** ở `03` (Upgrade enhancement, không có trong paper gốc — paper dùng random crop).

**Skippable**: Pipeline có guard fallback — nếu pose dict rỗng, LHP local view sẽ dùng random crop thay anatomical crop (paper-faithful). Tiết kiệm ~3h khi skip.

---

### 5.4 `02_uit_train.ipynb` — UIT Training (Paper-faithful)

**Strategy**: clone paper repo, generate runtime config + JSONL, launch paper's `Search.py` via subprocess.

**Pipeline**:

```python
# Step 1: Clone paper repo (Drive cache → local SSD)
UIT_REPO = LOCAL_ROOT / 'aio_repo'
rsync DRIVE_AIO_REPO → UIT_REPO   # contains uit/cmp/, lhp_2/, etc.

# Step 2: Stage data via symlink (NO copy)
DATA_ROOT = UIT_REPO / 'data'
os.symlink(INPUT_ROOT, DATA_ROOT / 'PAB')   # paper expects data/PAB/...
# → data/PAB/annotation/, data/PAB/train/imgs_*/, data/PAB/name-masked_test-set/

# Step 3: Generate train_trainable JSONL (filter val rows)
trainable_df = pd.read_parquet('train_manifest_trainable.parquet')
groups = group_by(trainable_df, 'annotation_file')
for name, items in groups.items():
    write_jsonl(INPUT_ROOT / 'annotation' / 'train_trainable' / name, items)

# Step 4: Override paper config
cfg = load_yaml('configs/cmp.yaml')
cfg['batch_size_train'] = 256                # A100-80GB (paper 84)
cfg['optimizer']['lr'] = 1e-5
cfg['schedular']['epochs'] = 22              # paper-faithful
cfg['train_file'] = [trainable_jsonls]       # exclude val rows
write_yaml('configs/cmp_a100_80gb.yaml', cfg)

# Step 5: Launch training subprocess
cmd = [python, '-m', 'torch.distributed.run', '--nproc_per_node=1',
       'Search.py', '--config', 'configs/cmp_a100_80gb.yaml',
       '--task', 'cmp', '--bs', '256', '--epo', '22', '--seed', '20260514']
subprocess.run(cmd, cwd=UIT_CODE)

# Step 6: Background thread sync checkpoint to Drive per epoch
# (Drive resume logic: latest checkpoint_<n>.pth restored ở Step 1)
```

**Losses (paper's CMP class)**:

```python
class CMP(nn.Module):
    def get_vision_embeds(self, image):     # Swin-B forward
    def get_text_embeds(self, ids, atts):   # BERT.bert(mode='text')
    def get_cross_embeds(self, img_e, img_a, txt_e, txt_a):
        # BERT.bert(encoder_embeds=txt, encoder_hidden_states=img, mode='fusion')
    def get_image_feat(self, embeds):       # avgpool + vision_proj → ITC feat
    def get_text_feat(self, embeds):        # avgpool + text_proj → ITC feat
    def get_contrastive_loss(...)           # L_itc
    def get_matching_loss(...)              # L_itm via itm_head(cross[:, 0])
    def get_mlm_loss(...)                   # L_mlm
    def get_mim_loss(...)                   # L_mim via SimMIM

# Combined: L = L_itc + L_itm + L_mlm + 0.1356 * L_mim
```

**Output**: `checkpoints/uit/checkpoint_best.pth` (~2-3GB).

---

### 5.5 `03_lhp_peg14_train.ipynb` — LHP với PE-G14 LoRA (Upgrade 3)

**Architecture diff vs paper**:

| | Paper | Our pipeline |
|---|-------|--------------|
| Backbone | BEiT-3-large @ 384 | PE-Core-G14-448 |
| Fine-tune | Full | **LoRA r=16 trên Q/K/V** |
| Trainable params | 600M | ~8-12M (~0.5%) |
| LR | 1e-4 | **3e-4** (LoRA-friendly) |
| Batch | 384 | 128 |
| Epochs | 3 | 3 |
| Image size | 384 | 448 |

**Why LoRA**: PE-G14 1.8B full FT trên 1M synthetic captions sẽ catastrophic-forget zero-shot pretraining → mất generalization về real-world test. LoRA chỉ train ~8-12M params, giữ alignment.

**Pipeline**:

```python
# Step 1: Load PE-G14, freeze 100%
pe_model = pe.CLIP.from_config('PE-Core-G14-448', pretrained=True).eval()
for p in pe_model.parameters(): p.requires_grad = False

# Step 2: Inject LoRA on Q/K/V (auto-detect target modules)
attn_targets = scan_attention_modules(pe_model)   # finds 'q_proj','k_proj','v_proj' or fused 'in_proj'/'qkv'
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      bias='none', target_modules=attn_targets)
pe_model = get_peft_model(pe_model, lora_cfg)
# Trainable: ~8-12M params

# Step 3: Projection head 1280 → 768 (extra ~1M params trainable)
proj_head = nn.Linear(1280, 768, bias=False).to(device)
# CRITICAL: proj_head MUST be applied at BOTH training AND inference (xem Fix 1, §9)

# Step 4: LHP transform (probabilistic local/global)
def lhp_transform(pil, kp):
    r = np.random.normal(0.5, 1 + LHP_DELTA)   # δ=3.0
    if r > 0.5 and kp is not None:
        return anatomical_crop(pil, kp)         # face / torso / lower body crop
    else:
        return global_aug(pil)                  # RandomResizedCrop + flip + jitter

# Step 5: ITC training loop (3 epoch)
for batch in train_loader:
    img_f = proj_head(pe_model.encode_image(imgs))   # (B, 768)
    txt_f = proj_head(pe_model.encode_text(tokens))  # (B, 768)
    img_f, txt_f = F.normalize(img_f), F.normalize(txt_f)
    logits = logit_scale * img_f @ txt_f.T
    loss = 0.5 * (CE(logits, range(B)) + CE(logits.T, range(B)))

    # Per-epoch val mAP@10 probe (single-positive: AP = 1/(rank+1) if rank<10)
    if val_mAP10 > best:
        save_pretrained(LHP_CKPT_DIR / 'lora_best')   # adapter ~40MB
        torch.save(proj_head.state_dict(), 'proj_head.pt')

# Step 6: Inference — MERGE LoRA + apply proj_head (Fix 1)
pe_model = pe_model.merge_and_unload()   # baked into base for speed
# Encode gallery, queries, val, val_zs through encode_image/text + proj_head

# Step 7: Algorithm 1 setup
S_lhp = F.normalize(q_emb) @ F.normalize(g_emb).T   # (1978, 36773)
torch.save({'scores': S_lhp, 'query_ids', 'gallery_ids'}, 'scores_lhp.pt')

# Step 8: Fallback gate
if best_val_mAP10 < zero_shot_PE_G14_val_mAP10:
    use_lora = False   # → 04 dùng vanilla PE-G14 cho LHP guidance
```

**Outputs**:
- `checkpoints/lhp_peg14/lora_best/` (LoRA adapter ~40MB + proj_head.pt ~3MB)
- `features/lhp_peg14/{gallery,queries,val,val_zs}.npz` (768-dim L2-norm fp16)
- `features/lhp_peg14/scores_lhp.pt` ← **input cho Algorithm 1 ở 04**

---

### 5.6 `04_uit_inference.ipynb` — Algorithm 1 (LHP-guided ITM Rerank)

**THE most important inference notebook**. Implement Algorithm 1 đầy đủ — bỏ qua bước này = ném đi 1/4 training signal (ITM loss).

**Pipeline**:

```python
# Step 1: Load UIT checkpoint + tokenizer + image transform
model = CMP(config=cfg)
model.load_state_dict(torch.load('checkpoint_best.pth')['model'])

# Step 2: Encode gallery → BOTH pooled feat AND token-level image_embeds
@torch.inference_mode()
def encode_images_full(df, id_col):
    pooled, tokens, atts = [], [], []
    for batch in loader:
        image_embeds, image_atts = model.get_vision_embeds(x)   # (B, T_v, D_v)
        image_feat = model.get_image_feat(image_embeds)         # (B, embed_dim)
        pooled.append(F.normalize(image_feat))
        tokens.append(image_embeds.half().cpu())                # offload for ITM later
        atts.append(image_atts.cpu())
    return pooled, tokens, atts

g_pooled, g_tokens, g_atts = encode_images_full(gallery_df)
# g_tokens shape: (36773, ~50, 1024) fp16 ≈ 3.7GB on CPU/GPU (fits 80GB)

# Step 3: Encode queries → BOTH pooled + token-level text_embeds
q_pooled, q_tokens, q_atts = encode_texts_full(query_captions)

# Step 4: ITC baseline (cosine sim of pooled feats)
S_itc = F.normalize(q_pooled) @ F.normalize(g_pooled).T   # (1978, 36773)
torch.save({'scores': S_itc}, 'scores_uit_itc_only.pt')   # ablation

# Step 5: Algorithm 1 — Feature Selection from LHP guidance
lhp_payload = torch.load('features/lhp_peg14/scores_lhp.pt')
lhp_S_aligned = align_ids(lhp_payload, q_ids, g_ids).to(device)
lhp_topvals, lhp_topidx = torch.topk(lhp_S_aligned, ITM_TOPK=256, dim=1)
# → cho mỗi query, có top-256 candidates do LHP đề xuất

# Step 6: ITM rerank trên top-K candidates
ITM_FUSION_ALPHA = 0.4
S_final = S_itc.clone()
for q in range(len(q_ids)):
    cand_idx = lhp_topidx[q]                          # (256,) gallery indices
    text_e = q_tokens[q:q+1].to(device)
    text_a = q_atts[q:q+1].to(device)
    # Process 32 candidates at a time
    for s in range(0, 256, ITM_BATCH=32):
        img_e = g_tokens[cand_idx[s:s+32]].to(device)
        img_a = g_atts[cand_idx[s:s+32]].to(device)
        cross = model.get_cross_embeds(img_e, img_a,
                                       text_e.expand(...), text_a.expand(...))
        logits = model.itm_head(cross[:, 0])          # (B, 2)
        p_match = F.softmax(logits, dim=1)[:, 1]      # P(match)
        itm_scores[s:s+32] = p_match
    # Convex combine: final = α · ITC + (1-α) · ITM cho top-K positions
    fused = 0.4 * S_itc[q, cand_idx] + 0.6 * itm_scores
    S_final[q, cand_idx] = fused

torch.save({'scores': S_final, 'note': 'ITC ∪ ITM-rerank top-256'}, 'scores_uit.pt')
```

**Critical**: cross-encoder ITM của UIT chỉ chạy trên **top-K=256 candidates do LHP chọn** (Algorithm 1). Chạy ITM trên cả 36,773 gallery sẽ tốn vô cùng (1978 × 36773 = 73M cross-encoder calls). Top-K=256 → 506K calls, ~30 min.

**Outputs**:
- `features/uit/{gallery,queries,val,val_zs}.npz` (ITC pooled embeds)
- `features/uit/scores_uit.pt` ← **finalRound 1 score** (Algorithm 1 applied)
- `features/uit/scores_uit_itc_only.pt` (ablation: ITC only, không có Algorithm 1)

---

### 5.7 `05_blip2_inference.ipynb` — BLIP-2 ITC + ITM Rerank (Round 2)

**Model**: LAVIS `blip2_image_text_matching` (Salesforce BLIP-2 với ViT-g vision tower + Q-Former).

**Pipeline** tương tự `04` nhưng dùng BLIP-2's own ITC cho top-K selection (không phải LHP):

```python
model = lavis.load('blip2_image_text_matching', model_type='pretrain', is_eval=True)

# 1. Extract ITC features
g_feats = [model.extract_features({'image': batch}, mode='image').image_embeds_proj[:, 0]
           for batch in gallery_loader]
q_feats = [model.extract_features({'text_input': txt}, mode='text').text_embeds_proj[:, 0]
           for txt in query_texts]

S_itc = F.normalize(q_feats) @ F.normalize(g_feats).T

# 2. ITM rerank top-K=1024 (BLIP-2 ITM nhẹ hơn UIT, K lớn được)
ITM_TOPK = 1024
ITC_ALPHA = 0.4
itc_topvals, itc_topidx = torch.topk(S_itc, ITM_TOPK, dim=1)
for q in range(len(query_texts)):
    text = query_texts[q]
    for batch in itc_topidx[q].chunked(32):
        pil_batch = [open_image(gallery_paths[i]) for i in batch]
        itm_out = model({'image': stack(pil_batch), 'text_input': [text]*len(batch)},
                        match_head='itm')
        p_match = F.softmax(itm_out, dim=1)[:, 1]
    fused = ITC_ALPHA * itc_topvals[q] + (1-ITC_ALPHA) * itm_scores
    S_final[q, itc_topidx[q]] = fused

torch.save({'scores': S_final}, 'scores_blip2.pt')
```

**Outputs**: `features/blip2/scores_blip2.pt`.

---

### 5.8 `06_clip_inference.ipynb` — OpenAI CLIP (Round 3)

**Model**: `openai/clip-vit-large-patch14-336` (KHÔNG phải EVA-CLIP — paper-faithful dùng OpenAI weights).

```python
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336').eval()
model = model.to(device, memory_format=torch.channels_last)

g_emb = model.get_image_features(...) → F.normalize
q_emb = model.get_text_features(...) → F.normalize
S = q_emb @ g_emb.T   # cosine only, no rerank

torch.save({'scores': S.half()}, 'scores_clip.pt')
```

**Outputs**: `features/clip/scores_clip.pt`.

---

### 5.9 `07_pe_g14_scores.ipynb` — PE-G14 Round 4 (Upgrade 1)

Tiny notebook — chỉ build score matrix từ embeddings đã có ở `01a`:

```python
z_g = np.load('features/pe_g14/gallery.npz')
z_q = np.load('features/pe_g14/queries.npz')
Q = F.normalize(torch.from_numpy(z_q['embeddings']).cuda())
G = F.normalize(torch.from_numpy(z_g['embeddings']).cuda())
S = (Q @ G.T).half().cpu()
torch.save({'scores': S, 'query_ids': z_q['ids'], 'gallery_ids': z_g['ids']},
           'features/pe_g14/scores_pe.pt')
```

**Outputs**: `features/pe_g14/scores_pe.pt`.

**ETA**: ~5 min (chỉ matmul 1978×1280 @ 1280×36773 fp32 = 1 GFLOP).

---

### 5.10 `08_kreciprocal_rerank.ipynb` — k-Reciprocal Re-ranking (Extra Upgrade)

**Algorithm** (Zhong et al. CVPR 2017, adapted cho text→image cross-modal):

```python
# Per model m, given score matrix S_m (Q × G) and gallery embeddings G_m:

# Step 1: Build gallery-gallery k-NN graph
GG = F.normalize(G_m) @ F.normalize(G_m).T              # (G, G)
GG_topk1_idx = torch.topk(GG, k1=20, dim=1).indices     # (G, k1)

# Step 2: k-reciprocal neighbor set per gallery item
# m ∈ R(g) iff m ∈ topk1(g) AND g ∈ topk1(m)  (mutual)
rec_sets = []
for g in range(G):
    cand = GG_topk1_idx[g]
    rec = [c for c in cand if g in topk1[c]]   # mutual check
    rec_sets.append(set(rec))

# Step 3: For each query, rerank top-200
TOPK_RR = 200
for q in range(Q):
    cands = topk_S(q)[:200]                    # original top-200 by S_m
    R_q = union(rec_sets[c] for c in cands[:k1])

    for i, g in enumerate(cands):
        R_g = rec_sets[g] | {g}
        d_jaccard[i] = 1 - |R_q ∩ R_g| / |R_q ∪ R_g|

    d_orig = 1 - S_m[q, cands]                  # cosine → distance
    d_final = (1-λ) * d_orig + λ * d_jaccard    # λ=0.3
    S_m[q, cands] = 1 - d_final                 # back to similarity
```

**Outputs**: `features/<m>/scores_<m>_rr.pt` cho mỗi m ∈ {uit, blip2, clip, pe_g14}.

**Why before fusion**: k-reciprocal cải thiện score distribution per-model → adaptive confidence ở `09` chính xác hơn.

---

### 5.11 `09_adaptive_ensemble_submit.ipynb` — Adaptive 4-way + Submission (Upgrades 2 + 4)

Notebook quan trọng nhất — chứa cả 2 upgrades cuối + generate `answer.zip`.

Xem [§6 Adaptive Ensemble Math](#6-adaptive-ensemble--math-chi-tiết) để hiểu thuật toán.

**Pipeline**:

```python
# Step 1: Load 4 score matrices (prefer _rr versions from 08)
score_payloads = [load_scores(m) for m in ['uit', 'blip2', 'clip', 'pe_g14']]
S_list = [align_ids(p, canon_q, canon_g) for p in score_payloads]   # 4 × (Q, G)

# Step 2: Per-query z-normalization (CRITICAL — BLIP-2 ITM logits vs cosine scale mismatch)
Stil_list = [per_query_zscore(S) for S in S_list]

# Step 3: 4 confidence signals per (model, query)
conf, topk_idx, topk_vals = compute_confidence(Stil_list, K=20)
# conf[M, Q, 4] = [top1_norm, margin_z, neg_entropy, agreement_with_other_models]

# Step 4: Adaptive weights
c[m, q] = α·top1 + β·margin - γ·entropy + λ·agreement
W[m, q] = (1 - M·η) · softmax(c[m,q] / T) + η   # floor η=0.05

# Step 5: Single-pass adaptive fusion
S_adaptive = Σ_m W[m, :, None] * Stil_list[m]

# Step 6: Iterative 4-round (Upgrade 2 — PE-G14 base, reversed order)
ITER_ORDER = ['pe_g14', 'clip', 'blip2', 'uit']
ITER_WEIGHTS = [0.85, 0.925, 0.9125]   # mix-in CLIP, BLIP-2, UIT
S_iter = Stil_list[SLOT['pe_g14']].clone()
for t, m in enumerate(ITER_ORDER[1:]):
    S_iter = ITER_WEIGHTS[t] * S_iter + (1 - ITER_WEIGHTS[t]) * Stil_list[SLOT[m]]
# Effective contribution: PE-G14 71.7%, CLIP 12.7%, BLIP-2 6.8%, UIT 8.8%

# Step 7: Val gate — pick winner by mAP@10
metrics = {cfg: eval(cfg, 'val') for cfg in ['uniform', 'iter4_paper', 'iter4_reversed',
                                              'iter4_grid_best', 'adaptive_F']}
chosen = argmax_with_threshold(metrics)   # auto-fallback nếu adaptive không beat fixed

# Step 8: Grid search ITER_WEIGHTS (5×4×4=80 configs) → iter4_grid_best
for (w2, w3, w4) in grid:
    score = mean(val_mAP10, val_zs_mAP10)
    if score > best: best_weights = (w2, w3, w4)

# Step 9: Top-10 submission
S_FINAL = {'adaptive_F': S_adaptive, 'iter4_grid_best': S_iter_grid, ...}[chosen]
top_vals, top_idx = torch.topk(S_FINAL, 10, dim=1)
lines = [' '.join(gallery_ids[idx] for idx in top_idx[q]) for q in range(1978)]
write_answer_txt(lines)
zip_answer()
```

**Outputs**:
- `submission/answer.txt` (1978 dòng × 10 gallery IDs)
- `submission/answer.zip`
- `submission/debug_topk.parquet` (per-query top-10 + per-model weights)
- `validation/adaptive_weights_meta.json` (chosen fusion + hyperparams)
- `validation/ablation_table.parquet` (mAP@10/R@1 cho từng config)
- `validation/iter_weight_grid.parquet` (top weight configs từ grid search)

---

## 6. Adaptive Ensemble — Math chi tiết

### 6.1 Per-query z-normalization (preprocessing)

```math
\tilde{s}_m(q, g) = \frac{s_m(q, g) - \mu_m(q)}{\sigma_m(q) + \epsilon}, \quad \epsilon = 10^{-6}
```

trong đó $\mu_m(q) = \frac{1}{|G|} \sum_g s_m(q, g)$, $\sigma_m(q) = \text{std}_g(s_m(q, g))$.

**Tại sao critical**: BLIP-2 ITM xuất logits cross-entropy, range khác hoàn toàn với CLIP/PE-G14 cosine $\in [-1, 1]$. Không z-norm → model có scale lớn nhất sẽ dominate fusion bất kể weight.

### 6.2 4 confidence signals

Cho mỗi model $m$ và query $q$, lấy top-K=20 similarity values $s^{(m)}_{(1)}(q) \geq s^{(m)}_{(2)}(q) \geq \dots \geq s^{(m)}_{(K)}(q)$:

**a) Normalized top-1**:
```math
\tilde{s}_{\text{top1}}^{(m)}(q) = \frac{s_{(1)}^{(m)}(q) - \min_g s_m(q,g)}{\max_g s_m(q,g) - \min_g s_m(q,g) + \epsilon}
```

**b) Margin (top-1 vs top-K mean, std-normalized)**:
```math
\text{margin}_m(q) = \frac{s_{(1)}^{(m)}(q) - \frac{1}{K-1}\sum_{k=2}^{K} s_{(k)}^{(m)}(q)}{\sigma_{\text{top-K}}^{(m)}(q) + \epsilon}
```

**c) Entropy (truncated top-K softmax)**:
```math
p_m^{(k)}(q) = \frac{\exp(s_{(k)}^{(m)}(q)/\tau_e)}{\sum_{j=1}^{K}\exp(s_{(j)}^{(m)}(q)/\tau_e)}, \quad \tau_e = 0.05
```
```math
\tilde{H}_m(q) = \frac{-\sum_{k} p_m^{(k)}(q) \log p_m^{(k)}(q)}{\log K} \in [0, 1]
```

Low entropy = peaked distribution = high confidence.

**d) Inter-model agreement** (fraction of top-K overlap với 3 model còn lại):
```math
\text{agr}_m(q) = \frac{1}{M-1} \sum_{m' \neq m} \frac{|\text{topK}_m(q) \cap \text{topK}_{m'}(q)|}{K}
```

### 6.3 Composite confidence + softmax weights

```math
c_m(q) = \alpha \cdot \tilde{s}_{\text{top1}}^{(m)}(q) + \beta \cdot \text{margin}_m(q) - \gamma \cdot \tilde{H}_m(q) + \lambda \cdot \text{agr}_m(q)
```

Defaults sau val tuning: $\alpha = 1.0$, $\beta = 1.0$, $\gamma = 0.5$, $\lambda = 0.5$.

**Softmax với floor**:
```math
w_m(q) = (1 - M \cdot \eta) \cdot \frac{\exp(c_m(q)/T)}{\sum_{m'} \exp(c_{m'}(q)/T)} + \eta
```

với $M = 4$, $\eta = 0.05$ (floor — không model nào bị disabled hoàn toàn), $T = 1.0$ (softmax temperature).

### 6.4 Final fusion

**Single-pass (recommended)**:
```math
S(q, g) = \sum_{m=1}^{M} w_m(q) \cdot \tilde{s}_m(q, g)
```

**Iterative 4-round (Upgrade 2, paper-style)**:
```math
S^{(0)} = \tilde{s}_{\text{PE-G14}}
```
```math
S^{(t)} = w_{t+1} \cdot S^{(t-1)} + (1 - w_{t+1}) \cdot \tilde{s}_{m_{t+1}}, \quad m_t \in [\text{CLIP}, \text{BLIP-2}, \text{UIT}]
```

với $w_2 = 0.85$, $w_3 = 0.925$, $w_4 = 0.9125$. Effective contribution sau 3 round:

| Model | Contribution |
|-------|--------------|
| PE-G14 (base) | $w_2 \cdot w_3 \cdot w_4 = 0.85 \cdot 0.925 \cdot 0.9125 \approx 71.7\%$ |
| CLIP | $(1-w_2) \cdot w_3 \cdot w_4 = 0.15 \cdot 0.925 \cdot 0.9125 \approx 12.7\%$ |
| BLIP-2 | $(1-w_3) \cdot w_4 = 0.075 \cdot 0.9125 \approx 6.8\%$ |
| UIT | $(1-w_4) = 0.0875 \approx 8.8\%$ |

### 6.5 Val gate

Trên held-out val (val_split + val_zeroshot_scene), eval 5 configs:

1. `uniform_4way` — 4-way mean (sanity baseline)
2. `iter4_paper` — UIT base, paper order (diagnostic)
3. `iter4_reversed` — PE-G14 base, paper weights
4. `iter4_grid_best` — PE-G14 base, val-tuned weights via 80-config grid
5. `adaptive_F` — single-pass adaptive

**Selection metric**: mAP@10. **Threshold**: chỉ promote nếu beat current chosen by ≥0.5%. Default winner: `iter4_reversed`.

---

## 7. Infrastructure (`aic_colab_utils.py`)

Module chia sẻ giữa toàn bộ notebooks, providing:

### 7.1 `setup_aic2026_environment()` — Bootstrap

6 bước tuần tự:

1. **Mount Drive** (`google.colab.drive.mount('/content/drive')`)
2. **Load Kaggle credentials** (`kaggle.json` từ Drive hoặc arg)
3. **Restore raw + manifests từ Drive → local SSD**:
   - Ưu tiên `restore_raw_from_tar_split()` (~30-40 min cho 100GB) nếu Drive có `raw_tar_parts/`
   - Fallback rsync many-files (legacy, ~30-60 min)
4. **Download missing datasets** qua `kagglehub.dataset_download()` (chỉ khi marker `.synced_*` không có)
5. **Resume output chunks** từ Drive (incremental)
6. **Resolve concrete subdirs** (`annotation_train_dir`, `test_dir`, `gallery_dir`) — auto-detect cả `PAB/` prefix lẫn không

**Returns** `paths` dict với 9 keys: `drive_root`, `local_root`, `input_root`, `manifest_dir`, `output_root`, `drive_output_root`, `annotation_train_dir`, `test_dir`, `gallery_dir`.

### 7.2 Drive persistence — 2 strategies

**Tar-split** (recommended cho 100GB raw):
```python
mirror_raw_as_tar_split(paths, part_size='4500M')
# tar -chf - -C local_root raw | split -b 4500M - drive/raw_tar_parts/raw.tar.part_
# -h flag DEREFERENCE symlinks (quan trọng nếu raw là symlink farm tới kagglehub cache)
```

**Rsync** (legacy, 3-10h cho many-small-files):
```python
mirror_dataset_to_drive(paths, include_raw=True, include_manifests=True)
```

### 7.3 Async chunk sync

```python
sync_chunk_to_drive(local_path, local_root, drive_output_root, background=True)
# Spawns daemon thread → shutil.copyfile(local_path, drive_path)
# Single _SYNC_LOCK throttles concurrent Drive writes (FUSE doesn't parallelize well)

wait_for_pending_syncs(timeout=600)   # cuối notebook để flush
```

### 7.4 Resume logic

```python
chunk_done(chunk_path, drive_output_root, local_root) → bool
# True nếu chunk đã tồn tại ở local SSD HOẶC Drive (auto-restore từ Drive nếu cần)

find_existing_chunks(*output_dirs, pattern='chunk_*.npz') → set[str]
# Union basename across multiple dirs (local + Drive)
```

### 7.5 GPU tuning

```python
select_a100_device(prefer_a100=True, verbose=True) → torch.device
# Auto-pick A100/H100 nếu có
# Enable: TF32 matmul, cuDNN benchmark, Flash SDPA, mem-efficient SDPA
```

### 7.6 Drive FUSE robustness

```python
_robust_drive_mkdir(path, retries=3, sleep_base=1.0) → bool
# 3 strategies trong mỗi attempt:
#   1. Python pathlib.mkdir(parents=True, exist_ok=True)
#   2. Shell `mkdir -p` qua subprocess (bypass FUSE quirks)
#   3. Re-check exists() — FUSE đôi khi raise nhưng folder vẫn tạo được
# Exponential backoff 1s → 2s → 4s
```

---

## 8. Validation strategy

### 8.1 Splits (deterministic, seed=20260514)

| Split | % of train | Stratification | Purpose |
|-------|-----------|----------------|---------|
| `val_split` | 5% (~50K rows) | `label_type × scene` | In-distribution val, tune hyperparams |
| `val_zeroshot_scene` | 1 random scene class | Per-scene | OOD proxy (Sim2Real-ish) |
| `train_manifest_trainable` | 90% còn lại | — | Train UIT + LHP |

### 8.2 Metric — mAP@10 (single-positive)

**KHÔNG dùng full mAP**. AIC submission yêu cầu top-10 → val gate phải match:

```python
def mAP_at_10(S, q_ids, g_ids):
    """Single-positive: AP = 1/(rank+1) if rank<10 else 0."""
    ranks = [rank_of_gt(S[q], q_ids[q], g_ids) for q in range(len(q_ids))]
    ap10 = [1.0 / (r + 1) if r < 10 else 0.0 for r in ranks]
    return mean(ap10)
```

### 8.3 Auto-fallback gate

Tất cả notebook training (02 UIT, 03 LHP) + ensemble (09):
- Compute mAP@10 trên val
- Compare với baseline (zero-shot model HOẶC fixed-weight ensemble)
- Auto-revert nếu trained version < baseline - 0.5%
- Ghi quyết định vào JSON: `fallback_decision.json` / `adaptive_weights_meta.json`

---

## 9. Critical bugs đã fix

### 9.1 Bug 1 — `proj_head` orphan trong `03_lhp_peg14_train.ipynb`

**Trước fix**: training tính `proj_head(encode_image(x))` cho ITC loss, nhưng val eval và final inference đều **bỏ qua proj_head**, dùng raw 1280-d. LoRA weights được tối ưu để hoạt động với proj_head → bị orphan.

**Fix**: pipe through proj_head ở cả 3 chỗ (train, val eval, inference). Fallback case (LoRA loses) set `proj_head = None` để dùng raw 1280-d zero-shot PE-G14.

### 9.2 Bug 2 — Bỏ qua Algorithm 1 trong `04_uit_inference.ipynb`

**Trước fix**: tính `S_uit = Q @ G.T` cosine only — đây chỉ là ITC. UIT trained 22 epoch với ITC + ITM + MLM + MIM → bỏ ITM nghĩa là ném đi 1/4 training signal.

**Fix**: implement đúng Algorithm 1:
1. Load `scores_lhp.pt` (LHP guidance)
2. Top-K=256 candidates per query
3. Cross-encoder + itm_head → P(match)
4. Replace top-K positions: `S_final[q, top_k] = α·ITC + (1-α)·ITM`

### 9.3 Bug 3 — Iterative ensemble order sai (PE-G14 chỉ 15%)

**Trước fix**: order paper UIT→BLIP-2→CLIP→PE-G14 với weights [0.9125, 0.925, 0.85] khiến UIT chiếm 71.7% effective contribution, PE-G14 (model mạnh nhất standalone) chỉ 15%.

**Fix**: đảo order PE-G14→CLIP→BLIP-2→UIT (strongest first). Effective contribution mới: PE-G14 71.7%, CLIP 12.7%, BLIP-2 6.8%, UIT 8.8%. Bonus: grid search 80 weight configs trên val mAP@10.

### 9.4 Bug 4 — Metric mismatch (Recall vs mAP)

**Trước fix**: paper tối ưu R@1, val gate dùng top-1 accuracy. AIC 2026 chấm bằng **mAP@10** — nếu GT ở rank 10 (Recall@10=1 nhưng AP=0.1) bị undercount.

**Fix**: đổi mọi val gate criterion → mAP@10. Compute đồng thời mAP@10, mAP@100, R@1/5/10 cho diagnostic.

### 9.5 Bug 5 — `torch.compile(mode='reduce-overhead')` crash với PE-G14

**Trước fix**: CUDA Graphs reuse memory buffers, PE-G14's `rope.py:338` lưu `self.freq = freq[None, ...]` cache → batch sau overwrite buffer → crash với "accessing tensor output of CUDAGraphs that has been overwritten".

**Fix**: `USE_COMPILE = False` mặc định. PE-G14 trên A100 BF16+channels_last+Flash-SDPA đã rất nhanh (~500-1500 img/s) không cần compile.

### 9.6 Bug 6 — Drive FUSE `[Errno 5]` Input/output error

**Trước fix**: `drive_output.mkdir()` raise OSError ngẫu nhiên do Drive FUSE state corruption / orphan folder trong trash.

**Fix**: `_robust_drive_mkdir()` với 3 strategies (pathlib mkdir → shell `mkdir -p` → recheck exists) × 3 retries. Setup tiếp tục nếu Drive fail (pipeline vẫn chạy local).

### 9.7 Bug 7 — `tar` archive symlinks rỗng

**Trước fix**: `mirror_raw_as_tar_split()` dùng `tar -cf` archive symlinks AS symlinks. Khi raw là symlink farm tới kagglehub cache (do disk full workaround), tar archive sẽ chứa broken links, restore session khác = data rỗng.

**Fix**: đổi `tar -cf` → `tar -chf` (`-h` dereference symlinks, archive actual content).

---

## 10. Storage & disk strategy

### 10.1 Colab disk reality

| Mount | Size | Persistence | Throughput | Note |
|-------|------|-------------|------------|------|
| `/` (overlay) | ~236 GB | Ephemeral | ~3-5 GB/s | Boot + system + cache. Limited free. |
| `/content/` | ~80 GB | Ephemeral | ~3-5 GB/s | Often pre-filled, ~6-50 GB free thực tế |
| `/var/colab/` | ~84 GB | Ephemeral | ~3-5 GB/s | Backup target nếu /content đầy |
| `/local-scratch/` | ~370 GB | Ephemeral | ~3-5 GB/s | A100 high-RAM only |
| `/content/drive/` | Drive quota | **Persistent** | **~10-50 MB/s** | FUSE bottleneck với many-small-files |

### 10.2 Data flow

```
Kaggle (CDN ~18MB/s, throttled)
    ↓ first session: 2h DL
kagglehub cache: /root/.cache/kagglehub/datasets/vnhtbo/...  (~136GB extracted)
    ↓ rsync (copy ~5-15 min) OR symlink (instant, 0 disk)
local SSD: <local_root>/raw/  (e.g., /var/colab/aic_local/raw)
    ↓ tar -chf | split  (~30-40 min, dereferences symlinks)
Drive: /content/drive/MyDrive/aic2026_data/raw_tar_parts/  (~22 parts × 4.5GB)
    ↓ session sau: cat parts | tar -xf  (~30-40 min)
local SSD restore (no Kaggle DL needed)
```

### 10.3 Symlink workaround (khi disk full)

Pipeline support trường hợp `<local_root>/raw/` là symlink farm tới kagglehub cache:

```
/var/colab/aic_local/raw/
├── annotation         → /root/.cache/kagglehub/.../annotation-testset/.../annotation/
├── name-masked_test-set → /root/.cache/.../name-masked_test-set/
└── train/
    ├── imgs_0         → /root/.cache/.../train-webp-part-01-05/.../train/imgs_0/
    ├── ...
    └── imgs_74        → /root/.cache/.../train-webp-part-06-10/.../train/imgs_74/
```

Disk usage: ~1MB (symlinks only), thay vì duplicate 136GB. `tar -chf` ở `mirror_raw_as_tar_split()` dereference → backup tar chứa actual files.

---

## 11. Run order & checklist

### 11.1 First-time setup (one-time, ~2-3h)

```bash
# 1. Upload Kaggle credentials lên Drive
#    /content/drive/MyDrive/aic2026_data/.kaggle/kaggle.json

# 2. Upload AIO paper repo lên Drive
#    /content/drive/MyDrive/aic2026_data/aio_repo/
#    (clone từ paper-released hoặc copy từ local workspace)

# 3. Upload pretrained checkpoint (optional, nếu paper-released)
#    /content/drive/MyDrive/aic2026_data/pretrained/uit_pretrained.pth

# 4. Upload aic_colab_utils.py + notebooks vào Colab session
```

### 11.2 Per-session bootstrap

```python
# Cell đầu mỗi notebook:
NOTEBOOK_DIR = Path('.').resolve()
sys.path.insert(0, str(NOTEBOOK_DIR))
from aic_colab_utils import setup_aic2026_environment

PATHS = setup_aic2026_environment(
    local_root='/var/colab/aic_local',          # ← disk lớn nhất available
    kaggle_token_path='/content/kaggle.json',
)
```

### 11.3 Execution order

```
✅ 00 manifest_qc        — chạy 1 lần (trên Kaggle hoặc Colab), upload kết quả lên Kaggle dataset
✅ 01a PE-G14 partial    — gallery + queries + val (~30 min)
⚠️  01b ViTPose          — OPTIONAL (~3h, chỉ nếu muốn anatomical crop ở 03)
✅ 02 UIT train          — 22 epoch (~8h, có checkpoint resume từ Drive)
✅ 03 LHP-PE-G14 LoRA    — 3 epoch (~2-3h)
✅ 04 UIT inference      — LHP-guided + ITM rerank (~45-60 min)
✅ 05 BLIP-2 inference   — ITC + ITM rerank top-1024 (~1.5-2h)
✅ 06 CLIP inference     — ViT-L/14@336 (~30 min)
✅ 07 PE-G14 scores      — build matrix từ 01a (~5 min)
✅ 08 k-reciprocal       — rerank per model (~40 min)
✅ 09 adaptive ensemble  — fusion + submission (~15 min)
```

**Total**: ~14-16h on A100-80GB (skip 01b). Split nhiều Colab session với Drive resume.

### 11.4 Final submission

`output/submission/answer.zip` chứa `answer.txt`:
- 1,978 dòng
- Mỗi dòng: 10 gallery IDs space-separated
- No duplicates per line
- Tất cả IDs phải nằm trong `gallery_manifest`

Verify trước khi submit:
```python
import zipfile
with zipfile.ZipFile('answer.zip') as zf:
    content = zf.read('answer.txt').decode()
lines = content.strip().split('\n')
assert len(lines) == 1978
for line in lines:
    ids = line.split()
    assert len(ids) == 10
    assert len(set(ids)) == 10   # no dups
print('Submission OK')
```

---

## 12. External References & Citations

Ngoài paper AIO (foundation), pipeline kết hợp thuật toán/model từ nhiều nguồn khác. Bảng dưới liệt kê đầy đủ + nơi xuất hiện trong code.

### 12.1 Pretrained models (non-AIO)

| Model | Paper / Source | Vai trò trong pipeline | Notebook |
|-------|---------------|------------------------|----------|
| **PE-Core-G14-448** | Bolya et al., *"Perception Encoder: The best visual embeddings are not at the output of the network"* (Meta FAIR, 2025). Repo: [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models) | Round 4 ensemble + LHP backbone (Upgrade 1, 3) | 01a, 03, 07 |
| **Swin Transformer (Swin-B)** | Liu et al., *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"* (ICCV 2021) | UIT image encoder (paper-faithful) | 02, 04 |
| **BERT-base-uncased** | Devlin et al., *"BERT: Pre-training of Deep Bidirectional Transformers"* (NAACL 2019) | UIT text encoder + cross-encoder (paper-faithful) | 02, 04 |
| **BLIP-2 ViT-g** | Li et al., *"BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"* (ICML 2023, Salesforce). HF: `Salesforce/blip2-itm-vit-g`, LAVIS lib | Round 2 ensemble (ITC + ITM rerank) | 05 |
| **CLIP ViT-L/14@336** | Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision"* (ICML 2021, OpenAI). HF: `openai/clip-vit-large-patch14-336` | Round 3 ensemble (cosine sim) | 06 |
| **ViTPose++** | Xu et al., *"ViTPose++: Vision Transformer for Generic Body Pose Estimation"* (TPAMI 2023). HF: `usyd-community/vitpose-plus-huge` | LHP anatomical crop (optional Upgrade) | 01b, 03 |
| **RT-DETR** | Zhao et al., *"DETRs Beat YOLOs on Real-time Object Detection"* (CVPR 2024). HF: `PekingU/rtdetr_r50vd_coco_o365` | Person detection cho ViTPose pipeline | 01b |
| **SimMIM** | Xie et al., *"SimMIM: A Simple Framework for Masked Image Modeling"* (CVPR 2022) | MIM loss trong UIT (paper-referenced inside CMP class) | 02 (via paper code) |

### 12.2 Algorithms (non-AIO)

| Algorithm | Paper | Vai trò | Notebook | Hyperparams |
|-----------|-------|---------|----------|-------------|
| **LoRA (Low-Rank Adaptation)** | Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"* (ICLR 2022). Lib: HuggingFace `peft` | Fine-tune PE-G14 mà không catastrophic-forget (Upgrade 3) | 03 | r=16, α=32, dropout=0.05, target=Q/K/V |
| **k-Reciprocal Re-ranking** | Zhong et al., *"Re-ranking Person Re-identification with k-reciprocal Encoding"* (CVPR 2017) | Refine ranking từ gallery-gallery k-NN graph trước khi fuse models (Extra Upgrade) | 08 | k1=20, k2=6, λ=0.3, TOPK_RR=200 |
| **InfoNCE / CLIP-style Contrastive Loss** | van den Oord et al., *"Representation Learning with Contrastive Predictive Coding"* (2018); Radford et al. (2021) | ITC loss trong UIT + LHP-LoRA training | 02, 03 | τ=0.07 |
| **AdamW optimizer** | Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (ICLR 2019) | Optimizer cho cả UIT + LHP-LoRA | 02, 03 | lr=1e-5 (UIT), 3e-4 (LoRA) |
| **Cosine LR schedule + linear warmup** | Loshchilov & Hutter, *"SGDR: Stochastic Gradient Descent with Warm Restarts"* (ICLR 2017) | LR schedule cho training | 02, 03 | warmup 5%, eta_min=1e-6 |
| **Flash Attention 2** | Dao, *"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"* (2023). PyTorch SDPA backend | Speed up attention trên A100 | All training/inference notebooks | enabled qua `torch.backends.cuda.enable_flash_sdp(True)` |
| **BF16 Mixed-Precision** | Kalamkar et al., *"A Study of BFLOAT16 for Deep Learning Training"* (2019). A100 native | Training + inference precision | All A100 notebooks | `torch.autocast(dtype=torch.bfloat16)` |
| **Stratified sampling** | Standard statistics | Val split deterministic theo `label_type × scene` | 00 | seed=20260514, frac=5% |

### 12.3 Adaptive Ensemble — sources of inspiration

Adaptive Weighting (Upgrade 4) là design **mới của pipeline này**, không copy paste từ paper nào, nhưng các component lấy ý tưởng từ:

| Component | Inspiration |
|-----------|-------------|
| **Per-query z-normalization** | Standard ML preprocessing — needed because BLIP-2 ITM logits scale ≠ cosine scale |
| **Top-1 margin confidence** | Active learning literature — model uncertainty quantification (Settles, *"Active Learning Literature Survey"*, 2009) |
| **Softmax entropy as confidence** | Information-theoretic uncertainty — Shannon entropy of probability distribution |
| **Inter-model agreement** | Ensemble diversity research — Krogh & Vedelsby, *"Neural Network Ensembles, Cross Validation, and Active Learning"* (NeurIPS 1995) |
| **Softmax with floor (η)** | Mixture-of-experts — KEEP all experts active, prevent dead model (similar to Switch Transformer's load balancing) |
| **Reciprocal Rank Fusion (RRF)** baseline | Cormack et al., *"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"* (SIGIR 2009). Considered as alternative to weighted-sum, but adaptive softmax fusion preferred |

### 12.4 Related person retrieval benchmarks (background reading)

| Paper | Năm | Insight áp dụng |
|-------|------|------------------|
| **CUHK-PEDES** (Li et al., CVPR 2017) | 2017 | Text-based person search benchmark establishment |
| **IRRA** (Jiang & Ye, CVPR 2023) | 2023 | SDM (Similarity Distribution Matching) loss — considered nhưng chưa apply |
| **CMP** (ICCV 2025, prior SOTA) | 2025 | 79.33% R@1 trên PAB — baseline để beat |
| **HFUT-LMC** (arXiv 2025) | 2025 | SCA (Similarity Coverage Analysis) re-ranking — considered nhưng dùng k-reciprocal thay |
| **AnomalyLMM** (arXiv 9/2025) | 2025 | Qwen2.5-VL training-free re-ranking — considered cho future upgrade A3 |
| **APTM** (ECCV 2024) | 2024 | 6-loss pretraining strategy — referenced trong BPAD-E roadmap |
| **MARS** (TOMM 2024) | 2024 | Attribute loss + visual reconstruction — considered |

### 12.5 Datasets / external resources

| Resource | Source | Vai trò |
|----------|--------|---------|
| **PAB (Pedestrian Anomaly Behavior)** | [github.com/Shuyu-XJTU/PAB-for-ECCV26-Workshop-Track4](https://github.com/Shuyu-XJTU/PAB-for-ECCV26-Workshop-Track4) | Main dataset, 1M train + 36k gallery + 1.9k queries |
| **AIC 2026 Track 4** | AI City Challenge 2026 workshop, ECCV 2026 | Competition platform + metric (mAP) |
| **BPAD-E v2.1 Strategic Roadmap** | Internal team document, [Document/BPAD_E_v2.1_Pipeline.docx](Document/BPAD_E_v2.1_Pipeline.docx) | Roadmap referenced cho future upgrades (Query Synthesis, MLLM rerank) |

### 12.6 Infrastructure dependencies

| Library | Version | Vai trò |
|---------|---------|---------|
| **PyTorch** | ≥2.3 (2.5 recommended) | DL framework |
| **HuggingFace Transformers** | 4.47.1 (paper-pinned), ≥4.40 (CLIP/BLIP-2) | Model loading |
| **HuggingFace `peft`** | ≥0.13.0 | LoRA implementation (Notebook 03) |
| **Salesforce LAVIS** | 1.0.2+ | BLIP-2 inference (Notebook 05) |
| **OpenCLIP / open_clip_torch** | — | (Was used for EVA-CLIP, hiện không cần) |
| **kagglehub** | ≥0.3.0 | Kaggle dataset download (`aic_colab_utils.py`) |
| **timm** | 0.6.13 (paper-pinned cho UIT) | Vision transformer utils |
| **torchscale** | 0.3.0 (paper-pinned cho UIT) | Microsoft DeepNet utils cho UIT |
| **kornia / torchvision** | — | Image transforms |
| **scipy** | — | k-reciprocal Spearman / cosine ops |
| **cupy** (optional) | — | GPU numpy cho k-reciprocal |
| **rsync** + **tar** + **split** (Unix) | — | Drive sync infrastructure (`mirror_raw_as_tar_split`) |
| **Google Drive FUSE** (`drivefs`) | — | Persistent storage trên Colab |

### 12.7 Tổng hợp — Algorithmic provenance map

```
┌──────────────────────────────────────────────────────────────────┐
│ AIC 2026 Track 4 Pipeline                                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ AIO Paper (WWW 2025) — Foundation                       │    │
│  │   ├─ LHP probabilistic sampling                         │    │
│  │   ├─ UIT 4-loss multi-task (ITC+ITM+MLM+MIM)           │    │
│  │   ├─ Algorithm 1: Feature Selection                     │    │
│  │   └─ Algorithm 2: Iterative Ensemble (3-round)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Our additions / modifications                            │    │
│  │   ├─ PE-Core-G14 (Meta 2025) ──→ replaces BEiT-3 + adds │    │
│  │   ├─ LoRA (Hu et al. ICLR 2022) ──→ for PE-G14 FT       │    │
│  │   ├─ k-Reciprocal (Zhong CVPR 2017) ──→ pre-fusion rerank│   │
│  │   ├─ Adaptive Weighting (our design) ──→ replaces fixed │    │
│  │   │     ├─ z-norm: standard ML                          │    │
│  │   │     ├─ entropy/margin: active learning lit          │    │
│  │   │     └─ agreement: ensemble diversity (Krogh 1995)   │    │
│  │   ├─ ViTPose++ (Xu TPAMI 2023) ──→ anatomical crop      │    │
│  │   ├─ RT-DETR (Zhao CVPR 2024) ──→ person detection     │    │
│  │   └─ mAP@10 metric ──→ match AIC submission style       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Backbone models (paper-faithful)                         │    │
│  │   ├─ Swin-B (Liu ICCV 2021)                             │    │
│  │   ├─ BERT-base (Devlin NAACL 2019)                      │    │
│  │   ├─ BLIP-2 (Li ICML 2023)                              │    │
│  │   ├─ OpenAI CLIP ViT-L/14 (Radford ICML 2021)           │    │
│  │   └─ SimMIM (Xie CVPR 2022) — for MIM loss              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Training infrastructure                                  │    │
│  │   ├─ AdamW (Loshchilov ICLR 2019)                       │    │
│  │   ├─ Cosine LR + warmup (Loshchilov ICLR 2017)          │    │
│  │   ├─ Flash Attention 2 (Dao 2023)                       │    │
│  │   ├─ BF16 mixed-precision (Kalamkar 2019)               │    │
│  │   └─ InfoNCE contrastive (van den Oord 2018)            │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

**Summary**: Pipeline foundation từ **1 paper chính (AIO WWW 2025)**, kết hợp **~15 paper khác** (model architectures + algorithms) + **5-6 component design mới của team** (Adaptive Weighting với 4-signal confidence, LoRA-based LHP, mAP@10 gate, Sim2Real val splits, tar-split Drive persistence, symlink workaround).

---

## Phụ lục — Key file references

| File | Vai trò |
|------|---------|
| `notebooks/aic_colab_utils.py` | Infrastructure shared across notebooks |
| `notebooks/00_manifest_qc.ipynb` | Build manifests + val splits |
| `notebooks/01a_pe_g14_features.ipynb` | PE-G14 encoding |
| `notebooks/01b_vitpose_features.ipynb` | ViTPose keypoints (optional) |
| `notebooks/02_uit_train.ipynb` | UIT training |
| `notebooks/03_lhp_peg14_train.ipynb` | LHP-PE-G14 LoRA training |
| `notebooks/04_uit_inference.ipynb` | UIT inference với Algorithm 1 |
| `notebooks/05_blip2_inference.ipynb` | BLIP-2 inference |
| `notebooks/06_clip_inference.ipynb` | CLIP inference |
| `notebooks/07_pe_g14_scores.ipynb` | PE-G14 score matrix |
| `notebooks/08_kreciprocal_rerank.ipynb` | k-reciprocal rerank |
| `notebooks/09_adaptive_ensemble_submit.ipynb` | Adaptive ensemble + submission |
| `Document/AIO_paper.pdf` | Paper foundation (WWW 2025) |
| `Document/pipeline_architecture.tex` | TikZ diagram source |
| `Hybrid-Unified-and-Iterative-.../uit/cmp/` | Paper's UIT training code (used by 02, 04) |
| `Hybrid-Unified-and-Iterative-.../lhp_2/` | Paper's LHP code (reference, not used directly) |

---

**Last updated**: 2026-05-15
**Author**: Pipeline implementation team
**Contact**: plehuyhoang@gmail.com
