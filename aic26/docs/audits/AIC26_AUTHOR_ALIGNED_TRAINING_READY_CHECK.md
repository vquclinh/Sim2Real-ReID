# AIC26 Author-Aligned Training Ready Check

> **Status:** Read-only audit. No training was run, no GPU was used, no checkpoint was downloaded, no author code was modified.
> **Primary source:** `aic26/docs/references/AIO_paper.pdf` — *"Hybrid, Unified and Iterative: A Novel Framework for Text-based Person Anomaly Retrieval"*, Nguyen, Tran, Phan-Nguyen, Dinh — WWW Companion '25, arXiv:2511.22470v1 (5 pages).
> **Companion audit:** `aic26/docs/audits/AIC26_AUTHOR_TRAINING_REPRODUCTION_PLAN.md`.
> **Labels used throughout:** `Paper says:` / `Code says:` / `Repo status:` / `Conclusion:`.

---

## Executive Summary

The paper's method is fully present in the repo as code, and most paper hyperparameters are reachable **without modifying author code** — via CLI flags that already exist (`--epo`, `--lr`, `--bs` in `uit/cmp/Search.py`; `--epochs`, `--batch_size`, `--lr` in `lhp_2/beit3/run_beit3_finetuning.py`). Three findings change the readiness picture relative to the previous audit:

1. **The BEiT-3 training entrypoint is `lhp_2/beit3/run_beit3_finetuning.py` (task `'356'`), not `run_beit3_retrieval.py`** (which does not exist). The LHP module is implemented exactly as the paper describes, in `lhp_2/beit3/datasets.py:83`: `torch.normal(mean=0.5, std=0.166667)` gating between a `RandomResizedCropAndInterpolation(384)` local transform and a full-image `Resize((384,384))` global transform.

2. **Bug found in `uit/cmp/inference.py:74`:** the call to `evaluation_itm()` passes `args.blip2_weight` into the `beit3_weight` parameter position. The `--beit3_weight` argument (default 0.925, the paper's best value) is **defined but never used**. With default flags the effective BEiT-3 weight is 0.9, not the paper's 0.925. Patch-only fix.

3. **Two config-vs-paper mismatches in UIT** that the saved run artifact does *not* resolve: `cmp.yaml` says 30 epochs / LR 1e-4 / `step` scheduler, while the paper says 22 epochs / LR 1e-5 / cosine annealing. Epochs and LR can be overridden by CLI (`--epo 22 --lr 1e-5`); the scheduler type **cannot** — it requires an `aic26/`-side copy of the config with `sched: cosine`. Note: `Search.py` dumps `config.yaml` to the output dir *before* applying CLI overrides, so the saved `uit/cmp/output/cmp/config.yaml` (30 epochs, 1e-4, step) is **not** evidence of what the author actually ran.

Blocking gaps that no patch can close: the author-trained checkpoints (`checkpoint/lhp/lhp_beit3.pth`, UIT trained weights) are absent and must be retrained from public inits; the identity of `uit/cmp/checkpoint/pretrained.pth` is unknown; the 0.1M training subset and the ensemble-tuning ground-truth set are not specified anywhere.

**Verdict:** training is *nearly* ready. The single blocker before any GPU work is verifying the PAB data layout matches what both dataloaders hard-code. That is the recommended first coding task (see final sections).

---

## Paper Facts Extracted From AIO_paper.pdf

### A. Dataset and evaluation

| Fact | Value | Paper location |
|---|---|---|
| PAB composition | 1,013,605 synthesized + 1,978 real-world image-text pairs | §3.2, p.3 |
| Train/test protocol | Test data is real-world; a diffusion model generates the training set | §3.2, p.3–4 |
| Metrics | Recall rates R@1, R@5, R@10 ("search successful if image perfectly matches the text query in top k-ranked images") | §3.2, p.4 |
| Training-size settings | 0.1M and 1M training-image settings, both reported | Table 1, p.4 |
| Validation/tuning protocol | Ensemble weights tuned by sweeping W with a scoring function against ground truth `gt` (Algorithm 2); **which split provides `gt` is not specified** | §2.3 + Algorithm 2, p.3 |

### B. BEiT-3 + LHP

| Fact | Value | Paper location |
|---|---|---|
| Backbone | BEiT-3 (baseline model, ref [12]); LHP module integrated into it | §3.1, p.3 |
| Image size | 384×384 | §3.1, p.3 |
| Epochs | 3 | §3.1, p.3 |
| Batch size | 184 | §3.1, p.3 |
| Optimizer | AdamW | §3.1, p.3 |
| Initial LR | 1e-5 | §3.1, p.3 |
| LR scheduler | Cosine annealing | §3.1, p.3 |
| LHP mechanism | Sample from normal distribution, mean 0.5, "variance of 1÷6"; if sample > 0.5 apply **local** transform, else **global** transform | §2.1, p.1; Figure 1(a), p.2 |
| Local transform | "a region of interest is cropped from the image to focus on fine-grained details" — crop parameters not specified in paper | §2.1, p.2 |
| Global transform | "retains the entire image" | §2.1, p.2 |
| Training loss | Contrastive loss, L_cl = −½ E[log S_I2T + log S_T2I], cosine similarity with temperature τ (Eq. 1–3) | §2.1, p.2 |
| Output role | Primary retrieval score matrix; also supplies the similarity matrix for Feature Selection (Algorithm 1) and initializes ensemble S | §2.2.2 + §2.3, p.3 |
| Reported metrics | LHP (0.1M): 85.39 / 99.49 / 99.95 (Table 2); Baseline+LHP (1M): R@1 87.11 (Table 3) | Tables 2–3, p.4 |

### C. UIT (CMP-style)

| Fact | Value | Paper location |
|---|---|---|
| Architecture | Image Encoder + Text Encoder + Cross Encoder + Decoder; "drawing inspiration from CMP [15]" | §2.2.1, p.2 |
| Image encoder | Swin-B | §3.1, p.3 |
| Text encoder | BERT-based | §3.1, p.3 |
| Image size | 224×224 | §3.1, p.3 |
| Epochs | 22 | §3.1, p.3 |
| Batch size | 84 | §3.1, p.3 |
| Optimizer | AdamW | §3.1, p.3 |
| Initial LR | 1e-5 | §3.1, p.3 |
| LR scheduler | Cosine annealing | §3.1, p.3 |
| Losses | L = L_itc + L_itm + L_mlm + α·L_mim (Eq. 8); MIM is L1 reconstruction (Eq. 4–5), ITM is binary cross-entropy via ITM head MLP (Eq. 6), MLM cross-entropy (Eq. 7) | §2.2.1, p.2–3 |
| MIM α | **0.1356** | Eq. 8, p.3 |
| Feature selection | Top-k image features per text embedding selected using the **LHP similarity matrix** (not UIT's own); selected features fed to cross-encoder for ITM scoring; applied at inference only ("Inference only" in Figure 1(b)). **k value not specified in paper** | §2.2.2 + Algorithm 1, p.3; Figure 1(b), p.2 |
| Output role | ITM reranking scores; Iteration 1 of the ensemble | §2.3 + Table 4 |
| Reported metrics | Baseline+LHP+UIT(FS) (1M): R@1 88.37 (Table 3) | Table 3, p.4 |

### D. BLIP-2 and CLIP

| Fact | Value | Paper location |
|---|---|---|
| Role | Inference/ensemble only — no fine-tuning mentioned anywhere in the paper | §3.1, p.3 |
| Variants | Not specified in paper beyond citations: BLIP-2 [4], CLIP [10]. `Code says:` BLIP-2 = LAVIS `blip2_feature_extractor` / `pretrain_vitL` (`blip/blip2.py:13`); CLIP = `ViT-L/14@336px` (`clip_infer.py:112`) | §3.1; code |
| Ensemble position | BLIP-2 = Iteration 2, CLIP = Iteration 3 | Table 4, p.4 |

### E. Iterative ensemble

| Fact | Value | Paper location |
|---|---|---|
| Algorithm | S ← 0; for each model θ: pred ← topk(w·S + (1−w)·t_θ(I,Q)); w ← argmax over W of f_s(pred, gt); S ← w·S + (1−w)·t_θ(I,Q) | Algorithm 2, p.3 |
| Model order | Base = BEiT-3+LHP (first model, S after iteration 0); Iter.1 = UIT, Iter.2 = BLIP-2, Iter.3 = CLIP | §2.3 + Table 4 column headers, p.3–4 |
| Best weights | Iter.1 = 0.925, Iter.2 = 0.9, Iter.3 = 0.8725 **or** 0.9 (two rows tie) → R@1 89.23, R@5 99.70, R@10 99.85 | Table 4, rows 14 and 16, p.4 |
| Tuning protocol | Hyperparameter sweep over W; "values close to 1 improve significantly"; the gt/scoring set used for the sweep is **not specified** | §2.3, p.3 |
| Best overall result | R@1 89.23 (1M images, LHP+UIT(FS)+IE) | Tables 3–4, p.4 |

---

## Paper-to-Code Mapping

| Paper component | Repo files | Entrypoint | Config | Input data | Required checkpoint/init | Output artifact |
|---|---|---|---|---|---|---|
| BEiT-3 + LHP training | `lhp_2/beit3/{run_beit3_finetuning.py, datasets.py, engine_for_finetuning.py, modeling_finetune.py, utils.py}` | `run_beit3_finetuning.py --task 356` (model hard-coded `beit3_large_patch16_384_retrieval` at line 259; task 356 → `BaseDataset` + `RetrievalHandler`) | CLI args only (no yaml) | `<data_path>/annotation/train/pair_{0..74}.json` (75 files hard-coded, `datasets.py:35-40`) + images under `<data_path>` | Public BEiT-3 init via `--finetune` (absent; see Missing Items) + `beit3.spm` (**present**: `lhp_2/beit3/beit3.spm`) | `checkpoint-*.pth` in `--output_dir` |
| BEiT-3 + LHP inference | `lhp_2/beit3/inference.py` (standalone, correct imports) | `python lhp_2/beit3/inference.py` | CLI args | gallery folder + `query.json` | `./checkpoint/lhp/lhp_beit3.pth` (absent) + `./checkpoint/lhp/beit3.spm` | score matrix `.pt` + top-10 `.txt` |
| UIT/CMP training | `uit/cmp/{Search.py, train.py, models/model_search.py, dataset/search_dataset.py, scheduler.py, optim.py}` | `Search.py --config configs/cmp.yaml --task 356 --output_dir ... --checkpoint ...` | `uit/cmp/configs/cmp.yaml` (or `baseline.yaml` for 8-file subset) | 75 `pair_*.json` + `annotation/test/pair.json` (caption must be a list) | `--checkpoint` → `uit/cmp/checkpoint/pretrained.pth` (absent, identity unknown) + `checkpoint/bert-base-uncased` (absent, public) | `output/356356/checkpoint_{epoch}.pth` (path hard-coded, `Search.py:135`) |
| UIT/CMP inference / reranking | `uit/cmp/{inference.py, eval.py, dataset/search_dataset.py::search_inference_dataset}` | `uit/cmp/inference.py` | `uit/cmp/configs/infer.yaml` (`k_test: 10`) | hidden-test `query.json` + gallery folder (via unsorted `os.listdir()`) | trained UIT checkpoint + bert-base-uncased + 3 score matrices via `--beit3_score/--blip2_score/--clip_score` | `answer.txt` (top-10 per query) |
| CLIP score generation | `clip_infer.py` | `python clip_infer.py` | CLI args | gallery folder + `query.json` | CLIP `ViT-L/14@336px` (auto-download) | `sims_score/score_clip_reproduce.pt` + `predictions/score_clip.txt` |
| BLIP-2 score generation | `blip2_infer.py`, `blip/blip2.py` | `python blip2_infer.py` (needs CWD = repo root for `sys.path.append('./blip')`) | CLI args | gallery folder + `query.json` | LAVIS `blip2_feature_extractor`/`pretrain_vitL` (auto-download) | `sims_score/score_blip2_reproduce.pt` |
| Iterative ensemble / score fusion | `uit/cmp/eval.py::evaluation_itm` (fusion formula at the end of the function) | via `uit/cmp/inference.py` | weights via CLI (`--beit3_weight 0.925 --blip2_weight 0.9 --clip_weight 0.9` — defaults match paper Table 4 row 16, **but see bug below**) | the three `.pt` score matrices | trained UIT checkpoint (for ITM rerank step) | fused score matrix → `answer.txt` |

`Code says:` the fusion in `eval.py` is exactly the paper's Algorithm 2 unrolled, with min-max normalization of every matrix first:
`S = (((1−w_b)·ITM + w_b·BEiT3) · w_2 + (1−w_2)·BLIP2) · w_3 + (1−w_3)·CLIP`
which is S₀=BEiT3 → S₁=w_b·S₀+(1−w_b)·UIT-ITM → S₂=w₂·S₁+(1−w₂)·BLIP2 → S₃=w₃·S₂+(1−w₃)·CLIP. `Conclusion:` code implements the paper formula; model order matches Table 4.

`Repo status:` `lhp/` exists but is **empty**. `beit3_infer.py` (repo root) imports from the non-existent `lhp.beit3` and would fail; the working BEiT-3 inference is `lhp_2/beit3/inference.py`.

---

## Paper Settings vs Current Code

| Requirement | Paper value | Code/config value | Match? | Action |
|---|---|---|---:|---|
| BEiT-3 image size | 384×384 (§3.1) | Model `beit3_large_patch16_384_retrieval` (run_beit3_finetuning.py:259); transforms `Resize((384,384))` / `RandomResizedCropAndInterpolation(384, scale=(0.5,1.0))` (datasets.py:58,65) | YES | None |
| BEiT-3 LHP gating | normal(mean 0.5, "variance 1÷6"), >0.5 → local (§2.1) | `torch.normal(mean=0.5, std=0.166667) > 0.5` → crop else nocrop (datasets.py:83) | YES* | None. *Code uses **std**=1/6 where paper writes "variance of 1÷6"; code is the ground truth implementation |
| BEiT-3 epochs | 3 (§3.1) | CLI default `--epochs 20` | NO (default) | Pass `--epochs 3` — patch-free |
| BEiT-3 batch size | 184 (§3.1) | CLI default `--batch_size 64` | NO (default) | Pass `--batch_size 184` (or smaller + `--update_freq` if VRAM-limited) |
| BEiT-3 LR | 1e-5 (§3.1) | CLI default `--lr 5e-4` | NO (default) | Pass `--lr 1e-5` |
| BEiT-3 scheduler | cosine annealing (§3.1) | `utils.cosine_scheduler(...)` (run_beit3_finetuning.py:338) | YES | None |
| BEiT-3 optimizer | AdamW (§3.1) | CLI default `--opt adamw` | YES | None |
| BEiT-3 checkpoint path (trained) | n/a | `lhp_2/beit3/inference.py` default `./checkpoint/lhp/lhp_beit3.pth` | ABSENT | Train ourselves (Category C) |
| BEiT-3 tokenizer path | n/a | `lhp_2/beit3/beit3.spm` | PRESENT | None |
| BEiT-3 init (`--finetune`) | Not specified in paper | Official guide (`get_started_for_retrieval.md:111`) inits large retrieval from `beit3_large_itc_patch16_224.pth`; `beit3_infer.py:53` default references public `beit3_large_patch16_384_coco_retrieval.pth` | UNKNOWN | Open question; both candidates are public downloads |
| UIT image size | 224×224 (§3.1) | `h: 224, w: 224` (cmp.yaml) | YES | None |
| UIT epochs | 22 (§3.1) | `epochs: 30` (cmp.yaml) | NO | Pass `--epo 22` (Search.py:40 override) — patch-free |
| UIT batch size | 84 (§3.1) | `batch_size_train: 84` (cmp.yaml) | YES | None |
| UIT LR | 1e-5 (§3.1) | `lr: 1e-4` (cmp.yaml optimizer + schedular) | NO | Pass `--lr 1e-5` (Search.py:42 overrides both) — **but see scheduler caveat below** |
| UIT scheduler | cosine annealing (§3.1) | `sched: step` (cmp.yaml); `scheduler.py` *does* implement `'cosine'` (line 45) | NO | Cannot override via CLI. Make an `aic26/`-side copy of cmp.yaml with `sched: cosine` and pass it via `--config`. Caveat: the cosine branch hard-codes `min_lr=1e-5, max_lr=1e-4` inside its `lr_lambda` — the actual LR curve must be reviewed against the optimizer LR before the full run |
| UIT optimizer | AdamW (§3.1) | `opt: adamW, weight_decay: 0.01, lr_mult: 2` (cmp.yaml) | YES | None (weight decay 0.01 not stated in paper — keep code value) |
| UIT losses | ITC + ITM + MLM + α·MIM (Eq. 8) | `loss = loss_itc + loss_itm + loss_mlm + loss_mim * 0.1356` (train.py:68); plus EDA-augmented terms `loss_itc += 0.8·loss_itc_eda`, `loss_itm += 0.8·loss_itm_eda` (model_search.py:28-29) **not mentioned in the paper** | PARTIAL | Keep code as-is (code is the author's actual implementation); note EDA terms as a paper-omission |
| UIT MIM α | 0.1356 (Eq. 8) | 0.1356 hard-coded (train.py:68) | YES | None |
| UIT k_test / FS top-k | "top-k", value not specified (§2.2.2) | `k_test: 128` (cmp.yaml, local eval), `k_test: 10` (infer.yaml, hidden test) | N/A | Not specified in paper/code beyond these values. Use code values as-is |
| UIT checkpoint/pretrained path | n/a | `Search.py --checkpoint` → expected `uit/cmp/checkpoint/pretrained.pth`; `load_params_vision: True, load_params_text: False` (cmp.yaml) | ABSENT | Unknown identity — inspect when obtainable (Category D) |
| BERT path | "BERT-based encoder" (§3.1) | `text_encoder: 'checkpoint/bert-base-uncased'` (cmp.yaml) | ABSENT | Public download (Category B) |
| CLIP model variant | Not specified in paper | `clip.load("ViT-L/14@336px")` (clip_infer.py:112) | CODE-ONLY | Use code value; auto-downloads |
| BLIP-2 model source | Not specified in paper | LAVIS `load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL")` (blip/blip2.py:13) | CODE-ONLY | Use code value; auto-downloads |
| Ensemble weights/order | BEiT3 base → UIT 0.925 → BLIP-2 0.9 → CLIP 0.9 (Table 4 row 16) | Defaults `--beit3_weight 0.925 --blip2_weight 0.9 --clip_weight 0.9` (inference.py:102-106) — **BUT** inference.py:74 passes `args.blip2_weight` into the `beit3_weight` position; `--beit3_weight` is never used | **BUG** | Patch-only: an `aic26/`-side wrapper must call `evaluation_itm` with the correct argument order (effective default today: 0.9/0.9/0.9, not 0.925/0.9/0.9) |
| Output answer format | top-k ranked images, R@K eval (§3.2) | top-10 gallery IDs per query line → `answer.txt` (inference.py / *_infer.py) | YES | None |

---

## Missing Items Classification

Categories: **A** Patch-only · **B** Public replacement available · **C** Train ourselves from public init · **D** Unknown until inspected · **E** Cannot match exactly without author artifact.

| Item | Needed by | Category | Why | Best action |
|---|---|---|---|---|
| `beit3_infer.py` imports `lhp.beit3` but `lhp/` is empty | BEiT-3 score generation (root script) | A | Wrong import path; working equivalent exists at `lhp_2/beit3/inference.py` | Use `lhp_2/beit3/inference.py`; or an `aic26/` wrapper sets the path. Never edit the original |
| `inference.py:74` weight-argument bug (`blip2_weight` passed as `beit3_weight`) | Ensemble reranking | A | `--beit3_weight 0.925` silently ignored; paper weights unreachable via CLI | `aic26/`-side inference wrapper calling `evaluation_itm` with correct args |
| Unsorted `os.listdir()` gallery (clip_infer.py, blip2_infer.py, `search_inference_dataset`) | All score generation + fusion alignment | A | Score-matrix columns are filesystem-order; matrices from different runs/machines silently misalign | `aic26/` wrappers with `sorted()` + persist `gallery_ids.json` per run |
| Hard-coded `output/356356/checkpoint_{epoch}.pth` (Search.py:135) | UIT training | A | Output dir not configurable; crashes if dir missing | Pre-create `output/356356/` in launch script |
| Hard-coded researcher path in beit3_infer.py:164 | BEiT-3 score generation | A | Developer leftover; only used if `--annotation` omitted | Always pass `--annotation` explicitly |
| UIT scheduler `step` vs paper `cosine` | UIT paper-aligned training | A | `scheduler.py` implements `cosine`; only the yaml selects `step`; no CLI override exists | Paper-aligned config copy under `aic26/` with `sched: cosine`, passed via `--config` |
| UIT epochs 30 / LR 1e-4 vs paper 22 / 1e-5 | UIT paper-aligned training | A | `--epo` / `--lr` CLI overrides already exist (Search.py:38-42) | Pass `--epo 22 --lr 1e-5`; no file change needed |
| `checkpoint/bert-base-uncased` | UIT training + inference | B | Standard HuggingFace asset, only the local copy is missing | Download `bert-base-uncased` into expected path (small, permitted when staging starts) |
| CLIP `ViT-L/14@336px` weights | CLIP score generation | B | Auto-downloads via `clip.load()` | Nothing to prepare |
| BLIP-2 `pretrain_vitL` weights | BLIP-2 score generation | B | Auto-downloads via LAVIS | Nothing to prepare |
| BEiT-3 public init checkpoint | BEiT-3 + LHP fine-tuning (`--finetune`) | B (+open question) | Both candidates public: `beit3_large_itc_patch16_224.pth` (official retrieval-finetune recipe) or `beit3_large_patch16_384_coco_retrieval.pth` (referenced by beit3_infer.py default). Which the author used is unstated | Default to `beit3_large_itc_patch16_224.pth` per official guide; document the choice; treat as Open Question #1 |
| `checkpoint/lhp/lhp_beit3.pth` (author's fine-tuned BEiT-3+LHP) | BEiT-3 score generation; exact-reproduction | C | Author-trained artifact, not public in repo | Train ourselves: 3 epochs, batch 184, 384×384, AdamW, cosine, LR 1e-5 (all from §3.1) |
| UIT/CMP trained checkpoint | ITM reranking; exact reproduction | C | Author-trained artifact, not public in repo | Train ourselves: 22 epochs, batch 84, 224×224, AdamW, cosine, LR 1e-5, α=0.1356 |
| `uit/cmp/checkpoint/pretrained.pth` (vision init for `load_params_vision: True`) | UIT training initialization | D | Referenced by config but absent; could be public SwinB ImageNet weights, CMP-released weights, or author-private. Key structure unknown until a candidate file is inspected | Obtain a candidate (e.g. from the CMP paper's release) and inspect `state_dict` keys against `models/cmp.py` expectations before training |
| Exact 0.1M training subset | Reproducing Table 1 0.1M rows | E | No split list in repo; paper does not say how the 0.1M subset was drawn from the 1M | If 0.1M runs are wanted, define our own deterministic subset (e.g. first N pairs) and document it. `Not specified in paper/code. Must choose our own value and document it.` |
| Ensemble-tuning ground truth set | Reproducing Table 4 weight sweep | E | Algorithm 2 requires `gt`; the paper never states which split supplied it | Tune only on a frozen local validation split (`annotation/test/pair.json`-based); never the official hidden test. Exact author protocol unrecoverable |
| Author random seeds / run logs | Bit-exact reproduction | E | `Search.py` defaults `--seed 42`, `run_beit3_finetuning.py` defaults `--seed 0`; the saved `output/cmp/config.yaml` was dumped **before** CLI overrides were applied (Search.py `__main__` order), so it proves nothing about actual run hyperparameters. tfevents logs exist (`lhp_2/beit3/lhp_reproduce/log/`) but contain curves, not configs | Use code-default seeds; accept statistical rather than bit-exact reproduction |

---

## What We Can Do Now

**1. Can do immediately with code patches only (no downloads, no GPU):**
- Write the `aic26/`-side launch/wrapper layer: corrected `evaluation_itm` argument order, sorted gallery listing + `gallery_ids.json` / `query_ids.json` persistence, pre-creation of `output/356356/`, paper-aligned UIT config copy (`sched: cosine`).
- Data format verification (see item 3 — patch-free, read-only).

**2. Can do after public downloads:**
- `bert-base-uncased` → unblocks UIT tokenizer/text-encoder load (training still needs `pretrained.pth` resolved).
- BEiT-3 public init (`beit3_large_itc_patch16_224.pth`, ~2 GB) → unblocks BEiT-3+LHP fine-tuning entirely (tokenizer `beit3.spm` already present).
- CLIP and BLIP-2 auto-download on first run → unblocks Stage 6 score generation.

**3. Can do after data format verification:**
- BEiT-3+LHP smoke training (Stage 1) — needs the 75 `pair_*.json` files verified against `datasets.py` expectations (JSONL, `image`/`caption` fields, image paths resolvable from `data_path`).
- UIT smoke training (Stage 4) — needs `pair.json` verified (`caption` as list) and image roots checked against `../../data/PAB/` from `uit/cmp/`.

**4. Can do only by training ourselves:**
- BEiT-3+LHP checkpoint (replaces absent `checkpoint/lhp/lhp_beit3.pth`).
- UIT/CMP checkpoint (replaces absent trained weights).
- Both follow §3.1 settings; results will approximate, not equal, Tables 1–4 (different init lineage possible, seeds unknown, 0.1M subset unknown).

**5. Cannot reproduce exactly without author artifact:**
- Bit-exact author checkpoints and their score matrices (the committed `sims_score/*.pt` are usable as frozen references, but their gallery column order is unrecorded).
- Table 1's 0.1M rows with the author's exact subset.
- Table 4's exact weight sweep (tuning gt set unspecified).

---

## Author-Aligned Training Plan

### Stage 0 — Data and config verification (no GPU)

- **Input:** PAB data tree at the path the configs expect; `uit/cmp/configs/*.yaml`; `lhp_2/beit3/datasets.py` expectations.
- **Checks:** all 75 `annotation/train/pair_{0..74}.json` exist and are valid JSONL with `image`/`caption` keys; `annotation/test/pair.json` exists with `caption` as a **list** and `image_id` present; `annotation/test/attr.json` valid; every `image` path in a sample of records resolves under the image root (`../../data/PAB/` relative to `uit/cmp/`, `data_path` arg for BEiT-3); report counts per file.
- **Output:** verification report (pass/fail per check) under `aic26/docs/audits/`.
- **Command/script to create later:** `aic26/tools/verify_pab_author_data.py` (read-only; see First Coding Task).
- **Blocker:** PAB dataset must be on disk (not in repo).
- **Success criterion:** every check passes; any failure is itemized with file and reason before any GPU minute is spent.

### Stage 1 — BEiT-3 + LHP tiny smoke training

- **Input:** verified data; public BEiT-3 init; `beit3.spm`.
- **Action:** run `run_beit3_finetuning.py --task 356` on a tiny subset (e.g. temporarily limit `num_files` via an `aic26/`-side dataset shim or a reduced copy of pair files — never edit `datasets.py`), 1 epoch, small batch. Verify dataloader yields LHP-transformed batches, forward/backward completes, checkpoint writes to `--output_dir`.
- **Output:** one tiny checkpoint + training log.
- **Command (to finalize later):** `python run_beit3_finetuning.py --task 356 --input_size 384 --batch_size 8 --epochs 1 --lr 1e-5 --sentencepiece_model beit3.spm --finetune <public_init> --data_path <PAB_root> --output_dir <aic26 run dir> --save_ckpt`
- **Blocker:** public init download; GPU availability.
- **Success criterion:** loss decreases over steps; checkpoint file loads back without key errors. No metric target.

### Stage 2 — BEiT-3 + LHP full fine-tuning (paper-aligned)

- **Input:** Stage 1 pass.
- **Settings (Paper says, §3.1):** image 384×384 (model-fixed), **epochs 3**, **batch 184** if GPU allows (else smaller batch + `--update_freq` to keep effective batch 184 — document the deviation), **AdamW**, **cosine annealing** (built-in), **LR 1e-5**, init from public BEiT-3 (`--finetune`) since author checkpoint is absent.
- **Output:** our `lhp_beit3` replacement checkpoint under an `aic26/`-tracked run dir.
- **Blocker:** VRAM for batch 184 at 384² on a 746M model (likely multi-GPU or heavy accumulation).
- **Success criterion:** training completes 3 epochs; checkpoint usable by `lhp_2/beit3/inference.py`; local eval on `attr.json` in the vicinity of paper Table 2 (R@1 ≈ 85, exactness not expected).

### Stage 3 — BEiT-3 score generation

- **Input:** Stage 2 checkpoint; gallery + query annotation.
- **Action:** run `lhp_2/beit3/inference.py` via an `aic26/` wrapper that enforces **deterministic (sorted) gallery ordering** and saves `query_ids.json` + `gallery_ids.json` alongside the matrix.
- **Output:** `score_beit3_*.pt` + the two ID-mapping JSONs.
- **Blocker:** Stage 2.
- **Success criterion:** matrix shape = (num_queries × num_gallery); IDs files row/column-consistent; re-running produces identical output.

### Stage 4 — UIT/CMP tiny smoke training

- **Input:** verified data; `bert-base-uncased` downloaded; `pretrained.pth` question resolved (Open Question #2) — or smoke-run with `load_pretrained: False` in an `aic26/` config copy purely to validate mechanics.
- **Action:** `Search.py --config <aic26 copy of baseline.yaml> --task 356 --output_dir <run dir> --epo 1 --bs 8`; pre-create `output/356356/`. Verify all four losses (ITC/ITM/MLM/MIM) log finite values and checkpoint writes.
- **Output:** tiny checkpoint + log with the four loss curves.
- **Blocker:** `pretrained.pth` identity (for a *meaningful* smoke; mechanics can be validated without it).
- **Success criterion:** `loss_itc/loss_itm/loss_mlm/loss_mim` all present and finite; `output/356356/checkpoint_1.pth` written.

### Stage 5 — UIT/CMP full training (paper-aligned)

- **Input:** Stage 4 pass; vision init resolved.
- **Settings (Paper says, §3.1 + Eq. 8):** image 224×224 (config), **epochs 22** (`--epo 22`), **batch 84** if GPU allows (config default), **AdamW** (config), **cosine annealing** via `aic26/` config copy with `sched: cosine` — after reviewing the hard-coded `min_lr=1e-5/max_lr=1e-4` inside `scheduler.py`'s cosine branch, **LR 1e-5** (`--lr 1e-5`), **α = 0.1356** (already hard-coded in `train.py:68` — code supports it natively).
- **Output:** `output/356356/checkpoint_22.pth` → copied into an `aic26/`-tracked run dir.
- **Blocker:** vision init; ~30 GPU-hours-class job (batch 84, 1M pairs, 22 epochs).
- **Success criterion:** training completes; local ITC eval on `pair.json` in the vicinity of CMP-class numbers (Table 1 context); checkpoint loads in `uit/cmp/inference.py`.

### Stage 6 — CLIP and BLIP-2 score generation

- **Input:** gallery + query annotation.
- **Action:** `aic26/` wrappers around `clip_infer.py` logic (`ViT-L/14@336px`) and `blip2_infer.py` logic (`blip2_feature_extractor`/`pretrain_vitL`) with sorted gallery + `gallery_ids.json`/`query_ids.json` saved. Variants exactly as in code (paper does not specify).
- **Output:** `score_clip_*.pt`, `score_blip2_*.pt` + ID mappings identical to Stage 3's.
- **Blocker:** auto-downloads only.
- **Success criterion:** all three matrices share identical shapes and identical `gallery_ids.json`.

### Stage 7 — Iterative ensemble

- **Input:** Stages 3, 5, 6 outputs.
- **Action:** run the reranking via an `aic26/` wrapper that fixes the `inference.py:74` argument bug, with **paper weights: BEiT-3 base, UIT w=0.925, BLIP-2 w=0.9, CLIP w=0.9** (Table 4 row 16; row 14's CLIP 0.8725 ties). If weight tuning is wanted, sweep W **only on a frozen local validation split** (built from `pair.json`/`attr.json` ground truth) — **never on the official hidden test**.
- **Output:** fused score matrix + tuning log (if tuned).
- **Blocker:** all prior stages.
- **Success criterion:** local R@1 of the fused matrix ≥ best single model; weights and tuning split documented.

### Stage 8 — Official answer generation

- **Input:** Stage 7 fused matrix + `gallery_ids.json`.
- **Action:** generate `answer.txt` (top-10 gallery IDs per query line, IDs = filename minus extension, matching `inference.py`'s `g_pids` convention); validate format against `aic26/docs/submissions/run_002/answer.txt`; zip.
- **Output:** `answer.txt`, `answer.zip` in an `aic26/` submission run dir.
- **Blocker:** Stage 7; a leaderboard attempt is spent on submission — submit only after local metrics beat the current PE-G14 baseline (mAP 0.8829 / R@1 0.7932 on `attr.json`).
- **Success criterion:** format validation passes; submission decision made deliberately.

---

## First Coding Task Recommendation

**Recommended task: `aic26/tools/verify_pab_author_data.py` — a read-only data format verifier for both training pipelines (Stage 0).**

One focused script, no GPU, no downloads, no author-code modification. It must check exactly what the two dataloaders hard-code:

1. All 75 `annotation/train/pair_{0..74}.json` files exist, parse as JSONL, and every sampled record has `image` (str) and `caption` (str) — required by `lhp_2/beit3/datasets.py:39-48` and `uit/cmp/dataset/search_dataset.py::search_train_dataset` (which additionally needs `image_id`).
2. `annotation/test/pair.json` parses, and `caption` is a **list** per record with `image_id` present — required by `search_test_dataset` (UIT eval crashes on single-string captions).
3. `annotation/test/attr.json` parses (local-eval reference split).
4. A sample of `image` paths from each file resolves to an existing file under the image root used by each pipeline (`../../data/PAB/` relative to `uit/cmp/`; `--data_path` for BEiT-3).
5. Report per-file pair counts (sanity vs. the paper's 1,013,605 total, §3.2).

**Why this over the alternatives:** the BEiT-3 smoke-test wrapper (Stage 1) and the public-init loader verification both *depend* on the data layout being correct and on a checkpoint download, while this task blocks nothing, requires nothing, and every later stage consumes its output. It is the smallest task that de-risks the first GPU minute.

Not recommended now: local_002/UCC work (not required before training) and PE-G14 fusion (not part of the author method).

---

## Open Questions

1. **Which public BEiT-3 init did the author fine-tune from?** Paper says: nothing. Code says: official guide inits large retrieval models from `beit3_large_itc_patch16_224.pth` (`get_started_for_retrieval.md:111`), while `beit3_infer.py:53`'s default string references `beit3_large_patch16_384_coco_retrieval.pth`. Conclusion: ambiguous; default to the official-guide init and document.
2. **What is `uit/cmp/checkpoint/pretrained.pth`?** Public SwinB, CMP-released weights, or author-private? `load_params_vision: True / load_params_text: False` (cmp.yaml) implies it carries vision weights at minimum. Unknown until a candidate file's `state_dict` is inspected.
3. **What `gt` set was used for the Algorithm 2 weight sweep (Table 4)?** Not specified in paper/code. Must choose our own value and document it (frozen local validation only).
4. **How was the 0.1M subset drawn for Table 1?** Not specified in paper/code. Must choose our own value and document it.
5. **Did the author run UIT with the cmp.yaml values (30 epochs / 1e-4 / step) or with CLI overrides matching the paper (22 / 1e-5 / cosine)?** The saved `output/cmp/config.yaml` cannot answer this — `Search.py` dumps the config before applying `--epo/--lr/--bs`. The paper's §3.1 values are taken as authoritative for our runs.
6. **UIT cosine scheduler internals:** `scheduler.py`'s `'cosine'` branch hard-codes `min_lr=1e-5, max_lr=1e-4` inside `lr_lambda` independent of the optimizer LR. Before Stage 5, verify what effective LR curve results when the optimizer LR is set to 1e-5, and document the choice.
7. **Gallery column order of the committed `sims_score/*.pt`:** produced by unsorted `os.listdir()` on the author's machine; no ID list is committed. The committed matrices can only be reused if that order can be recovered (it likely cannot) — our regenerated matrices with saved `gallery_ids.json` supersede them.
8. **Effective ensemble weights actually used by the author:** given the `inference.py:74` bug, running the script with defaults yields effective weights 0.9/0.9/0.9, not the paper's 0.925/0.9/0.9. Whether the author's submitted result used patched code, a different invocation, or the buggy path is unknowable from the repo.
