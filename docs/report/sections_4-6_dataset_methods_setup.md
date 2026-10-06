# 11_6_Report — Sections 4–6: Dataset, Methods, Experimental Setup

**Status:** draft for the report. Author: Trần Nam (assigned with Chu Ngọc Minh Khôi). Every fact comes from the repo at commit `564d49f`, which includes merged PR #12. Sources:
- `DATA.md` and `docs/open_vocab/YOLOE.md`;
- `configs/base.yaml`;
- `reports/00_setup` to `03_setup3_data_efficiency`;
- `reports/VERIFICATION.md`.

Interpretation of the results belongs to Section 7 (Le Minh / Thanh Thảo). These sections define *what* was measured and *how*.

---

## 4. Dataset and Data Preparation

### 4.1 Source

We use **Roboflow 100 Aquarium, version 2 (`release`)**:
- workspace `roboflow-100`, project `aquarium-qlnqy`;
- generated 30 August 2022, licensed CC BY 4.0 (https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2).

It is a bounding-box object-detection dataset of aquarium photographs. The version-2 page states that no preprocessing and no augmentation were applied when the version was generated. We deliberately avoid version 1 (`release-640`), which has different preprocessing.

The dataset has seven classes: **fish, jellyfish, penguin, puffin, shark, starfish, stingray**. Their IDs are 0–6 in the YOLO export and 1–7 in the derived COCO ground truth. Every model and every prompt set uses this one mapping.

### 4.2 Splits

We keep the published splits unchanged and never re-split them. The RF100 benchmark metadata independently confirms these counts.

| Split | Images | Boxes (derived COCO) | Role in this project |
|---|---:|---:|---|
| Train | 448 | 3,328 | Fine-tuning the YOLOv8n baseline (all images in Setup 1; nested subsets in Setup 3). YOLOE is not trained. |
| Validation | 127 | 909 | **Every model comparison** (Setups 1–3) |
| Test | 63 | 582 | Scored once, for YOLOv8n only. Never compared with YOLOE and never used for any choice. |
| **Total** | **638** | **4,819** | |

Validation boxes per class: fish 459, jellyfish 155, penguin 104, puffin 74, shark 57, starfish 27, stingray 33. The classes are strongly imbalanced. Fish account for half of the validation boxes, while starfish and stingray each have fewer than 35, so their per-class AP is noisy.

### 4.3 Data preparation pipeline

We downloaded the version-2 export in **YOLOv8 format** (YOLO TXT labels and `data.yaml`). We then processed it with three scripts. None of them edits, moves or re-splits the original files.

1. **Audit** (`scripts/data/prepare_data.py`). It checks:
   - the image count of each split;
   - that each image has exactly one label file;
   - empty label files;
   - that each label line has five normalized fields;
   - that class IDs are in range;
   - that the class mapping in `data.yaml` is the expected one.

   It writes `data/aquarium-v2-audit.json`.
2. **Conversion to COCO** (`scripts/data/yolo_to_coco.py`). It turns normalized YOLO `[class, xc, yc, w, h]` labels into COCO JSON with pixel `[x, y, width, height]` boxes and pixel² areas, one file per split. By default it refuses to convert while the audit fails; Section 4.4 describes the single exception.
3. **Validation** (`scripts/data/validate_coco.py`). All three JSON files load in `pycocotools` 2.0.11 and are checked for:
   - unique image and annotation IDs;
   - the seven expected category ID/name pairs;
   - positive, in-image boxes with matching areas;
   - no orphan annotations.

   All splits pass (`reports/00_setup/validate_coco.txt`).

For training, `python -m aquadet prepare` creates a hard-linked YOLO view of the same files (`data/aquarium-v2-yolo`). It omits the same two test boxes as in Section 4.4, and nothing else (`reports/00_setup/aquadet_prepare.txt`).

The derived `data/aquarium-v2-coco/valid.json` is the **single ground-truth file** that every validation number is scored against. Its SHA-256 is `684400c8…5702`, computed with LF line endings. A Windows checkout with CRLF line endings gives a different raw hash but the same content (`reports/VERIFICATION.md`). Images, labels, derived JSON and weights are not committed. The scripts regenerate them from the official export.

### 4.4 Data-quality findings and decisions

The source audit **fails**, and we report this rather than call the dataset clean.

- **Two zero-area test boxes.** Two `shark` labels in the test split have width = height = 0:
  - `IMG_2423…txt`, line 17;
  - `IMG_2570…txt`, line 15.

  These cannot be real boxes, and inventing a size for them would fabricate ground truth. Under an explicit, opt-in policy (`--invalid-policy omit-known-zero-boxes`), the converter omits **only these two records** from the derived COCO test file:
  - all 63 test images are kept;
  - the omissions are listed in `conversion_manifest.json`;
  - the converter refuses to run if any other invalid record appears.

  The policy affects **test only** and cannot change any validation number in this report. A detection at either point would count as a false positive, so test shark AP may be slightly biased.
- **One empty training label.** `train/labels/IMG_3133…txt` is empty. On visual inspection, the 1536×2048 image shows a rocky tank with no visible target animal. We keep it as a plausible negative image. This is a visual judgment, not proof.

### 4.5 Characteristics and limitations of the data

| Class | Val boxes | Val images containing it | Max per image | Median box area (% of image) | ≈ box side at 640 px input |
|---|---:|---:|---:|---:|---:|
| fish | 459 | 63 | 43 | 0.77 | ~48 px |
| jellyfish | 155 | 9 | 33 | 1.12 | ~59 px |
| penguin | 104 | 17 | 22 | 0.37 | ~34 px |
| puffin | 74 | 15 | 11 | 0.40 | ~35 px |
| shark | 57 | 28 | 6 | 1.29 | ~63 px |
| starfish | 27 | 17 | 4 | 2.27 | ~83 px |
| stingray | 33 | 23 | 3 | 3.11 | ~98 px |

Computed from `valid.json` with `python scripts/analysis/gt_stats.py`. The "≈ box side" column assumes a 3:4 image letterboxed to 640 px, since 87 of the 127 validation images are 1536×2048. It explains why we do not use COCO's small/medium/large size buckets: they are defined on original pixels, and on these 2–12 MP photos almost every box counts as "large".

Limitations:
- **Small dataset:** 127 validation images.
- **Class imbalance:** two classes have fewer than 35 validation boxes.
- **Crowding:** all 155 jellyfish boxes come from just 9 images. Penguins and puffins are the smallest objects and appear in groups.
- **Image quality:** some photos have poor lighting or rain-spattered glass (e.g. the puffin image `val_047` in `reports/05_qualitative_overlays`).
- **Single domain:** all images come from one domain, aquarium photos.
- **Fixed label set:** text prompts are model inputs. They do not create new ground-truth categories.

---

## 5. Methods

### 5.1 Task formulation

The input is an image plus a list of **K = 7 text prompts**, one per target class. The detector outputs boxes, each with a score and a prompt index, and prompt index *k* maps to COCO category *k + 1*. Rewording a prompt therefore changes only the model's **input**. The label space and the ground truth stay fixed, so any change in AP is caused by the wording.

### 5.2 Main method: text-prompted YOLOE-26s (zero-shot)

The main model is **YOLOE-26s** (Ultralytics), an open-vocabulary detector that accepts class names as free text at inference time. Its two assets are used unchanged:

| Asset | Details |
|---|---|
| `yoloe-26s-seg.pt` | Pretrained text-promptable checkpoint, 15.27 M parameters including an unused mask head. SHA-256 `48f24206…0654`. |
| `mobileclip2_b.ts` | MobileCLIP2-B text encoder. SHA-256 `35d7f213…982f`. |

We use the text-promptable checkpoint, not the prompt-free `-pf` variant.

**Inference** works in three steps:
1. `model.set_classes(prompts)` encodes the seven prompt strings once with the text encoder, giving one embedding per prompt.
2. The detector scores candidate regions against these seven embeddings.
3. Each box takes the index of its best-matching prompt.

Only the **bounding boxes** are evaluated; masks are discarded.

**YOLOE is not fine-tuned on Aquarium.** No Aquarium image or label adjusts it, so this is a zero-shot, text-conditioned evaluation. We cannot verify which concepts appeared in YOLOE's pretraining data. "Zero-shot" therefore means *no training on this dataset*, not that the class names were unseen.

**The runner,** `scripts/open_vocab/run_yoloe_baseline.py`, does the following:
- validates `valid.json`;
- processes images in sorted ID order;
- converts pixel `xyxy` boxes to COCO `xywh`;
- writes `config.json` (prompts, hashes, settings, environment), `predictions.json`, `runtime.json` and `metrics.json` to a new folder.

It **refuses the test split**, so prompts cannot be tuned on test.

### 5.3 Baseline: closed-set YOLOv8n (supervised fine-tuning)

The baseline is **YOLOv8n**, initialized from the official COCO-pretrained `yolov8n.pt` (3.01 M parameters, 8.1 GFLOPs). It is fine-tuned on the Aquarium training images with the project's `aquadet` pipeline. Unlike YOLOE, it has a fixed seven-class output head and **cannot accept text prompts**. It can detect only the classes it was trained on, and adding a class needs new labels and retraining.

We keep YOLOv8n as the **only** closed-set model; earlier YOLOv10n and YOLO11n work was removed in PR #12. It shows what a small, standard detector reaches when it *does* get in-domain labels, which puts YOLOE's zero-shot score in context.

### 5.4 Comparison strategy

**What is held equal.** Both methods are scored under **one evaluation protocol** (6.1): the same 127 images, the same `valid.json`, the same evaluator and the same inference settings. One script, `scripts/analysis/build_tables.py`, rescores every row from the saved predictions, so all rows go through identical scoring.

**What deliberately differs.** The two methods differ in regime (supervised vs zero-shot) and in model size. The comparison therefore measures how far a label-free, text-prompted detector is from a supervised detector trained on this exact domain. It **does not rank** the two architectures.

**Rules for a fair comparison:**
1. **`last.pt` is the baseline's headline checkpoint.** Ultralytics' `best.pt` is selected on validation fitness, so its validation AP is optimistic. `last.pt`, the final epoch, takes no input from validation, and with no early stopping it is the fair row. We show `best.pt` only for reference.
2. **YOLOE is reported on validation only.** YOLOv8n's test AP is never placed next to YOLOE's validation AP.
3. **Speed is not compared** between the models.
4. **Setup 3** turns the gap into an amount of labelled data: how many labelled images YOLOv8n needs before it beats zero-shot YOLOE.

---

## 6. Experimental Setup

### 6.1 Shared evaluation protocol (all setups)

| Item | Setting |
|---|---|
| Split | Validation: 127 images, 909 ground-truth boxes |
| Ground truth | `data/aquarium-v2-coco/valid.json` (derived, validated) |
| Input size | 640 px, batch 1, FP32, no test-time augmentation |
| Score threshold | conf = 0.001. Low on purpose, so that AP covers the full precision–recall curve. |
| NMS | IoU 0.7, class-aware (`agnostic_nms=False`) |
| Max detections from model | 300 per image |
| Evaluator | pycocotools 2.0.11 `COCOeval`, bbox, maxDets = 100 |
| Primary metric | **COCO AP@[0.50:0.95]** ×100 (mean over IoU 0.50 to 0.95 in steps of 0.05) |
| Secondary metrics | AP50, AP75, AR100, per-class AP and AP50 |

The model's `max_det = 300` and the evaluator's `maxDets = 100` are two separate limits; the evaluator keeps the top 100 detections per image. COCO's APs/APm/APl columns are reported in the repo, but they are noisy here (see 4.5).

### 6.2 Hardware and software

All results in `reports/` were produced on 2026-10-06 on one Linux machine with an NVIDIA RTX 3060 12 GB GPU. YOLOE and the closed-set benchmark pin different Ultralytics versions, so two Python 3.12 environments are used:

| Environment | Used for | Key packages |
|---|---|---|
| `.venv` | YOLOv8n training and evaluation, Setup 3, tables, overlays | torch 2.14.1+cu130, ultralytics **8.4.173**, pycocotools 2.0.11 |
| `.venv-yoloe` | YOLOE prompt runs and the demo | torch 2.14.1+cu130, ultralytics **8.4.172**, pycocotools 2.0.11, Ultralytics/CLIP @ `a13192f` |

The full package lists are in `reports/00_setup/*_freeze.txt`. All 13 unit tests pass in both environments.

### 6.3 Setup 1: Baseline vs main model

- **Goal:** put zero-shot text-prompted detection in context against supervised closed-set detection on the same data.
- **Main model:** YOLOE-26s with the bare class names as prompts. The synonym and description sets are also listed in the table.
- **Baseline:** YOLOv8n fine-tuned on all 448 training images. The settings come from the shared `configs/base.yaml` (protocol hash `42534e32b914`):
  - optimisation: 100 epochs with no early stopping, batch 16, 640 px, AdamW with lr0 0.001 and final LR factor 0.01, momentum 0.9, weight decay 0.0005, 3 warm-up epochs, linear LR schedule;
  - augmentation: Ultralytics default augmentation, with mosaic switched off for the last 10 epochs;
  - reproducibility: deterministic mode, mixed precision, no RAM caching, COCO-pretrained initialization.
- **Checkpoints and seeds:**
  - headline: **seed 0, `last.pt`**;
  - seed spread: `last.pt` of three seeds (0, 1, 2);
  - `best.pt` (best epoch 86/100) shown only as an optimistic reference.
- **Commands:**
  - `python -m aquadet run --model yolov8n` trains, scores val/test and runs the latency benchmark;
  - `python scripts/yolo/data_efficiency.py evaluate` also scores `last.pt`;
  - `scripts/analysis/build_tables.py` rebuilds the table from stored predictions, with no model run (`reports/01_setup1_baseline/README.md`).
- **Not compared:** YOLOv8n's test AP and latency are reported separately (`docs/yolo/YOLOV8N.md`), never against YOLOE.

### 6.4 Setup 2: Main experiment, prompt wording

- **Goal:** measure how the **wording** of class prompts affects zero-shot detection. This is the lecturer's second requirement.
- **Model:** YOLOE-26s with no training, under the protocol in 6.1.
- **Prompt sets:** three, versioned in `scripts/open_vocab/aquarium_prompts.json`.

| Class | Bare name | Synonym variant | Visual description |
|---|---|---|---|
| fish | fish | fish | swimming animal with fins and a tail |
| jellyfish | jellyfish | **sea jelly** | translucent sea animal with a bell and trailing tentacles |
| penguin | penguin | penguin | black-and-white swimming bird with flippers |
| puffin | puffin | puffin | seabird with a large colorful beak |
| shark | shark | shark | large fish with a dorsal fin and pointed snout |
| starfish | starfish | **sea star** | star-shaped sea animal with radiating arms |
| stingray | stingray | stingray | flat sea animal with broad wing-like fins and a long tail |

- **Synonym variant:** only jellyfish and starfish have widely accepted synonyms, so only those two prompts change. The other five stay identical; we did not invent weak synonyms. The descriptions are visual phrases written by the team: one choice of wording, not a canonical set.
- **Held fixed across the three runs:** the same image IDs, ground-truth hash, checkpoint and text-encoder hashes, prompt-file hash and inference settings. All of these are recorded in each run's `config.json` and checked automatically.
- **Two independent runs:** a documented CPU run (Windows, Python 3.14, torch 2.14.0+cpu) and a GPU reproduction (6.2, `--device cuda:0`) agree within **0.034 AP points** on every overall and per-class number (`reports/02_setup2_prompt_study`). The report uses the GPU reproduction, whose predictions are stored.
- **Command:** `python scripts/open_vocab/run_yoloe_baseline.py --source <export> --prompt-set {bare|synonym|description} --device cuda:0 --output results/<new_folder>`.

### 6.5 Setup 3: Data-efficiency ablation (how many labels equal zero-shot?)

- **Question:** how many labelled Aquarium training images does YOLOv8n need before its validation AP exceeds zero-shot YOLOE-26s with bare-name prompts (13.1 AP)?
- **Predeclared protocol:** written in `reports/03_setup3_data_efficiency/PROTOCOL.md` *before* any subset run was trained or scored.
- **Single factor:** the number of training images. YOLOv8n is trained on 10 / 25 / 50 / 100 % of the 448 images (45 / 112 / 224 / 448). Everything else follows the Setup 1 protocol.
- **Subset rule:**
  - for seed *s* ∈ {0, 1, 2}, the sorted training file list is shuffled with `random.Random(s)`, and the first `round(f × 448)` images form subset *f*;
  - subsets are therefore **nested** (10 % ⊂ 25 % ⊂ 50 %) and sampled uniformly, without class stratification;
  - the training seed equals the sampling seed.
  - Per-class coverage of each subset is recorded in `subsets.json`. A class missing from a subset scores AP 0, a genuine outcome of having few labels.
- **Measurement:**
  - `last.pt` validation COCO AP, as mean ± sample std over 3 seeds, with the same evaluator and settings as YOLOE (6.1);
  - per-class AP is reported as well;
  - the test split is not used.
- **Post-hoc follow-up:** 1 / 2.5 / 5 % (4 / 11 / 22 images), added *after* seeing that every 10 % run already beat YOLOE. It is recorded as a dated addendum to the protocol and reported separately from the predeclared table, in `posthoc_small_fractions/`. In total there are 21 YOLOv8n runs.
- **Limitation built into the design:** epochs are fixed at 100, so smaller subsets also get fewer optimizer steps (about 34 for 4 images vs 719 for 448). "Fewer labels" therefore includes "fewer updates".
- **Command:** `python scripts/yolo/data_efficiency.py {train|evaluate|summarize}`. See the section README for the exact flags.

### 6.6 Reproducibility

- **Stored predictions.** Every prediction set used in the report is stored gzip-compressed in `reports/`: the three YOLOE runs and every YOLOv8n checkpoint, all at conf ≥ 0.001. Every table, plot and overlay can be rebuilt **without running a model**.
- **Not committed:** trained weights, dataset images and raw training logs. The per-epoch metrics are kept in each run's `results.csv`.
- **No tuning on test.** No threshold, prompt or checkpoint was chosen using the test split. Validation was used to choose `best.pt` and the overlay display thresholds; this is why `last.pt` is the headline baseline.
- **Independent check.** Lê Thanh Thảo rescored all five Setup 1 prediction files and all 21 Setup 3 runs from the stored files on a second machine (Windows). Everything matched within CSV rounding (`reports/VERIFICATION.md`). No model was re-run in that check.
