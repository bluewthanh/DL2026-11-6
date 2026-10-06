# Dataset: Roboflow 100 Aquarium v2 (YOLO benchmark)

## Source and version

- **Official Roboflow Universe page, version 2 (`release`):** https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2
- **Workspace / project:** `roboflow-100` / `aquarium-qlnqy`. This is **not** the unrelated `/roboflow-100/aquarium` URL.
- **Dataset version:** 2 (`release`), generated 30 August 2022. Do not substitute version 1 (`release-640`), which has different preprocessing.
- **Task:** object detection (bounding boxes).
- **License shown on the Roboflow Universe project page:** CC BY 4.0. See the [version-2 page](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2) and [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/). Credit Roboflow 100 and check attribution for any redistributed material.

## Downloadable processed annotations (no images)

The group also distributes its **derived COCO train/valid/test annotations** as a small [downloadable ZIP](https://github.com/bluewthanh/DL2026-11-6/releases/download/aquarium-v2-derived-coco-v1/aquarium-v2-derived-coco.zip) (SHA-256 `1a30695ce8b6ee475f195ba3f0a9e8eb300b2da8baf7946749295e8c56cfe0ea`). This is a **processed annotations-only** derivative of the official Aquarium v2 YOLOv8 export: it includes `train.json`, `valid.json`, `test.json`, a portable omission-policy manifest and attribution notice. It contains **no images, original YOLO label files, weights, keys or machine-specific paths**. Download the original images separately from the [official version-2 dataset](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2/download/yolov8). Keep the extracted `*.json` under `data/aquarium-v2-coco/` and the original export at `data/aquarium.v2-release.yolov8/`: COCO `images[].file_name` is relative to that original export root. Check the ZIP hash before use, then run `python scripts/data/validate_coco.py --coco-dir data/aquarium-v2-coco`.

To **regenerate** instead of downloading, use the conversion command in the reproduction policy below, and then validate. `scripts/data/package_derived_coco.py` builds the same portable ZIP from validated local COCO JSON and strips absolute paths from its manifest. Do not equate the two omitted test shark boxes with a repaired source dataset; this derivative documents their omission. Roboflow 100 Aquarium v2 is credited under the CC BY 4.0 license shown by its official project/export, with [license terms](https://creativecommons.org/licenses/by/4.0/); keep the attribution and change notice when reusing the derivative.

## Images and original splits

| Split | Images |
|---|---:|
| Train | 448 |
| Validation (`valid` on Roboflow) | 127 |
| Test | 63 |
| **Total** | **638** |

These are the splits of the published version 2; **do not re-split or use the test split for model selection or threshold tuning**. Roboflow lists the split as approximately 70% / 20% / 10%. The [RF100 benchmark's dataset statistics](https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv) independently list 448 train, 127 valid, 63 test, 638 total and 7 classes for `aquarium-qlnqy`.

## Classes and text-prompt study

The open-vocabulary experiment uses the same seven annotated classes with three versioned prompt sets (bare names, two reliable synonyms and visual descriptions) in [`scripts/open_vocab/aquarium_prompts.json`](scripts/open_vocab/aquarium_prompts.json). Prompts are **model inputs**, not new ground-truth categories; mapping and splits stay fixed. The closed-set YOLO benchmark does not consume text prompts.

The seven classes in the [RF100 benchmark metadata](https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/labels_names.json) are **fish, jellyfish, penguin, puffin, shark, starfish, stingray**. The export's YOLO IDs are 0–6 in that order; derived COCO IDs are 1–7. The closed-set YOLOv8n baseline and the text-prompted YOLOE validation study use this same category mapping. See the audited export mapping below.

## Annotations and available exports

The [version-2 download page](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2/download) lists **COCO JSON** (also available; the benchmark derives COCO ground truth from its audited YOLO export), multiple **YOLO TXT + YAML** variants (e.g., YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv11, YOLOv12 and YOLO26), **Pascal VOC XML**, **YOLO Darknet TXT**, **TFRecord**, **PaliGemma JSONL**, and **CreateML JSON** among its popular formats. These are *available export formats*, not a claim that all have been downloaded or checked locally. The version-2 page states **no preprocessing and no augmentations** were applied when this version was generated. Model-specific resizing/normalization must be documented separately when implemented.

## Reproducible YOLO export and local audit

**Chosen annotation export:** version 2, YOLOv8 format (YOLO TXT labels, `data.yaml`); this is the annotation format used by the YOLOv8n baseline. [Official version-2 YOLOv8 download](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2/download/yolov8). `scripts/data/prepare_data.py` downloads this exact workspace/project/version/format into `data/aquarium-qlnqy-v2-yolov8/` by default, or audits an existing export **in place** via `--source`, without re-splitting, moving, or editing dataset files. A COCO JSON export is separately available for later evaluation, but is not what this audit script downloads.

```bash
python -m pip install roboflow PyYAML
# Set ROBOFLOW_API_KEY in your local environment privately (do not put it in a command or tracked file).
python scripts/data/prepare_data.py --download
# Or, for an already downloaded YOLOv8 export (no key needed):
python scripts/data/prepare_data.py --source /path/to/export-root
```

The export root contains `data.yaml` and the original `train`, `valid` (or `val`), and `test` image/label directories. The audit accepts either `val` or `valid` as the YAML validation key, resolving the directory named by the YAML; it does not change the splits. It checks expected image counts; one-to-one image/label stem matching; missing images/labels; empty labels (reported separately, not automatically errors because negative images can be legitimate); normalized 5-field YOLO detection boxes; class-ID range; and the `data.yaml` ID-to-name mapping against all seven classes. It writes `data/aquarium-v2-audit.json` and exits nonzero when checks fail. Label files contain **YOLO normalized `[class_id, x_center, y_center, width, height]`**, not COCO pixel `xywh`. The audit does **not** repair or remove invalid annotations.

### Actual local export audit (FAILED: annotation defects)

Run on the original export at `data/aquarium.v2-release.yolov8/` with:

```bash
python scripts/data/prepare_data.py --source data/aquarium.v2-release.yolov8 --report data/aquarium-v2-audit.json
```

The export's `data.yaml` declares `train: ../train/images`, **`val: ../valid/images`**, `test: ../test/images`, `nc: 7`, and Roboflow metadata `workspace: roboflow-100`, `project: aquarium-qlnqy`, `version: 2`, `license: CC BY 4.0`. Actual ID mapping is **0 fish; 1 jellyfish; 2 penguin; 3 puffin; 4 shark; 5 starfish; 6 stingray**. This is the same order used by the YOLOv8n baseline.

| Original directory | Images | Labels | Missing pairs | Empty label files | Invalid annotations |
|---|---:|---:|---:|---:|---:|
| `train` | 448 | 448 | 0 | 1 | 0 |
| `valid` (YAML key `val`) | 127 | 127 | 0 | 0 | 0 |
| `test` | 63 | 63 | 0 | 0 | **2** |

### Data-quality decision and evaluation impact

The one empty training label is `train/labels/IMG_3133_jpeg_jpg.rf.4da6a12d067cae0110409f3e2a21b31a.txt`. **Visual inspection of the corresponding 1536×2048 image** shows a rocky aquarium waterfall/tank, with no clearly visible fish, shark, starfish, jellyfish, penguin, stingray or puffin. Treat this as a **plausible negative image with zero labeled target objects**, not a demonstrated missing annotation. This is a visual judgment, not proof that no tiny/occluded object exists. It remains in train unchanged; no effect on val/test ground-truth counts, but if used in training, predictions on it can contribute negative examples. Reinspect if a team member finds a target object in the image.

Two **class 4 (`shark`) test labels** contain zero-width, zero-height boxes:

1. `test/labels/IMG_2423_jpeg_jpg.rf.39aca9cd118509b10f192b87e7ce9692.txt:17` — normalized xywh `0.5826822916666666 0.2958984375 0 0`.
2. `test/labels/IMG_2570_jpeg_jpg.rf.05001e7087160e744b14f28f1fa5c768.txt:15` — normalized xywh `0.059895833333333336 0.86572265625 0 0`.

**The audit fails (`passed: false`). Source annotations are NOT fully valid.** These two points cannot be valid detection boxes; a guessed positive-size replacement would fabricate ground truth. Under the policy below, evaluating all 63 test images with these two records absent from *derived* COCO ground truth means a prediction at either position could count as a false positive, and shark AP may be biased. State this limitation alongside test AP. Never describe the dataset itself as repaired or fully verified. The full structured findings (file, line, class ID, raw xywh, reason) are in the local `data/aquarium-v2-audit.json`; generated audit and images are ignored by git.

### Reproducible evaluation policy (original splits unchanged)

Use `scripts/data/yolo_to_coco.py` to create **derived**, auditable COCO JSON ground truth. By default it runs the strict audit and **refuses all conversion** while invalid boxes remain. For this exact export, an *explicit opt-in* `--invalid-policy omit-known-zero-boxes` keeps **all original images in their original splits** and every valid annotation, and omits **only these two identified zero-area source records** from the derived COCO JSON. It refuses any new/different invalid record or audit error; it writes a manifest listing each omission and `source_audit_passed: false`. The original YOLO files remain untouched. File names in the COCO JSON are paths relative to the original export root; category IDs are COCO 1–7 corresponding to source YOLO IDs 0–6, matching the audited export order. Boxes are pixel `[x, y, width, height]` and area is pixel². Do not change this rule between models. The policy affects **test ground truth only**; compare models on the same 63 test images and derived JSON, tune on the 127-image validation split, and report the two exclusions and possible shark-AP bias. If a lecturer requires a different treatment, decide and document it **before** test evaluation, then regenerate every model's evaluation with the same treatment.

```bash
python -m pip install PyYAML Pillow
# Strict default refuses this export: no JSON written.
python scripts/data/yolo_to_coco.py --source data/aquarium.v2-release.yolov8
# Explicit, reviewed policy; outputs data/aquarium-v2-coco/{train,valid,test}.json
# and conversion_manifest.json. Use a fresh output directory for each regeneration.
python scripts/data/yolo_to_coco.py --source data/aquarium.v2-release.yolov8 --invalid-policy omit-known-zero-boxes
```

Conversion performed locally with this policy: **448 train images / 3328 boxes**, **127 valid images / 909 boxes**, **63 test images / 582 valid boxes**. `data/aquarium-v2-coco/conversion_manifest.json` lists the two excluded zero-area test records. **Derived COCO validation passed** using `pycocotools 2.0.11` in the local Python 3.14 environment (Windows): all three JSON files load with `COCO`, have unique image/annotation IDs, the expected seven category ID/name pairs, valid positive in-image bboxes and matching positive areas, and no orphan annotations. This validates the *derived JSON*, not the original YOLO labels: the source audit still fails due to the two omitted zero-area shark boxes. The YOLOv8n benchmark has since evaluated val and test on this derived ground truth; see `docs/yolo/YOLOV8N.md`.

```bash
python -m pip install pycocotools
python scripts/data/validate_coco.py --coco-dir data/aquarium-v2-coco
# PASS train: 448 images, 3328 annotations
# PASS valid: 127 images, 909 annotations
# PASS test:   63 images,  582 annotations
# TOTAL:      638 images, 4819 annotations
```

Do **not** commit images, labels, credentials, or downloaded weights. The [official RF100 benchmark downloader](https://github.com/roboflow/roboflow-100-benchmark/blob/main/scripts/download_dataset.py) likewise requires an API key; its default export format is COCO, whereas this project's preparation script requests YOLOv8.

## Verification sources and limits

- [Official Universe version-2 page](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2): version, license displayed for project, counts, splits, processing, export choices. The site blocked direct automated access (HTTP 403); its page text was retrieved through a text mirror for verification. Confirm in browser if possible.
- [Official RF100 benchmark dataset links](https://github.com/roboflow/roboflow-100-benchmark/blob/main/datasets_links.txt): lists `aquarium-qlnqy/2`.
- [RF100 benchmark statistics](https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv): split counts, total, number of classes.
- [RF100 benchmark label names](https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/labels_names.json): seven class names (its numeric class frequencies are not COCO category IDs).
