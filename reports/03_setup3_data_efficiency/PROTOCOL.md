# Setup 3: YOLOv8n data-efficiency ablation (predeclared protocol)

Written on 2026-10-06 between 15:03 and 15:04 +0700, before the first subset run started training (15:04:50, `args.yaml` of `yolov8n_frac010_s0`) and before anything was scored. The time first recorded here, 15:05, was an estimate and was slightly too late. The file was later rewritten when `report/` was renamed to `reports/`, so its file-system timestamps no longer show the original writing time.

## Question

How much labelled Aquarium training data does the supervised closed-set YOLOv8n baseline need before its validation COCO AP exceeds that of the zero-shot, text-prompted YOLOE-26s (bare-name prompts), evaluated on the same 127 validation images with the same evaluator?

## Single factor that changes

The **number of training images**: 10 %, 25 %, 50 % and 100 % of the 448 official training images (45, 112, 224 and 448 images). Everything else is the shared `configs/base.yaml` protocol for `configs/models/yolov8n.yaml`: 100 epochs, no early stopping, the same optimizer and augmentation, and initialisation from COCO-pretrained `yolov8n.pt`. Training **epochs** stay fixed, so smaller subsets also get fewer optimizer steps. This is part of "less labelled data" and is stated as a limitation.

## Subset rule

- For sampling seed `s`, the sorted list of training image file names is shuffled with `random.Random(s)`, and the first `round(f × 448)` images form subset `f`. Subsets are therefore nested: 10 % ⊂ 25 % ⊂ 50 % for the same seed.
- The sampling is uniform over images, with no class stratification. Per-class image and box counts for every subset are recorded in `subsets.json`. A class with few training images may be missing from a small subset. That class then scores AP 0, which is a genuine outcome of having little labelled data.
- Seeds: `s ∈ {0, 1, 2}` for 10/25/50 %. The training seed `train.seed` equals the sampling seed. For 100 %, seed 0 reuses the existing run `runs/yolov8n_e100_s0`, and seeds 1 and 2 are trained fresh on the full training split.
- Validation and test images and labels are the unchanged derived views (`data/aquarium-v2-yolo`, `data/aquarium-v2-coco`). The test split is **not** used in Setup 3.

## Measurement

- pycocotools COCOeval bbox on `data/aquarium-v2-coco/valid.json`: 127 images, 909 boxes, IoU 0.50:0.95, maxDets 100. The inference settings match the YOLOE runs: imgsz 640, conf 0.001, NMS IoU 0.7, max_det 300.
- Both `last.pt` and `best.pt` are scored. **`last.pt` is the primary number**: `best.pt` is selected on validation fitness, so its validation AP is optimistically biased.
- Reported values: the mean ± sample std over the three seeds, per fraction, for overall AP, AP50 and per-class AP.
- Reference line: YOLOE-26s bare-name validation AP from this machine's reproduction (`reports/02_setup2_prompt_study`).

## What would count as an answer

The smallest tested fraction whose mean `last.pt` AP exceeds the YOLOE bare-name AP, together with its seed spread. If even 10 % exceeds it, the answer is "≤ 10 % (45 images)". No fractions are added after seeing results unless they are reported as a separate, post-hoc follow-up.

## Post-hoc addendum (2026-10-06 15:36 +0700)

Added **after** scoring the 10/25/50 % runs and finding that every 10 % run already exceeds the YOLOE bare-name AP: 16.9 / 18.6 / 22.2 against 13.1 (last.pt). The predeclared answer is therefore "≤ 10 % (45 images)". To locate the actual crossover, three smaller fractions are added with the same rule, seeds and settings: **1 %, 2.5 % and 5 % (4, 11 and 22 images)**. They are prefixes of the same per-seed permutation, so they are nested inside the 10 % subsets. These runs are reported separately as a post-hoc follow-up (`posthoc_small_fractions/`), and the predeclared table is left unchanged.
