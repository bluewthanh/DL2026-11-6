# Open-Vocabulary Object Detection with Text Prompts

## Group information

| Item | Details |
|---|---|
| Group ID | 11 |
| Project ID | 6 |
| Topic | Open-Vocabulary Object Detection with Text Prompts |

| No. | Member name | Student ID |
|---:|---|---|
| 1 | Lê Thanh Thảo | 23BA14265 |
| 2 | Chu Ngọc Minh Khôi | 2410478 |
| 3 | Nguyễn Đình Huy | 23BA14140 |
| 4 | Nguyễn Đức Minh | 23BA14193 |
| 5 | Lê Minh | 23BA14194 |
| 6 | Nguyễn Quang Minh | 23BA14195 |
| 7 | _To be filled in_ | _To be filled in_ |

Course project (Project 6): evaluate text-prompted detectors on the Roboflow 100 Aquarium v2 dataset. **Current status:** dataset audit and derived COCO validation are complete; a **pretrained YOLOE-26s** bare-name baseline has run on the original **validation** split. This is an in-progress project, not a final multi-model comparison. No fine-tuning or test-set inference has been done.

## Repository layout

```
DL2026-11-6/
├── README.md                 this file: install + reproduction commands
├── PROJECT_GUIDE.md          scope, evaluation rules, team plan
├── DATA.md                   dataset source/version, splits, audit, label policy (course-required)
├── docs/
│   ├── open_vocab/           zero-shot, text-prompted detectors
│   │   ├── YOLOE.md          YOLOE-26s baseline + prompt study
│   │   ├── OWLV2.md          OWLv2 validation workstream
│   │   └── visual_verification.md
│   └── yolo/                 closed-set fine-tuned YOLO benchmark
│       ├── README.md         setup, shared protocol, team workflow
│       └── YOLOV8N.md        YOLOv8n results (one file per model)
├── scripts/
│   ├── data/                 prepare_data.py, yolo_to_coco.py, validate_coco.py
│   └── open_vocab/           run_yoloe_baseline.py, run_owlv2.py (+ test), visualize_val.py, aquarium_prompts.json
├── aquadet/                  YOLO benchmark framework: python -m aquadet ...
├── configs/                  base.yaml (shared protocol) + models/{yolov8n,yolo11n,yolov10n}.yaml
├── reports/                  committed per-run summaries + generated benchmark.{md,csv}
├── tests/                    unit tests for aquadet (python -m unittest discover -s tests)
└── requirements-yolo.txt     pinned environment for the YOLO benchmark
```

Git-ignored local folders: `dataset/` (original export), `data/` (derived views), `weights/`, `runs/`, `results/`, `.venv/`. Run every command from the repository root.

| Workstream | Doc | Code |
|---|---|---|
| Dataset audit + COCO conversion | [`DATA.md`](DATA.md) | `scripts/data/` |
| YOLOE-26s zero-shot + prompt study | [`docs/open_vocab/YOLOE.md`](docs/open_vocab/YOLOE.md) | `scripts/open_vocab/run_yoloe_baseline.py` |
| OWLv2 zero-shot | [`docs/open_vocab/OWLV2.md`](docs/open_vocab/OWLV2.md) | `scripts/open_vocab/run_owlv2.py` |
| Closed-set YOLOv8n / YOLO11n / YOLOv10n | [`docs/yolo/README.md`](docs/yolo/README.md) | `aquadet/`, `configs/` |

<details>
<summary>Files moved on 6 Oct 2026 (for open branches)</summary>

| Old path | New path |
|---|---|
| `EXPERIMENTS.md` | `docs/open_vocab/YOLOE.md` |
| `OWLV2.md` | `docs/open_vocab/OWLV2.md` |
| `docs/visual_verification.md` | `docs/open_vocab/visual_verification.md` |
| `scripts/{prepare_data,yolo_to_coco,validate_coco}.py` | `scripts/data/` |
| `scripts/{run_yoloe_baseline,run_owlv2,test_run_owlv2,visualize_val}.py`, `scripts/aquarium_prompts.json` | `scripts/open_vocab/` |

Open-vocabulary scripts import `validate_coco` from `scripts/data/` through a one-line `sys.path` entry at the top of each file; copy it into new scripts in `scripts/open_vocab/` (for example `run_clip_rcnn.py`). New workstream docs go in `docs/open_vocab/` (for example `CLIP_RCNN.md`). After rebasing, Git applies edits to moved files automatically; only update path strings in any new text.
</details>

The YOLOv8 **export format** is used only for source annotations; the detector is **YOLOE-26s**, not YOLOv8. Original and derived datasets, downloaded weights, credentials, and generated results are excluded from Git.

## Closed-set YOLO benchmark (quick start)

Fine-tunes YOLOv8n, YOLO11n and YOLOv10n on the same splits under one protocol; full details in [`docs/yolo/README.md`](docs/yolo/README.md).

```bash
uv venv .venv --python 3.12 --managed-python && uv pip install --python .venv/bin/python -r requirements-yolo.txt
source .venv/bin/activate
python -m aquadet run --model yolov8n      # or yolo11n / yolov10n
python -m aquadet compare                  # -> reports/benchmark.md
```

## Install

Run commands from the repository root in a Python environment with PyTorch installed. The completed local baseline used Windows 11, Python 3.14.0, `torch==2.14.0+cpu`, `ultralytics==8.4.172`, and `pycocotools==2.0.11`. CPU inference works but is slow; use a compatible PyTorch CUDA installation for GPU runs. Install the remaining dependencies:

```bash
python -m pip install ultralytics==8.4.172 pycocotools==2.0.11 PyYAML Pillow
python -m pip install 'git+https://github.com/ultralytics/CLIP.git@a13192f8cb767260d7dfd98c843b0716593169e7'
# Only if downloading the dataset through the script:
python -m pip install roboflow
```

Git is needed for the CLIP install. See [`docs/open_vocab/YOLOE.md`](docs/open_vocab/YOLOE.md) for the upstream model and text-encoder links and SHA-256 checksums. This is **not yet a fully pinned environment**; a final `requirements.txt` and a GPU setup remain to be added before submission. Do not load checkpoints from untrusted sources.

## Reproduce the dataset and COCO checks

Official source: [Roboflow 100 Aquarium, version 2](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2). The published splits (448 train, 127 valid, 63 test) are kept unchanged. Download the [version-2 YOLOv8 export](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2/download/yolov8) into `data/aquarium-qlnqy-v2-yolov8/` yourself, **or** set `ROBOFLOW_API_KEY` privately and use `python scripts/data/prepare_data.py --download`. Never put the key in a tracked file. The commands below assume that default download path; replace `--source` if your existing export is elsewhere (the initial local copy used `data/aquarium.v2-release.yolov8/`).

```bash
# Audit the original export in place. Exit status 1 is EXPECTED for this exact
# source: two zero-width, zero-height test shark annotations; nothing is repaired.
python scripts/data/prepare_data.py --source data/aquarium-qlnqy-v2-yolov8 --report data/aquarium-v2-audit.json

# Strict conversion refuses this export. The reviewed policy omits ONLY those
# two records from derived test ground truth; use a fresh output directory.
python scripts/data/yolo_to_coco.py --source data/aquarium-qlnqy-v2-yolov8 --invalid-policy omit-known-zero-boxes
python scripts/data/validate_coco.py --coco-dir data/aquarium-v2-coco
```

Expected derived counts: **train 448 images / 3,328 annotations; valid 127 / 909; test 63 / 582**. The conversion manifest records the two exclusions. **Passing derived COCO validation does not mean the original source audit passed.** See [`DATA.md`](DATA.md) for details and possible shark-AP bias on a future test run. Do not re-split, modify or overwrite the original export or existing derived JSON.

## Reproduce the YOLOE-26s validation baseline

Download the text-promptable pretrained [`yoloe-26s-seg.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt) and [`mobileclip2_b.ts`](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts) into `weights/`, and check their hashes against [`docs/open_vocab/YOLOE.md`](docs/open_vocab/YOLOE.md). Do **not** use the prompt-free `-pf` checkpoint. The script expects the text encoder in the Ultralytics weights directory (`weights/` when run from this repository root).

```bash
# Replace --source with the path of your unmodified export if different.
# Choose a NEW output directory each time; the script will not overwrite runs.
python scripts/open_vocab/run_yoloe_baseline.py --source data/aquarium-qlnqy-v2-yolov8 --output results/yoloe26s_valid_bare
```

The script accepts the seven bare-name prompts (`fish`, `jellyfish`, `penguin`, `puffin`, `shark`, `starfish`, `stingray`), maps their indices to the verified COCO category IDs 1–7, converts pixel `xyxy` boxes to COCO pixel `xywh`, and evaluates on **all 127 validation images** using pycocotools bbox COCOeval. It refuses test JSON. The default CPU run uses image size 640, confidence 0.001, NMS IoU 0.7, and saves `predictions.json` (including scores), `config.json`, `runtime.json`, and `metrics.json` under the output folder. For a diagnostic subset only, pass `--limit 10` with another output folder; do not report subset AP as full-validation AP. To run on a GPU with a suitable CUDA PyTorch installation, pass `--device 0` and record that environment separately.

**Recorded full-validation result (CPU):** bbox AP@[0.50:0.95] **0.1311**, AP50 **0.2297**; per-class AP and exact configuration are in [`docs/open_vocab/YOLOE.md`](docs/open_vocab/YOLOE.md). These are validation numbers, **not** held-out test scores or fair GPU speed comparisons.

## Next steps and limitations

- Draw ground truth and predictions on several validation images to check box coordinates and class mapping before trusting comparisons.
- Run and save **separate** synonym and description variants on the same validation images. `scripts/open_vocab/aquarium_prompts.json` prepares the vocabularies, but those runs are **not done**. Only `sea jelly` and `sea star` are defensible synonyms; the other five entries remain unchanged in the synonym variant.
- Implement and evaluate Grounding DINO-T, OWLv2, and the CLIP + R-CNN baseline; compare latency on the **same warmed-up GPU**. Fine-tuning YOLOE is optional and has **not** been done.
- Record each completed experiment in `results/results.csv`, pin dependencies, and update this README with reproduction commands for **every number actually reported**. Do not tune thresholds/prompts on the test set. Run the held-out test split only when the validation protocol is frozen.

The team should fill in the confirmed group ID and member details above before submission. The lecturer also requires a **Member Contribution table in the report appendix**; the README roster does not replace it.
