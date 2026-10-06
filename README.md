# Aquarium v2: closed-set YOLO benchmark

## Group information

| Item | Details |
|---|---|
| Group ID | 11 |
| Project ID | 6 |
| Assigned course topic (confirm scope change with staff) | Open-Vocabulary Object Detection with Text Prompts |

| No. | Member name | Student ID |
|---:|---|---|
| 1 | Lê Thanh Thảo | 23BA14265 |
| 2 | Chu Ngọc Minh Khôi | 2410478 |
| 3 | Nguyễn Đình Huy | 23BA14140 |
| 4 | Nguyễn Đức Minh | 23BA14193 |
| 5 | Lê Minh | 23BA14194 |
| 6 | Nguyễn Quang Minh | 23BA14195 |
| 7 | _To be filled in_ | _To be filled in_ |

**Scope:** compare three COCO-pretrained, **closed-set** nano detectors fine-tuned on Roboflow 100 Aquarium v2: **YOLOv8n, YOLO11n and YOLOv10n**. These models do not accept arbitrary text prompts. This scope differs from the assigned open-vocabulary topic; obtain course-staff approval before presenting it as the Project 6 submission. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the revised plan and [DATA.md](DATA.md) for the dataset and annotation policy.

**Recorded status:** YOLOv8n completed a 100-epoch, seed-0 run, including validation, held-out test and batch-1 GPU benchmarking ([results](docs/yolo/YOLOV8N.md), [summary](reports/runs/yolov8n_e100_s0.json)). YOLO11n and YOLOv10n completed one-epoch pipeline smoke tests; full runs and their results are still pending. Do not report smoke-test metrics as full results.

## Repository layout

```text
DL2026-11-6/
├── README.md, PROJECT_GUIDE.md, DATA.md
├── aquadet/                  prepare, train, evaluate, benchmark, compare CLI
├── configs/base.yaml         shared protocol
├── configs/models/           yolov8n.yaml, yolo11n.yaml, yolov10n.yaml
├── docs/yolo/                benchmark instructions and per-model results
├── scripts/data/             source audit and derived COCO conversion/validation
├── reports/                  committed run summaries and leaderboard
├── tests/                    aquadet unit tests
└── requirements-yolo.txt     pinned benchmark dependencies
```

`dataset/` (original export), `data/` (derived views), `weights/`, `runs/` and `.venv/` are local/ignored. Do not commit images, labels, credentials or downloaded weights. Run all commands from the repository root.

## Install and run

Python 3.12 is recommended. With `uv`:

```bash
uv venv .venv --python 3.12 --managed-python
uv pip install --python .venv/bin/python -r requirements-yolo.txt
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m aquadet env
python -m unittest discover -s tests
```

For the full setup, supported overrides, hardware/latency rules and evaluation protocol, read [docs/yolo/README.md](docs/yolo/README.md). If you do not use `uv`, create a Python 3.12 venv, install the appropriate PyTorch build for your hardware and then `pip install -r requirements-yolo.txt`.

1. Obtain the unmodified [Roboflow 100 Aquarium v2 YOLOv8-format **annotation export**](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2/download/yolov8) and put its `data.yaml`, `train/`, `valid/` and `test/` at `dataset/aquarium.v2-release.yolov8/` (or pass `--source PATH` / set `AQUADET_SOURCE`). The YOLOv8 **export format** is not a restriction on which of the three detectors can be trained. The published 448/127/63 image splits are preserved.
2. Prepare derived data, then run a smoke test before committing to a full run:

```bash
python -m aquadet prepare
python -m aquadet run --model yolo11n --smoke
python -m aquadet run --model yolov8n      # or yolo11n / yolov10n; 100 epochs each
python -m aquadet compare                  # generates reports/benchmark.{md,csv}
```

`run` trains, scores validation and test with both Ultralytics and pycocotools, benchmarks latency, and writes `reports/runs/<run>.json`. Use validation for model selection and **do not tune on test**. YOLOv8n's recorded test COCO AP is **47.6** (AP50 **76.1**), 3.0M parameters and 4.4 ms/image batch-1 inference on an RTX 3060; see [YOLOV8N.md](docs/yolo/YOLOV8N.md) for the exact protocol and limitations. Do not infer results for the two pending models.

The source audit flags two zero-area test shark annotations; `prepare` omits only those records in the derived YOLO and COCO ground truth, without changing the original export. See [DATA.md](DATA.md). `reports/benchmark.md` currently contains only the completed YOLOv8n run; compare latency only on the same GPU under the same conditions.
