# Aquarium v2: text-prompted object detection and YOLO benchmark

## Group information

| Item | Details |
|---|---|
| Group ID | 11 |
| Project ID | 6 |
| Assigned course topic | Open-Vocabulary Object Detection with Text Prompts |

| No. | Member name | Student ID |
|---:|---|---|
| 1 | Lê Thanh Thảo | 23BA14265 |
| 2 | Chu Ngọc Minh Khôi | 2410478 |
| 3 | Nguyễn Đình Huy | 23BA14140 |
| 4 | Nguyễn Đức Minh | 23BA14193 |
| 5 | Lê Minh | 23BA14194 |
| 6 | Nguyễn Quang Minh | 23BA14195 |
| 7 | _To be filled in_ | _To be filled in_ |

**Lecturer's requirements:** detect objects using text descriptions rather than only fixed classes, and measure how object names, synonyms and descriptive prompts affect performance. The primary experiment uses text-prompted **YOLOE-26s** on Roboflow 100 Aquarium v2; [PROJECT_GUIDE.md](PROJECT_GUIDE.md) maps each requirement to the pipeline. **YOLOv8n, YOLO11n and YOLOv10n are supplementary, supervised closed-set controls**, not text-prompted models. See [DATA.md](DATA.md) for the dataset and annotation policy.

**Recorded status:** a pretrained YOLOE-26s bare-name validation baseline and a GPU validation prompt study (name/synonym/description) are documented in [YOLOE.md](docs/open_vocab/YOLOE.md). A full CPU bare/synonym/description comparison has also been run locally; generated `results/` artifacts are ignored by Git, so independent reproduction is still needed before submission. There is no reported prompt-based test evaluation. YOLOv8n and YOLOv10n completed 100-epoch, seed-0 supervised runs ([YOLOv8n](docs/yolo/YOLOV8N.md), [YOLOv10n](docs/yolo/YOLOV10N.md)); YOLO11n has only a one-epoch smoke test. Do not compare supervised test AP against zero-shot validation AP as if they shared a protocol.

## Repository layout

```text
DL2026-11-6/
├── README.md, PROJECT_GUIDE.md, DATA.md
├── aquadet/                  prepare, train, evaluate, benchmark, compare CLI
├── configs/base.yaml         shared protocol
├── configs/models/           yolov8n.yaml, yolo11n.yaml, yolov10n.yaml
├── docs/open_vocab/          YOLOE prompt study and recorded validation results
├── docs/yolo/                supplementary benchmark instructions and results
├── scripts/open_vocab/       text-prompted YOLOE runner and versioned prompts
├── scripts/data/             source audit and derived COCO conversion/validation
├── reports/                  committed run summaries and leaderboard
├── tests/                    benchmark and prompt-definition unit tests
└── requirements-yolo.txt     pinned benchmark dependencies
```

`dataset/` (benchmark source), `data/` (local export and derived views), `weights/`, `runs/`, `results/` and `.venv/` are local/ignored. Do not commit images, labels, credentials or downloaded weights. Run all commands from the repository root.

## Primary experiment: names, synonyms and descriptions

Reproduce the [YOLOE-26s prompt study](docs/open_vocab/YOLOE.md) using its separately documented environment (`ultralytics==8.4.172`, `pycocotools==2.0.11`, and the pinned Ultralytics/CLIP dependency). Download the trusted text-promptable checkpoint and MobileCLIP2 text encoder as described there; do not use the `-pf` checkpoint. The **YOLO benchmark environment below uses a different Ultralytics version** and should be kept separate. The seven prompt strings in each set are versioned in [`aquarium_prompts.json`](scripts/open_vocab/aquarium_prompts.json). Only jellyfish and starfish change in the synonym variant.

With the original export at `data/aquarium.v2-release.yolov8/` and derived `data/aquarium-v2-coco/valid.json` ready (see [DATA.md](DATA.md)), run from the repository root in the YOLOE environment:

```bash
python scripts/open_vocab/run_yoloe_baseline.py --source data/aquarium.v2-release.yolov8 --prompt-set bare --output results/yoloe_valid_bare_rerun
python scripts/open_vocab/run_yoloe_baseline.py --source data/aquarium.v2-release.yolov8 --prompt-set synonym --output results/yoloe_valid_synonym_rerun
python scripts/open_vocab/run_yoloe_baseline.py --source data/aquarium.v2-release.yolov8 --prompt-set description --output results/yoloe_valid_description_rerun
```

Use `--device cuda:0` for the GPU setup documented in [YOLOE.md](docs/open_vocab/YOLOE.md). Each run records its exact prompts, predicted boxes, per-class and overall COCO AP, and runtime in a **new** ignored output directory. All three sets use the same validation images and ground truth. The locally measured full-validation CPU study reports overall AP **0.13109 / 0.12771 / 0.05759** for bare/synonym/description respectively; per-class effects and the separately documented historical GPU study are in [YOLOE.md](docs/open_vocab/YOLOE.md). Generated artifacts are ignored by Git; rerun and verify before submission. The script refuses test evaluation to prevent accidental test tuning.

## Supplementary closed-set YOLO benchmark

Python 3.12 is recommended. With `uv`:

```bash
uv venv .venv --python 3.12 --managed-python
uv pip install --python .venv/bin/python -r requirements-yolo.txt
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m aquadet env
python -m unittest discover -s tests
```

For the full **closed-set** setup, supported overrides, hardware/latency rules and evaluation protocol, read [docs/yolo/README.md](docs/yolo/README.md). If you do not use `uv`, create a Python 3.12 venv, install the appropriate PyTorch build for your hardware and then `pip install -r requirements-yolo.txt`.

1. Obtain the unmodified [Roboflow 100 Aquarium v2 YOLOv8-format **annotation export**](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2/download/yolov8) and put its `data.yaml`, `train/`, `valid/` and `test/` at `dataset/aquarium.v2-release.yolov8/` (or pass `--source PATH` / set `AQUADET_SOURCE`). The YOLOv8 **export format** is not a restriction on which of the three detectors can be trained. The published 448/127/63 image splits are preserved.
2. Prepare derived data, then run a smoke test before committing to a full run:

```bash
python -m aquadet prepare
python -m aquadet run --model yolo11n --smoke
python -m aquadet run --model yolov8n      # or yolo11n / yolov10n; 100 epochs each
python -m aquadet compare                  # generates reports/benchmark.{md,csv}
```

`run` trains, scores validation and test with both Ultralytics and pycocotools, benchmarks latency, and writes `reports/runs/<run>.json`. Use validation for model selection and **do not tune on test**. YOLOv8n's recorded test COCO AP is **47.6** (AP50 **76.1**), 3.0M parameters and 4.4 ms/image batch-1 inference on an RTX 3060; see [YOLOV8N.md](docs/yolo/YOLOV8N.md) for the exact protocol and limitations. YOLOv10n results are also recorded in [YOLOV10N.md](docs/yolo/YOLOV10N.md); YOLO11n's full run remains pending.

The source audit flags two zero-area test shark annotations; `prepare` omits only those records in the derived YOLO and COCO ground truth, without changing the original export. See [DATA.md](DATA.md). `reports/benchmark.md` currently contains the completed YOLOv8n and YOLOv10n runs; compare latency only on the same GPU under the same conditions.
