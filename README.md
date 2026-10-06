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
| 7 | Trần Khoa Nam | 2410702 |

**Topic:** detect objects using text descriptions rather than only fixed classes, and measure how names, synonyms and descriptive prompts affect detection. The primary experiment uses text-prompted **YOLOE-26s** on Roboflow 100 Aquarium v2. **YOLOv8n is a supervised closed-set baseline**, not a text-prompted model. See [DATA.md](DATA.md) for dataset provenance and [reports/README.md](reports/README.md) for the three experiments and saved results.

**Recorded status:** the saved GPU validation prompt-study runs and predictions are in [reports/02_setup2_prompt_study/](reports/02_setup2_prompt_study/); a second member checked saved artifacts in [reports/VERIFICATION.md](reports/VERIFICATION.md), but did not independently rerun model inference. A separate local CPU study is in [YOLOE.md](docs/open_vocab/YOLOE.md). There is no reported prompt-based test evaluation. YOLOv8n completed supervised runs; its same-validation-split comparison with YOLOE, the data-efficiency study and qualitative analysis are in [reports/README.md](reports/README.md). Do not compare supervised test AP against zero-shot validation AP as if they shared a protocol.

## Repository layout

```text
DL2026-11-6/
├── README.md, DATA.md
├── aquadet/                  prepare, train, evaluate, benchmark, compare CLI
├── configs/base.yaml         shared protocol
├── configs/models/           yolov8n.yaml
├── docs/open_vocab/          YOLOE prompt study and recorded validation results
├── docs/yolo/                YOLOv8n baseline instructions and results
├── scripts/open_vocab/       text-prompted YOLOE runner and versioned prompts
├── scripts/data/             source audit and derived COCO conversion/validation
├── reports/                  run summaries and report results by section (see reports/README.md)
├── tests/                    benchmark and prompt-definition unit tests
└── requirements-yolo.txt     pinned benchmark dependencies
```

`dataset/` (benchmark source), `data/` (local export and derived views), `weights/`, `runs/`, `results/`, `.venv/` and `.venv-yoloe/` are local/ignored. Do not commit source images/labels, credentials or downloaded weights. A **separate attributed annotations-only derived COCO ZIP** is available from the link below. Run all commands from the repository root.

## Primary experiment: names, synonyms and descriptions

Reproduce the [YOLOE-26s prompt study](docs/open_vocab/YOLOE.md) in a **separate** Python 3.12 environment. For the exact GPU environment used for the saved results, see [reports/00_setup/README.md](reports/00_setup/README.md) (Ultralytics 8.4.172, pycocotools 2.0.11, pinned Ultralytics/CLIP). One installation route (bash; install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) first):

```bash
uv venv .venv-yoloe --python 3.12 --managed-python
uv pip install --python .venv-yoloe/bin/python torch==2.14.1 torchvision==0.29.1 ultralytics==8.4.172 pycocotools==2.0.11 'git+https://github.com/ultralytics/CLIP.git@a13192f8cb767260d7dfd98c843b0716593169e7'
source .venv-yoloe/bin/activate
```

On Windows, replace `.venv-yoloe/bin/python` above with `.venv-yoloe/Scripts/python.exe`, and activate with `.venv-yoloe\Scripts\Activate.ps1` in PowerShell (or `.venv-yoloe/Scripts/activate` in Git Bash). Install a CUDA-enabled PyTorch build compatible with your hardware if using GPU; the saved run's full environment is in `reports/00_setup/`. Download the text-promptable `yoloe-26s-seg.pt` and MobileCLIP2 encoder to `weights/` from the trusted links in [YOLOE.md](docs/open_vocab/YOLOE.md); check both SHA-256 hashes. Do **not** use the `-pf` checkpoint. The YOLO benchmark environment below uses a different Ultralytics version. Only jellyfish and starfish change in the synonym variant of [`aquarium_prompts.json`](scripts/open_vocab/aquarium_prompts.json).

Put the original Aquarium v2 YOLOv8 export at `data/aquarium.v2-release.yolov8/` (see [DATA.md](DATA.md)). The **derived annotations-only COCO ZIP** is [downloadable separately](https://github.com/bluewthanh/DL2026-11-6/releases/download/aquarium-v2-derived-coco-v1/aquarium-v2-derived-coco.zip); it does not contain images. Extract it so `train.json`, `valid.json` and `test.json` are under `data/aquarium-v2-coco/`, or regenerate the **same** derived ground truth from the original export (install `Pillow` and `PyYAML` in the conversion environment):

```bash
python scripts/data/yolo_to_coco.py --source data/aquarium.v2-release.yolov8 --invalid-policy omit-known-zero-boxes
python scripts/data/validate_coco.py --coco-dir data/aquarium-v2-coco
```

The separate audit command in [DATA.md](DATA.md) exits **nonzero** on the two known zero-area test shark annotations; the explicit conversion policy above allows only those records to be omitted from *derived* ground truth, not the original export. If you extract the ZIP, **do not run the conversion again into that nonempty folder**. Validate it instead. Every COCO image path is relative to the original export root; keep the original images available for inference.

For direct YOLOE runs, set `YOLO_CONFIG_DIR` to the **absolute** path of this repository's `.ultralytics` folder (and put `mobileclip2_b.ts` in `weights/`); otherwise Ultralytics may look for the encoder in a machine-wide directory. Check that the directory is writable; if Ultralytics falls back to a different location, configure its `weights_dir` to the repository's `weights/`. From the repository root, for example, on bash: `export YOLO_CONFIG_DIR="$PWD/.ultralytics"` (PowerShell: `$env:YOLO_CONFIG_DIR = (Join-Path (Get-Location) '.ultralytics')`). Then, with the original export and derived `valid.json` ready, run in the YOLOE environment:

```bash
python scripts/open_vocab/run_yoloe_baseline.py --source data/aquarium.v2-release.yolov8 --prompt-set bare --output results/yoloe_valid_bare_rerun
python scripts/open_vocab/run_yoloe_baseline.py --source data/aquarium.v2-release.yolov8 --prompt-set synonym --output results/yoloe_valid_synonym_rerun
python scripts/open_vocab/run_yoloe_baseline.py --source data/aquarium.v2-release.yolov8 --prompt-set description --output results/yoloe_valid_description_rerun
```

Use `--device cuda:0` for the GPU setup documented in [YOLOE.md](docs/open_vocab/YOLOE.md), or omit it for CPU. For a quick path/weights/inference check, add `--limit 1` and choose another new `--output` directory; **one-image AP is not the report result**. Each full run records its exact prompts, predicted boxes, per-class and overall COCO AP, and runtime in a **new** ignored output directory. All three sets use the same validation images and ground truth. The locally measured full-validation CPU study reports overall AP **0.13109 / 0.12771 / 0.05759** for bare/synonym/description respectively; per-class effects and the separately documented historical GPU study are in [YOLOE.md](docs/open_vocab/YOLOE.md). Generated local artifacts are ignored by Git; compressed predictions for the saved GPU reproduction are tracked in `reports/02_setup2_prompt_study/` and were checked as saved artifacts, not by a second independent inference rerun. The script refuses test evaluation to prevent accidental test tuning.

## Supplementary closed-set YOLOv8n baseline

Python 3.12 is recommended. With `uv`:

```bash
uv venv .venv --python 3.12 --managed-python
uv pip install --python .venv/bin/python -r requirements-yolo.txt
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m aquadet env
python -m unittest discover -s tests
```

For the full **closed-set** setup, supported overrides, hardware/latency rules and evaluation protocol, read [docs/yolo/README.md](docs/yolo/README.md). If you do not use `uv`, create a Python 3.12 venv, install the appropriate PyTorch build for your hardware and then `pip install -r requirements-yolo.txt`. The saved GPU experiments were run with separate Python 3.12 environments (see `reports/00_setup/`). A [fresh-environment Windows CPU walkthrough](reports/REPRODUCTION_CHECK.md) verified full YOLOE validation runs, a one-epoch YOLOv8n training/evaluation smoke pipeline, data conversion, unit tests and demo; it did **not** retrain YOLOv8n for 100 epochs or reproduce GPU latency.

1. Obtain the unmodified [Roboflow 100 Aquarium v2 YOLOv8-format **annotation export**](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2/download/yolov8) and put its `data.yaml`, `train/`, `valid/` and `test/` at `dataset/aquarium.v2-release.yolov8/` (or pass `--source PATH` / set `AQUADET_SOURCE`). For one shared export across both model environments, put it at `data/aquarium.v2-release.yolov8/` instead and pass `--source data/aquarium.v2-release.yolov8` to each `aquadet prepare/run` command. The YOLOv8 **export format** is not a restriction on which of the three detectors can be trained. The published 448/127/63 image splits are preserved.
2. Prepare derived data, then run a smoke test before committing to a full run:

```bash
python -m aquadet prepare
python -m aquadet run --model yolov8n --smoke
python -m aquadet run --model yolov8n      # 100 epochs
python -m aquadet compare                  # generates reports/benchmark.{md,csv}
```

`run` trains, scores validation and test with both Ultralytics and pycocotools, benchmarks latency, and writes `reports/runs/<run>.json`. Use validation for model selection and **do not tune on test**. YOLOv8n's recorded test COCO AP is **47.6** (AP50 **76.1**), 3.0M parameters and 4.4 ms/image batch-1 inference on an RTX 3060; see [YOLOV8N.md](docs/yolo/YOLOV8N.md) for the exact protocol and limitations.

The source audit flags two zero-area test shark annotations; `prepare` omits only those records in the derived YOLO and COCO ground truth, without changing the original export. See [DATA.md](DATA.md). `reports/benchmark.md` contains the completed YOLOv8n run; compare latency only on the same GPU under the same conditions. For saved **validation** YOLOv8n-vs-YOLOE numbers and all three setups, use [`reports/README.md`](reports/README.md) and each setup README. To demonstrate inference on an image using your own comma-separated prompts, run `python scripts/demo/detect.py --prompts "fish, sea star" --image path/to/image.jpg` in the YOLOE environment; see [`reports/06_demo/README.md`](reports/06_demo/README.md).
