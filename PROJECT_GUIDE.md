# Project plan: Aquarium v2 closed-set YOLO benchmark

This document describes the **current repository scope**, based on the benchmark added in commit `a63463c`. The original course topic was open-vocabulary detection with text prompts; YOLOv8n, YOLO11n and YOLOv10n are **closed-set supervised detectors**, not substitutes for a prompt study. Confirm with the teaching staff whether a change of project scope meets the course requirements before submitting. Do not claim that text-prompt experiments were conducted by this benchmark.

## Topic requirements (course brief)

- Detect objects using text descriptions instead of relying only on a fixed set of predefined classes.
- Investigate how different object names, synonyms, and descriptive prompts affect detection performance.

**Status:** The current three-model closed-set YOLO benchmark does not satisfy either requirement by itself: its predictions use fixed trained classes, and it does not accept or compare inference-time text prompts. How to address this gap is still to be discussed; no prompt-based results are claimed here.

## Objective and status

Fine-tune three COCO-pretrained nano models on the *same* RF100 Aquarium v2 training split, then compare validation/test accuracy, model size and batch-1 inference latency under a shared protocol:

| Model | Config | Full experiment status |
|---|---|---|
| YOLOv8n | `configs/models/yolov8n.yaml` | Complete: 100 epochs, seed 0; see [results](docs/yolo/YOLOV8N.md) |
| YOLO11n | `configs/models/yolo11n.yaml` | One-epoch smoke pipeline passed; full run pending |
| YOLOv10n | `configs/models/yolov10n.yaml` | One-epoch smoke pipeline passed; full run pending |

The [generated leaderboard](reports/benchmark.md) currently contains **only YOLOv8n**. There is no three-model ranking yet. Each model is initialized from its own COCO-pretrained checkpoint and fine-tuned on Aquarium training labels. Predictions are restricted to the dataset's seven learned classes; no inference-time text prompts or zero-shot claims apply.

## Dataset and reproducibility

Use the published Roboflow 100 Aquarium v2 splits: 448 train, 127 validation and 63 test images. The source is the YOLOv8-*format* export, not a requirement to use the YOLOv8 detector. [DATA.md](DATA.md) covers the download, class mapping, audit and two zero-area test shark boxes. `python -m aquadet prepare` builds a derived YOLO view and COCO ground truth that omit exactly those two invalid test annotations; the original export is never modified. Keep the policy identical for all three models.

Run from the repository root, with the pinned [requirements-yolo.txt](requirements-yolo.txt) environment. Full setup and run commands are in [docs/yolo/README.md](docs/yolo/README.md):

```bash
python -m unittest discover -s tests
python -m aquadet prepare
python -m aquadet run --model yolov8n
python -m aquadet run --model yolo11n
python -m aquadet run --model yolov10n
python -m aquadet compare
```

Do not rerun the completed YOLOv8n experiment solely to populate the leaderboard: its tracked [summary](reports/runs/yolov8n_e100_s0.json) is already included. `run` trains, evaluates val and test, benchmarks and exports a summary. Use `--smoke` for a one-epoch pipeline check; smoke results are not reportable as full results. Never commit `dataset/`, `data/`, `weights/` or `runs/`.

## Shared evaluation protocol

The single source of settings is [configs/base.yaml](configs/base.yaml); model configs select only the pretrained checkpoint. Train each model for 100 epochs, seed 0, 640 px and batch 16 with AdamW; choose `best.pt` using validation fitness, not test performance. For accuracy, evaluate on both the derived COCO ground truth with pycocotools (**primary COCO AP@[.50:.95], AP50 and per-class AP**) and the derived YOLO view with Ultralytics validation (cross-check). The protocol hash and environment are recorded in each run summary, and custom overrides are excluded from the default comparison. See [docs/yolo/README.md](docs/yolo/README.md) for all settings.

Benchmark size, GFLOPs and FP32/FP16 latency with warmed-up, in-memory, batch-1 640 px images. Speed numbers from different GPUs or system loads are **not directly comparable**; re-benchmark each finished model on the same machine. Inspect predicted/ground-truth overlays and the difference between the two AP scorers. Tune and select settings on validation only; the pipeline scores test after training, so do not alter the protocol in response to test numbers. If the shared protocol changes, rerun all models and document why.

## Remaining work and reporting

1. Confirm the change from the original text-prompted course topic with staff and confirm submission format, group roster and deadlines.
2. Finish YOLO11n and YOLOv10n full runs; commit only their run summaries and regenerate the leaderboard. Do not fabricate pending measurements.
3. Check protocol hashes, dataset policy, latency hardware and visual/metric sanity checks before interpreting a three-model comparison.
4. Report test-label limitations and the one-seed/single-dataset limitation. Include reproducible commands, actual measured results, per-class analysis and a truthful Member Contribution table in any submission.
