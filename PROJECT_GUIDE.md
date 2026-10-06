# Project plan: Aquarium v2 text-prompted detection and YOLO controls

The lecturer's **open-vocabulary** topic is the primary goal. The closed-set benchmark added in commit `a63463c` is a supplementary comparison, **not** the answer to the topic on its own. Do not present YOLOv8n, YOLO11n or YOLOv10n as text-promptable detectors.

## Topic requirements (course brief)

- Detect objects using text descriptions instead of relying only on a fixed set of predefined classes.
- Investigate how different object names, synonyms, and descriptive prompts affect detection performance.

## Primary experiment: text prompts

Use pretrained **YOLOE-26s** (`scripts/open_vocab/run_yoloe_baseline.py`) with its text encoder at inference time. `set_classes(prompts)` turns a supplied vocabulary into detections without fine-tuning on Aquarium; see [the model and prompt-study documentation](docs/open_vocab/YOLOE.md). Compare the versioned [bare names, synonym variant and visual descriptions](scripts/open_vocab/aquarium_prompts.json) on the **same 127 validation images**, with the same seven COCO category IDs and the same image size, confidence cutoff and evaluator. Save each run's prompts, predictions, per-class AP, overall AP and environment separately. Only two categories have reliable alternative names (`jellyfish` → `sea jelly`, `starfish` → `sea star`); five remain unchanged in the synonym set. This is **not** a seven-class synonym-effect estimate. Descriptions change all seven prompts.

Recorded results in `docs/open_vocab/YOLOE.md`: a full CPU validation comparison (bare AP **0.13109**, synonym AP **0.12771**, description AP **0.05759**) and a previously documented GPU prompt study (0.1313 / 0.1278 / 0.0576). CPU `results/` files exist locally but are ignored by Git; historical GPU artifacts are unavailable here. Independently reproduce and inspect saved predictions before submitting; see that document for commands and per-class effects. These results are **validation-only**; do not claim a prompt-based test result. Prompt flexibility does not by itself prove successful detection of categories absent from pretraining.

## Supplementary objective and status

Fine-tune three COCO-pretrained nano models on the *same* RF100 Aquarium v2 training split, then compare validation/test accuracy, model size and batch-1 inference latency under a shared protocol:

| Model | Config | Full experiment status |
|---|---|---|
| YOLOv8n | `configs/models/yolov8n.yaml` | Complete: 100 epochs, seed 0; see [results](docs/yolo/YOLOV8N.md) |
| YOLO11n | `configs/models/yolo11n.yaml` | One-epoch smoke pipeline passed; full run pending |
| YOLOv10n | `configs/models/yolov10n.yaml` | One-epoch smoke pipeline passed; full run pending |

The [closed-set leaderboard](reports/benchmark.md) currently contains **only YOLOv8n**. There is no three-model ranking yet. Each model is initialized from its own COCO-pretrained checkpoint and fine-tuned on Aquarium training labels. Predictions are restricted to the dataset's seven learned classes; no inference-time text prompts or zero-shot claims apply. The YOLOE validation-only zero-shot results and fine-tuned YOLO test scores are **not a fair head-to-head ranking**: splits and training regimes differ.

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

1. Independently rerun and verify the YOLOE name/synonym/description study (generated artifacts are ignored and historical GPU artifacts are unavailable here); inspect overlays and document errors and per-class effects. Never tune prompts on test.
2. If a held-out open-vocabulary test result is required, freeze prompts and evaluation settings on validation **first**, then implement/document an explicitly reviewed test protocol. The current YOLOE script intentionally refuses test inference.
3. Finish YOLO11n and YOLOv10n full runs if time/resources allow; commit only their summaries and regenerate the supplementary leaderboard. Do not fabricate pending measurements or imply supervised test AP is directly comparable to zero-shot validation AP.
4. Confirm report format, roster and deadlines with staff. Include reproducible commands, limitations (one dataset, two real synonyms, test-label defects, different training regimes), measured results and a truthful Member Contribution table.
