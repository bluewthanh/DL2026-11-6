# Setup 2: YOLOE-26s prompt study, independent GPU reproduction

Runs the three README commands (`scripts/open_vocab/run_yoloe_baseline.py --prompt-set {bare,synonym,description}`) on this machine and compares the results with the CPU table recorded in [`docs/open_vocab/YOLOE.md`](../../docs/open_vocab/YOLOE.md).

## Environment and inputs

- Machine: Linux, NVIDIA RTX 3060 12 GB (shared with another service), `--device cuda:0`, `3.12.12`, torch 2.14.1+cu130, ultralytics 8.4.172, Ultralytics/CLIP @ a13192f.
- Checkpoint `yoloe-26s-seg.pt` SHA-256 `48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654`; text encoder `mobileclip2_b.ts` SHA-256 `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`. Both match the hashes documented in YOLOE.md.
- Ground truth `valid.json` SHA-256 `684400c8936990f9fa20e41d8c69e5aeb8bb45e011e2b474e1fbc954b5be5702` (127 images, 909 boxes); images from `dataset/aquarium.v2-release.yolov8` (the unmodified export).
- Inference: {"imgsz": 640, "conf": 0.001, "iou": 0.7, "max_det": 300, "agnostic_nms": false, "batch": 1, "half": false, "augment": false, "task": "segment; bbox predictions only"}.
- Commands, run from the repository root with `YOLO_CONFIG_DIR=$PWD/.ultralytics`:

```bash
.venv-yoloe/bin/python scripts/open_vocab/run_yoloe_baseline.py --source dataset/aquarium.v2-release.yolov8 --prompt-set bare --device cuda:0 --output results/yoloe26s_valid_bare_gpu_repro
.venv-yoloe/bin/python scripts/open_vocab/run_yoloe_baseline.py --source dataset/aquarium.v2-release.yolov8 --prompt-set synonym --device cuda:0 --output results/yoloe26s_valid_synonym_gpu_repro
.venv-yoloe/bin/python scripts/open_vocab/run_yoloe_baseline.py --source dataset/aquarium.v2-release.yolov8 --prompt-set description --device cuda:0 --output results/yoloe26s_valid_description_gpu_repro
```

## Consistency checks across the three runs

- same image ids: **pass**
- same coco sha256: **pass**
- same checkpoint sha256: **pass**
- same text encoder sha256: **pass**
- same inference settings: **pass**
- same prompt file sha256: **pass**

## Results: documented CPU vs this GPU reproduction (AP ×100, IoU 0.50:0.95 unless AP50)

| Metric | Bare (doc / repro) | Synonym (doc / repro) | Description (doc / repro) |
|---|---:|---:|---:|
| AP | 13.11 / 13.11 | 12.77 / 12.77 | 5.76 / 5.77 |
| AP50 | 22.97 / 22.98 | 21.20 / 21.20 | 8.30 / 8.31 |
| fish | 19.61 / 19.59 | 19.85 / 19.82 | 1.60 / 1.62 |
| jellyfish | 12.85 / 12.84 | 2.98 / 2.98 | 3.05 / 3.06 |
| penguin | 11.18 / 11.19 | 11.21 / 11.23 | 3.31 / 3.32 |
| puffin | 6.21 / 6.22 | 6.06 / 6.09 | 0.08 / 0.08 |
| shark | 5.41 / 5.41 | 5.41 / 5.41 | 2.52 / 2.52 |
| starfish | 17.74 / 17.75 | 25.42 / 25.42 | 27.19 / 27.21 |
| stingray | 18.75 / 18.75 | 18.46 / 18.46 | 2.55 / 2.55 |

Largest absolute difference: **0.034 AP points** (×100), for synonym fish. The ranking and every per-class direction of change are the same. The small differences most likely come from the different hardware and software (CPU vs GPU; Windows, Python 3.14 and torch 2.14.0 vs Linux, Python 3.12 and torch 2.14.1). This was not isolated further. Saved detections: bare 16563, synonym 15777, description 19861.

## Wording effects (this reproduction, change vs bare, AP ×100)

| Class | Synonym − bare | Description − bare | Prompt change |
|---|---:|---:|---|
| fish | +0.23 | -17.98 | unchanged in synonym set |
| jellyfish | -9.87 | -9.78 | `jellyfish` → `sea jelly` |
| penguin | +0.03 | -7.88 | unchanged in synonym set |
| puffin | -0.13 | -6.13 | unchanged in synonym set |
| shark | +0.00 | -2.89 | unchanged in synonym set |
| starfish | +7.67 | +9.46 | `starfish` → `sea star` |
| stingray | -0.29 | -16.20 | unchanged in synonym set |
| **overall** | -0.34 | -7.34 | |

- The synonym set changes only two prompts, so only jellyfish and starfish can differ for real. The other five prompts are identical, and their AP moves by at most 0.29 points. Their detections still change: for example, fish detections rise from 8,214 with bare names to 9,096 with synonyms. This is plausibly because the prompts compete for the same boxes, and because of the per-image caps (`max_det` 300 at inference, top 100 in COCOeval). The mechanism was not isolated. `sea star` raises starfish AP by 7.7 points, while `sea jelly` lowers jellyfish AP by 9.9 points.
- Descriptions lower AP for six of the seven classes and raise only starfish (+9.5).
- Validation only. The runner refuses test inference, and nothing was tuned on test.
- GPU wall time per image (includes Python overhead; not a latency benchmark): bare 62.6 ms, synonym 34.7 ms, description 35.4 ms. The first run includes CUDA warm-up.

Every output of each run is stored in `runs/<prompt set>/`: `config.json`, `metrics.json`, `runtime.json` (per-image timing), `predictions.json.gz` (all detections at conf ≥ 0.001, gzip-compressed COCO results format) and `console.log`. Read them without rerunning, for example with `python -c "import gzip, json; print(len(json.load(gzip.open('reports/02_setup2_prompt_study/runs/bare/predictions.json.gz'))))"`. The commands above write the uncompressed originals to the Git-ignored `results/` folder.
