# YOLOE-26s pretrained validation baseline

**Status:** Full 127-image validation run completed; no training, no test inference/tuning. Dataset splits and derived COCO files were not changed. Dataset provenance and the two omitted, invalid **test-only** source boxes are documented in `DATA.md`.

## Verified API and weights

Official [Ultralytics YOLOE documentation](https://docs.ultralytics.com/models/yoloe/) documents `from ultralytics import YOLOE`, `YOLOE("yoloe-26s-seg.pt")`, `set_classes([...])`, and the text-promptable (not `-pf`) YOLOE-26s checkpoint. Installed `ultralytics==8.4.172`; inspected its `YOLOE.set_classes` and `YOLOE.predict` APIs and checked the loaded checkpoint is `YOLOESegModel`, `text_model=mobileclip2:b`. Segmentation pretrained checkpoint is used for **bbox detection**; masks are not evaluated. It uses the MobileCLIP2 text encoder, loaded once to generate prompt embeddings. Its tokenizer needs [Ultralytics/CLIP](https://github.com/ultralytics/CLIP) (installed from commit `a13192f8cb767260d7dfd98c843b0716593169e7`). Checkpoint and text encoder are local untracked assets, not fine-tuned:

- [yoloe-26s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt), SHA-256 `48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654`.
- [mobileclip2_b.ts](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts), SHA-256 `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`.

Reproduction from the project root (use a trusted source for PyTorch pickle checkpoints):

```bash
python -m pip install ultralytics==8.4.172 pycocotools==2.0.11
python -m pip install 'git+https://github.com/ultralytics/CLIP.git@a13192f8cb767260d7dfd98c843b0716593169e7'
# Download the two URLs above to weights/yoloe-26s-seg.pt and
# weights/mobileclip2_b.ts respectively (Ultralytics weights directory).
python scripts/validate_coco.py --coco-dir data/aquarium-v2-coco
python scripts/run_yoloe_baseline.py --output results/yoloe26s_valid_bare_final
# Optional diagnostic only: --limit 1 --output results/new_smoke_path
```

The script refuses `test.json`, validates the `valid.json` counts/categories, sorts validation image IDs, maps prompt index 0–6 to COCO ID 1–7, converts pixel xyxy boxes to COCO pixel xywh, and evaluates all 127 images with `pycocotools` COCOeval bbox. It saves `config.json` (prompt mapping/vocabularies, checkpoint and input hashes, environment/settings), `predictions.json` (COCO image/category/bbox/**score**), `runtime.json` (per-image timings), and `metrics.json` to a new ignored output folder; it never writes into dataset folders. Default inference: `imgsz=640`, `conf=0.001` (low fixed cutoff to preserve AP ranking), NMS IoU 0.7, `agnostic_nms=False`, `max_det=300`, batch 1, no augmentation, CPU float32. COCOeval uses IoU 0.50:0.95, area all, and default maxDets 100; model max_det and evaluator maxDets are distinct. Per-image wall time includes preprocessing, inference, postprocessing and Python overhead; not GPU latency. CPU is slow and timings are not comparable to a T4 run.

The baseline prompt list and **separate prompt-study** prompt study vocabularies are versioned in `scripts/aquarium_prompts.json`. Only `sea jelly` for jellyfish and `sea star` for starfish are defensible synonyms; the other five entries in the synonym variant remain **unchanged**. Descriptions are proposed visual phrases, not synonyms. Category IDs are the actual derived COCO IDs, not the order in the prompt table. No prompt variant was used for these baseline metrics.

## Validation results (full 127-image bare-name run; `results/yoloe26s_valid_bare_final/`)

- **AP@[0.50:0.95]:** 0.1311; **AP50:** 0.2297; 16,586 saved detections; CPU mean wall time **877 ms/image** (Windows 11, Python 3.14.0, torch 2.14.0+cpu, no CUDA).

| Class | AP@[0.50:0.95] |
|---|---:|
| fish | 0.1961 |
| jellyfish | 0.1285 |
| penguin | 0.1118 |
| puffin | 0.0621 |
| shark | 0.0541 |
| starfish | 0.1774 |
| stingray | 0.1875 |

These are **validation-only** zero-shot results, not test performance. No thresholds were tuned on the test split. Follow-up visual checking of GT and predictions on a few validation images is recommended before comparing additional models or prompts. The original dataset still has two invalid test shark annotations; see `DATA.md` before any test evaluation.
 
 
 ## Prompt study: full validation on RTX 3060 Laptop GPU

Evaluated all three prompt sets on the same 127 validation images and 909
ground-truth boxes, using pretrained YOLOE-26s without training.
These GPU results are separate from the earlier CPU baseline.

Environment: NVIDIA GeForce RTX 3060 Laptop GPU, torch 2.14.1+cu126,
ultralytics 8.4.172. All runs used device cuda:0 and identical defaults:
imgsz=640, conf=0.001, NMS IoU=0.7, max_det=300, batch=1, float32,
no augmentation. Evaluation used COCO bbox AP, maxDets=100.

### Results

AP values are on a 0–1 scale. Class AP uses IoU 0.50:0.95.

| Metric / class | Bare | Synonym | Description |
|---|---:|---:|---:|
| Overall AP | 0.1313 | 0.1278 | 0.0576 |
| AP50 | 0.2301 | 0.2121 | 0.0829 |
| fish | 0.1960 | 0.1983 | 0.0161 |
| jellyfish | 0.1294 | 0.0298 | 0.0305 |
| penguin | 0.1119 | 0.1123 | 0.0331 |
| puffin | 0.0622 | 0.0609 | 0.0008 |
| shark | 0.0543 | 0.0543 | 0.0252 |
| starfish | 0.1775 | 0.2542 | 0.2720 |
| stingray | 0.1876 | 0.1847 | 0.0256 |

The synonym set changes only jellyfish to "sea jelly" and starfish to
"sea star"; the other five prompts remain unchanged. Descriptions are
visual phrases, not synonyms.

Bare names achieved the highest overall AP. Relative to bare names,
synonyms reduced overall AP by 0.0035: starfish improved while jellyfish
declined substantially. Descriptions reduced overall AP by 0.0737,
although starfish achieved its highest AP with a description.
These findings apply to the tested vocabularies and validation split;
no test inference or test tuning was performed.

### Reproduction from project root

PowerShell commands below assume the export is at
./aquarium.v2-release.yolov8 and derived COCO JSON is already validated.
Each output directory must be new.

```powershell
python scripts/run_yoloe_baseline.py --checkpoint .\weights\yoloe-26s-seg.pt --coco .\data\aquarium-v2-coco\valid.json --source .\aquarium.v2-release.yolov8 --prompt-set bare --device cuda:0 --output .\results\yoloe26s_valid_bare_gpu
python scripts/run_yoloe_baseline.py --checkpoint .\weights\yoloe-26s-seg.pt --coco .\data\aquarium-v2-coco\valid.json --source .\aquarium.v2-release.yolov8 --prompt-set synonym --device cuda:0 --output .\results\yoloe26s_valid_synonym_gpu
python scripts/run_yoloe_baseline.py --checkpoint .\weights\yoloe-26s-seg.pt --coco .\data\aquarium-v2-coco\valid.json --source .\aquarium.v2-release.yolov8 --prompt-set description --device cuda:0 --output .\results\yoloe26s_valid_description_gpu
```

Each run saves config.json, metrics.json, predictions.json and runtime.json
locally. Dataset files, weights and generated result JSON are excluded
from the commit.