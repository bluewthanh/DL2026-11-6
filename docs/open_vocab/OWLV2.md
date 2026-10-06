# OWLv2 validation workstream (3b)

**Status: 1-, 10-, and full 127-image validation runs completed locally on CPU.** Results below are unreviewed; an independent team-member review is still required before including numbers in the team report. No test-set inference was performed.

## Model and protocol

- Hugging Face [`google/owlv2-base-patch16-ensemble`](https://huggingface.co/google/owlv2-base-patch16-ensemble), revision **`cfd3195ba4ea9592eec887ded089f4c08eff231d`**, pretrained `model.safetensors`, not fine-tuned. The revision is passed to both processor and model loaders, so they use the same frozen model snapshot. Download/cache contents stay untracked. Source: [official model card](https://huggingface.co/google/owlv2-base-patch16-ensemble) and [Transformers OWLv2 API](https://huggingface.co/docs/transformers/model_doc/owlv2).
- `scripts/open_vocab/run_owlv2.py` queries the seven **bare class names** in the order and ID mapping checked against `scripts/open_vocab/aquarium_prompts.json` and validated COCO categories. Hugging Face query labels 0–6 map to COCO category IDs 1–7. One best query per model image patch is returned by HF text-query post-processing (no additional NMS). The default score floor is 0.001; boxes are rescaled to the **original** height/width, clamped to image bounds, converted from xyxy to pixel xywh, sorted by score and limited to 300/image. COCOeval bbox uses its default maxDets **100**, IoU 0.50:0.95, area all. This is the same ID set, 127 validation images and COCO evaluator as YOLOE, though each model has its own architecture and postprocessing.
- `--limit` evaluates the first N **sorted image IDs**, not a random sample. Subset AP is diagnostic, **not** the full-validation AP. The script refuses `test.json` by requiring `valid.json` and validating the expected **127 images / 909 annotations**; it never writes into `data/`. Each run needs a new ignored `results/` directory and saves `config.json`, `predictions.json`, `metrics.json` and `runtime.json`. Wall time per image includes image loading, preprocessing, inference, postprocessing and Python overhead; it is **not** a comparable GPU latency benchmark. Use workstream 4's single-GPU protocol for model-to-model speed comparisons.

## Setup and exact run commands (pending execution)

This local run used Windows, Python 3.13, torch==2.6.0+cpu, transformers==4.48.3, pycocotools==2.0.11, Pillow==11.1.0, PyYAML==6.0.2 and safetensors==0.5.2. Install a compatible PyTorch CPU or CUDA build following [official install instructions](https://pytorch.org/get-started/locally/). Run from the repository root. Hugging Face access is needed on first download; do not load untrusted checkpoints. The pinned revision's `model.safetensors` downloaded for this run has SHA-256 `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`.

```bash
py -3.13 -m pip install 'torch==2.6.0' 'transformers==4.48.3' 'pycocotools==2.0.11' 'Pillow==11.1.0' 'PyYAML==6.0.2' 'safetensors==0.5.2'
# The YOLOv8 export was supplied at the repository root (train/, valid/, test/, data.yaml).
# Conversion requires output OUTSIDE the export root. Only after success, move to the ignored data/ directory:
py -3.13 scripts/data/yolo_to_coco.py --source . --output ../owlv2-derived-coco-temp --invalid-policy omit-known-zero-boxes
mkdir -p data
mv ../owlv2-derived-coco-temp data/aquarium-v2-coco
py -3.13 scripts/data/validate_coco.py --coco-dir data/aquarium-v2-coco
py -3.13 -m unittest discover -s scripts/open_vocab -p 'test_run_owlv2.py'
# Download the model card's revision cfd3195ba4ea9592eec887ded089f4c08eff231d
# to weights/owlv2-base-patch16-ensemble (model.safetensors and processor/tokenizer files).
# Verify model.safetensors SHA-256 above. The snapshot is ignored by Git.
py -3.13 scripts/open_vocab/run_owlv2.py --source . --model-dir weights/owlv2-base-patch16-ensemble --limit 1 --output results/owlv2_valid_1
py -3.13 scripts/open_vocab/run_owlv2.py --source . --model-dir weights/owlv2-base-patch16-ensemble --limit 10 --output results/owlv2_valid_10
py -3.13 scripts/open_vocab/run_owlv2.py --source . --model-dir weights/owlv2-base-patch16-ensemble --output results/owlv2_valid_full
# Without --model-dir, the script downloads the pinned revision via Hugging Face.
# Use a NEW --output path for reruns. CUDA: --device cuda:0; record that environment separately.
```

## Measured validation results (CPU, bare names; not reviewed)

| Images (sorted COCO IDs) | AP@[0.50:0.95] | AP50 | Detections | Mean wall ms/image |
|---:|---:|---:|---:|---:|
| 1 (diagnostic) | 0.6227 | 1.0000 | 300 | 11407 |
| 10 (diagnostic) | 0.3480 | 0.5940 | 3000 | 12147 |
| **127 (full validation)** | **0.2794** | **0.5151** | **37815** | **11710** |

| Class | Full-validation AP@[0.50:0.95] |
|---|---:|
| fish | 0.3889 |
| jellyfish | 0.2718 |
| penguin | 0.2480 |
| puffin | 0.2002 |
| shark | 0.2138 |
| starfish | 0.4641 |
| stingray | 0.1692 |

The derived `valid.json` SHA-256 is `86dcce246be3801d36cf65200d10988add8aac46583190fd745cef43e06e075c`. For each run see `results/owlv2_valid_{1,10,full}/{config,metrics,predictions,runtime}.json` (ignored by Git). Each detection is scored; 300/image is the inference maximum, **not** the COCOeval maxDets (100). CPU wall time is **not comparable GPU latency**. In addition to the unit tests and schema checks, inspected the first image's GT and top predictions (correct fish labels; fish boxes align), plus local overlays for IDs 1, 50, 100, 127 under `results/owlv2_valid_full/inspection/`. Green = GT; red = top 30 predictions, including some clear extra background boxes due to the deliberately low confidence floor. More exhaustive visual checking and independent review are still needed before report use.

### Checklist

- [x] Implementation, deterministic class/box conversion, validation guardrails and isolated output paths
- [x] Unit tests (3 passing)
- [x] 1-image validation inference and visual inspection
- [x] 10-image validation inference
- [x] 127-image validation inference and per-class AP
- [ ] Independent model implementation review (another team member)

Do not copy generated predictions, checkpoints or dataset files into Git; do not evaluate on the test set before the team freezes the validation policy.
