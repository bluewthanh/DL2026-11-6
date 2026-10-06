# Setup 1: YOLOv8n baseline vs text-prompted YOLOE-26s (main model)

## Comparison strategy

| | Baseline | Main method |
|---|---|---|
| Model | YOLOv8n, 3.01 M params | YOLOE-26s, 15.27 M params (segmentation checkpoint including the unused mask head; the MobileCLIP2 text encoder is separate) |
| How it gets the classes | Fine-tuned on the 448 labelled Aquarium training images (100 epochs, shared `configs/base.yaml`, seed 0), starting from COCO-pretrained weights | No Aquarium training. The 7 classes are given as text at inference time (`set_classes`) |
| Run | `runs/yolov8n_e100_s0` (`python -m aquadet run --model yolov8n`). Stored outputs: aquadet val/test evaluation and latency in `yolov8n_e100_s0_aquadet/`; training curves in `../03_setup3_data_efficiency/train_artifacts/yolov8n_e100_s0/`; val predictions in `../03_setup3_data_efficiency/eval/yolov8n_e100_s0/` | Predictions, metrics and configs in `../02_setup2_prompt_study/runs/{bare,synonym,description}/` |

**What is held equal:**

- The same 127 validation images and the same 909 ground-truth boxes (`data/aquarium-v2-coco/valid.json`).
- The same evaluator: pycocotools COCOeval bbox, IoU 0.50:0.95, maxDets 100.
- The same inference settings: imgsz 640, conf 0.001, NMS IoU 0.7, max_det 300, batch 1, FP32.

Every row below is recomputed from the saved predictions by one script, `scripts/analysis/build_tables.py`, so the scoring is identical across rows.

**What is deliberately not equal:**

- The training regime (supervised vs zero-shot) and the model size.
- Use of the validation split: YOLOv8n `best.pt` is selected on validation fitness, so its validation AP is optimistic. **`last.pt` is the fair row**, because it takes no input from validation and has no early stopping.
- YOLOE has no test-split result (its runner refuses test), so **no test numbers appear here**. YOLOv8n's test AP (47.6) must not be compared with YOLOE's validation AP.

## Results (validation, ×100)

| Model | Labelled Aquarium train images | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **YOLOv8n last.pt** (baseline) | 448 | **44.4** | **74.8** | 44.9 | 3.6 | 20.8 | 47.6 | 57.3 |
| YOLOv8n best.pt (val-selected, optimistic) | 448 | 45.4 | 75.8 | 46.0 | 3.4 | 20.2 | 48.5 | 57.9 |
| **YOLOE-26s, bare class names** (main) | 0 | **13.1** | **23.0** | 11.8 | 1.0 | 6.2 | 14.7 | 35.8 |
| YOLOE-26s, synonym set | 0 | 12.8 | 21.2 | 12.1 | 1.0 | 3.5 | 14.3 | 34.4 |
| YOLOE-26s, descriptions | 0 | 5.8 | 8.3 | 5.6 | 0.0 | 2.7 | 6.0 | 26.9 |

Seed spread for the baseline: three training seeds on the full training set give a `last.pt` AP of **44.7 ± 0.7**; see `../03_setup3_data_efficiency`. Machine-readable versions: `setup1_table.csv` and `scores.json`. Per-class results are in `../04_per_class_ap`.

## Interpretation

- **Size of the gap.** The supervised baseline reaches 3.4× the AP of the best text-prompted variant (44.4 vs 13.1). The gap is largest at strict localisation: AP75 is 44.9 vs 11.8, while AR100 is 57.3 vs 35.8. YOLOE often finds an object but mislabels or poorly fits it. At its F1-optimal display threshold, 291 of YOLOE bare's 1,123 boxes (26 %) sit on an object but carry the wrong class, against 22 of 814 for YOLOv8n best.pt (`../05_qualitative_overlays`).
- **Every class favours the baseline,** by 19 to 41 AP points for `last.pt` (`../04_per_class_ap`). Zero-shot YOLOE is weakest on shark (5.4) and puffin (6.2). At its display threshold, 41 of its boxes on sharks are labelled `fish` (there are 57 shark ground-truth boxes). Puffins are among the smallest objects relative to the image and appear in groups.
- **What the comparison does and does not show.** It does not show that YOLOE is a worse detector; the two are solving different problems. YOLOv8n cannot accept a new class without labelled data and retraining, whereas YOLOE does so from text alone (see the `rock` example in `../06_demo`). The supervised model is the ceiling that 448 labelled images buy. Setup 3 measures how quickly that ceiling is reached as labels are removed.
- **Small objects are hard for both.** APs is ≤ 3.6 for every model, but there are few small boxes in validation, so APs is noisy.

## Stored outputs of the main YOLOv8n run (`yolov8n_e100_s0_aquadet/`)

These are the aquadet outputs of `runs/yolov8n_e100_s0`, with `best.pt` scored by `python -m aquadet run`. They are the source of `docs/yolo/YOLOV8N.md` and `../benchmark.md`.

| File | Contents |
|---|---|
| `metrics.json` | Both scorers (pycocotools COCOeval and Ultralytics `model.val()`) on val and test, overall and per class, with the evaluation settings |
| `cocoeval_valid.txt`, `cocoeval_test.txt` | The full COCOeval summary printout for each split |
| `coco_predictions_test.json.gz` | All 7,945 test detections at conf ≥ 0.001 (gzip-compressed COCO results). The val predictions are in `../03_setup3_data_efficiency/eval/yolov8n_e100_s0/best/` |
| `benchmark.json` | Size and latency on the RTX 3060: settings, environment, and per-stage latency (mean, median, p90, std) |

Key numbers (×100):

| Split | COCO AP | AP50 | AP75 | APm | APl | AR100 | Ultralytics P / R | Ultralytics mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Val (127 images, 909 boxes) | 45.4 | 75.8 | 46.0 | 20.2 | 48.5 | 57.9 | 76.1 / 74.8 | 46.4 |
| Test (63 images, 582 boxes) | 47.6 | 76.1 | 54.3 | 27.0 | 50.1 | 57.4 | 81.5 / 72.2 | 47.9 |

- **Size:** 3.01 M parameters, 8.1 GFLOPs fused, 5.96 MB `best.pt`.
- **Latency** (batch 1, 640 px, 381 timed calls): FP32 inference 4.40 ms, end-to-end 6.57 ms (152 FPS); FP16 inference 4.89 ms, end-to-end 7.17 ms. FP16 is not faster for this nano model at batch 1.
- **Use of test:** test is reported only for YOLOv8n, which has no test counterpart from YOLOE, so it is not used in the Setup 1 comparison above. It was scored once and nothing was tuned on it. Two zero-area test shark boxes are omitted (`DATA.md`).

## Reproduce

Every prediction file needed for the table is stored in `reports/` (gzip-compressed), so the table can be rebuilt without running any model. The last command alone does that. The first three lines are only needed to regenerate the predictions from scratch.

```bash
# baseline (benchmark venv)
.venv/bin/python -m aquadet run --model yolov8n                 # trains + scores best.pt (val/test) + latency
.venv/bin/python scripts/yolo/data_efficiency.py evaluate        # also scores last.pt and best.pt of yolov8n_e100_s0 on val
# main model: see ../02_setup2_prompt_study
# table
E=reports/03_setup3_data_efficiency/eval/yolov8n_e100_s0; P=reports/02_setup2_prompt_study/runs
.venv/bin/python scripts/analysis/build_tables.py \
  --pred "YOLOv8n last.pt|supervised fine-tuning (COCO-pretrained)|448=$E/last/coco_predictions_valid.json.gz" \
  --pred "YOLOv8n best.pt (val-selected)|supervised fine-tuning (COCO-pretrained)|448=$E/best/coco_predictions_valid.json.gz" \
  --pred "YOLOE-26s bare names|zero-shot text prompts|0=$P/bare/predictions.json.gz" \
  --pred "YOLOE-26s synonym set|zero-shot text prompts|0=$P/synonym/predictions.json.gz" \
  --pred "YOLOE-26s descriptions|zero-shot text prompts|0=$P/description/predictions.json.gz" \
  --reference "YOLOE-26s bare names" --test-summary reports/runs/yolov8n_e100_s0.json
```
