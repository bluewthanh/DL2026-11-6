# Paired qualitative and error analysis (validation split)

Each figure shows the same validation image five times: the ground truth, then YOLOv8n (`runs/yolov8n_e100_s0` best.pt, supervised), then YOLOE-26s with the bare, synonym and description prompts (zero-shot, this machine's reproduction in `../02_setup2_prompt_study`). Generated from the prediction files stored in `reports/`, so no model needs to be run:

```bash
.venv/bin/python scripts/analysis/overlay.py \
  --pred "YOLOv8n best.pt=reports/03_setup3_data_efficiency/eval/yolov8n_e100_s0/best/coco_predictions_valid.json.gz" \
  --pred "YOLOE bare=reports/02_setup2_prompt_study/runs/bare/predictions.json.gz" \
  --pred "YOLOE synonym=reports/02_setup2_prompt_study/runs/synonym/predictions.json.gz" \
  --pred "YOLOE description=reports/02_setup2_prompt_study/runs/description/predictions.json.gz" \
  --contrast "YOLOE bare:YOLOE synonym" --contrast "YOLOE bare:YOLOE description" \
  --contrast "YOLOv8n best.pt:YOLOE bare" --out reports/05_qualitative_overlays
```

## How to read a figure

- **Display threshold, per model:** the score that maximises F1 at IoU 0.5 over **all** 127 validation images, searched over a 0.05 grid. The values are YOLOv8n 0.45 (F1 0.76), YOLOE bare 0.10 (F1 0.38), synonym 0.10 (F1 0.36) and description 0.05 (F1 0.10). Each model is therefore shown at its own best operating point; a single shared threshold would leave YOLOE nearly empty. AP is computed separately at conf 0.001 and does not depend on this threshold. All values are in `manifest.json`.
- **Box colours:**
  - Green: correct (right class, IoU ≥ 0.5).
  - Orange: right place, wrong class.
  - Yellow: right class, poor box (0.1 ≤ IoU < 0.5).
  - Purple: duplicate.
  - Red: background false positive.
  - Blue dashed: missed ground truth.
- **Panel title:** counts of each outcome (C, W, P, D, B, M).

## How images were chosen (fixed rule, decided before looking at any figure)

1. For each of the 7 classes, the validation image with the most ground-truth boxes of that class (7 images).
2. For each of three contrasts (bare vs synonym, bare vs description, YOLOv8n vs YOLOE bare), the 3 images with the largest absolute difference in per-image F1 at the display thresholds.

No image was selected twice, giving **16 figures** (7 + 3 × 3). `manifest.json` records the reason each image was chosen and its per-model counts. The contrast-selected images are by construction the most extreme cases, so they illustrate mechanisms and are not typical results. Single-object images give F1 differences of ±1.0, which is why several of them are chosen.

## Error breakdown over all 127 validation images (`error_breakdown.csv`)

| Model | Correct | Wrong class | Poor box | Duplicate | Background FP | Missed GT (of 909) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n best.pt | 654 | 22 | 43 | 8 | 87 | 255 |
| YOLOE bare | 390 | 291 | 81 | 25 | 336 | 519 |
| YOLOE synonym | 350 | 282 | 73 | 19 | 315 | 559 |
| YOLOE description | 121 | 734 | 55 | 14 | 558 | 788 |

The most frequent wrong-class confusions (ground truth → predicted) are in `manifest.json`:

- **YOLOE bare:** fish → jellyfish 50, shark → fish 41, jellyfish → fish 35, fish → stingray 27, fish → puffin 27, penguin → puffin 17.
- **YOLOE synonym:** jellyfish → fish **56** (35 with bare), while fish → jellyfish drops to 4. Renaming the class to `sea jelly` makes the model label real jellyfish as fish.
- **YOLOE description:** fish → shark 112, fish → penguin 110, fish → starfish 108. This is consistent with the generic fish description ("swimming animal with fins and a tail") losing to the more specific descriptions of other classes, a hypothesis that was not tested.
- **YOLOv8n:** fish ↔ shark (5 each way), fish → stingray 3. Its dominant error is missed objects (255), mostly fish (137 of 459 fish missed). The missed fish tend to be smaller than the fish it detects (median 0.57 % vs 0.88 % of the image area).

## Examples that explain the per-class AP changes

- **`val_022`, starfish: wording helps.** Bare YOLOE labels the starfish `jellyfish` (0.69). With `sea star`, the object gets a correct `sea star` box (0.58), plus a second, lower `sea jelly` box (0.45) on the same object, counted as wrong class. This one image illustrates the +7.7 starfish AP from the synonym. YOLOv8n gets it at 0.97.
- **`val_127`, jellyfish: wording hurts.** Bare YOLOE finds all 5 jellyfish. With `sea jelly`, all 5 boxes stay in place but are labelled `fish` or `puffin`, so the score is 0 correct, 5 missed. This matches the −9.9 jellyfish AP and the jellyfish → fish confusion count above. With the description prompts, the highest-scoring box on 4 of the 5 is `starfish`, though 3 of them are still matched by lower-scoring jellyfish boxes.
- **`val_047`, puffins: the zero-shot model cannot see this class here.** Eleven small puffins on rocks behind rain-spattered glass. YOLOE misses all 11 under every wording, while YOLOv8n finds 5. Puffin is YOLOE's second-weakest class (6.2 AP, after shark at 5.4) and YOLOv8n's weakest (24.9 for best.pt).
- **`val_034`, a mixed tank (17 fish, 6 sharks, 1 stingray).** YOLOv8n gets 19 correct. YOLOE bare gets 11 correct and 11 wrong-class. Over the whole split, shark → fish is YOLOE's second most common confusion; this figure has not been checked box by box. The description prompts produce 37 wrong-class boxes.
- **`val_119`, a crowded school of 43 fish.** YOLOv8n gets 30 correct, 14 missed and 11 background boxes. YOLOE bare gets 11 correct and 33 missed. With descriptions YOLOE gets 0 correct, consistent with fish AP falling from 19.6 to 1.6.

## Limitations

- These are visual explanations of aggregate numbers, not additional evidence. One image does not establish an effect.
- The display thresholds were chosen on the same validation data that is displayed.
- An IoU of 0.5 decides correct vs poor box; COCO AP averages over IoU 0.50–0.95.
- Some ground-truth labels are themselves debatable, for example tiny or occluded fish. The data were not relabelled.
