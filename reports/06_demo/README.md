# Demo: from a text prompt to labelled boxes

`scripts/demo/detect.py` takes one or more images and either:

- **`--prompts "a, b, c"`**: any comma-separated phrases become YOLOE-26s's class list for that call (open-vocabulary, zero-shot); or
- **`--weights best.pt`**: the closed-set YOLOv8n baseline, which can only output the 7 classes it was trained on.

It writes an annotated `.jpg` plus a `.json` with the prompts, the settings and every box. The display threshold is conf 0.25. This is a qualitative demo, not an evaluation. The AP numbers are in `../01_setup1_baseline` and `../02_setup2_prompt_study`.

## Setup (once)

Use the separate YOLOE environment, since it needs a different Ultralytics version from the benchmark venv:

```bash
uv venv .venv-yoloe --python 3.12 --managed-python
uv pip install --python .venv-yoloe/bin/python torch==2.14.1 torchvision==0.29.1 ultralytics==8.4.172 \
  pycocotools==2.0.11 'git+https://github.com/ultralytics/CLIP.git@a13192f8cb767260d7dfd98c843b0716593169e7'
curl -L -o weights/yoloe-26s-seg.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt
curl -L -o weights/mobileclip2_b.ts https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts
sha256sum weights/yoloe-26s-seg.pt weights/mobileclip2_b.ts   # compare with docs/open_vocab/YOLOE.md
```

## Commands used for the outputs in `outputs/` (validation images only)

```bash
V=data/aquarium-v2-yolo/valid/images
STAR=$(ls $V/IMG_2381_*); PENG=$(ls $V/IMG_2325_*); SHARK=$(ls $V/IMG_2425_*)
P=.venv-yoloe/bin/python
$P scripts/demo/detect.py --device cuda:0 --prompts "fish, jellyfish, penguin, puffin, shark, starfish, stingray" --image $STAR $SHARK $PENG
$P scripts/demo/detect.py --device cuda:0 --prompts "fish, sea jelly, penguin, puffin, shark, sea star, stingray" --image $STAR
$P scripts/demo/detect.py --device cuda:0 --prompts "penguin" --image $PENG
$P scripts/demo/detect.py --device cuda:0 --prompts "shark, rock" --image $SHARK
$P scripts/demo/detect.py --device cuda:0 --weights runs/yolov8n_e100_s0/weights/best.pt --image $STAR $SHARK $PENG
```

## What the outputs show (conf ≥ 0.25)

| Image (ground truth) | Input | Boxes |
|---|---|---|
| IMG_2381 (1 starfish) | YOLOE, 7 bare names | 1 × `jellyfish` 0.69 (wrong class) |
| | YOLOE, synonym set (`sea star`) | 1 × `sea star` 0.58 (correct), 1 × `sea jelly` 0.45 on the same object |
| | YOLOv8n | 1 × `starfish` 0.97 |
| IMG_2325 (22 penguins) | YOLOE, 7 bare names | 9 × `penguin`, 1 × `fish`, 1 × `jellyfish` |
| | YOLOE, only `penguin` | 9 × `penguin` |
| | YOLOv8n | 23 × `penguin` |
| IMG_2425 (17 fish, 6 sharks, 1 stingray) | YOLOE, 7 bare names | 13 × `fish`, 3 × `jellyfish`, 1 × `stingray` |
| | YOLOE, `shark, rock` | 1 × `rock` 0.38, **no shark** |
| | YOLOv8n | 23 × `fish`, 5 × `shark`, 1 × `stingray` |

Talking points for the demo slide:

1. **The text is the interface.** Changing the prompt list changes the label vocabulary with no retraining. `sea star` puts the right label on the starfish that `starfish` mislabels as `jellyfish` (a weaker `sea jelly` box remains), and the prompt `rock`, a concept that appears nowhere in the Aquarium labels, returns a plausible rock box. YOLOv8n cannot do either.
2. **The cost is accuracy on this domain.** Even with the right word, YOLOE misses most objects here: no shark is found at 0.25 among the six in IMG_2425, and IMG_2325 gets only 9 penguin boxes for 22 penguins. The supervised YOLOv8n, trained on 448 labelled Aquarium images, finds far more. This matches the validation AP gap in Setup 1 (YOLOE bare 13.1 vs YOLOv8n 44.4 for last.pt; the demo uses best.pt, 45.4).

The demo uses the same NMS settings as the evaluation runs (IoU 0.7, class-aware: `agnostic_nms=False`, max_det 300). At conf ≥ 0.25 it returns the same boxes as the stored evaluation predictions; this was checked for four image/prompt pairs, with scores equal up to rounding. `agnostic_nms=False` must be passed explicitly, because YOLOE otherwise suppresses overlapping boxes of different classes. Box counts are raw counts at the 0.25 threshold and can include false positives, so they are not the matched counts used in `../05_qualitative_overlays`.
