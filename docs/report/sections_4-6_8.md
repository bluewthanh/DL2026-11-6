# Sections 4–6 and 8: Dataset, Methods, Experimental Setup, Error Analysis

Author: Trần Nam. These are the report sections assigned to us; the other sections are written by the final-gate team. All numbers come from `reports/`, and every table and figure can be rebuilt from the stored predictions there.

Sources cited here, for the report's reference list: Ciaglia et al., 2022 (Roboflow 100, arXiv:2211.13523); Wang et al., 2025 (YOLOE, arXiv:2503.07465); Jocher et al., 2023 (Ultralytics YOLOv8).

---

## 4. Dataset and Data Preparation

We use **Roboflow 100 Aquarium, version 2** (Ciaglia et al., 2022) (CC BY 4.0, https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2). It contains photos taken in aquariums, with boxes for 7 classes: fish, jellyfish, penguin, puffin, shark, starfish and stingray.

| Split | Images | Boxes | Used for |
|---|---:|---:|---|
| Train | 448 | 3,328 | Training YOLOv8n only |
| Validation | 127 | 909 | All comparisons |
| Test | 63 | 582 | YOLOv8n only, never compared with YOLOE |

**Preparation.** We kept the original splits. Our scripts check the labels, convert them to COCO format and check the result again (`scripts/data/`).

**Problems we found.** Two shark boxes in the test set have zero size. We removed only these two boxes, and only from our converted test labels; the original files are unchanged (see `DATA.md`). This may slightly change the shark score on test, but it does not affect any validation result.

**What makes this dataset hard** (`scripts/analysis/gt_stats.py`):
- **Imbalance:** half of the validation boxes are fish, but there are only 27 starfish and 33 stingrays.
- **Small objects:** penguins and puffins are the smallest, with an equivalent box side of approximately 35 px when the image is resized to 640 px. They usually appear in groups.
- **Crowding:** all 155 jellyfish come from only 9 images.

## 5. Methods

**Main model: YOLOE-26s** (Wang et al., 2025). It is an open-vocabulary detector: we give it the class names as text, e.g. "fish, jellyfish, …". A text encoder turns each name into a vector, and every box gets the name whose vector matches it best. We did **not** fine-tune it on the Aquarium data (zero-shot), and we did not check whether its pretraining data overlaps with these images or classes. We use the pretrained `yoloe-26s-seg.pt` checkpoint and keep only its boxes.

**Baseline: YOLOv8n** (Jocher et al., 2023). A small, normal detector (3.0 M parameters) that starts from COCO-pretrained weights and is trained on the 448 training images. It can only find the 7 classes it was trained on and cannot take text.

**How we compare them.** Both models are tested on the same 127 validation images, with the same settings and the same scoring tool (COCO AP). The comparison is not meant to say which model is better. It shows how close a model with no training on our data gets to a model trained on it.

Two rules keep the comparison fair:
- For YOLOv8n we report the **last epoch** (`last.pt`), not the "best" epoch, because the best epoch was picked using the validation images.
- We never compare YOLOv8n's test score with YOLOE's validation score.

## 6. Experimental Setup

**Common settings.**
- Images resized to 640 px, score threshold 0.001, NMS IoU 0.7, at most 300 boxes per image.
- Main metric: COCO AP@[0.50:0.95] from pycocotools, reported on a 0–100 scale.
- The new experiments ran on one RTX 3060 GPU, with one Python environment for each model. An earlier CPU run of YOLOE was used only to check the GPU results.

**Setup 1: baseline vs main model.**
- YOLOE uses the plain class names as prompts.
- YOLOv8n is trained for 100 epochs with the shared settings in `configs/base.yaml` (AdamW, batch 16, no early stopping), using 3 seeds.

**Setup 2: prompt wording.** YOLOE is run 3 times, and only the text prompts change:

| Prompt set | What changes |
|---|---|
| Names | `fish`, `jellyfish`, `starfish`, … |
| Synonyms | only jellyfish → `sea jelly` and starfish → `sea star` |
| Descriptions | a short sentence for each class, e.g. "star-shaped sea animal with radiating arms" |

We ran this on both CPU and GPU. The results match to within 0.034 AP points.

**Setup 3: how many labels beat zero-shot?**
- We train YOLOv8n on 10 %, 25 %, 50 % and 100 % of the training images, 3 seeds each. The smaller subsets are part of the larger ones.
- We wrote the plan down before training (`reports/03_setup3_data_efficiency/PROTOCOL.md`).
- After seeing the first results, we added 1 %, 2.5 % and 5 % to narrow down the crossover. These extra runs are reported separately.

## 8. Error and Qualitative Analysis

### 8.1 What kind of mistakes each model makes

To count errors, each model is drawn at its own score threshold: YOLOv8n 0.45, YOLOE 0.10 (0.05 with descriptions). We picked these thresholds on the validation images, only to display and count errors. AP is computed separately, using every box with score ≥ 0.001. A box is correct when it has the right class and IoU ≥ 0.5 with the object. The YOLOv8n row uses `best.pt` (seed 0); the main comparison uses `last.pt`.

| Model | Correct | Wrong class | Poor box | Duplicate | Background | Missed (of 909) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n (`best.pt`, seed 0) | 654 | 22 | 43 | 8 | 87 | 255 |
| YOLOE, names | 390 | **291** | 81 | 25 | 336 | **519** |
| YOLOE, synonyms | 350 | 282 | 73 | 19 | 315 | 559 |
| YOLOE, descriptions | 121 | **734** | 55 | 14 | 558 | **788** |

Poor box = right class but IoU between 0.1 and 0.5. Duplicate = a second box on an object that is already found.

- **YOLOv8n** is usually right when it draws a box. Its main mistake is **missing objects**: 137 of its 255 misses are fish, and the missed fish are usually smaller than the ones it finds.
- **YOLOE** often puts a box on a real object but gives it the **wrong name** (291 boxes). It also misses more than half of the objects.

### 8.2 Main findings

Each finding lists what we **measured**, then a **possible reason**. The reasons are plausible explanations that we did not test.

**1. Sharks are called "fish".**
- Measured: YOLOE gets only 3 of the 57 sharks right and labels 41 sharks as `fish`. Sharks are not small, and YOLOv8n gets 44.7 AP on them (`last.pt`).
- Possible reason: a shark is also a kind of fish, so the prompt `fish` may match sharks better than `shark` does. (Figure 8.1)

**2. Small birds in groups.**
- Measured: puffins and penguins are the smallest objects. They are YOLOv8n's two weakest classes (25.0 and 30.5 AP, `last.pt`), and puffin is YOLOE's second weakest class after shark (6.2 AP). In one image with 11 puffins behind wet glass, YOLOE finds none and YOLOv8n finds 5.
- Possible reason: small, grouped objects and poor image quality are hard for both models. (Figure 8.3)

**3. One word can help or hurt one class.**
- Measured: with `sea jelly`, the number of jellyfish labelled `fish` rises from 35 to 56, and jellyfish AP drops by 9.9. With `sea star`, starfish AP rises by 7.7.
- Possible reason: the model may match some words to the images better than others. (Figure 8.2)

**4. Descriptions confuse the model.**
- Measured: with descriptions, YOLOE misses 442 of the 459 fish. More than 100 fish are labelled as shark, as penguin and as starfish (each). Only starfish improves.
- Possible reason: the fish description ("swimming animal with fins and a tail") may be too general, while the starfish one names a shape no other class has ("star-shaped").

### 8.3 Figures

Each figure shows one image five times: ground truth | YOLOv8n (`best.pt`) | YOLOE names | YOLOE synonyms | YOLOE descriptions. Colours: green = correct, orange = wrong class, red = background, blue dashed = missed. Images: Roboflow 100 Aquarium v2, CC BY 4.0.

**Figure 8.1: sharks labelled as fish (`val_034`).** YOLOv8n gets 19 correct; YOLOE gets 11 correct and 11 wrong class. We checked the 6 sharks one by one in the stored predictions. YOLOE puts a `fish` box on 5 of them and misses the sixth. YOLOv8n labels 5 correctly and calls one a stingray.

![val_034](../../reports/05_qualitative_overlays/figures/val_034.jpg)

**Figure 8.2: `sea jelly` hurts (`val_127`).** With names, YOLOE finds all 5 jellyfish. With `sea jelly`, the same boxes are labelled `fish` or `puffin`.

![val_127](../../reports/05_qualitative_overlays/figures/val_127.jpg)

**Figure 8.3: small puffins (`val_047`).** YOLOE misses all 11 puffins with every prompt set; YOLOv8n finds 5.

![val_047](../../reports/05_qualitative_overlays/figures/val_047.jpg)

### 8.4 Limits of the error analysis

- The figures explain the numbers; they are not extra proof.
- The display thresholds were picked on the same validation images.
- Some labels in the dataset are unclear, e.g. tiny or hidden fish.
