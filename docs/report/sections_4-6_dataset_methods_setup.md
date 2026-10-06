# Sections 4–6: Dataset, Methods, Experimental Setup

Author: Trần Nam. Draft for `11_6_Report`. All numbers come from `reports/`.

## 4. Dataset

We use **Roboflow 100 Aquarium, version 2** (CC BY 4.0, https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2). It has photos taken in aquariums, with boxes for 7 classes: fish, jellyfish, penguin, puffin, shark, starfish and stingray.

| Split | Images | Boxes | Used for |
|---|---:|---:|---|
| Train | 448 | 3,328 | Training YOLOv8n only |
| Validation | 127 | 909 | All comparisons |
| Test | 63 | 582 | YOLOv8n only, never compared with YOLOE |

**Preparation.** We kept the original splits. Our scripts check the labels, convert them to COCO format and check the result again (`scripts/data/`).

**Problems we found.** Two shark boxes in the test set have zero size. We removed these two boxes only from our converted test labels; the original files are unchanged (see `DATA.md`). This may slightly change the shark score on test, but not any validation result.

**What makes this dataset hard** (`scripts/analysis/gt_stats.py`):
- **Imbalance:** half of the validation boxes are fish, but there are only 27 starfish and 33 stingrays.
- **Small objects:** penguins and puffins are the smallest, about 35 px wide when the image is resized to 640 px. They usually appear in groups.
- **Crowding:** all 155 jellyfish come from only 9 images.

## 5. Methods

**Main model: YOLOE-26s.** It is an open-vocabulary detector, so we can tell it which classes to look for as text, e.g. "fish, jellyfish, …". We did **not** train it on our data (zero-shot). We cannot check whether these images were in its original training data.

**Baseline: YOLOv8n.** A normal detector that we trained on the 448 training images. It can only find the 7 classes it was trained on and cannot take text.

**How we compare them.** Both models are tested on the same 127 validation images, with the same settings and the same scoring tool (COCO AP). The comparison is not meant to say which model is better. It shows how close a model with no training gets to a model trained on our data.

Two rules keep it fair:
- For YOLOv8n we report the **last epoch** (`last.pt`), not the "best" epoch, because the best epoch was picked using the validation images.
- We never compare YOLOv8n's test score with YOLOE's validation score.

## 6. Experimental Setup

**Common settings.**
- Images resized to 640 px, score threshold 0.001, NMS IoU 0.7.
- Main metric: COCO AP@[0.50:0.95], from pycocotools.
- The new experiments ran on one RTX 3060 GPU, with one Python environment for each model. The earlier CPU run was used only to check the GPU results.

**Setup 1: baseline vs main model.**
- YOLOE uses the plain class names as prompts.
- YOLOv8n is trained for 100 epochs with the shared settings in `configs/base.yaml`, using 3 seeds.

**Setup 2: prompt wording.** YOLOE is run 3 times. Only the text prompts change:

| Prompt set | What changes |
|---|---|
| Names | `fish`, `jellyfish`, `starfish`, … |
| Synonyms | only jellyfish → `sea jelly` and starfish → `sea star` |
| Descriptions | a short sentence for each class, e.g. "star-shaped sea animal with radiating arms" |

We ran this twice, on CPU and on GPU. The results match to within 0.034 AP points (0–100 scale).

**Setup 3: how much labelled data beats zero-shot?**
- We train YOLOv8n on 10 %, 25 %, 50 % and 100 % of the training images, 3 seeds each.
- We wrote the plan down before training (`reports/03_setup3_data_efficiency/PROTOCOL.md`).
- Later we added 1 %, 2.5 % and 5 % to narrow down where YOLOv8n starts to beat YOLOE (between 22 and 45 images). These extra runs are reported separately.

**Reproducibility.** All predictions are saved in `reports/`, so every table and figure can be rebuilt without running a model again. Training and scoring commands are in each folder's `README.md`.
