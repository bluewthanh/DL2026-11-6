# Report-ready draft: Sections 3–9 (not yet integrated into the external PDF)

**Status:** This is a repository draft for the final-report writers to review and paste into their shared external document. It does **not** prove that the PDF is finished. Sections 4–6 and 8 adapt the existing draft in [`sections_4-6_8.md`](sections_4-6_8.md); Sections 3, 7 and 9 are new. Assign section owners and finalize numbering, layout, figures and references in the external document. AP below is COCO bbox AP@[0.50:0.95] on a **0–100 scale**, unless stated otherwise. The comparison is on the **validation** set, not the YOLOE test set.

## 3. Related Work

Traditional object detectors learn to predict a predefined set of categories. Ultralytics YOLOv8 is an example: its detector can be fine-tuned on a labelled dataset and then predicts the classes in that training task (Jocher, Chaurasia and Qiu, 2023). This makes YOLOv8n a useful supervised, closed-set reference for a seven-class Aquarium benchmark, but adding an entirely new category requires new labels and further training. Its accuracy therefore answers a different question from zero-shot text-prompted detection.

Vision–language learning offers another interface. CLIP demonstrated that image–text alignment learned from natural-language supervision can transfer to tasks specified with text rather than task-specific class training (Radford et al., 2021). CLIP itself is not the box detector evaluated in this project. YOLOE extends the promptable approach to detection and segmentation: the model accepts text-specified categories, among other prompt types (Wang et al., 2025). In the implementation used here, a pretrained YOLOE-26s segmentation checkpoint and MobileCLIP2 text encoder produce text-conditioned detections; only bounding boxes are scored. “Zero-shot” in this report means **no Aquarium-labelled fine-tuning**, not a verified absence of Aquarium categories or images from the model's pretraining data.

Roboflow 100 offers a collection of detection datasets spanning different domains (Ciaglia et al., 2022). We study its Aquarium v2 dataset rather than claiming broad benchmark coverage. Existing model descriptions establish that promptable detection is possible, but they do not determine whether the **exact wording** of prompts works equally well on these seven Aquarium classes. Our experiment keeps the images, ground truth and model fixed while comparing bare names, a two-class synonym substitution and seven visual descriptions. A separate label-efficiency experiment asks how much Aquarium supervision a closed-set YOLOv8n needs to exceed the bare-name YOLOE result. These are local comparisons, not a universal ranking of the model families.

*Reference-list leads, checked against the cited source pages:* Ciaglia et al. (2022), *Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark*, https://arxiv.org/abs/2211.13523; Jocher, Chaurasia and Qiu (2023), *Ultralytics YOLOv8* (software/documentation, **not** a formal YOLOv8 research paper), https://docs.ultralytics.com/models/yolov8/; Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*, https://arxiv.org/abs/2103.00020; Wang et al. (2025), *YOLOE: Real-Time Seeing Anything*, https://arxiv.org/abs/2503.07465. The final writers should make these entries consistent with the bibliography and check every citation in the PDF.

## 4. Dataset and Data Preparation

We use [Roboflow 100 Aquarium, version 2](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2) (Ciaglia et al., 2022), a bounding-box dataset distributed under the CC BY 4.0 license displayed on the Roboflow Universe project page. Its seven classes are **fish, jellyfish, penguin, puffin, shark, starfish, stingray**. We preserve the published split: 448 training images (3,328 valid boxes), 127 validation images (909 boxes) and 63 test images (582 valid boxes in our *derived* ground truth). YOLOv8n uses the training images; every model-versus-model and prompt comparison uses the same validation images. The test split is reported only for the supervised baseline, never compared against YOLOE validation AP.

We use the original YOLOv8-format annotation export, not an additional split or a claimed restriction to YOLOv8 detectors. The source audit checks image/label pairs, class IDs, normalized boxes and counts (`scripts/data/prepare_data.py`). COCO ground truth is derived from those audited labels for both models (`scripts/data/yolo_to_coco.py`) and checked again with `scripts/data/validate_coco.py`. Source class IDs 0–6 map to COCO category IDs 1–7 in the class order above. Resizing to a 640-pixel model input belongs to inference/training; it is not a change to the source dataset.

The **source audit fails** on two zero-width, zero-height *test* shark annotations. We do not fabricate replacement boxes or edit the original export. With the explicit `omit-known-zero-boxes` policy, exactly these two records are absent from the derived YOLO/COCO test ground truth; all 63 test images remain. A prediction near either omitted position could count as a false positive, so test shark AP may be biased. No validation boxes are affected. One empty training label remains in the source; a visual check regarded it as a plausible negative image, not proof of flawless annotation. See [`DATA.md`](../../DATA.md) for file-level provenance and the dataset license.

This dataset is challenging for several reasons. Fish account for 459 of the 909 validation boxes, whereas starfish and stingray have only 27 and 33 respectively. All 155 jellyfish boxes occur in just nine validation images. Penguins and puffins tend to be relatively small and grouped; their median box area corresponds to an approximate 35-pixel side under the project's 640-pixel input geometry (`scripts/analysis/gt_stats.py`). These observations describe the ground truth, not proven causes of any particular model error.

## 5. Methods

**Main method: text-prompted YOLOE-26s.** We load a pretrained, text-promptable `yoloe-26s-seg.pt` checkpoint (Wang et al., 2025), with MobileCLIP2-based text representations, and call `set_classes(prompts)` before inference. Although the checkpoint supports segmentation, we save and evaluate only its predicted boxes and confidence scores. The model is **not fine-tuned on Aquarium**. The seven prompt positions map to the same seven COCO category IDs for all runs; changing text never changes the validation labels. The exact prompt lists and checkpoint details are in [`aquarium_prompts.json`](../../scripts/open_vocab/aquarium_prompts.json) and [`docs/open_vocab/YOLOE.md`](../open_vocab/YOLOE.md). Only jellyfish → `sea jelly` and starfish → `sea star` change in the synonym set. The description set changes all seven prompts. This design tests these particular words, not every possible synonym or description.

**Supervised reference: YOLOv8n.** We initialize the closed-set detector from COCO-pretrained `yolov8n.pt`, then fine-tune on the 448 labelled Aquarium training images (Jocher, Chaurasia and Qiu, 2023). It outputs the seven learned classes and does not accept a new text class at inference. The training recipe uses 640-pixel inputs, batch size 16, AdamW, 100 epochs and no early stopping (`configs/base.yaml`). The full-data comparison uses seed-0 **`last.pt`**, not validation-selected `best.pt`, as its primary row. We also report a three-seed full-data mean separately. YOLOv8n and YOLOE differ in training regime and model size; sharing the split and evaluator does not make them equal-budget competitors.

**Comparison strategy.** We score both detectors against exactly the same derived validation COCO JSON with pycocotools COCOeval. YOLOE has zero Aquarium-labelled training examples; YOLOv8n has from 4 to 448 labelled training images, depending on the setup. We compare bbox AP, not mask quality or calibrated deployment thresholds. The best-epoch YOLOv8n checkpoint is selected with validation data and therefore is shown separately as a potentially optimistic validation result. The qualitative overlays use `best.pt` and are explicitly labelled as such; they must not silently replace the primary `last.pt` row.

## 6. Experimental Setup

All main comparisons use **127 validation images and 909 ground-truth boxes**, with a consistent seven-class mapping and COCOeval bbox AP@[0.50:0.95] (averaged over classes and IoU thresholds; evaluator maxDets 100). AP50 uses IoU 0.50. We multiply 0–1 outputs by 100 when reporting AP points. Both prediction pipelines use an input size of 640, low confidence floor **0.001** to retain the AP ranking, NMS IoU 0.7 and model-side `max_det=300`; model-side and evaluator limits are different. Bounding boxes, not masks, are evaluated. The GPU report experiments were produced on an RTX 3060 with separate pinned Ultralytics environments for YOLOE and YOLOv8n; runtime figures across the earlier CPU and GPU runs should not be compared as latency benchmarks. Exact settings, hashes and commands are recorded in [`reports/00_setup/`](../../reports/00_setup/), [`reports/02_setup2_prompt_study/`](../../reports/02_setup2_prompt_study/) and the scripts.

**Setup 1 — baseline versus main method.** Fine-tune YOLOv8n on all 448 training images and compare its seed-0 `last.pt` validation predictions to pretrained YOLOE-26s prompted with the seven bare names. For context, report the three-seed mean for full-data YOLOv8n separately. `best.pt` is validation-selected and is not the primary comparison row. This measures an observed gap between **different supervision regimes**, not an architecture-only effect.

**Setup 2 — prompt wording.** Run the same pretrained YOLOE checkpoint on the same validation split three times: bare category names, the synonym variant changing only **two** classes, and short visual descriptions changing all seven. Retain the same mapping and inference/evaluation settings. The three saved GPU prompt-study runs reproduce the separately documented CPU numbers closely; they are not an independent fresh rerun by the later artifact reviewer. We did not evaluate text prompts on test.

**Setup 3 — label efficiency, not prompt tuning.** Fine-tune YOLOv8n on uniformly sampled, nested subsets of 10%, 25%, 50% and 100% of the 448 training images (45/112/224/448), with three seeds each; compare primary `last.pt` AP to fixed, bare-name YOLOE AP. The protocol was documented before subset runs (`reports/03_setup3_data_efficiency/PROTOCOL.md`). **After** seeing the 10% result, 1%, 2.5% and 5% subsets (4/11/22 images, three seeds) were added and must be labelled *post-hoc*. The number of epochs stays at 100: smaller subsets thus receive fewer optimizer steps; the study does not hold training-update count constant. Subsets are not class-stratified, and test is not used.

For qualitative error counts only, each model gets a validation-selected F1 display threshold (YOLOv8n best.pt 0.45; YOLOE bare/synonym 0.10; description 0.05). Those thresholds do **not** define the AP results, which use detections from the 0.001 floor. The error categories use IoU 0.5 and differ from the 0.50–0.95 averaging of COCO AP.

## 7. Results and Discussion

### 7.1 Setup 1 — supervised baseline versus text-prompted main method

| Method / regime | Aquarium-labelled training images | Validation AP | AP50 |
|---|---:|---:|---:|
| YOLOv8n `last.pt`, seed 0 (supervised) | 448 | **44.4** | 74.8 |
| YOLOE-26s, bare names (no Aquarium fine-tuning) | 0 | **13.1** | 23.0 |
| YOLOv8n `best.pt`, seed 0 (validation-selected; context only) | 448 | 45.4 | 75.8 |

Source: [`reports/01_setup1_baseline/setup1_table.md`](../../reports/01_setup1_baseline/setup1_table.md), recomputed from saved validation predictions. Across three full-data YOLOv8n seeds, `last.pt` AP is **44.7 ± 0.7** (sample standard deviation; [`Setup 3 tables`](../../reports/03_setup3_data_efficiency/tables.md)). Do not mix that mean with the seed-0 44.4 row. The seed-0 gap is **31.3 AP points**. All seven classes favour the supervised baseline on this validation split: for instance, shark scores 44.7 against 5.4, and puffin 25.0 against 6.2 ([`per-class table`](../../reports/04_per_class_ap/per_class_ap.md)). At 448 labels, YOLOv8n is more accurate on this known-class domain, but this is not a controlled claim that its architecture is intrinsically superior: it has been fine-tuned on Aquarium while YOLOE has not. YOLOE's distinctive capability is accepting inference-time class text. YOLOv8n's **test** AP of 47.6 is a different-split supervised result, not a valid comparison with YOLOE validation AP.

### 7.2 Setup 2 — wording can help one class while hurting another

| YOLOE prompt set | Validation AP | AP50 | Jellyfish AP | Starfish AP |
|---|---:|---:|---:|---:|
| Bare names | **13.1** | 23.0 | 12.8 | 17.7 |
| Two-class synonym variant | 12.8 | 21.2 | **3.0** | **25.4** |
| Seven descriptions | 5.8 | 8.3 | 3.1 | **27.2** |

Sources: [`prompt-study reproduction`](../../reports/02_setup2_prompt_study/README.md) and [`per-class AP`](../../reports/04_per_class_ap/per_class_ap.md). Relative to bare names, `sea star` raises starfish AP by **7.7 points** while `sea jelly` lowers jellyfish AP by **9.9 points**. The net overall change (−0.3 points) conceals these opposing responses. The five unchanged prompts also show small AP shifts (no more than 0.3 points), so changing two members of a joint vocabulary can affect detections assigned to other classes; we did not isolate the mechanism. In validation image `val_127`, five jellyfish found with bare names lose their correct labels with the synonym vocabulary at the chosen display threshold. This image illustrates, but does not itself quantify, the aggregate effect.

Descriptions lower AP for **six of seven** classes, especially fish (19.6 → 1.6) and stingray (18.7 → 2.6), while starfish improves (17.7 → 27.2). The tested starfish description names its distinctive radiating shape; the fish description is comparatively broad. That is a possible interpretation, **not** a tested causal explanation. With only one selection of prompts and a single pretrained model, it would be unjustified to conclude that all descriptive prompts reduce performance or that seven synonyms were evaluated. The earlier local CPU study and saved GPU reproduction agree to within 0.034 AP points on the report's 0–100 scale (`reports/02_setup2_prompt_study/README.md`).

### 7.3 Setup 3 — how many labels exceed bare-name YOLOE?

| YOLOv8n training images | Study status | Validation AP, `last.pt` (3-seed mean ± sample SD) |
|---|---|---:|
| 4 (1%) | Post-hoc | 2.7 ± 1.7 |
| 11 (2.5%) | Post-hoc | 6.9 ± 1.2 |
| 22 (5%) | Post-hoc | **9.9 ± 1.8** |
| 45 (10%) | Predeclared | **19.2 ± 2.7** |
| 112 (25%) | Predeclared | 31.6 ± 2.2 |
| 224 (50%) | Predeclared | 38.0 ± 0.7 |
| 448 (100%) | Predeclared | 44.7 ± 0.7 |
| YOLOE bare-name reference (0 Aquarium labels) | Fixed reference; one run | 13.1 |

Source: [`Setup 3 tables`](../../reports/03_setup3_data_efficiency/posthoc_small_fractions/tables.md) and individual saved run records. Every 45-image YOLOv8n seed beats YOLOE bare AP (16.9–22.2 versus 13.1), whereas none of the 22-image seeds does (8.7–12.0). Therefore, **under this fixed 100-epoch schedule and among tested fractions, the crossover is bracketed between 22 and 45 training images**. Its precise position was not measured. The post-hoc smaller fractions cannot be presented as though predeclared. Moreover, lower fractions get fewer optimizer steps and the 4-/11-image runs were still improving at epoch 100; longer or step-matched training could change the bracket. The model sizes and supervision regimes also differ. The 45-image model exceeds YOLOE on overall AP and six of seven classes, but still trails on puffin (3.5 versus 6.2 AP), showing that an overall crossover is not a per-class crossover. This setup varies *training-data quantity*, not YOLOE prompts.

## 8. Error and Qualitative Analysis

The following is a **threshold-dependent diagnostic**, not a replacement for COCO AP. The paired panels use the same validation images for ground truth, YOLOv8n **`best.pt`** and YOLOE under each prompt set. Model-specific display thresholds maximize validation F1 on a 0.05 grid: 0.45 for YOLOv8n, 0.10 for YOLOE bare/synonym and 0.05 for descriptions. A correct detection requires the right class and IoU ≥ 0.5; other categories include wrong class, poor localization (right class, IoU 0.1–0.5), duplicate and background. Misses count ground-truth objects without a correct match. These counts are influenced by threshold choice and differ from the AP computation at confidence ≥ 0.001.

| Validation error category | YOLOv8n `best.pt` | YOLOE bare | YOLOE synonym | YOLOE description |
|---|---:|---:|---:|---:|
| Correct | 654 | 390 | 350 | 121 |
| Wrong class | 22 | 291 | 282 | 734 |
| Poor box | 43 | 81 | 73 | 55 |
| Duplicate | 8 | 25 | 19 | 14 |
| Background detection | 87 | 336 | 315 | 558 |
| Missed ground truth (of 909) | 255 | 519 | 559 | 788 |

Source: [`error_breakdown.csv`](../../reports/05_qualitative_overlays/error_breakdown.csv), with classification and thresholds in [`overlay.py`](../../scripts/analysis/overlay.py). YOLOE bare has many more wrong-class detections and misses than the supervised baseline at their respective display thresholds. In particular, the saved wrong-class tally assigns **41 shark-ground-truth overlaps to `fish` predictions** for YOLOE bare; sharks and fish are related words, but whether this semantic overlap causes the confusion has **not** been tested. At the description threshold, 442 of the 459 fish GT boxes have no correct match; fish is often predicted as shark, penguin or starfish. This observation is consistent with prompt competition, not proof of its mechanism. YOLOv8n's 255 misses include 137 fish, and the missed fish tend to be smaller than correctly detected fish, again an association rather than a causal result.

Three same-image examples for the final PDF (choose readable panel sizes; label all panels and credit **Roboflow 100 Aquarium v2, CC BY 4.0**):

1. **`val_034` — mixed fish/shark tank:** six shark GT boxes; the saved predictions show five paired with YOLOE `fish` predictions and one missed at the display threshold. YOLOv8n gets 19 of the image's 24 annotated objects correct, YOLOE bare 11 correct and 11 wrong-class. [`Figure`](../../reports/05_qualitative_overlays/figures/val_034.jpg). A separate box-by-box check was described in the existing §8 draft; the overlay README alone had not established it.
2. **`val_127` — jellyfish wording:** five jellyfish GT boxes. YOLOE bare has five correct detections (plus three wrong-class boxes); with the synonym set it has **zero correct**, eight wrong-class boxes and five missed GT. The figure makes the `sea jelly` failure visible without treating one image as a statistical test. [`Figure`](../../reports/05_qualitative_overlays/figures/val_127.jpg).
3. **`val_047` — small puffins:** 11 puffin GT boxes behind wet glass. At display thresholds, YOLOv8n finds five; YOLOE bare and synonym find none. A description run has one wrong-class box and no correct puffin match. [`Figure`](../../reports/05_qualitative_overlays/figures/val_047.jpg).

The 16 stored figures were selected using a fixed rule based on per-class GT counts and *extreme* between-model F1 differences (`reports/05_qualitative_overlays/README.md`); the extreme examples are **not representative samples**. The figures depend on validation-selected display thresholds; some tiny/occluded ground-truth annotations may be debatable. Overlay YOLOv8n uses `best.pt` (45.4 validation AP), whereas the primary Setup 1 row uses `last.pt` (44.4). The `rock` demo is an illustration of vocabulary flexibility, **not** evidence of quantified open-vocabulary accuracy.

## 9. Conclusion and Limitations

Text prompts let the pretrained YOLOE-26s change its inference-time label vocabulary without Aquarium fine-tuning. On 127 shared Aquarium v2 validation images, however, its best tested bare-name result is **13.1 COCO AP** versus **44.4 AP** for a YOLOv8n seed-0 `last.pt` model supervised on 448 labelled images. This comparison shows the value of domain-specific labels on this task, not an architecture-only contest. Prompt wording matters: changing just two names raises starfish AP by 7.7 points but lowers jellyfish AP by 9.9; seven descriptions reduce overall AP to 5.8, although the starfish description helps that class. No conclusion about *all* possible synonyms or descriptions follows from these specific wordings.

A three-seed data-efficiency study finds that YOLOv8n trained on 22 images does not beat bare YOLOE on overall AP, while every 45-image run does. The **post-hoc 22–45-image bracket** holds only for the tested subsets and fixed 100-epoch recipe, which also gives smaller subsets fewer optimizer steps. Limits include a single dataset, uneven class frequencies, small/clustered objects, two invalid test-only shark boxes omitted from derived ground truth, one pretrained YOLOE checkpoint and one prompt design. YOLOE was not fine-tuned on Aquarium, but potential overlap with its pretraining data was not checked. All prompt comparisons use validation only: **no YOLOE held-out test result** or independently rerun model result should be implied. Future work could predeclare additional prompt variants, compare step-matched small-data training, and evaluate a frozen validation-selected open-vocabulary protocol once on test if the course requires it.
