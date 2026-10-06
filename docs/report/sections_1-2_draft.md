# Report draft: Sections 1–2 (not yet integrated into the external PDF)

**Status:** Temporary repository draft for the two final-report writers to review and copy into the shared external document. It is not the submitted report. Numbers below use COCO bounding-box AP@[0.50:0.95] on a 0–100 scale and refer to the **validation split**. Check wording against [`sections_3-9_draft.md`](sections_3-9_draft.md) before final assembly.

## 1. Abstract

Open-vocabulary object detection allows users to specify target categories with text rather than training a detector for a fixed class list. We study this capability on Roboflow 100 Aquarium v2 using pretrained YOLOE-26s, without fine-tuning it on Aquarium, and compare it with a supervised YOLOv8n baseline. Both models are evaluated on the same 127 validation images using COCO bounding-box average precision (AP, 0–100 scale). YOLOE achieves **13.1 AP** with the seven class names, while YOLOv8n trained on all 448 Aquarium training images achieves **44.4 AP** with its seed-0 final checkpoint. Prompt wording has class-specific effects: replacing two names with synonyms gives **12.8 AP** overall, improving starfish by 7.7 AP points but reducing jellyfish by 9.9. Seven descriptive prompts give **5.8 AP**. In a separate data-efficiency study, the tested crossover from YOLOE to higher supervised YOLOv8n performance is bracketed between **22 and 45 labelled training images** under the fixed 100-epoch schedule; the smaller subsets were added post-hoc. Text prompts offer flexibility without Aquarium fine-tuning, but accuracy depends on wording and is substantially lower than the fully supervised baseline on this dataset. All YOLOE prompt results are validation results, not held-out test results.

## 2. Introduction and Research Questions

Conventional object detectors predict a set of categories defined during training. They can perform well when those categories are known and labelled examples are available, but adding another category generally requires new training data and further training. Open-vocabulary detection offers a more flexible interface: a user supplies category names or descriptions as text at inference time. That flexibility raises a practical question—does changing the wording of a prompt change what the model detects?

We investigate this question on **Roboflow 100 Aquarium v2**, a seven-class object-detection dataset containing fish, jellyfish, penguins, puffins, sharks, starfish and stingrays. Our main method is pretrained **YOLOE-26s**, which accepts text prompts at inference time and is **not fine-tuned on Aquarium**. We compare it with **YOLOv8n**, a fixed-class detector fine-tuned on the dataset's 448 training images. The models receive different amounts of Aquarium-specific supervision, so their comparison describes performance under different training regimes; it does not isolate which architecture is better.

This project addresses two research questions:

1. **How well does text-prompted YOLOE detect Aquarium objects without Aquarium fine-tuning, compared with a supervised fixed-class baseline on the same validation images?**
2. **How do class names, selected synonyms and descriptive prompts affect YOLOE's overall and per-class detection performance?**

To answer them, we compare the two models using the same 127 validation images and COCO AP evaluator. We then hold YOLOE, the images and the class mapping fixed while changing its text prompts. Finally, a separate data-efficiency experiment varies the number of labelled images used to train YOLOv8n, asking how much dataset-specific supervision is needed to exceed YOLOE's bare-name result. Paired prediction overlays and error analysis help interpret the measured outcomes.

The synonym experiment changes **only two** of the seven class names; it is not a comprehensive test of synonyms. Likewise, "without Aquarium fine-tuning" does not establish that YOLOE's original pretraining data contained no related images or categories. All text-prompt comparisons in this report use the **validation split**, not a held-out YOLOE test run.

**Handoff:** Before using these sections in the PDF, reconcile the two research questions and conclusion wording with the other writer's §§3–9. Sources for the numerical claims: [`Setup 1`](../../reports/01_setup1_baseline/setup1_table.md), [`Setup 2`](../../reports/02_setup2_prompt_study/README.md), [`per-class AP`](../../reports/04_per_class_ap/per_class_ap.md) and [`Setup 3`](../../reports/03_setup3_data_efficiency/posthoc_small_fractions/tables.md).
