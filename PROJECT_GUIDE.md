# Project guide: lecturer requirements and submission checklist

**Goal:** submit a complete project on **open-vocabulary object detection with text prompts**, not merely a closed-set YOLO benchmark. This checklist combines the topic requirements and the submission/examination instructions supplied by the team. It records repository evidence, not proof that the report, slides or submission are finished.

**Legend:** `[x]` implemented/documented in the repository · `[~]` partial or needs independent verification · `[ ]` not completed/evidenced. Update a box only after checking its deliverable. The full lecturer brief takes precedence if it contains further requirements.

## 1. Topic requirements: primary experiment

- [x] **Detect using text descriptions, not only fixed classes.** Pretrained YOLOE-26s accepts inference-time class text via `set_classes(prompts)` in [`scripts/open_vocab/run_yoloe_baseline.py`](scripts/open_vocab/run_yoloe_baseline.py). This is text-conditioned inference, not proof that the classes were unseen during pretraining. YOLOE was **not** fine-tuned on Aquarium.
- [x] **Investigate names, synonyms and descriptions.** [`scripts/open_vocab/aquarium_prompts.json`](scripts/open_vocab/aquarium_prompts.json) versions three prompt sets. The synonym variant changes **only** jellyfish → `sea jelly` and starfish → `sea star`; five class names stay unchanged. Descriptions change all seven. Do not claim a seven-class synonym effect.
- [x] **Measure prompt effects on consistent validation data.** The YOLOE pipeline maps prompt indices to the same seven COCO IDs, evaluates the same 127 validation images with pycocotools and writes `config.json`, `predictions.json`, `runtime.json`, `metrics.json` for each run. [`docs/open_vocab/YOLOE.md`](docs/open_vocab/YOLOE.md) documents CPU full-validation AP: bare **0.13109**, synonym **0.12771**, description **0.05759**, with per-class results. A historical GPU study is documented separately; its generated artifacts are unavailable here. These are **validation-only**, not prompt-based test scores.
- [~] **Independent reproduction and review.** CPU output JSON exists locally but `results/` is Git-ignored. Ask a second member to run all three [README commands](README.md#primary-experiment-names-synonyms-and-descriptions), compare image IDs, ground-truth/model hashes and inference settings, and inspect the saved metrics and predictions. Record reviewer, environment, outcome and any discrepancies. The existing `tests/test_prompts.py` checks prompt definitions, **not** model-result correctness.
- [ ] **Paired qualitative and error analysis.** On the *same validation images*, overlay ground truth and each prompt variant's predictions; show correct detections and failures, identify category/box errors and explain examples where wording helps or hurts. Keep example selection and limitations transparent. Save reproducible instructions and appropriate figures for report/slides.
- [ ] **Reproducible inference/demo evidence.** The runner performs inference, but there is no tracked visual demo command/output for a reader to follow from text input to labeled boxes. Create and verify one; avoid treating a single example as quantitative evidence.

## 2. Report experiments: decide the comparison before deleting code

The lecturer's report template explicitly calls for **Baseline Method, Main Method, Comparison Strategy**, and three setups. The brief does **not** require several closed-set YOLO variants; the project keeps one supervised baseline, YOLOv8n.

| Report setup | Proposed experiment and current evidence | Status / decision needed |
|---|---|---|
| **Setup 1 — Baseline vs Main Model** | Main = text-prompted pretrained YOLOE. Baseline = the fine-tuned closed-set YOLOv8n run; [`reports/benchmark.md`](reports/benchmark.md) contains its summary. | `[~]` Define the exact baseline and compare on the **same validation split and AP evaluator**. Disclose that supervised fine-tuning and zero-shot inference are different regimes. Never place YOLOE **validation** AP beside YOLO **test** AP as a fair ranking. |
| **Setup 2 — Main research experiment** | YOLOE bare-name vs synonym vs description prompts on the same 127 validation images; overall and per-class AP in [`docs/open_vocab/YOLOE.md`](docs/open_vocab/YOLOE.md). | `[~]` Quantitative comparison done locally; independent review, visual evidence and interpretation remain. |
| **Setup 3 — Analysis / Robustness / Ablation** | Candidate: a **separate, predeclared** focused experiment (e.g. change only one of the two real synonyms at a time while holding every other prompt fixed), with per-class AP and paired examples. | `[ ]` Agree on one feasible question and protocol, run it, interpret it. Do **not** merely rename Setup 2's existing table as Setup 3 or present unrun proposals as findings. |

**Supplementary work already in Git:** YOLOv8n has a completed 100-epoch seed-0 supervised run ([results](docs/yolo/YOLOV8N.md)). It learns fixed Aquarium classes and does **not** satisfy the text-prompt requirement. Do not fabricate unrun comparisons.

## 3. Required repository and dataset deliverables

- [x] **Official dataset, version, splits and preprocessing policy:** [`DATA.md`](DATA.md) documents Roboflow 100 Aquarium **v2**, its official URL, published **448 train / 127 valid / 63 test** splits, source export, annotation audit, class mapping and derived COCO conversion. Two zero-area test shark records are omitted **only in derived ground truth** under an explicit policy; the original export is unchanged. Disclose possible shark-AP bias if reporting test metrics.
- [x] **Data-preparation scripts:** `scripts/data/prepare_data.py`, `yolo_to_coco.py` and `validate_coco.py` audit/download the original export and reproduce/check derived COCO ground truth. Source images, annotations and generated COCO are not tracked.
- [~] **Processed-dataset download rule:** The lecturer requires a downloadable link **if the group creates a new or processed dataset**. The official source has a link and the derived COCO is reproducible from scripts, but confirm whether the lecturer also expects a separate downloadable derived-data artifact. Check redistribution terms before sharing data; record the decision in `DATA.md`/report.
- [x] **Training source code:** `aquadet/` and `configs/` cover the supervised closed-set experiments; YOLOE uses a pretrained checkpoint and **has no Aquarium training step**. If the baseline is removed, revisit how to fulfill the repository's training-source requirement and explain clearly what was and was not trained.
- [x] **Evaluation and inference source:** YOLOE validation inference/COCOeval in `scripts/open_vocab/`; closed-set evaluation in `aquadet/`. A standalone, documented **visual demo** is still outstanding (see §1).
- [~] **Complete README and reproducibility:** [`README.md`](README.md) documents setup, checkpoints, dataset preparation and main commands. Have a teammate follow it from a fresh environment; verify versions, paths, download hashes and all three results. YOLOE and closed-set benchmark need different Ultralytics versions. Do not commit keys, data, weights or generated results.
- [~] **Repository naming:** Current remote is `DL2026-11-6`, which appears to match `DL2026-GroupID-ProjectID` **if** group ID 11 and project ID 6 are confirmed. Verify roster and IDs before submission.

## 4. Required report PDF — not evidenced in tracked files

The report must be **10–15 pages excluding References and Appendix** and named `GroupID_ProjectID_Report.pdf` (likely `11_6_Report.pdf` **if IDs are confirmed**). No report PDF is tracked here; work in another location may exist but has not been checked. Use the lecturer's order:

| Status | Section | Required work |
|---|---|---|
| `[ ]` | 1. Abstract | 150–200 words; actual method, measured results and conclusion. |
| `[ ]` | 2. Introduction and Research Question | 0.5–1 page; state the two text-prompt requirements as research questions. |
| `[ ]` | 3. Related Work | 0.5–1 page; cite relevant detection and vision-language methods. |
| `[~]` | 4. Dataset and Data Preparation | Draft in [`docs/report/sections_4-6_dataset_methods_setup.md`](docs/report/sections_4-6_dataset_methods_setup.md) (source: `DATA.md`): official URL/version, splits, processing, quality decisions and limitations. Review, then move into the report. |
| `[~]` | 5. Methods | Draft in the same file: YOLOE-26s main method, YOLOv8n baseline (`last.pt` as the fair row) and comparison strategy. Review, then move into the report. |
| `[~]` | 6. Experimental Setup | Draft in the same file: shared protocol, environments, Setup 1, Setup 2 and Setup 3 (data efficiency) with reproduce commands. Review, then move into the report. |
| `[~]` | 7. Results and Discussion | Metrics are documented, but write **interpretation**, per-class effects and comparison caveats; do not list numbers alone. |
| `[~]` | 8. Error and Qualitative Analysis | Draft in [`docs/report/section_8_error_analysis.md`](docs/report/section_8_error_analysis.md): error breakdown, main reasons and three paired figures from `reports/05_qualitative_overlays`. Review, then move into the report. |
| `[~]` | 9. Conclusion and Limitations | About 0.5 page; address both topic questions; discuss one dataset/model, two real synonyms, validation-only prompt study and dataset defects. |
| `[ ]` | 10. References | Cite approximately 5–10 relevant sources, including dataset and methods. |
| `[ ]` | 11. Appendix | **Member Contribution Table** reflecting verified work; add reproducibility details as needed. |

Use only results backed by scripts/documentation; do not present the historical GPU numbers as independently verified or claim a prompt-based held-out test run. If a held-out open-vocabulary test result is required by further instructions, freeze the validation-selected protocol first; the current YOLOE runner intentionally refuses test inference.

## 5. Examination and submission

- [ ] **Exactly four slides**, in order: (1) Problem & Research Question, (2) Method & Experiments, (3) Key Results, (4) Conclusion / Demo. Choose **one presenter** and rehearse a project overview of **no more than 3 minutes**; examiners stop at 3 minutes.
- [ ] **Prepare every member for 12 minutes of Q&A:** prompts and mapping, COCO AP, experimental comparisons, errors, dataset policy, reproducibility and each person's actual contribution. Examiners may ask anyone.
- [ ] **Group leader submits via Google Classroom**, checks the required PDF filename and repository name and confirms successful upload. The stated deadline is **8:00am, 7 October 2026**; confirm course timezone and any updates with staff.

## 6. Priority order (focus on mandatory work)

1. **Agree on Setup 1 baseline and a distinct Setup 3** with the team, then freeze the evaluation plan. Do not remove training/baseline code until their role is decided.
2. **Independently reproduce/review YOLOE** and generate paired visual examples and an actual demo; record error cases and causes.
3. **Write and review the report** against every row in §4, especially comparison fairness, interpretation, limitations and contribution table.
4. **Test a clean README reproduction** and resolve the processed-dataset link question without sharing restricted data.
5. **Make and rehearse the four slides; prepare all members for Q&A; verify naming and submission.**
6. Keep YOLOv8n as the only closed-set baseline; do not add further closed-set models unless the course requires them.
