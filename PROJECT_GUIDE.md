# Project plan: Open-Vocabulary Object Detection with Text Prompts

Project 6 · Working plan originally drafted 4 Oct 2026. This document records the team's proposed scope and submission checklist; **it is not evidence that every planned experiment was completed**. See [README.md](README.md) for reproducible commands, [DATA.md](DATA.md) for dataset decisions, and [EXPERIMENTS.md](EXPERIMENTS.md) for runs actually performed. Confirm administrative details and the course's official instructions with the teaching staff.

**Current repository status:** Aquarium v2 preparation and derived COCO validation are complete. A pretrained YOLOE-26s bare-name baseline was evaluated on the validation split using a local CPU. Visual checks, prompt variants, the other model families, comparable GPU latency, and held-out test evaluation remain planned, not completed. The original source has two invalid test annotations; see DATA.md.

## 1. Submission and course checklist

- **Who submits:** the Group Leader (Lê Thanh Thảo), via Google Classroom.
- **Report PDF:** named `GroupID_ProjectID_Report.pdf` (ProjectID = 6), 10–15 pages excluding References and Appendix.
- **GitHub repo:** check the required name `DL2026-GroupID-ProjectID` against the official instructions before submission. Include code for the experiments actually run and a README with installation instructions and steps to reproduce reported results.
- **DATA.md:** official dataset URL, version, how the splits were made, preprocessing procedure, and scripts to reproduce the data. If the group creates or processes a new dataset, a download link is required.
- **Exam:** a 3-minute overview (exactly 4 slides, 1 presenter, stopped at exactly 3 minutes) + 12 minutes of Q/A. Examiners may ask **any team member**, about both the quantity and the quality of the work.
- **Report:** interpret measured results, document errors and limitations, and include a Member Contribution table in the Appendix and references. Verify formatting requirements against the official template.
- **Research integrity:** follow the course policy on AI-assisted work and disclose assistance if required. Do not attribute results to experiments that were not run.

## 2. Background and research goal

A closed-set detector scores an image region `f` with a learned weight vector per class: `s_c = w_c · f`. In a simplified open-vocabulary scoring model, a class can instead be represented by a **text embedding**: `s_c ≈ f · TextEnc(prompt_c)` (model-specific heads and normalization vary). So the class list becomes an **input at inference time**: to add a class, just type a new prompt.

Shared pipeline: image → image encoder; prompt → text encoder; (cross-attention fusion, **Grounding DINO only**); → score head (matches regions to text) + box head → threshold + NMS → boxes, labels, scores.

Terms to define once in the report:
- **Zero-shot detection:** detecting classes that had no labeled boxes during training.
- **Open-vocabulary detection:** classes are given as free text at test time; training may use image–text data that mentions them.
- **Open-set / open-world detection:** flagging unknown objects without naming them. A different problem, not part of this topic.

## 3. Proposed experimental scope

**Planned comparison:** three pretrained detectors plus one proposal-and-classify baseline. Only the YOLOE-26s validation baseline has been run in this repository; the shared-GPU comparison is pending.

| Model | Family | Library | Role |
|---|---|---|---|
| YOLOE-26s | CNN one-stage, CLIP text embeddings | `ultralytics` | Focus model, fine-tuning (stretch) |
| Grounding DINO-T | DETR-style, BERT text, cross-attention | `transformers` | The Transformer model, links to Lectures 8–9 |
| OWLv2 base | ViT + CLIP, no fusion | `transformers` | Third family |
| CLIP + R-CNN (baseline) | Two-stage: crop then classify | `torchvision`, `open_clip_torch` | Baseline built from Lecture 7 |

**Baseline (Lecture 7 + CLIP):** (1) take the top 100 proposals after NMS from the RPN of a pretrained Faster R-CNN (torchvision); (2) crop each, resize to 224×224, encode with the CLIP image encoder; (3) encode the prompt `a photo of a {class}` with the CLIP text encoder; (4) score by cosine similarity and apply per-class NMS. Possible weaknesses to test include loss of context from crops, lack of box refinement and the cost of one CLIP pass per crop. Do not assume its measured ranking in advance.

**Research questions:**

| RQ | Question | Priority | Goes into |
|---|---|---|---|
| RQ1 | How well do the 3 detectors find the dataset's objects zero-shot compared with the baseline? | Mandatory | Setup 1 |
| RQ2 | How much does prompt wording (names, synonyms, descriptions) change accuracy? | Mandatory, the core requirement of the topic | Setup 2 |
| RQ4 | Speed vs accuracy across models? | Mandatory (only needs latency timing) | Setup 1 |
| RQ3 | Does fine-tuning YOLOE beat zero-shot? | Stretch goal; run only if the validation pipeline and resources allow | Setup 3 |

**What we cut because we only have 3 days** (record these in the report's Limitations):

| Cut | Replaced by |
|---|---|
| Sanity check on a 500-image COCO val subset | Drawing boxes on 5–10 random images after every change |
| 100 self-taken photos | Not done |
| Fine-tuning with 10 / 50 / all images per class, multiple seeds | At most 1 fine-tuning run on the full train set, 1 seed |
| TIDE toolbox | Manually counting 5 error types on about 50 errors per model |
| Gradio live demo | Screenshots/clip of YOLOE with a custom prompt. Gradio only if Huy finishes early |
| Grounding DINO formatting-only variant | Not done |

### First end-to-end pipeline milestone

Before parallel model comparisons or optional fine-tuning, get **one** text-prompted detector working end to end: load a dataset image → supply an object name as text at inference time → predict boxes/scores → map prompt indices to dataset category IDs → convert boxes to COCO format → compute AP → save the prompts, settings and predictions. Verify by drawing ground truth and predicted boxes on a few images. Then change the prompt to a synonym and a description for the same category on the **same validation images** and check that the resulting predictions and metrics are recorded separately. This demonstrates both lecturer requirements before expanding to more models.

Each member must run this pipeline and explain where prompts are consumed, how labels and boxes are mapped, how AP is calculated, and how to reproduce one result. Model owners still implement their components, but another member must review each component before its results go into the report. If time is short, keep this working pipeline and the name/synonym/description study; cut fine-tuning and optional variants first. Do not silently drop mandatory comparisons.

## 4. Dataset

The selected dataset is Roboflow 100 Aquarium (`aquarium-qlnqy`, version 2). Its published train/valid/test split is preserved. [`DATA.md`](DATA.md) documents the source, annotation audit, derived COCO conversion, class mapping and the two omitted zero-area test records. `scripts/prepare_data.py` audits or downloads the source export, `scripts/yolo_to_coco.py` creates derived JSON, and `scripts/validate_coco.py` checks it with pycocotools. The source itself is not repaired.

## 5. Evaluation protocol (planned comparisons)

**Metrics:** mAP@[0.5:0.95] and AP50 via `pycocotools`, per-class AP, and latency (ms/image) on the same GPU after warm-up for cross-model speed comparisons. Existing YOLOE timing is CPU-only and must not be compared with GPU timings.

**Fairness rules:**
1. Same images, same GPU, same input size for every model.
2. Same class list and prompt template for every model, except in the prompt study.
3. When computing mAP, keep low-confidence boxes (threshold about 0.01–0.05), because a high threshold makes AP look artificially worse.
4. Grounding DINO reads at most 256 tokens: split long class lists into chunks; prompts are lower-case and separated by ` . `.
5. **Choose prompts and thresholds on validation. Evaluate the held-out test split only after freezing the protocol; do not tune on test.**

**5 checks before trusting a number:**
1. Draw predicted boxes and ground truth on 5 random images after every change.
2. Check the box format: COCO JSON is `[x, y, w, h]`, while models often return `[x1, y1, x2, y2]` or normalized `[cx, cy, w, h]`. A format mix-up is the #1 cause of near-zero mAP.
3. Check the class mapping: the label index must point to the right prompt, and each prompt must map to the right dataset category ID.
4. Run end to end on 10 images before running a whole split.
5. Record the settings next to every result in `results/results.csv`.

**Prompt study (RQ2), run on val, one row per variant for each model:** Use the same images and ground-truth categories for each variant. Define the exact prompts in a versioned list before running; map each prompt (including every synonym and description) back to its dataset category ID. Keep name, synonym and description variants even if scope is reduced. Record per-class AP and overall AP, plus examples where changing wording helps or hurts. Do not compare variants with different class coverage as though they were equivalent.

| Variant | Example for `helmet` | What it tests |
|---|---|---|
| Bare name | `helmet` | Baseline |
| Template | `a photo of a helmet` | CLIP-style phrasing |
| Synonyms | `crash helmet`, `motorcycle helmet` (averaged embedding for CLIP-style models) | Robustness to word choice |
| Descriptive phrase | `motorbike helmet worn on a head` | Does extra detail help or confuse? |
| Distractor classes (Setup 3) | add `hat`, `cap` to the list | Confusion between similar concepts |

## 6. Repository layout and tools

Current tracked files include `README.md`, `DATA.md`, `EXPERIMENTS.md`, `PROJECT_GUIDE.md` and the data-preparation and YOLOE baseline scripts under `scripts/`. Datasets, model assets and run outputs are ignored by Git. The following interface is **proposed for future multi-model comparisons**; it has not been implemented:

```python
class OVDetector:
    def predict(self, image, prompts: list[str], score_thr: float = 0.01):
        """Return boxes [N,4] xyxy pixels, scores [N], labels [N] (index into prompts)."""
        raise NotImplementedError
```

Conventions: never commit datasets or weights. Each run should record date, model, split, settings and metrics; a future shared `results.csv` is planned. The current YOLOE baseline saves `config.json`, `predictions.json`, `runtime.json` and `metrics.json` locally. Every number included in the report should be reproducible from a documented script.

## 7. Proposed task assignment

The assignments below are from the planning draft, **not verified records of who contributed to code or experiments**. Confirm them with the team and use actual work, not this table, for the report's contribution statement.

| Member | Role | Main tasks | Report part |
|---|---|---|---|
| Lê Thanh Thảo | Lead + evaluation | Decide the dataset with the team, create the repo, write `OVDetector` + `coco_eval.py`, maintain `results.csv`, merge and edit the report, submit | Abstract, Introduction, Conclusion |
| Lê Minh | Data | Shortlist and prepare the dataset, splits, COCO JSON, `DATA.md`, `prepare_data.py` | Section 4 (Dataset), Results RQ1 |
| Nguyễn Đức Minh | Grounding DINO | `gdino.py` (with 256-token chunking), zero-shot runs, prompt variants | Methods (Grounding DINO), Related Work |
| Nguyễn Minh | OWLv2 + prompt study | `owlv2.py`, zero-shot runs, run the prompt study (RQ2) for all models | Methods (OWLv2), Results RQ2, Related Work |
| Trung Ngọc Minh Côi | YOLOE | `yoloe.py`, zero-shot runs, 1 fine-tuning run (stretch, RQ3) | Methods (YOLOE), Setup 3, Results RQ3 if done |
| Trần Khoa Nam | Eval runs + error analysis | `latency.py`, speed tests (RQ4), error counting, pick 3 good + 3 bad examples per model | Experimental Setup, Results RQ4, Section 8 (Error Analysis) |
| Nguyễn Đình Huy | Baseline + slides/demo | `clip_rcnn.py`, the 4 slides, demo image/clip | Methods (Baseline + Comparison Strategy), Section 8 (with Nam) |

**Presenter:** confirm with the team. All members should be able to explain the actual experiments and limitations.

## 8. Original three-day work schedule (historical plan)

The table below records the original target schedule, **not a log of completed work or proof of submission**. Consult the README and EXPERIMENTS.md for current status. Confirm deadlines separately with the course staff.

| When | Task | Who |
|---|---|---|
| **Sun 4 Oct, first hour** | Decide the dataset, create repo + group chat, confirm the presenter, confirm outstanding course requirements | Whole team, Thảo, Minh |
| Sun 4 Oct, afternoon–evening | Convert to COCO JSON + split + `DATA.md`; `OVDetector` interface; each person runs their model on 1 image and draws the boxes | Lê Minh, Thảo, each owner |
| Sun 4 Oct, 22:00 | **Checkpoint A:** dataset ready, each model runs on 1 image | Whole team |
| Mon 5 Oct, morning | Finish the 4 wrappers; `coco_eval.py` works correctly on 10 images (draw boxes to check) | Owners, Thảo |
| Mon 5 Oct, afternoon | Zero-shot on val for 3 models + baseline (RQ1); measure latency | Owners, Nam |
| Mon 5 Oct, evening | Prompt study on val (RQ2); fix prompt template and threshold | Nguyễn Minh, owners |
| Mon 5 Oct, 20:00 | **Go/no-go on YOLOE fine-tuning:** if zero-shot is stable and GPU time remains, run 1 job overnight; otherwise cut it | Côi, Thảo |
| Mon 5 Oct, 22:00 | **Checkpoint B:** `results.csv` complete for RQ1 + RQ2 on val; start writing Methods and Related Work | Whole team |
| Tue 6 Oct, 08:00–10:00 | Run the test split **exactly once** for every model + latency (and fine-tuning if done) | Owners, Nam |
| Tue 6 Oct, 10:00–12:00 | Finish all numbers; **12:00 freeze on experiments** | Whole team |
| Tue 6 Oct, 10:00–15:00 | Error analysis, figures/tables, write Results, Setup, Section 8, in parallel | Nam, Huy, owners |
| Tue 6 Oct, 15:00–18:00 | Thảo merges the report; writes Abstract, Intro, Conclusion; make the 4 slides | Thảo, Huy |
| Tue 6 Oct, 18:00–20:30 | Run the pre-submission checklist, rename files, check the README reruns; rehearse the 3 minutes + Q/A | Whole team |
| **Tue 6 Oct, 21:00** | **Submit** (Thảo submits). Do not leave it until close to 8:00 AM | Thảo |

Prioritize the end-to-end pipeline, comparable validation images and clear reporting over optional fine-tuning. If the planned model coverage is not feasible, discuss the scope with the teaching staff and report what was actually completed.

## 9. Draft report outline

Use the official course template and verify its requirements independently. The following is a proposed division of writing tasks, not a record of completed sections.

| # | Section | What it contains | Length | Writer |
|---|---|---|---|---|
| 1 | Abstract | Problem, approach, 2–3 key numbers, conclusion | 150–200 words | Thảo (written last) |
| 2 | Introduction and Research Question | Why fixed class lists are a limitation, what open-vocabulary detection is, 3–4 RQs | 0.5–1 page | Thảo |
| 3 | Related Work | Closed-set detection → vision-language models → open-vocabulary detectors, grouped by idea | 0.5–1 page | Đức Minh, Nguyễn Minh |
| 4 | Dataset and Data Preparation | Dataset, **official URL + version**, splits, class list, preprocessing | ~1 page | Lê Minh |
| 5 | Methods | **Baseline Method** (CLIP + R-CNN); **Main Method** (YOLOE, Grounding DINO, OWLv2, one self-drawn diagram per model); **Comparison Strategy** (same images/GPU/prompts/metrics, why the comparison is fair) | 2.5–3 pages | Huy, Đức Minh, Nguyễn Minh, Côi |
| 6 | Experimental Setup | **Setup 1** Baseline vs Main (RQ1 + RQ4); **Setup 2** Main research experiment = prompt study (RQ2); **Setup 3** Analysis/Robustness/Ablation = distractor classes, per-class AP, and YOLOE fine-tuning if done (RQ3) | 1.5 pages | Nam, Côi |
| 7 | Results and Discussion | One subsection per RQ, opening with the answer, then **interpreting why**, not just listing numbers | 2.5–3 pages | Lê Minh (RQ1), Nguyễn Minh (RQ2), Nam (RQ4), Côi (RQ3) |
| 8 | Error and Qualitative Analysis | Error types (wrong box position, confused with a similar class, background false alarm, missed small/occluded object, duplicate boxes), counts per model, 3 good + 3 bad examples, **reasons** | 1.5 pages | Nam, Huy |
| 9 | Conclusion and Limitations | Answer each RQ; limitations (3 days, 1 dataset, 1 seed, what we cut); next steps | 0.5 page | Thảo |
| 10 | References | Sources used and cited in the report | Not counted | Đức Minh |
| 11 | Appendix | **Member Contribution Table** (required), full prompts, hyperparameters | Not counted | Thảo |

**Writing rules:**
- Every figure/table needs a caption stating the measured takeaway and should be referenced in the text; do not claim a prompt improves AP before measuring it.
- Tables put units in headers and bold the best result.
- Draw our own architecture diagrams; a figure taken from a paper must be cited in its caption.
- Cite every number and claim that comes from a paper. Do not claim state of the art; we are studying, not competing.
- Report negative results honestly; they are still findings.
- Check the course policy on AI writing tools and disclose use if required.
- Thảo does the final read: same terms everywhere, same model names, every RQ answered.

## 10. Proposed presentation outline

One presenter, stopped at exactly 3 minutes, so about 45 seconds per slide (roughly 90–100 spoken words). One message per slide, at most 6 short lines, font 24 pt or larger, pictures over text. Slide titles are takeaway sentences.

| Slide | Content | Suggested visual |
|---|---|---|
| 1. Problem & Research Question | Fixed-class detectors miss objects they never learned; open-vocabulary detectors take a text description. State the 3–4 RQs briefly | Same image: a regular YOLO misses an object, YOLOE finds it |
| 2. Method & Experiments | Class weights → text embeddings. 3 detectors + CLIP + R-CNN baseline, 1 dataset, same evaluation conditions, prompt study | The shared pipeline diagram (image encoder → text encoder → fusion → box head) |
| 3. Key Results | Main table/chart: mAP of the 4 models (RQ1), effect of prompts (RQ2), speed vs mAP (RQ4) | One chart whose title is the conclusion |
| 4. Conclusion / Demo | Answer the RQs, 1–2 main limitations, demo image/clip | YOLOE image with a custom prompt, or a 10–15 second clip |

With only 3 minutes, **no live demo**: use a pre-recorded image or clip on slide 4. Rehearse at least twice with a timer; if over time, cut content instead of speaking faster.

## 11. Preparation for Q/A

Examiners ask individuals about both the quantity and the quality of the work. Each person must (1) explain the core idea, (2) describe the pipeline of all 4 models, (3) know the main numbers and why they came out that way, (4) open the repo and point to the file that produces a given number.

| Likely question | Short answer |
|---|---|
| What is the key idea? | Per-class weights are replaced by text embeddings, so the class list is an input |
| Why mAP and not accuracy? | Detection must get both box position (IoU) and the ranking of confident predictions right |
| Did the model really never see the new classes? | No labeled boxes, but CLIP's web pre-training almost surely saw the words |
| Why does Grounding DINO use cross-attention? | So image features and text tokens can condition each other before boxes are predicted |
| What might limit the proposal-and-classify baseline? | Crops lose context, lack box refinement and require CLIP inference per crop; actual speed and AP need measurement. |
| Which prompts perform poorly? | Answer from saved prompt-study runs and examples once they exist; no results yet. |
| Why tune on val and run test once? | To avoid leaking test information into parameter choices |
| How will you check for box-format bugs? | Check xyxy versus xywh, category mapping, and draw ground truth and predictions on validation images; visual checks are still pending. |
| What was your part of the work? | Each person answers following the Member Contribution table |

Before the exam, hold a code walkthrough: every member runs a small inference/evaluation example, locates prompt handling, label/box conversion, AP computation and result logging in the repo, and explains one component they did not write. Have another member review each component's code and reproduce at least one reported result. Keep a short shared Q/A sheet with the actual commands and result-file paths.

When speaking, link back to the course: RPN and NMS from Lecture 7, cross-attention from Lecture 8, BERT from Lecture 9.

## 12. Pre-submission checklist (verify with official instructions)

- [ ] Report PDF is 10–15 pages excluding References/Appendix, named `GroupID_ProjectID_Report.pdf`
- [ ] Abstract is 150–200 words; Introduction and Related Work are 0.5–1 page each; Conclusion about 0.5 page
- [ ] All 11 template sections present, Member Contribution table in the Appendix
- [ ] 5–10 references; figures and numbers taken from papers are cited
- [ ] Official dataset URL + version appear in the report **and** in `DATA.md`
- [ ] Every RQ is answered in Results and Conclusion; results are interpreted, not just listed
- [ ] Repo named `DL2026-GroupID-ProjectID`, README has installation and rerun steps; every number in the report reruns from a script
- [ ] Every member has run the text-prompted pipeline, explained prompt/label/box/AP code, and reviewed a component they did not write
- [ ] Name, synonym and description variants use the same validation images and category mapping; exact prompts and settings are saved
- [ ] No datasets or weights committed; `requirements.txt` has pinned versions
- [ ] 4 slides done, rehearsed under 3 minutes; everyone has read the whole report and can answer Q/A
- [ ] The leader has submitted on Google Classroom and confirmed the upload succeeded

## 13. Suggested reading for the report

CLIP (Radford et al., 2021) · ViT (Dosovitskiy et al., 2020) · DETR (Carion et al., 2020) · OWL-ViT (Minderer et al., 2022) · OWLv2 (Minderer et al., 2023) · Grounding DINO (Liu et al., 2023) · YOLOE (Wang et al., 2025) · ViLD (Gu et al., 2021) · Faster R-CNN (Ren et al., 2015) · BERT and Attention Is All You Need (the basis of Lectures 8–9).

Implementation references include the Hugging Face zero-shot object detection guide (OWLv2 and Grounding DINO), the Grounding DINO model page, and the Ultralytics YOLOE text-prompting documentation. Cite the specific sources used and check model-specific APIs before implementation.

Numbers from official pages are for reference only; because protocols differ between papers, the report uses only numbers **we measure ourselves**.

## 14. Items to confirm with the team or course staff

- Group ID, final roster and report/repository naming requirements (fill in README.md after confirmation).
- Submission and oral-exam schedule, presentation format and report template.
- GPU access and consistent hardware for latency comparisons.
- Policy on AI-assisted code/writing and any required disclosure.

This plan does not certify that the checklist items above are done; mark them complete only after verification.
