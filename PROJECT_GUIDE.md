# Open-Vocabulary Object Detection with Text Prompts — Team Project Guide (3-Day Version)

Updated 4 Oct 2026 · Topic 6 · Submission deadline: **8:00 AM Wednesday 7 Oct 2026**. Our internal deadline is **9:00 PM Tuesday 6 Oct**, to leave a buffer.

## 1. What changed from the old guide

| Item | Old guide | New version (matches the lecturers' requirements) |
|---|---|---|
| Time | 10 weeks (placeholder) | 3 days: Sun 4 Oct → Tue 6 Oct |
| Report | 7 sections, 17-page guide | 11 sections following the template, 10–15 pages (excluding References/Appendix) |
| Exam | 12 slides, 7 speakers | 4 slides, 3 minutes, 1 speaker, then 12 minutes of Q/A |
| Submission | Report + code | Report PDF + repo + DATA.md, with the required file names |
| Scope | All 4 RQs, many experiments | RQ1, RQ2, RQ4 are mandatory; RQ3 (fine-tuning) is a stretch goal |

## 2. Mandatory requirements from the lecturers (cannot change)

- **Who submits:** the Group Leader (Lê Thanh Thảo), via Google Classroom.
- **Report PDF:** named `GroupID_ProjectID_Report.pdf` (ProjectID = 6), 10–15 pages excluding References and Appendix.
- **GitHub repo:** named `DL2026-GroupID-ProjectID`, containing code for data preparation, training, evaluation and inference/demo. The README must include installation instructions and the steps to reproduce the main results.
- **DATA.md:** official dataset URL, version, how the splits were made, preprocessing procedure, and scripts to reproduce the data. If the group creates or processes a new dataset, a download link is required.
- **Exam:** a 3-minute overview (exactly 4 slides, 1 presenter, stopped at exactly 3 minutes) + 12 minutes of Q/A. Examiners may ask **any team member**, about both the quantity and the quality of the work.
- **Report:** reporting results without interpretation is not enough. It needs error analysis with reasons, a Member Contribution table in the Appendix, and 5–10 references.

## 3. The core idea (everyone must be able to explain it)

A closed-set detector scores an image region `f` with a learned weight vector per class: `s_c = w_c · f`. An open-vocabulary detector replaces that vector with the **text embedding of the class name**: `s_c = f · TextEnc("a photo of a c") / τ`. So the class list becomes an **input at inference time**: to add a class, just type a new prompt.

Shared pipeline: image → image encoder; prompt → text encoder; (cross-attention fusion, **Grounding DINO only**); → score head (matches regions to text) + box head → threshold + NMS → boxes, labels, scores.

Terms to define once in the report:
- **Zero-shot detection:** detecting classes that had no labeled boxes during training.
- **Open-vocabulary detection:** classes are given as free text at test time; training may use image–text data that mentions them.
- **Open-set / open-world detection:** flagging unknown objects without naming them. A different problem, not part of this topic.

## 4. Reduced scope

**Models (3 pretrained detectors + 1 baseline we build), all running on a Colab/Kaggle T4:**

| Model | Family | Library | Role |
|---|---|---|---|
| YOLOE-26s | CNN one-stage, CLIP text embeddings | `ultralytics` | Focus model, fine-tuning (stretch) |
| Grounding DINO-T | DETR-style, BERT text, cross-attention | `transformers` | The Transformer model, links to Lectures 8–9 |
| OWLv2 base | ViT + CLIP, no fusion | `transformers` | Third family |
| CLIP + R-CNN (baseline) | Two-stage: crop then classify | `torchvision`, `open_clip_torch` | Baseline built from Lecture 7 |

**Baseline (Lecture 7 + CLIP):** (1) take the top 100 proposals after NMS from the RPN of a pretrained Faster R-CNN (torchvision); (2) crop each, resize to 224×224, encode with the CLIP image encoder; (3) encode the prompt `a photo of a {class}` with the CLIP text encoder; (4) score by cosine similarity and apply per-class NMS. It will lose to the other models, and explaining why (crops lose context, no box refinement, one CLIP pass per crop) is part of the analysis.

**Research questions:**

| RQ | Question | Priority | Goes into |
|---|---|---|---|
| RQ1 | How well do the 3 detectors find the dataset's objects zero-shot compared with the baseline? | Mandatory | Setup 1 |
| RQ2 | How much does prompt wording (names, synonyms, descriptions) change accuracy? | Mandatory, the core requirement of the topic | Setup 2 |
| RQ4 | Speed vs accuracy across models? | Mandatory (only needs latency timing) | Setup 1 |
| RQ3 | Does fine-tuning YOLOE beat zero-shot? | **Stretch**, go/no-go decision at 20:00 on Monday | Setup 3 |

**What we cut because we only have 3 days** (record these in the report's Limitations):

| Cut | Replaced by |
|---|---|
| Sanity check on a 500-image COCO val subset | Drawing boxes on 5–10 random images after every change |
| 100 self-taken photos | Not done |
| Fine-tuning with 10 / 50 / all images per class, multiple seeds | At most 1 fine-tuning run on the full train set, 1 seed |
| TIDE toolbox | Manually counting 5 error types on about 50 errors per model |
| Gradio live demo | Screenshots/clip of YOLOE with a custom prompt. Gradio only if Huy finishes early |
| Grounding DINO formatting-only variant | Not done |

### First milestone: a pipeline everyone can defend

Before parallel model comparisons or optional fine-tuning, get **one** text-prompted detector working end to end: load a dataset image → supply an object name as text at inference time → predict boxes/scores → map prompt indices to dataset category IDs → convert boxes to COCO format → compute AP → save the prompts, settings and predictions. Verify by drawing ground truth and predicted boxes on a few images. Then change the prompt to a synonym and a description for the same category on the **same validation images** and check that the resulting predictions and metrics are recorded separately. This demonstrates both lecturer requirements before expanding to more models.

Each member must run this pipeline and explain where prompts are consumed, how labels and boxes are mapped, how AP is calculated, and how to reproduce one result. Model owners still implement their components, but another member must review each component before its results go into the report. If time is short, keep this working pipeline and the name/synonym/description study; cut fine-tuning and optional variants first. Do not silently drop mandatory comparisons.

## 5. Dataset

Pick **one** labeled dataset from Roboflow 100 or ODinW with these criteria: objects that COCO lacks, roughly 5–15 classes, and a test split of a few hundred images or fewer so it runs fast on a T4. **Decide within the first hour today**, because everything else depends on it.

As soon as it is decided, Lê Minh writes into `DATA.md`: the official URL, version, number of images per split, and the class list. Then convert to COCO JSON and write a script that reproduces it (`scripts/prepare_data.py`). Use the dataset's own train/val/test split if it has one.

## 6. Evaluation rules

**Metrics:** mAP@[0.5:0.95] and AP50 via `pycocotools`, per-class AP, and latency (ms/image) on the same GPU after a few warm-up runs.

**Fairness rules:**
1. Same images, same GPU, same input size for every model.
2. Same class list and prompt template for every model, except in the prompt study.
3. When computing mAP, keep low-confidence boxes (threshold about 0.01–0.05), because a high threshold makes AP look artificially worse.
4. Grounding DINO reads at most 256 tokens: split long class lists into chunks; prompts are lower-case and separated by ` . `.
5. **Tune prompts and thresholds on val. Run the test split exactly once, on Tuesday morning.**

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

## 7. Repo and tools

```
DL2026-GroupID-6/
├── README.md        # installation + how to rerun every number
├── DATA.md          # URL, version, split, preprocessing
├── requirements.txt # pinned versions
├── src/models/      # yoloe.py, gdino.py, owlv2.py, clip_rcnn.py
├── src/eval/        # coco_eval.py, latency.py
├── scripts/         # prepare_data.py, run_zero_shot.py, prompt_study.py, finetune.py
├── results/results.csv
└── demo/            # inference/demo script
```

One shared interface, so the evaluation code never changes when a model is added:

```python
class OVDetector:
    def predict(self, image, prompts: list[str], score_thr: float = 0.01):
        """Return boxes [N,4] xyxy pixels, scores [N], labels [N] (index into prompts)."""
        raise NotImplementedError
```

Conventions: never commit datasets or weights; every run adds a row to `results.csv` (date, model, split, settings, metrics); every number in the report must be reproducible from a script in the repo. Colab/Kaggle sessions disconnect easily, so save results to Google Drive continuously. Use Colab and Kaggle in parallel so two people can run jobs at once.

## 8. Task assignment

Each person owns one piece from code to report. Three members are called Minh, so use full names in chat.

| Member | Role | Main tasks | Report part |
|---|---|---|---|
| Lê Thanh Thảo | Lead + evaluation | Decide the dataset with the team, create the repo, write `OVDetector` + `coco_eval.py`, maintain `results.csv`, merge and edit the report, submit | Abstract, Introduction, Conclusion |
| Lê Minh | Data | Shortlist and prepare the dataset, splits, COCO JSON, `DATA.md`, `prepare_data.py` | Section 4 (Dataset), Results RQ1 |
| Nguyễn Đức Minh | Grounding DINO | `gdino.py` (with 256-token chunking), zero-shot runs, prompt variants | Methods (Grounding DINO), Related Work |
| Nguyễn Minh | OWLv2 + prompt study | `owlv2.py`, zero-shot runs, run the prompt study (RQ2) for all models | Methods (OWLv2), Results RQ2, Related Work |
| Trung Ngọc Minh Côi | YOLOE | `yoloe.py`, zero-shot runs, 1 fine-tuning run (stretch, RQ3) | Methods (YOLOE), Setup 3, Results RQ3 if done |
| Trần Khoa Nam | Eval runs + error analysis | `latency.py`, speed tests (RQ4), error counting, pick 3 good + 3 bad examples per model | Experimental Setup, Results RQ4, Section 8 (Error Analysis) |
| Nguyễn Đình Huy | Baseline + slides/demo | `clip_rcnn.py`, the 4 slides, demo image/clip | Methods (Baseline + Comparison Strategy), Section 8 (with Nam) |

**Presenter of the 3-minute overview:** we suggest Lê Thanh Thảo (the person with the whole picture). The team confirms today. Whoever presents, **everyone must be able to answer Q/A about the whole project**.

## 9. Timeline (3 days)

Rule: **stop running new experiments at 12:00 on Tuesday**; after that, only analysis, writing and rehearsal.

| When | Task | Who |
|---|---|---|
| **Sun 4 Oct, first hour** | Decide the dataset, create repo + group chat, confirm the presenter, ask the lecturers if needed (section 15) | Whole team, Thảo, Minh |
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
| Tue 6 Oct, 18:00–20:30 | Run the checklist (section 13), rename files, check the README reruns; rehearse the 3 minutes + Q/A | Whole team |
| **Tue 6 Oct, 21:00** | **Submit** (Thảo submits). Do not leave it until close to 8:00 AM | Thảo |

If we fall behind, cut in this order: fine-tuning → distractor/template variants → baseline prompt variants (not the RQ1 baseline comparison). **Never cut the end-to-end text-prompted pipeline, the name/synonym/description comparison on the same val images, report-writing, or speaking-practice time.** If mandatory model coverage is impossible, discuss the change with the lecturers and report it honestly.

## 10. Report structure (following the lecturers' template)

Target 11–13 pages of content. The lecturers' template is mandatory, so this replaces the old 7-section structure.

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
| 10 | References | 5–10 sources (section 14) | Not counted | Đức Minh |
| 11 | Appendix | **Member Contribution Table** (required), full prompts, hyperparameters | Not counted | Thảo |

**Writing rules:**
- Every figure/table has a caption stating the takeaway (e.g. *Template prompts raise AP over bare names*) and is referenced in the text.
- Tables put units in headers and bold the best result.
- Draw our own architecture diagrams; a figure taken from a paper must be cited in its caption.
- Cite every number and claim that comes from a paper. Do not claim state of the art; we are studying, not competing.
- Report negative results honestly; they are still findings.
- Check the course policy on AI writing tools and disclose use if required.
- Thảo does the final read: same terms everywhere, same model names, every RQ answered.

## 11. 3-minute overview: exactly 4 slides

One presenter, stopped at exactly 3 minutes, so about 45 seconds per slide (roughly 90–100 spoken words). One message per slide, at most 6 short lines, font 24 pt or larger, pictures over text. Slide titles are takeaway sentences.

| Slide | Content | Suggested visual |
|---|---|---|
| 1. Problem & Research Question | Fixed-class detectors miss objects they never learned; open-vocabulary detectors take a text description. State the 3–4 RQs briefly | Same image: a regular YOLO misses an object, YOLOE finds it |
| 2. Method & Experiments | Class weights → text embeddings. 3 detectors + CLIP + R-CNN baseline, 1 dataset, same evaluation conditions, prompt study | The shared pipeline diagram (image encoder → text encoder → fusion → box head) |
| 3. Key Results | Main table/chart: mAP of the 4 models (RQ1), effect of prompts (RQ2), speed vs mAP (RQ4) | One chart whose title is the conclusion |
| 4. Conclusion / Demo | Answer the RQs, 1–2 main limitations, demo image/clip | YOLOE image with a custom prompt, or a 10–15 second clip |

With only 3 minutes, **no live demo**: use a pre-recorded image or clip on slide 4. Rehearse at least twice with a timer; if over time, cut content instead of speaking faster.

## 12. 12-minute Q/A: everyone must be able to answer

Examiners ask individuals about both the quantity and the quality of the work. Each person must (1) explain the core idea, (2) describe the pipeline of all 4 models, (3) know the main numbers and why they came out that way, (4) open the repo and point to the file that produces a given number.

| Likely question | Short answer |
|---|---|
| What is the key idea? | Per-class weights are replaced by text embeddings, so the class list is an input |
| Why mAP and not accuracy? | Detection must get both box position (IoU) and the ranking of confident predictions right |
| Did the model really never see the new classes? | No labeled boxes, but CLIP's web pre-training almost surely saw the words |
| Why does Grounding DINO use cross-attention? | So image features and text tokens can condition each other before boxes are predicted |
| Why is the baseline slow and weaker? | One CLIP pass per crop (the same weakness as R-CNN vs Fast R-CNN in Lecture 7), crops lose context, no box refinement |
| Which prompts fail? | Relations and states (relative position, damaged); have an example ready |
| Why tune on val and run test once? | To avoid leaking test information into parameter choices |
| How did you check for box-format bugs? | Drew boxes on images, checked xyxy/xywh/normalized, checked the class mapping |
| What was your part of the work? | Each person answers following the Member Contribution table |

Before the exam, hold a code walkthrough: every member runs a small inference/evaluation example, locates prompt handling, label/box conversion, AP computation and result logging in the repo, and explains one component they did not write. Have another member review each component's code and reproduce at least one reported result. Keep a short shared Q/A sheet with the actual commands and result-file paths.

When speaking, link back to the course: RPN and NMS from Lecture 7, cross-attention from Lecture 8, BERT from Lecture 9.

## 13. Checklist before submitting (Tue 6 Oct, 18:00–21:00)

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

## 14. References (pick 8–10 for the report)

CLIP (Radford et al., 2021) · ViT (Dosovitskiy et al., 2020) · DETR (Carion et al., 2020) · OWL-ViT (Minderer et al., 2022) · OWLv2 (Minderer et al., 2023) · Grounding DINO (Liu et al., 2023) · YOLOE (Wang et al., 2025) · ViLD (Gu et al., 2021) · Faster R-CNN (Ren et al., 2015) · BERT and Attention Is All You Need (the basis of Lectures 8–9).

Where to start coding (with copy-paste examples): the Hugging Face zero-shot object detection guide (OWLv2, Grounding DINO), the Grounding DINO model page on Hugging Face, and Ultralytics YOLOE (text prompts, fine-tuning, validation).

Numbers from official pages are for reference only; because protocols differ between papers, the report uses only numbers **we measure ourselves**.

## 15. Still unknown, confirm today

- **GroupID**, needed to name the report file and repo.
- The exact date and time of the oral exam (so we rehearse for the right deadline).
- Whether a university GPU is available (if not, use Colab T4 + Kaggle as planned).
- The course policy on AI writing tools in reports, and how to disclose use.
