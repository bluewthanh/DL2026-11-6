# Closed-set YOLO benchmark: YOLOv8n vs YOLO11n vs YOLOv10n

This benchmark fine-tunes three COCO-pretrained **closed-set** nano detectors on RF100 Aquarium v2 and compares their accuracy, size and speed under one shared protocol. All three train on the same training split and use the same derived evaluation ground truth. They do **not** support text-prompted or zero-shot detection; confirm with course staff whether this revised scope meets the original project brief.

| Model | Config | Owner | Status |
|---|---|---|---|
| YOLOv8n | `configs/models/yolov8n.yaml` | _fill in_ | done, see [`YOLOV8N.md`](YOLOV8N.md) |
| YOLO11n | `configs/models/yolo11n.yaml` | _fill in_ | pipeline smoke-tested, full run pending |
| YOLOv10n | `configs/models/yolov10n.yaml` | _fill in_ | pipeline smoke-tested, full run pending |

## 1. Setup (once per machine)

```bash
# Python 3.12 recommended. With uv (fast; --managed-python ships the headers Triton needs):
uv venv .venv --python 3.12 --managed-python
uv pip install --python .venv/bin/python -r requirements-yolo.txt
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# Without uv: python -m venv .venv, activate, install the PyTorch build for your
# CUDA from pytorch.org, then: python -m pip install -r requirements-yolo.txt

python -m aquadet env                # prints torch/CUDA/GPU that will be recorded with results
python -m unittest discover -s tests # 10 fast checks, no GPU needed
```

**Use the pinned `ultralytics==8.4.173`.** Training defaults and metric code change between releases, so a different version is a different experiment.

Put the **unmodified** Roboflow export (`data.yaml`, `train/`, `valid/`, `test/`) at `dataset/aquarium.v2-release.yolov8/`. If you keep it elsewhere, pass `--source PATH` or set `AQUADET_SOURCE=PATH`. Download instructions are in [`DATA.md`](../../DATA.md).

## 2. Running your model

```bash
python -m aquadet prepare                       # audit + build data/aquarium-v2-yolo and data/aquarium-v2-coco
python -m aquadet run --model yolo11n --smoke   # ~2 min: 1 epoch through every stage; checks your setup
python -m aquadet run --model yolo11n           # the real experiment (100 epochs + eval + benchmark)
git add reports/runs/yolo11n_e100_s0.json       # commit ONLY the summary; runs/ is git-ignored
python -m aquadet compare                       # regenerate reports/benchmark.md and .csv
```

`run` = train → evaluate on val and test with `best.pt` → benchmark → write the summary. Each stage can also be re-run on its own:

| Command | What it does |
|---|---|
| `python -m aquadet train --model M` | Train only, into `runs/<model>_e<epochs>_s<seed>/` |
| `python -m aquadet evaluate --run NAME [--split test]` | Re-score a trained run (merges into `eval/metrics.json`) |
| `python -m aquadet benchmark --run NAME` | Re-measure size and latency **on this machine** |
| `python -m aquadet report --run NAME` | Re-export `reports/runs/NAME.json` |
| `python -m aquadet compare [--include-custom]` | Leaderboard from all committed summaries |

Extra seeds: `--seed 1`, `--seed 2`. Seeds do not change the protocol hash, and `compare` reports mean ± std over seeds per model.

## 3. The shared protocol (`configs/base.yaml`)

Model configs only name the checkpoint. **Everything else is shared**, and `compare` enforces this. Each run stores a hash of the shared settings. A run with any `--set` override is named `*_custom` and left out of the leaderboard unless `--include-custom` is passed, which keeps ablations from mixing with the main comparison.

| Stage | Fixed settings | Why |
|---|---|---|
| Training | COCO-pretrained init, 100 epochs, **no early stopping** (`patience: 0`), 640 px, batch 16, AdamW lr0 = 0.001 (cosine off, lrf 0.01), 3 warm-up epochs, mosaic off for the last 10, seed 0, `deterministic: true`, AMP; all other augmentation = Ultralytics defaults (saved in `args.yaml`) | Identical budget and recipe; the optimizer is pinned rather than `auto` so the report can state it |
| Checkpoint | `best.pt` = highest val fitness (0.1·mAP50 + 0.9·mAP50-95) | Model selection uses val only; **test is never used for any choice** |
| Accuracy | conf 0.001, NMS IoU 0.7, max_det 300, FP32, 640 px; scored twice (below) | A low confidence floor keeps the full ranking, so AP is not artificially lowered |
| Speed | batch 1, 640 px, conf 0.25, images decoded in memory, 30 warm-up + 3 × 127 timed calls, FP32 and FP16; params/GFLOPs of the fused model | Deployment-like single-image latency; disk I/O excluded |

**Two scorers, on purpose:**

1. **COCO AP (primary)**: pycocotools `COCOeval` on `data/aquarium-v2-coco/{valid,test}.json`. This is the common evaluator and ground truth for all three benchmark models, so it is the number for cross-model tables.
2. **Ultralytics `model.val()`**: P, R, mAP50, mAP50-95 as YOLO papers and docs report them. Its AP integration differs slightly from COCOeval, so expect differences of about one point. A large gap means something is wrong.

**Test-label policy.** The original test labels contain two zero-area `shark` boxes (`DATA.md`). Ultralytics does *not* drop them, so they would count as guaranteed misses. `prepare` builds a derived YOLO view (`data/aquarium-v2-yolo/`) with hard-linked images (no extra disk). Its labels are identical except for **exactly those two records**, which is the same policy `scripts/data/yolo_to_coco.py` applies to the COCO JSON. The audit and the list of known records are imported from `scripts/data/`, not copied, and anything unexpected aborts the run. The original export is never modified, and Ultralytics' label caches are written into the derived view.

## 4. What each run produces

```
runs/<run>/                    (git-ignored, ~35 MB)
  weights/best.pt, last.pt     args.yaml, results.csv, *.png  (Ultralytics training outputs)
  experiment.yaml              resolved aquadet config + protocol hash
  train_info.json              wall time, best epoch
  eval/metrics.json            both scorers, val + test, per class
  eval/coco_predictions_*.json raw detections (COCO format) for error analysis
  eval/ultralytics_{val,test}/ PR curves, confusion matrices
  benchmark/benchmark.json     params, GFLOPs, latency per stage, environment
  summary.json
reports/runs/<run>.json        (tracked) compact copy of summary.json
reports/benchmark.{md,csv}     (tracked) generated by `compare`
```

## 5. Team workflow and fairness rules

1. **Same code, same config.** Pull `main` before running and never edit `configs/base.yaml` for one model. If the protocol has to change (for example, the lecturer asks for 50 epochs), change it in one PR and **re-run all three models**.
2. **Commit only `reports/runs/*.json`** (and code). Never commit `runs/`, `data/`, `dataset/`, `weights/`.
3. **Latency only compares on one GPU.** Accuracy is comparable across machines; latency is not. `compare` warns when benchmark rows come from different devices. To fix that, send the model owner who holds the benchmark GPU your `runs/<run>/` folder (zip of about 35 MB). They then run `python -m aquadet benchmark --run <run>` and commit the refreshed summary. The GPU on the reference machine is shared with another service, so close heavy jobs before benchmarking. `benchmark.json` records the GPU memory already in use.
4. **Check before you trust a number:** (a) the smoke run passes; (b) look at `runs/<run>/val_batch*_pred.jpg` against `*_labels.jpg`; (c) COCO AP and Ultralytics mAP50-95 agree within about 1–2 points; (d) the best epoch is not 1 or 100 (if it is 100, the model may still be improving; note it).
5. **Test once.** The pipeline scores test automatically at the end. Do not change settings after looking at test numbers. If a bug forces a change, re-run every model and say so in the report.

### Sanity checks already performed
- Ground truth fed back through the exact prediction-to-COCO conversion path gives **AP = 1.000** on val and test, which verifies class-ID offset, box format and image matching. Derived YOLO-view box counts equal the COCO JSON (909 val / 582 test).
- All three models complete the 1-epoch smoke pipeline on an RTX 3060 (Ultralytics 8.4.173, torch 2.14.1+cu130).
- FP16 is confirmed active (`torch.float16` weights) when `quantize=16`. At batch 1 nano models are launch-bound, so FP16 is not automatically faster.

## 6. Results

Each model has its own results file in this folder, named after the model, so owners never edit the same file. The generated cross-model table is [`reports/benchmark.md`](../../reports/benchmark.md).

| Model | Results file |
|---|---|
| YOLOv8n | [`YOLOV8N.md`](YOLOV8N.md) |
| YOLO11n | `YOLO11N.md` (to be added; copy the structure of `YOLOV8N.md`) |
| YOLOv10n | `YOLOV10N.md` (to be added; copy the structure of `YOLOV8N.md`) |

Cross-model interpretation will be added here once all three models are in.
