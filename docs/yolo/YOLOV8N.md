# YOLOv8n — closed-set fine-tuning results

**Status:** full run complete (100 epochs, seed 0); val and test evaluated once; benchmarked. Protocol, setup and team rules are in [`README.md`](README.md); this file covers YOLOv8n only. The cross-model table is generated in [`reports/benchmark.md`](../../reports/benchmark.md).

| Item | Value |
|---|---|
| Run | `yolov8n_e100_s0` (summary: [`reports/runs/yolov8n_e100_s0.json`](../../reports/runs/yolov8n_e100_s0.json)) |
| Init | official COCO-pretrained `yolov8n.pt` (Ultralytics) |
| Config | [`configs/models/yolov8n.yaml`](../../configs/models/yolov8n.yaml) + shared [`configs/base.yaml`](../../configs/base.yaml), protocol hash `42534e32b914` |
| Environment | RTX 3060 12 GB (shared with another service), torch 2.14.1+cu130, Ultralytics 8.4.173, Python 3.12 |
| Date | 6 Oct 2026 |

## Reproduce

```bash
python -m aquadet run --model yolov8n          # train -> eval val/test -> benchmark -> summary (~7 min on an RTX 3060)
```

Outputs go to `runs/yolov8n_e100_s0/` (git-ignored). Use `--name` for a new run directory if that one exists.

## Accuracy

| Metric (×100) | Val | Test |
|---|---:|---:|
| COCO AP@[.5:.95] | 45.4 | **47.6** |
| COCO AP50 | 75.8 | 76.1 |
| COCO AP75 | – | 54.3 |
| Ultralytics mAP50-95 / mAP50 | 46.4 / 76.9 | 47.9 / 78.5 |
| Ultralytics P / R (test) | – | 81.5 / 72.2 |

COCO AP uses pycocotools on the derived COCO ground truth, the same evaluator as the YOLOE/OWLv2 runs. Test has 63 images and 582 boxes after omitting the two zero-area shark boxes (see `DATA.md`).

| Per-class AP@[.5:.95] (×100) | fish | jellyfish | penguin | puffin | shark | starfish | stingray |
|---|---:|---:|---:|---:|---:|---:|---:|
| Test (GT boxes) | 41.8 (249) | 57.4 (154) | 31.6 (82) | 22.4 (35) | 56.5 (36) | 59.7 (11) | 64.0 (15) |
| Val (GT boxes) | 43.8 (459) | 53.5 (155) | 31.2 (104) | 24.9 (74) | 48.5 (57) | 55.3 (27) | 60.5 (33) |

## Size and speed

3.01 M params, 8.1 GFLOPs (fused), batch 1, 640 px, images pre-loaded in memory, 30 warm-up + 3 × 127 timed calls.

| Precision | Inference (ms) | End-to-end (ms) |
|---|---:|---:|
| FP32 | 4.40 | 6.57 |
| FP16 | 4.89 | 7.17 |

FP16 is *not* faster here. At batch 1 the nano model is limited by kernel-launch overhead rather than arithmetic, and the GPU was shared with another service. Compare these latencies only with models timed on the same GPU.

## Observations

- **Training:** 5.2 min, best epoch 86/100. Val mAP plateaus from about epoch 80, while val loss has been flat since about epoch 50. The 100-epoch budget is therefore sufficient, with mild late overfitting; `best.pt` is chosen on val.
- **Weak classes:** puffin and penguin. They are small, clustered and often in poor light. The rare classes (starfish, stingray) score high, but with 11 and 15 test boxes their AP is noisy.
- **Checks:** the two scorers agree within about 1 point on both splits, and the visual check of `runs/yolov8n_e100_s0/val_batch*_pred.jpg` shows correctly placed and labelled boxes.
- **Limitations:** a single seed, so there is no variance estimate yet (`--seed 1`, `--seed 2` would add it in about 5 min each).
