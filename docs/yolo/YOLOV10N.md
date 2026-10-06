# YOLOv10n — closed-set fine-tuning results

**Status:** full run complete (100 epochs, seed 0); val and test evaluated once; benchmarked. Protocol, setup and team rules are in [`README.md`](README.md); this file covers YOLOv10n only. The cross-model table is generated in [`reports/benchmark.md`](../../reports/benchmark.md).

| Item | Value |
|---|---|
| Run | `yolov10n_e100_s0` (summary: [`reports/runs/yolov10n_e100_s0.json`](../../reports/runs/yolov10n_e100_s0.json)) |
| Init | official COCO-pretrained `yolov10n.pt` (Ultralytics) |
| Config | [`configs/models/yolov10n.yaml`](../../configs/models/yolov10n.yaml) + shared [`configs/base.yaml`](../../configs/base.yaml), protocol hash `42534e32b914` |
| Environment | RTX 3060 12 GB (shared with another service), torch 2.14.1+cu130, Ultralytics 8.4.173, Python 3.12 |
| Date | 6 Oct 2026 |

## Reproduce

```bash
python -m aquadet run --model yolov10n         # train -> eval val/test -> benchmark -> summary (~7 min on an RTX 3060)
```

Outputs go to `runs/yolov10n_e100_s0/` (git-ignored). Use `--name` for a new run directory if that one exists.

## Accuracy

| Metric (×100) | Val | Test |
|---|---:|---:|
| COCO AP@[.5:.95] | 43.0 | **47.0** |
| COCO AP50 | 72.6 | 77.8 |
| COCO AP75 | – | 50.1 |
| Ultralytics mAP50-95 / mAP50 | 44.8 / 74.8 | 47.8 / 79.2 |
| Ultralytics P / R (test) | – | 87.5 / 68.6 |

COCO AP uses pycocotools on the derived COCO ground truth, the same evaluator as the YOLOE/OWLv2 runs. Test has 63 images and 582 boxes after omitting the two zero-area shark boxes (see `DATA.md`).

| Per-class AP@[.5:.95] (×100) | fish | jellyfish | penguin | puffin | shark | starfish | stingray |
|---|---:|---:|---:|---:|---:|---:|---:|
| Test (GT boxes) | 42.8 (249) | 52.3 (154) | 33.8 (82) | 23.0 (35) | 61.0 (36) | 55.4 (11) | 61.0 (15) |
| Val (GT boxes) | 44.5 (459) | 50.0 (155) | 28.8 (104) | 23.2 (74) | 42.8 (57) | 57.9 (27) | 54.0 (33) |

## Size and speed

2.27 M params, 6.6 GFLOPs (fused), batch 1, 640 px, images pre-loaded in memory, 30 warm-up + 3 × 127 timed calls.

| Precision | Inference (ms) | End-to-end (ms) |
|---|---:|---:|
| FP32 | 5.35 | 7.63 |
| FP16 | 5.89 | 8.19 |

YOLOv10n has fewer parameters and GFLOPs than YOLOv8n but is about 1 ms slower per image on this GPU. At batch 1 a nano model is limited by kernel-launch overhead rather than arithmetic, so FLOPs do not predict latency. FP16 is not faster for the same reason. Compare these latencies only with models timed on the same GPU.

## Observations

- **Training:** 6.8 min, best epoch 83/100. Val mAP plateaus from about epoch 80, and val classification loss is flat from about epoch 70. The 100-epoch budget is therefore sufficient; `best.pt` is chosen on val.
- **Against YOLOv8n (single seed each):** test AP is within 0.6 points (47.0 vs 47.6), while val AP is 2.4 points lower (43.0 vs 45.4). YOLOv10n has higher precision but lower recall on test (P 87.5 / R 68.6 vs 81.5 / 72.2). Without more seeds, these gaps cannot be separated from run-to-run noise.
- **Weak classes:** puffin and penguin, as for YOLOv8n. Puffin recall on test is low (36%). The rare classes (starfish, stingray) score high, but with 11 and 15 test boxes their AP is noisy.
- **Checks:** the two scorers agree within about 2 points on val (43.0 vs 44.8) and 1 point on test. The visual check of `runs/yolov10n_e100_s0/val_batch0_pred.jpg` against `val_batch0_labels.jpg` shows mostly correctly placed and labelled boxes. Two clear errors: the top prediction on the shark image in that batch is `fish` (0.9), and one image has an extra `puffin` (0.5) box covering most of the scene.
- **Limitations:** a single seed, so there is no variance estimate yet (`--seed 1`, `--seed 2` would add it in about 7 min each).
