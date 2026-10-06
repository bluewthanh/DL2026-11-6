# Setup and environment record

Everything in `reports/` was produced on one machine on 2026-10-06: Linux, an NVIDIA RTX 3060 12 GB, and 16 CPU threads. Two separate Python 3.12 environments are needed, because the two model families pin different Ultralytics versions.

| Environment | Used for | Key packages | Full list |
|---|---|---|---|
| `.venv` | aquadet benchmark, YOLOv8n training and evaluation, Setup 3, tables, overlays | torch 2.14.1+cu130, ultralytics **8.4.173**, pycocotools 2.0.11 | `benchmark_env_freeze.txt`, `aquadet_env.json` |
| `.venv-yoloe` | YOLOE prompt runs (Setup 2) and the demo | torch 2.14.1+cu130, ultralytics **8.4.172**, pycocotools 2.0.11, Ultralytics/CLIP @ a13192f | `yoloe_env_freeze.txt` |

```bash
# benchmark env (README "Supplementary closed-set YOLO benchmark")
uv venv .venv --python 3.12 --managed-python
uv pip install --python .venv/bin/python -r requirements-yolo.txt
# YOLOE env (docs/open_vocab/YOLOE.md); uv puts a .gitignore inside, so it is never committed
uv venv .venv-yoloe --python 3.12 --managed-python
uv pip install --python .venv-yoloe/bin/python torch==2.14.1 torchvision==0.29.1 ultralytics==8.4.172 \
  pycocotools==2.0.11 'git+https://github.com/ultralytics/CLIP.git@a13192f8cb767260d7dfd98c843b0716593169e7'
```

When running the YOLOE scripts directly, set `YOLO_CONFIG_DIR=$PWD/.ultralytics` so that Ultralytics looks for `mobileclip2_b.ts` in this repository's `weights/` folder and not in the machine-wide settings. `aquadet` and `scripts/demo/detect.py` set it automatically.

## Checks run

| Check | Result | File |
|---|---|---|
| Unit tests, both environments | 13 tests, OK in each | `unittests_benchmark_env.txt`, `unittests_yoloe_env.txt` |
| Derived COCO ground truth | train 448/3328, valid 127/909, test 63/582 images/boxes; all checks pass | `validate_coco.txt` |
| `python -m aquadet prepare` | Hard-linked YOLO view OK; exactly the 2 known zero-area test boxes omitted | `aquadet_prepare.txt` |
| Weights | YOLOE checkpoint and text encoder SHA-256 match `docs/open_vocab/YOLOE.md`; YOLOv8n checkpoints hashed | `weights_sha256.txt` |

The original export is at `dataset/aquarium.v2-release.yolov8/` (the README's `data/aquarium.v2-release.yolov8` path is passed with `--source`). Images, labels and weights are Git-ignored. The run outputs needed to read or rebuild every result are copied into `reports/` (see `../README.md`).
