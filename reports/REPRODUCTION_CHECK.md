# README reproduction walkthrough — Windows CPU (2026-10-07)

**Scope:** Executed on this checkout using newly created, separate Python 3.12.15 virtual environments on Windows 11, Intel i5-7300HQ (4 logical CPUs), **no CUDA GPU**. This is a new-environment walkthrough on the existing machine and locally available original export/weights, **not** an independent second person's download of every input or a reproduction of GPU training/latency. Reported GPU experiments remain documented under `reports/00_setup/`. Local commands write only Git-ignored `data/`, `results/`, `runs/`, `weights/` and virtual-environment files. The original export was not changed.

## Installation and inputs

- Used `python -m uv venv .venv-yoloe --python 3.12 --managed-python`, then installed the packages from the [README YOLOE command](../README.md#primary-experiment-names-synonyms-and-descriptions). Verified torch **2.14.1+cpu**, Ultralytics **8.4.172**, pycocotools **2.0.11**, and the pinned CLIP dependency.
- Used `python -m uv venv .venv --python 3.12 --managed-python`, then `python -m uv pip install --python .venv/Scripts/python.exe -r requirements-yolo.txt`. Verified torch **2.14.1+cpu**, Ultralytics **8.4.173**, pycocotools **2.0.11**. `uv` can be installed per https://docs.astral.sh/uv/getting-started/installation/; on Windows replace the README's `.venv/bin/python` with `.venv/Scripts/python.exe`.
- Original Aquarium v2 YOLOv8 export was already available locally at `data/aquarium.v2-release.yolov8/`; weight files were downloaded from the links in `docs/open_vocab/YOLOE.md`. YOLOE checkpoint and MobileCLIP2 encoder hashes matched that document. The pretrained `yolov8n.pt` was downloaded from `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt`; SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` matches `reports/00_setup/weights_sha256.txt`.
- Set `YOLO_CONFIG_DIR` to the absolute `./.ultralytics` path, created a writable `.ultralytics/Ultralytics` folder and verified Ultralytics' `WEIGHTS_DIR` points at this checkout's `weights/` with `mobileclip2_b.ts` present. No credentials were stored.

## Dataset and tests

- Regenerated COCO with `python scripts/data/yolo_to_coco.py --source data/aquarium.v2-release.yolov8 --output data/README_regenerated_coco_20261007 --invalid-policy omit-known-zero-boxes`; validated with `python scripts/data/validate_coco.py --coco-dir data/README_regenerated_coco_20261007`. All splits passed (train **448/3328**, valid **127/909**, test **63/582** images/boxes). Train/valid/test SHA-256 hashes after CRLF→LF normalization matched the earlier derived COCO exactly (`366f8ebc…`, `684400c8…`, `22e8246e…`). The two known zero-area **test** boxes were omitted only in derived labels.
- Downloaded and verified the [public annotations-only ZIP](https://github.com/bluewthanh/DL2026-11-6/releases/download/aquarium-v2-derived-coco-v1/aquarium-v2-derived-coco.zip), SHA-256 `1a30695ce8b6ee475f195ba3f0a9e8eb300b2da8baf7946749295e8c56cfe0ea`; extracted to a temporary local folder and validated all splits again.
- Ran `.venv/Scripts/python.exe -m unittest discover -s tests -q`: **13 passed**. A Windows-only test-reading issue was fixed by decoding generated UTF-8 explicitly in `tests/test_aquadet.py`.
- Ran `.venv/Scripts/python.exe -m aquadet prepare --source data/aquarium.v2-release.yolov8`: the first build reached the final `.partial` directory rename but Windows briefly returned `PermissionError [WinError 5]`; manually renaming the completed staging directory allowed a second `prepare` to pass with hardlinks and the correct split counts. This may be caused by local file locking and **means a first-attempt Windows prepare is not guaranteed**; repeat only after checking the staging manifest/counts, or rerun after a short wait. This did not affect the source export or COCO contents.

## Main experiment — full YOLOE validation on CPU

Ran all three full (127-image, **no `--limit`**) README prompt-study commands with `.venv-yoloe/Scripts/python.exe`, `--source data/aquarium.v2-release.yolov8`, and new local output paths `results/readme_clean312_{bare,synonym,description}`. Measured COCO bbox AP@[0.50:0.95] on a 0–100 scale:

| Prompt set | New CPU AP | Tracked GPU reproduction AP | Absolute difference (AP points) |
|---|---:|---:|---:|
| Bare names | 13.1086 | 13.1081 | 0.0005 |
| Two-synonym variant | 12.7709 | 12.7719 | 0.0009 |
| Descriptions | 5.7589 | 5.7658 | 0.0070 |

All three used the **same 127 image IDs**, COCO category mapping, inference settings, checkpoint and text-encoder hashes as the tracked GPU runs. The prompt-definition file's SHA-256 differs in raw bytes on Windows due to CRLF; CRLF→LF normalized SHA-256 equals the stored `701eb58f02caa3d1fc8d905e821baaf7096914c13f78433cd51bfc0d452f34ab`. Small AP and detection-count changes between CPU and GPU do not affect the published one-decimal AP values. New full-run results live only in Git-ignored `results/`; the tracked GPU artifacts remain the source for the report tables.

## Baseline and demo

- Ran `.venv/Scripts/python.exe -m aquadet run --model yolov8n --smoke --source data/aquarium.v2-release.yolov8 --device cpu`. The **one-epoch** run trained on 448 images, scored all 127 validation and 63 test images, performed a short CPU latency benchmark, and wrote `runs/yolov8n_e1_s0_smoke/summary.json` (not exported to tracked `reports/runs/`). It is *only* a pipeline check: val COCO AP was **0.0041** (0–1 scale), not the 100-epoch report result. Its CPU latency is not comparable with the recorded RTX 3060 latency.
- Ran `.venv-yoloe/Scripts/python.exe scripts/demo/detect.py --prompts "fish, sea star" --image <one official validation image> --device cpu --out results/README_demo_clean312`; it wrote a labelled `.jpg` and detection `.json` (one `sea star` 0.58 box).
- Re-scored the saved **100-epoch YOLOv8n `last.pt` validation predictions** plus the newly generated three YOLOE prediction files with `scripts/analysis/build_tables.py`, writing only to Git-ignored `results/`. The resulting AP rows round to **44.4 / 13.1 / 12.8 / 5.8**, matching the tracked Setup 1 report. Rebuilt predeclared and post-hoc Setup 3 tables with `scripts/yolo/data_efficiency.py summarize` from tracked run records, writing to ignored `results/`; the key rows remained **9.9 ± 1.8** (22 images), **19.2 ± 2.7** (45 images), **44.7 ± 0.7** (448 images).

**Not rerun:** a fresh **100-epoch, multi-seed YOLOv8n training/data-efficiency study**, full GPU inference, or GPU latency. This machine has no CUDA device. Those results have saved predictions, configs, curves and summaries under `reports/`; another member checked saved artifacts in `reports/VERIFICATION.md`. This README walkthrough verifies the main **text-prompt** inference result in clean environments and the baseline train→evaluate pipeline, but does not prove a second-machine end-to-end training reproduction. No one should claim that it does.
