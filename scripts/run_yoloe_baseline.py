"""Pretrained YOLOE-26s text-prompted Aquarium COCO validation baseline.

Requires ultralytics==8.4.172, pycocotools, torch, and Ultralytics/CLIP.
See EXPERIMENTS.md for installation, checkpoint and text encoder sources.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ultralytics
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLOE

from validate_coco import CATEGORIES, validate


PROMPT_FILE = Path(__file__).with_name("aquarium_prompts.json")
STUDY = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
PROMPTS = STUDY["bare"]
if STUDY["category_id_to_name"] != {str(i): CATEGORIES[i] for i in sorted(CATEGORIES)} or PROMPTS != list(CATEGORIES.values()):
    raise RuntimeError("Prompt vocabulary and validated COCO category order differ")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(gt, predictions, ids):
    # COCO.loadRes does not handle empty lists on all pycocotools versions.
    if not predictions:
        return {"status": "no predictions; AP is zero", "AP": 0.0, "AP50": 0.0,
                "per_class_AP": {name: 0.0 for name in PROMPTS}}
    dt = gt.loadRes(predictions)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = ids
    ev.params.catIds = sorted(CATEGORIES)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    # Precision dimensions: IoU, recall, category, area, maxDet. -1 = unavailable.
    precision = ev.eval["precision"]
    per_class = {}
    for index, category_id in enumerate(ev.params.catIds):
        values = precision[:, :, index, 0, -1]
        valid = values[values > -1]
        per_class[CATEGORIES[category_id]] = float(valid.mean()) if valid.size else None
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "per_class_AP": per_class,
            "iou": "0.50:0.95", "area": "all", "max_dets": int(ev.params.maxDets[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/yoloe-26s-seg.pt"))
    parser.add_argument("--coco", type=Path, default=Path("data/aquarium-v2-coco/valid.json"))
    parser.add_argument("--source", type=Path, default=Path("data/aquarium.v2-release.yolov8"))
    parser.add_argument("--output", type=Path, default=Path("results/yoloe26s_valid_bare"))
    parser.add_argument("--limit", type=int, default=None, help="Validation smoke run only; evaluate this subset")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="Fixed low cutoff for AP; not tuned on test")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    if args.coco.name != "valid.json" or args.limit is not None and args.limit < 1:
        parser.error("Only valid.json and a positive --limit are allowed (no test evaluation)")
    if not (0 < args.conf < 1 and 0 < args.iou < 1 and args.imgsz > 0 and args.max_det >= 1):
        parser.error("Invalid inference settings")
    if not args.checkpoint.is_file() or args.checkpoint.name != "yoloe-26s-seg.pt":
        parser.error("Provide the pretrained text-promptable yoloe-26s-seg.pt (not prompt-free)")
    if not args.source.is_dir():
        parser.error("Source image root missing")
    if args.output.exists():
        parser.error("Output directory exists; choose a new path to avoid overwriting results")
    count = validate(args.coco)
    if count[0] != 127 or count[1] != 909:
        parser.error("Unexpected validation COCO counts")
    gt = COCO(str(args.coco))
    if {item["id"]: item["name"] for item in gt.dataset["categories"]} != CATEGORIES:
        parser.error("Unexpected category mapping")
    images = sorted(gt.dataset["images"], key=lambda image: image["id"])
    if args.limit:
        images = images[:args.limit]
    ids = [image["id"] for image in images]
    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    model = YOLOE(str(checkpoint))
    if type(model.model).__name__ != "YOLOESegModel" or getattr(model.model, "text_model", None) != "mobileclip2:b":
        raise RuntimeError("Checkpoint is not the expected YOLOE-26s MobileCLIP2 text-prompt model")
    model.to(args.device)
    # Check the Ultralytics weights directory so an absent encoder cannot silently
    # trigger a download during the baseline run.
    from ultralytics.utils import WEIGHTS_DIR
    encoder = WEIGHTS_DIR / "mobileclip2_b.ts"
    if not encoder.is_file():
        raise FileNotFoundError(f"{encoder} missing; see EXPERIMENTS.md")
    model.set_classes(PROMPTS)
    if [model.names[i] for i in range(7)] != PROMPTS:
        raise RuntimeError("YOLOE prompt-index mapping mismatch")
    config = {
        "split": "valid", "subset": "all" if args.limit is None else f"first {len(ids)} sorted image IDs",
        "image_ids": ids, "image_root": str(source), "coco_json": str(args.coco.resolve()),
        "coco_sha256": sha256(args.coco), "model": "YOLOE-26s (pretrained segmentation checkpoint; bbox output)",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "text_encoder": "mobileclip2:b (mobileclip2_b.ts)", "text_encoder_sha256": sha256(encoder),
        "fine_tuned": False,
        "prompt_vocabulary": "bare", "prompts_by_category_id": dict(CATEGORIES),
        "prompt_file": str(PROMPT_FILE.resolve()), "prompt_file_sha256": sha256(PROMPT_FILE),
        "prompt_study_vocabularies_not_run": STUDY,
        "inference": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
                      "max_det": args.max_det, "agnostic_nms": False, "batch": 1, "half": False,
                      "augment": False, "task": "segment; bbox predictions only"},
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__, "device_requested": args.device,
                    "model_device": str(next(model.model.parameters()).device),
                    "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads()},
    }
    predictions = []
    runtimes = []
    for position, image in enumerate(images, 1):
        relative = Path(image["file_name"])
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Missing or unsafe image path: {relative}")
        start = time.perf_counter()
        result = model.predict(source=str(path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                               max_det=args.max_det, device=args.device, agnostic_nms=False,
                               augment=False, verbose=False, save=False)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        runtimes.append({"image_id": image["id"], "elapsed_ms": elapsed_ms,
                         "ultralytics_speed_ms": result.speed})
        if tuple(result.orig_shape) != (image["height"], image["width"]):
            raise RuntimeError(f"Image dimensions mismatch: {relative}")
        boxes = result.boxes
        if boxes is not None:
            for xyxy, score, cls in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                                        boxes.cls.cpu().tolist()):
                cls_index = int(cls)
                x1, y1, x2, y2 = map(float, xyxy)
                if not (0 <= cls_index < 7 and math.isfinite(score) and 0 <= score <= 1
                        and all(math.isfinite(v) for v in xyxy) and x2 > x1 and y2 > y1):
                    raise RuntimeError(f"Invalid model prediction on {relative}")
                predictions.append({"image_id": image["id"], "category_id": cls_index + 1,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
        print(f"[{position}/{len(images)}] image {image['id']}: {0 if boxes is None else len(boxes)} boxes, {elapsed_ms:.0f} ms", flush=True)
    metrics = evaluate(gt, predictions, ids)
    metrics["images_evaluated"] = len(ids)
    metrics["detections"] = len(predictions)
    metrics["mean_wall_ms_per_image"] = float(np.mean([r["elapsed_ms"] for r in runtimes]))
    args.output.mkdir(parents=True)  # write only after successful inference + evaluation
    for filename, data in (("config.json", config), ("predictions.json", predictions),
                           ("metrics.json", metrics), ("runtime.json", runtimes)):
        (args.output / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved results in {args.output}")


if __name__ == "__main__":
    main()
