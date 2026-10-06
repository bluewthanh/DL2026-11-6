"""Score a checkpoint with Ultralytics `model.val()` and with pycocotools.

COCOeval on the derived COCO JSON is the primary metric for all three benchmark models.
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from aquadet.dataset import COCO_SPLIT

COCO_STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def xyxy_to_coco(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def ultralytics_val(model, data_yaml: Path, split: str, settings: dict, out_dir: Path, device: str) -> dict:
    metrics = model.val(data=str(data_yaml), split=split, imgsz=settings["imgsz"], batch=settings["batch"],
                        conf=settings["conf"], iou=settings["iou"], max_det=settings["max_det"],
                        quantize=settings["quantize"], device=device, plots=True, project=str(out_dir),
                        name=f"ultralytics_{split}", exist_ok=True, verbose=False)
    box = metrics.box
    per_class = {}
    for i, class_id in enumerate(metrics.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[metrics.names[int(class_id)]] = {"P": float(p), "R": float(r), "AP50": float(ap50), "AP50-95": float(ap)}
    return {"P": float(box.mp), "R": float(box.mr), "mAP50": float(box.map50), "mAP75": float(box.map75),
            "mAP50-95": float(box.map), "per_class": per_class,
            "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()}}


def predict_coco(model, coco_gt, image_root: Path, settings: dict, device: str) -> list[dict]:
    """COCO detections for every GT image; category_id = YOLO class + 1."""
    detections = []
    for image_id in sorted(coco_gt.getImgIds()):
        info = coco_gt.imgs[image_id]
        result = model.predict(str(image_root / info["file_name"]), imgsz=settings["imgsz"], conf=settings["conf"],
                               iou=settings["iou"], max_det=settings["max_det"], quantize=settings["quantize"],
                               device=device, verbose=False)[0]
        if result.orig_shape != (info["height"], info["width"]):
            raise ValueError(f"Image {info['file_name']} shape {result.orig_shape} != COCO "
                             f"{(info['height'], info['width'])}")
        boxes = result.boxes
        for xyxy, cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            detections.append({"image_id": image_id, "category_id": int(cls) + 1,
                               "bbox": [round(v, 2) for v in xyxy_to_coco(xyxy)], "score": round(float(score), 5)})
    return detections


def per_class_ap(coco_eval, coco_gt) -> dict:
    """precision is [IoU, recall, class, area, maxDets]; uses area=all, maxDets=100."""
    precision = coco_eval.eval["precision"]
    result = {}
    for k, cat_id in enumerate(coco_eval.params.catIds):
        name = coco_gt.cats[cat_id]["name"]
        all_iou = precision[:, :, k, 0, -1]
        iou50 = precision[0, :, k, 0, -1]
        result[name] = {"AP": float(np.mean(all_iou[all_iou > -1])) if (all_iou > -1).any() else float("nan"),
                        "AP50": float(np.mean(iou50[iou50 > -1])) if (iou50 > -1).any() else float("nan"),
                        "gt_boxes": len(coco_gt.getAnnIds(catIds=[cat_id]))}
    return result


def coco_eval(model, coco_json: Path, image_root: Path, settings: dict, out_dir: Path, device: str) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(coco_json))
    detections = predict_coco(model, coco_gt, image_root, settings, device)
    (out_dir / f"coco_predictions_{coco_json.stem}.json").write_text(json.dumps(detections), encoding="utf-8")
    if not detections:
        return {name: 0.0 for name in COCO_STAT_NAMES} | {"detections": 0, "per_class": {}}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (out_dir / f"cocoeval_{coco_json.stem}.txt").write_text(log.getvalue(), encoding="utf-8")
    stats = {name: float(v) for name, v in zip(COCO_STAT_NAMES, evaluator.stats)}
    return stats | {"detections": len(detections), "images": len(coco_gt.getImgIds()),
                    "gt_boxes": len(coco_gt.getAnnIds()), "per_class": per_class_ap(evaluator, coco_gt)}


def evaluate(weights: Path, prepared: dict, settings: dict, splits: list[str], out_dir: Path, device: str) -> dict:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in splits:
        model = YOLO(str(weights))
        print(f"[evaluate] {weights.name} on {split}")
        native = ultralytics_val(model, prepared["data_yaml"], split, settings, out_dir, device)
        coco = coco_eval(YOLO(str(weights)), prepared["coco_dir"] / f"{COCO_SPLIT[split]}.json",
                         prepared["view"], settings, out_dir, device)
        results[split] = {"ultralytics": native, "coco": coco}
        print(f"[evaluate] {split}: COCO AP={coco['AP']:.4f} AP50={coco['AP50']:.4f} | "
              f"Ultralytics mAP50-95={native['mAP50-95']:.4f} mAP50={native['mAP50']:.4f}")
    # Keep splits scored by earlier calls.
    metrics_path = out_dir / "metrics.json"
    previous = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    if previous and (previous.get("weights") != str(weights) or previous.get("settings") != settings):
        previous = {}
    payload = {"weights": str(weights), "settings": settings, "results": previous.get("results", {}) | results}
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
