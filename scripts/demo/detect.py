"""Demo: text prompts in, labelled boxes out (YOLOE-26s), or the closed-set YOLOv8n baseline.

Run in the YOLOE environment (ultralytics==8.4.172; see docs/open_vocab/YOLOE.md):

    # Open-vocabulary: any comma-separated text becomes the class list for this call
    python scripts/demo/detect.py --prompts "fish, sea jelly, penguin" --image path/to/image.jpg

    # Closed-set baseline: fixed Aquarium classes learned in training, no text input
    python scripts/demo/detect.py --weights runs/yolov8n_e100_s0/weights/best.pt --image path/to/image.jpg

Writes <out>/<image stem>_<tag>.jpg with labelled boxes and <out>/<image stem>_<tag>.json with
the prompts (or class names), settings and every box. This is a qualitative demo, not an evaluation.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Use the repository's Ultralytics settings (weights dir = ROOT/weights), not the machine-wide ones.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def load_model(args):
    if args.prompts:
        from ultralytics import YOLOE
        from ultralytics.utils import WEIGHTS_DIR

        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        if not prompts:
            sys.exit("--prompts needs at least one non-empty phrase")
        if not (WEIGHTS_DIR / "mobileclip2_b.ts").is_file():
            sys.exit(f"{WEIGHTS_DIR / 'mobileclip2_b.ts'} missing; see docs/open_vocab/YOLOE.md")
        model = YOLOE(str(args.checkpoint))
        model.set_classes(prompts)
        tag = "yoloe_" + re.sub(r"[^a-z0-9]+", "-", "_".join(prompts).lower()).strip("-")[:60]
        return model, {"mode": "open-vocabulary text prompts", "model": args.checkpoint.name,
                       "prompts": prompts}, tag
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    return model, {"mode": "closed-set (fixed trained classes)", "model": str(args.weights),
                   "classes": list(model.names.values())}, "closed_set"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--prompts", help="Comma-separated text prompts for YOLOE")
    which.add_argument("--weights", type=Path, help="Closed-set YOLO checkpoint, e.g. YOLOv8n best.pt")
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights" / "yoloe-26s-seg.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Display threshold (default 0.25)")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default=None, help="e.g. cpu or cuda:0 (default: Ultralytics' choice)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "06_demo" / "outputs")
    args = parser.parse_args()

    model, meta, tag = load_model(args)
    args.out.mkdir(parents=True, exist_ok=True)
    for image in args.image:
        # Same NMS as the evaluation runs. agnostic_nms must be passed explicitly: when it is
        # omitted, YOLOE suppresses overlapping boxes of different classes.
        result = model.predict(str(image), conf=args.conf, iou=args.iou, imgsz=args.imgsz, max_det=args.max_det,
                               agnostic_nms=False, device=args.device, verbose=False)[0]
        stem = f"{image.stem[:24]}_{tag}"
        result.save(filename=str(args.out / f"{stem}.jpg"), masks=False)
        boxes = [{"label": result.names[int(c)], "score": round(float(s), 4),
                  "xyxy": [round(v, 1) for v in b]}
                 for b, s, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())]
        record = {**meta, "image": str(image), "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                  "max_det": args.max_det, "agnostic_nms": False,
                  "detections": sorted(boxes, key=lambda b: -b["score"])}
        (args.out / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        summary = ", ".join(f"{b['label']} {b['score']:.2f}" for b in record["detections"][:8]) or "no boxes"
        print(f"{image.name}: {len(boxes)} boxes ({summary}) -> {args.out / (stem + '.jpg')}")


if __name__ == "__main__":
    main()
