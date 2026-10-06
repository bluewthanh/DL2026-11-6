"""Zero-shot OWLv2 Aquarium v2 validation, using the shared COCO bbox protocol.

See docs/open_vocab/OWLV2.md for a pinned checkpoint, installation and reproduction commands.
Only valid.json is accepted; --limit 1/10 provides diagnostic subset runs.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

# validate_coco is in scripts/data/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from validate_coco import CATEGORIES, validate


MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
PROMPT_FILE = Path(__file__).with_name("aquarium_prompts.json")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predictions_for_image(result, image, score_floor, max_det):
    """Convert HF's xyxy/zero-based query labels to COCO xywh/IDs, top 300.

    No extra NMS: HF OWLv2 text-query post-processing does not apply it.
    One class (the highest scoring query) is selected for each image patch.
    """
    width, height = image["width"], image["height"]
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    labels = result["labels"].detach().cpu().tolist()
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("OWLv2 result tensor lengths differ")
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if (len(box) != 4 or type(label) is not int or not 0 <= label < len(CATEGORIES)
                or not math.isfinite(score) or not 0 <= score <= 1
                or not all(math.isfinite(v) for v in box)):
            raise ValueError(f"Invalid OWLv2 detection for image {image['id']}")
        # HF rescales xyxy to the original image; clamp numerical edge overshoot.
        x1, y1, x2, y2 = box
        x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
        y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
        if x2 <= x1 or y2 <= y1 or score < score_floor:
            continue
        detections.append({"image_id": image["id"], "category_id": label + 1,
                           "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    # Stable ties for reproducibility; COCOeval considers at most 100/image.
    return sorted(detections, key=lambda p: -p["score"])[:max_det]


def evaluate(gt, predictions, image_ids):
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        return {"status": "no predictions; AP is zero", "AP": 0.0, "AP50": 0.0,
                "per_class_AP": {name: 0.0 for name in CATEGORIES.values()},
                "iou": "0.50:0.95", "area": "all", "max_dets": 100}
    dt = gt.loadRes(predictions)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = image_ids
    ev.params.catIds = sorted(CATEGORIES)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    precision = ev.eval["precision"]
    per_class = {}
    for idx, category_id in enumerate(ev.params.catIds):
        valid = precision[:, :, idx, 0, -1]
        valid = valid[valid > -1]
        per_class[CATEGORIES[category_id]] = float(valid.mean()) if valid.size else None
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]),
            "per_class_AP": per_class, "iou": "0.50:0.95", "area": "all",
            "max_dets": int(ev.params.maxDets[-1])}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=Path("data/aquarium-v2-coco/valid.json"))
    parser.add_argument("--source", type=Path, default=Path("data/aquarium.v2-release.yolov8"))
    parser.add_argument("--output", type=Path, default=Path("results/owlv2_valid_bare"))
    parser.add_argument("--limit", type=int, help="Evaluate the first N sorted validation image IDs (diagnostic only)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--model-dir", type=Path, help="Local pinned snapshot with model.safetensors and processor files")
    parser.add_argument("--conf", type=float, default=0.001, help="Fixed pre-COCO score floor")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections/image before COCOeval")
    args = parser.parse_args(argv)
    if args.coco.name != "valid.json" or (args.limit is not None and args.limit < 1):
        parser.error("Only valid.json and a positive --limit are allowed; no test evaluation")
    if not (math.isfinite(args.conf) and 0 < args.conf < 1 and args.max_det >= 1):
        parser.error("Invalid inference settings")
    if args.output.exists():
        parser.error("Output directory exists; choose a fresh path")
    if not args.source.is_dir():
        parser.error("Source image root missing")
    return args


def main(argv=None):
    args = parse_args(argv)
    count = validate(args.coco)
    if count != (127, 909):
        raise ValueError(f"Unexpected validation COCO counts: {count}")

    import torch
    import transformers
    from PIL import Image
    from pycocotools.coco import COCO
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    gt = COCO(str(args.coco))
    if {cat["id"]: cat["name"] for cat in gt.dataset["categories"]} != CATEGORIES:
        raise ValueError("Unexpected category mapping")
    study = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    names = [CATEGORIES[i] for i in sorted(CATEGORIES)]
    if (study["category_id_to_name"] != {str(i): CATEGORIES[i] for i in sorted(CATEGORIES)}
            or study["bare"] != names):
        raise ValueError("COCO classes and bare prompt file disagree")
    # Match the project's bare-name study. HF expects a list of queries per image.
    prompts = names
    images = sorted(gt.dataset["images"], key=lambda image: image["id"])
    if args.limit is not None:
        images = images[:args.limit]
    ids = [image["id"] for image in images]
    source = args.source.resolve()
    paths = []
    for image in images:
        path = (source / image["file_name"]).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Missing or unsafe image path: {image['file_name']}")
        paths.append(path)
    if args.device != "cpu" and not args.device.startswith("cuda:"):
        raise ValueError("--device must be cpu or cuda:N")
    if args.device.startswith("cuda:") and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available")
    model_source = str(args.model_dir.resolve()) if args.model_dir else MODEL_ID
    if args.model_dir and not (args.model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing model.safetensors in {args.model_dir}")
    load_options = {"local_files_only": True} if args.model_dir else {"revision": MODEL_REVISION}
    processor = Owlv2Processor.from_pretrained(model_source, **load_options)
    model = Owlv2ForObjectDetection.from_pretrained(
        model_source, use_safetensors=True, **load_options).to(args.device).eval()
    config = {
        "split": "valid", "subset": "all" if args.limit is None else f"first {len(ids)} sorted image IDs",
        "image_ids": ids, "image_root": str(source), "coco_json": str(args.coco.resolve()),
        "coco_sha256": sha256(args.coco), "model": "OWLv2 base patch16 ensemble pretrained",
        "checkpoint": MODEL_ID, "revision": MODEL_REVISION, "weights_format": "safetensors",
        "model_source": model_source,
        "weights_sha256": sha256(args.model_dir / "model.safetensors") if args.model_dir else None,
        "fine_tuned": False, "prompt_vocabulary": "bare", "prompts_by_category_id": dict(CATEGORIES),
        "queries_in_order": prompts, "prompt_file": str(PROMPT_FILE.resolve()),
        "prompt_file_sha256": sha256(PROMPT_FILE),
        "inference": {"conf": args.conf, "max_det": args.max_det, "batch": 1,
                      "postprocess": "HF OWLv2; per-patch best query; no NMS; clip boxes; top scores",
                      "processor": "checkpoint default image size; original-size xyxy rescaling",
                      "half": False},
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "device": args.device, "cuda_available": torch.cuda.is_available()},
    }
    predictions, runtimes = [], []
    for position, (image, path) in enumerate(zip(images, paths), 1):
        start = time.perf_counter()
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        if rgb.size != (image["width"], image["height"]):
            raise ValueError(f"Image dimensions mismatch: {path}")
        inputs = processor(text=[prompts], images=rgb, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
        # target_sizes are (height, width), while PIL image.size is (width, height).
        target_sizes = torch.tensor([[image["height"], image["width"]]], device=args.device)
        result = processor.post_process_object_detection(
            outputs=output, target_sizes=target_sizes, threshold=args.conf)[0]
        detections = predictions_for_image(result, image, args.conf, args.max_det)
        predictions.extend(detections)
        if args.device.startswith("cuda:"):
            torch.cuda.synchronize(args.device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        runtimes.append({"image_id": image["id"], "elapsed_ms": elapsed_ms})
        print(f"[{position}/{len(images)}] image {image['id']}: {len(detections)} boxes, {elapsed_ms:.0f} ms", flush=True)
    metrics = evaluate(gt, predictions, ids)
    metrics.update(images_evaluated=len(ids), detections=len(predictions),
                   mean_wall_ms_per_image=sum(r["elapsed_ms"] for r in runtimes) / len(runtimes))
    args.output.mkdir(parents=True)  # only after successful inference and evaluation
    for filename, data in (("config.json", config), ("predictions.json", predictions),
                           ("metrics.json", metrics), ("runtime.json", runtimes)):
        (args.output / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved results in {args.output}")


if __name__ == "__main__":
    main()
