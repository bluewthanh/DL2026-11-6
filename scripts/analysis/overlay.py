"""Paired ground-truth / prediction overlays and error breakdown on the same validation images.

Each --pred is NAME=path/to/coco_predictions.json (COCO results format, category IDs 1-7).
For every selected image one figure is written with a ground-truth panel followed by one
panel per model. Predictions are shown at a per-model display threshold: by default the
score that maximises F1 at IoU 0.5 over the whole validation split (recorded in the output),
or a fixed value with --thr NAME=0.3. AP is unaffected by this display threshold.

Box colours in prediction panels:
  green  = correct (same class, IoU >= 0.5)       orange = right place, wrong class
  yellow = right class, poor box (0.1 <= IoU < 0.5) purple = duplicate of a matched object
  red    = background false positive             blue dashed = missed ground truth

    python scripts/analysis/overlay.py \
        --pred YOLOv8n=reports/03_setup3_data_efficiency/eval/yolov8n_e100_s0/best/coco_predictions_valid.json.gz \
        --pred "YOLOE bare=reports/02_setup2_prompt_study/runs/bare/predictions.json.gz" \
        --contrast "YOLOv8n:YOLOE bare" --out reports/05_qualitative_overlays
"""

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ERROR_TYPES = ("correct", "wrong_class", "poor_box", "duplicate", "background", "missed")
COLORS = {"gt": (0, 200, 0), "correct": (0, 200, 0), "wrong_class": (255, 140, 0),
          "poor_box": (230, 210, 0), "duplicate": (170, 60, 220), "background": (230, 20, 20),
          "missed": (30, 120, 255)}
PANEL_WIDTH = 480


def load_predictions(path) -> list[dict]:
    """COCO results JSON, plain or gzip-compressed (.json.gz, as stored in reports/)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xywh boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    w = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(ax1[:, None], bx1), 0, None)
    h = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(ay1[:, None], by1), 0, None)
    inter = w * h
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def classify(gts: list[dict], dets: list[dict]) -> tuple[list[str], list[bool], list[dict]]:
    """Greedy, score-ordered labelling; returns detection labels, GT-matched flags and the ordered detections."""
    dets = sorted(dets, key=lambda d: -d["score"])
    g = np.array([x["bbox"] for x in gts], dtype=float).reshape(-1, 4)
    d = np.array([x["bbox"] for x in dets], dtype=float).reshape(-1, 4)
    ious = iou_matrix(d, g)
    g_cls = np.array([x["category_id"] for x in gts])
    matched = [False] * len(gts)
    labels = []
    for i, det in enumerate(dets):
        same = g_cls == det["category_id"] if len(gts) else np.zeros(0, bool)
        candidates = [j for j in np.argsort(-ious[i]) if same[j] and not matched[j] and ious[i, j] >= 0.5]
        if candidates:
            matched[candidates[0]] = True
            labels.append("correct")
        elif len(gts) and (ious[i][same] >= 0.5).any():
            labels.append("duplicate")
        elif len(gts) and (ious[i][~same] >= 0.5).any():
            labels.append("wrong_class")
        elif len(gts) and (ious[i][same] >= 0.1).any():
            labels.append("poor_box")
        else:
            labels.append("background")
    return labels, matched, dets


def f1_threshold(gt_by_image: dict, det_by_image: dict, image_ids: list[int]) -> tuple[float, float]:
    best = (0.0, 0.25)
    n_gt = sum(len(gt_by_image[i]) for i in image_ids)
    for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
        tp = fp = 0
        for i in image_ids:
            labels, _, _ = classify(gt_by_image[i], [d for d in det_by_image[i] if d["score"] >= thr])
            tp += labels.count("correct")
            fp += len(labels) - labels.count("correct")
        f1 = 2 * tp / max(2 * tp + fp + (n_gt - tp), 1)
        if f1 > best[0]:
            best = (f1, float(thr))
    return best


def image_f1(gts: list[dict], dets: list[dict]) -> float:
    labels, _, _ = classify(gts, dets)
    tp = labels.count("correct")
    return 2 * tp / max(2 * tp + (len(labels) - tp) + (len(gts) - tp), 1)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_box(draw, box, scale, color, text=None, dashed=False, width=3):
    x, y, w, h = [v * scale for v in box]
    if dashed:
        for (x1, y1, x2, y2) in ((x, y, x + w, y), (x + w, y, x + w, y + h), (x, y + h, x + w, y + h), (x, y, x, y + h)):
            length = max(abs(x2 - x1), abs(y2 - y1))
            for s in np.arange(0, length, 10):
                t0, t1 = s / max(length, 1), min(s + 5, length) / max(length, 1)
                draw.line((x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0, x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1),
                          fill=color, width=2)
    else:
        draw.rectangle((x, y, x + w, y + h), outline=color, width=width)
    if text:
        f = font(13)
        tw, th = draw.textbbox((0, 0), text, font=f)[2:]
        ty = y - th - 3 if y - th - 3 > 0 else y + 1
        draw.rectangle((x, ty, x + tw + 4, ty + th + 2), fill=color)
        draw.text((x + 2, ty), text, fill=(0, 0, 0) if sum(color) > 400 else (255, 255, 255), font=f)


def panel(image: Image.Image, title: str, scale: float) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    resized = image.resize((round(image.width * scale), round(image.height * scale)))
    canvas = Image.new("RGB", (resized.width, resized.height + 28), (255, 255, 255))
    canvas.paste(resized, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(0, 0, 0), font=font(16))
    return canvas, draw


def render(info, image_root, gts, models, names, thresholds, out_path):
    image = Image.open(image_root / info["file_name"]).convert("RGB")
    scale = PANEL_WIDTH / image.width
    panels = []
    canvas, draw = panel(image, f"Ground truth ({len(gts)} boxes)", scale)
    for g in gts:
        shifted = [g["bbox"][0], g["bbox"][1] + 28 / scale, g["bbox"][2], g["bbox"][3]]
        draw_box(draw, shifted, scale, COLORS["gt"], names[g["category_id"]])
    panels.append(canvas)
    counts = {}
    for model, dets in models.items():
        shown = [d for d in dets if d["score"] >= thresholds[model]]
        labels, matched, ordered = classify(gts, shown)
        counts[model] = Counter(labels) | Counter({"missed": matched.count(False)})
        summary = " ".join(f"{k[0].upper()}{counts[model][k]}" for k in ERROR_TYPES if counts[model][k])
        canvas, draw = panel(image, f"{model} (thr {thresholds[model]:.2f})  {summary}", scale)
        for g, hit in zip(gts, matched):
            if not hit:
                draw_box(draw, [g["bbox"][0], g["bbox"][1] + 28 / scale, g["bbox"][2], g["bbox"][3]],
                         scale, COLORS["missed"], dashed=True)
        for det, label in sorted(zip(ordered, labels), key=lambda x: x[0]["score"]):
            draw_box(draw, [det["bbox"][0], det["bbox"][1] + 28 / scale, det["bbox"][2], det["bbox"][3]],
                     scale, COLORS[label], f"{names[det['category_id']]} {det['score']:.2f}")
        panels.append(canvas)
    legend_h = 30
    sheet = Image.new("RGB", (sum(p.width for p in panels) + 8 * (len(panels) - 1),
                              max(p.height for p in panels) + legend_h), (255, 255, 255))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 8
    draw = ImageDraw.Draw(sheet)
    x, y = 6, sheet.height - legend_h + 6
    for key, text in (("correct", "correct"), ("wrong_class", "wrong class"), ("poor_box", "poor box"),
                      ("duplicate", "duplicate"), ("background", "background FP"), ("missed", "missed GT (dashed)")):
        draw.rectangle((x, y, x + 14, y + 14), fill=COLORS[key])
        draw.text((x + 18, y), text, fill=(0, 0, 0), font=font(13))
        x += 30 + draw.textbbox((0, 0), text, font=font(13))[2]
    draw.text((x + 10, y), f"image_id {info['id']}  {Path(info['file_name']).name[:40]}", fill=(90, 90, 90), font=font(13))
    sheet.save(out_path, quality=90)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", type=Path, default=ROOT / "data/aquarium-v2-coco/valid.json")
    parser.add_argument("--images", type=Path, default=ROOT / "data/aquarium-v2-yolo",
                        help="Root that COCO file_name paths are relative to")
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--thr", action="append", default=[], metavar="NAME=SCORE",
                        help="Fixed display threshold for a model (default: its F1-optimal threshold)")
    parser.add_argument("--contrast", action="append", default=[], metavar="A:B",
                        help="Also select the images with the largest per-image F1 difference between A and B")
    parser.add_argument("--per-contrast", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.coco.name == "test.json":
        parser.error("Qualitative analysis is validation-only")

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}
    image_ids = sorted(images)
    gt_by_image = defaultdict(list)
    for a in coco["annotations"]:
        gt_by_image[a["image_id"]].append(a)
    models = {}
    for item in args.pred:
        name, _, path = item.partition("=")
        by_image = defaultdict(list)
        for d in load_predictions(path):
            by_image[d["image_id"]].append(d)
        models[name] = (path, by_image)
    fixed = {k: float(v) for k, _, v in (t.partition("=") for t in args.thr)}

    thresholds, threshold_info = {}, {}
    for name, (path, by_image) in models.items():
        if name in fixed:
            thresholds[name] = fixed[name]
            threshold_info[name] = {"threshold": fixed[name], "rule": "fixed (--thr)", "source": path}
        else:
            f1, thr = f1_threshold(gt_by_image, by_image, image_ids)
            thresholds[name] = thr
            threshold_info[name] = {"threshold": thr, "rule": "F1-optimal at IoU 0.5 on all validation images",
                                    "f1": round(f1, 4), "source": path}

    # Deterministic selection: for each class, the image with the most boxes of that class,
    # then the largest per-image F1 disagreements for each requested contrast.
    selected = {}
    for cat_id, cat in names.items():
        best = max(image_ids, key=lambda i: (sum(a["category_id"] == cat_id for a in gt_by_image[i]), -i))
        selected.setdefault(best, []).append(f"most '{cat}' boxes")
    for contrast in args.contrast:
        a, _, b = contrast.partition(":")
        diffs = []
        for i in image_ids:
            if not gt_by_image[i]:
                continue
            fa = image_f1(gt_by_image[i], [d for d in models[a][1][i] if d["score"] >= thresholds[a]])
            fb = image_f1(gt_by_image[i], [d for d in models[b][1][i] if d["score"] >= thresholds[b]])
            diffs.append((abs(fa - fb), fa - fb, i))
        for _, delta, i in sorted(diffs, key=lambda x: (-x[0], x[2]))[:args.per_contrast]:
            selected.setdefault(i, []).append(f"{a} vs {b}: per-image F1 {'+' if delta >= 0 else ''}{delta:.2f}")

    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in sorted(selected):
        out_path = figures / f"val_{i:03d}.jpg"
        counts = render(images[i], args.images, gt_by_image[i], {n: m[1][i] for n, m in models.items()},
                        names, thresholds, out_path)
        manifest.append({"image_id": i, "file_name": images[i]["file_name"], "reasons": selected[i],
                         "figure": str(out_path.relative_to(args.out)),
                         "gt_per_class": dict(Counter(names[a["category_id"]] for a in gt_by_image[i])),
                         "counts": {n: {k: c[k] for k in ERROR_TYPES} for n, c in counts.items()}})

    # Error breakdown over ALL validation images (not only the selected ones).
    rows = []
    for name, (_, by_image) in models.items():
        total, per_class = Counter(), defaultdict(Counter)
        for i in image_ids:
            gts = gt_by_image[i]
            shown = [d for d in by_image[i] if d["score"] >= thresholds[name]]
            labels, matched, ordered = classify(gts, shown)
            for det, label in zip(ordered, labels):
                total[label] += 1
                per_class[names[det["category_id"]]][label] += 1
            for g, hit in zip(gts, matched):
                if not hit:
                    total["missed"] += 1
                    per_class[names[g["category_id"]]]["missed"] += 1
        rows.append({"model": name, "class": "all", **{k: total[k] for k in ERROR_TYPES}})
        rows += [{"model": name, "class": c, **{k: per_class[c][k] for k in ERROR_TYPES}} for c in names.values()]
    with (args.out / "error_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "class", *ERROR_TYPES])
        writer.writeheader()
        writer.writerows(rows)
    confusion = {}
    for name, (_, by_image) in models.items():
        pairs = Counter()
        for i in image_ids:
            gts = gt_by_image[i]
            shown = [d for d in by_image[i] if d["score"] >= thresholds[name]]
            labels, _, ordered = classify(gts, shown)
            g = np.array([x["bbox"] for x in gts], dtype=float).reshape(-1, 4)
            for det, label in zip(ordered, labels):
                if label == "wrong_class":
                    j = int(np.argmax(iou_matrix(np.array([det["bbox"]], dtype=float), g)[0]))
                    pairs[f"{names[gts[j]['category_id']]} -> {names[det['category_id']]}"] += 1
        confusion[name] = dict(pairs.most_common())
    (args.out / "manifest.json").write_text(json.dumps(
        {"coco": str(args.coco), "display_thresholds": threshold_info, "selection_rule":
         "per class: image with the most GT boxes of that class (ties: lowest id); per contrast: "
         f"top {args.per_contrast} images by |per-image F1 difference| at the display thresholds",
         "wrong_class_pairs (gt -> predicted)": confusion, "images": manifest}, indent=2) + "\n", encoding="utf-8")
    print(f"{len(manifest)} figures in {figures}; thresholds: "
          + ", ".join(f"{k}={v['threshold']}" for k, v in threshold_info.items()))


if __name__ == "__main__":
    main()
