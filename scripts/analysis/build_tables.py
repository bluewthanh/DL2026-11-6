"""Setup 1 (YOLOv8n baseline vs text-prompted YOLOE) and per-class AP tables, validation split.

Every row is recomputed here from saved COCO-format predictions with one pycocotools COCOeval
call (bbox, IoU 0.50:0.95, maxDets 100, all 127 validation images), so all rows share the
same ground truth, image set and evaluator. Each --pred is NAME|REGIME|TRAIN_IMAGES=PATH.

    python scripts/analysis/build_tables.py \
        --pred "YOLOv8n last.pt|supervised fine-tuning|448=reports/03_setup3_data_efficiency/eval/yolov8n_e100_s0/last/coco_predictions_valid.json.gz" \
        --pred "YOLOE-26s bare|zero-shot text prompts|0=reports/02_setup2_prompt_study/runs/bare/predictions.json.gz" \
        --reference "YOLOE-26s bare" --test-summary reports/runs/yolov8n_e100_s0.json
"""

import argparse
import contextlib
import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
STATS = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100"]


def load_predictions(path) -> list[dict]:
    """COCO results JSON, plain or gzip-compressed (.json.gz, as stored in reports/)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def score(gt: COCO, predictions: list[dict]) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = sorted(gt.getImgIds())
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    precision = ev.eval["precision"]  # [IoU, recall, class, area, maxDets]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        all_iou, iou50 = precision[:, :, k, 0, -1], precision[0, :, k, 0, -1]
        per_class[gt.cats[cat_id]["name"]] = {
            "AP": float(all_iou[all_iou > -1].mean()), "AP50": float(iou50[iou50 > -1].mean()),
            "gt_boxes": len(gt.getAnnIds(catIds=[cat_id]))}
    return {**{n: float(v) for n, v in zip(STATS, ev.stats)}, "detections": len(predictions), "per_class": per_class}


def md_table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|",
            *("| " + " | ".join(str(c) for c in row) + " |" for row in rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", type=Path, default=ROOT / "data/aquarium-v2-coco/valid.json")
    parser.add_argument("--pred", action="append", required=True, metavar="NAME|REGIME|TRAIN_IMAGES=PATH")
    parser.add_argument("--reference", help="Row name that per-class deltas are computed against")
    parser.add_argument("--test-summary", type=Path, help="aquadet reports/runs/<run>.json for a test-split table")
    parser.add_argument("--setup1-out", type=Path, default=ROOT / "reports/01_setup1_baseline")
    parser.add_argument("--per-class-out", type=Path, default=ROOT / "reports/04_per_class_ap")
    args = parser.parse_args()
    if args.coco.name != "valid.json":
        parser.error("These tables are validation-only")

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(args.coco))
    rows = []
    for item in args.pred:
        spec, _, path = item.rpartition("=")
        name, regime, train_images = spec.split("|")
        result = score(gt, load_predictions(path))
        rows.append({"name": name, "regime": regime, "train_images": int(train_images), "source": path, **result})
    classes = list(rows[0]["per_class"])

    # Setup 1
    args.setup1_out.mkdir(parents=True, exist_ok=True)
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    table = md_table(["Model", "Regime", "Labelled Aquarium train images", "AP", "AP50", "AP75", "APs", "APm", "APl",
                      "AR100", "Detections"],
                     [[r["name"], r["regime"], r["train_images"], *(pct(r[s]) for s in STATS if s not in ("AR1", "AR10")),
                       r["detections"]] for r in rows])
    (args.setup1_out / "setup1_table.md").write_text(
        "All rows: 127 validation images, 909 GT boxes, pycocotools COCOeval bbox, maxDets 100, ×100. "
        "Inference for every model: imgsz 640, conf 0.001, NMS IoU 0.7, max_det 300.\n\n" + "\n".join(table) + "\n",
        encoding="utf-8")
    with (args.setup1_out / "setup1_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "regime", "train_images", *STATS, "detections", "source"])
        for r in rows:
            writer.writerow([r["name"], r["regime"], r["train_images"], *(round(r[s], 5) for s in STATS),
                             r["detections"], r["source"]])

    # Per-class AP (validation)
    args.per_class_out.mkdir(parents=True, exist_ok=True)
    gt_counts = rows[0]["per_class"]
    lines = ["# Per-class COCO AP, validation split (127 images, 909 boxes)", "",
             "Recomputed from saved predictions with one COCOeval call per model (`scripts/analysis/build_tables.py`). "
             "AP@[.5:.95] and AP50 ×100. Ground-truth box counts in the header.", ""]
    header = ["Model"] + [f"{c} ({gt_counts[c]['gt_boxes']})" for c in classes] + ["mean (= AP)"]
    for metric, title in (("AP", "AP@[.5:.95]"), ("AP50", "AP50")):
        lines += [f"## {title}", ""]
        lines += md_table(header, [[r["name"], *(pct(r["per_class"][c][metric]) for c in classes),
                                    pct(np.mean([r["per_class"][c][metric] for c in classes]))] for r in rows])
        lines.append("")
    if args.reference:
        ref = next(r for r in rows if r["name"] == args.reference)
        lines += [f"## Change in AP@[.5:.95] relative to {args.reference} (×100)", ""]
        lines += md_table(["Model"] + classes + ["overall"],
                          [[r["name"], *(f"{100 * (r['per_class'][c]['AP'] - ref['per_class'][c]['AP']):+.1f}" for c in classes),
                            f"{100 * (r['AP'] - ref['AP']):+.1f}"] for r in rows if r is not ref])
        lines.append("")
    if args.test_summary:
        summary = json.loads(args.test_summary.read_text(encoding="utf-8"))
        test = summary["eval"]["test"]["coco"]["per_class"]
        lines += [f"## Test split, `{summary['run']}` best.pt only (63 images, 582 boxes)", "",
                  "Supervised YOLOv8n only. No text-prompted model was run on test, so this row must **not** be ranked "
                  "against the validation rows above.", ""]
        lines += md_table(["Model"] + [f"{c} ({test[c]['gt_boxes']})" for c in classes] + ["AP"],
                          [[f"{summary['model']} best.pt (test)", *(pct(test[c]["AP"]) for c in classes),
                            pct(summary["eval"]["test"]["coco"]["AP"])]])
        lines.append("")
    (args.per_class_out / "per_class_ap.md").write_text("\n".join(lines), encoding="utf-8")
    with (args.per_class_out / "per_class_ap_valid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "class", "gt_boxes", "AP", "AP50"])
        for r in rows:
            for c in classes:
                pc = r["per_class"][c]
                writer.writerow([r["name"], c, pc["gt_boxes"], round(pc["AP"], 5), round(pc["AP50"], 5)])
    (args.setup1_out / "scores.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
