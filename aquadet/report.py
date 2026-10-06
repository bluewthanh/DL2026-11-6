"""Per-run summaries (reports/runs/*.json, committed) and the cross-model comparison."""

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from aquadet import ROOT

REPORTS_DIR = ROOT / "reports"
SUMMARY_DIR = REPORTS_DIR / "runs"
CLASSES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build_summary(run_dir: Path) -> dict:
    config = yaml.safe_load((run_dir / "experiment.yaml").read_text(encoding="utf-8"))
    metrics = load_json(run_dir / "eval" / "metrics.json") or {}
    bench = load_json(run_dir / "benchmark" / "benchmark.json")
    env = (bench or {}).get("environment_before")
    if env is None:
        from aquadet.env import environment
        env = environment()
    return {
        "run": run_dir.name,
        "model": config["model"]["name"],
        "seed": config["train"]["seed"],
        "epochs": config["train"]["epochs"],
        "smoke": config["smoke"],
        "protocol_hash": config["protocol_hash"],
        "protocol_matches_base": config["protocol_matches_base"],
        "overrides": config["overrides"],
        "train": load_json(run_dir / "train_info.json"),
        "eval": metrics.get("results", {}),
        "benchmark": None if bench is None else {"size": bench["size"], "latency_ms": bench["latency_ms"],
                                                  "settings": bench["settings"]},
        "environment": env,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def export_summary(run_dir: Path) -> Path:
    summary = build_summary(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["smoke"]:
        print(f"[report] smoke run: summary kept in {run_dir / 'summary.json'} only (not exported)")
        return run_dir / "summary.json"
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    target = SUMMARY_DIR / f"{summary['run']}.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[report] wrote {target.relative_to(ROOT)} (commit this file)")
    return target


def dig(data: dict | None, *keys, default=None):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


METRICS = [  # (column, path into summary, higher_is_better, scale, decimals)
    ("Params (M)", ("benchmark", "size", "params_M"), False, 1, 2),
    ("GFLOPs", ("benchmark", "size", "gflops"), False, 1, 1),
    ("Val AP", ("eval", "val", "coco", "AP"), True, 100, 1),
    ("Val AP50", ("eval", "val", "coco", "AP50"), True, 100, 1),
    ("Test AP", ("eval", "test", "coco", "AP"), True, 100, 1),
    ("Test AP50", ("eval", "test", "coco", "AP50"), True, 100, 1),
    ("Test AP75", ("eval", "test", "coco", "AP75"), True, 100, 1),
    ("FP32 infer (ms)", ("benchmark", "latency_ms", "fp32", "inference", "mean"), False, 1, 2),
    ("FP16 infer (ms)", ("benchmark", "latency_ms", "fp16", "inference", "mean"), False, 1, 2),
    ("FP16 post (ms)", ("benchmark", "latency_ms", "fp16", "postprocess", "mean"), False, 1, 2),
    ("FP16 end-to-end (ms)", ("benchmark", "latency_ms", "fp16", "end_to_end", "mean"), False, 1, 2),
    ("Train (min)", ("train", "wall_time_s"), False, 1 / 60, 1),
]
NATIVE = [("P", "P"), ("R", "R"), ("mAP50", "mAP50"), ("mAP50-95", "mAP50-95")]


def aggregate(values: list[float]) -> tuple[float, float | None]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def fmt(mean: float | None, std: float | None, decimals: int, bold: bool = False) -> str:
    if mean is None:
        return "–"
    text = f"{mean:.{decimals}f}" + (f" ± {std:.{decimals}f}" if std is not None else "")
    return f"**{text}**" if bold else text


def group_by_model(summaries: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for s in sorted(summaries, key=lambda s: (s["model"], s["seed"])):
        groups.setdefault(s["model"], []).append(s)
    return groups


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    align = ["---"] + ["---:"] * (len(header) - 1)
    return ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"] + ["| " + " | ".join(r) + " |" for r in rows]


def compare(include_custom: bool = False) -> Path:
    summaries = [load_json(p) for p in sorted(SUMMARY_DIR.glob("*.json"))]
    summaries = [s for s in summaries if s and not s["smoke"] and (include_custom or s["protocol_matches_base"])]
    if not summaries:
        raise ValueError(f"No run summaries in {SUMMARY_DIR.relative_to(ROOT)}; run `python -m aquadet run --model ...` first")
    groups = group_by_model(summaries)

    warnings = []
    hashes = {s["protocol_hash"] for s in summaries}
    if len(hashes) > 1:
        warnings.append(f"Runs use {len(hashes)} different protocols ({', '.join(sorted(hashes))}); AP is NOT directly comparable.")
    devices = {dig(s, "environment", "device") for s in summaries if s.get("benchmark")}
    if len(devices) > 1:
        warnings.append(f"Latency was measured on different devices ({', '.join(sorted(map(str, devices)))}); "
                        "re-run `python -m aquadet benchmark --run ...` for all models on one GPU before comparing speed.")
    missing = [m for m, runs in groups.items() for s in runs if "test" not in s["eval"] or not s.get("benchmark")]
    if missing:
        warnings.append(f"Incomplete runs (missing test eval or benchmark): {', '.join(sorted(set(missing)))}.")

    cells = {}
    for model, runs in groups.items():
        for column, path, _, scale, _ in METRICS:
            values = [v * scale for s in runs if (v := dig(s, *path)) is not None]
            cells[model, column] = aggregate(values) if values else (None, None)
    rows = []
    for model, runs in groups.items():
        row = [model, str(len(runs))]
        for column, _, higher, _, decimals in METRICS:
            means = [cells[m, column][0] for m in groups if cells[m, column][0] is not None]
            mean, std = cells[model, column]
            best = mean is not None and len(means) > 1 and mean == (max(means) if higher else min(means))
            row.append(fmt(mean, std, decimals, best))
        rows.append(row)

    lines = ["# Closed-set YOLO benchmark — RF100 Aquarium v2", "",
             f"Generated by `python -m aquadet compare` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC "
             "from `reports/runs/*.json`. Do not edit by hand.", ""]
    if warnings:
        lines += ["> **Warnings**", ">"] + [f"> - {w}" for w in warnings] + [""]
    lines += ["## Accuracy, size and speed", "",
              "AP = pycocotools COCO AP on the derived COCO ground truth (×100; shared evaluator for the three closed-set models). "
              "Mean ± std over seeds when more than one seed exists. Latency: batch 1, 640 px, warmed-up GPU, "
              "images pre-loaded in memory. Bold = best per column.", ""]
    lines += table(["Model", "Seeds"] + [m[0] for m in METRICS], rows) + [""]

    for split in ("test", "val"):
        lines += [f"## Per-class AP@[.5:.95] on {split} (COCO, ×100)", ""]
        per_class_rows = []
        for model, runs in groups.items():
            row = [model]
            for name in CLASSES:
                values = [v * 100 for s in runs if (v := dig(s, "eval", split, "coco", "per_class", name, "AP")) is not None]
                row.append(fmt(*aggregate(values), 1) if values else "–")
            per_class_rows.append(row)
        gt = next((dig(s, "eval", split, "coco", "per_class") for s in summaries if dig(s, "eval", split, "coco", "per_class")), {})
        header = ["Model"] + [f"{c} (n={dig(gt, c, 'gt_boxes', default='?')})" for c in CLASSES]
        lines += table(header, per_class_rows) + [""]

    lines += ["## Ultralytics-native metrics on test (×100)", "",
              "From `model.val()` on the derived YOLO view (same two zero-area boxes omitted). "
              "These use Ultralytics' own AP integration and differ slightly from COCOeval.", ""]
    native_rows = []
    for model, runs in groups.items():
        row = [model]
        for _, key in NATIVE:
            values = [v * 100 for s in runs if (v := dig(s, "eval", "test", "ultralytics", key)) is not None]
            row.append(fmt(*aggregate(values), 1) if values else "–")
        native_rows.append(row)
    lines += table(["Model"] + [n[0] for n in NATIVE], native_rows) + [""]

    lines += ["## Provenance", ""]
    prov_rows = [[s["run"], s["protocol_hash"] + ("" if s["protocol_matches_base"] else " (custom)"),
                  str(dig(s, "train", "best_epoch", default="–")), str(dig(s, "environment", "device", default="–")),
                  str(dig(s, "environment", "ultralytics", default="–")), str(dig(s, "environment", "torch", default="–"))]
                 for s in summaries]
    lines += table(["Run", "Protocol", "Best epoch", "Device", "Ultralytics", "Torch"], prov_rows) + [""]
    lines += ["Test split: 63 images / 582 boxes. Two zero-area `shark` boxes in the original test labels are omitted "
              "from ground truth for every model (see DATA.md); shark AP on test may be slightly biased.", ""]

    REPORTS_DIR.mkdir(exist_ok=True)
    md_path = REPORTS_DIR / "benchmark.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    write_csv(summaries, REPORTS_DIR / "benchmark.csv")
    for w in warnings:
        print(f"[compare] WARNING: {w}")
    print(f"[compare] wrote {md_path.relative_to(ROOT)} and reports/benchmark.csv ({len(summaries)} runs)")
    return md_path


def write_csv(summaries: list[dict], path: Path) -> None:
    columns = ["run", "model", "seed", "epochs", "protocol_hash"] + [m[0] for m in METRICS] + \
              [f"Test AP {c}" for c in CLASSES] + ["device", "ultralytics", "torch"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for s in summaries:
            row = [s["run"], s["model"], s["seed"], s["epochs"], s["protocol_hash"]]
            row += ["" if (v := dig(s, *p)) is None else round(v * scale, 4) for _, p, _, scale, _ in METRICS]
            row += ["" if (v := dig(s, "eval", "test", "coco", "per_class", c, "AP")) is None else round(v * 100, 4)
                    for c in CLASSES]
            row += [dig(s, "environment", k, default="") for k in ("device", "ultralytics", "torch")]
            writer.writerow(row)
