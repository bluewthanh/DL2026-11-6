"""Setup 3: YOLOv8n data-efficiency ablation (protocol: reports/03_setup3_data_efficiency/PROTOCOL.md).

Trains YOLOv8n on nested random subsets of the training images with the shared protocol,
scores best.pt and last.pt on the validation split with pycocotools, and tabulates the
result against the zero-shot YOLOE bare-name validation AP. Run in the benchmark venv:

    python scripts/yolo/data_efficiency.py train       # ~40 min on an RTX 3060
    python scripts/yolo/data_efficiency.py evaluate
    python scripts/yolo/data_efficiency.py summarize --yoloe reports/02_setup2_prompt_study/runs/bare/metrics.json
"""

import argparse
import contextlib
import csv
import io
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aquadet import config as cfg  # noqa: E402  (sets YOLO_CONFIG_DIR before ultralytics loads)
from aquadet.dataset import link_or_copy, prepare  # noqa: E402
from aquadet.train import RUNS_DIR, train  # noqa: E402

MODEL = "yolov8n"
FRACTIONS = (0.10, 0.25, 0.50, 1.00)  # predeclared in PROTOCOL.md
POSTHOC = (0.01, 0.025, 0.05)  # added after seeing that 10 % already beats YOLOE (post-hoc follow-up)
SEEDS = (0, 1, 2)
SUBSET_DIR = ROOT / "data" / "aquarium-v2-subsets"
OUT_DIR = ROOT / "reports" / "03_setup3_data_efficiency"
# The 100 %, seed-0 point is the existing full benchmark run (same protocol and seed).
EXISTING = {(1.00, 0): "yolov8n_e100_s0"}


def pct_label(fraction: float) -> str:
    """0.1 -> '010', 0.025 -> '002p5' (percent, zero-padded)."""
    pct = fraction * 100
    return f"{round(pct):03d}" if abs(pct - round(pct)) < 1e-9 else f"{int(pct):03d}p{str(pct).split('.')[1]}"


def run_name(fraction: float, seed: int) -> str:
    return EXISTING.get((fraction, seed), f"{MODEL}_frac{pct_label(fraction)}_s{seed}")


def subset_images(names: list[str], fraction: float, seed: int) -> list[str]:
    """Nested: for one seed, a smaller fraction is a prefix of the same permutation."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    return sorted(order[:round(fraction * len(order))])


def class_counts(label_dir: Path, images: list[str], names: list[str]) -> dict:
    boxes = {n: 0 for n in names}
    images_with = {n: 0 for n in names}
    for image in images:
        lines = (label_dir / (Path(image).stem + ".txt")).read_text(encoding="utf-8").split("\n")
        present = set()
        for line in filter(str.strip, lines):
            name = names[int(line.split()[0])]
            boxes[name] += 1
            present.add(name)
        for name in present:
            images_with[name] += 1
    return {"images": len(images), "boxes_per_class": boxes, "images_per_class": images_with}


def build_subset(prepared: dict, fraction: float, seed: int) -> tuple[Path, dict]:
    """Hard-linked train subset; val/test point at the unchanged derived view."""
    view, manifest = prepared["view"], prepared["manifest"]
    names = manifest["names"]
    train_dir = manifest["split_counts"]["train"]["directory"]
    all_images = [p.name for p in (view / train_dir / "images").iterdir() if p.is_file()]
    chosen = subset_images(all_images, fraction, seed)
    dest = SUBSET_DIR / f"frac{pct_label(fraction)}_s{seed}"
    images_out, labels_out = dest / "train" / "images", dest / "train" / "labels"
    if not (dest / "subset.json").is_file():
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image in chosen:
            label = Path(image).stem + ".txt"
            for src, dst in ((view / train_dir / "images" / image, images_out / image),
                             (view / train_dir / "labels" / label, labels_out / label)):
                if not dst.exists():
                    link_or_copy(src, dst)
    counts = class_counts(view / train_dir / "labels", chosen, names)
    info = {"fraction": fraction, "seed": seed, **counts, "image_files": chosen}
    (dest / "subset.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    import yaml

    splits = manifest["split_counts"]
    data = {"path": str(dest.resolve()), "train": "train/images",
            "val": str((view / splits["valid"]["directory"] / "images").resolve()),
            "test": str((view / splits["test"]["directory"] / "images").resolve()),
            "nc": len(names), "names": dict(enumerate(names))}
    data_yaml = dest / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml, info


def plan(fractions) -> list[tuple[float, int]]:
    return [(f, s) for f in fractions for s in SEEDS]


def cmd_train(args) -> None:
    subsets_path = OUT_DIR / "subsets.json"
    subsets = json.loads(subsets_path.read_text()) if subsets_path.is_file() else {}
    for fraction, seed in plan(args.fractions):
        config = cfg.load(MODEL, [f"train.seed={seed}"])
        prepared = prepare(config)
        name = run_name(fraction, seed)
        if fraction == 1.0:
            data_yaml = prepared["data_yaml"]
            names = prepared["manifest"]["names"]
            train_dir = prepared["manifest"]["split_counts"]["train"]["directory"]
            all_images = sorted(p.name for p in (prepared["view"] / train_dir / "images").iterdir())
            info = {"fraction": 1.0, "seed": seed,
                    **class_counts(prepared["view"] / train_dir / "labels", all_images, names)}
        else:
            data_yaml, info = build_subset(prepared, fraction, seed)
            info = {k: v for k, v in info.items() if k != "image_files"}
        subsets[name] = info
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        subsets_path.write_text(json.dumps(subsets, indent=2) + "\n", encoding="utf-8")
        if (RUNS_DIR / name / "weights" / "last.pt").is_file():
            print(f"[setup3] {name}: already trained, skipping")
            continue
        print(f"[setup3] training {name} on {info['images']} images")
        train(config, data_yaml, name, args.device)


def cmd_evaluate(args) -> None:
    from ultralytics import YOLO

    from aquadet.evaluate import coco_eval

    config = cfg.load(MODEL)
    prepared = prepare(config)
    settings = config["eval"]
    for fraction, seed in plan(args.fractions):
        name = run_name(fraction, seed)
        result_path = OUT_DIR / "runs" / f"{name}.json"
        if result_path.is_file() and not args.force:
            continue
        run_dir = RUNS_DIR / name
        if not (run_dir / "train_info.json").is_file():  # written only after training finishes
            print(f"[setup3] {name}: not trained yet, skipping")
            continue
        record = {"run": name, "fraction": fraction, "seed": seed, "split": "valid",
                  "eval_settings": settings,
                  "train_info": json.loads((run_dir / "train_info.json").read_text())}
        for weights in ("last", "best"):
            out = OUT_DIR / "eval" / name / weights
            out.mkdir(parents=True, exist_ok=True)
            print(f"[setup3] scoring {name}/{weights}.pt on valid")
            with contextlib.redirect_stdout(io.StringIO()):
                record[weights] = coco_eval(YOLO(str(run_dir / "weights" / f"{weights}.pt")),
                                            prepared["coco_dir"] / "valid.json", prepared["view"],
                                            settings, out, args.device)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def cmd_summarize(args) -> None:
    subsets = json.loads((OUT_DIR / "subsets.json").read_text())
    records = [json.loads((OUT_DIR / "runs" / f"{run_name(f, s)}.json").read_text()) for f, s in plan(args.fractions)]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    yoloe = json.loads(Path(args.yoloe).read_text())
    classes = list(records[0]["last"]["per_class"])

    rows, csv_rows = [], []
    for fraction in args.fractions:
        group = [r for r in records if r["fraction"] == fraction]
        row = {"fraction": fraction, "images": subsets[group[0]["run"]]["images"], "seeds": len(group)}
        for weights in ("last", "best"):
            for metric in ("AP", "AP50"):
                row[f"{weights}_{metric}"] = mean_std([r[weights][metric] for r in group])
            for c in classes:
                row[f"{weights}_{c}"] = mean_std([r[weights]["per_class"][c]["AP"] for r in group])
        row["epochs_best"] = [r["train_info"]["best_epoch"] for r in group]
        row["train_min"] = mean_std([r["train_info"]["wall_time_s"] / 60 for r in group])
        rows.append(row)
        for r in group:
            csv_rows.append({"run": r["run"], "fraction": fraction, "seed": r["seed"],
                             "train_images": subsets[r["run"]]["images"],
                             **{f"{w}_{m}": round(r[w][m], 5) for w in ("last", "best") for m in ("AP", "AP50")},
                             **{f"last_{c}": round(r["last"]["per_class"][c]["AP"], 5) for c in classes},
                             "best_epoch": r["train_info"]["best_epoch"],
                             "train_wall_s": r["train_info"]["wall_time_s"]})

    with (out_dir / "setup3_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    fmt = lambda ms: f"{100 * ms[0]:.1f} ± {100 * ms[1]:.1f}"  # noqa: E731
    yoloe_ap = yoloe["AP"]
    lines = ["| Train images | Seeds | AP (last.pt) | AP50 (last.pt) | AP (best.pt) | Beats YOLOE bare? |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['fraction'] * 100:g} % ({row['images']}) | {row['seeds']} | {fmt(row['last_AP'])} | "
                     f"{fmt(row['last_AP50'])} | {fmt(row['best_AP'])} | "
                     f"{'yes' if row['last_AP'][0] > yoloe_ap else 'no'} |")
    lines.append(f"| YOLOE-26s bare prompts, zero-shot (0) | 1 | {100 * yoloe_ap:.1f} | {100 * yoloe['AP50']:.1f} | – | – |")
    per_class = ["| Train images | " + " | ".join(classes) + " |", "|---|" + "---:|" * len(classes)]
    for row in rows:
        per_class.append(f"| {row['fraction'] * 100:g} % ({row['images']}) | "
                         + " | ".join(fmt(row[f"last_{c}"]) for c in classes) + " |")
    per_class.append("| YOLOE bare, zero-shot | "
                     + " | ".join(f"{100 * yoloe['per_class_AP'][c]:.1f}" for c in classes) + " |")
    counts = ["| Subset (seed) | Images | " + " | ".join(classes) + " |", "|---|---:|" + "---:|" * len(classes)]
    for name, info in ((run_name(f, s), subsets[run_name(f, s)]) for f, s in plan(args.fractions)):
        counts.append(f"| {name} | {info['images']} | "
                      + " | ".join(f"{info['images_per_class'][c]} / {info['boxes_per_class'][c]}" for c in classes)
                      + " |")
    (out_dir / "tables.md").write_text("\n".join(
        ["## Overall validation COCO AP (×100, mean ± std over seeds)", "", *lines, "",
         "## Per-class validation AP@[.5:.95] (last.pt, ×100, mean ± std over seeds)", "", *per_class, "",
         "## Training subsets: images containing the class / boxes of the class", "", *counts, ""]),
        encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(
        {"yoloe_bare_valid": {"AP": yoloe_ap, "AP50": yoloe["AP50"], "source": args.yoloe},
         "rows": rows}, indent=2) + "\n", encoding="utf-8")
    plot(rows, yoloe_ap, out_dir, posthoc=bool(set(args.fractions) - set(FRACTIONS)))
    print((out_dir / "tables.md").read_text())


def plot(rows: list[dict], yoloe_ap: float, out_dir: Path, posthoc: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["images"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for key, label, style in (("last_AP", "YOLOv8n last.pt", "o-"), ("best_AP", "YOLOv8n best.pt (val-selected)", "s--")):
        means = [100 * r[key][0] for r in rows]
        stds = [100 * r[key][1] for r in rows]
        ax.errorbar(x, means, yerr=stds, fmt=style, capsize=3, label=label)
    ax.axhline(100 * yoloe_ap, color="tab:red", linestyle=":", label="YOLOE-26s bare prompts (zero-shot)")
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{r['images']}\n({r['fraction'] * 100:g} %)" for r in rows])
    ax.minorticks_off()
    ax.set_xlabel("Labelled training images")
    ax.set_ylabel("Validation COCO AP@[.5:.95] (×100)")
    ax.set_title("YOLOv8n data efficiency vs zero-shot YOLOE (127 val images)"
                 + ("\nincludes post-hoc fractions below 10 %" if posthoc else ""), fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "data_efficiency.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fractions = dict(type=float, nargs="+", default=list(FRACTIONS),
                     help=f"Training fractions (default: the predeclared {FRACTIONS}; post-hoc: {POSTHOC})")
    for name, func in (("train", cmd_train), ("evaluate", cmd_evaluate)):
        sp = sub.add_parser(name)
        sp.add_argument("--device", default="0")
        sp.add_argument("--fractions", **fractions)
        sp.set_defaults(func=func)
    sub.choices["evaluate"].add_argument("--force", action="store_true", help="Re-score runs already scored")
    sp = sub.add_parser("summarize")
    sp.add_argument("--yoloe", required=True, help="metrics.json of the YOLOE bare-name validation run")
    sp.add_argument("--fractions", **fractions)
    sp.add_argument("--out", type=Path, default=OUT_DIR, help="Where tables and the plot are written")
    sp.set_defaults(func=cmd_summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
