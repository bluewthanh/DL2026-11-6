"""Closed-set YOLO benchmark.

    python -m aquadet run --model yolov8n   # prepare, train, evaluate, benchmark, summarize
    python -m aquadet compare               # summary table from reports/runs/*.json
"""

import argparse
import sys
from pathlib import Path

import yaml

from aquadet import ROOT, config as cfg


def default_device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def resolve_run(name: str) -> Path:
    from aquadet.train import RUNS_DIR

    path = Path(name)
    run_dir = path if path.is_dir() else RUNS_DIR / name
    if not (run_dir / "experiment.yaml").is_file():
        raise ValueError(f"{run_dir} is not an aquadet run (no experiment.yaml)")
    return run_dir.resolve()


def run_config(run_dir: Path, source: str | None) -> dict:
    config = yaml.safe_load((run_dir / "experiment.yaml").read_text(encoding="utf-8"))
    if source:
        config["dataset"]["source"] = source
    return config


def build_config(args) -> dict:
    overrides = list(args.set or [])
    if args.seed is not None:
        overrides.append(f"train.seed={args.seed}")
    return cfg.load(args.model, overrides, args.source, args.smoke)


def cmd_prepare(args) -> None:
    from aquadet.dataset import prepare

    config = cfg.load(cfg.available_models()[0], source=args.source)
    prepared = prepare(config, rebuild=args.rebuild)
    counts = prepared["manifest"]["split_counts"]
    print(f"[prepare] OK  view={prepared['view'].relative_to(ROOT)}  coco={prepared['coco_dir'].relative_to(ROOT)}  "
          f"link_mode={','.join(prepared['manifest']['link_modes'])}")
    for split, c in counts.items():
        print(f"           {split:5s}: {c['images']:4d} images, {c['boxes']:5d} boxes")
    print(f"           omitted: {len(prepared['manifest']['omitted_source_annotations'])} known zero-area boxes")


def cmd_train(args) -> Path:
    from aquadet.dataset import prepare
    from aquadet.train import train

    config = build_config(args)
    prepared = prepare(config)
    name = args.name or cfg.default_run_name(config)
    if not config["protocol_matches_base"]:
        print(f"[train] NOTE: overrides {config['overrides']} change the shared protocol; "
              "this run will be excluded from `compare` unless --include-custom is given")
    run_dir = train(config, prepared["data_yaml"], name, args.device or default_device())
    print(f"[train] finished: {run_dir.relative_to(ROOT)}")
    return run_dir


def cmd_evaluate(args, run_dir: Path | None = None) -> None:
    from aquadet.dataset import prepare
    from aquadet.evaluate import evaluate

    run_dir = run_dir or resolve_run(args.run)
    config = run_config(run_dir, args.source)
    prepared = prepare(config)
    splits = args.split or config["eval"]["splits"]
    evaluate(run_dir / "weights" / f"{args.weights}.pt", prepared, config["eval"], splits,
             run_dir / "eval", args.device or default_device())
    cmd_report(args, run_dir)


def cmd_benchmark(args, run_dir: Path | None = None) -> None:
    from aquadet.benchmark import benchmark
    from aquadet.dataset import prepare

    run_dir = run_dir or resolve_run(args.run)
    config = run_config(run_dir, args.source)
    prepared = prepare(config)
    image_dir = prepared["view"] / prepared["manifest"]["split_counts"]["valid"]["directory"] / "images"
    benchmark(run_dir / "weights" / f"{args.weights}.pt", image_dir, config["benchmark"],
              run_dir / "benchmark", args.device or default_device())
    cmd_report(args, run_dir)


def cmd_report(args, run_dir: Path | None = None) -> None:
    from aquadet.report import export_summary

    export_summary(run_dir or resolve_run(args.run))


def cmd_run(args) -> None:
    run_dir = cmd_train(args)
    args.split, args.weights = None, "best"
    cmd_evaluate(args, run_dir)
    cmd_benchmark(args, run_dir)
    if not run_dir.name.endswith("_smoke"):
        print(f"[run] done. Next: git add reports/runs/{run_dir.name}.json && python -m aquadet compare")


def cmd_compare(args) -> None:
    from aquadet.report import compare

    compare(include_custom=args.include_custom)


def cmd_env(args) -> None:
    import json

    from aquadet.env import environment

    print(json.dumps(environment(), indent=2))
    print("models:", ", ".join(cfg.available_models()))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m aquadet", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, device=True):
        sp.add_argument("--source", help="Original export root (default: configs/base.yaml or $AQUADET_SOURCE)")
        if device:
            sp.add_argument("--device", help="CUDA index like 0, or cpu (default: 0 if CUDA is available)")

    def training(sp):
        sp.add_argument("--model", required=True, choices=cfg.available_models())
        sp.add_argument("--seed", type=int, help="Shortcut for --set train.seed=N (seeds are aggregated in compare)")
        sp.add_argument("--name", help="Run directory name under runs/ (default: <model>_e<epochs>_s<seed>)")
        sp.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                        help="Override a configs/base.yaml value (recorded; marks the run as custom protocol)")
        sp.add_argument("--smoke", action="store_true", help="1 epoch + short benchmark to test the pipeline end to end")
        common(sp)

    def existing(sp):
        sp.add_argument("--run", required=True, help="Run name under runs/ or a path to a run directory")
        sp.add_argument("--weights", default="best", choices=("best", "last"))
        common(sp)

    sp = sub.add_parser("prepare", help="Audit the export and build the derived YOLO + COCO views")
    sp.add_argument("--rebuild", action="store_true", help="Rebuild the YOLO view even if it is up to date")
    common(sp, device=False)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("run", help="Full pipeline: prepare, train, evaluate (val+test), benchmark, summary")
    training(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("train", help="Train only")
    training(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="(Re-)evaluate a trained run")
    existing(sp)
    sp.add_argument("--split", action="append", choices=("val", "test"), help="Default: both")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("benchmark", help="(Re-)measure size and latency of a trained run on this machine")
    existing(sp)
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("report", help="Re-export reports/runs/<run>.json from a run directory")
    sp.add_argument("--run", required=True)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("compare", help="Build reports/benchmark.{md,csv} from reports/runs/*.json")
    sp.add_argument("--include-custom", action="store_true", help="Also include runs with protocol overrides")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("env", help="Print the environment that will be recorded with results")
    sp.set_defaults(func=cmd_env)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
