"""Download and audit RF100 Aquarium v2 YOLOv8 export without changing its splits.

Requires PyYAML; downloading additionally requires roboflow and ROBOFLOW_API_KEY.
Run `python scripts/data/prepare_data.py --help` for usage.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Install PyYAML first: python -m pip install PyYAML") from exc

WORKSPACE = "roboflow-100"
PROJECT = "aquarium-qlnqy"
VERSION = 2
EXPORT_FORMAT = "yolov8"
EXPECTED_COUNTS = {"train": 448, "valid": 127, "test": 63}
EXPECTED_CLASSES = {"fish", "shark", "starfish", "jellyfish", "penguin", "stingray", "puffin"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def download(root: Path) -> Path:
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Download target is not empty: {root}. Use --source to audit an existing export.")
    if not os.environ.get("ROBOFLOW_API_KEY"):
        raise ValueError("Set ROBOFLOW_API_KEY privately before downloading (never commit the key).")
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise ValueError("Install downloader: python -m pip install roboflow") from exc
    root.parent.mkdir(parents=True, exist_ok=True)
    dataset = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace(WORKSPACE).project(PROJECT).version(VERSION).download(EXPORT_FORMAT, location=str(root))
    return Path(dataset.location).resolve()


def image_index(directory: Path, errors: list[str]) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(directory).with_suffix("").as_posix()
        if rel in result:
            errors.append(f"Multiple images share label stem {rel!r}: {result[rel]} and {path}")
        result[rel] = path
    return result


def label_index(directory: Path) -> dict[str, Path]:
    return {p.relative_to(directory).with_suffix("").as_posix(): p
            for p in sorted(directory.rglob("*.txt")) if p.is_file()}


def parse_names(config: dict, errors: list[str]) -> list[str]:
    names = config.get("names")
    if isinstance(names, list) and all(isinstance(n, str) for n in names):
        result = names
    elif isinstance(names, dict):
        try:
            ids = [int(k) for k in names]
            if sorted(ids) != list(range(len(ids))):
                raise ValueError("IDs must be contiguous starting at zero")
            result = [names.get(i, names.get(str(i))) for i in range(len(ids))]
            if not all(isinstance(n, str) for n in result):
                raise ValueError("Each ID must have a name")
        except (ValueError, TypeError) as exc:
            errors.append(f"Invalid names mapping: {exc}")
            return []
    else:
        errors.append("data.yaml must contain a list or integer-keyed map of class names")
        return []
    if len(result) != 7 or len(set(result)) != 7 or set(result) != EXPECTED_CLASSES:
        errors.append(f"data.yaml classes differ from expected seven: {result!r}")
    if "nc" in config and (type(config["nc"]) is not int or config["nc"] != len(result)):
        errors.append(f"data.yaml nc={config['nc']!r} disagrees with names")
    return result


def check_label(path: Path, class_count: int, errors: list[str], class_hist: Counter,
                invalid: list[dict]) -> bool:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not any(line.strip() for line in lines):
        return True  # possible negative image; counted, never repaired
    for number, line in enumerate(lines, 1):
        parts = line.split()
        class_id = parts[0] if parts else None
        xywh = parts[1:] if len(parts) > 1 else []
        reason = None
        if len(parts) != 5:
            reason = f"expected 5 fields (class_id x_center y_center width height), got {len(parts)}"
        else:
            try:
                category = int(class_id)
            except (ValueError, TypeError):
                reason = "class ID is not an integer"
            try:
                values = [float(v) for v in xywh]
            except ValueError:
                reason = (reason + "; " if reason else "") + "xywh contains a non-numeric value"
            if reason is None:
                if not 0 <= category < class_count:
                    reason = f"class ID outside 0..{class_count - 1}"
                else:
                    x, y, w, h = values
                    if not all(0 <= v <= 1 for v in (x, y)):
                        reason = "center x/y must be finite and within [0, 1]"
                    elif not all(0 < v <= 1 for v in (w, h)):
                        reason = "width/height must be finite and strictly positive within (0, 1]"
                    elif not (0 <= x - w / 2 and x + w / 2 <= 1
                              and 0 <= y - h / 2 and y + h / 2 <= 1):
                        reason = "box extends beyond normalized image bounds [0, 1]"
        if reason:
            entry = {"file": str(path), "line": number, "class_id": class_id,
                     "normalized_xywh": xywh, "reason": reason}
            invalid.append(entry)
            errors.append(f"{path}:{number}: class_id={class_id!r} normalized_xywh={xywh!r}: {reason}")
        else:
            class_hist[category] += 1
    return False


def verify(root: Path) -> dict:
    root = root.resolve()
    errors: list[str] = []
    config_path = root / "data.yaml"
    if not config_path.is_file():
        raise ValueError(f"Missing data.yaml in export root {root}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("data.yaml must be a mapping")
    names = parse_names(config, errors)
    report = {"source": str(root), "workspace": WORKSPACE, "project": PROJECT,
              "version_expected": VERSION, "format_expected": EXPORT_FORMAT,
              "id_to_name": {str(i): name for i, name in enumerate(names)}, "splits": {},
              "invalid_annotations": [], "errors": errors}
    if "val" in config and "valid" in config:
        errors.append("data.yaml declares both val and valid; choose exactly one validation key")
    for split, expected in EXPECTED_COUNTS.items():
        # Accept both val/valid spellings, in data.yaml and on disk.
        yaml_key = "val" if split == "valid" and "val" in config else split
        declared = config.get(yaml_key)
        directory_name = split
        if declared is None:
            errors.append(f"data.yaml does not declare {split!r} split (val or valid accepted for validation)")
        elif not isinstance(declared, str):
            errors.append(f"data.yaml {yaml_key!r} must be a directory path")
        else:
            normalized = declared.replace("\\", "/")
            if normalized.startswith("../"):
                normalized = normalized[3:]
            elif normalized.startswith("./"):
                normalized = normalized[2:]
            allowed = {"val/images", "valid/images"} if split == "valid" else {f"{split}/images"}
            if normalized not in allowed:
                errors.append(f"Unexpected data.yaml {yaml_key} path {declared!r}; expected {sorted(allowed)}")
            else:
                directory_name = normalized.split("/")[0]
        images_dir = root / directory_name / "images"
        labels_dir = root / directory_name / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            errors.append(f"{split}: missing images/ or labels/ directory")
            report["splits"][split] = {"images": 0, "labels": 0}
            continue
        images = image_index(images_dir, errors)
        labels = label_index(labels_dir)
        missing_labels = sorted(images.keys() - labels.keys())
        missing_images = sorted(labels.keys() - images.keys())
        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} image(s) missing labels, examples: {missing_labels[:10]}")
        if missing_images:
            errors.append(f"{split}: {len(missing_images)} label(s) missing images, examples: {missing_images[:10]}")
        if len(images) != expected:
            errors.append(f"{split}: {len(images)} images, expected {expected}")
        histogram: Counter = Counter()
        empty = []
        invalid = []
        for stem, path in labels.items():
            try:
                if check_label(path, len(names), errors, histogram, invalid):
                    empty.append(stem)
            except (OSError, UnicodeError) as exc:
                errors.append(f"{path}: cannot read label: {exc}")
        report["invalid_annotations"].extend(invalid)
        report["splits"][split] = {"yaml_split_key": yaml_key, "directory": directory_name, "images": len(images), "labels": len(labels),
            "missing_labels": missing_labels, "missing_images": missing_images,
            "empty_labels": empty, "empty_label_count": len(empty), "invalid_annotation_count": len(invalid),
            "annotations_per_class_id": {str(i): histogram[i] for i in range(len(names))}}
    report["passed"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true", help="Download RF100 Aquarium v2 YOLOv8 using ROBOFLOW_API_KEY")
    group.add_argument("--source", type=Path, help="Existing, unmodified YOLOv8 export root (contains data.yaml)")
    parser.add_argument("--output", type=Path, default=Path("data/aquarium-qlnqy-v2-yolov8"), help="Download destination")
    parser.add_argument("--report", type=Path, default=Path("data/aquarium-v2-audit.json"), help="JSON audit report")
    args = parser.parse_args()
    try:
        root = download(args.output.resolve()) if args.download else args.source
        report = verify(root)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit {'PASS' if report['passed'] else 'FAIL'}: {root} (report: {args.report})")
    for split, item in report["splits"].items():
        print(f"  {split}: {item['images']} images, {item['labels']} labels, {item.get('empty_label_count', 0)} empty labels")
    print(f"  IDs: {report['id_to_name']}")
    print(f"  Invalid annotations: {len(report['invalid_annotations'])}")
    for error in report["errors"]:
        print(f"  ERROR: {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
