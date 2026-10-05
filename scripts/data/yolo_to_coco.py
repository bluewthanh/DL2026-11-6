"""Convert audited Aquarium v2 YOLO boxes to split-preserving COCO JSON.

Default is strict and writes nothing if any source annotation is invalid.
The explicit --invalid-policy omit-known-zero-boxes option permits ONLY the two
known zero-area test records; images and all other annotations remain present.
Never changes the source dataset. Requires PyYAML and Pillow.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from prepare_data import EXPECTED_COUNTS, verify

# (relative label path, line, class ID, normalized xywh as present in source)
KNOWN_INVALID = {
    ("test/labels/IMG_2423_jpeg_jpg.rf.39aca9cd118509b10f192b87e7ce9692.txt", 17,
     "4", ("0.5826822916666666", "0.2958984375", "0", "0")),
    ("test/labels/IMG_2570_jpeg_jpg.rf.05001e7087160e744b14f28f1fa5c768.txt", 15,
     "4", ("0.059895833333333336", "0.86572265625", "0", "0")),
}


def issue_key(issue: dict, source: Path) -> tuple:
    return (Path(issue["file"]).resolve().relative_to(source.resolve()).as_posix(),
            issue["line"], issue["class_id"], tuple(issue["normalized_xywh"]))


def convert(source: Path, destination: Path, policy: str) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be outside the original export directory")
    if any((destination / f"{split}.json").exists() for split in EXPECTED_COUNTS):
        raise ValueError("COCO output exists; use an empty output folder to avoid overwriting results")
    audit = verify(source)
    invalid = audit["invalid_annotations"]
    if policy == "strict" and not audit["passed"]:
        raise ValueError(f"Audit failed ({len(audit['errors'])} errors). No COCO output written. "
                         "Use --invalid-policy omit-known-zero-boxes only after reviewing DATA.md")
    known = {issue_key(item, source) for item in invalid}
    if policy == "omit-known-zero-boxes":
        if known != KNOWN_INVALID or len(invalid) != 2 or len(audit["errors"]) != 2 or any(
            item["reason"] != "width/height must be finite and strictly positive within (0, 1]"
            for item in invalid
        ):
            raise ValueError("Unexpected audit findings; refusing to omit any annotations")
    elif policy != "strict":
        raise ValueError(f"Unknown policy: {policy}")
    skipped = []
    outputs = {}
    categories = [{"id": int(i) + 1, "name": name, "supercategory": "object"}
                  for i, name in audit["id_to_name"].items()]
    # Build every split before writing anything. COCO category ID = YOLO class ID + 1.
    for split, expected in EXPECTED_COUNTS.items():
        directory = audit["splits"][split].get("directory", split)
        images_dir = source / directory / "images"
        labels_dir = source / directory / "labels"
        images = []
        annotations = []
        for image_path in sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower()
                                 in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}):
            image_id = len(images) + 1
            rel = image_path.relative_to(images_dir)
            label_path = (labels_dir / rel).with_suffix(".txt")
            with Image.open(image_path) as img:
                width, height = img.size
                img.verify()
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image size: {image_path}")
            images.append({"id": image_id, "file_name": image_path.relative_to(source).as_posix(),
                           "width": width, "height": height})
            for line_num, text in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not text.strip():
                    continue
                parts = text.split()
                key = (label_path.relative_to(source).as_posix(), line_num, parts[0], tuple(parts[1:]))
                if key in known:
                    skipped.append({"file": key[0], "line": line_num, "class_id": parts[0],
                                    "normalized_xywh": list(parts[1:]),
                                    "reason": "zero-area source box cannot be a COCO ground-truth box"})
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
                x, y, w_px, h_px = (cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height
                annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                    "category_id": class_id + 1, "bbox": [x, y, w_px, h_px],
                                    "area": w_px * h_px, "iscrowd": 0})
        if len(images) != expected:
            raise ValueError(f"{split}: expected {expected} images, found {len(images)}")
        outputs[split] = {"info": {"description": "RF100 aquarium-qlnqy v2 YOLOv8 export"},
                          "licenses": [], "images": images, "annotations": annotations,
                          "categories": categories}
    skipped_keys = [(item["file"], item["line"], item["class_id"],
                     tuple(item["normalized_xywh"])) for item in skipped]
    if len(skipped_keys) != len(known) or set(skipped_keys) != known:
        raise ValueError(f"Known invalid records not encountered exactly once: {skipped_keys}")
    destination.mkdir(parents=True, exist_ok=True)
    for split, data in outputs.items():
        (destination / f"{split}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    manifest = {"source": str(source), "policy": policy,
                "image_file_names_are_relative_to": str(source),
                "category_id_map": {str(int(i) + 1): name for i, name in audit["id_to_name"].items()},
                "split_counts": {s: {"images": len(d["images"]), "annotations": len(d["annotations"])}
                                 for s, d in outputs.items()},
                "omitted_source_annotations": skipped,
                "source_audit_passed": audit["passed"]}
    (destination / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Unmodified YOLO export root")
    parser.add_argument("--output", type=Path, default=Path("data/aquarium-v2-coco"))
    parser.add_argument("--invalid-policy", choices=("strict", "omit-known-zero-boxes"), default="strict")
    args = parser.parse_args()
    try:
        result = convert(args.source, args.output, args.invalid_policy)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Conversion refused: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
