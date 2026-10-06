"""Package Aquarium v2 derived COCO annotations (NO images or source labels).

Run after yolo_to_coco.py and validate_coco.py. Creates a portable ZIP for
sharing the processed ground truth; source images must be downloaded separately.
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

SPLITS = {"train": (448, 3328), "valid": (127, 909), "test": (63, 582)}
PREFIX = "aquarium-v2-derived-coco/"
NOTICE = """Roboflow 100 Aquarium v2 — derived COCO ground truth (annotations only)

Official dataset and images: https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2
Source: Roboflow 100 Aquarium v2, CC BY 4.0 (see official project page).
License terms: https://creativecommons.org/licenses/by/4.0/
Conversion code/policy: https://github.com/bluewthanh/DL2026-11-6/blob/main/DATA.md

The ZIP contains derived train/valid/test COCO JSON, a portable conversion
manifest, and this notice. It contains NO images, source YOLO labels, weights,
secrets, or machine paths. COCO image file_name values are relative to the
root of the *separately downloaded* original Aquarium v2 YOLOv8 export.
All source images and original annotations remain unchanged. Two zero-area
shark boxes are omitted ONLY from derived test ground truth; see manifest
and DATA.md. Attribution and these changes must be retained when reusing.

To reproduce from the official export, run scripts/data/yolo_to_coco.py with
--invalid-policy omit-known-zero-boxes, then scripts/data/validate_coco.py.
Use the same source images and derived ground truth for all model evaluations.
"""


def archive(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir() or output.is_relative_to(source):
        raise ValueError("Input must exist; output must be outside the source directory")
    entries = {}
    for split, (images, boxes) in SPLITS.items():
        path = source / f"{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if len(data["images"]) != images or len(data["annotations"]) != boxes:
            raise ValueError(f"Unexpected {split} counts; validate before packaging")
        if any(Path(item["file_name"]).is_absolute() or ".." in Path(item["file_name"]).parts
               for item in data["images"]):
            raise ValueError(f"Non-portable {split} image path")
        entries[f"{split}.json"] = path.read_bytes().replace(b"\r\n", b"\n")
    manifest = json.loads((source / "conversion_manifest.json").read_text(encoding="utf-8"))
    if manifest["policy"] != "omit-known-zero-boxes" or len(manifest["omitted_source_annotations"]) != 2:
        raise ValueError("Unexpected omission policy")
    # Local generation manifests contain absolute machine paths; they are not needed for reuse.
    manifest["source"] = "Obtain the original Aquarium v2 YOLOv8 export at the official URL in README.txt"
    manifest["image_file_names_are_relative_to"] = "root of the original Aquarium v2 YOLOv8 export"
    entries["conversion_manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    entries["README.txt"] = NOTICE.encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for name, content in sorted(entries.items()):
            info = ZipInfo(PREFIX + name, date_time=(2026, 10, 6, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, content, compress_type=ZIP_DEFLATED, compresslevel=9)
    return {"file": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "files": sorted(entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/aquarium-v2-coco"))
    parser.add_argument("--output", type=Path, default=Path("data/aquarium-v2-derived-coco.zip"))
    args = parser.parse_args()
    print(json.dumps(archive(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
