"""Derived YOLO and COCO views of the dataset, without the two zero-area test boxes (see DATA.md).

Ultralytics keeps zero-area boxes, which would count as guaranteed misses. The original
export is never modified; the audit and known-invalid list come from scripts/data/.
"""

import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from aquadet import ROOT

sys.path.insert(0, str(ROOT / "scripts" / "data"))
from prepare_data import EXPECTED_COUNTS, IMAGE_EXTENSIONS, verify  # noqa: E402
from validate_coco import validate as validate_coco_file  # noqa: E402
from yolo_to_coco import KNOWN_INVALID, convert as convert_to_coco, issue_key  # noqa: E402

COCO_SPLIT = {"train": "train", "val": "valid", "test": "test"}
MANIFEST = "view_manifest.json"


def check_policy(audit: dict, source: Path, policy: str) -> set[tuple]:
    """Records to omit; raises on any finding other than the reviewed ones."""
    found = {issue_key(item, source) for item in audit["invalid_annotations"]}
    if policy == "strict":
        if not audit["passed"]:
            raise ValueError(f"Audit failed ({len(audit['errors'])} errors) and policy is strict; see DATA.md")
        return set()
    if policy != "omit-known-zero-boxes":
        raise ValueError(f"Unknown invalid_policy {policy!r}")
    if found != KNOWN_INVALID or len(audit["errors"]) != len(KNOWN_INVALID):
        raise ValueError("Audit findings differ from the two reviewed zero-area boxes; refusing to continue. "
                         f"Errors: {audit['errors'][:5]}")
    return found


def filter_label_lines(text: str, rel_label: str, omit: set[tuple]) -> tuple[list[str], list[dict]]:
    kept, omitted = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if (rel_label, number, parts[0], tuple(parts[1:])) in omit:
            omitted.append({"file": rel_label, "line": number, "class_id": parts[0],
                            "normalized_xywh": parts[1:]})
            continue
        kept.append(line.strip())
    return kept, omitted


def link_or_copy(src: Path, dst: Path) -> str:
    for mode, action in (("hardlink", os.link), ("symlink", os.symlink)):
        try:
            action(src, dst)
            return mode
        except OSError:
            continue
    shutil.copy2(src, dst)
    return "copy"


def build_yolo_view(source: Path, dest: Path, policy: str) -> dict:
    source, dest = source.resolve(), dest.resolve()
    if source == dest or source in dest.parents:
        raise ValueError("The derived view must live outside the original export")
    audit = verify(source)
    omit = check_policy(audit, source, policy)

    staging = dest.with_name(dest.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    names = [audit["id_to_name"][str(i)] for i in range(len(audit["id_to_name"]))]
    link_modes, omitted, counts = set(), [], {}
    for split in EXPECTED_COUNTS:
        directory = audit["splits"][split]["directory"]
        (staging / directory / "images").mkdir(parents=True)
        (staging / directory / "labels").mkdir(parents=True)
        n_images = n_boxes = 0
        for image in sorted((source / directory / "images").iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            link_modes.add(link_or_copy(image, staging / directory / "images" / image.name))
            label = source / directory / "labels" / (image.stem + ".txt")
            rel_label = label.relative_to(source).as_posix()
            kept, dropped = filter_label_lines(label.read_text(encoding="utf-8-sig"), rel_label, omit)
            (staging / directory / "labels" / label.name).write_text(
                "".join(line + "\n" for line in kept), encoding="utf-8")
            omitted.extend(dropped)
            n_images += 1
            n_boxes += len(kept)
        counts[split] = {"directory": directory, "images": n_images, "boxes": n_boxes}

    if {issue_key({**o, "file": str(source / o["file"])}, source) for o in omitted} != omit:
        shutil.rmtree(staging)
        raise ValueError("Known invalid records were not each encountered exactly once")
    manifest = {"source": str(source), "policy": policy, "link_modes": sorted(link_modes),
                "names": names, "split_counts": counts, "omitted_source_annotations": omitted,
                "source_audit_passed": audit["passed"]}
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
    return manifest


def write_data_yaml(view: Path, manifest: dict) -> Path:
    """Absolute `path:` so Ultralytics does not resolve it against its datasets_dir."""
    counts = manifest["split_counts"]
    data = {"path": str(view.resolve()),
            "train": f"{counts['train']['directory']}/images",
            "val": f"{counts['valid']['directory']}/images",
            "test": f"{counts['test']['directory']}/images",
            "nc": len(manifest["names"]),
            "names": dict(enumerate(manifest["names"]))}
    path = view / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def prepare(config: dict, rebuild: bool = False) -> dict:
    """Build or verify both derived views."""
    from aquadet.config import resolve

    ds = config["dataset"]
    source, view, coco_dir = resolve(ds["source"]), resolve(ds["yolo_view"]), resolve(ds["coco_dir"])
    if not (source / "data.yaml").is_file():
        raise ValueError(f"Original export not found at {source}. Pass --source PATH or set AQUADET_SOURCE.")

    manifest_path = view / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    stale = manifest is None or manifest["source"] != str(source.resolve()) or manifest["policy"] != ds["invalid_policy"]
    if rebuild or stale:
        print(f"[prepare] building YOLO view {view} from {source}")
        manifest = build_yolo_view(source, view, ds["invalid_policy"])
    data_yaml = write_data_yaml(view, manifest)

    if not all((coco_dir / f"{s}.json").is_file() for s in COCO_SPLIT.values()):
        if coco_dir.exists() and any(coco_dir.iterdir()):
            raise ValueError(f"{coco_dir} is incomplete; remove it so it can be regenerated")
        print(f"[prepare] converting to COCO ground truth in {coco_dir}")
        convert_to_coco(source, coco_dir, ds["invalid_policy"])
    for split, coco_name in COCO_SPLIT.items():
        with contextlib.redirect_stdout(io.StringIO()):
            n_images, n_boxes = validate_coco_file(coco_dir / f"{coco_name}.json")
        expected = manifest["split_counts"][coco_name]
        if (n_images, n_boxes) != (expected["images"], expected["boxes"]):
            raise ValueError(f"{split}: COCO has {n_images} images/{n_boxes} boxes but YOLO view has "
                             f"{expected['images']}/{expected['boxes']}; rebuild with --rebuild")
    return {"source": source, "view": view, "data_yaml": data_yaml, "coco_dir": coco_dir, "manifest": manifest}
