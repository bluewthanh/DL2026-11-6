"""Load configs/base.yaml plus one model config, with recorded overrides."""

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from aquadet import ROOT

CONFIG_DIR = ROOT / "configs"
MODEL_DIR = CONFIG_DIR / "models"
SMOKE_OVERRIDES = {"train.epochs": 1, "benchmark.warmup": 3, "benchmark.images": 10, "benchmark.repeats": 1}


def available_models() -> list[str]:
    return sorted(p.stem for p in MODEL_DIR.glob("*.yaml"))


def parse_override(text: str) -> tuple[str, object]:
    """`train.epochs=50` -> ("train.epochs", 50)."""
    key, sep, value = text.partition("=")
    if not sep or "." not in key:
        raise ValueError(f"Override must look like section.key=value, got {text!r}")
    return key.strip(), yaml.safe_load(value)


def apply_overrides(config: dict, overrides: dict[str, object]) -> dict:
    result = copy.deepcopy(config)
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if section not in result or not isinstance(result[section], dict):
            raise ValueError(f"Unknown config section in override {dotted!r}")
        if key not in result[section]:
            raise ValueError(f"Unknown key {key!r} in section {section!r}; check configs/base.yaml")
        result[section][key] = value
    return result


def protocol_hash(config: dict) -> str:
    """Hash of the settings that must match across models. Seed, workers and plots are excluded."""
    shared = copy.deepcopy({k: config[k] for k in ("train", "eval", "benchmark")})
    for key in ("seed", "workers", "plots"):
        shared["train"].pop(key, None)
    shared["dataset_policy"] = config["dataset"]["invalid_policy"]
    blob = json.dumps(shared, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def load(model: str, overrides: list[str] | None = None, source: str | None = None,
         smoke: bool = False) -> dict:
    model_file = MODEL_DIR / f"{model}.yaml"
    if not model_file.exists():
        raise ValueError(f"Unknown model {model!r}; available: {', '.join(available_models())}")
    config = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    config["model"] = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    reference_hash = protocol_hash(config)

    parsed = dict(parse_override(item) for item in overrides or [])
    if smoke:
        parsed = {**SMOKE_OVERRIDES, **parsed}
    config = apply_overrides(config, parsed)
    source = source or os.environ.get("AQUADET_SOURCE")
    if source:
        config["dataset"]["source"] = source

    config["overrides"] = {k: v for k, v in parsed.items()}
    config["smoke"] = smoke
    config["protocol_hash"] = protocol_hash(config)
    config["protocol_matches_base"] = config["protocol_hash"] == reference_hash
    return config


def resolve(path: str | Path) -> Path:
    """Resolve relative to the repository root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def default_run_name(config: dict) -> str:
    name = f"{config['model']['name']}_e{config['train']['epochs']}_s{config['train']['seed']}"
    if config["smoke"]:
        return name + "_smoke"
    if not config["protocol_matches_base"]:
        return name + "_custom"
    return name
