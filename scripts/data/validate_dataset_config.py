#!/usr/bin/env python3
"""Validate DDG dataset and source download configuration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DEFAULT_CONFIG = Path("configs/datasets.yaml")
DATASET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
SOURCE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
DOWNLOAD_MODES = {"local_file", "direct_download", "manual_download", "remote_or_manual"}
REQUIRED_DATASET_KEYS = {
    "role",
    "description",
    "raw_dir",
    "interim_dir",
    "processed_dir",
    "results_dir",
    "reports_dir",
    "allowed_for_tuning",
    "allowed_for_final_benchmark",
    "sources",
}
REQUIRED_DOWNLOAD_KEYS = {
    "mode",
    "urls",
    "expected_path",
    "expected_filename_patterns",
    "required",
    "allow_parallel",
    "checksum",
    "notes",
}


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset config must be a mapping: {config_path}")
    return data


def is_relative_safe_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def validate_download(prefix: str, download: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(download, dict):
        return [f"{prefix}.download must be a mapping."]

    missing = sorted(REQUIRED_DOWNLOAD_KEYS - download.keys())
    if missing:
        errors.append(f"{prefix}.download: missing keys: {', '.join(missing)}")

    mode = download.get("mode")
    if mode not in DOWNLOAD_MODES:
        errors.append(f"{prefix}.download.mode must be one of: {', '.join(sorted(DOWNLOAD_MODES))}")

    urls = download.get("urls")
    if not isinstance(urls, list):
        errors.append(f"{prefix}.download.urls must be a list.")

    patterns = download.get("expected_filename_patterns")
    if not isinstance(patterns, list):
        errors.append(f"{prefix}.download.expected_filename_patterns must be a list.")

    expected_path = download.get("expected_path")
    if expected_path is not None and (
        not isinstance(expected_path, str) or not is_relative_safe_path(expected_path)
    ):
        errors.append(f"{prefix}.download.expected_path must be null or a safe relative path.")

    for key in ("required", "allow_parallel"):
        if not isinstance(download.get(key), bool):
            errors.append(f"{prefix}.download.{key} must be boolean.")

    checksum = download.get("checksum")
    if checksum is not None and not isinstance(checksum, str):
        errors.append(f"{prefix}.download.checksum must be null or string.")

    if not isinstance(download.get("notes"), str):
        errors.append(f"{prefix}.download.notes must be a string.")

    return errors


def validate(config_path: Path) -> list[str]:
    errors: list[str] = []
    config = load_config(config_path)
    root = config_path.parent.parent

    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        return ["'datasets' must be a non-empty mapping."]

    for schema_name, schema_path in (config.get("schemas") or {}).items():
        if not isinstance(schema_path, str) or not is_relative_safe_path(schema_path):
            errors.append(f"schemas.{schema_name} must be a safe relative path.")
        elif not (root / schema_path).exists():
            errors.append(f"schemas.{schema_name} does not exist: {schema_path}")

    manifest_dir = config.get("manifest_dir")
    if manifest_dir is not None and (
        not isinstance(manifest_dir, str) or not is_relative_safe_path(manifest_dir)
    ):
        errors.append("manifest_dir must be a safe relative path.")

    for dataset_name, dataset in datasets.items():
        prefix = f"datasets.{dataset_name}"
        if not DATASET_NAME_RE.match(dataset_name):
            errors.append(f"{prefix}: name must match {DATASET_NAME_RE.pattern}")
        if not isinstance(dataset, dict):
            errors.append(f"{prefix} must be a mapping.")
            continue

        missing = sorted(REQUIRED_DATASET_KEYS - dataset.keys())
        if missing:
            errors.append(f"{prefix}: missing keys: {', '.join(missing)}")

        for key in ("raw_dir", "interim_dir", "processed_dir", "results_dir", "reports_dir"):
            value = dataset.get(key)
            if not isinstance(value, str) or not is_relative_safe_path(value):
                errors.append(f"{prefix}.{key} must be a safe relative path.")

        for key in ("allowed_for_tuning", "allowed_for_final_benchmark"):
            if not isinstance(dataset.get(key), bool):
                errors.append(f"{prefix}.{key} must be boolean.")

        if dataset.get("allowed_for_final_benchmark") and dataset.get("allowed_for_tuning"):
            errors.append(f"{prefix}: final benchmark datasets cannot be allowed for tuning.")

        sources = dataset.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{prefix}.sources must be a non-empty list.")
            continue

        for index, source in enumerate(sources):
            source_prefix = f"{prefix}.sources[{index}]"
            if not isinstance(source, dict):
                errors.append(f"{source_prefix} must be a mapping.")
                continue

            source_name = source.get("name")
            if not isinstance(source_name, str) or not SOURCE_NAME_RE.match(source_name):
                errors.append(f"{source_prefix}.name must match {SOURCE_NAME_RE.pattern}")
            if not isinstance(source.get("source_type"), str):
                errors.append(f"{source_prefix}.source_type must be a string.")
            if source.get("raw_dir") is not None:
                raw_dir = source.get("raw_dir")
                if not isinstance(raw_dir, str) or not is_relative_safe_path(raw_dir):
                    errors.append(f"{source_prefix}.raw_dir must be a safe relative path.")

            errors.extend(validate_download(source_prefix, source.get("download")))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    try:
        errors = validate(args.config)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"OK: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
