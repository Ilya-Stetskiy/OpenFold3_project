#!/usr/bin/env python3
"""Create dataset directories declared in configs/datasets.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DEFAULT_CONFIG = Path("configs/datasets.yaml")
DIR_PATH_KEYS = ("raw_dir", "interim_dir", "processed_dir", "results_dir", "reports_dir")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Dataset config must be a mapping: {config_path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", help="Initialize only one dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Print directories without creating them.")
    args = parser.parse_args()

    config = load_config(args.config)
    root = args.config.parent.parent
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        raise SystemExit("'datasets' must be a mapping.")

    selected = {args.dataset: datasets.get(args.dataset)} if args.dataset else datasets
    if args.dataset and selected[args.dataset] is None:
        raise SystemExit(f"Unknown dataset: {args.dataset}")

    manifest_dir = config.get("manifest_dir")
    if isinstance(manifest_dir, str):
        selected_paths = [root / manifest_dir]
    else:
        selected_paths = []

    for dataset_name, dataset in selected.items():
        if not isinstance(dataset, dict):
            raise SystemExit(f"Dataset config must be a mapping: {dataset_name}")
        for key in DIR_PATH_KEYS:
            value = dataset.get(key)
            if not isinstance(value, str):
                raise SystemExit(f"Missing path '{key}' for dataset: {dataset_name}")
            selected_paths.append(root / value)
        for source in dataset.get("sources", []):
            if isinstance(source, dict) and isinstance(source.get("raw_dir"), str):
                selected_paths.append(root / source["raw_dir"])

    for path in selected_paths:
        if args.dry_run:
            print(path)
            continue
        path.mkdir(parents=True, exist_ok=True)
        keep_file = path / ".gitkeep"
        keep_file.touch(exist_ok=True)
        print(f"created {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
