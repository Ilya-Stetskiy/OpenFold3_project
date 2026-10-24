#!/usr/bin/env python3
"""List DDG datasets declared in configs/datasets.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DEFAULT_CONFIG = Path("configs/datasets.yaml")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Dataset config must be a mapping: {config_path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--names-only", action="store_true", help="Print only dataset identifiers.")
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        raise SystemExit("'datasets' must be a mapping.")

    for name, dataset in datasets.items():
        if args.names_only:
            print(name)
            continue
        role = dataset.get("role", "unknown")
        tuning = dataset.get("allowed_for_tuning", "unknown")
        final = dataset.get("allowed_for_final_benchmark", "unknown")
        print(f"{name}\trole={role}\tallowed_for_tuning={tuning}\tallowed_for_final_benchmark={final}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
