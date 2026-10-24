#!/usr/bin/env python3
"""Build or preview portable run bundles for current_internal stage groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_CONFIG = ROOT / "configs" / "server_paths.yaml"
EXAMPLE_SERVER_CONFIG = ROOT / "configs" / "server_paths.example.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def bundle_relative_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return Path("configs") / path.name


def iter_files(path: Path, *, include_server_config: bool = False) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        if path.name == "server_paths.yaml" and not include_server_config:
            return []
        return [path]
    return [
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and not is_machine_specific_config(item, include_server_config=include_server_config)
    ]


def is_machine_specific_config(path: Path, *, include_server_config: bool) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        relative = path
    text = relative.as_posix()
    name = path.name.lower()
    if name.endswith(".example.yaml"):
        return False
    if text == "configs/server_paths.yaml":
        return not include_server_config
    if text in {"configs/tools/foldx.yaml", "configs/tools/rosetta.yaml", "configs/tools/openfold.yaml"}:
        return True
    return text.startswith("configs/") and ("local" in name or "secret" in name) and name.endswith((".yaml", ".yml"))


def bundle_inputs(
    dataset: str,
    stage_group: str,
    include_structures: bool,
    include_raw: bool,
    include_results: bool,
    include_server_config: bool,
    server_config_path: Path,
) -> tuple[list[Path], dict[str, bool]]:
    categories = {
        "configs": True,
        "scripts": True,
        "docs": True,
        "processed": True,
        "manifests": True,
        "reports": True,
        "structures": include_structures or stage_group == "cpu_heavy",
        "raw": include_raw,
        "results": include_results,
        "server_config": include_server_config,
    }
    paths = [ROOT / "configs", ROOT / "scripts", ROOT / "docs"]
    if stage_group in {"preprepare", "postprocess", "all"}:
        paths.extend([ROOT / "data" / "processed" / dataset, ROOT / "data" / "manifests", ROOT / "reports" / dataset])
    if stage_group == "cpu_heavy":
        paths.extend(
            [
                ROOT / "data" / "processed" / dataset / f"{dataset}_recalc_jobs_resolved.csv",
                ROOT / "data" / "manifests",
                ROOT / "data" / "structures" / "chain" / dataset,
                ROOT / "data" / "sequences" / dataset,
            ]
        )
    if stage_group == "gpu_heavy":
        paths.extend(
            [
                ROOT / "data" / "processed" / dataset / f"{dataset}_recalc_jobs_resolved.csv",
                ROOT / "data" / "processed" / dataset / f"{dataset}_evaluation_tasks.csv",
                ROOT / "data" / "manifests" / f"{dataset}_sequence_manifest.csv",
                ROOT / "data" / "sequences" / dataset,
            ]
        )
    if include_structures and stage_group != "cpu_heavy":
        paths.append(ROOT / "data" / "structures" / "chain" / dataset)
    if include_raw:
        paths.append(ROOT / "data" / "raw" / dataset)
    if include_results:
        paths.extend([ROOT / "results" / dataset, ROOT / "results" / "evaluation" / dataset])
    if include_server_config:
        paths.append(server_config_path)
    return paths, categories


def load_server_config_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path if path.is_absolute() else ROOT / path
    used = path if path.exists() else EXAMPLE_SERVER_CONFIG
    config: dict[str, Any] = {}
    if used.exists():
        with used.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    server = config.get("server", {}) if isinstance(config, dict) else {}
    return config, {
        "server_config_path_used": display_path(used),
        "server_config_requested": display_path(path),
        "server_config_exists": path.exists(),
        "server_config_is_example_fallback": not path.exists() and used == EXAMPLE_SERVER_CONFIG,
        "server_project_root": server.get("project_root", ""),
        "intended_results_root": server.get("results_root", ""),
        "intended_logs_root": server.get("logs_root", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--stage-group", required=True, choices=["preprepare", "cpu_heavy", "gpu_heavy", "postprocess", "all"])
    parser.add_argument("--output-dir", type=Path, default=Path("bundles"))
    parser.add_argument("--include-structures", action="store_true")
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--include-results", action="store_true")
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER_CONFIG)
    parser.add_argument("--include-server-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with (ROOT / "configs" / "datasets.yaml").open("r", encoding="utf-8") as handle:
        datasets = yaml.safe_load(handle) or {}
    if args.dataset not in datasets.get("datasets", {}):
        raise SystemExit(f"Unknown dataset: {args.dataset}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_name = f"{args.dataset}_{args.stage_group}_{timestamp}"
    bundle_dir = ROOT / args.output_dir / bundle_name
    archive_path = ROOT / args.output_dir / f"{bundle_name}.tar.gz"
    manifest_path = ROOT / args.output_dir / f"{bundle_name}_manifest.json"
    server_config_path = args.server_config if args.server_config.is_absolute() else ROOT / args.server_config
    if args.include_server_config and not server_config_path.exists():
        raise SystemExit(f"--include-server-config was set, but server config does not exist: {server_config_path}")
    source_paths, categories = bundle_inputs(
        args.dataset,
        args.stage_group,
        args.include_structures,
        args.include_raw,
        args.include_results,
        args.include_server_config,
        server_config_path,
    )
    _, server_metadata = load_server_config_metadata(args.server_config)
    files = []
    for source in source_paths:
        files.extend(iter_files(source, include_server_config=args.include_server_config))
    unique_files = sorted(set(files))
    print(f"Dataset: {args.dataset}")
    print(f"Stage group: {args.stage_group}")
    print(f"Files planned: {len(unique_files)}")
    print(f"Bundle dir: {bundle_dir.relative_to(ROOT).as_posix()}")
    print(f"Archive: {archive_path.relative_to(ROOT).as_posix()}")
    print(f"Server config metadata: {server_metadata}")
    if args.include_server_config:
        print("WARNING: including machine-specific configs/server_paths.yaml in the bundle because --include-server-config was set.")
    if args.dry_run:
        for path in unique_files[:50]:
            print(f"  - {display_path(path)}")
        if len(unique_files) > 50:
            print(f"  ... {len(unique_files) - 50} more")
        print("Dry run: no bundle was written.")
        return 0
    if bundle_dir.exists() and not args.force:
        raise SystemExit(f"Bundle directory exists; use --force: {bundle_dir}")
    if archive_path.exists() and not args.force:
        raise SystemExit(f"Archive exists; use --force: {archive_path}")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    file_records: list[dict[str, Any]] = []
    for source in unique_files:
        relative = bundle_relative_path(source)
        target = bundle_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        file_records.append({"path": relative.as_posix(), "size": source.stat().st_size, "sha256": sha256_file(source)})
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(bundle_dir, arcname=bundle_dir.name)
    payload = {
        "dataset": args.dataset,
        "stage_group": args.stage_group,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "files": file_records,
        "included_categories": [name for name, enabled in categories.items() if enabled],
        "excluded_categories": [name for name, enabled in categories.items() if not enabled],
        "archive": archive_path.relative_to(ROOT).as_posix(),
        "server_config_metadata": server_metadata,
        "server_config_included": args.include_server_config,
        "server_config_included_path": display_path(server_config_path) if args.include_server_config else "",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Bundle written: {bundle_dir.relative_to(ROOT).as_posix()}")
    print(f"Archive written: {archive_path.relative_to(ROOT).as_posix()}")
    print(f"Manifest written: {manifest_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
