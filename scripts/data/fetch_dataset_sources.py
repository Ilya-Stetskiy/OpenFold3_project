#!/usr/bin/env python3
"""Fetch or register raw DDG dataset sources.

This script only obtains or registers raw source files. It does not normalize
datasets, parse mutations, fetch structures, run predictors, or merge datasets.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DATASETS_CONFIG = Path("configs/datasets.yaml")
DOWNLOADS_CONFIG = Path("configs/downloads.yaml")
MANIFEST_FIELDS = [
    "dataset_name",
    "dataset_role",
    "source_name",
    "source_type",
    "source_url",
    "local_path",
    "download_status",
    "download_mode",
    "downloaded_at",
    "file_size_bytes",
    "sha256",
    "requires_manual_download",
    "manual_instruction_path",
    "error_message",
    "notes",
    "file_label",
    "is_required",
    "allow_parallel",
    "attempts",
    "elapsed_seconds",
]
SUMMARY_KEYS = [
    "downloaded",
    "registered",
    "already_exists",
    "registered_manual_file",
    "requires_manual_download",
    "missing",
    "failed",
    "checksum_failed",
]


@dataclass(frozen=True)
class FetchOptions:
    workers: int
    retries: int
    timeout: int
    force: bool
    dry_run: bool
    no_progress_bar: bool
    log_dir: Path
    manifest_csv: Path
    manifest_json: Path


@dataclass(frozen=True)
class SourceJob:
    dataset_name: str
    dataset_role: str
    source_name: str
    source_type: str
    source_dir: Path
    mode: str
    url: str
    local_path: Path
    file_label: str
    expected_patterns: tuple[str, ...]
    required: bool
    allow_parallel: bool
    checksum: str | None
    notes: str


@dataclass
class JobResult:
    dataset_name: str
    dataset_role: str
    source_name: str
    source_type: str
    source_url: str
    local_path: str
    download_status: str
    download_mode: str
    downloaded_at: str
    file_size_bytes: int | str
    sha256: str
    requires_manual_download: bool
    manual_instruction_path: str
    error_message: str
    notes: str
    file_label: str
    is_required: bool
    allow_parallel: bool
    attempts: int
    elapsed_seconds: float


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return data


def project_root_from(config_path: Path) -> Path:
    return config_path.parent.parent.resolve()


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    if not name:
        token = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        return f"download_{token}"
    return name


def ensure_parent(path: Path, dry_run: bool = False) -> None:
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)


def setup_logging(root: Path, log_dir: Path) -> tuple[logging.Logger, Path]:
    full_log_dir = root / log_dir
    full_log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_log = full_log_dir / f"{timestamp}_{os.getpid()}_fetch_dataset_sources.log"
    latest_log = full_log_dir / "fetch_latest.log"

    logger = logging.getLogger("fetch_dataset_sources")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for path in (run_log, latest_log):
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, run_log


def source_raw_dir(dataset: dict[str, Any], source: dict[str, Any]) -> str:
    raw_dir = source.get("raw_dir")
    if isinstance(raw_dir, str):
        return raw_dir
    return dataset["raw_dir"]


def manual_instruction_path(dataset: dict[str, Any], source: dict[str, Any]) -> Path:
    mode = source.get("download", {}).get("mode")
    if mode == "local_file":
        return Path(source_raw_dir(dataset, source)) / "README_put_results_here.md"
    return Path(source_raw_dir(dataset, source)) / f"manual_download_{source['name']}.md"


def write_manual_stub(root: Path, dataset_name: str, dataset: dict[str, Any], source: dict[str, Any]) -> Path:
    relative_path = manual_instruction_path(dataset, source)
    output_path = root / relative_path
    download = source["download"]
    patterns = download.get("expected_filename_patterns", [])
    pattern_lines = "\n".join(f"- `{pattern}`" for pattern in patterns) or "- no specific pattern"
    expected_path = download.get("expected_path")
    if download["mode"] == "local_file":
        purpose = (
            "This source is a local file, not an internet download. Put the "
            "current internal DDG CSV with experimental values and existing "
            "FoldX/Rosetta/ESM2/OpenFold results at the expected path."
        )
        target = expected_path or source_raw_dir(dataset, source)
    else:
        purpose = (
            "Download or export this source manually, then rerun "
            "`python3 scripts/data/fetch_dataset_sources.py --dataset "
            f"{dataset_name}` to register the file."
        )
        target = source_raw_dir(dataset, source)

    content = f"""# Manual source setup: {source['name']}

Dataset: `{dataset_name}`
Dataset role: `{dataset['role']}`
Source type: `{source.get('source_type', 'unknown')}`
Download mode: `{download['mode']}`
Required: `{download['required']}`

{purpose}

Put the file(s) here:

```text
{target}
```

Expected filename patterns:

{pattern_lines}

Notes:

{download.get('notes') or 'No additional notes.'}
"""
    ensure_parent(output_path)
    output_path.write_text(content, encoding="utf-8")
    return relative_path


def iter_selected_datasets(config: dict[str, Any], selected: str) -> list[tuple[str, dict[str, Any]]]:
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict):
        raise SystemExit("'datasets' must be a mapping.")
    if selected == "all":
        return list(datasets.items())
    dataset = datasets.get(selected)
    if dataset is None:
        raise SystemExit(f"Unknown dataset: {selected}")
    return [(selected, dataset)]


def list_datasets(config: dict[str, Any]) -> None:
    for name, dataset in iter_selected_datasets(config, "all"):
        print(
            f"{name}\trole={dataset.get('role', 'unknown')}"
            f"\tallowed_for_tuning={dataset.get('allowed_for_tuning')}"
            f"\tallowed_for_final_benchmark={dataset.get('allowed_for_final_benchmark')}"
        )
        for source in dataset.get("sources", []):
            download = source.get("download", {})
            print(
                f"  - {source.get('name', 'unknown')}"
                f"\ttype={source.get('source_type', 'unknown')}"
                f"\tmode={download.get('mode', 'unknown')}"
                f"\trequired={download.get('required')}"
                f"\tallow_parallel={download.get('allow_parallel')}"
            )


def source_file_specs(source: dict[str, Any]) -> list[dict[str, Any]]:
    download = source.get("download", {})
    configured_files = source.get("files", download.get("files"))
    if isinstance(configured_files, list) and configured_files:
        specs = []
        for index, item in enumerate(configured_files):
            if not isinstance(item, dict):
                continue
            specs.append(
                {
                    "label": item.get("label") or item.get("name") or f"file_{index + 1}",
                    "url": item.get("url") or (item.get("urls") or [None])[0],
                    "expected_path": item.get("expected_path"),
                    "filename": item.get("filename"),
                    "checksum": item.get("checksum", download.get("checksum")),
                }
            )
        return specs

    urls = download.get("urls", [])
    if urls:
        return [
            {
                "label": f"file_{index + 1}",
                "url": url,
                "expected_path": None,
                "filename": safe_filename_from_url(url),
                "checksum": download.get("checksum"),
            }
            for index, url in enumerate(urls)
        ]

    return [
        {
            "label": source["name"],
            "url": "",
            "expected_path": download.get("expected_path"),
            "filename": None,
            "checksum": download.get("checksum"),
        }
    ]


def manifest_dataset_role(dataset_name: str, configured_role: str) -> str:
    if dataset_name == "independent_benchmark":
        return "benchmark"
    return configured_role


def build_jobs(
    root: Path,
    config: dict[str, Any],
    selected_dataset: str,
    selected_source: str | None,
) -> list[SourceJob]:
    jobs: list[SourceJob] = []
    for dataset_name, dataset in iter_selected_datasets(config, selected_dataset):
        for source in dataset.get("sources", []):
            if selected_source and source.get("name") != selected_source:
                continue
            download = source.get("download", {})
            source_dir = root / source_raw_dir(dataset, source)
            for spec in source_file_specs(source):
                url = spec.get("url") or ""
                expected_path = spec.get("expected_path")
                if expected_path:
                    local_path = root / expected_path
                elif spec.get("filename"):
                    local_path = source_dir / spec["filename"]
                else:
                    local_path = source_dir
                jobs.append(
                    SourceJob(
                        dataset_name=dataset_name,
                        dataset_role=manifest_dataset_role(dataset_name, dataset.get("role", "")),
                        source_name=source["name"],
                        source_type=source.get("source_type", ""),
                        source_dir=source_dir,
                        mode=download.get("mode", ""),
                        url=url,
                        local_path=local_path,
                        file_label=str(spec.get("label") or source["name"]),
                        expected_patterns=tuple(download.get("expected_filename_patterns", [])),
                        required=bool(download.get("required", False)),
                        allow_parallel=bool(download.get("allow_parallel", False)),
                        checksum=spec.get("checksum"),
                        notes=download.get("notes") or "",
                    )
                )
    return jobs


def matching_manual_files(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        return []
    files = [path for path in directory.iterdir() if path.is_file() and not path.name.startswith("manual_download_")]
    files = [path for path in files if not path.name.startswith("README_put_") and not path.name.endswith(".part")]
    if not patterns:
        return files
    return [path for path in files if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)]


def result_for(
    job: SourceJob,
    root: Path,
    status: str,
    options: FetchOptions,
    *,
    local_path: Path | None = None,
    sha256: str = "",
    size: int | str = "",
    manual_path: Path | None = None,
    error: str = "",
    attempts: int = 0,
    elapsed: float = 0.0,
) -> JobResult:
    path = local_path or job.local_path
    return JobResult(
        dataset_name=job.dataset_name,
        dataset_role=job.dataset_role,
        source_name=job.source_name,
        source_type=job.source_type,
        source_url=job.url,
        local_path=relpath(path, root),
        download_status=status,
        download_mode=job.mode,
        downloaded_at=utc_now(),
        file_size_bytes=size,
        sha256=sha256,
        requires_manual_download=status in {"missing", "requires_manual_download", "download_failed_manual_required"},
        manual_instruction_path=relpath(manual_path, root) if manual_path else "",
        error_message=error,
        notes=job.notes,
        file_label=job.file_label,
        is_required=job.required,
        allow_parallel=job.allow_parallel,
        attempts=attempts,
        elapsed_seconds=round(elapsed, 3),
    )


def register_existing_file(job: SourceJob, root: Path, status: str, options: FetchOptions) -> JobResult:
    digest = sha256_file(job.local_path)
    size = job.local_path.stat().st_size
    if job.checksum and digest.lower() != job.checksum.lower():
        return result_for(
            job,
            root,
            "checksum_failed",
            options,
            sha256=digest,
            size=size,
            error=f"Expected {job.checksum}, got {digest}",
        )
    return result_for(job, root, status, options, sha256=digest, size=size)


def register_manual_file(job: SourceJob, root: Path, path: Path, options: FetchOptions) -> JobResult:
    digest = sha256_file(path)
    size = path.stat().st_size
    if job.checksum and digest.lower() != job.checksum.lower():
        return result_for(
            job,
            root,
            "checksum_failed",
            options,
            local_path=path,
            sha256=digest,
            size=size,
            error=f"Expected {job.checksum}, got {digest}",
        )
    return result_for(job, root, "registered_manual_file", options, local_path=path, sha256=digest, size=size)


def download_to_file(job: SourceJob, root: Path, options: FetchOptions, logger: logging.Logger) -> JobResult:
    if not job.url:
        manual_path = write_manual_stub_for_job(root, job)
        return result_for(job, root, "requires_manual_download", options, manual_path=manual_path)

    ensure_parent(job.local_path)
    if job.local_path.exists() and not options.force:
        return register_existing_file(job, root, "already_exists", options)

    part_path = job.local_path.with_name(f"{job.local_path.name}.part")
    if part_path.exists():
        logger.info("Removing stale part file before retry: %s", relpath(part_path, root))
        part_path.unlink()

    start = time.monotonic()
    last_error = ""
    attempts = 0
    for attempts in range(1, options.retries + 1):
        try:
            logger.info("Download attempt %s/%s: %s -> %s", attempts, options.retries, job.url, relpath(job.local_path, root))
            request = urllib.request.Request(job.url, headers={"User-Agent": "OpenFold3-DDG-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=options.timeout) as response, part_path.open("wb") as output:
                shutil.copyfileobj(response, output)
            digest = sha256_file(part_path)
            size = part_path.stat().st_size
            if job.checksum and digest.lower() != job.checksum.lower():
                error = f"Expected {job.checksum}, got {digest}"
                logger.error("Checksum failed: %s", error)
                part_path.unlink()
                return result_for(
                    job,
                    root,
                    "checksum_failed",
                    options,
                    sha256=digest,
                    size=size,
                    error=error,
                    attempts=attempts,
                    elapsed=time.monotonic() - start,
                )
            part_path.replace(job.local_path)
            return result_for(
                job,
                root,
                "downloaded",
                options,
                sha256=digest,
                size=size,
                attempts=attempts,
                elapsed=time.monotonic() - start,
            )
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
            logger.warning("Download attempt failed (%s/%s): %s", attempts, options.retries, last_error)
            if attempts < options.retries:
                time.sleep(min(2.0, 0.25 * attempts))

    if part_path.exists():
        part_path.unlink()
    status = "download_failed_manual_required" if job.mode == "remote_or_manual" else "failed"
    manual_path = write_manual_stub_for_job(root, job) if job.mode == "remote_or_manual" else None
    return result_for(
        job,
        root,
        status,
        options,
        manual_path=manual_path,
        error=last_error,
        attempts=attempts,
        elapsed=time.monotonic() - start,
    )


def write_manual_stub_for_job(root: Path, job: SourceJob) -> Path:
    relative_path = job.source_dir.relative_to(root) / (
        "README_put_results_here.md" if job.mode == "local_file" else f"manual_download_{job.source_name}.md"
    )
    output_path = root / relative_path
    patterns = "\n".join(f"- `{pattern}`" for pattern in job.expected_patterns) or "- no specific pattern"
    target = relpath(job.local_path, root) if job.mode == "local_file" else relpath(job.source_dir, root)
    content = f"""# Manual source setup: {job.source_name}

Dataset: `{job.dataset_name}`
Dataset role: `{job.dataset_role}`
Source type: `{job.source_type}`
Download mode: `{job.mode}`
Required: `{job.required}`

Put the file(s) here:

```text
{target}
```

Expected filename patterns:

{patterns}

Notes:

{job.notes or 'No additional notes.'}

After placing the file, rerun:

```bash
python3 scripts/data/fetch_dataset_sources.py --dataset {job.dataset_name}
```
"""
    ensure_parent(output_path)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def process_job(job: SourceJob, root: Path, options: FetchOptions, logger: logging.Logger) -> list[JobResult]:
    logger.info("Start job: %s/%s mode=%s url=%s path=%s", job.dataset_name, job.source_name, job.mode, job.url, relpath(job.local_path, root))
    start = time.monotonic()

    if options.dry_run:
        status = "planned"
        return [result_for(job, root, status, options, elapsed=time.monotonic() - start)]

    job.source_dir.mkdir(parents=True, exist_ok=True)

    if job.mode == "local_file":
        if job.local_path.exists() and job.local_path.is_file():
            return [register_existing_file(job, root, "registered", options)]
        manual_path = write_manual_stub_for_job(root, job)
        return [result_for(job, root, "missing", options, manual_path=manual_path, error="Expected local file is missing")]

    if job.mode == "manual_download":
        files = matching_manual_files(job.source_dir, job.expected_patterns)
        if files:
            return [register_manual_file(job, root, path, options) for path in files]
        manual_path = write_manual_stub_for_job(root, job)
        return [result_for(job, root, "requires_manual_download", options, manual_path=manual_path)]

    if job.mode == "direct_download":
        return [download_to_file(job, root, options, logger)]

    if job.mode == "remote_or_manual":
        if job.url:
            return [download_to_file(job, root, options, logger)]
        files = matching_manual_files(job.source_dir, job.expected_patterns)
        if files:
            return [register_manual_file(job, root, path, options) for path in files]
        manual_path = write_manual_stub_for_job(root, job)
        return [result_for(job, root, "requires_manual_download", options, manual_path=manual_path)]

    return [result_for(job, root, "failed", options, error=f"Unknown download mode: {job.mode}")]


def result_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("dataset_name", "")),
        str(record.get("source_name", "")),
        str(record.get("file_label", "")),
        str(record.get("local_path", "")),
        str(record.get("source_url", "")),
    )


def load_manifest_csv(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {result_key(row): row for row in reader}


def backup_if_exists(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))


def write_manifests(root: Path, options: FetchOptions, results: list[JobResult]) -> None:
    csv_path = root / options.manifest_csv
    json_path = root / options.manifest_json
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_manifest_csv(csv_path)
    for result in results:
        record = asdict(result)
        existing[result_key(record)] = record

    rows = list(existing.values())
    rows.sort(key=result_key)
    backup_if_exists(csv_path)
    backup_if_exists(json_path)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"sources": rows}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def print_plan(jobs: list[SourceJob], root: Path) -> None:
    for job in jobs:
        print(f"\nDataset: {job.dataset_name}")
        print(f"  role: {job.dataset_role}")
        print(f"  source: {job.source_name}")
        print(f"  source_type: {job.source_type}")
        print(f"  file_label: {job.file_label}")
        print(f"  download_mode: {job.mode}")
        print(f"  required: {job.required}")
        print(f"  allow_parallel: {job.allow_parallel}")
        print(f"  url: {job.url or ''}")
        print(f"  local_path: {relpath(job.local_path, root)}")
        print(f"  expected_filename_patterns: {list(job.expected_patterns)}")
        print("  dry_run: no data will be downloaded and manifests will not be modified.")


def summarize(results: list[JobResult]) -> dict[str, int]:
    summary = {key: 0 for key in SUMMARY_KEYS}
    for result in results:
        if result.download_status in summary:
            summary[result.download_status] += 1
        elif result.download_status == "download_failed_manual_required":
            summary["failed"] += 1
            summary["requires_manual_download"] += 1
        elif result.download_status == "planned":
            summary.setdefault("planned", 0)
            summary["planned"] += 1
        else:
            summary["failed"] += 1
    return summary


def should_fail(results: list[JobResult], dry_run: bool) -> bool:
    if dry_run:
        return False
    bad_statuses = {"missing", "failed", "checksum_failed", "download_failed_manual_required", "requires_manual_download"}
    return any(result.is_required and result.download_status in bad_statuses for result in results)


def load_default_downloads_config(argv: list[str]) -> dict[str, Any]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--downloads-config", type=Path, default=DOWNLOADS_CONFIG)
    known, _ = pre_parser.parse_known_args(argv)
    return load_yaml(known.downloads_config)


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-config", type=Path, default=DATASETS_CONFIG)
    parser.add_argument("--downloads-config", type=Path, default=DOWNLOADS_CONFIG)
    parser.add_argument("--list", action="store_true", help="List configured datasets and sources.")
    parser.add_argument("--dataset", default="all", help="Dataset name to fetch/register, or 'all'.")
    parser.add_argument("--source", help="Restrict processing to one source name.")
    parser.add_argument("--workers", type=int, default=int(defaults.get("workers", 4)))
    parser.add_argument("--retries", type=int, default=int(defaults.get("retries", 3)))
    parser.add_argument("--timeout", type=int, default=int(defaults.get("timeout_seconds", 60)))
    parser.add_argument("--force", action="store_true", default=bool(defaults.get("force", False)))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path(defaults.get("log_dir", "logs/data_fetching")))
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path(defaults.get("manifest_csv", "data/manifests/source_manifest.csv")),
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path(defaults.get("manifest_json", "data/manifests/source_manifest.json")),
    )
    return parser.parse_args()


def main() -> int:
    downloads_config = load_default_downloads_config(sys.argv[1:])
    defaults = downloads_config.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    args = parse_args(defaults)

    datasets_config = load_yaml(args.datasets_config)
    downloads_config = load_yaml(args.downloads_config)
    root = project_root_from(args.datasets_config)

    if args.list:
        list_datasets(datasets_config)
        return 0

    options = FetchOptions(
        workers=max(1, args.workers),
        retries=max(1, args.retries),
        timeout=max(1, args.timeout),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        no_progress_bar=bool(args.no_progress_bar),
        log_dir=args.log_dir,
        manifest_csv=args.manifest_csv,
        manifest_json=args.manifest_json,
    )
    logger, run_log = setup_logging(root, options.log_dir)
    logger.info("argv=%s", sys.argv)
    logger.info("datasets_config_path=%s downloads_config_path=%s options=%s", args.datasets_config, args.downloads_config, options)
    logger.info("loaded_datasets_config=%s", datasets_config)
    logger.info("loaded_downloads_config=%s", downloads_config)

    jobs = build_jobs(root, datasets_config, args.dataset, args.source)
    if not jobs:
        raise SystemExit("No matching dataset/source jobs.")
    output_counts: dict[Path, int] = {}
    for job in jobs:
        if job.mode in {"direct_download", "remote_or_manual"} and job.url:
            output_counts[job.local_path] = output_counts.get(job.local_path, 0) + 1
    duplicate_outputs = [path for path, count in output_counts.items() if count > 1]
    if duplicate_outputs:
        for path in duplicate_outputs:
            print(f"ERROR: multiple download jobs target the same output path: {relpath(path, root)}")
        return 2
    logger.info("planned_jobs=%s", [job.__dict__ for job in jobs])

    if options.dry_run:
        print_plan(jobs, root)
        print(f"\nDry-run log: {relpath(run_log, root)}")
        return 0

    for job in jobs:
        job.source_dir.mkdir(parents=True, exist_ok=True)

    results: list[JobResult] = []
    with ThreadPoolExecutor(max_workers=options.workers) as executor:
        futures = {executor.submit(process_job, job, root, options, logger): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                job_results = future.result()
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.exception("Unhandled job error: %s/%s", job.dataset_name, job.source_name)
                job_results = [result_for(job, root, "failed", options, error=str(exc))]
            results.extend(job_results)
            for result in job_results:
                logger.info("job_result=%s", asdict(result))
                if options.no_progress_bar:
                    print(f"[{index}/{len(jobs)}] {result.dataset_name}/{result.source_name}: {result.download_status}")
                else:
                    print(f"{result.dataset_name}/{result.source_name}/{result.file_label}: {result.download_status}")

    write_manifests(root, options, results)
    summary = summarize(results)
    logger.info("summary=%s", summary)
    print("\nSummary:")
    for key in SUMMARY_KEYS:
        print(f"  {key}: {summary.get(key, 0)}")
    print(f"Log: {relpath(run_log, root)}")
    print(f"Manifest CSV: {options.manifest_csv}")
    print(f"Manifest JSON: {options.manifest_json}")

    return 1 if should_fail(results, options.dry_run) else 0


if __name__ == "__main__":
    raise SystemExit(main())
