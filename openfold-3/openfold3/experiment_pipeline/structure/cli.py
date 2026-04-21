from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .backends import FoldXBackend, OpenFold3Backend
from .models import CaseManifest, MutationCase
from .runner import StructureRunner


SEQUENCE_COLUMN = "sequence"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run structure-stage input preparation and backend orchestration.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to processed dataset CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for prepared cases.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare dry-run case inputs only.")
    parser.add_argument(
        "--backend",
        choices=("openfold3", "foldx", "both"),
        default="both",
        help="Backend selection for non-dry-run execution.",
    )
    parser.add_argument(
        "--foldx-binary",
        type=Path,
        default=Path(os.environ.get("FOLDX_BINARY", "foldx")),
        help="Path to FoldX binary.",
    )
    parser.add_argument(
        "--openfold-python",
        default=os.environ.get("PYTHON", os.sys.executable),
        help="Python executable used for `python -m openfold3.run_openfold`.",
    )
    parser.add_argument(
        "--openfold-runner-yaml",
        type=Path,
        default=None,
        help="Optional runner yaml passed to `run_openfold predict`.",
    )
    parser.add_argument(
        "--openfold-inference-ckpt-path",
        type=Path,
        default=None,
        help="Optional inference checkpoint path passed to `run_openfold predict`.",
    )
    parser.add_argument(
        "--strict-config-hash",
        action="store_true",
        help="Hash checkpoint file contents instead of metadata only when building config hash.",
    )
    return parser


def load_dataset_frame(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def load_cases(dataset_path: Path) -> list[MutationCase]:
    frame = load_dataset_frame(dataset_path)
    return [MutationCase.from_row(record) for record in frame.to_dict(orient="records")]


def load_case_sequences(dataset_path: Path) -> dict[str, str]:
    frame = load_dataset_frame(dataset_path)
    if SEQUENCE_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must contain '{SEQUENCE_COLUMN}' column for structure stage")

    sequences_by_case_id: dict[str, str] = {}
    for record in frame.to_dict(orient="records"):
        case = MutationCase.from_row(record)
        sequence = str(record[SEQUENCE_COLUMN]).strip().upper()
        if not sequence:
            raise ValueError(f"Empty sequence for case {case.case_id}")
        sequences_by_case_id[case.case_id] = sequence
    return sequences_by_case_id


def _normalized_backend_list(backend: str) -> list[str]:
    if backend == "both":
        return ["foldx", "openfold3"]
    return [backend]


def _normalize_path_string(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    resolved = shutil.which(str(path))
    if resolved is not None:
        return str(Path(resolved).resolve())
    return str(path)


def build_file_fingerprint(path: str | Path | None, *, hash_contents: bool) -> dict[str, Any]:
    normalized_path = _normalize_path_string(path)
    if normalized_path is None:
        return {
            "path": None,
            "exists": False,
            "size": None,
            "mtime_ns": None,
            "sha256": None,
        }

    candidate = Path(normalized_path)
    if not candidate.exists():
        return {
            "path": normalized_path,
            "exists": False,
            "size": None,
            "mtime_ns": None,
            "sha256": None,
        }

    stat_result = candidate.stat()
    sha256 = None
    if hash_contents:
        sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return {
        "path": str(candidate.resolve()),
        "exists": True,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "sha256": sha256,
    }


def build_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    runtime_config = {
        "backend": args.backend,
        "backend_list": _normalized_backend_list(args.backend),
        "dry_run": bool(args.dry_run),
        "strict_config_hash": bool(args.strict_config_hash),
        "foldx_binary": build_file_fingerprint(args.foldx_binary, hash_contents=True),
        "openfold_python": _normalize_path_string(args.openfold_python),
        "openfold_runner_yaml": build_file_fingerprint(
            args.openfold_runner_yaml,
            hash_contents=True,
        ),
        "openfold_inference_ckpt_path": build_file_fingerprint(
            args.openfold_inference_ckpt_path,
            hash_contents=bool(args.strict_config_hash),
        ),
    }
    return runtime_config


def hash_config(config_dict: dict[str, Any]) -> str:
    payload = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_backends(args: argparse.Namespace) -> tuple[object, ...]:
    if args.backend == "openfold3":
        return (
            OpenFold3Backend(
                python_executable=args.openfold_python,
                runner_yaml=args.openfold_runner_yaml,
                inference_ckpt_path=args.openfold_inference_ckpt_path,
            ),
        )
    if args.backend == "foldx":
        return (FoldXBackend(binary_path=args.foldx_binary),)
    return (
        OpenFold3Backend(
            python_executable=args.openfold_python,
            runner_yaml=args.openfold_runner_yaml,
            inference_ckpt_path=args.openfold_inference_ckpt_path,
        ),
        FoldXBackend(binary_path=args.foldx_binary),
    )


def summarize_manifests(manifests: list[CaseManifest]) -> dict[str, dict[str, int]]:
    case_counts = {
        "total_cases": len(manifests),
        "cases_all_success": 0,
        "cases_any_failed": 0,
        "cases_with_unavailable": 0,
        "cases_fully_cached": 0,
        "cases_partially_completed": 0,
    }
    backend_counts = {
        "total_backend_runs": 0,
        "backend_success": 0,
        "backend_failed": 0,
        "backend_cached": 0,
        "backend_unavailable": 0,
    }

    for manifest in manifests:
        statuses = [result.status for result in manifest.backend_results]
        if not statuses:
            continue

        backend_counts["total_backend_runs"] += len(statuses)
        backend_counts["backend_success"] += sum(status == "ok" for status in statuses)
        backend_counts["backend_failed"] += sum(status == "failed" for status in statuses)
        backend_counts["backend_cached"] += sum(status == "cached" for status in statuses)
        backend_counts["backend_unavailable"] += sum(status == "unavailable" for status in statuses)

        if any(status == "failed" for status in statuses):
            case_counts["cases_any_failed"] += 1
        elif any(status == "unavailable" for status in statuses):
            case_counts["cases_with_unavailable"] += 1
        elif all(status in {"ok", "cached"} for status in statuses):
            case_counts["cases_all_success"] += 1
            if all(status == "cached" for status in statuses):
                case_counts["cases_fully_cached"] += 1
        else:
            case_counts["cases_partially_completed"] += 1

    return {"case": case_counts, "backend": backend_counts}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    sequences_by_case_id = load_case_sequences(args.dataset)
    config_dict = build_config_dict(args)
    config_hash = hash_config(config_dict)
    print(f"[CONFIG HASH] {config_hash} (strict={bool(args.strict_config_hash)})")
    runner = StructureRunner(
        output_root=args.output_dir.resolve(),
        dry_run=args.dry_run,
        config_hash=config_hash,
        backends=() if args.dry_run else build_backends(args),
    )
    manifests = runner.run_all(cases, sequences_by_case_id)
    summary = summarize_manifests(manifests)
    print(f"Processed {len(manifests)} cases")
    print("=== CASE SUMMARY ===")
    print(f"Total cases: {summary['case']['total_cases']}")
    print(f"All success: {summary['case']['cases_all_success']}")
    print(f"Any failed: {summary['case']['cases_any_failed']}")
    print(f"With unavailable: {summary['case']['cases_with_unavailable']}")
    print(f"Fully cached: {summary['case']['cases_fully_cached']}")
    print(f"Partially completed: {summary['case']['cases_partially_completed']}")
    print("=== BACKEND SUMMARY ===")
    print(f"Total runs: {summary['backend']['total_backend_runs']}")
    print(f"Success: {summary['backend']['backend_success']}")
    print(f"Failed: {summary['backend']['backend_failed']}")
    print(f"Cached: {summary['backend']['backend_cached']}")
    print(f"Unavailable: {summary['backend']['backend_unavailable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
