from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openfold3.benchmark.foldx_panel import run_foldx_panel
from openfold3.benchmark.local_edit import run_local_mutation_case
from openfold3.benchmark.models import MutationInput


KNOWN_FOLDX_FILENAMES = (
    "foldx",
    "foldx5",
    "foldx_20270131",
    "foldx_20241231",
)


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Payload must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tail(value: str | None, *, limit: int = 2000) -> str:
    if not value:
        return ""
    return value[-limit:]


def _candidate_foldx_paths() -> list[Path]:
    candidates: list[Path] = []
    env_value = os.environ.get("FOLDX_BINARY")
    if env_value:
        resolved = shutil.which(env_value)
        candidates.append(Path(resolved or env_value).expanduser())
    for directory in (Path("/foldx"), Path("/tools/bin"), Path("/work/tools/bin")):
        for filename in KNOWN_FOLDX_FILENAMES:
            candidates.append(directory / filename)
    path_resolved = shutil.which("foldx")
    if path_resolved:
        candidates.append(Path(path_resolved))
    return candidates


def resolve_foldx_binary(*, required: bool = True) -> Path | None:
    for candidate in _candidate_foldx_paths():
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    if required:
        searched = ", ".join(str(path) for path in _candidate_foldx_paths())
        raise FileNotFoundError(
            "FoldX binary was not found or is not executable. "
            "Set FOLDX_BINARY or mount it at /foldx/foldx. "
            f"Searched: {searched}"
        )
    return None


def probe_foldx_binary(
    binary: Path,
    *,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Start FoldX once to catch wrong-OS binaries and missing shared libs."""
    command = [str(binary), "--help"]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "command": command,
            "returncode": None,
            "stdout_tail": _tail(exc.stdout if isinstance(exc.stdout, str) else None),
            "stderr_tail": _tail(exc.stderr if isinstance(exc.stderr, str) else None),
            "error": f"FoldX probe timed out after {timeout_seconds:g}s",
        }
    except OSError as exc:
        return {
            "status": "failed",
            "command": command,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    combined_output = f"{process.stdout}\n{process.stderr}".lower()
    looks_like_foldx = "foldx" in combined_output or "buildmodel" in combined_output
    if looks_like_foldx:
        status = "ok"
        error = None
    else:
        status = "failed"
        error = (
            "FoldX probe did not look like a working FoldX executable. "
            "Use the Linux FoldX binary inside this Linux container."
        )
    return {
        "status": status,
        "command": command,
        "returncode": process.returncode,
        "stdout_tail": _tail(process.stdout),
        "stderr_tail": _tail(process.stderr),
        "error": error,
    }


def _require_working_foldx(binary: Path, *, timeout_seconds: float) -> dict[str, Any]:
    probe = probe_foldx_binary(binary, timeout_seconds=timeout_seconds)
    if probe["status"] != "ok":
        raise RuntimeError(
            "FoldX binary probe failed. "
            f"binary={binary}; error={probe['error']}; "
            f"stdout={probe['stdout_tail']!r}; stderr={probe['stderr_tail']!r}"
        )
    return probe


def _mutation_from_payload(payload: dict[str, Any]) -> MutationInput:
    return MutationInput(
        chain_id=str(payload["chain_id"]),
        from_residue=str(payload["from_residue"]),
        position_1based=int(payload["position_1based"]),
        to_residue=str(payload["to_residue"]),
    )


def _input_source(payload: dict[str, Any]) -> dict[str, Any]:
    structure_path = payload.get("structure_path")
    pdb_id = payload.get("pdb_id")
    if (structure_path is None) == (pdb_id is None):
        raise ValueError("Payload must provide exactly one of structure_path or pdb_id")
    return {
        "structure_path": None if structure_path is None else str(structure_path),
        "pdb_id": None if pdb_id is None else str(pdb_id),
    }


def _aggregate_status(total: int, successful: int) -> str:
    if total <= 0 or successful <= 0:
        return "failed"
    if successful == total:
        return "completed"
    return "partial_failed"


def _row_from_result(index: int, result: Any) -> dict[str, Any]:
    return {
        "index": index,
        "case_id": result.case_id,
        "mutation_id": result.mutation.mutation_id,
        "local_edit_status": result.local_edit_status,
        "failure_reason": result.failure_reason,
        "runtime_seconds": result.runtime_seconds,
        "mutant_structure_path": (
            None
            if result.mutant_structure_path is None
            else str(result.mutant_structure_path)
        ),
        "report_path": str(result.report_path),
    }


def _failed_mutation_row(
    *,
    index: int,
    mutation_payload: Any,
    mutation: MutationInput | None,
    case_id: str | None,
    exc: BaseException,
) -> dict[str, Any]:
    payload_case_id = (
        mutation_payload.get("case_id") if isinstance(mutation_payload, dict) else None
    )
    mutation_id = mutation.mutation_id if mutation is not None else f"invalid_mutation_{index}"
    return {
        "index": index,
        "case_id": case_id or str(payload_case_id or mutation_id),
        "mutation_id": mutation_id,
        "local_edit_status": "failed",
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "runtime_seconds": None,
        "mutant_structure_path": None,
        "report_path": None,
    }


def preflight(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    import openfold3.benchmark.foldx_panel  # noqa: F401
    import openfold3.benchmark.local_edit  # noqa: F401

    try:
        foldx_binary = resolve_foldx_binary(required=not args.allow_missing_foldx)
    except Exception as exc:
        payload = {
            "status": "failed",
            "foldx_binary": None,
            "foldx_probe": None,
            "output_dir": str(output_dir.resolve()),
            "cache_dir": str(cache_dir.resolve()),
            "python": sys.executable,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(payload, indent=2))
        return 1

    foldx_probe = None
    status = "ok"
    error = None
    if foldx_binary is not None:
        foldx_probe = probe_foldx_binary(
            foldx_binary,
            timeout_seconds=args.probe_timeout_seconds,
        )
        if foldx_probe["status"] != "ok":
            status = "failed"
            error = str(foldx_probe["error"])
    payload = {
        "status": status,
        "foldx_binary": None if foldx_binary is None else str(foldx_binary),
        "foldx_probe": foldx_probe,
        "output_dir": str(output_dir.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "python": sys.executable,
        "error": error,
    }
    print(json.dumps(payload, indent=2))
    return 0 if status == "ok" else 1


def run_payload(args: argparse.Namespace) -> int:
    payload = _read_json(args.payload)
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        foldx_binary = resolve_foldx_binary(required=True)
        foldx_probe = _require_working_foldx(
            foldx_binary,
            timeout_seconds=args.probe_timeout_seconds,
        )
        os.environ["FOLDX_BINARY"] = str(foldx_binary)
    except Exception as exc:
        manifest = {
            "status": "failed",
            "mode": str(payload.get("mode") or "explicit_mutations"),
            "output_root": str(output_dir),
            "error": f"{type(exc).__name__}: {exc}",
            "total_mutations": 0,
            "successful_mutations": 0,
        }
        _write_json(output_dir / "foldx_worker_manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return 1

    mode = str(payload.get("mode") or "explicit_mutations")
    try:
        source = _input_source(payload)
    except Exception as exc:
        manifest = {
            "status": "failed",
            "mode": mode,
            "output_root": str(output_dir),
            "error": f"{type(exc).__name__}: {exc}",
            "total_mutations": 0,
            "successful_mutations": 0,
            "failed_mutations": 0,
            "foldx_binary": str(foldx_binary),
            "foldx_probe": foldx_probe,
        }
        _write_json(output_dir / "foldx_worker_manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return 1
    if mode == "panel":
        try:
            positions = tuple(int(value) for value in payload["positions"])
            result = run_foldx_panel(
                output_root=output_dir,
                chain_id=str(payload["chain_id"]),
                positions=positions,
                structure_path=source["structure_path"],
                pdb_id=source["pdb_id"],
                cache_dir=cache_dir,
                show_progress=bool(payload.get("show_progress", True)),
                num_workers=int(payload.get("num_workers", 1)),
                ranking_metric=str(payload.get("ranking_metric", "stability_ddg")),
            )
        except Exception as exc:
            summary = {
                "status": "failed",
                "mode": mode,
                "output_root": str(output_dir),
                "error": f"{type(exc).__name__}: {exc}",
                "total_mutations": 0,
                "successful_mutations": 0,
                "foldx_binary": str(foldx_binary),
                "foldx_probe": foldx_probe,
            }
            _write_json(output_dir / "foldx_worker_manifest.json", summary)
            print(json.dumps(summary, indent=2))
            return 1

        total_mutations = len(result.rows)
        successful_mutations = sum(row.local_edit_status == "ok" for row in result.rows)
        summary = {
            "status": _aggregate_status(total_mutations, successful_mutations),
            "mode": mode,
            "output_root": str(result.output_root),
            "summary_json_path": str(result.summary_json_path),
            "rows_csv_path": str(result.rows_csv_path),
            "ranking_csv_path": str(result.ranking_csv_path),
            "total_mutations": total_mutations,
            "successful_mutations": successful_mutations,
            "failed_mutations": total_mutations - successful_mutations,
            "foldx_binary": str(foldx_binary),
            "foldx_probe": foldx_probe,
        }
        _write_json(output_dir / "foldx_worker_manifest.json", summary)
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] == "completed" else 1

    if mode != "explicit_mutations":
        manifest = {
            "status": "failed",
            "mode": mode,
            "output_root": str(output_dir),
            "error": f"Unsupported mode: {mode}",
            "total_mutations": 0,
            "successful_mutations": 0,
            "failed_mutations": 0,
            "foldx_binary": str(foldx_binary),
            "foldx_probe": foldx_probe,
        }
        _write_json(output_dir / "foldx_worker_manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return 1

    cases_root = output_dir / "cases"
    rows: list[dict[str, Any]] = []
    mutation_payloads = payload.get("mutations") or []
    for index, mutation_payload in enumerate(mutation_payloads, start=1):
        mutation = None
        case_id = None
        try:
            mutation = _mutation_from_payload(mutation_payload)
            case_id = str(
                mutation_payload.get("case_id")
                or f"{payload.get('case_id', 'foldx_case')}_{mutation.mutation_id}"
            )
            result = run_local_mutation_case(
                mutation=mutation,
                work_dir=cases_root,
                structure_path=source["structure_path"],
                pdb_id=source["pdb_id"],
                case_id=case_id,
                cache_dir=cache_dir,
            )
            rows.append(_row_from_result(index, result))
        except Exception as exc:
            rows.append(
                _failed_mutation_row(
                    index=index,
                    mutation_payload=mutation_payload,
                    mutation=mutation,
                    case_id=case_id,
                    exc=exc,
                )
            )

    if not rows:
        manifest = {
            "status": "failed",
            "mode": mode,
            "output_root": str(output_dir),
            "error": "explicit_mutations payload requires a non-empty mutations list",
            "rows_csv_path": str(output_dir / "rows.csv"),
            "total_mutations": 0,
            "successful_mutations": 0,
            "failed_mutations": 0,
            "rows": rows,
            "foldx_binary": str(foldx_binary),
            "foldx_probe": foldx_probe,
        }
        _write_csv(output_dir / "rows.csv", rows)
        _write_json(output_dir / "foldx_worker_manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return 1

    _write_csv(output_dir / "rows.csv", rows)
    total_mutations = len(rows)
    successful_mutations = sum(row["local_edit_status"] == "ok" for row in rows)
    manifest = {
        "status": _aggregate_status(total_mutations, successful_mutations),
        "mode": mode,
        "output_root": str(output_dir),
        "rows_csv_path": str(output_dir / "rows.csv"),
        "total_mutations": total_mutations,
        "successful_mutations": successful_mutations,
        "failed_mutations": total_mutations - successful_mutations,
        "rows": rows,
        "foldx_binary": str(foldx_binary),
        "foldx_probe": foldx_probe,
    }
    _write_json(output_dir / "foldx_worker_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FoldX mutation jobs in Docker.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--output-dir",
        default=os.environ.get("FOLDX_OUTPUT_DIR", "/results"),
    )
    preflight_parser.add_argument(
        "--cache-dir",
        default=os.environ.get("FOLDX_CACHE_DIR", "/cache"),
    )
    preflight_parser.add_argument("--allow-missing-foldx", action="store_true")
    preflight_parser.add_argument(
        "--probe-timeout-seconds",
        type=float,
        default=15.0,
    )
    preflight_parser.set_defaults(func=preflight)

    run_parser = subparsers.add_parser("run-payload")
    run_parser.add_argument("--payload", type=Path, required=True)
    run_parser.add_argument(
        "--output-dir",
        default=os.environ.get("FOLDX_OUTPUT_DIR", "/results"),
    )
    run_parser.add_argument(
        "--cache-dir",
        default=os.environ.get("FOLDX_CACHE_DIR", "/cache"),
    )
    run_parser.add_argument(
        "--probe-timeout-seconds",
        type=float,
        default=15.0,
    )
    run_parser.set_defaults(func=run_payload)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
