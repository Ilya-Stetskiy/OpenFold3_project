from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from openfold3.benchmark.harness import HarnessContext
from openfold3.benchmark.cif_utils import parse_structure_records, write_pdb_atom_records
from openfold3.benchmark.methods import (
    FoldXBuildModelMethod,
    _prepare_local_pdb_copy,
    _resolve_executable_path,
)
from openfold3.benchmark.models import AtomRecord, BenchmarkCase, MutationInput

from ..data.canonical import CanonicalMutationRecord
from ..data.mutation_engine import apply_mutation
from ..structures.loader import extract_mutation_site, load_structure
from .base import BackendRunResult, DDGAdapter


def normalize_ddg(method: str, raw_value: float | None) -> float | None:
    if method.lower() not in {"foldx", "rosetta", "esm2"}:
        raise ValueError(f"Unsupported ddG normalization method: {method}")
    return None if raw_value is None else float(raw_value)


def _error_from_details(details: dict[str, Any]) -> str:
    reason = details.get("reason")
    error = details.get("error")
    if error is not None:
        return str(error)
    if reason is not None:
        return str(reason)
    stdout = details.get("stdout")
    stderr = details.get("stderr")
    if stdout or stderr:
        return f"stdout={stdout!s} stderr={stderr!s}".strip()
    return "FoldX backend failed"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_execution_trace(
    *,
    original_structure_path: Path,
    repair_input_path: Path,
    repaired_output_path: Path,
    buildmodel_input_path: Path,
    raw_output_path: Path,
    method_work_dir: Path | None,
    buildmodel_started: bool,
) -> str:
    return "\n".join(
        [
            "# FoldX Execution Trace",
            "",
            "## Code Path",
            "",
            "- ddg adapter: `openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.FoldXAdapter.run`",
            "- benchmark method: `openfold3.benchmark.methods.FoldXBuildModelMethod.run`",
            "",
            "## Runtime Paths",
            "",
            f"- original_structure_path: `{original_structure_path}`",
            f"- repair_input_path: `{repair_input_path}`",
            f"- repaired_output_path: `{repaired_output_path}`",
            f"- buildmodel_input_path: `{buildmodel_input_path}`",
            f"- benchmark_work_dir: `{method_work_dir}`",
            f"- raw_output_path: `{raw_output_path}`",
            f"- buildmodel_started: `{buildmodel_started}`",
            "",
            "## Equality Check",
            "",
            f"- buildmodel_input_path == repaired_output_path: `{buildmodel_input_path == repaired_output_path}`",
            "",
        ]
    )


def _discover_repaired_structure(repair_dir: Path, input_pdb_stem: str) -> Path:
    exact_candidates = [
        repair_dir / f"{input_pdb_stem}_Repair.pdb",
        repair_dir / "out" / f"{input_pdb_stem}_Repair.pdb",
    ]
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted({path.resolve() for path in repair_dir.rglob(f"{input_pdb_stem}_Repair.pdb") if path.is_file()})
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not locate repaired structure for {input_pdb_stem} under {repair_dir}")
    raise RuntimeError(
        f"Ambiguous repaired structure candidates for {input_pdb_stem} under {repair_dir}: "
        + ", ".join(str(path) for path in matches)
    )


def _repo_roots(start: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for candidate in (start.resolve(), *start.resolve().parents):
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def _workspace_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if candidate.name == "OpenFold_codex":
            return candidate
    return resolved.parent


def _to_windows_path(path: Path) -> str:
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith("/mnt/") and len(resolved.parts) >= 4:
        drive_letter = resolved.parts[2].upper()
        tail = resolved.parts[3:]
        return drive_letter + ":\\" + "\\".join(tail)
    return text


@lru_cache(maxsize=1)
def _candidate_rosetta_bundle_mains() -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in _repo_roots(Path(__file__)):
        tools_dir = root / "tools"
        if not tools_dir.exists():
            continue
        for candidate in tools_dir.glob("**/rosetta*_bundle/main"):
            resolved = candidate.resolve()
            if resolved.is_dir() and resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
    return tuple(candidates)


def _resolve_binary_override(env_var_name: str) -> str | None:
    override = os.environ.get(env_var_name)
    if not override:
        return None
    resolved = shutil.which(override) or (override if Path(override).exists() else None)
    if resolved is None:
        return None
    return str(Path(resolved).resolve())


@lru_cache(maxsize=1)
def _resolve_rosetta_ddg_binary() -> str | None:
    override = _resolve_binary_override("ROSETTA_DDG_MONOMER_BINARY")
    if override is not None:
        return override
    direct = _resolve_executable_path("ddg_monomer.static.linuxgccrelease", env_var_name="ROSETTA_DDG_MONOMER_BINARY")
    if direct is not None:
        return direct
    for bundle_main in _candidate_rosetta_bundle_mains():
        for relative in (
            "source/bin/ddg_monomer.static.linuxgccrelease",
            "source/bin/ddg_monomer.linuxgccrelease",
        ):
            candidate = bundle_main / relative
            if candidate.is_file():
                return str(candidate.resolve())
    return None


@lru_cache(maxsize=1)
def _resolve_rosetta_relax_binary() -> str | None:
    override = _resolve_binary_override("ROSETTA_RELAX_BINARY")
    if override is not None:
        return override
    direct = _resolve_executable_path("relax.static.linuxgccrelease", env_var_name="ROSETTA_RELAX_BINARY")
    if direct is not None:
        return direct
    for bundle_main in _candidate_rosetta_bundle_mains():
        for relative in (
            "source/bin/relax.static.linuxgccrelease",
            "source/bin/relax.linuxgccrelease",
        ):
            candidate = bundle_main / relative
            if candidate.is_file():
                return str(candidate.resolve())
    return None


@lru_cache(maxsize=1)
def _resolve_rosetta_database_path() -> str | None:
    override = os.environ.get("ROSETTA_DATABASE")
    if override and Path(override).exists():
        return str(Path(override).resolve())
    for binary_path in (_resolve_rosetta_ddg_binary(), _resolve_rosetta_relax_binary()):
        if binary_path is None:
            continue
        resolved_binary = Path(binary_path).resolve()
        for parent in resolved_binary.parents:
            candidate = parent / "database"
            if candidate.is_dir():
                return str(candidate.resolve())
    for bundle_main in _candidate_rosetta_bundle_mains():
        candidate = bundle_main / "database"
        if candidate.is_dir():
            return str(candidate.resolve())
    return None


def _collect_rosetta_environment_check() -> dict[str, Any]:
    ddg_binary = _resolve_rosetta_ddg_binary()
    relax_binary = _resolve_rosetta_relax_binary()
    database_path = _resolve_rosetta_database_path()
    missing: list[str] = []
    if ddg_binary is None:
        missing.append("ddg_monomer.static.linuxgccrelease")
    if relax_binary is None:
        missing.append("relax.static.linuxgccrelease")
    if database_path is None:
        missing.append("rosetta_database")
    return {
        "ddg_monomer_binary": ddg_binary,
        "relax_binary": relax_binary,
        "database_path": database_path,
        "missing_components": missing,
    }


@lru_cache(maxsize=1)
def _resolve_esm2_python() -> str | None:
    override = _resolve_binary_override("ESM2_PYTHON")
    if override is not None:
        return override
    for candidate in (
        Path("/mnt/d/Anaconda3/NewAnaconda/python.exe"),
        Path("/mnt/d/Anaconda3/python.exe"),
        Path("/mnt/d/Program files/Anacondas/python.exe"),
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _esm2_torch_home_windows() -> str:
    override = os.environ.get("TORCH_HOME")
    if override:
        return override
    repo_root = _workspace_root(Path(__file__))
    return _to_windows_path(repo_root / ".torch-cache")


def _esm2_device() -> str:
    return os.environ.get("ESM2_DEVICE", "auto")


def _esm2_inference_script_windows() -> str:
    return _to_windows_path(Path(__file__).with_name("esm2_inference.py"))


def _collect_esm2_environment_check() -> dict[str, Any]:
    python_executable = _resolve_esm2_python()
    missing: list[str] = []
    if python_executable is None:
        missing.append("esm2_python")
        return {
            "python_executable": None,
            "torch_home": _esm2_torch_home_windows(),
            "esm2_device": _esm2_device(),
            "missing_components": missing,
        }
    output_path = _workspace_root(Path(__file__)) / ".torch-cache" / "esm2_environment_check_runtime.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        _esm2_inference_script_windows(),
        "--mode",
        "check",
        "--output",
        _to_windows_path(output_path),
        "--torch-home",
        _esm2_torch_home_windows(),
        "--device",
        _esm2_device(),
    ]
    env = dict(os.environ)
    env["TORCH_HOME"] = _esm2_torch_home_windows()
    process = subprocess.run(command, capture_output=True, text=True, env=env)
    payload: dict[str, Any] = {
        "python_executable": python_executable,
        "torch_home": env["TORCH_HOME"],
        "esm2_device": _esm2_device(),
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "output_path": str(output_path),
        "missing_components": [],
    }
    if process.returncode != 0 or not output_path.exists():
        payload["missing_components"] = ["esm2_runtime"]
        return payload
    runtime_payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload.update(runtime_payload)
    if runtime_payload.get("torch_error") is not None:
        payload["missing_components"].append("torch")
    if runtime_payload.get("esm_import") is not True:
        payload["missing_components"].append("fair-esm")
    return payload


def _write_rosetta_protocol(path: Path, n_runs: int, top_k: int) -> None:
    _write_text(
        path,
        "\n".join(
            [
                "# Rosetta ddg_monomer Protocol",
                "",
                "1. Prepare input PDB and Rosetta mutfile.",
                "2. Renumber the Rosetta input consecutively before writing the mutfile.",
                "3. Relax the input structure once with relax.static.linuxgccrelease.",
                f"4. Launch {n_runs} independent ddg_monomer processes with -ddg::iterations 1.",
                "5. Parse ddg_predictions.out and the per-run wild-type / mutant energies.",
                f"6. Aggregate as mean(top-{top_k} mutant energies) - mean(top-{top_k} wild-type energies).",
                "",
                "Command template:",
                "relax.static.linuxgccrelease -database <db> -in:file:s input.pdb -out:path:all relax_out -nstruct 1",
                "ddg_monomer.static.linuxgccrelease -database <db> -in:file:s relaxed.pdb -ddg::mut_file mutations.txt -ddg::iterations 1",
                "",
                "Units: Rosetta Energy Units (REU).",
                "Convention: ddG = mutant energy - wildtype energy.",
            ]
        ),
    )


def _parse_rosetta_ddg_predictions(path: Path, mutation: str) -> float:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or not line.startswith("ddG:"):
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[1] == mutation:
            value = float(parts[2])
            if math.isnan(value):
                raise ValueError(f"NaN ddG in {path}")
            return value
    raise ValueError(f"Could not parse ddG prediction for {mutation} from {path}")


def _parse_rosetta_stdout_energies(stdout: str, mutation: str) -> tuple[float, float]:
    import re

    pattern = re.compile(
        rf"mutate {re.escape(mutation)}\s+wildtype_dG is:\s*([-+0-9.eE]+)\s+and mutant_dG is:\s*([-+0-9.eE]+)\s+ddG is:\s*([-+0-9.eE]+)"
    )
    matches = pattern.findall(stdout)
    if not matches:
        raise ValueError(f"Could not parse Rosetta stdout energies for {mutation}")
    wildtype_text, mutant_text, _ = matches[-1]
    wildtype_value = float(wildtype_text)
    mutant_value = float(mutant_text)
    if math.isnan(wildtype_value) or math.isnan(mutant_value):
        raise ValueError("Rosetta stdout produced NaN energies")
    return wildtype_value, mutant_value


def _renumber_pdb_for_rosetta(input_pdb: Path, output_pdb: Path) -> dict[tuple[str, str], int]:
    atoms = parse_structure_records(input_pdb)
    residue_index: "OrderedDict[tuple[str, str], int]" = OrderedDict()
    renumbered_atoms: list[AtomRecord] = []
    next_index = 1
    for atom in atoms:
        key = (str(atom.chain_id), str(atom.residue_id))
        if key not in residue_index:
            residue_index[key] = next_index
            next_index += 1
        renumbered_atoms.append(
            AtomRecord(
                chain_id=atom.chain_id,
                residue_name=atom.residue_name,
                residue_id=str(residue_index[key]),
                atom_name=atom.atom_name,
                x=atom.x,
                y=atom.y,
                z=atom.z,
                b_factor=atom.b_factor,
                group_pdb=atom.group_pdb,
            )
        )
    write_pdb_atom_records(output_pdb, renumbered_atoms)
    return dict(residue_index)


def _run_rosetta_ddg_worker(task: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(task["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    command = [
        str(task["ddg_binary"]),
        "-database",
        str(task["database_path"]),
        "-in:file:s",
        str(task["relaxed_pdb"]),
        "-ddg::mut_file",
        str(task["mutation_file"]),
        "-ddg::iterations",
        "1",
        "-ddg::dump_pdbs",
        "false",
        "-ddg::suppress_checkpointing",
        "true",
        "-ignore_unrecognized_res",
        "true",
        "-in:file:fullatom",
        "-ddg::local_opt_only",
        "true",
        "-ddg::mean",
        "true",
        "-ddg::min",
        "false",
        "-out:path:all",
        str(run_dir.resolve()),
    ]
    process = subprocess.run(command, cwd=run_dir, capture_output=True, text=True)
    _write_text(stdout_path, process.stdout)
    _write_text(stderr_path, process.stderr)
    if process.returncode != 0:
        return {
            "status": "failed",
            "run_index": task["run_index"],
            "command": command,
            "returncode": process.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error": "rosetta_ddg_process_failed",
        }
    prediction_path = run_dir / "ddg_predictions.out"
    if not prediction_path.exists():
        return {
            "status": "failed",
            "run_index": task["run_index"],
            "command": command,
            "returncode": process.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error": "rosetta_ddg_output_missing",
        }
    try:
        ddg_value = _parse_rosetta_ddg_predictions(prediction_path, str(task["mutation_label"]))
        wt_energy, mut_energy = _parse_rosetta_stdout_energies(process.stdout, str(task["mutation_label"]))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "run_index": task["run_index"],
            "command": command,
            "returncode": process.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "prediction_path": str(prediction_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "ok",
        "run_index": task["run_index"],
        "command": command,
        "returncode": process.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "prediction_path": str(prediction_path),
        "ddg": ddg_value,
        "wildtype_dg": wt_energy,
        "mutant_dg": mut_energy,
    }


@dataclass(frozen=True, slots=True)
class FoldXAdapter(DDGAdapter):
    method_name: str = "foldx"
    sequence_based: bool = False
    number_of_runs: int = 5

    def validate_environment(self) -> None:
        executable_path = _resolve_executable_path("foldx", env_var_name="FOLDX_BINARY")
        if executable_path is None:
            raise RuntimeError("FoldX executable is required but was not found")

    def prepare_input(self, record: CanonicalMutationRecord, structure_source: str, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        structure_path = getattr(record.structure_paths, structure_source)
        if structure_path is None:
            raise ValueError(f"Missing structure for source {structure_source}")
        structure = load_structure(structure_path)
        site = extract_mutation_site(structure, record.position, expected_wt=record.wt)
        payload = {
            "protein_id": record.protein_id,
            "mutation": record.mutation,
            "structure_source": structure_source,
            "structure_path": str(Path(structure_path).expanduser().resolve()),
            "chain_id": site.chain_id,
            "residue_id": site.residue_id,
            "wt": record.wt,
            "mut": record.mut,
            "number_of_runs": self.number_of_runs,
        }
        path = work_dir / "input.json"
        _write_text(path, json.dumps(payload, indent=2, sort_keys=True))
        return path

    def run(
        self,
        prepared_input_path: Path,
        record: CanonicalMutationRecord,
        structure_source: str,
        work_dir: Path,
    ) -> BackendRunResult:
        payload = json.loads(prepared_input_path.read_text(encoding="utf-8"))
        executable_path = _resolve_executable_path("foldx", env_var_name="FOLDX_BINARY")
        raw_output_path = work_dir / "raw_output.json"
        backend_logs_path = work_dir / "backend.log"
        execution_trace_path = work_dir / "foldx_execution_trace.md"
        pairing_artifact_path = work_dir / "foldx_model_pairing.json"

        if executable_path is None:
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "original_structure_path": payload["structure_path"],
                        "mutation": payload["mutation"],
                        "chain_id": payload["chain_id"],
                        "residue_id": payload["residue_id"],
                        "backend_status": "failed",
                        "backend_logs_path": str(backend_logs_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(
                status="failed",
                raw_output_path=raw_output_path,
                error_message="FoldX executable is required but was not found",
            )

        residue_id = str(payload["residue_id"])
        if not residue_id.isdigit():
            raise ValueError(f"Unsupported non-numeric residue id for FoldX: {residue_id}")

        original_structure_path = Path(str(payload["structure_path"])).expanduser().resolve()
        repair_dir = work_dir / "repair"
        if repair_dir.exists():
            shutil.rmtree(repair_dir)
        repair_dir.mkdir(parents=True, exist_ok=True)
        repair_input_pdb, repaired_from_cif, repair_prepare_seconds = _prepare_local_pdb_copy(original_structure_path, repair_dir)
        repair_command = [
            executable_path,
            "--command",
            "RepairPDB",
            "--pdb",
            repair_input_pdb.name,
            "--out-pdb",
            "true",
            "--screen",
            "false",
        ]
        repair_process = subprocess.run(
            repair_command,
            cwd=repair_dir,
            capture_output=True,
            text=True,
        )
        repair_stdout_path = repair_dir / "repair_stdout.txt"
        repair_stderr_path = repair_dir / "repair_stderr.txt"
        _write_text(repair_stdout_path, repair_process.stdout)
        _write_text(repair_stderr_path, repair_process.stderr)
        repaired_output_path = repair_dir / f"{repair_input_pdb.stem}_Repair.pdb"
        repair_output_persisted_by_foldx = False

        repair_payload = {
            "command": repair_command,
            "cwd": str(repair_dir),
            "returncode": repair_process.returncode,
            "prepared_input_pdb_path": str(repair_input_pdb),
            "repaired_output_path": str(repaired_output_path),
            "repair_output_persisted_by_foldx": repair_output_persisted_by_foldx,
            "prepared_from_cif": repaired_from_cif,
            "pdb_prepare_runtime_seconds": repair_prepare_seconds,
            "stdout_path": str(repair_stdout_path),
            "stderr_path": str(repair_stderr_path),
        }
        if repair_process.returncode != 0:
            _write_text(
                execution_trace_path,
                _build_execution_trace(
                    original_structure_path=original_structure_path,
                    repair_input_path=repair_input_pdb,
                    repaired_output_path=repaired_output_path,
                    buildmodel_input_path=repaired_output_path,
                    raw_output_path=raw_output_path,
                    method_work_dir=None,
                    buildmodel_started=False,
                ),
            )
            _write_text(
                backend_logs_path,
                "\n".join(
                    [
                        "=== RepairPDB stdout ===",
                        repair_process.stdout,
                        "=== RepairPDB stderr ===",
                        repair_process.stderr,
                    ]
                ),
            )
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "original_structure_path": str(original_structure_path),
                        "repaired_structure_path": str(repaired_output_path),
                        "buildmodel_input_path": str(repaired_output_path),
                        "mutation": payload["mutation"],
                        "chain_id": payload["chain_id"],
                        "residue_id": payload["residue_id"],
                        "generated_mutant_models": [],
                        "generated_wt_models": [],
                        "model_pairing": [],
                        "per_run_ddg": [],
                        "ddg": None,
                        "ddg_std": None,
                        "n_runs_requested": self.number_of_runs,
                        "n_runs_valid": 0,
                        "backend_status": "failed",
                        "backend_logs_path": str(backend_logs_path),
                        "execution_trace_path": str(execution_trace_path),
                        "repair": repair_payload,
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(
                status="failed",
                raw_output_path=raw_output_path,
                error_message="foldx_repair_failed",
            )
        try:
            repaired_output_path = _discover_repaired_structure(repair_dir, repair_input_pdb.stem)
            repair_output_persisted_by_foldx = True
            repair_payload["repaired_output_path"] = str(repaired_output_path)
            repair_payload["repair_output_persisted_by_foldx"] = True
        except FileNotFoundError:
            _write_text(
                execution_trace_path,
                _build_execution_trace(
                    original_structure_path=original_structure_path,
                    repair_input_path=repair_input_pdb,
                    repaired_output_path=repaired_output_path,
                    buildmodel_input_path=repaired_output_path,
                    raw_output_path=raw_output_path,
                    method_work_dir=None,
                    buildmodel_started=False,
                ),
            )
            _write_text(
                backend_logs_path,
                "\n".join(
                    [
                        "=== RepairPDB stdout ===",
                        repair_process.stdout,
                        "=== RepairPDB stderr ===",
                        repair_process.stderr,
                    ]
                ),
            )
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "original_structure_path": str(original_structure_path),
                        "repaired_structure_path": str(repaired_output_path),
                        "buildmodel_input_path": str(repaired_output_path),
                        "mutation": payload["mutation"],
                        "chain_id": payload["chain_id"],
                        "residue_id": payload["residue_id"],
                        "generated_mutant_models": [],
                        "generated_wt_models": [],
                        "model_pairing": [],
                        "per_run_ddg": [],
                        "ddg": None,
                        "ddg_std": None,
                        "n_runs_requested": self.number_of_runs,
                        "n_runs_valid": 0,
                        "backend_status": "failed",
                        "backend_logs_path": str(backend_logs_path),
                        "execution_trace_path": str(execution_trace_path),
                        "repair": repair_payload,
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(
                status="failed",
                raw_output_path=raw_output_path,
                error_message="foldx_repaired_structure_missing",
            )
        except RuntimeError as exc:
            _write_text(
                execution_trace_path,
                _build_execution_trace(
                    original_structure_path=original_structure_path,
                    repair_input_path=repair_input_pdb,
                    repaired_output_path=repaired_output_path,
                    buildmodel_input_path=repaired_output_path,
                    raw_output_path=raw_output_path,
                    method_work_dir=None,
                    buildmodel_started=False,
                ),
            )
            _write_text(
                backend_logs_path,
                "\n".join(
                    [
                        "=== RepairPDB stdout ===",
                        repair_process.stdout,
                        "=== RepairPDB stderr ===",
                        repair_process.stderr,
                        "=== RepairPDB discovery error ===",
                        str(exc),
                    ]
                ),
            )
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "original_structure_path": str(original_structure_path),
                        "repaired_structure_path": str(repaired_output_path),
                        "buildmodel_input_path": str(repaired_output_path),
                        "mutation": payload["mutation"],
                        "chain_id": payload["chain_id"],
                        "residue_id": payload["residue_id"],
                        "generated_mutant_models": [],
                        "generated_wt_models": [],
                        "model_pairing": [],
                        "per_run_ddg": [],
                        "ddg": None,
                        "ddg_std": None,
                        "n_runs_requested": self.number_of_runs,
                        "n_runs_valid": 0,
                        "backend_status": "failed",
                        "backend_logs_path": str(backend_logs_path),
                        "execution_trace_path": str(execution_trace_path),
                        "repair": repair_payload,
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(
                status="failed",
                raw_output_path=raw_output_path,
                error_message="foldx_repaired_structure_ambiguous",
            )

        mutation = MutationInput(
            chain_id=str(payload["chain_id"]),
            from_residue=record.wt,
            position_1based=int(residue_id),
            to_residue=record.mut,
        )
        buildmodel_case = BenchmarkCase(
            case_id=f"{record.protein_id}_{structure_source}_{record.mutation}",
            structure_path=repaired_output_path,
            mutations=(mutation,),
            experimental_ddg=record.experimental_ddg,
        )
        context = HarnessContext(case=buildmodel_case, structure_summary=None, confidence_payload=None)  # type: ignore[arg-type]
        method_result = FoldXBuildModelMethod(number_of_runs=self.number_of_runs).run(context)
        method_details = method_result.details
        buildmodel_input_path = Path(str(method_details.get("prepared_input_pdb_path", repaired_output_path))).resolve()
        per_run_ddg = [float(value) for value in method_details.get("per_run_buildmodel_ddg", [])]
        n_runs_valid = len(per_run_ddg)
        ddg = None if not per_run_ddg else _mean(per_run_ddg)
        ddg_std = None if not per_run_ddg else _std(per_run_ddg)
        model_pairing = list(method_details.get("model_pairing", []))
        generated_mutant_models = list(method_details.get("generated_mutant_models", []))
        generated_wt_models = list(method_details.get("generated_wt_models", []))
        _write_text(pairing_artifact_path, json.dumps(model_pairing, indent=2, sort_keys=True))
        _write_text(
            backend_logs_path,
            "\n".join(
                [
                    "=== RepairPDB stdout ===",
                    repair_process.stdout,
                    "=== RepairPDB stderr ===",
                    repair_process.stderr,
                    "=== BuildModel stdout tail ===",
                    str(method_details.get("stdout_tail", "")),
                ]
            ),
        )
        _write_text(
            execution_trace_path,
            _build_execution_trace(
                original_structure_path=original_structure_path,
                repair_input_path=repair_input_pdb,
                repaired_output_path=repaired_output_path,
                buildmodel_input_path=buildmodel_input_path,
                raw_output_path=raw_output_path,
                method_work_dir=Path(str(method_details.get("work_dir", ""))),
                buildmodel_started=True,
            ),
        )

        backend_status = "ok" if method_result.status == "ok" and n_runs_valid >= 1 else "failed"
        error_message = None if backend_status == "ok" else _error_from_details(method_details)
        payload_out = {
            "original_structure_path": str(original_structure_path),
            "repaired_structure_path": str(repaired_output_path),
            "buildmodel_input_path": str(buildmodel_input_path),
            "mutation": payload["mutation"],
            "chain_id": payload["chain_id"],
            "residue_id": payload["residue_id"],
            "generated_mutant_models": generated_mutant_models,
            "generated_wt_models": generated_wt_models,
            "model_pairing": model_pairing,
            "per_run_ddg": per_run_ddg,
            "ddg": ddg,
            "ddg_std": ddg_std,
            "n_runs_requested": self.number_of_runs,
            "n_runs_valid": n_runs_valid,
            "backend_status": backend_status,
            "backend_logs_path": str(backend_logs_path),
            "repair": repair_payload,
            "execution_trace_path": str(execution_trace_path),
            "model_pairing_path": str(pairing_artifact_path),
            "method_result": {
                "method": method_result.method,
                "status": method_result.status,
                "score": method_result.score,
                "units": method_result.units,
                "details": method_details,
            },
        }
        _write_text(raw_output_path, json.dumps(payload_out, indent=2, sort_keys=True))
        return BackendRunResult(
            status=backend_status,
            raw_output_path=raw_output_path,
            error_message=error_message,
            details=payload_out,
        )

    def parse_output(self, raw_output_path: Path) -> float | None:
        payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        if payload.get("backend_status") != "ok":
            return None
        ddg = payload.get("ddg")
        return None if ddg is None else float(ddg)

    def extract_summary(self, raw_output_path: Path) -> dict[str, Any]:
        payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        return {
            "ddg_std": payload.get("ddg_std"),
            "n_runs_requested": payload.get("n_runs_requested"),
            "n_runs_valid": payload.get("n_runs_valid"),
            "per_run_ddg": payload.get("per_run_ddg", []),
        }

    def normalize(self, raw_value: float | None) -> float | None:
        return normalize_ddg(self.method_name, raw_value)


@dataclass(frozen=True, slots=True)
class RosettaDDGAdapter(DDGAdapter):
    method_name: str = "rosetta"
    sequence_based: bool = False
    stochastic: bool = True
    number_of_runs: int = 20
    top_k: int = 3

    def validate_environment(self) -> None:
        status = _collect_rosetta_environment_check()
        if status["missing_components"]:
            raise RuntimeError(
                "Rosetta runtime is incomplete: " + ", ".join(str(component) for component in status["missing_components"])
            )

    def prepare_input(self, record: CanonicalMutationRecord, structure_source: str, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        structure_path = getattr(record.structure_paths, structure_source)
        if structure_path is None:
            raise ValueError(f"Missing structure for source {structure_source}")
        structure = load_structure(structure_path)
        site = extract_mutation_site(structure, record.position, expected_wt=record.wt)
        residue_id = str(site.residue_id)
        if not residue_id.isdigit():
            raise ValueError(f"Unsupported non-numeric residue id for Rosetta ddg_monomer: {residue_id}")
        source_input_pdb, prepared_from_cif, pdb_prepare_seconds = _prepare_local_pdb_copy(
            Path(str(structure_path)).expanduser().resolve(),
            work_dir,
        )
        renumbered_input_pdb = work_dir / "input_rosetta.pdb"
        residue_mapping = _renumber_pdb_for_rosetta(source_input_pdb, renumbered_input_pdb)
        mapping_key = (site.chain_id, residue_id)
        if mapping_key not in residue_mapping:
            raise ValueError(f"Rosetta residue mapping missing for {site.chain_id}:{residue_id}")
        rosetta_residue_number = residue_mapping[mapping_key]
        mutation_file = work_dir / "mutations.txt"
        mutation_file.write_text(
            "\n".join(
                [
                    "total 1",
                    "1",
                    f"{record.wt} {rosetta_residue_number} {record.mut}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        prepared_input_path = work_dir / "prepared_input.json"
        _write_text(
            prepared_input_path,
            json.dumps(
                {
                    "protein_id": record.protein_id,
                    "mutation": record.mutation,
                    "structure_source": structure_source,
                    "structure_path": str(Path(str(structure_path)).expanduser().resolve()),
                    "source_input_pdb": str(source_input_pdb),
                    "input_pdb": str(renumbered_input_pdb),
                    "mutation_file": str(mutation_file),
                    "chain_id": site.chain_id,
                    "residue_id": residue_id,
                    "rosetta_residue_number": rosetta_residue_number,
                    "residue_mapping": {
                        f"{chain_id}:{current_residue_id}": mapped_index
                        for (chain_id, current_residue_id), mapped_index in residue_mapping.items()
                    },
                    "prepared_from_cif": prepared_from_cif,
                    "pdb_prepare_runtime_seconds": pdb_prepare_seconds,
                    "n_runs_requested": self.number_of_runs,
                    "top_k": self.top_k,
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return prepared_input_path

    def run(
        self,
        prepared_input_path: Path,
        record: CanonicalMutationRecord,
        structure_source: str,
        work_dir: Path,
    ) -> BackendRunResult:
        payload = json.loads(prepared_input_path.read_text(encoding="utf-8"))
        environment_check = _collect_rosetta_environment_check()
        environment_check_path = work_dir / "rosetta_environment_check.json"
        _write_text(environment_check_path, json.dumps(environment_check, indent=2, sort_keys=True))
        protocol_path = work_dir / "rosetta_protocol.md"
        _write_rosetta_protocol(protocol_path, self.number_of_runs, min(self.top_k, self.number_of_runs))
        raw_output_path = work_dir / "raw_output.json"
        backend_logs_path = work_dir / "backend.log"
        parsed_output_path = work_dir / "parsed_output.json"
        if environment_check["missing_components"]:
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "backend_status": "failed",
                        "error": "rosetta_runtime_incomplete",
                        "environment_check_path": str(environment_check_path),
                        "protocol_path": str(protocol_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(
                status="failed",
                raw_output_path=raw_output_path,
                error_message="rosetta_runtime_incomplete",
            )

        relax_binary = str(environment_check["relax_binary"])
        ddg_binary = str(environment_check["ddg_monomer_binary"])
        database_path = str(environment_check["database_path"])
        relax_dir = work_dir / "relax"
        if relax_dir.exists():
            shutil.rmtree(relax_dir)
        relax_dir.mkdir(parents=True, exist_ok=True)
        relax_stdout_path = relax_dir / "stdout.txt"
        relax_stderr_path = relax_dir / "stderr.txt"
        relax_command = [
            relax_binary,
            "-database",
            database_path,
            "-in:file:s",
            str(Path(payload["input_pdb"]).resolve()),
            "-out:path:all",
            str(relax_dir.resolve()),
            "-nstruct",
            "1",
            "-ignore_unrecognized_res",
            "true",
        ]
        relax_process = subprocess.run(relax_command, cwd=work_dir, capture_output=True, text=True)
        _write_text(relax_stdout_path, relax_process.stdout)
        _write_text(relax_stderr_path, relax_process.stderr)
        relaxed_pdb = relax_dir / f"{Path(payload['input_pdb']).stem}_0001.pdb"
        if relax_process.returncode != 0:
            _write_text(
                backend_logs_path,
                "\n".join(
                    [
                        "=== Relax stdout ===",
                        relax_process.stdout,
                        "=== Relax stderr ===",
                        relax_process.stderr,
                    ]
                ),
            )
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "backend_status": "failed",
                        "error": "rosetta_relax_failed",
                        "environment_check_path": str(environment_check_path),
                        "protocol_path": str(protocol_path),
                        "relax_command": relax_command,
                        "relax_stdout_path": str(relax_stdout_path),
                        "relax_stderr_path": str(relax_stderr_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(status="failed", raw_output_path=raw_output_path, error_message="rosetta_relax_failed")
        if not relaxed_pdb.exists() or relaxed_pdb.stat().st_size == 0:
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "backend_status": "failed",
                        "error": "rosetta_relaxed_pdb_missing",
                        "environment_check_path": str(environment_check_path),
                        "protocol_path": str(protocol_path),
                        "relax_command": relax_command,
                        "relax_stdout_path": str(relax_stdout_path),
                        "relax_stderr_path": str(relax_stderr_path),
                        "expected_relaxed_pdb": str(relaxed_pdb),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(status="failed", raw_output_path=raw_output_path, error_message="rosetta_relaxed_pdb_missing")

        runs_dir = work_dir / "runs"
        if runs_dir.exists():
            shutil.rmtree(runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        max_workers = min(cpu_count(), self.number_of_runs)
        tasks = [
            {
                "run_index": run_index,
                "run_dir": str((runs_dir / f"run_{run_index:03d}").resolve()),
                "ddg_binary": ddg_binary,
                "database_path": database_path,
                "relaxed_pdb": str(relaxed_pdb.resolve()),
                "mutation_file": str(Path(payload["mutation_file"]).resolve()),
                "mutation_label": record.mutation,
            }
            for run_index in range(self.number_of_runs)
        ]
        with Pool(processes=max_workers) as pool:
            run_results = pool.map(_run_rosetta_ddg_worker, tasks)

        failures = [result for result in run_results if result["status"] != "ok"]
        if failures:
            _write_text(
                backend_logs_path,
                json.dumps({"relax_command": relax_command, "run_failures": failures}, indent=2, sort_keys=True),
            )
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "backend_status": "failed",
                        "error": "rosetta_ddg_run_failed",
                        "environment_check_path": str(environment_check_path),
                        "protocol_path": str(protocol_path),
                        "relax_command": relax_command,
                        "relax_stdout_path": str(relax_stdout_path),
                        "relax_stderr_path": str(relax_stderr_path),
                        "relaxed_pdb": str(relaxed_pdb),
                        "run_results": run_results,
                        "max_parallel_jobs": max_workers,
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(status="failed", raw_output_path=raw_output_path, error_message="rosetta_ddg_run_failed")

        per_run_ddg = [float(item["ddg"]) for item in run_results]
        wildtype_energies = [float(item["wildtype_dg"]) for item in run_results]
        mutant_energies = [float(item["mutant_dg"]) for item in run_results]
        selected_count = min(self.top_k, len(run_results))
        selected_wildtype_dg = sorted(wildtype_energies)[:selected_count]
        selected_mutant_dg = sorted(mutant_energies)[:selected_count]
        ddg_value = _mean(selected_mutant_dg) - _mean(selected_wildtype_dg)
        ddg_std = _std(per_run_ddg)
        parsed_payload = {
            "per_run_ddg": per_run_ddg,
            "per_run_wildtype_dg": wildtype_energies,
            "per_run_mutant_dg": mutant_energies,
            "selected_top_k_wildtype_dg": selected_wildtype_dg,
            "selected_top_k_mutant_dg": selected_mutant_dg,
            "ddg_raw": ddg_value,
            "n_runs_requested": self.number_of_runs,
            "n_runs_valid": len(per_run_ddg),
        }
        _write_text(parsed_output_path, json.dumps(parsed_payload, indent=2, sort_keys=True))
        normalization_payload = {
            "ddg_raw": ddg_value,
            "ddg": ddg_value,
            "unit_raw": "REU",
            "unit": "REU",
            "sign_convention_raw": "mutant_minus_wildtype",
            "sign_convention": "mutant_minus_wildtype",
            "normalization_note": "Rosetta ddg_monomer values are reported in REU and are not converted to kcal/mol.",
        }
        _write_text(
            backend_logs_path,
            json.dumps(
                {
                    "relax_command": relax_command,
                    "ddg_binary": ddg_binary,
                    "max_parallel_jobs": max_workers,
                    "run_results": run_results,
                },
                indent=2,
                sort_keys=True,
            ),
        )
        payload_out = {
            "backend_status": "ok",
            "representation": "structure",
            "structure_source": structure_source,
            "input_pdb": payload["input_pdb"],
            "source_input_pdb": payload.get("source_input_pdb"),
            "mutation_file": payload["mutation_file"],
            "chain_id": payload["chain_id"],
            "residue_id": payload["residue_id"],
            "rosetta_residue_number": payload.get("rosetta_residue_number"),
            "residue_mapping": payload.get("residue_mapping", {}),
            "relax_command": relax_command,
            "relax_stdout_path": str(relax_stdout_path),
            "relax_stderr_path": str(relax_stderr_path),
            "relaxed_pdb": str(relaxed_pdb),
            "command_template": [
                ddg_binary,
                "-database",
                database_path,
                "-in:file:s",
                str(relaxed_pdb.resolve()),
                "-ddg::mut_file",
                str(Path(payload["mutation_file"]).resolve()),
                "-ddg::iterations",
                "1",
            ],
            "run_results": run_results,
            "max_parallel_jobs": max_workers,
            "ddg_raw": ddg_value,
            "ddg": ddg_value,
            "ddg_std": ddg_std,
            "per_run_ddg": per_run_ddg,
            "per_run_wildtype_dg": wildtype_energies,
            "per_run_mutant_dg": mutant_energies,
            "selected_top_k_wildtype_dg": selected_wildtype_dg,
            "selected_top_k_mutant_dg": selected_mutant_dg,
            "n_runs_requested": self.number_of_runs,
            "n_runs_valid": len(per_run_ddg),
            "top_k": selected_count,
            "environment_check_path": str(environment_check_path),
            "protocol_path": str(protocol_path),
            "parsed_output_path": str(parsed_output_path),
            "backend_logs_path": str(backend_logs_path),
            **normalization_payload,
        }
        _write_text(raw_output_path, json.dumps(payload_out, indent=2, sort_keys=True))
        return BackendRunResult(status="ok", raw_output_path=raw_output_path, details=payload_out)

    def parse_output(self, raw_output_path: Path) -> float | None:
        payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        if payload.get("backend_status") != "ok":
            return None
        value = payload.get("ddg_raw")
        return None if value is None else float(value)

    def normalize(self, raw_value: float | None) -> float | None:
        return normalize_ddg(self.method_name, raw_value)

    def extract_summary(self, raw_output_path: Path) -> dict[str, Any]:
        payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        return {
            "ddg_std": payload.get("ddg_std"),
            "n_runs_requested": payload.get("n_runs_requested"),
            "n_runs_valid": payload.get("n_runs_valid"),
            "per_run_ddg": payload.get("per_run_ddg", []),
        }

    def result_unit(self, ddg: float | None) -> str | None:
        return "REU" if ddg is not None else None

    def normalization_details(self, raw_output_path: Path) -> dict[str, Any]:
        payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        return {
            "ddg_raw": payload.get("ddg_raw"),
            "unit_raw": payload.get("unit_raw"),
            "unit": payload.get("unit"),
            "sign_convention_raw": payload.get("sign_convention_raw"),
            "sign_convention": payload.get("sign_convention"),
            "normalization_note": payload.get("normalization_note"),
        }


@dataclass(frozen=True, slots=True)
class ESM2Adapter(DDGAdapter):
    method_name: str = "esm2"
    sequence_based: bool = True
    stochastic: bool = False

    def validate_environment(self) -> None:
        status = _collect_esm2_environment_check()
        if status["missing_components"]:
            raise RuntimeError(
                "ESM2 runtime is incomplete: " + ", ".join(str(component) for component in status["missing_components"])
            )

    def prepare_input(self, record: CanonicalMutationRecord, structure_source: str, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        wt_sequence = record.sequence
        mutant_sequence = apply_mutation(record.sequence, record.mutation)
        payload = {
            "protein_id": record.protein_id,
            "mutation": record.mutation,
            "position": record.position,
            "wt": record.wt,
            "mut": record.mut,
            "wt_sequence": wt_sequence,
            "mutant_sequence": mutant_sequence,
            "structure_source": structure_source,
        }
        prepared_input_path = work_dir / "prepared_input.json"
        _write_text(prepared_input_path, json.dumps(payload, indent=2, sort_keys=True))
        return prepared_input_path

    def run(
        self,
        prepared_input_path: Path,
        record: CanonicalMutationRecord,
        structure_source: str,
        work_dir: Path,
    ) -> BackendRunResult:
        environment_check = _collect_esm2_environment_check()
        environment_check_path = work_dir / "esm_environment_check.json"
        _write_text(environment_check_path, json.dumps(environment_check, indent=2, sort_keys=True))
        raw_output_path = work_dir / "raw_output.json"
        backend_logs_path = work_dir / "backend.log"
        model_info_path = work_dir / "esm_model_info.json"
        if environment_check["missing_components"]:
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "backend_status": "failed",
                        "error": "esm2_runtime_incomplete",
                        "environment_check_path": str(environment_check_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(status="failed", raw_output_path=raw_output_path, error_message="esm2_runtime_incomplete")

        python_executable = str(environment_check["python_executable"])
        runtime_dir = _workspace_root(Path(__file__)) / ".torch-cache" / "esm2_runtime" / record.protein_id / record.mutation
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_input_path = runtime_dir / "prepared_input.json"
        runtime_raw_output_path = runtime_dir / "raw_output.json"
        shutil.copy2(prepared_input_path, runtime_input_path)
        command = [
            python_executable,
            _esm2_inference_script_windows(),
            "--mode",
            "infer",
            "--prepared-input",
            _to_windows_path(runtime_input_path),
            "--output",
            _to_windows_path(runtime_raw_output_path),
            "--torch-home",
            _esm2_torch_home_windows(),
            "--device",
            _esm2_device(),
        ]
        env = dict(os.environ)
        env["TORCH_HOME"] = _esm2_torch_home_windows()
        process = subprocess.run(command, capture_output=True, text=True, env=env)
        _write_text(
            backend_logs_path,
            json.dumps(
                {
                    "command": command,
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "torch_home": env["TORCH_HOME"],
                    "esm2_device": _esm2_device(),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        if process.returncode != 0 or not runtime_raw_output_path.exists():
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "backend_status": "failed",
                        "error": "esm2_inference_failed",
                        "command": command,
                        "returncode": process.returncode,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                        "environment_check_path": str(environment_check_path),
                        "backend_logs_path": str(backend_logs_path),
                        "esm2_device": _esm2_device(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(status="failed", raw_output_path=raw_output_path, error_message="esm2_inference_failed")

        shutil.copy2(runtime_raw_output_path, raw_output_path)
        payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        if math.isnan(float(payload["ddg_raw"])):
            _write_text(
                raw_output_path,
                json.dumps(
                    {
                        "backend_status": "failed",
                        "error": "esm2_nan_output",
                        "command": command,
                        "environment_check_path": str(environment_check_path),
                        "backend_logs_path": str(backend_logs_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return BackendRunResult(status="failed", raw_output_path=raw_output_path, error_message="esm2_nan_output")

        _write_text(
            model_info_path,
            json.dumps(
                {
                    "loaded": payload.get("loaded"),
                    "device": payload.get("device"),
                    "requested_device": payload.get("requested_device"),
                    "selected_device": payload.get("selected_device"),
                    "dtype": payload.get("dtype"),
                    "parameter_count": payload.get("parameter_count"),
                    "alphabet_size": payload.get("alphabet_size"),
                    "torch_version": payload.get("torch_version"),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        payload.update(
            {
                "backend_status": "ok",
                "representation": "sequence",
                "structure_source": structure_source,
                "command": command,
                "environment_check_path": str(environment_check_path),
                "model_info_path": str(model_info_path),
                "backend_logs_path": str(backend_logs_path),
            }
        )
        _write_text(raw_output_path, json.dumps(payload, indent=2, sort_keys=True))
        return BackendRunResult(status="ok", raw_output_path=raw_output_path, details=payload)

    def parse_output(self, raw_output_path: Path) -> float | None:
        payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        if payload.get("backend_status") != "ok":
            return None
        value = payload.get("ddg_raw")
        return None if value is None else float(value)

    def normalize(self, raw_value: float | None) -> float | None:
        return normalize_ddg(self.method_name, raw_value)

    def result_unit(self, ddg: float | None) -> str | None:
        return "log-probability" if ddg is not None else None

    def normalization_details(self, raw_output_path: Path) -> dict[str, Any]:
        payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
        return {
            "ddg_raw": payload.get("ddg_raw"),
            "unit_raw": payload.get("unit_raw"),
            "unit": payload.get("unit"),
            "sign_convention_raw": payload.get("sign_convention_raw"),
            "sign_convention": payload.get("sign_convention"),
            "normalization_note": payload.get("normalization_note"),
        }


AVAILABLE_METHODS: dict[str, DDGAdapter] = {
    "foldx": FoldXAdapter(),
    "rosetta": RosettaDDGAdapter(),
    "esm2": ESM2Adapter(),
}


def default_adapters() -> list[DDGAdapter]:
    return [AVAILABLE_METHODS["foldx"], AVAILABLE_METHODS["rosetta"], AVAILABLE_METHODS["esm2"]]
