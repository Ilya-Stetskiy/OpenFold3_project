from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .cif_utils import parse_structure_records
from .harness import HarnessContext, MethodResult


def _candidate_repo_tool_paths(executable: str) -> list[Path]:
    current = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in current.parents:
        candidate = parent / "tools" / "bin" / executable
        if candidate.exists():
            candidates.append(candidate)
    return candidates


def _candidate_repo_glob_paths(pattern: str) -> list[Path]:
    current = Path(__file__).resolve()
    candidates: list[Path] = []
    seen: set[Path] = set()
    for parent in current.parents:
        tools_dir = parent / "tools"
        if not tools_dir.exists():
            continue
        for candidate in tools_dir.rglob(pattern):
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _candidate_repo_relative_paths(relative_path: str) -> list[Path]:
    current = Path(__file__).resolve()
    candidates: list[Path] = []
    seen: set[Path] = set()
    for parent in current.parents:
        candidate = parent / relative_path
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _resolve_existing_path(
    env_var_name: str | None = None,
    relative_path: str | None = None,
) -> str | None:
    if env_var_name:
        override = os.environ.get(env_var_name)
        if override and Path(override).exists():
            return str(Path(override).resolve())
    if relative_path is None:
        return None
    for candidate in _candidate_repo_relative_paths(relative_path):
        return str(candidate.resolve())
    return None


def _resolve_python_executable(env_var_name: str | None = None) -> str | None:
    return _resolve_executable_path("python3", env_var_name=env_var_name)


def _with_pythonpath(env: dict[str, str], pythonpath: str | None) -> dict[str, str]:
    merged = dict(env)
    if pythonpath is None:
        return merged
    existing = merged.get("PYTHONPATH")
    merged["PYTHONPATH"] = (
        pythonpath if not existing else pythonpath + os.pathsep + existing
    )
    return merged


def _extract_first_float(value: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if match is None:
        raise ValueError(f"Could not parse float from: {value!r}")
    return float(match.group(0))


def _parse_last_json_line(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Could not parse JSON payload from process stdout")


def _resolve_executable_path(
    executable: str | None, env_var_name: str | None = None
) -> str | None:
    if executable is None:
        return None

    if env_var_name:
        override = os.environ.get(env_var_name)
        if override:
            override_path = shutil.which(override) or (
                override if Path(override).exists() else None
            )
            if override_path is not None:
                return override_path

    resolved = shutil.which(executable)
    if resolved is not None:
        return resolved

    for candidate in _candidate_repo_tool_paths(executable):
        return str(candidate)
    return None


def _resolve_rosetta_binary(
    executable: str | None, env_var_name: str | None = None
) -> str | None:
    resolved = _resolve_executable_path(executable, env_var_name=env_var_name)
    if resolved is not None:
        return resolved
    if executable is None:
        return None
    for candidate in _candidate_repo_glob_paths(f"{executable}*linuxgccrelease"):
        if candidate.is_file():
            return str(candidate)
    return None


def _resolve_rosetta_database() -> str | None:
    override = os.environ.get("ROSETTA_DATABASE")
    if override and Path(override).exists():
        return override
    for candidate in _candidate_repo_glob_paths("database"):
        if candidate.is_dir():
            return str(candidate)
    return None


def _infer_rosetta_database_from_binary(executable_path: str) -> str | None:
    resolved_binary = Path(executable_path).resolve()
    for parent in resolved_binary.parents:
        candidate = parent / "database"
        if candidate.is_dir():
            return str(candidate)
    return None


def _foldx_mutation_token(chain_id: str, from_residue: str, position_1based: int, to_residue: str) -> str:
    return f"{from_residue.upper()}{chain_id}{position_1based}{to_residue.upper()};"


def _read_foldx_first_row(path: Path) -> dict[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("Pdb\t")
    )
    headers = lines[header_index].split("\t")
    values = lines[header_index + 1].split("\t")
    return dict(zip(headers, values, strict=True))


def _read_foldx_total_energy(path: Path) -> float:
    row = _read_foldx_first_row(path)
    return float(row["total energy"])


def _resolve_foldx_output_path(
    output_dir: Path,
    *,
    prefix: str,
    suffix: str,
) -> Path:
    exact_path = output_dir / f"{prefix}_{suffix}_AC.fxout"
    if exact_path.exists():
        return exact_path
    fallback_patterns = (
        f"{prefix}_{suffix}.fxout",
        f"{prefix}_{suffix}_*.fxout",
    )
    for pattern in fallback_patterns:
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not resolve FoldX output for {prefix=} {suffix=} in {output_dir}"
    )


def _element_from_atom_name(atom_name: str) -> str:
    letters = "".join(char for char in atom_name if char.isalpha())
    if not letters:
        return "X"
    if len(letters) >= 2 and letters[0].upper() == "H":
        return "H"
    return letters[0].upper()


def _write_atom_records_to_pdb(structure_path: Path, pdb_path: Path) -> None:
    atoms = parse_structure_records(structure_path)
    lines: list[str] = []
    serial = 1
    previous_chain = None
    for atom in atoms:
        if previous_chain is not None and atom.chain_id != previous_chain:
            lines.append("TER")
        previous_chain = atom.chain_id
        atom_name = atom.atom_name[:4]
        res_name = atom.residue_name[:3]
        chain_id = (atom.chain_id or "?")[:1]
        residue_token = atom.residue_id
        digits = "".join(char for char in residue_token if char.isdigit()) or "1"
        insertion = next((char for char in residue_token if char.isalpha()), " ")
        x = atom.x
        y = atom.y
        z = atom.z
        b_factor = 0.0 if atom.b_factor is None else atom.b_factor
        element = _element_from_atom_name(atom.atom_name)
        lines.append(
            f"ATOM  {serial:5d} {atom_name:>4} {res_name:>3} {chain_id}{int(digits):4d}{insertion:1}"
            f"   {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{b_factor:6.2f}          {element:>2}"
        )
        serial += 1
    lines.append("TER")
    lines.append("END")
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_local_pdb_copy(structure_path: Path, work_dir: Path) -> tuple[Path, bool, float]:
    started = time.perf_counter()
    if structure_path.suffix.lower() == ".pdb":
        structure_copy = work_dir / structure_path.name
        structure_copy.write_bytes(structure_path.read_bytes())
        prepared_from_cif = False
    else:
        structure_copy = work_dir / f"{structure_path.stem}.pdb"
        _write_atom_records_to_pdb(structure_path, structure_copy)
        prepared_from_cif = True
    return structure_copy, prepared_from_cif, time.perf_counter() - started


def _parse_rosetta_scorefile_total(scorefile_path: Path) -> float:
    score_lines = [
        line.strip()
        for line in scorefile_path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("SCORE:")
    ]
    if len(score_lines) < 2:
        raise ValueError(f"Could not parse Rosetta scorefile {scorefile_path}")
    header = score_lines[0].split()
    values = score_lines[-1].split()
    row = dict(zip(header[1:], values[1:], strict=True))
    return float(row["total_score"])


@dataclass(frozen=True)
class StructureInterfaceMethod:
    name: str = "structure_interface"

    def run(self, context: HarnessContext) -> MethodResult:
        summary = context.structure_summary
        score = float(summary.interface_ca_contacts_8a)
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="ca_contacts",
            details={
                "chain_ids": list(summary.chain_ids),
                "chain_lengths": {
                    chain_id: len(residues)
                    for chain_id, residues in summary.residues_by_chain.items()
                },
                "interface_atom_contacts_5a": summary.interface_atom_contacts_5a,
                "interface_ca_contacts_8a": summary.interface_ca_contacts_8a,
                "min_inter_chain_atom_distance": summary.min_inter_chain_atom_distance,
            },
        )


@dataclass(frozen=True)
class OpenFoldConfidenceMethod:
    name: str = "openfold_confidence"

    def run(self, context: HarnessContext) -> MethodResult:
        payload = context.confidence_payload
        if payload is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={"reason": "confidence_json_missing"},
            )
        score = payload.get("sample_ranking_score")
        return MethodResult(
            method=self.name,
            status="ok",
            score=None if score is None else float(score),
            units="ranking_score",
            details={
                "avg_plddt": payload.get("avg_plddt"),
                "iptm": payload.get("iptm"),
                "ptm": payload.get("ptm"),
                "gpde": payload.get("gpde"),
                "has_clash": payload.get("has_clash"),
                "chain_pair_iptm": payload.get("chain_pair_iptm", {}),
            },
        )


@dataclass(frozen=True)
class ExternalToolMethod:
    name: str
    executable: str | None = None
    env_var_name: str | None = None
    requires_mutation: bool = True
    supports_cif: bool = False
    server_only: bool = False

    def run(self, context: HarnessContext) -> MethodResult:
        if self.server_only:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "server_only_manual_submission",
                    "mutations_present": bool(context.case.mutations),
                },
            )
        if self.requires_mutation and not context.case.mutations:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={"reason": "mutation_spec_missing"},
            )
        if not self.supports_cif and context.case.structure_path.suffix.lower() == ".cif":
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={"reason": "pdb_conversion_required"},
            )
        executable_path = _resolve_executable_path(
            self.executable, env_var_name=self.env_var_name
        )
        if self.executable and executable_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "executable_not_found",
                    "expected_executable": self.executable,
                    "env_var_name": self.env_var_name,
                },
            )
        return MethodResult(
            method=self.name,
            status="ready",
            details={
                "resolved_executable": executable_path,
                "structure_path": str(context.case.structure_path),
                "mutation_ids": [mutation.mutation_id for mutation in context.case.mutations],
            },
        )


@dataclass(frozen=True)
class FoldXBuildModelMethod:
    name: str = "foldx"
    executable: str = "foldx"
    env_var_name: str = "FOLDX_BINARY"
    requires_mutation: bool = True
    supports_cif: bool = True
    number_of_runs: int = 1

    def run(self, context: HarnessContext) -> MethodResult:
        if self.requires_mutation and not context.case.mutations:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={"reason": "mutation_spec_missing"},
            )
        executable_path = _resolve_executable_path(
            self.executable, env_var_name=self.env_var_name
        )
        if executable_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "executable_not_found",
                    "expected_executable": self.executable,
                    "env_var_name": self.env_var_name,
                },
            )

        started_at = time.perf_counter()
        work_dir = Path(tempfile.mkdtemp(prefix="foldx-buildmodel-"))
        output_dir = work_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_copy, prepared_from_cif, pdb_prepare_seconds = _prepare_local_pdb_copy(
            context.case.structure_path, work_dir
        )
        mutant_file = work_dir / "individual_list.txt"
        mutant_file.write_text(
            "".join(
                _foldx_mutation_token(
                    mutation.chain_id,
                    mutation.from_residue,
                    mutation.position_1based,
                    mutation.to_residue,
                )
                + "\n"
                for mutation in context.case.mutations
            ),
            encoding="utf-8",
        )

        output_prefix = context.case.case_id.replace(" ", "_")
        build_command = [
            executable_path,
            "--command",
            "BuildModel",
            "--pdb",
            structure_copy.name,
            "--mutant-file",
            mutant_file.name,
            "--output-dir",
            output_dir.name,
            "--output-file",
            output_prefix,
            "--numberOfRuns",
            str(self.number_of_runs),
            "--screen",
            "false",
        ]
        build_started = time.perf_counter()
        build_process = subprocess.run(
            build_command,
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        build_runtime_seconds = time.perf_counter() - build_started
        structure_stem = structure_copy.stem
        dif_path = output_dir / f"Dif_{output_prefix}_{structure_stem}.fxout"
        average_path = output_dir / f"Average_{output_prefix}_{structure_stem}.fxout"
        raw_path = output_dir / f"Raw_{output_prefix}_{structure_stem}.fxout"
        pdb_list_path = output_dir / f"PdbList_{output_prefix}_{structure_stem}.fxout"

        if build_process.returncode != 0:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "foldx_buildmodel_failed",
                    "returncode": build_process.returncode,
                    "resolved_executable": executable_path,
                    "work_dir": str(work_dir),
                    "stdout": build_process.stdout[-4000:],
                    "stderr": build_process.stderr[-4000:],
                },
            )

        if not dif_path.exists():
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "foldx_output_missing",
                    "resolved_executable": executable_path,
                    "work_dir": str(work_dir),
                    "expected_dif_path": str(dif_path),
                    "stdout": build_process.stdout[-4000:],
                    "stderr": build_process.stderr[-4000:],
                },
            )

        buildmodel_total_energy_change = _read_foldx_total_energy(dif_path)
        pdb_names = [
            line.strip()
            for line in pdb_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if pdb_list_path.exists() else []
        mutant_model_name = pdb_names[0] if len(pdb_names) >= 1 else None
        wt_model_name = pdb_names[1] if len(pdb_names) >= 2 else None
        if mutant_model_name is None or wt_model_name is None:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "foldx_generated_pdbs_missing",
                    "resolved_executable": executable_path,
                    "work_dir": str(work_dir),
                    "pdb_list_path": str(pdb_list_path),
                    "generated_pdbs": pdb_names,
                },
            )

        def _run_analyse_complex(pdb_name: str, suffix: str) -> tuple[dict[str, str], Path, Path, float]:
            command = [
                executable_path,
                "--command",
                "AnalyseComplex",
                "--pdb",
                pdb_name,
                "--pdb-dir",
                output_dir.name,
                "--output-dir",
                output_dir.name,
                "--output-file",
                suffix,
                "--screen",
                "false",
            ]
            started = time.perf_counter()
            process = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
            runtime = time.perf_counter() - started
            if process.returncode != 0:
                raise RuntimeError(
                    f"AnalyseComplex failed for {pdb_name}: "
                    f"{process.stdout[-1000:]} {process.stderr[-1000:]}"
                )
            try:
                summary_path = _resolve_foldx_output_path(
                    output_dir,
                    prefix="Summary",
                    suffix=suffix,
                )
                interaction_path = _resolve_foldx_output_path(
                    output_dir,
                    prefix="Interaction",
                    suffix=suffix,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"AnalyseComplex produced no interaction outputs for {pdb_name}: "
                    f"{process.stdout[-1000:]} {process.stderr[-1000:]} ({exc})"
                ) from exc
            return _read_foldx_first_row(summary_path), summary_path, interaction_path, runtime

        try:
            mutant_summary, mutant_summary_path, mutant_interaction_path, mutant_runtime = _run_analyse_complex(
                mutant_model_name, f"{output_prefix}_mut"
            )
            wt_summary, wt_summary_path, wt_interaction_path, wt_runtime = _run_analyse_complex(
                wt_model_name, f"{output_prefix}_wt"
            )
            protocol = "BuildModel+AnalyseComplex"
            mutant_interaction_energy = float(mutant_summary["Interaction Energy"])
            wt_interaction_energy = float(wt_summary["Interaction Energy"])
            score = mutant_interaction_energy - wt_interaction_energy
        except RuntimeError as exc:
            protocol = "BuildModel"
            mutant_summary = {}
            wt_summary = {}
            mutant_summary_path = output_dir / "Summary_unavailable.fxout"
            wt_summary_path = output_dir / "Summary_unavailable.fxout"
            mutant_interaction_path = output_dir / "Interaction_unavailable.fxout"
            wt_interaction_path = output_dir / "Interaction_unavailable.fxout"
            mutant_runtime = 0.0
            wt_runtime = 0.0
            mutant_interaction_energy = None
            wt_interaction_energy = None
            score = buildmodel_total_energy_change
            analyse_complex_error = str(exc)
        else:
            analyse_complex_error = None
        runtime_seconds = time.perf_counter() - started_at
        details: dict[str, object] = {
            "protocol": protocol,
            "resolved_executable": executable_path,
            "work_dir": str(work_dir),
            "mutation_ids": [mutation.mutation_id for mutation in context.case.mutations],
            "dif_path": str(dif_path),
            "average_path": str(average_path),
            "raw_path": str(raw_path),
            "pdb_list_path": str(pdb_list_path),
            "prepared_input_pdb_path": str(structure_copy),
            "prepared_from_cif": prepared_from_cif,
            "pdb_prepare_runtime_seconds": pdb_prepare_seconds,
            "buildmodel_runtime_seconds": build_runtime_seconds,
            "analyse_mutant_runtime_seconds": mutant_runtime,
            "analyse_wt_runtime_seconds": wt_runtime,
            "runtime_seconds": runtime_seconds,
            "buildmodel_total_energy_change": buildmodel_total_energy_change,
            "mutant_interaction_energy": mutant_interaction_energy,
            "wt_interaction_energy": wt_interaction_energy,
            "mutant_summary_path": str(mutant_summary_path),
            "wt_summary_path": str(wt_summary_path),
            "mutant_interaction_path": str(mutant_interaction_path),
            "wt_interaction_path": str(wt_interaction_path),
            "stdout_tail": build_process.stdout[-2000:],
        }
        if analyse_complex_error is not None:
            details["analyse_complex_error"] = analyse_complex_error
        details["generated_pdbs"] = pdb_names
        details["mutant_model_path"] = str(output_dir / mutant_model_name)
        details["wt_model_path"] = str(output_dir / wt_model_name)
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="kcal/mol",
            details=details,
        )


@dataclass(frozen=True)
class RosettaScoreMethod:
    name: str = "rosetta_score"
    executable: str = "score_jd2"
    env_var_name: str = "ROSETTA_SCORE_JD2_BINARY"
    database_env_var_name: str = "ROSETTA_DATABASE"
    supports_cif: bool = True

    def run(self, context: HarnessContext) -> MethodResult:
        executable_path = _resolve_rosetta_binary(
            self.executable, env_var_name=self.env_var_name
        )
        if executable_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "executable_not_found",
                    "expected_executable": self.executable,
                    "env_var_name": self.env_var_name,
                },
            )
        database_path = _resolve_rosetta_database()
        if database_path is None:
            database_path = _infer_rosetta_database_from_binary(executable_path)
        if database_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "rosetta_database_not_found",
                    "database_env_var_name": self.database_env_var_name,
                    "resolved_executable": executable_path,
                    "database_inferred_from_binary": False,
                },
            )

        started_at = time.perf_counter()
        work_dir = Path(tempfile.mkdtemp(prefix="rosetta-score-"))
        output_dir = work_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_copy, prepared_from_cif, pdb_prepare_seconds = _prepare_local_pdb_copy(
            context.case.structure_path, work_dir
        )
        scorefile_path = output_dir / "score.sc"
        command = [
            executable_path,
            "-database",
            database_path,
            "-in:file:s",
            str(structure_copy),
            "-out:file:scorefile",
            str(scorefile_path),
            "-out:path:all",
            str(output_dir),
            "-overwrite",
        ]
        process = subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        runtime_seconds = time.perf_counter() - started_at
        if process.returncode != 0:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "rosetta_process_failed",
                    "returncode": process.returncode,
                    "resolved_executable": executable_path,
                    "database_path": database_path,
                    "work_dir": str(work_dir),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        if not scorefile_path.exists():
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "rosetta_scorefile_missing",
                    "resolved_executable": executable_path,
                    "database_path": database_path,
                    "work_dir": str(work_dir),
                    "expected_scorefile_path": str(scorefile_path),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        score = _parse_rosetta_scorefile_total(scorefile_path)
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="rosetta_energy",
            details={
                "protocol": "score_jd2",
                "resolved_executable": executable_path,
                "database_path": database_path,
                "database_inferred_from_binary": (
                    "ROSETTA_DATABASE" not in os.environ
                    and _resolve_rosetta_database() != database_path
                ),
                "work_dir": str(work_dir),
                "prepared_input_pdb_path": str(structure_copy),
                "prepared_from_cif": prepared_from_cif,
                "pdb_prepare_runtime_seconds": pdb_prepare_seconds,
                "runtime_seconds": runtime_seconds,
                "scorefile_path": str(scorefile_path),
                "mutation_ids": [mutation.mutation_id for mutation in context.case.mutations],
                "stdout_tail": process.stdout[-2000:],
            },
        )


def _parse_saambe_prediction(output_path: Path) -> tuple[float, str]:
    payload = output_path.read_text(encoding="utf-8").strip()
    if not payload:
        raise ValueError(f"Empty SAAMBE-3D output: {output_path}")
    value = _extract_first_float(payload)
    tokens = payload.split()
    label = tokens[-1] if len(tokens) >= 2 else "unknown"
    return value, label


def _parse_helixon_prediction(stdout: str) -> float:
    for line in reversed(stdout.splitlines()):
        if "Predicted ddG:" in line:
            return _extract_first_float(line)
    raise ValueError("Could not find 'Predicted ddG:' in HeliXon output")


@dataclass(frozen=True)
class Saambe3DMethod:
    name: str = "saambe_3d"
    python_env_var_name: str = "SAAMBE_3D_PYTHON"
    script_env_var_name: str = "SAAMBE_3D_SCRIPT"
    pythonpath_env_var_name: str = "SAAMBE_3D_PYTHONPATH"
    supports_cif: bool = True

    def run(self, context: HarnessContext) -> MethodResult:
        if not context.case.mutations:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={"reason": "mutation_spec_missing"},
            )
        if len(context.case.mutations) != 1:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "single_mutation_only",
                    "mutation_count": len(context.case.mutations),
                },
            )

        python_executable = _resolve_python_executable(self.python_env_var_name)
        if python_executable is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "python_not_found",
                    "env_var_name": self.python_env_var_name,
                },
            )

        script_path = _resolve_existing_path(
            env_var_name=self.script_env_var_name,
            relative_path="third_party/SAAMBE-3D/saambe-3d.py",
        )
        if script_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "saambe_script_not_found",
                    "script_env_var_name": self.script_env_var_name,
                },
            )

        dependency_path = _resolve_existing_path(
            env_var_name=self.pythonpath_env_var_name,
            relative_path=".deps/saambe3d",
        )

        mutation = context.case.mutations[0]
        started_at = time.perf_counter()
        work_dir = Path(tempfile.mkdtemp(prefix="saambe-3d-"))
        structure_copy, prepared_from_cif, pdb_prepare_seconds = _prepare_local_pdb_copy(
            context.case.structure_path, work_dir
        )
        output_path = work_dir / "output.out"
        env = dict(os.environ)
        env["HOME"] = str(work_dir)
        env["PRODYRC"] = str(work_dir / ".prodyrc")
        env = _with_pythonpath(env, dependency_path)
        command = [
            python_executable,
            script_path,
            "-i",
            str(structure_copy),
            "-c",
            mutation.chain_id,
            "-r",
            str(mutation.position_1based),
            "-w",
            mutation.from_residue.upper(),
            "-m",
            mutation.to_residue.upper(),
            "-d",
            "1",
            "-o",
            str(output_path),
        ]
        process = subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            text=True,
            env=env,
        )
        runtime_seconds = time.perf_counter() - started_at
        if process.returncode != 0:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "saambe_process_failed",
                    "returncode": process.returncode,
                    "python_executable": python_executable,
                    "script_path": script_path,
                    "dependency_path": dependency_path,
                    "work_dir": str(work_dir),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        if not output_path.exists():
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "saambe_output_missing",
                    "python_executable": python_executable,
                    "script_path": script_path,
                    "dependency_path": dependency_path,
                    "work_dir": str(work_dir),
                    "expected_output_path": str(output_path),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        try:
            score, label = _parse_saambe_prediction(output_path)
        except ValueError as exc:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "saambe_parse_failed",
                    "error": str(exc),
                    "output_path": str(output_path),
                    "output_text": output_path.read_text(encoding="utf-8")[-4000:],
                },
            )
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="kcal/mol",
            details={
                "protocol": "SAAMBE-3D",
                "python_executable": python_executable,
                "script_path": script_path,
                "dependency_path": dependency_path,
                "work_dir": str(work_dir),
                "prepared_input_pdb_path": str(structure_copy),
                "prepared_from_cif": prepared_from_cif,
                "pdb_prepare_runtime_seconds": pdb_prepare_seconds,
                "runtime_seconds": runtime_seconds,
                "output_path": str(output_path),
                "prediction_label": label,
                "mutation_ids": [mutation.mutation_id for mutation in context.case.mutations],
                "stdout_tail": process.stdout[-2000:],
                "stderr_tail": process.stderr[-2000:],
            },
        )


@dataclass(frozen=True)
class HeliXonBindingDdgMethod:
    name: str = "helixon_binding_ddg"
    python_env_var_name: str = "HELIXON_BINDING_DDG_PYTHON"
    script_env_var_name: str = "HELIXON_BINDING_DDG_SCRIPT"
    model_env_var_name: str = "HELIXON_BINDING_DDG_MODEL"
    pythonpath_env_var_name: str = "HELIXON_BINDING_DDG_PYTHONPATH"

    def run(self, context: HarnessContext) -> MethodResult:
        if not context.case.mutations:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={"reason": "mutation_spec_missing"},
            )

        python_executable = _resolve_python_executable(self.python_env_var_name)
        if python_executable is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "python_not_found",
                    "env_var_name": self.python_env_var_name,
                },
            )

        script_path = _resolve_existing_path(
            env_var_name=self.script_env_var_name,
            relative_path="third_party/binding-ddg-predictor/scripts/predict.py",
        )
        if script_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "helixon_script_not_found",
                    "script_env_var_name": self.script_env_var_name,
                },
            )

        model_path = _resolve_existing_path(
            env_var_name=self.model_env_var_name,
            relative_path="third_party/binding-ddg-predictor/data/model.pt",
        )
        if model_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "helixon_model_not_found",
                    "model_env_var_name": self.model_env_var_name,
                },
            )

        dependency_path = _resolve_existing_path(
            env_var_name=self.pythonpath_env_var_name,
            relative_path=".deps/helixon",
        )

        foldx_result = FoldXBuildModelMethod().run(context)
        if foldx_result.status != "ok":
            return MethodResult(
                method=self.name,
                status="failed" if foldx_result.status == "failed" else "unavailable",
                details={
                    "reason": "helixon_mutant_preparation_failed",
                    "foldx_status": foldx_result.status,
                    "foldx_details": foldx_result.details,
                },
            )

        wt_model_path = str(foldx_result.details["wt_model_path"])
        mutant_model_path = str(foldx_result.details["mutant_model_path"])
        started_at = time.perf_counter()
        env = dict(os.environ)
        env = _with_pythonpath(env, dependency_path)
        command = [
            python_executable,
            script_path,
            wt_model_path,
            mutant_model_path,
            "--model",
            model_path,
            "--device",
            "cpu",
        ]
        process = subprocess.run(
            command,
            cwd=str(Path(script_path).resolve().parents[1]),
            capture_output=True,
            text=True,
            env=env,
        )
        runtime_seconds = time.perf_counter() - started_at
        if process.returncode != 0:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "helixon_process_failed",
                    "returncode": process.returncode,
                    "python_executable": python_executable,
                    "script_path": script_path,
                    "model_path": model_path,
                    "dependency_path": dependency_path,
                    "wt_model_path": wt_model_path,
                    "mutant_model_path": mutant_model_path,
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        try:
            score = _parse_helixon_prediction(process.stdout)
        except ValueError as exc:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "helixon_parse_failed",
                    "error": str(exc),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="kcal/mol",
            details={
                "protocol": "binding-ddg-predictor",
                "python_executable": python_executable,
                "script_path": script_path,
                "model_path": model_path,
                "dependency_path": dependency_path,
                "runtime_seconds": runtime_seconds,
                "wt_model_path": wt_model_path,
                "mutant_model_path": mutant_model_path,
                "foldx_mutant_preparation_details": foldx_result.details,
                "stdout_tail": process.stdout[-2000:],
                "stderr_tail": process.stderr[-2000:],
            },
        )


@dataclass(frozen=True)
class PromptDdgMethod:
    name: str = "prompt_ddg"
    python_env_var_name: str = "PROMPT_DDG_PYTHON"
    wrapper_env_var_name: str = "PROMPT_DDG_INFER_SCRIPT"
    checkpoint_env_var_name: str = "PROMPT_DDG_CHECKPOINT"

    def run(self, context: HarnessContext) -> MethodResult:
        if not context.case.mutations:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={"reason": "mutation_spec_missing"},
            )

        python_executable = _resolve_python_executable(self.python_env_var_name)
        if python_executable is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "python_not_found",
                    "env_var_name": self.python_env_var_name,
                },
            )

        checkpoint_path = _resolve_existing_path(
            env_var_name=self.checkpoint_env_var_name,
            relative_path="third_party/Prompt-DDG/trained_models/ddg_model.ckpt",
        )
        if checkpoint_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "prompt_ddg_checkpoint_not_found",
                    "checkpoint_env_var_name": self.checkpoint_env_var_name,
                },
            )

        wrapper_path = _resolve_existing_path(
            env_var_name=self.wrapper_env_var_name,
            relative_path="scripts/dev/prompt_ddg_infer.py",
        )
        if wrapper_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "prompt_ddg_wrapper_not_found",
                    "wrapper_env_var_name": self.wrapper_env_var_name,
                    "checkpoint_path": checkpoint_path,
                },
            )

        started_at = time.perf_counter()
        work_dir = Path(tempfile.mkdtemp(prefix="prompt-ddg-"))
        structure_copy, prepared_from_cif, pdb_prepare_seconds = _prepare_local_pdb_copy(
            context.case.structure_path, work_dir
        )
        command = [
            python_executable,
            wrapper_path,
            str(structure_copy),
            "--checkpoint",
            checkpoint_path,
        ]
        for mutation in context.case.mutations:
            command.extend(
                [
                    "--mutation",
                    f"{mutation.chain_id}:{mutation.from_residue}{mutation.position_1based}{mutation.to_residue}",
                ]
            )
        process = subprocess.run(
            command,
            cwd=str(Path(wrapper_path).resolve().parents[2]),
            capture_output=True,
            text=True,
        )
        runtime_seconds = time.perf_counter() - started_at
        if process.returncode != 0:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "prompt_ddg_process_failed",
                    "returncode": process.returncode,
                    "python_executable": python_executable,
                    "wrapper_path": wrapper_path,
                    "checkpoint_path": checkpoint_path,
                    "work_dir": str(work_dir),
                    "prepared_input_pdb_path": str(structure_copy),
                    "prepared_from_cif": prepared_from_cif,
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        try:
            payload = _parse_last_json_line(process.stdout)
            score = float(payload["score"])
        except (KeyError, TypeError, ValueError) as exc:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "prompt_ddg_parse_failed",
                    "error": str(exc),
                    "python_executable": python_executable,
                    "wrapper_path": wrapper_path,
                    "checkpoint_path": checkpoint_path,
                    "work_dir": str(work_dir),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="kcal/mol",
            details={
                "protocol": "Prompt-DDG",
                "python_executable": python_executable,
                "wrapper_path": wrapper_path,
                "checkpoint_path": checkpoint_path,
                "work_dir": str(work_dir),
                "prepared_input_pdb_path": str(structure_copy),
                "prepared_from_cif": prepared_from_cif,
                "pdb_prepare_runtime_seconds": pdb_prepare_seconds,
                "runtime_seconds": runtime_seconds,
                "mutation_ids": [mutation.mutation_id for mutation in context.case.mutations],
                "payload": payload,
                "stdout_tail": process.stdout[-2000:],
                "stderr_tail": process.stderr[-2000:],
            },
        )


def multiscale_methods() -> list[object]:
    return [
        FoldXBuildModelMethod(),
        RosettaScoreMethod(),
        Saambe3DMethod(),
        HeliXonBindingDdgMethod(),
        PromptDdgMethod(),
    ]


def default_methods() -> list[object]:
    return [
        OpenFoldConfidenceMethod(),
        StructureInterfaceMethod(),
        *multiscale_methods(),
        ExternalToolMethod(
            name="mutabind2",
            server_only=True,
            requires_mutation=True,
        ),
        ExternalToolMethod(
            name="mcsm_ppi2",
            server_only=True,
            requires_mutation=True,
        ),
    ]
