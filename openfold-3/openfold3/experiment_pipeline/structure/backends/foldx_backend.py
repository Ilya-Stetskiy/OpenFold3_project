from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..models import BackendResult, MutationCase


def _foldx_mutation_token(case: MutationCase) -> str:
    return f"{case.wt_residue}{case.chain}{case.pdb_residue_id}{case.mut_residue};"


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _resolve_binary(binary_path: Path) -> Path | None:
    if binary_path.exists():
        return binary_path.resolve()
    resolved = shutil.which(str(binary_path))
    return None if resolved is None else Path(resolved).resolve()


def _resolve_generated_model(output_dir: Path, output_prefix: str, structure_stem: str) -> Path | None:
    pdb_list_path = output_dir / f"PdbList_{output_prefix}_{structure_stem}.fxout"
    if pdb_list_path.exists():
        pdb_names = [line.strip() for line in pdb_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if pdb_names:
            candidate = output_dir / pdb_names[0]
            if candidate.exists():
                return candidate
    exact = output_dir / f"{output_prefix}_1.pdb"
    if exact.exists():
        return exact
    return None


@dataclass(frozen=True, slots=True)
class FoldXBackend:
    binary_path: Path
    number_of_runs: int = 1
    backend_name: str = "foldx"

    def run(self, case: MutationCase, case_dir: Path, config_hash: str) -> BackendResult:
        resolved_binary = _resolve_binary(self.binary_path)
        foldx_dir = (case_dir / "foldx").resolve()
        output_dir = foldx_dir / "output"
        foldx_dir.mkdir(parents=True, exist_ok=True)

        if resolved_binary is None:
            return BackendResult(
                backend_name="foldx",
                status="unavailable",
                output_dir=foldx_dir,
                message=f"FoldX binary not found: {self.binary_path}",
            )

        structure_copy = foldx_dir / case.pdb_path.name
        try:
            shutil.copy2(case.pdb_path, structure_copy)
        except OSError as exc:
            return BackendResult(
                backend_name="foldx",
                status="failed",
                output_dir=foldx_dir,
                message=f"Could not prepare FoldX input structure: {type(exc).__name__}: {exc}",
            )

        try:
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print("[FOLDX] cleaned output dir")
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return BackendResult(
                backend_name="foldx",
                status="failed",
                output_dir=foldx_dir,
                message=f"Could not prepare FoldX output directory: {type(exc).__name__}: {exc}",
            )

        individual_list_path = foldx_dir / "individual_list.txt"
        individual_list_path.write_text(_foldx_mutation_token(case) + "\n", encoding="utf-8")

        output_prefix = _slug(f"{case.case_id}_{config_hash}")
        command = [
            str(resolved_binary),
            "--command",
            "BuildModel",
            "--pdb",
            structure_copy.name,
            "--mutant-file",
            individual_list_path.name,
            "--output-dir",
            output_dir.name,
            "--output-file",
            output_prefix,
            "--numberOfRuns",
            str(self.number_of_runs),
            "--screen",
            "false",
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=foldx_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            return BackendResult(
                backend_name="foldx",
                status="failed",
                output_dir=foldx_dir,
                message=f"FoldX launch failed: {type(exc).__name__}: {exc}",
            )

        if completed.returncode != 0:
            return BackendResult(
                backend_name="foldx",
                status="failed",
                output_dir=foldx_dir,
                message=(
                    f"FoldX BuildModel failed with return code {completed.returncode}: "
                    f"{completed.stderr[-500:] or completed.stdout[-500:]}"
                ),
            )

        generated_model = _resolve_generated_model(output_dir, output_prefix, structure_copy.stem)
        if generated_model is None:
            return BackendResult(
                backend_name="foldx",
                status="failed",
                output_dir=foldx_dir,
                message="FoldX completed but mutant model was not generated",
            )

        mutant_pdb_path = output_dir / "mutant.pdb"
        shutil.copy2(generated_model, mutant_pdb_path)
        return BackendResult(
            backend_name="foldx",
            status="ok",
            output_dir=foldx_dir,
            artifact_paths=(mutant_pdb_path,),
            message="FoldX BuildModel completed",
        )
