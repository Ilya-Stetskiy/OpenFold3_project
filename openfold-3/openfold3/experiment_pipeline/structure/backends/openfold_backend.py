from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..models import BackendResult, MutationCase


def _is_preferred_cif_name(path: Path) -> bool:
    name = path.name.lower()
    return "final" in name or ("model" in name and "intermediate" not in name)


def _select_output_cif(cif_candidates: list[Path]) -> Path:
    print(f"[OF3] Found {len(cif_candidates)} CIF candidates")
    if not cif_candidates:
        raise RuntimeError("OpenFold3 completed but no CIF output was found")
    if len(cif_candidates) == 1:
        return cif_candidates[0]

    preferred = [path for path in cif_candidates if _is_preferred_cif_name(path)]
    if len(preferred) == 1:
        return preferred[0]

    candidate_names = ", ".join(sorted(str(path) for path in cif_candidates))
    raise RuntimeError(
        "OpenFold3 completed with ambiguous CIF outputs: " f"{candidate_names}"
    )


def _build_query_payload(case: MutationCase, sequence: str) -> dict[str, object]:
    return {
        "queries": {
            case.case_id: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": [case.chain],
                        "sequence": sequence,
                    }
                ]
            }
        }
    }


@dataclass(frozen=True, slots=True)
class OpenFold3Backend:
    backend_name: str = "openfold3"
    python_executable: str = sys.executable
    runner_yaml: Path | None = None
    inference_ckpt_path: Path | None = None

    def run(self, case: MutationCase, case_dir: Path, config_hash: str, sequence: str) -> BackendResult:
        backend_dir = (case_dir / "openfold3").resolve()
        input_dir = backend_dir / "input"
        output_dir = backend_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = input_dir / "mutant.fasta"
        fasta_path.write_text(f">{case.case_id}\n{sequence}\n", encoding="utf-8")

        query_json_path = input_dir / "query.json"
        query_json_path.write_text(
            json.dumps(_build_query_payload(case, sequence), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        command = [
            self.python_executable,
            "-m",
            "openfold3.run_openfold",
            "predict",
            "--query-json",
            str(query_json_path),
            "--output-dir",
            str(output_dir),
        ]
        if self.runner_yaml is not None:
            command.extend(["--runner-yaml", str(self.runner_yaml)])
        if self.inference_ckpt_path is not None:
            command.extend(["--inference-ckpt-path", str(self.inference_ckpt_path)])

        try:
            completed = subprocess.run(
                command,
                cwd=case_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            return BackendResult(
                backend_name=self.backend_name,
                status="failed",
                output_dir=backend_dir,
                message=f"OpenFold3 launch failed: {type(exc).__name__}: {exc}",
            )

        if completed.returncode != 0:
            return BackendResult(
                backend_name=self.backend_name,
                status="failed",
                output_dir=backend_dir,
                message=(
                    f"OpenFold3 predict failed with return code {completed.returncode}: "
                    f"{completed.stderr[-500:] or completed.stdout[-500:]}"
                ),
            )

        cif_candidates = [path for path in output_dir.rglob("*.cif") if path.is_file()]
        source_cif = _select_output_cif(cif_candidates)

        model_cif_path = output_dir / "model.cif"
        if source_cif.resolve() != model_cif_path.resolve():
            shutil.copy2(source_cif, model_cif_path)

        return BackendResult(
            backend_name=self.backend_name,
            status="ok",
            output_dir=backend_dir,
            artifact_paths=(model_cif_path,),
            message="OpenFold3 predict completed",
        )
