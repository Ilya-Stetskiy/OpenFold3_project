from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..models import BackendResult, MutationCase


def _model_path_from_confidence_path(path: Path) -> Path | None:
    name = path.name
    for suffix in ("_confidences_aggregated.json", "_confidences.json"):
        if name.endswith(suffix):
            return path.with_name(f"{name.removesuffix(suffix)}_model.cif")
    return None


def _select_ranked_cif(output_dir: Path) -> Path | None:
    ranked_candidates: list[tuple[float, int, Path]] = []
    for summary_path in sorted(output_dir.rglob("summary.jsonl")):
        for line_number, line in enumerate(summary_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            score = payload.get("sample_ranking_score")
            confidence_path = payload.get("aggregated_confidence_path")
            if score is None or confidence_path is None:
                continue
            model_path = _model_path_from_confidence_path(Path(str(confidence_path)))
            if model_path is None:
                continue
            if not model_path.is_absolute():
                model_path = (summary_path.parent / model_path).resolve()
            if model_path.exists():
                ranked_candidates.append((float(score), -line_number, model_path))

    if not ranked_candidates:
        return None
    ranked_candidates.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
    best = ranked_candidates[0]
    tied = [candidate for candidate in ranked_candidates if candidate[:2] == best[:2]]
    if len(tied) > 1:
        names = ", ".join(sorted(str(candidate[2]) for candidate in tied))
        raise RuntimeError(f"OpenFold3 summary ranking is ambiguous: {names}")
    return best[2]


def _select_first_sample_cif(cif_candidates: list[Path]) -> Path | None:
    sample_pattern = re.compile(r"_sample_(\d+)_model\.cif$")
    indexed: list[tuple[int, Path]] = []
    for path in cif_candidates:
        match = sample_pattern.search(path.name)
        if match is not None:
            indexed.append((int(match.group(1)), path))
    if len(indexed) != len(cif_candidates):
        return None
    sample_one = [path for index, path in indexed if index == 1]
    return sample_one[0] if len(sample_one) == 1 else None


def _is_preferred_cif_name(path: Path) -> bool:
    name = path.name.lower()
    return "final" in name or ("model" in name and "intermediate" not in name)


def _select_output_cif(cif_candidates: list[Path], output_dir: Path) -> Path:
    print(f"[OF3] Found {len(cif_candidates)} CIF candidates")
    if not cif_candidates:
        raise RuntimeError("OpenFold3 completed but no CIF output was found")
    if len(cif_candidates) == 1:
        return cif_candidates[0]

    ranked = _select_ranked_cif(output_dir)
    if ranked is not None:
        return ranked

    preferred = [path for path in cif_candidates if _is_preferred_cif_name(path)]
    if len(preferred) == 1:
        return preferred[0]

    first_sample = _select_first_sample_cif(cif_candidates)
    if first_sample is not None:
        return first_sample

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

        fasta_path = input_dir / "wt.fasta"
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
        source_cif = _select_output_cif(cif_candidates, output_dir)

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
