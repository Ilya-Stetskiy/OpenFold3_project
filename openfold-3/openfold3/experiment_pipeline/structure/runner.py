from __future__ import annotations

import csv
import inspect
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Protocol

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - server environments normally provide tqdm
    def tqdm(iterable, **_kwargs):
        return iterable

from .cache import is_cached
from .manifest import load_manifest, manifest_entry_from_result, save_manifest
from .models import BackendResult, CaseManifest, MutationCase
from .sequence import apply_mutation


class BackendContractError(RuntimeError):
    pass


class StructureBackend(Protocol):
    backend_name: str


@dataclass(frozen=True, slots=True)
class StructureRunner:
    output_root: Path
    dry_run: bool = True
    config_hash: str = "dry-run"
    backends: tuple[StructureBackend, ...] = ()

    def run_case(self, case: MutationCase, sequence: str) -> CaseManifest:
        case_dir = self._case_dir(case)
        input_dir = case_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        mutated_sequence = apply_mutation(
            sequence,
            case.position,
            case.wt_residue,
            case.mut_residue,
        )
        case_json_path = input_dir / "case.json"
        mutant_fasta_path = input_dir / "mutant.fasta"
        manifest = CaseManifest(
            case=case,
            case_dir=case_dir,
            input_dir=input_dir,
            case_json_path=case_json_path,
            mutant_fasta_path=mutant_fasta_path,
            wild_type_sequence=sequence,
            mutated_sequence=mutated_sequence,
        )
        self._write_case_json(manifest)
        self._write_mutant_fasta(manifest)

        if self.dry_run:
            return manifest

        manifest_path = case_dir / "manifest.json"
        manifest_payload = load_manifest(manifest_path)
        backend_results: list[BackendResult] = []
        for backend in self.backends:
            backend_name = backend.backend_name
            if is_cached(manifest_payload, backend_name, self.config_hash):
                print(f"[SKIP] {case.case_id} {backend_name} cached")
                cached_state = manifest_payload.get(backend_name, {})
                backend_results.append(
                    BackendResult(
                        backend_name=backend_name,
                        status="cached",
                        output_dir=Path(str(cached_state.get("output_dir", case_dir / backend_name))),
                        artifact_paths=tuple(
                            Path(path) for path in cached_state.get("artifact_paths", [])
                        ),
                        message="cached",
                    )
                )
                continue

            result = self._run_backend_safely(backend, case, case_dir, sequence)
            manifest_payload[backend_name] = manifest_entry_from_result(result, self.config_hash)
            save_manifest(manifest_path, manifest_payload)
            backend_results.append(result)

        return replace(manifest, backend_results=tuple(backend_results))

    def run_all(
        self,
        cases: Iterable[MutationCase],
        sequences_by_case_id: Mapping[str, str],
    ) -> list[CaseManifest]:
        manifests: list[CaseManifest] = []
        case_list = list(cases)
        for case in tqdm(case_list, desc="Structure cases", unit="case", dynamic_ncols=True):
            sequence = sequences_by_case_id[case.case_id]
            manifests.append(self.run_case(case, sequence))
        if not self.dry_run:
            self.write_results_csv()
        return manifests

    def write_results_csv(self) -> Path:
        results_path = self.output_root / "results.csv"
        rows = []
        for manifest_path in sorted(self.output_root.glob("cases/*/manifest.json")):
            case_id = manifest_path.parent.name
            manifest_payload = load_manifest(manifest_path)
            openfold_state = manifest_payload.get("openfold3", {})
            foldx_state = manifest_payload.get("foldx", {})
            rows.append(
                {
                    "case_id": case_id,
                    "openfold3_status": str(openfold_state.get("status", "")),
                    "foldx_status": str(foldx_state.get("status", "")),
                    "openfold3_path": self._manifest_result_path(openfold_state),
                    "foldx_path": self._manifest_result_path(foldx_state),
                }
            )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "case_id",
                    "openfold3_status",
                    "foldx_status",
                    "openfold3_path",
                    "foldx_path",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return results_path

    def _case_dir(self, case: MutationCase) -> Path:
        return self.output_root / "cases" / case.case_id

    def _run_backend_safely(
        self,
        backend: StructureBackend,
        case: MutationCase,
        case_dir: Path,
        sequence: str,
    ) -> BackendResult:
        try:
            return self._run_backend(backend, case, case_dir, sequence)
        except BackendContractError as exc:
            print(f"[CONTRACT ERROR] backend={backend.backend_name} {exc}")
            raise
        except Exception as exc:  # noqa: BLE001 - backend isolation is intentional here
            backend_dir = (case_dir / backend.backend_name).resolve()
            backend_dir.mkdir(parents=True, exist_ok=True)
            return BackendResult(
                backend_name=backend.backend_name,
                status="failed",
                output_dir=backend_dir,
                artifact_paths=(),
                message=f"Unhandled backend exception: {type(exc).__name__}: {exc}",
            )

    def _run_backend(
        self,
        backend: StructureBackend,
        case: MutationCase,
        case_dir: Path,
        sequence: str,
    ) -> BackendResult:
        run = getattr(backend, "run")
        signature = inspect.signature(run)
        parameters = tuple(signature.parameters.values())
        positional_parameters = tuple(
            parameter
            for parameter in parameters
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if len(positional_parameters) != len(parameters):
            raise BackendContractError(
                f"Unsupported backend.run signature for {backend.backend_name}: {signature}"
            )
        if len(positional_parameters) == 3:
            return run(case, case_dir, self.config_hash)
        if len(positional_parameters) == 4:
            return run(case, case_dir, self.config_hash, sequence)
        raise BackendContractError(
            f"Unsupported backend.run signature for {backend.backend_name}: {signature}"
        )

    @staticmethod
    def _manifest_result_path(state: dict[str, object]) -> str:
        artifact_paths = state.get("artifact_paths")
        if isinstance(artifact_paths, list) and artifact_paths:
            return str(artifact_paths[0])
        output_dir = state.get("output_dir")
        return "" if output_dir is None else str(output_dir)

    @staticmethod
    def _write_case_json(manifest: CaseManifest) -> None:
        payload = asdict(manifest)
        payload["case"]["case_id"] = manifest.case.case_id
        payload["case"]["pdb_path"] = str(manifest.case.pdb_path)
        payload["case_dir"] = str(manifest.case_dir)
        payload["input_dir"] = str(manifest.input_dir)
        payload["case_json_path"] = str(manifest.case_json_path)
        payload["mutant_fasta_path"] = str(manifest.mutant_fasta_path)
        manifest.case_json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _write_mutant_fasta(manifest: CaseManifest) -> None:
        fasta = f">{manifest.case.case_id}\n{manifest.mutated_sequence}\n"
        manifest.mutant_fasta_path.write_text(fasta, encoding="utf-8")
