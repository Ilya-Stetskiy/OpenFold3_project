"""Local S669 ddG runner helpers for notebooks."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENFOLD_ROOT = REPO_ROOT / "openfold-3"
NOTEBOOK_HELPERS_ROOT = REPO_ROOT / "openfold_notebooks" / "helpers"
for candidate in (OPENFOLD_ROOT, NOTEBOOK_HELPERS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from of_notebook_lib.ddg_tools import collect_ddg_tool_status
from openfold3.benchmark.harness import DdgBenchmarkHarness, HarnessReport
from openfold3.benchmark.methods import FoldXBuildModelMethod, RosettaScoreMethod
from openfold3.benchmark.models import BenchmarkCase, MutationInput

from .external_benchmarks import S669Config, load_s669
from .s669_analysis import metric_table


@dataclass(frozen=True, slots=True)
class LocalCapability:
    method: str
    runnable: bool
    role: str
    reason: str
    path: str | None = None


def local_capabilities(project_dir: Path | None = None) -> pd.DataFrame:
    """Return methods that can be used locally in the current environment."""

    root = (project_dir or REPO_ROOT).resolve()
    statuses = collect_ddg_tool_status(root)
    rows = [
        LocalCapability(
            method="foldx",
            runnable=bool(statuses["foldx"].found),
            role="mutation_ddg",
            reason="FoldX BuildModel is the local kcal/mol ddG method."
            if statuses["foldx"].found
            else "FoldX binary was not found.",
            path=statuses["foldx"].path,
        ),
        LocalCapability(
            method="rosetta_score",
            runnable=bool(statuses["rosetta_score_jd2"].found and statuses["rosetta_database"].found),
            role="structure_score_proxy",
            reason=(
                "score_jd2 runs locally, but this project method is a structural energy proxy, not a mutation ddG protocol."
                if statuses["rosetta_score_jd2"].found and statuses["rosetta_database"].found
                else "Rosetta score_jd2 binary or database was not found."
            ),
            path=statuses["rosetta_score_jd2"].path,
        ),
        LocalCapability(
            method="saambe_3d",
            runnable=_module_available("prody"),
            role="mutation_ddg_optional",
            reason="Requires prody and SAAMBE-3D runtime dependencies.",
            path=str(root / "third_party" / "SAAMBE-3D" / "saambe-3d.py"),
        ),
        LocalCapability(
            method="helixon_binding_ddg",
            runnable=(root / "third_party" / "binding-ddg-predictor" / "data" / "model.pt").exists()
            and _module_available("torch"),
            role="optional_model_method",
            reason="Requires model.pt and torch.",
            path=str(root / "third_party" / "binding-ddg-predictor"),
        ),
        LocalCapability(
            method="prompt_ddg",
            runnable=(root / "third_party" / "Prompt-DDG" / "trained_models" / "ddg_model.ckpt").exists()
            and _module_available("torch"),
            role="optional_model_method",
            reason="Requires ddg_model.ckpt and torch.",
            path=str(root / "third_party" / "Prompt-DDG"),
        ),
    ]
    return pd.DataFrame(asdict(row) for row in rows)


def export_local_tool_env(project_dir: Path | None = None) -> dict[str, str]:
    """Export env vars needed by local harness methods."""

    root = (project_dir or REPO_ROOT).resolve()
    statuses = collect_ddg_tool_status(root)
    exported: dict[str, str] = {}
    mapping = {
        "FOLDX_BINARY": statuses["foldx"].path,
        "ROSETTA_SCORE_JD2_BINARY": statuses["rosetta_score_jd2"].path,
        "ROSETTA_DATABASE": statuses["rosetta_database"].path,
    }
    for key, value in mapping.items():
        if value:
            os.environ[key] = value
            exported[key] = value
    return exported


def run_s669_local_sample(
    *,
    sample_n: int = 3,
    include_rosetta_score: bool = False,
    output_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Run local available S669 methods on a small sample and return long results."""

    export_local_tool_env(REPO_ROOT)
    s669 = load_s669(S669Config())
    sample = s669.head(sample_n).copy()
    output_root = (output_dir or (REPO_ROOT / "analysis" / "s669_local_run")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    methods: list[object] = [FoldXBuildModelMethod()]
    if include_rosetta_score:
        methods.append(RosettaScoreMethod())
    harness = DdgBenchmarkHarness(methods=methods)

    rows: list[dict[str, Any]] = []
    for record in sample.to_dict(orient="records"):
        case_id = _case_id(record)
        report_path = output_root / "reports" / f"{case_id}.json"
        if force or not report_path.exists():
            report = harness.run_case(_benchmark_case(record, case_id))
            DdgBenchmarkHarness.write_report(report, report_path)
        else:
            report = _load_report(report_path)
        rows.extend(_report_rows(record, report, report_path))

    result = pd.DataFrame(rows)
    result_path = output_root / "s669_local_results_long.csv"
    result.to_csv(result_path, index=False)
    return result


def summarize_local_results(results: pd.DataFrame) -> pd.DataFrame:
    """Compute metric summary for local S669 run results."""

    ok = results[results["status"].eq("ok") & results["ddg"].notna()].copy()
    if ok.empty:
        return pd.DataFrame()
    metrics_input = ok.rename(columns={"unit": "units"})
    metrics_input["representation"] = metrics_input["method"].map(
        lambda method: "structure_proxy" if method == "rosetta_score" else "structure"
    )
    return metric_table(metrics_input)


def _benchmark_case(record: dict[str, Any], case_id: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        structure_path=Path(record["pdb_path"]),
        mutations=(
            MutationInput(
                chain_id=str(record["chain"]),
                from_residue=str(record["wt_residue"]),
                position_1based=int(record["pdb_residue_id"]),
                to_residue=str(record["mut_residue"]),
            ),
        ),
        experimental_ddg=float(record["experimental_ddg"]),
        notes=f"S669 {record['direction']} {record['mutation_id']}",
        pdb_id=str(record["pdb_id"]),
    )


def _case_id(record: dict[str, Any]) -> str:
    return f"S669_{record['protein_id']}_{record['mutation_id']}".replace("/", "_")


def _report_rows(
    record: dict[str, Any],
    report: HarnessReport,
    report_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in report.results:
        rows.append(
            {
                "benchmark": "S669",
                "direction": record["direction"],
                "protein_id": record["protein_id"],
                "pdb_id": record["pdb_id"],
                "chain": record["chain"],
                "position": int(record["position"]),
                "pdb_residue_id": str(record["pdb_residue_id"]),
                "mutation_id": record["mutation_id"],
                "method": result.method,
                "status": result.status,
                "ddg": result.score,
                "unit": result.units,
                "experimental_ddg": float(record["experimental_ddg"]),
                "report_path": str(report_path),
            }
        )
    return rows


def _load_report(path: Path) -> HarnessReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    from openfold3.benchmark.harness import MethodResult

    return HarnessReport(
        case_id=payload["case_id"],
        structure_path=payload["structure_path"],
        confidence_path=payload.get("confidence_path"),
        structure_summary=payload["structure_summary"],
        results=tuple(MethodResult(**item) for item in payload["results"]),
        experimental_ddg=payload.get("experimental_ddg"),
        notes=payload.get("notes"),
    )


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True
