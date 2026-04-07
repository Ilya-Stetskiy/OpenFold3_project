from __future__ import annotations

import json
from pathlib import Path

from openfold3.benchmark.foldx_panel import build_foldx_panel_mutations, run_foldx_panel
from openfold3.benchmark.harness import DdgBenchmarkHarness, MethodResult


class _MappedFoldxMethod:
    name = "foldx"

    def __init__(self, mutant_by_case_id: dict[str, tuple[Path, float]]):
        self.mutant_by_case_id = mutant_by_case_id

    def run(self, context):
        mutant_model_path, score = self.mutant_by_case_id[context.case.case_id]
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="kcal/mol",
            details={
                "prepared_from_cif": False,
                "runtime_seconds": 0.01,
                "mutant_model_path": str(mutant_model_path),
            },
        )


def _write_two_chain_complex(
    path: Path,
    *,
    chain_a: str = "GG",
    chain_b: str = "DE",
) -> None:
    aa3 = {"G": "GLY", "D": "ASP", "E": "GLU", "A": "ALA", "N": "ASN"}

    def line(serial: int, atom: str, res: str, chain: str, resid: int, x: float, y: float, z: float, elem: str) -> str:
        return f"ATOM  {serial:5d} {atom:>4} {res:>3} {chain}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00          {elem:>2}"

    lines = []
    serial = 1
    coords = [
        ("N", "N", (0.0, 0.0, 0.0)),
        ("CA", "C", (1.3, 0.0, 0.0)),
        ("C", "C", (1.9, 1.3, 0.0)),
        ("O", "O", (1.2, 2.3, 0.0)),
    ]
    for resid, residue in enumerate(chain_a, start=1):
        for atom_name, elem, (x, y, z) in coords:
            lines.append(line(serial, atom_name, aa3[residue], "A", resid, x + resid * 3.0, y, z, elem))
            serial += 1
    for resid, residue in enumerate(chain_b, start=1):
        for atom_name, elem, (x, y, z) in coords:
            lines.append(line(serial, atom_name, aa3[residue], "B", resid, x + resid * 3.0, y + 5.0, z, elem))
            serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_foldx_panel_mutations_expands_19xN(tmp_path: Path) -> None:
    structure_path = tmp_path / "wt.pdb"
    _write_two_chain_complex(structure_path)

    source_path, pdb_id, mutations = build_foldx_panel_mutations(
        structure_path=structure_path,
        chain_id="B",
        positions=(1, 2),
    )

    assert source_path == structure_path.resolve()
    assert pdb_id is None
    assert len(mutations) == 38
    assert mutations[0].from_residue == "D"


def test_run_foldx_panel_writes_rows_ranking_and_resume(tmp_path: Path) -> None:
    structure_path = tmp_path / "wt.pdb"
    mutant_a = tmp_path / "mut_a.pdb"
    mutant_b = tmp_path / "mut_b.pdb"
    _write_two_chain_complex(structure_path, chain_b="DE")
    _write_two_chain_complex(mutant_a, chain_b="AE")
    _write_two_chain_complex(mutant_b, chain_b="NE")

    cases = {}
    for residue in "ACDEFGHIKLMNPQRSTVWY":
        if residue == "D":
            continue
        case_id = f"wt_b_d1{residue.lower()}"
        if residue == "A":
            cases[case_id] = (mutant_a, -1.0)
        elif residue == "N":
            cases[case_id] = (mutant_b, 0.5)
        else:
            cases[case_id] = (mutant_a, 1.0)
    harness = DdgBenchmarkHarness(methods=[_MappedFoldxMethod(cases)])

    result = run_foldx_panel(
        output_root=tmp_path / "run",
        structure_path=structure_path,
        chain_id="B",
        positions=(1,),
        cache_dir=tmp_path / "cache",
        session=None,
        harness=harness,
    )

    resumed = run_foldx_panel(
        output_root=tmp_path / "run",
        structure_path=structure_path,
        chain_id="B",
        positions=(1,),
        harness=harness,
    )

    assert result.summary_json_path.exists()
    assert resumed.summary_json_path.exists()
    assert resumed.rows_csv_path.exists()
    assert resumed.ranking_csv_path.exists()
    payload = json.loads(resumed.summary_json_path.read_text(encoding="utf-8"))
    assert payload["total_mutations"] == 19
    assert payload["successful_mutations"] == 19
    assert payload["ranking"][0]["mutation_id"] == "B_D1A"
