from pathlib import Path

import pandas as pd

from pipeline.external_benchmarks import S669Config, preprocess_s669


def _write_pdb(path: Path, *, residue_name: str, chain: str, residue_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"ATOM      1  N   {residue_name} {chain}{residue_id:4d}      11.104  13.207   9.341  1.00 20.00           N",
                f"ATOM      2  CA  {residue_name} {chain}{residue_id:4d}      12.104  13.207   9.341  1.00 20.00           C",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_preprocess_s669_imports_direct_and_inverse_rows(tmp_path):
    raw_dir = tmp_path / "raw" / "s669"
    source_csv = raw_dir / "S669" / "S669.csv"
    pdb_dir = raw_dir / "S669" / "pdbs"
    source_csv.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Protein": "1abcA",
                "PDB_Mut": "S11A",
                "Seq_Mut": "S7A",
                "Temperature": 298.15,
                "pH": 7.0,
                "DOI": "10/example",
                "EXPDTA": "X-RAY DIFFRACTION",
                "Resolution": 1.5,
                "WT_PDB": "1abc",
                "Chain": "A",
                "INV_PDB": "1abcA_S11A",
                "Experimental_DDG_dir": -1.2,
                "Experimental_DDG_inv": 1.2,
            }
        ]
    ).to_csv(source_csv, index=False)
    _write_pdb(pdb_dir / "1abc.pdb", residue_name="SER", chain="A", residue_id=11)
    _write_pdb(pdb_dir / "1abcA_S11A.pdb", residue_name="ALA", chain="A", residue_id=7)

    config = S669Config(
        raw_dir=raw_dir,
        source_csv=source_csv,
        processed_path=tmp_path / "processed" / "s669.csv",
        include_inverse=True,
    )

    path = preprocess_s669(config)
    frame = pd.read_csv(path)

    assert len(frame) == 2
    direct = frame[frame["direction"] == "direct"].iloc[0]
    inverse = frame[frame["direction"] == "inverse"].iloc[0]
    assert direct["position"] == 7
    assert direct["pdb_residue_id"] == 11
    assert direct["mutation_id"] == "S7A"
    assert direct["experimental_ddg"] == -1.2
    assert inverse["mutation_id"] == "A7S"
    assert inverse["pdb_residue_id"] == 7
    assert inverse["experimental_ddg"] == 1.2
    assert inverse["structure_id"] == "1abcA_S11A"


def test_preprocess_s669_can_import_direct_only(tmp_path):
    raw_dir = tmp_path / "raw" / "s669"
    source_csv = raw_dir / "S669" / "S669.csv"
    pdb_dir = raw_dir / "S669" / "pdbs"
    source_csv.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Protein": "1abcA",
                "PDB_Mut": "S11A",
                "Seq_Mut": "S7A",
                "Temperature": 298.15,
                "pH": 7.0,
                "DOI": "",
                "EXPDTA": "",
                "Resolution": "",
                "WT_PDB": "1abc",
                "Chain": "A",
                "INV_PDB": "1abcA_S11A",
                "Experimental_DDG_dir": -1.2,
                "Experimental_DDG_inv": 1.2,
            }
        ]
    ).to_csv(source_csv, index=False)
    _write_pdb(pdb_dir / "1abc.pdb", residue_name="SER", chain="A", residue_id=11)

    config = S669Config(
        raw_dir=raw_dir,
        source_csv=source_csv,
        processed_path=tmp_path / "processed" / "s669.csv",
        include_inverse=False,
    )

    frame = pd.read_csv(preprocess_s669(config))

    assert frame["direction"].tolist() == ["direct"]
    assert frame["protein_id"].tolist() == ["1abcA_direct"]
