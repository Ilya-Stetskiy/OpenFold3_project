from __future__ import annotations

import pandas as pd


def preview_molecules(molecules: list[dict]) -> pd.DataFrame:
    rows = []
    for molecule in molecules:
        chain_ids = molecule.get("chain_ids") or [molecule.get("id")]
        sequence = molecule.get("sequence")
        rows.append(
            {
                "molecule_type": molecule.get("molecule_type") or molecule.get("type"),
                "chain_ids": ",".join(str(chain_id) for chain_id in chain_ids if chain_id),
                "length": len(sequence) if sequence else None,
                "sequence_preview": (
                    sequence[:20] + "..." + sequence[-10:]
                    if sequence and len(sequence) > 35
                    else sequence
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_molecules(molecules: list[dict]) -> list[str]:
    issues: list[str] = []

    if not molecules:
        issues.append("No molecules were provided.")
        return issues

    seen_chain_ids: set[str] = set()
    for index, molecule in enumerate(molecules, start=1):
        molecule_type = molecule.get("molecule_type") or molecule.get("type")
        if not molecule_type:
            issues.append(f"Molecule #{index} is missing molecule_type.")

        chain_ids = molecule.get("chain_ids") or ([molecule.get("id")] if molecule.get("id") else [])
        if not chain_ids:
            issues.append(f"Molecule #{index} is missing chain_ids.")
        else:
            for chain_id in chain_ids:
                if chain_id in seen_chain_ids:
                    issues.append(f"Chain id '{chain_id}' is duplicated.")
                seen_chain_ids.add(str(chain_id))

        if str(molecule_type).lower() in {"protein", "rna", "dna"} and not molecule.get("sequence"):
            issues.append(f"Molecule #{index} requires a sequence.")

    return issues


def format_sample_table(df_samples: pd.DataFrame) -> pd.DataFrame:
    if df_samples.empty:
        return df_samples.copy()

    columns = [
        "query_name",
        "sample_name",
        "sample_ranking_score",
        "iptm",
        "ptm",
        "avg_plddt",
        "gpde",
        "has_clash",
        "model_path",
    ]
    available = [column for column in columns if column in df_samples.columns]
    return df_samples[available].sort_values(
        ["sample_ranking_score", "iptm"],
        ascending=[False, False],
    )


def summarize_best_result(df_samples: pd.DataFrame) -> dict[str, str]:
    if df_samples.empty:
        return {"status": "No samples found."}

    best_row = df_samples.sort_values(
        ["sample_ranking_score", "iptm"],
        ascending=[False, False],
    ).iloc[0]

    iptm = best_row.get("iptm")
    avg_plddt = best_row.get("avg_plddt")

    if pd.notna(iptm) and iptm >= 0.7:
        interface_note = "interface looks promising"
    elif pd.notna(iptm) and iptm >= 0.5:
        interface_note = "interface may be plausible"
    elif pd.notna(iptm):
        interface_note = "interface signal is weak"
    else:
        interface_note = "interface score is missing"

    if pd.notna(avg_plddt) and avg_plddt >= 90:
        confidence_note = "local confidence is very high"
    elif pd.notna(avg_plddt) and avg_plddt >= 70:
        confidence_note = "local confidence is decent"
    elif pd.notna(avg_plddt):
        confidence_note = "local confidence is limited"
    else:
        confidence_note = "local confidence is missing"

    return {
        "best_sample": str(best_row.get("sample_name")),
        "best_query": str(best_row.get("query_name")),
        "interface_note": interface_note,
        "confidence_note": confidence_note,
    }


def format_mutation_ranking(df_summary: pd.DataFrame) -> pd.DataFrame:
    if df_summary.empty:
        return df_summary.copy()

    columns = [
        "mutation_label",
        "sample_ranking_score",
        "iptm",
        "ptm",
        "avg_plddt",
        "gpde",
        "has_clash",
        "is_wt",
    ]
    available = [column for column in columns if column in df_summary.columns]
    return df_summary[available].sort_values(
        ["is_wt", "sample_ranking_score", "iptm"],
        ascending=[False, False, False],
    )
