from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate single-point saturation mutagenesis CSV for an "
            "OpenFold3 query."
        )
    )
    parser.add_argument(
        "--query-json",
        type=Path,
        required=True,
        help="Path to OpenFold3 query JSON.",
    )
    parser.add_argument(
        "--query-name",
        required=True,
        help="Query key inside the JSON file.",
    )
    parser.add_argument(
        "--chain-id",
        required=True,
        help="Chain id to mutate, for example A.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--amino-acids",
        default=DEFAULT_AMINO_ACIDS,
        help=f"Target amino acid alphabet. Default: {DEFAULT_AMINO_ACIDS}",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include mutations where to_residue equals the wild-type residue.",
    )
    return parser.parse_args()


def load_sequence(query_json: Path, query_name: str, chain_id: str) -> str:
    payload = json.loads(query_json.read_text(encoding="utf-8"))
    query = payload["queries"][query_name]
    for chain in query["chains"]:
        if chain_id in chain["chain_ids"]:
            return chain["sequence"]
    raise ValueError(f"Chain id {chain_id!r} not found in query {query_name!r}")


def main() -> None:
    args = parse_args()
    sequence = load_sequence(args.query_json, args.query_name, args.chain_id)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chain_id", "position_1based", "from_residue", "to_residue"],
        )
        writer.writeheader()
        for index, wild_type in enumerate(sequence, start=1):
            for mutant in args.amino_acids:
                if not args.include_self and mutant == wild_type:
                    continue
                writer.writerow(
                    {
                        "chain_id": args.chain_id,
                        "position_1based": index,
                        "from_residue": wild_type,
                        "to_residue": mutant,
                    }
                )

    total_mutations = sum(
        1
        for wild_type in sequence
        for mutant in args.amino_acids
        if args.include_self or mutant != wild_type
    )
    print(
        f"Wrote {total_mutations} mutations for chain {args.chain_id} "
        f"({len(sequence)} residues) to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
