#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class MutationSpec:
    chain_id: str
    from_residue: str
    position_1based: int
    to_residue: str

    @property
    def token(self) -> str:
        return (
            f"{self.from_residue}{self.chain_id}"
            f"{self.position_1based}{self.to_residue}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_path")
    parser.add_argument(
        "--mutation",
        action="append",
        required=True,
        help="Mutation token, for example A:C182A or TH31W.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _infer_prompt_root(checkpoint_path: Path) -> Path:
    return checkpoint_path.resolve().parent.parent


def _bootstrap_paths(prompt_root: Path) -> None:
    workspace_root = prompt_root.parent.parent
    extra_paths = [
        prompt_root / "code",
        workspace_root / ".deps" / "promptddg",
        workspace_root / ".deps" / "helixon",
    ]
    for path in extra_paths:
        if path.exists():
            sys.path.insert(0, str(path))


def _parse_mutation_token(token: str) -> MutationSpec:
    colon_match = re.fullmatch(
        r"(?P<chain>[^:]+):(?P<wt>[A-Z])(?P<pos>\d+)(?P<mt>[A-Z])",
        token,
    )
    if colon_match is not None:
        return MutationSpec(
            chain_id=colon_match.group("chain"),
            from_residue=colon_match.group("wt"),
            position_1based=int(colon_match.group("pos")),
            to_residue=colon_match.group("mt"),
        )

    compact_match = re.fullmatch(
        r"(?P<wt>[A-Z])(?P<chain>[A-Za-z0-9])(?P<pos>\d+)(?P<mt>[A-Z])",
        token,
    )
    if compact_match is not None:
        return MutationSpec(
            chain_id=compact_match.group("chain"),
            from_residue=compact_match.group("wt"),
            position_1based=int(compact_match.group("pos")),
            to_residue=compact_match.group("mt"),
        )
    raise ValueError(f"Unsupported mutation token: {token!r}")


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    prompt_root = _infer_prompt_root(checkpoint_path)
    _bootstrap_paths(prompt_root)

    import torch
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.Polypeptide import index_to_one, one_to_index
    from dataset import PaddingCollate
    from model import DDG_RDE_Network
    from trainer import CrossValidation, recursive_to
    from common_utils.protein.parsers import parse_biopython_structure
    from common_utils.transforms import get_transform

    params = json.loads((prompt_root / "configs" / "param_configs.json").read_text())
    params["pre_epoch"] = 0
    config = SimpleNamespace(**params)
    device = torch.device(args.device)

    cv_mgr = CrossValidation(
        config=config,
        num_cvfolds=config.num_cvfolds,
        model_factory=DDG_RDE_Network,
    ).to(device)
    cv_mgr.load_state_dict(torch.load(checkpoint_path, map_location=device))

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(None, str(Path(args.pdb_path).resolve()))
    base_data, seq_map = parse_biopython_structure(structure[0])

    aa_mut = base_data["aa"].clone()
    mut_flag = torch.zeros(size=base_data["aa"].shape, dtype=torch.bool)
    parsed_mutations = [_parse_mutation_token(token) for token in args.mutation]
    for mutation in parsed_mutations:
        key = (mutation.chain_id, mutation.position_1based)
        if key not in seq_map:
            raise ValueError(f"Mutation position not found in structure: {mutation.token}")
        index = seq_map[key]
        observed = index_to_one(int(base_data["aa"][index].item()))
        if observed != mutation.from_residue:
            raise ValueError(
                "Prompt-DDG WT residue mismatch for "
                f"{mutation.token}: structure has {observed}"
            )
        mut_flag[index] = True
        aa_mut[index] = one_to_index(mutation.to_residue)

    sample = copy.deepcopy(base_data)
    sample["mut_flag"] = mut_flag
    sample["aa_mut"] = aa_mut
    sample["ddG"] = 0.0
    sample["mutstr"] = ",".join(mutation.token for mutation in parsed_mutations)
    transform = get_transform(
        [
            {"type": "select_atom", "resolution": "backbone+CB"},
            {
                "type": "selected_region_fixed_size_patch",
                "select_attr": "mut_flag",
                "patch_size": config.patch_size,
            },
        ]
    )
    sample = transform(sample)

    batch = PaddingCollate()([sample])
    batch = recursive_to(batch, device)
    fold_scores: list[float] = []
    for fold in range(config.num_cvfolds):
        model, _, _ = cv_mgr.get(fold)
        model.eval()
        with torch.no_grad():
            _, output = model(batch, None)
        fold_scores.append(float(output["ddG_pred"].detach().cpu()[0].item()))

    score = sum(fold_scores) / len(fold_scores)
    print(
        json.dumps(
            {
                "status": "ok",
                "score": score,
                "units": "kcal/mol",
                "fold_scores": fold_scores,
                "mutations": [mutation.token for mutation in parsed_mutations],
                "device": args.device,
                "checkpoint": str(checkpoint_path),
            }
        )
    )


if __name__ == "__main__":
    main()
