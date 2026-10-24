#!/usr/bin/env python3
"""Prepare current_internal structures and sequences for future recalculation jobs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DEFAULT_DATASETS_CONFIG = Path("configs/datasets.yaml")
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}
STRUCTURE_FIELDS = [
    "dataset_name",
    "mutation_record_id",
    "uniprot_id",
    "protein_id",
    "pdb_id",
    "chain_id",
    "mutation",
    "wild_type_aa",
    "position",
    "mutant_aa",
    "raw_structure_path",
    "chain_structure_path",
    "structure_download_status",
    "chain_prepare_status",
    "mutation_site_found",
    "structure_wt_aa",
    "wt_residue_matches_mutation",
    "missing_residue_near_mutation",
    "structure_quality_flags",
    "error_message",
]
SEQUENCE_FIELDS = [
    "dataset_name",
    "sequence_id",
    "uniprot_id",
    "protein_id",
    "pdb_id",
    "chain_id",
    "sequence_source",
    "sequence",
    "sequence_length",
    "fasta_path",
    "n_mutations",
    "quality_flags",
    "notes",
]
ERROR_FIELDS = [
    "dataset_name",
    "mutation_record_id",
    "pdb_id",
    "chain_id",
    "stage",
    "error_type",
    "error_message",
]
MAPPING_FIELDS = [
    "pdb_id",
    "chain_id",
    "original_residue_id",
    "original_insertion_code",
    "output_residue_id",
    "residue_name_3letter",
    "residue_name_1letter",
]


@dataclass(frozen=True)
class Paths:
    root: Path
    mutations_csv: Path
    recalc_jobs_csv: Path
    recalc_jobs_with_inputs_csv: Path
    raw_structures_dir: Path
    clean_structures_dir: Path
    chain_structures_dir: Path
    sequences_dir: Path
    manifests_dir: Path
    reports_dir: Path
    structure_download_log_dir: Path
    structure_preparation_log_dir: Path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return data


def project_root_from(config_path: Path) -> Path:
    return config_path.parent.parent.resolve()


def dataset_config(config: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict) or dataset_name not in datasets:
        raise SystemExit(f"Unknown dataset in configs/datasets.yaml: {dataset_name}")
    return datasets[dataset_name]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_paths(root: Path, dataset_name: str, dataset: dict[str, Any]) -> Paths:
    processed_dir = root / dataset["processed_dir"]
    return Paths(
        root=root,
        mutations_csv=processed_dir / f"{dataset_name}_mutations.csv",
        recalc_jobs_csv=processed_dir / f"{dataset_name}_recalc_jobs.csv",
        recalc_jobs_with_inputs_csv=processed_dir / f"{dataset_name}_recalc_jobs_with_inputs.csv",
        raw_structures_dir=root / "data" / "structures" / "raw" / dataset_name,
        clean_structures_dir=root / "data" / "structures" / "clean" / dataset_name,
        chain_structures_dir=root / "data" / "structures" / "chain" / dataset_name,
        sequences_dir=root / "data" / "sequences" / dataset_name,
        manifests_dir=root / "data" / "manifests",
        reports_dir=root / dataset["reports_dir"],
        structure_download_log_dir=root / "logs" / "structure_downloads" / dataset_name,
        structure_preparation_log_dir=root / "logs" / "structure_preparation" / dataset_name,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], force: bool = False) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing output without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_text(path: Path, text: str, force: bool = False) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing output without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fail_if_outputs_exist(paths: list[Path], root: Path, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if force or not existing:
        return
    formatted = "\n".join(f"  - {relpath(path, root)}" for path in existing)
    raise SystemExit(
        "Refusing to overwrite existing derived outputs without --force:\n"
        f"{formatted}\n\n"
        "Rerun with --force when you intentionally want to regenerate these outputs."
    )


def output_paths_for_mode(paths: Paths, *, prepare_structures: bool, prepare_sequences: bool, full_pipeline: bool) -> list[Path]:
    outputs: list[Path] = []
    if prepare_structures:
        outputs.extend(
            [
                paths.manifests_dir / "current_internal_structure_manifest.csv",
                paths.manifests_dir / "current_internal_structure_errors.csv",
                paths.manifests_dir / "current_internal_residue_mapping.csv",
            ]
        )
    if prepare_sequences:
        outputs.extend(
            [
                paths.manifests_dir / "current_internal_sequence_manifest.csv",
                paths.manifests_dir / "current_internal_sequence_errors.csv",
                paths.sequences_dir / "current_internal_pdb_chain_sequences.fasta",
                paths.sequences_dir / "current_internal_openfold_inputs.fasta",
                paths.sequences_dir / "current_internal_esm2_inputs.fasta",
            ]
        )
    if full_pipeline:
        outputs.extend(
            [
                paths.recalc_jobs_with_inputs_csv,
                paths.reports_dir / "current_internal_structure_sequence_qc.json",
                paths.reports_dir / "current_internal_structure_sequence_qc.md",
            ]
        )
    return outputs


def print_mode_plan(root: Path, outputs: list[Path], untouched: list[Path]) -> None:
    print("Outputs that may be changed:")
    if outputs:
        for path in outputs:
            print(f"  - {relpath(path, root)}")
    else:
        print("  - none of the protected derived manifests/jobs/reports")
    print("Outputs that will not be changed:")
    if untouched:
        for path in untouched:
            print(f"  - {relpath(path, root)}")
    else:
        print("  - none")


def canonical_pdb_id(value: str) -> str:
    return value.strip().upper()


def chain_file_name(pdb_id: str, chain_id: str) -> str:
    safe_chain = chain_id.replace("/", "_").replace(" ", "_")
    return f"{canonical_pdb_id(pdb_id)}_{safe_chain}.pdb"


def raw_cif_path(paths: Paths, pdb_id: str) -> Path:
    return paths.raw_structures_dir / f"{canonical_pdb_id(pdb_id)}.cif"


def chain_path(paths: Paths, pdb_id: str, chain_id: str) -> Path:
    return paths.chain_structures_dir / chain_file_name(pdb_id, chain_id)


def download_one(pdb_id: str, paths: Paths, force: bool, timeout: int = 60) -> dict[str, Any]:
    pdb_id = canonical_pdb_id(pdb_id)
    output = raw_cif_path(paths, pdb_id)
    part = output.with_suffix(output.suffix + ".part")
    if output.exists() and not force:
        return {"pdb_id": pdb_id, "status": "already_existing", "path": relpath(output, paths.root), "error": ""}
    output.parent.mkdir(parents=True, exist_ok=True)
    if part.exists():
        part.unlink()
    url = PDB_URL.format(pdb_id=pdb_id)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "OpenFold3-current-internal-prep/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as handle:
            handle.write(response.read())
        part.replace(output)
        return {"pdb_id": pdb_id, "status": "downloaded", "path": relpath(output, paths.root), "error": ""}
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        if part.exists():
            part.unlink()
        return {"pdb_id": pdb_id, "status": "failed", "path": relpath(output, paths.root), "error": str(exc)}


def tokenize_cif_line(line: str) -> list[str]:
    tokens: list[str] = []
    token = ""
    quote = ""
    i = 0
    while i < len(line):
        char = line[i]
        if quote:
            if char == quote:
                tokens.append(token)
                token = ""
                quote = ""
            else:
                token += char
        elif char in {"'", '"'}:
            quote = char
        elif char.isspace():
            if token:
                tokens.append(token)
                token = ""
        else:
            token += char
        i += 1
    if token:
        tokens.append(token)
    return tokens


def parse_atom_site(cif_path: Path) -> list[dict[str, str]]:
    lines = cif_path.read_text(encoding="utf-8", errors="replace").splitlines()
    atoms: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        tags: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            tags.append(lines[i].strip())
            i += 1
        if not tags or not all(tag.startswith("_atom_site.") for tag in tags):
            continue
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped == "#" or stripped == "loop_" or stripped.startswith("_"):
                break
            tokens = tokenize_cif_line(lines[i])
            if len(tokens) >= len(tags):
                atoms.append({tag.split(".", 1)[1]: tokens[index] for index, tag in enumerate(tags)})
            i += 1
        break
    return atoms


def atom_value(atom: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = atom.get(key, "")
        if value not in {"", ".", "?"}:
            return value
    return ""


def is_protein_atom(atom: dict[str, str]) -> bool:
    group = atom_value(atom, "group_PDB")
    comp = atom_value(atom, "label_comp_id", "auth_comp_id").upper()
    return group == "ATOM" and comp in AA3_TO_1


def chain_id_for_atom(atom: dict[str, str]) -> str:
    return atom_value(atom, "auth_asym_id", "label_asym_id")


def residue_id_for_atom(atom: dict[str, str]) -> str:
    return atom_value(atom, "auth_seq_id", "label_seq_id")


def insertion_code_for_atom(atom: dict[str, str]) -> str:
    value = atom_value(atom, "pdbx_PDB_ins_code")
    return "" if value in {".", "?"} else value


def first_model_atom(atom: dict[str, str]) -> bool:
    model = atom_value(atom, "pdbx_PDB_model_num")
    return model in {"", "1"}


def pdb_atom_line(serial: int, atom: dict[str, str], chain_id: str) -> str:
    atom_name = atom_value(atom, "auth_atom_id", "label_atom_id")[:4]
    comp = atom_value(atom, "auth_comp_id", "label_comp_id").upper()[:3]
    res_id_text = residue_id_for_atom(atom)
    try:
        res_id = int(float(res_id_text))
    except ValueError:
        res_id = 0
    ins = insertion_code_for_atom(atom)[:1]
    x = float(atom_value(atom, "Cartn_x") or 0.0)
    y = float(atom_value(atom, "Cartn_y") or 0.0)
    z = float(atom_value(atom, "Cartn_z") or 0.0)
    occupancy = float(atom_value(atom, "occupancy") or 1.0)
    b_factor = float(atom_value(atom, "B_iso_or_equiv") or 0.0)
    element = atom_value(atom, "type_symbol")[:2].rjust(2)
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {comp:>3s} {chain_id[:1]:1s}"
        f"{res_id:4d}{ins:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}          {element}\n"
    )


def prepare_chain(pdb_id: str, chain_id: str, paths: Paths, force: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pdb_id = canonical_pdb_id(pdb_id)
    raw_path = raw_cif_path(paths, pdb_id)
    out_path = chain_path(paths, pdb_id, chain_id)
    if not raw_path.exists():
        return (
            {"status": "failed", "error": "raw_structure_missing", "chain_path": relpath(out_path, paths.root), "sequence": ""},
            [],
            [],
        )
    if out_path.exists() and not force:
        try:
            residues = parse_chain_pdb(out_path, chain_id)
            if not residues:
                raise ValueError("no readable protein residues in existing chain PDB")
            sequence = "".join(item["aa1"] for item in residues)
            return ({"status": "already_existing", "error": "", "chain_path": relpath(out_path, paths.root), "sequence": sequence}, residues, [])
        except Exception as exc:
            return ({"status": "failed", "error": f"chain_file_validation_failed: {exc}", "chain_path": relpath(out_path, paths.root), "sequence": ""}, [], [])
    try:
        atoms = parse_atom_site(raw_path)
        selected = [atom for atom in atoms if first_model_atom(atom) and is_protein_atom(atom) and chain_id_for_atom(atom) == chain_id]
        if not selected:
            return ({"status": "failed", "error": "chain_not_found", "chain_path": relpath(out_path, paths.root), "sequence": ""}, [], [])
        residues = residues_for_chain(selected, chain_id)
        sequence = "".join(item["aa1"] for item in residues)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for serial, atom in enumerate(selected, start=1):
                handle.write(pdb_atom_line(serial, atom, chain_id))
            handle.write("END\n")
        return ({"status": "prepared", "error": "", "chain_path": relpath(out_path, paths.root), "sequence": sequence}, residues, [])
    except Exception as exc:
        return ({"status": "failed", "error": f"structure_parse_failed: {exc}", "chain_path": relpath(out_path, paths.root), "sequence": ""}, [], [])


def parse_chain_pdb(pdb_path: Path, chain_id: str) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    residues: list[dict[str, Any]] = []
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 27:
                continue
            if line[21].strip() != chain_id[:1]:
                continue
            res_id = line[22:26].strip()
            ins = line[26].strip()
            key = (res_id, ins)
            if key in seen:
                continue
            seen.add(key)
            comp = line[17:20].strip().upper()
            aa1 = AA3_TO_1.get(comp, "X")
            residues.append(
                {
                    "original_residue_id": res_id,
                    "original_insertion_code": ins,
                    "output_residue_id": res_id,
                    "residue_name_3letter": comp,
                    "residue_name_1letter": aa1,
                    "aa1": aa1,
                }
            )
    return residues


def load_existing_chain(pdb_id: str, chain_id: str, paths: Paths) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pdb_id = canonical_pdb_id(pdb_id)
    out_path = chain_path(paths, pdb_id, chain_id)
    if not out_path.exists():
        return (
            {"status": "failed", "error": "chain_file_missing", "chain_path": relpath(out_path, paths.root), "sequence": ""},
            [],
        )
    try:
        residues = parse_chain_pdb(out_path, chain_id)
        if not residues:
            raise ValueError("no readable protein residues in existing chain PDB")
        sequence = "".join(item["aa1"] for item in residues)
        return ({"status": "already_existing", "error": "", "chain_path": relpath(out_path, paths.root), "sequence": sequence}, residues)
    except Exception as exc:
        return (
            {"status": "failed", "error": f"chain_file_validation_failed: {exc}", "chain_path": relpath(out_path, paths.root), "sequence": ""},
            [],
        )


def residues_for_chain(atoms: list[dict[str, str]], chain_id: str) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    residues: list[dict[str, Any]] = []
    for atom in atoms:
        if chain_id_for_atom(atom) != chain_id or not first_model_atom(atom) or not is_protein_atom(atom):
            continue
        res_id = residue_id_for_atom(atom)
        ins = insertion_code_for_atom(atom)
        key = (res_id, ins)
        if key in seen:
            continue
        seen.add(key)
        comp = atom_value(atom, "auth_comp_id", "label_comp_id").upper()
        residues.append(
            {
                "original_residue_id": res_id,
                "original_insertion_code": ins,
                "output_residue_id": res_id,
                "residue_name_3letter": comp,
                "residue_name_1letter": AA3_TO_1.get(comp, "X"),
                "aa1": AA3_TO_1.get(comp, "X"),
            }
        )
    return residues


def residue_at_position(residues: list[dict[str, Any]], position: str) -> tuple[dict[str, Any] | None, bool]:
    matches = [item for item in residues if str(item["original_residue_id"]) == str(position)]
    if len(matches) == 1:
        return matches[0], False
    if len(matches) > 1:
        no_ins = [item for item in matches if not item["original_insertion_code"]]
        return (no_ins[0] if no_ins else matches[0]), True
    return None, False


def has_neighbor_gap(residues: list[dict[str, Any]], position: str) -> bool:
    try:
        pos = int(position)
    except ValueError:
        return False
    ids: set[int] = set()
    for item in residues:
        try:
            ids.add(int(str(item["original_residue_id"])))
        except ValueError:
            pass
    return (pos - 1 not in ids) or (pos + 1 not in ids)


def write_fasta(path: Path, records: list[tuple[str, str]], force: bool = False) -> None:
    lines: list[str] = []
    for name, sequence in records:
        lines.append(f">{name}")
        for index in range(0, len(sequence), 80):
            lines.append(sequence[index : index + 80])
    write_text(path, "\n".join(lines) + ("\n" if lines else ""), force=force)


def write_download_logs(paths: Paths, download_results: dict[str, dict[str, Any]]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_lines = [
        "current_internal structure download log",
        f"timestamp: {timestamp}",
        f"total_pdb_ids: {len(download_results)}",
        f"downloaded: {sum(1 for row in download_results.values() if row['status'] == 'downloaded')}",
        f"already_existing: {sum(1 for row in download_results.values() if row['status'] == 'already_existing')}",
        f"failed: {sum(1 for row in download_results.values() if row['status'] == 'failed')}",
        "",
    ]
    for pdb_id, result in sorted(download_results.items()):
        download_lines.append(f"{pdb_id}\t{result['status']}\t{result['path']}\t{result['error']}")
    write_text(paths.structure_download_log_dir / f"{timestamp}_download.log", "\n".join(download_lines) + "\n", force=True)
    write_text(paths.structure_download_log_dir / "latest_download.log", "\n".join(download_lines) + "\n", force=True)


def write_run_logs(paths: Paths, qc: dict[str, Any], download_results: dict[str, dict[str, Any]], chain_results: dict[tuple[str, str], dict[str, Any]]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_lines = [
        "current_internal structure download log",
        f"timestamp: {timestamp}",
        f"total_pdb_ids: {qc['structure_download']['total_pdb_ids']}",
        f"downloaded: {qc['structure_download']['downloaded']}",
        f"already_existing: {qc['structure_download']['already_existing']}",
        f"failed: {qc['structure_download']['failed']}",
        "",
    ]
    for pdb_id, result in sorted(download_results.items()):
        download_lines.append(f"{pdb_id}\t{result['status']}\t{result['path']}\t{result['error']}")
    preparation_lines = [
        "current_internal structure/sequence preparation log",
        f"timestamp: {timestamp}",
        f"total_pdb_chains: {qc['chain_preparation']['total_pdb_chains']}",
        f"prepared: {qc['chain_preparation']['prepared']}",
        f"already_existing: {qc['chain_preparation']['already_existing']}",
        f"failed: {qc['chain_preparation']['failed']}",
        f"unique_sequences: {qc['sequence_preparation']['unique_sequences']}",
        f"input_ready_jobs: {qc['recalc_jobs_readiness']['input_ready_jobs']}",
        "",
    ]
    for (pdb_id, chain_id), result in sorted(chain_results.items()):
        preparation_lines.append(f"{pdb_id}_{chain_id}\t{result['status']}\t{result['chain_path']}\t{result['error']}")
    write_text(paths.structure_download_log_dir / f"{timestamp}_download.log", "\n".join(download_lines) + "\n", force=True)
    write_text(paths.structure_download_log_dir / "latest_download.log", "\n".join(download_lines) + "\n", force=True)
    write_text(paths.structure_preparation_log_dir / f"{timestamp}_prepare.log", "\n".join(preparation_lines) + "\n", force=True)
    write_text(paths.structure_preparation_log_dir / "latest_prepare.log", "\n".join(preparation_lines) + "\n", force=True)


def build_sequence_id(row: dict[str, str]) -> str:
    uniprot = row.get("uniprot_id") or "unknown_uniprot"
    return f"current_internal__{uniprot}__{canonical_pdb_id(row['pdb_id'])}_{row['chain_id']}"


def summarize_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows if row.get(key, "") != ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-structures", action="store_true")
    parser.add_argument("--prepare-structures", action="store_true")
    parser.add_argument("--prepare-sequences", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--mutation-record-id")
    parser.add_argument("--pdb-id")
    args = parser.parse_args()

    if args.dataset != "current_internal":
        raise SystemExit("This script currently supports only --dataset current_internal")
    if args.all:
        args.download_structures = True
        args.prepare_structures = True
        args.prepare_sequences = True
    full_pipeline = bool(args.download_structures and args.prepare_structures and args.prepare_sequences)
    if not (args.download_structures or args.prepare_structures or args.prepare_sequences or args.dry_run):
        raise SystemExit("Choose --dry-run, --download-structures, --prepare-structures, --prepare-sequences, or --all")

    root = project_root_from(args.datasets_config)
    config = load_yaml(args.datasets_config)
    dataset = dataset_config(config, args.dataset)
    paths = build_paths(root, args.dataset, dataset)
    for required in (paths.mutations_csv, paths.recalc_jobs_csv):
        if not required.exists():
            raise SystemExit(f"Missing required input: {relpath(required, root)}")

    mutations = read_csv(paths.mutations_csv)
    recalc_jobs = read_csv(paths.recalc_jobs_csv)
    if args.mutation_record_id:
        mutations = [row for row in mutations if row["mutation_record_id"] == args.mutation_record_id]
    if args.pdb_id:
        mutations = [row for row in mutations if canonical_pdb_id(row["pdb_id"]) == canonical_pdb_id(args.pdb_id)]
    unique_pdb_ids = sorted({canonical_pdb_id(row["pdb_id"]) for row in mutations if row.get("pdb_id")})
    if args.max_structures is not None:
        unique_pdb_ids = unique_pdb_ids[: args.max_structures]
        mutations = [row for row in mutations if canonical_pdb_id(row["pdb_id"]) in set(unique_pdb_ids)]
    unique_chains = sorted({(canonical_pdb_id(row["pdb_id"]), row["chain_id"]) for row in mutations if row.get("pdb_id") and row.get("chain_id")})
    protected_outputs = output_paths_for_mode(
        paths,
        prepare_structures=bool(args.prepare_structures),
        prepare_sequences=bool(args.prepare_sequences),
        full_pipeline=full_pipeline,
    )
    all_protected_outputs = output_paths_for_mode(paths, prepare_structures=True, prepare_sequences=True, full_pipeline=True)
    untouched_outputs = [path for path in all_protected_outputs if path not in set(protected_outputs)]

    print(f"Dataset: {args.dataset}")
    print(f"Mutations input: {relpath(paths.mutations_csv, root)}")
    print(f"Recalc jobs input: {relpath(paths.recalc_jobs_csv, root)}")
    print(f"Mutation records selected: {len(mutations)}")
    print(f"Unique PDB IDs: {len(unique_pdb_ids)}")
    print(f"Unique PDB chains: {len(unique_chains)}")
    print_mode_plan(root, protected_outputs, untouched_outputs)

    if args.dry_run:
        print("Dry run PDB IDs:")
        for pdb_id in unique_pdb_ids:
            print(f"  {pdb_id}")
        print("Dry run: no structures, sequences, manifests, or jobs-with-inputs were written.")
        return 0
    fail_if_outputs_exist(protected_outputs, root, args.force)

    for directory in (
        paths.raw_structures_dir,
        paths.clean_structures_dir,
        paths.chain_structures_dir,
        paths.sequences_dir,
        paths.manifests_dir,
        paths.reports_dir,
        paths.structure_download_log_dir,
        paths.structure_preparation_log_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    download_results: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    if args.download_structures:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(download_one, pdb_id, paths, args.force): pdb_id for pdb_id in unique_pdb_ids}
            for future in as_completed(futures):
                result = future.result()
                download_results[result["pdb_id"]] = result
                if result["status"] == "failed":
                    errors.append({"dataset_name": args.dataset, "mutation_record_id": "", "pdb_id": result["pdb_id"], "chain_id": "", "stage": "download", "error_type": "structure_download_failed", "error_message": result["error"]})
                if not args.no_progress_bar:
                    print(f"{result['pdb_id']}: {result['status']}")
    else:
        for pdb_id in unique_pdb_ids:
            path = raw_cif_path(paths, pdb_id)
            download_results[pdb_id] = {"pdb_id": pdb_id, "status": "already_existing" if path.exists() else "not_requested", "path": relpath(path, root), "error": ""}

    if args.download_structures and not (args.prepare_structures or args.prepare_sequences):
        write_download_logs(paths, download_results)
        downloaded = sum(1 for row in download_results.values() if row["status"] == "downloaded")
        already_existing = sum(1 for row in download_results.values() if row["status"] == "already_existing")
        failed = sum(1 for row in download_results.values() if row["status"] == "failed")
        print("Download-only mode complete.")
        print(f"Downloaded: {downloaded}, already existing: {already_existing}, failed: {failed}")
        print("No structure/sequence manifests or jobs-with-inputs were modified.")
        return 1 if failed else 0

    chain_results: dict[tuple[str, str], dict[str, Any]] = {}
    residue_maps: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if args.prepare_structures:
        for pdb_id, chain_id in unique_chains:
            result, residues, _ = prepare_chain(pdb_id, chain_id, paths, args.force)
            chain_results[(pdb_id, chain_id)] = result
            residue_maps[(pdb_id, chain_id)] = residues
            if result["status"] == "failed":
                errors.append({"dataset_name": args.dataset, "mutation_record_id": "", "pdb_id": pdb_id, "chain_id": chain_id, "stage": "prepare_chain", "error_type": result["error"].split(":", 1)[0], "error_message": result["error"]})
            if not args.no_progress_bar:
                print(f"{pdb_id}_{chain_id}: {result['status']}")
    elif args.prepare_sequences:
        for pdb_id, chain_id in unique_chains:
            result, residues = load_existing_chain(pdb_id, chain_id, paths)
            chain_results[(pdb_id, chain_id)] = result
            residue_maps[(pdb_id, chain_id)] = residues
            if result["status"] == "failed":
                errors.append({"dataset_name": args.dataset, "mutation_record_id": "", "pdb_id": pdb_id, "chain_id": chain_id, "stage": "prepare_sequence", "error_type": result["error"].split(":", 1)[0], "error_message": result["error"]})
            if not args.no_progress_bar:
                print(f"{pdb_id}_{chain_id}: {result['status']}")
    else:
        for pdb_id, chain_id in unique_chains:
            path = chain_path(paths, pdb_id, chain_id)
            chain_results[(pdb_id, chain_id)] = {"status": "already_existing" if path.exists() else "not_requested", "error": "", "chain_path": relpath(path, root), "sequence": ""}
            residue_maps[(pdb_id, chain_id)] = []

    structure_manifest: list[dict[str, Any]] = []
    sequence_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in mutations:
        pdb_id = canonical_pdb_id(row.get("pdb_id", ""))
        chain_id = row.get("chain_id", "")
        key = (pdb_id, chain_id)
        download = download_results.get(pdb_id, {"status": "not_requested", "path": relpath(raw_cif_path(paths, pdb_id), root), "error": ""})
        chain = chain_results.get(key, {"status": "not_requested", "error": "", "chain_path": relpath(chain_path(paths, pdb_id, chain_id), root), "sequence": ""})
        residues = residue_maps.get(key, [])
        flags: list[str] = []
        error_messages: list[str] = []
        if not pdb_id:
            flags.append("missing_pdb_id")
        if not chain_id:
            flags.append("missing_chain_id")
        if download["status"] == "failed":
            flags.append("structure_download_failed")
            error_messages.append(download["error"])
        if chain["status"] == "failed":
            err = chain["error"]
            if err == "chain_not_found":
                flags.append("chain_not_found")
            elif err == "chain_file_missing":
                flags.append("chain_file_missing")
            elif err.startswith("chain_file_validation_failed"):
                flags.append("chain_file_validation_failed")
            else:
                flags.append("structure_parse_failed")
            error_messages.append(err)
        residue, insertion_ambiguous = residue_at_position(residues, row.get("position", ""))
        mutation_site_found = residue is not None
        structure_wt = residue["residue_name_1letter"] if residue else ""
        wt_match = bool(residue and structure_wt == row.get("wild_type_aa"))
        if insertion_ambiguous:
            flags.append("insertion_code_ambiguous")
        if residue and structure_wt == "X":
            flags.append("noncanonical_residue")
        if not mutation_site_found:
            flags.append("mutation_site_not_found")
        elif not wt_match:
            flags.append("wt_residue_mismatch")
        missing_near = has_neighbor_gap(residues, row.get("position", "")) if residue else False
        if missing_near:
            flags.append("missing_residue_near_mutation")
        if flags:
            errors.append({"dataset_name": args.dataset, "mutation_record_id": row["mutation_record_id"], "pdb_id": pdb_id, "chain_id": chain_id, "stage": "mutation_site_validation", "error_type": ";".join(flags), "error_message": "; ".join(error_messages)})
        structure_manifest.append(
            {
                "dataset_name": args.dataset,
                "mutation_record_id": row["mutation_record_id"],
                "uniprot_id": row.get("uniprot_id", ""),
                "protein_id": row.get("protein_id", ""),
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "mutation": row.get("mutation", ""),
                "wild_type_aa": row.get("wild_type_aa", ""),
                "position": row.get("position", ""),
                "mutant_aa": row.get("mutant_aa", ""),
                "raw_structure_path": download["path"],
                "chain_structure_path": chain["chain_path"],
                "structure_download_status": download["status"],
                "chain_prepare_status": chain["status"],
                "mutation_site_found": str(mutation_site_found).lower(),
                "structure_wt_aa": structure_wt,
                "wt_residue_matches_mutation": str(wt_match).lower(),
                "missing_residue_near_mutation": str(missing_near).lower(),
                "structure_quality_flags": ";".join(sorted(set(flags))),
                "error_message": "; ".join(error_messages),
            }
        )
        if chain.get("sequence"):
            sequence_groups.setdefault(
                key,
                {
                    "dataset_name": args.dataset,
                    "sequence_id": build_sequence_id(row),
                    "uniprot_id": row.get("uniprot_id", ""),
                    "protein_id": row.get("protein_id", ""),
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "sequence_source": "pdb_chain",
                    "sequence": chain["sequence"],
                    "mutation_ids": [],
                    "quality_flags": "",
                    "notes": "PDB chain sequence; may differ from full-length UniProt sequence.",
                },
            )["mutation_ids"].append(row["mutation_record_id"])

    sequence_rows: list[dict[str, Any]] = []
    for item in sequence_groups.values():
        sequence_rows.append(
            {
                "dataset_name": item["dataset_name"],
                "sequence_id": item["sequence_id"],
                "uniprot_id": item["uniprot_id"],
                "protein_id": item["protein_id"],
                "pdb_id": item["pdb_id"],
                "chain_id": item["chain_id"],
                "sequence_source": item["sequence_source"],
                "sequence": item["sequence"],
                "sequence_length": len(item["sequence"]),
                "fasta_path": relpath(paths.sequences_dir / "current_internal_pdb_chain_sequences.fasta", root),
                "n_mutations": len(item["mutation_ids"]),
                "quality_flags": item["quality_flags"],
                "notes": item["notes"],
            }
        )

    sequence_records = [(row["sequence_id"], row["sequence"]) for row in sequence_rows]
    if args.prepare_sequences:
        write_fasta(paths.sequences_dir / "current_internal_pdb_chain_sequences.fasta", sequence_records, force=args.force)
        write_fasta(paths.sequences_dir / "current_internal_openfold_inputs.fasta", sequence_records, force=args.force)
        write_fasta(paths.sequences_dir / "current_internal_esm2_inputs.fasta", sequence_records, force=args.force)

    mapping_rows: list[dict[str, Any]] = []
    for (pdb_id, chain_id), residues in residue_maps.items():
        for residue in residues:
            mapping_rows.append({"pdb_id": pdb_id, "chain_id": chain_id, **{key: residue.get(key, "") for key in MAPPING_FIELDS if key not in {"pdb_id", "chain_id"}}})

    structure_by_mutation = {row["mutation_record_id"]: row for row in structure_manifest}
    sequence_by_chain = {(row["pdb_id"], row["chain_id"]): row for row in sequence_rows}
    jobs_with_inputs: list[dict[str, Any]] = []
    for job in recalc_jobs:
        mutation_id = job["mutation_record_id"]
        sm = structure_by_mutation.get(mutation_id, {})
        key = (canonical_pdb_id(job.get("pdb_id", "")), job.get("chain_id", ""))
        seq = sequence_by_chain.get(key)
        method = job["method"]
        structure_source = job["structure_source"]
        input_ready = False
        input_status = "blocked"
        blocking_reason = ""
        experimental_structure_path = ""
        openfold_input_fasta_path = ""
        sequence_input_fasta_path = ""
        if method in {"foldx", "rosetta"} and structure_source == "experimental":
            experimental_structure_path = sm.get("chain_structure_path", "")
            input_ready = sm.get("chain_prepare_status") in {"prepared", "already_existing"} and sm.get("mutation_site_found") == "true" and sm.get("wt_residue_matches_mutation") == "true"
            input_status = "ready" if input_ready else "blocked_for_structure"
            if not input_ready:
                blocking_reason = sm.get("structure_quality_flags") or sm.get("error_message") or "missing_valid_experimental_structure"
        elif method in {"foldx", "rosetta"} and structure_source == "openfold":
            openfold_input_fasta_path = relpath(paths.sequences_dir / "current_internal_openfold_inputs.fasta", root) if seq else ""
            input_status = "pending_openfold_structure" if seq else "blocked_missing_sequence"
            blocking_reason = "" if seq else "missing_sequence"
        elif method == "esm2" and structure_source == "sequence_only":
            sequence_input_fasta_path = relpath(paths.sequences_dir / "current_internal_esm2_inputs.fasta", root) if seq else ""
            input_ready = bool(seq)
            input_status = "ready" if input_ready else "blocked_missing_sequence"
            blocking_reason = "" if input_ready else "missing_sequence"
        jobs_with_inputs.append(
            {
                **job,
                "input_ready": str(input_ready).lower(),
                "input_status": input_status,
                "experimental_structure_path": experimental_structure_path,
                "openfold_input_fasta_path": openfold_input_fasta_path,
                "sequence_input_fasta_path": sequence_input_fasta_path,
                "blocking_reason": blocking_reason,
            }
        )

    structure_manifest_path = paths.manifests_dir / "current_internal_structure_manifest.csv"
    sequence_manifest_path = paths.manifests_dir / "current_internal_sequence_manifest.csv"
    structure_errors_path = paths.manifests_dir / "current_internal_structure_errors.csv"
    sequence_errors_path = paths.manifests_dir / "current_internal_sequence_errors.csv"
    residue_mapping_path = paths.manifests_dir / "current_internal_residue_mapping.csv"
    if args.prepare_structures:
        write_csv(structure_manifest_path, structure_manifest, STRUCTURE_FIELDS, force=args.force)
        write_csv(structure_errors_path, errors, ERROR_FIELDS, force=args.force)
        write_csv(residue_mapping_path, mapping_rows, MAPPING_FIELDS, force=args.force)
    if args.prepare_sequences:
        write_csv(sequence_manifest_path, sequence_rows, SEQUENCE_FIELDS, force=args.force)
        write_csv(sequence_errors_path, [row for row in errors if "sequence" in row["error_type"]], ERROR_FIELDS, force=args.force)
    if full_pipeline:
        write_csv(
            paths.recalc_jobs_with_inputs_csv,
            jobs_with_inputs,
            list(recalc_jobs[0].keys()) + ["input_ready", "input_status", "experimental_structure_path", "openfold_input_fasta_path", "sequence_input_fasta_path", "blocking_reason"] if recalc_jobs else [],
            force=args.force,
        )

    if not full_pipeline:
        print("Partial preparation complete.")
        print("Jobs-with-inputs and structure/sequence QC reports were not modified.")
        return 1 if any(row["status"] == "failed" for row in chain_results.values()) else 0

    lengths = [int(row["sequence_length"]) for row in sequence_rows]
    qc = {
        "input_summary": {
            "mutations_input_path": relpath(paths.mutations_csv, root),
            "recalc_jobs_input_path": relpath(paths.recalc_jobs_csv, root),
            "mutation_records": len(mutations),
            "unique_pdb_ids": len(unique_pdb_ids),
            "unique_pdb_chains": len(unique_chains),
        },
        "structure_download": {
            "total_pdb_ids": len(unique_pdb_ids),
            "downloaded": sum(1 for row in download_results.values() if row["status"] == "downloaded"),
            "already_existing": sum(1 for row in download_results.values() if row["status"] == "already_existing"),
            "failed": sum(1 for row in download_results.values() if row["status"] == "failed"),
        },
        "chain_preparation": {
            "total_pdb_chains": len(unique_chains),
            "prepared": sum(1 for row in chain_results.values() if row["status"] == "prepared"),
            "already_existing": sum(1 for row in chain_results.values() if row["status"] == "already_existing"),
            "failed": sum(1 for row in chain_results.values() if row["status"] == "failed"),
            "top_failure_reasons": Counter(row["error"] for row in chain_results.values() if row["error"]).most_common(10),
        },
        "mutation_site_validation": {
            "total_mutations": len(structure_manifest),
            "mutation_site_found": sum(1 for row in structure_manifest if row["mutation_site_found"] == "true"),
            "wt_residue_matches": sum(1 for row in structure_manifest if row["wt_residue_matches_mutation"] == "true"),
            "wt_residue_mismatch": sum(1 for row in structure_manifest if "wt_residue_mismatch" in row["structure_quality_flags"]),
            "mutation_site_not_found": sum(1 for row in structure_manifest if "mutation_site_not_found" in row["structure_quality_flags"]),
            "blocked_mutations": sum(1 for row in structure_manifest if row["structure_quality_flags"]),
        },
        "sequence_preparation": {
            "unique_sequences": len(sequence_rows),
            "sequence_length_min": min(lengths) if lengths else None,
            "sequence_length_median": statistics.median(lengths) if lengths else None,
            "sequence_length_max": max(lengths) if lengths else None,
            "sequences_prepared_for_esm2": len(sequence_rows),
            "sequences_prepared_for_openfold": len(sequence_rows),
        },
        "recalc_jobs_readiness": {
            "total_jobs": len(jobs_with_inputs),
            "input_ready_jobs": sum(1 for row in jobs_with_inputs if row["input_ready"] == "true"),
            "blocked_jobs": sum(1 for row in jobs_with_inputs if row["input_ready"] != "true"),
            "jobs_by_method_and_input_status": dict(Counter(f"{row['method']}::{row['input_status']}" for row in jobs_with_inputs)),
            "top_blocking_reasons": Counter(row["blocking_reason"] for row in jobs_with_inputs if row["blocking_reason"]).most_common(10),
        },
        "warnings": [
            "PDB chain sequence may differ from UniProt full-length sequence.",
            "OpenFold on full-length UniProt and FoldX/Rosetta on PDB chain are not identical representations unless explicitly mapped.",
            "WT residue mismatches must be manually audited before scoring.",
        ],
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    qc_json_path = paths.reports_dir / "current_internal_structure_sequence_qc.json"
    qc_md_path = paths.reports_dir / "current_internal_structure_sequence_qc.md"
    write_text(qc_json_path, json.dumps(qc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", force=args.force)
    md = [
        "# Current Internal Structure/Sequence QC",
        "",
        f"- Mutation records: {qc['input_summary']['mutation_records']}",
        f"- Unique PDB IDs: {qc['input_summary']['unique_pdb_ids']}",
        f"- Unique PDB chains: {qc['input_summary']['unique_pdb_chains']}",
        "",
        "## Structure Download",
        f"- Downloaded: {qc['structure_download']['downloaded']}",
        f"- Already existing: {qc['structure_download']['already_existing']}",
        f"- Failed: {qc['structure_download']['failed']}",
        "",
        "## Chain Preparation",
        f"- Prepared: {qc['chain_preparation']['prepared']}",
        f"- Already existing: {qc['chain_preparation']['already_existing']}",
        f"- Failed: {qc['chain_preparation']['failed']}",
        f"- Top failure reasons: `{json.dumps(qc['chain_preparation']['top_failure_reasons'])}`",
        "",
        "## Mutation Site Validation",
        f"- Mutation site found: {qc['mutation_site_validation']['mutation_site_found']}",
        f"- WT residue matches: {qc['mutation_site_validation']['wt_residue_matches']}",
        f"- WT residue mismatch: {qc['mutation_site_validation']['wt_residue_mismatch']}",
        f"- Mutation site not found: {qc['mutation_site_validation']['mutation_site_not_found']}",
        f"- Blocked mutations: {qc['mutation_site_validation']['blocked_mutations']}",
        "",
        "## Sequence Preparation",
        f"- Unique sequences: {qc['sequence_preparation']['unique_sequences']}",
        f"- Length min/median/max: {qc['sequence_preparation']['sequence_length_min']} / {qc['sequence_preparation']['sequence_length_median']} / {qc['sequence_preparation']['sequence_length_max']}",
        "",
        "## Recalc Jobs Readiness",
        f"- Total jobs: {qc['recalc_jobs_readiness']['total_jobs']}",
        f"- Input-ready jobs: {qc['recalc_jobs_readiness']['input_ready_jobs']}",
        f"- Blocked jobs: {qc['recalc_jobs_readiness']['blocked_jobs']}",
        f"- Jobs by method/status: `{json.dumps(qc['recalc_jobs_readiness']['jobs_by_method_and_input_status'], sort_keys=True)}`",
        f"- Top blocking reasons: `{json.dumps(qc['recalc_jobs_readiness']['top_blocking_reasons'])}`",
        "",
        "## Warnings",
        *[f"- {warning}" for warning in qc["warnings"]],
        "",
    ]
    write_text(qc_md_path, "\n".join(md), force=args.force)
    write_run_logs(paths, qc, download_results, chain_results)

    print("Preparation complete.")
    print(f"Downloaded: {qc['structure_download']['downloaded']}, already existing: {qc['structure_download']['already_existing']}, failed: {qc['structure_download']['failed']}")
    print(f"Chain structures prepared: {qc['chain_preparation']['prepared']}, failed: {qc['chain_preparation']['failed']}")
    print(f"WT residue matches: {qc['mutation_site_validation']['wt_residue_matches']} / {qc['mutation_site_validation']['total_mutations']}")
    print(f"Input-ready jobs: {qc['recalc_jobs_readiness']['input_ready_jobs']} / {qc['recalc_jobs_readiness']['total_jobs']}")
    return 1 if qc["structure_download"]["failed"] or qc["chain_preparation"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
