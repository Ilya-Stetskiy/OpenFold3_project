"""External ddG benchmark importers.

The FireProt-like dataset stage is the project-local discovery dataset. This
module keeps independent evaluation benchmarks separate so article analyses can
compare within-dataset results against external checks without mixing schemas.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENFOLD_ROOT = REPO_ROOT / "openfold-3"
if str(OPENFOLD_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENFOLD_ROOT))

from openfold3.benchmark.cif_utils import parse_structure_records
from openfold3.benchmark.structure_source import CANONICAL_AA_1, CANONICAL_AA_3_TO_1

LOGGER = logging.getLogger(__name__)

S669_URL = "https://zenodo.org/records/7568094/files/S669.zip?download=1"
S669_MD5 = "fa6de0dbcee83656109a8e685720b94a"
S669_PREPROCESSING_VERSION = "s669-v1.0"
S669_ARCHIVE_NAME = "S669.zip"
S669_EXTRACTED_CSV = Path("S669") / "S669.csv"
S669_OUTPUT_COLUMNS = [
    "benchmark",
    "direction",
    "protein_id",
    "pdb_id",
    "chain",
    "position",
    "pdb_residue_id",
    "wt_residue",
    "mut_residue",
    "experimental_ddg",
    "mutation_id",
    "structure_id",
    "pdb_path",
    "doi",
    "temperature",
    "ph",
    "experimental_method",
    "resolution",
]
S669_CSV_DTYPES = {
    "benchmark": str,
    "direction": str,
    "protein_id": str,
    "pdb_id": str,
    "chain": str,
    "pdb_residue_id": str,
    "wt_residue": str,
    "mut_residue": str,
    "mutation_id": str,
    "structure_id": str,
    "pdb_path": str,
    "doi": str,
    "experimental_method": str,
}
MUTATION_PATTERN = re.compile(
    r"^\s*([ACDEFGHIKLMNPQRSTVWY])\s*(\d+)\s*([ACDEFGHIKLMNPQRSTVWY])\s*$",
    re.I,
)


@dataclass(frozen=True, slots=True)
class S669Config:
    """Configuration for the S669 external benchmark import."""

    data_dir: Path = REPO_ROOT / "data"
    raw_dir: Path | None = None
    processed_path: Path | None = None
    source_url: str = S669_URL
    source_md5: str | None = S669_MD5
    source_archive_name: str = S669_ARCHIVE_NAME
    source_csv: Path | None = None
    include_inverse: bool = True
    force_download: bool = False
    force_preprocess: bool = False

    def resolved(self) -> "S669Config":
        data_dir = Path(self.data_dir).expanduser().resolve()
        raw_dir = Path(self.raw_dir).expanduser().resolve() if self.raw_dir else data_dir / "raw" / "s669"
        processed_path = (
            Path(self.processed_path).expanduser().resolve()
            if self.processed_path
            else data_dir / "processed" / "s669_dataset_v1.csv"
        )
        source_csv = Path(self.source_csv).expanduser().resolve() if self.source_csv else None
        return replace(
            self,
            data_dir=data_dir,
            raw_dir=raw_dir,
            processed_path=processed_path,
            source_csv=source_csv,
        )

    @property
    def metadata_path(self) -> Path:
        resolved = self.resolved()
        assert resolved.processed_path is not None
        return resolved.processed_path.with_name("s669_dataset_v1_meta.json")

    @property
    def discarded_path(self) -> Path:
        resolved = self.resolved()
        assert resolved.processed_path is not None
        return resolved.processed_path.with_name("s669_discarded_rows.csv")


def _coerce_s669_config(config: S669Config | dict[str, Any] | None) -> S669Config:
    if config is None:
        return S669Config().resolved()
    if isinstance(config, S669Config):
        return config.resolved()
    if isinstance(config, dict):
        return S669Config(**config).resolved()
    raise TypeError(f"Unsupported S669 config type: {type(config)!r}")


def check_s669_exists(config: S669Config | dict[str, Any] | None = None) -> bool:
    """Return True when the processed S669 dataset and metadata are valid."""

    cfg = _coerce_s669_config(config)
    assert cfg.processed_path is not None
    if not cfg.processed_path.exists() or not cfg.metadata_path.exists():
        return False
    try:
        header = pd.read_csv(cfg.processed_path, nrows=0)
        metadata = json.loads(cfg.metadata_path.read_text(encoding="utf-8"))
        return (
            list(header.columns) == S669_OUTPUT_COLUMNS
            and metadata.get("preprocessing_version") == S669_PREPROCESSING_VERSION
            and metadata.get("output_columns") == S669_OUTPUT_COLUMNS
            and metadata.get("include_inverse") == cfg.include_inverse
            and metadata.get("dataset_hash") == _sha256(cfg.processed_path)
        )
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def download_s669(config: S669Config | dict[str, Any] | None = None) -> Path:
    """Download and extract the S669 benchmark archive."""

    cfg = _coerce_s669_config(config)
    assert cfg.raw_dir is not None
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    if cfg.source_csv is not None:
        if not cfg.source_csv.exists():
            raise FileNotFoundError(f"Configured source_csv does not exist: {cfg.source_csv}")
        return cfg.source_csv

    archive_path = cfg.raw_dir / cfg.source_archive_name
    extracted_csv = cfg.raw_dir / S669_EXTRACTED_CSV

    if cfg.force_download or not archive_path.exists():
        LOGGER.info("Downloading S669 archive to %s", archive_path)
        urlretrieve(cfg.source_url, archive_path)
    else:
        LOGGER.info("Using cached S669 archive: %s", archive_path)

    if cfg.source_md5 is not None:
        digest = _md5(archive_path)
        if digest != cfg.source_md5:
            raise ValueError(
                f"S669 archive checksum mismatch for {archive_path}: "
                f"expected {cfg.source_md5}, got {digest}"
            )

    if cfg.force_download or not extracted_csv.exists():
        LOGGER.info("Extracting S669 archive into %s", cfg.raw_dir)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(cfg.raw_dir)

    if not extracted_csv.exists():
        raise FileNotFoundError(f"Expected S669 CSV was not found: {extracted_csv}")
    return extracted_csv


def preprocess_s669(config: S669Config | dict[str, Any] | None = None) -> Path:
    """Create a canonical direct/inverse S669 CSV for independent ddG checks."""

    cfg = _coerce_s669_config(config)
    assert cfg.processed_path is not None
    if check_s669_exists(cfg) and not cfg.force_preprocess:
        LOGGER.info("Using cached processed S669 dataset: %s", cfg.processed_path)
        return cfg.processed_path

    raw_csv = download_s669(cfg)
    raw = pd.read_csv(raw_csv)
    _require_columns(
        raw,
        [
            "Protein",
            "PDB_Mut",
            "Seq_Mut",
            "WT_PDB",
            "Chain",
            "INV_PDB",
            "Experimental_DDG_dir",
            "Experimental_DDG_inv",
        ],
    )

    rows: list[dict[str, Any]] = []
    discarded_rows: list[dict[str, Any]] = []
    stats: OrderedDict[str, int] = OrderedDict(
        raw_rows=int(len(raw)),
        direct_rows=0,
        inverse_rows=0,
        invalid_mutation=0,
        invalid_ddg=0,
        missing_structure=0,
        structure_mismatch=0,
    )

    for raw_row in raw.to_dict(orient="records"):
        for direction in _directions(cfg):
            normalized, reason = _normalize_s669_row(cfg, raw_row, direction)
            if reason is not None:
                stats[reason] += 1
                discarded_rows.append(_discarded_row(raw_row, direction, reason))
                continue
            assert normalized is not None
            rows.append(normalized)
            stats[f"{direction}_rows"] += 1

    processed = pd.DataFrame(rows, columns=S669_OUTPUT_COLUMNS)
    if processed.empty:
        _write_s669_discarded(cfg, raw, discarded_rows)
        _write_s669_metadata(cfg, processed, stats)
        raise ValueError(f"No valid S669 rows after preprocessing. Stats: {dict(stats)}")

    processed = processed.sort_values(
        ["benchmark", "protein_id", "direction", "position", "mutation_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    _validate_s669_processed(processed)

    cfg.processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(cfg.processed_path, index=False)
    _write_s669_discarded(cfg, raw, discarded_rows)
    _write_s669_metadata(cfg, processed, stats)
    LOGGER.info(
        "Wrote processed S669 dataset: %s rows=%d proteins=%d stats=%s",
        cfg.processed_path,
        len(processed),
        processed["protein_id"].nunique(),
        dict(stats),
    )
    return cfg.processed_path


def load_s669(config: S669Config | dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the canonical S669 benchmark, creating it first when needed."""

    cfg = _coerce_s669_config(config)
    path = preprocess_s669(cfg) if not check_s669_exists(cfg) else cfg.processed_path
    assert path is not None
    return pd.read_csv(path, dtype=S669_CSV_DTYPES)


def _directions(cfg: S669Config) -> tuple[str, ...]:
    return ("direct", "inverse") if cfg.include_inverse else ("direct",)


def _normalize_s669_row(
    cfg: S669Config,
    row: dict[str, Any],
    direction: str,
) -> tuple[dict[str, Any] | None, str | None]:
    seq_mut = _parse_mutation(row.get("Seq_Mut"))
    pdb_mut = _parse_mutation(row.get("PDB_Mut"))
    if seq_mut is None or pdb_mut is None:
        return None, "invalid_mutation"

    seq_wt, seq_position, seq_mutant = seq_mut
    pdb_wt, direct_pdb_residue_id, pdb_mutant = pdb_mut
    if seq_wt != pdb_wt or seq_mutant != pdb_mutant:
        return None, "invalid_mutation"

    if direction == "direct":
        wt_residue = seq_wt
        mut_residue = seq_mutant
        ddg_raw = row.get("Experimental_DDG_dir")
        structure_id = _required_string(row.get("WT_PDB"))
        structure_residue_id = direct_pdb_residue_id
    elif direction == "inverse":
        wt_residue = seq_mutant
        mut_residue = seq_wt
        ddg_raw = row.get("Experimental_DDG_inv")
        structure_id = _required_string(row.get("INV_PDB"))
        structure_residue_id = seq_position
    else:
        raise ValueError(f"Unsupported S669 direction: {direction}")

    if structure_id is None:
        return None, "missing_structure"
    try:
        experimental_ddg = float(ddg_raw)
    except (TypeError, ValueError):
        return None, "invalid_ddg"

    pdb_path = _s669_structure_path(cfg, structure_id)
    if not pdb_path.exists():
        return None, "missing_structure"

    chain = _required_string(row.get("Chain"))
    protein_id = _required_string(row.get("Protein"))
    wt_pdb = _required_string(row.get("WT_PDB"))
    if chain is None or protein_id is None or wt_pdb is None:
        return None, "invalid_mutation"
    chain = chain.upper()
    pdb_id = wt_pdb[:4].upper()

    observed = _structure_residue(pdb_path, chain=chain, residue_id=structure_residue_id)
    if observed != wt_residue:
        return None, "structure_mismatch"

    mutation_id = f"{wt_residue}{seq_position}{mut_residue}"
    return {
        "benchmark": "S669",
        "direction": direction,
        "protein_id": f"{protein_id}_{direction}",
        "pdb_id": pdb_id,
        "chain": chain,
        "position": int(seq_position),
        "pdb_residue_id": str(structure_residue_id),
        "wt_residue": wt_residue,
        "mut_residue": mut_residue,
        "experimental_ddg": experimental_ddg,
        "mutation_id": mutation_id,
        "structure_id": structure_id,
        "pdb_path": str(pdb_path.resolve()),
        "doi": _optional_string(row.get("DOI")),
        "temperature": _optional_float(row.get("Temperature")),
        "ph": _optional_float(row.get("pH")),
        "experimental_method": _optional_string(row.get("EXPDTA")),
        "resolution": _optional_float(row.get("Resolution")),
    }, None


def _s669_structure_path(cfg: S669Config, structure_id: str) -> Path:
    assert cfg.raw_dir is not None
    pdb_dir = cfg.raw_dir / "S669" / "pdbs"
    exact = pdb_dir / f"{structure_id}.pdb"
    if exact.exists():
        return exact
    return pdb_dir / f"{structure_id.lower()}.pdb"


def _parse_mutation(value: Any) -> tuple[str, int, str] | None:
    text = _required_string(value)
    if text is None:
        return None
    match = MUTATION_PATTERN.fullmatch(text)
    if match is None:
        return None
    wt_residue, position, mut_residue = match.groups()
    wt_residue = wt_residue.upper()
    mut_residue = mut_residue.upper()
    if wt_residue not in CANONICAL_AA_1 or mut_residue not in CANONICAL_AA_1:
        return None
    return wt_residue, int(position), mut_residue


def _structure_residue(pdb_path: Path, *, chain: str, residue_id: int) -> str | None:
    normalized_chain = chain.strip().upper()
    target_residue_id = str(residue_id)
    for atom in parse_structure_records(pdb_path):
        if atom.chain_id.strip().upper() != normalized_chain:
            continue
        if str(atom.residue_id) != target_residue_id:
            continue
        return CANONICAL_AA_3_TO_1.get(atom.residue_name.upper())
    return None


def _required_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_string(value: Any) -> str:
    return _required_string(value) or ""


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"S669 source dataset is missing required columns: {missing}")


def _discarded_row(row: dict[str, Any], direction: str, reason: str) -> dict[str, Any]:
    discarded = dict(row)
    discarded["direction"] = direction
    discarded["discard_reason"] = reason
    return discarded


def _write_s669_discarded(
    cfg: S669Config,
    raw: pd.DataFrame,
    discarded_rows: list[dict[str, Any]],
) -> None:
    columns = list(raw.columns) + ["direction", "discard_reason"]
    frame = pd.DataFrame(discarded_rows, columns=columns)
    cfg.discarded_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.discarded_path, index=False)


def _write_s669_metadata(
    cfg: S669Config,
    processed: pd.DataFrame,
    stats: OrderedDict[str, int],
) -> None:
    dataset_hash = None
    if cfg.processed_path is not None and cfg.processed_path.exists():
        dataset_hash = _sha256(cfg.processed_path)
    metadata = {
        "benchmark": "S669",
        "dataset_source_url": cfg.source_url,
        "processed_dataset_path": str(cfg.processed_path),
        "discarded_rows_path": str(cfg.discarded_path),
        "include_inverse": cfg.include_inverse,
        "number_of_rows": int(len(processed)),
        "number_of_proteins": int(processed["protein_id"].nunique()) if not processed.empty else 0,
        "number_of_wt_pdbs": int(processed["pdb_id"].nunique()) if not processed.empty else 0,
        "ddg_range": (
            [
                float(processed["experimental_ddg"].min()),
                float(processed["experimental_ddg"].max()),
            ]
            if not processed.empty
            else None
        ),
        "filtering_steps": dict(stats),
        "output_columns": S669_OUTPUT_COLUMNS,
        "dataset_hash": dataset_hash,
        "preprocessing_version": S669_PREPROCESSING_VERSION,
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "preprocessing_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    cfg.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_s669_processed(frame: pd.DataFrame) -> None:
    if list(frame.columns) != S669_OUTPUT_COLUMNS:
        raise ValueError(f"S669 processed columns mismatch: {list(frame.columns)}")
    required = [
        "benchmark",
        "direction",
        "protein_id",
        "pdb_id",
        "chain",
        "position",
        "pdb_residue_id",
        "wt_residue",
        "mut_residue",
        "experimental_ddg",
        "mutation_id",
        "structure_id",
        "pdb_path",
    ]
    if frame[required].isna().any().any():
        raise ValueError("S669 processed dataset contains missing required values")
    if not set(frame["direction"]).issubset({"direct", "inverse"}):
        raise ValueError("S669 processed dataset contains unsupported directions")


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec: checksum for reproducibility, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
