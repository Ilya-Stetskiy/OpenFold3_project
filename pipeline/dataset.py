"""Dataset preparation for mutation ddG notebook experiments.

This module implements only the dataset stage. It downloads a FireProt-like
source dataset, normalizes single-point mutations, validates structure mapping,
and writes deterministic processed CSV outputs.
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
from openfold3.benchmark.structure_source import (
    CANONICAL_AA_1,
    CANONICAL_AA_3_TO_1,
    download_mmcif,
    normalize_pdb_id,
)

LOGGER = logging.getLogger(__name__)

FIREPROT_URL = "https://zenodo.org/records/8169289/files/fireprot_upload.zip?download=1"
FIREPROT_MD5 = "25cf04760665946b54b768eb7b6dfd1d"
FIREPROT_EXTRACTED_CSV = Path("fireprot_upload") / "csvs" / "4_fireprotDB_bestpH.csv"
OUTPUT_COLUMNS = [
    "protein_id",
    "pdb_id",
    "chain",
    "position",
    "wt_residue",
    "mut_residue",
    "experimental_ddg",
    "mutation_id",
    "pdb_residue_id",
    "pdb_path",
]
MUTATION_PATTERN = re.compile(
    r"^\s*([ACDEFGHIKLMNPQRSTVWY])\s*(\d+)\s*([ACDEFGHIKLMNPQRSTVWY])\s*$",
    re.I,
)
DISCARD_REASON_COLUMN = "discard_reason"
PREPROCESSING_VERSION = "v1.1"
CSV_DTYPES = {
    "protein_id": str,
    "pdb_id": str,
    "chain": str,
    "mutation_id": str,
    "pdb_residue_id": str,
    "pdb_path": str,
}
DISCARD_REASONS = (
    "invalid_pdb_id",
    "invalid_mutation",
    "invalid_ddg",
    "sequence_structure_mismatch",
    "mapping_failed",
    "duplicate",
)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for the dataset-only pipeline stage."""

    data_dir: Path = REPO_ROOT / "data"
    processed_path: Path | None = None
    raw_dir: Path | None = None
    structure_dir: Path | None = None
    source_url: str = FIREPROT_URL
    source_md5: str | None = FIREPROT_MD5
    source_archive_name: str = "fireprot_upload.zip"
    source_csv: Path | None = None
    force_download: bool = False
    force_preprocess: bool = False

    def resolved(self) -> "DatasetConfig":
        data_dir = Path(self.data_dir).expanduser().resolve()
        raw_dir = Path(self.raw_dir).expanduser().resolve() if self.raw_dir else data_dir / "raw"
        processed_path = (
            Path(self.processed_path).expanduser().resolve()
            if self.processed_path
            else data_dir / "processed" / "dataset_v1.csv"
        )
        structure_dir = (
            Path(self.structure_dir).expanduser().resolve()
            if self.structure_dir
            else raw_dir / "structures"
        )
        source_csv = Path(self.source_csv).expanduser().resolve() if self.source_csv else None
        return replace(
            self,
            data_dir=data_dir,
            raw_dir=raw_dir,
            processed_path=processed_path,
            structure_dir=structure_dir,
            source_csv=source_csv,
        )

    @property
    def discarded_path(self) -> Path:
        resolved = self.resolved()
        assert resolved.processed_path is not None
        return resolved.processed_path.with_name("discarded_rows.csv")

    @property
    def metadata_path(self) -> Path:
        resolved = self.resolved()
        assert resolved.processed_path is not None
        return resolved.processed_path.with_name("dataset_v1_meta.json")


def _coerce_config(config: DatasetConfig | dict[str, Any] | None) -> DatasetConfig:
    if config is None:
        return DatasetConfig().resolved()
    if isinstance(config, DatasetConfig):
        return config.resolved()
    if isinstance(config, dict):
        return DatasetConfig(**config).resolved()
    raise TypeError(f"Unsupported dataset config type: {type(config)!r}")


def check_dataset_exists(config: DatasetConfig | dict[str, Any] | None = None) -> bool:
    """Return True when the processed dataset and metadata are consistent."""

    cfg = _coerce_config(config)
    assert cfg.processed_path is not None
    if not cfg.processed_path.exists() or not cfg.metadata_path.exists():
        return False
    try:
        header = pd.read_csv(cfg.processed_path, nrows=0)
        if list(header.columns) != OUTPUT_COLUMNS:
            return False
        metadata = json.loads(cfg.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
            return False
        if metadata.get("output_columns") != OUTPUT_COLUMNS:
            return False
        if metadata.get("dataset_hash") != _sha256(cfg.processed_path):
            return False
        row_count = sum(1 for _ in cfg.processed_path.open("r", encoding="utf-8")) - 1
        return metadata.get("number_of_rows") == row_count
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def download_dataset(config: DatasetConfig | dict[str, Any] | None = None) -> Path:
    """Download and extract the FireProt-like raw dataset into data/raw/."""

    cfg = _coerce_config(config)
    assert cfg.raw_dir is not None
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cfg.raw_dir / cfg.source_archive_name
    extracted_csv = cfg.raw_dir / FIREPROT_EXTRACTED_CSV

    if cfg.source_csv is not None:
        if not cfg.source_csv.exists():
            raise FileNotFoundError(f"Configured source_csv does not exist: {cfg.source_csv}")
        LOGGER.info("Using configured source CSV: %s", cfg.source_csv)
        return cfg.source_csv

    if cfg.force_download or not archive_path.exists():
        LOGGER.info("Downloading dataset archive to %s", archive_path)
        urlretrieve(cfg.source_url, archive_path)
    else:
        LOGGER.info("Using cached dataset archive: %s", archive_path)

    if cfg.source_md5 is not None:
        digest = _md5(archive_path)
        if digest != cfg.source_md5:
            raise ValueError(
                f"Dataset archive checksum mismatch for {archive_path}: "
                f"expected {cfg.source_md5}, got {digest}"
            )

    if cfg.force_download or not extracted_csv.exists():
        LOGGER.info("Extracting dataset archive into %s", cfg.raw_dir)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(cfg.raw_dir)
    else:
        LOGGER.info("Using cached extracted dataset: %s", extracted_csv)

    if not extracted_csv.exists():
        raise FileNotFoundError(f"Expected FireProt CSV was not found: {extracted_csv}")
    return extracted_csv


def preprocess_dataset(config: DatasetConfig | dict[str, Any] | None = None) -> Path:
    """Create data/processed/dataset_v1.csv with strict deterministic filtering."""

    cfg = _coerce_config(config)
    assert cfg.processed_path is not None
    if check_dataset_exists(cfg) and not cfg.force_preprocess:
        LOGGER.info("Using cached processed dataset: %s", cfg.processed_path)
        return cfg.processed_path

    raw_csv = download_dataset(cfg)
    LOGGER.info("Loading raw dataset: %s", raw_csv)
    raw = pd.read_csv(raw_csv)
    _require_columns(raw, ["chain", "wild_type", "mutation", "ddG"])

    rows: list[dict[str, Any]] = []
    discarded_rows: list[dict[str, Any]] = []
    stats: OrderedDict[str, int] = OrderedDict(
        total=len(raw),
        invalid_pdb_id=0,
        invalid_mutation=0,
        invalid_ddg=0,
        sequence_structure_mismatch=0,
        mapping_failed=0,
        kept_before_dedup=0,
        duplicates_removed=0,
    )
    structure_cache: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for raw_row in raw.to_dict(orient="records"):
        normalized, reason = _normalize_raw_row(raw_row)
        if reason is not None:
            stats[reason] += 1
            discarded_rows.append(_discarded_row(raw_row, reason))
            continue
        assert normalized is not None

        if not _pdb_sequence_matches_wt(
            raw_row,
            pdb_position_index=int(normalized["_pdb_position_index"]),
            wt_residue=str(normalized["wt_residue"]),
        ):
            stats["sequence_structure_mismatch"] += 1
            discarded_rows.append(_discarded_row(raw_row, "sequence_structure_mismatch"))
            continue

        pdb_path = _resolve_pdb_path(cfg, str(normalized["pdb_id"]))
        pdb_residue_id = _resolve_pdb_residue_id(
            pdb_path=pdb_path,
            chain=str(normalized["chain"]),
            pdb_position_index=int(normalized["_pdb_position_index"]),
            wt_residue=str(normalized["wt_residue"]),
            cache=structure_cache,
        )
        if pdb_residue_id is None:
            stats["mapping_failed"] += 1
            discarded_rows.append(_discarded_row(raw_row, "mapping_failed"))
            continue

        normalized["pdb_path"] = str(pdb_path)
        normalized["pdb_residue_id"] = pdb_residue_id
        normalized["_raw_row"] = raw_row
        rows.append(normalized)

    stats["kept_before_dedup"] = len(rows)
    if not rows:
        _write_discarded_rows(cfg, raw, discarded_rows)
        _write_metadata(cfg, pd.DataFrame(columns=OUTPUT_COLUMNS), stats)
        raise ValueError(f"No valid dataset rows after preprocessing. Stats: {dict(stats)}")

    processed = pd.DataFrame(rows)
    processed = _sort_processed_rows(processed).reset_index(drop=True)
    duplicate_mask = processed.duplicated(
        subset=["protein_id", "mutation_id"],
        keep="first",
    )
    duplicate_rows = processed.loc[duplicate_mask]
    for duplicate_row in duplicate_rows.to_dict(orient="records"):
        discarded_rows.append(_discarded_row(duplicate_row["_raw_row"], "duplicate"))
    stats["duplicates_removed"] = int(duplicate_mask.sum())
    processed = processed.loc[~duplicate_mask, OUTPUT_COLUMNS].reset_index(drop=True)

    _validate_processed_dataset(processed)
    cfg.processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(cfg.processed_path, index=False)
    _write_discarded_rows(cfg, raw, discarded_rows)
    _write_metadata(cfg, processed, stats)
    _log_preprocessing_summary(processed, stats)
    LOGGER.info(
        "Wrote processed dataset: %s rows=%d proteins=%d stats=%s",
        cfg.processed_path,
        len(processed),
        processed["protein_id"].nunique(),
        dict(stats),
    )
    return cfg.processed_path


def load_dataset(config: DatasetConfig | dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the processed dataset, creating it first when needed."""

    cfg = _coerce_config(config)
    path = preprocess_dataset(cfg) if not check_dataset_exists(cfg) else cfg.processed_path
    assert path is not None
    return pd.read_csv(path, dtype=CSV_DTYPES)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec: checksum for reproducibility, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")


def _clean_optional_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _normalize_raw_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    pdb_raw = _clean_optional_string(row.get("pdb_id_corrected") or row.get("pdb_id"))
    chain_raw = _clean_optional_string(row.get("chain"))
    wt_raw = _clean_optional_string(row.get("wild_type") or row.get("wt_residue"))
    mut_raw = _clean_optional_string(row.get("mutation") or row.get("mut_residue"))
    ddg_raw = row.get("ddG", row.get("experimental_ddg"))
    protein_raw = _clean_optional_string(
        row.get("uniprot_id") or row.get("protein_id") or row.get("protein_name")
    )

    if pdb_raw is None:
        return None, "invalid_pdb_id"
    if chain_raw is None or wt_raw is None or mut_raw is None:
        return None, "invalid_mutation"
    if pd.isna(ddg_raw):
        return None, "invalid_ddg"

    try:
        pdb_id = normalize_pdb_id(str(pdb_raw).split("|")[0])
    except ValueError:
        return None, "invalid_pdb_id"

    parsed = _parse_mutation(row, wt_raw, mut_raw)
    if parsed is None:
        return None, "invalid_mutation"
    wt_residue, position, mut_residue = parsed

    pdb_position_index = _pdb_position_index_from_row(row)
    if pdb_position_index is None:
        return None, "mapping_failed"

    try:
        experimental_ddg = float(ddg_raw)
    except (TypeError, ValueError):
        return None, "invalid_ddg"
    if not -10.0 <= experimental_ddg <= 10.0:
        return None, "invalid_ddg"

    chain = chain_raw.strip().upper()
    protein_id = protein_raw or f"{pdb_id}_{chain}"
    mutation_id = f"{wt_residue}{position}{mut_residue}"
    return {
        "protein_id": str(protein_id),
        "pdb_id": pdb_id,
        "chain": chain,
        "position": int(position),
        "wt_residue": wt_residue,
        "mut_residue": mut_residue,
        "experimental_ddg": experimental_ddg,
        "mutation_id": mutation_id,
        "_pdb_position_index": pdb_position_index,
    }, None


def _parse_mutation(row: dict[str, Any], wt_raw: str, mut_raw: str) -> tuple[str, int, str] | None:
    wt_residue = wt_raw.strip().upper()
    mut_value = mut_raw.strip().upper()
    if wt_residue not in CANONICAL_AA_1:
        return None

    position = _sequence_position_from_row(row)
    if len(mut_value) == 1:
        mut_residue = mut_value
        if position is None:
            return None
    else:
        match = MUTATION_PATTERN.fullmatch(mut_value)
        if match is None:
            return None
        parsed_wt, parsed_position, mut_residue = match.groups()
        if parsed_wt.upper() != wt_residue:
            return None
        if position is None:
            position = int(parsed_position)

    if mut_residue not in CANONICAL_AA_1 or position is None or position < 1:
        return None
    return wt_residue, int(position), mut_residue


def _sequence_position_from_row(row: dict[str, Any]) -> int | None:
    position = row.get("position")
    if position is None or pd.isna(position):
        return None
    try:
        numeric_position = float(position)
    except (TypeError, ValueError):
        return None
    if not numeric_position.is_integer():
        return None
    return int(numeric_position)


def _pdb_position_index_from_row(row: dict[str, Any]) -> int | None:
    raw_value = row.get("pdb_position")
    if raw_value is None or pd.isna(raw_value):
        return None
    try:
        numeric_index = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not numeric_index.is_integer():
        return None
    index = int(numeric_index)
    return index if index >= 0 else None


def _pdb_sequence_matches_wt(
    row: dict[str, Any],
    *,
    pdb_position_index: int,
    wt_residue: str,
) -> bool:
    sequence = _clean_optional_string(row.get("pdb_sequence"))
    if sequence is None:
        return True
    normalized_sequence = sequence.upper()
    if pdb_position_index >= len(normalized_sequence):
        return False
    return normalized_sequence[pdb_position_index] == wt_residue


def _residue_sort_key(residue_id: str) -> tuple[int, str]:
    cleaned = str(residue_id).strip()
    match = re.fullmatch(r"(-?\d+)(.*)", cleaned)
    if match is None:
        return sys.maxsize, cleaned
    return int(match.group(1)), match.group(2)


def _sort_processed_rows(processed: pd.DataFrame) -> pd.DataFrame:
    return processed.sort_values(
        [
            "protein_id",
            "pdb_id",
            "chain",
            "pdb_residue_id",
            "position",
            "mutation_id",
            "mut_residue",
            "experimental_ddg",
        ],
        key=lambda values: (
            values.map(_residue_sort_key)
            if values.name == "pdb_residue_id"
            else values
        ),
        kind="mergesort",
    )


def _resolve_pdb_path(cfg: DatasetConfig, pdb_id: str) -> Path:
    assert cfg.raw_dir is not None
    assert cfg.structure_dir is not None
    local_candidates = [
        cfg.raw_dir / "fireprot_upload" / "pdbs" / f"{pdb_id}.pdb",
        cfg.raw_dir / "fireprot_upload" / "pdbs" / f"{pdb_id.lower()}.pdb",
        cfg.raw_dir / "pdbs" / f"{pdb_id}.pdb",
        cfg.raw_dir / "pdbs" / f"{pdb_id.lower()}.pdb",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate.resolve()

    LOGGER.info("Downloading missing structure for %s", pdb_id)
    path, _ = download_mmcif(pdb_id, cache_dir=cfg.structure_dir)
    return path.resolve()


def _ordered_chain_residues(pdb_path: Path, chain: str) -> list[tuple[str, str]]:
    residues: OrderedDict[str, str] = OrderedDict()
    normalized_chain = chain.strip().upper()
    for atom in parse_structure_records(pdb_path):
        if atom.chain_id.strip().upper() != normalized_chain:
            continue
        residue_name = CANONICAL_AA_3_TO_1.get(atom.residue_name.upper())
        if residue_name is None:
            continue
        residues.setdefault(str(atom.residue_id), residue_name)
    return list(residues.items())


def _resolve_pdb_residue_id(
    *,
    pdb_path: Path,
    chain: str,
    pdb_position_index: int,
    wt_residue: str,
    cache: dict[tuple[str, str], list[tuple[str, str]]],
) -> str | None:
    cache_key = (str(pdb_path), chain.strip().upper())
    if cache_key not in cache:
        cache[cache_key] = _ordered_chain_residues(pdb_path, chain)
    residues = cache[cache_key]
    if pdb_position_index < 0 or pdb_position_index >= len(residues):
        return None
    residue_id, observed_aa = residues[pdb_position_index]
    if observed_aa != wt_residue:
        return None
    return residue_id


def _discarded_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    discarded = dict(row)
    discarded[DISCARD_REASON_COLUMN] = reason
    return discarded


def _write_discarded_rows(
    cfg: DatasetConfig,
    raw: pd.DataFrame,
    discarded_rows: list[dict[str, Any]],
) -> None:
    path = cfg.discarded_path
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(raw.columns) + [DISCARD_REASON_COLUMN]
    frame = pd.DataFrame(discarded_rows, columns=columns)
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metadata(
    cfg: DatasetConfig,
    processed: pd.DataFrame,
    stats: OrderedDict[str, int],
) -> None:
    path = cfg.metadata_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if processed.empty:
        ddg_range = None
        protein_count = 0
    else:
        ddg_range = [
            float(processed["experimental_ddg"].min()),
            float(processed["experimental_ddg"].max()),
        ]
        protein_count = int(processed["protein_id"].nunique())
    dataset_hash = None
    if cfg.processed_path is not None and cfg.processed_path.exists():
        dataset_hash = _sha256(cfg.processed_path)
    metadata = {
        "dataset_source_url": cfg.source_url,
        "processed_dataset_path": str(cfg.processed_path),
        "discarded_rows_path": str(cfg.discarded_path),
        "number_of_rows": int(len(processed)),
        "number_of_proteins": protein_count,
        "ddg_range": ddg_range,
        "filtering_steps": dict(stats),
        "output_columns": OUTPUT_COLUMNS,
        "dataset_hash": dataset_hash,
        "preprocessing_version": PREPROCESSING_VERSION,
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "preprocessing_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log_preprocessing_summary(
    processed: pd.DataFrame,
    stats: OrderedDict[str, int],
) -> None:
    total = max(int(stats["total"]), 1)
    retained_pct = 100.0 * len(processed) / total
    discard_counts = {
        reason: int(stats.get(reason, 0))
        for reason in DISCARD_REASONS
        if reason != "duplicate"
    }
    discard_counts["duplicate"] = int(stats["duplicates_removed"])
    LOGGER.info("Discarded rows by reason: %s", discard_counts)
    LOGGER.info("Retained %.2f%% of raw rows (%d/%d)", retained_pct, len(processed), total)
    mapping_failed_pct = 100.0 * int(stats["mapping_failed"]) / total
    if mapping_failed_pct > 20.0:
        LOGGER.warning("mapping_failed is %.2f%% of raw rows", mapping_failed_pct)


def _validate_processed_dataset(frame: pd.DataFrame) -> None:
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Processed dataset columns mismatch: {list(frame.columns)}")
    if frame[OUTPUT_COLUMNS].isna().any().any():
        raise ValueError("Processed dataset contains missing values")
    if not frame["mutation_id"].map(lambda value: bool(MUTATION_PATTERN.fullmatch(str(value)))).all():
        raise ValueError("Processed dataset contains invalid mutation_id values")
    if not frame["wt_residue"].isin(CANONICAL_AA_1).all():
        raise ValueError("Processed dataset contains non-canonical wt_residue values")
    if not frame["mut_residue"].isin(CANONICAL_AA_1).all():
        raise ValueError("Processed dataset contains non-canonical mut_residue values")
    if not frame["experimental_ddg"].between(-10.0, 10.0).all():
        raise ValueError("Processed dataset contains ddG values outside [-10, 10]")
    if (frame["position"].astype(int) < 1).any():
        raise ValueError("Processed dataset contains non-positive sequence positions")
    if frame.duplicated(subset=["protein_id", "mutation_id"]).any():
        raise ValueError("Processed dataset contains duplicate protein_id + mutation_id rows")
    if frame.duplicated(subset=["pdb_id", "chain", "pdb_residue_id", "mut_residue"]).any():
        raise ValueError("Processed dataset contains duplicate structure mutation rows")

    structure_cache: dict[tuple[str, str], dict[str, str]] = {}
    for row in frame.to_dict(orient="records"):
        pdb_path = Path(str(row["pdb_path"]))
        if not pdb_path.exists():
            raise ValueError(f"Processed dataset references missing structure path: {pdb_path}")
        cache_key = (str(pdb_path), str(row["chain"]).strip().upper())
        if cache_key not in structure_cache:
            structure_cache[cache_key] = dict(_ordered_chain_residues(pdb_path, str(row["chain"])))
        observed = structure_cache[cache_key].get(str(row["pdb_residue_id"]))
        if observed != row["wt_residue"]:
            raise ValueError(
                "Processed dataset contains inconsistent structure mapping: "
                f"{row['pdb_id']} {row['chain']}:{row['pdb_residue_id']} "
                f"expected {row['wt_residue']} found {observed}"
            )
