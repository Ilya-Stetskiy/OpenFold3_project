from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

from .schemas import ArtifactManifest, LeaseJob, WorkerConfig
from .telemetry import utc_now


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_manifest(
    *,
    config: WorkerConfig,
    lease: LeaseJob,
    status: str,
    work_dir: Path,
    generated_inputs: list[Path],
    output_dir: Path | None,
    log_path: Path | None,
    telemetry_path: Path | None,
    error: str | None,
    timings: dict[str, float],
    summary: dict[str, Any],
) -> ArtifactManifest:
    return ArtifactManifest(
        job_id=lease.job_id,
        lease_id=lease.lease_id,
        worker_id=config.worker_id,
        job_type=lease.job_type,
        status=status,  # type: ignore[arg-type]
        created_at_utc=utc_now(),
        work_dir=str(work_dir),
        generated_inputs=[str(path) for path in generated_inputs],
        output_dir=str(output_dir) if output_dir else None,
        log_path=str(log_path) if log_path else None,
        telemetry_path=str(telemetry_path) if telemetry_path else None,
        error=error,
        timings=timings,
        summary=summary,
    )


def package_work_dir(work_dir: Path, manifest: ArtifactManifest) -> ArtifactManifest:
    manifest_path = work_dir / "artifact_manifest.json"
    write_json(manifest_path, manifest.model_dump(mode="json"))
    archive_path = work_dir.with_suffix(".tar.gz")
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(work_dir, arcname=work_dir.name)
    manifest.archive_path = str(archive_path)
    manifest.archive_size_bytes = archive_path.stat().st_size
    manifest.archive_sha256 = sha256_file(archive_path)
    write_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest


def cleanup_ttl_cache(results_dir: Path, ttl_days: int) -> list[Path]:
    if ttl_days <= 0 or not results_dir.exists():
        return []
    cutoff = time.time() - ttl_days * 24 * 60 * 60
    removed: list[Path] = []
    for child in results_dir.iterdir():
        if child.name == "state.json":
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        removed.append(child)
    return removed
