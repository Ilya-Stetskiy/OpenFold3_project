from __future__ import annotations

import tarfile
import time
from pathlib import Path

from gpu_worker.artifacts import build_manifest, cleanup_ttl_cache, package_work_dir
from gpu_worker.schemas import LeaseJob, WorkerConfig


def test_package_work_dir_includes_manifest_and_payload(tmp_path: Path) -> None:
    work_dir = tmp_path / "job-1"
    work_dir.mkdir()
    (work_dir / "output.txt").write_text("ok", encoding="utf-8")
    config = WorkerConfig(
        worker_id="worker-1",
        worker_token="token",
        server_url="http://server.test",
        openfold_project_dir=tmp_path,
        results_dir=tmp_path,
    )
    lease = LeaseJob.model_validate(
        {
            "job_id": "job-1",
            "lease_id": "lease-1",
            "job_type": "predict_batch",
            "payload": {"queries": []},
            "upload": {"url": "http://server.test/upload"},
        }
    )
    manifest = build_manifest(
        config=config,
        lease=lease,
        status="completed",
        work_dir=work_dir,
        generated_inputs=[],
        output_dir=work_dir,
        log_path=None,
        telemetry_path=None,
        error=None,
        timings={},
        summary={},
    )

    packaged = package_work_dir(work_dir, manifest)

    assert packaged.archive_sha256
    assert packaged.archive_size_bytes > 0
    with tarfile.open(packaged.archive_path, "r:gz") as archive:
        names = archive.getnames()
    assert "job-1/artifact_manifest.json" in names
    assert "job-1/output.txt" in names


def test_cleanup_ttl_cache_removes_old_artifacts(tmp_path: Path) -> None:
    old_dir = tmp_path / "old-job"
    old_dir.mkdir()
    fresh_dir = tmp_path / "fresh-job"
    fresh_dir.mkdir()
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    old_time = time.time() - 3 * 24 * 60 * 60
    old_dir.touch()
    Path(old_dir).chmod(0o755)
    import os

    os.utime(old_dir, (old_time, old_time))

    removed = cleanup_ttl_cache(tmp_path, ttl_days=1)

    assert old_dir in removed
    assert not old_dir.exists()
    assert fresh_dir.exists()
    assert state.exists()
