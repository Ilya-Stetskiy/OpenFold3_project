from __future__ import annotations

from pathlib import Path

from gpu_worker.artifacts import build_manifest, package_work_dir, write_json
import gpu_worker.main as worker_main
from gpu_worker.main import GPUWorker
from gpu_worker.runner import PreparedCommand, RunOutcome
from gpu_worker.schemas import LeaseJob, WorkerConfig


class _FakeClient:
    def __init__(self, *, fail_upload: bool = False) -> None:
        self.events = []
        self.completed = []
        self.heartbeats = []
        self.fail_upload = fail_upload

    def heartbeat(self, payload):
        self.heartbeats.append(payload)
        return {"ok": True}

    def event(self, job_id, event):
        self.events.append((job_id, event.event))
        return {"ok": True}

    def complete(self, manifest):
        self.completed.append(manifest)
        return {"ok": True}

    def upload_archive(self, target, archive_path, manifest):
        if self.fail_upload:
            raise RuntimeError("upload failed")
        return {"artifact_id": "artifact-1", "size": archive_path.stat().st_size}


class _FakeRunner:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def prepare(self, lease, work_dir):
        output_dir = work_dir / "output"
        output_dir.mkdir(parents=True)
        (output_dir / "result.txt").write_text("ok", encoding="utf-8")
        query_path = work_dir / "query.json"
        query_path.write_text("{}", encoding="utf-8")
        return PreparedCommand(
            cmd=["fake-openfold"],
            cwd=self.tmp_path,
            env={},
            generated_inputs=[query_path],
            output_dir=output_dir,
            log_path=work_dir / "run.log",
            summary={"mode": "test"},
        )


def test_worker_run_lease_packages_and_completes(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "openfold-3"
    repo.mkdir()
    config = WorkerConfig(
        worker_id="worker-1",
        worker_token="token",
        server_url="http://server.test",
        openfold_project_dir=tmp_path,
        openfold_repo_dir=repo,
        results_dir=tmp_path / "results",
        telemetry_interval_seconds=0.01,
    )
    worker = GPUWorker(config)
    fake_client = _FakeClient()
    worker.client = fake_client
    worker.runner = _FakeRunner(tmp_path)

    def fake_run_command(command, timeout_seconds, progress_callback=None):
        command.log_path.write_text("done\n", encoding="utf-8")
        if progress_callback is not None:
            progress_callback(0.01)
        return RunOutcome(return_code=0, timed_out=False, elapsed_seconds=0.01)

    monkeypatch.setattr(worker_main, "run_command", fake_run_command)
    lease = LeaseJob.model_validate(
        {
            "job_id": "job-1",
            "lease_id": "lease-1",
            "job_type": "predict_batch",
            "payload": {
                "queries": [
                    {
                        "molecules": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                ]
            },
            "upload": {"url": "http://server.test/upload"},
        }
    )

    worker._run_lease(lease)

    assert ("job-1", "accepted") in fake_client.events
    assert ("job-1", "completed") in fake_client.events
    assert len(fake_client.completed) == 1
    manifest = fake_client.completed[0]
    assert manifest.status == "completed"
    assert manifest.archive_sha256
    assert manifest.upload_response["artifact_id"] == "artifact-1"


def test_worker_keeps_upload_failed_state(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "openfold-3"
    repo.mkdir()
    config = WorkerConfig(
        worker_id="worker-1",
        worker_token="token",
        server_url="http://server.test",
        openfold_project_dir=tmp_path,
        openfold_repo_dir=repo,
        results_dir=tmp_path / "results",
        telemetry_interval_seconds=0.01,
    )
    worker = GPUWorker(config)
    worker.client = _FakeClient(fail_upload=True)
    worker.runner = _FakeRunner(tmp_path)

    def fake_run_command(command, timeout_seconds, progress_callback=None):
        command.log_path.write_text("done\n", encoding="utf-8")
        return RunOutcome(return_code=0, timed_out=False, elapsed_seconds=0.01)

    monkeypatch.setattr(worker_main, "run_command", fake_run_command)
    lease = LeaseJob.model_validate(
        {
            "job_id": "job-1",
            "lease_id": "lease-1",
            "job_type": "predict_batch",
            "payload": {
                "queries": [
                    {
                        "molecules": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                ]
            },
            "upload": {"url": "http://server.test/upload"},
        }
    )

    try:
        worker._run_lease(lease)
    except RuntimeError as exc:
        assert str(exc) == "upload failed"
    else:
        raise AssertionError("upload failure did not propagate")

    state = config.state_path.read_text(encoding="utf-8")
    assert '"status": "upload_failed"' in state
    assert '"manifest_path"' in state


def test_worker_recovers_upload_failed_state(tmp_path: Path) -> None:
    repo = tmp_path / "openfold-3"
    repo.mkdir()
    config = WorkerConfig(
        worker_id="worker-1",
        worker_token="token",
        server_url="http://server.test",
        openfold_project_dir=tmp_path,
        openfold_repo_dir=repo,
        results_dir=tmp_path / "results",
        telemetry_interval_seconds=0.01,
    )
    worker = GPUWorker(config)
    fake_client = _FakeClient()
    worker.client = fake_client
    work_dir = config.results_dir / "job-1"
    work_dir.mkdir(parents=True)
    (work_dir / "output.txt").write_text("ok", encoding="utf-8")
    lease = LeaseJob.model_validate(
        {
            "job_id": "job-1",
            "lease_id": "lease-1",
            "job_type": "predict_batch",
            "payload": {"queries": [{"molecules": [{"sequence": "ACDE"}]}]},
            "upload": {"url": "http://server.test/upload"},
        }
    )
    manifest = package_work_dir(
        work_dir,
        build_manifest(
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
        ),
    )
    write_json(
        config.state_path,
        {
            "status": "upload_failed",
            "job_id": "job-1",
            "lease_id": "lease-1",
            "lease": lease.model_dump(mode="json"),
            "work_dir": str(work_dir),
            "manifest_path": str(work_dir / "artifact_manifest.json"),
            "archive_path": manifest.archive_path,
        },
    )

    worker._recover_state()

    assert len(fake_client.completed) == 1
    assert '"status": "idle"' in config.state_path.read_text(encoding="utf-8")
