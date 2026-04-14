from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .artifacts import build_manifest, cleanup_ttl_cache, package_work_dir, write_json
from .client import WorkerClient
from .runner import OpenFoldWorkerRunner, PreparedCommand, run_command
from .schemas import (
    ArtifactManifest,
    HeartbeatPayload,
    JobEvent,
    LeaseJob,
    ResourceSnapshot,
    WorkerConfig,
    WorkerStatus,
)
from .telemetry import ResourceSampler, TelemetryRecorder, utc_now


class GPUWorker:
    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.client = WorkerClient(config)
        self.sampler = ResourceSampler(config.results_dir)
        self.runner = OpenFoldWorkerRunner(config)
        self.started_at = time.monotonic()
        self.status: WorkerStatus = "idle"
        self.active_job_id: str | None = None
        self._last_heartbeat_at = 0.0

    def run_forever(self) -> None:
        self._preflight()
        self._recover_state()
        while True:
            try:
                cleanup_ttl_cache(self.config.results_dir, self.config.cache_ttl_days)
                self._heartbeat(force=True)
                resources = self.sampler.snapshot().model_dump(mode="json")
                lease = self.client.lease(resources)
                if lease is None:
                    time.sleep(self.config.poll_interval_seconds)
                    continue
                self._run_lease(lease)
            except KeyboardInterrupt:
                self.status = "draining"
                self._heartbeat(force=True)
                raise
            except Exception as exc:
                self.status = "error"
                self.active_job_id = None
                self._heartbeat(force=True, error=str(exc))
                traceback.print_exc()
                time.sleep(self.config.poll_interval_seconds)

    def _preflight(self) -> None:
        if not self.config.effective_openfold_repo_dir.exists():
            raise FileNotFoundError(
                f"OpenFold repo not found: {self.config.effective_openfold_repo_dir}"
            )
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        probe_path = self.config.results_dir / ".write_probe"
        probe_path.write_text("ok\n", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        openfold_python = self.config.openfold_python
        executable = self._resolve_executable(openfold_python)
        probe = subprocess.run(
            [
                str(executable),
                "-c",
                "import click; import openfold3.run_openfold",
            ],
            cwd=self.config.effective_openfold_repo_dir,
            env=self.runner._build_env(),
            text=True,
            capture_output=True,
            timeout=min(30, self.config.request_timeout_seconds),
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "OpenFold runtime probe failed: "
                f"{(probe.stderr or probe.stdout).strip()}"
            )
        if self.config.min_free_disk_gb > 0:
            usage = shutil.disk_usage(self.config.results_dir)
            free_gb = usage.free / (1024**3)
            if free_gb < self.config.min_free_disk_gb:
                raise RuntimeError(
                    f"Low disk space: {free_gb:.2f} GB free, "
                    f"required {self.config.min_free_disk_gb:.2f} GB"
                )

    @staticmethod
    def _resolve_executable(path: Path) -> Path:
        if path.is_absolute() or path.parent != Path("."):
            if path.exists() and os.access(path, os.X_OK):
                return path
            raise FileNotFoundError(f"Python executable not found: {path}")
        resolved = shutil.which(str(path))
        if resolved:
            return Path(resolved)
        raise FileNotFoundError(f"Python executable not found in PATH: {path}")

    def _recover_state(self) -> None:
        state = self._read_state()
        if not state:
            return
        status = state.get("status")
        if status in {"uploading", "upload_failed"}:
            self._recover_upload(state)
            return
        if status == "running":
            state["status"] = "failed_recovered"
            state["recovered_at_utc"] = utc_now()
            self._write_state(state)

    def _recover_upload(self, state: dict[str, Any]) -> None:
        manifest_path = state.get("manifest_path")
        lease_payload = state.get("lease")
        if not manifest_path or not lease_payload:
            state["status"] = "failed_recovered"
            state["recovered_at_utc"] = utc_now()
            self._write_state(state)
            return
        manifest = ArtifactManifest.model_validate(
            json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        )
        lease = LeaseJob.model_validate(lease_payload)
        self.status = "uploading"
        self.active_job_id = manifest.job_id
        try:
            self._upload_and_complete(lease, manifest)
        except Exception as exc:
            state["status"] = "upload_failed"
            state["error"] = str(exc)
            state["updated_at_utc"] = utc_now()
            self._write_state(state)
            raise
        else:
            self._write_state(
                {
                    "status": "idle",
                    "last_job_id": manifest.job_id,
                    "last_lease_id": manifest.lease_id,
                    "last_status": manifest.status,
                    "recovered_upload": True,
                    "updated_at_utc": utc_now(),
                }
            )
        finally:
            self.status = "idle"
            self.active_job_id = None

    def _run_lease(self, lease: LeaseJob) -> None:
        self.status = "running"
        self.active_job_id = lease.job_id
        work_dir = self.config.results_dir / lease.job_id
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        self._write_state(
            {
                "status": "running",
                "job_id": lease.job_id,
                "lease_id": lease.lease_id,
                "lease": lease.model_dump(mode="json"),
                "work_dir": str(work_dir),
                "started_at_utc": utc_now(),
            }
        )
        self._event(lease, "accepted", "Lease accepted")
        command: PreparedCommand | None = None
        manifest: ArtifactManifest | None = None
        telemetry_path = work_dir / "telemetry.csv"
        telemetry = TelemetryRecorder(
            telemetry_path,
            self.sampler,
            self.config.telemetry_interval_seconds,
        )
        job_started = time.perf_counter()
        error: str | None = None
        status = "completed"
        try:
            self._enforce_limits(lease)
            command = self.runner.prepare(lease, work_dir)
            self._event(lease, "started", "Job prepared", {"cmd": command.cmd})
            telemetry.start()
            self._event(lease, "openfold_started", "OpenFold subprocess started")
            max_runtime = (
                lease.limits.max_runtime_seconds
                or self.config.max_job_runtime_seconds
            )
            progress_started = time.monotonic()
            last_progress_event_at = 0.0

            def report_progress(elapsed: float) -> None:
                nonlocal last_progress_event_at
                self._heartbeat()
                now = time.monotonic()
                if now - last_progress_event_at < self.config.heartbeat_interval_seconds:
                    return
                last_progress_event_at = now
                self._event(
                    lease,
                    "progress",
                    "OpenFold subprocess is still running",
                    {
                        "elapsed_seconds": elapsed,
                        "worker_elapsed_seconds": now - progress_started,
                    },
                )

            outcome = run_command(command, max_runtime, report_progress)
            self._event(
                lease,
                "openfold_finished",
                "OpenFold subprocess finished",
                {
                    "return_code": outcome.return_code,
                    "timed_out": outcome.timed_out,
                    "elapsed_seconds": outcome.elapsed_seconds,
                },
            )
            if outcome.timed_out:
                raise TimeoutError(f"Job exceeded max runtime of {max_runtime}s")
            if outcome.return_code != 0:
                raise RuntimeError(f"OpenFold exited with code {outcome.return_code}")
        except Exception as exc:
            status = "failed"
            error = str(exc)
            self._event(lease, "failed", error)
        finally:
            telemetry.stop()

        self.status = "uploading"
        self._write_state(
            {
                "status": "uploading",
                "job_id": lease.job_id,
                "lease_id": lease.lease_id,
                "lease": lease.model_dump(mode="json"),
                "work_dir": str(work_dir),
                "updated_at_utc": utc_now(),
            }
        )
        self._event(lease, "packaging", "Packaging artifacts")
        output_dir = command.output_dir if command is not None else None
        log_path = command.log_path if command is not None else None
        generated_inputs = command.generated_inputs if command is not None else []
        summary = command.summary if command is not None else {}
        manifest = build_manifest(
            config=self.config,
            lease=lease,
            status=status,
            work_dir=work_dir,
            generated_inputs=generated_inputs,
            output_dir=output_dir,
            log_path=log_path,
            telemetry_path=telemetry_path,
            error=error,
            timings={"total_worker_seconds": time.perf_counter() - job_started},
            summary=summary,
        )
        manifest = package_work_dir(work_dir, manifest)
        try:
            self._write_state(
                {
                    "status": "uploading",
                    "job_id": lease.job_id,
                    "lease_id": lease.lease_id,
                    "lease": lease.model_dump(mode="json"),
                    "work_dir": str(work_dir),
                    "manifest_path": str(work_dir / "artifact_manifest.json"),
                    "archive_path": manifest.archive_path,
                    "updated_at_utc": utc_now(),
                }
            )
            self._upload_and_complete(lease, manifest)
            final_event = "completed" if status == "completed" else "failed"
            self._event(lease, final_event, error)
        except Exception as exc:
            self.status = "error"
            self._write_state(
                {
                    "status": "upload_failed",
                    "job_id": lease.job_id,
                    "lease_id": lease.lease_id,
                    "lease": lease.model_dump(mode="json"),
                    "work_dir": str(work_dir),
                    "manifest_path": str(work_dir / "artifact_manifest.json"),
                    "archive_path": manifest.archive_path,
                    "error": str(exc),
                    "updated_at_utc": utc_now(),
                }
            )
            raise
        else:
            self.status = "idle"
            self.active_job_id = None
            self._write_state(
                {
                    "status": "idle",
                    "last_job_id": lease.job_id,
                    "last_lease_id": lease.lease_id,
                    "last_status": status,
                    "updated_at_utc": utc_now(),
                }
            )
            self._heartbeat(force=True)

    def _upload_and_complete(
        self,
        lease: LeaseJob,
        manifest: ArtifactManifest,
    ) -> None:
        if manifest.archive_path is None:
            raise RuntimeError("Artifact archive was not created")
        self._event(lease, "uploading", "Uploading artifact archive")
        manifest.upload_response = self.client.upload_archive(
            lease.upload,
            Path(manifest.archive_path),
            manifest,
        )
        write_json(
            Path(manifest.work_dir) / "artifact_manifest.json",
            manifest.model_dump(mode="json"),
        )
        self.client.complete(manifest)

    def _enforce_limits(self, lease: LeaseJob) -> None:
        min_free = lease.limits.min_free_disk_gb
        if min_free is None:
            min_free = self.config.min_free_disk_gb
        if min_free <= 0:
            return
        usage = shutil.disk_usage(self.config.results_dir)
        free_gb = usage.free / (1024**3)
        if free_gb < min_free:
            raise RuntimeError(
                f"Low disk space: {free_gb:.2f} GB free, "
                f"required {min_free:.2f} GB"
            )

    def _heartbeat(self, *, force: bool = False, error: str | None = None) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self._last_heartbeat_at
            < self.config.heartbeat_interval_seconds
        ):
            return
        details: dict[str, Any] = {}
        if error:
            details["error"] = error
        payload = HeartbeatPayload(
            worker_id=self.config.worker_id,
            status=self.status,
            active_job_id=self.active_job_id,
            uptime_seconds=now - self.started_at,
            resources=self.sampler.snapshot(),
            **details,
        )
        self.client.heartbeat(payload)
        self._last_heartbeat_at = now

    def _event(
        self,
        lease: LeaseJob,
        event: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        resources: ResourceSnapshot | None = None
        try:
            resources = self.sampler.snapshot()
        except Exception:
            resources = None
        self.client.event(
            lease.job_id,
            JobEvent(
                worker_id=self.config.worker_id,
                lease_id=lease.lease_id,
                event=event,  # type: ignore[arg-type]
                timestamp_utc=utc_now(),
                message=message,
                details=details or {},
                resources=resources,
            ),
        )

    def _read_state(self) -> dict[str, Any]:
        if not self.config.state_path.exists():
            return {}
        try:
            return json.loads(self.config.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        write_json(self.config.state_path, state)


def main() -> None:
    config = WorkerConfig()
    worker = GPUWorker(config)
    worker.run_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
