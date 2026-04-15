from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import write_json
from .main import GPUWorker
from .schemas import ArtifactManifest, HeartbeatPayload, JobEvent, LeaseJob, WorkerConfig


class LocalWorkerClient:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.upload_dir = results_dir / "local_uploads"
        self.events_path = results_dir / "local_events.jsonl"
        self.heartbeats_path = results_dir / "local_heartbeats.jsonl"
        self.completed_path = results_dir / "local_completed_manifest.json"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def heartbeat(self, payload: HeartbeatPayload) -> dict[str, Any]:
        self._append_jsonl(self.heartbeats_path, payload.model_dump(mode="json"))
        return {"ok": True, "local": True}

    def event(self, job_id: str, event: JobEvent) -> dict[str, Any]:
        row = event.model_dump(mode="json")
        row["job_id"] = job_id
        self._append_jsonl(self.events_path, row)
        return {"ok": True, "local": True}

    def complete(self, manifest: ArtifactManifest) -> dict[str, Any]:
        write_json(self.completed_path, manifest.model_dump(mode="json"))
        return {"ok": True, "local": True}

    def upload_archive(
        self,
        target: Any,
        archive_path: Path,
        manifest: ArtifactManifest,
    ) -> dict[str, Any]:
        destination = self.upload_dir / Path(archive_path).name
        shutil.copy2(archive_path, destination)
        return {
            "artifact_id": destination.name,
            "url": destination.resolve().as_uri(),
            "path": str(destination),
            "size": destination.stat().st_size,
            "local": True,
        }

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_lease(args: argparse.Namespace, config: WorkerConfig) -> LeaseJob:
    if args.lease_json is not None:
        return LeaseJob.model_validate(
            json.loads(Path(args.lease_json).read_text(encoding="utf-8"))
        )
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    job_id = args.job_id or f"local-{int(time.time())}"
    return LeaseJob.model_validate(
        {
            "job_id": job_id,
            "lease_id": f"{job_id}-lease",
            "job_type": args.job_type,
            "payload": payload,
            "limits": {
                "max_runtime_seconds": config.max_job_runtime_seconds,
                "min_free_disk_gb": config.min_free_disk_gb,
            },
            "upload": {"url": f"file://{config.results_dir}/local_uploads"},
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run one OpenFold3 worker payload locally without a server."
    )
    parser.add_argument("--payload", type=Path, help="High-level job payload JSON.")
    parser.add_argument("--lease-json", type=Path, help="Full lease JSON to execute.")
    parser.add_argument(
        "--job-type",
        choices=["predict_batch", "variant_batch"],
        default="predict_batch",
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    if args.payload is None and args.lease_json is None:
        parser.error("one of --payload or --lease-json is required")

    config_kwargs: dict[str, Any] = {"worker_id": "local-gpu-worker"}
    if args.results_dir is not None:
        config_kwargs["results_dir"] = args.results_dir
    config = WorkerConfig(**config_kwargs)
    client = LocalWorkerClient(config.results_dir)
    worker = GPUWorker(
        config,
        client=client,
        require_client=False,
    )

    if not args.skip_preflight:
        worker._preflight()
    lease = _load_lease(args, config)
    worker._run_lease(lease)
    completed = json.loads(client.completed_path.read_text(encoding="utf-8"))
    status = completed.get("status", "unknown")
    print(f"local job finished with status={status}: {config.results_dir / lease.job_id}")
    if status != "completed":
        print(
            f"local job failed; inspect {config.results_dir / lease.job_id / 'run_openfold.log'}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
