from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .schemas import (
    ArtifactManifest,
    HeartbeatPayload,
    JobEvent,
    LeaseJob,
    UploadTarget,
    WorkerConfig,
)


class WorkerClient:
    def __init__(self, config: WorkerConfig) -> None:
        if not config.worker_token:
            raise ValueError("WORKER_TOKEN must be set")
        self.config = config

    def heartbeat(self, payload: HeartbeatPayload) -> dict[str, Any]:
        return self._json_request(
            "POST",
            "/api/workers/heartbeat",
            payload.model_dump(mode="json"),
        )

    def lease(self, resources: dict[str, Any]) -> LeaseJob | None:
        response = self._json_request(
            "POST",
            "/api/workers/lease",
            {
                "worker_id": self.config.worker_id,
                "max_active_jobs": 1,
                "supported_job_types": ["predict_batch", "variant_batch"],
                "resources": resources,
            },
        )
        job = response.get("job", response)
        if job in ({}, None):
            return None
        return LeaseJob.model_validate(job)

    def event(self, job_id: str, event: JobEvent) -> dict[str, Any]:
        return self._json_request(
            "POST",
            f"/api/jobs/{urllib.parse.quote(job_id)}/events",
            event.model_dump(mode="json"),
        )

    def complete(self, manifest: ArtifactManifest) -> dict[str, Any]:
        return self._json_request(
            "POST",
            f"/api/jobs/{urllib.parse.quote(manifest.job_id)}/complete",
            manifest.model_dump(mode="json"),
        )

    def upload_archive(
        self,
        target: UploadTarget,
        archive_path: Path,
        manifest: ArtifactManifest,
    ) -> dict[str, Any]:
        archive_size = archive_path.stat().st_size
        if (
            target.max_size_bytes is not None
            and archive_size > target.max_size_bytes
        ):
            raise ValueError(
                f"Archive size {archive_size} exceeds upload limit "
                f"{target.max_size_bytes}"
            )
        headers = dict(target.headers)
        headers.setdefault("Content-Type", "application/gzip")
        headers.setdefault("Content-Length", str(archive_size))
        headers.setdefault(
            "X-Artifact-Manifest",
            json.dumps(manifest.model_dump(mode="json")),
        )
        with archive_path.open("rb") as archive_file:
            request = urllib.request.Request(
                target.url,
                data=archive_file,
                method=target.method,
                headers=headers,
            )
            with urllib.request.urlopen(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                body = response.read()
                status = response.status
            if not body:
                return {"status": status}
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {"body": body.decode("utf-8", errors="replace")}
            payload.setdefault("status", status)
            return payload

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.server_url}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.config.worker_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{method} {path} failed with HTTP {exc.code}: {detail}"
            ) from exc
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))
