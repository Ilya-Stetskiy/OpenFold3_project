from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from gpu_worker.client import WorkerClient
from gpu_worker.schemas import (
    ArtifactManifest,
    HeartbeatPayload,
    UploadTarget,
    WorkerConfig,
)
from gpu_worker.telemetry import ResourceSnapshot


class _Handler(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.__class__.calls.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": json.loads(body.decode("utf-8")),
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if self.path == "/api/workers/lease":
            self.wfile.write(json.dumps({"job": None}).encode("utf-8"))
        else:
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

    def do_PUT(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.__class__.calls.append(
            {
                "path": self.path,
                "content_length": self.headers.get("Content-Length"),
                "body": body,
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"artifact_id": "artifact-1"}).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return


def test_client_sends_bearer_token_and_heartbeat() -> None:
    _Handler.calls = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = WorkerConfig(
            worker_id="worker-1",
            worker_token="secret",
            server_url=f"http://127.0.0.1:{server.server_port}",
        )
        client = WorkerClient(config)
        resources = ResourceSnapshot(
            timestamp_utc="2026-01-01T00:00:00Z",
            cpu_percent=0,
            load_avg_1m=0,
            memory_used_gb=0,
            memory_percent=0,
            disk_free_gb=1,
            disk_percent=0,
            gpu=[],
        )
        client.heartbeat(
            HeartbeatPayload(
                worker_id="worker-1",
                status="idle",
                uptime_seconds=1,
                resources=resources,
            )
        )
        assert client.lease(resources.model_dump()) is None
    finally:
        server.shutdown()
        thread.join()

    assert _Handler.calls[0]["auth"] == "Bearer secret"
    assert _Handler.calls[0]["path"] == "/api/workers/heartbeat"
    assert _Handler.calls[1]["path"] == "/api/workers/lease"


def test_client_uploads_archive_from_path(tmp_path) -> None:
    _Handler.calls = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    archive_path = tmp_path / "job.tar.gz"
    archive_path.write_bytes(b"archive-bytes")
    try:
        config = WorkerConfig(
            worker_id="worker-1",
            worker_token="secret",
            server_url=f"http://127.0.0.1:{server.server_port}",
        )
        client = WorkerClient(config)
        manifest = ArtifactManifest(
            job_id="job-1",
            lease_id="lease-1",
            worker_id="worker-1",
            job_type="predict_batch",
            status="completed",
            created_at_utc="2026-01-01T00:00:00Z",
            work_dir=str(tmp_path),
        )

        response = client.upload_archive(
            UploadTarget(url=f"http://127.0.0.1:{server.server_port}/upload"),
            archive_path,
            manifest,
        )
    finally:
        server.shutdown()
        thread.join()

    assert response["artifact_id"] == "artifact-1"
    assert _Handler.calls[0]["body"] == b"archive-bytes"
    assert _Handler.calls[0]["content_length"] == str(len(b"archive-bytes"))
