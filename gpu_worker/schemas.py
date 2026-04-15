from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


WorkerStatus = Literal["idle", "running", "uploading", "draining", "error"]
JobType = Literal["predict_batch", "variant_batch"]
KNOWN_CHECKPOINT_FILENAMES = (
    "of3-p2-155k.pt",
    "of3-p2-145k.pt",
    "of3_ft3_v1.pt",
)
EventType = Literal[
    "accepted",
    "started",
    "openfold_started",
    "progress",
    "openfold_finished",
    "packaging",
    "uploading",
    "completed",
    "failed",
]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class WorkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(
        default_factory=lambda: os.environ.get("WORKER_ID", "gpu-worker-1")
    )
    worker_token: str = Field(
        default_factory=lambda: os.environ.get("WORKER_TOKEN", "")
    )
    server_url: str = Field(
        default_factory=lambda: os.environ.get("SERVER_URL", "http://127.0.0.1:8000")
    )
    openfold_project_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("OPENFOLD_PROJECT_DIR", Path.cwd())
        ).expanduser()
    )
    openfold_repo_dir: Path | None = Field(
        default_factory=lambda: (
            Path(os.environ["OPENFOLD_REPO_DIR"]).expanduser()
            if os.environ.get("OPENFOLD_REPO_DIR")
            else None
        )
    )
    openfold_prefix: Path | None = Field(
        default_factory=lambda: (
            Path(os.environ["OPENFOLD_PREFIX"]).expanduser()
            if os.environ.get("OPENFOLD_PREFIX")
            else None
        )
    )
    results_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("WORKER_RESULTS_DIR", Path.cwd() / ".runtime" / "gpu_worker")
        ).expanduser()
    )
    openfold_cache: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("OPENFOLD_CACHE", "/weights")
        ).expanduser()
    )
    triton_cache_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("TRITON_CACHE_DIR", "/triton_cache")
        ).expanduser()
    )
    torch_extensions_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("TORCH_EXTENSIONS_DIR", "/triton_cache/torch_extensions")
        ).expanduser()
    )
    default_inference_ckpt_path: Path | None = Field(
        default_factory=lambda: (
            Path(os.environ["OPENFOLD_INFERENCE_CKPT_PATH"]).expanduser()
            if os.environ.get("OPENFOLD_INFERENCE_CKPT_PATH")
            else None
        )
    )
    require_cuda: bool = Field(
        default_factory=lambda: _env_bool("WORKER_REQUIRE_CUDA", True)
    )
    require_checkpoint: bool = Field(
        default_factory=lambda: _env_bool("WORKER_REQUIRE_CHECKPOINT", True)
    )
    cache_ttl_days: int = Field(
        default_factory=lambda: int(os.environ.get("WORKER_CACHE_TTL_DAYS", "7")),
        ge=0,
    )
    max_job_runtime_seconds: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_JOB_RUNTIME_SECONDS", "86400")),
        ge=1,
    )
    min_free_disk_gb: float = Field(
        default_factory=lambda: float(os.environ.get("MIN_FREE_DISK_GB", "5")),
        ge=0,
    )
    poll_interval_seconds: float = Field(
        default_factory=lambda: float(
            os.environ.get("WORKER_POLL_INTERVAL_SECONDS", "10")
        ),
        ge=0.1,
    )
    heartbeat_interval_seconds: float = Field(
        default_factory=lambda: float(
            os.environ.get("WORKER_HEARTBEAT_INTERVAL_SECONDS", "30")
        ),
        ge=1,
    )
    request_timeout_seconds: float = Field(
        default_factory=lambda: float(
            os.environ.get("WORKER_REQUEST_TIMEOUT_SECONDS", "30")
        ),
        ge=1,
    )
    telemetry_interval_seconds: float = Field(
        default_factory=lambda: float(
            os.environ.get("WORKER_TELEMETRY_INTERVAL_SECONDS", "5")
        ),
        ge=0.01,
    )

    @field_validator("server_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def openfold_python(self) -> Path:
        if self.openfold_prefix is not None:
            candidate = self.openfold_prefix / "bin" / "python"
            if candidate.exists():
                return candidate
        configured_python = os.environ.get("PYTHON")
        if configured_python:
            return Path(configured_python)
        python3 = shutil.which("python3")
        if python3:
            return Path(python3)
        return Path("python")

    @property
    def effective_openfold_repo_dir(self) -> Path:
        return self.openfold_repo_dir or self.openfold_project_dir / "openfold-3"

    @property
    def state_path(self) -> Path:
        return self.results_dir / "state.json"

    def resolve_checkpoint_path(self, *, required: bool) -> Path | None:
        if self.default_inference_ckpt_path is not None:
            if self.default_inference_ckpt_path.exists():
                return self.default_inference_ckpt_path
            if required:
                raise FileNotFoundError(
                    "OPENFOLD_INFERENCE_CKPT_PATH does not exist: "
                    f"{self.default_inference_ckpt_path}"
                )
            return None

        root = self.openfold_cache
        ckpt_root_file = root / "ckpt_root"
        if ckpt_root_file.exists():
            root = Path(ckpt_root_file.read_text(encoding="utf-8").strip())

        for filename in KNOWN_CHECKPOINT_FILENAMES:
            candidate = root / filename
            if candidate.exists():
                return candidate

        if required:
            expected = ", ".join(KNOWN_CHECKPOINT_FILENAMES)
            raise FileNotFoundError(
                "No OpenFold checkpoint found. Mount weights at OPENFOLD_CACHE "
                f"({self.openfold_cache}) or set OPENFOLD_INFERENCE_CKPT_PATH. "
                f"Expected one of: {expected}"
            )
        return None


class RuntimeLimits(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_runtime_seconds: int | None = Field(default=None, ge=1)
    min_free_disk_gb: float | None = Field(default=None, ge=0)
    max_queries: int | None = Field(default=None, ge=1)
    max_variants: int | None = Field(default=None, ge=1)


class UploadTarget(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str
    method: Literal["PUT", "POST"] = "PUT"
    headers: dict[str, str] = Field(default_factory=dict)
    max_size_bytes: int | None = Field(default=None, ge=1)
    expires_at: datetime | None = None


class PointMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chain_id: str
    position_1based: int = Field(ge=1)
    from_residue: str | None = Field(default=None, min_length=1, max_length=1)
    to_residue: str = Field(min_length=1, max_length=1)

    @field_validator("from_residue", "to_residue")
    @classmethod
    def residue_upper(cls, value: str | None) -> str | None:
        return value.upper() if value is not None else None


class QuerySpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    query_id: str | None = None
    molecules: list[dict[str, Any]] = Field(min_length=1)


class PredictBatchPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    queries: list[QuerySpec] = Field(min_length=1)
    seeds: list[int] | None = None
    num_diffusion_samples: int | None = Field(default=None, ge=1)
    num_model_seeds: int | None = Field(default=None, ge=1)
    use_msa_server: bool = True
    use_templates: bool = True
    runner_yaml: str | None = None
    inference_ckpt_path: str | None = None
    inference_ckpt_name: str | None = None


class VariantSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_id: str
    mutations: list[PointMutation] | None = None
    molecules: list[dict[str, Any]] | None = None

    @field_validator("variant_id")
    @classmethod
    def nonempty_variant_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("variant_id must be non-empty")
        return value


class VariantBatchPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    base_query: QuerySpec
    variants: list[VariantSpec] = Field(min_length=1)
    include_wt: bool = True
    query_prefix: str = "variant"
    num_diffusion_samples: int | None = Field(default=None, ge=1)
    num_model_seeds: int | None = Field(default=None, ge=1)
    use_msa_server: bool = True
    use_templates: bool = True
    runner_yaml: str | None = None
    inference_ckpt_path: str | None = None
    inference_ckpt_name: str | None = None
    output_policy: Literal["full", "metrics_only"] = "full"
    num_cpu_workers: int = Field(default=1, ge=1)
    max_inflight_queries: int = Field(default=1, ge=1)
    subprocess_batch_size: int = Field(default=1, ge=1)
    dispatch_partial_batches: bool = False
    batch_gather_timeout_seconds: float | None = Field(default=None, ge=0)
    cache_query_results: bool = True


class LeaseJob(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    lease_id: str
    job_type: JobType
    payload: dict[str, Any]
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    upload: UploadTarget


class ResourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp_utc: str
    cpu_percent: float
    load_avg_1m: float
    memory_used_gb: float
    memory_percent: float
    disk_free_gb: float
    disk_percent: float
    gpu: list[dict[str, Any]] = Field(default_factory=list)


class HeartbeatPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    worker_id: str
    status: WorkerStatus
    active_job_id: str | None = None
    uptime_seconds: float
    max_active_jobs: int = 1
    supported_job_types: list[JobType] = Field(
        default_factory=lambda: ["predict_batch", "variant_batch"]
    )
    resources: ResourceSnapshot


class JobEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    worker_id: str
    lease_id: str
    event: EventType
    timestamp_utc: str
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    resources: ResourceSnapshot | None = None


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    lease_id: str
    worker_id: str
    job_type: JobType
    status: Literal["completed", "failed"]
    created_at_utc: str
    work_dir: str
    archive_path: str | None = None
    archive_sha256: str | None = None
    archive_size_bytes: int | None = None
    generated_inputs: list[str] = Field(default_factory=list)
    output_dir: str | None = None
    log_path: str | None = None
    telemetry_path: str | None = None
    error: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    upload_response: dict[str, Any] | None = None
