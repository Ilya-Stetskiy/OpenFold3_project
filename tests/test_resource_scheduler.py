import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "run"))


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


resource_scheduler = load_module("scripts/run/resource_scheduler.py", "resource_scheduler")
adaptive_runner = load_module("scripts/run/run_cpu_heavy_adaptive.py", "run_cpu_heavy_adaptive")


def test_cpu_heavy_plan_allocates_conservative_workers():
    config = {
        "defaults": {
            "reserve_cpu_cores": 1,
            "max_total_workers": "auto",
            "min_free_ram_gb": 8,
            "min_free_disk_gb": 20,
            "load_average_limit_per_core": 1.0,
            "allow_parallel_tools": True,
        },
        "cpu_heavy": {
            "tools": {
                "rosetta_experimental": {
                    "enabled": True,
                    "script_path": "scripts/executors/run_rosetta_experimental.py",
                    "cpu_per_job": 2,
                    "ram_gb_per_job": 4,
                    "min_workers": 1,
                    "max_workers": 4,
                    "priority": 10,
                },
                "foldx_experimental": {
                    "enabled": True,
                    "script_path": "scripts/executors/run_foldx_experimental.py",
                    "cpu_per_job": 1,
                    "ram_gb_per_job": 1,
                    "min_workers": 1,
                    "max_workers": 8,
                    "priority": 20,
                },
            }
        },
    }
    snapshot = resource_scheduler.ResourceSnapshot(
        cpu_count=8,
        load1=0.0,
        load5=0.0,
        load15=0.0,
        total_ram_gb=32.0,
        available_ram_gb=32.0,
        disk_free_gb=100.0,
        disk_path="/tmp",
    )

    plan = resource_scheduler.build_cpu_heavy_plan(config, snapshot)
    workers = {tool["tool_name"]: tool["workers"] for tool in plan["tools"]}

    assert plan["worker_budget"] == 7
    assert workers["rosetta_experimental"] == 3
    assert workers["foldx_experimental"] == 1


def test_low_ram_limits_rosetta_workers():
    config = {
        "defaults": {"min_free_ram_gb": 8, "min_free_disk_gb": 20, "allow_parallel_tools": True},
        "cpu_heavy": {
            "tools": {
                "rosetta_experimental": {
                    "enabled": True,
                    "script_path": "scripts/executors/run_rosetta_experimental.py",
                    "cpu_per_job": 2,
                    "ram_gb_per_job": 4,
                    "min_workers": 1,
                    "max_workers": 4,
                    "priority": 10,
                }
            }
        },
    }
    snapshot = resource_scheduler.ResourceSnapshot(
        cpu_count=16,
        load1=0.0,
        load5=0.0,
        load15=0.0,
        total_ram_gb=12.0,
        available_ram_gb=10.0,
        disk_free_gb=100.0,
        disk_path="/tmp",
    )

    plan = resource_scheduler.build_cpu_heavy_plan(config, snapshot)

    assert plan["tools"][0]["workers"] == 0
    assert "insufficient budget" in plan["tools"][0]["reason"]


def test_build_executor_command_passes_adaptive_workers():
    command = adaptive_runner.build_executor_command(
        tool={"script_path": "scripts/executors/run_foldx_experimental.py"},
        dataset="current_internal",
        mode="run",
        workers=5,
        max_jobs=10,
        resume=True,
        force=False,
        no_progress_bar=True,
    )

    assert "--run" in command
    assert command[command.index("--workers") + 1] == "5"
    assert command[command.index("--max-jobs") + 1] == "10"
    assert "--resume" in command
    assert "--no-progress-bar" in command
