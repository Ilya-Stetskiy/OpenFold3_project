from __future__ import annotations

import os
import shutil
import subprocess
import sys
from subprocess import CalledProcessError
from dataclasses import dataclass
from pathlib import Path


def _repo_roots(start: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for candidate in (start.resolve(), *start.resolve().parents):
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _first_matching_files(search_roots: tuple[Path, ...], patterns: tuple[str, ...]) -> Path | None:
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for candidate in root.glob(pattern):
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate.is_file() or candidate.is_symlink():
                    return candidate.resolve()
    return None


def _find_rosetta_database_near(binary_path: Path) -> Path | None:
    for parent in (binary_path.resolve().parent, *binary_path.resolve().parents):
        candidate = parent / "database"
        if candidate.is_dir():
            return candidate.resolve()
    return None


@dataclass(frozen=True)
class ToolStatus:
    name: str
    found: bool
    path: str | None
    source: str
    details: str | None = None


def resolve_foldx_binary(project_dir: Path) -> ToolStatus:
    override = os.environ.get("FOLDX_BINARY")
    if override:
        resolved = shutil.which(override) or (override if Path(override).exists() else None)
        if resolved is not None:
            return ToolStatus("foldx", True, str(Path(resolved).resolve()), "env:FOLDX_BINARY")
    from_path = shutil.which("foldx")
    if from_path is not None:
        return ToolStatus("foldx", True, str(Path(from_path).resolve()), "PATH")
    roots = _repo_roots(project_dir)
    candidate = _first_existing([root / "tools" / "bin" / "foldx" for root in roots])
    if candidate is not None:
        return ToolStatus("foldx", True, str(candidate), "repo:tools/bin/foldx")
    candidate = _first_matching_files(
        roots,
        (
            "foldx/**/foldx",
            "foldx/**/foldx_*",
            "tools/**/foldx",
            "tools/**/foldx_*",
        ),
    )
    if candidate is not None:
        return ToolStatus("foldx", True, str(candidate), "repo:foldx/**")
    return ToolStatus(
        "foldx",
        False,
        None,
        "missing",
        "Ожидается FOLDX_BINARY, PATH, tools/bin/foldx или локальная папка foldx/",
    )


def resolve_rosetta_score_binary(project_dir: Path) -> ToolStatus:
    override = os.environ.get("ROSETTA_SCORE_JD2_BINARY")
    if override:
        resolved = shutil.which(override) or (override if Path(override).exists() else None)
        if resolved is not None:
            return ToolStatus(
                "rosetta_score_jd2",
                True,
                str(Path(resolved).resolve()),
                "env:ROSETTA_SCORE_JD2_BINARY",
            )
    from_path = shutil.which("score_jd2")
    if from_path is not None:
        return ToolStatus("rosetta_score_jd2", True, str(Path(from_path).resolve()), "PATH")
    roots = _repo_roots(project_dir)
    candidate = _first_existing([root / "tools" / "bin" / "score_jd2" for root in roots])
    if candidate is not None:
        return ToolStatus("rosetta_score_jd2", True, str(candidate), "repo:tools/bin/score_jd2")
    candidate = _first_matching_files(
        roots,
        (
            "tools/**/score_jd2.static.linuxgccrelease",
            "tools/**/score_jd2.*linuxgccrelease",
            "rosetta*/**/score_jd2.static.linuxgccrelease",
            "rosetta*/**/score_jd2.*linuxgccrelease",
        ),
    )
    if candidate is not None:
        return ToolStatus("rosetta_score_jd2", True, str(candidate), "repo:rosetta/**/score_jd2")
    return ToolStatus(
        "rosetta_score_jd2",
        False,
        None,
        "missing",
        "Ожидается ROSETTA_SCORE_JD2_BINARY, PATH или локальный Rosetta subset в tools/",
    )


def resolve_rosetta_database(project_dir: Path, score_binary: str | None = None) -> ToolStatus:
    override = os.environ.get("ROSETTA_DATABASE")
    if override and Path(override).is_dir():
        return ToolStatus("rosetta_database", True, str(Path(override).resolve()), "env:ROSETTA_DATABASE")
    if score_binary is not None:
        inferred = _find_rosetta_database_near(Path(score_binary))
        if inferred is not None:
            return ToolStatus("rosetta_database", True, str(inferred), "relative_to_score_jd2")
    for root in _repo_roots(project_dir):
        for parent_name in ("tools", "rosetta", "rosetta3", "rosetta_bundle"):
            parent = root / parent_name
            if not parent.exists():
                continue
            matches = sorted(parent.glob("**/database"))
            for match in matches:
                if match.is_dir():
                    return ToolStatus("rosetta_database", True, str(match.resolve()), f"repo:{parent_name}/**/database")
    return ToolStatus(
        "rosetta_database",
        False,
        None,
        "missing",
        "Ожидается ROSETTA_DATABASE или локальный Rosetta subset с database/",
    )


def collect_ddg_tool_status(project_dir: Path) -> dict[str, ToolStatus]:
    foldx = resolve_foldx_binary(project_dir)
    rosetta_score = resolve_rosetta_score_binary(project_dir)
    rosetta_db = resolve_rosetta_database(project_dir, rosetta_score.path)
    return {
        "foldx": foldx,
        "rosetta_score_jd2": rosetta_score,
        "rosetta_database": rosetta_db,
    }


def ddg_tool_status_rows(project_dir: Path) -> list[tuple[str, object]]:
    statuses = collect_ddg_tool_status(project_dir)
    rows: list[tuple[str, object]] = []
    for key in ("foldx", "rosetta_score_jd2", "rosetta_database"):
        status = statuses[key]
        rows.append((f"{status.name}: found", status.found))
        rows.append((f"{status.name}: source", status.source))
        rows.append((f"{status.name}: path", status.path or "None"))
        if status.details:
            rows.append((f"{status.name}: details", status.details))
    return rows


def export_ddg_tool_env(project_dir: Path) -> dict[str, str]:
    statuses = collect_ddg_tool_status(project_dir)
    exported: dict[str, str] = {}
    foldx = statuses["foldx"]
    if foldx.path:
        exported["FOLDX_BINARY"] = foldx.path
    rosetta_score = statuses["rosetta_score_jd2"]
    if rosetta_score.path:
        exported["ROSETTA_SCORE_JD2_BINARY"] = rosetta_score.path
    rosetta_db = statuses["rosetta_database"]
    if rosetta_db.path:
        exported["ROSETTA_DATABASE"] = rosetta_db.path
    for key, value in exported.items():
        os.environ[key] = value
    return exported


def install_rosetta_subset(
    *,
    project_dir: Path,
    archive_path: Path,
    install_root: Path | None = None,
    python_executable: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    repo_root = project_dir.resolve()
    script_path = repo_root / "openfold-3" / "scripts" / "dev" / "install_rosetta_bundle_subset.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Rosetta installer script not found: {script_path}")
    if not archive_path.exists():
        raise FileNotFoundError(f"Rosetta archive not found: {archive_path}")
    resolved_install_root = (
        install_root.resolve()
        if install_root is not None
        else (repo_root / "tools" / "rosetta3.15_min").resolve()
    )
    resolved_python = (
        python_executable.resolve()
        if python_executable is not None
        else Path(sys.executable).resolve()
    )
    command = [
        str(resolved_python),
        str(script_path),
        "--archive",
        str(archive_path.resolve()),
        "--install-root",
        str(resolved_install_root),
    ]
    try:
        return subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except CalledProcessError as exc:
        stdout_tail = exc.stdout[-4000:] if exc.stdout else ""
        stderr_tail = exc.stderr[-4000:] if exc.stderr else ""
        raise RuntimeError(
            "Rosetta subset install failed.\n"
            f"command: {' '.join(command)}\n"
            f"stdout_tail:\n{stdout_tail}\n"
            f"stderr_tail:\n{stderr_tail}"
        ) from exc
