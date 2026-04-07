#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPERS_DIR = REPO_ROOT.parent / "openfold_notebooks" / "helpers"
for path in (REPO_ROOT, HELPERS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from of_notebook_lib.ddg_tools import collect_ddg_tool_status, export_ddg_tool_env, install_rosetta_subset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight and optional Rosetta subset install for ddG stand."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=REPO_ROOT.parent,
        help="Repo root that contains openfold-3 and openfold_notebooks",
    )
    parser.add_argument(
        "--rosetta-archive",
        type=Path,
        help="Path to Rosetta bundle archive. If provided and Rosetta is missing, install a minimal subset.",
    )
    parser.add_argument(
        "--rosetta-install-root",
        type=Path,
        help="Install root for the Rosetta subset. Defaults to tools/rosetta3.15_min under project-dir.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit non-zero unless FoldX, score_jd2, and Rosetta database are all available.",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    statuses = collect_ddg_tool_status(project_dir)
    need_rosetta = (
        args.rosetta_archive is not None
        and (
            not statuses["rosetta_score_jd2"].found
            or not statuses["rosetta_database"].found
        )
    )
    install_summary: dict[str, str] | None = None
    if need_rosetta:
        result = install_rosetta_subset(
            project_dir=project_dir,
            archive_path=args.rosetta_archive.resolve(),
            install_root=args.rosetta_install_root,
        )
        install_summary = {
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        statuses = collect_ddg_tool_status(project_dir)

    exported = export_ddg_tool_env(project_dir)
    payload = {
        "project_dir": str(project_dir),
        "statuses": {
            key: {
                "found": value.found,
                "path": value.path,
                "source": value.source,
                "details": value.details,
            }
            for key, value in statuses.items()
        },
        "exported_env": exported,
        "install_summary": install_summary,
    }
    print(json.dumps(payload, indent=2))
    if args.require_all and not all(item.found for item in statuses.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
