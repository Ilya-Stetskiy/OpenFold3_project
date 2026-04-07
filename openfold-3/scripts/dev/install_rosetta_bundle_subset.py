#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

DEFAULT_SUFFIXES = (
    "/main/source/bin/score_jd2.static.linuxgccrelease",
    "/main/source/bin/rosetta_scripts.static.linuxgccrelease",
    "/main/source/bin/relax.static.linuxgccrelease",
    "/main/database",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a minimal Rosetta subset needed by the ddG testbench."
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument(
        "--member",
        action="append",
        dest="members",
        default=[],
        help="Exact tar member to extract. Can be repeated.",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        dest="suffixes",
        default=[],
        help="Tar member suffix to extract. Can be repeated. Defaults cover score_jd2, rosetta_scripts, relax, and database.",
    )
    return parser.parse_args()


def _tar_mode_for_archive(archive: Path) -> str:
    name = archive.name.lower()
    if name.endswith((".tar.gz", ".tgz")):
        return "r:gz"
    if name.endswith((".tar.bz2", ".tbz2")):
        return "r:bz2"
    return "r:*"


def _resolve_members(
    archive: Path,
    *,
    exact_members: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    if exact_members:
        wanted = list(exact_members)
        found: set[str] = set()
        with tarfile.open(archive, _tar_mode_for_archive(archive)) as tf:
            names = {member.name for member in tf.getmembers()}
        missing = [member for member in wanted if member not in names]
        return wanted, missing

    if not suffixes:
        raise ValueError("No exact members or suffixes were provided")

    matched: dict[str, str] = {}
    with tarfile.open(archive, _tar_mode_for_archive(archive)) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                if member.isdir():
                    for suffix in suffixes:
                        if member.name.endswith(suffix):
                            matched.setdefault(suffix, member.name)
                continue
            for suffix in suffixes:
                if member.name.endswith(suffix):
                    matched.setdefault(suffix, member.name)
    resolved = [matched[suffix] for suffix in suffixes if suffix in matched]
    missing = [suffix for suffix in suffixes if suffix not in matched]
    return resolved, missing


def extract_subset(archive: Path, install_root: Path, members: tuple[str, ...]) -> tuple[list[str], list[str]]:
    install_root.mkdir(parents=True, exist_ok=True)
    wanted = set(members)
    found: set[str] = set()
    with tarfile.open(archive, _tar_mode_for_archive(archive)) as tf:
        for member in tf:
            if member.name not in wanted:
                continue
            tf.extract(member, path=install_root)
            found.add(member.name)
            print(f"EXTRACTED\t{member.name}")
            if found == wanted:
                break
    missing = sorted(wanted - found)
    return sorted(found), missing


def ensure_tool_symlinks(install_root: Path) -> list[Path]:
    candidates = sorted(install_root.glob("**/main/source/bin"))
    if not candidates:
        print("WARN\tCould not find main/source/bin under install root")
        return []
    source_bin = candidates[0]
    bundle_root = source_bin.parents[2]
    database = bundle_root / "database"
    tools_bin = install_root.parent / "bin"
    tools_bin.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for name, target_name in (
        ("score_jd2", "score_jd2.static.linuxgccrelease"),
        ("rosetta_scripts", "rosetta_scripts.static.linuxgccrelease"),
        ("relax", "relax.static.linuxgccrelease"),
    ):
        target = source_bin / target_name
        if not target.exists():
            continue
        link = tools_bin / name
        if link.exists() or link.is_symlink():
            link.unlink()
        relative_target = Path(os.path.relpath(target, tools_bin))
        link.symlink_to(relative_target)
        created.append(link)
        print(f"LINKED\t{link} -> {relative_target}")
    if database.exists():
        print(f"DATABASE\t{database}")
    return created


def main() -> int:
    args = parse_args()
    suffixes = tuple(args.suffixes) if args.suffixes else DEFAULT_SUFFIXES
    members, missing = _resolve_members(
        args.archive,
        exact_members=tuple(args.members),
        suffixes=suffixes,
    )
    if not members:
        print("FOUND=0")
        for item in missing:
            print(f"MISSING\t{item}")
        return 1
    found, extract_missing = extract_subset(args.archive, args.install_root, tuple(members))
    print(f"FOUND={len(found)}")
    for item in (missing + extract_missing):
        print(f"MISSING\t{item}")
    ensure_tool_symlinks(args.install_root)
    return 0 if not (missing or extract_missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
