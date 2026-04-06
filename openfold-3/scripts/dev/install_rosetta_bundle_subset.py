#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path


DEFAULT_MEMBERS = (
    "rosetta.binary.ubuntu.release-408/main/source/bin/score_jd2.static.linuxgccrelease",
    "rosetta.binary.ubuntu.release-408/main/source/bin/rosetta_scripts.static.linuxgccrelease",
    "rosetta.binary.ubuntu.release-408/main/source/bin/relax.static.linuxgccrelease",
    "rosetta.binary.ubuntu.release-408/main/source/build/src/release/linux/5.4/64/x86/gcc/7/static/score_jd2.static.linuxgccrelease",
    "rosetta.binary.ubuntu.release-408/main/source/build/src/release/linux/5.4/64/x86/gcc/7/static/rosetta_scripts.static.linuxgccrelease",
    "rosetta.binary.ubuntu.release-408/main/source/build/src/release/linux/5.4/64/x86/gcc/7/static/relax.static.linuxgccrelease",
    "rosetta.binary.ubuntu.release-408/main/database",
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
    return parser.parse_args()


def extract_subset(archive: Path, install_root: Path, members: tuple[str, ...]) -> tuple[list[str], list[str]]:
    install_root.mkdir(parents=True, exist_ok=True)
    wanted = set(members)
    found: set[str] = set()
    with tarfile.open(archive, "r|bz2") as tf:
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
    bundle_root = install_root / "rosetta.binary.ubuntu.release-408"
    source_bin = bundle_root / "main/source/bin"
    database = bundle_root / "main/database"
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
    members = tuple(args.members) if args.members else DEFAULT_MEMBERS
    found, missing = extract_subset(args.archive, args.install_root, members)
    print(f"FOUND={len(found)}")
    for item in missing:
        print(f"MISSING\t{item}")
    ensure_tool_symlinks(args.install_root)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
