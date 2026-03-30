from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "dev"
    / "install_rosetta_bundle_subset.py"
)
SPEC = importlib.util.spec_from_file_location("install_rosetta_bundle_subset", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extract_subset_extracts_requested_members(tmp_path):
    archive_path = tmp_path / "rosetta-mini.tar.bz2"
    source_dir = tmp_path / "source"
    wanted_file = source_dir / "rosetta.binary.ubuntu.release-408/main/source/build/src/release/linux/5.4/64/x86/gcc/7/static/score_jd2.static.linuxgccrelease"
    wanted_file.parent.mkdir(parents=True, exist_ok=True)
    wanted_file.write_text("fake-binary", encoding="utf-8")
    ignored_file = source_dir / "ignored.txt"
    ignored_file.write_text("ignore", encoding="utf-8")

    with tarfile.open(archive_path, "w:bz2") as tf:
        tf.add(
            wanted_file,
            arcname="rosetta.binary.ubuntu.release-408/main/source/build/src/release/linux/5.4/64/x86/gcc/7/static/score_jd2.static.linuxgccrelease",
        )
        tf.add(ignored_file, arcname="ignored.txt")

    install_root = tmp_path / "install"
    found, missing = MODULE.extract_subset(
        archive_path,
        install_root,
        (
            "rosetta.binary.ubuntu.release-408/main/source/build/src/release/linux/5.4/64/x86/gcc/7/static/score_jd2.static.linuxgccrelease",
        ),
    )

    assert missing == []
    assert found == [
        "rosetta.binary.ubuntu.release-408/main/source/build/src/release/linux/5.4/64/x86/gcc/7/static/score_jd2.static.linuxgccrelease"
    ]
    assert (
        install_root
        / "rosetta.binary.ubuntu.release-408/main/source/build/src/release/linux/5.4/64/x86/gcc/7/static/score_jd2.static.linuxgccrelease"
    ).read_text(encoding="utf-8") == "fake-binary"


def test_ensure_tool_symlinks_creates_links_for_existing_targets(tmp_path):
    install_root = tmp_path / "tools" / "rosetta3.15_min"
    source_bin = install_root / "rosetta.binary.ubuntu.release-408/main/source/bin"
    source_bin.mkdir(parents=True, exist_ok=True)
    database = install_root / "rosetta.binary.ubuntu.release-408/main/database"
    database.mkdir(parents=True, exist_ok=True)

    score_bin = source_bin / "score_jd2.static.linuxgccrelease"
    score_bin.write_text("fake", encoding="utf-8")
    rosetta_scripts_bin = source_bin / "rosetta_scripts.static.linuxgccrelease"
    rosetta_scripts_bin.write_text("fake", encoding="utf-8")

    created = MODULE.ensure_tool_symlinks(install_root)

    tools_bin = install_root.parent / "bin"
    score_link = tools_bin / "score_jd2"
    rosetta_scripts_link = tools_bin / "rosetta_scripts"
    relax_link = tools_bin / "relax"

    assert score_link in created
    assert rosetta_scripts_link in created
    assert relax_link not in created
    assert score_link.is_symlink()
    assert rosetta_scripts_link.is_symlink()
    assert score_link.resolve() == score_bin.resolve()
    assert rosetta_scripts_link.resolve() == rosetta_scripts_bin.resolve()
