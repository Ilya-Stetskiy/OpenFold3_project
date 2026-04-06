from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENFOLD_REPO_ROOT = REPO_ROOT.parent / "openfold-3"

COPY_DIRS = [
    "openfold3_length_benchmark",
    "openfold3_runtime_benchmark",
]
COPY_FILES = [
    "OpenFold3_Length_Benchmark.ipynb",
    "OpenFold3_Runtime_Benchmark.ipynb",
    "helpers/of_notebook_lib/config.py",
    "README.md",
    "requirements-test.txt",
    "run_tests.sh",
    "run_tests.ps1",
]
COPY_RUNTIME_SOURCE_FILES = [
    "openfold3/core/utils/profile_events.py",
    "openfold3/core/utils/callbacks.py",
    "openfold3/entry_points/experiment_runner.py",
    "openfold3/projects/of3_all_atom/runner.py",
]
RUNTIME_MIRROR_FILES = [
    "core/utils/profile_events.py",
    "core/utils/callbacks.py",
    "entry_points/experiment_runner.py",
    "projects/of3_all_atom/runner.py",
]
IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "cache",
    ".DS_Store",
}


def _ignore_filter(_src: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORE_NAMES}


def _copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    if src.resolve() == dst.resolve():
        print(f"SKIP same file {src}")
        return
    print(f"COPY {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    if src.resolve() == dst.resolve():
        print(f"SKIP same directory {src}")
        return
    print(f"SYNC {src} -> {dst}")
    if dry_run:
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore_filter)


def _replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found in {path}: {old!r}")
    return text.replace(old, new, 1)


def _patch_validator(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    target = 'model_update: ModelUpdate = ModelUpdate(presets=["predict", "pae_enabled"])'
    replacement = 'model_update: ModelUpdate = ModelUpdate(presets=["predict"])'
    if replacement in original:
        return False
    updated = _replace_once(original, target, replacement, path=path)
    path.write_text(updated, encoding="utf-8")
    return True


def _patch_colabfold_server(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original

    old_log = '    logger.warning(f"Using output directory: {output_directory} for ColabFold MSAs.")\n'
    new_log = '    logger.info("Using output directory: %s for ColabFold MSAs.", output_directory)\n'
    if old_log in updated:
        updated = updated.replace(old_log, new_log, 1)

    old_block = """        remapped = remap_colabfold_template_chain_ids(
            template_alignments=template_alignments,
            m_with_templates=m_with_templates,
            rep_ids=self.colabfold_mapper.rep_ids,
            rep_id_to_m=self.colabfold_mapper.rep_id_to_m,
        )
"""
    new_block = """        try:
            remapped = remap_colabfold_template_chain_ids(
                template_alignments=template_alignments,
                m_with_templates=m_with_templates,
                rep_ids=self.colabfold_mapper.rep_ids,
                rep_id_to_m=self.colabfold_mapper.rep_id_to_m,
            )
        except Exception as exc:
            logger.warning(
                "Failed to remap ColabFold template chain IDs; continuing without "
                "template alignments for this batch. Error: %s",
                exc,
            )
            return
"""
    if new_block in updated:
        changed = updated != original
        if changed:
            path.write_text(updated, encoding="utf-8")
        return changed

    updated = _replace_once(updated, old_block, new_block, path=path)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def _apply_runtime_patches(project_root: Path, *, dry_run: bool) -> None:
    patch_targets = [
        (
            project_root / "openfold3" / "entry_points" / "validator.py",
            _patch_validator,
        ),
        (
            project_root
            / "openfold3"
            / "core"
            / "data"
            / "tools"
            / "colabfold_msa_server.py",
            _patch_colabfold_server,
        ),
        (
            project_root
            / "runtime"
            / "openfold3"
            / "entry_points"
            / "validator.py",
            _patch_validator,
        ),
        (
            project_root
            / "runtime"
            / "openfold3"
            / "core"
            / "data"
            / "tools"
            / "colabfold_msa_server.py",
            _patch_colabfold_server,
        ),
    ]

    for target, patch_fn in patch_targets:
        if not target.exists():
            print(f"SKIP missing {target}")
            continue
        print(f"PATCH {target}")
        if dry_run:
            continue
        changed = patch_fn(target)
        if not changed:
            print(f"UNCHANGED {target}")


def sync_project(project_root: Path, *, dry_run: bool) -> None:
    for relative in COPY_FILES:
        src = REPO_ROOT / relative
        dst = project_root / relative
        _copy_file(src, dst, dry_run=dry_run)

    for relative in COPY_DIRS:
        src = REPO_ROOT / relative
        dst = project_root / relative
        _copy_tree(src, dst, dry_run=dry_run)

    for relative in COPY_RUNTIME_SOURCE_FILES:
        src = OPENFOLD_REPO_ROOT / relative
        dst = project_root / relative
        if not src.exists():
            print(f"SKIP missing source {src}")
            continue
        _copy_file(src, dst, dry_run=dry_run)

    for relative in RUNTIME_MIRROR_FILES:
        src = OPENFOLD_REPO_ROOT / "openfold3" / relative
        dst = project_root / "runtime" / "openfold3" / relative
        if not src.exists():
            print(f"SKIP missing source {src}")
            continue
        _copy_file(src, dst, dry_run=dry_run)

    _apply_runtime_patches(project_root, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Archived sync helper kept for manual recovery. "
            "It is not part of the active OpenFold3_project workflow."
        )
    )
    parser.add_argument(
        "project_root",
        type=Path,
        help="Path to the target OpenFold3_project directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without modifying files.",
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    if not project_root.exists():
        raise SystemExit(f"Target project root does not exist: {project_root}")

    sync_project(project_root, dry_run=args.dry_run)
    print(f"Done. Synced into {project_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
