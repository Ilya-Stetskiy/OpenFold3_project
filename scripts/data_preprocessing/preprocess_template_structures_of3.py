"""
Script to separately preprocess template structure files into structure arrays.
"""

from pathlib import Path

import click

if __package__ in (None, ""):
    _path = __import__("pathlib").Path(__file__).resolve()
    for _candidate in (_path.parent, *_path.parents):
        if (_candidate / "openfold3").exists() or (_candidate / "scripts").exists():
            __import__("sys").path.insert(0, str(_candidate))
            break

from openfold3.core.config import config_utils
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
    TemplateStructurePreprocessor,
)


@click.command()
@click.option(
    "--runner_yaml",
    required=True,
    help=(
        "Runner.yml file to be parsed into settings for the template structure "
        "array preprocessor pipeline."
    ),
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
def main(runner_yaml: Path):
    # Load runner YAML and extract template_preprocessor_settings if present
    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()

    # Extract template_preprocessor_settings from runner YAML or use default
    template_preprocessor_kwargs = runner_args.get("template_preprocessor_settings", {})

    # Create template preprocessor settings with defaults, overriding with YAML values
    template_preprocessor_settings = TemplatePreprocessorSettings(
        **template_preprocessor_kwargs
    )

    template_structure_preprocessor = TemplateStructurePreprocessor(
        config=template_preprocessor_settings
    )
    template_structure_preprocessor()


if __name__ == "__main__":
    main()

