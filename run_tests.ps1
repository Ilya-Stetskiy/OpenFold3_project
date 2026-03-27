$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$depsDir = Join-Path $projectRoot ".test_deps"
$helpersDir = Join-Path $projectRoot "helpers"
$testsDir = Join-Path $projectRoot "tests"
$requirementsPath = Join-Path $projectRoot "requirements-test.txt"

if (-not (Test-Path $depsDir)) {
    New-Item -ItemType Directory -Path $depsDir | Out-Null
}

$pytestDir = Join-Path $depsDir "pytest"
$pandasDir = Join-Path $depsDir "pandas"
$pytestCovDir = Join-Path $depsDir "pytest_cov"

if ((-not (Test-Path $pytestDir)) -or (-not (Test-Path $pandasDir)) -or (-not (Test-Path $pytestCovDir))) {
    python -m pip install --target $depsDir -r $requirementsPath
}

$env:PYTHONPATH = "$depsDir;$helpersDir"
python -m pytest $testsDir --cov=of_notebook_lib --cov-report=term-missing -q
