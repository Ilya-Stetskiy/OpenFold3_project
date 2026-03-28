$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyTag = python -c "import sys; print(f'py{sys.version_info.major}{sys.version_info.minor}')"
$platformTag = python -c "import platform; print(platform.system().lower())"
$depsDir = Join-Path $projectRoot ".test_deps\$platformTag" + "_$pyTag"
$helpersDir = Join-Path $projectRoot "helpers"
$testsDir = Join-Path $projectRoot "tests"
$requirementsPath = Join-Path $projectRoot "requirements-test.txt"

if (-not (Test-Path $depsDir)) {
    New-Item -ItemType Directory -Path $depsDir | Out-Null
}

$pytestDir = Join-Path $depsDir "pytest"
$pandasDir = Join-Path $depsDir "pandas"
$coverageDir = Join-Path $depsDir "coverage"

$envReady = $true
try {
    python -c "import coverage, pandas, pytest" | Out-Null
} catch {
    $envReady = $false
}

if ((-not $envReady) -and ((-not (Test-Path $pytestDir)) -or (-not (Test-Path $pandasDir)) -or (-not (Test-Path $coverageDir)))) {
    python -m pip install --target $depsDir -r $requirementsPath
}

$env:PYTHONPATH = "$depsDir;$helpersDir"
python -m pytest $testsDir --cov=of_notebook_lib --cov-report=term-missing -q
