$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$platform = if ($IsWindows) { "windows" } elseif ($IsLinux) { "linux" } elseif ($IsMacOS) { "macos" } else { "unknown" }
$pythonTag = python -c "import sys; print(f'py{sys.version_info[0]}{sys.version_info[1]}')"
$depsDir = Join-Path $projectRoot (Join-Path ".test_deps" "$platform-$pythonTag")
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
    python -c "import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in ('pytest', 'pandas', 'coverage')) else 1)"
    if ($LASTEXITCODE -eq 0) {
        $env:PYTHONPATH = $helpersDir
        python -m pytest $testsDir --cov=of_notebook_lib --cov-report=term-missing -q
        exit $LASTEXITCODE
    }

    python -m pip install --target $depsDir -r $requirementsPath
}

$env:PYTHONPATH = "$depsDir;$helpersDir"
python -m pytest $testsDir --cov=of_notebook_lib --cov-report=term-missing -q
