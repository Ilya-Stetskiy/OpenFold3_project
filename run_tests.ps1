$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$platform = if ($IsWindows) { "windows" } elseif ($IsLinux) { "linux" } elseif ($IsMacOS) { "macos" } else { "unknown" }
$pythonTag = python -c "import sys; print(f'py{sys.version_info[0]}{sys.version_info[1]}')"
$depsDir = Join-Path $projectRoot (Join-Path ".test_deps" "$platform-$pythonTag")
$packageRoot = $projectRoot
$helpersDir = Join-Path $projectRoot "helpers"
$testsDir = Join-Path $projectRoot "tests"
$lengthBenchTestsDir = Join-Path $projectRoot "openfold3_length_benchmark/tests"
$runtimeBenchTestsDir = Join-Path $projectRoot "openfold3_runtime_benchmark/tests"
$requirementsPath = Join-Path $projectRoot "requirements-test.txt"

if (-not (Test-Path $depsDir)) {
    New-Item -ItemType Directory -Path $depsDir | Out-Null
}

$pytestDir = Join-Path $depsDir "pytest"
$pandasDir = Join-Path $depsDir "pandas"
$pytestCovDir = Join-Path $depsDir "pytest_cov"

if ((-not (Test-Path $pytestDir)) -or (-not (Test-Path $pandasDir)) -or (-not (Test-Path $pytestCovDir))) {
    python -c "import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in ('pytest', 'pandas', 'coverage', 'numpy', 'requests')) else 1)"
    if ($LASTEXITCODE -eq 0) {
        $env:PYTHONPATH = "$packageRoot;$helpersDir"
        python -m pytest $testsDir $lengthBenchTestsDir $runtimeBenchTestsDir --cov=of_notebook_lib --cov=openfold3_length_benchmark --cov=openfold3_runtime_benchmark --cov-report=term-missing -q
        exit $LASTEXITCODE
    }

    python -m pip install --target $depsDir -r $requirementsPath
}

$env:PYTHONPATH = "$depsDir;$packageRoot;$helpersDir"
python -m pytest $testsDir $lengthBenchTestsDir $runtimeBenchTestsDir --cov=of_notebook_lib --cov=openfold3_length_benchmark --cov=openfold3_runtime_benchmark --cov-report=term-missing -q
