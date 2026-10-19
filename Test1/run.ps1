$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    python Test1/test_experiment.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python Test1/main.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python Test1/verify.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
