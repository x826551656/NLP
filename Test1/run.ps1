$ErrorActionPreference = 'Stop'
python "$PSScriptRoot/main.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python "$PSScriptRoot/check_modules.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python "$PSScriptRoot/verify.py"
exit $LASTEXITCODE
