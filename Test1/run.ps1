$ErrorActionPreference = 'Stop'
# 将工作目录设为 NLP 根目录，Python 参数与输出路径均相对该目录。
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    python Test1/main.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python Test1/check_modules.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python Test1/verify.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
