$ErrorActionPreference = 'Stop'
python "$PSScriptRoot/main.py" --segmented "$PSScriptRoot/data/1998人民日报（分词）.txt" --pos "$PSScriptRoot/data/1998人民日报（带词性标注）.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python "$PSScriptRoot/verify.py"
exit $LASTEXITCODE
