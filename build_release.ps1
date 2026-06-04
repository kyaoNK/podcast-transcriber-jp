# podcast-transcriber-jp 配布ZIPビルドスクリプト
# 使い方: PowerShellで .\build_release.ps1
# 出力:    dist\podcast-transcriber-jp.zip
#
# tkinter 込みの python-build-standalone を使うため、Windows embeddable Python の
# tkinter 欠落問題を回避している。CPU/GPU は ctranslate2 が自動検出するので
# 配布物は1つだけで両対応する。

[CmdletBinding()]
param(
    [string]$PbsTag = '20260602',
    [string]$PbsVersion = '3.11.15'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$build = Join-Path $root 'build'
$dist = Join-Path $root 'dist'
$pbsAsset = "cpython-$PbsVersion+$PbsTag-x86_64-pc-windows-msvc-install_only.tar.gz"
$pbsUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/$PbsTag/$pbsAsset"

Write-Host "[build] python-build-standalone $PbsVersion+$PbsTag"

# 1. クリーン
if (Test-Path $build) { Remove-Item $build -Recurse -Force }
New-Item -ItemType Directory -Path $build -Force | Out-Null
New-Item -ItemType Directory -Path $dist -Force | Out-Null
$pythonDir = Join-Path $build 'python'

# 2. python-build-standalone を取得・展開（tkinter 込み、pip 込み）
Write-Host '[build] downloading python-build-standalone...'
$pbsTar = Join-Path $build 'python.tar.gz'
Invoke-WebRequest -Uri $pbsUrl -OutFile $pbsTar
Write-Host '[build] extracting...'
tar -xzf $pbsTar -C $build
if ($LASTEXITCODE -ne 0) { throw "tar extract failed (exit $LASTEXITCODE)" }
Remove-Item $pbsTar

# 3. requirements インストール（pip は同梱済み）
Write-Host '[build] installing requirements...'
$reqFile = Join-Path $root 'apps\transcribe\requirements.txt'
& (Join-Path $pythonDir 'python.exe') -m pip install -r $reqFile --no-warn-script-location --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }

# 4. sitecustomize.py を仕込む（起動時にソースパスを sys.path に追加）
#    python-build-standalone は標準 Python 同様 site が有効なので sitecustomize が
#    自動的に読まれる。ここで apps/transcribe と core/src を sys.path に追加する。
$sitePackages = Join-Path $pythonDir 'Lib\site-packages'
$siteCustomize = @'
import os, sys

# sitecustomize.py の場所: <ZIP_ROOT>/python/Lib/site-packages/sitecustomize.py
# 3階層上 = <ZIP_ROOT>
_this_dir = os.path.dirname(os.path.abspath(__file__))
_zip_root = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
sys.path.insert(0, os.path.join(_zip_root, 'apps', 'transcribe'))
sys.path.insert(0, os.path.join(_zip_root, 'core', 'src'))
'@
Set-Content -Path (Join-Path $sitePackages 'sitecustomize.py') -Value $siteCustomize -Encoding UTF8

# 5. ソースコードをコピー
Write-Host '[build] copying source files...'
Copy-Item -Path (Join-Path $root 'apps') -Destination (Join-Path $build 'apps') -Recurse
Copy-Item -Path (Join-Path $root 'core') -Destination (Join-Path $build 'core') -Recurse
foreach ($f in 'start.bat','README.md','LICENSE','NOTICE.txt') {
    $src = Join-Path $root $f
    if (Test-Path $src) { Copy-Item -Path $src -Destination (Join-Path $build $f) }
}

# 6. 不要ファイル除去（__pycache__、pyc）
Get-ChildItem -Path $build -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
Get-ChildItem -Path $build -Recurse -File -Include *.pyc,*.pyo | Remove-Item -Force

# 7. ZIP 化
$zipName = 'podcast-transcriber-jp.zip'
$zipPath = Join-Path $dist $zipName
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Write-Host "[build] compressing -> $zipPath"
Compress-Archive -Path (Join-Path $build '*') -DestinationPath $zipPath -CompressionLevel Optimal

$sizeMB = [Math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "[build] done: $zipPath ($sizeMB MB)"
