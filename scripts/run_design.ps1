# 시안별 실행 (Windows).  기존: .\scripts\run_design.ps1 classic 8601 · 추천안: d 8602 · 테이스트안: f 8603
param([ValidateSet("classic", "d", "f")][string]$Design = "classic", [int]$Port = 8601)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:RESOURCE_DESIGN = $Design
$env:PYTHONPATH = $root  # 서버를 켠 직후 하위 화면 주소로 바로 들어와도 services 를 찾게
$args_ = @("run", "ui/app.py", "--server.port=$Port", "--server.address=127.0.0.1", "--server.headless=true")
if ($Design -ne "classic") { $args_ += "--theme.base=ui/designs/$Design.toml" }
& "$root\.venv\Scripts\streamlit.exe" @args_
