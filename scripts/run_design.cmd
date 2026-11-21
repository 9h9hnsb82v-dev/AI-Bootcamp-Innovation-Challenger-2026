@echo off
rem Run one design variant (works regardless of PowerShell execution policy). ASCII only: cmd reads this file as CP949.
rem   classic: scripts\run_design.cmd classic 8601   D: scripts\run_design.cmd d 8602   F: scripts\run_design.cmd f 8603
setlocal
set "DESIGN=%~1"
if "%DESIGN%"=="" set "DESIGN=classic"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8601"
if /I not "%DESIGN%"=="classic" if /I not "%DESIGN%"=="d" if /I not "%DESIGN%"=="f" (echo design must be classic, d or f & exit /b 2)
cd /d "%~dp0.."
set "RESOURCE_DESIGN=%DESIGN%"
set "PYTHONPATH=%CD%"
set "THEME="
if /I not "%DESIGN%"=="classic" set "THEME=--theme.base=ui/designs/%DESIGN%.toml"
".venv\Scripts\streamlit.exe" run ui/app.py --server.port=%PORT% --server.address=127.0.0.1 --server.headless=true %THEME%
