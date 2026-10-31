@echo off
rem Windows + NVIDIA RTX: install environment, then train. Run the same command again to resume.
rem   run_train.bat                 (default: hybridnets tln_large yolop yolopv2)
rem   run_train.bat --models hybridnets
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if "%TORCH_INDEX%"=="" set TORCH_INDEX=https://download.pytorch.org/whl/cu128
if exist .venv\.ready goto run
set PY=
py -3.10 -c "" >nul 2>&1 && set "PY=py -3.10"
if not defined PY py -3.11 -c "" >nul 2>&1 && set "PY=py -3.11"
if not defined PY py -3.12 -c "" >nul 2>&1 && set "PY=py -3.12"
if not defined PY python -c "" >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Python 3.10~3.12 is not installed. Install it from python.org and run again.
  pause
  exit /b 1
)
%PY% -m venv .venv || goto fail
.venv\Scripts\python -m pip install -U pip || goto fail
.venv\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url %TORCH_INDEX% || goto fail
.venv\Scripts\python -m pip install -r requirements_rtx.txt || goto fail
type nul > .venv\.ready
:run
.venv\Scripts\python scripts\run_train.py %*
if %errorlevel% neq 0 goto fail
exit /b 0
:fail
echo FAILED. Check the error above. Run this file again to resume.
pause
exit /b 1
