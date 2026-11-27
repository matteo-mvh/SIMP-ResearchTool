@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" start_worker.py
) else (
    python start_worker.py
)
pause
