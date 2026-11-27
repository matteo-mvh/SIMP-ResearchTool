@echo off
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "SIMP Control Center" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0start_app.py"
    exit /b
)
rem Launch in the background without a console where pythonw is available.
where pythonw >nul 2>nul
if not errorlevel 1 (
    start "SIMP Control Center" pythonw "%~dp0start_app.py"
) else (
    python start_app.py
    if errorlevel 1 (
        echo.
        echo Launch failed. Check that Python 3.11+ and Tkinter are installed.
        pause
    )
)
