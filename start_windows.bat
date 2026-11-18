@echo off
setlocal
cd /d "%~dp0"
title GradeForge
echo.
echo  GradeForge - starting. The first run installs things and takes a few minutes.
echo  Leave this window open while you use GradeForge. Close it to stop.
echo.

REM ---- find a Python version that has ready-made packages (3.10 - 3.13, 64-bit) ----
set PY=
for %%V in (3.13 3.12 3.11 3.10) do (
  if "%PY%"=="" (
    py -%%V -c "import struct,sys;sys.exit(0 if struct.calcsize('P')==8 else 1)" >nul 2>nul && set PY=py -%%V
  )
)
if "%PY%"=="" (
  where python >nul 2>nul && python -c "import struct,sys;sys.exit(0 if (struct.calcsize('P')==8 and (3,10)<=sys.version_info[:2]<=(3,13)) else 1)" >nul 2>nul && set PY=python
)
if "%PY%"=="" goto :needpython

echo  Using:
%PY% --version

if not exist ".venv\installed.ok" (
  if exist ".venv" rmdir /s /q ".venv"
  echo  Setting up for the first time...
  %PY% -m venv .venv || goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
  echo  Installing components, please wait...
  ".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt || goto :fail
  echo ok> ".venv\installed.ok"
)

echo.
echo  Opening GradeForge in your browser at http://localhost:8000
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
".venv\Scripts\python.exe" -m uvicorn gradeforge.app:app --port 8000
echo.
echo  GradeForge stopped.
pause
exit /b 0

:needpython
echo  GradeForge needs Python 3.12 (64-bit). Your computer has no suitable version.
echo.
echo  1. Go to https://www.python.org/downloads/windows/
echo  2. Download "Python 3.12" - choose "Windows installer (64-bit)".
echo  3. Run it. On the FIRST screen, tick "Add python.exe to PATH", then Install Now.
echo  4. Double-click start_windows.bat again.
echo.
pause
exit /b 1

:fail
echo.
echo  Setup failed. The most common cause is a Python version that is too new.
echo  Fix: install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/
echo  (tick "Add python.exe to PATH"), then double-click this file again.
echo  If it still fails, take a screenshot of this window and send it to Claude.
echo.
if exist ".venv" rmdir /s /q ".venv"
pause
exit /b 1
