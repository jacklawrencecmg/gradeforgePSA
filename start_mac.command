#!/bin/bash
# Double-click this file on a Mac (or run it in a terminal on Linux).
cd "$(dirname "$0")" || exit 1
echo
echo " GradeForge - starting. The first run installs things and takes a few minutes."
echo " Leave this window open while you use GradeForge. Close it to stop."
echo

PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo " Python is not installed."
  echo " Install it from https://www.python.org/downloads/ then double-click this file again."
  read -r -p " Press Return to close." _; exit 1
fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' || {
  echo " Your Python is too old (need 3.10 or newer). Install a newer one from https://www.python.org/downloads/"
  read -r -p " Press Return to close." _; exit 1; }

if [ ! -x .venv/bin/python ]; then
  echo " Setting up for the first time..."
  "$PY" -m venv .venv || { echo " Setup failed. Screenshot this window and send it to Claude."; read -r -p " Press Return." _; exit 1; }
fi
if [ ! -f .venv/installed.ok ]; then
  echo " Installing components, please wait..."
  .venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1
  .venv/bin/python -m pip install -r requirements.txt && touch .venv/installed.ok || {
    echo " Install failed. Screenshot this window and send it to Claude."; read -r -p " Press Return." _; exit 1; }
fi

echo
echo " Opening GradeForge in your browser at http://localhost:8000"
( sleep 4; (open http://localhost:8000 || xdg-open http://localhost:8000) >/dev/null 2>&1 ) &
.venv/bin/python -m uvicorn gradeforge.app:app --port 8000
