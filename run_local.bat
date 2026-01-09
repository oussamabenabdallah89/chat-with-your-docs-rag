@echo off
setlocal

set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "FRONTEND=%ROOT%frontend"
set "VENV_PY=%BACKEND%\.venv\Scripts\python.exe"

echo ==========================================
echo  START LOCAL (backend + frontend)
echo ==========================================

REM --- Backend ---
echo.
echo [1/2] Starting BACKEND...

if exist "%VENV_PY%" (
  start "backend" cmd /k "cd /d %BACKEND% && "%VENV_PY%" -m uvicorn app:app --reload --host 127.0.0.1 --port 8000"
) else (
  echo [ERROR] Virtualenv not found:
  echo         %VENV_PY%
  echo.
  echo Fix: create venv in backend/ or adjust the path in this bat.
  pause
  exit /b 1
)

REM --- Frontend ---
echo.
echo [2/2] Starting FRONTEND...

if exist "%FRONTEND%\package.json" (
  start "frontend" cmd /k "cd /d %FRONTEND% && npm run dev"
) else (
  echo [ERROR] Frontend not found (missing package.json):
  echo         %FRONTEND%\package.json
  pause
  exit /b 1
)

echo.
echo Done.
echo Backend:  http://127.0.0.1:8000/health
echo Frontend: check the frontend terminal (Vite URL)
endlocal
