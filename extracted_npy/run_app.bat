@echo off
title SpectraRestore Studio Launcher
echo ====================================================
echo  Launching SpectraRestore Studio Workbench...
echo ====================================================

:: Navigate directly to the directory containing this script
cd /d "%~dp0"

:: Start the browser in the background after a 3-second delay
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8501"

:: Run Streamlit
python -m streamlit run app.py
pause