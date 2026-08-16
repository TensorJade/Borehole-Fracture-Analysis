@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
python -m borehole_fracture_analysis %*
exit /b %errorlevel%
