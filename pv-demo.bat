@echo off
REM Convenience wrapper for the PV demo. Runs from the project root.
REM
REM Usage:
REM   pv-demo                             Run the full 4-panel arc
REM   pv-demo H500.jpeg                   Run on a single PV fault image
REM   pv-demo H500.jpeg "audience pick"   Same, with a custom label
REM   pv-demo "full\path\to\image.tif"    Use an explicit path (anywhere on disk)

cd /d %~dp0

if "%~1"=="" (
    venv\Scripts\python.exe src\demo_ensemble.py --model pv --equipment pv
    exit /b
)

REM If the filename has no path separator, assume it's in the PV fault folder.
echo %~1| findstr "[\\/:]" >nul
if errorlevel 1 (
    set "IMG=data\CA_Training_Data\test\fault\pv\%~1"
) else (
    set "IMG=%~1"
)

if "%~2"=="" (
    venv\Scripts\python.exe src\demo_ensemble.py --model pv --equipment pv --image "%IMG%"
) else (
    venv\Scripts\python.exe src\demo_ensemble.py --model pv --equipment pv --image "%IMG%" --expected "%~2"
)
