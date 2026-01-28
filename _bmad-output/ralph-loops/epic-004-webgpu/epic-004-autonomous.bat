@echo off
REM Ralph Loop Launcher - EPIC-004 WebGPU Migration
REM This script launches autonomous development loop for all Epic 4 stories

setlocal EnableDelayedExpansion

echo ========================================
echo EPIC-004 WebGPU Migration Loop
echo ========================================
echo.
echo This will autonomously execute all 8 stories:
echo   [1] 4.1 Visionary Integration
echo   [2] 4.2 WebGPU Renderer Init
echo   [3] 4.3 Vortex Shader TSL
echo   [4] 4.4 Water Material TSL
echo   [5] 4.5 Shell SDF TSL
echo   [6] 4.6 Jelly Shader TSL
echo   [7] 4.7 Fluid System Activation
echo   [8] 4.8 Performance Validation
echo.
echo Working Directory: vimana
echo.

REM Check if we're in the right directory
if not exist "vimana\src\main.js" (
    echo ERROR: Please run from shadowczarengine directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Create logs directory
if not exist "_bmad-output\ralph-loops\logs" mkdir "_bmad-output\ralph-loops\logs"

set LOGFILE=_bmad-output\ralph-loops\logs\epic-004-%date:~10,4%-%date:~4,2%-%date:~7,2%-%time:~0,2%-%time:~3,2%.log

echo Log file: %LOGFILE%
echo.

REM Prompt for confirmation
echo Press any key to START autonomous execution...
pause >nul

echo.
echo ========================================
echo STARTING AUTONOMOUS EXECUTION
echo ========================================
echo.

REM Get current branch
for /f "tokens=*" %%i in ('git -C vimana branch --show-current') do set CURRENT_BRANCH=%%i
echo Current branch: %CURRENT_BRANCH%
echo.

REM Check Three.js version
echo Checking Three.js version...
findstr "three" vimana\package.json
echo.

REM Start the loop - this would be triggered by calling Claude with the prompt
echo.
echo ========================================
echo INSTRUCTIONS
echo ========================================
echo.
echo 1. Copy the contents of:
echo    _bmad-output\ralph-loops\epic-004-webgpu-autonomous-loop.md
echo.
echo 2. Paste into Claude Code as your prompt
echo.
echo 3. Claude will execute all stories autonomously
echo.
echo ========================================
echo.

pause
