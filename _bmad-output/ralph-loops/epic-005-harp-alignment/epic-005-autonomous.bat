@echo off
REM Ralph Loop Launcher - EPIC-005 Harp Design Alignment
REM This script launches autonomous development loop for all Epic 5 stories

setlocal EnableDelayedExpansion

echo ========================================
echo EPIC-005 Harp Design Alignment Loop
echo ========================================
echo.
echo This will autonomously execute all 4 stories:
echo   [1] harp-101 Phrase-First Teaching Mode
echo   [2] harp-102 Synchronized Splash System
echo   [3] harp-103 Phrase Response Handling
echo   [4] harp-104 Visual Sequence Indicators
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

set LOGFILE=_bmad-output\ralph-loops\logs\epic-005-%date:~10,4%-%date:~4,2%-%date:~7,2%-%time:~0,2%-%time:~3,2%.log

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

REM Check existing code structure
echo Checking harp minigame files...
if exist "vimana\src\entities\PatientJellyManager.ts" (
    echo [FOUND] PatientJellyManager.ts
) else (
    echo [WARN] PatientJellyManager.ts not found
)
if exist "vimana\src\entities\JellyManager.ts" (
    echo [FOUND] JellyManager.ts
) else (
    echo [WARN] JellyManager.ts not found
)
if exist "vimana\src\audio\HarmonyChord.ts" (
    echo [FOUND] HarmonyChord.ts
) else (
    echo [WARN] HarmonyChord.ts not found
)
echo.

REM Start the loop - this would be triggered by calling Claude with the prompt
echo.
echo ========================================
echo INSTRUCTIONS
echo ========================================
echo.
echo 1. Copy the contents of:
echo    _bmad-output\ralph-loops\epic-005-harp-alignment\epic-005-harp-autonomous-loop.md
echo.
echo 2. Paste into Claude Code as your prompt
echo.
echo 3. Claude will execute all stories autonomously
echo.
echo ========================================
echo.

pause
