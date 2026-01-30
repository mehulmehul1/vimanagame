@echo off
REM Ralph Loop - EPIC-005: Harp Minigame Design Alignment
REM Autonomous AI agent loop for Claude Code (Windows)
REM Usage: ralph.bat [max_iterations]

setlocal EnableDelayedExpansion

set MAX_ITERATIONS=10
set TOOL=claude

REM Parse arguments
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--tool" (
    set TOOL=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--tool=" (
    set TOOL=%~1:~7%
    shift
    goto parse_args
)
echo %~1| findstr /r "^[0-9][0-9]*$">nul
if !errorlevel! equ 0 (
    set MAX_ITERATIONS=%~1
    shift
    goto parse_args
)
shift
goto parse_args
:end_parse

REM Validate tool
if not "%TOOL%"=="claude" if not "%TOOL%"=="amp" (
    echo Error: Invalid tool '%TOOL%'. Must be 'claude' or 'amp'.
    exit /b 1
)

REM Set paths
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..\..\vimana
set PRD_FILE=%SCRIPT_DIR%prd.json
set PROGRESS_FILE=%SCRIPT_DIR%progress.txt
set CLAUDE_PROMPT=%SCRIPT_DIR%CLAUDE.md

REM Convert to absolute paths
for %%i in ("%PROJECT_ROOT%") do set PROJECT_ROOT=%%~fi
for %%i in ("%PRD_FILE%") do set PRD_FILE=%%~fi
for %%i in ("%PROGRESS_FILE%") do set PROGRESS_FILE=%%~fi
for %%i in ("%CLAUDE_PROMPT%") do set CLAUDE_PROMPT=%%~fi

REM Initialize progress file if it doesn't exist
if not exist "%PROGRESS_FILE%" (
    echo # Ralph Progress Log - EPIC-005> "%PROGRESS_FILE%"
    echo Started: %date% %time%>> "%PROGRESS_FILE%"
    echo.>> "%PROGRESS_FILE%"
    echo ## Codebase Patterns>> "%PROGRESS_FILE%"
    echo # Add reusable patterns here as they are discovered>> "%PROGRESS_FILE%"
    echo.>> "%PROGRESS_FILE%"
    echo --- >> "%PROGRESS_FILE%"
)

echo.
echo ===============================================================
echo   Ralph Loop - EPIC-005: Harp Minigame Design Alignment
echo ===============================================================
echo Tool: %TOOL%
echo Max iterations: %MAX_ITERATIONS%
echo Project root: %PROJECT_ROOT%
echo PRD: %PRD_FILE%
echo.

REM Change to project directory
cd /d "%PROJECT_ROOT%"

REM Check git status
for /f "tokens=*" %%i in ('git branch --show-current') do set CURRENT_BRANCH=%%i
echo Current branch: %CURRENT_BRANCH%
echo.

REM Main loop
set i=1
:loop
if %i% gtr %MAX_ITERATIONS% goto max_reached

    echo.
    echo ===============================================================
    echo   Ralph Iteration %i% of %MAX_ITERATIONS% (%TOOL%)
    echo ===============================================================
    echo.

    REM Run Claude Code with the prompt
    cd /d "%PROJECT_ROOT%"
    claude --dangerously-skip-permissions --print < "%CLAUDE_PROMPT%"
    set CLAUDE_EXIT=!errorlevel!

    REM Check for completion signal in output (Claude prints to stdout)
    REM We'll check after each iteration via prd.json

    REM Check if all stories are complete
    node -e "const fs=require('fs'); const prd=JSON.parse(fs.readFileSync('%PRD_FILE%','utf8')); const allPass=prd.stories.every(s=>s.passes===true); if(allPass){console.log('ALL_COMPLETE');}else{console.log('CONTINUE');}" > "%TEMP%\ralph_check.txt" 2>&1
    set /p CHECK_RESULT=<"%TEMP%\ralph_check.txt"

    if "!CHECK_RESULT!"=="ALL_COMPLETE" (
        echo.
        echo ===============================================================
        echo   ALL STORIES COMPLETE!
        echo ===============================================================
        echo.
        echo <promise>COMPLETE</promise>
        exit /b 0
    )

    echo.
    echo Iteration %i% complete. Continuing...
    echo.

    set /a i=%i%+1
    timeout /t 1 /nobreak >nul
    goto loop

:max_reached
echo.
echo ===============================================================
echo   Ralph reached max iterations (%MAX_ITERATIONS%)
echo ===============================================================
echo Check %PROGRESS_FILE% for status.
echo.
exit /b 1
