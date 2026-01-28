#!/bin/bash
# Ralph Wiggum - EPIC-004 WebGPU Migration Loop
# Usage: ./ralph.sh [max_iterations]

set -e

# Configuration
TOOL="claude"
MAX_ITERATIONS="${1:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
ARCHIVE_DIR="$SCRIPT_DIR/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"

# Archive previous run if branch changed
if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  LAST_BRANCH=$(cat "$LAST_BRANCH_FILE" 2>/dev/null || echo "")

  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    DATE=$(date +%Y-%m-%d)
    FOLDER_NAME=$(echo "$LAST_BRANCH" | sed 's|^ralph/||')
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"

    echo "Archiving previous run: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    echo "   Archived to: $ARCHIVE_FOLDER"

    echo "# Ralph Progress Log - EPIC-004: WebGPU Migration" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
fi

# Track current branch
if [ -f "$PRD_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  if [ -n "$CURRENT_BRANCH" ]; then
    echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
  fi
fi

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Ralph Progress Log - EPIC-004: WebGPU Migration" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
  echo "" >> "$PROGRESS_FILE"
  echo "## Codebase Patterns" >> "$PROGRESS_FILE"
  echo "- Project: Vimana (Three.js + WebGPU + TSL + Visionary)" >> "$PROGRESS_FILE"
  echo "- Location: C:\\Users\\mehul\\OneDrive\\Desktop\\Studio\\PROJECTS\\shadowczarengine\\vimana" >> "$PROGRESS_FILE"
  echo "- Source: src/" >> "$PROGRESS_FILE"
  echo "- Shaders: TSL (Three.js Shading Language) → WGSL" >> "$PROGRESS_FILE"
  echo "- Renderer: WebGPURenderer with Visionary integration" >> "$PROGRESS_FILE"
  echo "- Platform: Windows 10/11 (Ubuntu not supported)" >> "$PROGRESS_FILE"
  echo "" >> "$PROGRESS_FILE"
  echo "## Visionary Platform Notes" >> "$PROGRESS_FILE"
  echo "- Ubuntu: NOT supported (fp16 WebGPU bug)" >> "$PROGRESS_FILE"
  echo "- macOS: Performance limited (M4 Max+ recommended)" >> "$PROGRESS_FILE"
  echo "- Windows: RECOMMENDED (discrete GPU)" >> "$PROGRESS_FILE"
  echo "" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

echo "Starting Ralph - EPIC-004: WebGPU Migration"
echo "Max iterations: $MAX_ITERATIONS"
echo ""

for i in $(seq 1 $MAX_ITERATIONS); do
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS"
  echo "==============================================================="
  echo ""

  # Run Claude Code with the CLAUDE prompt
  OUTPUT=$(claude --dangerously-skip-permissions --print < "$SCRIPT_DIR/CLAUDE.md" 2>&1) || true

  # Check for completion signal
  if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
    echo ""
    echo "🎉 Ralph completed EPIC-004 - WebGPU Migration!"
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    echo ""
    echo "All stories implemented:"
    jq -r '.userStories[] | "  - \(.id): \(.title)"' "$PRD_FILE"
    exit 0
  fi

  echo "Iteration $i complete. Continuing..."
  echo ""
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all stories."
echo "Check $PROGRESS_FILE for status."
echo ""
echo "Remaining stories:"
jq -r '.userStories[] | select(.passes == false) | "  - \(.id): \(.title)"' "$PRD_FILE"
exit 1
