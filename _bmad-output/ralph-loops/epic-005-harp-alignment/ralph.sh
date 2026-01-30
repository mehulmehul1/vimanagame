#!/bin/bash
# Ralph Loop - EPIC-005: Harp Minigame Design Alignment
# Autonomous AI agent loop for Claude Code
# Usage: ./ralph.sh [--tool claude] [max_iterations]

set -e

# Parse arguments
TOOL="claude"
MAX_ITERATIONS=10

while [[ $# -gt 0 ]]; do
  case $1 in
    --tool)
      TOOL="$2"
      shift 2
      ;;
    --tool=*)
      TOOL="${1#*=}"
      shift
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
  esac
done

# Validate tool choice
if [[ "$TOOL" != "amp" && "$TOOL" != "claude" ]]; then
  echo "Error: Invalid tool '$TOOL'. Must be 'amp' or 'claude'."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../vimana" && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
CLAUDE_PROMPT="$SCRIPT_DIR/CLAUDE.md"

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# Ralph Progress Log - EPIC-005" > "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "" >> "$PROGRESS_FILE"
  echo "## Codebase Patterns" >> "$PROGRESS_FILE"
  echo "# Add reusable patterns here as they are discovered" >> "$PROGRESS_FILE"
  echo "" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

echo ""
echo "==============================================================="
echo "  Ralph Loop - EPIC-005: Harp Minigame Design Alignment"
echo "==============================================================="
echo "Tool: $TOOL"
echo "Max iterations: $MAX_ITERATIONS"
echo "Project root: $PROJECT_ROOT"
echo "PRD: $PRD_FILE"
echo ""

# Change to project directory for git operations
cd "$PROJECT_ROOT"

# Check git status
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"
echo ""

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "==============================================================="
  echo ""

  # Check for completion in PRD
  ALL_PASS=true
  for story_id in $(jq -r '.stories[].id' "$PRD_FILE"); do
    passes=$(jq -r ".stories[] | select(.id == \"$story_id\") | .passes" "$PRD_FILE")
    if [ "$passes" != "true" ]; then
      ALL_PASS=false
      break
    fi
  done

  if [ "$ALL_PASS" = true ]; then
    echo ""
    echo "==============================================================="
    echo "  ALL STORIES COMPLETE!"
    echo "==============================================================="
    echo ""
    jq -r '.stories[] | select(.passes == true) | "\(.id): \(.title)"' "$PRD_FILE"
    echo ""
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  # Find next story to work on
  NEXT_STORY=$(jq -r '.stories[] | select(.passes == false) | .id' "$PRD_FILE" | head -1)

  # Check dependencies
  if [ -n "$NEXT_STORY" ]; then
    DEPS=$(jq -r ".stories[] | select(.id == \"$NEXT_STORY\") | .dependsOn[]?" "$PRD_FILE")
    if [ -n "$DEPS" ]; then
      echo "Checking dependencies for $NEXT_STORY..."
      for dep in $DEPS; do
        dep_passes=$(jq -r ".stories[] | select(.id == \"$dep\") | .passes" "$PRD_FILE")
        if [ "$dep_passes" != "true" ]; then
          echo "  Dependency $dep not complete, skipping..."
          # Mark this story as temporarily skipped
          NEXT_STORY=""
        fi
      done
    fi
  fi

  if [ -n "$NEXT_STORY" ]; then
    STORY_TITLE=$(jq -r ".stories[] | select(.id == \"$NEXT_STORY\") | .title" "$PRD_FILE")
    echo "Next story: $NEXT_STORY - $STORY_TITLE"
  else
    echo "No available story (dependencies not met). Scanning..."
  fi
  echo ""

  # Run the selected tool with the ralph prompt
  if [[ "$TOOL" == "amp" ]]; then
    OUTPUT=$(cat "$CLAUDE_PROMPT" | amp --dangerously-allow-all 2>&1 | tee /dev/stderr) || true
  else
    # Claude Code: use --dangerously-skip-permissions for autonomous operation, --print for output
    OUTPUT=$(cd "$PROJECT_ROOT" && claude --dangerously-skip-permissions --print < "$CLAUDE_PROMPT" 2>&1 | tee /dev/stderr) || true
  fi

  # Check for completion signal
  if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
    echo ""
    echo "==============================================================="
    echo "  Ralph completed all tasks!"
    echo "==============================================================="
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    echo ""
    echo "<promise>COMPLETE</promise>"
    exit 0
  fi

  echo ""
  echo "Iteration $i complete. Continuing..."
  sleep 1
done

echo ""
echo "==============================================================="
echo "  Ralph reached max iterations ($MAX_ITERATIONS)"
echo "==============================================================="
echo "Check $PROGRESS_FILE for status."
echo ""
exit 1
