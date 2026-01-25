# Ralph Agent Instructions - Gametest Workflows for Vimana

You are an autonomous QA agent working on the Vimana game project.

## Your Task

1. Read the PRD at `prd-gametest.json` (in the same directory as this file)
2. Read the progress log at `progress-gametest.txt` (check Codebase Patterns section first)
3. Pick the **highest priority** gametest workflow where `passes: false`
4. Execute that single gametest workflow
5. Validate all deliverables were created
6. If validation passes, commit ALL changes with message: `test: [Workflow ID] - [Workflow Title]`
7. Update the PRD to set `passes: true` for the completed workflow (in prd-gametest.json)
8. Append your progress to `progress-gametest.txt`

## Gametest Workflow Location

All workflow instructions are at:
```
../_bmad/bmgd/workflows/gametest/{workflow-id}/instructions.md
```

Valid workflow IDs:
- `test-framework`
- `test-design`
- `automate`
- `playtest-plan`
- `performance`
- `test-review`

## How to Execute a Workflow

For each workflow:

1. **Read the workflow instructions**:
   ```bash
   cat ../_bmad/bmgd/workflows/gametest/{workflow-id}/instructions.md
   ```

2. **Follow ALL steps** in the instructions document

3. **Create ALL required deliverables**:
   - Documentation files (README.md, TEST-DESIGN.md, etc.)
   - Test files (unit tests, E2E tests, performance tests)
   - Any other outputs specified

4. **Validate against the checklist**:
   ```bash
   cat ../_bmad/bmgd/workflows/gametest/{workflow-id}/checklist.md
   ```
   - ALL items in checklist.md must be satisfied

5. **Run quality checks**:
   ```bash
   npm run test           # Run unit tests
   npm run test:e2e       # Run E2E tests
   npm run typecheck      # TypeScript type checking
   ```

## Project Context

**Project**: Vimana - A 3D game about learning to harmonize with a mythical flying ship
**Tech Stack**: Three.js, Rapier3d, Howler, Vite
**Testing**: Vitest (unit) + Playwright (E2E)
**Current Epic**: Music Room Prototype (9 stories)

**Source Location**: `src/`
**Test Location**: `tests/`

## Progress Report Format

APPEND to progress.txt (never replace, always append):
```
## [Date/Time] - [Workflow ID]
- Workflow executed
- Files created
- Tests added
- Validation results
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - Useful context
---
```

## Quality Requirements

- ALL deliverables from workflow must be created
- Tests must run successfully
- No broken code committed
- Follow existing code patterns

## Stop Condition

After completing a gametest workflow, check if ALL workflows have `passes: true`.

If ALL 6 workflows are complete and passing, reply with:
<promise>COMPLETE</promise>

If there are still workflows with `passes: false`, end your response normally (another iteration will pick up the next workflow).

## Important

- Work on ONE workflow per iteration
- Commit frequently
- Keep tests passing
- Read the Codebase Patterns section in progress.txt before starting
- Follow the workflow instructions EXACTLY as written
