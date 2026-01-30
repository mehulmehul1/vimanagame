# Ralph Agent Instructions - EPIC-005: Harp Minigame Design Alignment

You are an autonomous game developer agent implementing the phrase-first teaching mode for the harp minigame in the Vimana game project.

## Skills to Use

**IMPORTANT**: Use the project-specific 3d-graphics and three-best-practices skills for all Three.js/visual decisions:

- `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana\.claude\skills\3d-graphics`
  - Three.js integration
  - Sprite materials
  - Animation patterns

- `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana\.claude\skills\three-best-practices`
  - Performance optimization
  - Mobile considerations
  - State management patterns

**Always consult these skills when making Three.js, animation, or architectural decisions.**

## Your Task

1. Read the PRD at `prd.json` (in the same directory as this file)
2. Read the progress log at `progress.txt` - check Codebase Patterns first
3. Pick the **highest priority** story where `passes: false` AND all dependencies are satisfied
4. Execute that single story (read the implementation artifact, implement ALL acceptance criteria)
5. Validate all deliverables were created
6. If validation passes, commit ALL changes with message: `feat: STORY-HARP-10X - [Story Title]`
7. Update the PRD to set `passes: true` for the completed story
8. Append your progress to `progress.txt`

## Story Artifacts Location

All implementation artifacts are at:
```
../implementation-artifacts/story-harp-10X-*.md
```

Story IDs:
- `harp-101` → `story-harp-101-phrase-first-teaching.md` (Phrase-First Teaching Mode)
- `harp-102` → `story-harp-102-synchronized-splash.md` (Synchronized Splash System)
- `harp-103` → `story-harp-103-phrase-response.md` (Phrase Response Handling)
- `harp-104` → `story-harp-104-visual-indicators.md` (Visual Sequence Indicators)

## How to Execute a Story

For each story:

1. **Read the implementation artifact**:
   ```bash
   cat ../implementation-artifacts/story-harp-10X-*.md
   ```

2. **Follow ALL technical specifications** in the artifact:
   - Create the specified classes/modules
   - Implement the state machine extensions
   - Create the file structure as specified
   - Implement ALL acceptance criteria
   - Maintain backward compatibility with existing APIs

3. **File structure for this epic**:
   ```
   src/entities/
   ├── PatientJellyManager.ts (extended)
   ├── JellyManager.ts (extended)
   ├── JellyLabelManager.ts (enhanced)
   └── SynchronizedSplashEffect.ts (new)

   src/audio/
   └── HarmonyChord.ts (extended)
   ```

4. **Import Pattern**:
   ```typescript
   import { THREE, ThreeGlobals } from '../core/three-globals';
   import type { DuetState, TeachingModeConfig } from './PatientJellyManager';
   ```

5. **Run quality checks**:
   ```bash
   npm run build         # TypeScript compilation
   npm run dev           # Verify game runs
   ```

## Project Context

**Project**: Vimana - A 3D contemplative game about harmonizing with a mythical ship
**Tech Stack**: Three.js r180.0+, WebGPU, TypeScript, Vite
**Current Epic**: EPIC-005 - Harp Minigame Design Alignment
**Design Reference**: `vimana_harp_minigame_design.md`

**Source Location**: `src/` (in vimana/ directory)
**Working Directory**: `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana`

## Design Philosophy

**CRITICAL**: The harp minigame embodies "The ship teaches you, it doesn't test you."

Key principles:
- No failure states - only patient redemonstration
- Gentle feedback - camera shake, discordant audio, visual highlights
- The synchronized splash is a TURN SIGNAL, not a sound effect
- Number indicators (①, ②, ③) help players remember sequence order
- Note-by-note mode remains as fallback (don't break existing functionality)

## State Machine Extension Pattern

The existing `DuetState` enum will be extended:

```typescript
// Existing states (preserve these!)
IDLE = 'IDLE'
DEMONSTRATING = 'DEMONSTRATING'
AWAITING_INPUT = 'AWAITING_INPUT'
PLAYING_HARMONY = 'PLAYING_HARMONY'
COMPLETE = 'COMPLETE'

// NEW states for phrase-first mode
PHRASE_DEMONSTRATION = 'PHRASE_DEMONSTRATION'
TURN_SIGNAL = 'TURN_SIGNAL'
AWAITING_PHRASE_RESPONSE = 'AWAITING_PHRASE_RESPONSE'
```

## Progress Report Format

APPEND to progress.txt (never replace, always append):
```
## [Date/Time] - Story harp-10X
- Story: [Story Title]
- Files created:
  - path/to/file1.ts
  - path/to/file2.ts
- Files modified:
  - path/to/existing.ts
- Acceptance criteria validated:
  - [x] Criterion 1
  - [x] Criterion 2
- Tests/verification performed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - Three.js specifics
---
```

## Quality Requirements

- ALL acceptance criteria from story must be satisfied
- Code must compile without errors
- Follow existing code patterns in src/entities/
- Maintain existing API compatibility (PatientJellyManager, JellyManager, etc.)
- No breaking changes to note-by-note mode
- Debug views must be implemented where specified
- No broken code committed

## Dependency Handling

Each story has `dependsOn` in the PRD. Only work on stories where:
- All dependencies have `passes: true`
- OR dependencies are empty array `[]`

Example:
- `harp-101` has no dependencies → can start immediately
- `harp-102` depends on `harp-101` → must wait for phrase-first mode
- `harp-103` depends on `harp-101` → must wait for phrase-first mode
- `harp-104` depends on `harp-101` → must wait for phrase-first mode

## Stop Condition

After completing a story, check if ALL stories have `passes: true`.

If ALL 4 stories are complete and passing, reply with:
```
<promise>COMPLETE</promise>
```

If there are still stories with `passes: false`, end your response normally (another iteration will pick up the next story).

## Important

- Work on ONE story per iteration
- Commit frequently after each story
- Keep code compiling
- Read the progress.txt Codebase Patterns before starting
- Follow the implementation artifact specifications EXACTLY as written
- This is a gameplay enhancement - maintain the "gentle teaching" philosophy
- The synchronized splash is THE TURN SIGNAL - make it clear
- Number labels help memory - make them visible and readable
