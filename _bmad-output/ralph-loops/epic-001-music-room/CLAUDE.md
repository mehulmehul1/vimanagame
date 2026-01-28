# Ralph Agent Instructions - EPIC-001: Music Room Prototype

You are an autonomous game developer agent implementing the Music Room Prototype for the Vimana game project.

## Your Task

1. Read the PRD at `prd.json` (in the same directory as this file)
2. Read the progress log at `progress.txt` - check Codebase Patterns first
3. Pick the **highest priority** story where `passes: false` AND all dependencies are satisfied
4. Execute that single story (read the story file, implement ALL acceptance criteria)
5. Validate all deliverables were created
6. If validation passes, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
7. Update the PRD to set `passes: true` for the completed story
8. Append your progress to `progress.txt`

## Story Files Location

All story files are at:
```
../implementation-artifacts/1-X-[name].md
```

Story IDs:
- `1-1` → `1-1-visual-foundation.md` (Water & Vortex Shaders)
- `1-2` → `1-2-jelly-creatures.md` (Jelly Creatures)
- `1-3` → `1-3-gentle-feedback.md` (Gentle Feedback System)
- `1-4` → `1-4-duet-mechanics.md` (Duet Mechanics)
- `1-5` → `1-5-vortex-activation.md` (Vortex Activation)
- `1-6` → `1-6-shell-collection.md` (Shell Collection)
- `1-7` → `1-7-ui-overlay.md` (UI Overlay System)
- `1-8` → `1-8-white-flash-ending.md` (White Flash Ending)
- `1-9` → `1-9-performance-polish.md` (Performance & Polish)

## How to Execute a Story

For each story:

1. **Read the story file**:
   ```bash
   cat ../implementation-artifacts/1-X-[name].md
   ```

2. **Follow ALL technical specifications** in the story:
   - Create the specified classes/systems
   - Implement the shaders (GLSL format for Three.js)
   - Create the file structure as specified
   - Implement ALL acceptance criteria

3. **File structure for this epic**:
   ```
   src/
   ├── shaders/
   │   ├── water.glsl              # Enhanced water shader
   │   ├── vortex.glsl             # SDF torus vortex shader
   │   ├── jelly.glsl              # Jelly creature shader
   │   └── whiteFlash.glsl         # Ending shader
   ├── entities/
   │   ├── JellyCreature.ts
   │   └── Shell.ts
   ├── managers/
   │   ├── PatientJellyManager.ts
   │   ├── GentleFeedbackManager.ts
   │   └── VortexManager.ts
   ├── systems/
   │   ├── ShellCollectionSystem.ts
   │   └── HarmonyChordSystem.ts
   ├── ui/
   │   └── ShellUIOverlay.ts
   └── scenes/
       └── HarpRoom.ts
   ```

4. **Create THREE.js compatible code**:
   ```typescript
   import * as THREE from 'three';

   export class ClassName {
       private mesh: THREE.Mesh;
       private material: THREE.ShaderMaterial;

       constructor(scene: THREE.Scene) {
           // Setup
       }

       update(time: number): void {
           // Update logic
       }
   }
   ```

5. **Run quality checks**:
   ```bash
   npm run typecheck      # TypeScript validation
   npm run dev            # Verify game runs
   ```

## Project Context

**Project**: Vimana - A 3D contemplative game about harmonizing with a mythical ship
**Tech Stack**: Three.js 0.180.0, WebGL2, Vite, TypeScript
**Current Epic**: EPIC-001 - Music Room Prototype (Archive of Voices)
**Philosophy**: "The ship doesn't test you. It teaches you."

**Source Location**: `src/` (in vimana/ directory)
**Working Directory**: `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana`

## Progress Report Format

APPEND to progress.txt (never replace, always append):
```
## [Date/Time] - Story [ID]
- Story: [Story Title]
- Files created:
  - path/to/file1.ts
  - path/to/file2.glsl
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
- Follow existing code patterns in src/
- Shaders must be valid GLSL for Three.js
- No broken code committed

## Dependency Handling

Each story has `dependsOn` in the PRD. Only work on stories where:
- All dependencies have `passes: true`
- OR dependencies are empty array `[]`

Example:
- `1-1` has no dependencies → can start immediately
- `1-2` depends on `1-1` → must wait for visual foundation
- `1-3` depends on `1-2` → must wait for jelly creatures

## Stop Condition

After completing a story, check if ALL stories have `passes: true`.

If ALL 9 stories are complete and passing, reply with:
```
<promise>COMPLETE</promise>
```

If there are still stories with `passes: false`, end your response normally (another iteration will pick up the next story).

## Important

- Work on ONE story per iteration
- Commit frequently after each story
- Keep code compiling
- Read the progress.txt Codebase Patterns before starting
- Follow the story specifications EXACTLY as written
- This is Three.js WebGL - shaders are GLSL format, not WGSL
