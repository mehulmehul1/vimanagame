# Ralph Agent Instructions - EPIC-002: WaterBall Fluid Simulation

You are an autonomous game developer agent implementing the WaterBall fluid simulation system for the Vimana game project.

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
../planning-artifacts/stories/story-XXX-[name].md
```

Story IDs:
- `002-001` → `story-001-particle-physics.md` (MLS-MPM Particle Physics)
- `002-002` → `story-002-depth-thickness.md` (Depth & Thickness Rendering)
- `002-003` → `story-003-fluid-surface.md` (Fluid Surface Shader)
- `002-004` → `story-004-sphere-animation.md` (Sphere Constraint Animation)
- `002-005` → `story-005-harp-interaction.md` (Harp-to-Water Interaction)
- `002-006` → `story-006-player-collision.md` (Player Collision & Displacement)

## How to Execute a Story

For each story:

1. **Read the story file**:
   ```bash
   cat ../planning-artifacts/stories/story-XXX-[name].md
   ```

2. **Follow ALL technical specifications** in the story:
   - Create the specified classes/systems
   - Implement the shaders (WGSL format for WebGPU)
   - Create the file structure as specified
   - Implement ALL acceptance criteria

3. **File structure for this epic**:
   ```
   src/systems/fluid/
   ├── MLSMPMSimulator.ts
   ├── compute/
   │   ├── clearGrid.wgsl
   │   ├── p2g_1.wgsl
   │   ├── p2g_2.wgsl
   │   ├── updateGrid.wgsl
   │   ├── g2p.wgsl
   │   └── copyPosition.wgsl
   ├── render/
   │   ├── DepthThicknessRenderer.ts
   │   ├── FluidSurfaceRenderer.ts
   │   └── shaders/
   │       ├── depthMap.wgsl
   │       ├── thicknessMap.wgsl
   │       ├── bilateral.wgsl
   │       ├── gaussian.wgsl
   │       └── fluid.wgsl
   ├── interaction/
   │   ├── HarpWaterInteraction.ts
   │   ├── PlayerWaterInteraction.ts
   │   └── StringRippleEffect.ts
   ├── animation/
   │   └── SphereConstraintAnimator.ts
   └── types.ts
   ```

4. **Create debug views** where specified:
   ```typescript
   // In a central debug file
   (window as any).debugVimana = {
       fluid: {
           getParticleCount: () => particleSystem.getParticleCount(),
           toggleWireframe: () => renderer.toggleWireframe(),
           showDepthMap: () => renderer.showDepthMap(),
           // ... etc
       }
   };
   ```

5. **Run quality checks**:
   ```bash
   npm run typecheck      # TypeScript validation
   npm run dev            # Verify game runs
   ```

## Project Context

**Project**: Vimana - A 3D contemplative game about harmonizing with a mythical ship
**Tech Stack**: Three.js 0.180.0, WebGPU, Vite, TypeScript
**Current Epic**: EPIC-002 - WaterBall Fluid Simulation System
**Source**: https://github.com/matsuoka-601/WaterBall (reference implementation)

**Source Location**: `src/` (in vimana/ directory)
**Working Directory**: `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana`

## Progress Report Format

APPEND to progress.txt (never replace, always append):
```
## [Date/Time] - Story [ID]
- Story: [Story Title]
- Files created:
  - path/to/file1.ts
  - path/to/file2.wgsl
- Acceptance criteria validated:
  - [x] Criterion 1
  - [x] Criterion 2
- Tests/verification performed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - WebGPU specifics
---
```

## Quality Requirements

- ALL acceptance criteria from story must be satisfied
- Code must compile without errors
- Follow existing code patterns in src/
- WebGPU shaders must be valid WGSL
- Debug views must be implemented
- No broken code committed

## Dependency Handling

Each story has `dependsOn` in the PRD. Only work on stories where:
- All dependencies have `passes: true`
- OR dependencies are empty array `[]`

Example:
- `002-001` has no dependencies → can start immediately
- `002-002` depends on `002-001` → must wait for particle physics
- `002-003` depends on `002-002` → must wait for depth/thickness rendering

## Stop Condition

After completing a story, check if ALL stories have `passes: true`.

If ALL 6 stories are complete and passing, reply with:
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
- This is WebGPU - all shaders are WGSL format, not GLSL
