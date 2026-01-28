# Ralph Agent Instructions - EPIC-004: WebGPU Migration

You are an autonomous game developer agent implementing the WebGPU migration with Visionary Gaussian splat support for the Vimana game project.

## Skills to Use

**IMPORTANT**: Use the project-specific 3d-graphics and three-best-practices skills for all TSL/WebGPU decisions:

- `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana\.claude\skills\3d-graphics`
  - WebGPU fundamentals
  - Node materials
  - Three.js integration

- `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana\.claude\skills\three-best-practices`
  - TSL complete reference
  - TSL compute shaders
  - TSL post-processing
  - Mobile optimization

**Always consult these skills when making TSL, WebGPU, or Three.js architectural decisions.**

## Your Task

1. Read the PRD at `prd.json` (in the same directory as this file)
2. Read the progress log at `progress.txt` - check Codebase Patterns first
3. Pick the **highest priority** story where `passes: false` AND all dependencies are satisfied
4. Execute that single story (read the implementation artifact, implement ALL acceptance criteria)
5. Validate all deliverables were created
6. If validation passes, commit ALL changes with message: `feat: STORY-004-0XX - [Story Title]`
7. Update the PRD to set `passes: true` for the completed story
8. Append your progress to `progress.txt`

## Story Artifacts Location

All implementation artifacts are at:
```
../implementation-artifacts/4-XX-[name].md
```

Story IDs:
- `004-001` → `4-1-visionary-integration.md` (Visionary Gaussian Splat Integration)
- `004-002` → `4-2-webgpu-renderer-init.md` (WebGPU Renderer Initialization)
- `004-003` → `4-3-vortex-shader-tsl.md` (Vortex Shader → TSL Migration)
- `004-004` → `4-4-water-material-tsl.md` (Water Material → TSL Migration)
- `004-005` → `4-5-shell-sdf-tsl.md` (Shell SDF → TSL Migration)
- `004-006` → `4-6-jelly-shader-tsl.md` (Jelly Shader → TSL Migration)
- `004-007` → `4-7-fluid-system-activation.md` (Fluid System Activation)
- `004-008` → `4-8-performance-validation.md` (Performance Validation)

## How to Execute a Story

For each story:

1. **Read the implementation artifact**:
   ```bash
   cat ../implementation-artifacts/4-XX-[name].md
   ```

2. **Follow ALL technical specifications** in the artifact:
   - Create the specified classes/modules
   - Implement the TSL shaders (Three.js Shading Language)
   - Create the file structure as specified
   - Implement ALL acceptance criteria
   - Maintain backward compatibility with existing APIs

3. **File structure for this epic**:
   ```
   src/rendering/
   ├── createWebGPURenderer.ts
   └── VisionarySplatRenderer.ts

   src/shaders/tsl/
   ├── VortexShader.ts
   ├── WaterShader.ts
   ├── ShellShader.ts
   └── JellyShader.ts

   src/systems/fluid/compute/
   ├── FluidCompute.ts
   ├── ParticleBuffers.ts
   └── MLSMPMStages.ts

   src/utils/
   ├── DeviceCapabilities.ts
   ├── PerformanceProfiler.ts
   ├── QualityManager.ts
   └── PerformanceHUD.ts
   ```

4. **TSL Import Pattern**:
   ```typescript
   import { Fn, positionLocal, normalLocal, uniform, uv,
            mix, distance, max, min, sin, cos, pow, float, vec3, vec2,
            normalize, dot, cameraPosition, timerLocal,
            MeshStandardNodeMaterial, MeshPhysicalNodeMaterial,
            If, smoothstep, storage, instanceIndex } from 'three/tsl';
   ```

5. **Run quality checks**:
   ```bash
   npm run build         # TypeScript compilation
   npm run dev           # Verify game runs
   ```

## Project Context

**Project**: Vimana - A 3D contemplative game about harmonizing with a mythical ship
**Tech Stack**: Three.js r171.0+, WebGPU, TSL, Vite, TypeScript
**Current Epic**: EPIC-004 - WebGPU Migration with Visionary
**Visionary**: https://github.com/Visionary-Laboratory/visionary

**Source Location**: `src/` (in vimana/ directory)
**Working Directory**: `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana`

## Visionary Platform Requirements

**CRITICAL**: Visionary has platform limitations:
- **Windows 10/11**: ✅ RECOMMENDED (discrete GPU)
- **Ubuntu/Linux**: ❌ NOT SUPPORTED (fp16 WebGPU bug)
- **macOS**: ⚠️ LIMITED (M4 Max+ recommended)
- **Mobile**: ⚠️ LIMITED (performance varies)

Platform detection MUST be implemented in Story 4.2.

## TSL vs GLSL Key Differences

- TSL uses `Fn()` for shader functions
- TSL uses `uniform()` for uniform declarations
- TSL uses `timerLocal()` instead of passing time uniform
- TSL uses `MeshStandardNodeMaterial` instead of `ShaderMaterial`
- TSL compiles to WGSL automatically for WebGPU

## Progress Report Format

APPEND to progress.txt (never replace, always append):
```
## [Date/Time] - Story [ID]
- Story: [Story Title]
- Files created:
  - path/to/file1.ts
  - path/to/file2.ts
- Acceptance criteria validated:
  - [x] Criterion 1
  - [x] Criterion 2
- Tests/verification performed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - TSL specifics
  - Visionary integration notes
---
```

## Quality Requirements

- ALL acceptance criteria from story must be satisfied
- Code must compile without errors
- Follow existing code patterns in src/
- Maintain existing API compatibility (VortexMaterial, WaterMaterial, etc.)
- TSL shaders must be valid
- Debug views must be implemented where specified
- No broken code committed

## Dependency Handling

Each story has `dependsOn` in the PRD. Only work on stories where:
- All dependencies have `passes: true`
- OR dependencies are empty array `[]`

Example:
- `004-001` has no dependencies → can start immediately
- `004-002` has no dependencies → can start in parallel with 004-001
- `004-003` depends on `004-002` → must wait for WebGPU init
- `004-004` depends on `004-003` → must wait for Vortex TSL

## Stop Condition

After completing a story, check if ALL stories have `passes: true`.

If ALL 8 stories are complete and passing, reply with:
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
- This is WebGPU + TSL - all shaders use Three.js Shading Language
- Visionary integration requires WebGPU context
