# Ralph Loop Prompt - EPIC-004 WebGPU Migration (Autonomous)

## Context

You are an autonomous development loop executing **EPIC-004: WebGPU Migration** for the VIMANA project.

**Agent Persona:** @C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\.claude\commands\bmad\bmgd\agents\game-dev.md

**Epic File:** `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\_bmad-output\planning-artifacts\epics\EPIC-004-webgpu-migration.md`

**Implementation Artifacts:** `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\_bmad-output\implementation-artifacts\4-*-*.md`

---

## Mission

Execute all 8 stories of EPIC-004 autonomously until complete. Do not stop for user input unless a critical decision point is reached.

---

## Stories to Execute (Dependency Order)

### Phase 1: Foundation (Can run in parallel)
- **Story 4.1**: Visionary Gaussian Splat Integration
- **Story 4.2**: WebGPU Renderer Initialization

### Phase 2: TSL Migrations (After 4.2, can run in parallel)
- **Story 4.3**: Vortex Shader → TSL Migration
- **Story 4.4**: Water Material → TSL Migration

### Phase 3: More TSL (After 4.4)
- **Story 4.5**: Shell SDF → TSL Migration
- **Story 4.6**: Jelly Shader → TSL Migration

### Phase 4: Fluid & Performance
- **Story 4.7**: Fluid System Activation (After 4.2, 4.4)
- **Story 4.8**: Performance Validation (After all above)

---

## Autonomous Execution Protocol

### For Each Story:

1. **READ** the implementation artifact
   - Example: `4-2-webgpu-renderer-init.md`

2. **LOAD** current codebase state
   - Read existing files mentioned in the artifact
   - Understand current architecture

3. **IMPLEMENT** following acceptance criteria
   - Create new files as specified
   - Modify existing files preserving compatibility
   - Follow code patterns from Dev Notes sections

4. **TEST** each implementation
   - Run `npm run dev` to check for errors
   - Verify TypeScript compiles
   - Check console for warnings

5. **COMMIT** after each story completion
   - Commit message format: `feat: STORY-004-0XX - [Story Name]`
   - Include co-author tag

6. **TRACK** progress
   - Mark completed tasks in implementation artifact
   - Update sprint status if needed

7. **CONTINUE** to next story
   - Do not pause for approval unless critical error
   - Log completion and proceed

---

## Critical Decision Points (Require Human Input)

PAUSE and ask user if:
1. **Package installation needed** - Confirm before running `npm install`
2. **Breaking changes detected** - If modifying public APIs
3. **Major architectural deviation** - If implementation differs significantly from spec
4. **Compilation failure** - If TypeScript errors cannot be resolved
5. **Performance regression** - If FPS drops below targets

**DO NOT PAUSE FOR:**
- Code formatting/style
- Minor refactoring within same approach
- Test writing (write tests as specified)
- Comments/documentation

---

## Project Context

**Working Directory:** `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana`

**Key Files:**
- `src/main.js` - Main entry point
- `src/core/renderer.js` - Existing renderer detection
- `src/shaders/index.ts` - GLSL shaders (to be converted to TSL)
- `src/entities/VortexMaterial.ts` - Vortex material class
- `src/entities/WaterMaterial.ts` - Water material class
- `src/entities/JellyCreature.ts` - Jelly entity
- `src/entities/ShellCollectible.ts` - Shell entity
- `src/systems/fluid/index.ts` - Fluid system

**Three.js Version:** Check `package.json` - requires r171.0+ for WebGPU/TSL

**Visionary Platform Notes:**
- Ubuntu: NOT supported (fp16 WebGPU bug)
- macOS: Performance limited (M4 Max+ recommended)
- Windows 10/11: RECOMMENDED

---

## Execution Checklist

Use this checklist to track progress. Mark each as completed:

### Foundation Phase
- [ ] Story 4.1: Visionary Integration
  - [ ] Install visionary-core package
  - [ ] Create VisionarySplatRenderer class
  - [ ] Integrate with render loop
  - [ ] Test depth occlusion
- [ ] Story 4.2: WebGPU Renderer Init
  - [ ] Create createWebGPURenderer.ts
  - [ ] Implement feature detection
  - [ ] Add platform detection
  - [ ] Configure renderer properties
  - [ ] Implement async shader compilation
  - [ ] Add pixel ratio limits
  - [ ] Set up animation loop

### TSL Migration Phase
- [ ] Story 4.3: Vortex Shader TSL
  - [ ] Create src/shaders/tsl/VortexShader.ts
  - [ ] Implement vertex displacement in TSL
  - [ ] Implement fragment shader in TSL
  - [ ] Create VortexMaterialTSL class
  - [ ] Update VortexSystem integration
- [ ] Story 4.4: Water Material TSL
  - [ ] Create src/shaders/tsl/WaterShader.ts
  - [ ] Implement wave animation with harp response
  - [ ] Implement jelly creature ripples
  - [ ] Implement fresnel and transmission
  - [ ] Implement bioluminescent color mixing
  - [ ] Implement sphere constraint for duet progress
  - [ ] Create WaterMaterialTSL class
- [ ] Story 4.5: Shell SDF TSL
  - [ ] Create src/shaders/tsl/ShellShader.ts
  - [ ] Implement nautilus spiral SDF
  - [ ] Implement iridescence effect
  - [ ] Implement dissolve effect
  - [ ] Implement appear animation
  - [ ] Create ShellMaterialTSL class
- [ ] Story 4.6: Jelly Shader TSL
  - [ ] Create src/shaders/tsl/JellyShader.ts
  - [ ] Implement vertex displacement in TSL
  - [ ] Implement fragment shader in TSL
  - [ ] Create JellyMaterialTSL class
  - [ ] Update JellyCreature entity

### Fluid & Performance Phase
- [ ] Story 4.7: Fluid System Activation
  - [ ] Create src/systems/fluid/compute/FluidCompute.ts
  - [ ] Create ParticleBuffers.ts
  - [ ] Create MLSMPMStages.ts
  - [ ] Integrate compute node with main render loop
  - [ ] Implement particle physics in TSL
  - [ ] Test performance with 10,000 particles
- [ ] Story 4.8: Performance Validation
  - [ ] Create src/utils/DeviceCapabilities.ts
  - [ ] Create src/utils/PerformanceProfiler.ts
  - [ ] Create src/utils/QualityManager.ts
  - [ ] Create src/utils/PerformanceHUD.ts (optional)
  - [ ] Implement async shader compilation
  - [ ] Test on desktop (60 FPS target)
  - [ ] Test on mobile (30 FPS target)
  - [ ] Verify Visionary depth occlusion
  - [ ] Document platform limitations

---

## Git Commit Pattern

After each story, commit with:

```
feat: STORY-004-0XX - [Story Name]

- [Key change 1]
- [Key change 2]
- [Key change 3]

Closes #[story-number-in-tracking]

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

---

## Error Handling Strategy

1. **TypeScript Errors**
   - Fix immediately if clear
   - Log and continue if minor type mismatch can be resolved later
   - PAUSE if blocking compilation

2. **Runtime Errors**
   - Check browser console
   - Verify WebGPU availability
   - Check Visionary platform support
   - PAUSE if cannot diagnose

3. **Import/Export Errors**
   - Verify file paths
   - Check module resolution
   - Fix before proceeding

---

## Success Criteria

**EPIC COMPLETE when:**
- All 8 stories implemented
- All TypeScript compiles without errors
- `npm run dev` runs successfully
- WebGPU renderer initializes
- Visionary splats render with depth occlusion
- All shaders converted to TSL
- Fluid compute shaders execute
- Performance targets met (60 FPS desktop, 30 FPS mobile)
- All commits pushed to branch

---

## Final Output

When complete, generate:
1. Summary of all changes
2. Files created/modified list
3. Performance metrics
4. Known issues or limitations
5. Next steps for Epic 5

---

## Start Command

Begin execution immediately with:
1. Check current git branch
2. Verify Three.js version in package.json
3. Start with Story 4.2 (WebGPU Init) OR 4.1 (Visionary) - your choice based on dependencies
4. Execute autonomously until complete

**GO NOW - DO NOT WAIT FOR USER INPUT**
