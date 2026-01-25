# Story 1.1: Visual Foundation - Water & Vortex Shaders

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **player entering the Archive of Voices (Music Room)**,
I want **the water surface to respond musically and the vortex to visually demonstrate the ship's latent power**,
so that **I feel the Vimana is a living entity awakening to my presence**.

## Acceptance Criteria

1. [ ] Water shader applied to ArenaFloor mesh with bioluminescent resonance that responds to 6 harp strings
2. [ ] Water ripples respond to all 6 string frequencies individually (C, D, E, F, G, A)
3. [ ] SDF torus vortex mesh created at proper position (0, 0.5, 2) in scene coordinates
4. [ ] Vortex particle system with 2000 particles with LOD optimization
5. [ ] Particles flow in spiral pattern through torus based on activation level
6. [ ] Glow intensity scales with activation level (0-1), representing ship awakening
7. [ ] 60 FPS maintained with all systems active (desktop target)

## Tasks / Subtasks

- [ ] Create EnhancedWaterMaterial class with GLSL shaders (AC: #1, #2)
  - [ ] Vertex shader with wave displacement from 6 string frequencies
  - [ ] Fragment shader with bioluminescent glow, caustics, and fresnel effects
  - [ ] Uniforms: uTime, uHarpFrequencies[6], uHarpVelocities[6], uDuetProgress, uShipPatience, uBioluminescentColor, uHarmonicResonance
  - [ ] Apply material to ArenaFloor mesh (find by name in GLB scene)
- [ ] Create VortexParticles class with LOD system (AC: #4, #5)
  - [ ] Initialize 2000 particles in torus distribution pattern
  - [ ] Implement LOD skip logic: 50% skip at <0.3 activation, 25% at <0.6, 0% at full
  - [ ] Spin particles around torus with proper trigonometry (CRITICAL BUG FIX: Z coordinate uses dist, not sinAngle)
  - [ ] Randomize tubeOffset on each reset (CRITICAL BUG FIX from Party Mode review)
  - [ ] Cyan-to-purple gradient colors with additive blending
- [ ] Create VortexMaterial with SDF torus shader (AC: #3, #6)
  - [ ] Vertex shader with activation-based spin and breathing displacement
  - [ ] Fragment shader with SDF torus distance field, swirl noise, edge glow
  - [ ] Uniforms: uTime, uVortexActivation, uDuetProgress, uInnerColor (cyan), uOuterColor (purple), uCoreColor (white)
  - [ ] Apply to torus geometry at position (0, 0.5, 2)
- [ ] Create VortexSystem wrapper class (AC: #6, #7)
  - [ ] Expose updateDuetProgress(progress) method for activation control
  - [ ] Implement destroy() cleanup method for memory management
  - [ ] Connect water harmonicResonance and duetProgress to visual intensity
- [ ] Performance testing and optimization (AC: #7)
  - [ ] Measure FPS with all systems active
  - [ ] Verify LOD particle counts on low-end devices
  - [ ] Test shader compilation time (<5 seconds target)

## Dev Notes

### Project Structure Notes

**Primary Framework:** Three.js r160+ (WebGPU/WebGL2)
**Shader Language:** GLSL ES 3.0 (WebGPU) with fallback to GLSL ES 1.0 (WebGL2)
**Scene Format:** GLB with Gaussian Splatting via Shadow Engine

**File Organization:**
```
vimana/
├── src/
│   ├── shaders/
│   │   ├── water-vertex.glsl
│   │   ├── water-fragment.glsl
│   │   ├── vortex-vertex.glsl
│   │   └── vortex-fragment.glsl
│   ├── entities/
│   │   ├── WaterMaterial.ts
│   │   ├── VortexMaterial.ts
│   │   ├── VortexParticles.ts
│   │   └── VortexSystem.ts
│   └── scenes/
│       └── HarpRoom.ts (main scene controller)
```

### Critical Implementation Constraints

**CRITICAL BUGS FIXED (Party Mode Review - MUST PRESERVE):**

1. **VortexParticles.update() line 415** - Z-coordinate MUST use `dist` NOT `sinAngle`:
   ```javascript
   // WRONG (original bug):
   this.positions[i * 3 + 2] = this.positions[i * 3 + 1];
   // CORRECT:
   this.positions[i * 3 + 2] = dist;
   ```

2. **VortexParticles.resetParticle() line 375** - tubeOffset MUST be randomized on each reset:
   ```javascript
   // FIX: Move randomization INSIDE resetParticle, not in init
   const tubeOffset = (Math.random() - 0.5) * 0.8;
   ```

3. **Particle LOD System** - MUST skip particles based on activation:
   ```javascript
   const lodSkip = activationLevel < 0.3 ? (i % 2 === 0) :
                   activationLevel < 0.6 ? (i % 4 === 0) : false;
   if (lodSkip) continue;
   ```

### Technical Requirements

**Water Shader Specifications:**
- 6-string frequency array uniforms (Float32Array)
- Per-string velocity tracking for intensity modulation
- Bioluminescent base color: #00ff88 (cyan-green)
- Fresnel exponent: 3.0 for surface transparency effect
- Transparency: 0.6-0.95 based on viewing angle
- DoubleSide rendering for proper visibility

**Vortex Specifications:**
- Torus major radius: 2.0 units
- Torus tube radius: 0.4 units
- Position: (0, 0.5, 2) world coordinates
- Max particles: 2000 (desktop), 500 (mobile fallback)
- Particle size: 0.05 units
- Additive blending for glow effect
- DepthWrite: false for transparency

**Performance Targets:**
- Desktop: 60 FPS with all systems
- Mobile: 30 FPS with reduced particles
- Shader compilation: <5 seconds
- Memory: Stable, no leaks on transition

### Memory Management Requirements

**CRITICAL:** All shader-based entities MUST implement destroy() methods:

```javascript
// Required cleanup pattern:
destroy() {
  if (this.particleSystem) {
    this.scene.remove(this.particleSystem);
    this.particleGeometry.dispose();
    this.particleSystem.material.dispose();
    this.particleSystem = null;
  }
}
```

**Scene Transition Pattern:**
- Call destroy() before loading next scene
- Clear all material.dispose() and geometry.dispose()
- Remove meshes from scene before disposal
- Nullify all references to enable GC

### Dependencies

**Previous Story:** None (first story in epic)
**Next Story:** 1.2 Jelly Creatures (depends on visual foundation being complete)

**External Dependencies:**
- Three.js core: scene graph, materials, geometries
- GLB loader: ArenaFloor mesh by name
- Web Audio API: for later audio integration (Story 1.3+)

### References

- [Source: VIMANA_HARP_IMPLEMENTATION_PLAN.md#Water Surface]
- [Source: VIMANA_HARP_IMPLEMENTATION_PLAN.md#Vortex Particles]
- [Source: VIMANA_HARP_IMPLEMENTATION_PLAN.md#Bug Fixes Applied]
- [Source: music-room-proto-epic.md#Story 1.1]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

### File List

- `src/shaders/water-vertex.glsl` (create)
- `src/shaders/water-fragment.glsl` (create)
- `src/shaders/vortex-vertex.glsl` (create)
- `src/shaders/vortex-fragment.glsl` (create)
- `src/entities/WaterMaterial.ts` (create)
- `src/entities/VortexMaterial.ts` (create)
- `src/entities/VortexParticles.ts` (create)
- `src/entities/VortexSystem.ts` (create)
