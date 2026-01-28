# EPIC-001: WaterBall Fluid Simulation System

**Epic ID**: `EPIC-001`
**Status**: `Ready for Dev`
**Points**: `29` (8+5+3+3+5+5)
**Sprint**: `Phase 4 - Fluid & Music`

---

## Overview

Replace the current static water shader in HarpRoom with a full MLS-MPM particle-based fluid simulation derived from [matsuoka-601/WaterBall](https://github.com/matsuoka-601/WaterBall). The water will respond to harp string vibrations, transform into a sphere tunnel as music progresses, and interact with player movement.

---

## Success Definition

**Done When:**
- Water visually matches WaterBall (side-by-side comparison)
- All 6 harp strings create unique ripple patterns in water
- Water transforms from flat plane → hollow sphere tunnel as `uDuetProgress` goes 0→1
- Player walking through water creates realistic displacement and wake
- Performance: 60fps with 10,000 particles on target hardware

---

## Stories

| ID | Title | Points | Status | Owner |
|----|-------|--------|--------|-------|
| STORY-001 | [MLS-MPM Particle Physics System](../stories/story-001-particle-physics.md) | 8 | Ready for Dev | TBD |
| STORY-002 | [Depth & Thickness Rendering Pipeline](../stories/story-002-depth-thickness.md) | 5 | Ready for Dev | TBD |
| STORY-003 | [Fluid Surface Shader](../stories/story-003-fluid-surface.md) | 3 | Ready for Dev | TBD |
| STORY-004 | [Sphere Constraint Animation](../stories/story-004-sphere-animation.md) | 3 | Ready for Dev | TBD |
| STORY-005 | [Harp-to-Water Interaction System](../stories/story-005-harp-interaction.md) | 5 | Ready for Dev | TBD |
| STORY-006 | [Player Collision & Water Displacement](../stories/story-006-player-collision.md) | 5 | Ready for Dev | TBD |

**Total Points: 29**

---

## Technical References

### Source Repositories
- [matsuoka-601/WaterBall](https://github.com/matsuoka-601/WaterBall) - Primary reference
  - MLS-MPM compute shaders
  - Multi-pass rendering pipeline
  - Sphere constraint implementation
- [matsuoka-601/WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) - Secondary reference
  - Similar MLS-MPM implementation
  - Alternative rendering approaches

### Key Files from WaterBall
| File | Purpose |
|------|---------|
| `mls-mpm/mls-mpm.ts` | Particle physics orchestration |
| `mls-mpm/clearGrid.wgsl` | Reset grid cells |
| `mls-mpm/p2g_1.wgsl` | Particle-to-grid mass/velocity transfer |
| `mls-mpm/p2g_2.wgsl` | Particle-to-grid MLS-MPM update |
| `mls-mpm/updateGrid.wgsl` | Grid force application |
| `mls-mpm/g2p.wgsl` | Grid-to-particle with sphere constraint |
| `render/fluidRender.ts` | Rendering pipeline orchestration |
| `render/depthMap.wgsl` | Depth pass shader |
| `render/thicknessMap.wgsl` | Thickness pass shader |
| `render/bilateral.wgsl` | Depth smoothing filter |
| `render/fluid.wgsl` | Final fluid surface shader |

---

## Dependencies

### Required
- WebGPU-capable browser (Chrome 113+, Edge 113+)
- Three.js 0.180.0 with WebGPU support
- Existing HarpRoom scene with 6 harp strings
- Existing `uDuetProgress` variable for music progression

### Blocked By
- **SparkRenderer (@sparkjsdev/spark)**: WebGL-only library
  - Current water uses SparkRenderer
  - WaterBall is pure WebGPU
  - **Decision Required**: Hybrid rendering OR migrate Spark to WebGPU (6-8 month effort)

### Blocking
- None (can begin STORY-001 immediately)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WATERBALL ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUTS                      COMPUTE                    RENDERING       │
│  ┌──────────┐              ┌─────────────┐          ┌──────────────┐   │
│  │ Harp     │─────────────││             │          │  Depth Pass  │   │
│  │ Strings  │  Forces     ││  MLS-MPM    │─────────│  (r32float)  │   │
│  │ (6x)     │              ││  Simulator  │          │      ↓       │   │
│  └──────────┘              ││             │          │  Bilateral   │   │
│                             ││  10,000     │          │  Filter (4x) │   │
│  ┌──────────┐              ││  particles  │          │      ↓       │   │
│  │ Player   │─────────────││             │          │Thickness Pass│   │
│  │ Movement │  Displace   ││             │          │  (r16float)  │   │
│  └──────────┘              ││             │          │      ↓       │   │
│                             └─────────────┘          │   Gaussian   │   │
│  ┌──────────┐                                       │  Blur (1x)   │   │
│  │Duet      │───────────────────────────────────────│      ↓       │   │
│  │Progress  │  Animate Box → Sphere                 │ Fluid Shader │   │
│  │(0→1)     │                                       │  (Fresnel)   │   │
│  └──────────┘                                       └──────────────┘   │
│                                                             ↓           │
│                                                        Final Water     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
src/systems/fluid/
├── MLSMPMSimulator.ts                 # Main physics orchestrator
├── compute/
│   ├── clearGrid.wgsl
│   ├── p2g_1.wgsl
│   ├── p2g_2.wgsl
│   ├── updateGrid.wgsl
│   ├── g2p.wgsl
│   └── copyPosition.wgsl
├── render/
│   ├── DepthThicknessRenderer.ts      # Depth + thickness passes
│   ├── FluidSurfaceRenderer.ts        # Final fluid shader
│   └── shaders/
│       ├── depthMap.wgsl
│       ├── thicknessMap.wgsl
│       ├── bilateral.wgsl
│       ├── gaussian.wgsl
│       └── fluid.wgsl
├── interaction/
│   ├── HarpWaterInteraction.ts        # String → water coupling
│   ├── PlayerWaterInteraction.ts      # Player → water coupling
│   └── StringRippleEffect.ts          # Visual ripple rings
├── animation/
│   └── SphereConstraintAnimator.ts    # Plane → sphere animation
└── types.ts                           # Particle, Cell structs
```

---

## Epic Acceptance Criteria

- [ ] All 6 stories marked as complete
- [ ] Side-by-side screenshot comparison with WaterBall shows visual parity
- [ ] Each harp string (1-6) produces identifiable ripple pattern
- [ ] `uDuetProgress` from 0→1 animates water to sphere tunnel
- [ ] Player can walk through hollow center of tunnel
- [ ] Performance: 60fps maintained during all interactions
- [ ] Debug views available: `window.debugVimana.fluid.*`
- [ ] No WebGPU validation errors in console

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| SparkRenderer incompatibility | High | Hybrid rendering: WebGPU for water, WebGL for spark effects |
| Performance at 10K particles | Medium | Profile compute shaders; reduce particle count if needed |
| Sphere constraint physics bugs | Medium | Test animation progression thoroughly; add fallback to flat water |
| Player collision edge cases | Low | Clamp forces; test swimming/jumping scenarios |

---

## Timeline Notes

- **STORY-001** is the foundation - blocks all other stories
- **STORY-002, STORY-003** can develop in parallel after STORY-001
- **STORY-004, STORY-005, STORY-006** are interaction layers - can parallelize

Estimated effort: **3-4 weeks** with one developer focused on this epic.

---

**Sources:**
- [WaterBall Repository](https://github.com/matsuoka-601/WaterBall)
- [WebGPU-Ocean Repository](https://github.com/matsuoka-601/WebGPU-Ocean)
- [WebGPU Fluid Simulations Article](https://tympanis.net/codrops/2025/02/26/webgpu-fluid-simulations-high-performance-real-time-rendering/)
