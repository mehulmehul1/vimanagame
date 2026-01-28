# Sprint Change Proposal: WebGPU Architecture - Visionary Single-Canvas Migration

**Date:** 2026-01-27
**Epic Affected:** EPIC-004-webgpu-migration
**Scope:** Major Architecture Change
**Status:** Awaiting Approval

---

## Section 1: Issue Summary

### Problem Statement

The **dual-canvas architecture** proposed in EPIC-004-webgpu-migration has a **critical technical flaw**:

> **WebGPU and WebGL2 are separate GPU contexts with separate depth buffers.**

This means:
- Gaussian splats rendered on the WebGL2 overlay canvas **cannot depth-test** against meshes rendered on the WebGPU canvas
- Splats would **always float on top**, visible through walls and other geometry
- Occlusion would be completely broken

### Discovery Context

- **When:** During story creation phase, before implementation began
- **How:** Architectural review identified depth buffer sharing limitation
- **Evidence:** WebGPU and WebGL2 specifications confirm contexts are isolated

### Visual Example of the Problem

```
DUAL-CANVAS (BROKEN):
┌─────────────────────────────────────┐
│  LAYER 2: WebGL2 Canvas (SPLATS)    │  ← Separate depth buffer
│  ✗ Splats always on top             │
├─────────────────────────────────────┤
│  LAYER 1: WebGPU Canvas (MESHES)    │  ← Separate depth buffer
│  ✗ Walls cannot occlude splats      │
└─────────────────────────────────────┘

SINGLE-CANVAS (VISIONARY - CORRECT):
┌─────────────────────────────────────┐
│  Single WebGPU Context              │  ← Shared depth buffer
│  ✓ Splats properly occluded         │
│  ✓ Walls block splats correctly     │
└─────────────────────────────────────┘
```

---

## Section 2: Impact Analysis

### Epic Impact

| Epic | Impact | Action Required |
|------|--------|-----------------|
| **EPIC-004-webgpu-migration** | MAJOR | Replace dual-canvas architecture with Visionary single-canvas |

**Stories Affected:**

| Story ID | Title | Action |
|----------|-------|--------|
| **4.1** | Dual Canvas Setup | ❌ OBSOLETE - Replace with Visionary integration |
| **4.2** | WebGPU Renderer Initialization | ✅ KEEP - Still needed for Visionary |
| **4.3** | Vortex Shader → TSL Migration | ✅ KEEP - Unchanged |
| **4.4** | Water Material → TSL Migration | ✅ KEEP - Unchanged |
| **4.5** | Shell SDF → TSL Migration | ✅ KEEP - Unchanged |
| **4.6** | Jelly Shader → TSL Migration | ✅ KEEP - Unchanged |
| **4.7** | Splat Overlay Adapter | ❌ OBSOLETE - Visionary handles this internally |
| **4.8** | Fluid System Activation | ✅ KEEP - Unchanged |
| **4.9** | Performance Validation | ✅ KEEP - Minor updates |

### Artifact Conflicts

| Artifact | Conflict | Changes Needed |
|----------|----------|----------------|
| **EPIC-004** | Architecture diagram | Replace dual-layer with single-canvas diagram |
| **EPIC-004** | Story list | Remove 4.1, 4.7; Add new Visionary story |
| **Architecture docs** | Rendering approach | Update to reflect Visionary integration |

### Technical Impact

| Area | Impact |
|------|--------|
| **Code changes** | No code written yet - no rollback needed |
| **Dependencies** | Add: `visionary-core` package |
| **Three.js version** | Requires r171.0+ (check compatibility) |
| **Platform support** | Ubuntu NOT supported by Visionary; macOS limited |

---

## Section 3: Recommended Approach

### Selection: **Option 1 - Direct Adjustment**

**Rationale:**
- No implementation code exists yet - zero rollback cost
- Stories 4.3-4.6, 4.8-4.9 remain valid and can proceed
- Only stories 4.1 and 4.7 need replacement
- Visionary provides **superior solution** with:
  - Proper depth compositing
  - Native WebGPU compute support
  - Better long-term architecture

### Effort Estimate

| Task | Estimate |
|------|----------|
| Revise EPIC-004 documentation | 2 hours |
| Create new Visionary integration story | 1 hour |
| Archive obsolete stories 4.1, 4.7 | 0.5 hours |
| **Total** | **3.5 hours** |

### Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Ubuntu platform support | Medium | Document limitation; target Windows/macOS |
| Three.js version compatibility | Low | Verify current version (likely compatible) |
| Visionary project stability | Low | Apache 2.0 license; active development |

---

## Section 4: Detailed Change Proposals

### EPIC-004 Architecture Diagram Replacement

**OLD (Lines 373-402):**
```markdown
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL-RENDERER ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LAYER 1 (Bottom): WebGPU Canvas - Primary Renderer     │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LAYER 2 (Top): WebGL2 Canvas - Overlay (Transparent)   │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LAYER 3: UI Overlay (pointer-events: auto)            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**NEW:**
```markdown
┌─────────────────────────────────────────────────────────────────┐
│                 VISIONARY HYBRID RENDERING ARCHITECTURE          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  SINGLE WebGPU Canvas + Context                          │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │  Pass A: Three.js Scene → RenderTarget            │  │    │
│  │  │        (Captures Color + Depth)                   │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │  Pass B: Gaussian Splats (uses captured depth)    │  │    │
│  │  │        Visionary: GaussianThreeJSRenderer         │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  │  ┌───────────────────────────────────────────────────┐  │    │
│  │  │  Result: Proper occlusion between meshes & splats │  │    │
│  │  └───────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Story Changes

#### OBSOLETE: Story 4.1 - Dual Canvas Setup

**Status:** ❌ DELETE
**Reason:** Dual-canvas approach replaced by Visionary single-canvas
**Action:** Archive to `_bmad-output/implementation-artifacts/obsolete/`

#### NEW: Story 4.1 - Visionary Integration

**Status:** ✅ CREATE
**Title:** Visionary Gaussian Splat Integration
**Description:** Integrate Visionary's WebGPU Gaussian splat renderer with proper depth compositing

**Acceptance Criteria:**
1. Install `visionary-core` package
2. Initialize `GaussianThreeJSRenderer` with WebGPU renderer
3. Load Gaussian splat files (PLY, SPLAT format)
4. Verify depth occlusion works (splats behind walls are hidden)
5. Integrate with existing render loop
6. Support 6 jelly creature splats

**Technical Implementation:**
```typescript
// src/rendering/VisionarySplatRenderer.ts
import { GaussianThreeJSRenderer } from 'visionary';
import { GaussianModel } from 'visionary';

export class VisionarySplatRenderer {
  private renderer: GaussianThreeJSRenderer;
  private models: GaussianModel[] = [];

  async initialize(
    webgpuRenderer: THREE.WebGPURenderer,
    scene: THREE.Scene
  ): Promise<void> {
    this.renderer = new GaussianThreeJSRenderer(
      webgpuRenderer,
      scene,
      this.models
    );
    await this.renderer.init();
    scene.add(this.renderer); // Gets onBeforeRender callbacks
  }

  async loadSplat(url: string): Promise<void> {
    const model = await GaussianModel.load(url);
    this.models.push(model);
  }

  render(camera: THREE.Camera): void {
    this.renderer.renderThreeScene(camera);  // Captures depth
    this.renderer.drawSplats(/* ... */);     // Uses depth
  }
}
```

#### OBSOLETE: Story 4.7 - Splat Overlay Adapter

**Status:** ❌ DELETE
**Reason:** Visionary handles splat rendering internally; no overlay adapter needed
**Action:** Archive to `_bmad-output/implementation-artifacts/obsolete/`

#### UPDATE: Story 4.2 - WebGPU Renderer Initialization

**Section:** Dev Notes
**Change:** Add Visionary compatibility note

**OLD:**
```markdown
### Integration in main.js:
```javascript
this.renderer = await createOptimalRenderer(
  { alpha: true, antialias: false },
  null,
  { requiresWebGL: true } // Currently forces WebGL2 for SparkRenderer
);
```
```

**NEW:**
```markdown
### Integration in main.js:
```javascript
this.renderer = await createOptimalRenderer(
  { alpha: true, antialias: false },
  null,
  { requiresWebGL: false } // Changed: Use WebGPU for Visionary
);

// Initialize Visionary after renderer
this.splatRenderer = new VisionarySplatRenderer();
await this.splatRenderer.initialize(this.renderer, this.scene);
```
```

---

## Section 5: Implementation Handoff

### Change Scope: **MODERATE**

**Rationale:** Requires backlog reorganization and story updates, but no code rollback.

### Handoff Responsibilities

| Role | Responsibilities |
|------|------------------|
| **Product Owner** | Approve story changes; update sprint backlog |
| **Development Team** | Implement new Story 4.1 (Visionary integration) |
| **Technical Lead** | Verify Three.js version compatibility |

### Success Criteria

1. ✅ EPIC-004 documentation updated with Visionary architecture
2. ✅ Stories 4.1 and 4.7 replaced with new Visionary integration story
3. ✅ Obsolete stories archived for reference
4. ✅ Development team can proceed with Story 4.1 implementation
5. ✅ Depth occlusion verified in testing (splats properly hidden behind walls)

### Next Steps

1. **Immediate:** Obtain user approval for this proposal
2. **Post-approval:** Update EPIC-004 documentation
3. **Post-approval:** Create new Story 4.1 file
4. **Post-approval:** Archive obsolete stories
5. **Post-approval:** Sprint planning can proceed

---

## Appendix: Visionary Platform Summary

### Key Features
- ✅ Native WebGPU powered
- ✅ **Hybrid Rendering Architecture** with depth compositing
- ✅ Universal asset loader (PLY, SPLAT, KSplat, SPZ, SOG, GLB, GLTF)
- ✅ Three.js integration plugin
- ✅ Apache 2.0 license

### Platform Requirements
| Platform | Status | Notes |
|----------|--------|-------|
| Windows 10/11 | ✅ Recommended | Discrete GPU (NVIDIA/AMD) |
| Ubuntu/Linux | ❌ NOT supported | fp16 WebGPU bug |
| macOS | ⚠️ Limited | M4 Max+ recommended |
| Browser | Chrome/Edge | WebGPU required |

### References
- GitHub: https://github.com/Visionary-Laboratory/visionary
- Documentation: https://ai4sports.opengvlab.com/help/index.html
- Three.js Integration: https://ai4sports.opengvlab.com/help/modules/12-three-integration/index.html

---

**Document prepared by:** Correct Course Workflow
**For approval by:** Mehul (Product Owner)
**Date:** 2026-01-27
