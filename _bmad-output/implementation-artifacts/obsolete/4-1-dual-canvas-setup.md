# Story 4.1: Dual Canvas Setup

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a rendering engineer,
I want to create a dual-canvas composition system with WebGPU and WebGL2 layers,
so that I can render WebGPU content and Gaussian splats simultaneously in a unified visual experience.

## Acceptance Criteria

1. `RenderComposer.ts` class created in `src/rendering/` directory
2. Dual HTML canvas elements created with CSS absolute positioning
3. WebGPU canvas at z-index: 1 (bottom layer) for primary rendering
4. WebGL2 canvas at z-index: 2 (transparent, pointer-events: none) for splat overlay
5. UI layer at z-index: 3 (pointer-events: auto) for game UI
6. Camera state synchronization between renderers
7. Resize handler for both canvases that maintains aspect ratio
8. Proper cleanup method that disposes both renderers

## Tasks / Subtasks

- [ ] Create RenderComposer class structure (AC: 1)
  - [ ] Define constructor accepting container element
  - [ ] Initialize private properties for both renderers
  - [ ] Define public interface (initialize, render, dispose, resize)
- [ ] Implement dual canvas HTML structure (AC: 2, 3, 4, 5)
  - [ ] Create WebGPU canvas element
  - [ ] Create WebGL2 canvas element with transparency
  - [ ] Apply CSS positioning (absolute, full width/height)
  - [ ] Set z-index layering correctly
  - [ ] Configure pointer-events for click-through behavior
- [ ] Implement camera synchronization (AC: 6)
  - [ ] Store single camera instance
  - [ ] Sync camera properties to both renderers each frame
  - [ ] Handle camera updates (position, rotation, projection matrix)
- [ ] Implement resize handling (AC: 7)
  - [ ] Listen to window resize events
  - [ ] Update both canvas dimensions
  - [ ] Update camera aspect ratio
  - [ ] Handle visualViewport for mobile browsers
- [ ] Implement disposal/cleanup (AC: 8)
  - [ ] Dispose WebGPU renderer properly
  - [ ] Dispose WebGL2 renderer properly
  - [ ] Remove canvas elements from DOM
  - [ ] Clear event listeners

## Dev Notes

### Existing Architecture Context

**Current Rendering Setup:**
- Single renderer created in `src/main.js` (line 100-104)
- Uses `createOptimalRenderer()` from `src/core/renderer.js`
- Currently forces WebGL2 for SparkRenderer compatibility (`requiresWebGL: true`)
- Canvas appended directly to `document.body`

**Key Integration Points:**
1. `main.js` - Game initialization, current renderer creation
2. `src/core/renderer.js` - WebGPU detection and renderer selection
3. `SparkRenderer` from `@sparkjsdev/spark` - WebGL2-only Gaussian splat renderer

**Modification Requirements:**
- Replace single renderer pattern with RenderComposer
- WebGPU canvas becomes the "main" renderer
- WebGL2 canvas is created solely for SparkRenderer overlay

### Project Structure Notes

**New file to create:**
```
src/rendering/
  RenderComposer.ts     # New class - dual canvas orchestration
```

**Files to modify:**
```
src/main.js                    # Integrate RenderComposer
src/core/renderer.js           # May need update for dual-canvas support
src/scenes/HarpRoom.ts         # Update to use dual-renderer if needed
```

**Alignment with unified project structure:**
- Follow existing TypeScript patterns in `src/entities/`
- Use class-based architecture matching existing entities
- Export patterns match `src/index.ts` barrel exports

### CSS Layering Specification

```css
/* Layer specification for dual canvas setup */
#webgpu-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1;  /* Bottom layer - primary rendering */
}

#webgl2-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 2;  /* Middle layer - splat overlay */
  background: transparent;
  pointer-events: none;  /* Click through to WebGPU layer */
}

#ui-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 3;  /* Top layer - UI elements */
  pointer-events: auto;  /* UI receives clicks */
}
```

### Camera Synchronization Pattern

```typescript
// Single camera shared between renderers
private camera: THREE.PerspectiveCamera;
private webgpuRenderer: THREE.WebGPURenderer;
private webglRenderer: SplatOverlayRenderer; // Story 4.7

// In render loop - sync camera before each render
public render(): void {
  // Camera is already updated by game logic
  // Both renderers share the same camera instance
  this.webgpuRenderer.render(this.scene, this.camera);
  this.webglRenderer.render(this.camera); // Splat overlay
}
```

### Memory Management

Critical for dual-context setup:
- Each context has its own GPU memory pool
- Must dispose BOTH renderers to avoid leaks
- Canvas elements must be removed from DOM
- Event listeners must be cleared

### Testing Requirements

1. Verify both canvases are created and visible in DOM
2. Confirm click events pass through overlay to WebGPU layer
3. Test resize behavior on desktop and mobile
4. Verify no memory leaks after disposal
5. Check that z-index layering produces correct visual result

### References

- Epic: `../_bmad-output/planning-artifacts/epics/EPIC-004-webgpu-migration.md` - Lines 46-78
- Architecture diagram: Epic lines 373-402 (dual-layer visualization)
- Existing renderer: `src/core/renderer.js` - WebGPU detection pattern
- Current integration: `src/main.js` lines 95-150 (renderer setup)

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

### Completion Notes List

### File List
