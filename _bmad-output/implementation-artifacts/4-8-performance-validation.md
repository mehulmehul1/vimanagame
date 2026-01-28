# Story 4.8: Performance Validation

Status: ready-for-dev

## Story

As a rendering engineer,
I want to profile and optimize the Visionary + WebGPU rendering system,
so that the game achieves target frame rates (60 FPS desktop, 30 FPS mobile) with acceptable memory usage.

## Acceptance Criteria

1. Frame time profiling implemented (WebGPU: 8ms, Fluid: 4ms, Splats: 3ms, Post: 1.5ms)
2. Desktop 60 FPS achieved (16.67ms budget)
3. Mobile 30 FPS achieved (33.33ms budget)
4. Memory profiling (<500MB on iOS Safari)
5. Device profile system (high/medium/low)
6. Pixel ratio limits enforced (1.5 mobile, 2 desktop)
7. Async shader compilation prevents frame drops
8. Visionary depth occlusion verified (splats hidden behind walls)
9. Platform capability detection (Ubuntu not supported)

## Tasks / Subtasks

- [ ] Create performance profiler module (AC: 1, 8)
  - [ ] Create `src/utils/PerformanceProfiler.ts`
  - [ ] Measure WebGPU render time
  - [ ] Measure Visionary splat render time
  - [ ] Measure compute shader execution time
  - [ ] Measure post-processing time
  - [ ] Calculate FPS and frame time
- [ ] Implement device capability detection (AC: 4, 5, 9)
  - [ ] Create `src/utils/DeviceCapabilities.ts`
  - [ ] Detect mobile platform
  - [ ] Detect GPU capabilities
  - [ ] Check Visionary compatibility (Ubuntu, macOS)
  - [ ] Return device profile (high/medium/low)
- [ ] Create adaptive quality system (AC: 2, 3, 6)
  - [ ] Create `src/utils/QualityManager.ts`
  - [ ] Apply device-specific settings
  - [ ] Dynamic quality adjustment based on FPS
  - [ ] Texture size scaling
  - [ ] Shadow quality adjustment
- [ ] Implement pixel ratio limits (AC: 6)
  - [ ] Cap pixel ratio at 1.5 for mobile
  - [ ] Cap pixel ratio at 2.0 for desktop
  - [ ] Handle visualViewport for mobile browsers
- [ ] Implement async shader compilation (AC: 7)
  - [ ] Call `renderer.compileAsync()` before render loop
  - [ ] Show loading indicator during compilation
  - [ ] Handle compilation errors gracefully
- [ ] Create performance HUD (optional)
  - [ ] Display FPS counter
  - [ ] Show frame time breakdown
  - [ ] Memory usage indicator
  - [ ] Toggle with debug key

## Dev Notes

### Target Frame Time Budgets

**Desktop (60 FPS):**
| System | Budget |
|--------|--------|
| WebGPU Rendering | 8ms |
| Fluid Compute | 4ms |
| Visionary Splats | 3ms |
| Post-Processing | 1.5ms |
| **Total** | **16.5ms** |

**Mobile (30 FPS):**
| System | Budget |
|--------|--------|
| WebGPU Rendering | 15ms |
| Fluid Compute | 8ms |
| Visionary Splats | 6ms |
| Post-Processing | 3ms |
| **Total** | **32ms** |

### Device Capability Detection

```typescript
// src/utils/DeviceCapabilities.ts

export interface DeviceProfile {
  name: string;
  tier: 'high' | 'medium' | 'low';
  pixelRatio: number;
  textureSize: number;
  shadows: boolean;
  postprocessing: boolean;
  antialias: boolean;
  visionarySupported: boolean; // Ubuntu not supported
  particleCount: number;        // Fluid particles
  targetFPS: number;
}

const getDeviceProfile = (): DeviceProfile => {
  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
  const isUbuntu = /Ubuntu/i.test(navigator.userAgent);
  const isMacOS = /Mac|iPod|iPhone|iPad/i.test(navigator.platform) &&
                   !/Mobile|Android/i.test(navigator.userAgent);

  // Check for discrete GPU (desktop only)
  const hasDiscreteGPU = !isMobile; // Simplified - could use WebGL context

  // Visionary platform support
  const visionarySupported = !isUbuntu; // Ubuntu has fp16 WebGPU bug

  if (isMobile) {
    return {
      name: 'mobile',
      tier: 'low',
      pixelRatio: Math.min(window.devicePixelRatio, 1.5),
      textureSize: 512,           // Reduced textures
      shadows: false,
      postprocessing: false,
      antialias: false,
      visionarySupported,
      particleCount: 3000,         // Reduced particles
      targetFPS: 30
    };
  }

  if (isMacOS) {
    return {
      name: 'macos',
      tier: 'medium',             // macOS has limited GPU performance
      pixelRatio: Math.min(window.devicePixelRatio, 2),
      textureSize: 1024,
      shadows: true,
      postprocessing: true,
      antialias: false,
      visionarySupported,
      particleCount: 5000,
      targetFPS: 60
    };
  }

  // Desktop (Windows)
  if (hasDiscreteGPU) {
    return {
      name: 'desktop-high',
      tier: 'high',
      pixelRatio: Math.min(window.devicePixelRatio, 2),
      textureSize: 2048,
      shadows: true,
      postprocessing: true,
      antialias: true,
      visionarySupported,
      particleCount: 10000,
      targetFPS: 60
    };
  }

  return {
    name: 'desktop-low',
    tier: 'medium',
    pixelRatio: Math.min(window.devicePixelRatio, 2),
    textureSize: 1024,
    shadows: true,
    postprocessing: false,
    antialias: false,
    visionarySupported,
    particleCount: 5000,
    targetFPS: 60
  };
};
```

### Performance Profiler

```typescript
// src/utils/PerformanceProfiler.ts

export class PerformanceProfiler {
  private measurements: Map<string, number> = new Map();
  private frameStartTime: number = 0;
  private fps: number = 0;
  private frameTimes: number[] = [];

  startFrame(): void {
    this.frameStartTime = performance.now();
  }

  measure(name: string): void {
    this.measurements.set(name, performance.now());
  }

  getMeasurement(name: string): number {
    const start = this.measurements.get(name) || 0;
    return performance.now() - start;
  }

  endFrame(): void {
    const frameTime = performance.now() - this.frameStartTime;
    this.frameTimes.push(frameTime);

    // Keep last 60 frames for FPS calculation
    if (this.frameTimes.length > 60) {
      this.frameTimes.shift();
    }

    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    this.fps = 1000 / avgFrameTime;
  }

  getStats() {
    return {
      fps: this.fps,
      frameTime: this.frameTimes[this.frameTimes.length - 1] || 0,
      measurements: Object.fromEntries(this.measurements)
    };
  }

  // Memory usage (where supported)
  getMemoryUsage(): number {
    if ('memory' in performance) {
      return (performance as any).memory.usedJSHeapSize / 1024 / 1024; // MB
    }
    return 0;
  }
}
```

### Quality Manager

```typescript
// src/utils/QualityManager.ts

export class QualityManager {
  private profile: DeviceProfile;
  private currentFPS: number = 60;
  private adaptiveQuality: boolean = true;

  constructor(profile: DeviceProfile) {
    this.profile = profile;
  }

  applySettings(renderer: THREE.WebGPURenderer): void {
    // Apply pixel ratio cap
    renderer.setPixelRatio(this.profile.pixelRatio);

    // Set texture size limit (for materials)
    // This would be applied to texture loaders

    // Shadow configuration
    if (!this.profile.shadows) {
      renderer.shadowMap.enabled = false;
    }
  }

  update(fps: number): void {
    this.currentFPS = fps;

    if (!this.adaptiveQuality) return;

    // Dynamic quality adjustment
    if (fps < this.profile.targetFPS * 0.8) {
      // FPS dropped below 80% of target - reduce quality
      this.downgradeQuality();
    } else if (fps > this.profile.targetFPS * 1.1) {
      // FPS above 110% of target - can increase quality
      this.upgradeQuality();
    }
  }

  private downgradeQuality(): void {
    // Reduce quality settings
    // - Lower particle count
    // - Reduce texture resolution
    // - Disable post-processing
    // - Reduce shadow resolution
  }

  private upgradeQuality(): void {
    // Increase quality settings
    // - More particles
    // - Higher texture resolution
    // - Enable post-processing
    // - Higher shadow resolution
  }

  getProfile(): DeviceProfile {
    return this.profile;
  }
}
```

### Async Shader Compilation

```typescript
// src/rendering/createWebGPURenderer.ts (updated)

import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';

export async function createWebGPURenderer(
  canvas: HTMLCanvasElement,
  options?: THREE.WebGPURendererParameters
): Promise<THREE.WebGPURenderer> {

  // Show loading indicator
  showLoadingIndicator('Initializing WebGPU renderer...');

  // Check WebGPU availability
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser');
  }

  // Create renderer
  const renderer = new WebGPURenderer({
    alpha: true,
    antialias: false,
    ...options
  });

  // REQUIRED: Async initialization
  await renderer.init();

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.5;

  // Async shader compilation
  showLoadingIndicator('Compiling shaders...');

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);

  try {
    // Compile all shaders asynchronously
    await renderer.compileAsync(scene, camera);
  } catch (error) {
    console.error('Shader compilation failed:', error);
    throw error;
  }

  hideLoadingIndicator();

  return renderer;
}

function showLoadingIndicator(message: string): void {
  // Create or update loading overlay
  const existing = document.getElementById('loading-indicator');
  if (existing) {
    existing.textContent = message;
    return;
  }

  const indicator = document.createElement('div');
  indicator.id = 'loading-indicator';
  indicator.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 20px 40px;
    border-radius: 8px;
    font-family: sans-serif;
    z-index: 10000;
  `;
  indicator.textContent = message;
  document.body.appendChild(indicator);
}

function hideLoadingIndicator(): void {
  const indicator = document.getElementById('loading-indicator');
  if (indicator) {
    indicator.remove();
  }
}
```

### Visionary Platform Check

```typescript
// src/rendering/VisionarySplatRenderer.ts (updated)

export class VisionarySplatRenderer {
  private renderer: GaussianThreeJSRenderer | null = null;
  private models: GaussianModel[] = [];

  async initialize(
    webgpuRenderer: THREE.WebGPURenderer,
    scene: THREE.Scene
  ): Promise<void> {
    // Platform check
    const isUbuntu = /Ubuntu/i.test(navigator.userAgent);

    if (isUbuntu) {
      console.warn('Visionary is not supported on Ubuntu due to fp16 WebGPU bug');
      console.warn('Gaussian splats will be disabled');
      // Don't initialize Visionary
      return;
    }

    // macOS performance check
    const isMacOS = /Mac/i.test(navigator.platform);
    if (isMacOS) {
      console.warn('Visionary performance on macOS may be limited');
      console.warn('M4 Max+ chip recommended for optimal performance');
    }

    this.renderer = new GaussianThreeJSRenderer(
      webgpuRenderer,
      scene,
      this.models
    );
    await this.renderer.init();
    scene.add(this.renderer);
  }

  isSupported(): boolean {
    return this.renderer !== null;
  }

  // ... rest of class
}
```

### Performance HUD (Optional)

```typescript
// src/utils/PerformanceHUD.ts

export class PerformanceHUD {
  private element: HTMLElement;
  private profiler: PerformanceProfiler;

  constructor(profiler: PerformanceProfiler) {
    this.profiler = profiler;
    this.element = this.createHUD();
  }

  private createHUD(): HTMLElement {
    const hud = document.createElement('div');
    hud.id = 'perf-hud';
    hud.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: #0f0;
      padding: 10px;
      font-family: monospace;
      font-size: 12px;
      z-index: 9999;
      pointer-events: none;
      display: none;
    `;
    document.body.appendChild(hud);
    return hud;
  }

  update(): void {
    const stats = this.profiler.getStats();
    const memory = this.profiler.getMemoryUsage();

    this.element.innerHTML = `
      FPS: ${stats.fps.toFixed(1)}<br>
      Frame: ${stats.frameTime.toFixed(2)}ms<br>
      Memory: ${memory.toFixed(1)}MB<br>
      WebGPU: ${this.profiler.getMeasurement('webgpu')?.toFixed(2) || 0}ms<br>
      Fluid: ${this.profiler.getMeasurement('fluid')?.toFixed(2) || 0}ms<br>
      Splats: ${this.profiler.getMeasurement('splats')?.toFixed(2) || 0}ms<br>
      Post: ${this.profiler.getMeasurement('post')?.toFixed(2) || 0}ms
    `;
  }

  toggle(): void {
    this.element.style.display =
      this.element.style.display === 'none' ? 'block' : 'none';
  }
}
```

### Memory Profiling

```typescript
// Memory tracking and warnings

export class MemoryMonitor {
  private warningThreshold: number = 400; // MB
  private criticalThreshold: number = 500; // MB

  checkMemory(): 'ok' | 'warning' | 'critical' {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      const usedMB = memory.usedJSHeapSize / 1024 / 1024;

      if (usedMB > this.criticalThreshold) {
        console.warn(`CRITICAL: Memory usage ${usedMB.toFixed(1)}MB exceeds threshold`);
        return 'critical';
      } else if (usedMB > this.warningThreshold) {
        console.warn(`WARNING: Memory usage ${usedMB.toFixed(1)}MB is high`);
        return 'warning';
      }
    }
    return 'ok';
  }
}
```

### File Structure

**New files:**
```
src/utils/
  DeviceCapabilities.ts    # Device detection and profiles
  PerformanceProfiler.ts    # Frame time and FPS tracking
  QualityManager.ts          # Adaptive quality settings
  PerformanceHUD.ts          # Debug HUD (optional)
  MemoryMonitor.ts           # Memory usage tracking
```

**Modified:**
```
src/rendering/createWebGPURenderer.ts    # Add async compilation
src/rendering/VisionarySplatRenderer.ts   # Add platform check
src/main.js                                # Profiler integration
```

### Testing Requirements

1. **Desktop Testing:**
   - Chrome/Edge on Windows with discrete GPU
   - Target 60 FPS with all features enabled
   - Memory < 500MB

2. **Mobile Testing:**
   - iOS Safari on iPhone
   - Chrome on Android
   - Target 30 FPS with reduced quality
   - Memory < 300MB

3. **Platform Limitations:**
   - Ubuntu: Visionary not supported - graceful degradation
   - macOS: Performance limited - warn user

4. **Performance Tests:**
   - 10,000 fluid particles at 60 FPS (desktop)
   - 6 jelly splats rendering
   - Water shader with all features
   - Vortex shader with animation

5. **Stress Tests:**
   - Run for 30 minutes - check for memory leaks
   - Rapid camera movement - check frame drops
   - Multiple splats on screen - check occlusion

### References

- Epic: `../_bmad-output/planning-artifacts/epics/EPIC-004-webgpu-migration.md` - Lines 326-361
- Mobile optimization: `.claude/skills/three-best-practices/rules/mobile-optimization.md`
- Visionary platform: https://ai4sports.opengvlab.com/help/index.html

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

### Completion Notes List

### File List
