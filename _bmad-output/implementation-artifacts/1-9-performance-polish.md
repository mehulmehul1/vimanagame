# Story 1.9: Performance & Polish

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **developer preparing the Archive of Voices for release**,
I want **all systems optimized, tested, and polished across target devices**,
so that **players experience the chamber at 60 FPS (desktop) or 30 FPS (mobile) without crashes or jank**.

## Acceptance Criteria

1. [ ] Async shader loading system with promise-based compilation
2. [ ] Performance targets met: 60 FPS desktop, 30 FPS mobile
3. [ ] Memory cleanup on all scene transitions (no leaks)
4. [ ] Device capability detection with quality presets
5. [ ] Performance monitor integration (FPS tracking)
6. [ ] Mobile particle count fallback (500 max particles)
7. [ ] All shaders compile in < 5 seconds on load
8. [ ] LOD system active on low-end devices

## Tasks / Subtasks

- [ ] Create async ShaderLoader system (AC: #1, #7)
  - [ ] Promise-based shader compilation
  - [ ] Load all 5 shaders in parallel: water, vortex, jelly, shell, white-flash
  - [ ] Progress callback for loading UI
  - [ ] Error handling with fallback shaders
  - [ ] Cache compiled shaders for reuse
  - [ ] Timeout protection (5 second max per shader)
  - [ ] Log compilation time for each shader
- [ ] Implement DeviceCapabilities detection (AC: #4)
  - [ ] GPU detection via WebGL/WebGPU context
  - [ ] Memory estimation (approximate GPU memory)
  - [ ] Screen size and pixel ratio detection
  - [ ] **Device tier classification:**
    - **Max:** Desktop GPU, >4GB VRAM, 1080p+
    - **Desktop:** Standard desktop, 2-4GB VRAM, 1080p
    - **Laptop:** Integrated GPU, <2GB VRAM, 720-1080p
    - **Mobile:** Mobile GPU, <1GB VRAM, 720p or lower
  - [ ] User agent hints for mobile detection
  - [ ] Fallback for unknown devices (assume laptop tier)
- [ ] Create quality presets based on device tier (AC: #4, #6)
  - [ ] **Max preset:** All effects, 2000 particles, full shaders
  - [ ] **Desktop preset:** All effects, 2000 particles, full shaders
  - [ ] **Laptop preset:** Reduced effects, 1000 particles, simplified shaders
  - [ ] **Mobile preset:** Minimal effects, 500 particles, basic shaders
  - [ ] Apply preset on scene initialization
  - [ ] Expose preset to all systems (particles, shaders, post-processing)
- [ ] Implement PerformanceMonitor (AC: #5)
  - [ ] FPS calculation using performance.now()
  - [ ] Rolling average over 60 frames
  - [ ] Frame time tracking (ms per frame)
  - [ ] **Performance thresholds:**
    - Excellent: >55 FPS (desktop), >28 FPS (mobile)
    - Good: 45-55 FPS (desktop), 24-28 FPS (mobile)
    - Fair: 30-45 FPS (desktop), 18-24 FPS (mobile)
    - Poor: <30 FPS (desktop), <18 FPS (mobile)
  - [ ] Debug overlay (toggle with key press)
  - [ ] Log performance stats every 10 seconds in dev mode
- [ ] Add particle LOD system (AC: #6, #8)
  - [ ] VortexParticles accepts max count parameter
  - [ ] **Mobile fallback:**
    - Max particles: 500
    - Skip rate: 50% (every other frame)
    - Size: Same (visual clarity maintained)
  - [ ] **Laptop fallback:**
    - Max particles: 1000
    - Skip rate: 25%
  - [ ] Set particle count on initialization based on device tier
  - [ ] Dynamic adjustment if FPS drops below threshold
- [ ] Implement memory management system (AC: #3)
  - [ ] Resource tracking for all created objects
  - [ ] **destroy() methods required for:**
    - VortexParticles
    - JellyCreature
    - JellyManager
    - ShellCollectible
    - WhiteFlashEnding
    - WaterMaterial
    - VortexMaterial
    - All audio contexts and nodes
  - [ ] Scene transition cleanup:
    - Call destroy() on all entities
    - Dispose all geometries
    - Dispose all materials
    - Remove all meshes from scene
    - Nullify all references
  - [ ] Memory profiler integration (DevTools)
- [ ] Create shader optimization passes (AC: #7)
  - [ ] Test compilation time on each device tier
  - [ ] **Mobile shader simplifications:**
    - Reduce spiral iterations in white-flash shader
    - Simplify fresnel calculations
    - Remove displacement where possible
    - Use lower precision (mediump vs highp)
  - [ ] **Desktop shader optimizations:**
    - Pre-compile variants
    - Cache uniform locations
    - Minimize branching in fragment shaders
  - [ ] Validation: All shaders compile in < 5 seconds
- [ ] Implement dynamic quality adjustment (AC: #8)
  - [ ] Monitor FPS continuously
  - [ ] If FPS drops below threshold for 3 seconds:
    - Reduce particle count by 25%
    - Simplify one shader effect
    - Log quality downgrade
  - [ ] Quality can recover if FPS improves
  - [ ] User notified of quality changes (subtle)
- [ ] Add performance testing utilities
  - [ ] Automated test mode: Run scene for 60 seconds
  - [ ] Record min/max/average FPS
  - [ ] Log memory usage before/after
  - [ ] Shader compilation timing report
  - [ ] Export results as JSON for analysis
- [ ] Create final polish checklist
  - [ ] [ ] No console errors or warnings in production build
  - [ ] [ ] Smooth frame timing (no stutters)
  - [ ] [ ] All animations are smooth (60fps feel)
  - [ ] [ ] Audio has no pops or clicks
  - [ ] [ ] Loading screen displays progress
  - [ ] [ ] All materials have destroy() methods
  - [ ] [ ] Event listeners are properly cleaned up
  - [ ] [ ] No memory leaks over 10 minute session
  - [ ] [ ] Touch controls work on mobile
  - [ ] [ ] Reduced motion respected

## Dev Notes

### Project Structure Notes

**Primary Framework:** Three.js r160+ (WebGPU/WebGL2)
**Profiling:** Performance API, DevTools Memory Profiler
**Testing:** Manual testing on target devices

**File Organization:**
```
vimana/
├── src/
│   ├── core/
│   │   ├── ShaderLoader.ts
│   │   ├── DeviceCapabilities.ts
│   │   ├── PerformanceMonitor.ts
│   │   ├── QualityPresets.ts
│   │   └── ResourceManager.ts
│   └── utils/
│       ├── PerformanceTest.ts
│       └── MemoryProfiler.ts
```

### Device Tier Detection

**DeviceCapabilities Class:**
```typescript
type DeviceTier = 'max' | 'desktop' | 'laptop' | 'mobile';

interface DeviceInfo {
    tier: DeviceTier;
    gpuMemory: number; // Estimated MB
    hasWebGPU: boolean;
    pixelRatio: number;
    screenWidth: number;
    screenHeight: number;
    isMobile: boolean;
}

class DeviceCapabilities {
    private static instance: DeviceCapabilities;
    private info: DeviceInfo;

    private constructor() {
        this.info = this.detectDevice();
    }

    static getInstance(): DeviceCapabilities {
        if (!DeviceCapabilities.instance) {
            DeviceCapabilities.instance = new DeviceCapabilities();
        }
        return DeviceCapabilities.instance;
    }

    private detectDevice(): DeviceInfo {
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const pixelRatio = window.devicePixelRatio || 1;
        const screenWidth = window.screen.width;
        const screenHeight = window.screen.height;

        // Detect WebGPU support
        let hasWebGPU = 'gpu' in navigator;

        // Estimate GPU memory based on renderer (simplified)
        let gpuMemory = 1024; // Default 1GB

        // Determine tier
        let tier: DeviceTier;

        if (isMobile) {
            tier = 'mobile';
            gpuMemory = 512;
        } else if (screenWidth < 1920 || pixelRatio < 1.5) {
            tier = 'laptop';
            gpuMemory = 1536;
        } else if (!hasWebGPU) {
            tier = 'desktop';
            gpuMemory = 2048;
        } else {
            tier = 'max';
            gpuMemory = 4096;
        }

        return {
            tier,
            gpuMemory,
            hasWebGPU,
            pixelRatio,
            screenWidth,
            screenHeight,
            isMobile
        };
    }

    getTier(): DeviceTier {
        return this.info.tier;
    }

    getInfo(): DeviceInfo {
        return this.info;
    }
}
```

### Quality Presets

**QualityPresets Configuration:**
```typescript
interface QualityPreset {
    name: string;
    particles: {
        maxCount: number;
        lodSkipRate: number;
    };
    shaders: {
        precision: 'highp' | 'mediump' | 'lowp';
        enableDisplacement: boolean;
        enableFresnel: boolean;
        enableIridescence: boolean;
        spiralIterations: number;
    };
    effects: {
        enableGlow: boolean;
        enableBloom: boolean;
        enableVignette: boolean;
    };
}

const QUALITY_PRESETS: Record<DeviceTier, QualityPreset> = {
    max: {
        name: 'Maximum',
        particles: { maxCount: 2000, lodSkipRate: 0 },
        shaders: {
            precision: 'highp',
            enableDisplacement: true,
            enableFresnel: true,
            enableIridescence: true,
            spiralIterations: 3
        },
        effects: {
            enableGlow: true,
            enableBloom: true,
            enableVignette: true
        }
    },
    desktop: {
        name: 'Desktop',
        particles: { maxCount: 2000, lodSkipRate: 0 },
        shaders: {
            precision: 'highp',
            enableDisplacement: true,
            enableFresnel: true,
            enableIridescence: true,
            spiralIterations: 3
        },
        effects: {
            enableGlow: true,
            enableBloom: false,
            enableVignette: true
        }
    },
    laptop: {
        name: 'Laptop',
        particles: { maxCount: 1000, lodSkipRate: 0.25 },
        shaders: {
            precision: 'mediump',
            enableDisplacement: true,
            enableFresnel: true,
            enableIridescence: false,
            spiralIterations: 2
        },
        effects: {
            enableGlow: true,
            enableBloom: false,
            enableVignette: true
        }
    },
    mobile: {
        name: 'Mobile',
        particles: { maxCount: 500, lodSkipRate: 0.5 },
        shaders: {
            precision: 'mediump',
            enableDisplacement: false,
            enableFresnel: true,
            enableIridescence: false,
            spiralIterations: 1
        },
        effects: {
            enableGlow: false,
            enableBloom: false,
            enableVignette: true
        }
    }
};

class QualityPresets {
    private static currentPreset: QualityPreset;

    static initialize() {
        const tier = DeviceCapabilities.getInstance().getTier();
        this.currentPreset = QUALITY_PRESETS[tier];
    }

    static getCurrent(): QualityPreset {
        return this.currentPreset;
    }

    static getPresetForTier(tier: DeviceTier): QualityPreset {
        return QUALITY_PRESETS[tier];
    }
}
```

### Shader Loader

**Async Shader Loading:**
```typescript
interface ShaderSource {
    vertex: string;
    fragment: string;
}

interface CompiledShader {
    vertex: string;
    fragment: string;
    compilationTime: number;
}

class ShaderLoader {
    private static shaders: Map<string, string> = new Map();

    static async loadShader(name: string): Promise<ShaderSource> {
        // Check cache first
        if (this.shaders.has(name + '-vertex')) {
            return {
                vertex: this.shaders.get(name + '-vertex')!,
                fragment: this.shaders.get(name + '-fragment')!
            };
        }

        // Load vertex shader
        const vertex = await fetch(`/src/shaders/${name}-vertex.glsl`)
            .then(r => r.text());

        // Load fragment shader
        const fragment = await fetch(`/src/shaders/${name}-fragment.glsl`)
            .then(r => r.text());

        // Cache for reuse
        this.shaders.set(name + '-vertex', vertex);
        this.shaders.set(name + '-fragment', fragment);

        return { vertex, fragment };
    }

    static async loadAllShaders(
        shaderNames: string[],
        onProgress?: (loaded: number, total: number) => void
    ): Promise<Map<string, CompiledShader>> {
        const results = new Map<string, CompiledShader>();
        const total = shaderNames.length;

        // Load all in parallel
        const promises = shaderNames.map(async (name, index) => {
            const startTime = performance.now();
            const source = await this.loadShader(name);

            // Compile (dry run to test compilation time)
            const compileTime = performance.now() - startTime;

            const compiled: CompiledShader = {
                vertex: source.vertex,
                fragment: source.fragment,
                compilationTime: compileTime
            };

            results.set(name, compiled);

            if (onProgress) {
                onProgress(index + 1, total);
            }

            console.log(`[ShaderLoader] ${name} compiled in ${compileTime.toFixed(2)}ms`);

            return compiled;
        });

        await Promise.all(promises);

        // Validate all under 5 seconds
        const totalTime = Array.from(results.values())
            .reduce((sum, s) => sum + s.compilationTime, 0);

        if (totalTime > 5000) {
            console.warn(`[ShaderLoader] Total compilation time ${totalTime.toFixed(2)}ms exceeds 5s target`);
        }

        return results;
    }
}
```

### Performance Monitor

**FPS Tracking:**
```typescript
interface PerformanceStats {
    fps: number;
    frameTime: number;
    minFps: number;
    maxFps: number;
    averageFps: number;
    tier: 'excellent' | 'good' | 'fair' | 'poor';
}

class PerformanceMonitor {
    private frameTimes: number[] = [];
    private maxSamples = 60;
    private minFps = Infinity;
    private maxFps = 0;
    private lastUpdate = performance.now();
    private isRecording = false;

    private isMobile(): boolean {
        return DeviceCapabilities.getInstance().getInfo().isMobile;
    }

    update(): void {
        const now = performance.now();
        const delta = now - this.lastUpdate;
        this.lastUpdate = now;

        const fps = 1000 / delta;

        // Track min/max
        this.minFps = Math.min(this.minFps, fps);
        this.maxFps = Math.max(this.maxFps, fps);

        // Rolling average
        this.frameTimes.push(fps);
        if (this.frameTimes.length > this.maxSamples) {
            this.frameTimes.shift();
        }
    }

    getStats(): PerformanceStats {
        const average = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        const frameTime = 1000 / average;

        // Determine tier
        const isMobile = this.isMobile();
        let tier: 'excellent' | 'good' | 'fair' | 'poor';

        if (isMobile) {
            if (average > 28) tier = 'excellent';
            else if (average >= 24) tier = 'good';
            else if (average >= 18) tier = 'fair';
            else tier = 'poor';
        } else {
            if (average > 55) tier = 'excellent';
            else if (average >= 45) tier = 'good';
            else if (average >= 30) tier = 'fair';
            else tier = 'poor';
        }

        return {
            fps: Math.round(average),
            frameTime: Math.round(frameTime * 100) / 100,
            minFps: Math.round(this.minFps),
            maxFps: Math.round(this.maxFps),
            averageFps: Math.round(average),
            tier
        };
    }

    startRecording(): void {
        this.isRecording = true;
        this.minFps = Infinity;
        this.maxFps = 0;
        this.frameTimes = [];
    }

    stopRecording(): PerformanceStats {
        this.isRecording = false;
        return this.getStats();
    }

    logStats(): void {
        const stats = this.getStats();
        console.log(`[PerformanceMonitor] FPS: ${stats.fps} | Frame Time: ${stats.frameTime}ms | Tier: ${stats.tier}`);
    }
}
```

### Memory Management

**Cleanup Checklist:**
```typescript
class ResourceManager {
    private static resources: Set<{ destroy(): void }> = new Set();

    static register(resource: { destroy(): void }): void {
        this.resources.add(resource);
    }

    static unregister(resource: { destroy(): void }): void {
        this.resources.delete(resource);
    }

    static cleanupAll(): void {
        console.log(`[ResourceManager] Cleaning up ${this.resources.size} resources...`);

        for (const resource of this.resources) {
            try {
                resource.destroy();
            } catch (e) {
                console.error('[ResourceManager] Error destroying resource:', e);
            }
        }

        this.resources.clear();

        // Force garbage collection if available
        if ((window as any).gc) {
            (window as any).gc();
        }
    }

    static getMemoryEstimate(): number {
        // Rough estimate in MB
        return this.resources.size * 10; // Assume ~10MB per resource
    }
}
```

### Usage Integration

**In HarpRoom:**
```typescript
class HarpRoom {
    private performanceMonitor: PerformanceMonitor;
    private qualityPreset: QualityPreset;

    async initialize() {
        // Detect device
        const device = DeviceCapabilities.getInstance();
        console.log(`[HarpRoom] Device tier: ${device.getTier()}`);

        // Load quality preset
        QualityPresets.initialize();
        this.qualityPreset = QualityPresets.getCurrent();

        // Load shaders
        const shaders = await ShaderLoader.loadAllShaders(
            ['water', 'vortex', 'jelly', 'shell', 'white-flash'],
            (loaded, total) => {
                console.log(`[HarpRoom] Loading shaders: ${loaded}/${total}`);
            }
        );

        // Initialize performance monitor
        this.performanceMonitor = new PerformanceMonitor();
        this.performanceMonitor.startRecording();

        // Create entities with quality settings
        this.vortexParticles = new VortexParticles(
            this.qualityPreset.particles.maxCount,
            this.qualityPreset.particles.lodSkipRate
        );
    }

    update(deltaTime: number) {
        this.performanceMonitor.update();

        // Log stats every 10 seconds
        if (Math.random() < 0.016) { // Approx once per 10 seconds at 60fps
            this.performanceMonitor.logStats();
        }
    }

    destroy() {
        ResourceManager.cleanupAll();
    }
}
```

### Performance Testing

**Automated Test:**
```typescript
class PerformanceTest {
    static async run(durationMs: number = 60000): Promise<TestResults> {
        const monitor = new PerformanceMonitor();
        monitor.startRecording();

        const startTime = performance.now();
        const frameTimes: number[] = [];

        while (performance.now() - startTime < durationMs) {
            await new Promise(resolve => requestAnimationFrame(resolve));

            const frameStart = performance.now();
            // ... scene update ...
            const frameTime = performance.now() - frameStart;

            frameTimes.push(frameTime);
            monitor.update();
        }

        const results = monitor.stopRecording();

        return {
            ...results,
            frameTimes,
            totalTime: durationMs,
            framesRendered: frameTimes.length
        };
    }
}

interface TestResults extends PerformanceStats {
    frameTimes: number[];
    totalTime: number;
    framesRendered: number;
}
```

### Dependencies

**Previous Story:** All previous stories (optimization pass after all features)

**Final Story:** This is the final story of the epic

**External Dependencies:**
- Performance API
- DevTools for profiling
- Manual device testing

### Success Criteria

**Before Release:**
- [ ] 60 FPS on desktop (Chrome, Firefox, Safari)
- [ ] 30 FPS on mobile (iOS Safari, Android Chrome)
- [ ] No memory leaks over 10 minute session
- [ ] All shaders compile in < 5 seconds
- [ ] No console errors in production build
- [ ] Touch controls work on all mobile devices

### References

- [Source: music-room-proto-epic.md#Story 1.9]
- [Source: gdd.md#Performance requirements]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

### File List

- `src/core/ShaderLoader.ts` (create)
- `src/core/DeviceCapabilities.ts` (create)
- `src/core/PerformanceMonitor.ts` (create)
- `src/core/QualityPresets.ts` (create)
- `src/core/ResourceManager.ts` (create)
- `src/utils/PerformanceTest.ts` (create)
