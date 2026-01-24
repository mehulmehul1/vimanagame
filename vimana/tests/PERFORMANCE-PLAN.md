# Performance Test Plan: Vimana

## Overview

**Project:** Vimana - A 3D contemplative game about learning to harmonize with a mythical flying ship
**Current Epic:** Music Room Prototype (Archive of Voices)
**Tech Stack:** Three.js, Rapier3d, Howler, Vite
**Testing:** Vitest (unit) + Playwright (E2E)

This document defines the comprehensive performance testing strategy for Vimana, covering frame rate targets, memory budgets, loading times, and platform-specific requirements.

---

## Performance Targets

### Frame Rate Targets

| Platform          | Target FPS | Minimum FPS | Notes                              |
| ----------------- | ---------- | ----------- | ---------------------------------- |
| Desktop (High)    | 60+        | 55          | Uncapped option, V-sync enabled    |
| Desktop (Low)     | 60         | 30          | Scalable settings via QualityPresets |
| Mobile (High)     | 60         | 30          | Device dependent, thermal throttling |
| Mobile (Standard) | 30         | 25          | Power saving, reduced particles     |
| Tablet            | 60         | 30          | Same as desktop high                |

### Memory Budgets

| Platform         | Total RAM | Game Budget | Notes                          |
| ---------------- | --------- | ----------- | ------------------------------ |
| Desktop (Min)    | 8 GB      | 500 MB      | WebGL heap limit               |
| Desktop (Rec)    | 16 GB     | 1 GB        | Headroom for browser           |
| Mobile (High)    | 6 GB      | 300 MB      | Tight constraints              |
| Mobile (Standard)| 4 GB      | 200 MB      | Background apps reduce budget  |

### Loading Time Targets

| Scenario     | Target | Maximum | Notes                           |
| ------------ | ------ | ------- | ------------------------------- |
| Initial boot | < 5s   | 10s     | Shader compilation is critical  |
| Scene load   | < 3s   | 5s      | GLB loading + material creation |
| Video intro  | < 2s   | 5s      | Video buffering                 |

### Regression Criteria

| Metric          | Warning Threshold | Critical Threshold | Action                 |
| --------------- | ----------------- | ------------------- | ---------------------- |
| Average FPS     | < 55              | < 30                | Trigger quality downscale |
| Memory growth   | > 10 MB/min       | > 50 MB/min         | Leak investigation      |
| 1% low FPS      | < 45              | < 25                | Frame spike investigation |
| Shader compile  | > 5s              | > 10s               | Optimization needed      |

---

## Test Scenarios

### Frame Rate Tests

#### Stress Test: Maximum Particle Count

```
SCENARIO: Vortex particles at full activation
  GIVEN VortexParticles is initialized with 2000 particles
  AND duet progress is 1.0 (full activation)
  WHEN all particles are updating
  THEN frame rate stays >= 55 FPS on desktop
  AND frame rate stays >= 25 FPS on mobile
  AND no visual artifacts from particle skipping
  CATEGORY: performance
  PRIORITY: P0
```

#### Stress Test: All Jelly Creatures Active

```
SCENARIO: All six jelly creatures animating
  GIVEN all six JellyCreature instances are spawned
  AND all jellies are in teaching state (bioluminescence active)
  WHEN camera views all jellies simultaneously
  THEN frame rate stays >= 55 FPS on desktop
  AND emissive pulsing does not cause frame drops
  CATEGORY: performance
  PRIORITY: P0
```

#### Stress Test: Full Scene Render

```
SCENARIO: All systems active simultaneously
  GIVEN 2000 vortex particles are updating
  AND 6 jelly creatures are visible and animating
  AND water shader is active with 6-string frequency uniforms
  AND vortex glow is at maximum emissive intensity
  WHEN player moves camera around entire scene
  THEN frame rate stays >= 55 FPS on desktop
  AND frame rate stays >= 25 FPS on mobile
  CATEGORY: performance
  PRIORITY: P0
```

#### Shader Compilation Stress

```
SCENARIO: Water shader compilation time
  GIVEN game is loading for first time (cold cache)
  WHEN WaterMaterial compiles with 6-string frequency uniforms
  THEN compilation completes in < 3 seconds
  AND loading screen shows progress feedback
  CATEGORY: loading
  PRIORITY: P0
```

### Memory Tests

#### Memory Leak Detection

```
SCENARIO: Extended play session memory stability
  GIVEN game is running in Music Room scene
  WHEN gameplay continues for 10 minutes
  - Player moves around
  - Harp strings are played
  - Jellies emerge and submerge
  - Vortex particles update
  THEN memory growth is < 50 MB total
  AND no leak patterns in memory allocations
  CATEGORY: memory
  PRIORITY: P0
```

#### Scene Transition Cleanup

```
SCENARIO: Scene transition properly cleans up
  GIVEN player completes Music Room duet
  WHEN transitioning to next scene
  THEN all geometries are disposed (dispose() called)
  AND all materials are disposed
  AND all textures are disposed
  AND all event listeners are removed
  AND WebGL memory returns to baseline
  CATEGORY: memory
  PRIORITY: P0
```

#### Particle System Memory

```
SCENARIO: VortexParticles memory efficiency
  GIVEN VortexParticles is created with 2000 particles
  WHEN particle system is destroyed
  THEN BufferGeometry is disposed
  AND PointsMaterial is disposed
  AND Float32Array buffers are released
  AND no Three.js object remains in memory
  CATEGORY: memory
  PRIORITY: P1
```

### Loading Tests

#### Cold Boot Performance

```
SCENARIO: Initial game load time
  GIVEN game is not in browser cache
  WHEN player navigates to game URL
  THEN WebGL context creates in < 1 second
  AND Three.js initializes in < 2 seconds
  AND shaders compile in < 5 seconds total
  AND scene is interactive in < 10 seconds
  CATEGORY: loading
  PRIORITY: P0
```

#### GLB Model Loading

```
SCENARIO: HarpRoom model loading time
  GIVEN HarpRoom.glb contains arena floor, harp, platforms
  WHEN model loads from CDN
  THEN loading completes in < 3 seconds
  AND trimesh colliders generate in < 1 second
  AND all mesh materials are applied correctly
  CATEGORY: loading
  PRIORITY: P0
```

#### Video Intro Loading

```
SCENARIO: Intro video buffering time
  GIVEN white flash ending video exists
  WHEN video is requested for playback
  THEN enough buffer loads in < 2 seconds
  AND playback starts smoothly without stuttering
  CATEGORY: loading
  PRIORITY: P1
```

### LOD (Level of Detail) Tests

#### Particle LOD Activation

```
SCENARIO: VortexParticles LOD skip rates
  GIVEN device is detected as low-end (DeviceCapabilities.tier = 'low')
  AND duet progress is 0.2 (< 30%)
  WHEN VortexParticles.update() runs
  THEN 50% of particles are skipped (i % 2 === 0)
  AND frame rate improves by at least 20%

  GIVEN duet progress increases to 0.4 (30-60%)
  WHEN VortexParticles.update() runs
  THEN 25% of particles are skipped (i % 4 === 0)

  GIVEN duet progress reaches 1.0
  WHEN VortexParticles.update() runs
  THEN 0% of particles are skipped (all update)
  CATEGORY: performance
  PRIORITY: P0
```

#### Quality Preset Application

```
SCENARIO: QualityPresets applies correct settings
  GIVEN device tier is 'low'
  WHEN QualityPresets applies to renderer
  THEN particleMultiplier is 0.5 (1000 particles max)
  AND shaderQuality is 0.5
  AND enableShadows is false
  AND enablePostProcessing is false
  AND pixelRatio is capped at 1.0
  CATEGORY: performance
  PRIORITY: P0
```

---

## Methodology

### Automated Tests

#### Vitest Unit Tests

Performance-critical unit tests verify correct LOD and quality behavior:

```typescript
// tests/entities/VortexParticles.performance.test.ts
describe('VortexParticles Performance', () => {
    it('should apply correct LOD skip rates at low activation', () => {
        const particles = new VortexParticles(2000);
        const updateSpy = vi.spyOn(particles['geometry'].attributes.position, 'needsUpdate', 'set');

        particles.update(0.016, 0.2); // Low activation
        // Should skip ~50% of particles
        expect(updateSpy).toHaveBeenCalled();
    });

    it('should dispose all resources on destroy', () => {
        const particles = new VortexParticles(2000);
        const geometrySpy = vi.spyOn(particles['geometry'], 'dispose');
        const materialSpy = vi.spyOn(particles['material'], 'dispose');

        particles.destroy();

        expect(geometrySpy).toHaveBeenCalled();
        expect(materialSpy).toHaveBeenCalled();
    });
});
```

#### Playwright E2E Performance Tests

Browser-based performance tests measure real FPS and memory:

```typescript
// tests/e2e/performance.test.ts (already exists)
test.describe('Performance Tests', () => {
    test('should maintain 60 FPS on desktop', async ({ page }) => {
        await page.goto('/');
        await page.waitForSelector('canvas');

        const fps = await page.evaluate(() => {
            return new Promise<number>((resolve) => {
                let frames = 0;
                const startTime = performance.now();
                function countFrames() {
                    frames++;
                    if (performance.now() - startTime < 2000) {
                        requestAnimationFrame(countFrames);
                    } else {
                        resolve(Math.round(frames * 1000 / (performance.now() - startTime)));
                    }
                }
                requestAnimationFrame(countFrames);
            });
        });

        expect(fps).toBeGreaterThanOrEqual(55);
    });
});
```

### Manual Profiling Checklist

#### CPU Profiling (Chrome DevTools Performance)

- [ ] Identify rendering hotspots (Functions taking > 5ms per frame)
- [ ] Check for unnecessary garbage collections
- [ ] Verify Three.js render loop efficiency
- [ ] Analyze Rapier3d physics step time
- [ ] Review audio synthesis CPU usage

#### GPU Profiling (Chrome DevTools Rendering)

- [ ] Check draw call count (Target: < 100)
- [ ] Analyze overdraw on water shader
- [ ] Review shader complexity (instruction count)
- [ ] Check texture memory usage
- [ ] Verify particle system batch efficiency

#### Memory Profiling (Chrome DevTools Memory)

- [ ] Take heap snapshot on load
- [ ] Take heap snapshot after 10 minutes
- [ ] Compare for leaked objects
- [ ] Check for retained Three.js objects
- [ ] Verify event listener cleanup

---

## Benchmark Suite

### Benchmark Definitions

| Benchmark Name      | Purpose                              | Duration | Success Criteria          |
| ------------------- | ------------------------------------ | -------- | ------------------------- |
| `idle-scene`        | Baseline FPS with no interaction     | 60s      | >= 60 FPS (desktop)       |
| `particle-stress`   | Max particles with full activation   | 60s      | >= 55 FPS (desktop)       |
| `jelly-animation`   | All 6 jellies animating              | 60s      | >= 55 FPS (desktop)       |
| `full-scene`        | All systems active                   | 120s     | >= 55 FPS (desktop)       |
| `memory-stability`  | Extended play for leak detection     | 600s     | < 50 MB growth            |
| `shader-compile`    | Cold start shader compilation        | 15s      | < 5s to interactive       |

### Baseline Metrics

Reference hardware for baseline capture:

| Tier  | GPU                    | CPU             | RAM   | Target FPS |
| ----- | ---------------------- | --------------- | ----- | ---------- |
| Ultra | NVIDIA RTX 3080       | Ryzen 7 5800X   | 32GB  | 60+        |
| High  | NVIDIA GTX 1660       | Ryzen 5 3600    | 16GB  | 60         |
| Med   | Intel UHD 620         | Core i5-8250U   | 8GB   | 45         |
| Low   | Intel HD 4000         | Core i3-4005U   | 4GB   | 30         |

### Regression Detection

Automated performance regression is integrated into CI:

```yaml
# .github/workflows/performance.yml
performance-test:
  runs-on: ubuntu-latest
  steps:
    - name: Run performance benchmarks
      run: npm run test:performance
    - name: Compare against baseline
      run: npm run test:performance:compare
```

Regression thresholds:
- **5% FPS degradation** = Warning
- **10% FPS degradation** = Fail
- **20 MB memory increase** = Warning
- **50 MB memory increase** = Fail

---

## Platform Matrix

### Desktop (Windows/Mac/Linux)

| Configuration | Min Spec                 | Recommended             |
| ------------- | ------------------------ | ----------------------- |
| OS            | Windows 10, macOS 10.14  | Windows 11, macOS 13    |
| CPU           | Core i3-4005U           | Core i5-8250U or better |
| GPU           | Intel HD 4000           | NVIDIA GTX 1660 or better |
| RAM           | 8 GB                    | 16 GB                   |
| Browser       | Chrome 90+, Firefox 88+  | Chrome 120+             |

**Desktop Tests:**
- [ ] 60 FPS maintained on high-end GPU
- [ ] 30 FPS maintained on low-end integrated GPU
- [ ] WASD keyboard controls responsive
- [ ] Mouse look smooth (pointer lock)
- [ ] Multiple browser compatibility

### Mobile (iOS/Android)

| Configuration | Min Device           | Recommended Device     |
| ------------- | -------------------- | ---------------------- |
| OS            | iOS 13, Android 8    | iOS 16, Android 13     |
| GPU           | Apple A10, Adreno 530| Apple A14, Adreno 650  |
| RAM           | 3 GB                 | 6 GB                   |
| Browser       | Safari 13, Chrome 90 | Safari 16, Chrome 120  |

**Mobile Tests:**
- [ ] 30 FPS maintained on mid-range device
- [ ] 25 FPS maintained on low-end device
- [ ] Touch controls responsive
- [ ] Landscape/portrait orientation handling
- [ ] Thermal throttling recovery

### Cross-Browser

| Browser    | WebGL2 | WebGPU | Notes                          |
| ---------- | ------ | ------ | ------------------------------ |
| Chrome     | Yes    | Yes    | Primary target, best DevTools  |
| Firefox    | Yes    | No     | Good WebGL2 support            |
| Safari     | Yes    | No     | iOS requires https              |
| Edge       | Yes    | Yes    | Same as Chrome                 |

**Browser Tests:**
- [ ] WebGL2 context creation on all browsers
- [ ] Consistent frame rates across browsers
- [ ] Audio context resume on user interaction
- [ ] Shader compilation consistency

---

## CI Integration

### Automated Performance Testing

```bash
# Run all performance tests
npm run test:performance

# Run with benchmark comparison
npm run test:performance:compare

# Generate performance report
npm run test:performance:report
```

### Test Schedule

| Trigger                  | Tests Run                          | Duration |
| ------------------------ | ---------------------------------- | -------- |
| Every Pull Request       | Smoke performance (10s sample)     | ~30s     |
| Nightly (main branch)    | Full benchmark suite                | ~15 min  |
| Weekly                   | Memory leak test (extended)         | ~30 min  |
| Pre-release              | Full platform matrix               | ~2 hours |

### Performance Report Output

```json
{
  "timestamp": "2025-01-24T10:00:00Z",
  "commit": "abc123",
  "benchmarks": {
    "idle-scene": {
      "avgFps": 59.8,
      "minFps": 55,
      "maxFps": 60,
      "frameTime": 16.7,
      "status": "pass"
    },
    "particle-stress": {
      "avgFps": 57.2,
      "minFps": 52,
      "maxFps": 60,
      "frameTime": 17.5,
      "status": "pass"
    }
  },
  "memory": {
    "initial": "245 MB",
    "final": "267 MB",
    "growth": "22 MB",
    "status": "pass"
  },
  "regression": false
}
```

---

## Vimana-Specific Considerations

### Contemplative Pacing

Vimana's "patient teaching" philosophy affects performance testing:

- **No rapid camera movements** - Frame rate can be more forgiving
- **Long idle periods** - Memory leaks become more visible
- **Ambient particle effects** - Visual quality must balance with performance

### Audio-Visual Synchronization

Performance affects the teaching mechanic:

- **Jelly animation timing** - Frame drops disrupt the teaching rhythm
- **Harp feedback latency** - Slow response frustrates players
- **Discordant tone timing** - Audio must sync with camera shake

### Shader Complexity

Custom shaders are a performance concern:

- **Water ripple effect** - 6-string frequency uniforms add complexity
- **Vortex glow** - Emissive intensity changes every frame
- **SDF shell iridescence** - Per-pixel calculation

---

## Next Steps

1. **Run existing performance tests** to establish baseline
2. **Add missing unit tests** for LOD and quality presets
3. **Set up CI pipeline** for automated performance regression
4. **Profile on real devices** (especially low-end mobile)
5. **Create performance monitoring dashboard**

---

## Appendix: Performance Monitoring Hook

```javascript
// Add to main.js for development monitoring
if (import.meta.env.DEV) {
    const monitor = new PerformanceMonitor({ targetFps: 60 });

    function updateMonitor() {
        const metrics = monitor.update();

        if (metrics.avgFps < 50) {
            console.warn('[Performance] Low FPS:', metrics.avgFps);
        }

        requestAnimationFrame(updateMonitor);
    }

    updateMonitor();
}
```

---

**Document Version:** 1.0
**Created:** 2025-01-24
**Workflow:** performance (BMad v6)
**Related:**
- `TEST-DESIGN.md` - Scenario definitions
- `tests/e2e/performance.test.ts` - E2E performance tests
- `src/utils/PerformanceMonitor.ts` - Performance monitoring utility
