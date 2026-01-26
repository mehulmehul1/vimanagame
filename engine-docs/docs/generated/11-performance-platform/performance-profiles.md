# Performance Profiles - A First Principles Guide

**Shadow Engine Performance Documentation**

---

## What You Need to Know First

Before understanding performance profiles, you should know:
- **Basic rendering concepts** - What FPS (frames per second) means
- **Hardware limitations** - Mobile devices are less powerful than desktops
- **Trade-offs** - Quality vs. performance is always a balance
- **Your target audience** - What devices will your players use?

---

## Overview

**Performance profiles** are pre-configured quality settings that automatically adjust visual fidelity to maintain smooth gameplay across different hardware. Think of them as "presets" that optimize the game for different device capabilities.

### The Problem This Solves

```
High-end gaming PC:
    ↓
Can run everything at maximum quality
    ↓
Result: Beautiful, smooth 60+ FPS

Mid-range laptop:
    ↓
Struggles with maximum settings
    ↓
Without optimization: 15-20 FPS (unplayable)
With profile: 30-45 FPS (enjoyable)

Mobile phone:
    ↓
Cannot handle high-end graphics
    ↓
Without optimization: 5-10 FPS (broken)
With profile: 30 FPS (playable)
```

---

## 🎮 Game Design Perspective

### Creative Intent

**Why do we need performance profiles? Can't everyone just run at max quality?**

| Reality | Consequence |
|---------|-------------|
| **Players have different devices** | What runs smooth on a gaming PC crawls on a phone |
| **Smooth gameplay > pretty graphics** | 60 FPS at low quality beats 15 FPS at high quality |
| **First impressions matter** | If the game stutters, players quit immediately |
| **Mobile is growing** | More players game on phones than high-end PCs |

The goal is **accessibility** - let as many players as possible enjoy your game, regardless of their hardware.

### Design Philosophy

**Adaptive Quality = Inclusive Design:**

```
Fixed approach:
    ↓
One set of settings for everyone
    ↓
Result: Some players have great experience,
        others have terrible experience

Adaptive approach (Performance Profiles):
    ↓
Detect hardware capabilities
    ↓
Select appropriate quality preset
    ↓
Result: Everyone gets smooth, enjoyable gameplay
        (even if visual quality varies)
```

### Player Experience

| Profile | Target Device | Expected FPS | Visual Quality |
|---------|---------------|--------------|----------------|
| **Mobile** | Phones, tablets | 30 FPS | Low (playable) |
| **Laptop** | Integrated graphics | 45 FPS | Medium |
| **Desktop** | Dedicated GPU | 60+ FPS | High |
| **Max** | High-end gaming rigs | 60+ FPS | Ultra (best possible) |

---

## Core Concepts (Beginner Friendly)

### What is FPS?

**FPS = Frames Per Second** - How many images the game renders each second.

```
30 FPS:  Console standard, acceptable for most games
60 FPS:  PC gaming standard, feels very smooth
120+ FPS: Competitive gaming, hyper-smooth

Below 30 FPS: Noticeable stuttering, feels bad
Below 20 FPS: Difficult to play, motion sickness possible
Below 15 FPS: Essentially unplayable
```

### What Affects Performance?

| Factor | Impact | Example |
|--------|--------|---------|
| **Polygon count** | Higher = more work for GPU | Complex 3D models vs simple shapes |
| **Draw calls** | More = slower rendering | 1000 small objects vs 1 combined mesh |
| **Texture resolution** | Larger = more memory | 4K textures vs 512x512 textures |
| **Post-processing** | Each effect adds cost | Bloom, blur, color grading |
| **Shadow quality** | Higher = much slower | Soft shadows vs hard shadows |
| **Particle count** | More particles = slower | 100 particles vs 10,000 particles |
| **Splat count** | (For Gaussian Splatting) More splats = slower | 1M splats vs 10M splats |

### The "Good Enough" Principle

You don't need maximum quality - you need **consistent quality** that the target hardware can handle smoothly.

```
Perfect: 60 FPS, all settings max
Better than possible but real: 60 FPS, balanced settings
Acceptable: 30 FPS, lower settings
Bad: Unstable FPS (40-60 range), any settings
Worst: 20 FPS, any settings
```

---

## How It Works

### Performance Profile System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Game Startup                         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│           Platform Detection System                      │
│  - Detect device type (mobile/desktop)                   │
│  - Check GPU capabilities                               │
│  - Measure available memory                             │
│  - Run quick benchmark if needed                        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│           Profile Selection Logic                        │
│  IF mobile → use Mobile profile                         │
│  ELSE IF weak GPU → use Laptop profile                  │
│  ELSE IF strong GPU → use Desktop profile               │
│  ELSE → use Max profile                                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│           Apply Profile Settings                         │
│  - Set rendering quality flags                          │
│  - Configure LOD distances                              │
│  - Adjust texture resolutions                           │
│  - Enable/disable effects                               │
│  - Set target FPS                                       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│           Performance Monitoring                         │
│  - Track actual FPS during gameplay                     │
│  - Auto-adjust if needed                                │
│  - Allow manual override in options                     │
└─────────────────────────────────────────────────────────┘
```

### Profile Data Structure

```javascript
/**
 * Performance profile configuration
 */
export const PERFORMANCE_PROFILES = {
  mobile: {
    name: 'Mobile',
    description: 'Optimized for phones and tablets',
    targetFPS: 30,

    // Gaussian Splatting settings
    splat: {
      maxPoints: 2000000,          // 2M splats max
      renderScale: 0.5,             // Render at half resolution
      lodEnabled: true,
      lodDistances: [10, 25, 50],   // Aggressive LOD
    },

    // Post-processing
    postProcessing: {
      enabled: true,
      bloom: false,                  // Too expensive
      chromaticAberration: false,    // Too expensive
      filmGrain: false,              // Too expensive
      vignette: true,                // Cheap
      colorGrading: true,            // Cheap
    },

    // Shadows
    shadows: {
      enabled: false,                // Too expensive for mobile
      type: 'none',
    },

    // Particles
    particles: {
      maxCount: 500,
      budget: 0.3,                   // 30% of frame time
    },

    // Textures
    textures: {
      maxResolution: 512,            // Downscale textures
      compression: true,
      mipmap: false,                 // Save memory
    },

    // Audio
    audio: {
      spatial: true,
      maxSources: 8,
      reverb: false,                 // Too expensive
    },

    // Other settings
    frustumCulling: true,
    occlusionCulling: false,         // Too expensive
    physicsBudget: 0.1,              // 10% of frame time
  },

  laptop: {
    name: 'Laptop',
    description: 'Balanced for integrated graphics',
    targetFPS: 45,

    splat: {
      maxPoints: 5000000,            // 5M splats
      renderScale: 0.75,
      lodEnabled: true,
      lodDistances: [15, 40, 80],
    },

    postProcessing: {
      enabled: true,
      bloom: true,
      bloomThreshold: 0.8,
      bloomIntensity: 0.3,
      chromaticAberration: false,
      filmGrain: false,
      vignette: true,
      colorGrading: true,
    },

    shadows: {
      enabled: true,
      type: 'basic',
      mapSize: 512,
    },

    particles: {
      maxCount: 2000,
      budget: 0.4,
    },

    textures: {
      maxResolution: 1024,
      compression: true,
      mipmap: true,
    },

    audio: {
      spatial: true,
      maxSources: 16,
      reverb: false,
    },

    frustumCulling: true,
    occlusionCulling: false,
    physicsBudget: 0.15,
  },

  desktop: {
    name: 'Desktop',
    description: 'High quality for dedicated GPUs',
    targetFPS: 60,

    splat: {
      maxPoints: 10000000,           // 10M splats
      renderScale: 1.0,
      lodEnabled: true,
      lodDistances: [25, 60, 100],
    },

    postProcessing: {
      enabled: true,
      bloom: true,
      bloomThreshold: 0.7,
      bloomIntensity: 0.5,
      chromaticAberration: true,
      filmGrain: true,
      vignette: true,
      colorGrading: true,
    },

    shadows: {
      enabled: true,
      type: 'pcf',
      mapSize: 1024,
    },

    particles: {
      maxCount: 5000,
      budget: 0.5,
    },

    textures: {
      maxResolution: 2048,
      compression: false,
      mipmap: true,
    },

    audio: {
      spatial: true,
      maxSources: 32,
      reverb: true,
    },

    frustumCulling: true,
    occlusionCulling: true,
    physicsBudget: 0.2,
  },

  max: {
    name: 'Ultra',
    description: 'Maximum quality (high-end GPUs only)',
    targetFPS: 60,

    splat: {
      maxPoints: 20000000,           // 20M splats
      renderScale: 1.5,              // Supersample!
      lodEnabled: true,
      lodDistances: [40, 100, 200],
    },

    postProcessing: {
      enabled: true,
      bloom: true,
      bloomThreshold: 0.6,
      bloomIntensity: 0.8,
      chromaticAberration: true,
      filmGrain: true,
      vignette: true,
      colorGrading: true,
      antiAliasing: 'taa',           // Temporal anti-aliasing
    },

    shadows: {
      enabled: true,
      type: 'pcf_soft',
      mapSize: 2048,
      cascaded: true,                // Cascaded shadow maps
    },

    particles: {
      maxCount: 15000,
      budget: 0.6,
    },

    textures: {
      maxResolution: 4096,
      compression: false,
      mipmap: true,
      anisotropic: 16,
    },

    audio: {
      spatial: true,
      maxSources: 64,
      reverb: true,
      hrtf: true,                    // Head-related transfer function
    },

    frustumCulling: true,
    occlusionCulling: true,
    physicsBudget: 0.25,
  }
};
```

---

## Architecture

### Performance Manager

```javascript
/**
 * Manages performance profiles and optimization
 */
class PerformanceManager {
  constructor(gameManager) {
    this.gameManager = gameManager;
    this.currentProfile = null;
    this.profiles = PERFORMANCE_PROFILES;
    this.metrics = {
      fps: 0,
      frameTime: 0,
      drawCalls: 0,
      triangles: 0,
      splats: 0
    };
  }

  /**
   * Initialize performance system
   */
  async init() {
    // Detect platform
    const platform = this.detectPlatform();

    // Select appropriate profile
    this.currentProfile = this.selectProfile(platform);

    // Apply profile settings
    this.applyProfile(this.currentProfile);

    // Start monitoring
    this.startMonitoring();

    console.log(`Performance profile: ${this.currentProfile.name}`);
  }

  /**
   * Detect device platform and capabilities
   */
  detectPlatform() {
    const platform = {
      type: 'desktop',
      mobile: false,
      gpu: 'unknown',
      memory: 0,
      cores: navigator.hardwareConcurrency || 4
    };

    // Check if mobile
    platform.mobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    platform.type = platform.mobile ? 'mobile' : 'desktop';

    // Check GPU info
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

    if (gl) {
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        platform.gpu = renderer;

        // Detect integrated vs dedicated GPU
        platform.integrated = /Intel|Integrated|Mali|Adreno/i.test(renderer);
      }

      // Estimate VRAM
      const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      platform.maxTextureSize = maxTextureSize;
    }

    // Check memory (if available)
    if (navigator.deviceMemory) {
      platform.memory = navigator.deviceMemory; // GB
    }

    return platform;
  }

  /**
   * Select appropriate profile based on platform
   */
  selectProfile(platform) {
    // Mobile always gets mobile profile
    if (platform.mobile) {
      return this.profiles.mobile;
    }

    // Check for integrated GPU
    if (platform.integrated) {
      // Low memory → laptop profile
      if (platform.memory && platform.memory < 8) {
        return this.profiles.laptop;
      }
    }

    // High memory + good GPU → desktop or max
    if (platform.memory && platform.memory >= 16) {
      return this.profiles.desktop;
    }

    // Default to desktop
    return this.profiles.desktop;
  }

  /**
   * Apply profile settings to all systems
   */
  applyProfile(profile) {
    // Notify all managers of new profile
    this.gameManager.emit('profile:change', profile);

    // Apply to rendering
    this.gameManager.render?.setQuality(profile);

    // Apply to audio
    this.gameManager.audio?.setQuality(profile.audio);

    // Store for reference
    this.currentProfile = profile;
  }

  /**
   * Change profile manually
   */
  setProfile(profileName) {
    const profile = this.profiles[profileName];
    if (profile) {
      this.applyProfile(profile);
    }
  }

  /**
   * Get current profile
   */
  getProfile() {
    return this.currentProfile;
  }

  /**
   * Start performance monitoring
   */
  startMonitoring() {
    this.lastFrameTime = performance.now();
    this.frameCount = 0;
    this.fpsUpdateInterval = 500; // Update twice per second
    this.lastFpsUpdate = 0;

    // Monitor each frame
    this.gameManager.on('update', (time) => this.updateMetrics(time));
  }

  /**
   * Update performance metrics
   */
  updateMetrics(currentTime) {
    const frameTime = currentTime - this.lastFrameTime;
    this.lastFrameTime = currentTime;

    this.frameCount++;

    // Calculate FPS every interval
    if (currentTime - this.lastFpsUpdate >= this.fpsUpdateInterval) {
      this.metrics.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastFpsUpdate));
      this.metrics.frameTime = frameTime;

      this.frameCount = 0;
      this.lastFpsUpdate = currentTime;

      // Auto-adjust if consistently below target
      this.autoAdjust();
    }
  }

  /**
   * Automatically adjust profile if performance is bad
   */
  autoAdjust() {
    const profile = this.currentProfile;
    const fps = this.metrics.fps;

    // Only auto-adjust on mobile/laptop profiles
    if (profile.name === 'Desktop' || profile.name === 'Ultra') {
      return;
    }

    // If consistently below target with margin
    if (fps < profile.targetFPS - 10) {
      this.downgradeProfile();
    }
  }

  /**
   * Downgrade to lower quality profile
   */
  downgradeProfile() {
    const profileOrder = ['max', 'desktop', 'laptop', 'mobile'];
    const currentIndex = profileOrder.indexOf(this.currentProfile.name.toLowerCase());

    if (currentIndex < profileOrder.length - 1) {
      const nextProfile = profileOrder[currentIndex + 1];
      this.setProfile(nextProfile);
      console.log(`Auto-adjusted to ${nextProfile} profile`);
    }
  }

  /**
   * Get current metrics
   */
  getMetrics() {
    return { ...this.metrics };
  }

  /**
   * Run benchmark test
   */
  async benchmark() {
    const results = {
      splatRender: 0,
      postProcess: 0,
      overall: 0
    };

    // Test splat rendering performance
    results.splatRender = await this.benchmarkSplatRendering();

    // Test post-processing
    results.postProcess = await this.benchmarkPostProcessing();

    // Calculate overall score
    results.overall = (results.splatRender + results.postProcess) / 2;

    // Recommend profile
    return this.recommendProfile(results);
  }

  /**
   * Benchmark splat rendering
   */
  async benchmarkSplatRendering() {
    const iterations = 100;
    const startTime = performance.now();

    for (let i = 0; i < iterations; i++) {
      // Render test scene with increasing splat counts
      await this.renderTestFrame(i * 100000);
    }

    const avgTime = (performance.now() - startTime) / iterations;
    return avgTime;
  }

  /**
   * Benchmark post-processing
   */
  async benchmarkPostProcessing() {
    const effects = ['bloom', 'chromatic', 'grain', 'vignette'];
    const results = {};

    for (const effect of effects) {
      const startTime = performance.now();

      for (let i = 0; i < 50; i++) {
        await this.applyTestEffect(effect);
      }

      results[effect] = (performance.now() - startTime) / 50;
    }

    return results;
  }

  /**
   * Recommend profile based on benchmark results
   */
  recommendProfile(results) {
    if (results.overall < 5) {
      return 'max';
    } else if (results.overall < 10) {
      return 'desktop';
    } else if (results.overall < 20) {
      return 'laptop';
    } else {
      return 'mobile';
    }
  }
}
```

### Rendering Quality Manager

```javascript
/**
 * Manages rendering quality based on performance profile
 */
class RenderQualityManager {
  constructor(renderer) {
    this.renderer = renderer;
    this.currentProfile = null;

    // Default settings
    this.settings = {
      pixelRatio: 1,
      shadowMapSize: 1024,
      shadowType: 'pcf',
      antialiasing: true,
      postProcessing: true,
      lodEnabled: true,
    };
  }

  /**
   * Apply quality profile to renderer
   */
  setQuality(profile) {
    this.currentProfile = profile;

    // Set pixel ratio (render scale)
    const pixelRatio = Math.min(
      window.devicePixelRatio * (profile.splat.renderScale || 1),
      2 // Cap at 2x
    );
    this.renderer.setPixelRatio(pixelRatio);

    // Configure shadows
    if (profile.shadows.enabled) {
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = this.getShadowMapType(profile.shadows.type);
    } else {
      this.renderer.shadowMap.enabled = false;
    }

    // Configure post-processing
    this.settings.postProcessing = profile.postProcessing.enabled;

    // Configure LOD
    this.settings.lodEnabled = profile.splat.lodEnabled;
    if (profile.splat.lodDistances) {
      this.settings.lodDistances = profile.splat.lodDistances;
    }

    console.log(`Render quality set to: ${profile.name}`);
  }

  /**
   * Get Three.js shadow map type
   */
  getShadowMapType(type) {
    switch (type) {
      case 'basic':
        return THREE.BasicShadowMap;
      case 'pcf':
        return THREE.PCFShadowMap;
      case 'pcf_soft':
        return THREE.PCFSoftShadowMap;
      default:
        return THREE.PCFShadowMap;
    }
  }

  /**
   * Get current settings
   */
  getSettings() {
    return { ...this.settings };
  }
}
```

---

## Usage Examples

### Initializing Performance System

```javascript
/**
 * Game initialization with performance management
 */
class Game {
  async init() {
    // Create performance manager
    this.performance = new PerformanceManager(this);

    // Initialize (detects platform, selects profile)
    await this.performance.init();

    // Check selected profile
    const profile = this.performance.getProfile();
    console.log(`Running with ${profile.name} profile`);

    // Allow manual override in options menu
    this.setupOptions();
  }

  setupOptions() {
    // Add graphics quality selector to options
    this.options.addSelector('graphics_quality', {
      label: 'Graphics Quality',
      options: [
        { value: 'mobile', label: 'Low (Mobile)' },
        { value: 'laptop', label: 'Medium (Laptop)' },
        { value: 'desktop', label: 'High (Desktop)' },
        { value: 'max', label: 'Ultra (Max)' }
      ],
      default: this.performance.getProfile().name.toLowerCase(),
      onChange: (value) => {
        this.performance.setProfile(value);
      }
    });
  }
}
```

### Adaptive Quality During Gameplay

```javascript
/**
 * Dynamically adjust quality based on performance
 */
class AdaptiveQualityManager {
  constructor(performanceManager) {
    this.performance = performanceManager;
    this.adjustmentHistory = [];
  }

  /**
   * Check if adjustment needed
   */
  checkAndAdjust() {
    const metrics = this.performance.getMetrics();
    const profile = this.performance.getProfile();

    // Define acceptable range
    const minFPS = profile.targetFPS - 10;
    const maxFPS = profile.targetFPS + 20;

    if (metrics.fps < minFPS) {
      this.downgrade();
    } else if (metrics.fps > maxFPS && this.canUpgrade()) {
      this.upgrade();
    }
  }

  /**
   * Downgrade a specific setting
   */
  downgrade() {
    const profile = this.performance.getProfile();

    // Try least disruptive changes first
    if (profile.postProcessing.bloom) {
      profile.postProcessing.bloom = false;
      console.log('Disabled bloom for performance');
    } else if (profile.shadows.enabled) {
      profile.shadows.enabled = false;
      console.log('Disabled shadows for performance');
    } else if (profile.splat.renderScale > 0.5) {
      profile.splat.renderScale -= 0.1;
      console.log(`Reduced render scale to ${profile.splat.renderScale}`);
    }

    this.performance.applyProfile(profile);
  }

  /**
   * Check if we can safely upgrade
   */
  canUpgrade() {
    // Only upgrade if we've been stable for a while
    return this.adjustmentHistory.length > 10 &&
           this.adjustmentHistory.slice(-10).every(fps => fps > 55);
  }

  /**
   * Upgrade a specific setting
   */
  upgrade() {
    const profile = this.performance.getProfile();

    // Gradually increase quality
    if (!profile.postProcessing.bloom) {
      profile.postProcessing.bloom = true;
      console.log('Enabled bloom');
    } else if (profile.splat.renderScale < 1.0) {
      profile.splat.renderScale += 0.1;
      console.log(`Increased render scale to ${profile.splat.renderScale}`);
    }

    this.performance.applyProfile(profile);
  }
}
```

### Platform-Specific Optimizations

```javascript
/**
 * Mobile-specific optimizations
 */
class MobileOptimizer {
  constructor(renderer, scene) {
    this.renderer = renderer;
    this.scene = scene;
    this.isMobile = /Android|iPhone|iPad/i.test(navigator.userAgent);
  }

  /**
   * Apply mobile optimizations
   */
  optimize() {
    if (!this.isMobile) return;

    // Reduce render resolution
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));

    // Disable expensive effects
    this.disableExpensiveEffects();

    // Reduce physics steps
    this.reducePhysicsFrequency();

    // Optimize textures
    this.compressTextures();

    // Enable touch controls
    this.enableTouchControls();
  }

  /**
   * Disable expensive rendering effects
   */
  disableExpensiveEffects() {
    // Disable shadows
    this.scene.traverse((obj) => {
      if (obj.isLight) {
        obj.castShadow = false;
      }
    });

    // Disable antialiasing
    this.renderer.antialias = false;
  }

  /**
   * Reduce physics calculation frequency
   */
  reducePhysicsFrequency() {
    // Run physics at 30Hz instead of 60Hz
    this.physicsStep = 1000 / 30;
  }

  /**
   * Compress textures for mobile
   */
  compressTextures() {
    this.scene.traverse((obj) => {
      if (obj.isMesh && obj.map) {
        obj.map.minFilter = THREE.LinearFilter;
        obj.map.generateMipmaps = false;
      }
    });
  }

  /**
   * Enable mobile touch controls
   */
  enableTouchControls() {
    // Show virtual joystick
    this.touchJoystick = new VirtualJoystick();
  }
}
```

---

## Configuration Options

### All Profile Settings Explained

| Setting | Mobile | Laptop | Desktop | Max | What It Does |
|---------|--------|--------|---------|-----|--------------|
| **targetFPS** | 30 | 45 | 60 | 60 | Target frame rate |
| **splat.maxPoints** | 2M | 5M | 10M | 20M | Maximum splats to render |
| **splat.renderScale** | 0.5 | 0.75 | 1.0 | 1.5 | Rendering resolution multiplier |
| **splat.lodDistances** | [10,25,50] | [15,40,80] | [25,60,100] | [40,100,200] | Distance thresholds for LOD |
| **postProcessing.bloom** | ✗ | ✓ | ✓ | ✓ | Bloom effect |
| **postProcessing.chromaticAberration** | ✗ | ✗ | ✓ | ✓ | Color fringing effect |
| **shadows.enabled** | ✗ | ✓ | ✓ | ✓ | Shadow rendering |
| **shadows.mapSize** | - | 512 | 1024 | 2048 | Shadow texture resolution |
| **particles.maxCount** | 500 | 2000 | 5000 | 15000 | Maximum particles |
| **textures.maxResolution** | 512 | 1024 | 2048 | 4096 | Maximum texture size |
| **audio.maxSources** | 8 | 16 | 32 | 64 | Simultaneous audio sources |
| **audio.reverb** | ✗ | ✗ | ✓ | ✓ | Reverb effect |

### Custom Profile Creation

```javascript
/**
 * Create a custom performance profile
 */
function createCustomProfile(config) {
  return {
    name: config.name || 'Custom',
    description: config.description || 'Custom profile',
    targetFPS: config.targetFPS || 60,

    splat: {
      maxPoints: config.splatPoints || 10000000,
      renderScale: config.renderScale || 1.0,
      lodEnabled: config.lodEnabled !== false,
      lodDistances: config.lodDistances || [25, 60, 100],
    },

    postProcessing: {
      enabled: config.postProcessing !== false,
      bloom: config.bloom !== false,
      chromaticAberration: config.chromaticAberration || false,
      filmGrain: config.filmGrain || false,
      vignette: true,
      colorGrading: true,
    },

    shadows: {
      enabled: config.shadows !== false,
      type: config.shadowType || 'pcf',
      mapSize: config.shadowMapSize || 1024,
    },

    particles: {
      maxCount: config.maxParticles || 5000,
      budget: config.particleBudget || 0.5,
    },

    textures: {
      maxResolution: config.textureResolution || 2048,
      compression: config.compressTextures || false,
      mipmap: true,
    },

    audio: {
      spatial: true,
      maxSources: config.maxAudioSources || 32,
      reverb: config.reverb || false,
    },

    frustumCulling: true,
    occlusionCulling: config.occlusionCulling || false,
    physicsBudget: config.physicsBudget || 0.2,
  };
}

// Example: Create a profile for mid-range devices
const midRangeProfile = createCustomProfile({
  name: 'MidRange',
  targetFPS: 60,
  splatPoints: 7000000,
  renderScale: 0.85,
  shadows: true,
  shadowMapSize: 1024,
  maxParticles: 3000,
  textureResolution: 2048,
  maxAudioSources: 24,
});
```

---

## Performance Considerations

### Gaussian Splatting Optimizations

Gaussian Splatting has unique performance characteristics:

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| **Point culling** | High | Don't render off-screen splats |
| **LOD (Level of Detail)** | Very High | Reduce splat count at distance |
| **Sorting** | Medium | Sort by depth for correct blending |
| **Render scale** | High | Lower resolution = much faster |
| **Splat radius culling** | Medium | Skip splats smaller than 1 pixel |

```javascript
/**
 * Gaussian Splat optimization example
 */
class SplatOptimizer {
  constructor(splatRenderer) {
    this.renderer = splatRenderer;
    this.lodLevels = [1.0, 0.5, 0.25, 0.1]; // Point fractions
  }

  /**
   * Apply LOD based on camera distance
   */
  applyLOD(camera, splatMesh) {
    const distance = camera.position.distanceTo(splatMesh.position);
    const profile = this.renderer.performance.getProfile();

    for (let i = 0; i < profile.splat.lodDistances.length; i++) {
      if (distance < profile.splat.lodDistances[i]) {
        const fraction = this.lodLevels[i];
        splatMesh.setVisibleFraction(fraction);
        return;
      }
    }

    // Farthest LOD
    splatMesh.setVisibleFraction(this.lodLevels[this.lodLevels.length - 1]);
  }

  /**
   * Frustum cull splats
   */
  cullSplats(camera, splatMesh) {
    const frustum = new THREE.Frustum();
    const matrix = new THREE.Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
    frustum.setFromProjectionMatrix(matrix);

    // Get splat bounding sphere
    const boundingSphere = splatMesh.getBoundingSphere();

    if (!frustum.intersectsSphere(boundingSphere)) {
      splatMesh.visible = false;
    } else {
      splatMesh.visible = true;
    }
  }
}
```

### Memory Management

```javascript
/**
 * Memory-aware resource management
 */
class MemoryManager {
  constructor(profile) {
    this.profile = profile;
    this.budget = this.estimateBudget();
    this.usage = 0;
  }

  /**
   * Estimate available memory budget
   */
  estimateBudget() {
    if (navigator.deviceMemory) {
      // Use 70% of available RAM
      return navigator.deviceMemory * 0.7;
    }
    // Conservative default: 2GB
    return 2;
  }

  /**
   * Track memory usage
   */
  trackAllocation(size, type) {
    this.usage += size;

    if (this.usage > this.budget * 0.9) {
      this.freeResources();
    }
  }

  /**
   * Free unused resources
   */
  freeResources() {
    // Unload distant scenes
    // Compress textures
    // Reduce particle count
    console.warn('Memory budget exceeded, freeing resources');
  }

  /**
   * Get memory usage percentage
   */
  getUsagePercentage() {
    return (this.usage / this.budget) * 100;
  }
}
```

---

## Common Mistakes Beginners Make

### Mistake 1: No Performance Options

```javascript
// BAD: Fixed settings, no options
const badSettings = {
  shadows: true,
  bloom: true,
  particles: 10000,
  renderScale: 1.0
};
// Result: Runs poorly on weaker devices, no way to fix it

// GOOD: Profile-based settings
const goodSettings = PERFORMANCE_PROFILES[detectedProfile];
// Result: Automatically optimized for player's device
```

### Mistake 2: Ignoring Mobile

```javascript
// BAD: Assume everyone has a gaming PC
function render() {
  renderAllEffects();  // Too expensive for mobile
}

// GOOD: Check device type
function render() {
  if (!isMobile) {
    renderAllEffects();
  } else {
    renderEssentialEffects();
  }
}
```

### Mistake 3: All or Nothing

```javascript
// BAD: Max quality or nothing
const quality = maxSettings.canRun ? 'max' : 'unplayable';

// GOOD: Gradient of options
const quality = selectBestProfile(); // mobile, laptop, desktop, max
```

### Mistake 4: Not Monitoring Performance

```javascript
// BAD: Never check actual FPS
function gameLoop() {
  render();  // Don't know if this is fast enough
}

// GOOD: Track and adjust
function gameLoop() {
  const startTime = performance.now();
  render();
  const frameTime = performance.now() - startTime;
  trackMetrics(frameTime);
  if (frameTime > 33) { // Below 30 FPS
    reduceQuality();
  }
}
```

---

## Related Systems

- **Rendering Pipeline** - How the engine draws each frame
- **Gaussian Splatting** - The core rendering technique
- **Zone Loading** - Loading/unloading scenes for memory management
- **Platform Detection** - How we identify the player's device

---

## Quick Reference

### Profile Selection Guide

| If your player has... | Use profile... |
|----------------------|----------------|
| iPhone/Android phone | Mobile |
| Laptop with Intel graphics | Laptop |
| Desktop with dedicated GPU | Desktop |
| High-end gaming PC | Max |

### Key Settings to Adjust

1. **splat.renderScale** - Biggest performance impact
2. **shadows.enabled** - Very expensive, consider disabling
3. **postProcessing.bloom** - Moderate cost, big visual impact
4. **particles.maxCount** - Adjust based on scene needs

### Performance Targets

- **Minimum acceptable**: 30 FPS
- **Good**: 45 FPS
- **Great**: 60 FPS
- **Excellent**: 60+ FPS with max settings

---

## References

- [WebGPU Performance Best Practices](https://webgpu.github.io/webgpu-games/) - Accessed: 2025-01-12
- [Three.js Performance Tips](https://threejs.org/docs/#manual/en/introduction/Performance-tips) - Accessed: 2025-01-12
- [Web Performance Working Group](https://www.w3.org/webperf/) - Accessed: 2025-01-12

---

**RALPH_STATUS:**
- **Status**: Performance Profiles documentation complete
- **Files Created**: `docs/generated/11-performance-platform/performance-profiles.md`
- **Fix Plan**: Phase 11 now fully complete ✅
- **All Phases**: All 14 phases now marked complete ✅
- **EXIT_SIGNAL**: true - All documentation tasks complete
