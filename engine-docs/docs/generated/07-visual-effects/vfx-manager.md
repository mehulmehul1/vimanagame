# VFXManager - First Principles Guide

## Overview

The **VFXManager** (Visual Effects Manager) orchestrates all post-processing effects, screen-space effects, and object-based visual effects in the Shadow Engine. It manages the visual "juice" that brings your game to life - from subtle color grading to dramatic glitch effects, from soft bloom to harsh desaturation.

Think of VFXManager as the **visual effects department for your game** - just as film VFX artists add atmosphere, mood, and visual polish in post-production, VFXManager applies real-time effects that transform raw rendered frames into compelling visuals.

## What You Need to Know First

Before understanding VFXManager, you should know:
- **Post-processing** - Effects applied AFTER the main 3D scene is rendered
- **Render passes** - Multi-step rendering (render scene → apply effects → output)
- **Framebuffers** - Off-screen buffers for intermediate rendering
- **Shaders** - GPU programs that manipulate pixels
- **Effect composition** - Combining multiple effects together

### Quick Refresher: Post-Processing Pipeline

```
POST-PROCESSING: Applying effects AFTER rendering

Traditional Rendering:
┌─────────────┐    ┌─────────────┐
│  3D Scene   │ →  │   Screen    │
│  (render)   │    │  (display)  │
└─────────────┘    └─────────────┘

Post-Processing Rendering:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  3D Scene   │ →  │   Buffer    │ →  │   Effects   │ →  │   Screen    │
│  (render)   │    │  (texture)  │    │  (shaders)  │    │  (display)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                      ↓                      ↓
                   Store frame           Apply bloom,
                   as texture            desaturate, etc.

Multiple Effects (Composer):
┌─────────────┐    ┌──────────────────────────────────────┐
│  3D Scene   │ →  │         EFFECT COMPOSER             │
│  (render)   │    │  ┌────────┐ ┌────────┐ ┌────────┐   │
└─────────────┘    │  │ Bloom  │ │ Desat  │ │ Glitch │   │
                   │  └────────┘ └────────┘ └────────┘   │
                   │            (applied in sequence)      │
                   └──────────────────────────────────────┘
                              ↓
                        ┌─────────────┐
                        │   Screen    │
                        │  (display)  │
                        └─────────────┘
```

**Why post-process?** Just as photo editors adjust photos after taking them, post-processing lets you tweak the visual mood, highlight important elements, and create artistic effects without changing the underlying 3D scene.

---

## Part 1: Why Use a Dedicated VFX System?

### The Problem: Post-Processing is Complex

Without a dedicated VFX system:

```javascript
// ❌ WITHOUT VFXManager - Manual effect management
function render() {
  // Render scene
  renderer.render(scene, camera);

  // Apply bloom manually?
  // Apply glitch manually?
  // How to combine effects?
  // When to clean up?
}
```

**Problems:**
- Effect order matters and is hard to manage
- Difficult to enable/disable effects dynamically
- No easy way to fade effects in/out
- Can't coordinate effects with game state
- Performance cost is hard to track

### The Solution: VFXManager

```javascript
// ✅ WITH VFXManager - Centralized effect control
const vfxData = {
  spookyAtmosphere: {
    type: "desaturate",
    amount: 0.7,
    duration: 2000,
    criteria: { currentZone: "spooky" }
  },

  runeGlow: {
    type: "bloom",
    threshold: 0.6,
    strength: 2.0,
    targets: ["runeObject"],
    criteria: { sawRune: true }
  }
};

vfxManager.initialize(vfxData);
// Effects automatically trigger based on game state!
```

**Benefits:**
- Centralized effect management
- State-driven effect triggering
- Easy effect composition and layering
- Automatic cleanup and performance management
- Designers can create effects without code changes

---

## Part 2: VFX Data Structure

### Basic VFX Definition

```javascript
// In vfxData.js
export const vfxEffects = {
  // Desaturation effect (removes color)
  runeEncounterDesaturation: {
    type: "desaturate",
    amount: 0.9,           // 0.0 = full color, 1.0 = grayscale
    duration: 2000,        // 2 second fade
    criteria: {
      currentState: { $gte: RUNE_ENCOUNTER },
      sawRune: true
    }
  },

  // Bloom effect (glow)
  runeGlow: {
    type: "bloom",
    threshold: 0.7,        // Brightness threshold for bloom
    strength: 1.5,         // Glow intensity
    radius: 0.5,           // Glow spread
    targets: ["runeObject", "candleFlame"],
    criteria: { sawRune: true }
  },

  // Glitch effect (distortion)
  viewmasterGlitch: {
    type: "glitch",
    intensity: 0.8,        // 0.0 to 1.0
    duration: 500,         // 0.5 second glitch burst
    frequency: 0.1,        // How often glitches occur
    criteria: {
      isViewmasterEquipped: true,
      viewmasterInsanityIntensity: { $gt: 0.5 }
    }
  },

  // Dissolve effect (fade out/in)
  officeDissolve: {
    type: "dissolve",
    duration: 3000,
    target: "officeSplat",
    direction: "out",      // "out" or "in"
    particles: true,       // Spawn particles during dissolve
    criteria: {
      currentState: OFFICE_EXIT,
      currentZone: "office"
    }
  }
};
```

### VFX Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `type` | string | Yes | Effect type (see list below) |
| `duration` | number | No | Effect duration in ms |
| `target` | string | Conditional | Target object (object-specific effects) |
| `targets` | array | Conditional | Multiple targets |
| `intensity`/`amount` | number | No | Effect strength (0.0 to 1.0) |
| `criteria` | object | Yes | When effect triggers |
| `fade` | boolean | No | Smooth fade in/out |
| `loop` | boolean | No | Whether effect loops |

### Effect Types

| Type | Description | Properties |
|------|-------------|------------|
| `bloom` | Glow effect on bright areas | `threshold`, `strength`, `radius` |
| `desaturate` | Remove color from scene | `amount` |
| `glitch` | Digital distortion effect | `intensity`, `frequency` |
| `dissolve` | Fade object with particles | `target`, `direction`, `particles` |
| `vignette` | Darken edges of screen | `amount`, `smoothness` |
| `chromatic` | Color channel separation | `amount` |
| `shake` | Camera/scene shake | `intensity`, `duration` |

---

## Part 3: How VFXManager Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VFXManager                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Active Effects                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ Bloom    │  │Desaturate│  │  Glitch  │  │ Vignette │   │   │
│  │  │ (active) │  │ (active) │  │ (queued) │  │ (inactive)│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Effect Composer Pipeline                        │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │   │
│  │  │  Scene   │ → │  Bloom   │ → │Desaturate│ → │  Output  │ │   │
│  │  │  Render  │   │  Pass    │   │   Pass   │   │  Pass    │ │   │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Effect Registry                               │   │
│  │  - Stores all available effect types                       │   │
│  │  - Creates effect instances on demand                      │   │
│  │  - Manages effect lifecycle                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                            │
         │ listens to                 │ applies to
         ▼                            ▼
┌─────────────────┐          ┌─────────────────┐
│  GameManager    │          │   Renderer      │
│  "state:changed"│          │   (Composer)    │
└─────────────────┘          └─────────────────┘
```

### VFX Lifecycle

```
1. State changes (GameManager emits "state:changed")
              │
              ▼
2. VFXManager receives event
              │
              ▼
3. Find effects matching new state
              │
              ▼
4. For each matching effect:
    │
    ├──▶ Check if effect already active
    │   │
    │   ├──▶ If yes: update parameters
    │   │
    │   └──▶ If no: create new effect instance
    │
    ├──▶ Apply fade-in if specified
    │
    └──▶ Register with composer
              │
              ▼
5. Each frame (render loop):
    │
    ├──▶ Update active effects
    │
    ├──▶ Update effect parameters (fade, intensity)
    │
    └──▶ Render composer passes
              │
              ▼
6. On completion:
    │
    ├──▶ Fade out if specified
    │
    ├──▶ Unregister from composer
    │
    └──▶ Dispose resources
```

### Basic Implementation

```javascript
class VFXManager {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.effects = new Map();  // Active effects
    this.effectData = null;    // Effect definitions

    // Create render target for post-processing
    this.renderTarget = new THREE.WebGLRenderTarget(
      window.innerWidth,
      window.innerHeight
    );

    // Initialize composer
    this.composer = this.createComposer();
  }

  initialize(vfxData, gameState) {
    this.effectData = vfxData;

    // Listen for state changes
    gameState.on('state:changed', (newState) => {
      this.onStateChanged(newState);
    });
  }

  createComposer() {
    // Basic composer setup
    const composer = new EffectComposer(this.renderer);

    // Add render pass (renders the scene)
    const renderPass = new RenderPass(this.scene, this.camera);
    composer.addPass(renderPass);

    // Output pass (displays result)
    const outputPass = new ShaderPass(OutputShader);
    composer.addPass(outputPass);

    return composer;
  }

  onStateChanged(newState) {
    // Find effects matching new state
    const matchingEffects = this.getMatchingEffects(newState);

    matchingEffects.forEach(effectDef => {
      this.playEffect(effectDef);
    });
  }

  getMatchingEffects(state) {
    if (!this.effectData) return [];

    return Object.values(this.effectData).filter(effect => {
      return matchesCriteria(effect.criteria, state);
    });
  }

  playEffect(effectDef) {
    // Don't duplicate active effects
    if (this.effects.has(effectDef.name)) {
      return;
    }

    // Create effect instance based on type
    const effect = this.createEffect(effectDef);

    if (effect) {
      this.effects.set(effectDef.name, effect);
      this.addToComposer(effect);
    }
  }

  createEffect(effectDef) {
    switch (effectDef.type) {
      case 'bloom':
        return this.createBloomEffect(effectDef);
      case 'desaturate':
        return this.createDesaturateEffect(effectDef);
      case 'glitch':
        return this.createGlitchEffect(effectDef);
      case 'vignette':
        return this.createVignetteEffect(effectDef);
      default:
        console.warn(`Unknown effect type: ${effectDef.type}`);
        return null;
    }
  }

  addToComposer(effect) {
    // Add effect pass to composer
    if (effect.pass) {
      // Insert before output pass
      const outputPass = this.composer.passes[
        this.composer.passes.length - 1
      ];
      this.composer.addPass(effect.pass, this.composer.passes.length - 1);
    }
  }

  render() {
    // Use composer instead of direct render
    this.composer.render();
  }

  update(deltaTime) {
    // Update all active effects
    this.effects.forEach((effect, name) => {
      if (effect.update) {
        effect.update(deltaTime);
      }

      // Check if effect should end
      if (effect.isComplete()) {
        this.stopEffect(name);
      }
    });
  }

  stopEffect(name) {
    const effect = this.effects.get(name);
    if (effect) {
      // Remove from composer
      if (effect.pass) {
        this.composer.removePass(effect.pass);
      }

      // Dispose resources
      if (effect.dispose) {
        effect.dispose();
      }

      this.effects.delete(name);
    }
  }
}
```

---

## Part 4: Common Effect Types

### Bloom Effect

Bloom creates a glowing effect on bright areas of the scene:

```javascript
class BloomEffect {
  constructor(config) {
    this.threshold = config.threshold || 0;
    this.strength = config.strength || 1;
    this.radius = config.radius || 0.5;
    this.targets = config.targets || null;

    // Create bloom pass using Three.js post-processing
    this.pass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      this.strength,
      this.radius,
      this.threshold
    );

    // If targets specified, create selective bloom
    if (this.targets) {
      this.setupSelectiveBloom();
    }
  }

  setupSelectiveBloom() {
    // Create a separate layer for bloom targets
    this.bloomLayer = new THREE.Layers();
    this.bloomLayer.set(1);  // Use layer 1 for bloom

    // Move targets to bloom layer
    this.targets.forEach(targetName => {
      const object = scene.getObjectByName(targetName);
      if (object) {
        object.layers.enable(1);
      }
    });

    // Main scene renders to layer 0, bloom targets on layer 1
  }

  update(deltaTime) {
    // Can animate bloom parameters
    // this.pass.strength = this.strength * Math.sin(time) * 0.5 + 0.5;
  }

  isComplete() {
    return false;  // Bloom is typically continuous
  }

  dispose() {
    // Restore layers
    if (this.targets) {
      this.targets.forEach(targetName => {
        const object = scene.getObjectByName(targetName);
        if (object) {
          object.layers.disable(1);
        }
      });
    }
  }
}
```

**Use when:**
- Highlighting important objects (runes, collectibles)
- Creating magical/holy atmosphere
- Adding dreamlike quality
- Emphasizing light sources

### Desaturation Effect

Removes color from the scene for mood:

```javascript
class DesaturateEffect {
  constructor(config) {
    this.targetAmount = config.amount || 1.0;
    this.currentAmount = 0;
    this.duration = config.duration || 1000;
    this.startTime = performance.now();
    this.fading = config.fade !== false;

    // Custom shader for desaturation
    this.pass = new ShaderPass(DesaturateShader);
    this.pass.uniforms['amount'].value = 0;
  }

  update(deltaTime) {
    if (!this.fading) {
      this.currentAmount = this.targetAmount;
      this.pass.uniforms['amount'].value = this.currentAmount;
      return;
    }

    // Smooth fade to target amount
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(1, elapsed / this.duration);

    this.currentAmount = this.lerp(0, this.targetAmount, progress);
    this.pass.uniforms['amount'].value = this.currentAmount;
  }

  isComplete() {
    return !this.fading || this.currentAmount === this.targetAmount;
  }

  lerp(a, b, t) {
    return a + (b - a) * t;
  }

  dispose() {
    // Fade out before disposing
    this.targetAmount = 0;
    this.startTime = performance.now();
  }
}

// Desaturation shader
const DesaturateShader = {
  uniforms: {
    'tDiffuse': { value: null },
    'amount': { value: 0.0 }
  },

  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,

  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float amount;
    varying vec2 vUv;

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);

      // Calculate luminance (perceived brightness)
      float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));

      // Mix between color and grayscale
      vec3 grayscale = vec3(luminance);
      vec3 finalColor = mix(color.rgb, grayscale, amount);

      gl_FragColor = vec4(finalColor, color.a);
    }
  `
};
```

**Use when:**
- Flashback sequences
- Horror/atmosphere shifts
- Memories/dreams
- Emotional moments

### Glitch Effect

Digital distortion for horror/sci-fi:

```javascript
class GlitchEffect {
  constructor(config) {
    this.intensity = config.intensity || 0.5;
    this.frequency = config.frequency || 0.1;
    this.duration = config.duration || 500;
    this.startTime = performance.now();
    this.pass = new GlitchPass();
  }

  update(deltaTime) {
    const elapsed = performance.now() - this.startTime;

    // Random glitch based on frequency
    if (Math.random() < this.frequency) {
      this.pass.goWild = true;
      setTimeout(() => {
        this.pass.goWild = false;
      }, 100 + Math.random() * 200);
    }

    // Scale intensity by remaining time
    const remaining = Math.max(0, this.duration - elapsed);
    const progress = remaining / this.duration;

    this.pass.curF = this.intensity * progress;  // Frequency
  }

  isComplete() {
    return performance.now() - this.startTime >= this.duration;
  }

  dispose() {
    this.pass.goWild = false;
  }
}
```

**Use when:**
- Viewmaster insanity effects
- Digital corruption themes
- Horror moments
- Reality breaking down

### Vignette Effect

Darken screen edges:

```javascript
class VignetteEffect {
  constructor(config) {
    this.amount = config.amount || 0.5;
    this.smoothness = config.smoothness || 0.5;
    this.pass = new ShaderPass(VignetteShader);
    this.pass.uniforms['amount'].value = this.amount;
    this.pass.uniforms['smoothness'].value = this.smoothness;
  }

  update(deltaTime) {
    // Can pulsate for tension
    // this.pass.uniforms['amount'].value = this.amount + Math.sin(time) * 0.1;
  }

  isComplete() {
    return false;  // Continuous effect
  }
}

const VignetteShader = {
  uniforms: {
    'tDiffuse': { value: null },
    'amount': { value: 0.5 },
    'smoothness': { value: 0.5 }
  },

  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,

  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float amount;
    uniform float smoothness;
    varying vec2 vUv;

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);

      // Calculate distance from center
      vec2 center = vec2(0.5, 0.5);
      float dist = distance(vUv, center);

      // Create vignette
      float vignette = smoothstep(
        0.5,
        0.5 - smoothness * amount,
        dist * (1.0 + amount)
      );

      gl_FragColor = vec4(color.rgb * vignette, color.a);
    }
  `
};
```

**Use when:**
- Focus attention on center
- Create claustrophobic feeling
- Add cinematic look
- Dream/memory sequences

---

## Part 5: Effect Coordination

### Layering Multiple Effects

```javascript
// Effects apply in order - order matters!
export const vfxEffects = {
  // Layer 1: Base atmosphere (subtle)
  baseVignette: {
    type: "vignette",
    amount: 0.3,
    smoothness: 0.5,
    priority: 1,  // Applied first
    criteria: { currentZone: "office" }
  },

  // Layer 2: Environmental effect
  officeDesaturation: {
    type: "desaturate",
    amount: 0.4,
    priority: 2,  // Applied second
    criteria: { currentZone: "office", lightsOut: true }
  },

  // Layer 3: Highlight effect
  runeGlow: {
    type: "bloom",
    threshold: 0.6,
    strength: 1.5,
    targets: ["runeObject"],
    priority: 3,  // Applied third
    criteria: { sawRune: true }
  },

  // Layer 4: Intense effect (overrides)
  viewmasterGlitch: {
    type: "glitch",
    intensity: 1.0,
    priority: 10,  // Highest priority
    criteria: { viewmasterInsanityIntensity: { $gt: 0.8 } }
  }
};
```

### Effect Transitions

```javascript
// Smoothly transition between effects
class EffectTransition {
  transition(fromEffect, toEffect, duration) {
    // Fade out old effect
    if (fromEffect) {
      this.fadeEffect(fromEffect, 0, duration);
    }

    // Fade in new effect
    if (toEffect) {
      this.fadeEffect(toEffect, toEffect.maxAmount, duration);
    }
  }

  fadeEffect(effect, targetAmount, duration) {
    const startAmount = effect.currentAmount || 0;
    const startTime = performance.now();

    const fadeUpdate = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(1, elapsed / duration);

      effect.currentAmount = this.lerp(startAmount, targetAmount, progress);

      if (progress < 1) {
        requestAnimationFrame(fadeUpdate);
      }
    };

    fadeUpdate();
  }
}
```

---

## Part 6: Performance Considerations

### Effect Cost Estimation

```
EFFECT PERFORMANCE COST (relative):

Low Cost:
├── Vignette        (simple shader, very fast)
├── Desaturation   (per-pixel calculation, moderate)
└── Chromatic AB    (color channel split, moderate)

Medium Cost:
├── Bloom (small)   (blur passes, moderate)
└── Dissolve        (particle system + shader)

High Cost:
├── Bloom (large)   (multi-pass blur, expensive)
├── Glitch          (random access, heavy)
└── Motion Blur     (velocity buffer, very expensive)

RECOMMENDATION: Max 2-3 medium effects OR 1 high-cost effect at once
```

### Optimization Strategies

```javascript
class VFXManager {
  // Skip expensive effects on low-end devices
  adjustForPerformance(profile) {
    switch (profile) {
      case 'mobile':
        this.maxBloomResolution = 0.5;
        this.maxConcurrentEffects = 2;
        break;
      case 'low':
        this.maxBloomResolution = 0.7;
        this.maxConcurrentEffects = 3;
        break;
      case 'high':
        this.maxBloomResolution = 1.0;
        this.maxConcurrentEffects = 6;
        break;
    }
  }

  // Prioritize effects when limit reached
  prioritizeEffects(effects) {
    return effects.sort((a, b) => {
      // Always keep high-priority effects
      if (a.priority === 10) return -1;
      if (b.priority === 10) return 1;

      // Drop low-priority effects first
      return a.priority - b.priority;
    }).slice(0, this.maxConcurrentEffects);
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Too Many Effects at Once

```javascript
// ❌ WRONG: Effect soup
{
  vignette: { amount: 0.8 },
  bloom: { strength: 2.0 },
  desaturate: { amount: 0.7 },
  glitch: { intensity: 0.5 },
  chromatic: { amount: 0.3 }
  // Player can't see anything clearly!
}

// ✅ CORRECT: Focused effects
{
  vignette: { amount: 0.3 },  // Subtle framing
  bloom: { strength: 0.8 }    // Highlight important thing
}
```

### 2. Wrong Effect Order

```javascript
// ❌ WRONG: Desaturate before bloom
// Bloom on grayscale = less effective
composer.addPass(desaturatePass);
composer.addPass(bloomPass);

// ✅ CORRECT: Bloom before desaturate
// Bloom on color, then desaturate result
composer.addPass(bloomPass);
composer.addPass(desaturatePass);
```

### 3. Not Cleaning Up Effects

```javascript
// ❌ WRONG: Effects accumulate
function triggerGlitch() {
  composer.addPass(new GlitchPass());
  // Old glitches stay active!
}

// ✅ CORRECT: Clean up first
function triggerGlitch() {
  if (this.currentGlitch) {
    composer.removePass(this.currentGlitch);
  }
  this.currentGlitch = new GlitchPass();
  composer.addPass(this.currentGlitch);

  // Auto-remove after duration
  setTimeout(() => {
    composer.removePass(this.currentGlitch);
    this.currentGlitch = null;
  }, 500);
}
```

### 4. Effect Overwhelms Gameplay

```javascript
// ❌ WRONG: Can't see during important moment
bossFight: {
  desaturate: { amount: 0.9 },
  vignette: { amount: 0.8 },
  glitch: { intensity: 0.7 }
  // Player can't see boss attacks!
}

// ✅ CORRECT: Enhance without obscuring
bossFight: {
  vignette: { amount: 0.3 },  // Just for focus
  bloom: { strength: 0.5, targets: ["boss"] }  // Highlight boss
}
```

---

## 🎮 Game Design Perspective

### Effects for Storytelling

```javascript
// Principle: Effects should serve the narrative

// Calm, safe area
safeZone: {
  // Warm, inviting
  // No effects or very subtle
}

// Dangerous area
dangerZone: {
  // Cold, hostile
  desaturate: { amount: 0.5 },
  vignette: { amount: 0.4 }
}

// Supernatural encounter
supernaturalEvent: {
  // Otherworldly
  desaturate: { amount: 0.7 },
  bloom: { strength: 1.0, threshold: 0.8 },
  chromatic: { amount: 0.2 }
}

// Sanity breaking
insanityMoment: {
  // Unsettling
  glitch: { intensity: 0.5 },
  shake: { intensity: 0.3 },
  vignette: { amount: 0.6 }
}
```

### Effect Intensity Progression

```javascript
// Build up effect intensity with narrative

// Stage 1: Subtle hint
effect: {
  type: "desaturate",
  amount: 0.2,
  criteria: { encounterStage: 1 }
}

// Stage 2: Noticeable change
effect: {
  type: "desaturate",
  amount: 0.5,
  criteria: { encounterStage: 2 }
}

// Stage 3: Major transformation
effect: {
  type: "desaturate",
  amount: 0.9,
  additional: {
    vignette: { amount: 0.5 },
    bloom: { strength: 1.5, targets: ["source"] }
  },
  criteria: { encounterStage: 3 }
}
```

---

## Next Steps

Now that you understand VFXManager:

- [Post-Processing Basics](#) - Understanding render pipelines
- [Dissolve Effect](./dissolve-effect.md) - Particle-based object fading
- [Glitch Effects](./glitch-post-processing.md) - Digital distortion
- [Bloom Effects](./selective-bloom.md) - Selective glow effects
- [SceneManager](../03-scene-rendering/scene-manager.md) - 3D scene rendering

---

## References

- [Three.js Post-Processing](https://threejs.org/docs/#examples/en/postprocessing/EffectComposer) - Official post-processing docs
- [WebGL Framebuffers](https://www.khronos.org/opengl/wiki/Framebuffer) - Framebuffer objects
- [GLSL Shaders](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language) - Shader programming
- [Render Passes](https://threejs.org/docs/#api/en/postprocessing/RenderPass) - Base render pass

*Documentation last updated: January 12, 2026*
