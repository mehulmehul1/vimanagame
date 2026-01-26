# Desaturation & Glitch Post-Processing - First Principles Guide

## Overview

**Desaturation** and **Glitch** are two powerful post-processing effects that dramatically alter the visual mood of your game. Desaturation removes color to create somber, eerie, or nostalgic atmospheres, while glitch effects introduce digital distortion for horror, sci-fi, or psychological breakdown themes.

Think of these effects as the **visual mood swings** of your game - just as music changes the emotional tone of a scene, desaturation and glitch effects transform how players *feel* about what they're seeing, often without changing a single pixel of the underlying 3D content.

## What You Need to Know First

Before understanding these effects, you should know:
- **Post-processing pipeline** - Effects applied after scene rendering
- **Fragment shaders** - GPU programs that manipulate pixel colors
- **Color theory** - How color affects mood and perception
- **RGB color model** - Red, Green, Blue channels per pixel
- **Luminance** - Perceived brightness of a color

### Quick Refresher: What These Effects Do

```
NORMAL COLOR:
┌────────────────────────────────┐
│ ████░░░░████░░░░░░░████░░░░░░░░ │  Full color scene
│ ████████░░░░░░░███████████░░░░ │
│ ░░░░████████████████████░░░░░ │
│ ░░░░░░░░██████████░░░░░░░░░░░ │
└────────────────────────────────┘

DESATURATED (Grayscale):
┌────────────────────────────────┐
│ ▓▓▓▓▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │  No color, only brightness
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒ │
└────────────────────────────────┘
Effect: Somber, serious, nostalgic

PARTIALLY DESATURATED:
┌────────────────────────────────┐
│ ████▓▓▓████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  Muted colors
│ ████████▓▓▓▓▓███████████▓▓▓▓ │
│ ▓▓▓▓████████████████████▓▓▓▓ │
│ ▓▓▓▓▓▓▓██████████▓▓▓▓▓▓▓▓▓▓▓ │
└────────────────────────────────┘
Effect: Unsettling, unnatural

GLITCH EFFECT:
┌────────────────────────────────┐
│ ████░░░░████░░▓░░░░░░████▓░░░░▓░ │  Color channels split
│ ████████░░░▒▒▒▒▒▒▓▓▓▓▓▓░░░░░ │  Horizontal tears
│ ░░░░████████▒▒▒░▒▒▒▒▒▒▓▓▓▓▓ │  Digital corruption
│ ░░░░░░░░██████████░░▒▒░▒░░░░▒▒ │
└────────────────────────────────┘
Effect: Horror, sci-fi, madness
```

---

## Part 1: Desaturation Effect

### Why Remove Color?

**Desaturation** transforms colorful scenes into grayscale (or partially gray) images. This powerful effect:

- Creates **somber, melancholic** moods
- Suggests **memories, dreams, or the past**
- Indicates **danger, death, or supernatural** presence
- Provides **visual contrast** to emphasize colorful elements
- Triggers **nostalgic feelings** (like old photographs)

### How Desaturation Works

```javascript
// The math behind grayscale conversion
// Human perception weights color channels differently

// Method 1: Rec. 601 Luma (standard for video)
luminance = 0.299 × red + 0.587 × green + 0.114 × blue

// Method 2: Simpler average
gray = (red + green + blue) / 3

// Method 3: Lightness (perceptually uniform)
gray = (max(r,g,b) + min(r,g,b)) / 2

// Desaturation mixes original color with grayscale
finalColor = mix(originalColor, grayColor, amount)
// amount = 0: full color
// amount = 1: grayscale
// amount = 0.5: half color
```

### Desaturate Shader Implementation

```javascript
const DesaturateShader = {
  name: 'DesaturateShader',

  uniforms: {
    // The rendered scene texture
    tDiffuse: { value: null },

    // How much to desaturate (0.0 = color, 1.0 = grayscale)
    amount: { value: 0.0 },

    // Which channels to affect (optional)
    desaturateR: { value: true },
    desaturateG: { value: true },
    desaturateB: { value: true }
  },

  vertexShader: `
    // Pass-through vertex shader
    // Just positions the fullscreen quad
    varying vec2 vUv;

    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,

  fragmentShader: `
    // Per-pixel desaturation
    uniform sampler2D tDiffuse;
    uniform float amount;
    uniform bool desaturateR;
    uniform bool desaturateG;
    uniform bool desaturateB;
    varying vec2 vUv;

    void main() {
      // Get original color
      vec4 color = texture2D(tDiffuse, vUv);

      // Calculate luminance (perceived brightness)
      // Using Rec. 601 weights for human perception
      float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));

      // Create grayscale version
      vec3 gray = vec3(luminance);

      // Selective desaturation (optional)
      if (!desaturateR) gray.r = color.r;
      if (!desaturateG) gray.g = color.g;
      if (!desaturateB) gray.b = color.b;

      // Mix between original and grayscale
      vec3 finalColor = mix(color.rgb, gray, amount);

      gl_FragColor = vec4(finalColor, color.a);
    }
  `
};
```

### Desaturate Effect Class

```javascript
class DesaturateEffect {
  constructor(config) {
    this.targetAmount = config.amount || 1.0;
    this.currentAmount = 0.0;
    this.duration = config.duration || 2000;
    this.startTime = performance.now();
    this.fade = config.fade !== false;

    // Create shader pass
    this.pass = new ShaderPass(DesaturateShader);
    this.pass.uniforms['amount'].value = 0;

    // Selective channel desaturation (optional)
    if (config.channels) {
      this.pass.uniforms['desaturateR'].value = config.channels.red !== false;
      this.pass.uniforms['desaturateG'].value = config.channels.green !== false;
      this.pass.uniforms['desaturateB'].value = config.channels.blue !== false;
    }
  }

  update(deltaTime) {
    if (!this.fade) {
      // Instant desaturation
      this.currentAmount = this.targetAmount;
      this.pass.uniforms['amount'].value = this.targetAmount;
      return;
    }

    // Smooth fade to target amount
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(1, elapsed / this.duration);
    const eased = this.easeInOutCubic(progress);

    this.currentAmount = this.lerp(0, this.targetAmount, eased);
    this.pass.uniforms['amount'].value = this.currentAmount;
  }

  setStrength(amount) {
    // Directly set desaturation amount (for dynamic control)
    this.targetAmount = Math.max(0, Math.min(1, amount));
    if (!this.fade) {
      this.pass.uniforms['amount'].value = this.targetAmount;
    } else {
      this.startTime = performance.now();  // Restart fade
    }
  }

  lerp(a, b, t) {
    return a + (b - a) * t;
  }

  easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  isComplete() {
    return !this.fade || this.currentAmount === this.targetAmount;
  }

  dispose() {
    // Fade out before cleanup
    this.targetAmount = 0;
    this.startTime = performance.now();
  }
}
```

### Desaturation Data Examples

```javascript
// In vfxData.js
export const vfxEffects = {
  // Full grayscale - horror moment
  horrorMoment: {
    type: "desaturate",
    amount: 1.0,           // Complete grayscale
    duration: 3000,
    fade: true,
    criteria: {
      currentState: JUMP_SCARE,
      playerHealth: { $lt: 0.3 }
    }
  },

  // Partial desaturation - unsettling atmosphere
  unsettlingAtmosphere: {
    type: "desaturate",
    amount: 0.7,           // Mostly gray, some color
    duration: 5000,
    fade: true,
    criteria: {
      currentZone: "spooky_area",
      sanityLevel: { $lt: 0.5 }
    }
  },

  // Selective desaturation - keep red only
  keepRedOnly: {
    type: "desaturate",
    amount: 1.0,
    channels: {
      red: false,     // Keep red
      green: true,    // Remove green
      blue: true      // Remove blue
    },
    duration: 2000,
    criteria: {
      currentState: DANGER_ZONE
    }
  },

  // Memory flash - sepia tone effect
  memoryFlash: {
    type: "desaturate",
    amount: 0.8,
    duration: 1000,
    fade: true,
    // Combined with color tint in composer
    criteria: {
      currentState: MEMORY_REVEAL
    }
  }
};
```

---

## Part 2: Glitch Effect

### Why Use Glitch?

The **glitch effect** simulates digital corruption - artifacts you'd see from corrupted video files, broken displays, or hacking in progress. This effect:

- Creates **unease, tension, and discomfort**
- Suggests **reality breaking down**
- Indicates **supernatural or sci-fi presence**
- Provides **horrific, nightmarish** imagery
- Triggers **technological anxiety**

### Types of Glitch Effects

```
GLITCH TYPE VISUAL REFERENCE:

1. RGB SHIFT (Chromatic Aberration)
┌────────────────────────────────┐
│ Colors separate at edges        │
│ ██████████ → ░███░███░         │
│ Red channel offset left/right   │
│ Green channel offset            │
│ Blue channel offset             │
└────────────────────────────────┘

2. SCANLINES
┌────────────────────────────────┐
│ Horizontal dark lines           │
│ ████████████████████           │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │ ← Line
│ ████████████████████           │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │ ← Line
└────────────────────────────────┘

3. TEARING
┌────────────────────────────────┐
│ Horizontal displacement          │
│ ████████████                   │
│         ███████████████████████ │
│                   ████████████   │
└────────────────────────────────┘

4. PIXELATION
┌────────────────────────────────┐
│ Blocky, low-res appearance       │
│ ██  ██  ██  ██  ██  ██  ██     │
│ ██  ██  ██  ██  ██  ██  ██     │
└────────────────────────────────┘

5. NOISE STATIC
┌────────────────────────────────┐
│ Random colored pixels            │
│ ████████░░░▒▒▒▓▓▓████░░░░▓▓▓    │
│ ▓▓▓░░░████████▓▓▓░░░▒▒▒███████  │
└────────────────────────────────┘

6. SCREEN SHAKE
┌────────────────────────────────┐
│ Entire frame jitters           │
│ (visual: frame moves rapidly)  │
└────────────────────────────────┘
```

### Glitch Shader Implementation

```javascript
const GlitchShader = {
  name: 'GlitchShader',

  uniforms: {
    tDiffuse: { value: null },

    // Overall intensity
    amount: { value: 0.5 },

    // RGB shift amount
    shiftAmount: { value: 0.02 },

    // Random trigger
    randomTrigger: { value: 0.0 },

    // Time for animation
    time: { value: 0.0 },

    // Glitch direction
    direction: { value: new THREE.Vector2(1, 0) },

    // Block size for tearing
    blockSize: { value: 50.0 }
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
    uniform float shiftAmount;
    uniform float randomTrigger;
    uniform float time;
    uniform vec2 direction;
    uniform float blockSize;
    varying vec2 vUv;

    float random(vec2 st) {
      return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
      vec2 uv = vUv;

      // Scanlines
      float scanline = sin(uv.y * 800.0 + time * 10.0) * 0.5 + 0.5;
      scanline = smoothstep(0.0, amount, scanline);

      // Tearing effect
      float tear = random(vec2(floor(uv.y * blockSize), time));
      tear = step(1.0 - amount, tear);

      float offset = tear * direction.x * shiftAmount * random(vec2(uv.y, time));

      // RGB split
      float shift = shiftAmount * amount * (sin(time * 20.0) * 0.5 + 0.5);
      vec4 colorR = texture2D(tDiffuse, uv + vec2(shift, 0.0));
      vec4 colorG = texture2D(tDiffuse, uv);
      vec4 colorB = texture2D(tDiffuse, uv - vec2(shift, 0.0));

      // Combine with tearing offset
      colorR = texture2D(tDiffuse, uv + vec2(offset, 0.0));
      colorG = texture2D(tDiffuse, uv);
      colorB = texture2D(tDiffuse, uv - vec2(offset, 0.0));

      // Combine channels
      vec3 color = vec3(colorR.r, colorG.g, colorB.b);

      // Add noise
      vec3 noise = vec3(
        random(uv + time),
        random(uv + time + 1.0),
        random(uv + time + 2.0)
      ) * amount * 0.3;

      color += noise;

      // Apply scanline darkening
      color *= (0.8 + 0.2 * scanline);

      gl_FragColor = vec4(color, 1.0);
    }
  `
};
```

### Glitch Effect Class

```javascript
class GlitchEffect {
  constructor(config) {
    this.intensity = config.intensity || 0.5;
    this.duration = config.duration || 500;
    this.frequency = config.frequency || 0.1;  // How often glitches occur
    this.startTime = performance.now();
    this.isGlitching = false;
    this.glitchEndTime = 0;

    // Create shader pass
    this.pass = new ShaderPass(GlitchShader);
    this.pass.uniforms['amount'].value = 0;
    this.pass.uniforms['time'].value = 0;

    // Glitch parameters
    this.shiftAmount = config.shiftAmount || 0.02;
    this.direction = config.direction || 'horizontal';
  }

  update(deltaTime) {
    const elapsed = performance.now() - this.startTime;
    const currentTime = performance.now() / 1000;
    this.pass.uniforms['time'].value = currentTime;

    // Check if glitch should trigger
    if (!this.isGlitching && Math.random() < this.frequency) {
      this.triggerGlitch();
    }

    // Update active glitch
    if (this.isGlitching) {
      if (currentTime < this.glitchEndTime) {
        // Active glitch - vary intensity
        const progress = (this.glitchEndTime - currentTime) / 0.2;
        const currentIntensity = this.intensity * (1 - progress);

        this.pass.uniforms['amount'].value = currentIntensity;
        this.pass.uniforms['shiftAmount'].value = this.shiftAmount * currentIntensity;
      } else {
        // Glitch complete
        this.isGlitching = false;
        this.pass.uniforms['amount'].value = 0;
      }
    }

    // Overall fade out
    if (elapsed > this.duration) {
      this.pass.uniforms['amount'].value = 0;
    }
  }

  triggerGlitch() {
    this.isGlitching = true;
    this.glitchEndTime = (performance.now() / 1000) + 0.1 + Math.random() * 0.2;

    // Randomize direction
    if (Math.random() < 0.3) {
      // Vertical glitch
      this.pass.uniforms['direction'].value = new THREE.Vector2(0, 1);
    } else {
      // Horizontal glitch
      this.pass.uniforms['direction'].value = new THREE.Vector2(1, 0);
    }
  }

  isComplete() {
    return performance.now() - this.startTime >= this.duration;
  }

  dispose() {
    this.pass.uniforms['amount'].value = 0;
  }
}
```

### Glitch Data Examples

```javascript
// In vfxData.js
export const vfxEffects = {
  // Viewmaster insanity glitch
  viewmasterGlitch: {
    type: "glitch",
    intensity: 0.8,
    duration: 500,
    frequency: 0.15,
    shiftAmount: 0.03,
    criteria: {
      isViewmasterEquipped: true,
      viewmasterInsanityIntensity: { $gt: 0.5 }
    }
  },

  // Subtle digital corruption
  subtleCorruption: {
    type: "glitch",
    intensity: 0.2,
    duration: 2000,
    frequency: 0.05,
    shiftAmount: 0.005,
    criteria: {
      currentState: DIGITAL_ZONE,
      corruptionLevel: { $gt: 0.3 }
    }
  },

  // Intense nightmare glitch
  nightmareGlitch: {
    type: "glitch",
    intensity: 1.0,
    duration: 200,
    frequency: 0.3,
    shiftAmount: 0.05,
    criteria: {
      currentState: NIGHTMARE_SEQUENCE,
      sanityLevel: { $lt: 0.1 }
    }
  },

  // Random static (TV noise style)
  staticNoise: {
    type: "glitch",
    intensity: 0.6,
    duration: 1000,
    frequency: 0.2,
    shiftAmount: 0.01,
    criteria: {
      currentState: STATIC_INTERFERENCE
    }
  }
};
```

---

## Part 3: Combined Effects

### Desaturation + Glitch (Horror Combo)

```javascript
export const vfxEffects = {
  // Combined effect for maximum unease
  nightmareFuel: {
    type: "composite",
    effects: [
      {
        type: "desaturate",
        amount: 0.7,
        duration: 5000,
        fade: true
      },
      {
        type: "glitch",
        intensity: 0.6,
        duration: 5000,
        frequency: 0.1
      },
      {
        type: "vignette",
        amount: 0.6,
        duration: 5000
      }
    ],
    criteria: {
      currentState: NIGHTMARE_MODE,
      sanityLevel: { $lt: 0.2 }
    }
  }
};
```

### Progressive Effect (Builds Over Time)

```javascript
class ProgressiveDesaturateWithGlitch {
  constructor(sanityLevel) {
    this.sanityLevel = sanityLevel;
    this.desaturateAmount = 0;
    this.glitchIntensity = 0;

    this.desaturatePass = new ShaderPass(DesaturateShader);
    this.glitchPass = new ShaderPass(GlitchShader);
  }

  update(deltaTime) {
    // As sanity decreases, effects increase
    const sanity = this.sanityLevel.get();

    // Desaturation scales with lost sanity
    const targetDesaturate = 1.0 - sanity;
    this.desaturateAmount = this.lerp(this.desaturateAmount, targetDesaturate, 0.01);
    this.desaturatePass.uniforms['amount'].value = this.desaturateAmount;

    // Glitch kicks in at low sanity
    const targetGlitch = Math.max(0, (0.5 - sanity) * 2);
    this.glitchIntensity = this.lerp(this.glitchIntensity, targetGlitch, 0.02);

    if (this.glitchIntensity > 0.1) {
      this.glitchPass.uniforms['amount'].value = this.glitchIntensity;

      // Random glitch bursts at high intensity
      if (this.glitchIntensity > 0.5 && Math.random() < 0.02) {
        this.triggerGlitchBurst();
      }
    }
  }

  lerp(a, b, t) {
    return a + (b - a) * t;
  }

  triggerGlitchBurst() {
    this.glitchPass.uniforms['amount'].value = 1.0;
    setTimeout(() => {
      this.glitchPass.uniforms['amount'].value = this.glitchIntensity;
    }, 100);
  }
}
```

---

## Part 4: Practical Use Cases

### Use Case 1: Safe → Dangerous Transition

```javascript
// Player enters dangerous area
export const areaTransitions = {
  safeZone: {
    // No effects - normal colorful world
  },

  transitionZone: {
    type: "desaturate",
    amount: 0.3,
    duration: 5000,
    fade: true,
    criteria: { currentZone: "transition" }
  },

  dangerZone: {
    type: "composite",
    effects: [
      {
        type: "desaturate",
        amount: 0.8,
        duration: 3000
      },
      {
        type: "vignette",
        amount: 0.5
      }
    ],
    criteria: { currentZone: "danger" }
  }
};
```

### Use Case 2: Memory/Dream Sequence

```javascript
// Flashback or memory reveal
export const memoryEffects = {
  memoryStart: {
    type: "desaturate",
    amount: 0.8,
    duration: 1000,
    fade: true,
    criteria: { currentState: MEMORY_BEGIN }
  },

  memoryActive: {
    type: "composite",
    effects: [
      {
        type: "desaturate",
        amount: 0.9
      },
      {
        type: "bloom",
        strength: 0.3,
        threshold: 0.8,
        radius: 0.3
      }
    ],
    criteria: { currentState: MEMORY_ACTIVE }
  },

  memoryEnd: {
    type: "desaturate",
    amount: 0,
    duration: 1500,
    fade: true,
    criteria: { currentState: MEMORY_END }
  }
};
```

### Use Case 3: Viewmaster Insanity Progression

```javascript
// Viewmaster gets more intense with use
export const viewmasterEffects = {
  // Stage 1: Just equipped
  stage1_subtle: {
    type: "vignette",
    amount: 0.3,
    smoothness: 0.5,
    criteria: {
      isViewmasterEquipped: true,
      viewmasterInsanityIntensity: { $gte: 0.1, $lt: 0.3 }
    }
  },

  // Stage 2: Unsettling
  stage2_unsettling: {
    type: "composite",
    effects: [
      {
        type: "desaturate",
        amount: 0.3,
        duration: 2000
      },
      {
        type: "vignette",
        amount: 0.5
      },
      {
        type: "glitch",
        intensity: 0.2,
        duration: -1,  // Continuous
        frequency: 0.02
      }
    ],
    criteria: {
      isViewmasterEquipped: true,
      viewmasterInsanityIntensity: { $gte: 0.3, $lt: 0.6 }
    }
  },

  // Stage 3: Terrifying
  stage3_nightmare: {
    type: "composite",
    effects: [
      {
        type: "desaturate",
        amount: 0.9,
        duration: 1000
      },
      {
        type: "vignette",
        amount: 0.8
      },
      {
        type: "glitch",
        intensity: 0.8,
        duration: -1,
        frequency: 0.15,
        shiftAmount: 0.04
      }
    ],
    criteria: {
      isViewmasterEquipped: true,
      viewmasterInsanityIntensity: { $gte: 0.8 }
    }
  }
};
```

---

## Common Mistakes Beginners Make

### 1. Over-Using Desaturation

```javascript
// ❌ WRONG: Everything is gray
{
  type: "desaturate",
  amount: 1.0,
  criteria: { currentState: { $gte: 0 } }  // Always on!
}
// Game looks boring and depressing

// ✅ CORRECT: Strategic desaturation
{
  type: "desaturate",
  amount: 0.6,
  criteria: { currentZone: "spooky_area" }  // Specific areas only
}
// Creates contrast when entering
```

### 2. Glitch Too Frequent

```javascript
// ❌ WRONG: Constant glitch
{
  type: "glitch",
  frequency: 0.5  // Glitches half the time!
}
// Annoying, hard to see, causes eye strain

// ✅ CORRECT: Occasional glitch
{
  type: "glitch",
  frequency: 0.05  // Only 5% of frames
}
// More impactful when it happens
```

### 3. No Fade (Abrupt Changes)

```javascript
// ❌ WRONG: Instant change
{
  type: "desaturate",
  amount: 0.8,
  fade: false  // Instant!
}
// Jarring, breaks immersion

// ✅ CORRECT: Smooth transition
{
  type: "desaturate",
  amount: 0.8,
  fade: true,
  duration: 3000  // 3 second fade
}
// Gradual build of tension
```

### 4. Too Many Effects Together

```javascript
// ❌ WRONG: Effect soup
{
  desaturate: { amount: 0.9 },
  glitch: { intensity: 0.7 },
  vignette: { amount: 0.8 },
  bloom: { strength: 1.5 },
  chromatic: { amount: 0.5 }
}
// Can't see anything clearly!

// ✅ CORRECT: Focused effects
{
  desaturate: { amount: 0.5 },  // Muted color
  vignette: { amount: 0.3 }     // Subtle framing
}
// Creates mood without obscuring
```

---

## Performance Considerations

### Shader Complexity

```
DESATURATE SHADER:
├── Complexity: Low
├── Operations per pixel: ~5
├── Cost: Very low
└── Safe for: All devices

GLITCH SHADER:
├── Complexity: Medium-High
├── Operations per pixel: ~20-30
├── Cost: Medium
└── Safe for: Desktop, high-end mobile

COMBINED EFFECTS:
├── Complexity: High
├── Operations per pixel: ~40-60
├── Cost: High
└── Safe for: Desktop only

RECOMMENDATION:
- Mobile: Use desaturate sparingly, avoid glitch
- Desktop: Full range of effects
- Consider performance profiles
```

### Optimization Techniques

```javascript
class EffectOptimizer {
  constructor() {
    this.qualityLevel = 'high';  // high, medium, low
  }

  setQuality(level) {
    this.qualityLevel = level;

    // Adjust effect quality based on level
    switch (level) {
      case 'low':
        // Reduce resolution of effect textures
        this.effectResolution = 0.5;
        // Disable expensive effects
        this.enableGlitch = false;
        break;

      case 'medium':
        this.effectResolution = 0.75;
        this.enableGlitch = true;
        this.glitchQuality = 'reduced';
        break;

      case 'high':
        this.effectResolution = 1.0;
        this.enableGlitch = true;
        this.glitchQuality = 'full';
        break;
    }
  }

  // Skip effect updates on low-end devices
  shouldUpdate(effect) {
    if (this.qualityLevel === 'low') {
      // Update every other frame
      return this.frameCounter % 2 === 0;
    }
    return true;
  }
}
```

---

## 🎮 Game Design Perspective

### Emotional Impact of Color Loss

```
COLOR PSYCHOLOGY:

Full Color:
- Normal, safe, familiar
- Everyday reality
- Comfort, warmth

Muted Colors (30-50% desaturation):
- Unsettling, unfamiliar
- Something is "off"
- Tension, mystery

Grayscale (100% desaturation):
- Death, supernatural
- Memories, past
- Horror, danger
- Clinical, cold
```

### Glitch as Horror Element

```

GLITCH INTENSITY LEVELS:

Subtle (0.1-0.3):
- "Is something wrong with my screen?"
- Uncertainty, confusion
- Good for building tension

Moderate (0.3-0.6):
- Digital corruption spreading
- Reality breaking down
- Supernatural presence

Intense (0.6-0.9):
- Nightmare territory
- Madness, insanity
- Can't trust what you see

Extreme (0.9-1.0):
- Completely overwhelming
- Disorienting, nauseating
- Use sparingly for maximum impact
```

---

## Next Steps

Now that you understand desaturation and glitch effects:

- [VFXManager](./vfx-manager.md) - Base visual effects system
- [Dissolve Effect](./dissolve-effect.md) - Particle-based transitions
- [Selective Bloom](./selective-bloom.md) - Glow effects
- [Game State System](../02-core-architecture/game-state-system.md) - State-driven effects

---

## References

- [Three.js ShaderMaterial](https://threejs.org/docs/#api/en/materials/ShaderMaterial) - Custom shaders
- [GLSL Shader Language](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language) - GPU programming
- [Post-Processing](https://threejs.org/docs/#examples/en/postprocessing/EffectComposer) - Effect composition
- [Color Theory](https://www.canva.com/colors/color-wheel/) - Understanding color psychology

*Documentation last updated: January 12, 2026*
