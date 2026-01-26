# Interactive Object: Viewmaster Controller

## Overview

The **Viewmaster** is a psychological horror mechanic that simulates the classic toy stereo viewer, progressively distorting the player's vision based on how long they stare through it. As the "insanity intensity" increases, visual effects intensify - creating a direct connection between player action (staring) and consequence (psychological distress).

Think of the viewmaster as the **"sanity meter"** made visible - just as prolonged exposure to disturbing content affects mental state, staring through the viewmaster makes the game world progressively more disturbed.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create unease through voluntary self-harm. The player chooses to look through the viewmaster, but prolonged viewing has consequences.

**Why a Viewmaster?**
- **Nostalgia Subversion**: Viewmasters are associated with childhood wonder - twisting this creates unease
- **Frame Within Frame**: Looking at the world through a restricted view creates vulnerability
- **Voluntary Exposure**: Player CHOOSES to look - complicity in the horror
- **Progressive Horror**: Effects build gradually, not immediately - tension through anticipation

**Player Psychology**:
```
Curiosity → Look Through Viewmaster → Normal Image
     ↓
Keep Looking → Subtle Distortion Begins → Notice Something Wrong
     ↓
Continue Staring → Distortion Intensifies → Unease Builds
     ↓
Prolonged Stare → Maximum Insanity → Can't Unsee
     ↓
Look Away → Lingering Effects → World Feels Wrong Now
```

### Design Decisions

**1. Progressive Insanity System**
Rather than binary (looking/not looking), the viewmaster uses a continuous intensity value (0-1) that increases with stare duration. This creates:
- Player agency (how long to stare)
- Clear feedback (effects intensify)
- Consequences (lingering distortion even after looking away)

**2. Visual Distortion Hierarchy**
Effects unlock at different intensity thresholds:
```
Intensity 0.0-0.3: Subtle chromatic aberration
Intensity 0.3-0.5: Color desaturation begins
Intensity 0.5-0.7: Glitch effects start
Intensity 0.7-0.9: Heavy distortion, screen effects
Intensity 0.9-1.0: Maximum insanity, reality breakdown
```

**3. Gaze-Triggered (Not Click-Triggered)**
Using head pose / gaze detection rather than input button creates:
- More immersive interaction (natural "looking" vs artificial "clicking")
- Fairer consequence (can't blame "misclick" for horror)
- Spatial awareness (must position correctly to view)

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the viewmaster implementation, you should know:
- **Head pose tracking** - Detecting where player is looking (desktop mouse vs mobile device orientation)
- **Raycasting** - Determining what object is being looked at
- **Post-processing effects** - Glitch, chromatic aberration, desaturation
- **Timer-based state** - Tracking duration of continuous action
- **Easing functions** - Smooth transitions between effect intensities

### Core Architecture

```
VIEWMASTER SYSTEM ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                   VIEWMASTER CONTROLLER                  │
│  - Tracks gaze duration                                 │
│  - Calculates insanity intensity                        │
│  - Triggers visual effects at thresholds                │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  GAZE TRACKER │  │  INSANITY     │  │   VFX        │
│  - Head pose  │  │  CALCULATOR   │  │   MANAGER    │
│  - Raycast    │  │  - Duration   │  │  - Glitch    │
│  - On/Off     │  │  - Intensity  │  │  - Desaturate│
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  POST-PROCESS │
                    │  - Apply      │
                    │    effects    │
                    │    based on   │
                    │    intensity  │
                    └───────────────┘
```

### Gaze Detection

The viewmaster detects when the player is looking through it:

```javascript
class ViewmasterGazeTracker {
  constructor(camera, scene) {
    this.camera = camera;
    this.scene = scene;
    this.isLooking = false;
    this.gazeStartTime = 0;
    this.totalGazeTime = 0;

    // Raycaster for gaze detection
    this.raycaster = new THREE.Raycaster();
    this.gazeDirection = new THREE.Vector3();
    this.gazeOrigin = new THREE.Vector3();
  }

  update() {
    // Get forward direction from camera
    this.camera.getWorldDirection(this.gazeDirection);
    this.gazeOrigin.copy(this.camera.position);

    // Cast ray
    this.raycaster.set(this.gazeOrigin, this.gazeDirection);

    // Check for viewmaster intersection
    const viewmaster = this.scene.getObjectByName("viewmaster");
    if (!viewmaster) {
      this.setLooking(false);
      return;
    }

    const intersects = this.raycaster.intersectObject(viewmaster, true);

    if (intersects.length > 0) {
      const hit = intersects[0];

      // Check if close enough and looking through lenses
      if (hit.distance < 2.0 && this.isLookingAtLenses(hit)) {
        this.setLooking(true);
        return;
      }
    }

    this.setLooking(false);
  }

  isLookingAtLenses(hit) {
    // Check if hit point is within lens area
    const localPoint = hit.object.worldToLocal(hit.point.clone());
    return Math.abs(localPoint.x) < 0.05 && Math.abs(localPoint.y) < 0.05;
  }

  setLooking(looking) {
    if (looking && !this.isLooking) {
      // Just started looking
      this.gazeStartTime = performance.now();
    } else if (!looking && this.isLooking) {
      // Stopped looking - accumulate time
      this.totalGazeTime += performance.now() - this.gazeStartTime;
    }

    this.isLooking = looking;
  }

  getCurrentGazeDuration() {
    if (!this.isLooking) return 0;
    return performance.now() - this.gazeStartTime;
  }

  getTotalGazeTime() {
    if (this.isLooking) {
      return this.totalGazeTime + (performance.now() - this.gazeStartTime);
    }
    return this.totalGazeTime;
  }
}
```

### Insanity Calculator

Converts gaze duration to insanity intensity:

```javascript
class ViewmasterInsanityCalculator {
  constructor(config = {}) {
    // Time to reach maximum insanity (milliseconds)
    this.maxInsanityTime = config.maxInsanityTime || 30000;  // 30 seconds

    // Current insanity state
    this.intensity = 0;  // 0-1
    this.lingerIntensity = 0;  // Lingering effects after looking away
    this.lingerDecay = 0.999;  // How fast lingering effects fade
  }

  update(gazeDuration, isLooking) {
    // Calculate target intensity based on gaze
    const targetIntensity = Math.min(1, gazeDuration / this.maxInsanityTime);

    if (isLooking) {
      // Smoothly increase intensity
      this.intensity = THREE.MathUtils.lerp(
        this.intensity,
        targetIntensity,
        0.01  // Lerp factor for smooth increase
      );

      // Update lingering intensity to match
      this.lingerIntensity = this.intensity;
    } else {
      // Not looking - intensity stays, but lingering effects decay
      this.intensity = targetIntensity;  // Don't decrease base intensity
      this.lingerIntensity *= this.lingerDecay;  // Decay lingering
    }
  }

  getInsanityIntensity() {
    return this.intensity;
  }

  getLingerIntensity() {
    return this.lingerIntensity;
  }

  reset() {
    this.intensity = 0;
    this.lingerIntensity = 0;
  }
}
```

### Visual Effect Application

Applies post-processing effects based on insanity intensity:

```javascript
class ViewmasterVFXController {
  constructor(vfxManager) {
    this.vfxManager = vfxManager;
    this.currentEffects = [];
  }

  update(insanityIntensity, lingerIntensity) {
    // Effective intensity combines current and lingering
    const effectiveIntensity = Math.max(insanityIntensity, lingerIntensity);

    // Apply effects based on intensity thresholds
    this.applyChromaticAberration(effectiveIntensity);
    this.applyDesaturation(effectiveIntensity);
    this.applyGlitch(effectiveIntensity);
    this.applyVignette(effectiveIntensity);
  }

  applyChromaticAberration(intensity) {
    if (intensity < 0.2) return;  // Threshold

    const amount = (intensity - 0.2) * 0.5;  // Scale to 0-0.4

    this.vfxManager.setEffectParams("chromatic", {
      amount: amount
    });
  }

  applyDesaturation(intensity) {
    if (intensity < 0.4) return;

    const amount = (intensity - 0.4) * 1.25;  // Scale to 0-0.75

    this.vfxManager.setEffectParams("desaturate", {
      amount: amount
    });
  }

  applyGlitch(intensity) {
    if (intensity < 0.5) return;

    const frequency = (intensity - 0.5) * 2;  // More frequent at higher intensity
    const amount = intensity * 0.8;

    this.vfxManager.setEffectParams("glitch", {
      amount: amount,
      frequency: frequency
    });
  }

  applyVignette(intensity) {
    // Always apply some vignette, increasing with intensity
    const amount = 0.2 + (intensity * 0.6);  // 0.2-0.8 range

    this.vfxManager.setEffectParams("vignette", {
      amount: amount,
      color: new THREE.Color(0x000000).lerp(
        new THREE.Color(0xff0000),
        intensity * 0.5  // Red tint at high intensity
      )
    });
  }
}
```

---

## 📝 How To Build A Mechanic Like This

### Step 1: Define the Progressive Mechanic

What are the stages of your effect?

```javascript
// Example: Sanity-based horror
const sanityStages = {
  calm: {
    threshold: 0.0,
    effects: []
  },
  uneasy: {
    threshold: 0.3,
    effects: ["subtle_flicker", "quiet_sounds"]
  },
  disturbed: {
    threshold: 0.6,
    effects: ["desaturate", "whisper_audio", "shadow_movement"]
  },
  panicked: {
    threshold: 0.8,
    effects: ["glitch", "heartbeat_audio", "tunnel_vision"]
  },
  broken: {
    threshold: 1.0,
    effects: ["reality_break", "scream_audio", "full_psychosis"]
  }
};
```

### Step 2: Choose Your Trigger Method

```javascript
// Options for triggering insanity buildup:

// 1. Gaze-based (viewmaster style)
const gazeTrigger = {
  type: "gaze",
  target: "object_name",
  distance: 2.0,
  buildRate: 1.0 / 30000  // Per-millisecond
};

// 2. Proximity-based (being near something disturbing)
const proximityTrigger = {
  type: "proximity",
  target: "ghost_entity",
  radius: 5.0,
  buildRate: 1.0 / 10000  // Faster buildup
};

// 3. Action-based (doing something wrong)
const actionTrigger = {
  type: "action",
  action: "look_at_forbidden",
  buildRate: 1.0 / 5000  // Very fast
};
```

### Step 3: Create the Intensity Calculator

```javascript
class ProgressiveIntensityCalculator {
  constructor(config) {
    this.trigger = config.trigger;
    this.maxIntensity = config.maxIntensity || 1.0;
    this.buildRate = config.buildRate;
    this.decayRate = config.decayRate || 0;

    this.currentIntensity = 0;
    this.isTriggering = false;
  }

  update(deltaTime, isTriggering) {
    this.isTriggering = isTriggering;

    if (isTriggering) {
      // Build intensity
      this.currentIntensity = Math.min(
        this.maxIntensity,
        this.currentIntensity + (this.buildRate * deltaTime)
      );
    } else if (this.decayRate > 0) {
      // Decay intensity
      this.currentIntensity = Math.max(
        0,
        this.currentIntensity - (this.decayRate * deltaTime)
      );
    }
  }

  getIntensity() {
    return this.currentIntensity;
  }
}
```

### Step 4: Apply Effects at Thresholds

```javascript
class ThresholdEffectManager {
  constructor(stages) {
    this.stages = stages;
    this.activeStage = null;
  }

  update(intensity) {
    // Find current stage
    let currentStage = "calm";
    for (const [stageName, stage] of Object.entries(this.stages)) {
      if (intensity >= stage.threshold) {
        currentStage = stageName;
      }
    }

    // Check if stage changed
    if (currentStage !== this.activeStage) {
      this.onStageChanged(currentStage);
      this.activeStage = currentStage;
    }
  }

  onStageChanged(stageName) {
    const stage = this.stages[stageName];

    // Enter new stage
    for (const effect of stage.effects) {
      this.enableEffect(effect);
    }

    // Exit old stage effects
    for (const [name, stage] of Object.entries(this.stages)) {
      if (name !== stageName) {
        for (const effect of stage.effects) {
          if (!stage.effects.includes(effect)) {
            this.disableEffect(effect);
          }
        }
      }
    }
  }
}
```

---

## 🔧 Variations For Your Game

### Sanity Meter (Amnesia-Style)

```javascript
class SanityMeter {
  // Global sanity that decreases with disturbing events
  // Affects all visuals, not just when viewing something

  config = {
    trigger: "event_based",  // Jump scares, disturbing sights
    decay: false,  // Sanity doesn't recover
    effects: {
      low: ["subtle_flicker"],
      medium: ["hallucinations_start"],
      critical: ["reality_breakdown"]
    }
  };
}
```

### Fear System (Alien: Isolation-Style)

```javascript
class FearSystem {
  // Dynamic fear that builds and recovers
  // Affects player behavior (shaking, breathing)

  config = {
    trigger: "proximity_to_threat",
    decay: true,  // Fear fades when safe
    effects: {
      low: ["heavy_breathing"],
      medium: ["hands_shake", "heart_pounding"],
      high: ["panic_movement", "tunnel_vision"]
    }
  };
}
```

### Corruption System (Progressive Mutation)

```javascript
class CorruptionSystem {
  // Permanent character transformation
  // Visual changes that persist and compound

  config = {
    trigger: "exposure_to_corruption",
    decay: false,  // Changes are permanent
    effects: {
      stage1: ["skin_discoloration"],
      stage2: ["voice_change", "movement_change"],
      stage3: ["full_transformation"]
    }
  };
}
```

---

## Common Mistakes Beginners Make

### 1. Insanity Builds Too Fast

```javascript
// ❌ WRONG: Reaches maximum too quickly
{ maxInsanityTime: 5000 }  // 5 seconds
// Player never experiences the buildup

// ✅ CORRECT: Slow, uncomfortable progression
{ maxInsanityTime: 30000 }  // 30 seconds
// Player has time to consider stopping, doesn't
```

### 2. No Lingering Effects

```javascript
// ❌ WRONG: Effects stop immediately when looking away
if (!isLooking) intensity = 0;
// No consequence for staring

// ✅ CORRECT: Lingering effects create unease
if (!isLooking) {
  lingerIntensity *= 0.999;  // Slow decay
}
// Player sees residual distortion even after stopping
```

### 3. Binary Intensity

```javascript
// ❌ WRONG: All-or-nothing effects
if (intensity > 0.5) {
  applyAllEffects();
}
// Jarring transition, no progression

// ✅ CORRECT: Gradual intensity
chromaticAmount = intensity * 0.5;
desaturateAmount = intensity > 0.4 ? (intensity - 0.4) : 0;
// Each effect scales independently with thresholds
```

---

## Performance Considerations

```
VIEWMASTER PERFORMANCE:

Gaze Tracking:
├── Raycast: One per frame (minimal cost)
├── Distance check: Negligible
└── Impact: Negligible

Insanity Calculation:
├── Timer updates: Negligible
├── Intensity lerp: Minimal
└── Impact: Negligible

Visual Effects:
├── Chromatic aberration: Low cost
├── Desaturation: Low cost
├── Glitch: Medium cost (shader-based)
└── Vignette: Low cost

Recommendations:
- Use shader-based effects (not post-processing passes)
- Combine effects into single pass when possible
- Reduce effect quality on mobile
```

---

## Related Systems

- [Head Pose Animation](../06-animation/head-pose-animation.md) - Gaze tracking
- [Glitch Post-Processing](../07-visual-effects/glitch-post-processing.md) - Visual distortion effects
- [VFXManager](../07-visual-effects/vfx-manager.md) - Effect orchestration
- [GameState System](../02-core-architecture/game-state-system.md) - State tracking

---

## References

- [Three.js Raycaster](https://threejs.org/docs/#api/en/core/Raycaster) - Gaze detection
- [Post-Processing Effects](https://threejs.org/docs/#examples/en/postprocessing/EffectComposer) - Visual effects
- [Psychological Horror Design](https://www.gamedeveloper.com/design/psychological-horror-in-games) - Design principles
- [Sanity Systems in Games](https://tvtropes.org/pmwiki/pmwiki.php/Main/SanityMeter) - Trope analysis

*Documentation last updated: January 12, 2026*
