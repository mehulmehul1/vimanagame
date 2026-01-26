# AnimationManager - First Principles Guide

## Overview

The **AnimationManager** handles camera and object animations in the Shadow Engine. It manages timed sequences of property changes - moving cameras through cinematic paths, rotating objects, scaling elements, and creating smooth visual transitions.

Think of AnimationManager as the **director of cinematography** for your game - just as a film director plans camera moves and object choreography, AnimationManager orchestrates how things move and change over time in your 3D world.

## What You Need to Know First

Before understanding AnimationManager, you should know:
- **Keyframes** - Values at specific points in time
- **Interpolation** - Calculating values between keyframes
- **Easing functions** - How values accelerate/decelerate
- **Three.js Object3D** - Position, rotation, scale properties
- **Time-based animation** - Updates per frame based on elapsed time

### Quick Refresher: Keyframe Animation

```
KEYFRAME ANIMATION: Values defined at specific times

Time:  0s      2s      4s      6s
       │       │       │       │

X Position:
       ┌───────┬───────┬───────┐
Value: │  0    │  5    │  2    │  (Keyframes defined)
       └───────┴───────┴───────┘
         │       │       │
         ▼       ▼       ▼
       KEY1    KEY2    KEY3

INTERPOLATION: Calculate values BETWEEN keyframes

Time:  0s    1s    2s    3s    4s
       │     │     │     │     │
       ●─────●─────●─────●─────●

Value: 0 → 2.5 → 5 → 3.5 → 2

       The AnimationManager calculates
       the intermediate values each frame!
```

**Why keyframes matter:** Instead of manually updating positions every frame, you define just a few key points and let the animation system calculate everything in between.

---

## Part 1: Why Use an Animation System?

### The Problem: Manual Animation is Tedious

Without an animation system:

```javascript
// ❌ WITHOUT AnimationManager - Manual updates
let time = 0;
function manualAnimation() {
  time += 1/60;  // 60 FPS

  // Manually calculate position
  camera.position.x = Math.sin(time * 0.5) * 5;
  camera.position.y = 1.6;
  camera.position.z = Math.cos(time * 0.5) * 5;

  // Manual rotation calculation
  camera.rotation.y = time * 0.2;

  // This gets complicated for complex animations!
}
```

**Problems:**
- Code is hard to read and modify
- Can't easily sequence animations
- Difficult to coordinate multiple objects
- No easing (everything looks robotic)
- Hard to design collaboratively

### The Solution: AnimationManager

```javascript
// ✅ WITH AnimationManager - Data-driven
const cameraAnimation = {
  type: "camera",
  duration: 8000,
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } },
    { time: 0.3, position: { x: 2, y: 1.6, z: 3 }, target: { x: 0, y: 1, z: 0 } },
    { time: 0.6, position: { x: -2, y: 1.6, z: 3 }, target: { x: 0, y: 1, z: 0 } },
    { time: 1.0, position: { x: 0, y: 1.6, z: 0 }, target: { x: 0, y: 1.5, z: -5 } }
  ],
  easing: "easeInOutCubic"
};

animationManager.play(cameraAnimation);
```

**Benefits:**
- Clear, readable animation definitions
- Easy sequencing with `playNext`
- Smooth easing functions built-in
- Data-driven (designers can edit)
- Reusable animations

---

## Part 2: Animation Data Structure

### Basic Animation Definition

```javascript
// In animationData.js
export const animations = {
  // Camera animation
  introCameraSweep: {
    type: "camera",           // What to animate
    duration: 8000,           // Total duration (ms)
    keyframes: [              // Values at specific times
      { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } },
      { time: 0.5, position: { x: 2, y: 1.6, z: 2 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1.0, position: { x: 0, y: 1.6, z: 0 }, target: { x: 0, y: 1.5, z: -5 } }
    ],
    easing: "easeInOutCubic", // How values change
    criteria: { currentState: INTRO }  // When to play
  },

  // Object animation
  phoneRinging: {
    type: "object",           // Animate an object
    target: "phoneBoothPhone",// Object ID to animate
    duration: 500,
    loop: true,               // Repeat animation
    keyframes: [
      { time: 0, rotation: { x: 0, y: 0, z: 0 } },
      { time: 0.25, rotation: { x: 0.05, y: 0, z: 0 } },
      { time: 0.5, rotation: { x: -0.05, y: 0, z: 0 } },
      { time: 0.75, rotation: { x: 0.02, y: 0, z: 0 } },
      { time: 1.0, rotation: { x: 0, y: 0, z: 0 } }
    ],
    easing: "easeInOutSine",
    criteria: { currentState: PHONE_RINGING }
  }
};
```

### Animation Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `type` | string | Yes | "camera" or "object" |
| `target` | string | Conditional | Object ID (for type="object") |
| `duration` | number | Yes | Total duration in milliseconds |
| `keyframes` | array | Yes | Array of keyframe objects |
| `easing` | string | No | Easing function (default: "linear") |
| `loop` | boolean | No | Whether to loop (default: false) |
| `playNext` | string | No | Animation to play after completion |
| `criteria` | object | Yes | When animation triggers |

### Keyframe Properties

| Property | Type | Description |
|----------|------|-------------|
| `time` | number | Time position (0.0 to 1.0, relative to duration) |
| `position` | object | { x, y, z } position |
| `rotation` | object | { x, y, z } Euler rotation (radians) |
| `scale` | object | { x, y, z } scale |
| `target` | object | { x, y, z } look-at target (camera only) |
| `fov` | number | Camera field of view (camera only) |

---

## Part 3: How AnimationManager Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AnimationManager                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Active Animations                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ Camera 1 │  │ Camera 2 │  │ Object 1 │  │ Object 2 │   │   │
│  │  │ Playing  │  │ Queued   │  │ Looping  │  │ Finished │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               Interpolation Engine                           │   │
│  │  - Calculates values between keyframes                      │   │
│  │  - Applies easing functions                                 │   │
│  │  - Updates camera/object properties each frame              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Animation Queue                               │   │
│  │  - Sequences animations with playNext                       │   │
│  │  - Manages transitions                                      │   │
│  │  - Handles looping                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                            │
         │ listens to                 │ updates
         ▼                            ▼
┌─────────────────┐          ┌─────────────────┐
│  GameManager    │          │   Three.js      │
│  "state:changed"│          │   Camera/Objects │
└─────────────────┘          └─────────────────┘
```

### Animation Playback Flow

```
1. State changes or animation explicitly triggered
              │
              ▼
2. AnimationManager receives request
              │
              ▼
3. Look up animation in animationData
              │
              ▼
4. Parse keyframes and validate
              │
              ▼
5. Create animation instance:
    │
    ├──▶ Store start time
    │
    ├──▶ Calculate keyframe timings
    │
    ├──▶ Set up easing function
    │
    └──▶ Register for updates
              │
              ▼
6. Each frame (update loop):
    │
    ├──▶ Calculate elapsed time
    │
    ├──▶ Find surrounding keyframes
    │
    ├──▶ Apply easing
    │
    ├──▶ Interpolate values
    │
    └──▶ Apply to camera/object
              │
              ▼
7. On completion:
    │
    ├──▶ If looping: return to start
    │
    ├──▶ If playNext: start next animation
    │
    └──▶ Otherwise: mark complete, cleanup
```

---

## Part 4: Interpolation and Easing

### Linear Interpolation (Lerp)

The simplest form of interpolation - values change at a constant rate:

```javascript
function lerp(a, b, t) {
  // t is between 0 and 1
  return a + (b - a) * t;
}

// Example:
// lerp(0, 10, 0.5) = 5     (halfway between 0 and 10)
// lerp(0, 10, 0.25) = 2.5  (quarter of the way)
// lerp(0, 10, 1.0) = 10    (at the end)
```

**Visual representation:**
```
Linear (no easing):
Value │
     10│                    ╱
      │                  ╱
      │                ╱
      │              ╱
      │            ╱
      │          ╱
      │        ╱
      │      ╱
      │    ╱
      │  ╱
      │╱
      0└──────────────────────▶ Time
       0                   1

Movement looks robotic and artificial!
```

### Easing Functions

Easing makes animations feel natural by varying the rate of change:

```javascript
// Common easing functions
const Easing = {
  linear: (t) => t,

  // Quadratic
  easeInQuad: (t) => t * t,
  easeOutQuad: (t) => t * (2 - t),
  easeInOutQuad: (t) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,

  // Cubic
  easeInCubic: (t) => t * t * t,
  easeOutCubic: (t) => (--t) * t * t + 1,
  easeInOutCubic: (t) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,

  // Sine
  easeInSine: (t) => 1 - Math.cos(t * Math.PI / 2),
  easeOutSine: (t) => Math.sin(t * Math.PI / 2),
  easeInOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,

  // Back (overshoots slightly)
  easeOutBack: (t) => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  }
};
```

**Visual comparison:**
```
          EASE OUT              LINEAR              EASE IN
          (starts fast)         (constant)          (starts slow)
Value 10 │╱                                          ┌────
        │                                          ┌─┘
        │                                       ┌──┘
        │                                    ┌───┘
        │                                 ┌────┘
        │                              ┌─────┘
        │                           ┌──────┘
        │                        ┌───────┘
        │                     ┌────────┘
        │                  ┌─────────┘
        │               ┌──────────┘
        │            ┌────────────┘
        │         ┌───────────────┘
        │      ╌──────────────────┘
        │   ╌─────────────────────┘
        0 ┌────────────────────────────────▶ Time

Use Ease Out for:     Use Linear for:      Use Ease In for:
- Camera movements    - Mechanical things   - Building tension
- UI transitions      - Constant speed      - Slow starts
- Object appearance   - Rotating fans       - Heavy objects
```

### Applying Easing in AnimationManager

```javascript
class AnimationManager {
  update(deltaTime) {
    this.activeAnimations.forEach(anim => {
      // Calculate progress (0 to 1)
      const elapsed = performance.now() - anim.startTime;
      const rawProgress = elapsed / anim.duration;

      // Apply easing
      const easedProgress = this.easingFunctions[anim.easing](rawProgress);

      // Apply to properties
      this.applyKeyframes(anim, easedProgress);
    });
  }

  applyKeyframes(animation, progress) {
    const keyframes = animation.keyframes;

    // Find the two keyframes surrounding current progress
    let startKeyframe = keyframes[0];
    let endKeyframe = keyframes[keyframes.length - 1];

    for (let i = 0; i < keyframes.length - 1; i++) {
      if (progress >= keyframes[i].time && progress <= keyframes[i + 1].time) {
        startKeyframe = keyframes[i];
        endKeyframe = keyframes[i + 1];
        break;
      }
    }

    // Calculate local progress between these two keyframes
    const localDuration = endKeyframe.time - startKeyframe.time;
    const localProgress = (progress - startKeyframe.time) / localDuration;

    // Interpolate each property
    const current = {};

    if (startKeyframe.position && endKeyframe.position) {
      current.position = {
        x: this.lerp(startKeyframe.position.x, endKeyframe.position.x, localProgress),
        y: this.lerp(startKeyframe.position.y, endKeyframe.position.y, localProgress),
        z: this.lerp(startKeyframe.position.z, endKeyframe.position.z, localProgress)
      };
    }

    // ... same for rotation, scale, etc.

    // Apply to target
    this.applyToTarget(animation, current);
  }

  lerp(a, b, t) {
    return a + (b - a) * t;
  }
}
```

---

## Part 5: Camera Animations

### Camera Movement Types

#### 1. Dolly/Truck (Position Movement)

```javascript
cameraDollyIn: {
  type: "camera",
  duration: 4000,
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 10 }, target: { x: 0, y: 1, z: 0 } },
    { time: 1, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } },
    { time: 1, position: { x: 0, y: 1.6, z: 2 }, target: { x: 0, y: 1, z: 0 } }
  ],
  easing: "easeOutCubic",
  criteria: { currentState: APPROACH_DIALOG }
}
```

**Visual:**
```
Side view of camera movement:

      Z-axis
       ↓
    10  [Cam] ──────────────→ Moving closer
     5       [Cam] ──────────→
     2           [Cam] ──────→
     0           ● Target

    Camera moves closer while keeping focus on target
```

#### 2. Orbit (Around Target)

```javascript
cameraOrbit: {
  type: "camera",
  duration: 6000,
  keyframes: [
    { time: 0, position: { x: 5, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } },
    { time: 0.25, position: { x: 0, y: 1.6, z: -5 }, target: { x: 0, y: 1, z: 0 } },
    { time: 0.5, position: { x: -5, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } },
    { time: 0.75, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } },
    { time: 1, position: { x: 5, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } }
  ],
  easing: "linear",  // Constant speed for orbit
  criteria: { currentState: INSPECT_OBJECT }
}
```

**Visual:**
```
Top-down view:

          Z
          ↑
          │
    X ←───┼── [Cam]
          │
      ╱   │   ╲
     ╱    │    ╲
    ╱     ●     ╲  ← Target (center)
     ╲    │    ╱
      ╲   │   ╱
          │

    Camera circles around the target
```

#### 3. Crane (Vertical Movement)

```javascript
cameraCraneUp: {
  type: "camera",
  duration: 3000,
  keyframes: [
    { time: 0, position: { x: 0, y: 0.5, z: 5 }, target: { x: 0, y: 1, z: 0 } },
    { time: 0.5, position: { x: 0, y: 3, z: 5 }, target: { x: 0, y: 1, z: 0 } },
    { time: 1, position: { x: 0, y: 6, z: 5 }, target: { x: 0, y: 1, z: 0 } }
  ],
  easing: "easeInOutCubic",
  criteria: { currentState: REVEAL_LANDSCAPE }
}
```

**Visual:**
```
Side view:

Y-axis ↑
     6  [Cam]  ← Bird's eye view
     3  [Cam]  ← Mid shot
     0.5[Cam]  ← Ground level
       ─────────
       ● Target

    Camera rises vertically, changing perspective
```

#### 4. FOV Change (Zoom Effect)

```javascript
cameraZoomIn: {
  type: "camera",
  duration: 2000,
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 }, fov: 75 },
    { time: 1, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 }, fov: 30 }
  ],
  easing: "easeInOutCubic",
  criteria: { currentState: DRAMATIC_REVEAL }
}
```

---

## Part 6: Object Animations

### Position Animation

```javascript
objectMove: {
  type: "object",
  target: "floatingOrb",
  duration: 3000,
  keyframes: [
    { time: 0, position: { x: 0, y: 1, z: 0 } },
    { time: 0.5, position: { x: 2, y: 1.5, z: -1 } },
    { time: 1, position: { x: 0, y: 2, z: -2 } }
  ],
  easing: "easeInOutSine",
  criteria: { currentState: ORB_FLOATS }
}
```

### Rotation Animation

```javascript
objectSpin: {
  type: "object",
  target: "propeller",
  duration: 1000,
  loop: true,
  keyframes: [
    { time: 0, rotation: { x: 0, y: 0, z: 0 } },
    { time: 1, rotation: { x: 0, y: Math.PI * 2, z: 0 } }  // Full 360° rotation
  ],
  easing: "linear",  // Constant speed for spinning
  criteria: { currentState: PROPELLER_ACTIVE }
}
```

### Scale Animation

```javascript
objectPulse: {
  type: "object",
  target: "heartbeatOrb",
  duration: 800,
  loop: true,
  keyframes: [
    { time: 0, scale: { x: 1, y: 1, z: 1 } },
    { time: 0.5, scale: { x: 1.2, y: 1.2, z: 1.2 } },
    { time: 1, scale: { x: 1, y: 1, z: 1 } }
  ],
  easing: "easeInOutSine",
  criteria: { sanityLevel: { $lt: 0.3 } }
}
```

### Combined Transform Animation

```javascript
complexTransform: {
  type: "object",
  target: "magicCube",
  duration: 4000,
  keyframes: [
    {
      time: 0,
      position: { x: -2, y: 1, z: 0 },
      rotation: { x: 0, y: 0, z: 0 },
      scale: { x: 0.5, y: 0.5, z: 0.5 }
    },
    {
      time: 0.5,
      position: { x: 0, y: 2, z: -2 },
      rotation: { x: Math.PI, y: Math.PI, z: 0 },
      scale: { x: 1, y: 1, z: 1 }
    },
    {
      time: 1,
      position: { x: 2, y: 1, z: 0 },
      rotation: { x: 0, y: Math.PI * 2, z: 0 },
      scale: { x: 0.5, y: 0.5, z: 0.5 }
    }
  ],
  easing: "easeInOutCubic",
  criteria: { currentState: CUBE_TRANSFORMATION }
}
```

---

## Part 7: Animation Chaining with playNext

### Sequential Animations

```javascript
export const animationSequences = {
  // Part 1: Slow approach
  intro_Part1: {
    type: "camera",
    duration: 5000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 20 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 0, y: 1.6, z: 10 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeOutCubic",
    playNext: "intro_Part2",  // Automatically continues!
    criteria: { currentState: INTRO_START }
  },

  // Part 2: Circle around
  intro_Part2: {
    type: "camera",
    duration: 4000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 10 }, target: { x: 0, y: 1, z: 0 } },
      { time: 0.5, position: { x: 7, y: 1.6, z: 7 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 10, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeInOutSine",
    playNext: "intro_Part3"
  },

  // Part 3: Final approach
  intro_Part3: {
    type: "camera",
    duration: 3000,
    keyframes: [
      { time: 0, position: { x: 10, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1, position: { x: 3, y: 1.6, z: 0 }, target: { x: 0, y: 1, z: 0 } }
    ],
    easing: "easeInCubic"
    // No playNext - sequence ends
  }
};
```

### Conditional Chaining

```javascript
// Branching animation paths
branchingIntro: {
  type: "camera",
  duration: 3000,
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 10 }, target: { x: 0, y: 1, z: 0 } },
    { time: 1, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } }
  ],
  easing: "easeOutCubic",
  // playNext is determined dynamically based on player choice
  playNext: null,  // Set programmatically
  criteria: { currentState: INTRO_CROSSROAD }
},

// Path A: Go left
pathA_Camera: {
  type: "camera",
  duration: 4000,
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: -5, y: 1, z: 0 } },
    { time: 1, position: { x: -3, y: 1.6, z: 3 }, target: { x: -5, y: 1, z: 0 } }
  ],
  easing: "easeInOutCubic",
  criteria: { playerChoice: "left" }
},

// Path B: Go right
pathB_Camera: {
  type: "camera",
  duration: 4000,
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: 5, y: 1, z: 0 } },
    { time: 1, position: { x: 3, y: 1.6, z: 3 }, target: { x: 5, y: 1, z: 0 } }
  ],
  easing: "easeInOutCubic",
  criteria: { playerChoice: "right" }
}
```

---

## Part 8: Common Animation Use Cases

### 1. Intro Cinematic

```javascript
introCinematic: {
  type: "camera",
  duration: 12000,
  keyframes: [
    { time: 0, position: { x: 0, y: 5, z: 15 }, target: { x: 0, y: 0, z: 0 } },
    { time: 0.3, position: { x: 0, y: 3, z: 10 }, target: { x: 0, y: 0, z: 0 } },
    { time: 0.5, position: { x: 5, y: 2, z: 5 }, target: { x: 0, y: 1, z: 0 } },
    { time: 0.7, position: { x: -3, y: 1.8, z: 3 }, target: { x: 0, y: 1, z: 0 } },
    { time: 1.0, position: { x: 0, y: 1.6, z: 0 }, target: { x: 0, y: 1.6, z: -1 } }
  ],
  easing: "easeInOutCubic",
  criteria: { currentState: GAME_START }
}
```

### 2. Object Reveals

```javascript
dramaticReveal: {
  type: "camera",
  duration: 3000,
  keyframes: [
    { time: 0, position: { x: -5, y: 2, z: 5 }, target: { x: 0, y: 1, z: 0 }, fov: 75 },
    { time: 0.5, position: { x: -2, y: 1.8, z: 3 }, target: { x: 0, y: 1, z: 0 }, fov: 60 },
    { time: 1, position: { x: -1, y: 1.7, z: 2 }, target: { x: 0, y: 1, z: 0 }, fov: 45 }
  ],
  easing: "easeInOutCubic",
  criteria: { currentState: DRAMATIC_REVEAL }
}
```

### 3. Idle Animations

```javascript
idleHover: {
  type: "object",
  target: "floatingCrystal",
  duration: 2000,
  loop: true,
  keyframes: [
    { time: 0, position: { x: 0, y: 1, z: 0 }, rotation: { x: 0, y: 0, z: 0 } },
    { time: 0.5, position: { x: 0, y: 1.2, z: 0 }, rotation: { x: 0, y: 0.1, z: 0 } },
    { time: 1, position: { x: 0, y: 1, z: 0 }, rotation: { x: 0, y: 0, z: 0 } }
  ],
  easing: "easeInOutSine",
  criteria: { currentState: IDLE }
}
```

### 4. Impact/Shake Effects

```javascript
cameraShake: {
  type: "camera",
  duration: 500,
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 0 } },
    { time: 0.1, position: { x: 0.1, y: 1.6, z: 0 } },
    { time: 0.2, position: { x: -0.15, y: 1.55, z: 0.05 } },
    { time: 0.3, position: { x: 0.1, y: 1.65, z: -0.05 } },
    { time: 0.4, position: { x: -0.05, y: 1.6, z: 0.02 } },
    { time: 0.5, position: { x: 0, y: 1.6, z: 0 } }
  ],
  easing: "linear",  // Fast, erratic movement
  criteria: { event: "explosion" }
}
```

### 5. Door Opening Animation

```javascript
doorOpen: {
  type: "object",
  target: "officeDoor",
  duration: 1000,
  keyframes: [
    { time: 0, rotation: { x: 0, y: 0, z: 0 } },
    { time: 1, rotation: { x: 0, y: -Math.PI / 2, z: 0 } }  // Swing 90°
  ],
  easing: "easeInOutCubic",
  criteria: { interaction: "door_open" }
}
```

---

## Common Mistakes Beginners Make

### 1. No Easing (Robotic Movement)

```javascript
// ❌ WRONG: Everything moves at constant speed
easing: "linear"  // Used for everything!

// ✅ CORRECT: Choose appropriate easing
easing: "easeOutCubic"  // Natural camera movement
easing: "easeInOutSine"  // Gentle floating
easing: "easeOutBack"   // Bouncy UI elements
```

### 2. Too Many Keyframes

```javascript
// ❌ WRONG: Over-specified animation
keyframes: [
  { time: 0, position: { x: 0, y: 0, z: 0 } },
  { time: 0.1, position: { x: 0.5, y: 0.1, z: 0 } },
  { time: 0.2, position: { x: 1.0, y: 0.2, z: 0 } },
  { time: 0.3, position: { x: 1.5, y: 0.3, z: 0 } },
  { time: 0.4, position: { x: 2.0, y: 0.4, z: 0 } },
  // ... 20 more keyframes!
]

// ✅ CORRECT: Let interpolation do the work
keyframes: [
  { time: 0, position: { x: 0, y: 0, z: 0 } },
  { time: 1, position: { x: 5, y: 1, z: 0 } }
]
// The system handles everything in between!
```

### 3. Wrong Easing for Context

```javascript
// ❌ WRONG: Wrong easing for the effect
explosion: {
  easing: "easeInOutCubic",  // Too slow/soft!
  keyframes: [...]
}

// ✅ CORRECT: Match easing to action
explosion: {
  easing: "easeOutQuad",  // Fast start, gradual slow
  keyframes: [...]
}

heavyDoor: {
  easing: "easeInOutCubic",  // Slow acceleration/deceleration
  keyframes: [...]
}
```

### 4. Forgetting Loop Property

```javascript
// ❌ WRONG: Animation plays once then stops
ambientRotation: {
  keyframes: [
    { time: 0, rotation: { x: 0, y: 0, z: 0 } },
    { time: 1, rotation: { x: 0, y: Math.PI * 2, z: 0 } }
  ],
  // Forgot "loop: true"!
}

// ✅ CORRECT: Specify loop for continuous animation
ambientRotation: {
  loop: true,
  keyframes: [
    { time: 0, rotation: { x: 0, y: 0, z: 0 } },
    { time: 1, rotation: { x: 0, y: Math.PI * 2, z: 0 } }
  ]
}
```

### 5. Duration Too Long or Short

```javascript
// ❌ WRONG: Animations drag on or feel rushed
slowDoor: { duration: 10000, ... },  // 10 seconds to open?!
fastExplosion: { duration: 50, ... },  // Too fast to see

// ✅ CORRECT: Match duration to action
normalDoor: { duration: 1000, ... },  // 1 second feels natural
explosion: { duration: 500, ... },  // Half second for impact
```

---

## Performance Considerations

### Limit Concurrent Animations

```javascript
// Active animation limits
const MAX_CAMERA_ANIMATIONS = 1;  // Only one camera animation
const MAX_OBJECT_ANIMATIONS = 50;  // Multiple object animations OK

// Prioritize important animations
const animationPriority = {
  cinematic: 10,
  gameplay: 5,
  ambient: 1
};
```

### Keyframe Optimization

```javascript
// Fewer keyframes = better performance
// Optimal: 2-4 keyframes per animation

// ❌ 100 keyframes = heavy processing each frame
// ✅ 2-4 keyframes = lightweight interpolation
```

### Avoid Continuous Updates When Not Needed

```javascript
// Pause off-screen animations
if (!isObjectVisible(object)) {
  animationManager.pause(object.id);
}

// Reduce update frequency for distant objects
if (getDistanceToPlayer(object) > 50) {
  animation.setUpdateRate(10);  // Update 10x per second instead of 60
}
```

---

## 🎮 Game Design Perspective

### Animation Functions in Games

1. **Storytelling** - Cinematic moments, reveals
2. **Feedback** - Confirm player actions
3. **Atmosphere** - Ambient movement, life
4. **Guidance** - Draw attention to important items
5. **Pacing** - Control flow and rhythm

### Camera Movement Principles

```javascript
// Principle 1: Subtle is better than flashy
gentleDrift: {
  duration: 10000,
  easing: "easeInOutSine",  // Very gentle
  // Almost imperceptible movement
}

// Principle 2: Movement serves a purpose
focusOnKeyItem: {
  // Moves camera TO important thing
  keyframes: [
    { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } },
    { time: 1, position: { x: 2, y: 1.6, z: 3 }, target: { x: 2, y: 1, z: -1 } }
  ]
  // Now player is looking at the key item
}

// Principle 3: Don't fight player control
// Only animate camera when player is NOT in control
criteria: {
  currentState: CINEMATIC_MODE,  // Or cutscene, etc.
  controlEnabled: false  // Player can't move during animation
}

// Principle 4: Respect motion sickness
// Avoid fast rotations, especially looking up/down
safeCameraMove: {
  easing: "easeInOutCubic",  // Smooth, no jerks
  duration: 3000,  // Give time to adjust
  keyframes: [
    // Keep Y changes minimal (prevent vertical motion sickness)
    { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1.5, z: 0 } },
    { time: 1, position: { x: 2, y: 1.6, z: 3 }, target: { x: 2, y: 1.5, z: 0 } }
  ]
}
```

### Easing Guidelines

| Situation | Recommended Easing | Why |
|-----------|-------------------|-----|
| Camera movement | easeInOutCubic | Natural, smooth |
| UI appearance | easeOutBack | Bouncy, playful |
| Door opening | easeInOutCubic | Heavy, smooth |
| Impact | easeOutQuad | Fast start, slow end |
| Floating/ambient | easeInOutSine | Gentle, repetitive |
| Building tension | easeInCubic | Slow start, fast end |

---

## Next Steps

Now that you understand AnimationManager:

- [VFXManager](../07-visual-effects/vfx-manager.md) - Visual effects system
- [SceneManager](../03-scene-rendering/scene-manager.md) - 3D content management
- [GameManager Deep Dive](../02-core-architecture/game-manager-deep-dive.md) - State-driven animation triggers
- [Data-Driven Design](../02-core-architecture/data-driven-design.md) - How animation data works

---

## References

- [Three.js MathUtils](https://threejs.org/docs/#api/en/math/MathUtils) - Math utilities including lerp, smoothstep
- [Easing Functions (easings.net)](https://easings.net/) - Visual easing function reference
- [Animation Best Practices](https://www.youtube.com/watch?v=Kz2M1dudq_Y) - Game animation principles
- [Cinematography for Games](https://www.gamedeveloper.com/disciplines/cinematography-for-games) - Camera design theory

*Documentation last updated: January 12, 2026*
