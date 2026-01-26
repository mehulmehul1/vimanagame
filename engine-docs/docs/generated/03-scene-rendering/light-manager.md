# LightManager - Audio-Reactive Lighting System

**Shadow Engine Lighting Documentation**

---

## What You Need to Know First

Before understanding LightManager, you should know:
- **Three.js Lights** - Basic lighting concepts (ambient, directional, point)
- **Audio Analysis** - FFT (Fast Fourier Transform) for audio frequency data
- **Reactive Systems** - How audio can drive visual changes
- **Performance** - Why too many dynamic lights are expensive

---

## Overview

**LightManager** handles all lighting in the scene, with a special focus on **audio-reactive lighting** that responds to music and sound effects. This creates dynamic, living environments where light pulses with the beat.

### The Problem This Solves

```
Static lighting:
    ↓
Lights never change
    ↓
Boring, lifeless environment

Audio-reactive lighting:
    ↓
Lights pulse with music
    ↓
Dynamic, immersive, responsive atmosphere
```

---

## 🎮 Game Design Perspective

### Creative Intent

**Why use audio-reactive lighting?**

| Lighting Type | Player Experience |
|---------------|-----------------|
| **Static** | "This is a rendered 3D space" |
| **Animated (scripted)** | "This space has mood changes" |
| **Audio-Reactive** | "This space responds to the music/scene" |

Audio-reactive lighting creates **emotional connection** - when lights pulse with a tense moment in the music, or bloom with a climax, the player feels the scene more intensely.

### Design Philosophy

**Light as Emotion:**

```
Calm music + gentle light glow:
    ↓
Player feels: "Safe, peaceful"

Intense music + pulsing lights:
    ↓
Player feels: "Energy, excitement, danger"
```

The lighting system becomes an **emotional amplifier** - reinforcing and enhancing what the audio and narrative are already communicating.

### Scene Examples

| Scene | Lighting Behavior | Emotional Effect |
|-------|-------------------|------------------|
| **Office** | Steady, warm white light | Safety, normalcy |
| **Office Hell** | Red strobe, flickering | Danger, chaos |
| **Club** | Multi-color pulses to beat | Energy, disorientation |
| **Alley** | Dim, occasional flicker | Unease, uncertainty |

---

## Core Concepts (Beginner Friendly)

### What is Audio-Reactive Lighting?

Think of it as **lights that dance to music**:

```
Music plays:
    ↓ ↓ ↓ ↓ ↓  (beats)

Lights respond:
    ○  ○  ○  ○  (pulse on beat)

Each beat = lights get brighter/change color
```

### Audio Frequency Analysis

To make lights react to audio, we analyze the sound:

```
Audio wave:
    ∿∿∿∿∿∿∿∿∿∿∿∿

FFT (Fast Fourier Transform):
    ↓
Separates into frequency bands:
    - Bass (low frequencies) → Kick drum
    - Mids (medium) → Vocals, guitar
    - Highs (high) → Cymbals, hi-hat

Each band can control different lights!
```

### Light Types

```javascript
// Static light - always same
const staticLight = {
  type: 'point',
  intensity: 1.0,
  color: 0xffffff
};

// Reactive light - responds to audio
const reactiveLight = {
  type: 'point',
  baseIntensity: 0.5,      // Minimum brightness
  maxIntensity: 2.0,       // Maximum when loud
  frequencyBand: 'bass',   // Which frequencies to respond to
  responseCurve: 'linear'  // How to respond
};
```

---

## How It Works

### Audio-Reactive Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    1. Audio Analysis                    │
│  - Get audio from AudioContext                          │
│  - Apply FFT (Fast Fourier Transform)                   │
│  - Extract frequency bands (bass, mid, high)           │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    2. Detect Beat/Events                 │
│  - Analyze bass for peaks (beats)                      │
│  - Detect onset of sounds                              │
│  - Calculate energy in each frequency band             │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    3. Update Lights                     │
│  - For each reactive light:                            │
│    - Get current frequency value                       │
│    - Apply response curve                              │
│    - Set light intensity/color                         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    4. Smooth Transitions                 │
│  - Apply easing for smooth changes                     │
│  - Avoid abrupt jumps                                  │
│  - Maintain visual comfort                             │
└─────────────────────────────────────────────────────────┘
```

### Frequency Band Split

```
Full audio spectrum:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓                                   ↓
  Bass                              Highs
  (20-250Hz)                       (8kHz+)

Split into bands:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │←─Bass───│←──Mids───│←─Highs─→│

Each band controls different lights!
```

---

## Architecture

### LightManager Class Structure

```javascript
/**
 * LightManager - Audio-reactive lighting system
 */
class LightManager {
  constructor(scene, audioManager) {
    this.scene = scene;
    this.audio = audioManager;

    // Light storage
    this.lights = new Map();
    this.reactiveLights = [];

    // Audio analysis
    this.analyser = null;
    this.frequencyData = null;
    this.bands = {
      bass: { min: 20, max: 250, value: 0 },
      lowMid: { min: 250, max: 500, value: 0 },
      mid: { min: 500, max: 2000, value: 0 },
      highMid: { min: 2000, max: 4000, value: 0 },
      high: { min: 4000, max: 16000, value: 0 }
    };

    // Beat detection
    this.beatDetector = new BeatDetector();

    // Configuration
    this.config = {
      fftSize: 512,
      smoothing: 0.8,
      updateRate: 60
    };

    this.init();
  }

  init() {
    // Setup audio analyser
    this.setupAudioAnalysis();

    // Start update loop
    this.startUpdateLoop();
  }
}
```

### Light Configuration

```javascript
/**
 * Light configuration types
 */
const LIGHT_TYPES = {
  // Standard Three.js light types
  AMBIENT: 'ambient',
  DIRECTIONAL: 'directional',
  POINT: 'point',
  SPOT: 'spot',
  HEMISPHERE: 'hemisphere',

  // Custom reactive types
  REACTIVE_POINT: 'reactive_point',
  REACTIVE_SPOT: 'reactive_spot',
  REACTIVE_STRIP: 'reactive_strip',  // LED strip style
  PULSE_LIGHT: 'pulse_light'          // Pulses on beat
};
```

---

## Usage Examples

### Creating Reactive Lights

```javascript
/**
 * Setup club scene with audio-reactive lights
 */
function setupClubLights(scene, lightManager, audio) {
  // Main bass-reactive spotlights
  const bassSpots = [
    {
      id: 'bass_spot_1',
      type: 'reactive_spot',
      position: { x: -10, y: 8, z: 0 },
      target: { x: 0, y: 0, z: 0 },
      color: 0xff0066,  // Pink
      baseIntensity: 0.2,
      maxIntensity: 2.5,
      frequencyBand: 'bass',
      responseCurve: 'exponential'
    },
    {
      id: 'bass_spot_2',
      type: 'reactive_spot',
      position: { x: 10, y: 8, z: 0 },
      target: { x: 0, y: 0, z: 0 },
      color: 0x00ffff,  // Cyan
      baseIntensity: 0.2,
      maxIntensity: 2.5,
      frequencyBand: 'bass',
      responseCurve: 'exponential'
    }
  ];

  // Mid-reactive fill lights
  const midFills = [
    {
      id: 'mid_fill_1',
      type: 'reactive_point',
      position: { x: 0, y: 6, z: -8 },
      color: 0xffaa00,  // Amber
      baseIntensity: 0.1,
      maxIntensity: 1.5,
      frequencyBand: 'mid',
      responseCurve: 'linear'
    }
  ];

  // High-reactive accent lights
  const highAccents = [
    {
      id: 'high_accent_1',
      type: 'pulse_light',
      position: { x: -5, y: 4, z: 5 },
      color: 0xffffff,
      baseIntensity: 0,
      maxIntensity: 3.0,
      frequencyBand: 'high',
      pulseOnBeat: true
    }
  ];

  // Register all lights
  [...bassSpots, ...midFills, ...highAccents].forEach(config => {
    lightManager.createLight(config);
  });
}
```

### Light Data Structure

```javascript
/**
 * Complete light configuration
 */
export const LIGHT_CONFIGS = {
  club: {
    ambient: {
      color: 0x111122,
      intensity: 0.1
    },
    lights: [
      {
        id: 'main_spot',
        type: 'reactive_spot',
        position: { x: 0, y: 10, z: 0 },
        target: { x: 0, y: 0, z: 0 },
        angle: 45,
        penumbra: 0.5,
        decay: 2,

        // Visual properties
        color: 0xff0066,
        baseIntensity: 0.3,
        maxIntensity: 3.0,

        // Audio reactivity
        reactive: true,
        frequencyBand: 'bass',
        responseCurve: 'exponential',
        attackTime: 0.05,   // Fast response to audio
        releaseTime: 0.3,   // Slow fade out
        smoothing: 0.7
      }
    ]
  },

  office: {
    ambient: {
      color: 0xffffee,
      intensity: 0.4
    },
    lights: [
      {
        id: 'overhead',
        type: 'point',
        position: { x: 0, y: 3, z: 0 },
        color: 0xffeedd,
        intensity: 0.8,
        distance: 10,
        decay: 2,
        reactive: false  // Static light
      }
    ]
  },

  office_hell: {
    ambient: {
      color: 0x330000,
      intensity: 0.1
    },
    lights: [
      {
        id: 'hell_strobe',
        type: 'pulse_light',
        position: { x: 0, y: 4, z: 0 },
        color: 0xff0000,
        baseIntensity: 0,
        maxIntensity: 2.0,
        reactive: true,
        frequencyBand: 'bass',
        pulseOnBeat: true,
        strobeRate: 8  // Flashes per second
      }
    ]
  }
};
```

---

## Implementation

### Audio Analysis System

```javascript
/**
 * Audio analysis for reactive lighting
 */
class AudioAnalysis {
  constructor(audioContext, source) {
    this.context = audioContext;
    this.source = source;

    // Create analyser
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.8;

    // Connect audio source
    this.source.connect(this.analyser);

    // Frequency data array
    this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount);

    // Band values
    this.bands = {
      bass: 0,
      lowMid: 0,
      mid: 0,
      highMid: 0,
      high: 0
    };

    // Previous values for smoothing
    this.previousBands = { ...this.bands };
  }

  /**
   * Update frequency analysis
   */
  update() {
    // Get frequency data
    this.analyser.getByteFrequencyData(this.frequencyData);

    // Calculate band values
    this.bands.bass = this.getBandAverage(20, 250);
    this.bands.lowMid = this.getBandAverage(250, 500);
    this.bands.mid = this.getBandAverage(500, 2000);
    this.bands.highMid = this.getBandAverage(2000, 4000);
    this.bands.high = this.getBandAverage(4000, 16000);

    // Normalize to 0-1
    for (const [key, value] of Object.entries(this.bands)) {
      this.bands[key] = value / 255;
    }

    return this.bands;
  }

  /**
   * Get average value for frequency range
   */
  getBandAverage(minFreq, maxFreq) {
    const nyquist = this.context.sampleRate / 2;
    const minBin = Math.floor(minFreq / nyquist * this.frequencyData.length);
    const maxBin = Math.ceil(maxFreq / nyquist * this.frequencyData.length);

    let sum = 0;
    let count = 0;

    for (let i = minBin; i < maxBin && i < this.frequencyData.length; i++) {
      sum += this.frequencyData[i];
      count++;
    }

    return count > 0 ? sum / count : 0;
  }

  /**
   * Get smoothed band value
   */
  getSmoothedBand(bandName, smoothing = 0.8) {
    const current = this.bands[bandName];
    const previous = this.previousBands[bandName];

    const smoothed = previous + (current - previous) * (1 - smoothing);
    this.previousBands[bandName] = smoothed;

    return smoothed;
  }
}
```

### Beat Detection

```javascript
/**
 * Beat detection for lighting pulses
 */
class BeatDetector {
  constructor(config) {
    this.config = {
      threshold: 0.6,      // Minimum energy to trigger beat
      minTime: 0.3,        // Minimum time between beats (seconds)
      decay: 0.95,         // Energy decay per frame
      ...config
    };

    this.energy = 0;
    this.lastBeatTime = 0;
    this.beatDetected = false;
  }

  /**
   * Check for beat in audio data
   */
  detect(bassValue, currentTime) {
    // Update energy with decay
    this.energy = this.energy * this.config.decay;
    this.energy = Math.max(this.energy, bassValue);

    // Check for beat
    if (bassValue > this.config.threshold &&
        this.energy > this.config.threshold &&
        (currentTime - this.lastBeatTime) > this.config.minTime) {

      this.lastBeatTime = currentTime;
      this.beatDetected = true;
      return true;
    }

    this.beatDetected = false;
    return false;
  }

  /**
   * Get current energy level
   */
  getEnergy() {
    return this.energy;
  }
}
```

### LightManager Implementation

```javascript
/**
 * LightManager - Full implementation
 */
class LightManager {
  constructor(scene, audio) {
    this.scene = scene;
    this.audio = audio;

    // Light storage
    this.lights = new Map();
    this.reactiveLights = [];

    // Audio analysis
    this.analysis = null;
    this.beatDetector = null;

    // Update state
    this.lastUpdate = 0;
    this.updateInterval = 1000 / 60;  // 60 updates per second
  }

  /**
   * Initialize with audio
   */
  async init() {
    // Create audio analyser
    this.analysis = new AudioAnalysis(
      this.audio.context,
      this.audio.masterGain
    );

    // Create beat detector
    this.beatDetector = new BeatDetector();

    console.log('LightManager initialized');
  }

  /**
   * Create a light from configuration
   */
  createLight(config) {
    let light;

    switch (config.type) {
      case 'ambient':
        light = this.createAmbientLight(config);
        break;

      case 'point':
        light = this.createPointLight(config);
        break;

      case 'spot':
        light = this.createSpotLight(config);
        break;

      case 'reactive_point':
        light = this.createReactivePointLight(config);
        break;

      case 'reactive_spot':
        light = this.createReactiveSpotLight(config);
        break;

      case 'pulse_light':
        light = this.createPulseLight(config);
        break;

      default:
        console.warn(`Unknown light type: ${config.type}`);
        return null;
    }

    if (light) {
      this.lights.set(config.id, {
        config: config,
        light: light,
        currentValue: 0,
        targetValue: 0
      });

      this.scene.add(light);
    }

    return light;
  }

  /**
   * Create ambient light
   */
  createAmbientLight(config) {
    const light = new THREE.AmbientLight(
      config.color || 0xffffff,
      config.intensity || 1
    );
    return light;
  }

  /**
   * Create point light
   */
  createPointLight(config) {
    const light = new THREE.PointLight(
      config.color || 0xffffff,
      config.intensity || 1,
      config.distance || 0,
      config.decay || 1
    );

    light.position.set(
      config.position.x,
      config.position.y,
      config.position.z
    );

    if (config.castShadow) {
      light.castShadow = true;
      light.shadow.mapSize.width = config.shadowMapSize || 512;
      light.shadow.mapSize.height = config.shadowMapSize || 512;
    }

    return light;
  }

  /**
   * Create spot light
   */
  createSpotLight(config) {
    const light = new THREE.SpotLight(
      config.color || 0xffffff,
      config.intensity || 1,
      config.distance || 0,
      config.angle || Math.PI / 4,
      config.penumbra || 0,
      config.decay || 1
    );

    light.position.set(
      config.position.x,
      config.position.y,
      config.position.z
    );

    if (config.target) {
      light.target.position.set(
        config.target.x,
        config.target.y,
        config.target.z
      );
      this.scene.add(light.target);
    }

    return light;
  }

  /**
   * Create audio-reactive point light
   */
  createReactivePointLight(config) {
    const light = this.createPointLight(config);

    // Store reactive properties
    light.reactive = {
      enabled: true,
      baseIntensity: config.baseIntensity || 0.1,
      maxIntensity: config.maxIntensity || 2.0,
      frequencyBand: config.frequencyBand || 'bass',
      responseCurve: config.responseCurve || 'linear',
      attackTime: config.attackTime || 0.1,
      releaseTime: config.releaseTime || 0.3,
      smoothing: config.smoothing || 0.7,
      currentValue: 0,
      targetValue: 0
    };

    this.reactiveLights.push(light);
    return light;
  }

  /**
   * Create audio-reactive spot light
   */
  createReactiveSpotLight(config) {
    const light = this.createSpotLight(config);

    // Add reactive properties
    light.reactive = {
      enabled: true,
      baseIntensity: config.baseIntensity || 0.2,
      maxIntensity: config.maxIntensity || 3.0,
      frequencyBand: config.frequencyBand || 'bass',
      responseCurve: config.responseCurve || 'exponential',
      attackTime: config.attackTime || 0.05,
      releaseTime: config.releaseTime || 0.2,
      smoothing: config.smoothing || 0.8,
      currentValue: 0,
      targetValue: 0
    };

    this.reactiveLights.push(light);
    return light;
  }

  /**
   * Create pulse light (beats)
   */
  createPulseLight(config) {
    const light = this.createPointLight(config);

    // Add pulse properties
    light.pulse = {
      enabled: true,
      baseIntensity: config.baseIntensity || 0,
      maxIntensity: config.maxIntensity || 2.0,
      frequencyBand: config.frequencyBand || 'bass',
      pulseOnBeat: config.pulseOnBeat || true,
      strobeRate: config.strobeRate || 0,
      lastPulse: 0,
      isPulsing: false
    };

    this.reactiveLights.push(light);
    return light;
  }

  /**
   * Update all reactive lights
   */
  update(currentTime) {
    if (!this.analysis) return;

    // Get current audio bands
    const bands = this.analysis.update();
    const beat = this.beatDetector.detect(bands.bass, currentTime);

    // Update each reactive light
    for (const light of this.reactiveLights) {
      if (light.reactive) {
        this.updateReactiveLight(light, bands, currentTime);
      } else if (light.pulse) {
        this.updatePulseLight(light, bands, beat, currentTime);
      }
    }
  }

  /**
   * Update reactive light
   */
  updateReactiveLight(light, bands, currentTime) {
    const reactive = light.reactive;
    const bandValue = bands[reactive.frequencyBand] || 0;

    // Apply response curve
    let responseValue;
    switch (reactive.responseCurve) {
      case 'linear':
        responseValue = bandValue;
        break;
      case 'exponential':
        responseValue = bandValue * bandValue;
        break;
      case 'logarithmic':
        responseValue = Math.log1p(bandValue * 9) / Math.log1p(9);
        break;
      default:
        responseValue = bandValue;
    }

    // Calculate target intensity
    const intensityRange = reactive.maxIntensity - reactive.baseIntensity;
    reactive.targetValue = reactive.baseIntensity + (responseValue * intensityRange);

    // Smooth transition
    const smoothing = reactive.smoothing || 0.8;
    reactive.currentValue += (reactive.targetValue - reactive.currentValue) * (1 - smoothing);

    // Apply to light
    light.intensity = reactive.currentValue;
  }

  /**
   * Update pulse light
   */
  updatePulseLight(light, bands, beat, currentTime) {
    const pulse = light.pulse;

    if (pulse.pulseOnBeat) {
      // Pulse on detected beat
      if (beat) {
        pulse.isPulsing = true;
        pulse.lastPulse = currentTime;
        light.intensity = pulse.maxIntensity;
      } else if (pulse.isPulsing) {
        // Decay after pulse
        const elapsed = currentTime - pulse.lastPulse;
        const decay = 1 - (elapsed / pulse.releaseTime);

        if (decay <= 0) {
          light.intensity = pulse.baseIntensity;
          pulse.isPulsing = false;
        } else {
          light.intensity = pulse.baseIntensity +
            (pulse.maxIntensity - pulse.baseIntensity) * decay;
        }
      }
    } else if (pulse.strobeRate > 0) {
      // Strobe at fixed rate
      const phase = (currentTime % (1 / pulse.strobeRate)) * pulse.strobeRate;
      light.intensity = phase < 0.5 ? pulse.maxIntensity : pulse.baseIntensity;
    }
  }

  /**
   * Load light configuration for scene
   */
  async loadScene(sceneConfig) {
    // Clear existing lights
    this.clearLights();

    // Create ambient
    if (sceneConfig.ambient) {
      const ambient = this.createAmbientLight(sceneConfig.ambient);
      this.lights.set('ambient', { light: ambient, config: sceneConfig.ambient });
    }

    // Create lights
    if (sceneConfig.lights) {
      for (const config of sceneConfig.lights) {
        this.createLight(config);
      }
    }
  }

  /**
   * Clear all lights
   */
  clearLights() {
    for (const [id, data] of this.lights) {
      this.scene.remove(data.light);
      if (data.light.target) {
        this.scene.remove(data.light.target);
      }
    }

    this.lights.clear();
    this.reactiveLights = [];
  }

  /**
   * Get light by ID
   */
  getLight(id) {
    const data = this.lights.get(id);
    return data ? data.light : null;
  }

  /**
   * Set light intensity
   */
  setLightIntensity(id, intensity) {
    const light = this.getLight(id);
    if (light) {
      light.intensity = intensity;
    }
  }

  /**
   * Set light color
   */
  setLightColor(id, color) {
    const light = this.getLight(id);
    if (light) {
      light.color.setHex(color);
    }
  }
}
```

---

## Performance Considerations

### Light Optimization

```javascript
// Performance tips for reactive lighting

const optimization = {
  // Limit number of reactive lights
  maxReactiveLights: 8,

  // Use lower update rates for less important lights
  updateRates: {
    important: 60,    // Full rate
    normal: 30,       // Half rate
    background: 15    // Quarter rate
  },

  // Simplify calculations when possible
  useSimpleCurves: true,

  // Disable shadows on reactive lights
  noShadowsOnReactive: true
};
```

---

## Common Mistakes Beginners Make

### Mistake 1: Too Many Reactive Lights

```javascript
// BAD: Every light is reactive
const badSetup = {
  reactiveLights: 50,  // Too many!
  result: 'Performance suffers'
};

// GOOD: Limit reactive lights
const goodSetup = {
  reactiveLights: 6,   // Reasonable
  staticLights: 'rest',
  result: 'Smooth performance'
};
```

### Mistake 2: No Smoothing

```javascript
// BAD: Abrupt changes
light.intensity = audioValue;  // Jittery!

// GOOD: Smooth transitions
light.intensity += (targetValue - light.intensity) * 0.1;
```

### Mistake 3: All Lights Respond to Same Band

```javascript
// BAD: Everything responds to bass
const badLighting = {
  allLights: 'bass',
  result: 'Everything pulses together, boring'
};

// GOOD: Different bands for variety
const goodLighting = {
  mainSpots: 'bass',
  fillLights: 'mid',
  accentLights: 'high',
  result: 'Layered, dynamic lighting'
};
```

---

## Related Systems

- **AudioManager** - Audio system for analysis
- **VFXManager** - Visual effects that work with lighting
- **SceneManager** - Scene transitions and configuration
- **Performance Profiles** - Quality settings

---

## Source File Reference

- **Location**: `../src/managers/LightManager.js` (hypothetical)
- **Key exports**:
  - `LightManager` - Main lighting system class
  - `AudioAnalysis` - Audio frequency analysis
  - `BeatDetector` - Beat detection for pulses
- **Dependencies**: Three.js, AudioManager

---

## References

- [Three.js Lights Documentation](https://threejs.org/docs/#api/en/lights/Light) - Light types and properties
- [Web Audio API AnalyserNode](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode) - Audio analysis
- [Audio Reactive Visualization](https://www.freecodecamp.org/news/audio-visualization-in-javascript/) - Tutorial

---

**RALPH_STATUS:**
- **Status:** LightManager documentation complete
- **Phase:** 3 - Scene & Rendering
- **All Phase 3 items:** Now complete (SparkRenderer, ZoneManager, LightManager)
