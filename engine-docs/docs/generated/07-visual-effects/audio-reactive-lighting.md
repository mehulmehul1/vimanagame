# Audio-Reactive Lighting - First Principles Guide

## Overview

**Audio-reactive lighting** makes lights in your game respond to music and sound effects in real-time. When the bass hits, lights pulse; when high frequencies play, lights shimmer. This creates an immersive connection between what players **hear** and what they **see**, making the world feel alive and responsive.

Think of audio-reactive lighting as the **"visual music"** system - just as a music visualizer at a concert makes sound visible through light and color, this system connects your game's audio to its lighting, creating synchronization between sound and vision.

## What You Need to Know First

Before understanding audio-reactive lighting, you should know:
- **Web Audio API** - Browser audio analysis and playback
- **Fast Fourier Transform (FFT)** - Converting audio to frequency data
- **Frequency bands** - Bass, midrange, treble ranges
- **Light intensity and color** - Three.js light properties
- **Frame updates** - Synchronizing visuals to audio
- **Audio frequency ranges** - What sounds are in what bands

### Quick Refresher: Audio to Visual

```
AUDIO WAVEFORM (time domain):
┌─────────────────────────────────────────┐
│  Amplitude over time                    │
│                                         │
│     ┌───┐   ┌─┐                         │
│     │   │   │ │     ┌──┐               │
│ ────┘   └───┘ └─────┘  └───            │
│                                         │
│  Time →                                │
└─────────────────────────────────────────┘
│
│  FFT (Fast Fourier Transform)
│  converts this to frequency data
▼

FREQUENCY SPECTRUM (frequency domain):
┌─────────────────────────────────────────┐
│  Volume at each frequency               │
│                                         │
│  Volume                                 │
│    ▲                                    │
│    │  ┌─┐   ┌───┐                       │
│    │  │ │   │   │              ┌─┐     │
│    │  │ │   │   │              │ │     │
│    │ ─┴─┴───┴───┴──────────────┴─┴────  │
│    └──────────────────────────────────  │
│      Bass  Mid  High                    │
│      20Hz  500Hz 10kHz                  │
│                                         │
│  We read this data to drive lights!     │
└─────────────────────────────────────────┘

MAPPED TO LIGHTS:
┌─────────────────────────────────────────┐
│                                         │
│  Low frequencies → Bass lights          │
│  ├─ Red ambient glow                    │
│  └─ Pulse on kick drum                  │
│                                         │
│  Mid frequencies → Rhythm lights        │
│  ├─ Flickering candle                   │
│  └─ Breathing glow                      │
│                                         │
│  High frequencies → Sparkle lights      │
│  ├─ Rune shimmer                        │
│  └─ Crystal flicker                     │
│                                         │
└─────────────────────────────────────────┘
```

---

## Part 1: Why Use Audio-Reactive Lighting?

### The Problem: Disconnect Between Sound and Vision

Without audio-reactive lighting:

```javascript
// ❌ WITHOUT Audio-Reactive Lighting
const scene = {
  musicPlaying: true,
  lights: {
    ambient: { intensity: 0.5, color: "red" },
    candle: { intensity: 1.0, flicker: "random" }
  }
};
// Music plays, but lights do their own thing.
// No connection between what you hear and see.
```

**Problems:**
- Audio and visuals feel disconnected
- Music doesn't affect atmosphere
- Sound effects lack visual impact
- Missing immersion and synchronicity

### The Solution: Audio-Reactive Lighting

```javascript
// ✅ WITH Audio-Reactive Lighting
const scene = {
  musicPlaying: true,
  lights: {
    ambient: {
      intensity: 0.5,
      audioReactive: {
        frequency: "bass",
        response: 0.5  // How much it responds
      }
    },
    candle: {
      intensity: 1.0,
      audioReactive: {
        frequency: "mid",
        mode: "flicker"
      }
    }
  }
};
// Lights pulse with the music.
// Every sound has visual impact.
// Complete audio-visual immersion!
```

**Benefits:**
- Synchronizes audio and visuals
- Creates immersive atmosphere
- Sound effects feel more impactful
- Music becomes a gameplay element
- Dynamic, living environments

---

## Part 2: Audio Analysis Fundamentals

### The Web Audio API

```javascript
class AudioAnalyzer {
  constructor() {
    this.audioContext = null;
    this.analyser = null;
    this.frequencyData = null;
    this.timeData = null;
  }

  async initialize(audioElement) {
    // Create audio context
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

    // Create analyser node
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 512;  // Size of FFT (must be power of 2)
    this.analyser.smoothingTimeConstant = 0.8;  // Smooth the data

    // Connect audio source
    const source = this.audioContext.createMediaElementSource(audioElement);
    source.connect(this.analyser);
    this.analyser.connect(this.audioContext.destination);

    // Create data arrays
    const bufferLength = this.analyser.frequencyBinCount;
    this.frequencyData = new Uint8Array(bufferLength);
    this.timeData = new Uint8Array(bufferLength);
  }

  getFrequencyData() {
    if (!this.analyser) return new Uint8Array(0);

    // Fill frequencyData with current frequency data (0-255)
    this.analyser.getByteFrequencyData(this.frequencyData);
    return this.frequencyData;
  }

  getTimeDomainData() {
    if (!this.analyser) return new Uint8Array(0);

    // Fill timeData with waveform data (0-255)
    this.analyser.getByteTimeDomainData(this.timeData);
    return this.timeData;
  }
}
```

### Understanding FFT Data

```javascript
// FFT size determines frequency resolution
// fftSize = 512 → 256 frequency bins (half of fftSize)

const fftSizeToBins = {
  256: 128,    // Low resolution, fast
  512: 256,    // Good balance
  1024: 512,   // Higher resolution
  2048: 1024   // High resolution, slower
};

// Frequency bins are arranged from low to high
// Bin 0 = lowest frequency (bass)
// Bin N = highest frequency (treble)

class FrequencyBandAnalyzer {
  constructor(frequencyData, sampleRate = 44100) {
    this.data = frequencyData;
    this.sampleRate = sampleRate;
    this.binCount = frequencyData.length;
  }

  // Calculate frequency range for each bin
  getBinFrequency(binIndex) {
    const nyquist = this.sampleRate / 2;
    return (binIndex / this.binCount) * nyquist;
  }

  // Get average value of a frequency range
  getBandRange(minFreq, maxFreq) {
    const minBin = Math.floor((minFreq / (this.sampleRate / 2)) * this.binCount);
    const maxBin = Math.ceil((maxFreq / (this.sampleRate / 2)) * this.binCount);

    let sum = 0;
    let count = 0;

    for (let i = minBin; i <= maxBin && i < this.binCount; i++) {
      sum += this.data[i];
      count++;
    }

    return count > 0 ? sum / count : 0;
  }

  // Common frequency bands
  getBass() {
    return this.getBandRange(20, 250);    // Kick drum, bass guitar
  }

  getLowMid() {
    return this.getBandRange(250, 500);   // Lower vocals, toms
  }

  getMid() {
    return this.getBandRange(500, 2000);  // Vocals, guitar
  }

  getHighMid() {
    return this.getBandRange(2000, 4000); // Upper vocals, snare
  }

  getTreble() {
    return this.getBandRange(4000, 16000); // Cymbals, high harmonics
  }

  // Get all bands at once
  getBands() {
    return {
      bass: this.getBass(),
      lowMid: this.getLowMid(),
      mid: this.getMid(),
      highMid: this.getHighMid(),
      treble: this.getTreble()
    };
  }
}
```

---

## Part 3: Audio-Reactive Light Implementation

### Base Audio-Reactive Light

```javascript
class AudioReactiveLight {
  constructor(light, config) {
    this.light = light;
    this.config = config;

    // Configuration
    this.frequency = config.frequency || "bass";  // bass, mid, treble, or all
    this.response = config.response || 1.0;       // How much light responds
    this.minIntensity = config.minIntensity || 0;
    this.maxIntensity = config.maxIntensity || 2;
    this.smoothing = config.smoothing || 0.8;     // Smooth transitions

    // Color react
    this.colorReactive = config.colorReactive || false;
    this.baseColor = new THREE.Color(config.color || 0xffffff);
    this.peakColor = new THREE.Color(config.peakColor || 0xffffff);

    // State
    this.currentIntensity = this.minIntensity;
    this.currentColor = this.baseColor.clone();
  }

  update(audioBands, deltaTime) {
    // Get audio value for our frequency
    const audioValue = this.getAudioValue(audioBands);

    // Normalize audio value (0-1)
    const normalizedValue = audioValue / 255;

    // Calculate target intensity
    const intensityRange = this.maxIntensity - this.minIntensity;
    const targetIntensity = this.minIntensity +
      (normalizedValue * this.response * intensityRange);

    // Smooth intensity
    this.currentIntensity = THREE.MathUtils.lerp(
      this.currentIntensity,
      targetIntensity,
      1 - this.smoothing
    );

    // Apply to light
    this.light.intensity = this.currentIntensity;

    // Update color if reactive
    if (this.colorReactive) {
      this.updateColor(normalizedValue);
    }
  }

  getAudioValue(audioBands) {
    switch (this.frequency) {
      case "bass":
        return audioBands.bass || 0;
      case "lowMid":
        return audioBands.lowMid || 0;
      case "mid":
        return audioBands.mid || 0;
      case "highMid":
        return audioBands.highMid || 0;
      case "treble":
        return audioBands.treble || 0;
      case "all":
        return (audioBands.bass + audioBands.mid + audioBands.treble) / 3;
      default:
        return audioBands.bass || 0;
    }
  }

  updateColor(normalizedValue) {
    // Interpolate between base and peak color
    this.currentColor.lerpColors(
      this.baseColor,
      this.peakColor,
      normalizedValue * this.response
    );

    this.light.color.copy(this.currentColor);
  }
}
```

### Point Light Reactivity

```javascript
class AudioReactivePointLight extends AudioReactiveLight {
  constructor(config) {
    const color = config.color || 0xffffff;
    const intensity = config.intensity || 1;
    const distance = config.distance || 10;

    const light = new THREE.PointLight(color, intensity, distance);
    light.position.set(
      config.position?.x || 0,
      config.position?.y || 2,
      config.position?.z || 0
    );

    super(light, config);
  }

  update(audioBands, deltaTime) {
    super.update(audioBands, deltaTime);

    // Optional: Distance pulse
    if (this.config.distancePulse) {
      const audioValue = this.getAudioValue(audioBands) / 255;
      const pulseAmount = audioValue * this.config.distancePulse;
      this.light.distance = this.config.baseDistance * (1 + pulseAmount);
    }
  }
}
```

### Ambient Light Reactivity

```javascript
class AudioReactiveAmbientLight extends AudioReactiveLight {
  constructor(config) {
    const color = config.color || 0x404040;
    const intensity = config.intensity || 0.5;

    const light = new THREE.AmbientLight(color, intensity);
    super(light, config);
  }

  // Ambient lights respond more subtly
  update(audioBands, deltaTime) {
    // Use lower response for ambient (more subtle)
    const originalResponse = this.response;
    this.response = originalResponse * 0.3;

    super.update(audioBands, deltaTime);

    this.response = originalResponse;
  }
}
```

### Spot Light Reactivity

```javascript
class AudioReactiveSpotLight extends AudioReactiveLight {
  constructor(config) {
    const color = config.color || 0xffffff;
    const intensity = config.intensity || 1;
    const distance = config.distance || 10;
    const angle = config.angle || Math.PI / 6;
    const penumbra = config.penumbra || 0;

    const light = new THREE.SpotLight(
      color,
      intensity,
      distance,
      angle,
      penumbra
    );

    light.position.set(
      config.position?.x || 0,
      config.position?.y || 5,
      config.position?.z || 0
    );

    if (config.target) {
      light.target.position.set(
        config.target.x || 0,
        config.target.y || 0,
        config.target.z || 0
      );
    }

    super(light, config);

    this.baseAngle = angle;
    this.anglePulse = config.anglePulse || 0;
  }

  update(audioBands, deltaTime) {
    super.update(audioBands, deltaTime);

    // Optional: Angle pulse (spotlight widens with audio)
    if (this.anglePulse) {
      const audioValue = this.getAudioValue(audioBands) / 255;
      this.light.angle = this.baseAngle * (1 + audioValue * this.anglePulse);
    }
  }
}
```

---

## Part 4: Audio-Reactive Lighting Manager

```javascript
class AudioReactiveLightingManager {
  constructor(scene, audioAnalyzer) {
    this.scene = scene;
    this.audioAnalyzer = audioAnalyzer;
    this.reactiveLights = new Map();
    this.frequencyAnalyzer = null;
  }

  async initialize(audioElement) {
    await this.audioAnalyzer.initialize(audioElement);
    this.frequencyAnalyzer = new FrequencyBandAnalyzer(
      this.audioAnalyzer.frequencyData,
      this.audioAnalyzer.audioContext.sampleRate
    );
  }

  createReactiveLight(config) {
    let reactiveLight;

    switch (config.type) {
      case "point":
        reactiveLight = new AudioReactivePointLight(config);
        break;
      case "ambient":
        reactiveLight = new AudioReactiveAmbientLight(config);
        break;
      case "spot":
        reactiveLight = new AudioReactiveSpotLight(config);
        break;
      default:
        reactiveLight = new AudioReactivePointLight(config);
    }

    // Add to scene
    this.scene.add(reactiveLight.light);

    // Store for updates
    this.reactiveLights.set(config.name, reactiveLight);

    return reactiveLight;
  }

  update(deltaTime) {
    if (!this.frequencyAnalyzer) return;

    // Get current frequency data
    this.audioAnalyzer.getFrequencyData();

    // Get frequency bands
    const audioBands = this.frequencyAnalyzer.getBands();

    // Update all reactive lights
    for (const [name, reactiveLight] of this.reactiveLights) {
      // Check if light should be active
      if (this.shouldUpdateLight(reactiveLight)) {
        reactiveLight.update(audioBands, deltaTime);
      }
    }
  }

  shouldUpdateLight(reactiveLight) {
    const config = reactiveLight.config;

    // Check criteria
    if (config.criteria) {
      return this.evaluateCriteria(config.criteria);
    }

    return true;
  }

  evaluateCriteria(criteria) {
    // Implement criteria evaluation (similar to VFXManager)
    // This would check game state, zones, etc.
    return true;  // Placeholder
  }

  removeLight(name) {
    const reactiveLight = this.reactiveLights.get(name);
    if (reactiveLight) {
      this.scene.remove(reactiveLight.light);
      this.reactiveLights.delete(name);
    }
  }

  setLightResponse(name, response) {
    const reactiveLight = this.reactiveLights.get(name);
    if (reactiveLight) {
      reactiveLight.response = response;
    }
  }

  setLightFrequency(name, frequency) {
    const reactiveLight = this.reactiveLights.get(name);
    if (reactiveLight) {
      reactiveLight.frequency = frequency;
    }
  }
}
```

---

## Part 5: Audio-Reactive Lighting Data Structure

### VFX Configuration

```javascript
// In vfxData.js
export const vfxEffects = {
  // Bass-reactive ambient glow
  ritualAmbientPulse: {
    type: "audioReactiveLight",
    lightType: "ambient",
    name: "ritualAmbient",
    color: "#440000",
    intensity: 0.5,
    frequency: "bass",
    response: 0.8,
    minIntensity: 0.2,
    maxIntensity: 1.5,
    smoothing: 0.7,
    colorReactive: true,
    peakColor: "#ff0000",
    criteria: {
      currentZone: "ritual_room",
      musicPlaying: true,
      currentState: RITUAL_ACTIVE
    }
  },

  // Mid-reactive candle flicker
  candleFlicker: {
    type: "audioReactiveLight",
    lightType: "point",
    name: "candle1",
    position: { x: 0, y: 1, z: 0 },
    color: "#ff8800",
    intensity: 1.0,
    distance: 5,
    frequency: "mid",
    response: 0.5,
    minIntensity: 0.7,
    maxIntensity: 1.5,
    smoothing: 0.6,
    criteria: {
      currentZone: "ritual_room",
      candleLit: true
    }
  },

  // Treble-reactive rune shimmer
  runeShimmer: {
    type: "audioReactiveLight",
    lightType: "point",
    name: "runeGlow",
    position: { x: 2, y: 0.5, z: -1 },
    color: "#ff00ff",
    intensity: 0.5,
    distance: 3,
    frequency: "treble",
    response: 1.0,
    minIntensity: 0.2,
    maxIntensity: 2.0,
    smoothing: 0.3,
    colorReactive: true,
    peakColor: "#00ffff",
    criteria: {
      runeActive: true,
      musicPlaying: true
    }
  },

  // Full-spectrum spotlight
  supernaturalSpotlight: {
    type: "audioReactiveLight",
    lightType: "spot",
    name: "ghostSpot",
    position: { x: 0, y: 5, z: 0 },
    target: { x: 0, y: 0, z: 0 },
    color: "#6644ff",
    intensity: 1.0,
    distance: 15,
    angle: Math.PI / 6,
    penumbra: 0.5,
    frequency: "all",
    response: 0.6,
    minIntensity: 0.5,
    maxIntensity: 3.0,
    smoothing: 0.5,
    anglePulse: 0.3,
    criteria: {
      currentState: SUPERNATURAL_EVENT,
      ghostVisible: true
    }
  }
};
```

### Audio-Reactive Light Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `type` | string | Yes | Must be "audioReactiveLight" |
| `lightType` | string | Yes | "point", "ambient", or "spot" |
| `name` | string | Yes | Unique identifier |
| `color` | string | Yes | Base light color (hex) |
| `intensity` | number | Yes | Base intensity |
| `frequency` | string | Yes | "bass", "mid", "treble", or "all" |
| `response` | number | Yes | How much light responds (0-1) |
| `minIntensity` | number | Yes | Minimum intensity |
| `maxIntensity` | number | Yes | Maximum intensity |
| `smoothing` | number | Yes | Smoothing factor (0-1) |
| `colorReactive` | boolean | No | Whether color changes with audio |
| `peakColor` | string | No | Color at peak audio |
| `distance` | number | Conditional | For point/spot lights |
| `angle` | number | Conditional | For spot lights |
| `position` | object | Conditional | Light position |
| `target` | object | Conditional | Spot light target |
| `criteria` | object | Yes | When light is active |

---

## Part 6: Practical Examples

### Example 1: Ritual Room Ambience

```javascript
class RitualRoomLighting {
  constructor(scene, audioManager) {
    this.scene = scene;
    this.audioManager = audioManager;
    this.lights = new Map();
  }

  setup() {
    // Bass-reactive ambient (room pulse)
    const ambientConfig = {
      type: "audioReactiveLight",
      lightType: "ambient",
      name: "ritualAmbient",
      color: "#330000",
      intensity: 0.3,
      frequency: "bass",
      response: 0.7,
      minIntensity: 0.1,
      maxIntensity: 1.0,
      smoothing: 0.8,
      colorReactive: true,
      peakColor: "#ff0000"
    };

    this.lights.set("ambient",
      this.audioManager.createReactiveLight(ambientConfig)
    );

    // Candle lights (mid-reactive)
    const candlePositions = [
      { x: -2, y: 1, z: 0 },
      { x: 2, y: 1, z: 0 },
      { x: 0, y: 1, z: -2 }
    ];

    candlePositions.forEach((pos, i) => {
      const candleConfig = {
        type: "audioReactiveLight",
        lightType: "point",
        name: `candle${i}`,
        position: pos,
        color: "#ff6600",
        intensity: 0.8,
        distance: 3,
        frequency: "lowMid",
        response: 0.4,
        minIntensity: 0.5,
        maxIntensity: 1.2,
        smoothing: 0.5
      };

      this.lights.set(`candle${i}`,
        this.audioManager.createReactiveLight(candleConfig)
      );
    });
  }

  update(deltaTime) {
    this.audioManager.update(deltaTime);
  }
}
```

### Example 2: Supernatural Event

```javascript
export const vfxEffects = {
  // Escalating supernatural lighting
  supernaturalStage1: {
    type: "audioReactiveLight",
    lightType: "ambient",
    name: "spookyAmbient",
    color: "#222233",
    intensity: 0.2,
    frequency: "bass",
    response: 0.3,
    minIntensity: 0.1,
    maxIntensity: 0.4,
    smoothing: 0.9,
    criteria: {
      currentState: SUPERNATURAL_BUILDING,
      eventProgress: { $gte: 0.0, $lt: 0.3 }
    }
  },

  supernaturalStage2: {
    type: "audioReactiveLight",
    lightType: "point",
    name: "spookyCenter",
    position: { x: 0, y: 2, z: 0 },
    color: "#4433aa",
    intensity: 1.0,
    distance: 10,
    frequency: "all",
    response: 0.6,
    minIntensity: 0.5,
    maxIntensity: 2.0,
    smoothing: 0.6,
    colorReactive: true,
    peakColor: "#aa00ff",
    criteria: {
      currentState: SUPERNATURAL_BUILDING,
      eventProgress: { $gte: 0.3, $lt: 0.7 }
    }
  },

  supernaturalStage3: {
    type: "audioReactiveLight",
    lightType: "spot",
    name: "climaxSpot",
    position: { x: 0, y: 8, z: 0 },
    target: { x: 0, y: 0, z: 0 },
    color: "#ff00ff",
    intensity: 2.0,
    distance: 20,
    angle: Math.PI / 4,
    penumbra: 0.8,
    frequency: "bass",
    response: 1.0,
    minIntensity: 1.0,
    maxIntensity: 5.0,
    smoothing: 0.3,
    colorReactive: true,
    peakColor: "#00ffff",
    anglePulse: 0.5,
    criteria: {
      currentState: SUPERNATURAL_CLIMAX,
      eventProgress: { $gte: 0.7 }
    }
  }
};
```

### Example 3: Music Visualizer Mode

```javascript
class MusicVisualizer {
  constructor(scene, audioManager) {
    this.scene = scene;
    this.audioManager = audioManager;
    this.lights = [];
  }

  createVisualizerLights(count = 16) {
    const radius = 5;
    const colors = [
      "#ff0000", "#ff4400", "#ff8800", "#ffcc00",
      "#ffff00", "#ccff00", "#88ff00", "#44ff00",
      "#00ff00", "#00ff44", "#00ff88", "#00ffcc",
      "#00ffff", "#00ccff", "#0088ff", "#0044ff"
    ];

    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      // Map each light to a frequency range
      const freqBin = Math.floor((i / count) * 256);  // FFT has 256 bins

      const config = {
        type: "audioReactiveLight",
        lightType: "point",
        name: `vizLight${i}`,
        position: { x, y: 1, z },
        color: colors[i % colors.length],
        intensity: 0,
        distance: 3,
        frequency: "custom",
        frequencyBin: freqBin,
        response: 2.0,
        minIntensity: 0,
        maxIntensity: 3.0,
        smoothing: 0.4
      };

      this.lights.push(
        this.audioManager.createReactiveLight(config)
      );
    }
  }
}
```

---

## Part 7: Advanced Techniques

### Beat Detection

```javascript
class BeatDetector {
  constructor(frequencyBinCount = 256) {
    this.energyHistory = [];
    this.historyLength = 43;  // ~1 second at 60fps
    this.beatThreshold = 1.3;  // Energy must be 1.3x average
    this.minBeatInterval = 0.25;  // Minimum 250ms between beats
    this.lastBeatTime = 0;
  }

  detectBeat(frequencyData, currentTime) {
    // Calculate average energy of bass frequencies
    let bassEnergy = 0;
    const bassBins = Math.floor(frequencyData.length * 0.1);  // Bottom 10%

    for (let i = 0; i < bassBins; i++) {
      bassEnergy += frequencyData[i];
    }
    bassEnergy /= bassBins;

    // Add to history
    this.energyHistory.push(bassEnergy);
    if (this.energyHistory.length > this.historyLength) {
      this.energyHistory.shift();
    }

    // Calculate average energy
    const averageEnergy = this.energyHistory.reduce((a, b) => a + b, 0) /
      this.energyHistory.length;

    // Check for beat
    const timeSinceLastBeat = currentTime - this.lastBeatTime;
    const isBeat = bassEnergy > averageEnergy * this.beatThreshold &&
      timeSinceLastBeat >= this.minBeatInterval;

    if (isBeat) {
      this.lastBeatTime = currentTime;
    }

    return {
      isBeat,
      energy: bassEnergy,
      averageEnergy,
      ratio: bassEnergy / averageEnergy
    };
  }
}

// Usage in light update
class BeatReactiveLight {
  constructor(light, config) {
    this.light = light;
    this.config = config;
    this.beatDetector = new BeatDetector();
    this.baseIntensity = config.intensity || 1;
    this.beatIntensity = config.beatIntensity || 3;
  }

  update(frequencyData, currentTime) {
    const beat = this.beatDetector.detectBeat(frequencyData, currentTime);

    if (beat.isBeat) {
      // Flash on beat
      this.light.intensity = this.beatIntensity;

      // Optional: Color flash
      if (this.config.beatColor) {
        const beatColor = new THREE.Color(this.config.beatColor);
        this.light.color.copy(beatColor);
      }
    } else {
      // Return to base
      this.light.intensity = THREE.MathUtils.lerp(
        this.light.intensity,
        this.baseIntensity,
        0.2
      );

      if (this.config.color) {
        const baseColor = new THREE.Color(this.config.color);
        this.light.color.lerp(baseColor, 0.1);
      }
    }
  }
}
```

### Audio-Reactive Emissive Materials

```javascript
// Make objects glow with audio
class AudioReactiveEmissive {
  constructor(mesh, config) {
    this.mesh = mesh;
    this.config = config;

    this.baseEmissive = new THREE.Color(config.baseEmissive || 0x000000);
    this.peakEmissive = new THREE.Color(config.peakEmissive || 0xffffff);
    this.baseIntensity = config.baseIntensity || 0;
    this.peakIntensity = config.peakIntensity || 1;

    // Ensure material has emissive
    if (!mesh.material.emissive) {
      mesh.material.emissive = new THREE.Color(0x000000);
    }
  }

  update(audioBands) {
    // Get audio value for frequency
    const value = this.getAudioValue(audioBands);
    const normalized = value / 255;

    // Interpolate emissive color
    this.mesh.material.emissive.lerpColors(
      this.baseEmissive,
      this.peakEmissive,
      normalized
    );

    // Set emissive intensity
    this.mesh.material.emissiveIntensity = THREE.MathUtils.lerp(
      this.baseIntensity,
      this.peakIntensity,
      normalized * this.config.response
    );
  }

  getAudioValue(audioBands) {
    switch (this.config.frequency) {
      case "bass": return audioBands.bass;
      case "mid": return audioBands.mid;
      case "treble": return audioBands.treble;
      default: return audioBands.bass;
    }
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Smoothing Too Low

```javascript
// ❌ WRONG: Jittery lights
{
  smoothing: 0.1  // Almost no smoothing
}
// Lights flicker nervously

// ✅ CORRECT: Smooth response
{
  smoothing: 0.7  // Smooth transitions
}
// Pleasing, natural movement
```

### 2. Response Too High

```javascript
// ❌ WRONG: Lights go crazy
{
  response: 2.0  // Too sensitive!
}
// Blinding flashes

// ✅ CORRECT: Balanced response
{
  response: 0.6  // Moderate reaction
}
// Noticeable but not overwhelming
```

### 3. Wrong Frequency for Effect

```javascript
// ❌ WRONG: High frequencies for ambient
{
  frequency: "treble",  // Wrong for room ambience
  lightType: "ambient"
}
// Fast, nervous flickering

// ✅ CORRECT: Bass for ambient
{
  frequency: "bass",  // Bass drives room feel
  lightType: "ambient"
}
// Slow, powerful pulses
```

### 4. Forgetting User Gesture Requirement

```javascript
// ❌ WRONG: Audio context won't start
async function init() {
  const context = new AudioContext();
  // This will be suspended!
}

// ✅ CORRECT: Require user interaction
async function init() {
  const context = new AudioContext();
  if (context.state === 'suspended') {
    await context.resume();
  }
}

// Or bind to user action
button.addEventListener('click', async () => {
  await audioSystem.initialize();
});
```

---

## Performance Considerations

```
AUDIO-REACTIVE LIGHTING COST:

Analyser overhead:
├── fftSize: 256   → Low cost (~1% CPU)
├── fftSize: 512   → Low cost (~2% CPU)
├── fftSize: 1024  → Medium cost (~4% CPU)
└── fftSize: 2048  → Higher cost (~8% CPU)

Light count:
├── 1-3:   Negligible
├── 4-10:  Minor impact
├── 11-20: Noticeable
└── 20+:   Significant

Update frequency:
├── Every frame:   Best quality, higher cost
├── Every 2 frames: Good balance
└── Every 4+ frames: Lower cost

Optimization tips:
├── Use fftSize 256 or 512 for mobile
├── Limit reactive lights to ~10
├── Share audio analyzer across all lights
├── Use smoothing to reduce calculations
└── Disable for distant/hidden zones
```

---

## 🎮 Game Design Perspective

### Audio-Reactive Lighting for Storytelling

```javascript
// Lighting intensity reflects narrative moment

Calm exploration:
{
  response: 0.3,  // Subtle
  frequency: "mid",
  smoothing: 0.9
  // Gentle background presence
}

Building tension:
{
  response: 0.6,
  frequency: "bass",
  smoothing: 0.6,
  colorReactive: true
  // Music starts to drive visuals
}

Climax:
{
  response: 1.0,
  frequency: "all",
  smoothing: 0.2,
  beatDetection: true
  // Audio completely controls atmosphere
}
```

### Emotional Color Mapping

```javascript
// Different frequencies evoke different emotions

Bass (power, danger):
{
  frequency: "bass",
  color: "#ff0000",  // Red
  response: 0.8
  // Threat, power, intensity
}

Mid (emotion, voice):
{
  frequency: "mid",
  color: "#ff8800",  // Orange
  response: 0.5
  // Warmth, humanity, emotion
}

Treble (magic, mystery):
{
  frequency: "treble",
  color: "#00ffff",  // Cyan
  response: 0.6
  // Magic, ethereal, supernatural
}
```

---

## Next Steps

Now that you understand audio-reactive lighting:

- [VFXManager](./vfx-manager.md) - Base visual effects system
- [Selective Bloom](./selective-bloom.md) - Glowing highlights
- [Splat Fractal Effect](./splat-fractal-effect.md) - Procedural patterns
- [MusicManager](../05-media-systems/music-manager.md) - Music playback system

---

## References

- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) - Audio analysis and playback
- [AnalyserNode](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode) - FFT analysis
- [Three.js Lights](https://threejs.org/docs/#api/en/lights/Light) - Light types and properties
- [Audio Visualization](https://www.youtube.com/watch?v=qi6nXXbA-mdI) - Coding Train tutorial
- [FFT Explained](https://www.youtube.com/watch?v=nM13LFJboZ4) - Fourier Transform visualization

*Documentation last updated: January 12, 2026*
