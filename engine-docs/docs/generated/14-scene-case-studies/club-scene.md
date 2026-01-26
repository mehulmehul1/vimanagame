# Scene Case Study: The Club

## 🎬 Scene Overview

**Location**: Nightclub environment, likely the final major interior space
**Narrative Context**: The climax setting—a space of sensory overload where audio-reactive technology creates an overwhelming, hypnotic atmosphere
**Player Experience: Awe → Disorientation → Hypnotic trance → Climactic realization

The Club scene represents the pinnacle of the engine's audio-reactive capabilities. This is where the Gaussian Splatting environment, WebGPU rendering, and audio analysis systems converge to create a space that feels alive—a club where lights pulse to music, shadows dance, and the environment responds to every sound. This scene demonstrates how to create immersive, responsive spaces that blur the line between game world and musical experience.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create sensory overload—a space so overwhelming that player loses track of reality, becoming immersed in the rhythmic, hypnotic atmosphere.

**Why This Scene Matters**:

```
THE CLUB AS CLIMAX SPACE:

Previous Scenes → Club → [Resolution]
     ↓              ↓            ↓
Exploration   Sensory    Narrative
Discovery    Overload   Climax

THE CLUB IS:
- Visual climax: Most impressive lighting
- Audio climax: Most immersive soundscape
- Interaction climax: All systems working together
- Technical showcase: What the engine can do

PLAYER EXPERIENCE:
"I've never seen anything like this.
 The lights, the music, everything moves together.
 I'm part of this experience."
```

### Design Philosophy

**1. Audio-Reactive Environment**

Every visual element responds to audio:

```
AUDIO-VISUAL SYNC HIERARCHY:

Beat Detection (BPM):
├── Strobe lights (on kick drum)
├── Floor pulses (on snare)
├── Ceiling spots (on chorus)
└── Effect: Rhythmic, driving

Frequency Analysis:
├── Bass → Floor vibration
├── Mids → Main color washes
├── Highs → Accent lights
└── Effect: Full spectrum response

Amplitude Response:
├── Quiet → Dim, intimate
├── Loud → Bright, intense
├── Build-ups → Increasing intensity
├── Drops → Explosive visuals
└── Effect: Dynamic, emotional

SPECIAL MOMENTS:
├── Beat drop → Everything peaks
├── Breakdown → Stripped back visuals
├── Build → Anticipation through effects
└── Effect: Musical structure reflected visually
```

**2. Spatial Design for Movement**

```
CLUB LAYOUT PSYCHOLOGY:

Entrance Zone:
├── Transition from exterior/previous scene
├── Volume gradually increases
├── Lighting hints at what's ahead
└── Purpose: Prepare player for overload

Main Floor:
├── Open space for movement
├── Visual focus on DJ booth/stage
├── Surround lighting (360° experience)
└── Purpose: Immersion, freedom of movement

VIP Areas:
├── Elevated, separated
├── Different lighting (more intimate)
├── May contain narrative elements
└── Purpose: Optional exploration

Bar Areas:
├── Side spaces, break from main floor
├── Interactive elements (drinks, NPCs)
├── Different audio mix (quieter)
└── Purpose: Pacing variety, detail discovery

Exit/Back Areas:
├── Path to final narrative moments
├── Transition to resolution
└── Purpose: Bridge to ending
```

**3. Progressive Intensity**

```
SENSORY PROGRESSION:

Entry (0-30 seconds):
├── Dim ambient lighting
├── Muffled audio from outside
├── Door opens → wave of sound/light
└── Player feels: Curiosity, anticipation

Initial Exposure (30-90 seconds):
├── Full club revealed
├── Audio at full volume
├── Lighting pulsing steadily
├── Player acclimating to intensity
└── Player feels: Awe, slight overwhelm

Exploration (90 seconds - 5 minutes):
├── Player moves through space
├── Different areas revealed
├── Narrative elements discovered
├── Audio-reactive systems experienced
└── Player feels: Immersed, engaged

Climactic Sequence (triggered by narrative):
├── Music peaks
├── All systems maximum
├── Visual transcendence
├── Reality breakdown begins
└── Player feels: Ecstatic, then transcendent

Resolution (after peak):
├── Systems gradually dim
├── Audio fades or transforms
├── Transition to final scene
└── Player feels: Changed, completed
```

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
                    CLUB LAYOUT DIAGRAM:

    ╔══════════════════════════════════════════════════════╗
   ║  [ENTRANCE FOYER]                                    ║
   ║  - Transition space                                   ║
   ║  - Coat check (visual detail)                         ║
   ║  - Sound gradually increases                          ║
   ║       ↓                                               ║
   ║  ╔══════════════════════════════════════════════════╗  ║
   ║  ║              MAIN CLUB FLOOR                      ║  ║
   ║  ║                                                  ║  ║
   ║  ║  [BAR AREA]           [VIP MEZZANINE]            ║  ║
   ║  ║  ┌──────────┐         ┌─────────────────┐        ║  ║
   ║  ║  │          │         │    (Elevated)    │        ║  ║
   ║  ║  │  Drinks  │         │    Seating      │        ║  ║
   ║  ║  │  NPC     │         │    Private      │        ║  ║
   ║  ║  │          │         │    Access       │        ║  ║
   ║  ║  └──────────┘         └─────────────────┘        ║  ║
   ║  ║       ↑                    ↑                     ║  ║
   ║  ║       │                    │                     ║  ║
   ║  ║    ╔══════════════════════════════════╗          ║  ║
   ║  ║    ║                                    ║          ║  ║
   ║  ║    ║      DANCE FLOOR (Open Space)      ║          ║  ║
   ║  ║    ║                                    ║          ║  ║
   ║  ║    ║    ╔════════════════════════╗      ║          ║  ║
   ║  ║    ║    ║   DJ BOOTH / STAGE    ║      ║          ║  ║
   ║  ║    ║    ║   (Focal Point)       ║      ║          ║  ║
   ║  ║    ║    ║   Audio-Reactive      ║      ║          ║  ║
   ║  ║    ║    ║   Centerpiece         ║      ║          ║  ║
   ║  ║    ║    ╚════════════════════════╝      ║          ║  ║
   ║  ║    ║                                    ║          ║  ║
   ║  ║    ╚════════════════════════════════════╝          ║  ║
   ║  ║                                                  ║  ║
   ║  ║  [BOOTHS] ← Seating areas along walls              ║  ║
   ║  ║  - Private feeling                                 ║  ║
   ║  ║  - Individual lighting controls                    ║  ║
   ║  ║  - Some contain narrative clues                    ║  ║
   ║  ║                                                  ║  ║
   ║  ╚══════════════════════════════════════════════════╝  ║
   ║       ↓                                               ║
   ║  [EXIT / BACK ROOMS] → Lead to resolution             ║
    ╚══════════════════════════════════════════════════════╝

LIGHTING ZONES:

Dance Floor:
├── Overhead array (moving spots)
├── Floor panels (LED, pressure-activated)
├── Perimeter lasers
└── Audio-reactive centerpieces

Bar Area:
├── Warm, dimmer lighting
├── Accent lights on bottles
├── Under-counter glow
└── More relaxed, conversation-friendly

VIP Mezzanine:
├── Exclusive color palette
├── Chandeliers or premium fixtures
├── Private atmosphere
└── Elevated perspective

Booths:
├── Individual color controls
├── Dimmable, intimate
├── Some may have special effects
└── Hidden narrative content
```

### Audio-Reactive Lighting System

```
AUDIO RESPONSE ARCHITECTURE:

Input: Web Audio API Analyser
├── FFT (Fast Fourier Transform)
│   ├── Frequency data (spectrum)
│   └── Used for: Color selection, intensity
├── Time-domain data (waveform)
│   ├── Peak detection (beats)
│   └── Used for: Strobes, pulses
└── BPM detection
    ├── Tempo analysis
    └── Used for: Sync speed, effect timing

Processing:
├── Band separation
│   ├── Bass (20-250 Hz)
│   ├── Mids (250-4000 Hz)
│   └── Highs (4000+ Hz)
├── Beat detection
│   ├── Kick (low peaks)
│   ├── Snare (mid peaks)
│   └── Onset detection
└── Feature extraction
    ├── Amplitude envelope
    ├── Frequency centroid
    └── Spectral flux

Output: Light Controls
├── Intensity modulation
├── Color shifting
├── Position/rotation (moving lights)
├── Effect triggers (strobes, etc.)
└── synchronized across all fixtures
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the Club implementation, you should know:
- **Audio Analysis**: FFT, frequency bands, beat detection
- **DMX/LED Control**: Light fixture control systems
- **Shader Programming**: GPU-based audio-reactive effects
- **Particle Systems**: Audio-driven particle effects
- **Spatial Audio**: Surround sound positioning

### Scene Data Structure

```javascript
// SceneData.js - Club configuration
export const SCENES = {
  club: {
    id: 'club',
    name: 'The Club',
    type: 'interior_club',

    // Club splat
    splat: {
      file: '/assets/splats/club.ply',
      settings: {
        renderScale: 1.0,
        splatSize: 1.2,
        // Higher opacity for rich atmosphere
        opacity: 1.0
      }
    },

    // Entry configuration
    entryPoint: {
      position: { x: 0, y: 1.7, z: 15 },
      rotation: { x: 0, y: Math.PI, z: 0 },
      transition: 'gradual_reveal',
      duration: 3.0
    },

    // Audio-reactive lighting system
    lighting: {
      // Base lighting (minimum level)
      base: {
        ambient: { color: 0x111122, intensity: 0.1 },
        floor: { color: 0x0a0a15, intensity: 0.2 }
      },

      // Audio-reactive fixtures
      fixtures: [
        // Main dance floor array
        {
          id: 'dance_floor_array',
          type: 'moving_spot',
          count: 8,
          positions: [
            { x: -4, y: 4, z: 0 },
            { x: -2, y: 4, z: 0 },
            { x: 0, y: 4.5, z: 0 },
            { x: 2, y: 4, z: 0 },
            { x: 4, y: 4, z: 0 },
            // ... more fixtures
          ],
          audioReactive: {
            enabled: true,
            frequency: 'bass',  // Respond to bass
            minIntensity: 0.2,
            maxIntensity: 2.0,
            colorMode: 'spectrum',  // Color based on frequency
            movementSpeed: 1.5
          }
        },

        // Strobe lights (beat detection)
        {
          id: 'strobe_array',
          type: 'strobe',
          count: 4,
          positions: [
            { x: -3, y: 3.5, z: -2 },
            { x: 3, y: 3.5, z: -2 },
            { x: 0, y: 4, z: 3 },
            { x: 0, y: 3, z: -3 }
          ],
          audioReactive: {
            enabled: true,
            trigger: 'kick',  // Fire on kick drum
            flashDuration: 0.05,  // 50ms flash
            maxFlashRate: 10  // Max 10 flashes/second
          }
        },

        // Floor panels
        {
          id: 'floor_panels',
          type: 'panel_array',
          layout: 'grid',
          dimensions: { x: 8, z: 8 },
          panelSize: 0.5,
          audioReactive: {
            enabled: true,
            frequency: 'full',  // Full spectrum
            mode: 'wave',  // Wave propagation from center
            responseSpeed: 0.9  // Fast response
          }
        },

        // Lasers
        {
          id: 'laser_array',
          type: 'laser',
          count: 4,
          audioReactive: {
            enabled: true,
            frequency: 'highs',  // Respond to highs
            pattern: 'audio_modulated'
          }
        }
      ],

      // Volumetric fog
      fog: {
        enabled: true,
        color: 0x220033,
        density: 0.03,
        animated: true
      }
    },

    // Audio system
    audio: {
      // Main music track
      music: {
        track: 'club_main_mix',
        bpm: 128,
        looping: true,
        layers: [
          { name: 'drums', file: 'club_drums.ogg' },
          { name: 'bass', file: 'club_bass.ogg' },
          { name: 'synth', file: 'club_synths.ogg' },
          { name: 'vocals', file: 'club_vocals.ogg' }
        ]
      },

      // Audio analysis configuration
      analysis: {
        fftSize: 2048,
        smoothing: 0.8,
        frequencyBands: {
          bass: { min: 20, max: 250 },
          lowMid: { min: 250, max: 500 },
          mid: { min: 500, max: 2000 },
          highMid: { min: 2000, max: 4000 },
          highs: { min: 4000, max: 20000 }
        },
        beatDetection: {
          threshold: 0.3,
          minInterval: 0.1  // Minimum 100ms between beats
        }
      },

      // Spatial positioning
      spatial: {
        system: 'surround',
        channels: 7.1,  // 7.1 surround
        positioning: '3d'
      }
    },

    // Interactive elements
    interactables: [
      {
        id: 'dj_booth',
        type: 'examine',
        position: { x: 0, y: 1, z: -4 },
        interaction: 'can_approach',
        narrative: 'DJ or automated system creating music'
      },
      {
        id: 'club_bar',
        type: 'interactive',
        position: { x: -6, y: 1, z: 0 },
        interactions: [
          { action: 'order_drink', result: 'npc_interaction' },
          { action: 'examine', result: 'bar_description' }
        ]
      },
      {
        id: 'vip_access',
        type: 'transition',
        position: { x: 2, y: 1, z: -8 },
        locked: true,
        unlockCriteria: 'narrative_progression',
        leadsTo: 'vip_area'
      }
    ],

    // Post-processing
    postProcessing: {
      bloom: {
        enabled: true,
        threshold: 0.6,
        strength: 1.5,
        radius: 0.8
      },
      chromaticAberration: {
        enabled: true,
        intensity: 0.01
      },
      filmGrain: {
        enabled: true,
        intensity: 0.1
      },
      motionBlur: {
        enabled: true,
        intensity: 0.3
      }
    }
  }
};
```

### Audio-Reactive Lighting Manager

```javascript
// AudioReactiveLighting.js - Manages audio-responsive lights
class AudioReactiveLighting {
  constructor(audioManager, sceneManager) {
    this.audio = audioManager;
    this.scene = sceneManager;

    this.fixtures = new Map();
    this.analyser = null;
    this.frequencyData = null;
    this.timeData = null;

    this.beatThreshold = 0.3;
    this.lastBeatTime = 0;
    this.minBeatInterval = 100;  // ms
  }

  initialize() {
    // Set up audio analyser
    this.analyser = this.audio.createAnalyser({
      fftSize: 2048,
      smoothingTimeConstant: 0.8
    });

    this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
    this.timeData = new Uint8Array(this.analyser.frequencyBinCount);

    // Create light fixtures
    this.createFixtures();

    // Start update loop
    this.startUpdate();
  }

  createFixtures() {
    const clubConfig = SCENES.club.lighting.fixtures;

    for (const fixtureConfig of clubConfig.fixtures) {
      const fixture = this.createFixture(fixtureConfig);
      this.fixtures.set(fixtureConfig.id, fixture);
    }
  }

  createFixture(config) {
    switch (config.type) {
      case 'moving_spot':
        return this.createMovingSpot(config);
      case 'strobe':
        return this.createStrobe(config);
      case 'panel_array':
        return this.createPanelArray(config);
      case 'laser':
        return this.createLaser(config);
      default:
        return null;
    }
  }

  createMovingSpot(config) {
    const spots = [];

    for (const pos of config.positions) {
      // Create spotlight
      const spot = new THREE.SpotLight(0xffffff, 0);
      spot.position.set(pos.x, pos.y, pos.z);
      spot.angle = Math.PI / 6;
      spot.penumbra = 0.3;
      spot.decay = 2;
      spot.distance = 20;
      spot.castShadow = false;

      // Create target helper (for rotation)
      const target = new THREE.Object3D();
      target.position.set(pos.x, 0, pos.z);
      spot.target = target;

      this.scene.add(spot);
      this.scene.add(target);

      spots.push({
        light: spot,
        target: target,
        basePosition: new THREE.Vector3(pos.x, pos.y, pos.z),
        baseIntensity: config.audioReactive.minIntensity,
        maxIntensity: config.audioReactive.maxIntensity,
        colorMode: config.audioReactive.colorMode,
        movementSpeed: config.audioReactive.movementSpeed,
        angle: 0,
        lastColor: new THREE.Color()
      });
    }

    return { type: 'moving_spot', spots: spots };
  }

  createStrobe(config) {
    const strobes = [];

    for (const pos of config.positions) {
      const strobe = new THREE.PointLight(0xffffff, 0, 10);
      strobe.position.set(pos.x, pos.y, pos.z);

      this.scene.add(strobe);

      strobes.push({
        light: strobe,
        position: new THREE.Vector3(pos.x, pos.y, pos.z),
        flashDuration: config.audioReactive.flashDuration,
        maxFlashRate: config.audioReactive.maxFlashRate,
        lastFlash: 0
      });
    }

    return { type: 'strobe', strobes: strobes };
  }

  createPanelArray(config) {
    const panels = [];
    const { x: xCount, z: zCount } = config.dimensions;
    const size = config.panelSize;

    // Create grid of panels
    for (let x = 0; x < xCount; x++) {
      for (let z = 0; z < zCount; z++) {
        const panelX = (x - xCount / 2) * size;
        const panelZ = (z - zCount / 2) * size;

        // Create panel light (area light approximation)
        const panel = new THREE.PointLight(0xff00ff, 0, size * 1.5);
        panel.position.set(panelX, 0.01, panelZ);

        // Create panel mesh
        const geometry = new THREE.PlaneGeometry(size * 0.95, size * 0.95);
        const material = new THREE.MeshBasicMaterial({
          color: 0xff00ff,
          side: THREE.DoubleSide
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(panelX, 0, panelZ);

        this.scene.add(panel);
        this.scene.add(mesh);

        panels.push({
          light: panel,
          mesh: mesh,
          basePosition: { x: panelX, z: panelZ },
          distanceFromCenter: Math.sqrt(panelX * panelX + panelZ * panelZ)
        });
      }
    }

    return { type: 'panel_array', panels: panels };
  }

  createLaser(config) {
    const lasers = [];

    for (let i = 0; i < config.count; i++) {
      // Laser using thin cylinder with emissive material
      const geometry = new THREE.CylinderGeometry(0.005, 0.005, 20, 8);
      const material = new THREE.MeshBasicMaterial({
        color: 0xff0000,
        transparent: true,
        opacity: 0.8
      });
      const laser = new THREE.Mesh(geometry, material);
      laser.position.set(0, 10, 0);

      this.scene.add(laser);

      lasers.push({
        mesh: laser,
        baseRotation: (i / config.count) * Math.PI * 2,
        speed: 0.5
      });
    }

    return { type: 'laser', lasers: lasers };
  }

  startUpdate() {
    const update = () => {
      this.update();
      requestAnimationFrame(update);
    };
    update();
  }

  update() {
    // Get audio data
    this.analyser.getByteFrequencyData(this.frequencyData);
    this.analyser.getByteTimeDomainData(this.timeData);

    // Detect beat
    const beatDetected = this.detectBeat();

    // Get frequency band values
    const bands = this.getFrequencyBands();

    // Update each fixture type
    for (const [id, fixture] of this.fixtures) {
      this.updateFixture(fixture, bands, beatDetected);
    }
  }

  detectBeat() {
    // Simple beat detection using time domain data
    let sum = 0;
    for (let i = 0; i < this.timeData.length; i++) {
      sum += Math.abs(this.timeData[i] - 128);
    }
    const average = sum / this.timeData.length;

    // Check if threshold exceeded and minimum time passed
    const now = Date.now();
    if (average > this.beatThreshold &&
        now - this.lastBeatTime > this.minBeatInterval) {
      this.lastBeatTime = now;
      return true;
    }
    return false;
  }

  getFrequencyBands() {
    // Calculate average value for each frequency band
    const bufferLength = this.frequencyData.length;
    const nyquist = this.audio.context.sampleRate / 2;
    const binWidth = nyquist / bufferLength;

    const bands = {
      bass: this.getBandAverage(20, 250, bufferLength, binWidth),
      lowMid: this.getBandAverage(250, 500, bufferLength, binWidth),
      mid: this.getBandAverage(500, 2000, bufferLength, binWidth),
      highMid: this.getBandAverage(2000, 4000, bufferLength, binWidth),
      highs: this.getBandAverage(4000, 20000, bufferLength, binWidth)
    };

    return bands;
  }

  getBandAverage(minFreq, maxFreq, bufferLength, binWidth) {
    let sum = 0;
    let count = 0;

    const minBin = Math.floor(minFreq / binWidth);
    const maxBin = Math.ceil(maxFreq / binWidth);

    for (let i = minBin; i <= maxBin && i < bufferLength; i++) {
      sum += this.frequencyData[i];
      count++;
    }

    return count > 0 ? sum / count / 255 : 0;  // Normalize to 0-1
  }

  updateFixture(fixture, bands, beat) {
    switch (fixture.type) {
      case 'moving_spot':
        this.updateMovingSpot(fixture, bands, beat);
        break;
      case 'strobe':
        this.updateStrobe(fixture, beat);
        break;
      case 'panel_array':
        this.updatePanelArray(fixture, bands);
        break;
      case 'laser':
        this.updateLaser(fixture, bands.highs);
        break;
    }
  }

  updateMovingSpot(fixture, bands, beat) {
    for (const spot of fixture.spots) {
      // Intensity based on bass
      const bassValue = bands.bass;
      const intensity = spot.baseIntensity +
        (spot.maxIntensity - spot.baseIntensity) * bassValue;
      spot.light.intensity = intensity;

      // Color based on frequency
      if (spot.colorMode === 'spectrum') {
        const color = this.frequencyToColor(bands);
        spot.light.color.lerp(color, 0.2);
      }

      // Movement
      spot.angle += 0.01 * spot.movementSpeed;
      const targetX = spot.basePosition.x + Math.sin(spot.angle) * 2;
      const targetZ = spot.basePosition.z + Math.cos(spot.angle * 0.7) * 2;
      spot.target.position.set(targetX, 0, targetZ);
    }
  }

  updateStrobe(fixture, beat) {
    const now = Date.now();

    for (const strobe of fixture.strobes) {
      // Check if we can flash
      if (beat && now - strobe.lastFlash > (1000 / strobe.maxFlashRate)) {
        strobe.lastFlash = now;
        strobe.light.intensity = 5;

        // Schedule turn off
        setTimeout(() => {
          strobe.light.intensity = 0;
        }, strobe.flashDuration * 1000);
      }
    }
  }

  updatePanelArray(fixture, bands) {
    // Create wave pattern based on audio
    const time = Date.now() / 1000;
    const overallLevel = bands.bass * 0.5 + bands.mid * 0.3 + bands.highs * 0.2;

    for (const panel of fixture.panels) {
      // Wave propagation from center
      const wave = Math.sin(
        panel.distanceFromCenter * 0.5 - time * 3
      ) * 0.5 + 0.5;

      // Audio modulation
      const audioMod = overallLevel;

      // Combined intensity
      const intensity = wave * audioMod * 2;

      // Color based on frequency mix
      const hue = (bands.highs * 0.6 + bands.mid * 0.3 + bands.bass * 0.1);
      const color = new THREE.Color().setHSL(hue, 1, 0.5);

      panel.light.intensity = intensity * 2;
      panel.light.color.copy(color);
      panel.mesh.material.color.copy(color);
    }
  }

  updateLaser(fixture, highsValue) {
    for (const laser of fixture.lasers) {
      // Rotate based on high frequencies
      laser.baseRotation += laser.speed * (0.5 + highsValue);

      laser.mesh.rotation.y = laser.baseRotation;
      laser.mesh.rotation.z = Math.sin(laser.baseRotation * 2) * 0.2;

      // Intensity based on highs
      laser.mesh.material.opacity = 0.3 + highsValue * 0.7;
    }
  }

  frequencyToColor(bands) {
    // Map frequency bands to color
    // Bass = red, Mids = green, Highs = blue
    const r = bands.bass;
    const g = bands.mid;
    const b = bands.highs;

    return new THREE.Color(r, g, b);
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Define the Club's Purpose

```
CLUB DESIGN BRIEF:

1. Narrative role?
    Club: Climactic setting, technical showcase,
           emotional peak before resolution

2. What should player feel?
    Club: Awe, immersion, hypnotic trance,
           then transcendence

3. What's the key feature?
    Club: Audio-reactive environment—everything
           responds to music

4. How long is player here?
    Club: Extended exploration (5-10 minutes)
           with climactic sequence

5. Exit condition?
    Club: Narrative event triggers resolution,
           or player finds exit after exploration
```

### Step 2: Design the Audio-Reactive System

```javascript
// Audio-reactive system architecture:

const audioReactiveSystem = {
  // Input
  audioSource: {
    type: 'music_track',
    bpm: 128,
    analysis: 'fft'
  },

  // Processing
  analysis: {
    fftSize: 2048,
    frequencyBands: ['bass', 'mids', 'highs'],
    beatDetection: 'peak_threshold',
    continuousAnalysis: true
  },

  // Output mapping
  lighting: {
    bass: 'intensity + main_color',
    mids: 'secondary_colors + movement',
    highs: 'accents + strobe',
    beat: 'special_effects + triggers'
  },

  // Visual effects
  effects: {
    bloom: 'audio_modulated',
    fog: 'color_shifts',
    particles: 'emit_on_beats'
  }
};
```

### Step 3: Create Spatial Hierarchy

```javascript
// Zone design for pacing:

const clubZones = [
  {
    name: 'foyer',
    purpose: 'transition',
    intensity: 'low',
    audio: 'muffled',
    lighting: 'dim',
    playerState: 'anticipation'
  },
  {
    name: 'main_floor',
    purpose: 'primary_experience',
    intensity: 'high',
    audio: 'full',
    lighting: 'audio_reactive_full',
    playerState: 'immersed'
  },
  {
    name: 'bar',
    purpose: 'break_space',
    intensity: 'medium',
    audio: 'reduced',
    lighting: 'warm',
    playerState: 'exploring'
  },
  {
    name: 'vip',
    purpose: 'optional_narrative',
    intensity: 'medium_low',
    audio: 'clear',
    lighting: 'exclusive',
    playerState: 'discovering'
  }
];
```

---

## 🔧 Variations For Your Game

### Variation 1: Empty Club

```javascript
const emptyClub = {
  // Club without people, music still playing
  atmosphere: 'haunting',

  // What changes:
  lights: 'still_active_but_feel_wrong',
  music: 'playing_but_source_unknown',
  narrative: 'something happened here'
};
```

### Variation 2: Crowded Club

```javascript
const crowdedClub = {
  // Full of NPC clubgoers
  population: 'dense',

  // NPC behavior:
  dancing: true,
  reactingToMusic: true,
  playerCanInteract: true,
  performanceImpact: 'high'
};
```

### Variation 3: Retro Club

```javascript
const retroClub = {
  // Different era aesthetic
  era: '1980s',

  // Visual style:
  neon: 'abundant',
  fixtures: 'retro',
  music: 'synth_wave',
  colors: 'cyan_magenta_pink'
};
```

---

## Performance Considerations

```
CLUB PERFORMANCE:

Audio Analysis:
├── FFT every frame
├── 2048 bins = moderate CPU
├── Can reduce for mobile
└── Target: 60 FPS with analysis

Lighting:
├── Many fixtures = many draw calls
├── Consider instancing for similar lights
├── Limit shadow casting lights
├── Use baked lighting where possible
└── Target: Optimized fixture count

Post-Processing:
├── Bloom is expensive
├── Motion blur adds GPU load
├── Quality settings important
└── Target: Quality slider

Particles:
├── Audio-driven emit = many particles
├── Use pool, limit max count
├── Consider LOD for distance
└── Target: Stable particle count

RECOMMENDATION:
Club is performance-heavy.
Test on minimum spec,
provide quality options.
```

---

## Common Mistakes Beginners Make

### 1. Too Much Strobe

```javascript
// ❌ WRONG: Constant strobing
// Seizure risk, visually painful

// ✅ CORRECT: Rhythmic, deliberate strobe
// Synced to beats, breaks between
```

### 2. All Lights Reacting Identically

```javascript
// ❌ WRONG: Everything pulses together
// Boring, no visual interest

// ✅ CORRECT: Different response patterns
// Bass → floor, highs → accents, etc.
```

### 3. No Visual Hierarchy

```javascript
// ❌ WRONG: Everything equally bright
// Player doesn't know where to look

// ✅ CORRECT: Clear focal points
// DJ booth/stage is brightest
```

### 4. Music Doesn't Match Visuals

```javascript
// ❌ WRONG: Slow music with frantic lights
// Disconnect breaks immersion

// ✅ CORRECT: Visual tempo matches music
// Everything feels synchronized
```

---

## Related Systems

- [Audio-Reactive Lighting](../07-visual-effects/audio-reactive-lighting.md) - Lighting system
- [MusicManager](../05-media-systems/music-manager.md) - Audio system
- [VFXManager](../07-visual-effects/vfx-manager.md) - Effects
- [Post-Processing](../07-visual-effects/selective-bloom.md) - Bloom effects

---

## Source File Reference

**Scene Data**:
- `content/SceneData.js` - Club configuration

**Managers**:
- `managers/AudioReactiveLighting.js` - Audio-responsive lights
- `managers/ClubManager.js` - Club-specific logic

**Assets**:
- `assets/splats/club.ply` - Club splat
- `assets/audio/club_main_mix.ogg` - Music track

---

## 🧠 Creative Process Summary

**From Concept to Club Scene**:

```
1. CLIMAX NEED
   "Player needs overwhelming experience"

2. AUDIO-FIRST DESIGN
   "Music drives everything"

3. REACTIVE SYSTEMS
   "Every visual responds to audio"

4. SPATIAL HIERARCHY
   "Zones for different experiences"

5. PROGRESSIVE INTENSITY
   "Entry → immersion → climax → resolution"

6. TECHNICAL SHOWCASE
   "Demonstrate engine capabilities"

7. EMOTIONAL PEAK
   "Player feels part of something larger"
```

---

## References

- [Audio-Reactive Art](https://www.youtube.com/watch?v=M4skP6bN_Ks) - Tutorial series
- [Web Audio API](https://webaudioapi.com/book/) - Complete reference
- [Club Lighting Design](https://www.cljlighting.com/) - Industry techniques
- [Shader Audio Reactive](https://www.shadertoy.com/) - Shader examples

*Documentation last updated: January 12, 2026*
