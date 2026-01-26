# Scene Case Study: The Edison Cylinder

## 🎬 Scene Overview

**Location**: Interior office space, typically on or near a desk
**Narrative Context: A vintage audio playback device—the Edison cylinder phonograph—creates an anachronistic moment, playing recorded content that reveals narrative through period-authentic technology**
**Player Experience: Curiosity → Discovery → Immersion in the past → Understanding

The Edison Cylinder scene combines physical interaction with historical atmosphere. Unlike digital audio that players take for granted, the cylinder phonograph represents a tangible, mechanical way of experiencing sound—a ritualistic process that makes each playback feel meaningful. This scene demonstrates how to create atmospheric interactions using period technology, where the mechanical nature of playback becomes part of the experience.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create nostalgia for a time player never knew—the ritual of mechanical audio playback creates connection to the past.

**Why This Technology Matters**:

```

DIGITAL VS ANALOG AUDIO EXPERIENCE:

Digital (What We're Used To):
├── Instant playback
├── Perfect quality
├── No effort required
├── Infinite repeat
└─→ CONVENIENT, MEANINGLESS

Edison Cylinder (What This Scene Offers):
├── Manual process to play
├── Imperfect, degraded audio
├── Ritualistic interaction
├── Each playback feels special
└─→ MEANINGFUL, IMMERSIVE

THE CYLINDER IS:
- A window to the past
- A physical object to interact with
- A storytelling device with character
- An anachronism that raises questions
```

### Design Philosophy

**1. The Ritual of Playback**

```

INTERCTION SEQUENCE:

APPROACH:
├── Player sees phonograph on desk
├── Vintage appearance draws curiosity
├── Cylinder visible (what is it?)
└─→ "What does this do?"

INITIAL INTERACTION:
├── Player examines the device
├── Description explains its function
├── Prompt: "Insert cylinder to play"
└─→ "I need to find a cylinder"

FIND CYLINDER:
├── Cylinder located nearby
├── Physical object to pick up
├── Carried to phonograph
└─→ Discovery quest

INSERT CYLINDER:
├── Manual placement animation
├── Satisfying mechanical feedback
├── Player feels they accomplished something
└─→ Ritual complete

PLAYBACK:
├── Hand crank or automatic start
├── Mechanical whirring sounds
├── Audio degrades over time (period feel)
├── Content reveals narrative
└─→ Immersion in the moment
```

**2. Period Authenticity**

```

HISTORICAL ACCURACY AS ATMOSPHERE:

Visual Design:
├── Real phonograph proportions
├── Materials: brass, wood, enamel
├── Wear: scratches, patina, dust
├─→ Looks like actual antique

Audio Design:
├── Real cylinder recording (or simulation)
├── Surface noise, crackle
├── Limited frequency range
├── Volume fluctuations
├─→ Sounds like period recording

Mechanical Animation:
├── Rotating cylinder
├── Tone arm movement
├── Governor spinning
├── Belt/Drive motion
├─→ Feels alive, mechanical

RESULT:
Player believes this is real.
That belief makes content more impactful.
```

---

## 🎨 Level Design Breakdown

### Object Design

```

                    EDISON PHONOGRAPH DESIGN:

        ╔═══════════════════════════════╗
        ║         [HORN/SPEAKER]         ║
        ║              ║                  ║
        ║             ╱                   ║
        ║            ╱                    ║
        ║           ╱  BRASS              ║
        ║          ╱                      ║
        ║    ┌─────────────────────┐      ║
        ║    │    [GOVERNOR]      │      ║
        ║    │     (spinning)      │      ║
        ║    └─────────────────────┘      ║
        ║           ↓                      ║
        ║    ╔═════════════════════╗       ║
        ║    ║   [TONE ARM]       ║       ║
        ║    ║    →              ║       ║
        ║    ║   [NEEDLE]        ║       ║
        ║    ║    ↓              ║       ║
        ║    ╚═════════════════════╝       ║
        ║           ↓                      ║
        ║    ┌─────────────────────┐      ║
        ║    │   [CYLINDER]       │      ║
        ║    │   (rotating)       │      ║
        ║    │   ╔══════════╗      │      ║
        ║    │   ║ Recording ║      │      ║
        ║    │   ╚══════════╝      │      ║
        ║    └─────────────────────┘      ║
        ║                                  ║
        ║    [BASE/WOOD BOX]               ║
        ║    ════════════════════════════   ║
        ╚═══════════════════════════════════╝

KEY INTERACTIVE PARTS:

Horn/Speaker:
├── Emitter for audio
├── Visual point of interest
└── Audio origin point

Tone Arm:
├── Animated during playback
├── Moves across cylinder
├── Can be manually positioned
└── Satisfying mechanical detail

Cylinder:
├── Recording medium
├── Inserts into mandrel
├── Rotates during playback
├── Can be removed/swapped
└── Core interactive element

Crank (if manual):
├── Player winds mechanism
├── Determines play duration
├── Mechanical feedback
└── Optional (adds to ritual)
```

### Player Interaction Flow

```

PLAYER EXPERIENCE FLOW:

DISCOVERY:
┌─────────────────────────────────────────────────────────┐
│ Player explores office space                           │
│ Sees phonograph on desk                               │
│ "What is this old thing?"                             │
│ Action: Examine phonograph                            │
└─────────────────────────────────────────────────────────┘
                          ↓
EXAMINATION:
┌─────────────────────────────────────────────────────────┐
│ Close-up view of phonograph                            │
│ Description explains it's an Edison cylinder player     │
│ Shows where cylinder goes                               │
│ "I need to find a cylinder to play"                   │
│ Action: Look for cylinder                              │
└─────────────────────────────────────────────────────────┘
                          ↓
FINDING THE CYLINDER:
┌─────────────────────────────────────────────────────────┐
│ Cylinder located nearby (drawer, shelf, etc.)          │
│ Pick up interaction                                     │
│ Cylinder added to inventory                            │
│ Action: Return to phonograph                          │
└─────────────────────────────────────────────────────────┘
                          ↓
INSERTION:
┌─────────────────────────────────────────────────────────┐
│ Approach phonograph with cylinder                     │
│ Placement animation (aligns with mandrel)             │
│ Mechanical feedback (clicks into place)               │
│ "Ready to play"                                       │
│ Action: Trigger playback                              │
└─────────────────────────────────────────────────────────┘
                          ↓
PLAYBACK:
┌─────────────────────────────────────────────────────────┐
│ Mechanism animates (cylinder spins, arm moves)        │
│ Audio plays (period recording)                         │
│ Player listens to content                               │
│ May have option to stop, swap cylinders               │
│ Action: Listen until complete or interrupt            │
└─────────────────────────────────────────────────────────┘
                          ↓
NARRATIVE REVEALED:
┌─────────────────────────────────────────────────────────┐
│ Audio content reveals story information                 │
│ Player gains understanding                              │
│ World enriched                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the Edison cylinder implementation, you should know:
- **Object Interaction**: Pickup, placement, and inventory systems
- **Animation Blending**: Smooth transitions between animation states
- **Audio degradation**: Applying filters to simulate vintage recording quality
- **State Machines**: Managing device states (empty, loaded, playing, stopped)
- **Particle Effects**: Dust motes in light for period atmosphere

### Scene Data Structure

```javascript
// InteractiveObjectData.js - Edison cylinder phonograph
export const INTERACTIVE_OBJECTS = {
  edison_phonograph: {
    id: 'edison_phonograph',
    name: 'Edison Cylinder Phonograph',
    type: 'audio_playback_device',
    era: 'late_1800s',

    // Visual model
    model: {
      file: '/assets/models/edison_phonograph.glb',
      scale: 1.0,
      materials: {
        brass: { metalness: 0.9, roughness: 0.3 },
        wood: { roughness: 0.8 },
        enamel: { roughness: 0.2 }
      }
    },

    // Position in scene
    position: {
      scene: 'office_interior',
      location: 'desk_surface',
      position: { x: 0.5, y: 1.1, z: -1 },
      rotation: { x: 0, y: -0.3, z: 0 }
    },

    // Interaction configuration
    interaction: {
      // States
      states: {
        empty: {
          cylinderPresent: false,
          canInteract: true,
          prompt: 'Need a cylinder to play',
          availableActions: ['examine', 'insert_cylinder']
        },
        loaded: {
          cylinderPresent: true,
          canInteract: true,
          prompt: 'Press E to start playback',
          availableActions: ['examine', 'play', 'remove_cylinder']
        },
        playing: {
          cylinderPresent: true,
          isPlaying: true,
          canInteract: true,
          prompt: 'Press E to stop',
          availableActions: ['stop']
        },
        stopped: {
          cylinderPresent: true,
          isPlaying: false,
          canInteract: true,
          prompt: 'Press E to play again',
          availableActions: ['play', 'remove_cylinder']
        }
      },

      // Current state
      currentState: 'empty',

      // Cylinder to load (if present)
      cylinderId: null
    },

    // Playback configuration
    playback: {
      type: 'edison_cylinder',

      // Audio characteristics
      audio: {
        baseRecording: '/assets/audio/edison_cylinder_recording.mp3',
        // Applied filters for period feel
        filters: {
          highPass: 200,      // Remove low-frequency rumble
          lowPass: 3500,       // Limit high frequencies
          surfaceNoise: 0.15,  // Crackles and pops
          volumeFluctuation: 0.1, // Wobbly volume
          wowFlutter: 0.08,    // Pitch variation
          saturation: 0.85     // Slight reduction
        },

        // Playback duration (short for actual cylinders)
        duration: 120,  // 2 minutes max per wind
        loop: false
      },

      // Mechanical animation
      animation: {
        cylinderRPM: 120,  // Rotations per minute
        toneArmSpeed: 0.5,  // Movement speed across cylinder

        // Moving parts
        rotating: [
          { part: 'cylinder', axis: 'y', speed: 1 },
          { part: 'governor', axis: 'z', speed: 4 },
          { part: 'drive_belt', path: 'loop' }
        ],

        // Tone arm movement
        toneArm: {
          idlePosition: { x: 0, y: 0.5, z: 0.3 },
          playPosition: { x: 0, y: 0.5, z: -0.2 },
          travelDuration: 5  // Seconds to travel across
        }
      }
    },

    // Cylinder that works with this phonograph
    compatibleCylinders: [
      {
        id: 'cylinder_office_log',
        name: 'Office Log - October 14',
        content: 'dialog:office_log_cylinder',
        model: '/assets/models/edison_cylinder.glb',
        position: {
          scene: 'office_interior',
          container: 'desk_drawer_02',
          locked: false
        }
      }
    ]
  }
};
```

### Edison Phonograph Manager

```javascript
// EdisonPhonographManager.js - Period audio device controller
class EdisonPhonographManager {
  constructor(sceneManager, audioManager, dialogManager) {
    this.scene = sceneManager;
    this.audio = audioManager;
    this.dialog = dialogManager;

    this.phonographs = new Map();
    this.cylinders = new Map();
  }

  createPhonograph(config) {
    // Create the phonograph object
    const phonograph = {
      id: config.id,
      config: config,
      state: config.interaction.currentState,
      loadedCylinder: null,

      // Visual components
      parts: {},

      // Animation state
      animating: false,
      playbackTimer: null
    };

    // Load model
    this.loadPhonographModel(phonograph, config);

    // Set up interaction zones
    this.setupInteractionZone(phonograph, config);

    // Store reference
    this.phonographs.set(config.id, phonograph);

    return phonograph;
  }

  async loadPhonographModel(phonograph, config) {
    const gltf = await this.scene.loadModel(config.model.file);

    // Set up individual parts for animation
    gltf.traverse((child) => {
      if (child.isMesh) {
        // Identify parts by name
        if (child.name.includes('cylinder') || child.name.includes('mandrel')) {
          phonograph.parts.cylinder = child;
        } else if (child.name.includes('tone_arm') || child.name.includes('arm')) {
          phonograph.parts.toneArm = child;
        } else if (child.name.includes('governor')) {
          phonograph.parts.governor = child;
        } else if (child.name.includes('horn') || child.name.includes('speaker')) {
          phonograph.parts.horn = child;
        }
      }
    });

    phonograph.model = gltf;
    this.scene.add(gltf);
  }

  setupInteractionZone(phonograph, config) {
    const zone = {
      position: config.position.position,
      radius: 1.5,
      onEnter: (entity) => this.onPlayerApproach(phonograph, entity),
      onExit: (entity) => this.onPlayerLeave(phonograph, entity),
      onInteract: (entity) => this.onInteract(phonograph, entity)
    };

    this.scene.registerTrigger(config.id, zone);
  }

  onPlayerApproach(phonograph, entity) {
    if (entity.type !== 'player') return;

    // Show appropriate prompt based on state
    const stateConfig = phonograph.config.interaction.states[phonograph.state];
    this.showPrompt(phonograph, stateConfig.prompt);
  }

  onPlayerLeave(phonograph, entity) {
    this.hidePrompt(phonograph);
  }

  async onInteract(phonograph, entity) {
    if (entity.type !== 'player') return;

    switch (phonograph.state) {
      case 'empty':
        await this.handleEmptyState(phonograph);
        break;

      case 'loaded':
        await this.handleLoadedState(phonograph);
        break;

      case 'playing':
        await this.handlePlayingState(phonograph);
        break;

      case 'stopped':
        await this.handleStoppedState(phonograph);
        break;
    }
  }

  async handleEmptyState(phonograph) {
    // Check if player has a compatible cylinder
    const player = game.getManager('player');
    const cylinderId = player.hasItem('edison_cylinder');

    if (!cylinderId) {
      // No cylinder - informative message
      this.dialog.show('This appears to be an Edison cylinder phonograph. It needs a wax cylinder to play recordings.');
      return;
    }

    // Player has cylinder - animate insertion
    await this.animateCylinderInsertion(phonograph, cylinderId);

    // Remove from player inventory
    player.removeItem('edison_cylinder');

    // Change state
    this.changeState(phonograph, 'loaded');
    phonograph.loadedCylinder = cylinderId;
  }

  async handleLoadedState(phonograph) {
    // Start playback
    await this.startPlayback(phonograph);
    this.changeState(phonograph, 'playing');
  }

  async handlePlayingState(phonograph) {
    // Stop playback
    await this.stopPlayback(phonograph);
    this.changeState(phonograph, 'stopped');
  }

  async handleStoppedState(phonograph) {
    // Restart from beginning
    await this.startPlayback(phonograph);
    this.changeState(phonograph, 'playing');
  }

  async animateCylinderInsertion(phonograph, cylinderId) {
    // Get cylinder model
    const cylinderConfig = phonograph.config.compatibleCylinders.find(c => c.id === cylinderId);

    // Spawn cylinder in player's "hand"
    const cylinder = await this.scene.loadModel(cylinderConfig.model);

    // Position in front of camera, then move to phonograph
    const camera = this.scene.getCamera();
    const handOffset = new THREE.Vector3(0.3, -0.2, -0.5).applyMatrix4(camera.matrixWorld);

    cylinder.position.copy(handOffset);

    // Animate to phonograph
    const targetPosition = new THREE.Vector3(
      phonograph.model.position.x,
      phonograph.model.position.y + 0.6,
      phonograph.model.position.z
    );

    await this.animateToPosition(cylinder, targetPosition, 1.0);

    // Attach to phonograph
    phonograph.parts.cylinder.attach(cylinder);

    // Sound effect
    this.audio.playOneShot('cylinder_insert', {
      volume: 0.5,
      position: phonograph.model.position
    });
  }

  async startPlayback(phonograph) {
    const cylinderId = phonograph.loadedCylinder;
    const cylinderConfig = phonograph.config.compatibleCylinders.find(c => c.id === cylinderId);

    // Get recording content
    const recordingContent = this.getRecordingContent(cylinderConfig.content);

    // Start mechanical animations
    this.startMechanicalAnimation(phonograph);

    // Start audio with filters
    await this.playFilteredAudio(recordingContent, phonograph);

    // Start dialog if it's a dialog recording
    if (recordingContent.type === 'dialog') {
      this.playDialogRecording(recordingContent.dialogId);
    }

    // Set timer for playback end
    const duration = phonograph.config.playback.audio.duration;
    phonograph.playbackTimer = setTimeout(() => {
      this.onPlaybackComplete(phonograph);
    }, duration * 1000);
  }

  startMechanicalAnimation(phonograph) {
    const animConfig = phonograph.config.playback.animation;

    // Rotate cylinder
    const cylinderSpeed = (animConfig.cylinderRPM * 360) / 60;  // degrees per second

    // Rotate governor
    const governorSpeed = cylinderSpeed * animConfig.toneArm.speed;

    // Start animation loop
    const animate = () => {
      if (phonograph.state !== 'playing') {
        return;
      }

      // Rotate cylinder
      if (phonograph.parts.cylinder) {
        phonograph.parts.cylinder.rotation.y += cylinderSpeed * 0.016;
      }

      // Rotate governor
      if (phonograph.parts.governor) {
        phonograph.parts.governor.rotation.z += governorSpeed * 0.016;
      }

      // Move tone arm
      if (phonograph.parts.toneArm) {
        const arm = phonograph.parts.toneArm;
        const start = animConfig.toneArm.idlePosition;
        const end = animConfig.toneArm.playPosition;
        const duration = animConfig.toneArm.travelDuration;

        // Calculate progress
        const progress = Math.min(1, this.getPlaybackProgress(phonograph) / duration);

        arm.position.x = start.x + (end.x - start.x) * progress;
        arm.position.y = start.y + (end.y - start.y) * progress;
        arm.position.z = start.z + (end.z - start.z) * progress;
      }

      requestAnimationFrame(animate);
    };

    animate();
  }

  async playFilteredAudio(content, phonograph) {
    const filters = phonograph.config.playback.audio.filters;

    // Create audio context with filters
    const source = this.audio.createSource(content.file);

    // Apply period filters
    this.audio.applyFilters(source, filters);

    // Play at phonograph position
    source.play({
      position: phonograph.model.position,
      volume: 0.8
    });

    phonograph.audioSource = source;
  }

  playDialogRecording(dialogId) {
    // Play dialog with phonograph styling
    this.dialog.play(dialogId, {
      style: 'phonograph',
      subtitles: true,
      filter: 'vintage'
    });
  }

  async stopPlayback(phonograph) {
    // Stop mechanical animation
    // (animation loop checks state)

    // Stop audio
    if (phonograph.audioSource) {
      phonograph.audioSource.stop();
      phonograph.audioSource = null;
    }

    // Cancel timer
    if (phonograph.playbackTimer) {
      clearTimeout(phonograph.playbackTimer);
      phonograph.playbackTimer = null;
    }

    // Stop dialog
    this.dialog.stop();

    // Reset tone arm position
    if (phonograph.parts.toneArm) {
      const idle = phonograph.config.playback.animation.toneArm.idlePosition;
      await this.animateToPosition(phonograph.parts.toneArm, idle, 1.0);
    }
  }

  onPlaybackComplete(phonograph) {
    this.changeState(phonograph, 'stopped');
  }

  changeState(phonograph, newState) {
    phonograph.state = newState;

    // Update prompt
    const stateConfig = phonograph.config.interaction.states[newState];
    this.showPrompt(phonograph, stateConfig.prompt);

    // Emit state change event
    game.emit('phonograph:state_changed', {
      id: phonograph.id,
      state: newState
    });
  }

  getRecordingProgress(phonograph) {
    if (!phonograph.audioSource) return 0;

    return phonograph.audioSource.currentTime / phonograph.audioSource.duration;
  }

  getRecordingContent(contentId) {
    // Parse content ID (e.g., "dialog:office_log_cylinder")
    const [type, id] = contentId.split(':');

    switch (type) {
      case 'dialog':
        return {
          type: 'dialog',
          dialogId: id,
          file: `/assets/audio/cylinders/${id}.mp3`
        };

      case 'music':
        return {
          type: 'music',
          trackId: id,
          file: `/assets/audio/cylinders/${id}.mp3`
        };

      case 'recording':
        return {
          type: 'recording',
          fileId: id,
          file: `/assets/audio/cylinders/${id}.mp3`
        };

      default:
        return null;
    }
  }

  showPrompt(phonograph, text) {
    const ui = game.getManager('ui');
    ui.showInteractionPrompt({
      target: phonograph.id,
      text: text,
      position: phonograph.model.position
    });
  }

  hidePrompt(phonograph) {
    const ui = game.getManager('ui');
    ui.hideInteractionPrompt(phonograph.id);
  }

  async animateToPosition(object, target, duration) {
    const start = object.position.clone();
    const startTime = Date.now();

    return new Promise(resolve => {
      const animate = () => {
        const elapsed = (Date.now() - startTime) / 1000;
        const progress = Math.min(1, elapsed / duration);
        const eased = this.easeInOutQuad(progress);

        object.position.lerpVectors(start, target, eased);

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          resolve();
        }
      };

      animate();
    });
  }

  easeInOutQuad(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }
}
```

### Period Audio Filter System

```javascript
// PeriodAudioFilters.js - Audio degradation for vintage feel
class PeriodAudioFilters {
  constructor(audioContext) {
    this.context = audioContext;
  }

  applyFilters(source, config) {
    const filters = [];

    // High-pass filter (remove low rumble)
    if (config.highPass) {
      const highPass = this.context.createBiquadFilter();
      highPass.type = 'highpass';
      highPass.frequency.value = config.highPass;
      filters.push(highPass);
    }

    // Low-pass filter (limit highs)
    if (config.lowPass) {
      const lowPass = this.context.createBiquadFilter();
      lowPass.type = 'lowpass';
      lowPass.frequency.value = config.lowPass;
      filters.push(lowPass);
    }

    // Connect filters
    source.connect(filters[0]);
    for (let i = 0; i < filters.length - 1; i++) {
      filters[i].connect(filters[i + 1]);
    }
    filters[filters.length - 1].connect(this.context.destination);

    // Add effects
    if (config.surfaceNoise > 0) {
      this.addSurfaceNoise(filters[filters.length - 1], config.surfaceNoise);
    }

    if (config.wowFlutter > 0) {
      this.addWowFlutter(filters[filters.length - 1], config.wowFlutter);
    }
  }

  addSurfaceNoise(inputNode, intensity) {
    // Create noise buffer
    const bufferSize = 2 * this.context.sampleRate;
    const noiseBuffer = this.context.createBuffer(1, bufferSize, this.context.sampleRate);
    const data = noiseBuffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * intensity;
    }

    const noise = this.context.createBufferSource();
    noise.buffer = noiseBuffer;
    noise.loop = true;

    // Gain node for noise
    const noiseGain = this.context.createGain();
    noiseGain.gain.value = intensity * 0.5;

    noise.connect(noiseGain);
    noiseGain.connect(inputNode);
    noise.start();
  }

  addWowFlutter(inputNode, intensity) {
    // Create oscillating delay for pitch variation
    // (Simplified - real wow/flutter is more complex)
    const delay = this.context.createDelay();
    delay.delayTime.value = 0.05;

    const depth = this.context.createGain();
    depth.gain.value = intensity * 0.002;

    inputNode.connect(depth);
    depth.connect(delay);
    delay.connect(inputNode);
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Define the Period Object

```
PERIOD ITEM DESIGN:

1. What time period?
    Edison = late 1800s, early 1900s

2. What does it do?
    Plays audio recordings on wax cylinders

3. How does player interact?
    Find cylinder → insert → wind → play

4. What content does it reveal?
    Narrative from someone in that era

5. Why this specific technology?
    Anachronism creates mystery
    Ritual creates meaning
```

### Step 2: Design the Interaction Ritual

```javascript
// Interaction ritual stages:

const ritual = {
  discovery: 'see device, want to use',
  quest: 'find compatible cylinder',
  insertion: 'place cylinder, mechanical feedback',
  activation: 'start mechanism, anticipation',
  playback: 'listen to content',
  completion: 'content delivered, understanding gained'
};
```

---

## 🔧 Variations For Your Game

### Variation 1: Vinyl Record Player

```javascript
const vinylPlayer = {
  // 1950s instead of 1890s
  era: 'mid_20th_century',

  // Records instead of cylinders
  media: 'vinyl_records',

  // Electric or wind-up
  power: 'electric',

  // Different ritual (place record, drop needle)
  interaction: 'place_record_and_needle'
};
```

### Variation 2: Cassette Tape

```javascript
const cassettePlayer = {
  // 1980s instead of 1890s
  era: 'late_20th_century',

  // Tapes instead of cylinders
  media: 'cassette_tapes',

  // Electric playback
  power: 'electric',

  // Different ritual (insert tape, press play)
  interaction: 'insert_and_play'
};
```

### Variation 3: Magical Talking Object

```javascript
const magicalObject = {
  // Not period-accurate, but stylized
  type: 'magical_artifact',

  // Plays voice when touched
  trigger: 'touch',

  // No physical media required
  media: 'none_required',

  // Different feel entirely
  atmosphere: 'magical_not_mechanical'
};
```

---

## Performance Considerations

```

EDISON CYLINDER PERFORMANCE:

Model:
├── Single complex model
├── Multiple parts for animation
├── Can use instancing for duplicates
└─→ Moderate impact

Animation:
├── Rotating parts (simple transforms)
├── Tone arm movement (lerp)
├── All CPU-side (minimal GPU)
└─→ Negligible

Audio Filters:
├── Web Audio nodes (efficient)
├── Real-time processing
├── Can be pre-filtered for mobile
└─→ Acceptable overall

Particles:
├── Dust motes in light beam
├── Low count (<30)
├─→ Minimal impact
```

---

## Common Mistakes Beginners Make

### 1. Too Convenient

```javascript
// ❌ WRONG: Cylinder right next to phonograph
// No discovery, no ritual

// ✅ CORRECT: Cylinder hidden elsewhere
// Player must explore to find it
```

### 2: Digital Audio Quality

```javascript
// ❌ WRONG: Perfect digital audio
// Breaks immersion, feels fake

// ✅ CORRECT: Degraded, filtered audio
// Sounds like period recording
```

### 3: No Mechanical Feedback

```javascript
// ❌ WRONG: Audio plays, nothing moves
// Feels like digital audio player

// ✅ CORRECT: Parts rotate, arm moves
// Mechanical feel creates immersion
```

---

## Related Systems

- [DialogManager](../05-media-systems/dialog-manager.md) - Audio playback
- [Interactive Object System](../05-interactive-objects/interactive-object-system.md) - Object interaction
- [AnimationManager](../06-animation/animation-manager.md) - Mechanical animations

---

## Source File Reference

**Object Data**:
- `content/InteractiveObjectData.js` - Phonograph configuration

**Managers**:
- `managers/EdisonPhonographManager.js` - Device controller
- `managers/PeriodAudioFilters.js` - Audio degradation

**Assets**:
- `assets/models/edison_phonograph.glb` - Phonograph model
- `assets/models/edison_cylinder.glb` - Cylinder model
- `assets/audio/cylinders/*.mp3` - Recordings

---

## References

- [Edison Phonograph History](https://www.youtube.com/watch?v=T9S_e6jX8kA) - Historical reference
- [Audio Filtering](https://webaudioapi.com/book/chapter-8/) - Web Audio filters
- [Period Object Design](https://www.youtube.com/watch?v=M4skP6bN_Ks) - Video essay

*Documentation last updated: January 12, 2026*
