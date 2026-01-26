# Phone Booth Scene - First Interactive Moment

**Scene Case Study #13**

---

## What You Need to Know First

- **Gaussian Splatting Basics** (See: *Rendering System*)
- **Interactive Object System** (See: *Interactive Objects*)
- **Cord Physics** (See: *Physics System*)
- **Dialog System** (See: *DialogManager*)

---

## Scene Overview

| Property | Value |
|----------|-------|
| **Location** | Plaza, near four-way intersection |
| **Narrative Context** | First interactive moment - introduces communication from the "other side" |
| **Player Experience** | Discovery → Approach → Ringing → Interaction → Mysterious Dialog |
| **Atmosphere** | Mystery, intrusion, communication from beyond |
| **Technical Focus** | Interactive prop with physics-based cord, dialog trigger, state machine |

### The Scene's Purpose

The phone booth is the **player's first active interaction** with the game world. Unlike the rusted car (passive observation), this scene demands participation:

1. **Discovery** - Player sees the booth and is drawn to investigate
2. **Provocation** - Phone rings, creating urgency and curiosity
3. **Interaction** - Player must lift the receiver (cord physics)
4. **Revelation** - Mysterious voice initiates the main narrative

This is the **inciting incident** - the call to adventure that transforms the player from observer to participant.

---

## 🎮 Game Design Perspective

### Creative Intent

**Why a phone booth? Why not a radio, note, or ghostly apparition?**

| Medium | Player Experience |
|---------|-------------------|
| **Ghost/Apparition** | Passive - spectacles happen TO you |
| **Note/Journal** | One-way - you read, no response possible |
| **Radio** | Broadcast - same message for everyone |
| **Phone Booth** | **Dialog - two-way communication, feels personal** |

The phone booth creates **intimacy through one-on-one communication**. When the phone rings, it's ringing for YOU.

### Design Philosophy

**The Ringing Phone as Irresistible Hook:**

```
Player sees phone booth
    ↓
Curiosity - "Can I interact with this?"
    ↓
PHONE RINGS
    ↓
Urgency - "Someone is calling NOW!"
    ↓
Compulsion - "I must answer this"
```

The ringing is a **game design contract** - when a phone rings in a game, players are culturally conditioned to answer it. We're leveraging decades of Pavlovian training.

### Mood Building

The phone booth establishes mood through:

1. **Temporal Dissonance** - Old-fashioned booth in (maybe) modern setting
2. **Isolation** - You're the only one who can hear this ring
3. **Intrusion** - Something from "outside" is breaking through
4. **Physicality** - Cord weight, receiver texture, rotary dial mechanics

### Player Psychology

| Psychological Effect | How the Booth Achieves It |
|---------------------|---------------------------|
| **Agency** | Player chooses to answer (or not) |
| **Vulnerability** | Picking up receiver = accepting communication |
| **Mystery** | Who knows this number? Why here? |
| **Connection** | Voice confirms "you are not alone" (for better or worse) |

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
┌─────────────────────────────────────────────┐
│                                             │
│   [Alley West]           [Alley East]       │
│         │                     │             │
│         └──────────┬──────────┘             │
│                    │                        │
│              ╔═══════════╗                  │
│              ║ RUSTED    ║                  │
│              ║ CAR       ║                  │
│              ╚═══════════╝                  │
│                    │                        │
│              [Plaza North]                  │
│                    │                        │
│              ═════════════                 │
│              │           │                  │
│          [PHONE BOOTH]  [PLAZA CENTER]      │
│          ◉ Ringing     ◉ Open space         │
│              │           │                  │
│                                             │
│   Player approaches from plaza center       │
│   Booth is visible from spawn               │
│   Ringing audible from ~15m away            │
└─────────────────────────────────────────────┘
```

### Player Path

```
1. Player spawns in plaza
   ↓
2. Explores, notices phone booth
   ↓
3. [Trigger: Player enters proximity zone]
   ↓
4. Phone begins ringing
   ↓
5. Player approaches booth
   ↓
6. "Press E to interact" prompt appears
   ↓
7. Player presses E, character reaches for receiver
   ↓
8. Cord physics simulation activates
   ↓
9. Receiver lifts to ear
   ↓
10. Dialog begins - mysterious voice speaks
   ↓
11. Dialog completes, receiver hangs up (or player hangs up)
    ↓
12. Phone booth enters "quiet" state for remainder
```

### Atmosphere Layers

| Layer | Elements |
|-------|----------|
| **Visual** | Weathered booth, tangled cord, dim interior, flickering light |
| **Audio** | Ringing (bell), dial tone, cord creak, voice through receiver |
| **Lighting** | Booth interior slightly darker, ambient light glow |
| **Interaction** | Physical cord weight, receiver lift animation, hangup physics |

---

## Technical Implementation

### Phone Booth Data Structure

```javascript
export const PHONE_BOOTH = {
  id: 'phone_booth_plaza',
  type: 'interactive_prop',

  // Splat/Model data
  splat: {
    file: '/assets/splats/phone-booth.ply',
    maxPoints: 1500000,
    renderScale: 1.0
  },

  // World placement
  transform: {
    position: { x: -8, y: 0, z: 3 },
    rotation: { x: 0, y: -30, z: 0 },
    scale: { x: 1, y: 1, z: 1 }
  },

  // Interaction configuration
  interaction: {
    type: 'phone_booth',
    prompt: 'Answer Phone',
    promptKey: 'e',
    maxDistance: 2.5,
    triggerRadius: 5.0  // Ringing starts when player enters this radius
  },

  // Cord physics
  cord: {
    boneCount: 20,          // Number of segments in cord
    cordLength: 1.5,        // Total length in meters
    cordRadius: 0.005,      // Cord thickness
    anchorPoint: {          // Where cord connects to booth
      x: 0, y: 1.8, z: 0.3
    },
    receiver: {             // Receiver attachment point
      offset: { x: 0, y: 0, z: 0.15 }
    },
    stiffness: 0.8,         // Cord physics properties
    damping: 0.5,
    gravity: { x: 0, y: -9.8, z: 0 }
  },

  // Phone states
  states: {
    idle: {
      ringing: false,
      receiverHung: true,
      dialogActive: false
    },
    ringing: {
      ringing: true,
      receiverHung: true,
      dialogActive: false
    },
    lifted: {
      ringing: false,
      receiverHung: false,
      dialogActive: true
    },
    finished: {
      ringing: false,
      receiverHung: true,
      dialogActive: false,
      canReinteract: false  // One-time interaction
    }
  },

  // Audio configuration
  audio: {
    ringing: {
      file: '/assets/audio/phone-ring.ogg',
      volume: 0.8,
      loop: true,
      spatial: true,
      position: { x: -8, y: 1.5, z: 3 }
    },
    dialTone: {
      file: '/assets/audio/dial-tone.ogg',
      volume: 0.6,
      loop: true,
      spatial: true
    },
    click: {
      file: '/assets/audio/receiver-click.ogg',
      volume: 0.5
    },
    cord: {
      creak: '/assets/audio/cord-creak.ogg',
      volume: 0.3
    }
  },

  // Dialog to play when answered
  dialog: {
    id: 'phone_booth_first_call',
    speaker: 'unknown_voice',
    lines: [
      {
        text: "...hello?",
        duration: 2.0,
        audio: '/assets/audio/dialog/phone_01.ogg'
      },
      {
        text: "Is someone there? I... I can't see you but...",
        duration: 4.5,
        audio: '/assets/audio/dialog/phone_02.ogg'
      },
      {
        text: "You came. The door... it opened for you.",
        duration: 4.0,
        audio: '/assets/audio/dialog/phone_03.ogg'
      },
      {
        text: "Listen carefully. You shouldn't be here.",
        duration: 3.5,
        audio: '/assets/audio/dialog/phone_04.ogg'
      },
      {
        text: "But now that you are... find the office. The radio.",
        duration: 4.0,
        audio: '/assets/audio/dialog/phone_05.ogg'
      },
      {
        text: "They're watching. I'll call again.",
        duration: 3.0,
        audio: '/assets/audio/dialog/phone_06.ogg'
      }
    ],
    onComplete: 'phone_booth_first_call_complete'
  },

  // Visual effects
  effects: {
    lightFlicker: {
      enabled: true,
      intensity: 0.3,
      frequency: 0.2  // Hz
    },
    condensation: {
      enabled: true,  // Moisture on glass
      opacity: 0.15
    }
  }
};
```

### Phone Booth Interaction Manager

```javascript
/**
 * Manages phone booth interactions including cord physics and dialog
 */
class PhoneBoothManager {
  constructor(scene) {
    this.scene = scene;
    this.gameManager = scene.gameManager;
    this.booths = new Map();
  }

  /**
   * Initialize a phone booth
   */
  async loadBooth(config) {
    const booth = {
      id: config.id,
      config: config,
      state: 'idle',
      mesh: null,
      cord: null,
      receiver: null,
      audio: {},
      playerInTrigger: false,
      dialogActive: false
    };

    // Load booth mesh/splat
    booth.mesh = await this.loadBoothMesh(config);

    // Setup cord physics
    booth.cord = this.setupCordPhysics(config.cord);

    // Setup receiver
    booth.receiver = this.setupReceiver(booth.cord);

    // Setup audio sources
    booth.audio = this.setupAudio(config.audio);

    // Create interaction trigger zone
    booth.triggerZone = this.createTriggerZone(config);

    // Register for interaction prompts
    this.scene.interaction.register(config.interaction);

    this.booths.set(config.id, booth);
    return booth;
  }

  /**
   * Load booth visual mesh
   */
  async loadBoothMesh(splatConfig) {
    const mesh = await this.scene.splatLoader.load(splatConfig.file);

    mesh.position.set(
      splatConfig.transform.position.x,
      splatConfig.transform.position.y,
      splatConfig.transform.position.z
    );

    mesh.rotation.set(
      THREE.MathUtils.degToRad(splatConfig.transform.rotation.x),
      THREE.MathUtils.degToRad(splatConfig.transform.rotation.y),
      THREE.MathUtils.degToRad(splatConfig.transform.rotation.z)
    );

    return mesh;
  }

  /**
   * Setup physics-based cord
   */
  setupCordPhysics(cordConfig) {
    // Create chain of physics bodies for cord simulation
    const bones = [];
    const boneLength = cordConfig.cordLength / cordConfig.boneCount;

    for (let i = 0; i < cordConfig.boneCount; i++) {
      const bone = this.scene.physics.createSphere({
        radius: cordConfig.cordRadius,
        mass: 0.01,  // Very light
        friction: 0.5,
        restitution: 0.1
      });

      // Position bones hanging down
      const y = cordConfig.anchorPoint.y - (i * boneLength);
      bone.setPosition(
        cordConfig.anchorPoint.x,
        Math.max(y, 0.5),  // Don't go below ground
        cordConfig.anchorPoint.z
      );

      // Connect to previous bone
      if (i > 0) {
        this.scene.physics.createJoint(
          bones[i - 1],
          bone,
          {
            type: 'ball',
            position: {
              x: cordConfig.anchorPoint.x,
              y: cordConfig.anchorPoint.y - ((i - 0.5) * boneLength),
              z: cordConfig.anchorPoint.z
            }
          }
        );
      }

      bones.push(bone);
    }

    // Anchor first bone to booth
    this.scene.physics.createFixedJoint(
      this.scene.physics.getStaticBody(),
      bones[0],
      cordConfig.anchorPoint
    );

    return {
      bones: bones,
      config: cordConfig,
      receiverAttached: true
    };
  }

  /**
   * Setup receiver object
   */
  setupReceiver(cord) {
    // Receiver is attached to last cord bone
    const receiver = this.scene.createGameObject('phone_receiver');

    receiver.attachToBone(cord.bones[cord.bones.length - 1]);
    receiver.setLocalOffset(cord.config.receiver.offset);

    receiver.on('grab', () => this.onReceiverGrabbed(receiver));
    receiver.on('release', () => this.onReceiverReleased(receiver));

    return receiver;
  }

  /**
   * Setup all audio sources
   */
  setupAudio(audioConfig) {
    const audio = {};

    // Ringing sound (looping)
    audio.ringing = this.scene.audio.createSpatialSource({
      url: audioConfig.ringing.file,
      volume: audioConfig.ringing.volume,
      loop: audioConfig.ringing.loop,
      position: audioConfig.ringing.position
    });

    // Dial tone (starts when receiver lifted)
    audio.dialTone = this.scene.audio.createSpatialSource({
      url: audioConfig.dialTone.file,
      volume: 0,  // Start silent
      loop: true,
      position: audioConfig.ringing.position
    });

    // One-shot sounds
    audio.click = this.scene.audio.createOneShot(audioConfig.click.file);
    audio.cordCreak = this.scene.audio.createOneShot(audioConfig.cord.creak);

    return audio;
  }

  /**
   * Create trigger zone for ringing activation
   */
  createTriggerZone(config) {
    const zone = this.scene.physics.createTrigger({
      shape: 'sphere',
      radius: config.interaction.triggerRadius,
      position: config.transform.position,
      onEnter: (entity) => this.onTriggerEnter(entity, config.id),
      onExit: (entity) => this.onTriggerExit(entity, config.id)
    });

    return zone;
  }

  /**
   * Handle player entering trigger zone
   */
  onTriggerEnter(entity, boothId) {
    if (entity !== this.scene.player) return;

    const booth = this.booths.get(boothId);
    if (!booth || booth.state !== 'idle') return;

    // Start ringing after short delay
    setTimeout(() => {
      if (booth.state === 'idle' && booth.playerInTrigger) {
        this.startRinging(boothId);
      }
    }, 500);
  }

  /**
   * Handle player exiting trigger zone
   */
  onTriggerExit(entity, boothId) {
    if (entity !== this.scene.player) return;

    const booth = this.booths.get(boothId);
    if (!booth) return;

    booth.playerInTrigger = false;

    // Stop ringing if player walks away
    if (booth.state === 'ringing') {
      this.stopRinging(boothId);
    }
  }

  /**
   * Start phone ringing
   */
  startRinging(boothId) {
    const booth = this.booths.get(boothId);
    if (!booth) return;

    booth.state = 'ringing';
    booth.playerInTrigger = true;

    // Start ringing audio
    booth.audio.ringing.play();

    // Add visual indicator
    this.scene.vfx.trigger('phone_ringing', {
      position: booth.mesh.position,
      intensity: 1.0
    });

    // Enable interaction prompt
    this.scene.interaction.enablePrompt(boothId);
  }

  /**
   * Stop phone ringing
   */
  stopRinging(boothId) {
    const booth = this.booths.get(boothId);
    if (!booth) return;

    booth.state = 'idle';
    booth.playerInTrigger = false;

    booth.audio.ringing.stop();

    this.scene.interaction.disablePrompt(boothId);
  }

  /**
   * Handle player interacting with booth (answering)
   */
  async onInteract(boothId) {
    const booth = this.booths.get(boothId);
    if (!booth || booth.state !== 'ringing') return;

    booth.state = 'lifting';

    // Stop ringing
    booth.audio.ringing.stop();

    // Play click sound
    booth.audio.click.play();

    // Animate receiver lift
    await this.animateReceiverLift(booth);

    // Start dialog
    this.startDialog(booth);
  }

  /**
   * Animate receiver lifting to ear
   */
  async animateReceiverLift(booth) {
    const duration = 800;  // ms
    const startTime = performance.now();

    const receiverStart = booth.receiver.getWorldPosition().clone();
    const receiverEnd = new THREE.Vector3(
      booth.config.transform.position.x + 0.3,
      this.scene.player.camera.position.y - 0.2,
      booth.config.transform.position.z + 0.5
    );

    return new Promise((resolve) => {
      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = this.easeOutQuad(progress);

        booth.receiver.lerpPosition(receiverStart, receiverEnd, eased);

        // Play cord creak sound periodically
        if (Math.random() < 0.05) {
          booth.audio.cordCreak.play();
        }

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          booth.state = 'lifted';
          resolve();
        }
      };

      animate();
    });
  }

  /**
   * Start dialog sequence
   */
  startDialog(booth) {
    booth.state = 'dialog';
    booth.dialogActive = true;

    // Start dial tone briefly
    booth.audio.dialTone.setVolume(booth.config.audio.dialTone.volume);
    booth.audio.dialTone.play();

    // Fade out dial tone after 1 second
    setTimeout(() => {
      this.fadeAudio(booth.audio.dialTone, 0, 500);
    }, 1000);

    // Play dialog
    this.scene.dialog.play(booth.config.dialog, {
      onComplete: () => this.onDialogComplete(booth)
    });
  }

  /**
   * Handle dialog completion
   */
  onDialogComplete(booth) {
    booth.dialogActive = false;
    booth.state = 'finished';

    // Hang up receiver
    this.animateReceiverHangup(booth);

    // Disable interaction
    this.scene.interaction.unregister(booth.config.interaction);

    // Flag for quest system
    this.gameManager.setCriteria('answered_first_phone', true);
  }

  /**
   * Animate receiver hanging up
   */
  async animateReceiverHangup(booth) {
    const duration = 600;
    const startTime = performance.now();

    const receiverStart = booth.receiver.getWorldPosition().clone();
    const receiverEnd = new THREE.Vector3(
      booth.config.transform.position.x,
      booth.config.cord.anchorPoint.y - booth.config.cord.cordLength + 0.3,
      booth.config.transform.position.z + 0.3
    );

    return new Promise((resolve) => {
      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = this.easeInQuad(progress);

        booth.receiver.lerpPosition(receiverStart, receiverEnd, eased);

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          // Play click when receiver hits cradle
          booth.audio.click.play({ volume: 0.3 });
          resolve();
        }
      };

      animate();
    });
  }

  /**
   * Update cord physics
   */
  update(deltaTime) {
    for (const booth of this.booths.values()) {
      if (booth.cord) {
        this.updateCordPhysics(booth.cord, deltaTime);
      }

      // Update light flicker effect
      if (booth.config.effects?.lightFlicker?.enabled) {
        this.updateLightFlicker(booth);
      }
    }
  }

  /**
   * Update cord physics simulation
   */
  updateCordPhysics(cord, deltaTime) {
    // Physics is handled by Rapier, but we may need
    // to apply constraints or dampening here

    // Ensure cord doesn't go through booth geometry
    for (const bone of cord.bones) {
      const pos = bone.getPosition();
      if (pos.y < 0.1) {
        pos.y = 0.1;
        bone.setPosition(pos.x, pos.y, pos.z);
      }
    }
  }

  /**
   * Update light flicker effect
   */
  updateLightFlicker(booth) {
    const config = booth.config.effects.lightFlicker;
    const time = performance.now() / 1000;

    const flicker = Math.sin(time * Math.PI * 2 * config.frequency) * 0.5 + 0.5;
    const intensity = 1 - (flicker * config.intensity);

    // Apply to booth light
    if (booth.boothLight) {
      booth.boothLight.intensity = intensity;
    }
  }

  /**
   * Fade audio volume
   */
  fadeAudio(source, targetVolume, duration) {
    const startVolume = source.getVolume();
    const startTime = performance.now();

    const fade = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      source.setVolume(startVolume + (targetVolume - startVolume) * progress);

      if (progress < 1) {
        requestAnimationFrame(fade);
      } else if (targetVolume === 0) {
        source.stop();
      }
    };

    fade();
  }

  /**
   * Easing function for smooth animation
   */
  easeOutQuad(t) {
    return t * (2 - t);
  }

  easeInQuad(t) {
    return t * t;
  }
}
```

### Scene Integration

```javascript
/**
 * Phone booth integrated into plaza scene
 */
class PlazaScene extends BaseScene {
  async onLoad() {
    // Load base scene
    await this.loadSplat('/assets/splats/plaza.ply');

    // Initialize phone booth manager
    this.phoneBooths = new PhoneBoothManager(this);

    // Load phone booth
    const booth = await this.phoneBooths.loadBooth(PHONE_BOOTH);

    // Set player spawn (facing toward booth for visual interest)
    this.player.spawn.set({ x: 0, y: 1.7, z: 5 });
    this.player.spawn.lookTowards(booth.mesh.position);

    // Register quest criteria check
    this.gameManager.on('criteria:answered_first_phone', () => {
      console.log('First phone call completed!');
    });
  }

  onUpdate(deltaTime) {
    // Update phone booth physics and effects
    this.phoneBooths.update(deltaTime);
  }
}
```

---

## How To Build A Scene Like This

### Step 1: Define the Interaction Hook

What makes the player want to interact?

```javascript
const hook = {
  // Visual curiosity
  visual: 'Weathered booth stands out from environment',

  // Auditory provocation
  audio: 'Ringing creates urgency - must answer NOW',

  // Cultural conditioning
  conditioning: 'Players know ringing phones = answer them',

  // Mystery
  mystery: 'Who is calling? Why here?'
};
```

### Step 2: Design the Interaction Flow

```
Idle → Ringing → Interact → Animating → Dialog → Finished
  ↑                                            ↓
  └─────────────── Player walks away ──────────┘
```

Each state should have clear entry/exit conditions.

### Step 3: Setup Physics-Based Elements

```javascript
// Cord physics breakdown
const cordPhysics = {
  // Chain of connected bodies
  bones: Array(20).fill().map(() => physicsBody),

  // Each bone connected to previous
  joints: bones.map((bone, i) =>
    i === 0 ? fixedJoint : connectTo(bones[i-1])
  ),

  // Receiver attached to last bone
  receiver: attachTo(bones.last)
};
```

### Step 4: Layer Audio Feedback

```javascript
const audioLayers = {
  ambient: 'Wind, environment',

  ringing: 'Gets attention, creates urgency',

  interaction: 'Click, cord creak - tactile feedback',

  dialog: 'Voice through receiver (filtered audio)',

  completion: 'Hangup click - closure'
};
```

### Step 5: Create Dialog System Integration

```javascript
const dialogIntegration = {
  play: (dialogId) => {
    // Show captions
    // Play voice audio
    // Trigger any animations
    // Call onComplete when done
  }
};
```

### Step 6: Test and Iterate

**Playtest Questions:**
1. Is the booth visible from spawn?
2. Does the ringing feel urgent or annoying?
3. Is the interaction prompt clear?
4. Does the cord feel weighty and realistic?
5. Does the dialog feel personal and mysterious?

---

## Variations For Your Game

### Variation 1: Horror Version

```javascript
const horrorVersion = {
  ringing: {
    behavior: 'erratic',  // Starts and stops randomly
    sound: 'distorted_ring',
    visualEffect: 'lights_flicker violently when ringing'
  },

  dialog: {
    speaker: 'threatening_voice',
    content: 'They know you answered...',
    effect: 'sanity_decrease'
  },

  consequence: {
    immediate: 'Enemies become aware of player',
    longTerm: 'Phone rings at dangerous times'
  }
};
```

### Variation 2: Puzzle Integration

```javascript
const puzzleVersion = {
  ringing: {
    trigger: 'Only after specific event',
    pattern: 'Morse code or number hints'
  },

  dialog: {
    content: 'Clues to current puzzle',
    repeatable: true
  },

  usage: 'Information hotline for puzzles'
};
```

### Variation 3: Multiple Booths

```javascript
const multipleBooths = {
  design: 'Several booths, only one rings',

  selection: {
    criteria: 'Random or story-triggered',
    visual: 'Ringing booth has light on'
  },

  progression: 'Each booth reveals more story'
};
```

### Variation 4: Two-Way Conversation

```javascript
const twoWayVersion = {
  dialog: {
    type: 'branching',
    playerCanRespond: true,

    choices: [
      { text: "Who is this?", next: 'dialog_branch_a' },
      { text: "Where am I?", next: 'dialog_branch_b' },
      { text: "*Hang up*", next: 'dialog_hangup' }
    ]
  }
};
```

### Variation 5: Time-Limited

```javascript
const timedVersion = {
  ringing: {
    duration: 10,  // seconds
    consequence: 'If not answered, important story missed'
  },

  urgency: 'Creates tension and replay value'
};
```

---

## Performance Considerations

### Cord Physics Optimization

```javascript
// Reduce bone count when possible
const optimization = {
  highQuality: { bones: 20 },
  mediumQuality: { bones: 12 },
  lowQuality: { bones: 6 }
};

// Use simpler joint types
const jointTypes = {
  expensive: 'ball_joint',  // Full rotation
  cheap: 'hinge_joint'      // Restricted rotation
};
```

### Audio Optimization

```javascript
// Don't play ringing if player can't hear
const audioOptimization = {
  maxHearingDistance: 50,

  update: (playerPos, boothPos) => {
    if (distance > maxHearingDistance) {
      ringingSound.pause();
    } else {
      ringingSound.resume();
    }
  }
};
```

---

## Common Mistakes Beginners Make

### Mistake 1: No Clear Hook

```javascript
// BAD: Phone sits silently, player may miss it
const badHook = {
  ringing: false,
  visual: 'blends in with environment'
};

// GOOD: Ringing grabs attention
const goodHook = {
  ringing: true,
  visual: 'distinctive silhouette, light draws eye'
};
```

### Mistake 2: Weak Cord Physics

```javascript
// BAD: Cord doesn't swing or has weight
const badCord = {
  bones: 2,  // Too few for realistic movement
  physics: 'static'
};

// GOOD: Cord swings naturally
const goodCord = {
  bones: 15-20,  // Enough for smooth curves
  physics: 'full_simulation'
};
```

### Mistake 3: Generic Dialog

```javascript
// BAD: Exposition dump
const badDialog = {
  text: "Welcome to the game. Your objective is to find the key..."
};

// GOOD: Mysterious, personal
const goodDialog = {
  text: "...hello? Is someone there? You came...",
  tone: 'uncertain, human, intriguing'
};
```

### Mistake 4: No Feedback

```javascript
// BAD: Silent interaction
const badFeedback = {
  clickSound: null,
  cordCreak: null,
  dialTone: null
};

// GOOD: Tactile audio feedback
const goodFeedback = {
  clickSound: 'receiver_click',
  cordCreak: 'cord_movement',
  dialTone: 'dial_tone_fade'
};
```

---

## Related Systems

- **DialogManager** - For playing phone conversations
- **PhysicsManager** - For cord simulation
- **InteractionSystem** - For player interaction prompts
- **AudioManager** - For spatial audio effects
- **Other Interactive Objects** - Viewmaster, Candlestick Phone, Amplifier

---

## References

- **Shadow Engine Documentation**: `docs/`
- **Cord Physics**: See *Physics System*
- **Dialog System**: See *DialogManager*

---

**RALPH_STATUS:**
- **Status**: Phone Booth Scene documentation complete
- **Files Created**: `docs/generated/14-scene-case-studies/phone-booth-scene.md`
- **Related Documentation**: All Phase 14 scene case studies
- **Next**: Viewmaster Scene documentation
