# Candlestick Phone Scene - Period-Authentic Dialog Trigger

**Scene Case Study #15**

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
| **Location** | Office interior, on desk |
| **Narrative Context** | Period-authentic communication device for dialog delivery |
| **Player Experience** | Discovery → Ringing → Lift Receiver → Dialog Sequence |
| **Atmosphere** | Nostalgic, grounded, mysterious communication |
| **Technical Focus** | Period-specific interaction design, cord physics, dialog triggering |

### The Scene's Purpose

The candlestick phone is a **period-authentic dialog trigger** that advances the narrative while maintaining the game's historical atmosphere. Unlike the phone booth (outdoor, first contact), the candlestick phone:

1. **Reinforces setting** - Late 1800s/early 1900s technology grounds the world
2. **Delivers narrative** - Specific plot information through dialog
3. **Demonstrates evolution** - Shows communication methods of the era
4. **Creates ritual** - The physical process of answering becomes memorable

This is an **environmentally-grounded exposition delivery** - information arrives through a period-appropriate device that feels natural in the space.

---

## 🎮 Game Design Perspective

### Creative Intent

**Why a candlestick phone specifically?**

| Phone Type | Era | Association |
|------------|-----|-------------|
| **Smartphone** | 2000s+ | Modern, breaks historical immersion |
| **Rotary Phone** | 1930s-1970s | Mid-20th century, specific era |
| **Candlestick Phone** | 1880s-1930s | **Turn of the century, period-authentic for this setting** |

The candlestick phone is **historically resonant** - it's what people of that era would have used, making it feel like a genuine artifact rather than a game mechanic.

### Design Philosophy

**Period Authenticity as World-Building:**

```
Modern game: "Press E for dialog"
          ↓
Player thinks: "I'm playing a game"

Period device: "The candlestick phone rings"
          ↓
Player thinks: "I'm in this world, experiencing its reality"
```

Every interaction that feels **historically authentic** reinforces the illusion that this world exists and has its own technology, culture, and history.

### Mood Building

The candlestick phone contributes to atmosphere through:

1. **Tactile Reality** - Separate mouthpiece and earpiece create physical ritual
2. **Auditory Character** - Distinctive ring from brass bells
3. **Temporal Anchoring** - Places the setting in a specific historical period
4. **Communication Mystery** - Who is calling on a dead line?

### Player Psychology

| Psychological Effect | How the Phone Achieves It |
|---------------------|---------------------------|
| **Immersion** | Period device feels like part of real history |
| **Ritual** | Separate mouthpiece/earpiece creates memorable interaction |
| **Mystery** | "How does this phone work? Who's calling?" |
| **Continuity** | Connects to phone booth, shows communication theme |

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
┌─────────────────────────────────────────────┐
│                                             │
│   [Office Interior - Desk Area]             │
│                                             │
│         ┌─────────────────────┐              │
│         │                     │              │
│         │   [DESK]            │              │
│         │                     │              │
│         │  ┌───────┐  ┌───┐   │              │
│         │  │ VIEW- │  │TEL│   │              │
│         │  │ MASTER│  │   │   │              │
│         │  └───────┘  └───┘   │              │
│         │                     │              │
│         │                     │              │
│         │         ┌─────┐     │              │
│         │         │CORD  │     │              │
│         │         └─────┘     │              │
│         └─────────────────────┘              │
│                                             │
│   Candlestick Phone:                        │
│   - Base on desk surface                    │
│   - Mouthpiece on stand                     │
│   - Receiver on hook at side                │
│   - Cord drapes naturally                   │
└─────────────────────────────────────────────┘
```

### Player Path

```
1. Player explores office after initial phone booth encounter
   ↓
2. Notices candlestick phone on desk
   ↓
3. [Trigger: Story progression or proximity]
   ↓
4. Phone begins ringing (distinctive brass bell sound)
   ↓
5. Player approaches desk
   ↓
6. "Press E to answer" prompt appears
   ↓
7. Player presses E - character reaches for receiver
   ↓
8. Cord physics activates as receiver lifts
   ↓
9. Player must hold receiver to ear (no mouthpiece pickup)
   ↓
10. Dialog plays - voice through earpiece
   ↓
11. Dialog completes, receiver can be hung up
    ↓
12. Phone enters quiet state or rings again later
```

### Atmosphere Layers

| Layer | Elements |
|-------|----------|
| **Visual** | Brass finish, wooden base, separate mouthpiece stand, curled cord |
| **Audio** | Brass bell ring (double-ring), voice through receiver, cord creak |
| **Tactile** | Lift receiver, hold to ear, no microphone to speak into |
| **Lighting** | Brass catches light, subtle reflections |

---

## Technical Implementation

### Candlestick Phone Data Structure

```javascript
export const CANDLESTICK_PHONE = {
  id: 'candlestick_phone_office',
  type: 'interactive_prop',

  // Splat/Model data
  splat: {
    file: '/assets/splats/candlestick-phone.ply',
    maxPoints: 600000,
    renderScale: 1.0
  },

  // World placement
  transform: {
    position: { x: 1.8, y: 0.78, z: -1.2 },  // On desk
    rotation: { x: 0, y: 15, z: 0 },
    scale: { x: 1, y: 1, z: 1 }
  },

  // Interaction configuration
  interaction: {
    type: 'candlestick_phone',
    prompt: 'Answer Phone',
    promptKey: 'e',
    maxDistance: 2.0,
    triggerRadius: 6.0
  },

  // Phone parts
  parts: {
    base: {
      mesh: 'phone_base',
      position: { x: 0, y: 0, z: 0 },
      material: 'brass',
      static: true
    },
    mouthpiece: {
      mesh: 'mouthpiece_stand',
      position: { x: 0, y: 0.12, z: -0.03 },
      material: 'brushed_metal',
      static: true,
      // Mouthpiece is separate from receiver - player holds receiver to ear
      // and speaks toward mouthpiece stand
    },
    receiver: {
      mesh: 'receiver',
      restPosition: { x: 0.06, y: 0.08, z: 0 },  // On hook
      liftPosition: { x: 0.04, y: 0.15, z: 0.05 },  // To ear
      material: 'brass',
      animated: true
    },
    hook: {
      mesh: 'receiver_hook',
      position: { x: 0.06, y: 0.08, z: 0 },
      material: 'brass'
    },
    bells: {
      mesh: 'bells',
      position: { x: 0, y: 0.15, z: 0 },
      material: 'brass',
      animated: true  // Animate when ringing
    },
    cord: {
      boneCount: 15,
      cordLength: 1.2,
      cordRadius: 0.003,
      anchorPoint: { x: 0, y: 0.05, z: 0.03 },
      receiverAttachment: { x: 0, y: 0, z: -0.04 },
      material: 'cotton_covered_copper'
    }
  },

  // Cord physics
  cord: {
    boneCount: 15,
    cordLength: 1.2,
    cordRadius: 0.003,
    stiffness: 0.6,
    damping: 0.6,
    gravity: { x: 0, y: -9.8, z: 0 },
    restPosition: {
      // Natural drape when receiver on hook
      curve: 'natural_gravity_drape'
    }
  },

  // Ringing configuration
  ringing: {
    sound: '/assets/audio/phones/candlestick-ring.ogg',
    pattern: 'double_ring',  // "brr-brrr" pattern
    interval: 3.0,  // Seconds between rings
    duration: 0.8,  // Ring duration
    bellAnimation: {
      amplitude: 0.02,  // Bell vibration amount
      frequency: 15  // Hz
    }
  },

  // Audio configuration
  audio: {
    ringing: {
      file: '/assets/audio/phones/candlestick-ring.ogg',
      volume: 0.9,
      loop: false,  // Loop manually for pattern
      spatial: true
    },
    click: {
      hook: '/assets/audio/phones/hook-click.ogg',
      receiver: '/assets/audio/phones/receiver-pickup.ogg'
    },
    cord: {
      creak: '/assets/audio/phones/cord-creak.ogg'
    },
    dialog: {
      filter: 'telephone',  // Apply EQ to simulate telephone speaker
      bandwidth: 'narrowband',  // 300Hz-3kHz typical telephone range
      distortion: 'subtle'  // Slight vintage character
    }
  },

  // Dialog sequence
  dialog: {
    id: 'candlestick_phone_call',
    trigger: {
      type: 'story_event',
      event: 'office_explored',
      delay: 2000  // ms after trigger
    },
    speaker: 'mysterious_caller',
    lines: [
      {
        text: "You found it. The office.",
        duration: 3.0,
        audio: '/assets/audio/dialog/candlestick_01.ogg'
      },
      {
        text: "I tried to warn you. The door... it shouldn't have opened.",
        duration: 4.5,
        audio: '/assets/audio/dialog/candlestick_02.ogg'
      },
      {
        text: "Listen carefully. There's a cylinder. Blue cardboard, yellow label.",
        duration: 4.0,
        audio: '/assets/audio/dialog/candlestick_03.ogg'
      },
      {
        text: "Edson cylinder. Plays the truth. Find it.",
        duration: 3.5,
        audio: '/assets/audio/dialog/candlestick_04.ogg'
      },
      {
        text: "And be careful. They hear the bells too.",
        duration: 3.0,
        audio: '/assets/audio/dialog/candlestick_05.ogg'
      }
    ],
    onComplete: {
      event: 'candlestick_call_complete',
      criteria: { 'heard_candlestick_call': true }
    }
  },

  // States
  states: {
    idle: {
      receiverOnHook: true,
      ringing: false,
      dialogActive: false
    },
    ringing: {
      receiverOnHook: true,
      ringing: true,
      dialogActive: false
    },
    answered: {
      receiverOnHook: false,
      ringing: false,
      dialogActive: true
    },
    finished: {
      receiverOnHook: true,
      ringing: false,
      dialogActive: false
    }
  }
};
```

### Candlestick Phone Manager

```javascript
/**
 * Manages candlestick phone interactions
 */
class CandlestickPhoneManager {
  constructor(scene) {
    this.scene = scene;
    this.phones = new Map();
    this.activeRings = new Map();
  }

  /**
   * Initialize a candlestick phone
   */
  async loadPhone(config) {
    const phone = {
      id: config.id,
      config: config,
      mesh: null,
      parts: {},
      cord: null,
      state: 'idle',
      audio: {},
      ringingInterval: null
    };

    // Load phone mesh
    phone.mesh = await this.loadPhoneMesh(config);

    // Load individual parts
    await this.loadParts(phone, config);

    // Setup cord physics
    phone.cord = this.setupCordPhysics(config);

    // Setup audio
    phone.audio = this.setupAudio(config);

    // Create interaction trigger
    this.createTriggerZone(phone, config);

    // Register dialog trigger
    this.setupDialogTrigger(phone, config);

    this.phones.set(config.id, phone);
    return phone;
  }

  /**
   * Load phone visual mesh
   */
  async loadPhoneMesh(splatConfig) {
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
   * Load phone parts
   */
  async loadParts(phone, config) {
    for (const [partName, partConfig] of Object.entries(config.parts)) {
      const part = await this.loadPart(partConfig);

      // Position relative to phone base
      const worldPos = new THREE.Vector3(
        config.transform.position.x + partConfig.position.x,
        config.transform.position.y + partConfig.position.y,
        config.transform.position.z + partConfig.position.z
      );

      part.position.copy(worldPos);

      phone.parts[partName] = {
        mesh: part,
        config: partConfig,
        worldPosition: worldPos
      };
    }
  }

  /**
   * Load individual part mesh
   */
  async loadPart(partConfig) {
    const mesh = await this.scene.assetLoader.load(partConfig.mesh);
    return mesh;
  }

  /**
   * Setup cord physics
   */
  setupCordPhysics(config) {
    const cordConfig = config.cord;
    const bones = [];
    const boneLength = cordConfig.cordLength / cordConfig.boneCount;

    for (let i = 0; i < cordConfig.boneCount; i++) {
      const bone = this.scene.physics.createSphere({
        radius: cordConfig.cordRadius,
        mass: 0.008,
        friction: 0.6,
        restitution: 0.1
      });

      // Calculate position along natural drape curve
      const t = i / (cordConfig.boneCount - 1);
      const curveOffset = Math.sin(t * Math.PI) * 0.1;  // Natural sag

      const y = config.transform.position.y + cordConfig.anchorPoint.y -
                (i * boneLength) - curveOffset;

      bone.setPosition(
        config.transform.position.x + cordConfig.anchorPoint.x,
        Math.max(y, config.transform.position.y),
        config.transform.position.z + cordConfig.anchorPoint.z
      );

      // Connect to previous bone
      if (i > 0) {
        this.scene.physics.createJoint(bones[i - 1], bone, {
          type: 'ball',
          position: {
            x: config.transform.position.x,
            y: config.transform.position.y + cordConfig.anchorPoint.y -
               ((i - 0.5) * boneLength),
            z: config.transform.position.z + cordConfig.anchorPoint.z
          }
        });
      }

      bones.push(bone);
    }

    // Anchor first bone to phone base
    const anchorBody = this.scene.physics.getStaticBody(
      config.transform.position
    );
    this.scene.physics.createFixedJoint(anchorBody, bones[0]);

    // Attach receiver to last bone
    const receiver = phone.parts.receiver;
    if (receiver) {
      receiver.attachedToBone = bones[bones.length - 1];
      receiver.attachOffset = config.parts.cord.receiverAttachment;
    }

    return {
      bones: bones,
      config: cordConfig,
      receiverAttached: true
    };
  }

  /**
   * Setup audio sources
   */
  setupAudio(config) {
    const audio = {};

    // Ringing sound
    audio.ringing = this.scene.audio.createSpatialSource({
      url: config.audio.ringing.file,
      volume: 0,
      loop: false,
      position: {
        x: config.transform.position.x,
        y: config.transform.position.y + 0.15,
        z: config.transform.position.z
      }
    });

    // One-shot sounds
    audio.hookClick = this.scene.audio.createOneShot(config.audio.click.hook);
    audio.receiverPickup = this.scene.audio.createOneShot(config.audio.click.receiver);
    audio.cordCreak = this.scene.audio.createOneShot(config.audio.cord.creak);

    return audio;
  }

  /**
   * Create trigger zone
   */
  createTriggerZone(phone, config) {
    const zone = this.scene.physics.createTrigger({
      shape: 'sphere',
      radius: config.interaction.triggerRadius,
      position: config.transform.position,
      onEnter: (entity) => this.onTriggerEnter(entity, phone.id),
      onExit: (entity) => this.onTriggerExit(entity, phone.id)
    });

    phone.triggerZone = zone;
  }

  /**
   * Setup dialog trigger
   */
  setupDialogTrigger(phone, config) {
    const trigger = config.dialog.trigger;

    if (trigger.type === 'story_event') {
      this.scene.gameManager.on(trigger.event, () => {
        setTimeout(() => {
          if (phone.state === 'idle') {
            this.startRinging(phone.id);
          }
        }, trigger.delay);
      });
    }
  }

  /**
   * Handle player entering trigger zone
   */
  onTriggerEnter(entity, phoneId) {
    if (entity !== this.scene.player) return;

    const phone = this.phones.get(phoneId);
    if (!phone) return;

    phone.playerInRange = true;

    // Enable interaction prompt
    this.scene.interaction.enablePrompt(phoneId);
  }

  /**
   * Handle player exiting trigger zone
   */
  onTriggerExit(entity, phoneId) {
    if (entity !== this.scene.player) return;

    const phone = this.phones.get(phoneId);
    if (!phone) return;

    phone.playerInRange = false;

    // Disable interaction prompt
    this.scene.interaction.disablePrompt(phoneId);
  }

  /**
   * Start phone ringing
   */
  startRinging(phoneId) {
    const phone = this.phones.get(phoneId);
    if (!phone || phone.state !== 'idle') return;

    phone.state = 'ringing';

    // Start double-ring pattern
    this.startRingPattern(phone);

    // Animate bells
    this.startBellAnimation(phone);
  }

  /**
   * Start ringing pattern
   */
  startRingPattern(phone) {
    const config = phone.config.ringing;

    const playRing = () => {
      if (phone.state !== 'ringing') return;

      // Play ring sound
      phone.audio.ringing.setVolume(config.sound ? config.sound.volume : 0.9);
      phone.audio.ringing.play();

      // Schedule next ring
      phone.ringingInterval = setTimeout(playRing, config.interval * 1000);
    };

    playRing();
  }

  /**
   * Animate bells vibrating
   */
  startBellAnimation(phone) {
    const config = phone.config.ringing.bellAnimation;
    const bells = phone.parts.bells;

    if (!bells) return;

    const animate = () => {
      if (phone.state !== 'ringing') {
        // Reset bell position
        bells.mesh.rotation.z = 0;
        return;
      }

      const time = performance.now() / 1000;
      const vibration = Math.sin(time * Math.PI * 2 * config.frequency) *
                       config.amplitude;

      bells.mesh.rotation.z = vibration;

      requestAnimationFrame(animate);
    };

    animate();
  }

  /**
   * Stop ringing
   */
  stopRinging(phoneId) {
    const phone = this.phones.get(phoneId);
    if (!phone) return;

    if (phone.ringingInterval) {
      clearTimeout(phone.ringingInterval);
      phone.ringingInterval = null;
    }

    phone.audio.ringing.stop();
  }

  /**
   * Handle player interacting with phone (answering)
   */
  async onInteract(phoneId) {
    const phone = this.phones.get(phoneId);
    if (!phone || phone.state !== 'ringing') return;

    phone.state = 'answering';

    // Stop ringing
    this.stopRinging(phoneId);

    // Play pickup sound
    phone.audio.receiverPickup.play();

    // Animate receiver lift
    await this.animateReceiverLift(phone);

    // Start dialog
    this.startDialog(phone);
  }

  /**
   * Animate receiver lifting to ear
   */
  async animateReceiverLift(phone) {
    const duration = 700;
    const startTime = performance.now();

    const receiver = phone.parts.receiver;
    const restPos = new THREE.Vector3(
      receiver.worldPosition.x,
      receiver.worldPosition.y,
      receiver.worldPosition.z
    );

    // Calculate ear position relative to phone
    const earPos = new THREE.Vector3(
      phone.config.transform.position.x + 0.15,
      this.scene.player.camera.position.y - 0.1,
      phone.config.transform.position.z + 0.2
    );

    return new Promise((resolve) => {
      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = this.easeOutQuad(progress);

        // Move receiver
        const currentPos = new THREE.Vector3().lerpVectors(restPos, earPos, eased);

        // Update cord physics to follow receiver
        if (phone.cord && phone.cord.receiverAttached) {
          const lastBone = phone.cord.bones[phone.cord.bones.length - 1];
          lastBone.setPosition(currentPos.x, currentPos.y, currentPos.z);
        }

        // Random cord creak
        if (Math.random() < 0.03) {
          phone.audio.cordCreak.play();
        }

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          phone.state = 'answered';
          resolve();
        }
      };

      animate();
    });
  }

  /**
   * Start dialog sequence
   */
  startDialog(phone) {
    const dialogConfig = phone.config.dialog;

    // Apply telephone filter to dialog audio
    const filterConfig = phone.config.audio.dialog;

    this.scene.dialog.play(dialogConfig, {
      audioFilter: {
        type: 'bandpass',
        lowFreq: 300,
        highFreq: 3000,
        distortion: filterConfig.distortion
      },
      onComplete: () => this.onDialogComplete(phone)
    });
  }

  /**
   * Handle dialog completion
   */
  async onDialogComplete(phone) {
    // Set completion criteria
    const onComplete = phone.config.dialog.onComplete;
    if (onComplete.criteria) {
      for (const [key, value] of Object.entries(onComplete.criteria)) {
        this.scene.gameManager.setCriteria(key, value);
      }
    }

    // Trigger completion event
    if (onComplete.event) {
      this.scene.gameManager.emit(onComplete.event);
    }

    // Hang up receiver
    await this.animateReceiverHangup(phone);

    phone.state = 'finished';
  }

  /**
   * Animate receiver hanging up
   */
  async animateReceiverHangup(phone) {
    const duration = 500;
    const startTime = performance.now();

    const receiver = phone.parts.receiver;
    const currentPos = receiver.getWorldPosition().clone();

    const restPos = receiver.worldPosition.clone();

    return new Promise((resolve) => {
      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = this.easeInQuad(progress);

        const newPos = new THREE.Vector3().lerpVectors(currentPos, restPos, eased);

        // Update cord physics
        if (phone.cord && phone.cord.receiverAttached) {
          const lastBone = phone.cord.bones[phone.cord.bones.length - 1];
          lastBone.setPosition(newPos.x, newPos.y, newPos.z);
        }

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          // Play hook click
          phone.audio.hookClick.play();
          resolve();
        }
      };

      animate();
    });
  }

  /**
   * Update phone state
   */
  update(deltaTime) {
    for (const phone of this.phones.values()) {
      // Update cord physics
      if (phone.cord) {
        this.updateCordPhysics(phone.cord, deltaTime);
      }
    }
  }

  /**
   * Update cord physics
   */
  updateCordPhysics(cord, deltaTime) {
    // Ensure cord doesn't clip through desk
    const deskHeight = this.scene.deskHeight || 0.75;

    for (const bone of cord.bones) {
      const pos = bone.getPosition();
      if (pos.y < deskHeight) {
        pos.y = deskHeight + 0.01;
        bone.setPosition(pos.x, pos.y, pos.z);
      }
    }
  }

  /**
   * Easing functions
   */
  easeOutQuad(t) {
    return t * (2 - t);
  }

  easeInQuad(t) {
    return t * t;
  }
}
```

### Telephone Audio Filter

```javascript
/**
 * Audio filter for telephone effect
 */
class TelephoneAudioFilter {
  constructor(audioContext) {
    this.context = audioContext;
    this.filters = [];
  }

  /**
   * Create telephone filter chain
   */
  createFilterChain(source) {
    // High-pass filter at 300Hz
    const highPass = this.context.createBiquadFilter();
    highPass.type = 'highpass';
    highPass.frequency.value = 300;
    highPass.Q.value = 0.5;

    // Low-pass filter at 3kHz
    const lowPass = this.context.createBiquadFilter();
    lowPass.type = 'lowpass';
    lowPass.frequency.value = 3000;
    lowPass.Q.value = 0.5;

    // Subtle distortion for vintage character
    const distortion = this.context.createWaveShaper();
    distortion.curve = this.makeDistortionCurve(50);  // 50% distortion
    distortion.oversample = '4x';

    // Connect chain
    source.connect(highPass);
    highPass.connect(lowPass);
    lowPass.connect(distortion);

    this.filters = [highPass, lowPass, distortion];

    return distortion;
  }

  /**
   * Create distortion curve
   */
  makeDistortionCurve(amount) {
    const samples = 44100;
    const curve = new Float32Array(samples);
    const deg = Math.PI / 180;

    for (let i = 0; i < samples; i++) {
      const x = (i * 2) / samples - 1;
      curve[i] = ((3 + amount) * x * 20 * deg) / (Math.PI + amount * Math.abs(x));
    }

    return curve;
  }

  /**
   * Disconnect and cleanup
   */
  disconnect() {
    for (const filter of this.filters) {
      filter.disconnect();
    }
    this.filters = [];
  }
}
```

---

## How To Build A Scene Like This

### Step 1: Research Period Technology

```javascript
const research = {
  candlestickPhone: {
    era: '1880s-1930s',

    features: [
      'Separate mouthpiece and receiver',
      'Receiver hangs on hook',
      'Mouthpiece fixed on stand',
      'Brass bells for ringer',
      'Curled cloth-covered cord'
    ],

    interaction: 'Lift receiver, hold to ear, speak toward mouthpiece'
  }
};
```

### Step 2: Design the Interaction Ritual

```javascript
const ritual = {
  steps: [
    'Phone rings (distinctive double-ring)',
    'Player approaches',
    'Player presses interaction key',
    'Character reaches for receiver',
    'Cord physics activates',
    'Receiver lifts to ear',
    'Dialog plays through receiver',
    'Receiver returns to hook'
  ],

  keyMoments: [
    'Ring creates urgency',
    'Lift is tactile and memorable',
    'Audio through receiver feels intimate'
  ]
};
```

### Step 3: Create Authentic Audio

```javascript
const audioDesign = {
  ring: {
    type: 'brass_bell',
    pattern: 'double_ring',  // "brr-brrr, pause, brr-brrr"
    recording: 'Real candlestick phone or sample library'
  },

  dialog: {
    filter: 'bandpass 300Hz-3kHz',
    character: 'Vintage telephone frequency response',
    processing: 'Add subtle distortion'
  },

  interaction: {
    click: 'Receiver leaving/returning to hook',
    creak: 'Cord moving and stretching'
  }
};
```

### Step 4: Setup Cord Physics

```javascript
const cordSetup = {
  bones: '10-20 segments for realistic movement',

  drape: 'Natural gravity curve when receiver on hook',

  follow: 'Last bone follows receiver during lift',

  collision: 'Prevent cord from passing through desk'
};
```

### Step 5: Integrate Dialog

```javascript
const dialogIntegration = {
  trigger: 'Story event or proximity',

  content: 'Plot-relevant information',

  delivery: 'Voice through filtered audio',

  completion: 'Set criteria, trigger next event'
};
```

---

## Variations For Your Game

### Variation 1: Puzzle Integration

```javascript
const puzzleVersion = {
  rings: 'Only when puzzle conditions met',

  content: 'Hints or codes for current puzzle',

  repeatable: true
};
```

### Variation 2: Horror Variant

```javascript
const horrorVersion = {
  rings: 'At random or during dangerous moments',

  voice: 'Threatening or warning',

  effect: 'Attracts enemies or signals danger'
};
```

### Variation 3: Multiple Phones

```javascript
const multiPhoneVersion = {
  several: 'Multiple candlestick phones in different locations',

  each: 'Different caller or information',

  sequence: 'Story progression through phone calls'
};
```

---

## Performance Considerations

```javascript
const optimization = {
  cord: 'Reduce bone count on lower quality settings',

  audio: 'Pre-filter dialog audio instead of runtime filtering',

  bells: 'Simple rotation animation, not full physics'
};
```

---

## Common Mistakes Beginners Make

### Mistake 1: Inaccurate Period Detail

```javascript
// BAD: Modern phone with candlestick skin
const badPeriod = {
  design: 'Modern smartphone behavior',
  visual: 'Candlestick phone mesh'
};

// GOOD: Authentic period interaction
const goodPeriod = {
  design: 'Separate mouthpiece/receiver, no speaking into receiver',
  visual: 'Accurate candlestick phone'
};
```

### Mistake 2: No Audio Character

```javascript
// BAD: Clean, full-frequency dialog
const badAudio = {
  dialog: 'Standard recording, no filtering'
};

// GOOD: Telephone-character audio
const goodAudio = {
  dialog: 'Bandpass filtered, subtle distortion',
  ring: 'Authentic brass bell'
};
```

### Mistake 3: Weak Cord Physics

```javascript
// BAD: Static cord or too few bones
const badCord = {
  bones: 3,
  physics: 'static animation'
};

// GOOD: Realistic cord movement
const goodCord = {
  bones: 15,
  physics: 'full simulation'
};
```

---

## Related Systems

- **Phone Booth** - Outdoor counterpart for same mechanic
- **Dialog System** - For delivering narrative content
- **Cord Physics** - Shared with other interactive objects
- **Audio System** - For telephone audio filtering

---

## References

- **Shadow Engine Documentation**: `docs/`
- **Cord Physics**: See *Physics System*
- **Dialog System**: See *DialogManager*

---

**RALPH_STATUS:**
- **Status**: Candlestick Phone Scene documentation complete
- **Files Created**: `docs/generated/14-scene-case-studies/candlestick-phone-scene.md`
- **Related Documentation**: All Phase 14 scene case studies
- **Next**: Amplifier Cord Scene documentation
