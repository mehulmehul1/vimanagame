# Amplifier + Cord Scene - Physics Puzzle Interaction

**Scene Case Study #16**

---

## What You Need to Know First

- **Gaussian Splatting Basics** (See: *Rendering System*)
- **Interactive Object System** (See: *Interactive Objects*)
- **Cord Physics** (See: *Physics System*)
- **State Machine Design** (See: *Game Architecture*)

---

## Scene Overview

| Property | Value |
|----------|-------|
| **Location** | Office interior, in corner |
| **Narrative Context** | Physics puzzle - player must connect amplifier to progress |
| **Player Experience** | Discovery → Examination → Pick Up Cord → Carry → Connect → Resolution |
| **Atmosphere** | Tactile, puzzle-solving, physical interaction satisfaction |
| **Technical Focus** | Physics-based cord carrying, connection detection, state progression |

### The Scene's Purpose

The Amplifier + Cord scene is a **physics-based puzzle interaction** that:
1. **Creates spatial puzzle** - player must carry object from point A to point B
2. **Demonstrates physics interaction** - cord drapes and swings realistically
3. **Provides tactile satisfaction** - physical connection feels rewarding
4. **Advances narrative** - connecting amplifier enables next story element

Unlike other interactive objects that are stationary (phone booth, candlestick phone), this interaction involves **carrying a physics object through space**, making the player's movement and navigation part of the puzzle.

---

## 🎮 Game Design Perspective

### Creative Intent

**Why a cord-carrying puzzle? Why not a simple switch or key?**

| Puzzle Type | Player Experience |
|-------------|-------------------|
| **Switch/Button** | Static - walk up and press |
| **Key Hunt** | Collection - find and bring |
| **Cord Carry** | **Physical - object affects movement, creates spatial awareness** |

The cord-carrying puzzle creates **embodied gameplay** - the physics object becomes part of the player's avatar. When you carry a cord:

1. **Movement changes** - you're aware of the trailing cord
2. **Routing matters** - you must navigate around obstacles
3. **Connection is deliberate** - you manually place the plug
4. **Success feels earned** - physical effort invested

### Design Philosophy

**Physicality as Puzzle:**

```
Simple interaction: "Press E to connect"
    ↓
Player thinks: "I clicked a thing"

Physical interaction: "Pick up cord, carry it across room,
                      navigate around desk, insert into socket"
    ↓
Player thinks: "I did this. I connected these things."
```

The additional effort creates **investment and satisfaction**. The puzzle isn't hard intellectually, but the physicality makes it memorable.

### Mood Building

The amplifier scene contributes to atmosphere through:

1. **Grounded Reality** - Mundane task in supernatural space
2. **Tactile Engagement** - Physical interaction with world
3. **Problem-Solving** - Player figure out how to progress
4. **Quiet Moment** - Non-threatening interaction breaks tension

### Player Psychology

| Psychological Effect | How the Scene Achieves It |
|---------------------|---------------------------|
| **Agency** | I chose to carry this, I connected it |
| **Competence** | I figured out what to do |
| **Satisfaction** | Plug snapping in feels good |
| **Progress** | Connection advances story |

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
┌─────────────────────────────────────────────┐
│                                             │
│   [Office Interior]                         │
│                                             │
│         ┌─────────────────────┐              │
│         │                     │              │
│         │   [DESK]            │              │
│         │                     │              │
│         │  [PHONE] [VIEW]     │              │
│         │                     │              │
│         └─────────────────────┘              │
│                                             │
│   [Corner]                    [Wall Outlet]  │
│     ┌───┐                         ◉          │
│     │AMP│                       SOCKET       │
│     │   │   ← Cord must connect →           │
│     └───┘                                    │
│                                             │
│   Player must:                              │
│   1. Find amplifier in corner               │
│   2. Pick up cord from amplifier            │
│   3. Navigate around desk to outlet          │
│   4. Insert plug                            │
└─────────────────────────────────────────────┘
```

### Player Path

```
1. Player enters office, explores
   ↓
2. Notices amplifier in corner (unpowered, quiet)
   ↓
3. Examines amplifier - sees cord hanging loose
   ↓
4. "Press E to pick up cord" prompt appears
   ↓
5. Player presses E - character grabs cord end
   ↓
6. Cord physics activates - drapes from player hand
   ↓
7. Player moves toward wall outlet
   ↓
8. Cord trails behind, drapes over furniture
   ↓
9. Player reaches outlet
   ↓
10. "Press E to connect" prompt appears
    ↓
11. Player presses E - plug animates to socket
    ↓
12. Connection made - amplifier powers on
    ↓
13. Audio/visual feedback confirms success
    ↓
14. Story progression unlocked
```

### Atmosphere Layers

| Layer | Elements |
|-------|----------|
| **Visual** | Vintage amplifier, loose cord, wall outlet, power indicator |
| **Audio** | Cord rustling, plug insertion click, amplifier power-on hum |
| **Tactile** | Cord weight, dragging physics, connection resistance |
| **Lighting** | Amplifier power LED, warm glow from tubes |

---

## Technical Implementation

### Amplifier Data Structure

```javascript
export const AMPLIFIER = {
  id: 'amplifier_office',
  type: 'interactive_prop',

  // Splat/Model data
  splat: {
    file: '/assets/splats/amplifier.ply',
    maxPoints: 1200000,
    renderScale: 1.0
  },

  // World placement
  transform: {
    position: { x: -3.5, y: 0, z: -3 },  // Corner of office
    rotation: { x: 0, y: 45, z: 0 },
    scale: { x: 1, y: 1, z: 1 }
  },

  // Interaction configuration
  interaction: {
    type: 'amplifier',
    prompt: 'Pick Up Cord',
    promptKey: 'e',
    maxDistance: 2.0
  },

  // Amplifier parts
  parts: {
    chassis: {
      mesh: 'amp_chassis',
      position: { x: 0, y: 0.4, z: 0 },
      material: 'wooden_cabinet',
      static: true
    },
    speaker: {
      mesh: 'amp_speaker',
      position: { x: 0, y: 0.5, z: 0.1 },
      material: 'speaker_mesh',
      static: true
    },
    knobs: {
      mesh: 'amp_knobs',
      position: { x: 0, y: 0.6, z: -0.1 },
      material: 'brass',
      static: true
    },
    powerLight: {
      mesh: 'power_led',
      position: { x: 0.15, y: 0.3, z: -0.15 },
      material: 'emissive_red',
      animated: true,
      initialState: 'off'  // Off until connected
    }
  },

  // Cord configuration
  cord: {
    id: 'amplifier_cord',
    boneCount: 25,
    cordLength: 3.0,  // Long enough to reach across room
    cordRadius: 0.004,
    anchorPoint: {  // Where cord attaches to amplifier
      x: -0.2,
      y: 0.1,
      z: 0
    },
    plugEnd: {
      mesh: 'ac_plug',
      position: { x: 0, y: 0, z: 0 }
    },
    stiffness: 0.4,  // More flexible than phone cord
    damping: 0.4,
    gravity: { x: 0, y: -9.8, z: 0 }
  },

  // Wall outlet target
  outlet: {
    id: 'wall_outlet',
    position: { x: 4.0, y: 0.3, z: -3 },  // Opposite wall
    rotation: { x: 0, y: 180, z: 0 },
    socketRadius: 0.08,  // Connection detection radius
    prompt: 'Connect Cord',
    promptKey: 'e'
  },

  // States
  states: {
    disconnected: {
      cordAttached: true,
      cordCarried: false,
      amplifierPowered: false,
      canPickUp: true
    },
    carrying: {
      cordAttached: true,
      cordCarried: true,
      amplifierPowered: false,
      canPickUp: false
    },
    connected: {
      cordAttached: true,
      cordCarried: false,
      amplifierPowered: true,
      canPickUp: false
    }
  },

  // Audio configuration
  audio: {
    pickup: {
      cord: '/assets/audio/amp/cord-pickup.ogg'
    },
    movement: {
      rustle: '/assets/audio/amp/cord-rustle.ogg'
    },
    connection: {
      insert: '/assets/audio/amp/plug-insert.ogg',
      powerOn: '/assets/audio/amp/amp-power-on.ogg',
      hum: '/assets/audio/amp/amp-hum.ogg'
    }
  },

  // Connection effects
  onConnect: {
    powerLight: {
      from: { r: 0, g: 0, b: 0 },
      to: { r: 1, g: 0, b: 0 },  // Red
      duration: 500
    },
    sound: {
      play: 'powerOn',
      loop: 'hum'
    },
    story: {
      event: 'amplifier_connected',
      criteria: { 'amplifier_powered': true }
    }
  }
};
```

### Amplifier Interaction Manager

```javascript
/**
 * Manages amplifier and cord-carrying puzzle
 */
class AmplifierManager {
  constructor(scene) {
    this.scene = scene;
    this.amplifiers = new Map();
  }

  /**
   * Initialize an amplifier
   */
  async loadAmplifier(config) {
    const amplifier = {
      id: config.id,
      config: config,
      mesh: null,
      parts: {},
      cord: null,
      state: 'disconnected',
      outlet: null
    };

    // Load amplifier mesh
    amplifier.mesh = await this.loadAmplifierMesh(config);

    // Load parts
    await this.loadParts(amplifier, config);

    // Setup cord physics
    amplifier.cord = this.setupCordPhysics(config);

    // Create outlet
    amplifier.outlet = this.createOutlet(config);

    // Create interaction triggers
    this.createInteractions(amplifier, config);

    this.amplifiers.set(config.id, amplifier);
    return amplifier;
  }

  /**
   * Load amplifier visual mesh
   */
  async loadAmplifierMesh(splatConfig) {
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
   * Load amplifier parts
   */
  async loadParts(amplifier, config) {
    for (const [partName, partConfig] of Object.entries(config.parts)) {
      const part = await this.loadPart(partConfig);

      const worldPos = new THREE.Vector3(
        config.transform.position.x + partConfig.position.x,
        config.transform.position.y + partConfig.position.y,
        config.transform.position.z + partConfig.position.z
      );

      part.position.copy(worldPos);
      part.material = partConfig.material;

      amplifier.parts[partName] = {
        mesh: part,
        config: partConfig,
        worldPosition: worldPos
      };
    }
  }

  /**
   * Load individual part
   */
  async loadPart(partConfig) {
    return await this.scene.assetLoader.load(partConfig.mesh);
  }

  /**
   * Setup cord physics
   */
  setupCordPhysics(config) {
    const cordConfig = config.cord;
    const bones = [];
    const boneLength = cordConfig.cordLength / cordConfig.boneCount;

    // Calculate cord anchor in world space
    const anchorWorld = new THREE.Vector3(
      config.transform.position.x + cordConfig.anchorPoint.x,
      config.transform.position.y + cordConfig.anchorPoint.y,
      config.transform.position.z + cordConfig.anchorPoint.z
    );

    for (let i = 0; i < cordConfig.boneCount; i++) {
      const bone = this.scene.physics.createSphere({
        radius: cordConfig.cordRadius,
        mass: 0.005,
        friction: 0.4,
        restitution: 0.2
      });

      // Natural drape on ground when not carried
      const t = i / (cordConfig.boneCount - 1);
      const drapeHeight = Math.sin(t * Math.PI) * 0.15;

      const y = anchorWorld.y - drapeHeight;

      bone.setPosition(
        anchorWorld.x,
        Math.max(y, 0.02),  // Above floor
        anchorWorld.z
      );

      // Connect to previous bone
      if (i > 0) {
        this.scene.physics.createJoint(bones[i - 1], bone, {
          type: 'ball',
          position: {
            x: anchorWorld.x,
            y: anchorWorld.y - ((i - 0.5) * boneLength),
            z: anchorWorld.z
          }
        });
      }

      bones.push(bone);
    }

    // Anchor first bone to amplifier
    const anchorBody = this.scene.physics.getStaticBody(anchorWorld);
    this.scene.physics.createFixedJoint(anchorBody, bones[0]);

    // Plug is at the last bone
    const plug = {
      bone: bones[bones.length - 1],
      mesh: null,
      carried: false,
      attachedToPlayer: false
    };

    return {
      bones: bones,
      plug: plug,
      config: cordConfig,
      anchorPoint: anchorWorld
    };
  }

  /**
   * Create wall outlet
   */
  createOutlet(config) {
    const outletConfig = config.outlet;

    const outlet = {
      position: new THREE.Vector3(
        outletConfig.position.x,
        outletConfig.position.y,
        outletConfig.position.z
      ),
      rotation: outletConfig.rotation,
      socketRadius: outletConfig.socketRadius,
      connected: false
    };

    // Create trigger zone for connection
    outlet.triggerZone = this.scene.physics.createTrigger({
      shape: 'sphere',
      radius: outletConfig.socketRadius * 2,
      position: outletConfig.position,
      onEnter: (entity) => this.onOutletEnter(entity, config.id),
      onExit: (entity) => this.onOutletExit(entity, config.id)
    });

    return outlet;
  }

  /**
   * Create interaction zones
   */
  createInteractions(amplifier, config) {
    // Cord pickup interaction
    this.scene.interaction.register({
      id: `${config.id}_pickup`,
      type: 'cord_pickup',
      prompt: config.interaction.prompt,
      key: config.interaction.promptKey,
      maxDistance: config.interaction.maxDistance,
      position: amplifier.cord.plug.bone.getPosition(),
      onInteract: () => this.onCordPickup(config.id)
    });
  }

  /**
   * Handle player entering outlet trigger zone
   */
  onOutletEnter(entity, amplifierId) {
    if (entity !== this.scene.player) return;

    const amplifier = this.amplifiers.get(amplifierId);
    if (!amplifier || amplifier.state !== 'carrying') return;

    // Enable connect prompt
    this.scene.interaction.enablePrompt(`${amplifierId}_connect`, {
      prompt: amplifier.config.outlet.prompt,
      key: amplifier.config.outlet.promptKey,
      onInteract: () => this.onConnect(amplifierId)
    });

    amplifier.playerNearOutlet = true;
  }

  /**
   * Handle player exiting outlet trigger zone
   */
  onOutletExit(entity, amplifierId) {
    if (entity !== this.scene.player) return;

    const amplifier = this.amplifiers.get(amplifierId);
    if (!amplifier) return;

    this.scene.interaction.disablePrompt(`${amplifierId}_connect`);
    amplifier.playerNearOutlet = false;
  }

  /**
   * Handle cord pickup
   */
  async onCordPickup(amplifierId) {
    const amplifier = this.amplifiers.get(amplifierId);
    if (!amplifier || amplifier.state !== 'disconnected') return;

    amplifier.state = 'carrying';

    // Play pickup sound
    this.scene.audio.playOneShot(amplifier.config.audio.pickup.cord);

    // Attach plug to player
    const plug = amplifier.cord.plug;
    plug.carried = true;
    plug.attachedToPlayer = true;

    // Disable pickup prompt
    this.scene.interaction.unregister(`${amplifierId}_pickup`);

    // Start cord movement audio
    amplifier.movementAudio = this.scene.audio.createLoopingSource({
      url: amplifier.config.audio.movement.rustle,
      volume: 0,
      loop: true,
      spatial: true
    });

    console.log('Cord picked up - carry to outlet');
  }

  /**
   * Handle connection to outlet
   */
  async onConnect(amplifierId) {
    const amplifier = this.amplifiers.get(amplifierId);
    if (!amplifier || amplifier.state !== 'carrying') return;

    amplifier.state = 'connecting';

    // Play connection sounds
    this.scene.audio.playOneShot(amplifier.config.audio.connection.insert);
    await this.delay(200);
    this.scene.audio.playOneShot(amplifier.config.audio.connection.powerOn);

    // Animate plug to socket
    await this.animatePlugConnection(amplifier);

    // Disable connect prompt
    this.scene.interaction.disablePrompt(`${amplifierId}_connect`);

    // Stop movement audio
    if (amplifier.movementAudio) {
      amplifier.movementAudio.stop();
      delete amplifier.movementAudio;
    }

    // Power on amplifier
    await this.powerOnAmplifier(amplifier);

    amplifier.state = 'connected';
    amplifier.outlet.connected = true;

    // Set story criteria
    const onConnect = amplifier.config.onConnect;
    if (onConnect.story) {
      if (onConnect.story.criteria) {
        for (const [key, value] of Object.entries(onConnect.story.criteria)) {
          this.scene.gameManager.setCriteria(key, value);
        }
      }
      if (onConnect.story.event) {
        this.scene.gameManager.emit(onConnect.story.event);
      }
    }

    console.log('Amplifier connected!');
  }

  /**
   * Animate plug connecting to socket
   */
  async animatePlugConnection(amplifier) {
    const duration = 300;
    const startTime = performance.now();

    const plug = amplifier.cord.plug;
    const outlet = amplifier.outlet;

    const startPos = plug.bone.getPosition().clone();
    const endPos = outlet.position.clone();

    return new Promise((resolve) => {
      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = this.easeOutQuad(progress);

        const currentPos = new THREE.Vector3().lerpVectors(startPos, endPos, eased);
        plug.bone.setPosition(currentPos.x, currentPos.y, currentPos.z);

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          resolve();
        }
      };

      animate();
    });
  }

  /**
   * Power on amplifier effects
   */
  async powerOnAmplifier(amplifier) {
    const onConnect = amplifier.config.onConnect;

    // Animate power light
    if (amplifier.parts.powerLight) {
      await this.animatePowerLight(
        amplifier.parts.powerLight,
        onConnect.powerLight
      );
    }

    // Start hum sound
    const humSound = this.scene.audio.createLoopingSource({
      url: onConnect.sound.loop,
      volume: 0.3,
      loop: true,
      spatial: true,
      position: amplifier.config.transform.position
    });
    humSound.play();

    amplifier.powerHum = humSound;
  }

  /**
   * Animate power light turning on
   */
  async animatePowerLight(powerLight, config) {
    const duration = config.duration;
    const startTime = performance.now();
    const startColor = new THREE.Color(config.from.r, config.from.g, config.from.b);
    const endColor = new THREE.Color(config.to.r, config.to.g, config.to.b);

    return new Promise((resolve) => {
      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const currentColor = new THREE.Color().lerpColors(startColor, endColor, progress);
        powerLight.mesh.material.color = currentColor;
        powerLight.mesh.material.emissive = currentColor;

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          resolve();
        }
      };

      animate();
    });
  }

  /**
   * Update amplifier and cord physics
   */
  update(deltaTime) {
    for (const amplifier of this.amplifiers.values()) {
      this.updateCordPhysics(amplifier, deltaTime);
      this.updateCarriedCord(amplifier, deltaTime);
    }
  }

  /**
   * Update cord physics
   */
  updateCordPhysics(amplifier, deltaTime) {
    const cord = amplifier.cord;

    // Keep bones above floor
    for (const bone of cord.bones) {
      const pos = bone.getPosition();
      if (pos.y < 0.02) {
        pos.y = 0.02;
        bone.setPosition(pos.x, pos.y, pos.z);
      }
    }
  }

  /**
   * Update carried cord - follows player
   */
  updateCarriedCord(amplifier, deltaTime) {
    if (amplifier.state !== 'carrying') return;

    const plug = amplifier.cord.plug;
    const playerHandPos = this.getPlayerHandPosition();

    // Update plug position to follow player hand
    plug.bone.setPosition(playerHandPos.x, playerHandPos.y, playerHandPos.z);

    // Play movement audio based on cord velocity
    const velocity = this.calculateCordVelocity(amplifier);
    if (amplifier.movementAudio && velocity > 0.1) {
      const volume = Math.min(velocity / 2.0, 1.0) * 0.3;
      amplifier.movementAudio.setVolume(volume);
    } else if (amplifier.movementAudio) {
      amplifier.movementAudio.setVolume(0);
    }
  }

  /**
   * Get player hand position for cord attachment
   */
  getPlayerHandPosition() {
    const playerPos = this.scene.player.position;
    const playerForward = this.scene.player.forward;

    // Hand position extends slightly in front of player
    return new THREE.Vector3(
      playerPos.x + playerForward.x * 0.3,
      playerPos.y - 0.3,  // Below eye level
      playerPos.z + playerForward.z * 0.3
    );
  }

  /**
   * Calculate cord velocity for audio
   */
  calculateCordVelocity(amplifier) {
    const plug = amplifier.cord.plug;
    const currentPos = plug.bone.getPosition();

    if (!plug.lastPos) {
      plug.lastPos = currentPos.clone();
      return 0;
    }

    const velocity = currentPos.distanceTo(plug.lastPos) / 0.016;  // Per frame at 60fps
    plug.lastPos = currentPos.clone();

    return velocity;
  }

  /**
   * Easing function
   */
  easeOutQuad(t) {
    return t * (2 - t);
  }

  /**
   * Delay helper
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

### Scene Integration

```javascript
/**
 * Amplifier integrated into office scene
 */
class OfficeScene extends BaseScene {
  async onLoad() {
    // Load base scene
    await this.loadSplat('/assets/splats/office.ply');

    // Initialize amplifier manager
    this.amplifierManager = new AmplifierManager(this);

    // Load amplifier
    const amplifier = await this.amplifierManager.loadAmplifier(AMPLIFIER);

    // Set player spawn
    this.player.spawn.set({ x: 0, y: 1.7, z: 2 });

    // Listen for connection event
    this.gameManager.on('amplifier_connected', () => {
      console.log('Amplifier connected - story progression unlocked');
    });
  }

  onUpdate(deltaTime) {
    // Update amplifier and cord physics
    this.amplifierManager.update(deltaTime);
  }
}
```

---

## How To Build A Scene Like This

### Step 1: Define the Puzzle Goal

```javascript
const puzzleGoal = {
  objective: 'Connect amplifier to power',

  playerUnderstanding: {
    visual: 'Amplifier has cord, outlet on wall',
    spatial: 'Must carry cord from A to B',
    action: 'Insert plug into socket'
  },

  satisfaction: 'Physical connection feels rewarding'
};
```

### Step 2: Design the Cord Physics

```javascript
const cordDesign = {
  length: 'Long enough to reach, but creates routing challenge',

  bones: 'More bones = smoother physics, more CPU',

  flexibility: {
    stiff: 'Cable-like, holds shape',
    loose: 'Rope-like, drapes more'
  }
};
```

### Step 3: Create Spatial Relationship

```javascript
const spatialDesign = {
  startPosition: 'Amplifier in corner, somewhat hidden',

  endPosition: 'Outlet visible but requires navigation',

  obstacles: 'Desk, furniture create routing challenge',

  path: 'Player must move around, not straight line'
};
```

### Step 4: Implement State Machine

```javascript
const states = {
  disconnected: 'Cord hanging loose, can be picked up',

  carrying: 'Player has plug, cord drags behind',

  connected: 'Plug in socket, amplifier powered'
};
```

### Step 5: Add Feedback

```javascript
const feedback = {
  visual: 'Power light turns on',

  audio: 'Click of plug, hum of amplifier',

  tactile: 'Cord drags and swings'
};
```

---

## Variations For Your Game

### Variation 1: Multiple Connections

```javascript
const multiConnection = {
  design: 'Several items to connect in sequence',

  complexity: 'Each connection enables next'
};
```

### Variation 2: Timed Puzzle

```javascript
const timedVersion = {
  constraint: 'Must connect within time limit',

  tension: 'Amplifier powers something urgent'
};
```

### Variation 3: Hazard Environment

```javascript
const hazardVersion = {
  danger: 'Cord can't touch certain surfaces',

  challenge: 'Navigate around hazards'
};
```

---

## Performance Considerations

```javascript
const optimization = {
  boneCount: 'Reduce on lower quality settings',

  physics: 'Use simpler collision for cord',

  audio: 'Only play rustle when moving'
};
```

---

## Common Mistakes Beginners Make

### Mistake 1: Cord Too Short/Long

```javascript
// BAD: Cord too short to reach
const badLength = {
  cordLength: 0.5,
  result: 'Frustrating, impossible puzzle'
};

// GOOD: Cord reaches with some routing required
const goodLength = {
  cordLength: 3.0,
  result: 'Solvable with spatial awareness'
};
```

### Mistake 2: No Feedback

```javascript
// BAD: Silent connection
const badFeedback = {
  connected: { visual: null, audio: null }
};

// GOOD: Clear confirmation
const goodFeedback = {
  connected: { visual: 'light_on', audio: 'power_on' }
};
```

---

## Related Systems

- **Cord Physics** - Shared with phone systems
- **State Machines** - For managing interaction states
- **Physics System** - For cord simulation
- **Interactive Objects** - For player interactions

---

## References

- **Shadow Engine Documentation**: `docs/`
- **Cord Physics**: See *Physics System*
- **State Machines**: See *Game Architecture*

---

**RALPH_STATUS:**
- **Status**: Amplifier Cord Scene documentation complete
- **Files Created**: `docs/generated/14-scene-case-studies/amplifier-cord-scene.md`
- **Related Documentation**: All Phase 14 scene case studies
- **Next**: Drawing Minigame documentation
