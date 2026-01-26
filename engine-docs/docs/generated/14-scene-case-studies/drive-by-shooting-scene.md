# Scene Case Study: Drive-By Shooting

## 🎬 Scene Overview

**Location**: Exterior near phone booth, usually triggered after player answers phone
**Narrative Context**: Violent interruption—a car speeds past, shots fired, escalating tension and signaling danger
**Player Experience: Curiosity (answering phone) → Connection (dialog delivery) → SUDDEN VIOLENCE (shock) → Disorientation → Realization stakes are raised

The Drive-By Shooting scene is a pivotal action moment that shatters the relative calm of exploration. Unlike the gradual horror of other scenes, this is sudden, violent, and chaotic—a burst of action that demonstrates the engine's ability to choreograph complex sequences combining audio, visual effects, animation, and timing. It serves as a narrative turning point, raising stakes and signaling that the player is not alone in this world.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create shock through disruption—player feels safe during dialog, then sudden violence shatters that safety.

**Why This Moment Matters**:

```

THE SHOCK OF DISRUPTION:

Player State Before:
├── Standing still, listening to phone
├── Focused on dialog content
├── Feeling relatively safe
├── Processing narrative information
└─→ COMFORTABLE, ENGAGED

Event Happens:
├── Sudden loud noise (car horn)
├── Motion blur (fast movement)
├── Gunshots (staccato violence)
├── Phone drops (connection broken)
├── Car speeds away
└─→ SUDDEN CHAOS, DISORIENTED

Player State After:
├── Adrenaline spike
├── No longer feels safe
├── Stakes are raised
├── "I'm not alone here"
└─→ ALERT, VULNERABLE

NARRATIVE FUNCTION:
This isn't just action—it's a statement.
The world is dangerous. Someone fired
those shots. And they were aiming
near where YOU are standing.
```

### Design Philosophy

**1. Choreographed Chaos**

```

ACTION SEQUENCE STRUCTURE:

BUILD-UP (What player doesn't notice):
├── Distant car engine (2 seconds before)
├── Approaching rapidly
├── Player focused on dialog, misses it
└─→ Subtle, unnoticed

TRIGGER (The moment it starts):
├── CAR HORN (impossible to ignore)
├── Audio overwhelms
├── Player reflexively turns
└─→ IMMEDIATE, SHARP

ACTION (1-2 seconds of chaos):
├── Car blurs past (motion blur effect)
├── Multiple shots (3-5 gunshots)
├── Muzzle flashes visible
├── Impacts hit environment
├── Phone drops from player's "hand"
└─→ FAST, CONFUSING, OVERWHELMING

AFTERMATH (3-5 seconds):
├── Car recedes (engine fades)
├── Tinnitus ringing (ear damage simulation)
├── Dust settles
├── Player processes what happened
└─→ ECHOES, DISORIENTED

RESOLUTION (Return to exploration):
├── Normal ambience slowly returns
├── Player can investigate impacts
├── Phone on ground (can't re-answer)
├── World has changed
└─→ NEW NORMAL, HEIGHTENED AWARENESS
```

**2. Audio-Visual Coordination**

```

TIMELINE OF EVENTS:

T-2.0s: [Car engine] distant, getting louder
T-1.0s: [Tires on road] screech approaching
T-0.0s: [CAR HORN] BLAAAAST!

T+0.1s: [Motion blur] starts
T+0.2s: Car enters frame (blur)
T+0.3s: [GUNSHOT 1] loud crack!
T+0.4s: [Impact 1] wall/sparks
T+0.5s: [GUNSHOT 2] crack!
T+0.6s: Phone drops (animates)
T+0.7s: [GUNSHOT 3] crack!
T+0.8s: [Impact 2] more sparks
T+0.9s: [GUNSHOT 4] crack!
T+1.0s: Car exits frame (blur)
T+1.5s: Engine fading
T+2.0s: [Tinnitus] ringing starts
T+5.0s: Normal ambience returning

TOTAL: 7 seconds of chaos
```

---

## 🎨 Level Design Breakdown

### Spatial Setup

```

                    DRIVE-BY SCENE LAYOUT:

    [PLAYER] [PHONE BOOTH]
      ↓          ↓
    ╔══════════════════════════════════════════╗
    ║           STREET (Alley/Exterior)          ║
    ║                                                ║
    ║  [CAR APPROACH PATH] → → → →                ║
    ║                         ║  ║              ║
    ║  [PLAYER] ☎️           ║  ║              ║
    ║     ↓                  ║ ↓              ║
    ║  [IMPACT POINTS] on wall behind player      ║
    ║     ✦               ✦    ✦               ║
    ║    (sparks from bullet hits)                ║
    ║                                                ║
    ╚════════════════════════════════════════════╝

KEY ELEMENTS:

Player Position:
├── Near phone booth (answering call)
├── Facing phone booth initially
├── Will turn toward car on horn
└── Fixed during sequence (no control)

Car Path:
├── Approaches from one side
├── Passes close to player (for threat)
├── Exits opposite side quickly
└── Camera follows briefly

Impact Points:
├── Visible sparks where bullets hit
├── Near player (threatening but not hitting)
├── On environment (destructible marks)
└─→ Shows danger, tells story
```

### Camera Choreography

```

CAMERA BEHAVIOR:

BEFORE (Cinematic/Player Control):
├── Player in first-person view
├── Looking at phone booth
├── Normal control
└─→ GROUNDED, EXPLORING

DURING (Forced cinematic):
├── Horn triggers camera swivel
├── Tracks car as it passes
├── Motion blur during fast movement
├── Slight screen shake on gunshots
└─→ REACTIVE, FOLLOWING ACTION

AFTER (Return to player):
├── Smooth interpolation back to player control
├── Player can look around
├── Investigate bullet impacts
└─→ REGAINING ORIENTATION

CAMERA PATH:

Frame T-1.0 to T-0: Player POV (normal)
Frame T+0.0 to T+0.1: Quick swivel toward sound
Frame T+0.1 to T+1.0: Track car, motion blur
Frame T+1.0 to T+3.0: Watch car recede
Frame T+3.0 onward: Return to player control

TRANSITIONS:
All use smooth easing (no cuts)
Maintains spatial continuity
Player never feels "teleported"
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the drive-by implementation, you should know:
- **Animation Sequencing**: Coordinating multiple simultaneous animations
- **Camera Control**: Temporary cinematic control vs player control
- **Motion Blur**: Post-processing effect for fast movement
- **Sound Layering**: Multiple simultaneous audio with priority
- **Object State Changes**: Phone transitioning from held to dropped

### Scene Data Structure

```javascript
// AnimationData.js - Drive-by sequence configuration
export const ANIMATIONS = {
  drive_by_shooting: {
    id: 'drive_by_shooting',
    name: 'Drive-By Shooting Event',
    type: 'action_sequence',
    trigger: 'after_phone_answered',

    // Sequence timeline
    duration: 7.0,

    // Stages
    stages: [
      {
        name: 'buildup',
        startTime: -2.0,
        duration: 2.0,
        elements: [
          {
            type: 'audio',
            sound: 'car_approaching',
            volume: 0,
            fadeIn: 2.0,
            spatial: true,
            position: { x: -30, y: 0, z: 0 }
          }
        ]
      },
      {
        name: 'trigger',
        startTime: 0,
        duration: 0.5,
        elements: [
          {
            type: 'audio',
            sound: 'car_horn',
            volume: 1.0,
            spatial: false  // Full volume
          },
          {
            type: 'camera',
            action: 'swivel_toward_sound',
            duration: 0.3,
            easing: 'easeOutQuad'
          }
        ]
      },
      {
        name: 'pass_by',
        startTime: 0.5,
        duration: 1.5,
        elements: [
          {
            type: 'spawn',
            object: 'car',
            model: '/assets/models/sedan_car.glb',
            startPosition: { x: -20, y: 0, z: 2 },
            speed: 25,  // m/s
            path: 'straight_line'
          },
          {
            type: 'vfx',
            effect: 'motion_blur',
            intensity: 0.8,
            duration: 1.5
          },
          {
            type: 'gunfire',
            shots: 4,
            timing: [0.3, 0.5, 0.7, 0.9],  // Relative to stage start
            source: 'car_window'
          },
          {
            type: 'impacts',
            count: 3,
            positions: [
              { x: 0.5, y: 1.2, z: -1 },
              { x: -0.3, y: 1.5, z: -0.8 },
              { x: 0.8, y: 0.8, z: -1.5 }
            ],
            vfx: 'sparks'
          },
          {
            type: 'phone_drop',
            trigger: 'second_shot',
            animation: 'drop_with_physics'
          }
        ]
      },
      {
        name: 'aftermath',
        startTime: 2.0,
        duration: 5.0,
        elements: [
          {
            type: 'audio',
            sound: 'tinnitus_ring',
            volume: 0.4,
            fadeIn: 1.0,
            loop: true,
            duration: 4.0
          },
          {
            type: 'restore_control',
            delay: 2.0,
            duration: 1.0
          }
        ]
      }
    ]
  }
};
```

### Drive-By Sequence Manager

```javascript
// DriveBySequenceManager.js - Action sequence choreography
class DriveBySequenceManager {
  constructor(animationManager, audioManager, sceneManager, vfxManager) {
    this.animation = animationManager;
    this.audio = audioManager;
    this.scene = sceneManager;
    this.vfx = vfxManager;

    this.isPlaying = false;
    this.currentStage = null;
    this.stageTimeouts = [];
  }

  async trigger() {
    if (this.isPlaying) return;
    this.isPlaying = true;

    const sequence = ANIMATIONS.drive_by_shooting;

    // Disable player control temporarily
    game.getManager('player').disableControl(8);  // Disable for 8 seconds

    // Start buildup stage
    await this.playBuildup(sequence);

    // Wait for trigger moment
    await this.delay(2000);

    // Play main sequence
    await this.playPassBy(sequence);

    // Play aftermath
    await this.playAftermath(sequence);

    // Complete
    this.onComplete();
  }

  async playBuildup(sequence) {
    const stage = sequence.stages.find(s => s.name === 'buildup');

    // Start approaching car audio
    for (const element of stage.elements) {
      if (element.type === 'audio') {
        this.audio.playPositional(element.sound, element.position, {
          volume: element.volume,
          fadeIn: element.fadeIn,
          loop: true
        });
      }
    }
  }

  async playPassBy(sequence) {
    const stage = sequence.stages.find(s => s.name === 'pass_by');
    const trigger = sequence.stages.find(s => s.name === 'trigger');

    // Play trigger elements first
    await this.playStageElements(trigger.elements);

    // Wait for horn
    await this.delay(500);

    // Spawn car and choreograph pass-by
    await this.executePassBy(stage);
  }

  async playStageElements(elements) {
    for (const element of elements) {
      await this.executeElement(element);
    }
  }

  async executePassBy(stage) {
    const carElement = stage.elements.find(e => e.type === 'spawn');
    const blurElement = stage.elements.find(e => e.type === 'vfx');
    const gunfireElement = stage.elements.find(e => e.type === 'gunfire');

    // Spawn car
    const car = await this.spawnCar(carElement);

    // Enable motion blur
    if (blurElement) {
      this.vfx.enableEffect('motion_blur', {
        intensity: blurElement.intensity,
        duration: blurElement.duration
      });
    }

    // Animate car movement
    await this.animateCarPassBy(car, carElement);

    // Schedule gunshots
    if (gunfireElement) {
      for (const shotTime of gunfireElement.timing) {
        this.scheduleAction(shotTime * 1000, () => {
          this.fireShot(car);
        });
      }
    }

    // Wait for car to pass
    await this.delay(1500);

    // Despawn car (it's driven off)
    this.despawnCar(car);
  }

  async spawnCar(config) {
    // Load car model
    const car = await this.scene.loadModel(config.model);
    car.position.set(
      config.startPosition.x,
      config.startPosition.y,
      config.startPosition.z
    );

    // Set up car for animation
    car.userData.isDriveByCar = true;
    car.userData.speed = config.speed;

    this.scene.add(car);
    return car;
  }

  async animateCarPassBy(car, config) {
    const startTime = Date.now();
    const duration = 1500;  // ms to cross scene

    // Animate car moving across
    while (Date.now() - startTime < duration) {
      const progress = (Date.now() - startTime) / duration;
      const distance = progress * 50;  // Car travels 50 meters

      car.position.x = config.startPosition.x + distance;
      car.position.y = config.startPosition.y + Math.sin(progress * Math.PI) * 0.3;

      // Camera follows car briefly
      if (progress < 0.7) {
        const camera = this.scene.getCamera();
        const lookTarget = car.position.clone();
        lookTarget.y += 1;
        camera.lookAt(lookTarget);
      }

      await this.frameDelay();
    }
  }

  fireShot(car) {
    // Play gunshot audio
    this.audio.playOneShot('gunshot', {
      volume: 0.8,
      spatial: true,
      position: car.position
    });

    // Muzzle flash
    this.vfx.trigger('muzzle_flash', {
      position: car.position.clone().add(new THREE.Vector3(1, 1.2, 0)),
      duration: 0.1
    });

    // Create impact after delay (bullet travel time)
    setTimeout(() => {
      this.createBulletImpact();
    }, 100);

    // Screen shake
    this.vfx.trigger('screen_shake', {
      intensity: 0.3,
      duration: 0.2
    });

    // Drop phone on second shot (approximate check)
    if (!this.phoneDropped && Math.random() > 0.5) {
      this.dropPhone();
      this.phoneDropped = true;
    }
  }

  createBulletImpact() {
    // Random impact position near player
    const impactPos = new THREE.Vector3(
      (Math.random() - 0.5) * 2,
      1 + Math.random() * 0.5,
      -1 - Math.random()
    );

    // Spark VFX
    this.vfx.trigger('sparks', {
      position: impactPos,
      count: 10,
      spread: 0.3,
      duration: 0.5
    });

    // Impact mark on environment
    this.scene.addBulletHole(impactPos);
  }

  dropPhone() {
    const phone = this.scene.getObjectByName('phone_receiver');
    if (!phone) return;

    // Animate phone dropping
    this.animation.play('phone_drop', {
      target: phone,
      to: { y: 0 },
      duration: 0.5,
      easing: 'gravity',
      onComplete: () => {
        // Enable physics for phone on ground
        phone.userData.physicsEnabled = true;
      }
    });

    // Dialog stops
    game.getManager('dialog').stop();
  }

  async playAftermath(sequence) {
    const stage = sequence.stages.find(s => s.name === 'aftermath');

    // Tinnitus ringing
    const tinnitusElement = stage.elements.find(e => e.sound === 'tinnitus_ring');
    if (tinnitusElement) {
      this.audio.play(tinnitusElement.sound, {
        volume: tinnitusElement.volume,
        fadeIn: tinnitusElement.fadeIn,
        loop: true
      });
    }

    // Wait before restoring control
    await this.delay(2000);

    // Fade out tinnitus
    this.audio.fadeOut(tinnitusElement.sound, 2);

    // Restore player control smoothly
    const restoreElement = stage.elements.find(e => e.type === 'restore_control');
    if (restoreElement) {
      await this.delay(restoreElement.delay);
      game.getManager('player').enableControl(restoreElement.duration);
    }
  }

  scheduleAction(delayMs, callback) {
    const timeout = setTimeout(callback, delayMs);
    this.stageTimeouts.push(timeout);
  }

  despawnCar(car) {
    // Fade out car visually if still visible
    // Or just remove (it's off screen)
    this.scene.remove(car);
  }

  onComplete() {
    this.isPlaying = false;

    // Clean up timeouts
    for (const timeout of this.stageTimeouts) {
      clearTimeout(timeout);
    }
    this.stageTimeouts = [];

    // Emit event for other systems
    game.emit('drive_by_complete', {
      time: Date.now(),
      playerShaken: true
    });
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  frameDelay() {
    return new Promise(resolve => requestAnimationFrame(resolve));
  }

  async executeElement(element) {
    switch (element.type) {
      case 'audio':
        this.audio.play(element.sound, {
          volume: element.volume,
          spatial: element.spatial || false
        });
        break;

      case 'camera':
        await this.executeCameraAction(element);
        break;
    }
  }

  async executeCameraAction(config) {
    const camera = this.scene.getCamera();
    const player = game.getManager('player');

    // Swivel camera toward sound
    const targetRotation = new THREE.Quaternion();
    targetRotation.setFromEuler(new THREE.Euler(0, Math.PI / 4, 0));

    const startQuat = camera.quaternion.clone();

    // Animate rotation
    const startTime = Date.now();
    while (Date.now() - startTime < config.duration * 1000) {
      const progress = (Date.now() - startTime) / (config.duration * 1000);
      const eased = this.easeOutQuad(progress);

      camera.quaternion.slerpQuaternions(startQuat, targetRotation, eased);

      await this.frameDelay();
    }
  }

  easeOutQuad(t) {
    return t * (2 - t);
  }
}
```

### Impact Effects System

```javascript
// ImpactEffects.js - Bullet impact visual effects
class ImpactEffects {
  constructor(sceneManager, vfxManager) {
    this.scene = sceneManager;
    this.vfx = vfxManager;

    this.impacts = [];
    this.maxImpacts = 10;
  }

  createImpact(position) {
    // Spark particles
    this.createSparks(position);

    // Bullet hole decal
    this.createBulletHole(position);

    // Sound effect
    this.vfx.audio.playOneShot('bullet_impact', {
      position: position,
      volume: 0.3
    });
  }

  createSparks(position) {
    const sparkCount = 8 + Math.floor(Math.random() * 5);

    for (let i = 0; i < sparkCount; i++) {
      const spark = {
        position: position.clone(),
        velocity: new THREE.Vector3(
          (Math.random() - 0.5) * 5,
          Math.random() * 3,
          (Math.random() - 0.5) * 5
        ),
        life: 1.0,
        decay: 0.02 + Math.random() * 0.03
      };

      this.impacts.push(spark);
    }
  }

  createBulletHole(position) {
    // Create a small dark circle on affected surface
    const holeGeometry = new THREE.CircleGeometry(0.02, 8);
    const holeMaterial = new THREE.MeshBasicMaterial({
      color: 0x111111,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide
    });

    const hole = new THREE.Mesh(holeGeometry, holeMaterial);
    hole.position.copy(position);

    // Orient to face the impact direction
    hole.lookAt(position.x + 1, position.y, position.z);

    this.scene.add(hole);

    // Add to cleanup list
    this.scene.registerTemporaryObject(hole, {
      lifetime: 300  // Remove after 5 minutes
    });
  }

  update(deltaTime) {
    // Update particle effects
    for (let i = this.impacts.length - 1; i >= 0; i--) {
      const spark = this.impacts[i];

      // Update position
      spark.position.add(
        spark.velocity.clone().multiplyScalar(deltaTime)
      );

      // Update life
      spark.life -= spark.decay;

      // Apply gravity to velocity
      spark.velocity.y -= 9.8 * deltaTime;

      // Remove dead particles
      if (spark.life <= 0) {
        this.impacts.splice(i, 1);
      }
    }
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Define the Action Purpose

```
ACTION SCENE BRIEF:

1. What's the narrative purpose?
    Raise stakes, show danger, escalate tension

2. What should player feel?
    Shock → Fear → Heightened awareness

3. How long should it last?
    Short (5-10 seconds) - shock is brief

4. What are the consequences?
    Phone drops (dialog ends), world feels unsafe

5. What changes after?
    Player is more cautious, atmosphere changed
```

### Step 2: Choreograph the Timeline

```javascript
// Action sequence timeline:

const timeline = [
  { time: -2, event: 'buildup_audio_start' },
  { time: 0, event: 'horn_blast' },
  { time: 0.1, event: 'camera_swivel' },
  { time: 0.3, event: 'car_visible', shot: 1 },
  { time: 0.5, event: 'phone_drops', shot: 2 },
  { time: 0.7, event: 'shot: 3' },
  { time: 0.9, event: 'shot: 4', car_exits },
  { time: 2, event: 'tinnitus_starts' },
  { time: 5, event: 'normal_audio_returns' }
];
```

---

## 🔧 Variations For Your Game

### Variation 1: Non-Violent Disruption

```javascript
const nonViolentDriveBy = {
  // Instead of shooting, just menacing passage
  action: 'drive_by_threat',

  // Car slows, occupants look at player
  behavior: 'menacing_stare',

  // Still unsettling, but not violent
  emotional: 'threatening_not_deadly'
};
```

### Variation 2: Chase Sequence

```javascript
const chaseSequence = {
  // Car doesn't just pass by, but pursues
  action: 'car_chase',

  // Player must flee or hide
  playerCanMove: true,

  // More extended sequence
  duration: 'extended'
};
```

### Variation 3: Missed Shots

```javascript
const missedShots = {
  // Gunshots fired but not aimed at player
  targeting: 'environment_only',

  // Creates tension without direct threat
  effect: 'nearby_not_direct',

  // Different emotional beat
  narrative: 'someone else is target'
};
```

---

## Performance Considerations

```
DRIVE-BY PERFORMANCE:

Car Model:
├── Single model, briefly visible
├── Can be lower LOD
├── Removed after sequence
└─→ Minimal impact

Particles (sparks):
├── Short-lived (1-2 seconds)
├── Low count (<20)
├── Simple geometry
└─→ Negligible

Motion Blur:
├── Post-processing effect
├── Can be expensive on mobile
├── Quality setting option
└─→ Consider optional on low-end

Audio:
├── Layered sounds (horn + shots + tires)
├── All short duration
├── Tinnitus can be simpler on mobile
└─→ Acceptable overall
```

---

## Common Mistakes Beginners Make

### 1. Too Long

```javascript
// ❌ WRONG: 30+ second action scene
// Becomes boring, loses shock value

// ✅ CORRECT: 5-10 seconds
// Brief, shocking, memorable
```

### 2. No Build-Up

```javascript
// ❌ WRONG: Shots come from nowhere
// Confusing, feels unfair

// ✅ CORRECT: Car approaches first
// Audio cue before visual (horn)
// Player understands what's happening
```

### 3: Player Can't React

```javascript
// ❌ WRONG: Player frozen too long
// Frustrating, breaks agency

// ✅ CORRECT: Quick cinematic, return control
// Player can investigate aftermath
```

---

## Related Systems

- [AnimationManager](../06-animation/animation-manager.md) - Animation sequences
- [SFXManager](../05-media-systems/sfx-manager.md) - Gunshot and impact sounds
- [VFXManager](../07-visual-effects/vfx-manager.md) - Spark effects
- [Phone Booth Scene](../08-interactive-objects/phone-booth-scene.md) - Trigger context

---

## Source File Reference

**Animation Data**:
- `content/AnimationData.js` - Drive-by sequence configuration

**Managers**:
- `managers/DriveBySequenceManager.js` - Action choreography
- `managers/ImpactEffects.js` - Bullet impact effects

**Assets**:
- `assets/models/sedan_car.glb` - Vehicle model
- `assets/audio/gunshot.wav` - Gun sounds
- `assets/audio/car_horn.wav` - Horn sound

---

## References

- [Action Sequence Design](https://www.youtube.com/watch?v=nK0FvK_8f6Q) - Video essay
- [Choreographing Action](https://www.youtube.com/watch?v=8FpigqfcqlM) - Tutorial
- [Game Audio for Gunshots](https://www.youtube.com/watch?v=M4skP6bN_Ks) - Sound design

*Documentation last updated: January 12, 2026*
