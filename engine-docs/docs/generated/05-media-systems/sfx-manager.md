# SFXManager - First Principles Guide

## Overview

The **SFXManager** (Sound Effects Manager) handles short audio clips with **3D spatial positioning** in the Shadow Engine. Unlike music or dialog which provide background atmosphere, sound effects are tied to specific game events and objects - footsteps, door creaks, impacts, and environmental sounds.

Think of SFXManager as the **foley artist for your game world** - just as movie foley artists create synchronized sounds for actions on screen, SFXManager plays the right sound at the right time from the right position in 3D space.

## What You Need to Know First

Before understanding SFXManager, you should know:
- **Howler.js library** - Web audio library (v2.2.4+) for audio playback
- **3D coordinate systems** - X, Y, Z positions in world space
- **Spatial audio** - Sound that changes based on listener position
- **Audio attenuation** - How sound fades over distance
- **Event triggers** - What causes a sound to play

### Quick Refresher: Spatial Audio

```
SPATIAL AUDIO: Sound changes based on player position

          [Sound Source]
              ●
              │
              │  ←─ 5 meters
              │
         ┌────┴────┐
         │  👤     │  ← Player (Listener)
         │         │
         └─────────┘

Player at 5m hears:  ██████████░░░░  (loud volume)
Player at 10m hears: ██████░░░░░░░░░  (medium volume)
Player at 20m hears: ██░░░░░░░░░░░░░  (quiet)
Player at 30m hears: ░░░░░░░░░░░░░░░  (silent)

PANNING: Stereo positioning based on angle
Player left of source →  Left: ████  Right: ░░░
Player right of source → Left: ░░░  Right: ████
Player centered        → Left: ██  Right: ██
```

**Why spatial audio matters:** It tells players WHERE things are happening - footsteps around a corner, an explosion behind them, a dialog from a character to their left.

---

## Part 1: Why Use a Dedicated SFX Manager?

### The Problem: Basic Audio Can't Do 3D

Without a dedicated sound effects system:

```javascript
// ❌ WITHOUT SFXManager - No spatial positioning
function playFootstep() {
  const audio = new Audio("sfx/footstep.mp3");
  audio.volume = 1.0;
  audio.play();

  // Player can't tell WHERE the sound came from!
  // All sounds come from "center" of speakers
}
```

**Problems:**
- No 3D positioning
- No distance attenuation
- Can't tell direction of sounds
- Manual cleanup required
- No sound pooling (performance waste)

### The Solution: SFXManager

```javascript
// ✅ WITH SFXManager - Full 3D spatial audio
sfxManager.playSpatial({
  file: "sfx/footstep.mp3",
  position: { x: 5, y: 0, z: -3 },  // Where the sound is
  volume: 1.0,
  maxDistance: 20,  // How far it can be heard
  referenceDistance: 3  // Distance at which attenuation begins
});

// Sound appears to come from that location!
// Volume and panning adjust based on player position.
```

**Benefits:**
- True 3D spatial positioning
- Distance-based attenuation
- Stereo panning based on angle
- Sound pooling for performance
- Automatic cleanup

---

## Part 2: Sound Effect Data Structure

### Basic SFX Definition

```javascript
// In sfxData.js
export const soundEffects = {
  // Footstep sounds
  footstep: {
    file: "sfx/footsteps/wood-1.mp3",
    volume: 0.5,
    spatial: true,
    maxDistance: 15,
    referenceDistance: 2
  },

  // Door sound
  doorCreak: {
    file: "sfx/doors/creaky-open.mp3",
    volume: 0.8,
    spatial: true,
    position: { x: 5, y: 1, z: -10 },
    maxDistance: 20,
    referenceDistance: 3
  },

  // UI click (not spatial)
  uiClick: {
    file: "sfx/ui/click.mp3",
    volume: 0.3,
    spatial: false  // 2D UI sound
  }
};
```

### Sound Effect Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | string | Yes | Path to audio file |
| `volume` | number | No | Base volume (0.0-1.0) |
| `spatial` | boolean | No | Whether to use 3D positioning |
| `position` | object | Conditional | { x, y, z } world position (if spatial) |
| `maxDistance` | number | No | Maximum hearing distance (meters) |
| `referenceDistance` | number | No | Distance for full volume |
| `rolloffFactor` | number | No | How fast sound fades (0-1) |
| `loop` | boolean | No | Whether to loop (default: false) |
| `randomize` | boolean | No | Pick random from file pattern |
| `pitch` | number | No | Playback rate (0.5-2.0) |

---

## Part 3: How SFXManager Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SFXManager                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Sound Pool                                │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐        │   │
│  │  │ SFX1 │  │ SFX2 │  │ SFX3 │  │ SFX4 │  │ SFX5 │  ...   │   │
│  │  │      │  │      │  │      │  │      │  │      │        │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘        │   │
│  │  Reuse sounds instead of creating/destroying                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Spatial Audio Engine                            │   │
│  │  - Calculates distance from listener                        │   │
│  │  - Applies attenuation curve                                │   │
│  │  - Calculates stereo panning                                │   │
│  │  - Updates each frame                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Howler.js Integration                         │   │
│  │  - Creates Howl instances with spatial: true                │   │
│  │  - Uses Web Audio PannerNode                                │   │
│  │  - Updates position/orientation each frame                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                            │
         │ receives from              │ plays to
         ▼                            ▼
┌─────────────────┐          ┌─────────────────┐
│  Event Triggers │          │   Web Audio     │
│  - Collider     │          │   (Speakers)    │
│  - Interaction  │          │                 │
│  - Animation    │          │                 │
└─────────────────┘          └─────────────────┘
```

### Sound Playback Flow

```
1. Game event occurs (collision, interaction, etc.)
              │
              ▼
2. Trigger action: { action: "playSound", sound: "doorCreak" }
              │
              ▼
3. SFXManager receives request
              │
              ▼
4. Look up sound in sfxData
              │
              ▼
5. Get pooled Howl instance or create new one
              │
              ▼
6. For spatial sounds:
    │
    ├──▶ Calculate distance to player
    │
    ├──▶ Apply attenuation (volume drops with distance)
    │
    ├──▶ Calculate stereo panning (left/right balance)
    │
    └──▶ Update PannerNode position
              │
              ▼
7. Play sound
              │
              ▼
8. Return instance to pool when finished
```

---

## Part 4: Spatial Audio Implementation

### Distance Attenuation

```javascript
class SFXManager {
  calculateVolume(distance, maxDistance, referenceDistance, rolloffFactor) {
    // Inverse distance attenuation model
    // Based on OpenAL attenuation formula

    if (distance <= referenceDistance) {
      return 1.0;  // Full volume when close
    }

    if (distance >= maxDistance) {
      return 0.0;  // Silent when too far
    }

    // Calculate attenuation
    const distanceBeyondReference = distance - referenceDistance;
    const effectiveDistance = referenceDistance +
      (distanceBeyondReference * rolloffFactor);

    const attenuation = referenceDistance / effectiveDistance;
    return Math.max(0, Math.min(1, attenuation));
  }
}
```

### Howler.js Spatial Audio Setup

```javascript
class SFXManager {
  playSpatial(soundData, worldPosition) {
    // Create Howl with spatial enabled
    const howl = new Howl({
      src: [soundData.file],
      volume: soundData.volume || 1.0,
      spatial: true,  // Enable 3D spatial audio
      orientation: [0, 0, -1],  // Forward direction
      pannerAttr: {
        panningModel: 'HRTF',  // Head-related transfer function (best for 3D)
        distanceModel: 'inverse',
        refDistance: soundData.referenceDistance || 1,
        maxDistance: soundData.maxDistance || 10000,
        rolloffFactor: soundData.rolloffFactor || 1
      }
    });

    // Set sound position in 3D space
    howl.pos(
      worldPosition.x,
      worldPosition.y,
      worldPosition.z
    );

    // Set listener position (usually camera/player)
    this.updateListener();

    // Play the sound
    howl.play();

    return howl;
  }

  updateListener() {
    const player = this.getPlayerPosition();  // Get from game state
    const orientation = this.getPlayerOrientation();

    // Update listener position and orientation
    Howler.pos(player.x, player.y, player.z);
    Howler.orientation(
      orientation.forward.x,
      orientation.forward.y,
      orientation.forward.z,
      orientation.up.x,
      orientation.up.y,
      orientation.up.z
    );
  }
}
```

### Per-Frame Updates

```javascript
class SFXManager {
  update(deltaTime) {
    // Update listener position each frame
    this.updateListener();

    // Update any moving sound sources
    this.activeSpatialSounds.forEach(sound => {
      if (sound.moving) {
        const newPosition = this.calculateNewPosition(sound);
        sound.howl.pos(newPosition.x, newPosition.y, newPosition.z);
      }
    });
  }
}
```

---

## Part 5: Sound Pooling for Performance

### Why Pool Sounds?

```
WITHOUT POOLING:
Footstep event → Create Howl → Play → Destroy
Footstep event → Create Howl → Play → Destroy
Footstep event → Create Howl → Play → Destroy
... 100 times/second = 100 allocations!

WITH POOLING:
Pre-create 10 Howl instances:
Footstep event → Get from pool → Play → Return to pool
Footstep event → Get from pool → Play → Return to pool
... No allocations after initial pool creation!
```

### Pool Implementation

```javascript
class SFXManager {
  constructor() {
    this.pools = new Map();  // Map<soundName, Howl[]>
    this.maxPoolSize = 10;   // Max instances per sound
  }

  play(soundName) {
    let pool = this.pools.get(soundName);

    if (!pool) {
      pool = [];
      this.pools.set(soundName, pool);
    }

    // Try to get a non-playing instance from pool
    let howl = pool.find(h => !h.playing());

    // If none available and pool not full, create new
    if (!howl && pool.length < this.maxPoolSize) {
      const soundData = this.sfxData[soundName];
      howl = this.createHowl(soundData);
      pool.push(howl);
    }

    // If we have an instance, play it
    if (howl) {
      // Reset position if spatial
      if (soundData.spatial) {
        howl.stop();
      }
      howl.play();
      return howl;
    }

    console.warn(`Sound pool exhausted for: ${soundName}`);
    return null;
  }

  createHowl(soundData) {
    return new Howl({
      src: [soundData.file],
      volume: soundData.volume || 1.0,
      spatial: soundData.spatial || false,
      pool: this.maxPoolSize
    });
  }
}
```

---

## Part 6: Common SFX Use Cases

### 1. Footsteps

```javascript
footsteps: {
  wood: {
    file: "sfx/footsteps/wood.mp3",
    volume: 0.4,
    spatial: true,
    maxDistance: 15,
    referenceDistance: 2,
    randomize: true,  // Pick from wood-1.mp3, wood-2.mp3, etc.
    criteria: {
      isMoving: true,
      groundType: "wood"
    }
  },

  grass: {
    file: "sfx/footsteps/grass.mp3",
    volume: 0.3,
    spatial: true,
    maxDistance: 12,
    referenceDistance: 2,
    randomize: true,
    criteria: {
      isMoving: true,
      groundType: "grass"
    }
  },

  metal: {
    file: "sfx/footsteps/metal.mp3",
    volume: 0.5,
    spatial: true,
    maxDistance: 20,
    referenceDistance: 3,
    randomize: true,
    criteria: {
      isMoving: true,
      groundType: "metal"
    }
  }
}
```

### 2. Door Sounds

```javascript
doors: {
  openCreaky: {
    file: "sfx/doors/creaky-open.mp3",
    volume: 0.7,
    spatial: true,
    maxDistance: 25,
    referenceDistance: 2,
    criteria: { interaction: "door_open", doorType: "creaky" }
  },

  closeHeavy: {
    file: "sfx/doors/heavy-close.mp3",
    volume: 0.9,
    spatial: true,
    maxDistance: 30,
    referenceDistance: 3,
    criteria: { interaction: "door_close", doorType: "heavy" }
  },

  lockUnlock: {
    file: "sfx/doors/lock-unlock.mp3",
    volume: 0.6,
    spatial: true,
    maxDistance: 10,
    referenceDistance: 1,
    criteria: { interaction: "lock_unlock" }
  }
}
```

### 3. Impact/Hit Sounds

```javascript
impacts: {
  punch: {
    file: "sfx/combat/punch.mp3",
    volume: 0.8,
    spatial: true,
    maxDistance: 20,
    referenceDistance: 3,
    criteria: { hitType: "melee" }
  },

  bulletHit: {
    file: "sfx/combat/bullet-hit.mp3",
    volume: 0.9,
    spatial: true,
    maxDistance: 50,
    referenceDistance: 5,
    criteria: { hitType: "ranged" }
  },

  bodyFall: {
    file: "sfx/combat/body-fall.mp3",
    volume: 1.0,
    spatial: true,
    maxDistance: 30,
    referenceDistance: 2,
    criteria: { event: "knockdown" }
  }
}
```

### 4. Ambient Looping Sounds

```javascript
ambient: {
  fireCrackling: {
    file: "sfx/ambient/fire-loop.mp3",
    volume: 0.5,
    spatial: true,
    position: { x: 5, y: 0, z: -8 },
    maxDistance: 15,
    referenceDistance: 2,
    loop: true,
    criteria: { inRoom: "fireplace_room" }
  },

  drippingFaucet: {
    file: "sfx/ambient/drip-loop.mp3",
    volume: 0.3,
    spatial: true,
    position: { x: -3, y: 1, z: 5 },
    maxDistance: 8,
    referenceDistance: 1,
    loop: true,
    criteria: { inRoom: "bathroom" }
  },

  machineryHum: {
    file: "sfx/ambient/machinery-loop.mp3",
    volume: 0.4,
    spatial: true,
    position: { x: 0, y: 2, z: -15 },
    maxDistance: 40,
    referenceDistance: 5,
    loop: true,
    criteria: { currentZone: "factory" }
  }
}
```

### 5. UI Sounds (2D, Not Spatial)

```javascript
ui: {
  click: {
    file: "sfx/ui/click.mp3",
    volume: 0.3,
    spatial: false  // UI sounds are 2D
  },

  hover: {
    file: "sfx/ui/hover.mp3",
    volume: 0.2,
    spatial: false
  },

  confirm: {
    file: "sfx/ui/confirm.mp3",
    volume: 0.4,
    spatial: false
  },

  cancel: {
    file: "sfx/ui/cancel.mp3",
    volume: 0.4,
    spatial: false
  },

  popup: {
    file: "sfx/ui/popup.mp3",
    volume: 0.5,
    spatial: false
  }
}
```

### 6. One-Shot Events

```javascript
events: {
  glassBreak: {
    file: "sfx/events/glass-break.mp3",
    volume: 0.9,
    spatial: true,
    maxDistance: 25,
    referenceDistance: 3,
    criteria: { event: "glass_shatter" }
  },

  explosion: {
    file: "sfx/events/explosion.mp3",
    volume: 1.0,
    spatial: true,
    maxDistance: 100,
    referenceDistance: 10,
    criteria: { event: "explosion" }
  },

  thunder: {
    file: "sfx/events/thunder.mp3",
    volume: 0.9,
    spatial: false,  // Environmental, not positioned
    criteria: { event: "thunder_strike" }
  }
}
```

---

## Part 7: Triggering Sound Effects

### From Collider Zones

```javascript
export const triggerZones = {
  creakyFloor: {
    name: "creaky_floor",
    size: { x: 2, y: 2, z: 2 },
    position: { x: 5, y: 0, z: -10 },
    shape: "box",
    onEnter: {
      action: "playSound",
      sound: "floor-creak.mp3",
      volume: 0.6,
      spatial: true,
      maxDistance: 15
    }
  },

  spookyWhisper: {
    name: "spooky_whisper",
    size: { x: 3, y: 3, z: 3 },
    position: { x: -8, y: 0, z: -5 },
    shape: "box",
    onEnter: {
      action: "playSound",
      sound: "whisper.mp3",
      volume: 0.4,
      spatial: true,
      position: { x: -10, y: 1, z: -5 },
      maxDistance: 10
    }
  }
};
```

### From Object Interactions

```javascript
export const interactiveObjects = {
  door: {
    id: "office_door",
    file: "models/door.glb",
    position: { x: 5, y: 0, z: 0 },
    interactionType: "click",
    onInteract: {
      sound: "door-open.mp3",
      volume: 0.7,
      spatial: true,
      maxDistance: 20
    }
  },

  switch: {
    id: "light_switch",
    file: "models/switch.glb",
    position: { x: 2, y: 1.5, z: -3 },
    interactionType: "proximity",
    onInteract: {
      sound: "switch-click.mp3",
      volume: 0.5,
      spatial: true,
      maxDistance: 10
    }
  }
};
```

### From Animation Events

```javascript
// Sync sounds with animation frames
export const animationSounds = {
  walkCycle: {
    sound: "footstep.mp3",
    triggers: [0.1, 0.6],  // Time in animation cycle
    spatial: true,
    maxDistance: 15
  },

  runCycle: {
    sound: "footstep-run.mp3",
    triggers: [0.05, 0.25, 0.45, 0.65, 0.85],
    spatial: true,
    maxDistance: 20
  }
};
```

---

## Part 8: Advanced Features

### Randomized Sound Variation

```javascript
class SFXManager {
  playWithVariation(soundName, position) {
    const soundData = this.sfxData[soundName];

    // Check if this sound has variations
    if (soundData.variations) {
      // Pick random variation
      const variation = soundData.variations[
        Math.floor(Math.random() * soundData.variations.length)
      ];
      return this.playSpatial(variation, position);
    }

    // Or use file pattern
    if (soundData.randomize) {
      const index = Math.floor(Math.random() * soundData.variantCount) + 1;
      const variantFile = soundData.file.replace('.mp3', `-${index}.mp3`);
      return this.playSpatial({ ...soundData, file: variantFile }, position);
    }

    return this.playSpatial(soundData, position);
  }
}
```

### Pitch Variation

```javascript
// Add subtle pitch variation for realism
class SFXManager {
  playWithPitch(soundData, position) {
    const howl = this.createHowl(soundData);

    // Random pitch between 0.9 and 1.1
    const pitchVariation = 0.9 + Math.random() * 0.2;
    howl.rate(pitchVariation);

    this.playSpatial(howl, position);
  }
}
```

### Moving Sound Sources

```javascript
class SFXManager {
  update(deltaTime) {
    // Update positions of moving sounds
    this.movingSounds.forEach(moving => {
      const newPosition = {
        x: moving.position.x + moving.velocity.x * deltaTime,
        y: moving.position.y + moving.velocity.y * deltaTime,
        z: moving.position.z + moving.velocity.z * deltaTime
      };

      moving.howl.pos(newPosition.x, newPosition.y, newPosition.z);
      moving.position = newPosition;
    });
  }

  attachToObject(soundName, object) {
    const sound = this.play(soundName);

    // Store reference for updating
    this.movingSounds.push({
      howl: sound,
      object: object,
      position: object.position,
      velocity: { x: 0, y: 0, z: 0 }
    });

    return sound;
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Forgetting to Update Listener

```javascript
// ❌ WRONG: Listener never moves
playSpatial(sound) {
  const howl = new Howl({ src: [sound.file], spatial: true });
  howl.pos(sound.position.x, sound.position.y, sound.position.z);
  howl.play();
  // Listener position never updated!
}

// ✅ CORRECT: Update listener each frame
update(deltaTime) {
  const player = this.gameManager.getPlayerPosition();
  Howler.pos(player.x, player.y, player.z);
}
```

### 2. Wrong Reference Distance

```javascript
// ❌ WRONG: Reference distance too large
referenceDistance: 50  // Sound is full volume 50m away!

// ✅ CORRECT: Smaller reference distance
referenceDistance: 3  // Sound starts fading after 3m
```

### 3. Not Using Pooling

```javascript
// ❌ WRONG: Creating new Howl every frame
onFootstep() {
  const howl = new Howl({ src: ['footstep.mp3'] });  // Wasteful!
  howl.play();
}

// ✅ CORRECT: Use sound pool
onFootstep() {
  this.sfxManager.play('footstep');  // Uses pooled instance
}
```

### 4. Spatial Sounds for UI

```javascript
// ❌ WRONG: UI sounds shouldn't be spatial
uiClick: {
  file: "ui/click.mp3",
  spatial: true  // UI isn't in 3D space!
}

// ✅ CORRECT: UI sounds are 2D
uiClick: {
  file: "ui/click.mp3",
  spatial: false
}
```

### 5. Not Cleaning Up Looping Sounds

```javascript
// ❌ WRONG: Ambient sounds keep playing
enterZone() {
  const fire = new Howl({ src: ['fire.mp3'], loop: true });
  fire.play();
  // Never stops!
}

// ✅ CORRECT: Clean up when leaving zone
enterZone() {
  this.ambientFire = this.sfxManager.playLoop('fire', position);
}

exitZone() {
  this.sfxManager.stop(this.ambientFire);
}
```

---

## Performance Considerations

### Concurrent Sound Limits

```javascript
// Limit number of simultaneous sounds
const MAX_CONCURRENT_SOUNDS = 32;

// Prioritize important sounds
const soundPriority = {
  'explosion': 10,
  'dialog': 9,
  'footstep': 1,
  'ambient': 2
};

// Drop lowest priority when limit reached
```

### Distance Culling

```javascript
// Don't play sounds too far to hear
class SFXManager {
  playIfAudible(soundData, position) {
    const distance = this.getDistanceToPlayer(position);

    if (distance > soundData.maxDistance) {
      return null;  // Don't play at all
    }

    return this.playSpatial(soundData, position);
  }
}
```

### File Size Optimization

| Sound Type | Duration | Format | Bitrate | Approx Size |
|------------|----------|--------|---------|-------------|
| Footstep | 0.2 sec | MP3 | 128 kbps | ~3 KB |
| Door creak | 0.8 sec | MP3 | 128 kbps | ~13 KB |
| Explosion | 2.0 sec | MP3 | 192 kbps | ~50 KB |
| Ambient loop | 10 sec | OGG | 128 kbps | ~160 KB |

---

## 🎮 Game Design Perspective

### Sound Design Principles

1. **Feedback** - Confirm player actions
2. **Information** - Convey game state
3. **Immersion** - Enhance atmosphere
4. **Direction** - Guide player attention
5. **Realism** - Ground the world

### Sound Layering

```
GAME AUDIO HIERARCHY:

1. Dialog (highest priority)
   └── Spoken lines, narrative

2. Music
   └── Emotional backdrop, mood

3. Important SFX
   └── Explosions, major events

4. Gameplay SFX
   └── Footsteps, interactions

5. Ambient SFX (lowest priority)
   └── Background atmosphere

All layers mix together for complete experience
```

### Audio Cues for Player Guidance

```javascript
// Use sound to direct attention
export const guidanceSounds = {
  // Sound draws attention to important object
  keyJingle: {
    file: "sfx/items/key-jingle.mp3",
    volume: 0.6,
    spatial: true,
    position: { x: 10, y: 1, z: -5 },
    maxDistance: 25,
    criteria: { canPickupKey: true }
  },

  // Sound indicates danger nearby
  enemyGrowl: {
    file: "sfx/enemy/growl.mp3",
    volume: 0.5,
    spatial: true,
    maxDistance: 30,
    criteria: { enemyNearby: true }
  }
};
```

---

## Next Steps

Now that you understand SFXManager:

- [DialogManager](./dialog-manager.md) - Spoken dialogue system
- [MusicManager](./music-manager.md) - Background music with crossfades
- [VideoManager](./video-manager.md) - Video playback with alpha
- [ColliderManager](../04-input-physics/collider-manager.md) - Trigger zones for sounds

---

## References

- [Howler.js Spatial Audio](https://howlerjs.com/) - Official documentation (v2.2.4+)
- [Web Audio API - PannerNode](https://developer.mozilla.org/en-US/docs/Web/API/PannerNode) - MDN reference
- [HRTF Spatial Audio](https://webaudio-hrtf.org/) - Head-related transfer function explanation
- [Game Audio Programming](https://www.gameaudio101.com/) - Game sound design principles

*Documentation last updated: January 12, 2026*
