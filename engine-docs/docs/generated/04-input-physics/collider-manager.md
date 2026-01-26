# ColliderManager - First Principles Guide

## Overview

The **ColliderManager** handles **trigger zones** and **collision detection** in the Shadow Engine. Trigger zones are invisible 3D areas that detect when the player enters or exits them - perfect for story events, areas of effect, and interactive spaces.

Think of ColliderManager as the **game's invisible boundaries** - it knows when the player walks into a specific area and can trigger events like "play scary music" or "show dialog" automatically.

## What You Need to Know First

Before understanding ColliderManager, you should know:
- **What collision detection is** - Determining when two objects overlap
- **3D space and boundaries** - How positions (x, y, z) define areas
- **Event-driven programming** - Reacting to things happening
- **Rapier colliders** - The physics shapes used for detection

### Quick Refresher: Colliders

```javascript
// A collider is a 3D shape used for collision detection
// Common shapes:

// Box collider - rectangular area
const boxCollider = {
  shape: "box",
  halfExtents: { x: 1, y: 1, z: 1 },  // Size from center
  position: { x: 0, y: 0, z: 0 }
};

// Sphere collider - round area
const sphereCollider = {
  shape: "sphere",
  radius: 1.5,
  position: { x: 0, y: 0, z: 0 }
};
```

---

## Part 1: What Problem Does ColliderManager Solve?

### The Problem: How to Detect When Player Enters an Area?

Imagine you want something to happen when the player walks into a room:

```javascript
// ❌ WITHOUT ColliderManager - You need to check manually every frame:
function update() {
  const playerPos = player.position;
  const roomMin = { x: -5, y: 0, z: -5 };
  const roomMax = { x: 5, y: 3, z: 5 };

  if (playerPos.x > roomMin.x && playerPos.x < roomMax.x &&
      playerPos.y > roomMin.y && playerPos.y < roomMax.y &&
      playerPos.z > roomMin.z && playerPos.z < roomMax.z) {
    // Player is in room! But wait... did we just enter? Or have we been here?
    // We need extra tracking...
  }
}

// Now multiply this by 50+ trigger zones in your game!
```

### The Solution: Sensor Colliders + Events

```javascript
// ✅ WITH ColliderManager - Define once, get automatic events:

// In your scene data:
triggerZones: {
  spookyRoom: {
    position: { x: 0, y: 0, z: 0 },
    shape: "box",
    size: { x: 10, y: 3, z: 10 },
    onEnter: "playSpookyMusic",
    onExit: "stopSpookyMusic"
  }
}

// ColliderManager handles:
// - Detecting when player enters/exits
// - Only firing events ONCE per entry/exit
// - Managing multiple overlapping zones
// - Performance optimization
```

---

## Part 2: Solid Colliders vs Sensor Colliders

In Rapier (the physics engine), there are two types of colliders:

### Solid Colliders

**Generate physical contact forces** - objects can't pass through them.

```
┌─────────────┐
│   WALL      │  ← Solid collider
│             │     Player cannot walk through
├─────────────�
             │
    Player stops here
```

### Sensor Colliders

**Only detect overlap** - no physical force, just events.

```
╔═════════════╗
║   ZONE      ║  ← Sensor collider
║             │     Player CAN walk through
║     👤      ║     But engine KNOWS they're here
╚═════════════╝

Triggers: onEnter, onExit, onStay
```

**ColliderManager works with SENSOR colliders.** They're perfect for:
- Story trigger areas
- Music region changes
- Interactive object detection
- Minigame boundaries
- Safe/danger zones

---

## Part 3: Trigger Zone Architecture

### Data Structure

```javascript
// In interactiveObjectData.js or sceneData.js
export const triggerZones = {
  spookyRoomTrigger: {
    // Zone properties
    name: "spookyRoomTrigger",
    position: { x: 2.5, y: 0, z: -8 },
    size: { x: 6, y: 3, z: 6 },
    shape: "box",  // or "sphere", "capsule"

    // What happens when triggered
    onEnter: {
      action: "setState",
      state: { inSpookyRoom: true }
    },

    onExit: {
      action: "setState",
      state: { inSpookyRoom: false }
    },

    // Optional: Only trigger if player has specific state
    criteria: {
      currentState: { $gte: OFFICE_INTERIOR }
    }
  },

  runeDiscoveryZone: {
    name: "runeDiscoveryZone",
    position: { x: -4, y: 0, z: 2 },
    size: { x: 3, y: 2, z: 3 },
    shape: "sphere",

    onEnter: {
      action: "emit",
      event: "rune:approached"
    },

    // Only trigger once
    once: true,

    criteria: {
      sawRune: false
    }
  },

  dangerZone: {
    name: "lavaZone",
    position: { x: 0, y: -1, z: 0 },
    size: { x: 10, y: 1, z: 10 },
    shape: "box",

    // Called every frame while inside
    onStay: {
      action: "damage",
      amount: 0.1,
      interval: 500  // ms
    }
  }
};
```

### Properties Reference

| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Unique identifier for this zone |
| `position` | object | { x, y, z } center position |
| `size` | object | { x, y, z } dimensions (box) or radius (sphere) |
| `shape` | string | "box", "sphere", "capsule" |
| `onEnter` | object | Action when player enters zone |
| `onExit` | object | Action when player exits zone |
| `onStay` | object | Action every frame while in zone |
| `once` | boolean | Only trigger once then disable |
| `enabled` | boolean | Whether zone is active (default: true) |
| `criteria` | object | State requirements for zone to be active |

---

## Part 4: How ColliderManager Works

### Initialization Flow

```
1. ColliderManager.initialize()
   │
   ├── Import trigger zone data
   │
   ├── Create Rapier sensor colliders for each zone
   │
   ├── Set up Rapier collision events
   │
   └── Subscribe to game loop updates
```

### Runtime Flow

```
Each frame:
┌─────────────────────────────────────────────────────────┐
│  1. PhysicsManager steps the physics simulation        │
│     Rapier detects all collider overlaps               │
│                                                         │
│  2. ColliderManager checks for new overlaps            │
│     ┌───────────────────────────────────────────────┐   │
│     │ For each player collider vs trigger zone:     │   │
│     │                                               │   │
│     │  If NOT overlapping last frame:              │   │
│     │    → Fire onEnter event                       │   │
│     │    → Mark as "inside"                         │   │
│     │                                               │   │
│     │  If IS overlapping:                           │   │
│     │    → Fire onStay event (if configured)        │   │
│     │                                               │   │
│     │  If WAS overlapping but NOT this frame:       │   │
│     │    → Fire onExit event                        │   │
│     │    → Mark as "outside"                        │   │
│     └───────────────────────────────────────────────┘   │
│                                                         │
│  3. Execute triggered actions                         │
│     → setState(), emit(), playSound(), etc.          │
└─────────────────────────────────────────────────────────┘
```

---

## Part 5: Creating Trigger Zones

### Box Trigger Zone

```javascript
// Creates a rectangular trigger area
export const doorTrigger = {
  name: "officeDoorTrigger",
  position: { x: 0, y: 1, z: 5 },
  size: { x: 3, y: 2, z: 1 },
  shape: "box",

  onEnter: {
    action: "showPrompt",
    text: "Press E to enter office"
  },

  onExit: {
    action: "hidePrompt"
  }
};
```

**Visual representation:**
```
     Top view
        ↓
    ┌───┐
    │   │ ← Door (trigger zone)
    │   │
    └───┘
     ↑
   Player approaching
```

### Sphere Trigger Zone

```javascript
// Creates a spherical trigger area
export const npcDetectionZone = {
  name: "npcDetection",
  position: { x: 10, y: 0, z: 0 },
  size: { radius: 5 },  // Only radius for sphere
  shape: "sphere",

  onEnter: {
    action: "setState",
    state: { npcNearby: true }
  },

  onExit: {
    action: "setState",
    state: { npcNearby: false }
  }
};
```

**Visual representation:**
```
     Top view
        ↓
     .---.
   .'     '.  ← Sphere (radius 5)
   |   NPC   |
    '.     .'
      '---'
```

### Capsule Trigger Zone

```javascript
// Creates a vertical capsule (good for doorways, hallways)
export const hallwayTrigger = {
  name: "hallwayAmbience",
  position: { x: 0, y: 1.5, z: 0 },
  size: {
    radius: 1,     // Width of capsule
    height: 4      // Height of capsule (excluding hemispheres)
  },
  shape: "capsule",

  onEnter: {
    action: "playMusic",
    track: "hallway-ambience.mp3"
  }
};
```

**Visual representation:**
```
   Side view
       ↓
     ┌─┐
     │ │  ← Capsule
     │ │    (good for vertical passages)
     │ │
     └─┘
```

---

## Part 6: Action Types

### setState Action

Changes game state when triggered:

```javascript
onEnter: {
  action: "setState",
  state: {
    inDangerZone: true,
    health: 0.9  // Direct value assignment
  }
}
```

### emit Action

Fires a custom event:

```javascript
onEnter: {
  action: "emit",
  event: "player:enteredSpookyRoom",
  data: { roomName: "spookyRoom" }
}

// Other systems can listen:
gameManager.on("player:enteredSpookyRoom", (data) => {
  console.log(`Player entered ${data.roomName}`);
});
```

### playSound Action

Plays a sound effect:

```javascript
onEnter: {
  action: "playSound",
  sound: "creaky-floor.mp3",
  volume: 0.5,
  spatial: true,
  position: { x: 0, y: 0, z: 0 }
}
```

### showDialog Action

Shows a dialog or prompt:

```javascript
onEnter: {
  action: "showDialog",
  text: "It's dark in here...",
  duration: 3000
}

onExit: {
  action: "hideDialog"
}
```

### damage Action

Applies damage over time:

```javascript
onStay: {
  action: "damage",
  amount: 0.1,
  interval: 500,  // Every 500ms
  type: "fire"
}
```

### custom Action

Calls a custom function:

```javascript
onEnter: {
  action: "custom",
  function: "triggerRuneSequence",
  params: { runeId: "rune3" }
}

// You register the handler:
colliderManager.registerHandler("triggerRuneSequence", (params) => {
  startRuneSequence(params.runeId);
});
```

---

## Part 7: Advanced Features

### One-Time Triggers

Zones that only trigger once:

```javascript
export const oneTimeDiscovery = {
  name: "secretDiscovery",
  position: { x: -10, y: 0, z: -10 },
  size: { x: 2, y: 2, z: 2 },
  shape: "box",

  once: true,  // ← Only triggers first time!

  onEnter: {
    action: "setState",
    state: { discoveredSecret: true }
  }
};
```

### Conditional Triggers

Zones that only work under certain conditions:

```javascript
export const nightOnlyTrigger = {
  name: "nightMusic",
  position: { x: 0, y: 0, z: 0 },
  size: { x: 20, y: 5, z: 20 },
  shape: "box",

  // Criteria determine when zone is active
  criteria: {
    currentState: { $in: [NIGHT_TIME, MIDNIGHT_HOUR] }
  },

  onEnter: {
    action: "playMusic",
    track: "night-ambience.mp3"
  }
};
```

### Enabled/Disabled Zones

Dynamically enable or disable zones:

```javascript
// In game code:
gameManager.on("rune:destroyed", (runeId) => {
  // Disable that rune's trigger zone
  colliderManager.setZoneEnabled(`rune${runeId}Zone`, false);
});
```

### Nested Zones

Zones can overlap - both will trigger:

```javascript
// Large outer zone
export const outerZone = {
  name: "roomAmbience",
  size: { x: 20, y: 5, z: 20 },
  shape: "box",
  onEnter: { action: "playMusic", track: "room.mp3" }
};

// Smaller inner zone
export const innerZone = {
  name: "nearFire",
  size: { x: 3, y: 3, z: 3 },
  shape: "box",
  onEnter: { action: "playSound", sound: "crackling.mp3" }
};

// Both fire when player is in center:
// - roomAmbience (from outer zone)
// - crackling fire (from inner zone)
```

---

## Part 8: Rapier Integration Details

### Creating Sensor Colliders

```javascript
// Based on official Rapier documentation
// https://rapier.rs/docs/user_guides/javascript/colliders/

class ColliderManager {
  createTriggerZone(zoneData) {
    // Create a sensor collider (not solid)
    const colliderDesc = RAPIER.ColliderDesc.cuboid(
      zoneData.size.x / 2,  // Half-extents
      zoneData.size.y / 2,
      zoneData.size.z / 2
    )
      .setTranslation(
        zoneData.position.x,
        zoneData.position.y,
        zoneData.position.z
      )
      .setSensor(true)  // ← KEY: Makes it a sensor (no physical force)
      .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS);  // Get events

    // Create without rigid body (fixed in world space)
    const collider = this.world.createCollider(colliderDesc);

    // Store reference
    this.zones.set(zoneData.name, {
      data: zoneData,
      collider: collider,
      isInside: new Set()  // Track what's inside
    });
  }
}
```

### Handling Collision Events

```javascript
class ColliderManager {
  setupCollisionEvents() {
    // Rapier provides collision events through the world
    this.world.addEventListener("collision", (event) => {
      const { collider1, collider2, started } = event;

      // Find which is our trigger zone
      const zone = this.findZoneForCollider(collider1) ||
                   this.findZoneForCollider(collider2);

      if (!zone) return;

      // Check if player is involved
      const playerCollider = this.getPlayerCollider(collider1, collider2);
      if (!playerCollider) return;

      if (started) {
        // Collision started (onEnter)
        this.handleEnter(zone, playerCollider);
      } else {
        // Collision ended (onExit)
        this.handleExit(zone, playerCollider);
      }
    });
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Forgetting setSensor(true)

```javascript
// ❌ WRONG: Creates a solid wall!
const collider = RAPIER.ColliderDesc.cuboid(1, 1, 1);
world.createCollider(collider);

// ✅ CORRECT: Creates a trigger zone
const collider = RAPIER.ColliderDesc.cuboid(1, 1, 1)
  .setSensor(true);  // ← Critical!
world.createCollider(collider);
```

### 2. Not Setting Active Events

```javascript
// ❌ WRONG: No events will fire!
const collider = RAPIER.ColliderDesc.cuboid(1, 1, 1)
  .setSensor(true);

// ✅ CORRECT: Enable collision events
const collider = RAPIER.ColliderDesc.cuboid(1, 1, 1)
  .setSensor(true)
  .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS);  // ← Critical!
```

### 3. Wrong Size Format

```javascript
// ❌ WRONG: Using full dimensions
const collider = RAPIER.ColliderDesc.cuboid(2, 2, 2);
// This creates a 4x4x4 box! (half-extents * 2)

// ✅ CORRECT: Rapier uses half-extents
const collider = RAPIER.ColliderDesc.cuboid(1, 1, 1);
// This creates a 2x2x2 box
```

### 4. Checking Position Manually

```javascript
// ❌ WRONG: Manual position checking (expensive!)
function update() {
  if (player.position.x > zone.min.x && player.position.x < zone.max.x && ...) {
    // Complex and error-prone
  }
}

// ✅ CORRECT: Let ColliderManager handle it
// Zone fires events automatically when entered
```

### 5. Not Handling onStay Properly

```javascript
// ❌ WRONG: Firing onStay every frame (60+ calls per second!)
onStay: {
  action: "damage",
  amount: 0.1  // That's 6 damage per second!
}

// ✅ CORRECT: Use interval
onStay: {
  action: "damage",
  amount: 0.1,
  interval: 500  // Only every 500ms
}
```

---

## Performance Considerations

### Broad Phase Optimization

Rapier uses a "broad phase" to quickly eliminate impossible collisions:

```
1. Broad Phase:
   - Check bounding boxes (very fast)
   - Only check pairs that might overlap

2. Narrow Phase:
   - Precise collision detection
   - Generate collision events
```

### Zone Count Guidelines

| Zone Count | Performance | Recommendation |
|------------|-------------|----------------|
| 0-20 | Excellent | No optimization needed |
| 20-50 | Good | Consider spatial partitioning |
| 50-100 | Fair | Use collision groups, spatial partitioning |
| 100+ | Poor | Reconsider design, use quadtrees/octrees |

### Optimization Tips

```javascript
// 1. Use collision groups to skip unnecessary checks
collider.setCollisionGroups(0x00010001);  // Only interact with group 0

// 2. Disable zones when not needed
colliderManager.setZoneEnabled("distantZone", false);

// 3. Use larger zones instead of many small ones
// ❌ 10 small zones
for (let i = 0; i < 10; i++) {
  createZone({ position: i * 2, size: 1 });
}

// ✅ 1 large zone (fewer collision checks)
createZone({ position: 10, size: 20 });
```

---

## 🎮 Game Design Perspective

### Creative Intent

Trigger zones are powerful tools for **environmental storytelling** and **pacing control**:

- **Atmosphere transitions** - Change ambience when entering themed areas
- **Pacing gates** - Slow players down before important moments
- **Discovery rewards** - Trigger content when players explore off-path
- **Tension building** - Subtle audio/visual changes as danger approaches

### Design Examples

**The Slow Reveal:**
```
Zone 1 (far): Crickets chirping, peaceful
Zone 2 (mid): Silence, occasional owl hoot
Zone 3 (near): Unsettling drone sound
Zone 4 (at): Scary music swells

Each zone builds tension progressively!
```

**The Invisible Guide:**
```
Zone 1: Hint at something ahead ("Strange smell...")
Zone 2: Audio cue (faint humming)
Zone 3: Visual cue (flickering light)
Zone 4: The reveal (door slowly opens)

Players feel like they discovered it organically!
```

---

## 🎨 Level Design Perspective

### How To Design Trigger Zones Like This

#### Step 1: Map Your Player Journey

```
Draw a top-down map of your level:

Start ─────► [Hallway] ─────► [Room A]
                        │
                        └──► [Room B]

Mark where you want triggers to happen.
```

#### Step 2: Define Emotional Beats

```
Hallway:     Curiosity, mystery
Room A:      Relief, safety
Room B:      Tension, danger

Place triggers to support these emotions.
```

#### Step 3: Create Trigger Zones

```
// At hallway entrance
hallwayTrigger: {
  onEnter: { action: "playMusic", track: "mystery.mp3" }
}

// At Room A entrance
roomASafe: {
  onEnter: { action: "setState", state: { safeZone: true } }
}

// At Room B entrance
roomBDanger: {
  onEnter: { action: "playSound", sound: "heartbeat.mp3" }
}
```

---

## Next Steps

Now that you understand ColliderManager:

- [PhysicsManager](./physics-manager.md) - Rapier physics integration
- [CharacterController](./character-controller.md) - Player movement
- [InputManager](./input-manager.md) - How player input works
- [Interactive Objects](../08-interactive-objects/) - Using triggers with interactables

---

## References

- [Rapier Colliders Documentation](https://rapier.rs/docs/user_guides/javascript/colliders/) - Official guide
- [Rapier Getting Started](https://rapier.rs/docs/user_guides/javascript/getting_started_js/) - Installation and setup
- [Rapier Scene Queries](https://rapier.rs/docs/user_guides/javascript/scene_queries/) - Manual collision queries
- [Collision Detection Basics](https://en.wikipedia.org/wiki/Collision_detection) - Wikipedia overview

*Documentation last updated: January 12, 2026*
