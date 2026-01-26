# ZoneManager - Dynamic Scene Loading System

**Shadow Engine Scene Management Documentation**

---

## What You Need to Know First

Before understanding ZoneManager, you should know:
- **SceneManager** - Basic scene loading (See: *SceneManager*)
- **Gaussian Splatting** - Point cloud rendering basics
- **Memory Management** - Why we can't load everything at once
- **Async/Await** - JavaScript asynchronous programming

---

## Overview

**ZoneManager** handles dynamic loading and unloading of scene zones (areas) based on player position. This enables large, seamless worlds without loading screens by only keeping nearby zones in memory.

### The Problem This Solves

```
Load everything at once:
    ↓
10 zones × 10M splats = 100M splats
    ↓
Memory exhausted! Browser crashes!

Zone loading:
    ↓
Only load current zone + adjacent zones
    ↓
~20M splats in memory = Runs smoothly!
```

---

## 🎮 Game Design Perspective

### Creative Intent

**Why use zone-based loading instead of individual loading screens?**

| Approach | Player Experience |
|---------|-------------------|
| **Loading Screens** | "Wait 10 seconds..." - Breaks immersion |
| **Corridors** | Walk through tunnel while loading - Feels artificial |
| **Zone Loading** | Seamless exploration - World feels continuous and real |

Zone loading creates **unbroken immersion** - the player can explore freely without interruption, maintaining the feeling of being in a real, continuous space.

### Design Philosophy

**Invisible Boundaries = Believable World:**

```
Obvious loading:
    Player walks → "LOADING..." → New area appears
    ↓
Player thinks: "I'm playing a game with loading zones"

Seamless zone loading:
    Player walks → World continues smoothly
    ↓
Player feels: "I'm exploring this place. It just keeps going."
```

The goal is for players to never notice the technical boundaries.

---

## Core Concepts (Beginner Friendly)

### What is a Zone?

Think of a zone as a **chunk of the game world**:

```
Whole game world:
    ┌─────────────────────────────────────┐
    │                                     │
    │  [Zone A]  [Zone B]  [Zone C]       │
    │    Plaza     Alley     Office        │
    │                                     │
    └─────────────────────────────────────┘

Each zone = One splat file + objects + data
```

### Active vs Inactive Zones

```
Active Zones:
    ↓
Currently loaded in memory
    Player can see or enter soon

Inactive Zones:
    ↓
    Unloaded from memory
    Save memory, faster rendering
```

### Overlap Zones

```
Zone A    Zone B
   │          │
   └────┬─────┘
        │
    Overlap area
    Both zones loaded briefly
    Smooth transition!
```

---

## How It Works

### Zone Detection and Loading

```
┌─────────────────────────────────────────────────────────┐
│                    1. Track Player Position             │
│  - Every frame: update player position                │
│  - Check which zone player is in                      │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    2. Determine Active Zones            │
│  - Current zone (player is here)                       │
│  - Adjacent zones (player might enter)                 │
│  - Unload distant zones                                │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    3. Load Needed Zones                  │
│  - Async load new zones                                │
│  - Keep old zones until new ones ready                 │
│  - Crossfade at boundaries                              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    4. Unload Old Zones                  │
│  - Remove zones too far away                           │
│  - Free memory                                          │
└─────────────────────────────────────────────────────────┘
```

### Zone Transition Flow

```
Player approaches Zone B boundary
    ↓
ZoneManager detects proximity
    ↓
Start loading Zone B in background
    ↓
Player crosses boundary
    ↓
Zone B is already loaded (hopefully!)
    ↓
Seamless transition - no loading screen
    ↓
Unload Zone A (now far away)
```

---

## Architecture

### ZoneManager Class Structure

```javascript
/**
 * ZoneManager - Dynamic zone loading system
 */
class ZoneManager {
  constructor(scene, config) {
    this.scene = scene;
    this.config = config;

    // Zone definitions
    this.zones = new Map();
    this.zoneGraph = new Map();  // Adjacency relationships

    // State tracking
    this.activeZones = new Set();
    this.loadingZones = new Set();
    this.currentZone = null;

    // Player tracking
    this.playerPosition = new THREE.Vector3();

    // Configuration
    this.loadDistance = config.loadDistance || 50;
    this.unloadDistance = config.unloadDistance || 80;
    this.loadAhead = config.loadAhead || 1;  // Zones to preload

    // Initialize
    this.init();
  }

  async init() {
    // Load zone definitions
    await this.loadZoneDefinitions();

    // Start player tracking
    this.startPlayerTracking();
  }
}
```

### Zone Data Structure

```javascript
/**
 * Zone configuration
 */
const ZONE_DEFINITION = {
  id: 'plaza',
  name: 'Plaza Area',

  // Boundaries
  bounds: {
    center: { x: 0, y: 0, z: 0 },
    size: { x: 100, y: 50, z: 100 }
  },

  // Splat data
  splat: {
    file: '/assets/splats/plaza.ply',
    maxPoints: 10000000,
    renderScale: 1.0
  },

  // Connections to other zones
  connections: [
    { to: 'alley_west', trigger: { x: -45, z: 0, radius: 10 } },
    { to: 'intersection', trigger: { x: 45, z: 0, radius: 10 } },
    { to: 'office_exterior', trigger: { x: 0, z: -45, radius: 10 } }
  ],

  // Objects in this zone
  objects: [
    { id: 'phone_booth', position: { x: -8, y: 0, z: 3 } },
    { id: 'rusted_car', position: { x: 0, y: 0, z: -8 } }
  ],

  // Preload priority
  priority: 1  // Lower = loads first
};
```

---

## Usage Examples

### Defining Zones

```javascript
/**
 * Zone configuration setup
 */
const ZONES = {
  plaza: {
    id: 'plaza',
    bounds: {
      center: new THREE.Vector3(0, 0, 0),
      size: new THREE.Vector3(80, 40, 80)
    },
    splat: '/assets/splats/plaza.ply',
    connections: ['intersection', 'alley_west', 'office_exterior']
  },

  intersection: {
    id: 'intersection',
    bounds: {
      center: new THREE.Vector3(50, 0, 0),
      size: new THREE.Vector3(60, 40, 60)
    },
    splat: '/assets/splats/intersection.ply',
    connections: ['plaza', 'alley_east', 'street_south']
  },

  alley_west: {
    id: 'alley_west',
    bounds: {
      center: new THREE.Vector3(-60, 0, 0),
      size: new THREE.Vector3(100, 40, 30)
    },
    splat: '/assets/splats/alley_west.ply',
    connections: ['plaza', 'dead_end']
  }
};
```

### Zone Loading

```javascript
/**
 * Basic zone loading example
 */
async function initializeZoneManager(scene) {
  const zoneManager = new ZoneManager(scene, {
    loadDistance: 60,    // Start loading at 60m
    unloadDistance: 100, // Unload at 100m
    loadAhead: 2,        // Preload 2 zones ahead
    transitionTime: 1.0  // Crossfade duration
  });

  // Register all zones
  for (const [id, config] of Object.entries(ZONES)) {
    await zoneManager.registerZone(id, config);
  }

  // Start with initial zone
  await zoneManager.loadZone('plaza');

  return zoneManager;
}
```

### Zone Transitions

```javascript
/**
 * Zone transition handling
 */
class ZoneTransition {
  constructor(zoneManager) {
    this.zoneManager = zoneManager;
    this.transitions = new Map();
  }

  /**
   * Handle zone transition
   */
  async transitionTo(fromZone, toZone) {
    const transitionId = `${fromZone}_${toZone}`;

    // Check if already transitioning
    if (this.transitions.has(transitionId)) {
      return this.transitions.get(transitionId);
    }

    // Start transition
    const transition = {
      from: fromZone,
      to: toZone,
      progress: 0,
      complete: false
    };

    this.transitions.set(transitionId, transition);

    // Ensure target zone is loaded
    if (!this.zoneManager.isZoneLoaded(toZone)) {
      await this.zoneManager.loadZone(toZone);
    }

    // Crossfade between zones
    await this.crossfadeZones(fromZone, toZone);

    // Unload old zone if safe
    if (this.canUnloadZone(fromZone)) {
      await this.zoneManager.unloadZone(fromZone);
    }

    transition.complete = true;
    this.transitions.delete(transitionId);

    return transition;
  }

  /**
   * Crossfade between two zones
   */
  async crossfadeZones(fromZone, toZone) {
    const duration = 1000; // 1 second
    const startTime = performance.now();

    const fromSplat = this.zoneManager.getSplat(fromZone);
    const toSplat = this.zoneManager.getSplat(toZone);

    return new Promise((resolve) => {
      const fade = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Fade out old zone
        fromSplat.setOpacity(1 - progress);

        // Fade in new zone
        toSplat.setOpacity(progress);

        if (progress < 1) {
          requestAnimationFrame(fade);
        } else {
          fromSplat.visible = false;
          resolve();
        }
      };

      fade();
    });
  }

  /**
   * Check if zone can be safely unloaded
   */
  canUnloadZone(zoneId) {
    const zone = this.zoneManager.zones.get(zoneId);
    const playerPos = this.zoneManager.playerPosition;

    // Unload if player is far enough and not coming back soon
    const distance = zone.bounds.center.distanceTo(playerPos);
    return distance > this.zoneManager.config.unloadDistance;
  }
}
```

---

## Implementation

### Complete ZoneManager

```javascript
/**
 * ZoneManager - Full implementation
 */
class ZoneManager {
  constructor(scene, config) {
    this.scene = scene;
    this.config = config;

    // Storage
    this.zones = new Map();
    this.loadedZones = new Map();
    this.activeZones = new Set();

    // Player tracking
    this.playerPosition = new THREE.Vector3();
    this.currentZone = null;

    // Preloading
    this.loadQueue = [];
    this.maxConcurrentLoads = config.maxConcurrentLoads || 2;

    // Events
    this.onZoneEnter = null;
    this.onZoneLeave = null;
    this.onZoneLoad = null;
    this.onZoneUnload = null;
  }

  /**
   * Register a zone definition
   */
  registerZone(zoneId, config) {
    const zone = {
      id: zoneId,
      config: config,
      bounds: new THREE.Box3(
        new THREE.Vector3(
          config.bounds.center.x - config.bounds.size.x / 2,
          config.bounds.center.y - config.bounds.size.y / 2,
          config.bounds.center.z - config.bounds.size.z / 2
        ),
        new THREE.Vector3(
          config.bounds.center.x + config.bounds.size.x / 2,
          config.bounds.center.y + config.bounds.size.y / 2,
          config.bounds.center.z + config.bounds.size.z / 2
        )
      ),
      connections: new Set(config.connections || []),
      loaded: false,
      loading: false,
      splat: null
    };

    this.zones.set(zoneId, zone);
  }

  /**
   * Update zone states based on player position
   */
  update(playerPosition, deltaTime) {
    this.playerPosition.copy(playerPosition);

    // Find which zone player is in
    const currentZone = this.findZoneAtPosition(playerPosition);

    // Handle zone change
    if (currentZone && currentZone !== this.currentZone) {
      this.onEnterZone(currentZone);
      this.currentZone = currentZone;
    }

    // Update loading/unloading
    this.updateZoneLoads(playerPosition);
  }

  /**
   * Find zone containing position
   */
  findZoneAtPosition(position) {
    for (const [id, zone] of this.zones) {
      if (zone.bounds.containsPoint(position)) {
        return id;
      }
    }
    return null;
  }

  /**
   * Handle entering a zone
   */
  async onEnterZone(zoneId) {
    const previousZone = this.currentZone;

    // Fire enter event
    if (this.onZoneEnter) {
      this.onZoneEnter(zoneId, previousZone);
    }

    // Ensure zone is loaded
    if (!this.loadedZones.has(zoneId)) {
      await this.loadZone(zoneId);
    }

    // Mark as active
    this.activeZones.add(zoneId);

    // Preload connected zones
    const zone = this.zones.get(zoneId);
    for (const connection of zone.connections) {
      this.preloadZone(connection);
    }
  }

  /**
   * Load a zone
   */
  async loadZone(zoneId) {
    const zone = this.zones.get(zoneId);
    if (!zone || zone.loaded || zone.loading) {
      return;
    }

    zone.loading = true;

    try {
      // Load splat
      zone.splat = await this.scene.splatLoader.load(zone.config.splat);
      this.scene.add(zone.splat);

      zone.loaded = true;
      zone.loading = false;
      this.loadedZones.set(zoneId, zone);

      // Fire load event
      if (this.onZoneLoad) {
        this.onZoneLoad(zoneId);
      }

      console.log(`Zone loaded: ${zoneId}`);
    } catch (error) {
      console.error(`Failed to load zone ${zoneId}:`, error);
      zone.loading = false;
    }
  }

  /**
   * Preload a zone in background
   */
  async preloadZone(zoneId) {
    const zone = this.zones.get(zoneId);
    if (!zone || zone.loaded || zone.loading) {
      return;
    }

    // Add to load queue
    this.loadQueue.push(zoneId);

    // Process queue
    this.processLoadQueue();
  }

  /**
   * Process load queue (limit concurrent loads)
   */
  async processLoadQueue() {
    const loadingCount = Array.from(this.zones.values())
      .filter(z => z.loading).length;

    if (loadingCount >= this.maxConcurrentLoads) {
      return; // Already loading max concurrent
    }

    while (this.loadQueue.length > 0) {
      const zoneId = this.loadQueue.shift();
      const zone = this.zones.get(zoneId);

      if (zone && !zone.loaded && !zone.loading) {
        await this.loadZone(zoneId);

        // Check if still at concurrent load limit
        const newLoadingCount = Array.from(this.zones.values())
          .filter(z => z.loading).length;

        if (newLoadingCount >= this.maxConcurrentLoads) {
          break;
        }
      }
    }
  }

  /**
   * Update zone loading/unloading
   */
  updateZoneLoads(playerPosition) {
    for (const [id, zone] of this.zones) {
      if (!zone.loaded) continue;

      const distance = this.getDistanceToZone(playerPosition, zone);

      // Unload distant zones
      if (distance > this.config.unloadDistance) {
        if (this.activeZones.has(id)) {
          this.activeZones.delete(id);
        }
        this.unloadZone(id);
      }
    }
  }

  /**
   * Get distance to zone
   */
  getDistanceToZone(position, zone) {
    return zone.bounds.distanceToPoint(position);
  }

  /**
   * Unload a zone
   */
  async unloadZone(zoneId) {
    const zone = this.zones.get(zoneId);
    if (!zone || !zone.loaded || zone.loading) {
      return;
    }

    // Check if still needed (active or adjacent to active)
    if (this.isZoneNeeded(zoneId)) {
      return;
    }

    // Remove splat from scene
    if (zone.splat) {
      this.scene.remove(zone.splat);
      zone.splat.dispose();
      zone.splat = null;
    }

    zone.loaded = false;
    this.loadedZones.delete(zoneId);

    // Fire unload event
    if (this.onZoneUnload) {
      this.onZoneUnload(zoneId);
    }

    console.log(`Zone unloaded: ${zoneId}`);
  }

  /**
   * Check if zone is still needed
   */
  isZoneNeeded(zoneId) {
    // Active zones are always needed
    if (this.activeZones.has(zoneId)) {
      return true;
    }

    // Zones adjacent to active zones are needed
    for (const activeId of this.activeZones) {
      const activeZone = this.zones.get(activeId);
      if (activeZone && activeZone.connections.has(zoneId)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Get zone by ID
   */
  getZone(zoneId) {
    return this.zones.get(zoneId);
  }

  /**
   * Get all active zones
   */
  getActiveZones() {
    return Array.from(this.activeZones);
  }

  /**
   * Get current zone
   */
  getCurrentZone() {
    return this.currentZone;
  }
}
```

---

## Performance Considerations

### Zone Size Tuning

```javascript
// Small zones = frequent loading but less memory per load
const smallZones = {
  size: 30,  // meters
  memory: ~3MB per zone,
  loadFrequency: 'high'
};

// Large zones = less frequent loading but more memory
const largeZones = {
  size: 100,  // meters
  memory: ~10MB per zone,
  loadFrequency: 'low'
};

// Recommended balance
const balancedZones = {
  size: 50-75,  // meters
  loadFrequency: 'medium'
};
```

### Preload Strategy

```javascript
/**
 * Smart preloading based on player movement
 */
class SmartPreloader {
  constructor(zoneManager) {
    this.zoneManager = zoneManager;
    this.playerHistory = [];
    this.historyLength = 10;
  }

  /**
   * Predict next zones based on movement
   */
  predictNextZones(currentZone) {
    const zone = this.zoneManager.getZone(currentZone);
    if (!zone) return [];

    // Track player movement direction
    const direction = this.getMovementDirection();

    // Prioritize zones in movement direction
    const predictions = [];

    for (const connection of zone.connections) {
      const connectionZone = this.zoneManager.getZone(connection);
      if (!connectionZone) continue;

      const toCenter = connectionZone.bounds.center;
      const score = this.scoreZone(toCenter, direction);

      predictions.push({
        zoneId: connection,
        score: score
      });
    }

    // Sort by score and return top N
    predictions.sort((a, b) => b.score - a.score);
    return predictions.slice(0, 2).map(p => p.zoneId);
  }

  /**
   * Get recent movement direction
   */
  getMovementDirection() {
    if (this.playerHistory.length < 2) {
      return new THREE.Vector3();
    }

    const recent = this.playerHistory.slice(-5);
    const direction = new THREE.Vector3();

    for (let i = 1; i < recent.length; i++) {
      direction.subVectors(recent[i], recent[i - 1]);
    }

    return direction.normalize();
  }

  /**
   * Score zone based on movement direction
   */
  scoreZone(zoneCenter, movementDirection) {
    const toZone = zoneCenter.clone()
      .sub(this.playerHistory[this.playerHistory.length - 1])
      .normalize();

    return toZone.dot(movementDirection);
  }
}
```

---

## Common Mistakes Beginners Make

### Mistake 1: Zones Too Small

```javascript
// BAD: Tiny zones = constant loading
const badZones = {
  size: 10,  // 10 meters = constant loading
  result: 'Players see loading hitches'
};

// GOOD: Reasonable zone sizes
const goodZones = {
  size: 60,  // 60 meters = smooth gameplay
  result: 'Seamless exploration'
};
```

### Mistake 2: No Preloading

```javascript
// BAD: Load zone only when entering
async onPlayerEnter(zoneId) {
  await loadZone(zoneId);  // Player waits!
}

// GOOD: Preload before entering
async onPlayerApproach(zoneId) {
  await preloadZone(zoneId);  // Already loaded
}
```

### Mistake 3: Forgetting to Unload

```javascript
// BAD: Never unload zones
const badManager = {
  loadOnly: true,
  unload: false,
  result: 'Memory leak, eventual crash'
};

// GOOD: Unload distant zones
const goodManager = {
  loadAndUnload: true,
  result: 'Stable memory usage'
};
```

---

## Related Systems

- **SceneManager** - Overall scene management
- **SparkRenderer** - Splat rendering
- **Performance Profiles** - Memory and performance settings
- **Memory Management** - Overall memory tracking

---

## Source File Reference

- **Location**: `../src/managers/ZoneManager.js` (hypothetical)
- **Key exports**:
  - `ZoneManager` - Main zone management class
  - `ZoneTransition` - Zone transition handling
- **Dependencies**: SceneManager, SparkRenderer

---

## References

- [Open World Loading Patterns](https://www.gamedeveloper.com/programming/open-world-streaming-) - Industry best practices
- [Three.js LoadingManager](https://threejs.org/docs/#api/en/loaders/LoadingManager) - Asset loading utilities

---

**RALPH_STATUS:**
- **Status**: ZoneManager documentation complete
- **Phase**: 3 - Scene & Rendering
- **Related**: LightManager (1 remaining)
