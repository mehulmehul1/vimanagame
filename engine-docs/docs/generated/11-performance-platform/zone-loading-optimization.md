# Zone Loading Optimization - First Principles Guide

## Overview

**Zone Loading Optimization** is the system that dynamically loads and unloads game content based on player position. Instead of loading the entire game at once (which would take forever and consume all memory), the world is divided into "zones" or "chunks." As the player moves through the world, zones ahead are pre-loaded and zones behind are unloaded, keeping memory usage manageable while maintaining smooth gameplay.

Think of zone loading like the **"scrolling spotlight"**—like a stage spotlight that illuminates only what's immediately needed while the rest remains in darkness, zone loading keeps only the relevant content active, loading ahead as you move and unloading what you've left behind.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create a seamless, boundless world without loading screens interrupting exploration. Players should feel like they're in a continuous environment, not discrete "levels" that pause the experience when transitioning.

**Why Zone Loading Matters?**
- **Seamless World**: No loading breaks between areas
- **Memory Efficiency**: Only load what's needed
- **Faster Initial Load**: Start playing sooner
- **Large Worlds**: Support massive environments
- **Platform Scaling**: Adjust zone size based on device

**Zone Loading Flow**:
```
Player in Zone A
    ↓
Moving toward Zone B
    ↓
Preload Zone B (in background)
    ↓
Player enters Zone B
    ↓
 seamless (no loading screen)
    ↓
Unload Zone A (after delay)
    ↓
Repeat for Zone C
```

### Design Principles

**1. Imperceptible Transitions**
Players shouldn't notice zones loading:
- Preload before reaching zone boundary
- Fade in/out content smoothly
- Keep previous zone briefly overlapping

**2. Memory Budget**
Know your limits and stick to them:
- Set maximum memory per platform
- Count assets actively in memory
- Unload aggressively when over budget

**3. Predictive Loading**
Load what player will likely need:
- Load zones in movement direction
- Load connected zones (doorways, paths)
- Consider quest objectives

**4. Graceful Degradation**
Handle slow storage/network:
- Show loading indicators if needed
- Reduce zone size on slow devices
- Fall back to loading screens if necessary

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding zone loading, you should know:
- **Asynchronous loading** - Using Promises/async for non-blocking loads
- **Memory management** - Tracking and releasing resources
- **Spatial partitioning** - Dividing world into regions
- **Asset bundles** - Grouping related assets
- **Distance checking** - Determining what's near the player

### Core Architecture

```
ZONE LOADING ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                    ZONE MANAGER                         │
│  - Track active zones                                   │
│  - Monitor player position                             │
│  - Trigger load/unload                                 │
│  - Manage memory budget                                │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   ZONES       │  │   LOADING    │  │   MEMORY     │
│  - Active     │  │  - Preload   │  │  - Budget    │
│  - Nearby     │  │  - Unload    │  │  - Tracking  │
│  - Distant    │  │  - Priority  │  │  - Cleanup   │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   ASSETS     │
                    │  - Models    │
                    │  - Textures  │
                    │  - Audio     │
                    │  - Splats    │
                    └───────────────┘
```

### ZoneManager Class

```javascript
class ZoneManager {
  constructor(options = {}) {
    this.gameManager = options.gameManager;
    this.sceneManager = options.sceneManager;
    this.assetLoader = options.assetLoader;
    this.logger = options.logger || console;

    // Zone definitions
    this.zones = new Map();  // id -> zone data
    this.zoneBounds = new Map();  // id -> bounding box

    // Current state
    this.activeZones = new Set();  // Currently loaded
    this.loadingZones = new Set();  // Currently loading
    this.unloadingZones = new Set();  // Currently unloading

    // Player tracking
    this.playerPosition = new THREE.Vector3();
    this.previousZone = null;
    this.currentZone = null;

    // Configuration
    this.config = {
      preloadDistance: options.preloadDistance || 50,  // Distance to start loading
      unloadDistance: options.unloadDistance || 60,   // Distance to unload
      unloadDelay: options.unloadDelay || 5000,        // ms before unloading
      maxActiveZones: options.maxActiveZones || 5,     // Maximum zones loaded
      enableStreaming: options.enableStreaming !== false,
      streamingInterval: options.streamingInterval || 500,  // ms between checks
      memoryBudget: options.memoryBudget || 500 * 1024 * 1024  // 500MB default
    };

    // Memory tracking
    this.memoryUsage = new Map();  // zoneId -> bytes used
    this.totalMemoryUsed = 0;

    // Loading queue
    this.loadQueue = [];
    this.isProcessingQueue = false;

    // Unload timers
    this.unloadTimers = new Map();  // zoneId -> timer

    // Callbacks
    this.onZoneLoaded = options.onZoneLoaded || null;
    this.onZoneUnloaded = options.onZoneUnloaded || null;
    this.onZoneActivated = options.onZoneActivated || null;

    // Initialize
    this.initialize();
  }

  /**
   * Initialize zone manager
   */
  initialize() {
    // Start player position tracking
    this.trackPlayer();

    // Start streaming loop
    if (this.config.enableStreaming) {
      this.startStreaming();
    }

    this.logger.log('ZoneManager initialized');
  }

  /**
   * Register a zone
   */
  registerZone(zone) {
    // Zone structure:
    // {
    //   id: string,
    //   name: string,
    //   position: THREE.Vector3,
    //   size: THREE.Vector3,  // bounding box size
    //   assets: {
    //     models: string[],
    //     textures: string[],
    //     audio: string[],
    //     splats: string[]
    //   },
    //   connections: string[],  // adjacent zone IDs
    //   priority: number,
    //   estimatedMemory: number
    // }

    this.zones.set(zone.id, zone);

    // Create bounding box
    const min = new THREE.Vector3(
      zone.position.x - zone.size.x / 2,
      zone.position.y - zone.size.y / 2,
      zone.position.z - zone.size.z / 2
    );
    const max = new THREE.Vector3(
      zone.position.x + zone.size.x / 2,
      zone.position.y + zone.size.y / 2,
      zone.position.z + zone.size.z / 2
    );
    this.zoneBounds.set(zone.id, new THREE.Box3(min, max));

    this.logger.log(`Registered zone: ${zone.id}`);
  }

  /**
   * Unregister a zone
   */
  unregisterZone(zoneId) {
    this.zones.delete(zoneId);
    this.zoneBounds.delete(zoneId);
  }

  /**
   * Track player position
   */
  trackPlayer() {
    if (this.gameManager.player) {
      this.gameManager.player.on('positionChanged', (position) => {
        this.playerPosition.copy(position);
        this.checkZones();
      });
    }
  }

  /**
   * Start the streaming loop
   */
  startStreaming() {
    this.streamingInterval = setInterval(() => {
      this.updateStreaming();
    }, this.config.streamingInterval);
  }

  /**
   * Stop the streaming loop
   */
  stopStreaming() {
    if (this.streamingInterval) {
      clearInterval(this.streamingInterval);
      this.streamingInterval = null;
    }
  }

  /**
   * Main streaming update
   */
  updateStreaming() {
    // Get all zones with their distances
    const zoneDistances = [];

    for (const [zoneId, zone] of this.zones) {
      const distance = this.playerPosition.distanceTo(zone.position);
      zoneDistances.push({ zoneId, distance, zone });
    }

    // Sort by distance
    zoneDistances.sort((a, b) => a.distance - b.distance);

    // Determine which zones should be loaded
    const zonesToLoad = [];
    const zonesToUnload = new Set(this.activeZones);

    for (const { zoneId, distance, zone } of zoneDistances) {
      if (distance <= this.config.preloadDistance) {
        // Should be loaded
        zonesToLoad.push(zoneId);
        zonesToUnload.delete(zoneId);
      }
    }

    // Enforce max active zones limit
    while (zonesToLoad.length > this.config.maxActiveZones) {
      zonesToLoad.pop();
    }

    // Queue loads
    for (const zoneId of zonesToLoad) {
      if (!this.activeZones.has(zoneId) && !this.loadingZones.has(zoneId)) {
        this.queueLoad(zoneId);
      }
    }

    // Queue unloads
    for (const zoneId of zonesToUnload) {
      const distance = this.playerPosition.distanceTo(
        this.zones.get(zoneId).position
      );

      if (distance >= this.config.unloadDistance) {
        this.queueUnload(zoneId);
      }
    }
  }

  /**
   * Check which zone the player is in
   */
  checkZones() {
    let playerZone = null;

    for (const [zoneId, bounds] of this.zoneBounds) {
      if (bounds.containsPoint(this.playerPosition)) {
        playerZone = zoneId;
        break;
      }
    }

    if (playerZone !== this.currentZone) {
      this.previousZone = this.currentZone;
      this.currentZone = playerZone;

      // Emit zone change event
      this.gameManager.emit('zone:changed', {
        previous: this.previousZone,
        current: this.currentZone
      });

      this.logger.log(`Zone changed: ${this.previousZone} -> ${this.currentZone}`);
    }
  }

  /**
   * Queue a zone for loading
   */
  queueLoad(zoneId) {
    const zone = this.zones.get(zoneId);
    if (!zone) return;

    this.loadQueue.push({
      action: 'load',
      zoneId,
      priority: this.calculateLoadPriority(zoneId),
      zone
    });

    this.processQueue();
  }

  /**
   * Queue a zone for unloading
   */
  queueUnload(zoneId) {
    if (!this.activeZones.has(zoneId)) return;

    // Delay unload for smoother experience
    const timer = setTimeout(() => {
      this.loadQueue.push({
        action: 'unload',
        zoneId,
        priority: 0  // Unload is low priority
      });

      this.unloadTimers.delete(zoneId);
      this.processQueue();
    }, this.config.unloadDelay);

    this.unloadTimers.set(zoneId, timer);
  }

  /**
   * Calculate loading priority
   */
  calculateLoadPriority(zoneId) {
    const zone = this.zones.get(zoneId);
    let priority = 0;

    // Distance: closer = higher priority
    const distance = this.playerPosition.distanceTo(zone.position);
    priority += Math.max(0, 100 - distance);

    // In current path: higher priority
    if (this.currentZone && zone.connections.includes(this.currentZone)) {
      priority += 50;
    }

    // Explicit priority from zone data
    if (zone.priority) {
      priority += zone.priority;
    }

    // Quest objective: higher priority
    if (this.isQuestObjective(zoneId)) {
      priority += 100;
    }

    return priority;
  }

  /**
   * Check if zone is a quest objective
   */
  isQuestObjective(zoneId) {
    // Check with quest manager if available
    if (this.gameManager.questManager) {
      return this.gameManager.questManager.isZoneObjective(zoneId);
    }
    return false;
  }

  /**
   * Process the load/unload queue
   */
  async processQueue() {
    if (this.isProcessingQueue || this.loadQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;

    // Sort by priority
    this.loadQueue.sort((a, b) => b.priority - a.priority);

    const task = this.loadQueue.shift();

    try {
      if (task.action === 'load') {
        await this.loadZone(task.zoneId);
      } else if (task.action === 'unload') {
        await this.unloadZone(task.zoneId);
      }
    } catch (error) {
      this.logger.error(`Zone task failed:`, error);
    }

    this.isProcessingQueue = false;

    // Process next if any
    if (this.loadQueue.length > 0) {
      this.processQueue();
    }
  }

  /**
   * Load a zone
   */
  async loadZone(zoneId) {
    if (this.activeZones.has(zoneId) || this.loadingZones.has(zoneId)) {
      return;
    }

    const zone = this.zones.get(zoneId);
    if (!zone) {
      this.logger.warn(`Zone "${zoneId}" not found`);
      return null;
    }

    this.logger.log(`Loading zone: ${zoneId}`);
    this.loadingZones.add(zoneId);

    const startTime = performance.now();
    let memoryUsed = 0;

    try {
      // Load assets
      const loadPromises = [];

      // Models
      if (zone.assets.models) {
        for (const modelPath of zone.assets.models) {
          loadPromises.push(
            this.assetLoader.loadModel(modelPath).then(model => {
              memoryUsed += this.estimateModelMemory(model);
              return { type: 'model', data: model, path: modelPath };
            })
          );
        }
      }

      // Textures
      if (zone.assets.textures) {
        for (const texturePath of zone.assets.textures) {
          loadPromises.push(
            this.assetLoader.loadTexture(texturePath).then(texture => {
              memoryUsed += this.estimateTextureMemory(texture);
              return { type: 'texture', data: texture, path: texturePath };
            })
          );
        }
      }

      // Audio
      if (zone.assets.audio) {
        for (const audioPath of zone.assets.audio) {
          loadPromises.push(
            this.assetLoader.loadAudio(audioPath).then(audio => {
              memoryUsed += this.estimateAudioMemory(audio);
              return { type: 'audio', data: audio, path: audioPath };
            })
          );
        }
      }

      // Gaussian Splats
      if (zone.assets.splats) {
        for (const splatPath of zone.assets.splats) {
          loadPromises.push(
            this.assetLoader.loadSplat(splatPath).then(splat => {
              memoryUsed += this.estimateSplatMemory(splat);
              return { type: 'splat', data: splat, path: splatPath };
            })
          );
        }
      }

      // Wait for all assets to load
      const loadedAssets = await Promise.all(loadPromises);

      // Add to scene
      this.addZoneToScene(zoneId, loadedAssets);

      // Update memory tracking
      this.memoryUsage.set(zoneId, memoryUsed);
      this.totalMemoryUsed += memoryUsed;

      // Mark as active
      this.activeZones.add(zoneId);
      this.loadingZones.delete(zoneId);

      const loadTime = performance.now() - startTime;
      this.logger.log(`Zone loaded: ${zoneId} (${loadTime.toFixed(0)}ms, ${(memoryUsed / 1024 / 1024).toFixed(1)}MB)`);

      // Emit event
      this.gameManager.emit('zone:loaded', {
        zoneId,
        loadTime,
        memoryUsed
      });

      // Call callback
      if (this.onZoneLoaded) {
        this.onZoneLoaded(zoneId, loadedAssets);
      }

      // Check memory budget
      this.checkMemoryBudget();

      return { zoneId, loadedAssets };

    } catch (error) {
      this.loadingZones.delete(zoneId);
      this.logger.error(`Failed to load zone "${zoneId}":`, error);
      throw error;
    }
  }

  /**
   * Add loaded assets to scene
   */
  addZoneToScene(zoneId, assets) {
    const zoneGroup = new THREE.Group();
    zoneGroup.name = `zone_${zoneId}`;

    for (const asset of assets) {
      switch (asset.type) {
        case 'model':
          zoneGroup.add(asset.data);
          break;

        case 'splat':
          // Splats are handled by SparkRenderer
          if (this.sceneManager.sparkRenderer) {
            this.sceneManager.sparkRenderer.addSplat(asset.data, zoneId);
          }
          break;

        case 'texture':
        case 'audio':
          // These are managed by their respective managers
          break;
      }
    }

    this.sceneManager.scene.add(zoneGroup);
  }

  /**
   * Unload a zone
   */
  async unloadZone(zoneId) {
    if (!this.activeZones.has(zoneId)) {
      return;
    }

    // Cancel any pending unload timer
    const timer = this.unloadTimers.get(zoneId);
    if (timer) {
      clearTimeout(timer);
      this.unloadTimers.delete(zoneId);
    }

    this.logger.log(`Unloading zone: ${zoneId}`);
    this.unloadingZones.add(zoneId);

    try {
      // Remove from scene
      const zoneGroup = this.sceneManager.scene.getObjectByName(`zone_${zoneId}`);
      if (zoneGroup) {
        this.sceneManager.scene.remove(zoneGroup);

        // Dispose resources
        zoneGroup.traverse((object) => {
          if (object.geometry) {
            object.geometry.dispose();
          }
          if (object.material) {
            if (Array.isArray(object.material)) {
              object.material.forEach(m => m.dispose());
            } else {
              object.material.dispose();
            }
          }
        });
      }

      // Unload splats
      if (this.sceneManager.sparkRenderer) {
        this.sceneManager.sparkRenderer.removeZoneSplats(zoneId);
      }

      // Unload assets from cache
      const zone = this.zones.get(zoneId);
      if (zone && zone.assets) {
        if (zone.assets.models) {
          for (const path of zone.assets.models) {
            this.assetLoader.unloadModel(path);
          }
        }
        if (zone.assets.textures) {
          for (const path of zone.assets.textures) {
            this.assetLoader.unloadTexture(path);
          }
        }
        if (zone.assets.audio) {
          for (const path of zone.assets.audio) {
            this.assetLoader.unloadAudio(path);
          }
        }
      }

      // Update memory tracking
      const memoryFreed = this.memoryUsage.get(zoneId) || 0;
      this.totalMemoryUsed -= memoryFreed;
      this.memoryUsage.delete(zoneId);

      // Mark as inactive
      this.activeZones.delete(zoneId);
      this.unloadingZones.delete(zoneId);

      this.logger.log(`Zone unloaded: ${zoneId} (${(memoryFreed / 1024 / 1024).toFixed(1)}MB freed)`);

      // Emit event
      this.gameManager.emit('zone:unloaded', { zoneId, memoryFreed });

      // Call callback
      if (this.onZoneUnloaded) {
        this.onZoneUnloaded(zoneId);
      }

    } catch (error) {
      this.unloadingZones.delete(zoneId);
      this.logger.error(`Failed to unload zone "${zoneId}":`, error);
      throw error;
    }
  }

  /**
   * Check if over memory budget
   */
  checkMemoryBudget() {
    if (this.totalMemoryUsed > this.config.memoryBudget) {
      this.logger.warn(`Over memory budget: ${this.totalMemoryUsed} / ${this.config.memoryBudget}`);

      // Unload least recently used zones
      const zonesByDistance = Array.from(this.activeZones)
        .map(zoneId => ({
          zoneId,
          distance: this.playerPosition.distanceTo(this.zones.get(zoneId).position)
        }))
        .sort((a, b) => b.distance - a.distance);

      for (const { zoneId } of zonesByDistance) {
        if (this.totalMemoryUsed <= this.config.memoryBudget * 0.8) {
          break;
        }
        if (zoneId !== this.currentZone) {
          this.unloadZone(zoneId);
        }
      }
    }
  }

  /**
   * Estimate memory usage for different asset types
   */
  estimateModelMemory(model) {
    // Rough estimate based on geometry
    let triangles = 0;
    model.traverse((child) => {
      if (child.geometry) {
        triangles += child.geometry.index ? child.geometry.index.count / 3 : child.geometry.attributes.position.count / 3;
      }
    });

    // Assume ~100 bytes per triangle
    return triangles * 100;
  }

  estimateTextureMemory(texture) {
    // Width * Height * 4 bytes (RGBA) * mipmap levels
    const size = texture.image.width * texture.image.height * 4;
    const mipmaps = Math.log2(Math.min(texture.image.width, texture.image.height)) + 1;
    return size * mipmaps * 1.33;  // 1.33x for mipmap overhead
  }

  estimateAudioMemory(audio) {
    // Duration * sample rate * channels * bytes per sample
    // Assuming standard format
    return audio.duration * 44100 * 2 * 2;  // 2 channels, 2 bytes
  }

  estimateSplatMemory(splat) {
    // Splat count * parameters * bytes per parameter
    const splatCount = splat.numSplats || 100000;
    return splatCount * 48;  // 48 bytes per splat (position, color, covariance)
  }

  /**
   * Get currently active zones
   */
  getActiveZones() {
    return Array.from(this.activeZones);
  }

  /**
   * Get zone at position
   */
  getZoneAt(position) {
    for (const [zoneId, bounds] of this.zoneBounds) {
      if (bounds.containsPoint(position)) {
        return this.zones.get(zoneId);
      }
    }
    return null;
  }

  /**
   * Get current zone
   */
  getCurrentZone() {
    return this.currentZone ? this.zones.get(this.currentZone) : null;
  }

  /**
   * Force load a zone (for immediate needs)
   */
  async forceLoadZone(zoneId) {
    // Cancel any unload timer
    const timer = this.unloadTimers.get(zoneId);
    if (timer) {
      clearTimeout(timer);
      this.unloadTimers.delete(zoneId);
    }

    return this.loadZone(zoneId);
  }

  /**
   * Clean up all zones
   */
  async unloadAll() {
    const zonesToUnload = Array.from(this.activeZones);
    await Promise.all(zonesToUnload.map(zoneId => this.unloadZone(zoneId)));
  }

  /**
   * Get memory usage statistics
   */
  getMemoryStats() {
    return {
      totalUsed: this.totalMemoryUsed,
      budget: this.config.memoryBudget,
      usagePercent: (this.totalMemoryUsed / this.config.memoryBudget) * 100,
      byZone: Object.fromEntries(this.memoryUsage),
      activeZones: this.activeZones.size
    };
  }

  /**
   * Destroy the zone manager
   */
  destroy() {
    this.stopStreaming();
    this.unloadAll();
  }
}

export default ZoneManager;
```

---

## 📝 How To Implement Zone Loading

### Step 1: Divide Your World Into Zones

```javascript
// Define zones as bounding boxes
const zones = [
  {
    id: 'plaza',
    name: 'Central Plaza',
    position: new THREE.Vector3(0, 0, 0),
    size: new THREE.Vector3(100, 50, 100),
    assets: {
      models: ['models/plaza.glb'],
      splats: ['splats/plaza.splat'],
      audio: ['audio/plaza_ambience.mp3']
    }
  },
  {
    id: 'street',
    name: 'Main Street',
    position: new THREE.Vector3(100, 0, 0),
    size: new THREE.Vector3(150, 50, 50),
    connections: ['plaza'],
    assets: { /* ... */ }
  }
];

zones.forEach(zone => zoneManager.registerZone(zone));
```

### Step 2: Enable Streaming

```javascript
const zoneManager = new ZoneManager({
  sceneManager,
  assetLoader,
  preloadDistance: 50,   // Start loading 50 units away
  unloadDistance: 80,    // Unload 80 units away
  maxActiveZones: 3,     // Keep 3 zones max
  memoryBudget: 500 * 1024 * 1024  // 500MB
});
```

### Step 3: Handle Zone Transitions

```javascript
gameManager.on('zone:changed', ({ previous, current }) => {
  if (previous !== current) {
    // Trigger zone-specific events
    onEnterZone(current);
    onLeaveZone(previous);
  }
});
```

---

## Common Mistakes Beginners Make

### 1. Loading Everything at Start

```javascript
// ❌ WRONG: Load all zones upfront
async function init() {
  for (const zone of zones) {
    await loadZone(zone);
  }
}
// Takes forever, uses all memory

// ✅ CORRECT: Load only starting zone
async function init() {
  await loadZone(startZone);
  zoneManager.enableStreaming();
}
// Fast startup, loads as you go
```

### 2. No Preload Buffer

```javascript
// ❌ WRONG: Load only current zone
if (distance < zoneRadius) {
  loadZone(zone);
}
// Player sees loading hiccups

// ✅ CORRECT: Preload nearby zones
if (distance < preloadDistance) {
  preloadZone(zone);
}
// Seamless transitions
```

### 3. Forgetting to Unload

```javascript
// ❌ WRONG: Never unload
function loadZone(zone) {
  zones.add(zone);
}
// Memory leak

// ✅ CORRECT: Unload far zones
function update() {
  for (const zone of activeZones) {
    if (distanceTo(zone) > unloadDistance) {
      unloadZone(zone);
    }
  }
}
// Memory stays manageable
```

---

## Performance Considerations

```
ZONE LOADING PERFORMANCE:

Memory Impact:
├── Without zone loading: All assets in memory
├── With zone loading: Only ~20% of assets at once
└── Savings: 80% memory reduction

Loading Impact:
├── Initial load: Single zone vs all zones
├── Time: 2-5 seconds vs 30+ seconds
└── Player experience: Much faster start

Streaming Overhead:
├── CPU: Periodic distance checks (negligible)
├── I/O: Background loading (async)
└── Impact: Minimal when done right

Optimization:
├── Use coarse bounds for distance checks
├── Prioritize zones in player's path
├── Delay unloading for backtracking
└── Group assets by zone for efficient unloading
```

---

## Related Systems

- [Performance Profiles](./performance-profiles.md) - Platform-based settings
- [Memory Management](./memory-management.md) - Resource tracking
- [Platform Detection](./platform-detection.md) - Hardware capabilities

---

## Source File Reference

**Primary Files**:
- `../src/managers/ZoneManager.js` - Zone loading system (estimated)

**Key Classes**:
- `ZoneManager` - Dynamic loading/unloading

**Dependencies**:
- THREE.Box3 (bounds checking)
- THREE.Vector3 (position tracking)
- AssetLoader (async loading)

---

## References

- [Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API) - Background loading
- [Performance API](https://developer.mozilla.org/en-US/docs/Web/API/Performance) - Monitoring
- [Memory API](https://developer.mozilla.org/en-US/docs/Web/API/DeviceMemory_API) - Memory info

*Documentation last updated: January 12, 2026*
