# Memory Management - First Principles Guide

## Overview

**Memory Management** is the practice of efficiently allocating, tracking, and releasing memory resources. In a browser-based 3D game, poor memory management leads to crashes, freezes, and sluggish performance. Good memory management ensures the game runs smoothly within the browser's memory limits while providing the best possible visual quality.

Think of memory management like the **"warehouse manager"**—like a warehouse with limited shelf space, you must track what's stored, remove what's no longer needed, and organize efficiently to fit everything without running out of room.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create a game that never crashes or stutters due to memory issues. Players should be immersed in the experience without technical interruptions like "out of memory" errors or sudden frame drops when garbage collection happens.

**Why Memory Management Matters?**
- **Browser Limits**: Tabs share memory, browser kills tabs that use too much
- **Mobile Constraints**: Phones have much less memory than desktops
- **Smooth Performance**: Garbage collection pauses cause frame drops
- **Asset Quality**: More memory available = higher quality assets
- **Session Length**: Good memory management enables longer play sessions

**Memory Hierarchy of Needs**:
```
CRITICAL (Must Fit)
├── Core engine code
├── Current zone assets
├── Player character
└── Active gameplay systems

IMPORTANT (Keep Loaded)
├── Adjacent zone assets
├── Common UI elements
├── Audio engine
└── Input handling

NICE TO HAVE (Load When Possible)
├── Distant zones
├── High-res textures
├── Cached sounds
└── Particle effects

DISCARDABLE (Unload First)
├── Left-behind zones
├── Unused textures
├── Finished audio
└── Completed effects
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding memory management, you should know:
- **JavaScript memory model** - Heap vs stack, garbage collection
- **Three.js disposal** - geometry.dispose(), material.dispose()
- **Texture memory** - How GPU memory works
- **Audio buffers** - How audio data is stored
- **Weak references** - For caching without preventing GC

### Core Architecture

```
MEMORY MANAGEMENT ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                    MEMORY MANAGER                       │
│  - Track allocations                                    │
│  - Monitor usage                                        │
│  - Enforce budgets                                      │
│  - Trigger cleanup                                      │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   TRACKING    │  │   ALLOCATIONS │  │   CLEANUP    │
│  - Resources  │  │  - Geometry   │  │  - Disposal   │
│  - Zones      │  │  - Textures   │  │  - Unloading  │
│  - Audio      │  │  - Audio      │  │  - Cache purge│
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   BUDGETS    │
                    │  - Per type   │
                    │  - Per zone   │
                    │  - Total cap  │
                    └───────────────┘
```

### MemoryManager Class

```javascript
class MemoryManager {
  constructor(options = {}) {
    this.gameManager = options.gameManager;
    this.logger = options.logger || console;

    // Memory budgets (in bytes)
    this.budgets = {
      total: options.totalBudget || this.getDefaultBudget(),
      geometry: options.geometryBudget || 100 * 1024 * 1024,      // 100MB
      textures: options.textureBudget || 200 * 1024 * 1024,      // 200MB
      audio: options.audioBudget || 50 * 1024 * 1024,           // 50MB
      splats: options.splatBudget || 300 * 1024 * 1024,         // 300MB
      other: options.otherBudget || 50 * 1024 * 1024            // 50MB
    };

    // Current usage tracking
    this.usage = {
      geometry: new Map(),   // id -> { size, refCount }
      textures: new Map(),   // id -> { size, refCount }
      audio: new Map(),      // id -> { size, refCount }
      splats: new Map(),     // id -> { size, refCount }
      zones: new Map()       // zoneId -> total size
    };

    // Resource registry
    this.resources = {
      geometries: new Map(),   // id -> THREE.Geometry
      materials: new Map(),    // id -> THREE.Material
      textures: new Map(),     // id -> THREE.Texture
      audio: new Map(),        // id -> AudioBuffer
      splats: new Map()        // id -> SplatData
    };

    // Cache for reusable resources
    this.cache = new Map();
    this.cacheMaxAge = options.cacheMaxAge || 300000;  // 5 minutes

    // Monitoring
    this.isMonitoring = false;
    this.monitoringInterval = null;
    this.warnThreshold = 0.8;  // Warn at 80% usage

    // Cleanup settings
    this.autoCleanup = options.autoCleanup !== false;
    this.cleanupInterval = options.cleanupInterval || 60000;  // 1 minute

    // Initialize
    this.initialize();
  }

  /**
   * Get default memory budget based on platform
   */
  getDefaultBudget() {
    // Check device memory (limited browser support)
    const deviceMemory = navigator.deviceMemory || 4;  // GB

    // Assume we can use ~50% of available RAM
    // (browser, other tabs, OS need the rest)
    const budgetBytes = (deviceMemory * 1024 * 1024 * 1024) * 0.5;

    // Cap based on device category
    if (this.isMobile()) {
      return Math.min(budgetBytes, 400 * 1024 * 1024);  // Max 400MB on mobile
    } else {
      return Math.min(budgetBytes, 2000 * 1024 * 1024); // Max 2GB on desktop
    }
  }

  /**
   * Check if mobile device
   */
  isMobile() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
           ('ontouchstart' in window);
  }

  /**
   * Initialize memory manager
   */
  initialize() {
    // Start monitoring
    this.startMonitoring();

    // Start periodic cleanup
    if (this.autoCleanup) {
      this.startCleanup();
    }

    // Log initial budget
    this.logger.log('MemoryManager initialized:', {
      budgets: {
        total: `${(this.budgets.total / 1024 / 1024).toFixed(0)}MB`,
        geometry: `${(this.budgets.geometry / 1024 / 1024).toFixed(0)}MB`,
        textures: `${(this.budgets.textures / 1024 / 1024).toFixed(0)}MB`,
        audio: `${(this.budgets.audio / 1024 / 1024).toFixed(0)}MB`,
        splats: `${(this.budgets.splats / 1024 / 1024).toFixed(0)}MB`
      }
    });
  }

  /**
   * Register a resource
   */
  registerResource(type, id, resource, options = {}) {
    const category = this.getCategory(type);
    if (!category) return;

    // Calculate size
    const size = options.size || this.estimateSize(type, resource);

    // Check budget
    const currentUsage = this.getCategoryUsage(category);
    if (currentUsage + size > this.budgets[category]) {
      this.logger.warn(`Memory budget exceeded for ${category}: ${(currentUsage / 1024 / 1024).toFixed(1)}MB + ${(size / 1024 / 1024).toFixed(1)}MB > ${(this.budgets[category] / 1024 / 1024).toFixed(1)}MB`);

      // Try to free memory
      this.freeMemory(category, size * 1.5);  // Free extra buffer
    }

    // Store resource
    this.resources[category].set(id, {
      resource,
      size,
      type,
      refCount: options.refCount || 1,
      lastAccess: Date.now(),
      persistent: options.persistent || false,
      category
    });

    // Track usage
    this.usage[category].set(id, { size, refCount: options.refCount || 1 });

    this.logger.log(`Registered ${type}: ${id} (${(size / 1024).toFixed(0)}KB)`);

    return resource;
  }

  /**
   * Unregister (dispose) a resource
   */
  unregisterResource(type, id) {
    const category = this.getCategory(type);
    if (!category) return;

    const resourceData = this.resources[category].get(id);
    if (!resourceData) {
      this.logger.warn(`Resource not found: ${type}/${id}`);
      return;
    }

    // Decrease ref count
    resourceData.refCount--;

    // Dispose if no more references
    if (resourceData.refCount <= 0) {
      this.disposeResource(category, id);
    }
  }

  /**
   * Actually dispose of a resource
   */
  disposeResource(category, id) {
    const resourceData = this.resources[category].get(id);
    if (!resourceData) return;

    const { resource, type, size } = resourceData;

    // Type-specific disposal
    switch (type) {
      case 'geometry':
      case 'mesh':
        if (resource.geometry) {
          resource.geometry.dispose();
        }
        if (resource.material) {
          if (Array.isArray(resource.material)) {
            resource.material.forEach(m => this.disposeMaterial(m));
          } else {
            this.disposeMaterial(resource.material);
          }
        }
        break;

      case 'texture':
        resource.dispose();
        break;

      case 'material':
        this.disposeMaterial(resource);
        break;

      case 'audio':
        // Audio buffers are released by closing the source
        break;

      case 'splat':
        // Splat data cleanup
        if (resource.dispose) {
          resource.dispose();
        }
        break;
    }

    // Remove from tracking
    this.resources[category].delete(id);
    this.usage[category].delete(id);

    this.logger.log(`Disposed ${type}: ${id} (${(size / 1024).toFixed(0)}KB freed)`);
  }

  /**
   * Dispose material and its textures
   */
  disposeMaterial(material) {
    // Dispose maps
    if (material.map) material.map.dispose();
    if (material.normalMap) material.normalMap.dispose();
    if (material.roughnessMap) material.roughnessMap.dispose();
    if (material.metalnessMap) material.metalnessMap.dispose();
    if (material.emissiveMap) material.emissiveMap.dispose();
    if (material.aoMap) material.aoMap.dispose();

    // Dispose material itself
    material.dispose();
  }

  /**
   * Get a resource (with reference counting)
   */
  getResource(type, id) {
    const category = this.getCategory(type);
    if (!category) return null;

    const resourceData = this.resources[category].get(id);
    if (!resourceData) return null;

    // Update last access
    resourceData.lastAccess = Date.now();
    resourceData.refCount++;

    return resourceData.resource;
  }

  /**
   * Cache a resource for potential reuse
   */
  cacheResource(key, resource, size) {
    this.cache.set(key, {
      resource,
      size,
      cached: Date.now()
    });
  }

  /**
   * Get cached resource
   */
  getCached(key) {
    const cached = this.cache.get(key);
    if (cached) {
      cached.lastAccess = Date.now();
      return cached.resource;
    }
    return null;
  }

  /**
   * Clear old cache entries
   */
  clearExpiredCache() {
    const now = Date.now();
    let freed = 0;

    for (const [key, value] of this.cache.entries()) {
      if (now - value.cached > this.cacheMaxAge) {
        this.cache.delete(key);
        freed += value.size;
      }
    }

    if (freed > 0) {
      this.logger.log(`Cache cleared: ${(freed / 1024).toFixed(0)}KB freed`);
    }
  }

  /**
   * Clear entire cache
   */
  clearCache() {
    let freed = 0;
    for (const [key, value] of this.cache.entries()) {
      freed += value.size;
    }
    this.cache.clear();
    this.logger.log(`Cache cleared: ${(freed / 1024 / 1024).toFixed(1)}MB freed`);
  }

  /**
   * Get category for resource type
   */
  getCategory(type) {
    const categories = {
      geometry: 'geometry',
      mesh: 'geometry',
      bufferGeometry: 'geometry',
      texture: 'textures',
      image: 'textures',
      canvas: 'textures',
      material: 'other',
      audio: 'audio',
      sound: 'audio',
      splat: 'splats',
      gaussianSplat: 'splats'
    };

    return categories[type] || 'other';
  }

  /**
   * Estimate resource size
   */
  estimateSize(type, resource) {
    switch (type) {
      case 'geometry':
      case 'mesh':
      case 'bufferGeometry':
        return this.estimateGeometrySize(resource);

      case 'texture':
      case 'image':
        return this.estimateTextureSize(resource);

      case 'audio':
      case 'sound':
        return this.estimateAudioSize(resource);

      case 'splat':
      case 'gaussianSplat':
        return this.estimateSplatSize(resource);

      default:
        return 0;
    }
  }

  /**
   * Estimate geometry memory size
   */
  estimateGeometrySize(geometry) {
    if (!geometry) return 0;

    let size = 0;

    // Position attribute (3 floats per vertex)
    if (geometry.attributes.position) {
      size += geometry.attributes.position.count * 3 * 4;
    }

    // Normal attribute (3 floats per vertex)
    if (geometry.attributes.normal) {
      size += geometry.attributes.normal.count * 3 * 4;
    }

    // UV attribute (2 floats per vertex)
    if (geometry.attributes.uv) {
      size += geometry.attributes.uv.count * 2 * 4;
    }

    // Index buffer (if present)
    if (geometry.index) {
      size += geometry.index.count * 4;
    }

    return size;
  }

  /**
   * Estimate texture memory size
   */
  estimateTextureSize(texture) {
    if (!texture || !texture.image) return 0;

    const image = texture.image;
    const width = image.width || image.videoWidth;
    const height = image.height || image.videoHeight;

    // Base texture size (RGBA = 4 bytes per pixel)
    let size = width * height * 4;

    // Mipmaps add ~33% more
    size *= 1.33;

    return size;
  }

  /**
   * Estimate audio buffer size
   */
  estimateAudioSize(audio) {
    if (!audio) return 0;

    // Duration * sample rate * channels * bytes per sample
    const duration = audio.duration || 0;
    const sampleRate = audio.sampleRate || 44100;
    const channels = audio.numberOfChannels || 2;

    return duration * sampleRate * channels * 2;  // 2 bytes per sample (16-bit)
  }

  /**
   * Estimate Gaussian Splat size
   */
  estimateSplatSize(splat) {
    if (!splat) return 0;

    // Each splat has:
    // - Position: 3 floats (12 bytes)
    // - Color/Scale: 6 floats (24 bytes)
    // - Covariance: 6 floats (24 bytes)
    // Total: ~48 bytes per splat

    const numSplats = splat.numSplats || splat.count || 100000;
    return numSplats * 48;
  }

  /**
   * Get usage for a category
   */
  getCategoryUsage(category) {
    let total = 0;
    for (const [id, data] of this.usage[category]) {
      total += data.size;
    }
    return total;
  }

  /**
   * Get total memory usage
   */
  getTotalUsage() {
    let total = 0;
    for (const category of Object.keys(this.usage)) {
      total += this.getCategoryUsage(category);
    }
    return total;
  }

  /**
   * Get memory statistics
   */
  getStats() {
    const stats = {
      budgets: {},
      usage: {},
      percent: {},
      resources: {},
      total: { used: 0, budget: this.budgets.total, percent: 0 }
    };

    for (const category of Object.keys(this.budgets)) {
      if (category === 'total') continue;

      const used = this.getCategoryUsage(category);
      const budget = this.budgets[category];
      const percent = (used / budget) * 100;

      stats.budgets[category] = budget;
      stats.usage[category] = used;
      stats.percent[category] = percent;
      stats.resources[category] = this.usage[category].size;

      stats.total.used += used;
    }

    stats.total.percent = (stats.total.used / stats.total.budget) * 100;

    return stats;
  }

  /**
   * Start monitoring memory
   */
  startMonitoring() {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    this.monitoringInterval = setInterval(() => {
      this.checkMemory();
    }, 5000);  // Check every 5 seconds
  }

  /**
   * Stop monitoring
   */
  stopMonitoring() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    this.isMonitoring = false;
  }

  /**
   * Check memory usage and warn if needed
   */
  checkMemory() {
    const stats = this.getStats();

    // Check total usage
    if (stats.total.percent > this.warnThreshold * 100) {
      this.logger.warn(`Memory usage high: ${stats.total.percent.toFixed(0)}% (${(stats.total.used / 1024 / 1024).toFixed(0)}MB / ${(stats.total.budget / 1024 / 1024).toFixed(0)}MB)`);

      // Trigger cleanup
      if (this.autoCleanup) {
        this.cleanup();
      }
    }

    // Check per-category usage
    for (const category of Object.keys(this.usage)) {
      const used = this.getCategoryUsage(category);
      const budget = this.budgets[category];
      const percent = (used / budget) * 100;

      if (percent > this.warnThreshold * 100) {
        this.logger.warn(`${category} memory high: ${percent.toFixed(0)}%`);
      }
    }

    // Emit stats event
    this.gameManager.emit('memory:stats', stats);
  }

  /**
   * Free memory from a category
   */
  freeMemory(category, amount) {
    let freed = 0;
    const toFree = [];

    // Collect resources that can be freed
    for (const [id, resourceData] of this.resources[category]) {
      if (freed >= amount) break;

      // Skip persistent resources
      if (resourceData.persistent) continue;

      // Skip recently accessed (within last minute)
      if (Date.now() - resourceData.lastAccess < 60000) continue;

      // Skip if ref count > 0
      if (resourceData.refCount > 0) continue;

      toFree.push(id);
      freed += resourceData.size;
    }

    // Dispose resources
    for (const id of toFree) {
      this.disposeResource(category, id);
    }

    if (freed > 0) {
      this.logger.log(`Freed ${(freed / 1024 / 1024).toFixed(1)}MB from ${category}`);
    }

    return freed;
  }

  /**
   * Start automatic cleanup
   */
  startCleanup() {
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, this.cleanupInterval);
  }

  /**
   * Stop automatic cleanup
   */
  stopCleanup() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
  }

  /**
   * Perform cleanup of unused resources
   */
  cleanup() {
    this.logger.log('Starting memory cleanup...');

    let totalFreed = 0;

    // Clear expired cache
    const beforeCache = this.cache.size;
    this.clearExpiredCache();

    // Free resources from each category
    for (const category of Object.keys(this.usage)) {
      if (category === 'zones') continue;

      // Free resources with no references
      const toFree = [];
      for (const [id, data] of this.usage[category]) {
        const resourceData = this.resources[category].get(id);
        if (resourceData && !resourceData.persistent && resourceData.refCount <= 0) {
          toFree.push(id);
        }
      }

      for (const id of toFree) {
        const resourceData = this.resources[category].get(id);
        if (resourceData) {
          totalFreed += resourceData.size;
          this.disposeResource(category, id);
        }
      }
    }

    // Suggest garbage collection (hint to browser)
    if (window.gc) {
      window.gc();
    }

    if (totalFreed > 0) {
      this.logger.log(`Cleanup complete: ${(totalFreed / 1024 / 1024).toFixed(1)}MB freed`);
    } else {
      this.logger.log('Cleanup complete: nothing to free');
    }

    return totalFreed;
  }

  /**
   * Set memory budget for a category
   */
  setBudget(category, amount) {
    this.budgets[category] = amount;
    this.logger.log(`Budget set: ${category} = ${(amount / 1024 / 1024).toFixed(0)}MB`);
  }

  /**
   * Get memory budget for a category
   */
  getBudget(category) {
    return this.budgets[category] || 0;
  }

  /**
   * Create a memory report
   */
  generateReport() {
    const stats = this.getStats();

    let report = '=== Memory Report ===\n\n';

    report += 'Budgets:\n';
    for (const category of Object.keys(stats.budgets)) {
      if (category === 'total') continue;
      report += `  ${category}: ${(stats.usage[category] / 1024 / 1024).toFixed(1)}MB / ${(stats.budgets[category] / 1024 / 1024).toFixed(0)}MB (${stats.percent[category].toFixed(0)}%)\n`;
    }

    report += `\nTotal: ${(stats.total.used / 1024 / 1024).toFixed(1)}MB / ${(stats.total.budget / 1024 / 1024).toFixed(0)}MB (${stats.total.percent.toFixed(0)}%)\n`;

    report += `\nResources by category:\n`;
    for (const category of Object.keys(this.usage)) {
      report += `  ${category}: ${this.usage[category].size} items\n`;
    }

    report += `\nCached items: ${this.cache.size}\n`;

    return report;
  }

  /**
   * Destroy memory manager and clean up everything
   */
  destroy() {
    this.stopMonitoring();
    this.stopCleanup();

    // Dispose all resources
    for (const category of Object.keys(this.resources)) {
      for (const [id, resourceData] of this.resources[category]) {
        this.disposeResource(category, id);
      }
    }

    // Clear cache
    this.cache.clear();

    this.logger.log('MemoryManager destroyed');
  }
}

export default MemoryManager;
```

---

## 📝 How To Implement Memory Management

### Step 1: Track Your Resources

```javascript
const memoryManager = new MemoryManager();

// Register loaded assets
const texture = await loadTexture('diffuse.jpg');
memoryManager.registerResource('texture', 'diffuse.jpg', texture);
```

### Step 2: Dispose When Done

```javascript
// Unregister when done
memoryManager.unregisterResource('texture', 'diffuse.jpg');

// Or explicitly dispose
memoryManager.disposeResource('textures', 'diffuse.jpg');
```

### Step 3: Monitor Memory

```javascript
// Get stats
const stats = memoryManager.getStats();
console.log(`Memory: ${(stats.total.used / 1024 / 1024).toFixed(1)}MB used`);

// Listen for warnings
gameManager.on('memory:stats', (stats) => {
  if (stats.total.percent > 80) {
    reduceQuality();
  }
});
```

---

## Common Mistakes Beginners Make

### 1. Never Disposing Three.js Objects

```javascript
// ❌ WRONG: Just remove from scene
scene.remove(mesh);
// Memory leak

// ✅ CORRECT: Dispose properly
scene.remove(mesh);
mesh.geometry.dispose();
mesh.material.dispose();
// Memory freed
```

### 2. Not Clearing Event Listeners

```javascript
// ❌ WRONG: Never remove listeners
object.addEventListener('event', handler);
// Memory leak (object kept alive)

// ✅ CORRECT: Remove when done
object.addEventListener('event', handler);
// Later:
object.removeEventListener('event', handler);
// Proper cleanup
```

### 3. Loading Everything at Once

```javascript
// ❌ WRONG: Load all assets
const allTextures = await Promise.all(textureList.map(load));
// Out of memory on large projects

// ✅ CORRECT: Load on demand
const texture = await loadTexture(neededId);
memoryManager.registerResource('texture', neededId, texture);
// Manageable memory
```

---

## Browser Memory Limits

```
TYPICAL MEMORY LIMITS:

Desktop Chrome:
├── ~2-4GB per tab (64-bit)
├── Can exceed with user gesture
└── Subject to system RAM

Desktop Firefox:
├── ~2-4GB per tab
├── More aggressive GC
└── Better with many tabs

Desktop Safari:
├── ~1-2GB per tab
├── Stricter limits
└── Harder to exceed limits

Mobile Chrome:
├── ~200-500MB per tab
├── Kills tab if exceeded
└── Very strict limits

Mobile Safari:
├── ~400-800MB per tab (iOS)
├── Aggressive tab reloading
└── Will reload page on memory pressure
```

---

## Related Systems

- [Performance Profiles](./performance-profiles.md) - Memory budgets by profile
- [Zone Loading](./zone-loading-optimization.md) - Dynamic memory management
- [Platform Detection](./platform-detection.md) - Hardware capabilities

---

## Source File Reference

**Primary Files**:
- `../src/managers/MemoryManager.js` - Memory management (estimated)

**Key Classes**:
- `MemoryManager` - Resource tracking and disposal

**Dependencies**:
- THREE.js (geometry/material/texture disposal)
- GameManager (event emission)
- Device Memory API (budget detection)

---

## References

- [MDN: Memory Management](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Memory_Management) - JS memory
- [Three.js Disposal](https://threejs.org/docs/#manual/en/introduction/How-to-dispose-of-objects) - Cleanup guide
- [Performance.Memory API](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory) - Memory info

*Documentation last updated: January 12, 2026*
