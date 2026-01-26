# SceneManager - First Principles Guide

## Overview

The **SceneManager** is responsible for managing all 3D content in the Shadow Engine. It handles loading Gaussian Splat scenes (.sog files), traditional 3D models (.glb/.gltf), positioning objects, managing the camera, and coordinating with the Three.js rendering loop and SparkJS splat renderer.

Think of SceneManager as the **stage manager** for a theater production - it decides what props, scenery, and lighting are visible at any given moment.

## What You Need to Know First

Before understanding SceneManager, you should know:
- **Basic 3D concepts** - X, Y, Z coordinates, what a "scene" is
- **Three.js basics** - Scene, Camera, Renderer pattern
- **Gaussian Splatting** - Covered in [Gaussian Splatting Explained](../01-foundation/gaussian-splatting-explained.md)
- **JavaScript Promises** - How async loading works
- **Event-driven programming** - How state changes trigger actions

### Quick Refresher: Three.js Scene Graph

```
┌─────────────────────────────────────────────────────────────┐
│                       THREE.JS SCENE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                      scene                              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │ │
│  │  │  camera  │  │   light   │  │  mesh1   │  ...       │ │
│  │  └──────────┘  └──────────┘  └──────────┘            │ │
│  │                                                         │ │
│  │  Children can be nested:                               │ │
│  │  ┌─────────────┐                                        │ │
│  │  │   group    │────▶ mesh1, mesh2, light              │ │
│  │  └─────────────┘                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  renderer.render(scene, camera);  // Draw one frame        │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: SceneManager Responsibilities

### What SceneManager Does

| Responsibility | Description |
|----------------|-------------|
| **Scene Management** | Creates and manages the Three.js Scene object |
| **Camera Control** | Manages the camera and its properties |
| **Renderer Setup** | Initializes WebGL/WebGL2 renderer |
| **Splat Loading** | Loads and manages Gaussian Splat scenes via SparkJS |
| **Model Loading** | Loads GLTF 3D models |
| **Object Management** | Adds, removes, and positions 3D objects |
| **Zone Management** | Loads/unloads content based on player location |
| **Render Loop** | Coordinates the frame-by-frame rendering |
| **Resize Handling** | Responds to window resize events |

---

## Part 2: Initialization

### SceneManager Initialization Pattern

```javascript
class SceneManager {
  async initialize(gameManager, platform) {
    console.log('[SceneManager] Initializing...');

    // Store references
    this.gameManager = gameManager;
    this.platform = platform;

    // 1. Create the Three.js renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: !platform.isMobile,  // Disable on mobile
      powerPreference: 'high-performance',
      alpha: false,
      stencil: true,
      depth: true
    });

    // 2. Configure renderer
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // 3. Create the scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);

    // 4. Create the camera
    this.camera = new THREE.PerspectiveCamera(
      75,  // Field of view
      window.innerWidth / window.innerHeight,  // Aspect ratio
      0.1,  // Near clipping plane
      1000  // Far clipping plane
    );
    this.camera.position.set(0, 1.6, 2);  // Eye level (1.6m = average human height)

    // 5. Add renderer to DOM
    document.body.appendChild(this.renderer.domElement);

    // 6. Initialize SparkJS (Gaussian Splatting)
    await this.initializeSpark();

    // 7. Listen for state changes
    gameManager.on('state:changed', this.onStateChanged.bind(this));

    // 8. Listen for window resize
    window.addEventListener('resize', this.onResize.bind(this));

    console.log('[SceneManager] Ready!');
  }
}
```

### Creating the Renderer

The renderer is the engine that draws everything to the screen:

```javascript
// Renderer configuration choices explained
const renderer = new THREE.WebGLRenderer({
  // Antialiasing smooths jagged edges, but costs performance
  antialias: !platform.isMobile,  // Disable on mobile for FPS

  // Request "high-performance" GPU (vs "low-power")
  powerPreference: 'high-performance',

  // Alpha channel (transparent background)
  alpha: false,  // We use opaque black for better performance

  // Stencil buffer for advanced effects
  stencil: true,

  // Depth buffer for proper 3D rendering
  depth: true
});
```

**Verified from Three.js docs (r180+):**
- `powerPreference: 'high-performance'` requests the discrete GPU
- `setPixelRatio` with cap prevents performance issues on high-DPI screens
- `antialias: false` is recommended for mobile performance

---

## Part 3: Gaussian Splat Loading with SparkJS

### What is a Splat Scene?

A Gaussian Splat scene is captured from the real world using specialized photography. It contains millions of "splats" - each with:
- Position (X, Y, Z)
- Color (R, G, B)
- Opacity (Alpha)
- 3D Gaussian parameters (covariance for the splat shape)

The Shadow Engine uses the **.SOG format** (Spark Splat Object) from SparkJS.

### Loading a Splat Scene

```javascript
async initializeSpark() {
  // Import SplatMesh from SparkJS
  const { SplatMesh } = await import('@sparkjsdev/spark');

  // Create the splat mesh
  this.splatMesh = new SplatMesh({
    url: null,  // Will be set per scene
    position: new THREE.Vector3(0, 0, 0),
    quaternion: new THREE.Quaternion(0, 0, 0, 1),
    scale: new THREE.Vector3(1, 1, 1)
  });

  // Add to scene
  this.scene.add(this.splatMesh);

  console.log('[SceneManager] Spark initialized');
}
```

### Loading Different Splat Scenes

```javascript
async loadSplatScene(splatData) {
  console.log(`[SceneManager] Loading splat: ${splatData.file}`);

  // Unload previous splat
  if (this.splatMesh.url) {
    await this.splatMesh.dispose();
  }

  // Load new splat
  await this.splatMesh.load({
    url: splatData.file,
    position: new THREE.Vector3(
      splatData.position?.x || 0,
      splatData.position?.y || 0,
      splatData.position?.z || 0
    ),
    quaternion: new THREE.Quaternion(
      splatData.rotation?.x || 0,
      splatData.rotation?.y || 0,
      splatData.rotation?.z || 0,
      splatData.rotation?.w || 1
    ),
    scale: new THREE.Vector3(
      splatData.scale?.x || 1,
      splatData.scale?.y || 1,
      splatData.scale?.z || 1
    )
  });

  console.log(`[SceneManager] Splat loaded: ${splatData.file}`);
}
```

**Verified from SparkJS docs (v0.1.10):**
- `SplatMesh.load({ url, position, quaternion, scale })` is the correct API
- Splats use `position` (Vector3), `quaternion` (Quaternion), and `scale` (Vector3)
- The `dispose()` method unloads splat data from memory

---

## Part 4: Loading 3D Models (GLTF)

### GLTF Model Loading

For traditional 3D models (phones, doors, interactive objects), we use GLTF:

```javascript
// Import GLTFLoader from Three.js
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

class SceneManager {
  constructor() {
    this.gltfLoader = new GLTFLoader();
    this.loadedModels = new Map();  // Cache loaded models
  }

  async loadGLTF(modelData) {
    const { id, file, position, rotation, scale } = modelData;

    // Check if already loaded
    if (this.loadedModels.has(id)) {
      return this.loadedModels.get(id);
    }

    console.log(`[SceneManager] Loading GLTF: ${file}`);

    // Load the model
    const gltf = await new Promise((resolve, reject) => {
      this.gltfLoader.load(file, resolve, undefined, reject);
    });

    // Get the root scene
    const model = gltf.scene;

    // Set transform
    model.position.set(position.x, position.y, position.z);
    model.rotation.set(rotation.x, rotation.y, rotation.z);
    model.scale.set(scale.x, scale.y, scale.z);

    // Add to scene
    this.scene.add(model);

    // Cache for later
    this.loadedModels.set(id, model);

    return model;
  }
}
```

### Model Data Structure

```javascript
// From sceneData.js
const phoneBooth = {
  type: 'gltf',
  id: 'phone-booth',
  file: '/models/phone-booth.glb',
  position: { x: 5, y: 0, z: -10 },
  rotation: { x: 0, y: Math.PI / 4, z: 0 },  // 45 degrees
  scale: { x: 1, y: 1, z: 1 },
  interactive: true,
  criteria: { currentState: { $gte: PLAZA_ARRIVAL } }
};
```

---

## Part 5: State-Driven Scene Loading

### Loading Objects Based on Game State

When the game state changes, SceneManager loads/unloads objects:

```javascript
class SceneManager {
  onStateChanged(newState, oldState) {
    // Find all objects that should be visible in new state
    const objectsToLoad = Object.values(sceneData).filter(obj =>
      matchesCriteria(obj.criteria, newState)
    );

    // Find all objects currently loaded
    const objectsToUnload = this.currentObjects.filter(obj =>
      !matchesCriteria(obj.criteria, newState)
    );

    // Load new objects
    objectsToLoad.forEach(obj => {
      if (!this.currentObjects.has(obj.id)) {
        this.loadObject(obj);
      }
    });

    // Unload old objects
    objectsToUnload.forEach(obj => {
      this.unloadObject(obj);
    });

    // Update current objects set
    this.currentObjects = new Set(objectsToLoad.map(o => o.id));
  }

  async loadObject(obj) {
    switch (obj.type) {
      case 'splat':
        await this.loadSplatScene(obj);
        break;
      case 'gltf':
        await this.loadGLTF(obj);
        break;
      case 'light':
        this.loadLight(obj);
        break;
      // ... etc
    }
  }

  unloadObject(obj) {
    // Remove from scene
    const object = this.loadedModels.get(obj.id);
    if (object) {
      this.scene.remove(object);
      this.loadedModels.delete(obj.id);
    }
  }
}
```

---

## Part 6: Camera Management

### Camera Types

The Shadow Engine uses different camera modes:

```javascript
class SceneManager {
  // Camera modes
  static CAMERA_MODES = {
    FIRST_PERSON: 'first-person',
    CINEMATIC: 'cinematic',
    FIXED: 'fixed',
    DIALOG: 'dialog'
  };

  setCameraMode(mode, options = {}) {
    this.cameraMode = mode;

    switch (mode) {
      case SceneManager.CAMERA_MODES.FIRST_PERSON:
        // Controlled by CharacterController
        this.camera.position.set(0, 1.6, 0);
        break;

      case SceneManager.CAMERA_MODES.CINEMATIC:
        // Controlled by AnimationManager
        break;

      case SceneManager.CAMERA_MODES.DIALOG:
        // Focus on dialog target
        if (options.target) {
          this.camera.lookAt(options.target);
        }
        break;
    }
  }
}
```

### Camera Properties

```javascript
// Field of view (FOV) - how wide the camera sees
this.camera.fov = 75;  // Degrees, typically 60-90

// Clipping planes - what's visible
this.camera.near = 0.1;  // Closest visible distance
this.camera.far = 1000;  // Furthest visible distance

// Aspect ratio - width/height
this.camera.aspect = window.innerWidth / window.innerHeight;

// After changing properties, update projection matrix
this.camera.updateProjectionMatrix();
```

---

## Part 7: The Render Loop

### Per-Frame Update

```javascript
class SceneManager {
  update(deltaTime) {
    // Update splat mesh (SparkJS internal updates)
    if (this.splatMesh) {
      this.splatMesh.update(deltaTime);
    }

    // Update animations
    this.animationManager?.update(deltaTime);

    // Update effects
    this.vfxManager?.update(deltaTime);

    // Render the scene
    this.renderer.render(this.scene, this.camera);
  }
}
```

### The Animation Loop

```javascript
// In main.js or GameManager
function animate(time) {
  // Calculate delta time
  const deltaTime = clock.getDelta();

  // Update all managers
  gameManager.update(deltaTime);

  // Request next frame
  requestAnimationFrame(animate);
}

// Start the loop
requestAnimationFrame(animate);
```

**Verified from Three.js docs:**
- `requestAnimationFrame` is the standard way to create a render loop
- `renderer.render(scene, camera)` draws one frame
- `clock.getDelta()` gives time in seconds since last frame

---

## Part 8: Zone Management

### What Are Zones?

Zones are spatial areas in the game world. Each zone has its own:
- Splat scene (environment)
- 3D objects
- Lighting
- Audio ambience

### Zone Transitions

```javascript
class SceneManager {
  async loadZone(zoneName) {
    console.log(`[SceneManager] Loading zone: ${zoneName}`);

    // Get zone data
    const zoneData = zoneDefinitions[zoneName];
    if (!zoneData) {
      console.error(`[SceneManager] Unknown zone: ${zoneName}`);
      return;
    }

    // Show loading screen
    this.gameManager.emit('loading:start');

    // 1. Fade out current scene
    await this.fadeOut(1000);

    // 2. Unload current zone
    await this.unloadCurrentZone();

    // 3. Load new splat
    if (zoneData.splat) {
      await this.loadSplatScene(zoneData.splat);
    }

    // 4. Load zone objects
    for (const object of zoneData.objects) {
      await this.loadObject(object);
    }

    // 5. Set player spawn point
    if (zoneData.spawnPoint) {
      this.gameManager.characterController.setPosition(
        zoneData.spawnPoint.position
      );
      this.gameManager.characterController.setRotation(
        zoneData.spawnPoint.rotation
      );
    }

    // 6. Fade in new scene
    await this.fadeIn(1000);

    // 7. Hide loading screen
    this.gameManager.emit('loading:complete');

    // Update current zone
    this.currentZone = zoneName;
  }
}
```

### Zone Data Example

```javascript
// zoneData.js
export const zoneDefinitions = {
  plaza: {
    name: 'plaza',
    splat: {
      file: '/splats/plaza.sog',
      position: { x: 0, y: 0, z: 0 }
    },
    objects: [
      {
        type: 'gltf',
        id: 'phone-booth',
        file: '/models/phone-booth.glb',
        position: { x: 5, y: 0, z: -10 },
        rotation: { x: 0, y: 0.78, z: 0 },  // 45 degrees
        scale: { x: 1, y: 1, z: 1 }
      }
    ],
    spawnPoint: {
      position: { x: 0, y: 1.6, z: 5 },
      rotation: { x: 0, y: Math.PI, z: 0 }  // Face the scene
    }
  },

  office: {
    name: 'office',
    splat: {
      file: '/splats/office.sog',
      position: { x: 0, y: 0, z: 0 }
    },
    objects: [
      {
        type: 'gltf',
        id: 'office-door',
        file: '/models/door.glb',
        position: { x: 0, y: 0, z: 3 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1, z: 1 }
      }
    ],
    spawnPoint: {
      position: { x: 0, y: 1.6, z: 2 },
      rotation: { x: 0, y: 0, z: 0 }
    }
  }
};
```

---

## Part 9: Window Resize Handling

### Responsive Rendering

```javascript
class SceneManager {
  onResize() {
    // Update camera aspect ratio
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();

    // Update renderer size
    this.renderer.setSize(window.innerWidth, window.innerHeight);

    // Notify other managers
    this.gameManager.emit('window:resize', {
      width: window.innerWidth,
      height: window.innerHeight
    });
  }
}
```

---

## Part 10: Performance Optimization

### Splat Quality Settings

```javascript
// Adjust splat quality based on performance profile
setSplatQuality(profile) {
  const settings = {
    mobile: {
      splatCount: 500000,  // Fewer splats
      renderScale: 0.75,
      splatSize: 1.2
    },
    laptop: {
      splatCount: 1000000,
      renderScale: 1.0,
      splatSize: 1.0
    },
    desktop: {
      splatCount: 2000000,
      renderScale: 1.0,
      splatSize: 1.0
    },
    max: {
      splatCount: -1,  // All splats
      renderScale: 1.5,  // Supersampling
      splatSize: 0.8
    }
  };

  const quality = settings[profile] || settings.desktop;

  // Apply settings
  this.splatMesh.setMaxSplats(quality.splatCount);
  this.renderer.setPixelRatio(quality.renderScale);
}
```

### Memory Management

```javascript
class SceneManager {
  dispose() {
    console.log('[SceneManager] Disposing...');

    // Dispose splat mesh
    if (this.splatMesh) {
      this.splatMesh.dispose();
      this.splatMesh = null;
    }

    // Dispose all loaded models
    for (const [id, model] of this.loadedModels) {
      this.scene.remove(model);
      model.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(m => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
    }
    this.loadedModels.clear();

    // Dispose renderer
    this.renderer.dispose();
    this.renderer = null;

    // Remove canvas
    const canvas = this.renderer?.domElement;
    if (canvas && canvas.parentNode) {
      canvas.parentNode.removeChild(canvas);
    }

    console.log('[SceneManager] Disposed');
  }
}
```

**Verified from Three.js docs:**
- `geometry.dispose()` frees GPU memory for geometry
- `material.dispose()` frees GPU memory for materials
- `renderer.dispose()` cleans up WebGL context

---

## Common Mistakes Beginners Make

### 1. Not Disposing Resources

```javascript
// ❌ WRONG: Memory leak
function loadModel(url) {
  gltfLoader.load(url, (gltf) => {
    scene.add(gltf.scene);
  });
  // Never removed - memory leak!
}

// ✅ CORRECT: Proper disposal
async loadModel(url) {
  const gltf = await this.loadGLTF(url);
  this.loadedModels.set(id, gltf.scene);
}

function dispose() {
  // Clean up when done
  this.loadedModels.forEach(model => {
    scene.remove(model);
    model.traverse(child => {
      child.geometry?.dispose();
      child.material?.dispose();
    });
  });
}
```

### 2. Forgetting to Update Camera on Resize

```javascript
// ❌ WRONG: Stretched image when resized
window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  // Camera aspect not updated - looks stretched!
});

// ✅ CORRECT: Update camera too
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
```

### 3. Loading Large Assets Synchronously

```javascript
// ❌ WRONG: Blocks the main thread
function loadScene() {
  const data = loadLargeFileSync();  // Blocks!
  processData(data);
}

// ✅ CORRECT: Async loading
async function loadScene() {
  const data = await loadLargeFileAsync();  // Doesn't block
  processData(data);
}
```

### 4. Not Caching Loaded Models

```javascript
// ❌ WRONG: Reloads every state change
onStateChanged(state) {
  this.loadGLTF('phone-booth.glb');  // Loads again!
}

// ✅ CORRECT: Cache and reuse
onStateChanged(state) {
  if (!this.loadedModels.has('phone-booth')) {
    await this.loadGLTF('phone-booth.glb');
  }
  // Reuse loaded model
}
```

### 5. Incorrect Quaternion Usage

```javascript
// ❌ WRONG: Euler angles don't work the same for quaternions
mesh.quaternion.set(0, 45, 0, 1);  // Wrong! quat is (x, y, z, w)

// ✅ CORRECT: Convert Euler to Quaternion
const euler = new THREE.Euler(0, Math.PI / 4, 0);
mesh.quaternion.setFromEuler(euler);

// Or use setFromAxisAngle for simple rotations
mesh.quaternion.setFromAxisAngle(
  new THREE.Vector3(0, 1, 0),  // Y-axis
  Math.PI / 4  // 45 degrees
);
```

---

## Performance Considerations

1. **Splat Count** - More splats = better quality but slower
2. **Render Scale** - Higher pixelRatio = sharper but slower
3. **Model Complexity** - High-poly models should be LOD'd
4. **Draw Calls** - Batch objects when possible
5. **Shadow Casting** - Only enable on important objects

---

## Next Steps

Now that you understand SceneManager:

- [Gaussian Splatting Explained](../01-foundation/gaussian-splatting-explained.md) - How splats work
- [SparkJS Integration](./sparkjs-integration.md) - Spark renderer deep dive
- [ZoneManager](./zone-manager.md) - Dynamic loading/unloading
- [GameManager Deep Dive](./game-manager-deep-dive.md) - How SceneManager coordinates
- [VFXManager](../07-visual-effects/vfx-manager.md) - Visual effects

---

## Source File Reference

- **Location:** `src/managers/SceneManager.js`
- **Key exports:** `SceneManager` class
- **Dependencies:** Three.js, @sparkjsdev/spark, GLTFLoader
- **Used by:** GameManager, AnimationManager, VFXManager

---

## References

- [Three.js Documentation](https://threejs.org/docs/) - Version r180+ (verified January 2026)
- [Three.js Scene Graph](https://threejs.org/docs/#api/en/core/Scene) - Scene API
- [Three.js WebGLRenderer](https://threejs.org/docs/#api/en/renderers/WebGLRenderer) - Renderer API
- [Three.js PerspectiveCamera](https://threejs.org/docs/#api/en/cameras/PerspectiveCamera) - Camera API
- [SparkJS Documentation](https://sparkjs.dev/docs/) - Version 0.1.10 (verified January 2026)
- [SparkJS SplatMesh](https://sparkjs.dev/docs/#splatmesh) - SplatMesh API
- [GLTF Loader](https://threejs.org/docs/#examples/en/loaders/GLTFLoader) - Model loading
- [GLTF File Format](https://www.khronos.org/gltf/) - glTF specification

*Documentation last updated: January 12, 2026*
