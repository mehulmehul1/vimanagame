# Project Initialization & Boot Sequence - First Principles Guide

## Overview

The **boot sequence** is everything that happens from the moment a player opens your game to when the first frame appears on screen. It's the critical startup process that initializes all managers, loads assets, and prepares the game world.

This document explains:
- How the project is structured
- How Vite bundles the game
- The step-by-step initialization process
- How to configure the game for different platforms

## What You Need to Know First

Before understanding the boot sequence, you should know:
- **JavaScript modules** - Import/export for organizing code
- **Async/await** - Handling asynchronous operations
- **The DOM** - Basic HTML structure
- **Canvas elements** - Where graphics are rendered
- **Promises** - Handling operations that take time

### Quick Refresher: Async/Await

```javascript
// Synchronous code - blocks execution
const data = loadData();  // Everything waits here
console.log("This waits for loadData");

// Asynchronous code - doesn't block
async function init() {
  const data = await loadData();  // Other code can run
  console.log("This waits for loadData to finish");
}
```

---

## Project Structure

### Directory Layout

```
shadow-engine/
├── index.html              # Entry point HTML
├── package.json            # Dependencies and scripts
├── vite.config.js          # Vite configuration
├── public/                 # Static assets
│   ├── models/            # 3D models (.glb)
│   ├── splats/            # Gaussian splat files (.sog)
│   ├── audio/             # Music, dialog, SFX
│   └── video/             # Video files (.webm)
└── src/
    ├── main.js            # Entry point - initializes game
    ├── gameManager.js     # Central state manager
    ├── managers/          # All manager classes
    │   ├── sceneManager.js
    │   ├── dialogManager.js
    │   ├── musicManager.js
    │   ├── videoManager.js
    │   ├── sfxManager.js
    │   ├── inputManager.js
    │   ├── physicsManager.js
    │   ├── characterController.js
    │   ├── animationManager.js
    │   ├── vfxManager.js
    │   └── uiManager.js
    ├── util/              # Utility functions
    │   └── criteria.js    # Criteria matching
    └── data/              # All data files
        ├── dialogData.js
        ├── musicData.js
        ├── sfxData.js
        ├── videoData.js
        ├── sceneData.js
        ├── animationData.js
        ├── lightData.js
        └── vfxData.js
```

---

## Vite Configuration

### What is Vite?

**Vite** is a build tool that:
- Serves your code during development with hot reload
- Bundles your code for production
- Handles modern JavaScript features
- Processes WASM files (needed for Rapier physics)

### vite.config.js

```javascript
// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  // Base path for serving
  base: './',

  // Development server configuration
  server: {
    port: 3000,
    open: true,  // Open browser automatically
    host: true   // Listen on all addresses
  },

  // Build configuration
  build: {
    // Output directory
    outDir: 'dist',

    // Public assets base path
    assetsDir: 'assets',

    // Source maps for debugging
    sourcemap: true,

    // Minify in production
    minify: 'terser',

    // Target browsers
    target: 'es2020',

    // Chunk size warning limit
    chunkSizeWarningLimit: 1000
  },

  // Optimizations
  optimizeDeps: {
    // Include Three.js in optimization
    include: ['three']
  },

  // How to handle .wasm files (Rapier physics)
  assetsInclude: ['**/*.wasm']
});
```

### package.json Scripts

```json
{
  "name": "shadow-engine-game",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",           // Start development server
    "build": "vite build",   // Build for production
    "preview": "vite preview" // Preview production build
  },
  "dependencies": {
    "three": "^0.180.0",
    "@sparkjsdev/spark": "^0.1.10",
    "@dimforge/rapier3d": "^0.19.0",
    "howler": "^2.2.4"
  },
  "devDependencies": {
    "vite": "^7.1.7"
  }
}
```

---

## The Boot Sequence

### Step-by-Step Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BOOT SEQUENCE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. HTML LOADS                                                          │
│     └──> Browser loads index.html                                       │
│                                                                          │
│  2. JAVASCRIPT ENTRY                                                    │
│     └──> main.js executes                                               │
│                                                                          │
│  3. PLATFORM DETECTION                                                  │
│     └──> Detect: iOS, Android, Safari, Mobile                           │
│                                                                          │
│  4. PERFORMANCE PROFILE                                                 │
│     └──> Select: mobile, laptop, desktop, max                           │
│                                                                          │
│  5. CANVAS SETUP                                                        │
│     └──> Create and configure canvas for rendering                       │
│                                                                          │
│  6. THREE.JS INITIALIZATION                                             │
│     └──> Setup scene, camera, renderer                                  │
│                                                                          │
│  7. GAME MANAGER CREATION                                              │
│     └──> Create central state coordinator                               │
│                                                                          │
│  8. MANAGER INITIALIZATION                                              │
│     └──> Initialize all managers in dependency order                    │
│                                                                          │
│  9. DATA LOADING                                                        │
│     └──> Load all data files (dialog, music, etc.)                      │
│                                                                          │
│ 10. ASSET PRELOADING                                                    │
│     └──> Preload critical assets                                        │
│                                                                          │
│  11. START SCREEN                                                       │
│     └──> Show start screen / loading screen                             │
│                                                                          │
│  12. GAME LOOP START                                                    │
│     └──> Begin requestAnimationFrame loop                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Detailed Boot Sequence

#### Phase 1: HTML Loading (index.html)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Shadow Engine Game</title>
  <style>
    /* Basic reset */
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { overflow: hidden; background: #000; }
    canvas { display: block; }
  </style>
</head>
<body>
  <!-- Main canvas for rendering -->
  <canvas id="gameCanvas"></canvas>

  <!-- Loading screen -->
  <div id="loadingScreen">
    <div class="loading-bar">
      <div class="progress" id="loadingProgress"></div>
    </div>
    <div class="loading-text" id="loadingText">Loading...</div>
  </div>

  <!-- Import main.js as module -->
  <script type="module" src="/src/main.js"></script>
</body>
</html>
```

#### Phase 2: Main Entry Point (main.js)

```javascript
// src/main.js
import * as THREE from 'three';
import { GameManager } from './gameManager.js';
import { SceneManager } from './managers/sceneManager.js';
import { DialogManager } from './managers/dialogManager.js';
import { MusicManager } from './managers/musicManager.js';
import { SFXManager } from './managers/sfxManager.js';
import { VideoManager } from './managers/videoManager.js';
import { InputManager } from './managers/inputManager.js';
import { PhysicsManager } from './managers/physicsManager.js';
import { CharacterController } from './managers/characterController.js';
import { AnimationManager } from './managers/animationManager.js';
import { VFXManager } from './managers/vfxManager.js';
import { UIManager } from './managers/uiManager.js';

// Import data files
import * as dialogData from './data/dialogData.js';
import * as musicData from './data/musicData.js';
import * as sfxData from './data/sfxData.js';
import * as videoData from './data/videoData.js';
import * as sceneData from './data/sceneData.js';
import * as animationData from './data/animationData.js';
import * as lightData from './data/lightData.js';
import * as vfxData from './data/vfxData.js';

class Game {
  constructor() {
    this.canvas = document.getElementById('gameCanvas');
    this.container = document.body;
  }

  async init() {
    console.log('🎮 Game initializing...');

    // Phase 1: Platform detection
    await this.detectPlatform();

    // Phase 2: Performance profile
    await this.selectPerformanceProfile();

    // Phase 3: Canvas setup
    await this.setupCanvas();

    // Phase 4: Three.js initialization
    await this.initThree();

    // Phase 5: Create GameManager
    await this.createGameManager();

    // Phase 6: Initialize managers
    await this.initManagers();

    // Phase 7: Load data
    await this.loadData();

    // Phase 8: Preload critical assets
    await this.preloadAssets();

    // Phase 9: Show start screen
    await this.showStartScreen();

    // Phase 10: Start game loop
    this.startGameLoop();

    console.log('✅ Game initialized!');
  }

  async detectPlatform() {
    const ua = navigator.userAgent;

    this.platform = {
      isIOS: /iPad|iPhone|iPod/.test(ua) && !window.MSStream,
      isAndroid: /Android/.test(ua),
      isSafari: /Safari/.test(ua) && !/Chrome/.test(ua),
      isMobile: /Mobi|Android/i.test(ua) || /iPad|iPhone|iPod/.test(ua)
    };

    console.log('📱 Platform:', this.platform);
  }

  async selectPerformanceProfile() {
    // Auto-detect or use URL override
    const urlProfile = new URLSearchParams(window.location.search).get('profile');

    if (urlProfile) {
      this.performanceProfile = urlProfile;
    } else if (this.platform.isMobile) {
      this.performanceProfile = 'mobile';
    } else if (this.platform.isIOS || this.platform.isSafari) {
      this.performanceProfile = 'laptop';
    } else {
      this.performanceProfile = 'desktop';
    }

    console.log('⚡ Performance profile:', this.performanceProfile);
  }

  async setupCanvas() {
    // Resize canvas to window
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;

    // Handle resize
    window.addEventListener('resize', () => this.onResize());

    console.log('🖼️ Canvas setup complete');
  }

  async initThree() {
    // Create scene
    this.scene = new THREE.Scene();

    // Create camera
    this.camera = new THREE.PerspectiveCamera(
      75, // FOV
      window.innerWidth / window.innerHeight, // Aspect
      0.1, // Near
      1000 // Far
    );
    this.camera.position.set(0, 1.6, 0); // Eye level

    // Create renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(
      Math.min(window.devicePixelRatio, 2)
    );

    console.log('🎨 Three.js initialized');
  }

  async createGameManager() {
    this.gameManager = new GameManager();

    // Set initial state
    this.gameManager.setState({
      currentState: GAME_STATES.LOADING,
      isPlaying: false,
      isPaused: false,
      controlEnabled: false,
      currentZone: null,
      // ... all other state properties
    });

    console.log('🎮 GameManager created');
  }

  async initManagers() {
    // Create all managers
    this.sceneManager = new SceneManager(this.scene, this.renderer);
    this.dialogManager = new DialogManager();
    this.musicManager = new MusicManager();
    this.sfxManager = new SFXManager();
    this.videoManager = new VideoManager(this.scene);
    this.inputManager = new InputManager(this.canvas);
    this.physicsManager = new PhysicsManager();
    this.characterController = new CharacterController();
    this.animationManager = new AnimationManager(this.camera, this.scene);
    this.vfxManager = new VFXManager(this.renderer, this.scene);
    this.uiManager = new UIManager(this.container);

    // Initialize all managers with GameManager and data
    await this.gameManager.initialize({
      sceneManager: this.sceneManager,
      dialogManager: this.dialogManager,
      musicManager: this.musicManager,
      sfxManager: this.sfxManager,
      videoManager: this.videoManager,
      inputManager: this.inputManager,
      physicsManager: this.physicsManager,
      characterController: this.characterController,
      animationManager: this.animationManager,
      vfxManager: this.vfxManager,
      uiManager: this.uiManager
    });

    console.log('🔧 All managers initialized');
  }

  async loadData() {
    // Load data into managers
    await this.dialogManager.initialize(this.gameManager, dialogData.dialogTracks);
    await this.musicManager.initialize(this.gameManager, musicData.musicTracks);
    await this.sfxManager.initialize(this.gameManager, sfxData.sfxTracks);
    await this.videoManager.initialize(this.gameManager, videoData.videos);
    await this.sceneManager.initialize(this.gameManager, sceneData.sceneObjects);
    await this.animationManager.initialize(this.gameManager, animationData.animations);
    await this.vfxManager.initialize(this.gameManager, vfxData.vfxEffects);

    console.log('📦 All data loaded');
  }

  async preloadAssets() {
    // Show loading progress
    const loadingProgress = document.getElementById('loadingProgress');
    const loadingText = document.getElementById('loadingText');

    // Preload critical assets
    const assetsToPreload = [
      'models/start-screen.glb',
      'audio/music/start-screen.mp3',
      // ... other critical assets
    ];

    for (let i = 0; i < assetsToPreload.length; i++) {
      const asset = assetsToPreload[i];
      loadingText.textContent = `Loading ${asset}...`;
      // Load asset...
      loadingProgress.style.width = `${((i + 1) / assetsToPreload.length) * 100}%`;
    }

    console.log('📥 Critical assets preloaded');
  }

  async showStartScreen() {
    // Update state to start screen
    this.gameManager.setState({
      currentState: GAME_STATES.START_SCREEN
    });

    console.log('🏠 Start screen ready');
  }

  startGameLoop() {
    let lastTime = performance.now();

    const loop = (currentTime) => {
      requestAnimationFrame(loop);

      const deltaTime = (currentTime - lastTime) / 1000; // Convert to seconds
      lastTime = currentTime;

      // Update all managers
      this.update(deltaTime);

      // Render scene
      this.renderer.render(this.scene, this.camera);
    };

    requestAnimationFrame(loop);

    console.log('🔄 Game loop started');
  }

  update(deltaTime) {
    // Update in dependency order
    this.inputManager.update(deltaTime);
    this.physicsManager.update(deltaTime);
    this.characterController.update(deltaTime);
    this.dialogManager.update(deltaTime);
    this.musicManager.update(deltaTime);
    this.videoManager.update(deltaTime);
    this.sfxManager.update(deltaTime);
    this.animationManager.update(deltaTime);
    this.vfxManager.update(deltaTime);
    this.uiManager.update(deltaTime);
  }

  onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }
}

// Start the game
window.addEventListener('DOMContentLoaded', async () => {
  const game = new Game();
  await game.init();
});
```

---

## Performance Profiles

The engine supports different performance profiles for different devices:

| Profile | Use Case | Settings |
|---------|----------|----------|
| **mobile** | Low-end phones | Reduced resolution, no shadows, fewer particles |
| **laptop** | Integrated GPU | Medium quality, basic shadows |
| **desktop** | Dedicated GPU | High quality, full shadows |
| **max** | High-end PCs | Maximum quality, all effects |

### Profile Configuration

```javascript
const PERFORMANCE_PROFILES = {
  mobile: {
    pixelRatio: 1,
    shadowMapSize: 512,
    particleCount: 100,
    antialiasing: false,
    textureQuality: 'low'
  },

  laptop: {
    pixelRatio: 1.5,
    shadowMapSize: 1024,
    particleCount: 500,
    antialiasing: true,
    textureQuality: 'medium'
  },

  desktop: {
    pixelRatio: 2,
    shadowMapSize: 2048,
    particleCount: 2000,
    antialiasing: true,
    textureQuality: 'high'
  },

  max: {
    pixelRatio: window.devicePixelRatio,
    shadowMapSize: 4096,
    particleCount: 5000,
    antialiasing: true,
    textureQuality: 'ultra'
  }
};
```

---

## Initialization Order (Critical!)

Managers must be initialized in **dependency order**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INITIALIZATION ORDER                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. GameManager        (must be first - provides state)                 │
│  2. SceneManager        (sets up 3D scene)                              │
│  3. InputManager        (needs canvas)                                  │
│  4. PhysicsManager      (independent)                                   │
│  5. CharacterController (depends on: Input, Physics)                     │
│  6. DialogManager       (independent)                                   │
│  7. MusicManager        (independent)                                   │
│  8. SFXManager          (independent)                                   │
│  9. VideoManager        (needs scene)                                   │
│ 10. AnimationManager    (needs camera, scene)                           │
│ 11. VFXManager          (needs renderer, scene)                         │
│ 12. UIManager           (needs container)                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this order matters:**

```javascript
// ❌ WRONG: CharacterController before PhysicsManager
characterController.initialize();  // Will crash - needs PhysicsManager!
physicsManager.initialize();

// ✅ CORRECT: Dependencies first
physicsManager.initialize();
characterController.initialize();  // Can reference PhysicsManager
```

---

## Common Boot Issues

### Issue 1: Canvas Not Found

```javascript
// ❌ WRONG: Script runs before DOM is ready
const canvas = document.getElementById('gameCanvas'); // null!

// ✅ CORRECT: Wait for DOM
window.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('gameCanvas');
});
```

### Issue 2: Import Errors

```javascript
// ❌ WRONG: Missing .js extension
import { Game } from './game';  // Won't work!

// ✅ CORRECT: Include .js extension
import { Game } from './game.js';
```

### Issue 3: Async Initialization Race Conditions

```javascript
// ❌ WRONG: Not waiting for initialization
gameManager.init();
gameManager.setState({ ... });  // Might not be ready!

// ✅ CORRECT: Wait for init
await gameManager.init();
gameManager.setState({ ... });  // Safe!
```

### Issue 4: WebGL Not Supported

```javascript
// Always check for WebGL support
function checkWebGLSupport() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

  if (!gl) {
    alert('Your browser does not support WebGL. Please try a different browser.');
    return false;
  }

  return true;
}
```

---

## Debugging Boot Issues

### Console Logging Strategy

```javascript
class Game {
  async init() {
    console.group('🎮 Game Initialization');

    try {
      console.log('1/10 Platform detection...');
      await this.detectPlatform();

      console.log('2/10 Performance profile...');
      await this.selectPerformanceProfile();

      // ... etc

      console.groupEnd();
      console.log('✅ Initialization complete!');
    } catch (error) {
      console.groupEnd();
      console.error('❌ Initialization failed:', error);
      throw error;
    }
  }
}
```

### URL Debug Overrides

```javascript
// Allow skipping to specific states via URL
const urlParams = new URLSearchParams(window.location.search);
const debugState = urlParams.get('state');
const debugZone = urlParams.get('zone');

if (debugState || debugZone) {
  console.log('🐛 Debug mode detected');
  gameManager.setState({
    currentState: parseInt(debugState) || GAME_STATES.START_SCREEN,
    currentZone: debugZone || null
  });
}
```

---

## Next Steps

Now that you understand the boot sequence:

- [Tech Stack Overview](../01-foundation/tech-stack-overview.md) - All technologies used
- [Vite Configuration](https://vitejs.dev/config/) - Official Vite docs
- [GameManager Deep Dive](./game-manager-deep-dive.md) - State management
- [Manager-Based Architecture](./manager-based-architecture-pattern.md) - How managers work

---

## References

- [Vite Guide](https://vitejs.dev/guide/) - Official Vite documentation
- [Vite Build Configuration](https://vitejs.dev/guide/build.html) - Build options
- [Vite Development Server](https://vitejs.dev/guide/development.html) - Dev server setup
- [JavaScript Modules (MDN)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules) - ES modules
- [WebGL Setup (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Getting_started_with_WebGL) - WebGL basics

*Documentation last updated: January 12, 2026*
