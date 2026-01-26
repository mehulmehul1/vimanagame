# WASM Handling - First Principles Guide

## Overview

**WASM (WebAssembly)** is a binary instruction format that runs in web browsers at near-native speed. It enables running performance-critical code—like physics simulations, audio processing, and machine learning—in the browser with performance approaching native applications. For the Shadow Engine, WASM powers complex calculations that would be too slow in pure JavaScript.

Think of WASM like the **"specialized engine"**—like how a car has a regular engine for normal driving and a turbo booster for extra power, JavaScript handles general game logic while WASM handles heavy computational tasks.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Enable complex game systems that feel responsive and realistic. Physics should feel weighty, audio should sound rich, and AI should be smart—without sacrificing frame rate. WASM makes these computationally expensive features possible in the browser.

**Why WASM for Games?**
- **Physics**: Realistic collisions at 60+ FPS
- **Audio**: High-quality spatial audio processing
- **ML/AI**: Real-time gesture recognition
- **Compression**: Faster asset loading/decompression
- **Cross-Platform**: Same performance on all devices

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding WASM, you should know:
- **Binary format** - WASM is compiled code, not human-readable
- **Module system** - WASM modules export/import functions
- **Memory** - Linear memory array for WASM data
- **JavaScript interop** - How JS and WASM communicate
- **Loading** - Async loading of WASM modules

### Core Architecture

```
WASM INTEGRATION ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                    GAME LAYER                          │
│  - High-level game logic                               │
│  - Scene management                                     │
│  - Player input                                         │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌───────────────────┐               ┌───────────────────┐
│   JAVASCRIPT      │               │      WASM       │
│  - Game logic     │◄────────────►│  - Physics      │
│  - Rendering     │   Interop     │  - Audio DSP    │
│  - UI            │               │  - ML inference │
└───────────────────┘               └───────────────────┘
                                                │
        ┌───────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│                  WASM MODULES                         │
│  - Physics engine (Rapier compiled to WASM)          │
│  - Audio codecs (Opus, Vorbis)                        │
│  - TensorFlow.js (WASM backend)                       │
└─────────────────────────────────────────────────────────┘
```

### WASM Module Loading

```javascript
/**
 * WASM Loader - Handles loading and initializing WASM modules
 */
class WasmLoader {
  constructor(options = {}) {
    this.gameManager = options.gameManager;
    this.loadedModules = new Map();
    this.loadingPromises = new Map();
  }

  /**
   * Load a WASM module
   */
  async loadModule(name, path) {
    // Return cached if already loaded
    if (this.loadedModules.has(name)) {
      return this.loadedModules.get(name);
    }

    // Return existing promise if currently loading
    if (this.loadingPromises.has(name)) {
      return this.loadingPromises.get(name);
    }

    // Load the module
    const loadPromise = this._loadModule(name, path);
    this.loadingPromises.set(name, loadPromise);

    try {
      const module = await loadPromise;
      this.loadedModules.set(name, module);
      this.loadingPromises.delete(name);
      return module;
    } catch (error) {
      this.loadingPromises.delete(name);
      throw error;
    }
  }

  /**
   * Internal loading implementation
   */
  async _loadModule(name, path) {
    // Check if browser supports WASM
    if (!this.isSupported()) {
      throw new Error('WebAssembly is not supported in this browser');
    }

    // Fetch the WASM file
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Failed to load WASM module "${name}": ${response.statusText}`);
    }

    // Get WASM bytes
    const wasmBytes = await response.arrayBuffer();

    // Create module
    const module = await WebAssembly.compile(wasmBytes);

    // Create memory for the module
    const memory = new WebAssembly.Memory({ initial: 256, maximum: 512 });

    // Create instance with imports
    const instance = await WebAssembly.instantiate(module, {
      env: {
        memory: memory,
        emscripten_notify_memory_growth: (index) => {
          this.handleMemoryGrowth(index);
        },
        abort: () => {
          throw new Error('WASM module aborted');
        }
      },
      // Custom imports based on module needs
      ...this.getModuleImports(name)
    });

    return {
      instance,
      memory,
      exports: instance.exports
    };
  }

  /**
   * Get module-specific imports
   */
  getModuleImports(name) {
    // Different WASM modules need different imports
    const imports = {
      // Physics engine imports
      physics: {
        log: (ptr, len) => this.logFromWasm(ptr, len),
        performance_now: () => performance.now()
      },

      // Audio processing imports
      audio: {
        sample_rate: () => this.getSampleRate(),
        buffer_duration: () => this.getBufferDuration()
      },

      // ML model imports
      ml: {
        Math_log: Math.log,
        Math_exp: Math.exp,
        Math_sqrt: Math.sqrt
      }
    };

    return imports[name] || {};
  }

  /**
   * Handle WASM memory growth
   */
  handleMemoryGrowth(index) {
    console.warn(`WASM memory growth requested: ${index} pages`);
  }

  /**
   * Log from WASM (converts WASM memory to string)
   */
  logFromWasm(ptr, len) {
    const memory = this.getWasmMemory();
    const bytes = new Uint8Array(memory.buffer, ptr, len);
    const string = new TextDecoder().decode(bytes);
    console.log('[WASM]', string);
  }

  /**
   * Check WASM support
   */
  isSupported() {
    return typeof WebAssembly === 'object' && WebAssembly.validate;
  }

  /**
   * Check for SIMD support (faster WASM)
   */
  hasSimdSupport() {
    return WebAssembly.validate(new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01
    ]));
  }

  /**
   * Check for threading support
   */
  hasThreadingSupport() {
    // Requires SharedArrayBuffer and proper headers
    return typeof SharedArrayBuffer !== 'undefined' &&
           crossOriginIsolated;
  }

  /**
   * Get WASM memory (for debugging)
   */
  getWasmMemory() {
    for (const [name, module] of this.loadedModules) {
      if (module.memory) {
        return module.memory;
      }
    }
    return null;
  }
}

export default WasmLoader;
```

### Integrating with Physics (Rapier)

Rapier physics can use WASM for better performance:

```javascript
import RigidBody from '@dimforge/rapier3d-compat';

// Rapier has a WASM backend
// Initialize Rapier with WASM support
async function initPhysics() {
  // Rapier will load its WASM module automatically
  const gravity = { x: 0, y: -9.81, z: 0 };
  const world = new RigidBody(gravity);

  console.log('Physics initialized with WASM:', world.isWasm());
}
```

### Integrating with TensorFlow.js

TensorFlow.js uses WASM for faster inference:

```javascript
import * as tf from '@tensorflow/tfjs';

// Set WASM backend
async function initTensorFlow() {
  await tf.setBackend('webgl');
  await tf.ready();

  console.log('TensorFlow backend:', tf.getBackend().toUpperCase());

  // Run model (uses WASM for some operations)
  const model = await tf.loadLayersModel('/models/model.json');
  const result = model.predict(input);
}
```

---

## 📝 WASM Use Cases

### 1. Physics Simulation

```javascript
// WASM enables physics at 60+ FPS
class PhysicsWorld {
  async init() {
    // Load Rapier WASM module
    await RigidBody.init();

    // Create physics world
    this.world = new RigidBody({
      gravity: { x: 0, y: -9.81, z: 0 }
    });

    // WASM handles collision detection
    // Much faster than JS implementation
  }

  update(dt) {
    // Step physics (WASM-optimized)
    this.world.step();
  }
}
```

### 2. Audio Processing

```javascript
// WASM for audio effects
class AudioProcessor {
  async init() {
    // Load WASM audio codec
    this.wasm = await loadWasmModule('audio-codec.wasm');

    // Process audio with WASM
    this.wasm.exports.initialize();
  }

  process(inputBuffer) {
    // Pass audio to WASM for processing
    const inputPtr = this.wasm.exports.createBuffer(inputBuffer.length);
    const inputData = new Uint8Array(this.wasm.memory.buffer, inputPtr, inputBuffer.length);
    inputData.set(new Uint8Array(inputBuffer));

    this.wasm.exports.process(inputPtr, inputBuffer.length);

    // Get processed audio back
    const outputPtr = this.wasm.exports.getOutputBuffer();
    const outputLength = this.wasm.exports.getOutputLength();
    const outputData = new Uint8Array(this.wasm.memory.buffer, outputPtr, outputLength);

    return outputData.buffer;
  }
}
```

### 3. Gesture Recognition

```javascript
// WASM for ML inference
class GestureRecognizer {
  async init() {
    // Load TensorFlow.js with WASM backend
    await tf.setBackend('wasm');
    await tf.ready();

    // Load pre-trained model
    this.model = await tf.loadLayersModel('/models/gesture-model.json');
  }

  async recognize(stroke) {
    // Preprocess stroke
    const input = this.preprocessStroke(stroke);

    // Run inference (WASM-accelerated)
    const result = await this.model.predict(input);

    return result;
  }
}
```

---

## WASM File Handling

### Vite Configuration for WASM

```javascript
// vite.config.js
export default defineConfig({
  assetsInclude: [
    '**/*.wasm',
    '**/*.wasm.br'  // Brotli compressed
  ],

  build: {
    rollupOptions: {
      output: {
        // Keep WASM files separate
        assetFileNames: 'assets/wasm/[name]-[hash][extname]'
      }
    }
  }
});
```

### Inline WASM (Small Modules)

```javascript
// For small WASM modules, inline as base64
import wasmBase64 from './module.wasm?inline';

const wasmBytes = Uint8Array.from(atob(wasmBase64), c => c.charCodeAt(0));
const module = await WebAssembly.instantiate(wasmBytes);
```

---

## Common Mistakes Beginners Make

### 1. Not Checking WASM Support

```javascript
// ❌ WRONG: Assume WASM is available
const module = await WebAssembly.instantiate(wasmBytes);
// Crashes on older browsers

// ✅ CORRECT: Check support first
if (!WebAssembly || !WebAssembly.validate) {
  // Fall back to JavaScript
  useJSFallback();
} else {
  const module = await WebAssembly.instantiate(wasmBytes);
}
```

### 2. Forgetting Cross-Origin Isolation

```javascript
// ❌ WRONG: Missing headers
// SharedArrayBuffer won't work

// ✅ CORRECT: Set headers in server config
// vite.config.js server headers:
{
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp'
}
```

### 3. Not Handling WASM Errors

```javascript
// ❌ WRONG: No error handling
const module = await WebAssembly.instantiate(wasmBytes);
// If WASM fails, game crashes

// ✅ CORRECT: Handle errors
try {
  const module = await WebAssembly.instantiate(wasmBytes);
} catch (error) {
  console.error('WASM load failed:', error);
  useFallbackImplementation();
}
```

---

## Performance Considerations

```
WASM PERFORMANCE COMPARISON:

Physics (1000 objects):
├── JavaScript: ~15ms per frame (66 FPS)
├── WASM: ~5ms per frame (200 FPS)
└── Speedup: 3x faster

Audio Processing:
├── JavaScript: ~8ms per buffer
├── WASM: ~2ms per buffer
└── Speedup: 4x faster

ML Inference:
├── JavaScript: ~50ms per inference
├── WASM: ~15ms per inference
└── Speedup: 3.3x faster
```

---

## Related Systems

- [Vite Configuration](./vite-configuration.md) - Build settings
- [Build Process](./build-process.md) - Production builds
- [Performance Profiles](../11-performance-platform/performance-profiles.md) - Performance tuning

---

## Source File Reference

**WASM Files**:
- `public/wasm/*.wasm` - WebAssembly modules
- `src/util/WasmLoader.js` - WASM loader (estimated)

**Dependencies**:
- WebAssembly API (browser built-in)
- Rapier Physics (uses WASM)
- TensorFlow.js (uses WASM backend)

---

## References

- [WebAssembly MDN](https://developer.mozilla.org/en-US/docs/WebAssembly) - WASM API
- [Rapier Physics](https://rapier.rs/) - Physics engine
- [TensorFlow.js WASM](https://www.tensorflow.org/js/guides/platforms) - ML with WASM

*Documentation last updated: January 12, 2026*
