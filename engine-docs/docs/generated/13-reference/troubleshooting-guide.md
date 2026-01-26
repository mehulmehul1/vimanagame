# Troubleshooting Guide - Shadow Engine

## Overview

This **Troubleshooting Guide** helps you diagnose and fix common problems when developing with the Shadow Engine. Every developer encounters errors and unexpected behavior—knowing how to quickly identify and resolve issues is an essential skill.

Think of this guide as your **"first aid kit"**—like a medical first aid kit helps you treat common injuries quickly, this guide helps you treat common engine problems without wasting hours of development time.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Transform frustration into problem-solving. When something breaks, developers should feel equipped to diagnose and fix it, not stuck and helpless.

**Why Troubleshooting Matters?**
- **Fast Recovery**: Get back to creating, not debugging
- **Learning Opportunity**: Each bug teaches you how the engine works
- **Player Experience**: Players encounter fewer bugs if you catch them early
- **Confidence**: Know you can handle whatever goes wrong

---

## 🔍 How to Approach Problems

### The Scientific Method for Debugging

```
1. OBSERVE
   What exactly is happening?
   What should be happening instead?

2. ISOLATE
   When does it occur?
   What triggers it?
   Can you reproduce it consistently?

3. HYPOTHESIZE
   What could cause this?
   What changed recently?

4. TEST
   Try your fix
   Did it work?

5. VERIFY
   Does it work in all cases?
   Did you break anything else?
```

### Essential Debugging Tools

| Tool | Purpose | How to Use |
|------|---------|------------|
| **Console** | See errors/logs | F12 → Console tab |
| **Network Tab** | Check asset loading | F12 → Network tab |
| **Performance** | Profile FPS/memory | F12 → Performance |
| **Sources** | Set breakpoints | F12 → Sources tab |
| **Vue/React DevTools** | Inspect component state | Browser extension |

---

## 🚨 Common Issues by Category

---

## Graphics & Rendering Issues

### Problem: Black Screen, No 3D Content

**Symptoms:**
- Game loads but screen is black
- UI appears but 3D scene is missing
- No error messages in console

**Possible Causes:**

1. **WebGPU Not Supported**
```javascript
// Check if WebGPU is available
if (!navigator.gpu) {
  console.error('WebGPU is not supported in this browser');
  // Fall back to WebGL or show message
  showBrowserCompatibilityMessage();
}
```

**Solution:** Check browser compatibility and update:
- Chrome 113+ supports WebGPU
- Edge 113+ supports WebGPU
- Firefox: Enable `dom.webgpu.enabled` in `about:config`
- Safari: Experimental support as of 2025

2. **Splat File Failed to Load**
```javascript
// Check Network tab for failed requests
// Verify .splat file path is correct

// Common mistake: Wrong path
loadScene({
  splats: ['/assets/splats/plaza.splat']  // ✅ Leading slash
  // NOT: ['assets/splats/plaza.splat']   // ❌ Relative path
});
```

**Solution:** Verify paths and check Network tab for 404 errors.

3. **Camera Position Wrong**
```javascript
// Camera might be underground or facing away
const camera = sceneManager.getCamera();
camera.position.set(0, 2, 5);  // Set above ground
camera.lookAt(0, 0, 0);        // Face origin
```

---

### Problem: Poor Performance / Low FPS

**Symptoms:**
- Frame rate drops below 30 FPS
- Stuttering during movement
- Audio glitches or delays

**Diagnostic Steps:**

```javascript
// Enable performance monitoring
const perf = game.getManager('performance');
perf.enableProfiling(true);

// Check after 30 seconds
const stats = perf.getStats();
console.log('FPS:', stats.fps);
console.log('Frame Time:', stats.frameTime, 'ms');
console.log('Draw Calls:', stats.drawCalls);
console.log('Triangles:', stats.triangles);
```

**Common Solutions:**

1. **Too Many Splat Points**
```javascript
// Reduce splat resolution
sceneManager.loadScene({
  splats: ['/assets/splats/plaza.splat'],
  splatOptions: {
    maxPoints: 5000000,  // Reduce from default
    lod: true            // Enable level of detail
  }
});
```

2. **High Render Scale**
```javascript
// Lower internal resolution
game.options.renderScale = 0.7;  // 70% of screen resolution
```

3. **No Performance Profile Applied**
```javascript
// Apply appropriate profile
const perf = game.getManager('performance');
const profile = perf.detectBestProfile();
perf.applyProfile(profile);
```

---

### Problem: Visual Glitches / Artifacts

**Symptoms:**
- Flickering objects
- Z-fighting (surfaces flickering)
- Objects disappearing
- Wrong colors

**Solutions:**

1. **Z-Fighting - Move Surfaces Apart**
```javascript
// Add offset between overlapping surfaces
const floor1 = new THREE.Mesh(geometry, material);
floor1.position.y = 0;

const floor2 = new THREE.Mesh(geometry, material);
floor2.position.y = 0.01;  // Small offset to prevent z-fighting
```

2. **Enable Antialiasing**
```javascript
// In renderer config
const renderer = new THREE.WebGPURenderer({
  antialias: true,
  alpha: false
});
```

3. **Check Material Settings**
```javascript
// Ensure correct material properties
material.side = THREE.DoubleSide;  // Or FrontSide
material.transparent = true;
material.depthWrite = true;
```

---

## Input & Control Issues

### Problem: Keyboard/Mouse Not Working

**Symptoms:**
- No response to key presses
- Camera doesn't move with mouse
- Click events not registering

**Solutions:**

1. **Click to Focus**
```javascript
// Browser requires user interaction for keyboard/pointer lock
document.addEventListener('click', async () => {
  await document.body.requestPointerLock();
});

// Show prompt to user
showMessage('Click to enable controls');
```

2. **Check Input Binding**
```javascript
// Verify action is bound
const input = game.getManager('input');
input.on('jump', () => {
  console.log('Jump pressed!');  // Debug message
  player.jump();
});

// Check if action is mapped to key
input.bindKey('Space', 'jump');
```

3. **Pointer Lock Requirements**
```javascript
// Pointer lock requires:
// 1. User interaction (click/keypress)
// 2. Secure context (HTTPS or localhost)
// 3. Fullscreen or fullscreen flag

// Check status
if (document.pointerLockElement !== document.body) {
  console.warn('Pointer not locked');
}
```

---

### Problem: Gamepad Not Detected

**Symptoms:**
- Gamepad doesn't work
- `navigator.getGamepads()` returns empty array

**Solutions:**

1. **Press a Button First**
```javascript
// Browsers require button press to detect gamepad
window.addEventListener('gamepadconnected', (e) => {
  console.log('Gamepad connected:', e.gamepad.id);
});

// Check after button press
function pollGamepad() {
  const gamepads = navigator.getGamepads();
  const gp = gamepads[0];
  if (gp) {
    // Gamepad is now active
    const button = gp.buttons[0].pressed;
  }
}
```

2. **Browser Support**
```javascript
// Check if Gamepad API is supported
if (!navigator.getGamepads) {
  showMessage('Gamepad not supported in this browser');
}
```

---

## Audio Issues

### Problem: No Sound Playing

**Symptoms:**
- Music doesn't start
- Sound effects are silent
- No error messages

**Solutions:**

1. **User Interaction Required**
```javascript
// Audio context requires user gesture
document.addEventListener('click', async () => {
  const audio = game.getManager('audio');
  await audio.initialize();  // Initialize after user interaction
  await audio.playMusic('main_theme');
});

// Show "Click to start" overlay
```

2. **Check Volume Settings**
```javascript
// Verify volume is not zero
const audio = game.getManager('audio');
console.log('Master volume:', audio.getMasterVolume());
console.log('Music volume:', audio.getMusicVolume());

audio.setMasterVolume(0.8);
```

3. **Verify File Paths**
```javascript
// Check Network tab for failed audio loads
audio.playMusic('main_theme');
// Looks for: /assets/audio/music/main_theme.mp3

// Verify files exist:
// /assets/audio/music/main_theme.mp3
// /assets/audio/sfx/jump.ogg
```

---

### Problem: Audio Cutting Out / Glitching

**Symptoms:**
- Audio stops playing after a while
- Popping or crackling sounds
- Delay between sound trigger and audio

**Solutions:**

1. **Too Many Concurrent Sounds**
```javascript
// Limit concurrent SFX
const audio = game.getManager('audio');
audio.maxConcurrentSFX = 16;  // Default to 16

// Prioritize important sounds
audio.playSFX('explosion', { priority: 'high' });
audio.playSFX('footstep', { priority: 'low' });
```

2. **Preload Audio**
```javascript
// Preload frequently used sounds
await audio.preloadMusic(['main_theme', 'combat_theme']);
await audio.preloadSFX(['jump', 'footstep', 'explosion']);
```

3. **Check Audio Context State**
```javascript
// Verify context is running
if (audio.context.state === 'suspended') {
  await audio.context.resume();
}
```

---

## Physics Issues

### Problem: Objects Falling Through Floor

**Symptoms:**
- Objects sink into ground
- Player falls through world
- Physics bodies not colliding

**Solutions:**

1. **Check Collision Layers**
```javascript
// Ensure objects are on colliding layers
physics.createCollider(body, {
  shape: 'box',
  layers: 1,  // Layer 1
  mask: 0xFFFF  // Collides with everything
});

// Player needs ground layer
playerCollider.mask |= LAYER_GROUND;
```

2. **Increase Solver Iterations**
```javascript
// For more stable stacking
physics.world.numSolverIterations = 8;  // Default: 4
```

3. **Continuous Collision Detection**
```javascript
// For fast-moving objects
physics.createBody({
  type: 'dynamic',
  CCD: true,  // Enable continuous collision
  position: { x: 0, y: 10, z: 0 }
});
```

---

### Problem: Jittery Physics Movement

**Symptoms:**
- Objects vibrate
- Camera shakes when moving
- Unstable stacking

**Solutions:**

1. **Fixed Time Step**
```javascript
// Physics should run at fixed time step
const FIXED_TIMESTEP = 1 / 60;
let accumulator = 0;

function update(deltaTime) {
  accumulator += deltaTime;

  while (accumulator >= FIXED_TIMESTEP) {
    physics.step(FIXED_TIMESTEP);
    accumulator -= FIXED_TIMESTEP;
  }
}
```

2. **Adjust Damping**
```javascript
// Add damping to reduce oscillation
physics.createBody({
  type: 'dynamic',
  linearDamping: 0.1,  // Reduce velocity over time
  angularDamping: 0.1  // Reduce rotation
});
```

---

## Build & Deployment Issues

### Problem: Build Fails with Errors

**Symptoms:**
- `npm run build` exits with error
- Module not found errors
- Type errors

**Solutions:**

1. **Clean and Rebuild**
```bash
# Remove all build artifacts and dependencies
rm -rf node_modules
rm -rf dist
rm package-lock.json

# Reinstall
npm install

# Build again
npm run build
```

2. **Check Import Paths**
```javascript
// ❌ WRONG: File extension missing
import GameManager from './GameManager';

// ✅ CORRECT: Include .js extension
import GameManager from './GameManager.js';
```

3. **TypeScript Errors**
```bash
# Run type check to see all errors
npm run type-check

# Common fix: Add type declarations
// @ts-check
// @ts-ignore for specific lines
```

---

### Problem: Production Build Has Different Behavior

**Symptoms:**
- Works in dev, breaks in production
- Assets not loading in production
- Different behavior after build

**Solutions:**

1. **Check Environment Variables**
```javascript
// Some code may have environment-specific branches
if (import.meta.env.DEV) {
  // This code only runs in development
}

// Use consistent behavior across environments
const isDebug = import.meta.env.VITE_DEBUG === 'true';
if (isDebug) {
  // Debug code
}
```

2. **Verify Asset Paths**
```javascript
// Use relative paths for assets
const splatPath = new URL('./assets/splats/plaza.splat', import.meta.url).href;

// Or put static assets in public/ folder
// They will be copied to dist/ as-is
```

3. **Source Maps for Debugging**
```javascript
// vite.config.js
export default {
  build: {
    sourcemap: true  // Enable for production debugging
  }
};
```

---

## WASM Loading Issues

### Problem: WASM Module Fails to Load

**Symptoms:**
- Error loading .wasm file
- "WebAssembly.instantiate" failed
- MIME type error

**Solutions:**

1. **Check Server Configuration**
```nginx
# nginx.conf
location ~* \.wasm$ {
  application/wasm;
  add_header Cache-Control "public, max-age=31536000";
}
```

2. **Enable COOP/COEP Headers**
```javascript
// vite.config.js
export default {
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'credentialless',
      'Cross-Origin-Opener-Policy': 'same-origin'
    }
  }
};
```

3. **Verify WASM Support**
```javascript
if (typeof WebAssembly === 'undefined') {
  showMessage('WebAssembly is not supported');
  // Provide fallback
}
```

---

## Memory Issues

### Problem: Browser Crashes / Out of Memory

**Symptoms:**
- Tab crashes after playing for a while
- "Out of memory" errors
- Performance degrades over time

**Solutions:**

1. **Check for Memory Leaks**
```javascript
// Use Chrome Memory Profiler
// F12 → Memory → Take Heap Snapshot

// Look for:
// - Detached DOM nodes
// - Event listeners not removed
// - Textures not disposed
```

2. **Properly Dispose Resources**
```javascript
// When removing objects
sceneManager.removeObject(mesh);

// Also dispose geometry and material
mesh.geometry.dispose();
mesh.material.dispose();

// For textures
mesh.material.map.dispose();

// For splats
splatRenderer.dispose();
```

3. **Enable Memory Budgeting**
```javascript
const memory = game.getManager('memory');
memory.setBudget('textures', 200 * 1024 * 1024);  // 200MB
memory.setBudget('geometry', 50 * 1024 * 1024);   // 50MB

memory.enableAutoCleanup(true);
```

---

## Browser-Specific Issues

### Chrome/Edge

**Issue: "SharedArrayBuffer is not defined"**

**Cause:** Missing COOP/COEP headers

**Solution:**
```javascript
// vite.config.js
server: {
  headers: {
    'Cross-Origin-Embedder-Policy': 'credentialless',
    'Cross-Origin-Opener-Policy': 'same-origin'
  }
}
```

---

### Firefox

**Issue: WebGPU not available**

**Cause:** WebGPU is behind a flag

**Solution:**
1. Navigate to `about:config`
2. Set `dom.webgpu.enabled` to `true`
3. Restart browser

---

### Safari

**Issue: Various WebGL/WebGPU issues**

**Cause:** Safari has different implementation

**Solution:**
```javascript
// Feature detection and fallback
if (navigator.gpu) {
  renderer = new WebGPURenderer();
} else {
  renderer = new WebGLRenderer({ antialias: true });
}
```

---

## Mobile-Specific Issues

### Problem: Performance is Very Poor on Mobile

**Solutions:**

1. **Apply Mobile Profile**
```javascript
const perf = game.getManager('performance');
perf.applyProfile('mobile');
```

2. **Reduce Texture Resolution**
```javascript
// Use smaller textures for mobile
const textureSize = isMobile ? 512 : 2048;
```

3. **Disable Post-Processing**
```javascript
renderer.postProcessing.enabled = isDesktop;
```

---

### Problem: Touch Controls Not Working

**Solutions:**

1. **Prevent Default Touch Actions**
```css
/* Prevent zoom/scroll on canvas */
canvas {
  touch-action: none;
}
```

2. **Handle Touch Events Properly**
```javascript
joystick.element.addEventListener('touchstart', (e) => {
  e.preventDefault();  // Prevent default
  // Handle touch
}, { passive: false });
```

---

## Getting Help

### Information to Gather

When asking for help, include:

1. **Browser Info:**
```javascript
console.log({
  userAgent: navigator.userAgent,
  platform: navigator.platform,
  webgl: getWebGLInfo(),
  webgpu: !!navigator.gpu
});
```

2. **Console Errors:**
   - Screenshot or copy all red errors
   - Include stack traces

3. **Reproduction Steps:**
   - What exactly you did
   - What you expected to happen
   - What actually happened

4. **Minimal Example:**
   - Simple code that reproduces the issue
   - Remove unrelated code

### Resources

| Resource | Purpose |
|----------|---------|
| **Console** | See errors immediately |
| **Network Tab** | Check asset loading |
| **Performance Tab** | Profile FPS and memory |
| **Source Tab** | Set breakpoints and debug |
| [Three.js Discord](https://discord.gg/7GJSaXC) | Community help |
| [Stack Overflow](https://stackoverflow.com) | Search for similar issues |

---

## Quick Reference Checklist

```
When Something Breaks:

☐ Check Console for errors
☐ Check Network tab for failed loads
☐ Verify all file paths are correct
☐ Ensure user interaction occurred (for audio/input)
☐ Test in different browsers
☐ Try clearing cache and rebuilding
☐ Check for recent code changes
☐ Create minimal reproduction case
☐ Search for similar issues online
☐ Ask for help with all relevant information
```

---

## Related Systems

- [Development Setup](../12-build-deployment/development-setup.md) - Environment configuration
- [Performance Profiles](../11-performance-platform/performance-profiles.md) - Optimization
- [Memory Management](../11-performance-platform/memory-management.md) - Resource cleanup
- [API Reference](./api-reference.md) - Complete API documentation

---

## Source File Reference

**Debug Utilities:**
- `src/util/Debug.js` - Debug helpers
- `src/util/Logger.js` - Logging system
- `src/managers/PerformanceManager.js` - Performance profiling

**Error Handling:**
- `src/core/ErrorHandler.js` - Centralized error handling
- `src/core/Validation.js` - Input validation

---

## References

- [Chrome DevTools](https://developer.chrome.com/docs/devtools/) - Browser debugging
- [Firefox DevTools](https://firefox-source-docs.mozilla.org/devtools-user/) - Firefox debugging
- [Three.js Debugging](https://threejs.org/docs/#manual/en/introduction/Creating-a-scene) - Scene debugging
- [WebGPU Troubleshooting](https://gpuweb.github.io/gpuweb/) - WebGPU spec and issues

*Documentation last updated: January 12, 2026*
