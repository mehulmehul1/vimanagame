# Platform Detection - First Principles Guide

## Overview

**Platform Detection** is the system that identifies what device, browser, and capabilities the player is using. This information drives critical decisions: which performance profile to use, whether to show touch controls, what graphics features are available, and how to optimize the experience. Good platform detection happens once at startup and provides reliable data for all other systems.

Think of platform detection as the **"identity check"**—like a border agent checking your passport, it determines who you are (device type), where you're from (browser), and what you're allowed to do (capabilities) before letting you into the country.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Ensure the game feels native and optimized for whatever platform the player uses. Mobile players should get touch controls, PC players should get keyboard prompts, and everyone should get the best possible version of the experience their hardware can deliver.

**Why Platform Detection Matters?**
- **Input Method**: Touch vs keyboard vs gamepad require different UI
- **Performance**: Mobile needs lower settings, desktop can go higher
- **Features**: Some browsers don't support WebGPU or specific WebGL features
- **Layout**: Portrait vs landscape, screen size affects UI
- **Accessibility**: Some platforms have accessibility APIs to leverage

**Platform Categories**:
```
DESKTOP (Windows, macOS, Linux)
├── Keyboard + Mouse primary
├── Gamepad optional
├── Powerful GPU available
├── Large screen (1920x1080+)
└── Features: WebGPU, WebGL2, Workers

MOBILE (iOS, Android)
├── Touch input primary
├── No keyboard (usually)
├── Integrated GPU
├── Small screen (varies widely)
└── Features: Limited WebGL, no WebGPU (iOS)

TABLET (iPad, Android Tablets)
├── Touch input
├── Possible keyboard accessory
├── Mid-range GPU
├── Medium screen
└── Features: Hybrid of mobile/desktop

WEB (Various browsers)
├── Chrome: Best support
├── Firefox: Good support
├── Safari: Limited support
├── Edge: Chrome-based
└── Features: Varies by browser version
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding platform detection, you should know:
- **User Agent string** - Browser identification string (but unreliable)
- **Feature detection** - Testing for specific capabilities
- **Viewport** - Visible area of the browser window
- **Device Pixel Ratio** - Screen pixel density
- **Touch events** - API for touchscreen detection

### Core Architecture

```
PLATFORM DETECTION ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                  PLATFORM DETECTOR                      │
│  - Detect device type                                   │
│  - Detect browser                                       │
│  - Detect capabilities                                  │
│  - Cache results                                        │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   DEVICE      │  │   BROWSER    │  │   CAPABILITY │
│  - Desktop    │  │  - Chrome    │  │  - WebGL     │
│  - Mobile     │  │  - Firefox   │  │  - WebGPU    │
│  - Tablet     │  │  - Safari    │  │  - Workers   │
│  - VR/AR      │  │  - Edge      │  │  - Threads   │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   OUTPUT     │
                    │  - Platform  │
                    │  - Features  │
                    │  - Profile   │
                    └───────────────┘
```

### PlatformDetector Class

```javascript
class PlatformDetector {
  constructor() {
    // Cache results
    this.cache = null;
    this.detectionComplete = false;

    // Detection results
    this.results = {
      // Device type
      device: null,           // 'desktop', 'mobile', 'tablet', 'vr'
      os: null,               // 'windows', 'macos', 'linux', 'ios', 'android'

      // Browser
      browser: null,          // 'chrome', 'firefox', 'safari', 'edge'
      browserVersion: null,

      // Graphics
      webgl: null,            // 0, 1, or 2
      webgpu: false,
      maxTextureSize: null,
      gpuVendor: null,
      gpuRenderer: null,

      // Hardware
      cores: null,
      memory: null,           // GB
      pixelRatio: null,

      // Input
      touch: false,
      gamepad: false,
      keyboard: true,

      // Screen
      screenWidth: null,
      screenHeight: null,
      screenDepth: null,
      orientation: null,

      // Network
      connectionType: null,
      saveData: false,

      // Features
      workers: false,
      wasm: false,
      modules: false,
      sharedArrayBuffer: false,
      offscreenCanvas: false,

      // Limitations
      limitations: []         // Array of identified limitations
    };

    // Perform detection
    this.detect();
  }

  /**
   * Perform all platform detection
   */
  detect() {
    if (this.detectionComplete) return this.cache;

    // Detect in order of reliability
    this.detectCanvas();
    this.detectWebGL();
    this.detectBrowser();
    this.detectOS();
    this.detectDevice();
    this.detectScreen();
    this.detectInput();
    this.detectHardware();
    this.detectFeatures();
    this.detectNetwork();
    this.identifyLimitations();

    this.detectionComplete = true;
    this.cache = { ...this.results };

    return this.cache;
  }

  /**
   * Canvas detection (basic graphics support)
   */
  detectCanvas() {
    const canvas = document.createElement('canvas');
    this.results.canvasSupport = !!canvas.getContext;
  }

  /**
   * WebGL and GPU detection
   */
  detectWebGL() {
    const canvas = document.createElement('canvas');

    // Try WebGL 2 first
    let gl = canvas.getContext('webgl2');
    if (gl) {
      this.results.webgl = 2;
    } else {
      // Fall back to WebGL 1
      gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (gl) {
        this.results.webgl = 1;
      } else {
        this.results.webgl = 0;
      }
    }

    if (gl) {
      // Get GPU info
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');

      if (debugInfo) {
        this.results.gpuVendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
        this.results.gpuRenderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      }

      // Get max texture size
      this.results.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      this.results.maxRenderbufferSize = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE);

      // Get other limits
      this.results.maxVertexAttributes = gl.getParameter(gl.MAX_VERTEX_ATTRIBS);
      this.results.maxViewportDims = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
      this.results.maxCombinedTextureUnits = gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS);
    }

    // Check WebGPU
    this.results.webgpu = 'gpu' in navigator;
  }

  /**
   * Browser detection
   */
  detectBrowser() {
    const ua = navigator.userAgent;
    const vendor = navigator.vendor || '';

    // Chrome
    if (/Chrome|CriOS/.test(ua) && !/Edge|OPR/.test(ua)) {
      this.results.browser = 'chrome';
      const match = ua.match(/Chrome\/(\d+\.\d+\.\d+\.\d+)/);
      this.results.browserVersion = match ? match[1] : null;
    }
    // Firefox
    else if (/Firefox/.test(ua)) {
      this.results.browser = 'firefox';
      const match = ua.match(/Firefox\/(\d+\.\d+)/);
      this.results.browserVersion = match ? match[1] : null;
    }
    // Safari
    else if (/Safari/.test(ua) && /Apple Computer/.test(vendor)) {
      this.results.browser = 'safari';
      const match = ua.match(/Version\/(\d+\.\d+)/);
      this.results.browserVersion = match ? match[1] : null;
    }
    // Edge (Chromium)
    else if (/Edg/.test(ua)) {
      this.results.browser = 'edge';
      const match = ua.match(/Edg\/(\d+\.\d+\.\d+\.\d+)/);
      this.results.browserVersion = match ? match[1] : null;
    }
    // Opera
    else if (/OPR/.test(ua)) {
      this.results.browser = 'opera';
      const match = ua.match(/OPR\/(\d+\.\d+\.\d+\.\d+)/);
      this.results.browserVersion = match ? match[1] : null;
    }
    // Unknown
    else {
      this.results.browser = 'unknown';
    }

    // Check if in iframe
    this.results.inIframe = window.self !== window.top;

    // Check for private browsing (some browsers)
    this.detectPrivateBrowsing();
  }

  /**
   * Detect private browsing mode
   */
  detectPrivateBrowsing() {
    // This is unreliable and varies by browser
    try {
      this.results.privateBrowsing = false;

      // Safari (old method)
      if (this.results.browser === 'safari') {
        try {
          localStorage.setItem('test', 'test');
          localStorage.removeItem('test');
        } catch (e) {
          this.results.privateBrowsing = true;
        }
      }
    } catch (e) {
      // Detection failed
    }
  }

  /**
   * Operating system detection
   */
  detectOS() {
    const ua = navigator.userAgent;
    const platform = navigator.platform || '';

    // Windows
    if (/Win/.test(platform) || /Windows/.test(ua)) {
      this.results.os = 'windows';

      // Detect Windows version
      if (/Windows NT 10.0/.test(ua)) this.results.osVersion = '10';
      else if (/Windows NT 6.3/.test(ua)) this.results.osVersion = '8.1';
      else if (/Windows NT 6.2/.test(ua)) this.results.osVersion = '8';
      else if (/Windows NT 6.1/.test(ua)) this.results.osVersion = '7';
    }
    // macOS
    else if (/Mac/.test(platform) || /Macintosh/.test(ua)) {
      this.results.os = 'macos';

      // Detect macOS version
      const match = ua.match(/Mac OS X (\d+[_\.]\d+)/);
      if (match) {
        this.results.osVersion = match[1].replace('_', '.');
      }
    }
    // Linux
    else if (/Linux/.test(platform) && !/Android/.test(ua)) {
      this.results.os = 'linux';
    }
    // iOS
    else if (/iPhone|iPad|iPod/.test(ua) || /iPhone|iPad|iPod/.test(platform)) {
      this.results.os = 'ios';

      // Detect iOS version
      const match = ua.match(/OS (\d+_\d+)/);
      if (match) {
        this.results.osVersion = match[1].replace('_', '.');
      }
    }
    // Android
    else if (/Android/.test(ua)) {
      this.results.os = 'android';

      // Detect Android version
      const match = ua.match(/Android (\d+\.\d+)/);
      if (match) {
        this.results.osVersion = match[1];
      }
    }
    // Unknown
    else {
      this.results.os = 'unknown';
    }
  }

  /**
   * Device type detection
   */
  detectDevice() {
    const ua = navigator.userAgent;
    const screen = this.results.screen;

    // Mobile detection (comprehensive)
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua) ||
                     ('ontouchstart' in window) ||
                     (navigator.maxTouchPoints > 0);

    // Tablet vs phone distinction
    if (isMobile) {
      // iPad
      if (/iPad/.test(ua)) {
        this.results.device = 'tablet';
      }
      // Android tablet (usually larger screen)
      else if (/Android/.test(ua) && !/Mobile/.test(ua)) {
        this.results.device = 'tablet';
      }
      // iPhone or Android phone
      else {
        this.results.device = 'mobile';
      }
    }
    // Desktop
    else {
      this.results.device = 'desktop';
    }

    // Check for VR headsets
    this.detectVR();
  }

  /**
   * VR/AR device detection
   */
  detectVR() {
    // WebXR support
    if ('xr' in navigator) {
      navigator.xr.isSessionSupported('immersive-vr').then(supported => {
        if (supported) {
          this.results.vrCapable = true;
        }
      });
      navigator.xr.isSessionSupported('immersive-ar').then(supported => {
        if (supported) {
          this.results.arCapable = true;
        }
      });
    }

    // Oculus Quest via user agent
    if (/OculusBrowser/.test(navigator.userAgent)) {
      this.results.device = 'vr';
      this.results.vrDevice = 'oculus';
    }
  }

  /**
   * Screen detection
   */
  detectScreen() {
    this.results.screenWidth = window.screen.width;
    this.results.screenHeight = window.screen.height;
    this.results.screenColorDepth = window.screen.colorDepth;
    this.results.screenPixelDepth = window.screen.pixelDepth;

    // Viewport (actual visible area)
    this.results.viewportWidth = window.innerWidth;
    this.results.viewportHeight = window.innerHeight;

    // Device pixel ratio (for retina displays)
    this.results.pixelRatio = window.devicePixelRatio || 1;

    // Orientation
    this.results.orientation = this.getOrientation();

    // Screen size category
    const minDimension = Math.min(this.results.screenWidth, this.results.screenHeight);
    if (minDimension < 600) {
      this.results.screenSize = 'small';
    } else if (minDimension < 1200) {
      this.results.screenSize = 'medium';
    } else {
      this.results.screenSize = 'large';
    }

    // HDR support
    this.results.hdr = window.matchMedia('(dynamic-range: high)').matches;
  }

  /**
   * Get current orientation
   */
  getOrientation() {
    if (window.screen.orientation) {
      return window.screen.orientation.type;
    }
    return window.innerWidth < window.innerHeight ? 'portrait' : 'landscape';
  }

  /**
   * Input detection
   */
  detectInput() {
    // Touch support
    this.results.touch = 'ontouchstart' in window ||
                         navigator.maxTouchPoints > 0;

    if (this.results.touch) {
      this.results.maxTouchPoints = navigator.maxTouchPoints;
    }

    // Gamepad support
    this.results.gamepad = 'getGamepads' in navigator;

    // Keyboard (assume true unless explicitly disabled)
    // Hard to detect absence, so default to true
    this.results.keyboard = true;

    // Hover capability (distinguish mouse from touch)
    this.results.hover = window.matchMedia('(hover: hover)').matches;
    this.results.pointer = window.matchMedia('(pointer: fine)').matches ? 'precise' : 'coarse';
  }

  /**
   * Hardware detection
   */
  detectHardware() {
    // CPU cores
    this.results.cores = navigator.hardwareConcurrency || 4;

    // Memory (limited browser support)
    if (navigator.deviceMemory) {
      this.results.memory = navigator.deviceMemory;  // in GB
    }

    // Battery status (limited browser support)
    if ('getBattery' in navigator) {
      navigator.getBattery().then(battery => {
        this.results.battery = {
          level: battery.level,
          charging: battery.charging,
          chargingTime: battery.chargingTime,
          dischargingTime: battery.dischargingTime
        };

        // Listen for changes
        battery.addEventListener('levelchange', () => {
          if (this.results.battery) {
            this.results.battery.level = battery.level;
          }
        });
      }).catch(() => {
        // Battery API not supported or denied
      });
    }

    // Concurrency (for threading)
    this.results.logicalProcessors = navigator.hardwareConcurrency || 4;
  }

  /**
   * Feature detection
   */
  detectFeatures() {
    // Web Workers
    this.results.workers = typeof Worker !== 'undefined';

    // OffscreenCanvas
    this.results.offscreenCanvas = typeof OffscreenCanvas !== 'undefined';

    // WebAssembly
    this.results.wasm = typeof WebAssembly === 'object';

    // WebAssembly Modules (ESM integration)
    this.results.wasmModules = this.results.wasm && WebAssembly.compileStreaming;

    // SharedArrayBuffer (required for multi-threading)
    this.results.sharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';

    // Atomics (required for synchronization)
    this.results.atomics = typeof Atomics !== 'undefined';

    // BigInt
    this.results.bigInt = typeof BigInt !== 'undefined';

    // Gamepad API
    this.results.gamepadAPI = 'getGamepads' in navigator;

    // WebVR / WebXR
    this.results.webvr = 'getVRDisplays' in navigator;
    this.results.webxr = 'xr' in navigator;

    // Media Capabilities (for encoding/decoding)
    this.results.mediaCapabilities = 'mediaCapabilities' in navigator;

    // Content Visibility (performance optimization)
    this.results.contentVisibility = CSS.supports('content-visibility', 'auto');

    // Clipboard API
    this.results.clipboard = 'clipboard' in navigator;

    // File System Access API
    this.results.fileSystemAccess = 'showOpenFilePicker' in window;

    // WebCodecs (video encoding/decoding)
    this.results.webcodecs = 'VideoEncoder' in window;

    // Web Locks API
    this.results.webLocks = 'locks' in navigator;

    // Reporting API (for performance monitoring)
    this.results.reporting = 'ReportingObserver' in window;

    // Performance Observer API
    this.results.performanceObserver = 'PerformanceObserver' in window;
  }

  /**
   * Network detection
   */
  detectNetwork() {
    if (navigator.connection) {
      const conn = navigator.connection;

      this.results.connection = {
        type: conn.effectiveType,        // 'slow-2g', '2g', '3g', '4g'
        downlink: conn.downlink,          // Mbps
        rtt: conn.rtt,                    // Round-trip time (ms)
        saveData: conn.saveData           // Data saver mode
      };

      this.results.connectionType = conn.effectiveType;
      this.results.saveData = conn.saveData;

      // Listen for changes
      conn.addEventListener('change', () => {
        this.results.connection = {
          type: conn.effectiveType,
          downlink: conn.downlink,
          rtt: conn.rtt,
          saveData: conn.saveData
        };
      });
    }
  }

  /**
   * Identify platform limitations
   */
  identifyLimitations() {
    const limitations = this.results.limitations;

    // Mobile limitations
    if (this.results.device === 'mobile') {
      limitations.push('limited_memory');
      limitations.push('battery_constrained');
      limitations.push('touch_input_only');
    }

    // Browser-specific limitations
    if (this.results.browser === 'safari') {
      limitations.push('no_webgpu');
      limitations.push('indexeddb_quotas');
    }

    // Graphics limitations
    if (this.results.webgl < 2) {
      limitations.push('webgl1_only');
    }

    if (!this.results.webgpu) {
      limitations.push('no_webgpu');
    }

    // Low-end device indicators
    if (this.results.memory && this.results.memory < 4) {
      limitations.push('low_memory');
    }

    if (this.results.cores && this.results.cores < 4) {
      limitations.push('low_cpu_cores');
    }

    // Slow network
    if (this.results.connection) {
      if (this.results.connection.type === 'slow-2g' || this.results.connection.type === '2g') {
        limitations.push('slow_network');
      }
    }

    // No multi-threading
    if (!this.results.sharedArrayBuffer || !this.results.atomics) {
      limitations.push('no_multithreading');
    }

    // Data saver mode
    if (this.results.saveData) {
      limitations.push('data_saver_enabled');
    }
  }

  /**
   * Check if a specific feature is supported
   */
  hasFeature(feature) {
    if (!this.detectionComplete) {
      this.detect();
    }

    // Direct feature map
    const featureMap = {
      'webgl2': this.results.webgl >= 2,
      'webgpu': this.results.webgpu,
      'touch': this.results.touch,
      'gamepad': this.results.gamepad,
      'workers': this.results.workers,
      'wasm': this.results.wasm,
      'offscreen-canvas': this.results.offscreenCanvas,
      'shared-array-buffer': this.results.sharedArrayBuffer,
      'webxr': this.results.webxr
    };

    if (feature in featureMap) {
      return featureMap[feature];
    }

    // Check in limitations
    const negation = feature.startsWith('no_');
    const baseFeature = negation ? feature.substring(3) : feature;

    if (negation) {
      return !this.results.limitations.includes(baseFeature);
    }

    return this.results.limitations.includes(feature);
  }

  /**
   * Get recommended performance profile
   */
  getRecommendedProfile() {
    if (!this.detectionComplete) {
      this.detect();
    }

    // Mobile → mobile profile
    if (this.results.device === 'mobile') {
      return 'mobile';
    }

    // Tablet → laptop profile
    if (this.results.device === 'tablet') {
      return 'laptop';
    }

    // Low memory → laptop profile
    if (this.results.memory && this.results.memory < 8) {
      return 'laptop';
    }

    // No WebGPU or weak GPU → desktop profile
    if (!this.results.webgpu || this.results.webgl < 2) {
      return 'desktop';
    }

    // High-end → max profile
    if (this.results.memory >= 16 && this.results.webgl >= 2) {
      return 'max';
    }

    // Default to desktop
    return 'desktop';
  }

  /**
   * Get display string for platform
   */
  getDisplayString() {
    if (!this.detectionComplete) {
      this.detect();
    }

    return `${this.results.browser} ${this.results.browserVersion || ''} on ${this.results.os} ${this.results.osVersion || ''} (${this.results.device})`;
  }

  /**
   * Export as JSON
   */
  toJSON() {
    if (!this.detectionComplete) {
      this.detect();
    }
    return JSON.stringify(this.results, null, 2);
  }

  /**
   * Export specific subset of results
   */
  getResults(keys) {
    if (!this.detectionComplete) {
      this.detect();
    }

    const output = {};
    for (const key of keys) {
      if (key in this.results) {
        output[key] = this.results[key];
      }
    }
    return output;
  }
}

/**
 * Singleton instance
 */
let detectorInstance = null;

/**
 * Get the platform detector singleton
 */
function getPlatformDetector() {
  if (!detectorInstance) {
    detectorInstance = new PlatformDetector();
  }
  return detectorInstance;
}

export default PlatformDetector;
export { getPlatformDetector };
```

---

## 📝 How To Use Platform Detection

### Basic Usage

```javascript
import { getPlatformDetector } from './PlatformDetector.js';

// Detect platform
const platform = getPlatformDetector();

// Check device type
if (platform.results.device === 'mobile') {
  // Show touch controls
  touchJoysticks.enable();
}

// Check for specific features
if (platform.hasFeature('webgpu')) {
  // Use WebGPU renderer
  useWebGPURenderer();
} else if (platform.results.webgl >= 2) {
  // Fall back to WebGL 2
  useWebGL2Renderer();
}

// Get recommended profile
const profile = platform.getRecommendedProfile();
performanceManager.setProfile(profile);
```

### Feature Detection Pattern

```javascript
// Always use feature detection, not device detection
const supportsWebGL2 = () => {
  const canvas = document.createElement('canvas');
  return !!canvas.getContext('webgl2');
};

// Use it
if (supportsWebGL2()) {
  initAdvancedFeatures();
} else {
  useFallbackRendering();
}
```

---

## 🔧 Browser-Specific Quirks

### Safari (iOS and macOS)

**Limitations:**
- No WebGPU support (as of 2025)
- IndexedDB has strict quotas
- Auto-play restrictions for audio
- No SharedArrayBuffer without specific headers

**Workarounds:**
```javascript
// Check for Safari
const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

// Handle audio auto-play
if (isSafari) {
  // Require user interaction first
  document.addEventListener('click', unlockAudio, { once: true });
}
```

### Chrome/Edge (Chromium)

**Features:**
- Best WebGPU support
- Full WebGL2 support
- SharedArrayBuffer (with headers)
- Workers, OffscreenCanvas

**Requirements:**
```javascript
// COOP/COEP headers needed for SharedArrayBuffer
// Cross-Origin-Opener-Policy: same-origin
// Cross-Origin-Embedder-Policy: require-corp
```

### Firefox

**Features:**
- Good WebGL2 support
- WebGPU (behind flag in some versions)
- Strong privacy controls

**Gotchas:**
```javascript
// Private browsing mode limits storage
// Always handle QuotaExceededError
try {
  await saveToLocalStorage(data);
} catch (e) {
  if (e.name === 'QuotaExceededError') {
    // Use in-memory fallback
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Using User Agent for Feature Detection

```javascript
// ❌ WRONG: Assume Safari = no feature
const isSafari = /Safari/.test(navigator.userAgent);
if (isSafari) {
  disableFeature();
}

// ✅ CORRECT: Test the feature directly
const canvas = document.createElement('canvas');
if (!canvas.getContext('webgl2')) {
  useFallback();
}
```

### 2. Hardcoding for Specific Devices

```javascript
// ❌ WRONG: Assume all iOS devices are the same
if (platform.os === 'ios') {
  maxTextureSize = 4096;
}

// ✅ CORRECT: Detect actual capability
maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
```

### 3. Not Handling Orientation Changes

```javascript
// ❌ WRONG: Set once and forget
const isLandscape = window.innerWidth > window.innerHeight;

// ✅ CORRECT: Listen for changes
window.addEventListener('resize', () => {
  const isLandscape = window.innerWidth > window.innerHeight;
  updateLayout(isLandscape);
});
```

---

## Feature Detection Reference Table

| Feature | Detection Method | Fallback |
|---------|-----------------|----------|
| WebGL2 | `canvas.getContext('webgl2')` | WebGL1 |
| WebGPU | `'gpu' in navigator` | WebGL |
| Workers | `typeof Worker !== 'undefined'` | Main thread |
| WASM | `typeof WebAssembly === 'object'` | JS fallback |
| SharedArrayBuffer | `typeof SharedArrayBuffer !== 'undefined'` | Workers |
| OffscreenCanvas | `typeof OffscreenCanvas !== 'undefined'` | Canvas |
| Touch | `'ontouchstart' in window` | Mouse |
| Gamepad | `'getGamepads' in navigator` | Keyboard |

---

## Related Systems

- [Performance Profiles](./performance-profiles.md) - Hardware-based settings
- [Memory Management](./memory-management.md) - Resource allocation
- [Touch Joystick](../10-user-interface/touch-joystick.md) - Mobile controls

---

## Source File Reference

**Primary Files**:
- `../src/util/PlatformDetector.js` - Platform detection (estimated)

**Key Classes**:
- `PlatformDetector` - Complete platform detection
- `getPlatformDetector()` - Singleton accessor

**Dependencies**:
- Navigator APIs (browser info)
- WebGL/WebGPU APIs (graphics detection)
- Screen APIs (display info)

---

## References

- [MDN: Browser Detection](https://developer.mozilla.org/en-US/docs/Web/API/Navigator) - Navigator API
- [Feature Detection](https://developer.mozilla.org/en-US/docs/Learn/Tools_and_testing/Cross_browser_testing/Feature_detection) - Best practices
- [Device Memory API](https://developer.mozilla.org/en-US/docs/Web/API/DeviceMemory_API) - Memory detection
- [Network Information API](https://developer.mozilla.org/en-US/docs/Web/API/NetworkInformation) - Network status

*Documentation last updated: January 12, 2026*
