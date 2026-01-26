# Drawing Minigame - First Principles Guide

## Overview

The **Drawing Minigame** is an innovative gesture recognition system that allows players to draw symbols in 3D space using mouse or touch input. The game then recognizes these drawings using machine learning (TensorFlow.js) and triggers corresponding game events. This system transforms simple input into magical interaction - players aren't just clicking buttons, they're casting spells through gesture.

Think of the drawing minigame as **"incantation through gesture"** - just as a wizard might draw mystical symbols in the air, players draw runes that the game recognizes and responds to, creating a sense of genuine magical agency.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create empowerment through magical expression. Drawing symbols that the game recognizes makes players feel like they're actually performing magic, not just selecting menu options.

**Why a Drawing System?**
- **Tangible Magic**: Drawing feels more magical than clicking
- **Player Expression**: Each drawing is unique - personal connection
- **Gesture Recognition**: Real-time feedback creates anticipation ("Did it work?")
- **Accessibility**: Simple input (draw a circle) that anyone can understand

**Player Psychology**:
```
Prompt Appears → "Draw This Symbol" → Understanding
     ↓
Start Drawing → Visual Feedback (particles trail) → Immersion
     ↓
Complete Drawing → Anticipation (will it recognize?) → Tension
     ↓
Recognition Success → "I Drew That!" → Agency & Satisfaction
     ↓
Game Responds → World Changes → "My Magic Has Effect"
```

### Design Decisions

**1. 3D Drawing, Not 2D**
Drawing happens in 3D space, not on a 2D surface. This:
- Creates more connection to the 3D game world
- Allows for spatial gestures (drawing "in the air")
- Feels more magical/mystical

**2. Particle Trail Feedback**
As players draw, particles follow their cursor/stroke. This:
- Provides immediate visual confirmation
- Looks magical (sparkles, glow)
- Helps players see what they've drawn

**3. Forgiving Recognition**
The ML model recognizes imperfect drawings. A circle doesn't need to be perfect - close enough counts. This:
- Reduces frustration
- Maintains magical feeling (not a CAD tool)
- Accommodates different input methods (mouse vs touch)

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the drawing minigame, you should know:
- **TensorFlow.js** - Machine learning in the browser
- **Canvas API** - 2D drawing context for stroke capture
- **3D particle systems** - Visual feedback trails
- **Gesture recognition** - Converting strokes to predictions
- **Input events** - Mouse, touch, pointer events
- **Coordinate systems** - 2D canvas vs 3D world space

### Core Architecture

```
DRAWING MINIGAME SYSTEM ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                    DRAWING MANAGER                       │
│  - Manages drawing state and lifecycle                   │
│  - Coordinates input capture and rendering               │
│  - Triggers recognition on stroke completion             │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  STROKE DATA  │  │   PARTICLE    │  │   RECOGNITION│
│  - Capture    │  │   CANVAS 3D  │  │   MANAGER    │
│  - Normalize  │  │  - Trail      │  │  - ML Model  │
│  - Store      │  │  - Feedback   │  │  - Predict   │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   GAME STATE  │
                    │   CHANGE      │
                    │  - Trigger    │
                    │    event      │
                    └───────────────┘
```

### Drawing Manager Class

```javascript
class DrawingManager {
  constructor(options = {}) {
    this.scene = options.scene;
    this.camera = options.camera;
    this.gameManager = options.gameManager;
    this.recognitionManager = options.recognitionManager;
    this.particleCanvas = options.particleCanvas;

    // Drawing state
    this.isDrawing = false;
    this.currentStroke = [];
    this.allStrokes = [];  // For multi-stroke symbols

    // Configuration
    this.config = {
      minStrokeLength: 10,      // Minimum points to be valid
      maxStrokeLength: 500,     // Prevent infinite strokes
      strokeSmoothing: 0.5,     // Smooth input
      capturePlaneDistance: 3,  // Distance from camera
      autoRecognize: true,      // Auto-recognize on stroke end
      clearOnRecognize: true    // Clear drawing after success
    };

    // Visual feedback
    this.strokeMesh = null;
    this.strokeMaterial = null;

    // Events
    this.setupInputHandlers();
  }

  /**
   * Set up input event handlers
   */
  setupInputHandlers() {
    // Mouse events
    document.addEventListener('mousedown', (e) => this.onPointerDown(e));
    document.addEventListener('mousemove', (e) => this.onPointerMove(e));
    document.addEventListener('mouseup', (e) => this.onPointerUp(e));

    // Touch events
    document.addEventListener('touchstart', (e) => this.onTouchStart(e));
    document.addEventListener('touchmove', (e) => this.onTouchMove(e));
    document.addEventListener('touchend', (e) => this.onTouchEnd(e));

    // Pointer events (unified)
    document.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    document.addEventListener('pointermove', (e) => this.onPointerMove(e));
    document.addEventListener('pointerup', (e) => this.onPointerUp(e));
  }

  /**
   * Start drawing - called on pointer down
   */
  startDrawing(position) {
    if (this.isDrawing) return;

    this.isDrawing = true;
    this.currentStroke = [];

    // Add first point
    this.addStrokePoint(position);

    // Start particle trail
    if (this.particleCanvas) {
      this.particleCanvas.startTrail();
    }

    this.logger.log("Started drawing");
  }

  /**
   * Add a point to the current stroke
   */
  addStrokePoint(position) {
    // Position is in world coordinates
    const point = {
      x: position.x,
      y: position.y,
      z: position.z,
      timestamp: performance.now()
    };

    this.currentStroke.push(point);

    // Update visual feedback
    this.updateStrokeVisual();

    // Add particles
    if (this.particleCanvas) {
      this.particleCanvas.addParticle(position);
    }
  }

  /**
   * Continue drawing - called on pointer move
   */
  continueDrawing(position) {
    if (!this.isDrawing) return;

    // Throttle points (don't capture every frame)
    const lastPoint = this.currentStroke[this.currentStroke.length - 1];
    const distance = Math.sqrt(
      Math.pow(position.x - lastPoint.x, 2) +
      Math.pow(position.y - lastPoint.y, 2) +
      Math.pow(position.z - lastPoint.z, 2)
    );

    // Only add point if moved enough
    if (distance > 0.02) {  // 2cm minimum distance
      this.addStrokePoint(position);
    }
  }

  /**
   * End drawing - called on pointer up
   */
  endDrawing() {
    if (!this.isDrawing) return;

    this.isDrawing = false;

    // Stop particle trail
    if (this.particleCanvas) {
      this.particleCanvas.stopTrail();
    }

    // Check if stroke is valid
    if (this.currentStroke.length >= this.config.minStrokeLength) {
      this.completeStroke();
    } else {
      // Stroke too short, discard
      this.currentStroke = [];
      this.clearStrokeVisual();
    }

    this.logger.log("Ended drawing");
  }

  /**
   * Stroke completed - process for recognition
   */
  completeStroke() {
    // Normalize stroke for recognition
    const normalizedStroke = this.normalizeStroke(this.currentStroke);

    // Send to recognition manager
    if (this.config.autoRecognize && this.recognitionManager) {
      const result = this.recognitionManager.recognize(normalizedStroke);
      this.onRecognitionResult(result);
    }

    // Store stroke for multi-stroke symbols
    this.allStrokes.push(normalizedStroke);

    // Clear if configured
    if (this.config.clearOnRecognize) {
      this.clearStroke();
    }
  }

  /**
   * Normalize stroke for consistent recognition
   */
  normalizeStroke(stroke) {
    if (stroke.length === 0) return [];

    // Find bounding box
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (const point of stroke) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
      minZ = Math.min(minZ, point.z);
      maxZ = Math.max(maxZ, point.z);
    }

    // Calculate center and scale
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;

    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

    // Normalize to [-1, 1] range
    const normalized = stroke.map(point => ({
      x: ((point.x - centerX) / size) * 2,
      y: ((point.y - centerY) / size) * 2,
      z: ((point.z - centerZ) / size) * 2,
      timestamp: point.timestamp
    }));

    return normalized;
  }

  /**
   * Handle recognition result
   */
  onRecognitionResult(result) {
    if (result.confidence > 0.7) {
      // Recognition successful
      this.logger.log(`Recognized: ${result.label} (${result.confidence})`);

      // Trigger game event
      this.gameManager.emit(`drawing:recognized`, {
        label: result.label,
        confidence: result.confidence,
        stroke: this.currentStroke
      });

      // Visual feedback for success
      this.showSuccessEffect();

      // Clear drawing
      this.clearStroke();
    } else {
      // Recognition failed
      this.logger.log(`Not recognized (confidence: ${result.confidence})`);
      this.showFailureEffect();
    }
  }

  /**
   * Clear current drawing
   */
  clearStroke() {
    this.currentStroke = [];
    this.allStrokes = [];
    this.clearStrokeVisual();
  }

  /**
   * Input event handlers
   */
  onPointerDown(event) {
    // Calculate world position at draw plane
    const worldPos = this.getDrawPlanePosition(event);
    if (worldPos) {
      this.startDrawing(worldPos);
    }
  }

  onPointerMove(event) {
    if (!this.isDrawing) return;
    const worldPos = this.getDrawPlanePosition(event);
    if (worldPos) {
      this.continueDrawing(worldPos);
    }
  }

  onPointerUp(event) {
    this.endDrawing();
  }

  /**
   * Get world position on drawing plane
   */
  getDrawPlanePosition(event) {
    // Get normalized device coordinates
    const ndc = {
      x: (event.clientX / window.innerWidth) * 2 - 1,
      y: -(event.clientY / window.innerHeight) * 2 + 1
    };

    // Create ray from camera
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(ndc.x, ndc.y), this.camera);

    // Create drawing plane at configured distance
    const planePosition = new THREE.Vector3(0, 0, -this.config.capturePlaneDistance);
    const planeNormal = new THREE.Vector3(0, 0, 1);
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
      planeNormal,
      planePosition
    );

    // Find intersection
    const intersection = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, intersection);

    if (intersection) {
      return {
        x: intersection.x,
        y: intersection.y,
        z: intersection.z
      };
    }

    return null;
  }

  /**
   * Update stroke visualization
   */
  updateStrokeVisual() {
    if (!this.strokeMesh) {
      this.createStrokeMesh();
    }

    // Create line from current stroke points
    const points = this.currentStroke.map(p => new THREE.Vector3(p.x, p.y, p.z));

    if (points.length > 1) {
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      this.strokeMesh.geometry.dispose();
      this.strokeMesh.geometry = geometry;
    }
  }

  /**
   * Create stroke mesh
   */
  createStrokeMesh() {
    const material = new THREE.LineBasicMaterial({
      color: 0x00ff88,
      linewidth: 2,
      transparent: true,
      opacity: 0.8
    });

    this.strokeMesh = new THREE.Line(new THREE.BufferGeometry(), material);
    this.scene.add(this.strokeMesh);
  }

  /**
   * Clear stroke visualization
   */
  clearStrokeVisual() {
    if (this.strokeMesh) {
      this.strokeMesh.geometry.dispose();
      this.strokeMesh.geometry = new THREE.BufferGeometry();
    }
  }

  /**
   * Show success effect
   */
  showSuccessEffect() {
    // Burst particles, play sound, etc.
    if (this.particleCanvas) {
      this.particleCanvas.burst(this.currentStroke);
    }

    this.sfxManager?.play('drawing_success');
  }

  /**
   * Show failure effect
   */
  showFailureEffect() {
    // Shake effect, fade out, etc.
    this.sfxManager?.play('drawing_fail');
  }

  /**
   * Update loop
   */
  update(dt) {
    // Update particles
    if (this.particleCanvas) {
      this.particleCanvas.update(dt);
    }
  }

  /**
   * Enable/disable drawing mode
   */
  setEnabled(enabled) {
    this.enabled = enabled;
    if (!enabled) {
      this.clearStroke();
    }
  }

  /**
   * Clean up
   */
  destroy() {
    // Remove event listeners
    document.removeEventListener('pointerdown', this.onPointerDown);
    document.removeEventListener('pointermove', this.onPointerMove);
    document.removeEventListener('pointerup', this.onPointerUp);

    // Clear visuals
    if (this.strokeMesh) {
      this.scene.remove(this.strokeMesh);
      this.strokeMesh.geometry.dispose();
      this.strokeMesh.material.dispose();
    }
  }
}

export default DrawingManager;
```

---

## 📝 How To Build A Drawing System Like This

### Step 1: Define Your Symbols

What can players draw?

```javascript
const gameSymbols = {
  circle: {
    name: "Circle",
    examples: ["Perfect circle", "Oval", "Rough circle"],
    gameEffect: "shield_activate"
  },
  triangle: {
    name: "Triangle",
    examples: ["Upright", "Inverted", "Pointing right"],
    gameEffect: "fireball_cast"
  },
  lightning: {
    name: "Lightning Bolt",
    examples: ["Z-shape", "S-shape", "Jagged line"],
    gameEffect: "lightning_strike"
  },
  pentagram: {
    name: "Pentagram",
    examples: ["5-pointed star", "Rough pentagram"],
    gameEffect: "summon_demon"
  }
};
```

### Step 2: Capture Input

```javascript
// Basic input capture
const captureInput = (event) => {
  const point = {
    x: event.clientX,
    y: event.clientY,
    timestamp: performance.now()
  };

  currentStroke.push(point);

  // Visual feedback
  drawToCanvas(point);
};
```

### Step 3: Process for Recognition

```javascript
const processStroke = (stroke) => {
  // 1. Simplify (reduce number of points)
  const simplified = simplifyStroke(stroke, tolerance: 5);

  // 2. Normalize (scale to standard size)
  const normalized = normalizeStroke(simplified);

  // 3. Recognize
  const result = recognizer.recognize(normalized);

  return result;
};
```

---

## 🔧 Variations For Your Game

### Rhythm Drawing

```javascript
class RhythmDrawing {
  // Draw in time with music
  config = {
    rhythmWindow: 100,  // ms tolerance
    symbols: ["circle", "square", "triangle"],
    tempo: 120  // BPM
  };
}
```

### Sequenced Gestures

```javascript
class GestureSequence {
  // Draw symbols in specific order
  config = {
    sequence: ["circle", "triangle", "circle"],
    allowErrors: false,
    timeLimit: 5000
  };
}
```

### Collaborative Drawing

```javascript
class CollaborativeDrawing {
  // Multiple players draw together
  config = {
    players: ["player1", "player2"],
    sharedCanvas: true,
    mergeStrokes: true
  };
}
```

---

## Common Mistakes Beginners Make

### 1. No Visual Feedback

```javascript
// ❌ WRONG: Invisible drawing
onPointerMove(event) {
  currentStroke.push(event.position);
}
// Player can't see what they're drawing

// ✅ CORRECT: Immediate visual feedback
onPointerMove(event) {
  currentStroke.push(event.position);
  updateStrokeLine();  // Draw line
  addParticle(event.position);  // Add sparkle
}
// Clear visual confirmation
```

### 2. Too Strict Recognition

```javascript
// ❌ WRONG: Must be perfect
if (similarity < 0.95) return "not_recognized";
// Players can't get it to work

// ✅ CORRECT: Forgiving recognition
if (similarity > 0.6) return "recognized";
// Feels magical, not frustrating
```

### 3. Confusing Controls

```javascript
// ❌ WRONG: Unclear how to draw
// No prompt, no tutorial, no hint

// ✅ CORRECT: Clear instructions
showPrompt("Draw a circle in the air");
showGuide("Start drawing: Hold mouse button");
showGuide("Complete: Release when done");
```

---

## Performance Considerations

```
DRAWING SYSTEM PERFORMANCE:

Input Capture:
├── Events: Mouse/Touch/Pointer (minimal overhead)
├── Point storage: Array of objects (minimal)
└── Impact: Negligible

Visual Feedback:
├── Stroke line: Recreated each frame (minor)
├── Particles: Depends on count (moderate)
└── Impact: Minor to moderate

Recognition:
├── TensorFlow.js: Can be heavy on first load
├── Prediction: Usually fast (tens of ms)
└── Impact: Moderate (one-time model loading)

Optimization:
- Limit particle count
- Use simplified stroke geometry
- Pre-load ML model
- Use web workers for recognition
```

---

## Related Systems

- [ParticleCanvas3D](./particle-canvas-3d.md) - Visual particle feedback
- [DrawingRecognitionManager](./drawing-recognition-manager.md) - ML-based recognition
- [Stroke Data System](./stroke-data-system.md) - Stroke storage and format
- [InputManager](../04-input-physics/input-manager.md) - Unified input handling

---

## Source File Reference

**Primary Files**:
- `../src/content/DrawingManager.js` - Main drawing controller (estimated)
- `../src/content/DrawingRecognitionManager.js` - ML recognition (estimated)
- `../src/content/ParticleCanvas3D.js` - Particle effects (estimated)

**Key Classes**:
- `DrawingManager` - Drawing state and input coordination
- `StrokeData` - Stroke normalization and storage
- `ParticleCanvas3D` - Visual feedback system

**Dependencies**:
- Three.js (Line, BufferGeometry, Raycaster, Plane)
- TensorFlow.js (model recognition)
- Pointer Events API (input)

---

## References

- [TensorFlow.js](https://www.tensorflow.org/js) - Machine learning in JavaScript
- [Pointer Events API](https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events) - Unified input
- [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) - 2D drawing
- [Gesture Recognition](https://www.tensorflow.org/lite/models/gesture_recognition/overview) - ML models
- [Dollar Recognizer](https://depts.washington.edu/aimgroup/proj/dollar/) - Gesture recognition algorithm

*Documentation last updated: January 12, 2026*
