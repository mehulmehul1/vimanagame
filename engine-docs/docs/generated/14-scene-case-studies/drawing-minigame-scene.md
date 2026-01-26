# The Drawing Minigame - ML Recognition & 3D Particle Drawing

**Scene Case Study #17**

---

## What You Need to Know First

- **Gaussian Splatting Basics** (See: *Rendering System*)
- **Input Systems** (See: *InputManager*)
- **Canvas & Drawing** (See: *DrawingManager*)
- **Machine Learning Basics** (See: *DrawingRecognitionManager*)

---

## Scene Overview

| Property | Value |
|----------|-------|
| **Location** | Unlocks at specific narrative moment |
| **Narrative Context** | Supernatural drawing mechanic - player draws symbols that manifest in 3D |
| **Player Experience** | Drawing Gesture → Symbol Recognition → 3D Manifestation → World Change |
| **Atmosphere** | Magical, ritualistic, empowering, mysterious |
| **Technical Focus** | Gesture input, ML recognition, 3D particle effects, spatial drawing |

### The Scene's Purpose

The Drawing Minigame is a **supernatural gameplay mechanic** that:
1. **Empowers the player** - their actions have visible magical effects
2. **Creates ritual** - drawing symbols feels ceremonial
3. **Bridges 2D/3D** - flat drawings become 3D reality
4. **Progresses story** - specific symbols unlock narrative content

Unlike standard minigames, this is **integrated into the world** - the player isn't "playing a game" but **performing magic**. The drawing recognition makes the player feel like their intent matters, not just their precision.

---

## 🎮 Game Design Perspective

### Creative Intent

**Why a drawing mechanic? Why not simple button prompts or spell selection?**

| Interaction Type | Player Experience |
|-----------------|-------------------|
| **Button Prompt** | Mechanical - "I pressed X" |
| **Spell Menu** | Tactical - "I chose fireball" |
| **Drawing Gesture** | **Expressive - "I cast this spell"** |

The drawing mechanic creates **personal expression and intent**. When a player draws a symbol:
1. **They focus** - must concentrate on the shape
2. **They invest** - time and effort spent drawing
3. **They own it** - "I drew this, I made this happen"
4. **They remember** - muscle memory for symbols

### Design Philosophy

**Drawing as Magic Ritual:**

```
Simple input: "Press X to cast spell"
    ↓
Player thinks: "I triggered an effect"

Drawing input: "Draw the circle symbol"
    ↓
Player thinks: "I am performing a ritual. The shape matters.
                I am casting magic."
```

The additional effort creates **weight and meaning**. Magic shouldn't be trivial - it should feel like something you learned and practiced.

### Mood Building

The drawing minigame contributes to atmosphere through:

1. **Learning Curve** - Discovering symbols feels like uncovering secrets
2. **Mastery** - Getting better at drawing symbols feels like growing in power
3. **Visual Payoff** - 3D manifestation is spectacular
4. **Integration** - Symbols appear in world, not just UI

### Player Psychology

| Psychological Effect | How the Drawing Achieves It |
|---------------------|----------------------------|
| **Agency** | I drew this, I created this effect |
| **Competence** | I learned the symbols, I'm getting better |
| **Mystery** | What other symbols exist? What can they do? |
| **Empowerment** | My drawings change the world |

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
┌─────────────────────────────────────────────┐
│                                             │
│   [Drawing Interface Overlay]               │
│                                             │
│   ╔═══════════════════════════════════════╗ │
│   ║                                       ║ │
│   ║         [DRAWING CANVAS]              ║ │
│   ║                                       ║ │
│   ║           (Transparent)               ║ │
│   ║                                       ║ │
│   ║     Player draws symbols here         ║ │
│   ║                                       ║ │
│   ╚═══════════════════════════════════════╝ │
│                                             │
│   Background: 3D game world visible         │
│   Particles float up from completed drawings │
│   Recognized symbols glow and transform      │
└─────────────────────────────────────────────┘
```

### Player Path

```
1. Player discovers drawing ability (story event)
   ↓
2. Drawing interface unlocked
   ↓
3. First symbol revealed (tutorial)
   ↓
4. Player practices drawing
   ↓
5. Successful recognition → 3D particle effect
   ↓
6. Symbol manifests in world
   ↓
7. Story content unlocked
   ↓
8. New symbols discovered through exploration
   ↓
9. Player builds repertoire of magical symbols
```

### Symbol Examples

| Symbol | Shape | Effect | Narrative Context |
|--------|-------|--------|-------------------|
| **Circle** | Simple closed loop | Protection/Warding | First symbol, basic defense |
| **Spiral** | Inward spiral | Reveal/Insight | Shows hidden things |
| **Triangle** | Three-sided shape | Fire/Energy | Destructive magic |
| **Wave** | Sine wave pattern | Water/Flow | Movement and change |
| **Rune** | Complex pattern | Unique effect | Story-specific, powerful |

### Atmosphere Layers

| Layer | Elements |
|-------|----------|
| **Visual** | Glowing stroke, particles coalescing, 3D manifestation |
| **Audio** | Drawing scratch sound, recognition chime, manifestation rumble |
| **Tactile** | Haptic feedback on stroke (if supported) |
| **Spatial** | Drawing exists in 3D space, not just on screen |

---

## Technical Implementation

### Drawing Minigame Data Structure

```javascript
export const DRAWING_MINIGAME = {
  id: 'drawing_minigame',
  type: 'gameplay_system',

  // Canvas configuration
  canvas: {
    width: 512,
    height: 512,
    transparent: true,
    overlay: true,  // Shows over 3D scene
    inputLayer: 'top'
  },

  // Stroke configuration
  stroke: {
    width: 8,
    color: '#00ffff',  // Cyan glow
    glowIntensity: 0.8,
    smoothing: true,
    simplify: true,  // Reduce points for recognition
    minPoints: 10,   // Minimum points for valid stroke
    maxPoints: 500
  },

  // Recognition configuration
  recognition: {
    model: 'tensorflow_lite',
    confidence: {
      threshold: 0.6,  // Minimum confidence for match
      strictness: 0.8  // Higher = must be closer match
    },
    timeout: 5000,  // ms before auto-submit
    autoSubmit: true
  },

  // Known symbols
  symbols: [
    {
      id: 'circle',
      name: 'Circle of Protection',
      description: 'A simple closed loop',
      unlockTrigger: 'tutorial_complete',
      effect: 'protection_ward',
      template: generateCircleTemplate(),
      variations: ['circle', 'oval', 'rounded']
    },
    {
      id: 'spiral',
      name: 'Spiral of Insight',
      description: 'A spiral starting from outside, going in',
      unlockTrigger: 'chapter_2',
      effect: 'reveal_hidden',
      template: generateSpiralTemplate(),
      variations: ['spiral_cw', 'spiral_ccw']
    },
    {
      id: 'triangle',
      name: 'Triangle of Fire',
      description: 'A three-sided pointed shape',
      unlockTrigger: 'chapter_3',
      effect: 'fire_blast',
      template: generateTriangleTemplate(),
      variations: ['triangle_up', 'triangle_down']
    }
  ],

  // Particle effect configuration
  particles: {
    count: 100,
    lifetime: 2.0,
    colors: ['#00ffff', '#0088ff', '#ffffff'],
    size: { min: 2, max: 8 },
    speed: { min: 1, max: 3 },
    behavior: 'coalesce_and_manifest'
  },

  // 3D manifestation
  manifestation: {
    scale: 1.0,
    duration: 1000,  // ms to fully manifest
    glow: true,
    float: true,
    hoverHeight: 1.5
  },

  // Audio configuration
  audio: {
    drawing: {
      stroke: '/assets/audio/drawing/stroke.ogg',
      loop: true,
      volume: 0.3
    },
    recognition: {
      success: '/assets/audio/drawing/success.ogg',
      fail: '/assets/audio/drawing/fail.ogg'
    },
    manifestation: {
      buildup: '/assets/audio/drawing/buildup.ogg',
      complete: '/assets/audio/drawing/complete.ogg'
    }
  },

  // Tutorial progression
  tutorial: {
    firstSymbol: 'circle',
    guidance: {
      showTemplate: true,
      showPath: true,
      allowTracing: true
    },
    attempts: 3,
    strictness: 0.5  // More lenient during tutorial
  }
};

// Helper functions to generate symbol templates
function generateCircleTemplate() {
  const points = [];
  const segments = 32;
  const radius = 100;

  for (let i = 0; i <= segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    points.push({
      x: Math.cos(angle) * radius + 256,
      y: Math.sin(angle) * radius + 256
    });
  }

  return points;
}

function generateSpiralTemplate() {
  const points = [];
  const turns = 3;
  const segmentsPerTurn = 16;
  const maxRadius = 120;

  for (let i = 0; i <= turns * segmentsPerTurn; i++) {
    const progress = i / (turns * segmentsPerTurn);
    const angle = progress * turns * Math.PI * 2;
    const radius = maxRadius * (1 - progress);

    points.push({
      x: Math.cos(angle) * radius + 256,
      y: Math.sin(angle) * radius + 256
    });
  }

  return points;
}

function generateTriangleTemplate() {
  const points = [];
  const sides = 3;
  const segmentsPerSide = 10;
  const radius = 100;

  for (let i = 0; i <= sides * segmentsPerSide; i++) {
    const sideProgress = (i % segmentsPerSide) / segmentsPerSide;
    const sideIndex = Math.floor(i / segmentsPerSide);

    const angle1 = (sideIndex / sides) * Math.PI * 2 - Math.PI / 2;
    const angle2 = ((sideIndex + 1) / sides) * Math.PI * 2 - Math.PI / 2;

    const x1 = Math.cos(angle1) * radius + 256;
    const y1 = Math.sin(angle1) * radius + 256;
    const x2 = Math.cos(angle2) * radius + 256;
    const y2 = Math.sin(angle2) * radius + 256;

    points.push({
      x: x1 + (x2 - x1) * sideProgress,
      y: y1 + (y2 - y1) * sideProgress
    });
  }

  return points;
}
```

### Drawing Manager

```javascript
/**
 * Manages drawing canvas and stroke capture
 */
class DrawingManager {
  constructor(scene) {
    this.scene = scene;
    this.canvas = null;
    this.ctx = null;
    this.currentStroke = null;
    this.strokes = [];
    this.isDrawing = false;
    this.onStrokeComplete = null;
  }

  /**
   * Initialize drawing system
   */
  async init(config) {
    this.config = config;

    // Create canvas element
    this.canvas = document.createElement('canvas');
    this.canvas.width = config.canvas.width;
    this.canvas.height = config.canvas.height;
    this.canvas.style.position = 'absolute';
    this.canvas.style.top = '0';
    this.canvas.style.left = '0';
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.pointerEvents = 'auto';
    this.canvas.style.zIndex = '1000';
    this.canvas.classList.add('drawing-canvas');

    // Get 2D context
    this.ctx = this.canvas.getContext('2d');
    this.setupContext();

    // Setup input handlers
    this.setupInputHandlers();

    // Initially hidden
    this.canvas.style.display = 'none';

    return this.canvas;
  }

  /**
   * Setup canvas context
   */
  setupContext() {
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.strokeStyle = this.config.stroke.color;
    this.ctx.lineWidth = this.config.stroke.width;
    this.ctx.shadowBlur = 10;
    this.ctx.shadowColor = this.config.stroke.color;
  }

  /**
   * Setup input handlers for drawing
   */
  setupInputHandlers() {
    // Mouse events
    this.canvas.addEventListener('mousedown', (e) => this.startStroke(e));
    this.canvas.addEventListener('mousemove', (e) => this.continueStroke(e));
    this.canvas.addEventListener('mouseup', () => this.endStroke());
    this.canvas.addEventListener('mouseleave', () => this.endStroke());

    // Touch events
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.startStroke(e.touches[0]);
    });
    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      this.continueStroke(e.touches[0]);
    });
    this.canvas.addEventListener('touchend', () => this.endStroke());
  }

  /**
   * Start a new stroke
   */
  startStroke(event) {
    this.isDrawing = true;
    const pos = this.getCanvasPosition(event);

    this.currentStroke = {
      points: [pos],
      startTime: performance.now(),
      color: this.config.stroke.color
    };

    // Start drawing sound
    this.scene.audio.playOneShot(this.config.audio.drawing.stroke, {
      loop: true,
      volume: this.config.audio.drawing.volume
    });

    // Clear canvas for new stroke
    this.clearCanvas();
  }

  /**
   * Continue current stroke
   */
  continueStroke(event) {
    if (!this.isDrawing || !this.currentStroke) return;

    const pos = this.getCanvasPosition(event);
    this.currentStroke.points.push(pos);

    // Draw stroke
    this.drawStroke(this.currentStroke);
  }

  /**
   * End current stroke
   */
  endStroke() {
    if (!this.isDrawing || !this.currentStroke) return;

    this.isDrawing = false;

    // Stop drawing sound
    this.scene.audio.stopOneShot(this.config.audio.drawing.stroke);

    // Simplify stroke (reduce points)
    this.currentStroke.points = this.simplifyStroke(
      this.currentStroke.points,
      this.config.stroke.simplify ? 0.5 : 0
    );

    // Store stroke
    this.strokes.push(this.currentStroke);

    // Notify completion
    if (this.onStrokeComplete) {
      this.onStrokeComplete(this.currentStroke);
    }

    this.currentStroke = null;
  }

  /**
   * Get canvas coordinates from event
   */
  getCanvasPosition(event) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;

    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
      timestamp: performance.now()
    };
  }

  /**
   * Draw stroke to canvas
   */
  drawStroke(stroke) {
    if (stroke.points.length < 2) return;

    this.ctx.beginPath();
    this.ctx.moveTo(stroke.points[0].x, stroke.points[0].y);

    for (let i = 1; i < stroke.points.length; i++) {
      // Apply smoothing if enabled
      if (this.config.stroke.smoothing && i > 1) {
        const prev = stroke.points[i - 1];
        const curr = stroke.points[i];
        const next = stroke.points[i + 1] || curr;

        const cpX = (prev.x + curr.x) / 2;
        const cpY = (prev.y + curr.y) / 2;

        this.ctx.quadraticCurveTo(prev.x, prev.y, cpX, cpY);
      } else {
        this.ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
      }
    }

    this.ctx.stroke();
  }

  /**
   * Simplify stroke using Douglas-Peucker algorithm
   */
  simplifyStroke(points, tolerance) {
    if (points.length < 3) return points;

    // Find point with maximum distance
    let maxDist = 0;
    let maxIndex = 0;
    const start = points[0];
    const end = points[points.length - 1];

    for (let i = 1; i < points.length - 1; i++) {
      const dist = this.perpendicularDistance(points[i], start, end);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }

    // Recursively simplify
    if (maxDist > tolerance) {
      const left = this.simplifyStroke(points.slice(0, maxIndex + 1), tolerance);
      const right = this.simplifyStroke(points.slice(maxIndex), tolerance);
      return left.slice(0, -1).concat(right);
    } else {
      return [start, end];
    }
  }

  /**
   * Calculate perpendicular distance from point to line
   */
  perpendicularDistance(point, lineStart, lineEnd) {
    const dx = lineEnd.x - lineStart.x;
    const dy = lineEnd.y - lineStart.y;
    const len = Math.sqrt(dx * dx + dy * dy);

    if (len === 0) return Math.sqrt(
      (point.x - lineStart.x) ** 2 +
      (point.y - lineStart.y) ** 2
    );

    const t = Math.max(0, Math.min(1,
      ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / (len * len)
    ));

    const projX = lineStart.x + t * dx;
    const projY = lineStart.y + t * dy;

    return Math.sqrt(
      (point.x - projX) ** 2 +
      (point.y - projY) ** 2
    );
  }

  /**
   * Clear canvas
   */
  clearCanvas() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Show drawing interface
   */
  show() {
    this.canvas.style.display = 'block';
  }

  /**
   * Hide drawing interface
   */
  hide() {
    this.canvas.style.display = 'none';
  }

  /**
   * Get captured stroke data
   */
  getStrokes() {
    return this.strokes;
  }

  /**
   * Clear all strokes
   */
  reset() {
    this.strokes = [];
    this.clearCanvas();
  }

  /**
   * Draw template symbol for guidance
   */
  drawTemplate(points) {
    this.ctx.save();
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([5, 5]);

    this.ctx.beginPath();
    if (points.length > 0) {
      this.ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        this.ctx.lineTo(points[i].x, points[i].y);
      }
    }
    this.ctx.stroke();

    this.ctx.restore();
  }
}
```

### Drawing Recognition Manager

```javascript
/**
 * Recognizes drawn symbols using machine learning
 */
class DrawingRecognitionManager {
  constructor(scene) {
    this.scene = scene;
    this.model = null;
    this.symbols = new Map();
  }

  /**
   * Initialize recognition system
   */
  async init(config) {
    this.config = config;

    // Load TensorFlow.js
    await this.loadTensorFlow();

    // Load or create recognition model
    this.model = await this.loadOrCreateModel(config.recognition.model);

    // Register symbols
    for (const symbol of config.symbols) {
      this.symbols.set(symbol.id, symbol);
    }

    return this.model;
  }

  /**
   * Load TensorFlow.js
   */
  async loadTensorFlow() {
    if (typeof tf !== 'undefined') return;

    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js';
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  /**
   * Load or create recognition model
   */
  async loadOrCreateModel(modelType) {
    // For this implementation, we'll use a simpler gesture recognition
    // In production, you might train a custom model

    return {
      recognize: (stroke) => this.recognizeSymbol(stroke)
    };
  }

  /**
   * Recognize symbol from stroke
   */
  recognizeSymbol(stroke) {
    const results = [];

    for (const [id, symbol] of this.symbols) {
      // Skip if not unlocked
      if (symbol.unlockTrigger &&
          !this.scene.gameManager.getCriteria(symbol.unlockTrigger)) {
        continue;
      }

      // Calculate similarity score
      const score = this.calculateSimilarity(stroke, symbol);

      results.push({
        id: symbol.id,
        symbol: symbol,
        score: score
      });
    }

    // Sort by score
    results.sort((a, b) => b.score - a.score);

    // Return best match if confident enough
    if (results.length > 0 && results[0].score >= this.config.recognition.confidence.threshold) {
      return {
        recognized: true,
        symbol: results[0].symbol,
        confidence: results[0].score
      };
    }

    return {
      recognized: false,
      symbol: null,
      confidence: 0
    };
  }

  /**
   * Calculate similarity between stroke and symbol template
   */
  calculateSimilarity(stroke, symbol) {
    const strokePoints = stroke.points;
    const templatePoints = symbol.template;

    if (strokePoints.length < this.config.stroke.minPoints) return 0;

    // Normalize both strokes
    const normalizedStroke = this.normalizeStroke(strokePoints);
    const normalizedTemplate = this.normalizeStroke(templatePoints);

    // Calculate shape similarity using multiple metrics
    const shapeScore = this.calculateShapeSimilarity(
      normalizedStroke,
      normalizedTemplate
    );

    const directionScore = this.calculateDirectionSimilarity(
      normalizedStroke,
      normalizedTemplate
    );

    // Weighted combination
    return shapeScore * 0.7 + directionScore * 0.3;
  }

  /**
   * Normalize stroke to standard position and scale
   */
  normalizeStroke(points) {
    if (points.length === 0) return [];

    // Calculate bounding box
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (const point of points) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }

    // Calculate scale and offset
    const width = maxX - minX;
    const height = maxY - minY;
    const scale = Math.max(width, height) || 1;
    const targetSize = 200;

    // Normalize points
    const normalized = points.map(point => ({
      x: (point.x - minX - width / 2) * (targetSize / scale) + targetSize,
      y: (point.y - minY - height / 2) * (targetSize / scale) + targetSize
    }));

    return normalized;
  }

  /**
   * Calculate shape similarity using Hausdorff distance
   */
  calculateShapeSimilarity(stroke1, stroke2) {
    // Resample to same number of points
    const sampleCount = 32;
    const resampled1 = this.resampleStroke(stroke1, sampleCount);
    const resampled2 = this.resampleStroke(stroke2, sampleCount);

    // Calculate average distance between corresponding points
    let totalDistance = 0;
    for (let i = 0; i < sampleCount; i++) {
      const dx = resampled1[i].x - resampled2[i].x;
      const dy = resampled1[i].y - resampled2[i].y;
      totalDistance += Math.sqrt(dx * dx + dy * dy);
    }

    const avgDistance = totalDistance / sampleCount;
    const maxDistance = 50; // Maximum expected distance

    return Math.max(0, 1 - avgDistance / maxDistance);
  }

  /**
   * Calculate drawing direction similarity
   */
  calculateDirectionSimilarity(stroke1, stroke2) {
    // Compare drawing directions between adjacent points
    const dirs1 = this.getDirections(stroke1);
    const dirs2 = this.getDirections(stroke2);

    // Align by finding best rotation
    let bestScore = 0;
    for (let offset = 0; offset < dirs2.length; offset++) {
      let matches = 0;
      for (let i = 0; i < dirs1.length; i++) {
        const idx2 = (i + offset) % dirs2.length;
        if (Math.abs(dirs1[i] - dirs2[idx2]) < Math.PI / 4) {
          matches++;
        }
      }
      bestScore = Math.max(bestScore, matches / dirs1.length);
    }

    return bestScore;
  }

  /**
   * Get direction angles between consecutive points
   */
  getDirections(stroke) {
    const directions = [];

    for (let i = 1; i < stroke.length; i++) {
      const dx = stroke[i].x - stroke[i - 1].x;
      const dy = stroke[i].y - stroke[i - 1].y;
      directions.push(Math.atan2(dy, dx));
    }

    return directions;
  }

  /**
   * Resample stroke to have exactly n points
   */
  resampleStroke(stroke, n) {
    if (stroke.length < 2) return stroke;

    const resampled = [stroke[0]];
    const totalLength = this.calculateStrokeLength(stroke);
    const targetSegmentLength = totalLength / (n - 1);

    let accumulatedLength = 0;
    let lastPoint = stroke[0];

    for (let i = 1; i < stroke.length && resampled.length < n; i++) {
      const dx = stroke[i].x - lastPoint.x;
      const dy = stroke[i].y - lastPoint.y;
      const segmentLength = Math.sqrt(dx * dx + dy * dy);

      accumulatedLength += segmentLength;

      while (accumulatedLength >= targetSegmentLength && resampled.length < n) {
        const ratio = (accumulatedLength - targetSegmentLength) / segmentLength;
        resampled.push({
          x: lastPoint.x + dx * ratio,
          y: lastPoint.y + dy * ratio
        });
        accumulatedLength -= targetSegmentLength;
      }

      lastPoint = stroke[i];
    }

    while (resampled.length < n) {
      resampled.push(stroke[stroke.length - 1]);
    }

    return resampled;
  }

  /**
   * Calculate total stroke length
   */
  calculateStrokeLength(stroke) {
    let length = 0;

    for (let i = 1; i < stroke.length; i++) {
      const dx = stroke[i].x - stroke[i - 1].x;
      const dy = stroke[i].y - stroke[i - 1].y;
      length += Math.sqrt(dx * dx + dy * dy);
    }

    return length;
  }
}
```

### Particle Canvas 3D

```javascript
/**
 * 3D particle effects for drawing manifestation
 */
class ParticleCanvas3D {
  constructor(scene) {
    this.scene = scene;
    this.particles = [];
    this.maxParticles = 5000;
  }

  /**
   * Create particles from stroke
   */
  createFromStroke(stroke, config) {
    const particleCount = config.particles.count;
    const strokeBounds = this.getBounds(stroke.points);

    const particles = [];

    for (let i = 0; i < particleCount; i++) {
      const point = stroke.points[Math.floor(Math.random() * stroke.points.length)];

      const particle = {
        position: new THREE.Vector3(
          (point.x - 256) * 0.01,
          (point.y - 256) * -0.01,  // Flip Y
          0
        ),
        velocity: new THREE.Vector3(
          (Math.random() - 0.5) * config.particles.speed.min,
          Math.random() * config.particles.speed.max,
          (Math.random() - 0.5) * config.particles.speed.min
        ),
        color: this.randomColor(config.particles.colors),
        size: this.randomRange(config.particles.size),
        lifetime: config.particles.lifetime,
        age: 0,
        behavior: config.particles.behavior
      };

      particles.push(particle);
    }

    return particles;
  }

  /**
   * Get bounding box of stroke points
   */
  getBounds(points) {
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (const point of points) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }

    return { minX, minY, maxX, maxY };
  }

  /**
   * Update particles
   */
  update(deltaTime) {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const particle = this.particles[i];

      // Update age
      particle.age += deltaTime;

      // Remove dead particles
      if (particle.age >= particle.lifetime) {
        this.particles.splice(i, 1);
        continue;
      }

      // Update based on behavior
      switch (particle.behavior) {
        case 'coalesce_and_manifest':
          this.updateCoalescing(particle, deltaTime);
          break;
        case 'explode':
          this.updateExploding(particle, deltaTime);
          break;
        case 'float_up':
          this.updateFloating(particle, deltaTime);
          break;
      }

      // Apply velocity
      particle.position.add(particle.velocity.clone().multiplyScalar(deltaTime));
    }
  }

  /**
   * Update coalescing particles (forming 3D shape)
   */
  updateCoalescing(particle, deltaTime) {
    const progress = particle.age / particle.lifetime;

    // Slow down as they form the shape
    particle.velocity.multiplyScalar(0.95);

    // Fade in then out
    if (progress > 0.7) {
      particle.size *= 0.98;
    }
  }

  /**
   * Update exploding particles
   */
  updateExploding(particle, deltaTime) {
    // Accelerate outward
    particle.velocity.multiplyScalar(1.02);
  }

  /**
   * Update floating particles
   */
  updateFloating(particle, deltaTime) {
    // Float upward with slight wobble
    particle.velocity.y = 1;
    particle.velocity.x += Math.sin(particle.age * 3) * 0.1;
  }

  /**
   * Render particles
   */
  render(renderer, camera) {
    // Render particles as points or sprites
    // Implementation depends on rendering system
  }

  /**
   * Get random color from array
   */
  randomColor(colors) {
    return colors[Math.floor(Math.random() * colors.length)];
  }

  /**
   * Get random number in range
   */
  randomRange(range) {
    return range.min + Math.random() * (range.max - range.min);
  }
}
```

### Complete Drawing Minigame Manager

```javascript
/**
 * Orchestrates the complete drawing minigame
 */
class DrawingMinigameManager {
  constructor(scene) {
    this.scene = scene;
    this.drawing = null;
    this.recognition = null;
    this.particles = null;
    this.isActive = false;
    this.currentSymbol = null;
  }

  /**
   * Initialize all systems
   */
  async init(config) {
    this.config = config;

    // Initialize drawing
    this.drawing = new DrawingManager(this.scene);
    await this.drawing.init(config);

    // Initialize recognition
    this.recognition = new DrawingRecognitionManager(this.scene);
    await this.recognition.init(config);

    // Initialize particles
    this.particles = new ParticleCanvas3D(this.scene);

    // Setup stroke completion handler
    this.drawing.onStrokeComplete = (stroke) => this.onStrokeComplete(stroke);

    // Add canvas to DOM
    document.body.appendChild(this.drawing.canvas);

    return this;
  }

  /**
   * Start drawing minigame
   */
  start(symbolId = null) {
    this.isActive = true;
    this.currentSymbol = symbolId;
    this.drawing.show();
    this.drawing.reset();

    // Show template if specified
    if (symbolId) {
      const symbol = this.recognition.symbols.get(symbolId);
      if (symbol) {
        this.drawing.drawTemplate(symbol.template);
      }
    }
  }

  /**
   * Handle stroke completion
   */
  async onStrokeComplete(stroke) {
    // Recognize symbol
    const result = this.recognition.recognizeSymbol(stroke);

    if (result.recognized) {
      await this.onRecognitionSuccess(result.symbol);
    } else {
      await this.onRecognitionFail();
    }
  }

  /**
   * Handle successful recognition
   */
  async onRecognitionSuccess(symbol) {
    // Play success sound
    this.scene.audio.playOneShot(this.config.audio.recognition.success);

    // Create particle effect
    const stroke = this.drawing.strokes[this.drawing.strokes.length - 1];
    const particles = this.particles.createFromStroke(stroke, this.config);

    // Play manifestation buildup
    this.scene.audio.playOneShot(this.config.audio.manifestation.buildup);

    // Animate 3D manifestation
    await this.manifestSymbol3D(symbol, particles);

    // Play completion sound
    this.scene.audio.playOneShot(this.config.audio.manifestation.complete);

    // Apply symbol effect
    await this.applySymbolEffect(symbol);

    // Hide drawing interface
    this.drawing.hide();
    this.isActive = false;
  }

  /**
   * Handle recognition failure
   */
  async onRecognitionFail() {
    // Play fail sound
    this.scene.audio.playOneShot(this.config.audio.recognition.fail);

    // Show feedback
    this.showFeedback('Not recognized. Try again.');

    // Clear after delay
    setTimeout(() => {
      this.drawing.reset();
    }, 1000);
  }

  /**
   * Manifest symbol in 3D
   */
  async manifestSymbol3D(symbol, particles) {
    const config = this.config.manifestation;
    const duration = config.duration;
    const startTime = performance.now();

    // Create 3D mesh for symbol
    const geometry = new THREE.ShapeGeometry(
      this.createShapeFromPoints(symbol.template)
    );
    const material = new THREE.MeshBasicMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(geometry, material);

    mesh.position.set(
      this.scene.player.position.x,
      config.hoverHeight,
      this.scene.player.position.z + 1
    );

    this.scene.add(mesh);

    // Animate manifestation
    return new Promise((resolve) => {
      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Fade in and scale up
        material.opacity = progress;
        mesh.scale.setScalar(progress);

        // Update particles
        this.particles.update(0.016);

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          // Keep symbol visible briefly
          setTimeout(() => {
            this.scene.remove(mesh);
            resolve();
          }, 2000);
        }
      };

      animate();
    });
  }

  /**
   * Create THREE.Shape from points
   */
  createShapeFromPoints(points) {
    const shape = new THREE.Shape();

    if (points.length > 0) {
      shape.moveTo(
        (points[0].x - 256) * 0.01,
        (points[0].y - 256) * -0.01
      );

      for (let i = 1; i < points.length; i++) {
        shape.lineTo(
          (points[i].x - 256) * 0.01,
          (points[i].y - 256) * -0.01
        );
      }
    }

    return shape;
  }

  /**
   * Apply symbol effect
   */
  async applySymbolEffect(symbol) {
    // Trigger game event
    this.scene.gameManager.emit(`symbol_${symbol.id}`, symbol);

    // Set criteria
    this.scene.gameManager.setCriteria(`drew_${symbol.id}`, true);

    console.log(`Symbol effect applied: ${symbol.effect}`);
  }

  /**
   * Show feedback message
   */
  showFeedback(message) {
    // Implement UI feedback
    console.log(message);
  }

  /**
   * Update loop
   */
  update(deltaTime) {
    if (this.isActive) {
      // Update particles
      this.particles.update(deltaTime);
    }
  }
}
```

---

## How To Build A Scene Like This

### Step 1: Define the Magic System

```javascript
const magicSystem = {
  question: 'What does drawing accomplish?',

  answers: [
    'Unlock story content',
    'Defeat enemies',
    'Reveal secrets',
    'Change environment'
  ],

  choice: 'Each symbol should have a unique purpose'
};
```

### Step 2: Design Recognizable Symbols

```javascript
const symbolDesign = {
  principles: [
    'Simple shapes are easier to draw',
    'Distinct shapes avoid confusion',
    'Cultural symbols are intuitive',
    '5-10 symbols maximum for recall'
  ],

  examples: {
    circle: 'Protection, completeness',
    spiral: 'Change, insight',
    triangle: 'Power, direction'
  }
};
```

### Step 3: Implement Stroke Capture

```javascript
const strokeCapture = {
  input: 'Mouse/touch to canvas',

  storage: 'Array of {x, y, timestamp} points',

  processing: 'Simplify, normalize, resample for recognition'
};
```

### Step 4: Create Recognition Algorithm

```javascript
const recognition = {
  approach: 'Shape matching',

  methods: [
    'Template matching',
    'Neural network',
    'Gesture recognizer'
  ],

  confidence: 'Score similarity, require threshold'
};
```

### Step 5: Design Visual Feedback

```javascript
const feedback = {
  drawing: 'Glowing stroke, particle trail',

  recognition: 'Sound, visual confirmation',

  manifestation: '3D formation, spectacular effect'
};
```

---

## Variations For Your Game

### Variation 1: Combat Spells

```javascript
const combatVersion = {
  symbols: 'Attack, defend, heal spells',

  timing: 'Must draw quickly during combat',

  consequence: 'Failure = vulnerability'
};
```

### Variation 2: Puzzle Codes

```javascript
const puzzleVersion = {
  symbols: 'Puzzle solutions',

  discovery: 'Find symbols in environment',

  application: 'Draw to unlock doors, devices'
};
```

### Variation 3: Creation System

```javascript
const creationVersion = {
  symbols: 'Create items, creatures',

  complexity: 'Combine symbols for new effects',

  experimentation: 'Discover combinations'
};
```

---

## Performance Considerations

```javascript
const optimization = {
  particles: 'Limit count, use GPU instancing',

  recognition: 'Cache results, batch processing',

  canvas: 'Use offscreen canvas for processing'
};
```

---

## Common Mistakes Beginners Make

### Mistake 1: Symbols Too Complex

```javascript
// BAD: Intricate symbols
const badSymbols = {
  complexity: 'Detailed, hard to draw',
  result: 'Frustrating, low recognition rate'
};

// GOOD: Simple, distinct shapes
const goodSymbols = {
  complexity: 'Simple geometric shapes',
  result: 'Satisfying, high recognition rate'
};
```

### Mistake 2: No Feedback

```javascript
// BAD: Silent recognition
const badFeedback = {
  recognized: null
};

// GOOD: Clear confirmation
const goodFeedback = {
  recognized: { sound: 'chime', visual: 'glow' }
};
```

---

## Related Systems

- **InputManager** - For mouse/touch input
- **VFXManager** - For particle effects
- **AudioManager** - For drawing and manifestation sounds
- **GameState** - For symbol unlock tracking

---

## References

- **Shadow Engine Documentation**: `docs/`
- **TensorFlow.js**: https://www.tensorflow.org/js
- **Gesture Recognition**: https://developer.mozilla.org/en-US/docs/Web/API/Pointer Events

---

**RALPH_STATUS:**
- **Status**: Drawing Minigame Scene documentation complete
- **Files Created**: `docs/generated/14-scene-case-studies/drawing-minigame-scene.md`
- **Phase 14**: All scene case studies now complete (17 scenes documented)
- **Next**: Review fix plan and address remaining tasks
