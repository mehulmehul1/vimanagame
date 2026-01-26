# Stroke Data System - First Principles Guide

## Overview

The **Stroke Data System** is the data structure and format used throughout the drawing minigame to represent player gestures. It defines how a stroke (a continuous drawing motion) is captured, stored, normalized, and processed for recognition. This system is the "language" that connects raw input to magical output.

Think of the stroke data system as the **"symbol definition format"** - just as text has characters and images have pixels, drawings have strokes, and this system defines how those strokes are represented digitally so the game can understand them.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Make drawing feel responsive and predictable. Players should feel their input is being captured accurately and fairly.

**Why a Dedicated Stroke Format?**
- **Consistency**: Same format for capture, storage, and recognition
- **Optimization**: Efficient representation (no wasted data)
- **Extensibility**: Easy to add new metadata (pressure, speed, etc.)
- **Debugging**: Human-readable for testing and verification

### Design Principles

**1. Minimal but Complete**
Store only what's needed:
- Point positions (x, y, z)
- Timestamps (for speed/pressure)
- Optional metadata (stroke ID, device type)

**2. Normalization-Ready**
Format makes it easy to:
- Center the stroke
- Scale to standard size
- Remove temporal variation
- Prepare for ML model input

**3. Multi-Stroke Support**
Some symbols require multiple strokes (like a pentagram drawn in 5 parts):
- Store stroke sequence
- Track stroke count
- Validate completeness

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the stroke data system, you should know:
- **Data structures** - Arrays, objects, serialization
- **Coordinate systems** - 3D world vs 2D normalized space
- **Time series data** - Recording values over time
- **Normalization** - Scaling data to standard ranges
- **JSON serialization** - Saving/loading stroke data

### Stroke Data Structure

```javascript
/**
 * Raw stroke data format (as captured from input)
 */
export interface RawStroke {
  id: string;              // Unique identifier
  points: StrokePoint[];   // Array of points in the stroke
  startTime: number;       // When stroke started (timestamp)
  endTime: number;         // When stroke ended (timestamp)
  device: 'mouse' | 'touch' | 'pointer';
  metadata?: {
    pressure?: number[];   // For pressure-sensitive input
    tilt?: number[];       // For stylus tilt
    rotation?: number[];   // For stylus rotation
  };
}

/**
 * Single point in a stroke
 */
export interface StrokePoint {
  x: number;              // World X position
  y: number;              // World Y position
  z: number;              // World Z position
  timestamp: number;      // Milliseconds from stroke start
}

/**
 * Normalized stroke (for recognition)
 */
export interface NormalizedStroke {
  points: NormalizedPoint[];
  boundingBox: BoundingBox;
  startPoint: NormalizedPoint;
  endPoint: NormalizedPoint;
  length: number;          // Total path length
  features?: StrokeFeatures;
}

/**
 * Point normalized to [-1, 1] range
 */
export interface NormalizedPoint {
  x: number;              // Normalized X [-1, 1]
  y: number;              // Normalized Y [-1, 1]
  z: number;              // Normalized Z (optional, for 3D)
}

/**
 * Bounding box for normalization
 */
export interface BoundingBox {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
  width: number;
  height: number;
  depth: number;
  centerX: number;
  centerY: number;
  centerZ: number;
}

/**
 * Extracted features for classification
 */
export interface StrokeFeatures {
  aspectRatio: number;    // width / height
  closure: number;        // Distance from start to end
  corners: number;        // Sharp direction changes
  directionChanges: number;
  totalLength: number;
  averageSpeed: number;
}
```

### StrokeData Class

```javascript
class StrokeData {
  constructor() {
    this.strokes = [];
    this.currentStroke = null;
  }

  /**
   * Start a new stroke
   */
  beginStroke(id, device = 'pointer') {
    this.currentStroke = {
      id: id,
      points: [],
      startTime: performance.now(),
      endTime: null,
      device: device,
      metadata: {}
    };
  }

  /**
   * Add a point to current stroke
   */
  addPoint(x, y, z) {
    if (!this.currentStroke) {
      throw new Error('No active stroke. Call beginStroke() first.');
    }

    const timestamp = performance.now() - this.currentStroke.startTime;

    this.currentStroke.points.push({ x, y, z, timestamp });
  }

  /**
   * End current stroke
   */
  endStroke() {
    if (!this.currentStroke) {
      throw new Error('No active stroke to end.');
    }

    this.currentStroke.endTime = performance.now();
    this.strokes.push(this.currentStroke);
    this.currentStroke = null;
  }

  /**
   * Get all strokes
   */
  getStrokes() {
    return this.strokes;
  }

  /**
   * Get combined multi-stroke data
   */
  getMultiStroke() {
    return {
      strokes: this.strokes,
      pointCount: this.strokes.reduce((sum, s) => sum + s.points.length, 0),
      boundingBox: this.calculateMultiStrokeBounds(),
      duration: this.strokes.length > 0
        ? this.strokes[this.strokes.length - 1].endTime - this.strokes[0].startTime
        : 0
    };
  }

  /**
   * Normalize a stroke for recognition
   */
  normalize(stroke) {
    // Find bounding box
    const bbox = this.calculateBoundingBox(stroke.points);

    // Calculate scale to fit in [-1, 1] range
    const maxSize = Math.max(bbox.width, bbox.height, bbox.depth);
    const scale = maxSize > 0 ? 2 / maxSize : 1;

    // Normalize each point
    const normalizedPoints = stroke.points.map(point => ({
      x: (point.x - bbox.centerX) * scale,
      y: (point.y - bbox.centerY) * scale,
      z: (point.z - bbox.centerZ) * scale,
      timestamp: point.timestamp
    }));

    // Extract features
    const features = this.extractFeatures(normalizedPoints, bbox);

    return {
      points: normalizedPoints,
      boundingBox: bbox,
      startPoint: normalizedPoints[0],
      endPoint: normalizedPoints[normalizedPoints.length - 1],
      length: this.calculatePathLength(stroke.points),
      features
    };
  }

  /**
   * Calculate bounding box of points
   */
  calculateBoundingBox(points) {
    if (points.length === 0) {
      return null;
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (const point of points) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
      minZ = Math.min(minZ, point.z);
      maxZ = Math.max(maxZ, point.z);
    }

    return {
      minX, maxX, minY, maxY, minZ, maxZ,
      width: maxX - minX,
      height: maxY - minY,
      depth: maxZ - minZ,
      centerX: (minX + maxX) / 2,
      centerY: (minY + maxY) / 2,
      centerZ: (minZ + maxZ) / 2
    };
  }

  /**
   * Calculate multi-stroke bounding box
   */
  calculateMultiStrokeBounds() {
    if (this.strokes.length === 0) {
      return null;
    }

    // Collect all points from all strokes
    const allPoints = this.strokes.flatMap(s => s.points);
    return this.calculateBoundingBox(allPoints);
  }

  /**
   * Calculate total path length
   */
  calculatePathLength(points) {
    if (points.length < 2) return 0;

    let length = 0;
    for (let i = 1; i < points.length; i++) {
      const dx = points[i].x - points[i - 1].x;
      const dy = points[i].y - points[i - 1].y;
      const dz = points[i].z - points[i - 1].z;
      length += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    return length;
  }

  /**
   * Extract stroke features for classification
   */
  extractFeatures(points, bbox) {
    return {
      aspectRatio: bbox.width / (bbox.height || 1),
      closure: this.calculateClosure(points),
      corners: this.countCorners(points),
      directionChanges: this.countDirectionChanges(points),
      totalLength: this.calculatePathLength(points),
      averageSpeed: this.calculateAverageSpeed(points)
    };
  }

  /**
   * Calculate closure (distance from start to end)
   */
  calculateClosure(points) {
    if (points.length < 2) return 0;

    const start = points[0];
    const end = points[points.length - 1];

    const distance = Math.sqrt(
      Math.pow(end.x - start.x, 2) +
      Math.pow(end.y - start.y, 2) +
      Math.pow(end.z - start.z, 2)
    );

    // Normalize by bounding box size
    const bbox = this.calculateBoundingBox(points);
    const diagonal = Math.sqrt(
      Math.pow(bbox.width, 2) +
      Math.pow(bbox.height, 2) +
      Math.pow(bbox.depth, 2)
    );

    return diagonal > 0 ? distance / diagonal : 0;
  }

  /**
   * Count corners (sharp direction changes)
   */
  countCorners(points) {
    if (points.length < 3) return 0;

    let corners = 0;
    let lastAngle = null;
    const cornerThreshold = Math.PI / 4; // 45 degrees

    for (let i = 1; i < points.length - 1; i++) {
      const angle = this.getAngleAt(points, i);

      if (lastAngle !== null) {
        let diff = Math.abs(angle - lastAngle);
        if (diff > Math.PI) diff = 2 * Math.PI - diff;

        if (diff > cornerThreshold) {
          corners++;
        }
      }

      lastAngle = angle;
    }

    return corners;
  }

  /**
   * Get angle at point index
   */
  getAngleAt(points, index) {
    if (index < 1 || index >= points.length - 1) {
      return 0;
    }

    const prev = points[index - 1];
    const curr = points[index];
    const next = points[index + 1];

    const dx1 = curr.x - prev.x;
    const dy1 = curr.y - prev.y;
    const dx2 = next.x - curr.x;
    const dy2 = next.y - curr.y;

    return Math.atan2(dy2, dx2) - Math.atan2(dy1, dx1);
  }

  /**
   * Count direction changes
   */
  countDirectionChanges(points) {
    if (points.length < 3) return 0;

    let changes = 0;
    let lastDirection = null;

    for (let i = 1; i < points.length; i++) {
      const direction = Math.atan2(
        points[i].y - points[i - 1].y,
        points[i].x - points[i - 1].x
      );

      if (lastDirection !== null) {
        let diff = Math.abs(direction - lastDirection);
        if (diff > Math.PI) diff = 2 * Math.PI - diff;

        if (diff > Math.PI / 2) { // 90 degree change
          changes++;
        }
      }

      lastDirection = direction;
    }

    return changes;
  }

  /**
   * Calculate average drawing speed
   */
  calculateAverageSpeed(points) {
    if (points.length < 2) return 0;

    let totalSpeed = 0;

    for (let i = 1; i < points.length; i++) {
      const dx = points[i].x - points[i - 1].x;
      const dy = points[i].y - points[i - 1].y;
      const dz = points[i].z - points[i - 1].z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

      const dt = points[i].timestamp - points[i - 1].timestamp;

      totalSpeed += dt > 0 ? distance / dt : 0;
    }

    return totalSpeed / (points.length - 1);
  }

  /**
   * Serialize to JSON
   */
  toJSON() {
    return JSON.stringify({
      strokes: this.strokes,
      multiStroke: this.getMultiStroke()
    });
  }

  /**
   * Deserialize from JSON
   */
  static fromJSON(json) {
    const data = JSON.parse(json);
    const strokeData = new StrokeData();
    strokeData.strokes = data.strokes;
    return strokeData;
  }

  /**
   * Clear all strokes
   */
  clear() {
    this.strokes = [];
    this.currentStroke = null;
  }

  /**
   * Get stroke count
   */
  getStrokeCount() {
    return this.strokes.length + (this.currentStroke ? 1 : 0);
  }

  /**
   * Get total point count
   */
  getTotalPointCount() {
    let count = this.currentStroke ? this.currentStroke.points.length : 0;
    count += this.strokes.reduce((sum, s) => sum + s.points.length, 0);
    return count;
  }
}

export default StrokeData;
```

### Training Data Format

```javascript
/**
 * Training data format for ML model
 */
export interface TrainingData {
  examples: TrainingExample[];
  labels: string[];
  numClasses: number;
}

export interface TrainingExample {
  stroke: NormalizedStroke;
  label: string;  // The correct class
  labelIndex: number;  // Numeric label
}

/**
 * Sample training dataset
 */
export const sampleTrainingData = {
  labels: ['circle', 'triangle', 'square', 'lightning', 'pentagram'],

  examples: [
    {
      stroke: { /* normalized circle stroke */ },
      label: 'circle',
      labelIndex: 0
    },
    {
      stroke: { /* normalized triangle stroke */ },
      label: 'triangle',
      labelIndex: 1
    },
    // ... more examples
  ]
};

/**
 * Stroke recording utility for collecting training data
 */
class StrokeRecorder {
  constructor() {
    this.recordings = new Map();  // label -> Array of strokes
  }

  /**
   * Start recording a new example
   */
  startRecording(label) {
    this.currentLabel = label;
    this.strokeData = new StrokeData();
    this.strokeData.beginStroke(generateId());
  }

  /**
   * Record a point during capture
   */
  recordPoint(x, y, z) {
    this.strokeData.addPoint(x, y, z);
  }

  /**
   * Finish recording and save
   */
  finishRecording() {
    this.strokeData.endStroke();

    if (!this.recordings.has(this.currentLabel)) {
      this.recordings.set(this.currentLabel, []);
    }

    this.recordings.get(this.currentLabel).push(
      this.strokeData.normalize(this.strokeData.strokes[0])
    );
  }

  /**
   * Export as training data
   */
  exportTrainingData() {
    const examples = [];
    let labelIndex = 0;

    for (const [label, strokes] of this.recordings) {
      for (const stroke of strokes) {
        examples.push({
          stroke,
          label,
          labelIndex
        });
      }
      labelIndex++;
    }

    return {
      examples,
      labels: Array.from(this.recordings.keys()),
      numClasses: this.recordings.size
    };
  }
}
```

---

## 📝 How To Use This System

### Recording Training Examples

```javascript
// 1. Create recorder
const recorder = new StrokeRecorder();

// 2. Start recording
recorder.startRecording('circle');

// 3. Capture input (from DrawingManager)
drawingManager.on('point', (p) => recorder.recordPoint(p.x, p.y, p.z));

// 4. Finish when done
recorder.finishRecording();

// 5. Repeat for all examples
// 6. Export training data
const trainingData = recorder.exportTrainingData();
```

### Recognizing a Stroke

```javascript
// 1. Create stroke data
const strokeData = new StrokeData();

// 2. Capture stroke
strokeData.beginStroke('stroke_1');
// ... add points during drawing
strokeData.endStroke();

// 3. Normalize
const normalized = strokeData.normalize(strokeData.strokes[0]);

// 4. Extract features
const features = normalized.features;

// 5. Classify
if (features.closure < 0.2 && features.aspectRatio > 0.8) {
  console.log('Probably a circle!');
}
```

---

## 🔧 Variations For Your Game

### Pressure-Sensitive Strokes

```javascript
interface StrokePoint {
  x, y, z, timestamp,
  pressure: number;  // 0-1 range from stylus
}

// Use pressure for line width
lineWidth = 0.02 + point.pressure * 0.03;
```

### Speed-Aware Recognition

```javascript
// Fast vs slow drawing means different intent
const speed = features.averageSpeed;

if (speed > 50) {
  // Fast stroke = aggressive spell
} else {
  // Slow stroke = careful ritual
}
```

### Multi-Stroke Symbols

```javascript
// Define symbols requiring multiple strokes
const multiStrokeSymbols = {
  pentagram: {
    strokes: 5,
    orderMatters: true,
    timeLimit: 5000  // Must complete within 5 seconds
  },
  infinity: {
    strokes: 2,
    orderMatters: false,
    timeLimit: 3000
  }
};
```

---

## Common Mistakes Beginners Make

### 1. Not Normalizing

```javascript
// ❌ WRONG: Raw coordinates vary wildly
recognize(rawStroke);

// ✅ CORRECT: Always normalize first
const normalized = normalizeStroke(rawStroke);
recognize(normalized);
```

### 2. Losing Precision

```javascript
// ❌ WRONG: Rounding too much
const point = {
  x: Math.round(x * 10) / 10,
  y: Math.round(y * 10) / 10
};
// Too coarse

// ✅ CORRECT: Keep precision
const point = { x, y, z, timestamp };
// Accurate representation
```

### 3. No Bounds Checking

```javascript
// ❌ WRONG: No validation
addPoint(x, y, z);

// ✅ CORRECT: Validate input
if (!isFinite(x) || !isFinite(y) || !isFinite(z)) {
  throw new Error('Invalid point');
}
addPoint(x, y, z);
```

---

## Related Systems

- [DrawingManager](./drawing-manager.md) - Input capture
- [DrawingRecognitionManager](./drawing-recognition-manager.md) - ML recognition
- [ParticleCanvas3D](./particle-canvas-3d.md) - Visual feedback

---

## Source File Reference

**Primary Files**:
- `../src/content/StrokeData.js` - Stroke data structure (estimated)

**Key Classes**:
- `StrokeData` - Main data class
- `StrokeRecorder` - Training data collection
- Interface definitions for type safety

**Dependencies**:
- None (pure data structure)

---

## References

- [JSON Serialization](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/stringify) - Data persistence
- [TypeScript Interfaces](https://www.typescriptlang.org/docs/handbook/2/interfaces.html) - Type definitions
- [Dollar Recognizer](https://depts.washington.edu/aimgroup/proj/dollar/) - Gesture recognition algorithm
- [MNIST Dataset](https://yann.lecun.com/exdb/mnist/) - Training data reference

*Documentation last updated: January 12, 2026*
