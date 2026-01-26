# Drawing Recognition Manager - First Principles Guide

## Overview

The **Drawing Recognition Manager** is the machine learning component of the drawing minigame. It takes raw stroke data from the DrawingManager and uses TensorFlow.js to recognize what symbol the player drew. This system transforms gesture input into game events - a circle becomes a shield, a lightning bolt becomes an attack, etc.

Think of the recognition manager as the **"spell interpreter"** - just as a wizard understands the meaning of mystical symbols, this system interprets player drawings and determines their intent, converting physical gesture into digital action.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create the feeling that your drawing MATTERS. When the game recognizes your gesture and responds, it should feel like your magic actually worked.

**Why Machine Learning Recognition?**
- **Natural Interaction**: Players draw naturally, not pixel-perfect
- **Forgiving**: ML can recognize variation (imperfect circles still count)
- **Scalable**: Easy to add new symbols without hard-coded rules
- **Magical Feeling**: Recognition feels like the game "understands" you

**Player Psychology**:
```
Draw Symbol → "I hope this works" → Uncertainty
     ↓
ML Processing → Brief pause → Anticipation
     ↓
Success! → "It understood me!" → Empowerment
     ↓
World Changes → "My drawing did that!" → Agency
```

### Design Decisions

**1. Pre-Trained Model + Fine-Tuning**
Use an existing gesture recognition model, fine-tuned on game-specific symbols:
- Faster than training from scratch
- Better generalization
- Proven architecture

**2. Confidence Threshold**
Don't require 100% match - accept "good enough" drawings:
- Reduces frustration
- Maintains magical feeling
- Accommodates different playstyles

**3. Fallback to Simple Recognition**
If ML fails or isn't available, use geometric heuristics:
- Circle detection (bounding box ratio, closure)
- Line detection (stroke direction)
- Always has a backup

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the recognition system, you should know:
- **TensorFlow.js** - ML framework for JavaScript
- **Neural networks** - Input layers, hidden layers, output layers
- **Training data** - Labeled examples for learning
- **Inference** - Running the model on new input
- **Feature extraction** - Converting raw strokes to model input

### Core Architecture

```
DRAWING RECOGNITION ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│              DRAWING RECOGNITION MANAGER                 │
│  - Loads TensorFlow.js model                            │
│  - Preprocesses stroke data                             │
│  - Runs inference on strokes                            │
│  - Returns predictions with confidence                   │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌───────────────┐                       ┌───────────────┐
│ TENSORFLOW.JS │                       │   FALLBACK    │
│   MODEL       │                       │   RECOGNITION │
│  - CNN/LSTM   │                       │  - Geometric  │
│  - Classes    │                       │  - Heuristics │
└───────────────┘                       └───────────────┘
        │                                       │
        └───────────────┬───────────────────────┘
                        ▼
                ┌───────────────┐
                │   RESULT      │
                │  - label      │
                │  - confidence │
                └───────────────┘
```

### Recognition Manager Class

```javascript
class DrawingRecognitionManager {
  constructor(options = {}) {
    this.logger = options.logger || console;
    this.model = null;
    this.modelLoaded = false;
    this.labels = [];  // Symbol classes
    this.confidenceThreshold = options.confidenceThreshold || 0.7;

    // Fallback recognition (geometric)
    this.useFallback = true;
    this.fallbackRecognizer = new GeometricRecognizer();

    // Model configuration
    this.modelConfig = {
      inputSize: 28,        // 28x28 input (like MNIST)
      numClasses: 5,        // Number of symbols
      modelUrl: options.modelUrl || '/models/drawing_model.json'
    };
  }

  /**
   * Initialize TensorFlow.js model
   */
  async initialize() {
    try {
      // Load TensorFlow.js
      const tf = await import('@tensorflow/tfjs');

      // Load model
      this.model = await tf.loadLayersModel(this.modelConfig.modelUrl);

      // Get labels from model
      this.labels = this.getModelLabels();

      this.modelLoaded = true;
      this.logger.log("Drawing recognition model loaded");

      // Warm up model (run one inference)
      const dummyInput = tf.zeros([1, 28, 28, 1]);
      this.model.predict(dummyInput);
      dummyInput.dispose();

    } catch (error) {
      this.logger.warn("Failed to load ML model, using fallback:", error);
      this.modelLoaded = false;
      this.useFallback = true;
    }
  }

  /**
   * Recognize a stroke
   */
  recognize(stroke) {
    // Preprocess stroke
    const processed = this.preprocessStroke(stroke);

    // Try ML recognition first
    if (this.modelLoaded && !this.useFallback) {
      const mlResult = this.recognizeWithModel(processed);
      if (mlResult.confidence >= this.confidenceThreshold) {
        return mlResult;
      }
    }

    // Fallback to geometric recognition
    return this.recognizeWithFallback(processed);
  }

  /**
   * Preprocess stroke for model input
   */
  preprocessStroke(stroke) {
    // 1. Simplify stroke (reduce points)
    const simplified = this.simplifyStroke(stroke, tolerance: 0.02);

    // 2. Convert to 2D (project to XY plane)
    const points2D = simplified.map(p => ({ x: p.x, y: p.y }));

    // 3. Create bounding box
    const bbox = this.getBoundingBox(points2D);

    // 4. Create image from stroke
    const image = this.strokeToImage(points2D, bbox, this.modelConfig.inputSize);

    // 5. Normalize to [0, 1]
    const normalized = this.normalizeImage(image);

    return normalized;
  }

  /**
   * Simplify stroke using Ramer-Douglas-Peucker algorithm
   */
  simplifyStroke(stroke, tolerance) {
    if (stroke.length <= 2) return stroke;

    // Find the point with maximum distance from line between start and end
    let maxDist = 0;
    let maxIndex = 0;

    const start = stroke[0];
    const end = stroke[stroke.length - 1];

    for (let i = 1; i < stroke.length - 1; i++) {
      const dist = this.pointToLineDistance(stroke[i], start, end);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }

    // If max distance is greater than tolerance, recursively simplify
    if (maxDist > tolerance) {
      const left = this.simplifyStroke(stroke.slice(0, maxIndex + 1), tolerance);
      const right = this.simplifyStroke(stroke.slice(maxIndex), tolerance);
      return [...left.slice(0, -1), ...right];
    }

    // Otherwise, return just the endpoints
    return [start, end];
  }

  /**
   * Calculate point-to-line distance
   */
  pointToLineDistance(point, lineStart, lineEnd) {
    const A = point.x - lineStart.x;
    const B = point.y - lineStart.y;
    const C = lineEnd.x - lineStart.x;
    const D = lineEnd.y - lineStart.y;

    const dot = A * C + B * D;
    const lenSq = C * C + D * D;

    let param = -1;
    if (lenSq !== 0) param = dot / lenSq;

    let xx, yy;

    if (param < 0) {
      xx = lineStart.x;
      yy = lineStart.y;
    } else if (param > 1) {
      xx = lineEnd.x;
      yy = lineEnd.y;
    } else {
      xx = lineStart.x + param * C;
      yy = lineStart.y + param * D;
    }

    const dx = point.x - xx;
    const dy = point.y - yy;

    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Convert stroke to image tensor
   */
  strokeToImage(points, bbox, size) {
    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Fill white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, size, size);

    // Draw stroke
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Scale points to image
    const scaleX = size / (bbox.maxX - bbox.minX);
    const scaleY = size / (bbox.maxY - bbox.minY);
    const scale = Math.min(scaleX, scaleY) * 0.8;  // 80% fill

    const offsetX = (size - (bbox.maxX - bbox.minX) * scale) / 2;
    const offsetY = (size - (bbox.maxY - bbox.minY) * scale) / 2;

    ctx.beginPath();
    points.forEach((point, i) => {
      const x = (point.x - bbox.minX) * scale + offsetX;
      const y = (point.y - bbox.minY) * scale + offsetY;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    return canvas;
  }

  /**
   * Recognize with TensorFlow model
   */
  async recognizeWithModel(processedStroke) {
    const tf = await import('@tensorflow/tfjs');

    // Convert canvas to tensor
    const imageTensor = tf.browser.fromPixels(processedStroke);
    const resized = tf.image.resizeBilinear(imageTensor, [28, 28]);
    const grayscale = tf.image.rgbToGrayscale(resized);
    const batched = grayscale.expandDims(0);
    const normalized = batched.div(255.0);

    // Run inference
    const prediction = this.model.predict(normalized);
    const probabilities = await prediction.data();

    // Get top result
    const maxProb = Math.max(...probabilities);
    const labelIndex = probabilities.indexOf(maxProb);
    const label = this.labels[labelIndex];

    // Clean up tensors
    imageTensor.dispose();
    resized.dispose();
    grayscale.dispose();
    batched.dispose();
    normalized.dispose();
    prediction.dispose();

    return {
      label: label,
      confidence: maxProb,
      allProbabilities: probabilities.map((p, i) => ({
        label: this.labels[i],
        probability: p
      }))
    };
  }

  /**
   * Fallback geometric recognition
   */
  recognizeWithFallback(stroke) {
    return this.fallbackRecognizer.recognize(stroke);
  }

  /**
   * Train model with new examples
   */
  async train(examples) {
    // examples: [{ stroke: [...], label: "circle" }, ...]
    const tf = await import('@tensorflow/tfjs');

    // Convert examples to training data
    const trainingData = this.prepareTrainingData(examples);

    // Create model
    const model = this.createModel();

    // Train
    await model.fit(trainingData.inputs, trainingData.labels, {
      epochs: 50,
      batchSize: 32,
      validationSplit: 0.2
    });

    // Save model
    await model.save(this.modelConfig.modelUrl);

    this.model = model;
    this.modelLoaded = true;
  }

  /**
   * Create a simple CNN model for gesture recognition
   */
  createModel() {
    const tf = await import('@tensorflow/tfjs');

    const model = tf.sequential();

    // Input: 28x28 grayscale image
    model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 32,
      kernelSize: 3,
      activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    // Output: probability for each class
    model.add(tf.layers.dense({
      units: this.modelConfig.numClasses,
      activation: 'softmax'
    }));

    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    return model;
  }

  /**
   * Get model labels
   */
  getModelLabels() {
    // Default labels - should match model training
    return ['circle', 'triangle', 'square', 'lightning', 'pentagram'];
  }

  /**
   * Clean up
   */
  destroy() {
    if (this.model) {
      this.model.dispose();
    }
  }
}

/**
 * Geometric fallback recognizer
 */
class GeometricRecognizer {
  recognize(stroke) {
    // Analyze stroke geometry
    const features = this.extractFeatures(stroke);

    // Simple classification rules
    if (this.isCircle(features)) {
      return { label: 'circle', confidence: 0.8 };
    }
    if (this.isTriangle(features)) {
      return { label: 'triangle', confidence: 0.75 };
    }
    if (this.isSquare(features)) {
      return { label: 'square', confidence: 0.75 };
    }
    if (this.isLightning(features)) {
      return { label: 'lightning', confidence: 0.7 };
    }

    return { label: 'unknown', confidence: 0 };
  }

  extractFeatures(stroke) {
    // Calculate bounding box
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (const point of stroke) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }

    const width = maxX - minX;
    const height = maxY - minY;
    const aspectRatio = width / height;

    // Calculate if closed (start and end near each other)
    const start = stroke[0];
    const end = stroke[stroke.length - 1];
    const closure = Math.sqrt(
      Math.pow(start.x - end.x, 2) +
      Math.pow(start.y - end.y, 2)
    ) / Math.sqrt(width * width + height * height);

    // Calculate corners (sharp direction changes)
    const corners = this.countCorners(stroke);

    // Calculate stroke direction changes
    const directionChanges = this.countDirectionChanges(stroke);

    return {
      width,
      height,
      aspectRatio,
      closure,
      corners,
      directionChanges,
      startPoint: start,
      endPoint: end
    };
  }

  isCircle(features) {
    // Circle: roughly equal width/height, closed
    return features.aspectRatio > 0.7 &&
           features.aspectRatio < 1.3 &&
           features.closure < 0.2;  // Start/end close
  }

  isTriangle(features) {
    // Triangle: 3 corners
    return features.corners === 3;
  }

  isSquare(features) {
    // Square: 4 corners, roughly equal width/height
    return features.corners === 4 &&
           features.aspectRatio > 0.7 &&
           features.aspectRatio < 1.3;
  }

  isLightning(features) {
    // Lightning: zig-zag pattern, not closed
    return features.directionChanges >= 3 &&
           features.closure > 0.3;  // Not closed
  }

  countCorners(stroke) {
    // Count sharp direction changes
    let corners = 0;
    let lastAngle = null;

    for (let i = 1; i < stroke.length - 1; i++) {
      const prev = stroke[i - 1];
      const curr = stroke[i];
      const next = stroke[i + 1];

      const angle1 = Math.atan2(curr.y - prev.y, curr.x - prev.x);
      const angle2 = Math.atan2(next.y - curr.y, next.x - curr.x);

      let angleDiff = Math.abs(angle2 - angle1);
      if (angleDiff > Math.PI) angleDiff = 2 * Math.PI - angleDiff;

      // Sharp turn = corner
      if (angleDiff > Math.PI / 4) {  // 45 degrees
        corners++;
      }
    }

    return corners;
  }

  countDirectionChanges(stroke) {
    // Count overall direction changes
    let changes = 0;
    let lastDirection = null;

    for (let i = 1; i < stroke.length; i++) {
      const dx = stroke[i].x - stroke[i - 1].x;
      const dy = stroke[i].y - stroke[i - 1].y;

      const direction = Math.atan2(dy, dx);

      if (lastDirection !== null) {
        let diff = Math.abs(direction - lastDirection);
        if (diff > Math.PI) diff = 2 * Math.PI - diff;

        if (diff > Math.PI / 2) {  // 90 degree change
          changes++;
        }
      }

      lastDirection = direction;
    }

    return changes;
  }
}

export default DrawingRecognitionManager;
