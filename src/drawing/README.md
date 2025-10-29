# Quick Draw Drawing Recognition System

This module integrates Google's Quick Draw CNN model into the game as an in-world drawing mechanic.

## Files

- `drawingRecognitionManager.js` - Main manager for loading model, handling predictions, and coordinating drawing
- `drawingCanvas3D.js` - Creates a 3D canvas mesh with drawing input handling
- `drawingLabels.js` - Contains label definitions (3 active labels: car, star, house)
- `imagePreprocessor.js` - Processes stroke data into 28x28 images for the model
- `drawingExample.js` - Example integration showing how to use the system

## Quick Start

```javascript
import { DrawingRecognitionManager } from "./drawing/drawingRecognitionManager.js";

// In your scene setup
const drawingManager = new DrawingRecognitionManager(gameManager);

// Initialize the model (async)
await drawingManager.initialize();

// Create a drawing canvas in the 3D world
const canvas = drawingManager.createDrawingCanvas(
  scene,
  { x: 0, y: 1.5, z: -2 }, // position
  2 // scale
);

// Set what the player should draw
drawingManager.setExpectedDrawing("star");

// Enable drawing input
drawingManager.enableDrawingMode(camera, renderer.domElement);

// In your game loop
drawingManager.update();

// When ready to check the drawing
const result = await drawingManager.predictAndEvaluate();
// result = { success: true/false, prediction: 'star', confidence: 0.85, expected: 'star' }

// Clear the canvas for a new attempt
drawingManager.clearCanvas();
```

## API Reference

### DrawingRecognitionManager

#### `constructor(gameManager)`

Creates a new drawing recognition manager.

#### `async initialize()`

Loads the TensorFlow Lite model. Must be called before using any other methods.

#### `createDrawingCanvas(scene, position, scale)`

Creates a 3D drawing canvas mesh in the scene.

- `scene` - THREE.Scene to add the canvas to
- `position` - Object with x, y, z coordinates
- `scale` - Size of the canvas plane (default: 1)

Returns: DrawingCanvas3D instance

#### `setExpectedDrawing(label)`

Sets what drawing the player should create. Must be one of: 'car', 'star', 'house'

Returns: true if successful, false if label not in active set

#### `enableDrawingMode(camera, domElement)`

Enables drawing input on the canvas using raycasting.

- `camera` - THREE.Camera for raycasting
- `domElement` - DOM element to attach event listeners to

#### `disableDrawingMode()`

Disables drawing input.

#### `async predict()`

Runs prediction on the current drawing.

Returns: Array of top 3 predictions with {className, probability, index}

#### `async predictAndEvaluate()`

Predicts the drawing and evaluates against expected drawing.
Automatically calls `gameManager.setState()` with result.

Returns: Object with {success, prediction, confidence, expected, allPredictions}

#### `clearCanvas()`

Clears the drawing canvas.

#### `update()`

Updates the canvas texture. Call in your game loop.

#### `dispose()`

Cleans up resources, removes event listeners, disposes of 3D objects.

## Game Manager Integration

When `predictAndEvaluate()` is called, it automatically triggers:

```javascript
gameManager.setState(
    recognized ? 'drawingSuccess' : 'drawingFailed',
    {
        success: true/false,
        prediction: 'star',
        confidence: 0.85,
        expected: 'star',
        allPredictions: [...]
    }
);
```

You can listen for these state changes:

```javascript
gameManager.on("state:changed", (newState, oldState) => {
  if (newState.currentState === "drawingSuccess") {
    console.log("Player successfully drew:", newState.prediction);
    // Trigger next game event
  } else if (newState.currentState === "drawingFailed") {
    console.log("Drawing not recognized or incorrect");
    // Allow retry
  }
});
```

## Configuration

Default settings in `drawingRecognitionManager.js`:

- Recognition threshold: 0.4 (40% confidence minimum)
- Canvas size: 500x500 pixels
- Stroke weight: 3 pixels
- Active labels: ['car', 'star', 'house']

To modify the active labels, edit `DRAWING_LABELS` in `drawingLabels.js`.

## Model Information

- Model: TensorFlow Lite format
- Input: 28x28 grayscale image
- Output: 347 class probabilities (Quick Draw dataset)
- Location: `/public/models/quickdraw-model.tflite`

## Example Use Case

A puzzle where the player must draw specific symbols to progress:

```javascript
const drawingPuzzle = {
  sequence: ["star", "house", "car"],
  currentIndex: 0,

  async checkDrawing() {
    const result = await drawingManager.predictAndEvaluate();

    if (result.success) {
      this.currentIndex++;

      if (this.currentIndex < this.sequence.length) {
        drawingManager.setExpectedDrawing(this.sequence[this.currentIndex]);
        drawingManager.clearCanvas();
      } else {
        // Puzzle complete!
        gameManager.setState({ currentState: GAME_STATES.PUZZLE_SOLVED });
      }
    }
  },
};
```
