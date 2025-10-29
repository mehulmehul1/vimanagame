import { DRAWING_LABELS, FULL_LABEL_SET } from "./drawingLabels.js";
import { ImagePreprocessor } from "./imagePreprocessor.js";
import { DrawingCanvas3D } from "./drawingCanvas3D.js";
import * as THREE from "three";

export class DrawingRecognitionManager {
  constructor(gameManager) {
    this.gameManager = gameManager;
    this.model = null;
    this.preprocessor = new ImagePreprocessor();
    this.isModelLoaded = false;
    this.drawingCanvas = null;
    this.raycaster = new THREE.Raycaster();
    this.expectedDrawing = null;
    this.recognitionThreshold = 0.4;
    this.isDrawingMode = false;
  }

  loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) {
        if (existing.dataset.loaded === "true") {
          resolve();
        } else {
          existing.addEventListener("load", resolve);
          existing.addEventListener("error", reject);
        }
        return;
      }

      const script = document.createElement("script");
      script.src = src;
      script.onload = () => {
        script.dataset.loaded = "true";
        resolve();
      };
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  async initialize() {
    console.log("Loading Quick Draw model...");

    try {
      // Load TensorFlow.js and TFLite from local files in /models/
      console.log("Loading TensorFlow.js from /models/...");
      await this.loadScript("/models/local-tf.min.js");

      console.log("Loading TFLite from /models/...");
      await this.loadScript("/models/local-tf-tflite.min.js");

      // console.log("Loading TFLite from /models/...");
      // await this.loadScript("/models/tfjs-backend-cpu.js");

      // Wait a bit for the global objects to be set
      await new Promise((resolve) => setTimeout(resolve, 500));

      if (!window.tf || !window.tflite) {
        throw new Error("TensorFlow.js or TFLite not available after loading");
      }

      console.log("TensorFlow.js and TFLite loaded from local files");

      // Set WASM path to local /models/ directory
      window.tflite.setWasmPath("/models/");

      await window.tf.ready();
      console.log("TensorFlow.js backend ready");

      console.log("Loading TFLite model from /models/quickdraw-model.tflite");
      this.model = await window.tflite.loadTFLiteModel(
        "/models/quickdraw-model.tflite"
      );

      console.log("Running warmup prediction");
      this.model.predict(window.tf.zeros([1, 28, 28, 1]));

      this.isModelLoaded = true;
      console.log(
        `Quick Draw model loaded! (${FULL_LABEL_SET.length} total classes, ${DRAWING_LABELS.length} active labels)`
      );
    } catch (error) {
      console.error("Failed to load Quick Draw model:", error);
      throw error;
    }
  }

  createDrawingCanvas(scene, position, scale = 1) {
    if (this.drawingCanvas) {
      this.drawingCanvas.dispose();
    }

    this.drawingCanvas = new DrawingCanvas3D(scene, position, scale);
    return this.drawingCanvas;
  }

  setExpectedDrawing(label) {
    console.log(
      `🔧 [RecognitionManager] setExpectedDrawing called with: "${label}"`
    );
    console.trace("Call stack:");

    if (!DRAWING_LABELS.includes(label)) {
      console.warn(`Label "${label}" is not in the active drawing labels`);
      return false;
    }

    this.expectedDrawing = label;
    console.log(
      `🔧 [RecognitionManager] expectedDrawing is now: "${this.expectedDrawing}"`
    );
    return true;
  }

  enableDrawingMode(camera, domElement) {
    if (!this.drawingCanvas) {
      console.error("No drawing canvas created");
      return;
    }

    this.isDrawingMode = true;
    this.camera = camera;
    this.domElement = domElement;

    this.onPointerDown = this.handlePointerDown.bind(this);
    this.onPointerMove = this.handlePointerMove.bind(this);
    this.onPointerUp = this.handlePointerUp.bind(this);

    domElement.addEventListener("pointerdown", this.onPointerDown);
    domElement.addEventListener("pointermove", this.onPointerMove);
    domElement.addEventListener("pointerup", this.onPointerUp);
  }

  disableDrawingMode() {
    if (!this.domElement) return;

    this.isDrawingMode = false;
    this.domElement.removeEventListener("pointerdown", this.onPointerDown);
    this.domElement.removeEventListener("pointermove", this.onPointerMove);
    this.domElement.removeEventListener("pointerup", this.onPointerUp);
  }

  handlePointerDown(event) {
    if (!this.isDrawingMode || !this.drawingCanvas) return;

    const uv = this.getCanvasUV(event);
    if (uv) {
      this.drawingCanvas.startStroke(uv);
    }
  }

  handlePointerMove(event) {
    if (!this.isDrawingMode || !this.drawingCanvas) return;

    const uv = this.getCanvasUV(event);
    if (uv) {
      this.drawingCanvas.addPoint(uv);
    }
  }

  handlePointerUp(event) {
    if (!this.isDrawingMode || !this.drawingCanvas) return;

    this.drawingCanvas.endStroke();
  }

  getCanvasUV(event) {
    if (!this.camera || !this.domElement || !this.drawingCanvas) return null;

    const rect = this.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );

    this.raycaster.setFromCamera(mouse, this.camera);
    const intersects = this.raycaster.intersectObject(
      this.drawingCanvas.getMesh()
    );

    if (intersects.length > 0) {
      return intersects[0].uv;
    }

    return null;
  }

  async predict() {
    if (!this.isModelLoaded) {
      console.error("Model not loaded yet");
      return null;
    }

    if (!this.drawingCanvas || !this.drawingCanvas.hasStrokes()) {
      console.warn("No drawing to predict");
      return null;
    }

    const imageStrokes = this.drawingCanvas.getStrokes();
    console.log("Raw strokes:", imageStrokes);

    const canvas = await this.preprocessor.preprocessImage(imageStrokes);

    if (!canvas) {
      console.error("Failed to preprocess image");
      return null;
    }

    console.log("Canvas size:", canvas.width, "x", canvas.height);

    // Debug: show the preprocessed 28x28 image (already resized by canvas API)
    console.log("Preprocessed to 28x28 using canvas drawImage (like PIL)");
    const debugCanvas = document.createElement("canvas");
    debugCanvas.width = 28;
    debugCanvas.height = 28;
    const debugCtx = debugCanvas.getContext("2d");

    // Draw the canvas directly (already 28x28)
    debugCtx.drawImage(canvas, 0, 0);

    debugCanvas.style.cssText =
      "position: fixed; top: 10px; right: 10px; width: 280px; height: 280px; image-rendering: pixelated; border: 2px solid red; z-index: 10000; background: white;";
    debugCanvas.title = "28x28 canvas (resized with drawImage like PIL)";
    document.body.appendChild(debugCanvas);
    setTimeout(() => {
      debugCanvas.remove();
    }, 5000);

    // Now do the actual prediction (matching working demo exactly)
    // Canvas is already 28x28 from preprocessor (resized using canvas API like PIL)
    const tensor = window.tf.tidy(() => {
      // Convert 28x28 canvas to tensor (grayscale)
      const imgTensor = window.tf.browser.fromPixels(canvas, 1);

      // Add batch dimension
      return window.tf.expandDims(imgTensor, 0);
    });

    console.log("Tensor:", tensor);

    // Log the actual 28x28 pixel values
    const tensorData = await tensor.data();
    console.log("28x28 Tensor values (784 pixels):", Array.from(tensorData));
    console.log("Min value:", Math.min(...tensorData));
    console.log("Max value:", Math.max(...tensorData));
    console.log("First 10 values:", Array.from(tensorData).slice(0, 10));

    const predictions = this.model.predict(tensor).dataSync();
    tensor.dispose();

    // Filter predictions to only include our active labels
    const filteredPredictions = Array.from(predictions)
      .map((p, i) => ({
        probability: p,
        className: FULL_LABEL_SET[i],
        index: i,
      }))
      .filter((pred) => DRAWING_LABELS.includes(pred.className))
      .sort((a, b) => b.probability - a.probability);

    console.log(
      "Filtered predictions (active labels only):",
      filteredPredictions
    );

    // Also show top 3 from all classes for debugging
    const top3All = Array.from(predictions)
      .map((p, i) => ({
        probability: p,
        className: FULL_LABEL_SET[i],
        index: i,
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 3);

    console.log("Top 3 predictions (all classes):", top3All);

    return filteredPredictions;
  }

  async predictAndEvaluate() {
    const predictions = await this.predict();

    if (!predictions || predictions.length === 0) {
      console.log("❌ No predictions returned");
      return {
        success: false,
        prediction: null,
        expected: this.expectedDrawing,
      };
    }

    const topPrediction = predictions[0];

    console.log("=== EVALUATION DEBUG ===");
    console.log("Expected drawing:", this.expectedDrawing);
    console.log("Top prediction:", topPrediction.className);
    console.log("Probability:", topPrediction.probability);
    console.log(
      "Class match:",
      topPrediction.className === this.expectedDrawing
    );

    // NO THRESHOLD - just match the top prediction
    const recognized = topPrediction.className === this.expectedDrawing;

    console.log("RECOGNIZED:", recognized);
    console.log("=======================");

    const result = {
      success: recognized,
      prediction: topPrediction.className,
      confidence: topPrediction.probability,
      expected: this.expectedDrawing,
      allPredictions: predictions,
    };

    if (this.gameManager) {
      this.gameManager.setState(
        recognized ? "drawingSuccess" : "drawingFailed",
        result
      );
    }

    return result;
  }

  clearCanvas() {
    if (this.drawingCanvas) {
      this.drawingCanvas.clearCanvas();
    }
  }

  dispose() {
    this.disableDrawingMode();

    if (this.drawingCanvas) {
      this.drawingCanvas.dispose();
      this.drawingCanvas = null;
    }

    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }

  update() {
    if (this.drawingCanvas) {
      this.drawingCanvas.update();
    }
  }
}
