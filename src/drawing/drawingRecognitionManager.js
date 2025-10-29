import { DRAWING_LABELS, FULL_LABEL_SET } from "./drawingLabels.js";
import { ImagePreprocessor } from "./imagePreprocessor.js";
import { ParticleCanvas3D } from "./particleCanvas3D.js";
import { Logger } from "../utils/logger.js";
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
    this.logger = new Logger("DrawingRecognitionManager", false);
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
    this.logger.log("Loading Quick Draw model...");

    try {
      // Load TensorFlow.js and TFLite from local files in /models/
      this.logger.log("Loading TensorFlow.js from /models/...");
      await this.loadScript("/models/local-tf.min.js");

      this.logger.log("Loading TFLite from /models/...");
      await this.loadScript("/models/local-tf-tflite.min.js");

      // Wait a bit for the global objects to be set
      await new Promise((resolve) => setTimeout(resolve, 500));

      if (!window.tf || !window.tflite) {
        throw new Error("TensorFlow.js or TFLite not available after loading");
      }

      this.logger.log("TensorFlow.js and TFLite loaded from local files");

      // Set WASM path to local /models/ directory
      window.tflite.setWasmPath("/models/");

      await window.tf.ready();
      this.logger.log("TensorFlow.js backend ready");

      this.logger.log(
        "Loading TFLite model from /models/quickdraw-model.tflite"
      );
      this.model = await window.tflite.loadTFLiteModel(
        "/models/quickdraw-model.tflite"
      );

      this.logger.log("Running warmup prediction");
      this.model.predict(window.tf.zeros([1, 28, 28, 1]));

      this.isModelLoaded = true;
      this.logger.log(
        `Quick Draw model loaded! (${FULL_LABEL_SET.length} total classes, ${DRAWING_LABELS.length} active labels)`
      );
    } catch (error) {
      this.logger.error("Failed to load Quick Draw model:", error);
      throw error;
    }
  }

  createDrawingCanvas(scene, position, scale = 1, enableParticles = true) {
    if (this.drawingCanvas) {
      this.drawingCanvas.dispose();
    }

    this.drawingCanvas = new ParticleCanvas3D(
      scene,
      position,
      scale,
      enableParticles
    );
    return this.drawingCanvas;
  }

  setExpectedDrawing(label) {
    this.logger.log(`setExpectedDrawing called with: "${label}"`);

    if (!DRAWING_LABELS.includes(label)) {
      this.logger.warn(`Label "${label}" is not in the active drawing labels`);
      return false;
    }

    this.expectedDrawing = label;
    this.logger.log(`expectedDrawing is now: "${this.expectedDrawing}"`);
    return true;
  }

  enableDrawingMode(camera, domElement) {
    if (!this.drawingCanvas) {
      this.logger.error("No drawing canvas created");
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
      this.logger.error("Model not loaded yet");
      return null;
    }

    if (!this.drawingCanvas || !this.drawingCanvas.hasStrokes()) {
      this.logger.warn("No drawing to predict");
      return null;
    }

    const imageStrokes = this.drawingCanvas.getStrokes();
    this.logger.log("Raw strokes:", imageStrokes);

    const canvas = await this.preprocessor.preprocessImage(imageStrokes);

    if (!canvas) {
      this.logger.error("Failed to preprocess image");
      return null;
    }

    this.logger.log("Canvas size:", canvas.width, "x", canvas.height);

    this.logger.log("Preprocessed to 28x28 using canvas drawImage (like PIL)");
    const debugCanvas = document.createElement("canvas");
    debugCanvas.width = 28;
    debugCanvas.height = 28;
    const debugCtx = debugCanvas.getContext("2d");

    debugCtx.drawImage(canvas, 0, 0);

    debugCanvas.style.cssText =
      "position: fixed; top: 10px; right: 10px; width: 280px; height: 280px; image-rendering: pixelated; border: 2px solid red; z-index: 10000; background: white;";
    debugCanvas.title = "28x28 canvas (resized with drawImage like PIL)";
    document.body.appendChild(debugCanvas);
    setTimeout(() => {
      debugCanvas.remove();
    }, 5000);

    const tensor = window.tf.tidy(() => {
      const imgTensor = window.tf.browser.fromPixels(canvas, 1);
      return window.tf.expandDims(imgTensor, 0);
    });

    this.logger.log("Tensor:", tensor);

    const tensorData = await tensor.data();
    this.logger.log(
      "28x28 Tensor values (784 pixels):",
      Array.from(tensorData)
    );
    this.logger.log("Min value:", Math.min(...tensorData));
    this.logger.log("Max value:", Math.max(...tensorData));
    this.logger.log("First 10 values:", Array.from(tensorData).slice(0, 10));

    const predictions = this.model.predict(tensor).dataSync();
    tensor.dispose();

    const filteredPredictions = Array.from(predictions)
      .map((p, i) => ({
        probability: p,
        className: FULL_LABEL_SET[i],
        index: i,
      }))
      .filter((pred) => DRAWING_LABELS.includes(pred.className))
      .sort((a, b) => b.probability - a.probability);

    this.logger.log(
      "Filtered predictions (active labels only):",
      filteredPredictions
    );

    const top3All = Array.from(predictions)
      .map((p, i) => ({
        probability: p,
        className: FULL_LABEL_SET[i],
        index: i,
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 3);

    this.logger.log("Top 3 predictions (all classes):", top3All);

    return filteredPredictions;
  }

  async predictAndEvaluate() {
    const predictions = await this.predict();

    if (!predictions || predictions.length === 0) {
      this.logger.log("No predictions returned");
      return {
        success: false,
        prediction: null,
        expected: this.expectedDrawing,
      };
    }

    const topPrediction = predictions[0];

    this.logger.log("=== EVALUATION DEBUG ===");
    this.logger.log("Expected drawing:", this.expectedDrawing);
    this.logger.log("Top prediction:", topPrediction.className);
    this.logger.log("Probability:", topPrediction.probability);
    this.logger.log(
      "Class match:",
      topPrediction.className === this.expectedDrawing
    );

    const recognized = topPrediction.className === this.expectedDrawing;

    this.logger.log("RECOGNIZED:", recognized);
    this.logger.log("=======================");

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

  update(dt = 0.016) {
    if (this.drawingCanvas) {
      this.drawingCanvas.update(dt);
    }
  }
}
