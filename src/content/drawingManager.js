import * as THREE from "three";
import { GAME_STATES } from "../gameData.js";
import { Logger } from "../utils/logger.js";
import { DrawingRecognitionManager } from "../drawing/drawingRecognitionManager.js";

/**
 * DrawingManager - Manages drawing canvas interactions and AI recognition
 *
 * Features:
 * - Creates 3D drawing canvas in front of player during CURSOR state
 * - Integrates Quick Draw CNN model for drawing recognition
 * - Handles drawing input via raycasting
 * - Triggers game state changes based on recognition results
 *
 * Usage:
 * const drawingManager = new DrawingManager({
 *   sceneManager,
 *   scene,
 *   camera,
 * });
 * drawingManager.initialize(gameManager);
 */
class DrawingManager {
  constructor(options = {}) {
    this.sceneManager = options.sceneManager;
    this.scene = options.scene;
    this.camera = options.camera;
    this.renderer = options.renderer;
    this.logger = new Logger("DrawingManager", true);

    this.drawingRecognition = null;
    this.gameManager = null;
    this.isActive = false;
    this.canvasPosition = new THREE.Vector3(0, 0, -1.5);
    this.canvasScale = 1.2;

    this.config = {
      expectedDrawing: "star",
      canvasDistance: 1.5,
      canvasHeight: 0,
      canvasScale: 1.2,
      autoPredict: false,
    };
  }

  /**
   * Initialize the drawing manager
   * Sets up event listeners and prepares the drawing recognition system
   */
  async initialize(gameManager = null) {
    this.gameManager = gameManager;

    if (!this.scene || !this.camera) {
      this.logger.warn("Missing scene or camera references");
      return;
    }

    this.drawingRecognition = new DrawingRecognitionManager(this.gameManager);

    try {
      await this.drawingRecognition.initialize();
      this.logger.log("Drawing recognition system initialized");
    } catch (error) {
      this.logger.error("Failed to initialize drawing recognition:", error);
      return;
    }

    if (this.gameManager) {
      this.gameManager.on("state:changed", (newState, oldState) => {
        if (
          newState.currentState === GAME_STATES.CURSOR &&
          oldState.currentState !== GAME_STATES.CURSOR
        ) {
          this.logger.log("Entering CURSOR state - activating drawing canvas");
          this.activateDrawingCanvas();
        } else if (
          oldState.currentState === GAME_STATES.CURSOR &&
          newState.currentState !== GAME_STATES.CURSOR
        ) {
          this.logger.log("Leaving CURSOR state - deactivating drawing canvas");
          this.deactivateDrawingCanvas();
        }

        if (newState.currentState === "drawingSuccess") {
          this.logger.log(
            "Drawing recognized successfully:",
            newState.prediction
          );
          this.handleDrawingSuccess(newState);
        } else if (newState.currentState === "drawingFailed") {
          this.logger.log("Drawing not recognized or incorrect");
          this.handleDrawingFailure(newState);
        }
      });
    }

    this.handleKeyPress = this.handleKeyPress.bind(this);
    window.addEventListener("keydown", this.handleKeyPress);

    if (
      this.gameManager?.state?.currentState === GAME_STATES.CURSOR &&
      !this.isActive
    ) {
      this.logger.log("Starting in CURSOR state - activating drawing canvas");
      this.activateDrawingCanvas();
    }
  }

  /**
   * Activate the drawing canvas in front of the camera
   */
  activateDrawingCanvas() {
    if (this.isActive) {
      this.logger.warn("Drawing canvas already active");
      return;
    }

    const cameraDirection = new THREE.Vector3();
    this.camera.getWorldDirection(cameraDirection);

    const cameraPosition = new THREE.Vector3();
    this.camera.getWorldPosition(cameraPosition);

    const canvasPosition = cameraPosition
      .clone()
      .add(cameraDirection.multiplyScalar(this.config.canvasDistance));
    canvasPosition.y += this.config.canvasHeight;

    this.drawingRecognition.createDrawingCanvas(
      this.scene,
      canvasPosition,
      this.config.canvasScale
    );

    const canvas = this.drawingRecognition.drawingCanvas;
    if (canvas) {
      canvas.mesh.lookAt(cameraPosition);
    }

    this.drawingRecognition.setExpectedDrawing(this.config.expectedDrawing);

    if (this.renderer) {
      this.drawingRecognition.enableDrawingMode(
        this.camera,
        this.renderer.domElement
      );
    } else {
      this.logger.warn("No renderer provided - drawing input disabled");
    }

    this.isActive = true;
    this.logger.log(
      `Drawing canvas activated at position: ${canvasPosition.toArray()}`
    );
  }

  /**
   * Deactivate the drawing canvas
   */
  deactivateDrawingCanvas() {
    if (!this.isActive) {
      return;
    }

    this.drawingRecognition.disableDrawingMode();

    if (this.drawingRecognition.drawingCanvas) {
      this.drawingRecognition.drawingCanvas.dispose();
      this.drawingRecognition.drawingCanvas = null;
    }

    this.isActive = false;
    this.logger.log("Drawing canvas deactivated");
  }

  /**
   * Handle keyboard input for drawing controls
   */
  handleKeyPress(event) {
    if (!this.isActive) return;

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      this.logger.log("Predicting drawing...");
      this.predict();
    } else if (
      event.key === "c" ||
      event.key === "C" ||
      event.key === "r" ||
      event.key === "R"
    ) {
      event.preventDefault();
      this.logger.log("Clearing canvas...");
      this.clearCanvas();
    }
  }

  /**
   * Handle successful drawing recognition
   */
  handleDrawingSuccess(stateData) {
    this.logger.log("Drawing success!", stateData);
  }

  /**
   * Handle failed drawing recognition
   */
  handleDrawingFailure(stateData) {
    this.logger.log("Drawing failed:", stateData);
  }

  /**
   * Trigger prediction on current drawing
   */
  async predict() {
    if (!this.isActive) {
      this.logger.warn("Drawing canvas not active");
      return null;
    }

    return await this.drawingRecognition.predictAndEvaluate();
  }

  /**
   * Clear the drawing canvas
   */
  clearCanvas() {
    if (this.isActive) {
      this.drawingRecognition.clearCanvas();
    }
  }

  /**
   * Set what the player should draw
   */
  setExpectedDrawing(label) {
    this.config.expectedDrawing = label;
    if (this.isActive) {
      this.drawingRecognition.setExpectedDrawing(label);
    }
  }

  /**
   * Update method - called in animation loop
   */
  update(dt) {
    if (this.isActive && this.drawingRecognition) {
      this.drawingRecognition.update();
    }
  }

  /**
   * Dispose of resources
   */
  dispose() {
    window.removeEventListener("keydown", this.handleKeyPress);
    this.deactivateDrawingCanvas();
    if (this.drawingRecognition) {
      this.drawingRecognition.dispose();
      this.drawingRecognition = null;
    }
  }
}

export default DrawingManager;
