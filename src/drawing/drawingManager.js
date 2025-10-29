import { GAME_STATES } from "../gameData.js";
import * as THREE from "three";
import { Logger } from "../utils/logger.js";

const EMOJI_MAP = {
  lightning: "⚡",
  star: "⭐",
  circle: "🔴",
};

const GAME_LABELS = ["lightning", "star", "circle"];

export class DrawingManager {
  constructor(scene, drawingRecognitionManager, gameManager) {
    this.scene = scene;
    this.recognitionManager = drawingRecognitionManager;
    this.gameManager = gameManager;
    this.logger = new Logger("DrawingManager", true);

    this.targetLabel = null;
    this.targetEmojiElement = null;
    this.feedbackContainer = null;
    this.submitButton = null;
    this.clearButton = null;
    this.isActive = false;

    this.canvasPosition = { x: 0, y: 1.5, z: -2 };
    this.canvasScale = 1;
    this.enableParticles = true; // Toggle to disable particle effects

    this.labelPool = [];
    this.refillLabelPool();

    this.setupUI();
    this.setupKeyboardShortcuts();
    this.bindGameStateListener();
  }

  refillLabelPool() {
    this.labelPool = [...GAME_LABELS];
    for (let i = this.labelPool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.labelPool[i], this.labelPool[j]] = [
        this.labelPool[j],
        this.labelPool[i],
      ];
    }
    this.logger.log("Label pool refilled and shuffled:", this.labelPool);
  }

  setupKeyboardShortcuts() {
    this.onKeyDown = (event) => {
      if (!this.isActive) return;

      // Ignore if typing in an input
      if (
        document.activeElement.tagName === "INPUT" ||
        document.activeElement.tagName === "TEXTAREA"
      ) {
        return;
      }

      switch (event.key.toLowerCase()) {
        case "r":
          this.handleClear();
          break;
        case "enter":
        case " ":
          event.preventDefault();
          this.handleSubmit();
          break;
      }
    };

    window.addEventListener("keydown", this.onKeyDown);
  }

  bindGameStateListener() {
    if (this.gameManager) {
      this.gameManager.on("state:changed", (newState, oldState) => {
        const isCursorState = newState.currentState === GAME_STATES.CURSOR;

        if (isCursorState && !this.isActive) {
          this.startGame();
        } else if (!isCursorState && this.isActive) {
          this.stopGame();
        }
      });
    }
  }

  setupUI() {
    const style = document.createElement("style");
    style.textContent = `
      .drawing-game-target {
        position: fixed;
        top: 5%;
        left: 50%;
        transform: translateX(-50%);
        font-size: 80px;
        background: rgba(0, 0, 0, 0.8);
        padding: 20px 40px;
        border-radius: 15px;
        border: 3px solid white;
        display: none;
        z-index: 999;
      }

      .drawing-game-target.active {
        display: flex;
        align-items: center;
        gap: 20px;
      }

      .drawing-game-result {
        font-size: 60px;
        opacity: 0;
        transition: opacity 0.3s;
      }

      .drawing-game-result.show {
        opacity: 1;
      }
    `;
    document.head.appendChild(style);

    this.targetEmojiElement = document.createElement("div");
    this.targetEmojiElement.className = "drawing-game-target";

    this.targetEmoji = document.createElement("span");
    this.resultEmoji = document.createElement("span");
    this.resultEmoji.className = "drawing-game-result";

    this.targetEmojiElement.appendChild(this.targetEmoji);
    this.targetEmojiElement.appendChild(this.resultEmoji);

    document.body.appendChild(this.targetEmojiElement);
  }

  startGame() {
    this.logger.log("Starting drawing game...");
    this.isActive = true;

    if (this.targetEmojiElement) {
      this.targetEmojiElement.classList.add("active");
      this.logger.log("Target emoji element activated");
    }

    this.logger.log("About to pick new target...");
    this.pickNewTarget();
    this.logger.log("Target picked, game ready");

    this.logger.log(
      "Checking if canvas exists:",
      this.recognitionManager.drawingCanvas
    );

    if (!this.recognitionManager.drawingCanvas) {
      this.logger.log("Creating drawing canvas...");

      // Use characterController's getPosition to place canvas in front of player
      // z: positive = forward, y: 0 = at eye level, x: 0 = centered
      const canvasPos = window.characterController.getPosition({
        x: 0,
        y: 0.8,
        z: -1.5,
      });

      this.logger.log("Canvas position:", canvasPos);

      this.recognitionManager.createDrawingCanvas(
        this.scene,
        canvasPos,
        this.canvasScale,
        this.enableParticles
      );

      const canvas = this.recognitionManager.drawingCanvas;
      if (canvas) {
        // Make canvas face the camera
        const camera = window.camera;
        const mesh = canvas.getMesh();
        if (mesh) {
          mesh.lookAt(camera.position);
        }

        // Position particles toward camera from the plane
        if (canvas.particleSystem) {
          const direction = new THREE.Vector3()
            .subVectors(camera.position, mesh.position)
            .normalize();
          canvas.particleSystem.position
            .copy(mesh.position)
            .add(direction.multiplyScalar(canvas.particleOffset));
          canvas.particleSystem.lookAt(camera.position);
        }
      }

      this.logger.log("Canvas created:", this.recognitionManager.drawingCanvas);

      const domElement =
        window.inputManager?.domElement || document.querySelector("canvas");

      this.logger.log(
        "Enabling drawing mode with camera:",
        camera,
        "domElement:",
        domElement
      );

      if (camera && domElement) {
        this.recognitionManager.enableDrawingMode(camera, domElement);
        this.logger.log("Drawing mode enabled");
      } else {
        this.logger.warn("Missing camera or domElement!");
      }
    } else {
      this.logger.log("Canvas already exists, skipping creation");
    }
  }

  stopGame() {
    this.logger.log("Stopping drawing game...");
    this.isActive = false;

    if (this.targetEmojiElement) {
      this.targetEmojiElement.classList.remove("active");
    }

    this.clearResult();

    if (this.recognitionManager.drawingCanvas) {
      this.recognitionManager.disableDrawingMode();
      this.recognitionManager.drawingCanvas.dispose();
      this.recognitionManager.drawingCanvas = null;
    }
  }

  pickNewTarget() {
    if (this.labelPool.length === 0) {
      this.refillLabelPool();
    }

    this.targetLabel = this.labelPool.pop();

    this.logger.log(
      `Picked new target: ${this.targetLabel} (${this.labelPool.length} remaining in pool)`
    );

    const setResult = this.recognitionManager.setExpectedDrawing(
      this.targetLabel
    );
    this.logger.log(`setExpectedDrawing returned: ${setResult}`);
    this.logger.log(
      `RecognitionManager.expectedDrawing is now: ${this.recognitionManager.expectedDrawing}`
    );

    const emoji = EMOJI_MAP[this.targetLabel];
    this.logger.log(`Emoji to display: ${emoji}`);

    if (this.targetEmoji) {
      this.targetEmoji.textContent = emoji;
      this.logger.log(`Updated display element to: ${emoji}`);
    } else {
      this.logger.warn(`targetEmoji is null!`);
    }
  }

  async handleSubmit() {
    if (!this.isActive) {
      this.logger.log("Submit ignored - game not active");
      return;
    }

    this.logger.log("Submitting drawing...");

    const result = await this.recognitionManager.predictAndEvaluate();

    this.logger.log("Result received:", result);

    if (!result || !result.prediction) {
      this.logger.log("No prediction, showing failure");
      this.showResult("❌");
      return;
    }

    const predictedEmoji = EMOJI_MAP[result.prediction] || "❓";

    this.logger.log(
      `Result: predicted=${result.prediction} (${result.confidence?.toFixed(
        2
      )}), expected=${result.expected}, success=${result.success}`
    );

    if (result.success) {
      this.logger.log("Success! Will pick new target in 2s");
      this.showResult("✅");
      setTimeout(() => {
        this.handleClear();
        this.pickNewTarget();
      }, 2000);
    } else {
      this.logger.log("Failed. Try again!");
      this.showResult(`${predictedEmoji} ❌`);
    }
  }

  handleClear() {
    if (!this.isActive) return;

    this.recognitionManager.clearCanvas();
    this.clearResult();
  }

  showResult(text) {
    if (!this.resultEmoji) return;

    this.resultEmoji.textContent = text;
    this.resultEmoji.classList.add("show");

    setTimeout(() => {
      this.clearResult();
    }, 3000);
  }

  clearResult() {
    if (!this.resultEmoji) return;

    this.resultEmoji.classList.remove("show");
    setTimeout(() => {
      this.resultEmoji.textContent = "";
    }, 300);
  }

  dispose() {
    this.stopGame();

    if (this.targetEmojiElement) {
      this.targetEmojiElement.remove();
    }

    if (this.onKeyDown) {
      window.removeEventListener("keydown", this.onKeyDown);
    }
  }
}
