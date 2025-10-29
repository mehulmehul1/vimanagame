import { GAME_STATES } from "../gameData.js";
import * as THREE from "three";

const EMOJI_MAP = {
  lightning: "⚡",
  star: "⭐",
  circle: "🔴",
};

const GAME_LABELS = ["lightning", "star", "circle"];

export class DrawingGame {
  constructor(scene, drawingRecognitionManager, gameManager) {
    this.scene = scene;
    this.recognitionManager = drawingRecognitionManager;
    this.gameManager = gameManager;

    this.targetLabel = null;
    this.targetEmojiElement = null;
    this.feedbackContainer = null;
    this.submitButton = null;
    this.clearButton = null;
    this.isActive = false;

    this.canvasPosition = { x: 0, y: 1.5, z: -2 };
    this.canvasScale = 1;

    this.setupUI();
    this.setupKeyboardShortcuts();
    this.bindGameStateListener();
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
    console.log("🎮 [DrawingGame] Starting drawing game...");
    this.isActive = true;

    if (this.targetEmojiElement) {
      this.targetEmojiElement.classList.add("active");
      console.log("🎮 [DrawingGame] Target emoji element activated");
    }

    console.log("🎮 [DrawingGame] About to pick new target...");
    this.pickNewTarget();
    console.log("🎮 [DrawingGame] Target picked, game ready");

    console.log(
      "🎮 [DrawingGame] Checking if canvas exists:",
      this.recognitionManager.drawingCanvas
    );

    if (!this.recognitionManager.drawingCanvas) {
      console.log("🎮 [DrawingGame] Creating drawing canvas...");

      // Use characterController's getPosition to place canvas in front of player
      // z: positive = forward, y: 0 = at eye level, x: 0 = centered
      const canvasPos = window.characterController.getPosition({
        x: 0,
        y: 0.8,
        z: -1.5,
      });

      console.log("🎮 [DrawingGame] Canvas position:", canvasPos);

      this.recognitionManager.createDrawingCanvas(
        this.scene,
        canvasPos,
        this.canvasScale
      );

      const canvas = this.recognitionManager.drawingCanvas;
      if (canvas) {
        // Make canvas face the camera
        const camera = window.camera;
        canvas.mesh.lookAt(camera.position);
      }

      console.log(
        "🎮 [DrawingGame] Canvas created:",
        this.recognitionManager.drawingCanvas
      );

      const domElement =
        window.inputManager?.domElement || document.querySelector("canvas");

      console.log(
        "🎮 [DrawingGame] Enabling drawing mode with camera:",
        camera,
        "domElement:",
        domElement
      );

      if (camera && domElement) {
        this.recognitionManager.enableDrawingMode(camera, domElement);
        console.log("🎮 [DrawingGame] Drawing mode enabled");
      } else {
        console.log("🎮 [DrawingGame] ⚠️ Missing camera or domElement!");
      }
    } else {
      console.log("🎮 [DrawingGame] Canvas already exists, skipping creation");
    }
  }

  stopGame() {
    console.log("Stopping drawing game...");
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
    const randomIndex = Math.floor(Math.random() * GAME_LABELS.length);
    this.targetLabel = GAME_LABELS[randomIndex];

    console.log(`🎮 [DrawingGame] Picked new target: ${this.targetLabel}`);

    const setResult = this.recognitionManager.setExpectedDrawing(
      this.targetLabel
    );
    console.log(`🎮 [DrawingGame] setExpectedDrawing returned: ${setResult}`);
    console.log(
      `🎮 [DrawingGame] RecognitionManager.expectedDrawing is now: ${this.recognitionManager.expectedDrawing}`
    );

    const emoji = EMOJI_MAP[this.targetLabel];
    console.log(`🎮 [DrawingGame] Emoji to display: ${emoji}`);

    if (this.targetEmoji) {
      this.targetEmoji.textContent = emoji;
      console.log(`🎮 [DrawingGame] Updated display element to: ${emoji}`);
    } else {
      console.log(`🎮 [DrawingGame] ⚠️ targetEmoji is null!`);
    }
  }

  async handleSubmit() {
    if (!this.isActive) {
      console.log("❌ Submit ignored - game not active");
      return;
    }

    console.log("🎮 [DrawingGame] Submitting drawing...");

    const result = await this.recognitionManager.predictAndEvaluate();

    console.log("🎮 [DrawingGame] Result received:", result);

    if (!result || !result.prediction) {
      console.log("🎮 [DrawingGame] No prediction, showing failure");
      this.showResult("❌");
      return;
    }

    const predictedEmoji = EMOJI_MAP[result.prediction] || "❓";

    console.log(
      `🎮 [DrawingGame] Result: predicted=${
        result.prediction
      } (${result.confidence?.toFixed(2)}), expected=${
        result.expected
      }, success=${result.success}`
    );

    if (result.success) {
      console.log("🎮 [DrawingGame] Success! Will pick new target in 2s");
      this.showResult("✅");
      setTimeout(() => {
        this.handleClear();
        this.pickNewTarget();
      }, 2000);
    } else {
      console.log("🎮 [DrawingGame] Failed. Try again!");
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
