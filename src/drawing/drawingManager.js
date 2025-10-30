import { GAME_STATES } from "../gameData.js";
import * as THREE from "three";
import { Logger } from "../utils/logger.js";

// ==================== PARTICLE & DRAWING CONFIG ====================
// Tweak these values to customize particle behavior and gameplay
const DRAWING_CONFIG = {
  // Particle physics
  strokeRepulsionDistance: 0.05, // How far strokes push particles (lower = tighter margin)
  strokeRepulsionFalloff: "smooth", // "linear" for hard edge, "smooth" for soft gradient
  strokeRepulsionSmoothness: 1.0, // 0=linear, 1.0=smoothstep, higher=even smoother

  // Add more configs here as needed for other tunable parameters
};

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
    this.strokeRepulsionDistance = DRAWING_CONFIG.strokeRepulsionDistance;
    this.strokeRepulsionFalloff = DRAWING_CONFIG.strokeRepulsionFalloff;
    this.strokeRepulsionSmoothness = DRAWING_CONFIG.strokeRepulsionSmoothness;

    // Input manager reference (will be set via setInputManager)
    this.inputManager = null;
    this.originalGizmoProbe = null; // Store original probe to restore later

    // Pointer lock monitor
    this.pointerLockChangeHandler = null;

    this.labelPool = [];
    this.refillLabelPool();

    this.successCount = 0;
    this.maxRounds = 3;

    this.setupUI();
    this.setupKeyboardShortcuts();
    this.bindGameStateListener();
  }

  setInputManager(inputManager) {
    this.inputManager = inputManager;
    this.logger.log("Input manager reference set");
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
        const isCursorState =
          newState.currentState === GAME_STATES.CURSOR ||
          newState.currentState === GAME_STATES.CURSOR_FINAL;

        if (isCursorState && !this.isActive) {
          this.startGame(newState.currentState === GAME_STATES.CURSOR_FINAL);
        } else if (!isCursorState && this.isActive) {
          this.stopGame();
        }

      });
    }
  }

  setupUI() {
    const style = document.createElement("style");
    style.textContent = `
      body.drawing-game-cursor {
        cursor: grab !important;
      }

      body.drawing-game-cursor:active {
        cursor: grabbing !important;
      }

      body.drawing-game-cursor * {
        cursor: grab !important;
      }

      body.drawing-game-cursor *:active {
        cursor: grabbing !important;
      }

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

  startGame(isFinalRound = false) {
    this.logger.log(
      `Starting drawing game... ${isFinalRound ? "(FINAL ROUND)" : ""}`
    );
    this.logger.log("InputManager reference:", this.inputManager);
    this.isActive = true;
    this.successCount = isFinalRound ? 2 : 0; // Start at 2/3 for final round

    // Use window.inputManager as fallback if not set
    const inputMgr = this.inputManager || window.inputManager;
    this.logger.log("Using inputManager:", inputMgr);

    // Block pointer lock and enable drag-to-look
    if (inputMgr) {
      this.logger.log("Calling setPointerLockBlocked(true)...");
      inputMgr.setPointerLockBlocked(true);

      // Store the existing gizmo probe (if any) and combine it with our canvas probe
      const existingProbe = inputMgr.gizmoProbe;
      this.originalGizmoProbe = existingProbe;

      inputMgr.setGizmoProbe(() => {
        // Check both the drawing canvas AND any existing probe (like gizmo manager)
        const overCanvas = this.recognitionManager.isPointerOverDrawingCanvas();
        const overExisting = existingProbe ? existingProbe() : false;
        return overCanvas || overExisting;
      });

      this.logger.log(
        "Pointer lock blocked, drag-to-look enabled, canvas probe combined"
      );
    } else {
      this.logger.warn("No inputManager available!");
    }

    // Add gloved hand cursor to body
    document.body.classList.add("drawing-game-cursor");

    // Monitor pointer lock changes and force exit if engaged during CURSOR state
    this.pointerLockChangeHandler = () => {
      if (this.isActive && document.pointerLockElement) {
        this.logger.log(
          "Pointer lock engaged during CURSOR state, forcing exit"
        );
        document.exitPointerLock();
      }
    };
    document.addEventListener(
      "pointerlockchange",
      this.pointerLockChangeHandler
    );
    document.addEventListener(
      "mozpointerlockchange",
      this.pointerLockChangeHandler
    );
    document.addEventListener(
      "webkitpointerlockchange",
      this.pointerLockChangeHandler
    );

    if (this.targetEmojiElement) {
      this.targetEmojiElement.classList.add("active");
      this.logger.log("Target emoji element activated");
    }

    this.logger.log("About to pick new target...");
    this.pickNewTarget();
    this.logger.log("Target picked, game ready");

    this.recognitionManager.setOnStrokeEndCallback(() => {
      this.autoSubmitDrawing();
    });

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
        this.enableParticles,
        this.strokeRepulsionDistance,
        this.strokeRepulsionFalloff,
        this.strokeRepulsionSmoothness
      );

      const canvas = this.recognitionManager.drawingCanvas;
      if (canvas) {
        if (canvas.particleSystem) {
          canvas.particleSystem.visible = true;
        }
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

        // If starting in final round, set to red particles immediately
        if (isFinalRound) {
          this.logger.log("Setting canvas to FINAL ROUND state (red, stage 2)");
          canvas.colorStage = 2;
          canvas.currentColor.copy(canvas.redColor);
          canvas.currentJitterIntensity = 0.5;

          if (canvas.particleSystem && canvas.particleSystem.material) {
            canvas.particleSystem.material.uniforms.uColor.value.copy(
              canvas.redColor
            );
            canvas.particleSystem.material.uniforms.uJitterIntensity.value = 0.5;
          }

          if (canvas.strokeMesh && canvas.strokeMesh.material) {
            const strokeColor = canvas.redColor.clone().multiplyScalar(0.8);
            canvas.strokeMesh.material.uniforms.uColor.value.copy(strokeColor);
          }
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
      if (this.recognitionManager.drawingCanvas?.particleSystem) {
        this.recognitionManager.drawingCanvas.particleSystem.visible = true;
      }
    }
  }

  stopGame() {
    this.logger.log("Stopping drawing game...");
    this.isActive = false;

    this.recognitionManager.setOnStrokeEndCallback(null);

    if (this.pointerLockChangeHandler) {
      document.removeEventListener(
        "pointerlockchange",
        this.pointerLockChangeHandler
      );
      document.removeEventListener(
        "mozpointerlockchange",
        this.pointerLockChangeHandler
      );
      document.removeEventListener(
        "webkitpointerlockchange",
        this.pointerLockChangeHandler
      );
      this.pointerLockChangeHandler = null;
    }

    // Use window.inputManager as fallback if not set
    const inputMgr = this.inputManager || window.inputManager;

    // Unblock pointer lock (allow pointer lock outside CURSOR state)
    if (inputMgr) {
      inputMgr.setPointerLockBlocked(false);

      // Restore the original gizmo probe (if there was one)
      inputMgr.setGizmoProbe(this.originalGizmoProbe || null);

      this.logger.log("Pointer lock unblocked, canvas probe restored");
    }

    // Remove gloved hand cursor from body
    document.body.classList.remove("drawing-game-cursor");

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

  async autoSubmitDrawing() {
    if (!this.isActive) {
      return;
    }

    if (!this.recognitionManager.drawingCanvas?.hasStrokes()) {
      return;
    }

    await this.handleSubmit();
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
      this.logger.log("No prediction");
      return;
    }

    this.logger.log(
      `Result: predicted=${result.prediction} (${result.confidence?.toFixed(
        2
      )}), expected=${result.expected}, success=${result.success}`
    );

    if (result.success) {
      this.successCount++;
      this.logger.log(`Success ${this.successCount}/${this.maxRounds}!`);
      this.showResult("✅");

      // Trigger particle pulse immediately on success
      if (this.recognitionManager.drawingCanvas) {
        const canvas = this.recognitionManager.drawingCanvas;
        canvas.triggerPulse();
        canvas.triggerStrokePulse();

        // Check if this is the third success (red stage -> explosion)
        this.logger.log(
          `Checking for explosion: colorStage=${canvas.colorStage}`
        );

        if (canvas.colorStage === 2) {
          // Trigger explosion instead of normal color cycle
          this.logger.log("EXPLOSION TRIGGERED!");

          // Set explosion state directly
          canvas.isExploding = true;
          canvas.explosionProgress = 0;

          // If this is the final round, don't recreate particles after explosion
          if (this.successCount >= this.maxRounds) {
            canvas.skipRecreateAfterExplosion = true;
            this.logger.log("Final explosion - particles will not recreate");
          }

          this.logger.log(
            `After setting - isExploding=${canvas.isExploding}, explosionProgress=${canvas.explosionProgress}`
          );

          // Wait for explosion to complete (2.5s) + buffer
          const explosionDuration = 2.5;
          setTimeout(() => {
            this.logger.log("Explosion complete");

            // After 3 successes, end the game and transition to POST_CURSOR
            if (this.successCount >= this.maxRounds) {
              this.logger.log(
                "Drawing game complete! Transitioning to POST_CURSOR"
              );
              this.stopGame();

              // Transition to POST_CURSOR state
              if (this.gameManager) {
                this.gameManager.setState({
                  currentState: GAME_STATES.POST_CURSOR,
                });
              }
            } else {
              this.logger.log("Picking new target");
              this.pickNewTarget();
            }
          }, explosionDuration * 1000 + 300);
        } else {
          // Normal color cycle (blue -> orange, orange -> red)
          this.logger.log(
            `Normal color cycle: advancing from stage ${canvas.colorStage}`
          );
          setTimeout(() => {
            this.handleClear(true); // true = success, trigger color change

            // If this was the second success (advancing to red stage), update game state to CURSOR_FINAL
            if (this.successCount === 2) {
              this.logger.log("Advancing to CURSOR_FINAL state");
              if (this.gameManager) {
                this.gameManager.setState({
                  currentState: GAME_STATES.CURSOR_FINAL,
                });
              }
            }

            this.pickNewTarget();
          }, 1500);
        }
      }
    }
  }

  handleClear(advanceColor = false) {
    if (!this.isActive) return;

    // Trigger color cycle on success (syncs with particle reformation)
    if (advanceColor && this.recognitionManager.drawingCanvas) {
      this.recognitionManager.drawingCanvas.triggerColorCycle();
    }

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
