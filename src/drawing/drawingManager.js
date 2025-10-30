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

  // Debug
  canvasGizmo: false, // Enable gizmo for positioning/scaling canvas (disabled - final position set)
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
    this.canvasGizmo = DRAWING_CONFIG.canvasGizmo;

    // Input manager reference (will be set via setInputManager)
    this.inputManager = null;
    this.originalGizmoProbe = null; // Store original probe to restore later
    this.canvasMesh = null; // Store canvas mesh reference
    this.canvasParticleSystem = null; // Store particle system reference for gizmo
    this.gizmoRegistered = false; // Track if gizmo has been registered

    // Pointer lock monitor
    this.pointerLockChangeHandler = null;

    this.labelPool = [];
    this.refillLabelPool();

    this.successCount = 0;
    this.maxRounds = 3;
    this.currentGoalRune = null;
    this.goalRunePositions = {
      lightning: null, // Will be set dynamically near canvas
      star: { x: 2.17, y: 2.06, z: 77.8 },
      circle: { x: -1.81, y: 6.31, z: 74.67 },
    };
    this.runeTargetOpacity = 0;
    this.runeCurrentOpacity = 0;

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

        // Update rune visibility based on viewmaster equipped state
        this.updateRuneVisibility(newState);
      });
    }
  }

  updateRuneVisibility(gameState) {
    const isEquipped = gameState?.isViewmasterEquipped || false;
    this.runeTargetOpacity = isEquipped ? 1.0 : 0.0;
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
    this.logger.log("GizmoManager available:", !!window.gizmoManager);
    this.logger.log("Canvas gizmo enabled:", this.canvasGizmo);
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

    if (this.recognitionManager.showEmojiUI && this.targetEmojiElement) {
      this.targetEmojiElement.classList.add("active");
      this.logger.log("Target emoji element activated");
    }

    this.recognitionManager.setOnStrokeEndCallback(() => {
      this.autoSubmitDrawing();
    });

    this.logger.log(
      "Checking if canvas exists:",
      this.recognitionManager.drawingCanvas
    );

    if (!this.recognitionManager.drawingCanvas) {
      this.logger.log("Creating drawing canvas...");

      // Fixed canvas position, rotation, and scale
      const canvasPos = { x: -2.85, y: 2.85, z: 87.45 };
      const canvasRot = { x: -3.1416, y: 0.1655, z: 3.1416 };
      const canvasScale = 1.5;

      this.logger.log("Canvas position:", canvasPos);

      this.recognitionManager.createDrawingCanvas(
        this.scene,
        canvasPos,
        canvasScale,
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
        const mesh = canvas.getMesh();
        if (mesh) {
          // Set position, rotation, and scale for all components
          mesh.position.set(canvasPos.x, canvasPos.y, canvasPos.z);
          mesh.rotation.set(canvasRot.x, canvasRot.y, canvasRot.z);
          mesh.scale.set(canvasScale, canvasScale, canvasScale);

          this.canvasMesh = mesh;
          this.canvasParticleSystem = canvas.particleSystem;

          // Register particle system (visible component) with gizmo manager if enabled
          if (this.canvasGizmo && canvas.particleSystem) {
            if (window.gizmoManager) {
              window.gizmoManager.registerObject(
                canvas.particleSystem,
                "drawing-canvas-particles",
                "drawing"
              );
              this.gizmoRegistered = true;
              this.logger.log(
                "✅ Registered drawing particle system with gizmo manager"
              );
            } else {
              this.logger.warn("❌ GizmoManager not found on window!");
            }
          } else if (this.canvasGizmo) {
            this.logger.warn(
              "Canvas gizmo enabled but particleSystem not found"
            );
          } else {
            this.logger.log("Canvas gizmo disabled in config");
          }
        }

        // Set particle system position, rotation, and scale to match mesh
        if (canvas.particleSystem) {
          canvas.particleSystem.position.set(
            canvasPos.x,
            canvasPos.y,
            canvasPos.z
          );
          canvas.particleSystem.rotation.set(
            canvasRot.x,
            canvasRot.y,
            canvasRot.z
          );
          canvas.particleSystem.scale.set(
            canvasScale,
            canvasScale,
            canvasScale
          );
        }

        // Set stroke mesh position, rotation, and scale to match
        if (canvas.strokeMesh) {
          canvas.strokeMesh.position.set(canvasPos.x, canvasPos.y, canvasPos.z);
          canvas.strokeMesh.rotation.set(canvasRot.x, canvasRot.y, canvasRot.z);
          canvas.strokeMesh.scale.set(canvasScale, canvasScale, canvasScale);
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

        // Store lightning rune position (near canvas)
        this.goalRunePositions.lightning = {
          x: canvasPos.x + 1.2,
          y: canvasPos.y,
          z: canvasPos.z,
        };
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

      // Get mesh reference for gizmo registration
      const mesh = this.recognitionManager.drawingCanvas.getMesh();
      const canvas = this.recognitionManager.drawingCanvas;
      if (mesh) {
        this.canvasMesh = mesh;
        this.canvasParticleSystem = canvas.particleSystem;

        // Register particle system (visible component) with gizmo manager if enabled
        if (this.canvasGizmo && canvas.particleSystem) {
          if (window.gizmoManager) {
            window.gizmoManager.registerObject(
              canvas.particleSystem,
              "drawing-canvas-particles",
              "drawing"
            );
            this.gizmoRegistered = true;
            this.logger.log(
              "✅ Registered existing drawing particle system with gizmo manager"
            );
          } else {
            this.logger.warn("❌ GizmoManager not found on window!");
          }
        } else if (this.canvasGizmo) {
          this.logger.warn("Canvas gizmo enabled but particleSystem not found");
        } else {
          this.logger.log("Canvas gizmo disabled in config");
        }

        // Set lightning rune position if not already set
        if (!this.goalRunePositions.lightning) {
          this.goalRunePositions.lightning = {
            x: mesh.position.x + 1.2,
            y: mesh.position.y,
            z: mesh.position.z,
          };
        }
      }
    }

    // Pick new target after canvas and position are set up
    this.logger.log("About to pick new target...");
    this.pickNewTarget();
    this.logger.log("Target picked, game ready");
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

    if (this.recognitionManager.showEmojiUI && this.targetEmojiElement) {
      this.targetEmojiElement.classList.remove("active");
    }

    this.clearResult();

    if (this.recognitionManager.drawingCanvas) {
      this.recognitionManager.disableDrawingMode();
      this.recognitionManager.drawingCanvas.dispose();
      this.recognitionManager.drawingCanvas = null;
    }

    // Unregister particle system from gizmo manager
    if (
      this.gizmoRegistered &&
      this.canvasParticleSystem &&
      window.gizmoManager
    ) {
      window.gizmoManager.unregisterObject(this.canvasParticleSystem);
      this.logger.log(
        "Unregistered drawing particle system from gizmo manager"
      );
      this.gizmoRegistered = false;
    }
    this.canvasParticleSystem = null;
    this.canvasMesh = null;

    if (this.currentGoalRune) {
      this.currentGoalRune.dispose();
      this.currentGoalRune = null;
    }

    if (window.runeManager) {
      window.runeManager.clearRunes();
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

    if (this.recognitionManager.showEmojiUI) {
      const emoji = EMOJI_MAP[this.targetLabel];
      this.logger.log(`Emoji to display: ${emoji}`);

      if (this.targetEmoji) {
        this.targetEmoji.textContent = emoji;
        this.logger.log(`Updated display element to: ${emoji}`);
      } else {
        this.logger.warn(`targetEmoji is null!`);
      }
    }

    // Update goal rune to show current target at its specific position
    const runePosition = this.goalRunePositions[this.targetLabel];
    if (window.runeManager && runePosition) {
      // Remove old rune if it exists
      if (this.currentGoalRune) {
        this.currentGoalRune.dispose();
      }

      // Create new rune for current target at its designated position
      const rune = window.runeManager.createRune(
        this.targetLabel,
        runePosition,
        {
          scale: 1.5,
          rotation: { x: 0, y: 0, z: 0 },
        }
      );

      // Make rune face the camera
      if (rune && rune.mesh) {
        const camera = window.camera;
        if (camera) {
          rune.mesh.lookAt(camera.position);
        }
      }

      this.currentGoalRune = rune;
      this.logger.log(
        `Created ${this.targetLabel} goal rune at position`,
        runePosition
      );
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
    if (!this.recognitionManager.showEmojiUI || !this.resultEmoji) return;

    this.resultEmoji.textContent = text;
    this.resultEmoji.classList.add("show");

    setTimeout(() => {
      this.clearResult();
    }, 3000);
  }

  clearResult() {
    if (!this.recognitionManager.showEmojiUI || !this.resultEmoji) return;

    this.resultEmoji.classList.remove("show");
    setTimeout(() => {
      this.resultEmoji.textContent = "";
    }, 300);
  }

  update(dt = 0.016) {
    if (!this.isActive) return;

    // Late registration: if gizmo is enabled but not yet registered, try again
    if (this.canvasGizmo && !this.gizmoRegistered) {
      if (this.canvasParticleSystem && window.gizmoManager) {
        window.gizmoManager.registerObject(
          this.canvasParticleSystem,
          "drawing-canvas-particles",
          "drawing"
        );
        this.gizmoRegistered = true;
        this.logger.log(
          "✅ Late-registered drawing particle system with gizmo manager"
        );
      }
    }

    // Smoothly interpolate rune opacity
    const opacityLerpSpeed = 3.0; // How fast opacity changes
    const opacityDiff = this.runeTargetOpacity - this.runeCurrentOpacity;
    this.runeCurrentOpacity += opacityDiff * opacityLerpSpeed * dt;

    // Apply opacity and color to current goal rune via shader uniforms
    if (this.currentGoalRune && this.currentGoalRune.material) {
      if (
        this.currentGoalRune.material.uniforms &&
        this.currentGoalRune.material.uniforms.uOpacity
      ) {
        this.currentGoalRune.material.uniforms.uOpacity.value =
          this.runeCurrentOpacity;
      }

      // Sync color with particle canvas color stage
      if (this.recognitionManager.drawingCanvas) {
        const canvas = this.recognitionManager.drawingCanvas;
        if (
          canvas.currentColor &&
          this.currentGoalRune.material.uniforms &&
          this.currentGoalRune.material.uniforms.uColor
        ) {
          // Apply the same color as the particle canvas (slightly darkened for stroke)
          this.currentGoalRune.material.uniforms.uColor.value
            .copy(canvas.currentColor)
            .multiplyScalar(0.8);
        }
      }

      this.currentGoalRune.visible = this.runeCurrentOpacity > 0.01;
    }
  }

  dispose() {
    this.stopGame();

    if (this.targetEmojiElement) {
      this.targetEmojiElement.remove();
    }

    if (this.onKeyDown) {
      window.removeEventListener("keydown", this.onKeyDown);
    }

    // Ensure particle system is unregistered
    if (
      this.gizmoRegistered &&
      this.canvasParticleSystem &&
      window.gizmoManager
    ) {
      window.gizmoManager.unregisterObject(this.canvasParticleSystem);
      this.gizmoRegistered = false;
    }
    this.canvasParticleSystem = null;
    this.canvasMesh = null;
  }
}
