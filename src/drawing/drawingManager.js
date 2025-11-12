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
  canvasGizmo: false, // Enable gizmo for positioning/scaling canvas
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
    this.logger = new Logger("DrawingManager", false);

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
    this.previousFractalIntensity = 0; // Track previous fractal intensity for delay detection
    this.jitterResetTimeout = null; // Timeout for delayed jitter reset

    // Pointer lock monitor
    this.pointerLockChangeHandler = null;

    this.labelPool = [];
    this.usedRunes = []; // Track which runes have been used in current game session
    this.refillLabelPool();

    this.successCount = 0;
    this.maxRounds = 3;
    this.currentGoalRune = null;
    this.gameStartTime = 0; // Track when game starts to prevent immediate predictions
    this.goalRunePositions = {
      lightning: null, // Will be set dynamically near canvas
      star: { x: 2.17, y: 2.06, z: 77.8 },
      circle: { x: -1.81, y: 6.31, z: 74.67 },
    };
    this.runeTargetOpacity = 0;
    this.runeCurrentOpacity = 0;

    // Billboard rotation easing
    this.billboardLerpSpeed = 2.0; // How fast rotation catches up (higher = faster)
    this.targetQuaternion = new THREE.Quaternion();
    this.currentQuaternion = new THREE.Quaternion();

    // Shake animation state (for incorrect guesses)
    this.isShaking = false;
    this.shakeProgress = 0;
    this.shakeDuration = 2.5; // 50% faster than before (was 2.0)
    this.shakeAmplitude = 0.15; // Radians (~8.6 degrees)
    this.shakeFrequency = 8.0; // Number of oscillations during shake (2x wag)
    this.shakeBaseQuaternion = new THREE.Quaternion(); // Store base rotation before shake

    this.setupUI();
    this.setupKeyboardShortcuts();
    this.bindGameStateListener();
  }

  setInputManager(inputManager) {
    this.inputManager = inputManager;
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
      // Check initial state in case we're already in CURSOR state (e.g., debug spawn)
      // Delay slightly to ensure dependencies (camera, inputManager) are ready
      setTimeout(() => {
        const currentState = this.gameManager.getState();
        if (currentState) {
          const isCursorState =
            currentState.currentState === GAME_STATES.CURSOR ||
            currentState.currentState === GAME_STATES.CURSOR_FINAL;

          if (isCursorState && !this.isActive) {
            this.startGame(
              currentState.currentState === GAME_STATES.CURSOR_FINAL
            );
            this.updateRuneVisibility(currentState);
          }
        }
      }, 100);

      // Listen for future state changes
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

      #space-bar-hint {
        position: fixed;
        bottom: 7%;
        left: 5%;
        opacity: 0;
        pointer-events: none;
        z-index: 1000;
        transition: opacity 0.3s ease-in;
      }

      #space-bar-hint:not(.progress-mode):not(.captions-active) {
        pointer-events: all !important;
      }

      #space-bar-hint.mobile-button {
        bottom: auto;
        top: 5%;
        left: 50%;
        transform: translateX(-50%);
        pointer-events: all;
      }

      #space-bar-hint.pressed {
        transform: scale(0.95);
        opacity: 0.8 !important;
      }

      #space-bar-hint img {
        width: auto;
        height: 60px;
        display: block;
      }

      #space-bar-hint.mobile-button.pressed {
        transform: translateX(-50%) scale(0.95);
        opacity: 0.8;
      }

      /* Hide mobile button when captions are showing */
      #space-bar-hint.mobile-button.captions-active {
        opacity: 0 !important;
        pointer-events: none;
      }

      @keyframes space-bar-hint-cycle {
        0% {
          opacity: 1;
        }
        20% {
          opacity: 0.4;
        }
        30% {
          opacity: 1;
        }
        40% {
          opacity: 0.4;
        }
        50% {
          opacity: 1;
        }
        60% {
          opacity: 0.4;
        }
        70% {
          opacity: 1;
        }
        80% {
          opacity: 0.4;
        }
        100% {
          opacity: 1;
        }
      }

      #space-bar-hint.animating {
        animation: space-bar-hint-cycle 7s ease-in-out infinite;
      }

      /* Make hint fully white when viewmaster can be toggled on (not in progress mode) */
      #space-bar-hint:not(.progress-mode) img {
        filter: none !important;
      }

      /* Prevent captions from overlapping space bar hint */
      body.drawing-game-cursor #dialog-caption {
        left: calc(5% + 200px);
        right: calc(5% + 200px);
        max-width: none;
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

    // Create space bar hint
    const currentState = this.gameManager?.getState?.();
    const isIOS = currentState?.isIOS === true;
    const isMobile = currentState?.isMobile === true;

    this.spaceBarHintElement = document.createElement("div");
    this.spaceBarHintElement.id = "space-bar-hint";

    // Make clickable on all platforms
    this.spaceBarHintElement.style.cursor = "pointer";

    // Click handler for all platforms - prioritize viewmaster toggle like spacebar
    this.spaceBarHintElement.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();

      // Debounce: ignore clicks within 1 second of last click
      const now = Date.now();
      if (this._lastButtonClickTime && now - this._lastButtonClickTime < 1000) {
        return;
      }
      this._lastButtonClickTime = now;

      // Check viewmaster toggle first (same as spacebar) - works whether equipped or not
      if (
        window.viewmasterController &&
        window.viewmasterController.isToggleEnabled
      ) {
        const currentState = this.gameManager?.getState();
        if (currentState) {
          const newEquippedState = !currentState.isViewmasterEquipped;
          // Reset manually removed flag when user clicks to toggle
          this.gameManager.setState({
            isViewmasterEquipped: newEquippedState,
            viewmasterManuallyRemoved: false,
          });
          return;
        }
      }

      // Fall back to drawing submission if drawing game is active
      if (this.isActive) {
        this.handleSubmit();
      }
    });

    if (isMobile || isIOS) {
      // Mobile: button at top center
      this.spaceBarHintElement.classList.add("mobile-button");

      // Also handle touch events for better mobile support
      this.spaceBarHintElement.addEventListener("touchend", (e) => {
        e.preventDefault();
        e.stopPropagation();

        // Debounce: ignore touches within 1 second of last click/touch
        const now = Date.now();
        if (
          this._lastButtonClickTime &&
          now - this._lastButtonClickTime < 1000
        ) {
          return;
        }
        this._lastButtonClickTime = now;

        // Check viewmaster toggle first (same as click handler) - works whether equipped or not
        if (
          window.viewmasterController &&
          window.viewmasterController.isToggleEnabled
        ) {
          const currentState = this.gameManager?.getState();
          if (currentState) {
            const newEquippedState = !currentState.isViewmasterEquipped;
            // Reset manually removed flag when user touches to toggle
            this.gameManager.setState({
              isViewmasterEquipped: newEquippedState,
              viewmasterManuallyRemoved: false,
            });
            return;
          }
        }

        // Fall back to drawing submission if drawing game is active
        if (this.isActive) {
          this.handleSubmit();
        }
      });

      this.spaceBarHintElement.addEventListener("touchstart", (e) => {
        e.preventDefault();
        this.spaceBarHintElement.classList.add("pressed");
      });

      this.spaceBarHintElement.addEventListener("touchend", (e) => {
        e.preventDefault();
        this.spaceBarHintElement.classList.remove("pressed");
      });
    } else {
      // Desktop: add click feedback
      this.spaceBarHintElement.addEventListener("mousedown", (e) => {
        e.preventDefault();
        this.spaceBarHintElement.classList.add("pressed");
      });

      this.spaceBarHintElement.addEventListener("mouseup", (e) => {
        e.preventDefault();
        this.spaceBarHintElement.classList.remove("pressed");
      });

      this.spaceBarHintElement.addEventListener("mouseleave", () => {
        this.spaceBarHintElement.classList.remove("pressed");
      });
    }

    const img = document.createElement("img");
    img.src = "/images/SpaceBarHint.svg";
    img.alt = "Press Space to submit";

    this.spaceBarHintElement.appendChild(img);
    document.body.appendChild(this.spaceBarHintElement);
  }

  async startGame(isFinalRound = false) {
    // Ensure recognition manager is initialized (lazy loading)
    // This will load TensorFlow.js assets (~15MB) only when needed
    if (!this.recognitionManager.isModelLoaded) {
      this.logger.log(
        "Initializing DrawingRecognitionManager (lazy load TensorFlow.js)..."
      );
      try {
        await this.recognitionManager.ensureInitialized();
        this.logger.log("✅ DrawingRecognitionManager initialized");
      } catch (error) {
        this.logger.error(
          "❌ Failed to initialize DrawingRecognitionManager:",
          error
        );
        return; // Don't start game if initialization failed
      }
    }

    // Check if camera and inputManager are available before proceeding
    const camera = window.camera;
    const inputMgr = this.inputManager || window.inputManager;

    if (!camera) {
      this.logger.warn("Camera not available yet, retrying in 100ms...");
      setTimeout(() => {
        if (!this.isActive) {
          this.startGame(isFinalRound);
        }
      }, 100);
      return;
    }

    if (!inputMgr) {
      this.logger.warn("InputManager not available yet, retrying in 100ms...");
      setTimeout(() => {
        if (!this.isActive) {
          this.startGame(isFinalRound);
        }
      }, 100);
      return;
    }

    this.isActive = true;
    this.successCount = isFinalRound ? 2 : 0; // Start at 2/3 for final round
    this.gameStartTime = Date.now(); // Track when game starts

    // Initialize drawing state in game manager
    if (this.gameManager) {
      this.gameManager.setState({
        drawingSuccessCount: this.successCount,
        drawingFailureCount: 0,
        lastDrawingSuccess: null,
        currentDrawingTarget: null,
      });
    }

    // Block pointer lock and enable drag-to-look
    if (inputMgr) {
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
    } else {
      this.logger.warn("No inputManager available!");
    }

    // Add gloved hand cursor to body
    document.body.classList.add("drawing-game-cursor");

    // Monitor pointer lock changes and force exit if engaged during CURSOR state
    this.pointerLockChangeHandler = () => {
      if (this.isActive && document.pointerLockElement) {
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
    }

    // Delay registering stroke callback to prevent initial false triggers
    setTimeout(() => {
      if (this.isActive) {
        this.recognitionManager.setOnStrokeEndCallback(() => {
          this.autoSubmitDrawing();
        });
      }
    }, 1500);

    if (!this.recognitionManager.drawingCanvas) {
      // Fixed canvas position, rotation, and scale
      const canvasPos = { x: -8.77, y: 2.85, z: 81.87 };
      const canvasRot = { x: 0.0, y: 1.4047, z: -0.0 };
      const canvasScale = 1.5;

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
        if (canvas.strokeMesh && canvas.strokeMesh.mesh) {
          canvas.strokeMesh.mesh.visible = true;
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

        // If starting in final round, set to white particles immediately
        if (isFinalRound) {
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
          x: canvasPos.x + 2.2,
          y: canvasPos.y,
          z: canvasPos.z,
        };
      }

      const domElement =
        window.inputManager?.domElement || document.querySelector("canvas");
      const camera = window.camera;

      if (camera && domElement) {
        this.recognitionManager.enableDrawingMode(camera, domElement);
      } else {
        this.logger.warn("Missing camera or domElement!");
      }
    } else {
      if (this.recognitionManager.drawingCanvas?.particleSystem) {
        this.recognitionManager.drawingCanvas.particleSystem.visible = true;
      }
      if (this.recognitionManager.drawingCanvas?.strokeMesh?.mesh) {
        this.recognitionManager.drawingCanvas.strokeMesh.mesh.visible = true;
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

    // Clear any existing strokes from previous sessions
    if (this.recognitionManager.drawingCanvas) {
      this.recognitionManager.clearCanvas();

      // Initialize billboard quaternion from current canvas rotation
      const mesh = this.recognitionManager.drawingCanvas.getMesh();
      if (mesh) {
        this.currentQuaternion.copy(mesh.quaternion);
      }
    }

    // Pick new target after canvas and position are set up
    // Ensure pool is fresh at game start
    this.refillLabelPool();
    this.pickNewTarget();

    // Start flashing space bar hint on desktop (not mobile)
    if (this.spaceBarHintElement) {
      const currentState = this.gameManager?.getState?.();
      const isMobile = currentState?.isMobile === true;
      const isIOS = currentState?.isIOS === true;

      if (!isMobile && !isIOS) {
        this.startSpaceBarHintAnimation();
      } else {
        // Mobile: show button only if captions are not active
        this.updateMobileButtonVisibility();
      }
    }
  }

  updateMobileButtonVisibility() {
    if (
      !this.spaceBarHintElement ||
      !this.spaceBarHintElement.classList.contains("mobile-button")
    ) {
      return;
    }

    // Only show button between CURSOR and POST_CURSOR states
    const gameState = this.gameManager?.getState?.();
    const currentState = gameState?.currentState;
    if (
      currentState === undefined ||
      currentState < GAME_STATES.CURSOR ||
      currentState >= GAME_STATES.POST_CURSOR
    ) {
      this.spaceBarHintElement.style.opacity = "0";
      this.spaceBarHintElement.style.pointerEvents = "none";
      return;
    }

    // Check if captions are showing
    const captionElement = document.getElementById("dialog-caption");
    const hasCaptions =
      captionElement && captionElement.textContent.trim().length > 0;

    if (hasCaptions) {
      this.spaceBarHintElement.classList.add("captions-active");
    } else {
      this.spaceBarHintElement.classList.remove("captions-active");
      // Show button if viewmaster toggle is enabled or drawing game is active
      const isViewmasterEnabled =
        window.viewmasterController?.isToggleEnabled === true;
      const isViewmasterEquipped = gameState?.isViewmasterEquipped === true;

      if (this.isActive || isViewmasterEnabled || isViewmasterEquipped) {
        this.spaceBarHintElement.style.opacity = "1";
        this.spaceBarHintElement.style.pointerEvents = "all";
      } else {
        this.spaceBarHintElement.style.opacity = "0";
        this.spaceBarHintElement.style.pointerEvents = "none";
      }
    }
  }

  stopGame() {
    this.isActive = false;
    this.gameStartTime = 0;
    this.usedRunes = []; // Reset used runes when game stops

    // Stop space bar hint animation
    if (this.spaceBarHintElement) {
      this.stopSpaceBarHintAnimation();
    }

    // Reset shake animation state
    this.isShaking = false;
    this.shakeProgress = 0;

    // Clear jitter reset timeout
    if (this.jitterResetTimeout) {
      clearTimeout(this.jitterResetTimeout);
      this.jitterResetTimeout = null;
    }
    this.previousFractalIntensity = 0;

    // Reset drawing state in game manager
    if (this.gameManager) {
      this.gameManager.setState({
        drawingSuccessCount: 0,
        drawingFailureCount: 0,
        lastDrawingSuccess: null,
        currentDrawingTarget: null,
      });
    }

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
    }

    // Remove gloved hand cursor from body
    document.body.classList.remove("drawing-game-cursor");

    if (this.recognitionManager.showEmojiUI && this.targetEmojiElement) {
      this.targetEmojiElement.classList.remove("active");
    }

    this.clearResult();

    // Clear fractal effects and hide canvas (don't dispose - just hide)
    if (this.recognitionManager.drawingCanvas) {
      const canvas = this.recognitionManager.drawingCanvas;

      // Reset canvas internal state to stop any ongoing animations
      canvas.isExploding = false;
      canvas.explosionProgress = 0;
      canvas.isSuccessAnimating = false;
      canvas.successAnimProgress = 0;
      canvas.isPulsing = false;
      canvas.pulseProgress = 0;
      canvas.isStrokePulsing = false;
      canvas.strokePulseProgress = 0;
      canvas.isTransitioningColor = false;
      canvas.colorTransitionProgress = 0;
      canvas.currentJitterIntensity = 0;

      // Reset particle jitter to base state (no fractal effects)
      if (canvas.particleSystem && canvas.particleSystem.material) {
        canvas.particleSystem.material.uniforms.uJitterIntensity.value = 0;
        canvas.particleSystem.material.uniforms.uPulseScale.value = 1.0;
        canvas.particleSystem.material.uniforms.uExplosionFactor.value = 0;
        canvas.particleSystem.material.uniforms.uImplosionFactor.value = 0;
        // Hide particle system
        canvas.particleSystem.visible = false;
      }

      // Reset stroke mesh fractal warping and hide it
      if (canvas.strokeMesh) {
        if (
          canvas.strokeMesh.material &&
          canvas.strokeMesh.material.uniforms.uFractalIntensity
        ) {
          canvas.strokeMesh.material.uniforms.uFractalIntensity.value = 0;
        }
        // Hide stroke mesh
        if (canvas.strokeMesh.mesh) {
          canvas.strokeMesh.mesh.visible = false;
        }
      }

      this.recognitionManager.disableDrawingMode();
      // Don't dispose - just hide it so we can reuse it
      // this.recognitionManager.drawingCanvas.dispose();
      // this.recognitionManager.drawingCanvas = null;
    }

    // Unregister particle system from gizmo manager
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

    if (this.currentGoalRune) {
      this.currentGoalRune.dispose();
      this.currentGoalRune = null;
    }

    if (window.runeManager) {
      window.runeManager.clearRunes();
    }
  }

  pickNewTarget() {
    // If pool is empty, refill it
    if (this.labelPool.length === 0) {
      this.refillLabelPool();
    }

    // If all 3 runes have been used in this game session, reset the used list
    // This ensures we use all 3 runes before any repeats
    if (this.usedRunes.length >= GAME_LABELS.length) {
      this.usedRunes = [];
      this.logger.log("All runes used, resetting used runes list");
    }

    // Filter out already-used runes from the pool
    const availableRunes = this.labelPool.filter(
      (rune) => !this.usedRunes.includes(rune)
    );

    // If no unused runes available, use any rune from pool (all have been used)
    const runesToChooseFrom =
      availableRunes.length > 0 ? availableRunes : this.labelPool;

    // Pick a random rune from available ones
    const randomIndex = Math.floor(Math.random() * runesToChooseFrom.length);
    this.targetLabel = runesToChooseFrom[randomIndex];

    // Remove the selected rune from the pool
    const poolIndex = this.labelPool.indexOf(this.targetLabel);
    if (poolIndex !== -1) {
      this.labelPool.splice(poolIndex, 1);
    }

    // Track that this rune has been used
    if (!this.usedRunes.includes(this.targetLabel)) {
      this.usedRunes.push(this.targetLabel);
    }

    // Update game state with new target
    if (this.gameManager) {
      this.gameManager.setState({
        currentDrawingTarget: this.targetLabel,
      });
    }

    this.recognitionManager.setExpectedDrawing(this.targetLabel);

    if (this.recognitionManager.showEmojiUI) {
      const emoji = EMOJI_MAP[this.targetLabel];

      if (this.targetEmoji) {
        this.targetEmoji.textContent = emoji;
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

    // Ignore predictions that happen too soon after game start (within 1.5 seconds)
    const timeSinceStart = Date.now() - this.gameStartTime;
    if (timeSinceStart < 1500) {
      this.logger.log(
        `Submit ignored - too soon after init (${timeSinceStart}ms)`
      );
      return;
    }

    // Verify there are strokes to evaluate
    if (!this.recognitionManager.drawingCanvas?.hasStrokes()) {
      this.logger.log("Submit ignored - no strokes");
      return;
    }

    this.logger.log("Submitting drawing...");

    const result = await this.recognitionManager.predictAndEvaluate();

    this.logger.log("Result received:", result);

    if (!result || !result.prediction) {
      this.logger.log("No prediction - ignoring for gameplay");
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

      // Update game state and clear flag after a delay to allow dialog to trigger
      if (this.gameManager) {
        this.gameManager.setState({
          drawingSuccessCount: this.successCount,
          lastDrawingSuccess: true,
        });

        // Clear flag after a short delay to allow dialog system to process the state change
        setTimeout(() => {
          if (this.gameManager && this.isActive) {
            this.gameManager.setState({
              lastDrawingSuccess: null,
            });
          }
        }, 100);
      }

      // Trigger success animation or explosion
      if (this.recognitionManager.drawingCanvas) {
        const canvas = this.recognitionManager.drawingCanvas;

        // Check if this is the third success (white stage -> explosion)
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

              // Unequip viewmaster if it's on (including all VFX) and transition to POST_CURSOR
              if (this.gameManager) {
                const currentState = this.gameManager.getState();
                const stateUpdate = {
                  currentState: GAME_STATES.POST_CURSOR,
                };

                if (currentState?.isViewmasterEquipped) {
                  this.logger.log(
                    "Unequipping viewmaster after puzzle completion"
                  );
                  stateUpdate.isViewmasterEquipped = false;
                  stateUpdate.viewmasterManuallyRemoved = false;
                  stateUpdate.viewmasterOverheatDialogIndex = null;
                }

                this.gameManager.setState(stateUpdate);
              }
            } else {
              this.logger.log("Picking new target");
              this.pickNewTarget();
            }
          }, explosionDuration * 1000 + 300);
        } else {
          // First or second success - trigger scaled-down explosion animation
          if (this.successCount === 1) {
            canvas.triggerSuccessAnimation(0.25); // 25% scale for first success
          } else if (this.successCount === 2) {
            canvas.triggerSuccessAnimation(0.5); // 50% scale for second success
          }
          canvas.triggerStrokePulse();

          // Normal color cycle (blue -> orange, orange -> red)
          this.logger.log(
            `Normal color cycle: advancing from stage ${canvas.colorStage}`
          );
          setTimeout(() => {
            this.handleClear(true); // true = success, trigger color change

            // If this was the second success (advancing to white stage), update game state to CURSOR_FINAL
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
    } else {
      // Drawing failed - fade strokes over 2 seconds
      if (this.recognitionManager.drawingCanvas) {
        this.recognitionManager.drawingCanvas.fadeIncorrectGuess();

        // Trigger shake animation
        this.isShaking = true;
        this.shakeProgress = 0;
        if (this.recognitionManager.drawingCanvas.mesh) {
          this.shakeBaseQuaternion.copy(this.currentQuaternion);
        }
      }

      if (this.gameManager) {
        const currentFailureCount =
          this.gameManager.state.drawingFailureCount || 0;
        this.gameManager.setState({
          lastDrawingSuccess: false,
          drawingFailureCount: currentFailureCount + 1,
        });

        // Clear flag after a short delay to allow dialog system to process the state change
        setTimeout(() => {
          if (this.gameManager && this.isActive) {
            this.gameManager.setState({
              lastDrawingSuccess: null,
            });
          }
        }, 100);
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
    // Update mobile button visibility based on captions (even when game not active)
    if (
      this.spaceBarHintElement &&
      this.spaceBarHintElement.classList.contains("mobile-button")
    ) {
      this.updateMobileButtonVisibility();
    }

    if (!this.isActive) return;

    // Billboard canvas components to camera with smooth easing (or shake on incorrect guess)
    const camera = window.camera;
    if (camera && this.recognitionManager.drawingCanvas) {
      const canvas = this.recognitionManager.drawingCanvas;

      if (canvas.mesh) {
        if (this.isShaking) {
          // Update shake progress
          this.shakeProgress += dt / this.shakeDuration;

          if (this.shakeProgress >= 1.0) {
            // Shake complete, return to billboard
            this.isShaking = false;
            this.shakeProgress = 0;
            // Reset to billboard target
            const tempObject = new THREE.Object3D();
            tempObject.position.copy(canvas.mesh.position);
            tempObject.lookAt(camera.position);
            this.currentQuaternion.copy(tempObject.quaternion);
          } else {
            // Update base quaternion to follow camera (but slower during shake)
            const tempObject = new THREE.Object3D();
            tempObject.position.copy(canvas.mesh.position);
            tempObject.lookAt(camera.position);
            this.shakeBaseQuaternion.slerp(
              tempObject.quaternion,
              this.billboardLerpSpeed * dt * 0.5 // Slower follow during shake
            );

            // Calculate shake rotation on world Y axis (decay over time)
            const shakeAngle =
              Math.sin(this.shakeProgress * Math.PI * this.shakeFrequency) *
              this.shakeAmplitude *
              (1.0 - this.shakeProgress);

            // Rotate around world Y axis (vertical) for "shaking head no" effect
            const worldYAxis = new THREE.Vector3(0, 1, 0);
            const shakeQuaternion = new THREE.Quaternion().setFromAxisAngle(
              worldYAxis,
              shakeAngle
            );
            // Apply shake rotation to base quaternion
            this.currentQuaternion.multiplyQuaternions(
              shakeQuaternion,
              this.shakeBaseQuaternion
            );
          }
        } else {
          // Normal billboard behavior
          const tempObject = new THREE.Object3D();
          tempObject.position.copy(canvas.mesh.position);
          tempObject.lookAt(camera.position);
          this.targetQuaternion.copy(tempObject.quaternion);

          // Smoothly interpolate current rotation towards target
          this.currentQuaternion.slerp(
            this.targetQuaternion,
            this.billboardLerpSpeed * dt
          );
        }

        // Apply rotation to all components
        canvas.mesh.quaternion.copy(this.currentQuaternion);

        if (canvas.particleSystem) {
          canvas.particleSystem.quaternion.copy(this.currentQuaternion);
        }

        if (canvas.strokeMesh) {
          canvas.strokeMesh.quaternion.copy(this.currentQuaternion);
        }
      }
    }

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

    // Apply fractal intensity from world disintegration to drawing canvas
    if (this.recognitionManager.drawingCanvas && window.vfxManager) {
      const fractalEffect = window.vfxManager.effects?.splatFractal;
      const canvas = this.recognitionManager.drawingCanvas;

      const currentIntensity = fractalEffect?.currentIntensity || 0;
      const intensityDropped =
        this.previousFractalIntensity > 0 && currentIntensity === 0;

      if (currentIntensity > 0) {
        // Clear any pending reset timeout since intensity is active
        if (this.jitterResetTimeout) {
          clearTimeout(this.jitterResetTimeout);
          this.jitterResetTimeout = null;
        }

        // Apply extra jitter to particles based on fractal intensity
        if (canvas.particleSystem && canvas.particleSystem.material) {
          const baseJitter = canvas.currentJitterIntensity || 0;
          const fractalJitter = currentIntensity * 2.0; // Scale up for dramatic effect
          canvas.particleSystem.material.uniforms.uJitterIntensity.value =
            baseJitter + fractalJitter;
        }

        // Apply warping to stroke mesh
        if (canvas.strokeMesh && canvas.strokeMesh.material) {
          if (canvas.strokeMesh.material.uniforms.uFractalIntensity) {
            canvas.strokeMesh.material.uniforms.uFractalIntensity.value =
              currentIntensity;
          }
        }
      } else if (intensityDropped) {
        // Intensity just dropped from >0 to 0 - delay reset by 0.5s
        if (this.jitterResetTimeout) {
          clearTimeout(this.jitterResetTimeout);
        }
        this.jitterResetTimeout = setTimeout(() => {
          this.jitterResetTimeout = null;
          if (
            canvas.particleSystem &&
            canvas.particleSystem.material &&
            this.isActive
          ) {
            const baseJitter = canvas.currentJitterIntensity || 0;
            canvas.particleSystem.material.uniforms.uJitterIntensity.value =
              baseJitter;
          }
          if (
            canvas.strokeMesh &&
            canvas.strokeMesh.material &&
            this.isActive
          ) {
            if (canvas.strokeMesh.material.uniforms.uFractalIntensity) {
              canvas.strokeMesh.material.uniforms.uFractalIntensity.value = 0;
            }
          }
        }, 500);
      } else if (!this.jitterResetTimeout) {
        // No active intensity and no pending reset - apply reset immediately (e.g., initial state)
        if (canvas.particleSystem && canvas.particleSystem.material) {
          const baseJitter = canvas.currentJitterIntensity || 0;
          canvas.particleSystem.material.uniforms.uJitterIntensity.value =
            baseJitter;
        }
        if (canvas.strokeMesh && canvas.strokeMesh.material) {
          if (canvas.strokeMesh.material.uniforms.uFractalIntensity) {
            canvas.strokeMesh.material.uniforms.uFractalIntensity.value = 0;
          }
        }
      }

      this.previousFractalIntensity = currentIntensity;
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

    if (this.spaceBarHintElement) {
      this.stopSpaceBarHintAnimation();
      this.spaceBarHintElement.remove();
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

  startSpaceBarHintAnimation() {
    if (!this.spaceBarHintElement) return;

    // Only show between CURSOR and POST_CURSOR states
    const gameState = this.gameManager?.getState?.();
    const currentState = gameState?.currentState;
    if (
      currentState === undefined ||
      currentState < GAME_STATES.CURSOR ||
      currentState >= GAME_STATES.POST_CURSOR
    ) {
      return;
    }

    // Don't animate on mobile (it's a button)
    const isMobile = gameState?.isMobile === true;
    const isIOS = gameState?.isIOS === true;
    if (isMobile || isIOS) {
      return;
    }

    // Check if viewmaster is equipped - if so, don't start animation
    if (gameState?.isViewmasterEquipped) {
      return;
    }

    // Remove any existing listener to prevent duplicates
    if (this.spaceBarHintAnimationHandler) {
      this.spaceBarHintElement.removeEventListener(
        "animationend",
        this.spaceBarHintAnimationHandler
      );
    }

    // Create handler for animation end to restart it (loop)
    this.spaceBarHintAnimationHandler = () => {
      if (this.isActive && this.spaceBarHintElement) {
        // Check if viewmaster is equipped - if so, stop animation
        const currentState = this.gameManager?.getState?.();
        if (currentState?.isViewmasterEquipped) {
          this.stopSpaceBarHintAnimation();
          return;
        }

        // Restart animation after a brief pause
        setTimeout(() => {
          if (this.isActive && this.spaceBarHintElement) {
            const state = this.gameManager?.getState?.();
            if (state?.isViewmasterEquipped) {
              this.stopSpaceBarHintAnimation();
              return;
            }
            this.spaceBarHintElement.classList.remove("animating");
            // Force reflow to restart animation
            void this.spaceBarHintElement.offsetWidth;
            this.spaceBarHintElement.classList.add("animating");
          }
        }, 2000);
      }
    };

    // Add the animating class to trigger CSS animation
    this.spaceBarHintElement.classList.add("animating");
    // Ensure pointer events are enabled when visible
    this.spaceBarHintElement.style.pointerEvents = "all";
    this.spaceBarHintElement.addEventListener(
      "animationend",
      this.spaceBarHintAnimationHandler
    );
  }

  stopSpaceBarHintAnimation() {
    if (this.spaceBarHintElement) {
      this.spaceBarHintElement.classList.remove("animating");
      if (this.spaceBarHintAnimationHandler) {
        this.spaceBarHintElement.removeEventListener(
          "animationend",
          this.spaceBarHintAnimationHandler
        );
        this.spaceBarHintAnimationHandler = null;
      }
    }
  }
}
