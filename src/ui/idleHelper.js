import { Logger } from "../utils/logger.js";
import "../styles/idleHelper.css";

export class IdleHelper {
  constructor(
    dialogManager = null,
    cameraAnimationSystem = null,
    dialogChoiceUI = null,
    gameManager = null,
    inputManager = null,
    characterController = null
  ) {
    this.helperElement = null;
    this.lastMovementTime = null; // Don't start tracking until controls are enabled
    this.idleThreshold = 10000; // 10 seconds
    this.isAnimating = false;
    this.dialogManager = dialogManager;
    this.cameraAnimationSystem = cameraAnimationSystem;
    this.dialogChoiceUI = dialogChoiceUI;
    this.gameManager = gameManager;
    this.inputManager = inputManager;
    this.characterController = characterController;
    this.wasControlEnabled = false; // Track previous control state
    this.wasCameraAnimating = false; // Track previous camera animation state
    this.wasBlocked = false; // Track previous overall blocked state
    this.globalDisable = false; // If true, idle behaviors are fully disabled
    this.logger = new Logger("IdleHelper", false);

    this.init();
    this.setupMovementListeners();
    this.startIdleCheck();
  }

  /**
   * Globally enable/disable all idle behaviors (e.g., when gizmo editing is active)
   * @param {boolean} disabled
   */
  setGlobalDisable(disabled) {
    this.globalDisable = !!disabled;
    if (this.globalDisable) {
      // Hide any active animation immediately
      if (this.isAnimating) {
        this.stopAnimation();
      }
    } else {
      // Reset timer on re-enable to avoid immediate popup
      this.lastMovementTime = Date.now();
    }
  }

  init() {
    // Don't create helper on iOS (keyboard/mouse controls don't apply)
    const currentState = this.gameManager?.getState?.();
    const isIOS = currentState?.isIOS === true;
    if (isIOS) {
      this.logger.log("Skipping initialization on iOS device");
      return;
    }

    // Create the helper image element
    this.helperElement = document.createElement("div");
    this.helperElement.id = "idle-helper";

    // Create the image
    const img = document.createElement("img");
    img.src = "/images/WASD.svg";
    img.alt = "WASD controls";

    this.helperElement.appendChild(img);
    document.body.appendChild(this.helperElement);
  }

  setupMovementListeners() {
    // Listen for WASD and arrow keys
    const movementKeys = [
      "w",
      "a",
      "s",
      "d",
      "W",
      "A",
      "S",
      "D",
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
    ];

    window.addEventListener("keydown", (e) => {
      if (movementKeys.includes(e.key)) {
        this.onMovement();
      }
    });

    // Count mouse movement as activity when pointer is locked (in gameplay)
    let lastMouseMoveTime = 0;
    const mouseMoveThrottle = 100; // Throttle to once per 100ms

    window.addEventListener("mousemove", () => {
      // Only reset idle timer if pointer is locked (actively playing)
      if (document.pointerLockElement) {
        const now = Date.now();
        if (now - lastMouseMoveTime >= mouseMoveThrottle) {
          lastMouseMoveTime = now;
          this.onMovement();
        }
      }
    });
  }

  onMovement() {
    // Only track movement if controls are enabled
    const isControlEnabled =
      this.gameManager && this.gameManager.isControlEnabled();
    const isInputEnabled = this.inputManager && this.inputManager.isEnabled();
    const isCharacterInputEnabled =
      !this.characterController || !this.characterController.inputDisabled;

    if (!isControlEnabled || !isInputEnabled || !isCharacterInputEnabled) {
      return;
    }

    this.lastMovementTime = Date.now();

    // If currently animating, smoothly fade out from current position
    if (this.isAnimating) {
      this.interruptWithFadeOut();
    }
  }

  /**
   * Check if idle behaviors should be allowed (glances, etc.)
   * @returns {boolean} True if player is idle and no blocking conditions are active
   */
  shouldAllowIdleBehavior() {
    // Block all idle behaviors if globally disabled (e.g., gizmo mode)
    if (this.globalDisable) {
      return false;
    }
    // Also block if game state declares gizmo presence in data (pre-instantiation)
    const currentState = this.gameManager?.getState?.();
    const hasGizmoInData = currentState?.hasGizmoInData === true;
    if (hasGizmoInData) {
      return false;
    }
    // Block on iOS devices (WASD/mouse controls don't apply)
    const isIOS = currentState?.isIOS === true;
    if (isIOS) {
      return false;
    }
    // Check if controls are enabled at all levels
    const isControlEnabled =
      this.gameManager && this.gameManager.isControlEnabled();
    const isInputEnabled = this.inputManager && this.inputManager.isEnabled();
    const isCharacterInputEnabled =
      !this.characterController || !this.characterController.inputDisabled;

    if (
      !isControlEnabled ||
      !isInputEnabled ||
      !isCharacterInputEnabled ||
      this.lastMovementTime === null
    ) {
      return false;
    }

    // Check blocking conditions first (including camera animations)
    const isDialogPlaying = this.dialogManager && this.dialogManager.isPlaying;
    const hasPendingDialog =
      this.dialogManager &&
      this.dialogManager.pendingDialogs &&
      this.dialogManager.pendingDialogs.size > 0;
    const isCameraAnimating =
      this.cameraAnimationSystem && this.cameraAnimationSystem.isPlaying;
    const isCharacterLookingAt =
      this.characterController && this.characterController.isLookingAt;
    const isCharacterMovingTo =
      this.characterController && this.characterController.isMovingTo;
    const isChoiceUIOpen = this.dialogChoiceUI && this.dialogChoiceUI.isVisible;

    // Don't allow idle behaviors during any blocking condition
    if (
      isDialogPlaying ||
      hasPendingDialog ||
      isCameraAnimating ||
      isCharacterLookingAt ||
      isCharacterMovingTo ||
      isChoiceUIOpen
    ) {
      return false;
    }

    // Check time since last movement
    const timeSinceLastMovement = Date.now() - this.lastMovementTime;
    if (timeSinceLastMovement < this.idleThreshold) {
      return false;
    }

    return true;
  }

  interruptWithFadeOut() {
    // Skip if helper element doesn't exist (e.g., on iOS)
    if (!this.helperElement) {
      this.isAnimating = false;
      return;
    }

    // Remove animation class and let CSS transition handle the fade out
    this.helperElement.classList.remove("animating");
    this.isAnimating = false;
  }

  startIdleCheck() {
    // Check frequently if user is idle (100ms for responsive gamepad detection)
    setInterval(() => {
      // Check if controls are enabled at all levels
      const isControlEnabled =
        this.gameManager && this.gameManager.isControlEnabled();
      const isInputEnabled = this.inputManager && this.inputManager.isEnabled();
      const isCharacterInputEnabled =
        !this.characterController || !this.characterController.inputDisabled;

      const isFullyEnabled =
        isControlEnabled && isInputEnabled && isCharacterInputEnabled;

      // If controls just got enabled, reset the idle timer
      if (isFullyEnabled && !this.wasControlEnabled) {
        this.lastMovementTime = Date.now();
      }

      // If controls got disabled, hide any active animation and stop tracking
      if (!isFullyEnabled && this.wasControlEnabled) {
        if (this.isAnimating) {
          this.stopAnimation();
        }
        // Don't reset lastMovementTime here - it will be reset when controls are re-enabled
      }

      // Update the previous control state
      this.wasControlEnabled = isFullyEnabled;

      // Check if camera animation just ended and reset idle timer
      const isCameraAnimating =
        this.cameraAnimationSystem && this.cameraAnimationSystem.isPlaying;

      if (this.wasCameraAnimating && !isCameraAnimating) {
        // Camera animation just ended - reset idle timer
        this.lastMovementTime = Date.now();
        this.logger.log("Camera animation ended, resetting idle timer");
      }

      // Update the previous camera animation state
      this.wasCameraAnimating = isCameraAnimating;

      // If globally disabled, gizmo flag is set, or iOS device, force hide and skip
      const currentState = this.gameManager?.getState?.();
      const hasGizmoInData = currentState?.hasGizmoInData === true;
      const isIOS = currentState?.isIOS === true;
      if (this.globalDisable || hasGizmoInData || isIOS) {
        if (this.isAnimating) {
          this.stopAnimation();
        }
        return;
      }

      // Calculate current blocked state (all blocking conditions)
      const isDialogPlaying =
        this.dialogManager && this.dialogManager.isPlaying;
      const hasPendingDialog =
        this.dialogManager &&
        this.dialogManager.pendingDialogs &&
        this.dialogManager.pendingDialogs.size > 0;
      const isCharacterLookingAt =
        this.characterController && this.characterController.isLookingAt;
      const isCharacterMovingTo =
        this.characterController && this.characterController.isMovingTo;
      const isChoiceUIOpen =
        this.dialogChoiceUI && this.dialogChoiceUI.isVisible;

      const isCurrentlyBlocked =
        !isFullyEnabled ||
        isDialogPlaying ||
        hasPendingDialog ||
        isCameraAnimating ||
        isCharacterLookingAt ||
        isCharacterMovingTo ||
        isChoiceUIOpen;

      // If transitioning from blocked to unblocked, reset idle timer
      if (this.wasBlocked && !isCurrentlyBlocked) {
        this.lastMovementTime = Date.now();
        this.logger.log(
          "Transitioned from blocked to unblocked, resetting idle timer"
        );
      }

      // Update the previous blocked state
      this.wasBlocked = isCurrentlyBlocked;

      // Don't check idle state if controls haven't been enabled yet at all levels
      if (!isFullyEnabled || this.lastMovementTime === null) {
        return;
      }

      // Don't track idle time while camera animation is playing
      if (isCameraAnimating) {
        return;
      }

      // Check for gamepad and touch input and reset idle timer if active
      if (this.inputManager) {
        const movementInput = this.inputManager.getMovementInput();
        const hasMovementInput =
          Math.abs(movementInput.x) > 0.01 || Math.abs(movementInput.y) > 0.01;

        // Check for gamepad camera input (right stick)
        const gamepad = this.inputManager.getGamepad();
        let hasCameraInput = false;
        if (gamepad) {
          const rightX = this.inputManager.applyDeadzone(
            gamepad.axes[this.inputManager.gamepadMapping.AXIS_RIGHT_STICK_X]
          );
          const rightY = this.inputManager.applyDeadzone(
            gamepad.axes[this.inputManager.gamepadMapping.AXIS_RIGHT_STICK_Y]
          );
          hasCameraInput = Math.abs(rightX) > 0.01 || Math.abs(rightY) > 0.01;
        }

        // Check for touch joystick input
        let hasTouchInput = false;
        if (
          this.inputManager.leftJoystick &&
          this.inputManager.leftJoystick.isActive()
        ) {
          hasTouchInput = true;
        }
        if (
          this.inputManager.rightJoystick &&
          this.inputManager.rightJoystick.isActive()
        ) {
          hasTouchInput = true;
        }

        // If there's any gamepad or touch input, reset idle timer
        if (hasMovementInput || hasCameraInput || hasTouchInput) {
          this.onMovement();
        }
      }

      const timeSinceLastMovement = Date.now() - this.lastMovementTime;

      // Note: All blocking condition variables (isDialogPlaying, hasPendingDialog,
      // isCameraAnimating, isCharacterLookingAt, isCharacterMovingTo, isChoiceUIOpen)
      // are already defined above when calculating the blocked state

      if (
        timeSinceLastMovement >= this.idleThreshold &&
        !this.isAnimating &&
        !isDialogPlaying &&
        !hasPendingDialog &&
        !isCameraAnimating &&
        !isCharacterLookingAt &&
        !isCharacterMovingTo &&
        !isChoiceUIOpen
      ) {
        this.startAnimation();
      }

      // Hide helper if any blocking condition becomes active while it's showing
      if (
        (isDialogPlaying ||
          hasPendingDialog ||
          isCameraAnimating ||
          isCharacterLookingAt ||
          isCharacterMovingTo ||
          isChoiceUIOpen) &&
        this.isAnimating
      ) {
        this.stopAnimation();
      }
    }, 100); // Check every 100ms for responsive gamepad detection
  }

  startAnimation() {
    // Skip if helper element doesn't exist (e.g., on iOS)
    if (!this.helperElement) {
      this.isAnimating = false;
      return;
    }

    this.isAnimating = true;

    // Add the animating class to trigger CSS animation
    this.helperElement.classList.add("animating");

    // Listen for animation end to reset state
    const handleAnimationEnd = () => {
      this.helperElement.classList.remove("animating");
      this.isAnimating = false;
      this.helperElement.removeEventListener(
        "animationend",
        handleAnimationEnd
      );
    };

    this.helperElement.addEventListener("animationend", handleAnimationEnd);
  }

  stopAnimation() {
    if (this.helperElement) {
      this.helperElement.classList.remove("animating");
    }
    this.isAnimating = false;
  }

  destroy() {
    this.stopAnimation();
    if (this.helperElement && this.helperElement.parentNode) {
      this.helperElement.parentNode.removeChild(this.helperElement);
    }
  }
}
