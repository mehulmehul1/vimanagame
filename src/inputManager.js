import { TouchJoystick } from "./ui/touchJoystick.js";
import { GAME_STATES } from "./gameData.js";
import { Logger } from "./utils/logger.js";

/**
 * InputManager - Unified input handling for keyboard, mouse, gamepad, and touch
 *
 * Features:
 * - Keyboard input (WASD + Shift for sprint)
 * - Mouse input (for camera rotation via pointer lock)
 * - Gamepad API support (left stick = movement, right stick = camera, triggers/buttons = sprint)
 * - Touch joysticks for mobile (left = movement, right = camera)
 * - Unified interface for character controller
 * - Dead zone handling for gamepad sticks
 * - Automatic gamepad detection and connection handling
 */

class InputManager {
  constructor(rendererDomElement, gameManager = null) {
    this.rendererDomElement = rendererDomElement;
    this.gameManager = gameManager;
    this.logger = new Logger("InputManager", false);

    // Input state
    this.keys = { w: false, a: false, s: false, d: false, shift: false };
    this.mouseDelta = { x: 0, y: 0 };
    this.gamepadIndex = null;
    this.enabled = true;
    this.movementEnabled = true; // Separate control for movement
    this.rotationEnabled = true; // Separate control for look rotation
    this.hasSelectiveDisable = false; // Track if we have selective disables active (to prevent enable() from overriding)
    this.pointerLockBlocked = false; // When true, do not request pointer lock
    this.dragToLookEnabled = true; // Allow click-drag to look when pointer lock is blocked
    this.isMouseDown = false;
    this.lastMousePos = { x: 0, y: 0 };
    this.gizmoProbe = null; // Optional function returning boolean: is pointer over/dragging gizmo

    // Touch joysticks
    this.leftJoystick = null;
    this.rightJoystick = null;

    // Gamepad settings
    this.deadzone = 0.15; // Dead zone for analog sticks (0-1)
    this.stickSensitivity = 1.0; // Sensitivity multiplier for gamepad sticks
    this.triggerThreshold = 0.5; // Threshold for trigger buttons (0-1)

    // Mouse settings
    this.mouseSensitivity = 0.0025;

    // Gamepad mappings (standard gamepad layout)
    this.gamepadMapping = {
      // Axes
      AXIS_LEFT_STICK_X: 0,
      AXIS_LEFT_STICK_Y: 1,
      AXIS_RIGHT_STICK_X: 2,
      AXIS_RIGHT_STICK_Y: 3,

      // Buttons
      BUTTON_A: 0, // Cross/A
      BUTTON_B: 1, // Circle/B
      BUTTON_X: 2, // Square/X
      BUTTON_Y: 3, // Triangle/Y
      BUTTON_LB: 4, // L1/LB
      BUTTON_RB: 5, // R1/RB
      BUTTON_LT: 6, // L2/LT (sometimes axis)
      BUTTON_RT: 7, // R2/RT (sometimes axis)
      BUTTON_SELECT: 8, // Select/Back
      BUTTON_START: 9, // Start/Options
      BUTTON_L3: 10, // Left stick press
      BUTTON_R3: 11, // Right stick press
      BUTTON_DPAD_UP: 12,
      BUTTON_DPAD_DOWN: 13,
      BUTTON_DPAD_LEFT: 14,
      BUTTON_DPAD_RIGHT: 15,
    };

    this.setupEventListeners();
    this.setupGamepadListeners();
    this.setupTouchJoysticks();

    this.logger.log(
      "Initialized with keyboard, mouse, gamepad, and touch support"
    );
  }

  /**
   * Set up touch joysticks for mobile devices
   */
  setupTouchJoysticks() {
    // Get touch support from gameManager state (set by platformDetection utility)
    const isTouchDevice = this.gameManager?.getState?.()?.isMobile || false;

    // Left joystick for movement
    this.leftJoystick = new TouchJoystick({
      side: "left",
      size: 120,
      stickSize: 50,
      deadzone: 0.15,
      isTouchDevice: isTouchDevice,
    });

    // Right joystick for camera
    this.rightJoystick = new TouchJoystick({
      side: "right",
      size: 120,
      stickSize: 50,
      deadzone: 0.15,
      isTouchDevice: isTouchDevice,
    });
  }

  /**
   * Set up keyboard event listeners
   */
  setupEventListeners() {
    // Keyboard input
    window.addEventListener("keydown", (event) => {
      if (!this.enabled) return;
      const k = event.key.toLowerCase();
      if (k in this.keys) this.keys[k] = true;
      if (event.key === "Shift") this.keys.shift = true;
    });

    window.addEventListener("keyup", (event) => {
      if (!this.enabled) return;
      const k = event.key.toLowerCase();
      if (k in this.keys) this.keys[k] = false;
      if (event.key === "Shift") this.keys.shift = false;
    });

    // Pointer lock + mouse look
    this.rendererDomElement.addEventListener("click", (event) => {
      // Check if drawing manager is active - if so, don't request pointer lock (needs cursor)
      const drawingActive = window.drawingManager?.isActive || false;
      if (drawingActive) {
        console.log(
          "[InputManager] Drawing active, NOT requesting pointer lock"
        );
        return;
      }

      // Check if gizmo mode is active - if so, don't request pointer lock (gizmos need cursor)
      const gizmoActive =
        window.gizmoManager?.enabled &&
        (window.gizmoManager.objects?.length > 0 ||
          window.gizmoManager.hasGizmoInDefinitions ||
          window.gizmoManager.hasGizmoURLParam);
      if (gizmoActive) {
        console.log(
          "[InputManager] Gizmo mode active, NOT requesting pointer lock"
        );
        return;
      }

      // Only allow pointer lock after game has started (past start screen and title sequence)
      if (this.gameManager && this.gameManager.state) {
        const currentState = this.gameManager.state.currentState;
        if (currentState < GAME_STATES.TITLE_SEQUENCE) {
          return; // Don't request pointer lock during start screen
        }
      }

      // Don't request pointer lock on mobile devices
      const isMobile =
        this.gameManager?.getState?.()?.isMobile || false;
      if (isMobile) {
        console.log("[InputManager] Mobile device detected, NOT requesting pointer lock");
        return;
      }

      // Allow pointer lock even if movement/rotation are disabled
      // (user might want to look around even if they can't move)
      // Only block if pointerLockBlocked is true (set by gizmo mode or drawing mode)
      if (this.pointerLockBlocked) {
        console.log("[InputManager] Pointer lock blocked, NOT requesting");
        return;
      }

      console.log("[InputManager] Requesting pointer lock");
      this.rendererDomElement.requestPointerLock();
    });

    document.addEventListener("mousemove", (event) => {
      if (!this.enabled) return;
      const isLocked = document.pointerLockElement === this.rendererDomElement;
      const overGizmo = this.gizmoProbe ? this.gizmoProbe() : false;

      if (isLocked) {
        // Pointer lock mode: use movement deltas
        // But only if rotation is actually enabled (prevents accumulation during title sequence)
        if (this.rotationEnabled) {
          // Validate movementX/Y before accumulating
          const moveX = isFinite(event.movementX) ? event.movementX : 0;
          const moveY = isFinite(event.movementY) ? event.movementY : 0;
          this.mouseDelta.x += moveX;
          this.mouseDelta.y += moveY;
        }
        return;
      }

      // Drag-to-look when pointer lock is blocked, only if mouse is down and not over gizmo
      if (
        this.pointerLockBlocked &&
        this.dragToLookEnabled &&
        this.isMouseDown &&
        !overGizmo
      ) {
        // Validate clientX/Y and lastMousePos before calculating delta
        const clientX = isFinite(event.clientX)
          ? event.clientX
          : this.lastMousePos.x;
        const clientY = isFinite(event.clientY)
          ? event.clientY
          : this.lastMousePos.y;
        const lastX = isFinite(this.lastMousePos.x)
          ? this.lastMousePos.x
          : clientX;
        const lastY = isFinite(this.lastMousePos.y)
          ? this.lastMousePos.y
          : clientY;

        const dx = clientX - lastX;
        const dy = clientY - lastY;

        // Only accumulate if delta is finite
        if (isFinite(dx)) this.mouseDelta.x += dx;
        if (isFinite(dy)) this.mouseDelta.y += dy;

        this.lastMousePos.x = clientX;
        this.lastMousePos.y = clientY;
      }
    });

    // Track mouse down/up for drag-to-look
    document.addEventListener("mousedown", (event) => {
      if (!this.enabled) return;
      this.isMouseDown = true;
      this.lastMousePos.x = event.clientX;
      this.lastMousePos.y = event.clientY;
    });
    document.addEventListener("mouseup", () => {
      if (!this.enabled) return;
      this.isMouseDown = false;
    });
  }

  /**
   * Block or allow pointer lock acquisition (e.g., when editing with gizmo)
   * @param {boolean} blocked
   */
  setPointerLockBlocked(blocked) {
    this.pointerLockBlocked = !!blocked;
    console.log(
      "[InputManager] setPointerLockBlocked:",
      this.pointerLockBlocked
    );
    if (this.pointerLockBlocked && document.pointerLockElement) {
      console.log("[InputManager] Exiting existing pointer lock");
      document.exitPointerLock();
    }
  }

  /**
   * Provide a function so InputManager can know if pointer is on gizmo
   * @param {Function} probeFn - returns boolean
   */
  setGizmoProbe(probeFn) {
    this.gizmoProbe = typeof probeFn === "function" ? probeFn : null;
  }

  /**
   * Set up gamepad event listeners
   */
  setupGamepadListeners() {
    window.addEventListener("gamepadconnected", (e) => {
      this.logger.log(
        `Gamepad connected - ${e.gamepad.id} (index: ${e.gamepad.index})`
      );

      // Use the first connected gamepad
      if (this.gamepadIndex === null) {
        this.gamepadIndex = e.gamepad.index;
        this.logger.log(`Using gamepad index ${this.gamepadIndex}`);
      }
    });

    window.addEventListener("gamepaddisconnected", (e) => {
      this.logger.log(
        `Gamepad disconnected - ${e.gamepad.id} (index: ${e.gamepad.index})`
      );

      // Clear gamepad index if this was our active gamepad
      if (this.gamepadIndex === e.gamepad.index) {
        this.gamepadIndex = null;
        this.logger.log("Active gamepad disconnected");
      }
    });
  }

  /**
   * Get the active gamepad if connected
   * @returns {Gamepad|null}
   */
  getGamepad() {
    if (this.gamepadIndex === null) return null;

    const gamepads = navigator.getGamepads();
    return gamepads[this.gamepadIndex] || null;
  }

  /**
   * Apply dead zone to analog stick value
   * @param {number} value - Raw axis value (-1 to 1)
   * @returns {number} Processed value with dead zone applied
   */
  applyDeadzone(value) {
    if (Math.abs(value) < this.deadzone) return 0;

    // Remap value from [deadzone, 1] to [0, 1] for smooth transition
    const sign = Math.sign(value);
    const magnitude = Math.abs(value);
    const normalized = (magnitude - this.deadzone) / (1 - this.deadzone);

    return sign * normalized;
  }

  /**
   * Get movement input from keyboard and gamepad
   * Returns normalized movement vector
   * @returns {{x: number, y: number}} Movement input (-1 to 1 for each axis)
   */
  getMovementInput() {
    // Return zero if movement is disabled
    if (!this.movementEnabled) {
      return { x: 0, y: 0 };
    }

    let x = 0;
    let y = 0;

    // Keyboard input (WASD)
    if (this.keys.w) y += 1;
    if (this.keys.s) y -= 1;
    if (this.keys.a) x -= 1;
    if (this.keys.d) x += 1;

    // Gamepad input (left stick)
    const gamepad = this.getGamepad();
    if (gamepad) {
      const leftX = this.applyDeadzone(
        gamepad.axes[this.gamepadMapping.AXIS_LEFT_STICK_X]
      );
      const leftY = this.applyDeadzone(
        gamepad.axes[this.gamepadMapping.AXIS_LEFT_STICK_Y]
      );

      // Add gamepad input (Y is inverted on gamepad)
      x += leftX;
      y -= leftY; // Invert Y axis (forward is negative on gamepad)
    }

    // Touch joystick input (left joystick for movement)
    if (this.leftJoystick && this.leftJoystick.isActive()) {
      const touchInput = this.leftJoystick.getValue();
      x += touchInput.x;
      y += touchInput.y;
    }

    // Clamp to [-1, 1] range (in case multiple inputs combine)
    x = Math.max(-1, Math.min(1, x));
    y = Math.max(-1, Math.min(1, y));

    return { x, y };
  }

  /**
   * Get camera rotation input from mouse, gamepad, and touch
   * Returns camera delta for this frame
   * @param {number} dt - Delta time in seconds (for gamepad/touch scaling)
   * @returns {{x: number, y: number, hasGamepad: boolean}} Camera rotation delta and source info
   */
  getCameraInput(dt = 0.016) {
    // Return zero if rotation is disabled
    if (!this.rotationEnabled) {
      return { x: 0, y: 0, hasGamepad: false };
    }

    let deltaX = 0;
    let deltaY = 0;
    let hasGamepadInput = false;

    // Validate and sanitize mouseDelta before use
    const mouseX = isFinite(this.mouseDelta.x) ? this.mouseDelta.x : 0;
    const mouseY = isFinite(this.mouseDelta.y) ? this.mouseDelta.y : 0;

    // Mouse input (accumulated during frame)
    // Scale by dt for frame-rate independence (normalize to 60fps reference)
    // At 60fps, dt = 0.0167, so we scale by dt/0.0167 to maintain same feel
    const dtScale = dt / 0.0167; // Normalize to 60fps
    deltaX += mouseX * this.mouseSensitivity * dtScale;
    deltaY += mouseY * this.mouseSensitivity * dtScale;

    // Gamepad input (right stick)
    const gamepad = this.getGamepad();
    if (gamepad) {
      const rawRightX = gamepad.axes[this.gamepadMapping.AXIS_RIGHT_STICK_X];
      const rawRightY = gamepad.axes[this.gamepadMapping.AXIS_RIGHT_STICK_Y];

      // Validate gamepad axes before use
      const rightX = isFinite(rawRightX) ? this.applyDeadzone(rawRightX) : 0;
      const rightY = isFinite(rawRightY) ? this.applyDeadzone(rawRightY) : 0;

      if (rightX !== 0 || rightY !== 0) {
        hasGamepadInput = true;

        // Convert stick input to camera delta (scale by dt for frame-rate independence)
        const stickScale = 2.5; // Radians per second at full deflection
        const scaledX = rightX * stickScale * this.stickSensitivity * dt;
        const scaledY = rightY * stickScale * this.stickSensitivity * dt;

        // Only add if finite
        if (isFinite(scaledX)) deltaX += scaledX;
        if (isFinite(scaledY)) deltaY += scaledY;
      }
    }

    // Touch joystick input (right joystick for camera)
    if (this.rightJoystick && this.rightJoystick.isActive()) {
      hasGamepadInput = true; // Treat touch like gamepad (direct control, no smoothing)

      const touchInput = this.rightJoystick.getValue();
      // Validate touch input
      const touchX = isFinite(touchInput?.x) ? touchInput.x : 0;
      const touchY = isFinite(touchInput?.y) ? touchInput.y : 0;

      // Convert touch input to camera delta (scale by dt for frame-rate independence)
      const touchScale = 2.5; // Radians per second at full deflection
      const scaledX = touchX * touchScale * dt;
      const scaledY = touchY * touchScale * dt;

      // Only add if finite
      if (isFinite(scaledX)) deltaX += scaledX;
      if (isFinite(scaledY)) deltaY -= scaledY; // Invert Y for natural camera movement
    }

    // Ensure final values are finite
    deltaX = isFinite(deltaX) ? deltaX : 0;
    deltaY = isFinite(deltaY) ? deltaY : 0;

    return { x: deltaX, y: deltaY, hasGamepad: hasGamepadInput };
  }

  /**
   * Get touch joystick speed multiplier (0 to 1)
   * Returns 0 if touch joystick is not active
   * @returns {number} Speed multiplier (0 to 1)
   */
  getTouchSpeedMultiplier() {
    if (this.leftJoystick && this.leftJoystick.isActive()) {
      return this.leftJoystick.getSpeedMultiplier();
    }
    return 0;
  }

  /**
   * Check if sprint is active (Shift key or gamepad trigger/button)
   * @returns {boolean}
   */

  isSprinting() {
    // Keyboard input
    if (this.keys.shift) return true;

    // Gamepad input (L2/LT trigger, R2/RT trigger, L3 button, or R3 button)
    const gamepad = this.getGamepad();
    if (gamepad) {
      // Check L3 (left stick press)
      if (gamepad.buttons[this.gamepadMapping.BUTTON_L3]?.pressed) return true;

      // Check R3 (right stick press)
      if (gamepad.buttons[this.gamepadMapping.BUTTON_R3]?.pressed) return true;

      // Check LT trigger (button 6)
      const ltButton = gamepad.buttons[this.gamepadMapping.BUTTON_LT];
      if (ltButton && ltButton.value > this.triggerThreshold) return true;

      // Check RT trigger (button 7)
      const rtButton = gamepad.buttons[this.gamepadMapping.BUTTON_RT];
      if (rtButton && rtButton.value > this.triggerThreshold) return true;
    }

    return false;
  }

  /**
   * Reset accumulated input (call after processing each frame)
   */
  resetFrameInput() {
    // Reset mouseDelta, but also ensure values are finite
    this.mouseDelta.x = 0;
    this.mouseDelta.y = 0;

    // If mouseDelta somehow got corrupted, reset it
    if (!isFinite(this.mouseDelta.x)) this.mouseDelta.x = 0;
    if (!isFinite(this.mouseDelta.y)) this.mouseDelta.y = 0;
  }

  /**
   * Enable input processing (both movement and rotation)
   * Only enables if it was disabled by disable() - doesn't override selective disables
   */
  enable() {
    // Don't override selective disables (e.g., from moveTo with selective inputControl)
    if (this.hasSelectiveDisable) {
      this.logger.log(
        "enable() called but selective disable is active (ignoring to preserve selective disables)"
      );
      return;
    }

    this.enabled = true;
    this.movementEnabled = true;
    this.rotationEnabled = true;
    // Clear any accumulated input when enabling
    this.mouseDelta = { x: 0, y: 0 };
    this.showTouchControls();
    this.logger.log("Input enabled");
  }

  /**
   * Disable input processing and clear all input states (both movement and rotation)
   */
  disable() {
    this.enabled = false;
    this.movementEnabled = false;
    this.rotationEnabled = false;
    this.hasSelectiveDisable = false; // Full disable clears selective flag
    this.keys = { w: false, a: false, s: false, d: false, shift: false };
    this.mouseDelta = { x: 0, y: 0 };
    this.hideTouchControls();
    this.logger.log("Input disabled");
  }

  /**
   * Enable movement input only
   */
  enableMovement() {
    this.movementEnabled = true;
    this.enabled = true; // Enable input system so pointer lock can work
    // Clear selective disable flag if both are now enabled
    if (this.rotationEnabled) {
      this.hasSelectiveDisable = false;
    }
    // Show left joystick only if on touch device
    if (this.leftJoystick) {
      this.leftJoystick.show();
    }
    this.logger.log("Movement enabled");
  }

  /**
   * Disable movement input only
   */
  disableMovement() {
    this.movementEnabled = false;
    this.hasSelectiveDisable = true; // Mark that we have a selective disable
    this.keys.w = false;
    this.keys.a = false;
    this.keys.s = false;
    this.keys.d = false;
    this.keys.shift = false;
    // Hide left joystick
    if (this.leftJoystick) {
      this.leftJoystick.hide();
    }
    this.logger.log("Movement disabled");
  }

  /**
   * Enable rotation input only
   */
  enableRotation() {
    this.rotationEnabled = true;
    this.enabled = true; // Enable input system so pointer lock can work
    // Clear selective disable flag if both are now enabled
    if (this.movementEnabled) {
      this.hasSelectiveDisable = false;
    }
    // Clear any accumulated mouse input when enabling rotation
    this.mouseDelta = { x: 0, y: 0 };
    // Show right joystick only if on touch device
    if (this.rightJoystick) {
      this.rightJoystick.show();
    }
    this.logger.log("Rotation enabled");
  }

  /**
   * Disable rotation input only
   */
  disableRotation() {
    this.rotationEnabled = false;
    this.hasSelectiveDisable = true; // Mark that we have a selective disable
    this.mouseDelta = { x: 0, y: 0 };
    // Hide right joystick
    if (this.rightJoystick) {
      this.rightJoystick.hide();
    }
    this.logger.log("Rotation disabled");
  }

  /**
   * Show touch controls (for mobile)
   */
  showTouchControls() {
    // Don't show if dialog choice UI is visible
    if (this.isDialogChoiceVisible()) {
      return;
    }

    if (this.leftJoystick) {
      this.leftJoystick.show();
    }
    if (this.rightJoystick) {
      this.rightJoystick.show();
    }
  }

  /**
   * Hide touch controls
   */
  hideTouchControls() {
    if (this.leftJoystick) {
      this.leftJoystick.hide();
    }
    if (this.rightJoystick) {
      this.rightJoystick.hide();
    }
  }

  /**
   * Fade out touch controls (for UI overlays)
   */
  fadeOutTouchControls() {
    if (this.leftJoystick) {
      this.leftJoystick.fadeOut();
    }
    if (this.rightJoystick) {
      this.rightJoystick.fadeOut();
    }
  }

  /**
   * Fade in touch controls (for UI overlays)
   */
  fadeInTouchControls() {
    // Don't fade in if dialog choice UI is visible
    if (this.isDialogChoiceVisible()) {
      return;
    }

    if (this.leftJoystick) {
      this.leftJoystick.fadeIn();
    }
    if (this.rightJoystick) {
      this.rightJoystick.fadeIn();
    }
  }

  /**
   * Check if dialog choice UI is currently visible
   * Uses multiple methods for reliability across different browsers/devices
   * @returns {boolean}
   */
  isDialogChoiceVisible() {
    const dialogChoiceContainer = document.getElementById("dialog-choices");
    if (!dialogChoiceContainer) {
      return false;
    }

    // Check computed style first (most reliable, catches all hiding methods)
    const computedStyle = window.getComputedStyle(dialogChoiceContainer);
    if (computedStyle.display === "none" || computedStyle.visibility === "hidden" || computedStyle.opacity === "0") {
      return false;
    }

    // Check inline style as secondary check
    if (dialogChoiceContainer.style.display === "none") {
      return false;
    }

    // For fixed position elements, check if they're actually visible
    // offsetParent is null for fixed elements that are visible, but also for hidden elements
    // So we check the bounding rect instead
    const rect = dialogChoiceContainer.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) {
      return false;
    }

    return true;
  }

  /**
   * Check if input is currently enabled
   * @returns {boolean}
   */
  isEnabled() {
    return this.enabled;
  }

  /**
   * Check if movement input is currently enabled
   * @returns {boolean}
   */
  isMovementEnabled() {
    return this.movementEnabled;
  }

  /**
   * Check if rotation input is currently enabled
   * @returns {boolean}
   */
  isRotationEnabled() {
    return this.rotationEnabled;
  }

  /**
   * Set mouse sensitivity
   * @param {number} sensitivity - Mouse sensitivity multiplier
   */
  setMouseSensitivity(sensitivity) {
    this.mouseSensitivity = sensitivity;
  }

  /**
   * Set gamepad stick sensitivity
   * @param {number} sensitivity - Stick sensitivity multiplier
   */
  setStickSensitivity(sensitivity) {
    this.stickSensitivity = sensitivity;
  }

  /**
   * Set gamepad dead zone
   * @param {number} deadzone - Dead zone value (0-1)
   */
  setDeadzone(deadzone) {
    this.deadzone = Math.max(0, Math.min(1, deadzone));
  }

  /**
   * Get gamepad connection status
   * @returns {boolean}
   */
  isGamepadConnected() {
    return this.getGamepad() !== null;
  }

  /**
   * Get gamepad info for display/debugging
   * @returns {Object|null}
   */
  getGamepadInfo() {
    const gamepad = this.getGamepad();
    if (!gamepad) return null;

    return {
      id: gamepad.id,
      index: gamepad.index,
      connected: gamepad.connected,
      buttons: gamepad.buttons.length,
      axes: gamepad.axes.length,
    };
  }

  /**
   * Update method - call in animation loop to update gamepad state
   * @param {number} dt - Delta time (for future use with gamepad input scaling)
   */
  update(dt) {
    // Gamepad state is automatically updated by the browser
    // This method is here for future extensions and consistency
    // Could add gamepad rumble/haptics here in the future
  }
}

export default InputManager;
