/**
 * touchJoystick.js - VIRTUAL JOYSTICK FOR MOBILE TOUCH CONTROLS
 * =============================================================================
 *
 * ROLE: Provides a visual virtual joystick for mobile devices. Returns
 * normalized X/Y values (-1 to 1) for use by InputManager.
 *
 * KEY RESPONSIBILITIES:
 * - Create joystick DOM elements
 * - Handle touch events with multi-touch support
 * - Calculate normalized stick position
 * - Apply dead zone and speed multiplier
 * - RAF-based visual updates for performance
 *
 * =============================================================================
 */

export class TouchJoystick {
  constructor(options = {}) {
    this.side = options.side || "left"; // 'left' or 'right'
    this.size = options.size || 120; // Outer circle diameter in pixels
    this.stickSize = options.stickSize || 50; // Inner stick diameter in pixels
    this.maxDistance = (this.size - this.stickSize) / 2; // Max stick travel distance
    this.deadzone = options.deadzone || 0.15; // Dead zone (0-1)
    this.isTouchDevice = options.isTouchDevice || false; // Touch support flag (set by UIManager)

    // State
    this.active = false;
    this.touchId = null;
    this.centerX = 0;
    this.centerY = 0;
    this.currentX = 0;
    this.currentY = 0;
    this.deltaX = 0;
    this.deltaY = 0;
    this.speedMultiplier = 0; // Speed multiplier based on stick distance (0 to 1)

    // Performance optimization state
    this.unlockedThisSession = false; // Track if we've unlocked during this touch session
    this.pendingVisualUpdate = false; // Track if RAF is scheduled
    this.pendingDx = 0; // Pending visual position X
    this.pendingDy = 0; // Pending visual position Y

    // Create DOM elements
    this.createElements();
    this.setupEventListeners();
  }

  /**
   * Create the joystick DOM elements
   */
  createElements() {
    // Container
    this.container = document.createElement("div");
    this.container.className = `touch-joystick touch-joystick-${this.side}`;
    this.container.style.cssText = `
      position: fixed;
      ${this.side}: 30px;
      bottom: 30px;
      width: ${this.size}px;
      height: ${this.size}px;
      background: rgba(255, 255, 255, 0.1);
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      touch-action: none;
      user-select: none;
      z-index: 1000;
      display: none;
      opacity: 0.6;
      transition: opacity 0.2s;
    `;

    // Stick
    this.stick = document.createElement("div");
    this.stick.className = "touch-joystick-stick";
    this.stick.style.cssText = `
      position: absolute;
      width: ${this.stickSize}px;
      height: ${this.stickSize}px;
      background: rgba(255, 255, 255, 0.5);
      border: 2px solid rgba(255, 255, 255, 0.8);
      border-radius: 50%;
      left: 50%;
      top: 50%;
      transform: translate3d(-50%, -50%, 0);
      pointer-events: none;
      transition: background 0.2s;
    `;

    this.container.appendChild(this.stick);
    document.body.appendChild(this.container);
  }

  /**
   * Set up touch event listeners
   */
  setupEventListeners() {
    // Touch start
    this.container.addEventListener(
      "touchstart",
      (e) => {
        // Unlock audio/video once per touch session
        // The global unlockAudioOnInteraction handler also fires in capture phase,
        // but we call it here explicitly to ensure gesture context is established
        if (window.unlockAudioOnInteraction && !this.unlockedThisSession) {
          window.unlockAudioOnInteraction();
          this.unlockedThisSession = true;
        }

        e.preventDefault();

        if (this.active) return; // Already handling a touch

        const touch = e.changedTouches[0];
        this.touchId = touch.identifier;

        const rect = this.container.getBoundingClientRect();
        this.centerX = rect.left + rect.width / 2;
        this.centerY = rect.top + rect.height / 2;

        this.active = true;
        this.unlockedThisSession = false; // Reset for new touch session
        this.container.style.opacity = "1";
        this.stick.style.background = "rgba(255, 255, 255, 0.8)";

        this.updatePosition(touch.clientX, touch.clientY);
      },
      { passive: false }
    );

    // Touch move
    window.addEventListener(
      "touchmove",
      (e) => {
        if (!this.active) return;

        // Find our touch
        for (let i = 0; i < e.changedTouches.length; i++) {
          const touch = e.changedTouches[i];
          if (touch.identifier === this.touchId) {
            // Unlock only once per touch session (already done on touchstart)
            // No need to unlock on touchmove - gesture context is established on touchstart

            e.preventDefault();
            this.updatePosition(touch.clientX, touch.clientY);
            break;
          }
        }
      },
      { passive: false }
    );

    // Touch end
    const handleTouchEnd = (e) => {
      if (!this.active) return;

      // Check if our touch ended
      for (let i = 0; i < e.changedTouches.length; i++) {
        const touch = e.changedTouches[i];
        if (touch.identifier === this.touchId) {
          // CRITICAL: Unlock on touchend for long holds (only if not already unlocked)
          // iOS Safari may not establish gesture context on long holds (touchstart),
          // but it WILL establish it on touchend after a hold
          // This ensures audio/video can play after releasing a long hold
          if (
            window.unlockAudioOnInteraction &&
            !this.unlockedThisSession
          ) {
            window.unlockAudioOnInteraction();
            this.unlockedThisSession = true;
          }

          this.reset();
          break;
        }
      }
    };

    window.addEventListener("touchend", handleTouchEnd, { passive: false });
    window.addEventListener("touchcancel", handleTouchEnd, { passive: false });
  }

  /**
   * Update stick position based on touch coordinates
   */
  updatePosition(touchX, touchY) {
    // Calculate delta from center
    let dx = touchX - this.centerX;
    let dy = touchY - this.centerY;

    // Calculate squared distance first (avoid sqrt if possible)
    const distanceSq = dx * dx + dy * dy;
    const maxDistanceSq = this.maxDistance * this.maxDistance;

    // Constrain to max distance if needed
    if (distanceSq > maxDistanceSq) {
      const distance = Math.sqrt(distanceSq);
      const angle = Math.atan2(dy, dx);
      dx = Math.cos(angle) * this.maxDistance;
      dy = Math.sin(angle) * this.maxDistance;
    }

    // Store current position
    this.currentX = dx;
    this.currentY = dy;

    // Calculate normalized delta (-1 to 1) - immediate for input responsiveness
    this.deltaX = dx / this.maxDistance;
    this.deltaY = dy / this.maxDistance;

    // Calculate raw magnitude (before deadzone) - reuse distance calculation
    const rawMagnitude = Math.sqrt(
      this.deltaX * this.deltaX + this.deltaY * this.deltaY
    );

    // Calculate speed multiplier based on raw distance (0 to 1)
    // At center (0) = 0 speed, at max distance (1) = full speed
    // Smoothly ramp from 0 to 1 as stick moves from center to edge
    if (rawMagnitude < this.deadzone) {
      this.speedMultiplier = 0;
      this.deltaX = 0;
      this.deltaY = 0;
    } else {
      // Speed multiplier: remap from [deadzone, 1] to [0, 1]
      // This gives smooth speed ramp from deadzone to max distance
      this.speedMultiplier =
        (rawMagnitude - this.deadzone) / (1 - this.deadzone);

      // Direction: remap from [deadzone, 1] to [0, 1] for normalized direction
      const scale =
        (rawMagnitude - this.deadzone) / (1 - this.deadzone) / rawMagnitude;
      this.deltaX *= scale;
      this.deltaY *= scale;
    }

    // Schedule visual update via requestAnimationFrame (debounced to 60fps)
    this.pendingDx = dx;
    this.pendingDy = dy;
    if (!this.pendingVisualUpdate) {
      this.pendingVisualUpdate = true;
      requestAnimationFrame(() => {
        this.updateVisualPosition();
      });
    }
  }

  /**
   * Update visual stick position (called via requestAnimationFrame)
   * @private
   */
  updateVisualPosition() {
    this.pendingVisualUpdate = false;
    // Use transform3d for better performance (GPU acceleration)
    this.stick.style.transform = `translate3d(calc(-50% + ${this.pendingDx}px), calc(-50% + ${this.pendingDy}px), 0)`;
  }

  /**
   * Reset joystick to center
   */
  reset() {
    this.active = false;
    this.touchId = null;
    this.currentX = 0;
    this.currentY = 0;
    this.deltaX = 0;
    this.deltaY = 0;
    this.speedMultiplier = 0;
    this.pendingDx = 0;
    this.pendingDy = 0;
    this.pendingVisualUpdate = false;
    this.unlockedThisSession = false;

    // Reset visual
    this.stick.style.transform = "translate3d(-50%, -50%, 0)";
    this.container.style.opacity = "0.6";
    this.stick.style.background = "rgba(255, 255, 255, 0.5)";
  }

  /**
   * Get current joystick values
   * @returns {{x: number, y: number}} Normalized values (-1 to 1)
   */
  getValue() {
    return {
      x: this.deltaX,
      y: -this.deltaY, // Invert Y for game coordinates (up = positive)
    };
  }

  /**
   * Get speed multiplier based on how far the stick is pushed
   * @returns {number} Speed multiplier (0 to 1), where 1 = maximum speed at full extension
   */
  getSpeedMultiplier() {
    return this.speedMultiplier;
  }

  /**
   * Check if joystick is currently active
   * @returns {boolean}
   */
  isActive() {
    return this.active;
  }

  /**
   * Show the joystick (only on touch devices)
   */
  show() {
    if (this.isTouchDevice) {
      this.container.style.display = "block";
    }
  }

  /**
   * Hide the joystick
   */
  hide() {
    this.container.style.display = "none";
    this.reset();
  }

  /**
   * Fade out the joystick (reduce opacity)
   */
  fadeOut() {
    if (this.container.style.display !== "none") {
      this.container.style.opacity = "0";
      this.container.style.pointerEvents = "none";
    }
  }

  /**
   * Fade in the joystick (restore opacity)
   */
  fadeIn() {
    if (this.isTouchDevice) {
      // Ensure joystick is visible and can receive input
      if (this.container.style.display === "none") {
        this.container.style.display = "block";
      }
      this.container.style.opacity = "0.6";
      this.container.style.pointerEvents = "auto";
    }
  }

  /**
   * Destroy the joystick and remove from DOM
   */
  destroy() {
    if (this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }
}
