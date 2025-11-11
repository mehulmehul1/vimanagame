/**
 * TouchJoystick - Virtual joystick for mobile touch controls
 *
 * Creates a visual joystick that responds to touch input within a range of motion.
 * Returns normalized values (-1 to 1) for X and Y axes.
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
      transform: translate(-50%, -50%);
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
        // Audio/video unlocking is handled by global unlockAudioOnInteraction handler
        // (in capture phase, so it fires before this handler)
        // Just call it to refresh gesture context if needed
        if (window.unlockAudioOnInteraction) {
          window.unlockAudioOnInteraction();
        }

        e.preventDefault();

        if (this.active) return; // Already handling a touch

        const touch = e.changedTouches[0];
        this.touchId = touch.identifier;

        const rect = this.container.getBoundingClientRect();
        this.centerX = rect.left + rect.width / 2;
        this.centerY = rect.top + rect.height / 2;

        this.active = true;
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
            // Refresh gesture context during long holds (iOS Safari gesture context can expire)
            // Call unlock on touchmove to maintain gesture context throughout the hold
            if (window.unlockAudioOnInteraction) {
              window.unlockAudioOnInteraction();
            }

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
          // CRITICAL: Unlock on touchend for long holds
          // iOS Safari may not establish gesture context on long holds (touchstart),
          // but it WILL establish it on touchend after a hold
          // This ensures audio/video can play after releasing a long hold
          if (window.unlockAudioOnInteraction) {
            window.unlockAudioOnInteraction();
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

    // Calculate distance and constrain to max
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance > this.maxDistance) {
      const angle = Math.atan2(dy, dx);
      dx = Math.cos(angle) * this.maxDistance;
      dy = Math.sin(angle) * this.maxDistance;
    }

    // Update stick visual position
    this.stick.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;

    // Store current position
    this.currentX = dx;
    this.currentY = dy;

    // Calculate normalized delta (-1 to 1)
    this.deltaX = dx / this.maxDistance;
    this.deltaY = dy / this.maxDistance;

    // Calculate raw magnitude (before deadzone)
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

    // Reset visual
    this.stick.style.transform = "translate(-50%, -50%)";
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
