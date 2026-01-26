# Touch Joystick (Mobile) - First Principles Guide

## Overview

The **Touch Joystick** is the primary input method for mobile players, providing virtual analog stick controls for movement and camera. Since touchscreens lack physical buttons, the joystick recreates analog stick functionality through touch input, allowing players to move in any direction with varying speed based on how far they "push" the virtual stick.

Think of the touch joystick as the **"invisible gamepad"**—like a physical controller's analog stick brought to life on screen, it translates finger position into character movement with full 360-degree control.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Make mobile controls feel responsive and natural, not clunky. Players should forget they're using a touchscreen and focus on the game, with the joystick becoming an extension of their intent.

**Why Virtual Joysticks for Mobile?**
- **Precision**: 360-degree movement vs tap-to-move
- **Speed Control**: Partial push = walk, full push = run
- **Muscle Memory**: Similar to physical controllers
- **Screen Space**: Can be positioned optimally
- **Multi-touch**: Two joysticks for dual-stick controls

**Player Psychology**:
```
Touch Screen → "Where do I touch?" → Uncertainty
     ↓
Joystick Appears → "Oh, like a controller" → Familiarity
     ↓
Push Slightly → Character walks → "I have control!"
     ↓
Push Further → Character runs → Gradation understood
     ↓
Release → Character stops → Responsive feedback
```

### Design Principles

**1. Appear on Touch**
Don't show joystick constantly—it appears when player touches in the joystick zone and fades when released. Keeps screen clean while providing controls when needed.

**2. Snap to Thumb Position**
Place the joystick center where the player first touches, not at a fixed position. This adapts to how the player holds their device.

**3. Visual Feedback**
Show clearly:
- Joystick base (outer circle)
- Thumb position (inner stick)
- Active direction indicator
- Visual boundary (how far you can push)

**4. Haptic Feedback**
Vibrate slightly when:
- Joystick first appears
- Reaching maximum push
- Direction changes sharply

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the touch joystick, you should know:
- **Touch Events** - touchstart, touchmove, touchend, touchcancel
- **Touch Points** - Tracking multiple simultaneous touches
- **Vector Math** - Calculating direction and magnitude
- **Dead Zones** - Small movements shouldn't register
- **Clamping** - Limiting joystick to maximum radius

### Core Architecture

```
TOUCH JOYSTICK ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                  TOUCH JOYSTICK MANAGER                 │
│  - Joystick zone detection                             │
│  - Touch point tracking                                 │
│  - Vector calculation (x, y magnitude)                 │
│  - Dead zone handling                                   │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   LEFT STICK  │  │  RIGHT STICK │  │   EVENTS     │
│  - Movement   │  │  - Camera    │  │  - On move   │
│  - Position   │  │  - Look      │  │  - On stop   │
└───────────────┘  └───────────────┘  └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   INPUT      │
                    │  MANAGER     │
                    │  - Normalize │
                    │  - Route to  │
                    │    game      │
                    └───────────────┘
```

### TouchJoystick Class

```javascript
class TouchJoystick {
  constructor(options = {}) {
    this.gameManager = options.gameManager;
    this.inputManager = options.inputManager;
    this.container = options.container || document.body;

    // Joystick configuration
    this.config = {
      side: options.side || 'left',  // 'left' or 'right'
      maxRadius: options.maxRadius || 50,  // Maximum stick movement
      deadZone: options.deadZone || 0.15,   // 15% dead zone
      fadeOpacity: options.fadeOpacity || 0.3,  // When inactive
      activeOpacity: options.activeOpacity || 0.8,  // When active
      snapToTouch: options.snapToTouch !== false,  // Center at touch point
      hideDelay: options.hideDelay || 200,  // ms before fading
      enableHaptics: options.enableHaptics !== false,
      allowDiagonal: true
    };

    // State
    this.isActive = false;
    this.touchId = null;
    this.center = { x: 0, y: 0 };
    this.current = { x: 0, y: 0 };
    this.normalized = { x: 0, y: 0 };  // -1 to 1
    this.magnitude = 0;
    this.angle = 0;

    // DOM elements
    this.element = null;
    this.baseElement = null;
    this.stickElement = null;

    // Timers
    this.hideTimeout = null;
    this.hapticCooldown = null;

    // Create joystick
    this.createElement();

    // Setup touch handlers
    this.setupTouchHandlers();

    // Setup keyboard fallback (for testing)
    this.setupKeyboardFallback();
  }

  /**
   * Create joystick DOM elements
   */
  createElement() {
    // Main joystick container
    this.element = document.createElement('div');
    this.element.className = `touch-joystick touch-joystick-${this.config.side}`;
    this.element.style.cssText = `
      position: fixed;
      ${this.config.side}: 0;
      top: 0;
      width: 50%;
      height: 100%;
      z-index: 1000;
      touch-action: none;
      pointer-events: auto;
    `;

    // Joystick base (outer circle)
    this.baseElement = document.createElement('div');
    this.baseElement.className = 'joystick-base';
    this.baseElement.style.cssText = `
      position: absolute;
      width: ${this.config.maxRadius * 2}px;
      height: ${this.config.maxRadius * 2}px;
      border-radius: 50%;
      background: radial-gradient(circle,
        rgba(255, 255, 255, 0.1) 0%,
        rgba(255, 255, 255, 0.05) 70%,
        transparent 100%);
      border: 2px solid rgba(255, 255, 255, 0.2);
      opacity: ${this.config.fadeOpacity};
      transition: opacity 0.2s ease;
      pointer-events: none;
      transform: translate(-50%, -50%);
    `;

    // Joystick stick (inner circle)
    this.stickElement = document.createElement('div');
    this.stickElement.className = 'joystick-stick';
    this.stickElement.style.cssText = `
      position: absolute;
      width: ${this.config.maxRadius * 0.6}px;
      height: ${this.config.maxRadius * 0.6}px;
      border-radius: 50%;
      background: radial-gradient(circle,
        rgba(0, 255, 136, 0.6) 0%,
        rgba(0, 255, 136, 0.3) 70%,
        transparent 100%);
      border: 2px solid rgba(0, 255, 136, 0.5);
      opacity: ${this.config.fadeOpacity};
      transition: opacity 0.2s ease;
      pointer-events: none;
      transform: translate(-50%, -50%);
      box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
    `;

    // Direction indicators (optional visual aid)
    this.createDirectionIndicators();

    // Assemble
    this.baseElement.appendChild(this.stickElement);
    this.element.appendChild(this.baseElement);
    this.container.appendChild(this.element);

    // Initially hidden
    this.setVisible(false);
  }

  /**
   * Create direction indicator lines
   */
  createDirectionIndicators() {
    const directions = [
      { x: 0, y: -1, rotation: 0 },    // Up
      { x: 1, y: -1, rotation: 45 },  // Up-Right
      { x: 1, y: 0, rotation: 90 },   // Right
      { x: 1, y: 1, rotation: 135 },  // Down-Right
      { x: 0, y: 1, rotation: 180 },  // Down
      { x: -1, y: 1, rotation: 225 }, // Down-Left
      { x: -1, y: 0, rotation: 270 }, // Left
      { x: -1, y: -1, rotation: 315 } // Up-Left
    ];

    directions.forEach(dir => {
      const indicator = document.createElement('div');
      indicator.className = 'direction-indicator';
      indicator.style.cssText = `
        position: absolute;
        width: ${this.config.maxRadius * 0.3}px;
        height: 2px;
        background: rgba(255, 255, 255, 0.1);
        top: 50%;
        left: 50%;
        transform-origin: left center;
        transform: rotate(${dir.rotation}deg) translateX(${this.config.maxRadius * 0.4}px);
        pointer-events: none;
      `;
      this.baseElement.appendChild(indicator);
    });
  }

  /**
   * Setup touch event handlers
   */
  setupTouchHandlers() {
    this.element.addEventListener('touchstart', (e) => this.onTouchStart(e), { passive: false });
    this.element.addEventListener('touchmove', (e) => this.onTouchMove(e), { passive: false });
    this.element.addEventListener('touchend', (e) => this.onTouchEnd(e), { passive: false });
    this.element.addEventListener('touchcancel', (e) => this.onTouchEnd(e), { passive: false });
  }

  /**
   * Handle touch start
   */
  onTouchStart(event) {
    event.preventDefault();

    // Find the touch that started in our zone
    for (const touch of event.changedTouches) {
      if (!this.touchId) {
        // This is our touch
        this.touchId = touch.identifier;
        this.isActive = true;

        // Set center position (snap to touch point)
        this.center = {
          x: touch.clientX,
          y: touch.clientY
        };

        // Position joystick base at touch point
        this.baseElement.style.left = `${this.center.x}px`;
        this.baseElement.style.top = `${this.center.y}px`;

        // Reset stick position
        this.current = { ...this.center };
        this.normalized = { x: 0, y: 0 };
        this.magnitude = 0;

        // Make visible
        this.setVisible(true);

        // Cancel hide timeout
        if (this.hideTimeout) {
          clearTimeout(this.hideTimeout);
          this.hideTimeout = null;
        }

        // Haptic feedback
        this.triggerHaptic('start');

        // Emit event
        this.emitInput();

        break;
      }
    }
  }

  /**
   * Handle touch move
   */
  onTouchMove(event) {
    event.preventDefault();

    // Find our touch point
    for (const touch of event.changedTouches) {
      if (touch.identifier === this.touchId) {
        // Calculate offset from center
        const deltaX = touch.clientX - this.center.x;
        const deltaY = touch.clientY - this.center.y;

        // Calculate distance from center
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

        // Clamp to max radius
        const clampedDistance = Math.min(distance, this.config.maxRadius);

        // Calculate angle
        this.angle = Math.atan2(deltaY, deltaX);

        // Calculate current stick position
        if (distance > this.config.maxRadius) {
          // Outside max radius, stick at edge
          this.current = {
            x: this.center.x + Math.cos(this.angle) * this.config.maxRadius,
            y: this.center.y + Math.sin(this.angle) * this.config.maxRadius
          };
        } else {
          // Within max radius, stick at touch position
          this.current = {
            x: touch.clientX,
            y: touch.clientY
          };
        }

        // Calculate normalized output (-1 to 1)
        const rawNormalizedX = deltaX / this.config.maxRadius;
        const rawNormalizedY = deltaY / this.config.maxRadius;

        // Apply dead zone
        const magnitude = Math.sqrt(
          rawNormalizedX * rawNormalizedX +
          rawNormalizedY * rawNormalizedY
        );

        if (magnitude < this.config.deadZone) {
          this.normalized = { x: 0, y: 0 };
          this.magnitude = 0;
        } else {
          // Scale dead zone to 0
          const scale = (magnitude - this.config.deadZone) / (1 - this.config.deadZone);
          this.normalized = {
            x: (rawNormalizedX / magnitude) * scale,
            y: (rawNormalizedY / magnitude) * scale
          };
          this.magnitude = scale;
        }

        // Constrain to octagon if diagonal not allowed
        if (!this.config.allowDiagonal) {
          this.constrainToCardinal();
        }

        // Update stick position
        this.updateStickPosition();

        // Haptic feedback on max push
        if (magnitude >= 0.95 && !this.atMax) {
          this.atMax = true;
          this.triggerHaptic('max');
        } else if (magnitude < 0.9) {
          this.atMax = false;
        }

        // Emit input event
        this.emitInput();

        break;
      }
    }
  }

  /**
   * Constrain to cardinal directions (no diagonal)
   */
  constrainToCardinal() {
    const absX = Math.abs(this.normalized.x);
    const absY = Math.abs(this.normalized.y);

    // Only allow dominant axis
    if (absX > absY * 2) {
      // Horizontal dominant
      this.normalized.y = 0;
    } else if (absY > absX * 2) {
      // Vertical dominant
      this.normalized.x = 0;
    } else {
      // Near diagonal, snap to closer axis
      if (absX > absY) {
        this.normalized.y = 0;
      } else {
        this.normalized.x = 0;
      }
    }

    // Recalculate magnitude
    this.magnitude = Math.max(Math.abs(this.normalized.x), Math.abs(this.normalized.y));
  }

  /**
   * Update stick visual position
   */
  updateStickPosition() {
    const stickDeltaX = this.current.x - this.center.x;
    const stickDeltaY = this.current.y - this.center.y;

    this.stickElement.style.transform = `
      translate(calc(-50% + ${stickDeltaX}px), calc(-50% + ${stickDeltaY}px))
    `;

    // Update stick appearance based on push amount
    const intensity = this.magnitude;
    this.stickElement.style.opacity = this.config.activeOpacity * (0.5 + intensity * 0.5);
    this.stickElement.style.boxShadow = `0 0 ${15 + intensity * 10}px rgba(0, 255, 136, ${0.3 + intensity * 0.3})`;
  }

  /**
   * Handle touch end
   */
  onTouchEnd(event) {
    for (const touch of event.changedTouches) {
      if (touch.identifier === this.touchId) {
        this.touchId = null;
        this.isActive = false;

        // Reset values
        this.normalized = { x: 0, y: 0 };
        this.magnitude = 0;

        // Center stick
        this.stickElement.style.transform = 'translate(-50%, -50%)';

        // Emit final input (zeroed)
        this.emitInput();

        // Schedule hide (with delay for visual smoothness)
        this.hideTimeout = setTimeout(() => {
          this.setVisible(false);
        }, this.config.hideDelay);

        // Haptic feedback
        this.triggerHaptic('end');

        break;
      }
    }
  }

  /**
   * Set joystick visibility
   */
  setVisible(visible) {
    this.element.style.display = visible ? '' : 'none';
    this.baseElement.style.opacity = visible ? this.config.activeOpacity : this.config.fadeOpacity;
    this.stickElement.style.opacity = visible ? this.config.activeOpacity : this.config.fadeOpacity;
  }

  /**
   * Trigger haptic feedback
   */
  triggerHaptic(type) {
    if (!this.config.enableHaptics) return;

    // Check cooldown
    if (this.hapticCooldown && Date.now() < this.hapticCooldown) {
      return;
    }

    try {
      if (navigator.vibrate) {
        switch (type) {
          case 'start':
            navigator.vibrate(10);
            break;
          case 'end':
            navigator.vibrate(5);
            break;
          case 'max':
            navigator.vibrate(15);
            this.hapticCooldown = Date.now() + 200;  // 200ms cooldown
            break;
        }
      }
    } catch (e) {
      // Vibration not supported or permission denied
    }
  }

  /**
   * Emit input event
   */
  emitInput() {
    const eventData = {
      side: this.config.side,
      x: this.normalized.x,
      y: this.normalized.y,
      magnitude: this.magnitude,
      angle: this.angle,
      isActive: this.isActive
    };

    // Emit to input manager
    if (this.inputManager) {
      if (this.config.side === 'left') {
        this.inputManager.handleJoystickLeft(eventData);
      } else {
        this.inputManager.handleJoystickRight(eventData);
      }
    }

    // Emit to game manager
    this.gameManager.emit(`joystick:${this.config.side}`, eventData);
  }

  /**
   * Setup keyboard fallback for testing
   */
  setupKeyboardFallback() {
    // Only for development/testing on desktop
    if (!this.isMobile()) return;

    this.keyboardState = { x: 0, y: 0 };

    const keyMap = this.config.side === 'left' ? {
      up: ['KeyW', 'ArrowUp'],
      down: ['KeyS', 'ArrowDown'],
      left: ['KeyA', 'ArrowLeft'],
      right: ['KeyD', 'ArrowRight']
    } : {
      up: ['ArrowUp'],
      down: ['ArrowDown'],
      left: ['ArrowLeft'],
      right: ['ArrowRight']
    };

    const updateKeyboardInput = () => {
      if (!this.keyboardState.x && !this.keyboardState.y) {
        if (this.isActive) {
          // Simulate release
          this.isActive = false;
          this.normalized = { x: 0, y: 0 };
          this.magnitude = 0;
          this.emitInput();
          this.setVisible(false);
        }
        return;
      }

      // Activate if not already
      if (!this.isActive) {
        this.isActive = true;
        this.center = { x: 200, y: window.innerHeight - 200 };
        this.baseElement.style.left = `${this.center.x}px`;
        this.baseElement.style.top = `${this.center.y}px`;
        this.setVisible(true);
      }

      // Set normalized input
      this.normalized = {
        x: this.keyboardState.x,
        y: this.keyboardState.y
      };
      this.magnitude = Math.sqrt(
        this.normalized.x * this.normalized.x +
        this.normalized.y * this.normalized.y
      );

      // Update visual
      const visualX = this.normalized.x * this.config.maxRadius * 0.8;
      const visualY = this.normalized.y * this.config.maxRadius * 0.8;
      this.stickElement.style.transform = `
        translate(calc(-50% + ${visualX}px), calc(-50% + ${visualY}px))
      `;

      this.emitInput();
    };

    document.addEventListener('keydown', (e) => {
      const code = e.code;

      if (keyMap.up.includes(code)) {
        this.keyboardState.y = -1;
        e.preventDefault();
      }
      if (keyMap.down.includes(code)) {
        this.keyboardState.y = 1;
        e.preventDefault();
      }
      if (keyMap.left.includes(code)) {
        this.keyboardState.x = -1;
        e.preventDefault();
      }
      if (keyMap.right.includes(code)) {
        this.keyboardState.x = 1;
        e.preventDefault();
      }

      updateKeyboardInput();
    });

    document.addEventListener('keyup', (e) => {
      const code = e.code;

      if (keyMap.up.includes(code) && this.keyboardState.y < 0) {
        this.keyboardState.y = 0;
      }
      if (keyMap.down.includes(code) && this.keyboardState.y > 0) {
        this.keyboardState.y = 0;
      }
      if (keyMap.left.includes(code) && this.keyboardState.x < 0) {
        this.keyboardState.x = 0;
      }
      if (keyMap.right.includes(code) && this.keyboardState.x > 0) {
        this.keyboardState.x = 0;
      }

      updateKeyboardInput();
    });
  }

  /**
   * Check if running on mobile
   */
  isMobile() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
           ('ontouchstart' in window) ||
           (navigator.maxTouchPoints > 0);
  }

  /**
   * Get current values
   */
  getValue() {
    return {
      x: this.normalized.x,
      y: this.normalized.y,
      magnitude: this.magnitude,
      angle: this.angle,
      isActive: this.isActive
    };
  }

  /**
   * Reset joystick state
   */
  reset() {
    this.isActive = false;
    this.touchId = null;
    this.normalized = { x: 0, y: 0 };
    this.magnitude = 0;
    this.setVisible(false);
  }

  /**
   * Clean up
   */
  destroy() {
    // Remove event listeners
    this.element.removeEventListener('touchstart', this.onTouchStart);
    this.element.removeEventListener('touchmove', this.onTouchMove);
    this.element.removeEventListener('touchend', this.onTouchEnd);
    this.element.removeEventListener('touchcancel', this.onTouchEnd);

    // Remove from DOM
    this.element.remove();

    // Clear timers
    if (this.hideTimeout) {
      clearTimeout(this.hideTimeout);
    }
  }
}

/**
 * Manager for dual joysticks
 */
class TouchJoystickManager {
  constructor(options = {}) {
    this.gameManager = options.gameManager;
    this.inputManager = options.inputManager;
    this.container = options.container || document.body;

    // Create joysticks
    this.leftJoystick = new TouchJoystick({
      side: 'left',
      maxRadius: options.maxRadius,
      deadZone: options.deadZone,
      gameManager: this.gameManager,
      inputManager: this.inputManager,
      container: this.container
    });

    this.rightJoystick = new TouchJoystick({
      side: 'right',
      maxRadius: options.maxRadius,
      deadZone: options.deadZone,
      gameManager: this.gameManager,
      inputManager: this.inputManager,
      container: this.container
    });

    // Auto-detect mobile and enable/disable
    if (this.isMobile()) {
      this.enable();
    } else {
      this.disable();
    }

    // Listen for resize
    window.addEventListener('resize', () => this.onResize());
  }

  /**
   * Enable joysticks
   */
  enable() {
    this.leftJoystick.element.style.display = '';
    this.rightJoystick.element.style.display = '';
    this.enabled = true;
  }

  /**
   * Disable joysticks
   */
  disable() {
    this.leftJoystick.element.style.display = 'none';
    this.rightJoystick.element.style.display = 'none';
    this.enabled = false;
  }

  /**
   * Toggle based on device
   */
  onResize() {
    if (this.isMobile() && !this.enabled) {
      this.enable();
    } else if (!this.isMobile() && this.enabled) {
      this.disable();
    }
  }

  /**
   * Check if mobile
   */
  isMobile() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
           ('ontouchstart' in window) ||
           (navigator.maxTouchPoints > 0);
  }

  /**
   * Get both joystick values
   */
  getValues() {
    return {
      left: this.leftJoystick.getValue(),
      right: this.rightJoystick.getValue()
    };
  }

  /**
   * Clean up
   */
  destroy() {
    this.leftJoystick.destroy();
    this.rightJoystick.destroy();
    window.removeEventListener('resize', this.onResize);
  }
}

export default TouchJoystick;
export { TouchJoystickManager };
```

---

## 📝 How To Build A Touch Joystick Like This

### Step 1: Create Basic Joystick

```javascript
class SimpleJoystick {
  constructor(container) {
    this.container = container;
    this.active = false;
    this.center = { x: 0, y: 0 };
    this.current = { x: 0, y: 0 };
    this.touchId = null;

    // Create elements
    this.base = document.createElement('div');
    this.base.className = 'joystick-base';
    this.stick = document.createElement('div');
    this.stick.className = 'joystick-stick';
    this.base.appendChild(this.stick);
    this.container.appendChild(this.base);

    // Add styles
    this.base.style.cssText = `
      position: absolute;
      width: 100px; height: 100px;
      border-radius: 50%;
      background: rgba(255,255,255,0.2);
      display: none;
    `;
    this.stick.style.cssText = `
      position: absolute;
      width: 40px; height: 40px;
      border-radius: 50%;
      background: rgba(0,255,136,0.5);
      transform: translate(-50%, -50%);
    `;

    // Touch handling
    this.base.addEventListener('touchstart', (e) => this.onStart(e));
    this.base.addEventListener('touchmove', (e) => this.onMove(e));
    this.base.addEventListener('touchend', (e) => this.onEnd(e));
  }

  onStart(e) {
    e.preventDefault();
    const touch = e.changedTouches[0];
    this.touchId = touch.identifier;
    this.active = true;
    this.center = { x: touch.clientX, y: touch.clientY };
    this.base.style.left = (this.center.x - 50) + 'px';
    this.base.style.top = (this.center.y - 50) + 'px';
    this.base.style.display = 'block';
  }

  onMove(e) {
    e.preventDefault();
    for (const touch of e.changedTouches) {
      if (touch.identifier === this.touchId) {
        const dx = touch.clientX - this.center.x;
        const dy = touch.clientY - this.center.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const maxDist = 50;

        if (dist > maxDist) {
          const angle = Math.atan2(dy, dx);
          this.current.x = this.center.x + Math.cos(angle) * maxDist;
          this.current.y = this.center.y + Math.sin(angle) * maxDist;
        } else {
          this.current.x = touch.clientX;
          this.current.y = touch.clientY;
        }

        const stickX = this.current.x - this.center.x;
        const stickY = this.current.y - this.center.y;
        this.stick.style.transform = `translate(calc(-50% + ${stickX}px), calc(-50% + ${stickY}px))`;

        // Output normalized value
        const output = {
          x: Math.max(-1, Math.min(1, stickX / 50)),
          y: Math.max(-1, Math.min(1, stickY / 50))
        };
        console.log('Joystick:', output);
      }
    }
  }

  onEnd(e) {
    for (const touch of e.changedTouches) {
      if (touch.identifier === this.touchId) {
        this.active = false;
        this.touchId = null;
        this.base.style.display = 'none';
      }
    }
  }
}
```

---

## 🔧 Variations For Your Game

### Dynamic Joystick Zones

```javascript
// Detect which half of screen was touched
document.addEventListener('touchstart', (e) => {
  for (const touch of e.changedTouches) {
    if (touch.clientX < window.innerWidth / 2) {
      // Left half - movement joystick
      leftJoystick.activateAt(touch.clientX, touch.clientY);
    } else {
      // Right half - camera joystick
      rightJoystick.activateAt(touch.clientX, touch.clientY);
    }
  }
});
```

### Swipe Gesture Mode

```javascript
// Instead of joystick, use swipes for direction
class SwipeController {
  constructor() {
    this.startPos = { x: 0, y: 0 };
    this.threshold = 50;  // Minimum swipe distance

    document.addEventListener('touchstart', (e) => {
      this.startPos = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    });

    document.addEventListener('touchend', (e) => {
      const dx = e.changedTouches[0].clientX - this.startPos.x;
      const dy = e.changedTouches[0].clientY - this.startPos.y;

      if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > this.threshold) {
        this.emit(dx > 0 ? 'right' : 'left');
      } else if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > this.threshold) {
        this.emit(dy > 0 ? 'down' : 'up');
      }
    });
  }
}
```

### Context-Sensitive Buttons

```javascript
// Show action buttons near right joystick when interactive objects nearby
class ActionButtonManager {
  constructor(joystick) {
    this.joystick = joystick;
    this.buttons = [];
  }

  showAction(action, callback) {
    const btn = document.createElement('button');
    btn.className = 'context-action';
    btn.textContent = action.icon;
    btn.style.position = 'fixed';
    btn.style.right = '20px';
    btn.style.bottom = (100 + this.buttons.length * 60) + 'px';
    btn.onclick = callback;
    document.body.appendChild(btn);
    this.buttons.push(btn);
  }

  clearActions() {
    this.buttons.forEach(btn => btn.remove());
    this.buttons = [];
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Fixed Joystick Position

```javascript
// ❌ WRONG: Joystick always in corner
base.style.left = '50px';
base.style.top = 'calc(100% - 150px)';
// Uncomfortable for different hand sizes

// ✅ CORRECT: Snap to touch point
base.style.left = (touch.clientX - 50) + 'px';
base.style.top = (touch.clientY - 50) + 'px';
// Adapts to player's hand position
```

### 2. No Dead Zone

```javascript
// ❌ WRONG: Tiny movements register
const x = deltaX / maxRadius;
const y = deltaY / maxRadius;
// Drift when player's thumb isn't perfectly still

// ✅ CORRECT: Apply dead zone
const magnitude = Math.sqrt(x * x + y * y);
if (magnitude < 0.15) {
  return { x: 0, y: 0 };  // Dead zone
}
const scaled = (magnitude - 0.15) / (1 - 0.15);
return { x: (x / magnitude) * scaled, y: (y / magnitude) * scaled };
// Smooth control near center
```

### 3. No Visual Feedback

```javascript
// ❌ WRONG: Invisible joystick
// Player doesn't know where they're touching

// ✅ CORRECT: Show joystick on touch
base.style.display = 'block';
base.style.opacity = '0.8';
// Clear visual indication
```

### 4. Canceling Other Touches

```javascript
// ❌ WRONG: preventDefault on all touches
document.addEventListener('touchstart', (e) => {
  e.preventDefault();
});
// Breaks multi-touch (can't use both joysticks)

// ✅ CORRECT: Only handle our touch
document.addEventListener('touchstart', (e) => {
  for (const touch of e.changedTouches) {
    if (touch.identifier === this.touchId) {
      e.preventDefault();
    }
  }
});
// Multi-touch works
```

---

## CSS for Touch Joysticks

```css
/* Container zones */
.touch-joystick-left,
.touch-joystick-right {
  position: fixed;
  top: 0;
  width: 50%;
  height: 100%;
  touch-action: none;
  z-index: 1000;
}

.touch-joystick-left {
  left: 0;
}

.touch-joystick-right {
  right: 0;
}

/* Joystick base */
.joystick-base {
  position: absolute;
  border-radius: 50%;
  background: radial-gradient(circle,
    rgba(255, 255, 255, 0.15) 0%,
    rgba(255, 255, 255, 0.05) 60%,
    transparent 70%);
  border: 2px solid rgba(255, 255, 255, 0.2);
  pointer-events: none;
  user-select: none;
  will-change: transform, opacity;
}

/* Joystick stick */
.joystick-stick {
  position: absolute;
  border-radius: 50%;
  background: radial-gradient(circle,
    rgba(0, 255, 136, 0.5) 0%,
    rgba(0, 255, 136, 0.2) 70%);
  border: 2px solid rgba(0, 255, 136, 0.5);
  box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
  pointer-events: none;
  user-select: none;
  will-change: transform;
}

/* Direction indicators */
.direction-indicator {
  position: absolute;
  background: rgba(255, 255, 255, 0.1);
  pointer-events: none;
}

/* Active state */
.joystick-base.active,
.joystick-stick.active {
  opacity: 0.8;
}

/* Pressed effect */
.joystick-stick.pressed {
  background: radial-gradient(circle,
    rgba(0, 255, 136, 0.7) 0%,
    rgba(0, 255, 136, 0.4) 70%);
  transform: translate(-50%, -50%) scale(0.95);
}
```

---

## Performance Considerations

```
TOUCH JOYSTICK PERFORMANCE:

Touch Events:
├── Frequency: Every frame during movement
├── Processing: Minimal math
└── Impact: Negligible

DOM Updates:
├── Position changes: transform (GPU-accelerated)
├── Opacity changes: CSS transition
└── Impact: Minimal

Optimization:
├── Use transform, not left/top
├── Use requestAnimationFrame for updates
├── Debounce non-critical updates
└── Avoid garbage collection during active use
```

---

## Related Systems

- [UIManager](./ui-manager.md) - Mobile UI management
- [InputManager](../04-input-physics/input-manager.md) - Input coordination
- [Fullscreen Button](./fullscreen-button.md) - Mobile control

---

## Source File Reference

**Primary Files**:
- `../src/ui/TouchJoystick.js` - Touch joystick implementation (estimated)

**Key Classes**:
- `TouchJoystick` - Single virtual joystick
- `TouchJoystickManager` - Dual joystick coordination

**Dependencies**:
- Touch Events API (input handling)
- Vibration API (haptic feedback)
- Vector math (direction/magnitude)

---

## References

- [Touch Events MDN](https://developer.mozilla.org/en-US/docs/Web/API/Touch_events) - Touch input
- [Vibration API](https://developer.mozilla.org/en-US/docs/Web/API/Vibration_API) - Haptic feedback
- [Pointer Events](https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events) - Unified input

*Documentation last updated: January 12, 2026*
