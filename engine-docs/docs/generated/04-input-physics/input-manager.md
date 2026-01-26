# InputManager - First Principles Guide

## Overview

The **InputManager** is the unified input system for the Shadow Engine. It handles all player input from multiple sources - keyboard, mouse, gamepad, and touch - and converts them into consistent actions that the rest of the engine can use.

Think of InputManager as a **universal translator** - it takes raw input signals (key presses, mouse movements, touch events) and translates them into game actions like "move forward", "interact", or "jump".

## What You Need to Know First

Before understanding InputManager, you should know:
- **Event listeners** - How JavaScript responds to user input
- **Keyboard events** - `keydown`, `keyup` events
- **Mouse events** - `mousemove`, `mousedown`, `mouseup`, `wheel`
- **Touch events** - `touchstart`, `touchmove`, `touchend`
- **Gamepad API** - Browser API for game controllers
- **Normalization** - Converting different input types to the same action

### Quick Refresher: Input Events

```javascript
// Keyboard
window.addEventListener('keydown', (e) => {
  console.log('Key pressed:', e.code);  // e.g., "KeyW", "KeyA"
});

// Mouse
window.addEventListener('mousemove', (e) => {
  console.log('Mouse position:', e.clientX, e.clientY);
});

// Touch
window.addEventListener('touchstart', (e) => {
  console.log('Touch:', e.touches[0].clientX, e.touches[0].clientY);
});

// Gamepad
window.addEventListener('gamepadconnected', (e) => {
  console.log('Gamepad connected:', e.gamepad.id);
});
```

---

## Part 1: Why a Unified Input System?

### The Problem: Multiple Input Sources

Without a unified system, each input type requires separate handling:

```javascript
// ❌ WITHOUT InputManager: Scattered input handling

// In CharacterController
window.addEventListener('keydown', (e) => {
  if (e.code === 'KeyW') moveForward();
  if (e.code === 'KeyS') moveBackward();
  // ... handle all keys
});

// In UI code
document.getElementById('interactButton').addEventListener('click', () => {
  interact();
});

// In gamepad code (if you even added it)
window.addEventListener('gamepaddown', (e) => {
  // What button? What does it do?
});
```

**Problems:**
- Input handling scattered across the codebase
- Inconsistent behavior between input types
- Difficult to add new input types
- No central place to remap controls

### The Solution: InputManager

```javascript
// ✅ WITH InputManager: Unified, abstracted input

const inputManager = new InputManager();

// All components use the same API
if (inputManager.isActionPressed('moveForward')) {
  moveForward();
}

if (inputManager.isActionJustPressed('interact')) {
  interact();
}

// Works with keyboard, mouse, gamepad, touch - automatically!
```

---

## Part 2: Actions vs Raw Input

### Input Abstraction Layer

InputManager translates raw input into **actions**:

```
RAW INPUT                    INPUT MAPPING                GAME ACTIONS
─────────────────────────────────────────────────────────────────────
Keyboard "W" key         │                               │
Gamepad "Left Stick"     ├────▶ "moveForward"     ├────▶ Character moves forward
Touch "Up" virtual       │                               │
Swipe up gesture          │                               │
─────────────────────────────────────────────────────────────────────
Keyboard "E" key         │                               │
Gamepad "A" button       ├────▶ "interact"         ├────▶ Interaction with object
Touch "Tap" screen       │                               │
─────────────────────────────────────────────────────────────────────
Keyboard "Space" key     │                               │
Gamepad "X" button       ├────▶ "jump"              ├────▶ Character jumps
Touch "Up" swipe         │                               │
```

### Supported Actions

| Action | Description | Default Bindings |
|--------|-------------|-------------------|
| `moveForward` | Move forward | W, Up arrow, Gamepad L-stick up |
| `moveBackward` | Move backward | S, Down arrow, Gamepad L-stick down |
| `moveLeft` | Move left (strafe) | A, Left arrow, Gamepad L-stick left |
| `moveRight` | Move right (strafe) | D, Right arrow, Gamepad L-stick right |
| `look` | Look around | Mouse movement, Gamepad R-stick |
| `interact` | Interact with object | E, Gamepad A, Touch tap |
| `jump` | Jump | Space, Gamepad X, Touch swipe up |
| `crouch` | Crouch | Ctrl, Gamepad B, Touch swipe down |
| `menu` | Open menu | Escape, Gamepad Start, Touch menu button |
| `nextDialog` | Skip dialog/next choice | Enter, Gamepad A, Touch tap |

---

## Part 3: InputManager Structure

### Class Overview

```javascript
class InputManager {
  constructor() {
    // Current state of all actions
    this.actions = {
      moveForward: false,
      moveBackward: false,
      moveLeft: false,
      moveRight: false,
      interact: false,
      jump: false,
      crouch: false,
      menu: false
    };

    // Previous frame state (for "just pressed" detection)
    this.previousActions = { ...this.actions };

    // Input values (for analog input)
    this.axes = {
      moveX: 0,    // -1 to 1 (left to right)
      moveY: 0,    // -1 to 1 (down to up)
      lookX: 0,    // -1 to 1
      lookY: 0     // -1 to 1
    };

    // Input sources
    this.keyboard = new Map();  // Pressed keys
    this.mouse = { x: 0, y: 0, dx: 0, dy: 0, buttons: 0 };
    this.gamepads = [];           // Connected gamepads
    this.touch = { active: false, position: null };

    // Configuration
    this.sensitivity = {
      mouse: 0.002,
      gamepad: 1.0,
      touch: 1.0
    };

    // Is pointer locked?
    this.pointerLocked = false;
  }
}
```

---

## Part 4: Keyboard Input

### Key Mapping

```javascript
class InputManager {
  initializeKeyboard() {
    // Key code to action mapping
    this.keyMap = {
      'KeyW': 'moveForward',
      'ArrowUp': 'moveForward',

      'KeyS': 'moveBackward',
      'ArrowDown': 'moveBackward',

      'KeyA': 'moveLeft',
      'ArrowLeft': 'moveLeft',

      'KeyD': 'moveRight',
      'ArrowRight': 'moveRight',

      'KeyE': 'interact',
      'Enter': 'interact',
      'Space': 'jump',
      'ShiftLeft': 'sprint',
      'ControlLeft': 'crouch',
      'Escape': 'menu'
    };

    // Listen for key events
    window.addEventListener('keydown', this.onKeyDown.bind(this));
    window.addEventListener('keyup', this.onKeyUp.bind(this));
  }

  onKeyDown(event) {
    // Map key to action
    const action = this.keyMap[event.code];
    if (action) {
      this.actions[action] = true;
    }

    // Store key state
    this.keyboard.set(event.code, true);
  }

  onKeyUp(event) {
    const action = this.keyMap[event.code];
    if (action) {
      this.actions[action] = false;
    }

    this.keyboard.set(event.code, false);
  }
}
```

### Preventing Default Behavior

```javascript
onKeyDown(event) {
  const action = this.keyMap[event.code];
  if (action) {
    this.actions[action] = true;

    // Prevent default browser behavior
    // (e.g., prevent scrolling with arrow keys)
    if (['Space', 'ArrowUp', 'ArrowDown', 'Tab'].includes(event.code)) {
      event.preventDefault();
    }
  }
}
```

---

## Part 5: Mouse Input

### Mouse Movement and Buttons

```javascript
class InputManager {
  initializeMouse() {
    // Mouse movement
    document.addEventListener('mousemove', this.onMouseMove.bind(this));

    // Mouse buttons
    document.addEventListener('mousedown', this.onMouseDown.bind(this));
    document.addEventListener('mouseup', this.onMouseUp.bind(this));

    // Mouse wheel
    document.addEventListener('wheel', this.onWheel.bind(this));

    // Pointer lock (for first-person look)
    document.addEventListener('pointerlockchange', this.onPointerLockChange.bind(this));
  }

  onMouseMove(event) {
    // Calculate delta movement
    this.mouse.dx = event.movementX;
    this.mouse.dy = event.movementY;

    // Update absolute position
    this.mouse.x = event.clientX;
    this.mouse.y = event.clientY;

    // Map mouse movement to look axes
    if (this.pointerLocked) {
      this.axes.lookX = this.mouse.dx * this.sensitivity.mouse;
      this.axes.lookY = this.mouse.dy * this.sensitivity.mouse;
    }
  }

  onMouseDown(event) {
    this.mouse.buttons |= (1 << event.button);
  }

  onMouseUp(event) {
    this.mouse.buttons &= ~(1 << event.button);
  }

  onWheel(event) {
    // Scroll wheel can be used for menu navigation
    this.mouse.scrollY = event.deltaY;
  }
}
```

### Pointer Lock API

```javascript
requestPointerLock() {
  const canvas = document.body;
  canvas.requestPointerLock();
}

onPointerLockChange() {
  this.pointerLocked = (document.pointerLockElement === document.body);

  if (this.pointerLocked) {
    console.log('[InputManager] Pointer locked - mouse look enabled');
  } else {
    console.log('[InputManager] Pointer unlocked');
  }
}
```

**Verified from MDN docs:**
- `requestPointerLock()` hides cursor and locks pointer to element
- `document.pointerLockElement` indicates if lock is active
- `movementX` and `movementY` only work while pointer locked

---

## Part 6: Gamepad Input

### Gamepad Detection

```javascript
class InputManager {
  initializeGamepad() {
    // Check for already connected gamepads
    this.pollGamepads();

    // Listen for connect/disconnect
    window.addEventListener('gamepadconnected', this.onGamepadConnected.bind(this));
    window.addEventListener('gamepaddisconnected', this.onGamepadDisconnected.bind(this));

    // Poll gamepad state every frame
    // (Gamepad API requires polling)
    this.gameManager.on('update', this.pollGamepads.bind(this));
  }

  onGamepadConnected(event) {
    console.log(`[InputManager] Gamepad connected: ${event.gamepad.id}`);
    this.gamepads.push(event.gamepad);
    this.vibrate(event.gamepad, 0, 100);  // Feedback
  }

  onGamepadDisconnected(event) {
    console.log(`[InputManager] Gamepad disconnected: ${event.gamepad.id}`);
    this.gamepads = this.gamepads.filter(g => g.id !== event.gamepad.id);
  }
}
```

### Gamepad Polling

```javascript
pollGamepads() {
  // Get all gamepads
  const gamepads = navigator.getGamepads();

  for (const gamepad of gamepads) {
    if (!gamepad) continue;

    // Read buttons (standard mapping)
    // Button 0: A/Cross | Button 1: B/Circle | Button 2: X/Square | Button 3: Y/Triangle
    const aPressed = gamepad.buttons[0].pressed;
    const bPressed = gamepad.buttons[1].pressed;
    const xPressed = gamepad.buttons[2].pressed;
    const yPressed = gamepad.buttons[3].pressed;

    // Map buttons to actions
    this.actions.interact = aPressed;  // A button
    this.actions.jump = xPressed;        // X button
    this.actions.crouch = bPressed;      // B button
    this.actions.menu = gamepad.buttons[9].pressed;  // Start button

    // Read axes (left stick for movement, right stick for look)
    const deadzone = 0.15;

    // Left stick (axes 0, 1)
    let moveX = gamepad.axes[0];
    let moveY = gamepad.axes[1];

    // Apply deadzone
    if (Math.abs(moveX) < deadzone) moveX = 0;
    if (Math.abs(moveY) < deadzone) moveY = 0;

    // Update axes (Y is inverted in gamepads)
    this.axes.moveX = moveX * this.sensitivity.gamepad;
    this.axes.moveY = -moveY * this.sensitivity.gamepad;

    // Right stick (axes 2, 3) - for looking
    let lookX = gamepad.axes[2];
    let lookY = gamepad.axes[3];

    if (Math.abs(lookX) < deadzone) lookX = 0;
    if (Math.abs(lookY) < deadzone) lookY = 0;

    this.axes.lookX = lookX * this.sensitivity.gamepad;
    this.axes.lookY = lookY * this.sensitivity.gamepad;

    // Triggers/bumpers
    const leftTrigger = gamepad.buttons[6].value;  // 0 to 1
    const rightTrigger = gamepad.buttons[7].value;

    // Can be used for smooth actions (sprint, zoom, etc.)
    this.axes.sprint = leftTrigger;
    this.axes.aim = rightTrigger;
  }
}
```

**Verified from MDN Gamepad API docs:**
- `navigator.getGamepads()` returns array of connected gamepads
- `gamepad.buttons[n].pressed` is boolean for digital buttons
- `gamepad.buttons[n].value` is 0-1 for analog triggers
- `gamepad.axes[n]` returns -1 to 1 for analog sticks
- Standard mapping: Buttons 0-3 are A/B/X/Y, Axes 0-1 is left stick, 2-3 is right stick

---

## Part 7: Touch Input

### Touch Handling

```javascript
class InputManager {
  initializeTouch() {
    this.touch = {
      active: false,
      touches: new Map(),
      startX: 0,
      startY: 0,
      currentX: 0,
      currentY: 0,
      swipeThreshold: 30  // Minimum pixels for swipe
    };

    document.addEventListener('touchstart', this.onTouchStart.bind(this));
    document.addEventListener('touchmove', this.onTouchMove.bind(this));
    document.addEventListener('touchend', this.onTouchEnd.bind(this));
  }

  onTouchStart(event) {
    this.touch.active = true;

    for (const touch of event.changedTouches) {
      this.touch.touches.set(touch.identifier, {
        startX: touch.clientX,
        startY: touch.clientY,
        currentX: touch.clientX,
        currentY: touch.clientY
      });
    }
  }

  onTouchMove(event) {
    event.preventDefault();  // Prevent scrolling

    for (const touch of event.changedTouches) {
      const touchData = this.touch.touches.get(touch.identifier);
      if (touchData) {
        touchData.currentX = touch.clientX;
        touchData.currentY = touch.clientY;
      }
    }

    // Calculate movement
    const touches = Array.from(this.touch.touches.values());

    if (touches.length === 1) {
      // Single touch - emulate mouse
      const touch = touches[0];
      this.axes.lookX = (touch.currentX - touch.startX) * 0.005;
      this.axes.lookY = (touch.currentY - touch.startY) * 0.005;
    }
  }

  onTouchEnd(event) {
    for (const touch of event.changedTouches) {
      const touchData = this.touch.touches.get(touch.identifier);
      if (touchData) {
        // Check for tap (minimal movement)
        const dx = touchData.currentX - touchData.startX;
        const dy = touchData.currentY - touchData.startY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 10) {
          // Tap - trigger interact
          this.actions.interact = true;
          setTimeout(() => {
            this.actions.interact = false;
          }, 100);
        }

        // Check for swipe
        if (Math.abs(dx) > this.touch.swipeThreshold) {
          // Horizontal swipe
          if (dx > 0) {
            this.actions.nextDialog = true;
          } else {
            this.actions.previousDialog = true;
          }
        }

        if (Math.abs(dy) > this.touch.swipeThreshold) {
          // Vertical swipe
          if (dy > 0) {
            this.actions.crouch = true;
          } else {
            this.actions.jump = true;
          }
        }
      }

      this.touch.touches.delete(touch.identifier);
    }

    if (this.touch.touches.size === 0) {
      this.touch.active = false;
    }
  }
}
```

---

## Part 8: Virtual Joystick (Mobile)

### On-Screen Joystick

```javascript
class InputManager {
  createVirtualJoystick() {
    // Only show on mobile
    if (!this.platform.isMobile) return;

    // Create joystick element
    const joystick = document.createElement('div');
    joystick.className = 'virtual-joystick';
    joystick.innerHTML = `
      <div class="joystick-base">
        <div class="joystick-stick"></div>
      </div>
    `;

    document.body.appendChild(joystick);

    // Add touch handling
    this.setupJoystickTouch(joystick);
  }

  setupJoystickTouch(joystick) {
    const base = joystick.querySelector('.joystick-base');
    const stick = joystick.querySelector('.joystick-stick');

    let touchId = null;
    let centerX = 0;
    let centerY = 0;

    joystick.addEventListener('touchstart', (e) => {
      e.preventDefault();
      touchId = e.changedTouches[0].identifier;

      const rect = base.getBoundingClientRect();
      centerX = rect.left + rect.width / 2;
      centerY = rect.top + rect.height / 2;

      this.updateStick(stick, centerX, centerY, centerX, centerY);
    });

    joystick.addEventListener('touchmove', (e) => {
      e.preventDefault();

      for (const touch of e.changedTouches) {
        if (touch.identifier === touchId) {
          const dx = touch.clientX - centerX;
          const dy = touch.clientY - centerY;

          // Clamp to joystick radius
          const distance = Math.sqrt(dx * dx + dy * dy);
          const maxDistance = 40;

          let clampedX = dx;
          let clampedY = dy;

          if (distance > maxDistance) {
            const angle = Math.atan2(dy, dx);
            clampedX = Math.cos(angle) * maxDistance;
            clampedY = Math.sin(angle) * maxDistance;
          }

          this.updateStick(stick, centerX, centerY, centerX + clampedX, centerY + clampedY);

          // Normalize to -1 to 1
          this.axes.moveX = clampedX / maxDistance;
          this.axes.moveY = -clampedY / maxDistance;
        }
      }
    });

    joystick.addEventListener('touchend', (e) => {
      for (const touch of e.changedTouches) {
        if (touch.identifier === touchId) {
          touchId = null;
          this.updateStick(stick, centerX, centerY, centerX, centerY);
          this.axes.moveX = 0;
          this.axes.moveY = 0;
        }
      }
    });
  }

  updateStick(stick, baseX, baseY, stickX, stickY) {
    stick.style.transform = `translate(${stickX - baseX}px, ${stickY - baseY}px)`;
  }
}
```

---

## Part 9: Action Query API

### Public API for Game Code

```javascript
class InputManager {
  // Check if action is currently pressed
  isActionPressed(action) {
    return this.actions[action] === true;
  }

  // Check if action was just pressed this frame
  isActionJustPressed(action) {
    return this.actions[action] === true && this.previousActions[action] === false;
  }

  // Check if action was just released this frame
  isActionJustReleased(action) {
    return this.actions[action] === false && this.previousActions[action] === true;
  }

  // Get analog axis value
  getAxis(axis) {
    return this.axes[axis] || 0;
  }

  // Update (called once per frame)
  update(deltaTime) {
    // Store current state as previous
    this.previousActions = { ...this.actions };

    // Poll gamepads
    this.pollGamepads();

    // Decay momentary actions
    this.decayActions();
  }

  decayActions() {
    // Reset "just pressed" actions
    this.actions.interact = false;
    this.actions.jump = false;
    this.actions.menu = false;
  }
}
```

### Usage Examples

```javascript
// In CharacterController
class CharacterController {
  update(deltaTime) {
    const input = this.inputManager;

    // Movement (continuous)
    const forward = input.isActionPressed('moveForward') ? 1 : 0;
    const backward = input.isActionPressed('moveBackward') ? 1 : 0;
    const left = input.isActionPressed('moveLeft') ? 1 : 0;
    const right = input.isActionPressed('moveRight') ? 1 : 0;

    // Or use analog values directly
    const moveX = input.getAxis('moveX');
    const moveY = input.getAxis('moveY');

    // Interaction (momentary)
    if (input.isActionJustPressed('interact')) {
      this.tryInteract();
    }
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Not Using Delta Time

```javascript
// ❌ WRONG: Frame-rate dependent
position.x += input.getAxis('moveX') * speed;

// ✅ CORRECT: Frame-rate independent
position.x += input.getAxis('moveX') * speed * deltaTime;
```

### 2. Forgetting to Update Previous State

```javascript
// ❌ WRONG: "Just pressed" never works
if (this.actions.interact) {
  doSomething();
}

// ✅ CORRECT: Track previous state
if (this.actions.interact && !this.previousActions.interact) {
  doSomething();
}

// Or use helper
if (this.isActionJustPressed('interact')) {
  doSomething();
}
```

### 3. Not Applying Deadzone

```javascript
// ❌ WRONG: Drift when thumbstick at rest
const moveX = gamepad.axes[0];
position.x += moveX * speed;

// ✅ CORRECT: Apply deadzone
const moveX = gamepad.axes[0];
const deadzone = 0.15;
const clampedX = Math.abs(moveX) < deadzone ? 0 : moveX;
position.x += clampedX * speed;
```

### 4. Forgetting PreventDefault

```javascript
// ❌ WRONG: Arrow keys scroll the page
window.addEventListener('keydown', (e) => {
  if (e.code === 'ArrowDown') {
    player.moveBackward();
  }
  // Page scrolls!
});

// ✅ CORRECT: Prevent default
window.addEventListener('keydown', (e) => {
  if (e.code === 'ArrowDown') {
    player.moveBackward();
    e.preventDefault();  // Stop scrolling
  }
});
```

### 5. Not Handling Multiple Touches

```javascript
// ❌ WRONG: Only tracks first touch
document.addEventListener('touchmove', (e) => {
  const touch = e.touches[0];
  handleTouch(touch.clientX, touch.clientY);
});

// ✅ CORRECT: Track all touches
document.addEventListener('touchmove', (e) => {
  for (const touch of e.touches) {
    handleTouch(touch.identifier, touch.clientX, touch.clientY);
  }
});
```

---

## Next Steps

Now that you understand InputManager:

- [CharacterController](./character-controller.md) - How input drives player movement
- [PhysicsManager](./physics-manager.md) - How physics simulation works
- [ColliderManager](./collider-manager.md) - Trigger zones and collisions
- [UIManager](../10-user-interface/ui-manager.md) - UI input handling
- [Touch Joystick](../10-user-interface/touch-joystick.md) - Virtual controls

---

## Source File Reference

- **Location:** `src/managers/InputManager.js`
- **Key exports:** `InputManager` class
- **Dependencies:** None (uses browser APIs)
- **Used by:** CharacterController, UIManager, any interactive system

---

## References

- [Gamepad API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Gamepad_API) - Gamepad API (verified January 2026)
- [Pointer Lock API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Pointer_Lock_API) - Pointer Lock API
- [KeyboardEvent - MDN](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent) - Keyboard events
- [MouseEvent - MDN](https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent) - Mouse events
- [TouchEvent - MDN](https://developer.mozilla.org/en-US/docs/Web/API/TouchEvent) - Touch events
- [Three.js Controls](https://threejs.org/docs/#examples/en/controls/FirstPersonControls) - FirstPersonControls reference

*Documentation last updated: January 12, 2026*
