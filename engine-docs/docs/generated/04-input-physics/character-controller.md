# CharacterController - First Principles Guide

## Overview

The **CharacterController** handles first-person player movement in the Shadow Engine. It translates input from the InputManager into actual movement through the 3D world - including walking, looking around, crouching, and interacting with objects.

Think of CharacterController as the **player's physical presence** in the game world - it's where "press W" becomes "move forward through 3D space."

## What You Need to Know First

Before understanding CharacterController, you should know:
- **3D coordinates** - X (left/right), Y (up/down), Z (forward/backward)
- **Vectors** - Direction and magnitude in 3D space
- **Quaternions** - How rotation is represented in 3D
- **InputManager** - How input is captured (see [InputManager guide](./input-manager.md))
- **Delta time** - Time between frames for frame-rate independent movement
- **Collision detection** - Preventing movement through walls

### Quick Refresher: 3D Movement Basics

```javascript
// A 3D position
const position = { x: 0, y: 1.7, z: 0 };  // 1.7m tall (eye level)

// A 3D direction (normalized = length of 1)
const forward = { x: 0, y: 0, z: -1 };    // Looking towards -Z
const right = { x: 1, y: 0, z: 0 };       // Right direction

// Movement: position + direction × speed × time
position.x += forward.x * speed * deltaTime;
position.y += forward.y * speed * deltaTime;
position.z += forward.z * speed * deltaTime;
```

---

## Part 1: How First-Person Movement Works

### The Basic Concept

First-person movement consists of two independent parts:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FIRST-PERSON MOVEMENT                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  POSITION MOVEMENT              LOOK ROTATION                  │
│  ────────────────               ──────────────                  │
│  W/A/S/D keys         Mouse movement /                         │
│  Gamepad stick        Right stick                              │
│  Touch swipe                                                  │
│       │                              │                         │
│       ▼                              ▼                         │
│  Move through 3D space      Rotate camera/player               │
│  (X, Y, Z position)         (yaw, pitch angles)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** Movement and rotation are independent - you can move forward while looking up, or strafe left while looking right.

### The Player Object

The player in Shadow Engine consists of:

```javascript
{
  // Position in world space (meters)
  position: new THREE.Vector3(0, 1.7, 0),

  // Where we're looking (degrees)
  rotation: {
    yaw: 0,      // Left/right rotation (around Y axis)
    pitch: 0     // Up/down rotation (around X axis)
  },

  // Current speed
  velocity: new THREE.Vector3(0, 0, 0),

  // Movement state
  isGrounded: true,
  isCrouching: false,
  isRunning: false
}
```

---

## Part 2: CharacterController Structure

### Class Overview

```javascript
class CharacterController {
  constructor() {
    // Dependencies
    this.inputManager = null;  // For reading input
    this.sceneManager = null;  // For camera attachment
    this.physicsManager = null; // For collision detection

    // Player transform
    this.position = new THREE.Vector3(0, 1.7, 0);  // Eye level
    this.yaw = 0;    // Horizontal rotation (radians)
    this.pitch = 0;  // Vertical rotation (radians)

    // Movement settings
    this.settings = {
      walkSpeed: 2.5,      // m/s (normal walking)
      runSpeed: 5.0,       // m/s (sprinting)
      crouchSpeed: 1.0,    // m/s (crouching)
      lookSensitivity: 0.002,
      jumpForce: 5.0,      // m/s upward velocity
      gravity: 9.81        // m/s²
    };

    // Current state
    this.velocity = new THREE.Vector3();
    this.isGrounded = true;
    this.isCrouching = false;
    this.isRunning = false;
    this.isEnabled = false;

    // Camera (attached to player)
    this.camera = null;

    // Collision settings
    this.playerHeight = 1.8;     // Standing height
    this.crouchHeight = 1.2;     // Crouching height
    this.playerRadius = 0.3;     // Collision capsule radius
  }
}
```

---

## Part 3: Reading Input for Movement

### Movement Input

CharacterController reads actions from InputManager:

```javascript
update(deltaTime) {
  if (!this.isEnabled) return;

  // Get movement input from InputManager
  const input = this.inputManager;

  // Digital movement (keyboard)
  const forward = input.isActionPressed('moveForward') ? 1 : 0;
  const backward = input.isActionPressed('moveBackward') ? 1 : 0;
  const left = input.isActionPressed('moveLeft') ? 1 : 0;
  const right = input.isActionPressed('moveRight') ? 1 : 0;

  // Analog movement (gamepad stick)
  const moveX = input.getAxis('moveX');   // -1 (left) to 1 (right)
  const moveY = input.getAxis('moveY');   // -1 (down) to 1 (up)

  // Combine: analog overrides digital
  const moveDirection = new THREE.Vector3(
    moveX !== 0 ? moveX : right - left,
    0,
    moveY !== 0 ? moveY : backward - forward
  );

  // Check for running/crouching
  this.isRunning = input.isActionPressed('sprint');
  this.isCrouching = input.isActionPressed('crouch');

  // Jump input (momentary - just pressed)
  if (input.isActionJustPressed('jump') && this.isGrounded) {
    this.jump();
  }
}
```

### Look Input

```javascript
updateLookRotation(deltaTime) {
  const input = this.inputManager;

  // Get look input from mouse or gamepad
  const lookX = input.getAxis('lookX');
  const lookY = input.getAxis('lookY');

  // Apply rotation with sensitivity
  this.yaw -= lookX * this.settings.lookSensitivity;
  this.pitch -= lookY * this.settings.lookSensitivity;

  // Clamp pitch to prevent over-rotation (looking too far up/down)
  const maxPitch = Math.PI / 2 - 0.01;  // ~89 degrees
  this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch));

  // Apply rotation to camera
  this.updateCameraRotation();
}
```

---

## Part 4: Calculating Movement

### From Local to World Space

Input is in "local" space (relative to where we're looking), but movement happens in "world" space:

```javascript
calculateMovement(deltaTime) {
  const input = this.inputManager;

  // Get input values
  let moveX = input.getAxis('moveX');
  let moveY = input.getAxis('moveY');

  // Fallback to digital input if analog is 0
  if (moveX === 0) {
    moveX = (input.isActionPressed('moveRight') ? 1 : 0) -
            (input.isActionPressed('moveLeft') ? 1 : 0);
  }
  if (moveY === 0) {
    moveY = (input.isActionPressed('moveBackward') ? 1 : 0) -
            (input.isActionPressed('moveForward') ? 1 : 0);
  }

  // Calculate direction vectors based on yaw (horizontal rotation)
  const forward = new THREE.Vector3(
    Math.sin(this.yaw),
    0,
    Math.cos(this.yaw)
  );

  const right = new THREE.Vector3(
    Math.sin(this.yaw + Math.PI / 2),
    0,
    Math.cos(this.yaw + Math.PI / 2)
  );

  // Combine input with direction vectors
  const moveDirection = new THREE.Vector3();
  moveDirection.addScaledVector(right, moveX);
  moveDirection.addScaledVector(forward, moveY);

  // Normalize (so diagonal movement isn't faster)
  if (moveDirection.length() > 1) {
    moveDirection.normalize();
  }

  // Apply speed
  let speed = this.settings.walkSpeed;
  if (this.isRunning) speed = this.settings.runSpeed;
  if (this.isCrouching) speed = this.settings.crouchSpeed;

  // Calculate velocity
  this.velocity.x = moveDirection.x * speed;
  this.velocity.z = moveDirection.z * speed;

  // Vertical movement (gravity/jump) handled separately
}
```

### Applying Movement

```javascript
applyMovement(deltaTime) {
  if (!this.isEnabled) return;

  // Apply horizontal velocity
  this.position.x += this.velocity.x * deltaTime;
  this.position.z += this.velocity.z * deltaTime;

  // Apply gravity if not grounded
  if (!this.isGrounded) {
    this.velocity.y -= this.settings.gravity * deltaTime;
    this.position.y += this.velocity.y * deltaTime;
  }

  // Update camera position
  if (this.camera) {
    this.camera.position.copy(this.position);

    // Crouching: lower camera
    if (this.isCrouching) {
      this.camera.position.y -= (this.playerHeight - this.crouchHeight);
    }
  }
}
```

---

## Part 5: Camera Control

### First-Person Camera

The camera follows the player's position and rotation:

```javascript
updateCameraRotation() {
  if (!this.camera) return;

  // Create rotation from yaw and pitch
  const quaternion = new THREE.Quaternion();
  quaternion.setFromEuler(new THREE.Euler(
    this.pitch,  // X axis rotation (look up/down)
    this.yaw,    // Y axis rotation (look left/right)
    0,          // Z axis rotation (roll - usually 0 for FPS)
    'YXZ'       // Rotation order
  ));

  // Apply to camera
  this.camera.quaternion.copy(quaternion);
}
```

### Attaching Camera

```javascript
attachCamera(camera) {
  this.camera = camera;

  // Set initial position
  camera.position.copy(this.position);

  // Set initial rotation
  this.updateCameraRotation();

  console.log('[CharacterController] Camera attached at player position');
}
```

---

## Part 6: Collision Detection

### Ground Check

The simplest collision is checking if we're on the ground:

```javascript
checkGrounded() {
  // Simple height check (no physics engine)
  const groundLevel = 0;  // Assuming Y=0 is ground

  // Raycast downward from player position
  const rayOrigin = this.position.clone();
  const rayDirection = new THREE.Vector3(0, -1, 0);

  const raycaster = new THREE.Raycaster(
    rayOrigin,
    rayDirection,
    0,
    this.playerHeight * 1.1  // Check slightly below feet
  );

  // Check against scene objects
  const intersects = raycaster.intersectObjects(
    this.sceneManager.getCollidableObjects(),
    true
  );

  this.isGrounded = intersects.length > 0;

  if (this.isGrounded) {
    // Reset vertical velocity when on ground
    this.velocity.y = 0;

    // Snap to ground if needed
    if (this.position.y < groundLevel) {
      this.position.y = groundLevel;
    }
  }
}
```

### Wall Collision

```javascript
checkWallCollision() {
  // Create a capsule for the player
  const capsule = {
    radius: this.playerRadius,
    height: this.isCrouching ? this.crouchHeight : this.playerHeight,
    position: this.position
  };

  // Check against nearby colliders
  const colliders = this.sceneManager.getNearbyColliders(this.position);

  for (const collider of colliders) {
    if (this.intersectsCapsule(capsule, collider)) {
      // Push player out of collision
      this.resolveCollision(capsule, collider);
    }
  }
}
```

---

## Part 7: Jumping and Crouching

### Jumping

```javascript
jump() {
  if (!this.isGrounded) return;  // Can't jump while in air

  // Apply upward velocity
  this.velocity.y = this.settings.jumpForce;

  this.isGrounded = false;

  console.log('[CharacterController] Jumped');
}
```

**Jump physics:**
1. Initial upward velocity (`jumpForce`)
2. Gravity reduces velocity each frame
3. When velocity becomes negative, player falls
4. Ground collision stops the fall

### Crouching

```javascript
toggleCrouch(forceState = null) {
  if (forceState !== null) {
    this.isCrouching = forceState;
  } else {
    this.isCrouching = !this.isCrouching;
  }

  if (this.isCrouching) {
    // Lower camera and reduce speed
    this.settings.currentSpeed = this.settings.crouchSpeed;
  } else {
    // Return to normal height
    this.settings.currentSpeed = this.settings.walkSpeed;
  }
}
```

---

## Part 8: Enabling/Disabling Control

The game needs to disable player control during cutscenes, dialog, etc.:

```javascript
enable() {
  this.isEnabled = true;

  // Request pointer lock for mouse look
  document.body.requestPointerLock();

  // Notify GameManager
  this.gameManager.emit('character-controller:enabled');

  console.log('[CharacterController] Controls enabled');
}

disable() {
  this.isEnabled = false;

  // Release pointer lock
  document.exitPointerLock();

  // Reset input state
  this.velocity.set(0, 0, 0);

  // Notify GameManager
  this.gameManager.emit('character-controller:disabled');

  console.log('[CharacterController] Controls disabled');
}

setEnabled(enabled) {
  if (enabled) {
    this.enable();
  } else {
    this.disable();
  }
}
```

---

## Part 9: Integration with GameManager

CharacterController responds to game state changes:

```javascript
initialize(gameManager, inputManager, sceneManager) {
  this.gameManager = gameManager;
  this.inputManager = inputManager;
  this.sceneManager = sceneManager;

  // Listen for state changes
  gameManager.on('state:changed', this.onStateChanged.bind(this));

  // Listen for control enable/disable
  gameManager.on('character-controller:enable', () => this.enable());
  gameManager.on('character-controller:disable', () => this.disable());

  // Subscribe to update loop
  gameManager.on('update', this.update.bind(this));
}

onStateChanged(newState, oldState) {
  // Enable/disable controls based on state
  const shouldEnable = newState.controlEnabled === true;

  if (shouldEnable && !this.isEnabled) {
    this.enable();
  } else if (!shouldEnable && this.isEnabled) {
    this.disable();
  }
}
```

---

## Part 10: Complete Update Loop

```javascript
update(deltaTime) {
  if (!this.isEnabled) return;

  // 1. Process look input (mouse/gamepad)
  this.updateLookRotation(deltaTime);

  // 2. Process movement input (keyboard/gamepad)
  this.calculateMovement(deltaTime);

  // 3. Apply gravity
  if (!this.isGrounded) {
    this.velocity.y -= this.settings.gravity * deltaTime;
  }

  // 4. Check collisions
  this.checkGrounded();
  this.checkWallCollision();

  // 5. Apply movement
  this.applyMovement(deltaTime);

  // 6. Update camera
  if (this.camera) {
    this.camera.position.copy(this.position);

    // Adjust for crouching
    const eyeHeight = this.isCrouching
      ? this.crouchHeight
      : this.playerHeight;

    this.camera.position.y = this.position.y + eyeHeight;
    this.updateCameraRotation();
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Not Using Delta Time

```javascript
// ❌ WRONG: Frame-rate dependent
position.x += speed;

// ✅ CORRECT: Frame-rate independent
position.x += speed * deltaTime;
```

### 2. Forgetting to Normalize Diagonal Movement

```javascript
// ❌ WRONG: Diagonal is 1.4x faster
if (forward && right) {
  velocity.x = 1;
  velocity.z = 1;
}

// ✅ CORRECT: Normalize for consistent speed
velocity.set(forward, 0, right);
if (velocity.length() > 1) velocity.normalize();
velocity.multiplyScalar(speed);
```

### 3. Not Clamping Pitch

```javascript
// ❌ WRONG: Can look completely upside down
this.pitch += lookY;

// ✅ CORRECT: Clamp to ~89 degrees
const maxPitch = Math.PI / 2 - 0.01;
this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch));
```

### 4. Mixing Up Local and World Space

```javascript
// ❌ WRONG: Moving in world Z instead of forward direction
position.z += input.getAxis('moveY') * speed;

// ✅ CORRECT: Rotate input by player's yaw
const forward = new THREE.Vector3(
  Math.sin(this.yaw), 0, Math.cos(this.yaw)
);
position.addScaledVector(forward, input.getAxis('moveY') * speed);
```

### 5. Forgetting to Reset Velocity

```javascript
// ❌ WRONG: Velocity accumulates forever
if (input.isActionPressed('moveForward')) {
  velocity.z = -speed;
}

// ✅ CORRECT: Recalculate velocity each frame
velocity.set(0, 0, 0);
if (input.isActionPressed('moveForward')) {
  velocity.z = -speed;
}
```

---

## Part 11: Settings and Tuning

### Movement Speed Reference

| Action | Speed (m/s) | Speed (ft/s) | Notes |
|--------|-------------|--------------|-------|
| Crouch walk | 1.0 | 3.3 | Slow, stealthy |
| Normal walk | 2.5 | 8.2 | Comfortable walking |
| Run/Sprint | 5.0 | 16.4 | Jogging speed |
| Jump velocity | 5.0 m/s | - | Launch speed |

### Camera Sensitivity

| Setting | Value | Notes |
|---------|-------|-------|
| Mouse look | 0.002 | Standard FPS sensitivity |
| Gamepad look | 1.0-2.0 | Higher for stick input |
| Touch look | 0.005 | Swipe-based look |

### Player Dimensions

```javascript
{
  playerHeight: 1.8,      // 6 feet tall
  crouchHeight: 1.2,      // 4 feet when crouching
  playerRadius: 0.3,      // Shoulder width / 2
  eyeHeight: 1.7          // Camera height when standing
}
```

---

## Part 12: Advanced Features

### Head Bobbing

Simulate walking head motion:

```javascript
updateHeadBob(deltaTime) {
  if (!this.isGrounded) return;

  // Only bob when moving
  const isMoving = this.velocity.length() > 0.1;
  if (!isMoving) {
    this.bobTime = 0;
    return;
  }

  // Accumulate time based on speed
  this.bobTime += deltaTime * this.settings.bobFrequency;

  // Calculate bob offset
  const bobAmount = Math.sin(this.bobTime) * this.settings.bobAmount;

  // Apply to camera Y position
  this.camera.position.y += bobAmount;
}
```

### Footstep Sounds

```javascript
updateFootsteps(deltaTime) {
  if (!this.isGrounded) return;

  const speed = this.velocity.length();
  if (speed < 0.1) return;

  // Time between steps based on speed
  const stepInterval = this.settings.stepDistance / speed;

  this.stepTimer += deltaTime;

  if (this.stepTimer >= stepInterval) {
    this.stepTimer = 0;
    this.playFootstepSound();
  }
}

playFootstepSound() {
  // Determine surface type from raycast
  const surface = this.getSurfaceType();

  // Play appropriate sound
  this.gameManager.sfxManager.playFootstep(surface);
}
```

### Interaction Raycast

```javascript
getInteractable() {
  // Raycast forward from camera
  const ray = new THREE.Raycaster(
    this.camera.position,
    this.camera.getWorldDirection(new THREE.Vector3()),
    0,
    3.0  // 3 meter interaction range
  );

  const hits = ray.intersectObjects(
    this.sceneManager.getInteractableObjects(),
    true
  );

  if (hits.length > 0) {
    return hits[0].object;
  }

  return null;
}
```

---

## Next Steps

Now that you understand CharacterController:

- [PhysicsManager](./physics-manager.md) - Rapier physics integration
- [ColliderManager](./collider-manager.md) - Trigger zones and collisions
- [InputManager](./input-manager.md) - How input is captured
- [Interaction System](../08-interactive-objects/interaction.md) - Object interaction

---

## Source File Reference

- **Location:** `src/managers/CharacterController.js`
- **Key exports:** `CharacterController` class
- **Dependencies:** Three.js (r180+), InputManager
- **Used by:** GameManager, interaction systems

---

## References

- [Three.js Math](https://threejs.org/docs/#api/en/math/Vector3) - Vector3 math reference
- [First-Person Controls](https://threejs.org/docs/#examples/en/controls/FirstPersonControls) - Three.js first-person controls
- [Pointer Lock API](https://developer.mozilla.org/en-US/docs/Web/API/Pointer_Lock_API) - MDN pointer lock docs
- [Euler Angles](https://en.wikipedia.org/wiki/Euler_angles) - Rotation math
- [Quaternions](https://threejs.org/docs/#api/en/math/Quaternion) - Quaternion rotation

*Documentation last updated: January 12, 2026*
