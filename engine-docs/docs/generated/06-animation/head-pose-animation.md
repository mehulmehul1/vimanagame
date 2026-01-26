# Head-Pose Animation System - First Principles Guide

## Overview

The **Head-Pose Animation System** tracks and responds to the player's head orientation to create immersive, responsive character animations. This system enables VR-style head tracking for desktop (using mouse) and mobile (using device orientation), allowing NPCs to look at the player, triggering effects when the player stares at certain objects, and creating the infamous "Viewmaster insanity" mechanic.

Think of head-pose animation as the **eye contact system** for your game world - just as real people notice when someone is staring at them, head-pose tracking lets your game know WHERE the player is looking and respond accordingly.

## What You Need to Know First

Before understanding head-pose animation, you should know:
- **Euler angles** - Rotation around X, Y, Z axes (pitch, yaw, roll)
- **DeviceOrientation API** - Browser API for mobile device orientation
- **Raycasting** - Shooting a ray from camera to find what player looks at
- **Look-at quaternion math** - Making objects face a target
- **Animation blending** - Smoothly transitioning between poses

### Quick Refresher: Head Pose Explained

```
HEAD POSE: Where the player is looking

         Looking Direction
              ↓
              │
              │
    ╱─────────┴─────────╲
   │   👤 Player         │
   │   (Camera)          │
    ╲─────────────────────╲

    HEAD ROTATION (Euler Angles):

    PITCH (X-axis):  ↑  Looking up / down  ↓
                    +30°         -30°

    YAW (Y-axis):    ←  Looking left / right →
                   -45°         +45°

    ROLL (Z-axis):    ↺  Tilting head  ↻
                    (rarely used in FPS)

    WHAT WE TRACK:
    1. Rotation angles (where looking)
    2. Look target (what looking at)
    3. Duration (how long looking)
    4. Changes (sudden movements)
```

---

## Part 1: Why Use Head-Pose Animation?

### The Problem: Static Characters Feel Lifeless

Without head-pose tracking:

```javascript
// ❌ WITHOUT Head-Pose - Characters never acknowledge player
function updateNPC(npc) {
  npc.playAnimation("idle");  // Always same animation
  // Character stares blankly ahead, ignoring player
}
```

**Problems:**
- Characters feel like robots
- No player presence acknowledgment
- Missed opportunities for interactive storytelling
- Can't detect "staring" for game mechanics

### The Solution: Head-Pose Animation

```javascript
// ✅ WITH Head-Pose - Characters react to player attention
function updateNPC(npc) {
  const playerGaze = headPoseSystem.getGazeDirection();
  const isLookingAtMe = headPoseSystem.isLookingAt(npc.position);

  if (isLookingAtMe) {
    // Make eye contact!
    npc.head.lookAt(camera.position);
    npc.playAttentionReaction();
  } else {
    // Casual glance
    npc.head.lookAt(playerGaze.origin, 0.3);  // Partial turn
  }
}
```

**Benefits:**
- Characters feel alive and aware
- Player feels seen and acknowledged
- Enables "staring" mechanics for puzzles/horror
- Creates more immersive social interactions

---

## Part 2: Head Pose Detection

### Desktop (Mouse-Based) Head Tracking

On desktop, we approximate head direction from camera rotation (controlled by mouse look):

```javascript
class DesktopHeadPoseTracker {
  constructor(camera) {
    this.camera = camera;
    this.gazeDirection = new THREE.Vector3();
    this.gazeOrigin = new THREE.Vector3();
  }

  update() {
    // Get forward direction from camera rotation
    this.camera.getWorldDirection(this.gazeDirection);

    // Gaze origin is camera position
    this.gazeOrigin.copy(this.camera.position);

    return {
      origin: this.gazeOrigin,
      direction: this.gazeDirection,
      pitch: this.camera.rotation.x,  // X-axis rotation
      yaw: this.camera.rotation.y      // Y-axis rotation
    };
  }
}
```

### Mobile (DeviceOrientation) Head Tracking

On mobile, we use the DeviceOrientation API for physical device rotation:

```javascript
class MobileHeadPoseTracker {
  constructor() {
    this.orientation = { alpha: 0, beta: 0, gamma: 0 };
    this.hasPermission = false;

    // Request permission (required on iOS 13+)
    this.requestPermission();
  }

  async requestPermission() {
    if (typeof DeviceOrientationEvent !== 'undefined' &&
        typeof DeviceOrientationEvent.requestPermission === 'function') {
      try {
        const permission = await DeviceOrientationEvent.requestPermission();
        if (permission === 'granted') {
          this.hasPermission = true;
          this.startListening();
        }
      } catch (error) {
        console.error('Device orientation permission denied:', error);
      }
    } else {
      // Non-iOS or older devices don't need permission
      this.hasPermission = true;
      this.startListening();
    }
  }

  startListening() {
    window.addEventListener('deviceorientation', (event) => {
      this.orientation = {
        alpha: event.alpha,  // Z-axis rotation (0-360°)
        beta: event.beta,    // X-axis rotation (-180 to 180°)
        gamma: event.gamma   // Y-axis rotation (-90 to 90°)
      };
    });
  }

  getHeadPose() {
    // Convert device orientation to look direction
    const pitch = this.orientation.beta * (Math.PI / 180);   // X
    const yaw = this.orientation.alpha * (Math.PI / 180);     // Y
    const roll = this.orientation.gamma * (Math.PI / 180);    // Z

    // Create direction vector from Euler angles
    const direction = new THREE.Vector3(0, 0, -1);
    const euler = new THREE.Euler(pitch, yaw, roll, 'YXZ');
    direction.applyEuler(euler);

    return {
      origin: this.camera.position,
      direction: direction,
      pitch: pitch,
      yaw: yaw,
      roll: roll,
      raw: this.orientation
    };
  }
}
```

### Platform Detection

```javascript
class HeadPoseSystem {
  constructor(camera) {
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

    if (isMobile) {
      this.tracker = new MobileHeadPoseTracker(camera);
    } else {
      this.tracker = new DesktopHeadPoseTracker(camera);
    }
  }

  getCurrentPose() {
    return this.tracker.getHeadPose();
  }
}
```

---

## Part 3: Raycasting for Look Target Detection

### Finding What Player Is Looking At

```javascript
class GazeTargetDetector {
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;
    this.raycaster = new THREE.Raycaster();
    this.gazeTarget = null;
    this.gazeStartTime = 0;
    this.gazeDuration = 0;
  }

  update() {
    // Get current gaze direction
    const headPose = headPoseSystem.getCurrentPose();

    // Set up raycaster
    this.raycaster.set(
      headPose.origin,
      headPose.direction
    );

    // Check for intersections
    const interactableObjects = this.scene.getInteractableObjects();
    const intersects = this.raycaster.intersectObjects(interactableObjects);

    if (intersects.length > 0) {
      const newTarget = intersects[0].object;

      if (newTarget !== this.gazeTarget) {
        // New target!
        this.gazeTarget = newTarget;
        this.gazeStartTime = performance.now();
        this.emit('gaze:start', { target: newTarget });
      }

      // Update duration
      this.gazeDuration = performance.now() - this.gazeStartTime;

      return {
        target: this.gazeTarget,
        point: intersects[0].point,
        distance: intersects[0].distance,
        duration: this.gazeDuration
      };
    } else {
      // Not looking at anything
      if (this.gazeTarget) {
        this.emit('gaze:end', {
          target: this.gazeTarget,
          totalDuration: this.gazeDuration
        });
        this.gazeTarget = null;
        this.gazeDuration = 0;
      }

      return {
        target: null,
        duration: 0
      };
    }
  }

  isLookingAt(objectOrPosition, maxDuration = Infinity) {
    if (!this.gazeTarget) return false;

    // Check if looking at specific object
    if (objectOrPosition instanceof THREE.Object3D) {
      return this.gazeTarget === objectOrPosition;
    }

    // Check if looking at position (within radius)
    if (objectOrPosition instanceof THREE.Vector3) {
      const gazePoint = this.getCurrentGazePoint();
      return gazePoint.distanceTo(objectOrPosition) < 0.5;
    }

    return false;
  }

  getGazeDuration() {
    return this.gazeDuration;
  }
}
```

---

## Part 4: Character Look-At Animations

### Making Characters Look at Player

```javascript
class CharacterLookAtController {
  constructor(characterMesh) {
    this.character = characterMesh;
    this.headBone = this.findHeadBone();
    this.neckBone = this.findNeckBone();
    this.targetRotation = new THREE.Euler();
    this.currentRotation = new THREE.Euler();
    this.maxTurnAngle = Math.PI / 4;  // 45 degrees
  }

  findHeadBone() {
    return this.character.getObjectByName('head') ||
           this.character.getObjectByName('mixamorigHead');
  }

  lookAt(targetPosition, intensity = 1.0) {
    if (!this.headBone) return;

    // Calculate direction to target
    const direction = new THREE.Vector3()
      .subVectors(targetPosition, this.headBone.position)
      .normalize();

    // Calculate target rotation
    const targetQuaternion = new THREE.Quaternion();
    const matrix = new THREE.Matrix4();
    matrix.lookAt(
      this.headBone.position,
      targetPosition,
      new THREE.Vector3(0, 1, 0)
    );
    targetQuaternion.setFromRotationMatrix(matrix);

    // Clamp rotation to realistic neck movement
    const maxAngle = this.maxTurnAngle * intensity;
    this.clampRotation(targetQuaternion, maxAngle);

    this.targetRotation.setFromQuaternion(targetQuaternion);
  }

  clampRotation(quaternion, maxAngle) {
    const currentEuler = new THREE.Euler().setFromQuaternion(
      this.headBone.quaternion
    );
    const targetEuler = new THREE.Euler().setFromQuaternion(quaternion);

    // Limit pitch (X) and yaw (Y)
    targetEuler.x = THREE.MathUtils.clamp(
      targetEuler.x,
      currentEuler.x - maxAngle,
      currentEuler.x + maxAngle
    );
    targetEuler.y = THREE.MathUtils.clamp(
      targetEuler.y,
      currentEuler.y - maxAngle,
      currentEuler.y + maxAngle
    );
    // Keep roll (Z) at 0 typically

    quaternion.setFromEuler(targetEuler);
  }

  update(deltaTime) {
    if (!this.headBone) return;

    // Smoothly interpolate to target rotation
    const speed = 5.0;  // Look speed
    this.currentRotation.x = THREE.MathUtils.lerp(
      this.currentRotation.x,
      this.targetRotation.x,
      speed * deltaTime
    );
    this.currentRotation.y = THREE.MathUtils.lerp(
      this.currentRotation.y,
      this.targetRotation.y,
      speed * deltaTime
    );
    this.currentRotation.z = THREE.MathUtils.lerp(
      this.currentRotation.z,
      this.targetRotation.z,
      speed * deltaTime
    );

    // Apply to head bone
    this.headBone.rotation.copy(this.currentRotation);
  }

  reset() {
    this.lookAt(this.getForwardPosition(), 0);
  }

  getForwardPosition() {
    const forward = new THREE.Vector3(0, 0, 1);
    forward.applyQuaternion(this.character.quaternion);
    forward.add(this.character.position);
    return forward;
  }
}
```

### Hierarchical Look-At (Eyes → Head → Torso)

```javascript
class HierarchicalLookAt {
  constructor(character) {
    this.character = character;
    this.eyes = character.getObjectByName('eyes');
    this.head = character.getObjectByName('head');
    this.spine = character.getObjectByName('spine');
  }

  lookAt(targetPosition) {
    // Eyes have fastest, widest movement
    if (this.eyes) {
      this.rotateTowards(this.eyes, targetPosition, 0.8, Math.PI / 3);
    }

    // Head moves more slowly, limited range
    if (this.head) {
      this.rotateTowards(this.head, targetPosition, 0.5, Math.PI / 6);
    }

    // Torso moves slowest, most limited
    if (this.spine) {
      this.rotateTowards(this.spine, targetPosition, 0.3, Math.PI / 8);
    }
  }

  rotateTowards(bone, target, intensity, maxAngle) {
    const direction = new THREE.Vector3()
      .subVectors(target, bone.position)
      .normalize();

    const targetRotation = new THREE.Quaternion();
    const up = new THREE.Vector3(0, 1, 0);

    const lookMatrix = new THREE.Matrix4();
    lookMatrix.lookAt(bone.position, target, up);
    targetRotation.setFromRotationMatrix(lookMatrix);

    // Apply with intensity and clamp
    bone.quaternion.slerp(targetRotation, intensity);
  }
}
```

---

## Part 5: Viewmaster Insanity System

### The Viewmaster Mechanic

The Viewmaster is a special item that creates visual distortion based on how long the player stares through it:

```javascript
class ViewmasterInsanitySystem {
  constructor() {
    this.isActive = false;
    this.intensity = 0.0;  // 0.0 to 1.0
    this.gazeStartTime = 0;
    this.overheatCount = 0;
    this.maxOverheat = 3;
  }

  equip() {
    this.isActive = true;
    this.intensity = 0.0;
    this.gazeStartTime = performance.now();

    // Apply initial visual effects
    vfxManager.enable('viewmaster-vignette');
  }

  unequip() {
    this.isActive = false;
    this.intensity = 0.0;

    // Remove effects
    vfxManager.disable('viewmaster-vignette');
    vfxManager.disable('viewmaster-glitch');
    vfxManager.disable('viewmaster-desaturate');
  }

  update(currentTime) {
    if (!this.isActive) return;

    // Check if player is "looking through" viewmaster
    const isLookingThrough = this.isPlayerLookingThrough();

    if (isLookingThrough) {
      // Increase insanity over time
      const gazeDuration = (currentTime - this.gazeStartTime) / 1000;
      this.intensity = Math.min(1.0, gazeDuration / 30);  // 30s to max

      // Apply progressive effects
      this.applyInsanityEffects();
    } else {
      // Cool down when not looking
      this.intensity = Math.max(0, this.intensity - 0.01);
      this.gazeStartTime = currentTime;
    }
  }

  isPlayerLookingThrough() {
    // Player is looking through if:
    // 1. Viewmaster is equipped
    // 2. Camera pitch is downward (as if holding to face)
    // 3. Not in menu/pause
    const headPose = headPoseSystem.getCurrentPose();
    const pitch = headPose.pitch;

    // Looking down = positive pitch in Three.js
    return pitch > 0.3 && this.isActive;
  }

  applyInsanityEffects() {
    const intensity = this.intensity;

    // Stage 1: Vignette (0.0 - 0.3)
    if (intensity > 0.1) {
      vfxManager.setStrength('viewmaster-vignette', intensity * 0.5);
    }

    // Stage 2: Desaturation (0.3 - 0.6)
    if (intensity > 0.3) {
      const desatAmount = (intensity - 0.3) / 0.7;
      vfxManager.setStrength('viewmaster-desaturate', desatAmount);
    }

    // Stage 3: Glitch effects (0.6 - 1.0)
    if (intensity > 0.6) {
      const glitchAmount = (intensity - 0.6) / 0.4;
      vfxManager.setStrength('viewmaster-glitch', glitchAmount);

      // Random screen shake at high intensity
      if (intensity > 0.8 && Math.random() < 0.1) {
        cameraController.shake(0.1, 100);
      }
    }

    // Update game state
    gameState.set('viewmasterInsanityIntensity', this.intensity);
  }

  triggerOverheat() {
    this.overheatCount++;

    if (this.overheatCount >= this.maxOverheat) {
      // Game over or bad ending
      this.unequip();
      gameState.set('hasViewmasterOverheat', true);
      sceneManager.loadScene('bad-ending-viewmaster');
    } else {
      // Temporary disable
      this.unequip();
      gameState.set('viewmasterManuallyRemoved', true);

      // Show warning
      uiManager.showWarning('The Viewmaster feels dangerously hot...');

      // Allow re-equip after cooldown
      setTimeout(() => {
        gameState.set('viewmasterManuallyRemoved', false);
      }, 30000);
    }
  }
}
```

---

## Part 6: NPC Gaze Reactions

### Attention States

```javascript
class NPCGazeReaction {
  constructor(npc, lookAtController) {
    this.npc = npc;
    this.lookAt = lookAtController;
    this.attentionState = 'idle';
    this.gazeMemory = [];  // Track recent gaze events
  }

  update(playerGaze) {
    const isLookingAtMe = this.isPlayerLookingAtNPC();
    const glanceAngle = this.getGlanceAngle(playerGaze);

    switch (this.attentionState) {
      case 'idle':
        if (isLookingAtMe) {
          this.transitionTo('noticed');
        }
        break;

      case 'noticed':
        // Make brief eye contact
        this.lookAt.lookAt(playerGaze.origin, 0.8);
        if (playerGaze.duration > 1000) {
          this.transitionTo('acknowledged');
        } else if (!isLookingAtMe) {
          this.transitionTo('idle');
        }
        break;

      case 'acknowledged':
        // Sustained eye contact
        this.lookAt.lookAt(playerGaze.origin, 1.0);
        if (playerGaze.duration > 3000) {
          this.transitionTo('unsettled');
        } else if (!isLookingAtMe) {
          this.transitionTo('idle');
        }
        break;

      case 'unsettled':
        // Player has been staring too long - creepy!
        this.lookAt.lookAt(playerGaze.origin, 0.3);  // Avert gaze slightly
        this.npc.playAnimation('uncomfortable');

        if (playerGaze.duration > 5000) {
          this.transitionTo('react');
        } else if (!isLookingAtMe) {
          this.transitionTo('idle');
        }
        break;

      case 'react':
        // Do something about being stared at
        this.npc.say("Is... something wrong?");
        this.transitionTo('post-react');
        break;

      case 'post-react':
        // Return to normal after reaction
        if (!isLookingAtMe) {
          this.transitionTo('idle');
        }
        break;
    }
  }

  transitionTo(newState) {
    this.attentionState = newState;
    this.gazeMemory.push({
      state: newState,
      time: performance.now()
    });
  }

  isPlayerLookingAtNPC() {
    return gazeTargetDetector.isLookingAt(this.npc.position);
  }

  getGlanceAngle(playerGaze) {
    // Angle between player's gaze and NPC position
    const toNPC = new THREE.Vector3()
      .subVectors(this.npc.position, playerGaze.origin)
      .normalize();

    return playerGaze.direction.angleTo(toNPC);
  }
}
```

---

## Part 7: Gaze-Based Game Mechanics

### Triggering Events by Staring

```javascript
// In interactionData.js
export const gazeTriggers = {
  spookyRune: {
    object: 'rune_on_wall',
    requiredStareTime: 3000,  // Must stare for 3 seconds
    maxLookAngle: 30,  // Within 30 degrees
    onTrigger: {
      action: 'setState',
      state: { sawRune: true, runeSightings: { $inc: 1 } }
    },
    onProgress: {
      vfx: 'rune-glow',
      strength: 'progress'  // Increases as player stares
    }
  },

  secretDoor: {
    object: 'crack_in_wall',
    requiredStareTime: 5000,
    maxLookAngle: 20,
    onTrigger: {
      action: 'animation',
      target: 'secret_door',
      animation: 'open'
    },
    hint: {
      text: "The crack seems to catch your eye...",
      triggerTime: 2000  // Show hint after 2 seconds of staring
    }
  }
};

// System to handle gaze triggers
class GazeTriggerSystem {
  constructor(triggers) {
    this.triggers = triggers;
    this.activeTriggers = new Map();
  }

  update(gazeInfo) {
    if (!gazeInfo.target) {
      this.activeTriggers.clear();
      return;
    }

    // Find matching trigger
    const trigger = Object.values(this.triggers).find(
      t => t.object === gazeInfo.target.name
    );

    if (!trigger) return;

    // Check if looking within angle threshold
    const lookAngle = this.getLookAngle(gazeInfo.target.position);
    if (lookAngle > trigger.maxLookAngle) return;

    // Get or create active trigger state
    let triggerState = this.activeTriggers.get(trigger);
    if (!triggerState) {
      triggerState = {
        startTime: performance.now(),
        triggered: false
      };
      this.activeTriggers.set(trigger, triggerState);
    }

    // Calculate progress
    const gazeDuration = performance.now() - triggerState.startTime;
    const progress = Math.min(1, gazeDuration / trigger.requiredStareTime);

    // Handle progress feedback
    if (trigger.onProgress) {
      this.handleProgress(trigger, progress);
    }

    // Show hint at halfway point
    if (trigger.hint && gazeDuration > trigger.hint.triggerTime) {
      uiManager.showHint(trigger.hint.text);
    }

    // Trigger when threshold reached
    if (progress >= 1 && !triggerState.triggered) {
      this.executeTrigger(trigger);
      triggerState.triggered = true;
    }
  }

  executeTrigger(trigger) {
    switch (trigger.onTrigger.action) {
      case 'setState':
        gameState.merge(trigger.onTrigger.state);
        break;
      case 'animation':
        animationManager.play(
          trigger.onTrigger.animation,
          trigger.onTrigger.target
        );
        break;
    }
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Not Smoothing Head Rotations

```javascript
// ❌ WRONG: Instant snapping
function lookAt(target) {
  head.quaternion.setFromRotationMatrix(
    lookAt(head.position, target)
  );
}

// ✅ CORRECT: Smooth interpolation
function lookAt(target) {
  const targetQuat = calculateLookAt(target);
  head.quaternion.slerp(targetQuat, 0.1);  // Smooth!
}
```

### 2. Unrealistic Neck Limits

```javascript
// ❌ WRONG: Head can spin 360°
head.lookAt(anywhere);  // Owl mode!

// ✅ CORRECT: Limit to human range
function lookAt(target) {
  const maxAngle = Math.PI / 4;  // 45 degrees
  // Clamp rotation to maxAngle
}
```

### 3. Forgetting Mobile Permission

```javascript
// ❌ WRONG: Assumes permission
window.addEventListener('deviceorientation', handler);

// ✅ CORRECT: Request permission first
async function initHeadTracking() {
  if (typeof DeviceOrientationEvent.requestPermission === 'function') {
    const permission = await DeviceOrientationEvent.requestPermission();
    if (permission === 'granted') {
      window.addEventListener('deviceorientation', handler);
    }
  }
}
```

### 4. Too Sensitive Gaze Detection

```javascript
// ❌ WRONG: Triggers even when glancing
const isLooking = raycastHit && hit.distance < 100;

// ✅ CORRECT: Require sustained gaze
const isLooking = raycastHit &&
                 hit.distance < 50 &&
                 gazeDuration > 1000;  // At least 1 second
```

---

## Performance Considerations

### Raycast Optimization

```javascript
// Limit raycast frequency
const RAYCAST_INTERVAL = 100;  // ms between raycasts
let lastRaycastTime = 0;

function update(currentTime) {
  if (currentTime - lastRaycastTime < RAYCAST_INTERVAL) {
    return;  // Skip this frame
  }
  lastRaycastTime = currentTime;

  // Perform raycast...
}

// Only raycast against relevant objects
const interactableObjects = scene.children.filter(
  obj => obj.userData.isGazable
);
```

### Look-At Update Rates

```javascript
// Different update rates for different body parts
const UPDATE_RATES = {
  eyes: 60,     // Fast - every frame
  head: 30,     // Medium - every other frame
  spine: 15     // Slow - every 4th frame
};
```

---

## 🎮 Game Design Perspective

### Why Head-Pose Matters

1. **Player Presence** - The game notices when you look at things
2. **Atmosphere** - Characters that make eye contact feel alive
3. **Pacing** - Control information reveal based on attention
4. **Horror** - Staring can trigger scares - creates tension
5. **Accessibility** - Gaze-based interactions for players with limited mobility

### Designing Gaze Mechanics

```javascript
// Principle 1: Always give feedback
playerLooksAtObject: {
  // Immediate: visual indicator
  onGazeStart: { highlight: true },

  // Sustained: build up
  onGazeHold: { vfx: 'glow', strength: 'progress' },

  // Complete: trigger
  onGazeComplete: { action: 'revealSecret' }
}

// Principle 2: Don't punish looking
bad: {
  // Player looks = instant death
  onGazeStart: { gameOver: true }
}

good: {
  // Player looks = warning, then escalation
  onGazeStart: { playSound: 'heartbeat' },
  onGazeHold_2s: { playSound: 'heartbeat-faster' },
  onGazeHold_5s: { showWarning: true },
  onGazeHold_10s: { gameOver: true }  // Gave plenty of warning
}

// Principle 3: Different thresholds for different contexts
subtleHint: { requiredStareTime: 2000 },
puzzleTrigger: { requiredStareTime: 5000 },
dangerZone: { requiredStareTime: 500 }  // Fast response
```

---

## Next Steps

Now that you understand head-pose animation:

- [AnimationManager](./animation-manager.md) - Camera and object animations
- [CharacterController](../04-input-physics/character-controller.md) - First-person controls
- [VFXManager](../07-visual-effects/vfx-manager.md) - Visual effects for insanity
- [Game State System](../02-core-architecture/game-state-system.md) - Gaze-based state changes

---

## References

- [DeviceOrientationEvent (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/DeviceOrientationEvent) - Web API for device orientation
- [Using Device Orientation (MDN Guide)](https://developer.mozilla.org/en-US/docs/Web/API/Device_orientation_events/Using_device_orientation) - Practical guide
- [Three.js Quaternion](https://threejs.org/docs/#api/en/math/Quaternion) - Rotation math
- [Three.js Raycaster](https://threejs.org/docs/#api/en/core/Raycaster) - Intersection testing
- [WebXR - Head Tracking](https://www.w3.org/TR/webxr/) - VR/AR head tracking standards

*Documentation last updated: January 12, 2026*
