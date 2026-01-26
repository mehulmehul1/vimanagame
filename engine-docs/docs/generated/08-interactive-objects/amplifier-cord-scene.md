# Interactive Object: Amplifier Cord

## Overview

The **Amplifier Cord** is a physics puzzle interaction where players must connect a cord between an amplifier and a power source to progress. Building on the reusable PhoneCord module, the amplifier cord demonstrates how the same physics system can serve different gameplay purposes - from visual immersion (phone booth) to interactive puzzle (amplifier).

Think of the amplifier cord as the **"physical puzzle"** counterpart to the phone's "physical immersion" - the cord isn't just atmospheric, it's a mechanic the player must solve.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create satisfaction through physical problem-solving. Connecting the cord should feel tangible and rewarding - a clear "I did that" moment.

**Why an Amplifier Connection Puzzle?**
- **Familiar Interaction**: Everyone knows plugging something in feels
- **Clear Cause-Effect**: Plug in → power on → obvious feedback
- **Physical Tangibility**: Cord physics makes the connection feel real
- **Progress Marker**: Amplifier turning on signals progress

**Player Psychology**:
```
See Amplifier → It's Off → "I need to power this"
     ↓
See Cord → It's Unplugged → "Connect them"
     ↓
Drag Cord → Physics Responds → "I'm doing it"
     ↓
Connect → Amplifier Powers On → Satisfaction!
     ↓
Audio Plays → "Something new is available"
```

### Design Decisions

**1. Physics-Based Connection**
Rather than a simple "click to connect," the player must drag the cord plug to the socket. This creates:
- Physical engagement (not abstract)
- Clear feedback (cord follows mouse/cursor)
- Spatial understanding (where is the socket?)

**2. Audio Feedback on Connection**
When properly connected, the amplifier should produce sound - immediate, unmistakable feedback.

**3. State Progression**
The amplifier connection unlocks new content (music, dialog, etc.) - serves as a gate for progression.

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the amplifier cord implementation, you should know:
- **PhoneCord module reuse** - Same physics, different configuration
- **Drag interaction** - Moving objects with mouse/touch
- **Distance checking** - Determining when plug is close enough to socket
- **Socket detection** - Raycasting to find connection point
- **State changes** - Triggering game state on connection

### Core Architecture

```
AMPLIFIER CORD SYSTEM ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                   AMPLIFIER CORD CONTROLLER               │
│  - Manages cord physics (via PhoneCord)                  │
│  - Handles drag interaction                              │
│  - Detects socket proximity                              │
│  - Triggers connection events                            │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  PHONE CORD   │  │   DRAG       │  │   SOCKET     │
│  (Customized) │  │   HANDLER    │  │   DETECTION  │
│  - Longer     │  │  - Raycast   │  │  - Distance  │
│  - Heavy plug │  │  - Physics   │  │  - Snap      │
└───────────────┘  └───────────────┘  └───────────────┘
```

### Cord Configuration for Amplifier

```javascript
// Amplifier cord configuration
const amplifierCordConfig = {
  // Longer than phone cords
  cordSegments: 15,
  cordSegmentLength: 0.06,  // 90cm total length

  // Heavy plug end (more mass, different feel)
  plugMass: 0.1,  // 100g plug (heavier than receiver)
  plugDamping: 5.0,  // Less swing, more control

  // Socket attachment
  socketPosition: { x: 2, y: 0.1, z: 0 },
  plugAttachMesh: this.amplifierPlug,
  socketMesh: this.amplifierSocket,

  // Connection threshold
  connectionDistance: 0.15,  // 15cm - must be close to connect

  // Visual
  cordColor: 0x222222,  // Dark cord
  cordVisualRadius: 0.01,
  plugModel: "amplifier_plug.glb"
};
```

### Drag Interaction Handler

```javascript
class AmplifierCordDragHandler {
  constructor(cordController, config) {
    this.cordController = cordController;
    this.socketPosition = config.socketPosition;
    this.connectionDistance = config.connectionDistance;

    this.isDragging = false;
    this.dragPlane = new THREE.Plane();
    this.dragOffset = new THREE.Vector3();
    this.raycaster = new THREE.Raycaster();
  }

  /**
   * Start dragging the plug
   */
  startDrag(event, camera, plugMesh) {
    // Calculate mouse position
    const mouse = this.getMousePosition(event);

    // Set up raycaster
    this.raycaster.setFromCamera(mouse, camera);

    // Create drag plane at plug position, facing camera
    const plugPos = new THREE.Vector3();
    plugMesh.getWorldPosition(plugPos);
    const cameraDir = new THREE.Vector3();
    camera.getWorldDirection(cameraDir);

    this.dragPlane.setFromNormalAndCoplanarPoint(
      cameraDir,
      plugPos
    );

    // Calculate offset
    const intersectPoint = new THREE.Vector3();
    this.raycaster.ray.intersectPlane(this.dragPlane, intersectPoint);
    this.dragOffset.subVectors(plugPos, intersectPoint);

    this.isDragging = true;
    this.cordController.onDragStart();
  }

  /**
   * Update drag position
   */
  updateDrag(event, camera) {
    if (!this.isDragging) return false;

    const mouse = this.getMousePosition(event);
    this.raycaster.setFromCamera(mouse, camera);

    // Find new position on drag plane
    const intersectPoint = new THREE.Vector3();
    const hit = this.raycaster.ray.intersectPlane(this.dragPlane, intersectPoint);

    if (hit) {
      // Apply offset
      const targetPos = intersectPoint.add(this.dragOffset);

      // Update plug position (via cord anchor)
      this.cordController.movePlug(targetPos);

      // Check for socket proximity
      const distance = targetPos.distanceTo(this.socketPosition);
      if (distance < this.connectionDistance) {
        this.connectToSocket();
        return true;  // Connected!
      }
    }

    return false;  // Still dragging
  }

  /**
   * End dragging
   */
  endDrag() {
    this.isDragging = false;
    this.cordController.onDragEnd();
  }

  /**
   * Connect plug to socket
   */
  connectToSocket() {
    this.isDragging = false;

    // Snap plug to socket position
    this.cordController.snapToSocket(this.socketPosition);

    // Trigger connection effects
    this.onConnected();
  }

  /**
   * Handle successful connection
   */
  onConnected() {
    // Audio feedback
    this.cordController.playSound("amp_plug_in");

    // Visual feedback
    this.cordController.showSparkEffect(this.socketPosition);

    // Game state change
    this.cordController.gameManager.emit("amplifier:connected");

    // Power on amplifier
    this.cordController.powerOnAmplifier();
  }

  /**
   * Get normalized mouse position
   */
  getMousePosition(event) {
    return new THREE.Vector2(
      (event.clientX / window.innerWidth) * 2 - 1,
      -(event.clientY / window.innerHeight) * 2 + 1
    );
  }
}
```

### Amplifier Cord Controller

```javascript
class AmplifierCordController {
  constructor(options = {}) {
    this.scene = options.scene;
    this.physicsManager = options.physicsManager;
    this.gameManager = options.gameManager;
    this.sfxManager = options.sfxManager;

    this.amplifierPlug = options.amplifierPlug;
    this.amplifierSocket = options.amplifierSocket;
    this.isConnected = false;

    // Create cord with custom config
    this.cord = new PhoneCord({
      scene: this.scene,
      physicsManager: this.physicsManager,
      cordAttach: this.amplifierSocket,  // Cord from socket
      receiver: this.amplifierPlug,      // To plug
      loggerName: "AmplifierCord",
      config: {
        cordSegments: 15,
        cordSegmentLength: 0.06,
        cordDroopAmount: 1.5,  // Less droop (stiffer cord)
        initMode: "straight",
        cordColor: 0x222222,
        cordVisualRadius: 0.01
      }
    });

    // Create drag handler
    this.dragHandler = new AmplifierCordDragHandler(this, {
      socketPosition: options.socketPosition,
      connectionDistance: 0.15
    });

    // Setup interaction
    this.setupInteraction();
  }

  setupInteraction() {
    // Mouse events
    document.addEventListener("mousedown", (e) => this.onMouseDown(e));
    document.addEventListener("mousemove", (e) => this.onMouseMove(e));
    document.addEventListener("mouseup", (e) => this.onMouseUp(e));

    // Touch events
    document.addEventListener("touchstart", (e) => this.onTouchStart(e));
    document.addEventListener("touchmove", (e) => this.onTouchMove(e));
    document.addEventListener("touchend", (e) => this.onTouchEnd(e));
  }

  onMouseDown(event) {
    if (this.isConnected) return;

    // Raycast to check if clicking on plug
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(
      (event.clientX / window.innerWidth) * 2 - 1,
      -(event.clientY / window.innerHeight) * 2 + 1
    );

    raycaster.setFromCamera(mouse, this.camera);
    const intersects = raycaster.intersectObject(this.amplifierPlug, true);

    if (intersects.length > 0) {
      this.dragHandler.startDrag(event, this.camera, this.amplifierPlug);
    }
  }

  onMouseMove(event) {
    if (!this.isConnected) {
      this.dragHandler.updateDrag(event, this.camera);
    }
  }

  onMouseUp(event) {
    this.dragHandler.endDrag();
  }

  /**
   * Move plug to target position (during drag)
   */
  movePlug(targetPos) {
    // Update the receiver anchor to follow target
    if (this.cord.receiverAnchor) {
      this.cord.receiverAnchor.setNextKinematicPosition({
        x: targetPos.x,
        y: targetPos.y,
        z: targetPos.z
      });
    }
  }

  /**
   * Snap plug to socket position
   */
  snapToSocket(socketPos) {
    this.isConnected = true;

    // Set plug position to socket
    this.amplifierPlug.position.copy(socketPos);

    // Update cord anchor
    if (this.cord.receiverAnchor) {
      this.cord.receiverAnchor.setNextKinematicPosition({
        x: socketPos.x,
        y: socketPos.y,
        z: socketPos.z
      });

      // Make it kinematic (locked in place)
      this.cord.receiverAnchor.setBodyType(
        this.physicsManager.RAPIER.RigidBodyType.KinematicPositionBased,
        true
      );
    }
  }

  /**
   * Power on the amplifier
   */
  powerOnAmplifier() {
    // Light up amplifier
    const amplifierLights = this.amplifierSocket.parent.children.filter(
      c => c.name.includes("light") || c.name.includes("LED")
    );

    amplifierLights.forEach(light => {
      if (light.material) {
        light.material.emissive = new THREE.Color(0x00ff00);
        light.material.emissiveIntensity = 1.0;
      }
    });

    // Play power-on sound
    this.sfxManager.play("amplifier_power_on");

    // Emit event for other systems
    this.gameManager.emit("amplifier:powered", {
      amplifierId: this.amplifierSocket.parent.name
    });
  }

  /**
   * Show spark effect at connection point
   */
  showSparkEffect(position) {
    // Create particle burst
    const sparkCount = 20;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(sparkCount * 3);
    const velocities = [];

    for (let i = 0; i < sparkCount; i++) {
      positions[i * 3] = position.x;
      positions[i * 3 + 1] = position.y;
      positions[i * 3 + 2] = position.z;

      velocities.push({
        x: (Math.random() - 0.5) * 2,
        y: Math.random() * 2,
        z: (Math.random() - 0.5) * 2
      });
    }

    geometry.setAttribute("position",
      new THREE.BufferAttribute(positions, 3)
    );

    const material = new THREE.PointsMaterial({
      color: 0xffff00,
      size: 0.05,
      transparent: true,
      opacity: 1,
      blending: THREE.AdditiveBlending
    });

    const sparks = new THREE.Points(geometry, material);
    this.scene.add(sparks);

    // Animate sparks
    const startTime = performance.now();
    const duration = 500;  // 500ms

    const animateSparks = () => {
      const elapsed = performance.now() - startTime;
      const progress = elapsed / duration;

      if (progress >= 1) {
        this.scene.remove(sparks);
        return;
      }

      const positions = sparks.geometry.attributes.position.array;
      for (let i = 0; i < sparkCount; i++) {
        positions[i * 3] += velocities[i].x * 0.02;
        positions[i * 3 + 1] += velocities[i].y * 0.02;
        positions[i * 3 + 2] += velocities[i].z * 0.02;
      }
      sparks.geometry.attributes.position.needsUpdate = true;
      sparks.material.opacity = 1 - progress;

      requestAnimationFrame(animateSparks);
    };

    animateSparks();
  }

  /**
   * Update loop
   */
  update(dt) {
    // Update cord physics
    if (this.cord) {
      this.cord.update();
    }
  }

  /**
   * Clean up
   */
  destroy() {
    if (this.cord) {
      this.cord.destroy();
    }

    // Remove event listeners
    document.removeEventListener("mousedown", this.onMouseDown);
    document.removeEventListener("mousemove", this.onMouseMove);
    document.removeEventListener("mouseup", this.onMouseUp);
  }
}

export default AmplifierCordController;
```

---

## 📝 How To Build A Puzzle Like This

### Step 1: Define the Puzzle Mechanics

What makes this a puzzle?

```javascript
const puzzleDefinition = {
  type: "connection",
  objective: "Connect amplifier to power",

  components: {
    plug: "movable_end",
    socket: "fixed_target",
    cord: "physics_connection"
  },

  solution: "Drag plug to socket and release",

  feedback: {
    near: "Plug glows when close",
    connect: "Spark effect + sound",
    success: "Amplifier powers on"
  }
};
```

### Step 2: Configure Your Cord

```javascript
const myCordConfig = {
  // Adjust for your puzzle
  cordSegments: 15,        // Length
  cordSegmentLength: 0.06, // Segment size
  cordDroopAmount: 1.5,    // Stiffness

  // Visual
  cordColor: 0x222222,
  cordVisualRadius: 0.01,

  // Connection
  connectionDistance: 0.15,  // How close to connect
  snapOnConnect: true
};
```

### Step 3: Implement Drag Detection

```javascript
// Simple drag detection
const isClickable = (mesh) => {
  return mesh.userData.draggable === true;
};

const onPointerDown = (event) => {
  const hit = raycast(event);

  if (hit && isClickable(hit.object)) {
    startDrag(hit.object);
  }
};
```

---

## 🔧 Variations For Your Game

### Pipe Connection Puzzle

```javascript
class PipeConnection {
  // Connect pipes between valves
  config = {
    connectionType: "threaded",
    interaction: "rotate_to_connect",
    feedback: "hissing_sound + water_flow"
  };
}
```

### Wire Stripping Puzzle

```javascript
class WireStripping {
  // Strip wire ends before connecting
  config = {
    tools: ["stripper", "wire"],
    sequence: ["strip", "twist", "connect"],
    timing: true  // Must do quickly
  };
}
```

### Circuit Board Puzzle

```javascript
class CircuitPuzzle {
  // Place components, connect with traces
  config = {
    components: ["resistor", "capacitor", "chip"],
    connections: "draw_traces",
    verify: "check_circuit"
  };
}
```

---

## Common Mistakes Beginners Make

### 1. Cord Too Short to Reach

```javascript
// ❌ WRONG: Can't quite reach
{ cordSegments: 8, cordSegmentLength: 0.05 }
// 40cm total - frustrating!

// ✅ CORRECT: Generous length
{ cordSegments: 15, cordSegmentLength: 0.06 }
// 90cm total - comfortable reach with slack
```

### 2. Connection Distance Too Small

```javascript
// ❌ WRONG: Must be pixel-perfect
{ connectionDistance: 0.02 }  // 2cm
// frustrating precision required

// ✅ CORRECT: Forgiving
{ connectionDistance: 0.15 }  // 15cm
// Satisfying without being too easy
```

### 3. No Connection Feedback

```javascript
// ❌ WRONG: Silent connection
connectToSocket() {
  this.isConnected = true;
}
// Did it work?

// ✅ CORRECT: Clear feedback
connectToSocket() {
  this.isConnected = true;
  this.playSound("plug_in");
  this.showSparks();
  this.amplifier.powerOn();
}
// Unmistakable confirmation
```

---

## Performance Considerations

```
AMPLIFIER CORD PERFORMANCE:

Cord Physics:
├── Segments: 15 (moderate)
├── Update: Every frame
└── Impact: Minor

Drag Detection:
├── Raycast: On drag only
├── Plane intersection: Minimal math
└── Impact: Negligible

Visual Effects:
├── Cord tube: Recreated each frame
├── Spark particles: One-time burst
└── Impact: Minimal

Optimization:
- Use lower segment count on mobile
- Share cord physics logic
- Cache spark particle system
```

---

## Related Systems

- [Phone Booth Scene](./phone-booth-scene.md) - Cord physics origin
- [PhoneCord Module](`../src/content/phoneCord.js`) - Shared cord implementation
- [Candlestick Phone](./candlestick-phone-scene.md) - Another cord usage
- [PhysicsManager](../04-input-physics/physics-manager.md) - Physics simulation
- [VFXManager](../07-visual-effects/vfx-manager.md) - Spark effects

---

## Source File Reference

**Primary Files**:
- `../src/content/phoneCord.js` - Shared cord physics (634 lines)
- `../src/content/amplifierCord.js` - Amplifier cord controller (estimated)

**Key Classes**:
- `PhoneCord` - Reusable cord module
- `AmplifierCordController` - Puzzle-specific implementation
- `AmplifierCordDragHandler` - Drag interaction

**Dependencies**:
- Three.js (Raycaster, Plane, Vector3)
- Rapier (RigidBody, kinematic positioning)
- GameManager (events, state)

---

## References

- [Three.js Raycaster](https://threejs.org/docs/#api/en/core/Raycaster) - Click detection
- [Three.js Plane](https://threejs.org/docs/#api/en/math/Plane) - Drag surface
- [Rapier Kinematic Bodies](https://rapier.rs/docs/user_guides/javascript/rigid_bodies) - Controlled movement
- [Puzzle Design Principles](https://www.gamedeveloper.com/design/puzzle-design-101) - Design theory

*Documentation last updated: January 12, 2026*
