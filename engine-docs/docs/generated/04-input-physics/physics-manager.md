# PhysicsManager - First Principles Guide

## Overview

The **PhysicsManager** integrates the Rapier physics engine into the Shadow Engine. It handles realistic physics simulation including gravity, collisions, forces, and joint constraints - all running at near-native speed through WebAssembly.

Think of PhysicsManager as the **laws of physics** for your game world - it makes objects fall, bounce, roll, and interact realistically.

## What You Need to Know First

Before understanding PhysicsManager, you should know:
- **Basic physics concepts** - Gravity, velocity, acceleration, mass
- **Collision detection** - How objects know when they touch
- **Rigid bodies** - Solid objects that don't deform
- **WebAssembly (WASM)** - Code that runs at near-native speed in browsers
- **Rapier physics engine** - The underlying physics library

### Quick Refresher: Physics Basics

```javascript
// Position - where something is (x, y, z)
const position = { x: 0, y: 10, z: 0 };

// Velocity - how fast something is moving (m/s)
const velocity = { x: 0, y: -5, z: 0 };

// Acceleration - how velocity is changing (m/s²)
const gravity = { x: 0, y: -9.81, z: 0 };

// Each frame:
// 1. Add acceleration to velocity
// 2. Add velocity to position
velocity.y += gravity.y * deltaTime;
position.y += velocity.y * deltaTime;
```

---

## Part 1: Why Use a Physics Engine?

### The Problem: Manual Physics Is Hard

Without a physics engine, you'd need to calculate:

```javascript
// ❌ WITHOUT physics engine - you write all this:

// Detect collision between two boxes
function checkBoxCollision(box1, box2) {
  return box1.minX < box2.maxX &&
         box1.maxX > box2.minX &&
         box1.minY < box2.maxY &&
         box1.maxY > box2.minY;
}

// Calculate bounce
function calculateBounce(velocity, normal, restitution) {
  // Complex dot product math...
}

// Handle stacking (A on B on C)
function handleStacking() {
  // Even more complex...
}

// And this is just simple boxes!
// Rotated boxes? Spheres? Complex shapes?
}
```

### The Solution: Rapier Physics Engine

```javascript
// ✅ WITH Rapier - the engine handles it:

// Create a physics body
const body = world.createRigidBody(
  RAPIER.RigidBodyDesc.fixed()
    .setTranslation(x, y, z)
);

// Add a collider shape
const collider = world.createCollider(
  RAPIER.ColliderDesc.cuboid(width, height, depth),
  body
);

// Rapier handles:
// - Collision detection (all shape types)
// - Gravity simulation
// - Velocity integration
// - Stacking, friction, bouncing
// - Joints and constraints
```

---

## Part 2: Rapier Physics Engine Overview

### What Is Rapier?

Rapier is a **cross-platform physics engine** written in Rust and compiled to WebAssembly:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAPIER PHYSICS ENGINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Written in Rust           ─────▶  Compiled to WebAssembly      │
│  (Fast, memory-safe)              (Runs near-native in JS)      │
│                                                                 │
│  Features:                                                      │
│  - Rigid body dynamics     - Collision detection               │
│  - Joints & constraints    - Continuous collision detection    │
│  - Island-based solver     - Sleeping bodies                   │
│  - Scene queries           - Raycasting                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Rapier for Shadow Engine?

| Feature | Benefit |
|---------|---------|
| **WASM-based** | Near-native performance in browsers |
| **Cross-platform** | Same physics on desktop, mobile, web |
| **Modern** | Active development, modern algorithms |
| **JavaScript bindings** | Easy integration with Three.js |
| **Free & open source** | No licensing fees |

---

## Part 3: PhysicsManager Structure

### Class Overview

```javascript
class PhysicsManager {
  constructor() {
    // Rapier physics world
    this.world = null;
    this.gravity = { x: 0, y: -9.81, z: 0 };

    // Track physics bodies
    this.bodies = new Map();     // bodyId -> RigidBody
    this.colliders = new Map();  // colliderId -> Collider

    // Track user data for collisions
    this.bodyUserData = new Map();   // bodyId -> user data
    this.colliderUserData = new Map(); // colliderId -> user data

    // Integration with Three.js
    this.meshMap = new Map();  // bodyId -> Three.js mesh

    // Physics settings
    this.settings = {
      timeStep: 1/60,      // Fixed timestep
      maxSubSteps: 8,      // Max physics steps per frame
      iterations: 4,       // Solver iterations
      fixedTimestep: true  // Use fixed timestep
    };

    // Event callbacks
    this.collisionCallbacks = [];
  }
}
```

---

## Part 4: Initialization

### Loading Rapier WASM

```javascript
async initialize(options = {}) {
  console.log('[PhysicsManager] Initializing...');

  // Import and init Rapier
  const RAPIER = await import('@dimforge/rapier3d');
  await RAPIER.init();

  this.RAPIER = RAPIER;

  // Create physics world
  this.world = new RAPIER.World({
    x: options.gravityX ?? 0,
    y: options.gravityY ?? -9.81,
    z: options.gravityZ ?? 0
  });

  console.log('[PhysicsManager] Initialized with gravity:',
    this.world.gravity);

  return this;
}
```

### Gravity Configuration

```javascript
setGravity(x, y, z) {
  if (!this.world) {
    console.warn('[PhysicsManager] Cannot set gravity: world not initialized');
    return;
  }

  this.world.gravity = { x, y, z };
  console.log('[PhysicsManager] Gravity set to:', { x, y, z });
}

// Common presets
setGravityEarth() { this.setGravity(0, -9.81, 0); }
setGravityMoon() { this.setGravity(, -1.62, 0); }
setGravityZero() { this.setGravity(0, 0, 0); }  // Space mode
```

---

## Part 5: Rigid Bodies

### What Is a Rigid Body?

A **rigid body** is a physics object that:
- Has mass (how heavy it is)
- Has position and rotation
- Responds to forces and gravity
- Can collide with other bodies

### Body Types

```javascript
const RAPIER = this.RAPIER;

// 1. FIXED - Never moves (walls, floors, buildings)
const floorBody = this.world.createRigidBody(
  RAPIER.RigidBodyDesc.fixed()
    .setTranslation(0, -1, 0)
);

// 2. DYNAMIC - Moves and falls (player, boxes, balls)
const boxBody = this.world.createRigidBody(
  RAPIER.RigidBodyDesc.dynamic()
    .setTranslation(0, 10, 0)
    .setCanSleep(true)  // Optimize by sleeping when still
);

// 3. KINEMATIC - Moves but isn't affected by forces (elevators, doors)
const doorBody = this.world.createRigidBody(
  RAPIER.RigidBodyDesc.kinematicPositionBased()
    .setTranslation(5, 0, 0)
);
```

### Creating Bodies with Helper Methods

```javascript
createBody(type, position, rotation, options = {}) {
  const RAPIER = this.RAPIER;

  // Create body description based on type
  let bodyDesc;

  switch (type) {
    case 'fixed':
      bodyDesc = RAPIER.RigidBodyDesc.fixed();
      break;
    case 'dynamic':
      bodyDesc = RAPIER.RigidBodyDesc.dynamic();
      break;
    case 'kinematic':
      bodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased();
      break;
    default:
      bodyDesc = RAPIER.RigidBodyDesc.dynamic();
  }

  // Set position
  bodyDesc.setTranslation(
    position.x ?? 0,
    position.y ?? 0,
    position.z ?? 0
  );

  // Set rotation (quaternion)
  if (rotation) {
    bodyDesc.setRotation({
      x: rotation.x ?? 0,
      y: rotation.y ?? 0,
      z: rotation.z ?? 0,
      w: rotation.w ?? 1
    });
  }

  // Set other properties
  if (options.mass !== undefined) {
    // Note: In Rapier, use .additionalMass to add mass
    // or use a collider with specific density
  }

  if (options.linearDamping !== undefined) {
    bodyDesc.setLinearDamping(options.linearDamping);
  }

  if (options.angularDamping !== undefined) {
    bodyDesc.setAngularDamping(options.angularDamping);
  }

  if (options.gravityScale !== undefined) {
    bodyDesc.setGravityScale(options.gravityScale);
  }

  // Create the body
  const body = this.world.createRigidBody(bodyDesc);
  const bodyId = body.handle;

  // Store reference
  this.bodies.set(bodyId, body);

  // Store user data
  if (options.userData) {
    this.bodyUserData.set(bodyId, options.userData);
  }

  return { body, bodyId };
}
```

---

## Part 6: Colliders

### What Is a Collider?

A **collider** defines the shape of a physics object. It's attached to a rigid body and determines how the object collides.

### Collider Shapes

```javascript
const RAPIER = this.RAPIER;

// 1. CUBOID (box)
const boxCollider = RAPIER.ColliderDesc.cuboid(
  halfWidth,   // Half-extent X
  halfHeight,  // Half-extent Y
  halfDepth    // Half-extent Z
);

// 2. BALL (sphere)
const sphereCollider = RAPIER.ColliderDesc.ball(radius);

// 3. CYLINDER
const cylinderCollider = RAPIER.ColliderDesc.cylinder(
  halfHeight,  // Half height
  radius
);

// 4. CAPSULE (common for characters)
const capsuleCollider = RAPIER.ColliderDesc.capsule(
  halfHeight,  // Half height (without hemispheres)
  radius
);

// 5. TRIMESH (complex shapes from 3D models)
const trimeshCollider = RAPIER.ColliderDesc.trimesh(
  vertices,  // Float32Array of positions
  indices    // Uint32Array of triangle indices
);

// 6. HEIGHTFIELD (terrain)
const heightfieldCollider = RAPIER.ColliderDesc.heightfield(
  rows, cols, heights, scale
);
```

### Creating Colliders

```javascript
createCollider(body, shapeType, shapeParams, options = {}) {
  const RAPIER = this.RAPIER;

  let colliderDesc;

  // Create collider description based on shape type
  switch (shapeType) {
    case 'box':
    case 'cuboid':
      colliderDesc = RAPIER.ColliderDesc.cuboid(
        shapeParams.halfWidth ?? 0.5,
        shapeParams.halfHeight ?? 0.5,
        shapeParams.halfDepth ?? 0.5
      );
      break;

    case 'sphere':
    case 'ball':
      colliderDesc = RAPIER.ColliderDesc.ball(
        shapeParams.radius ?? 0.5
      );
      break;

    case 'capsule':
      colliderDesc = RAPIER.ColliderDesc.capsule(
        shapeParams.halfHeight ?? 0.5,
        shapeParams.radius ?? 0.3
      );
      break;

    case 'cylinder':
      colliderDesc = RAPIER.ColliderDesc.cylinder(
        shapeParams.halfHeight ?? 0.5,
        shapeParams.radius ?? 0.5
      );
      break;

    case 'trimesh':
      colliderDesc = RAPIER.ColliderDesc.trimesh(
        shapeParams.vertices,
        shapeParams.indices
      );
      break;

    default:
      console.warn('[PhysicsManager] Unknown shape type:', shapeType);
      return null;
  }

  // Set collision properties
  if (options.isSensor !== undefined) {
    colliderDesc.setSensor(options.isSensor);
  }

  if (options.friction !== undefined) {
    colliderDesc.setFriction(options.friction);
  }

  if (options.restitution !== undefined) {
    colliderDesc.setRestitution(options.restitution);
  }

  if (options.density !== undefined) {
    colliderDesc.setDensity(options.density);
  }

  // Set collision layers/masks
  if (options.collisionGroups !== undefined) {
    colliderDesc.setCollisionGroups(options.collisionGroups);
  }

  if (options.solverGroups !== undefined) {
    colliderDesc.setSolverGroups(options.solverGroupsGroups);
  }

  // Create the collider attached to body
  const collider = this.world.createCollider(colliderDesc, body);
  const colliderId = collider.handle;

  // Store reference
  this.colliders.set(colliderId, collider);

  // Store user data
  if (options.userData) {
    this.colliderUserData.set(colliderId, options.userData);
  }

  return { collider, colliderId };
}
```

### Creating a Complete Physics Object

```javascript
createBox(position, size, options = {}) {
  // Create rigid body
  const { body, bodyId } = this.createBody(
    options.type ?? 'dynamic',
    position,
    options.rotation
  );

  // Create collider
  const { collider, colliderId } = this.createCollider(
    body,
    'box',
    {
      halfWidth: size.x / 2,
      halfHeight: size.y / 2,
      halfDepth: size.z / 2
    },
    {
      friction: options.friction ?? 0.5,
      restitution: options.restitution ?? 0.3,
      density: options.density ?? 1.0,
      isSensor: options.isSensor ?? false,
      userData: options.userData
    }
  );

  // Link to Three.js mesh if provided
  if (options.mesh) {
    this.meshMap.set(bodyId, options.mesh);
  }

  return { body, collider, bodyId, colliderId };
}
```

---

## Part 7: The Physics Step

### Advancing the Simulation

```javascript
step(deltaTime) {
  if (!this.world) return;

  // Fixed timestep for consistent physics
  if (this.settings.fixedTimestep) {
    this.world.step();
  } else {
    // Custom timestep
    this.world.step(this.settings.timeStep);
  }

  // Sync Three.js meshes with physics bodies
  this.syncMeshes();
}
```

### Syncing with Three.js

```javascript
syncMeshes() {
  for (const [bodyId, mesh] of this.meshMap) {
    const body = this.bodies.get(bodyId);
    if (!body || !mesh) continue;

    // Get position from physics body
    const position = body.translation();
    mesh.position.set(position.x, position.y, position.z);

    // Get rotation from physics body
    const rotation = body.rotation();
    mesh.quaternion.set(rotation.x, rotation.y, rotation.z, rotation.w);
  }
}
```

---

## Part 8: Collision Detection

### Collision Events

```javascript
setupCollisionEvents() {
  const RAPIER = this.RAPIER;

  // Create collision event listener
  this.collisionEventQueue = new RAPIER.EventQueue();

  // Check for collisions each step
  this.world.forEachCollider((collider) => {
    // Collision handler setup
  });
}

checkCollisions() {
  const queue = this.collisionEventQueue;

  // Drain the collision event queue
  while (queue.hasCollisionEvent()) {
    const event = queue.getNextCollisionEvent();

    const collider1 = event.collider1();
    const collider2 = event.collider2();

    // Get user data
    const data1 = this.colliderUserData.get(collider1);
    const data2 = this.colliderUserData.get(collider2);

    // Check if collision started or ended
    if (event.started()) {
      this.onCollisionStarted(collider1, collider2, data1, data2);
    } else if (event.stopped()) {
      this.onCollisionEnded(collider1, collider2, data1, data2);
    }
  }
}

onCollisionStarted(collider1, collider2, data1, data2) {
  console.log('[PhysicsManager] Collision started:',
    data1?.name, '-', data2?.name);

  // Trigger callbacks
  for (const callback of this.collisionCallbacks) {
    callback(collider1, collider2, data1, data2, true);
  }
}

onCollisionEnded(collider1, collider2, data1, data2) {
  console.log('[PhysicsManager] Collision ended:',
    data1?.name, '-', data2?.name);

  // Trigger callbacks
  for (const callback of this.collisionCallbacks) {
    callback(collider1, collider2, data1, data2, false);
  }
}
```

### Sensor Colliders

Sensors detect collisions but don't physically react:

```javascript
createTriggerZone(position, size, onEnter, onLeave) {
  // Create a fixed body
  const { body, bodyId } = this.createBody('fixed', position);

  // Create a sensor collider
  const { collider, colliderId } = this.createCollider(
    body,
    'box',
    {
      halfWidth: size.x / 2,
      halfHeight: size.y / 2,
      halfDepth: size.z / 2
    },
    {
      isSensor: true,  // KEY: Makes it a sensor
      userData: {
        type: 'trigger',
        onEnter,
        onLeave
      }
    }
  );

  return { body, collider, bodyId, colliderId };
}
```

---

## Part 9: Applying Forces and Impulses

### Force vs Impulse

| Type | Description | Use Case |
|------|-------------|----------|
| **Force** | Continuous push over time | Wind, thrusters, gravity |
| **Impulse** | Instant push | Jumping, explosions, impacts |
| **Torque** | Rotational force | Spinning objects |
| **Angular Impulse** | Instant rotational push | Bounce spin |

### Applying Forces

```javascript
applyForce(bodyId, force, wakeUp = true) {
  const body = this.bodies.get(bodyId);
  if (!body) {
    console.warn('[PhysicsManager] Body not found:', bodyId);
    return;
  }

  body.applyForce(
    { x: force.x, y: force.y, z: force.z },
    wakeUp
  );
}

// Example: Apply wind force
applyWindForce(bodyId, windDirection, windStrength) {
  this.applyForce(bodyId, {
    x: windDirection.x * windStrength,
    y: windDirection.y * windStrength,
    z: windDirection.z * windStrength
  });
}
```

### Applying Impulses

```javascript
applyImpulse(bodyId, impulse, wakeUp = true) {
  const body = this.bodies.get(bodyId);
  if (!body) return;

  body.applyImpulse(
    { x: impulse.x, y: impulse.y, z: impulse.z },
    wakeUp
  );
}

// Example: Jump impulse
jump(bodyId, jumpStrength) {
  // Apply upward impulse
  this.applyImpulse(bodyId, {
    x: 0,
    y: jumpStrength,
    z: 0
  });
}

// Example: Explosion force
explode(origin, force, radius) {
  for (const [bodyId, body] of this.bodies) {
    const pos = body.translation();
    const distance = Math.sqrt(
      Math.pow(pos.x - origin.x, 2) +
      Math.pow(pos.y - origin.y, 2) +
      Math.pow(pos.z - origin.z, 2)
    );

    if (distance < radius) {
      // Direction away from explosion
      const dir = {
        x: (pos.x - origin.x) / distance,
        y: (pos.y - origin.y) / distance,
        z: (pos.z - origin.z) / distance
      };

      // Scale by distance
      const strength = force * (1 - distance / radius);

      this.applyImpulse(bodyId, {
        x: dir.x * strength,
        y: dir.y * strength,
        z: dir.z * strength
      });
    }
  }
}
```

---

## Part 10: Scene Queries

### Raycasting

```javascript
castRay(origin, direction, maxDistance = Infinity) {
  const RAPIER = this.RAPIER;

  // Create ray
  const ray = new RAPIER.Ray(origin, direction);

  // Cast ray
  const hit = this.world.castRay(
    ray,
    maxDistance,
    true  // Solid only
  );

  if (hit) {
    const colliderId = hit.collider.handle;
    const point = ray.pointAt(hit.toi);

    return {
      hit: true,
      collider: this.colliders.get(colliderId),
      colliderId,
      point: { x: point.x, y: point.y, z: point.z },
      distance: hit.toi,
      normal: hit.normal ? { ...hit.normal } : null
    };
  }

  return { hit: false };
}
```

### Point Projection

```javascript
projectPoint(point) {
  // Find nearest collider to a point
  const projection = this.world.projectPoint(
    point,
    true  // Solid only
  );

  if (projection) {
    return {
      collider: this.colliders.get(projection.collider.handle),
      point: projection.point,
      isInside: projection.isInside
    };
  }

  return null;
}
```

### Shape Casting

```javascript
castShape(shape, position, direction, maxDistance) {
  // Cast a shape (like a capsule for character controller)
  const hit = this.world.castShape(
    shape,
    position,
    0,  // Shape rotation
    direction,
    maxDistance,
    true  // Solid only
  );

  return hit;
}
```

---

## Part 11: Joints and Constraints

### Fixed Joint

```javascript
createFixedJoint(body1Id, body2Id, anchor1, anchor2) {
  const RAPIER = this.RAPIER;

  const body1 = this.bodies.get(body1Id);
  const body2 = this.bodies.get(body2Id);

  const joint = RAPIER.JointDesc.fixed(
    { x: anchor1.x, y: anchor1.y, z: anchor1.z },
    { x: anchor2.x, y: anchor2.y, z: anchor2.z }
  );

  this.world.createImpulseJoint(joint, body1, body2);
}
```

### Revolute Joint (Hinge)

```javascript
createRevoluteJoint(body1Id, body2Id, anchor, axis) {
  const RAPIER = this.RAPIER;

  const body1 = this.bodies.get(body1Id);
  const body2 = this.bodies.get(body2Id);

  const joint = RAPIER.JointDesc.revolute(
    { x: anchor.x, y: anchor.y, z: anchor.z },
    { x: anchor.x, y: anchor.y, z: anchor.z },
    axis
  );

  this.world.createImpulseJoint(joint, body1, body2);
}
```

---

## Part 12: Character Controller Integration

### Kinematic Character Controller

```javascript
createCharacterController(config) {
  const RAPIER = this.RAPIER;

  // Create a kinematic body for the character
  const { body, bodyId } = this.createBody(
    'kinematic',
    config.position ?? { x: 0, y: 0, z: 0 }
  );

  // Create capsule collider
  const { collider } = this.createCollider(
    body,
    'capsule',
    {
      halfHeight: config.height ?? 0.9,
      radius: config.radius ?? 0.3
    },
    {
      friction: 0.0,
      density: 1.0
    }
  );

  // Create character controller helper
  const controller = this.world.createCharacterController(
    config.maxSlopeClimbAngle ?? 0.785,  // ~45 degrees
    config.minSlopeSlideAngle ?? 0.1,
    config.autostep ?? { height: 0.1, minWidth: 0.2 },
    config.offset ?? 0.01
  );

  return {
    body,
    controller,
    move: (direction) => {
      controller.computeColliderMovement(
        collider,
        direction,
        { x: 0, y: -1, z: 0 }
      );

      body.setNextKinematicTranslation({
        x: body.translation().x + controller.computedMovement().x,
        y: body.translation().y + controller.computedMovement().y,
        z: body.translation().z + controller.computedMovement().z
      });
    }
  };
}
```

---

## Part 13: Cleanup

### Removing Bodies and Colliders

```javascript
removeBody(bodyId) {
  const body = this.bodies.get(bodyId);
  if (!body) return;

  // Remove all colliders attached to this body
  body.forEachColliderAttached((collider) => {
    const colliderId = collider.handle;
    this.colliders.delete(colliderId);
    this.colliderUserData.delete(colliderId);
  });

  // Remove body from world
  this.world.removeRigidBody(body);

  // Clean up references
  this.bodies.delete(bodyId);
  this.bodyUserData.delete(bodyId);
  this.meshMap.delete(bodyId);
}

removeCollider(colliderId) {
  const collider = this.colliders.get(colliderId);
  if (!collider) return;

  this.world.removeCollider(collider);
  this.colliders.delete(colliderId);
  this.colliderUserData.delete(colliderId);
}
```

### Destroying the Physics World

```javascript
destroy() {
  console.log('[PhysicsManager] Destroying physics world...');

  // Clear all references
  this.bodies.clear();
  this.colliders.clear();
  this.bodyUserData.clear();
  this.colliderUserData.clear();
  this.meshMap.clear();

  // Free the world
  this.world = null;

  console.log('[PhysicsManager] Physics world destroyed');
}
```

---

## Common Mistakes Beginners Make

### 1. Forgetting to Step the World

```javascript
// ❌ WRONG: Physics never advances
createBox({ x: 0, y: 10, z: 0 }, { x: 1, y: 1, z: 1 });
// Box never falls!

// ✅ CORRECT: Step the simulation
createBox({ x: 0, y: 10, z: 0 }, { x: 1, y: 1, z: 1 });
world.step();  // Box falls!
```

### 2. Not Syncing Visuals

```javascript
// ❌ WRONG: Physics moves but visuals don't update
world.step();
// Three.js mesh still at old position!

// ✅ CORRECT: Sync mesh with physics body
world.step();
const pos = body.translation();
mesh.position.set(pos.x, pos.y, pos.z);
```

### 3. Wrong Scale

```javascript
// ❌ WRONG: Rapier uses meters, Three.js might use different units
const body = createBody({
  x: 0, y: 100, z: 0  // Is this 100 meters or 100 units?
});

// ✅ CORRECT: Establish a scale convention (1 unit = 1 meter)
const body = createBody({
  x: 0, y: 1.7, z: 0  // Eye height in meters
});
```

### 4. Forgetting to Wake Bodies

```javascript
// ❌ WRONG: Body is sleeping, won't respond
body.applyForce({ x: 0, y: 100, z: 0 });
// Nothing happens if body is asleep!

// ✅ CORRECT: Wake the body
body.applyForce({ x: 0, y: 100, z: 0 }, true);  // wakeUp = true
```

### 5. Creating Too Many Small Objects

```javascript
// ❌ WRONG: Creating 1000 individual boxes
for (let i = 0; i < 1000; i++) {
  createBox({ x: i, y: 10, z: 0 }, { x: 0.1, y: 0.1, z: 0.1 });
}
// Performance nightmare!

// ✅ CORRECT: Use fewer, larger objects or instanced rendering
// Or use compound shapes for complex objects
```

---

## Next Steps

Now that you understand PhysicsManager:

- [ColliderManager](./collider-manager.md) - Trigger zones and collision handling
- [CharacterController](./character-controller.md) - First-person movement
- [Interactive Objects](../08-interactive-objects/interaction.md) - Object interactions
- [Rapier Documentation](https://rapier.rs/docs/) - Official Rapier docs

---

## Source File Reference

- **Location:** `src/managers/PhysicsManager.js`
- **Key exports:** `PhysicsManager` class
- **Dependencies:** @dimforge/rapier3d-compat (v0.19.0+), Three.js
- **Used by:** CharacterController, ColliderManager, interactive objects

---

## References

- [Rapier Physics Documentation](https://rapier.rs/docs/) - Official Rapier docs
- [Rapier JavaScript Guide](https://rapier.rs/docs/user_guides/javascript/getting_started_js/) - Getting started with JS
- [Rapier User Guide](https://rapier.rs/docs/user_guides/rigid_bodies/) - Rigid bodies
- [Rapier Collision](https://rapier.rs/docs/user_guides/collision_detection/) - Collision detection
- [Rapier Joints](https://rapier.rs/docs/user_guides/joints/) - Constraints and joints
- [WebAssembly](https://webassembly.org/) - WASM documentation

*Documentation last updated: January 12, 2026*
