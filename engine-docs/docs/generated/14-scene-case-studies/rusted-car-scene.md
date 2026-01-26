# Rusted Car Scene - Environmental Obstruction

**Scene Case Study #12**

---

## What You Need to Know First

- **Gaussian Splatting Basics** (See: *Rendering System*)
- **Static Prop Integration** (See: *Scene Management*)
- **Collision Detection** (See: *Physics System*)
- **Environmental Storytelling** (See: *Game Design Principles*)

---

## Scene Overview

| Property | Value |
|----------|-------|
| **Location** | Four-way intersection, blocking south exit |
| **Narrative Context** | Environmental storytelling prop suggesting long-term abandonment |
| **Player Experience** | Passive observation - the car is scenery, not interactive |
| **Atmosphere** | Decay, neglect, passage of time |
| **Technical Focus** | Static prop placement, collision shaping, visual storytelling |

### The Scene's Purpose

The rusted car serves as **environmental storytelling through obstruction**. Unlike interactive props like the phonograph or radio, this car exists solely to:
1. **Block a path** - physically preventing player access to the south exit
2. **Convey abandonment** - visual evidence that this area has been deserted for decades
3. **Ground the world** - adds verisimilitude through mundane, realistic decay

---

## 🎮 Game Design Perspective

### Creative Intent

**Why a rusted car? Why not a barrier, wall, or debris?**

A rusted car is a **specific storytelling choice** that conveys information without text or dialogue:

| Element | What It Communicates |
|---------|---------------------|
| **Rust pattern** | Decades of exposure to elements |
| **Missing parts** | Stripped for scrap over time |
| **Position blocking road** | Abandoned in place, not parked |
| **Make/model (era-specific)** | Rough timeline of when abandonment occurred |
| **Interior condition** | Was it evacuated? Hastily left? |

### Design Philosophy

**Passive Environmental Storytelling:**

```
Active Storytelling: "The town was abandoned in 1987 after the mill closed."
                      ↑ Player is told explicitly

Passive Storytelling: [Rusted 1970s car blocking the road]
                      ↑ Player deduces abandonment era and circumstances
```

The rusted car is an **environmental inference engine** - players construct their own narrative from visual evidence.

### Mood Building

The car contributes to mood through:

1. **Scale** - Large object makes the player feel small in the abandoned space
2. **Permanence** - Cannot be moved, emphasizes player's lack of control
3. **Familiarity made alien** - Cars are normally functional; this one is dead and frozen
4. **Gradual realization** - Player notices details on approach (stripped parts, personal items left inside)

### Player Psychology

| Psychological Effect | How the Car Achieves It |
|---------------------|-------------------------|
| **Unease** | Familiar object rendered hostile by decay |
| **Curiosity** | What happened to the owner? Why was it left? |
| **Isolation** | Roads are for travel; blocked road = trapped |
| **Temporal dislocation** | Old car in modern-ish setting = time is wrong here |

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
┌─────────────────────────────────────────────┐
│                                             │
│   [Alley West]           [Alley East]       │
│         │                     │             │
│         └──────────┬──────────┘             │
│                    │                        │
│              ╔═══════════╗                  │
│              ║ RUSTED    ║  ← Blocks path   │
│              ║ CAR       ║    to south      │
│              ╚═══════════╝                  │
│                    │                        │
│              [Plaza North]                  │
│                    │                        │
│               ═════════                     │
│              ROAD ENDS                      │
│                                             │
└─────────────────────────────────────────────┘
```

### Player Path

```
1. Player approaches intersection
   ↓
2. Sees car blocking south path from distance
   ↓
3. Moves closer to investigate
   ↓
4. Discovers cannot pass (collision boundary)
   ↓
5. Circles car, observes details
   ↓
6. Turns back to explore available paths
```

### Atmosphere Layers

| Layer | Elements |
|-------|----------|
| **Visual** | Oxidized metal, broken glass, faded paint, vegetation growth |
| **Audio** | Wind through chassis, metal creak, hollow thud if bumped |
| **Lighting** | Harsh shadows from rusted surfaces, ambient occlusion in wheel wells |
| **Interaction** | Solid collision, no movement, no response to player actions |

---

## Technical Implementation

### Asset Data Structure

```javascript
// Static prop configuration
export const RUSTED_CAR = {
  id: 'rusted_car_intersection',
  type: 'static_prop',

  // Splat data
  splat: {
    file: '/assets/splats/rusted-car.ply',
    maxPoints: 2500000,
    renderScale: 1.0,
    lodDistances: [0, 20, 50]  // High, medium, low detail
  },

  // World placement
  transform: {
    position: { x: 0, y: 0, z: -8 },
    rotation: { x: 0, y: 15, z: 0 },  // Slight angle, not perfectly straight
    scale: { x: 1, y: 1, z: 1 }
  },

  // Collision shape (simplified for performance)
  collision: {
    type: 'compound',
    shapes: [
      {
        type: 'box',
        offset: { x: 0, y: 0.5, z: 0 },
        halfExtents: { x: 1.8, y: 0.6, z: 4.0 },  // Main body
        friction: 0.8,
        restitution: 0.1
      },
      {
        type: 'box',
        offset: { x: 0, y: 1.2, z: 1.5 },
        halfExtents: { x: 0.7, y: 0.5, z: 0.8 },  // Cabin/roof
        friction: 0.8,
        restitution: 0.1
      }
    ]
  },

  // Visual storytelling elements
  details: {
    stripped: ['wheels', 'radio', 'engine', 'trunk_lid'],
    remaining: ['frame', 'body_panels', 'seats', 'steering_column'],
    vegetation: {
      vines: true,
      moss: true,
      weeds_in_wheel_wells: true
    },
    personalItems: [
      {
        item: 'baby_shoe',
        position: { x: 0.3, y: 0.1, z: 2.1 },
        visible: true  // On back seat, visible through missing glass
      }
    ]
  },

  // Environmental responses
  audio: {
    ambient: 'wind_through_car_chassis',
    impact: 'metal_hollow_thud',
    volume: 0.3
  },

  // Path blocking
  obstruction: {
    blockedPath: 'south_exit',
    blockerType: 'static',
    playerMessage: null  // No tutorial text - let player observe
  }
};
```

### Static Prop Manager

```javascript
/**
 * Manages non-interactive environmental props
 */
class StaticPropManager {
  constructor(scene) {
    this.scene = scene;
    this.props = new Map();
    this.physics = scene.physics;
  }

  /**
   * Load a static prop into the scene
   */
  async loadProp(config) {
    const prop = {
      id: config.id,
      config: config,
      mesh: null,
      collisionBody: null
    };

    // Load Gaussian Splat
    prop.mesh = await this.loadSplatProp(config.splat);

    // Apply transform
    prop.mesh.position.set(
      config.transform.position.x,
      config.transform.position.y,
      config.transform.position.z
    );

    prop.mesh.rotation.set(
      THREE.MathUtils.degToRad(config.transform.rotation.x),
      THREE.MathUtils.degToRad(config.transform.rotation.y),
      THREE.MathUtils.degToRad(config.transform.rotation.z)
    );

    // Create collision body
    prop.collisionBody = this.createCollisionBody(config.collision);
    prop.collisionBody.setPosition(
      config.transform.position.x,
      config.transform.position.y,
      config.transform.position.z
    );

    // Setup ambient audio response
    if (config.audio) {
      this.setupAmbientAudio(prop, config.audio);
    }

    // Register as obstruction if applicable
    if (config.obstruction) {
      this.registerObstruction(prop, config.obstruction);
    }

    this.props.set(config.id, prop);
    return prop;
  }

  /**
   * Load Gaussian Splat for prop
   */
  async loadSplatProp(splatConfig) {
    const splat = await this.scene.splatLoader.load(splatConfig.file);

    // Configure LOD
    if (splatConfig.lodDistances) {
      splat.setLOD(splatConfig.lodDistances);
    }

    return splat;
  }

  /**
   * Create physics collision body
   */
  createCollisionBody(collisionConfig) {
    if (collisionConfig.type === 'compound') {
      const shapes = collisionConfig.shapes.map(shape => {
        return this.createShape(shape);
      });

      return this.physics.createCompoundBody(shapes, {
        type: 'static',
        friction: collisionConfig.shapes[0].friction,
        restitution: collisionConfig.shapes[0].restitution
      });
    }

    return this.physics.createBody(collisionConfig);
  }

  /**
   * Create individual collision shape
   */
  createShape(shapeConfig) {
    switch (shapeConfig.type) {
      case 'box':
        return this.physics.createBoxShape({
          halfExtents: shapeConfig.halfExtents,
          offset: shapeConfig.offset
        });

      case 'cylinder':
        return this.physics.createCylinderShape({
          radius: shapeConfig.radius,
          height: shapeConfig.height,
          offset: shapeConfig.offset
        });

      case 'mesh':
        return this.physics.createMeshShape({
          mesh: shapeConfig.mesh,
          convex: shapeConfig.convex || false
        });

      default:
        console.warn(`Unknown collision shape type: ${shapeConfig.type}`);
        return null;
    }
  }

  /**
   * Setup ambient audio responses
   */
  setupAmbientAudio(prop, audioConfig) {
    // Create positional ambient sound
    const ambient = this.scene.audio.createAmbientSource({
      url: audioConfig.ambient,
      position: prop.mesh.position,
      volume: audioConfig.volume,
      loop: true,
      spatial: true
    });

    // Impact sound setup
    prop.impactSound = this.scene.audio.createOneShot(audioConfig.impact, {
      position: prop.mesh.position,
      volume: audioConfig.volume
    });

    ambient.play();

    // Listen for collision events
    this.physics.on('collision', (event) => {
      if (event.bodyB === prop.collisionBody) {
        this.handleImpact(prop, event);
      }
    });
  }

  /**
   * Handle player/other collision with prop
   */
  handleImpact(prop, event) {
    const impactVelocity = event.contact.getImpactVelocity();

    // Only play sound for significant impacts
    if (impactVelocity > 0.5) {
      const volume = Math.min(impactVelocity / 5.0, 1.0);
      prop.impactSound.play({ volume });
    }
  }

  /**
   * Register path obstruction
   */
  registerObstruction(prop, obstruction) {
    this.scene.pathManager.blockPath(
      obstruction.blockedPath,
      prop.id,
      obstruction.blockerType
    );
  }

  /**
   * Unload a prop (for dynamic loading/unloading)
   */
  unloadProp(propId) {
    const prop = this.props.get(propId);
    if (!prop) return;

    // Remove splat mesh
    this.scene.remove(prop.mesh);

    // Remove collision body
    this.physics.removeBody(prop.collisionBody);

    // Clean up audio
    if (prop.ambientSound) {
      prop.ambientSound.stop();
    }

    // Unregister obstruction
    if (prop.config.obstruction) {
      this.scene.pathManager.unblockPath(
        prop.config.obstruction.blockedPath,
        propId
      );
    }

    this.props.delete(propId);
  }

  /**
   * Update LOD based on player distance
   */
  update(playerPosition) {
    for (const prop of this.props.values()) {
      if (prop.mesh && prop.mesh.setLOD) {
        const distance = prop.mesh.position.distanceTo(playerPosition);
        prop.mesh.updateLOD(distance);
      }
    }
  }
}
```

### Environmental Storytelling System

```javascript
/**
 * System for managing passive narrative elements
 */
class EnvironmentalStorytelling {
  constructor(scene) {
    this.scene = scene;
    this.observedElements = new Set();
    this.storyTriggers = new Map();
  }

  /**
   * Register a story-telling element
   */
  registerElement(elementId, config) {
    this.storyTriggers.set(elementId, {
      id: elementId,
      details: config.details || [],
      observationDistance: config.observationDistance || 2.0,
      observationAngle: config.observationAngle || 45,
      observed: false,
      hintDelay: config.hintDelay || 3000  // ms before hint
    });
  }

  /**
   * Check for player observation
   */
  update(playerPosition, playerForward) {
    for (const [id, element] of this.storyTriggers) {
      if (element.observed) continue;

      const prop = this.scene.staticProps.get(id);
      if (!prop) continue;

      const distance = playerPosition.distanceTo(prop.mesh.position);
      const direction = new THREE.Vector3()
        .subVectors(prop.mesh.position, playerPosition)
        .normalize();

      const angle = THREE.MathUtils.radToDeg(
        Math.acos(playerForward.dot(direction))
      );

      // Player is observing this element
      if (distance < element.observationDistance && angle < element.observationAngle) {
        this.handleObservation(id, element);
      }
    }
  }

  /**
   * Handle player observing a story element
   */
  handleObservation(elementId, element) {
    element.observed = true;
    this.observedElements.add(elementId);

    // Don't show popup - let player observe
    // But we might trigger subtle audio or lighting changes
    this.onElementObserved(elementId, element);
  }

  /**
   * Callback when element is observed
   */
  onElementObserved(elementId, element) {
    // Could trigger:
    // - Subtle music cue
    // - Lighting shift
    // - Ghostly whisper
    // - Just internal tracking for achievements

    console.log(`Player observed: ${elementId}`);
  }

  /**
   * Check if player has observed specific details
   */
  hasObserved(elementId) {
    return this.observedElements.has(elementId);
  }

  /**
   * Get observation progress
   */
  getObservationProgress() {
    return {
      total: this.storyTriggers.size,
      observed: this.observedElements.size,
      percentage: (this.observedElements.size / this.storyTriggers.size) * 100
    };
  }
}
```

### Scene Integration Example

```javascript
/**
 * Complete scene setup with rusted car prop
 */
class IntersectionScene extends BaseScene {
  async onLoad() {
    // Load base scene splat
    await this.loadSplat('/assets/splats/intersection.ply');

    // Setup static props
    this.staticProps = new StaticPropManager(this);

    // Load rusted car
    const rustedCar = await this.staticProps.loadProp(RUSTED_CAR);

    // Setup environmental storytelling
    this.envStorytelling = new EnvironmentalStorytelling(this);

    // Register the car as a story element
    this.envStorytelling.registerElement('rusted_car_intersection', {
      details: RUSTED_CAR.details,
      observationDistance: 3.0,
      observationAngle: 60
    });

    // Set player spawn
    this.player.spawn.set({ x: 0, y: 1.7, z: 5 });
  }

  onUpdate(deltaTime) {
    // Update static props (LOD, audio)
    this.staticProps.update(this.player.position);

    // Update environmental storytelling
    this.envStorytelling.update(
      this.player.position,
      this.player.forward
    );
  }

  onUnload() {
    // Cleanup
    this.staticProps.unloadProp('rusted_car_intersection');
  }
}
```

---

## How To Build A Scene Like This

### Step 1: Define the Prop's Story Purpose

Before modeling or coding, answer:

1. **What does this prop communicate?**
   - Rusted car = abandonment, neglect, passage of time

2. **Is it interactive or passive?**
   - Passive = player observes but doesn't interact
   - Interactive = player can examine, use, or affect it

3. **What questions should it raise?**
   - "Who left this car here?"
   - "Why didn't they come back for it?"
   - "What happened to the people?"

### Step 2: Create or Source the Asset

**Option A: Photogrammetry Scan**
- Scan a real rusted car (ideal for authenticity)
- Process into Gaussian Splat format
- Clean up unwanted elements (trash, bystanders)

**Option B: Procedural Generation**
- Model car in 3D software
- Apply rust/decay shaders
- Export and convert to splat

**Option C: Stock Asset**
- Purchase rusted vehicle model
- Verify license for commercial use
- Convert to required format

### Step 3: Configure Placement

```javascript
// Consider these factors:
const placement = {
  // Narrative: Does it make sense here?
  // - Cars belong on roads, not in living rooms
  // - Is the path blocking believable?

  // Visual: Is it visible from approach?
  // - Player should see it from distance
  // - Foreshadowing creates anticipation

  // Collision: Is the boundary clear?
  // - Player should understand they can't pass
  // - Invisible walls feel cheap

  // Atmosphere: Does it enhance mood?
  // - Lighting should cast interesting shadows
  // - Audio should respond to proximity
};
```

### Step 4: Setup Collision

```javascript
// Start simple, refine if needed
const collision = {
  // Simple box collision for most props
  type: 'box',
  halfExtents: { x: 2, y: 1, z: 4 },

  // Use compound for complex shapes
  type: 'compound',
  shapes: [
    // Main body
    { type: 'box', ... },
    // Cabin
    { type: 'box', ... }
  ]
};
```

**Collision Design Tips:**
- Slightly larger than visual = forgiving, feels better
- Exact to visual = precise, can feel finicky
- Use simple shapes when possible (performance)

### Step 5: Add Environmental Responses

```javascript
// Audio feedback
const audio = {
  ambient: 'wind_through_structure',
  impact: 'metal_thud',
  volume: 0.3
};

// Visual details
const details = {
  // What tells the story?
  stripped: ['parts_removed'],
  remaining: ['parts_still_there'],
  personal: ['items_left_behind']
};
```

### Step 6: Test and Iterate

**Playtest Questions:**
1. Do players understand they can't pass?
2. Do they notice the storytelling details?
3. Does it feel grounded or game-y?
4. Is the mood appropriately affected?

**Common Adjustments:**
- Move prop for better visibility
- Add/rem details for clarity
- Adjust collision boundary
- Modify audio for better feedback

---

## Variations For Your Game

### Variation 1: Forced Path Reveal

Instead of blocking, use the prop to **reveal** the correct path:

```javascript
// Car blocks obvious path, draws attention to hidden alley
const variation = {
  blockedPath: 'main_road',  // Expected route
  revealedPath: 'alley_way', // Actual route
  revealMethod: 'visual_line_of_sight'
};
```

### Variation 2: Temporal Progression

Car changes based on story progress:

```javascript
const temporalVariation = {
  states: {
    chapter1: {
      appearance: 'lightly_rusted',
      damage: 'minimal',
      message: 'Recently abandoned'
    },
    chapter3: {
      appearance: 'heavily_rusted',
      damage: 'stripped',
      message: 'Years have passed'
    }
  }
};
```

### Variation 3: Interactive Investigation

Allow detailed examination of specific elements:

```javascript
const interactiveVariation = {
  examinationPoints: [
    {
      id: 'baby_shoe',
      position: { x: 0.3, y: 0.1, z: 2.1 },
      prompt: 'Examine',
      onExamine: () => {
        showThought('Who leaves a child behind?');
      }
    },
    {
      id: 'open_trunk',
      position: { x: 0, y: 0.5, z: -2 },
      prompt: 'Look inside',
      onExamine: () => {
        showThought('Empty. Whatever was here is gone.');
      }
    }
  ]
};
```

### Variation 4: Horror Transformation

Static prop becomes threatening:

```javascript
const horrorVariation = {
  normalState: {
    appearance: 'rusted_car',
    threat: 'none',
    audio: 'wind'
  },
  nightmareState: {
    appearance: 'rusted_car_with_figures',
    threat: 'entities_hiding',
    audio: 'whispers_from_inside',
    triggers: {
      distance: 2.0,
      event: 'figures_emerge'
    }
  }
};
```

### Variation 5: Puzzle Integration

Prop is part of a puzzle:

```javascript
const puzzleVariation = {
  puzzleType: 'symbol_matching',
  carSymbols: ['hood_ornament', 'trunk_emblem', 'door_handle'],
  requiredItems: ['crowbar'],
  sequence: [
    'pry_trunk_open',
    'find_key_in_trunk',
    'unlock_building'
  ]
};
```

---

## Performance Considerations

### Gaussian Splat Optimization

```javascript
// LOD configuration for distant props
const lodConfig = {
  high: { distance: 10, pointFraction: 1.0 },
  medium: { distance: 30, pointFraction: 0.5 },
  low: { distance: 100, pointFraction: 0.1 }
};

// Frustum culling
if (!this.camera.frustum.contains(prop.boundingBox)) {
  prop.visible = false;
}
```

### Collision Optimization

```javascript
// Use simple shapes when possible
const collisionPerformance = {
  // Fast: Box, Sphere, Cylinder
  simpleShapes: ['box', 'sphere', 'cylinder'],

  // Slower: Convex hull
  convexMesh: 'moderate',

  // Slowest: Triangle mesh (use sparingly)
  concaveMesh: 'expensive'
};
```

### Audio Optimization

```javascript
// Only update spatial audio when player is nearby
const audioOptimization = {
  updateDistance: 50,  // meters
  stopCompletelyAt: 100  // meters
};

if (distance > audioOptimization.stopCompletelyAt) {
  ambientAudio.stop();
} else if (distance > audioOptimization.updateDistance) {
  ambientAudio.pause();
}
```

---

## Common Mistakes Beginners Make

### Mistake 1: Over-Blocking

```javascript
// BAD: Blocks too much of the playable space
const badBlocking = {
  position: { x: 0, y: 0, z: 0 },
  collision: {
    halfExtents: { x: 10, y: 5, z: 10 }  // Too large!
  }
};

// GOOD: Blocks only the intended path
const goodBlocking = {
  position: { x: 0, y: 0, z: -8 },
  collision: {
    halfExtents: { x: 1.8, y: 0.6, z: 4.0 }  // Car-sized
  }
};
```

### Mistake 2: No Visual Feedback

```javascript
// BAD: Player walks into invisible wall
const invisibleWall = {
  collision: true,
  visualMesh: null  // Nothing visible!
};

// GOOD: Player sees and understands the obstruction
const visibleObstruction = {
  collision: true,
  visualMesh: rustedCarMesh,
  impactSound: 'metal_thud'  // Audio feedback
};
```

### Mistake 3: Over-Storytelling

```javascript
// BAD: Too many explicit clues
const badStorytelling = {
  note: 'This car was left here in 1987 when the mill closed.',
  newspaper: 'Headline: MILL TO CLOSE PERMANENTLY',
  skeleton: 'Obviously dead driver inside'
};

// GOOD: Let player infer from subtle details
const goodStorytelling = {
  strippedParts: ['radio', 'engine'],
  personalItem: 'baby_shoe_on_back_seat',
  rustPattern: 'consistent_with_20_years_exposure'
  // Player connects the dots
};
```

### Mistake 4: Wrong Context

```javascript
// BAD: Modern electric car in 1970s setting
const anachronism = {
  make: 'Tesla Model 3',
  year: 2020,
  setting: { era: '1970s', location: 'rural_town' }
};

// GOOD: Era-appropriate vehicle
const eraAppropriate = {
  make: 'Ford Galaxie',
  year: 1968,
  condition: 'heavily_rusted',
  setting: { era: '1990s', location: 'abandoned_town' }
};
```

---

## Related Systems

- **Interactive Props** (Phone, Radio, Phonograph) - For when objects need interaction
- **Path Manager** - For handling navigation around obstructions
- **Physics System** - For collision and spatial queries
- **Audio System** - For ambient and impact sounds
- **Environmental Storytelling** - For tracking player observations

---

## References

- **Shadow Engine Documentation**: `docs/`
- **Gaussian Splat Rendering**: See *Rendering System*
- **Environmental Storytelling**: See *Game Design Principles*

---

**RALPH_STATUS:**
- **Status**: Rusted Car Scene documentation complete
- **Files Created**: `docs/generated/14-scene-case-studies/rusted-car-scene.md`
- **Related Documentation**: All Phase 14 scene case studies
- **Next Steps**: Review remaining Phase 14 tasks or proceed to Phase 15 (if applicable)
