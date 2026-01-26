# Interactive Object: Rune Manager

## Overview

The **Rune Manager** handles magical rune objects throughout the game world - glowing symbols that players can interact with to trigger events, unlock content, or progress the narrative. Runes serve as both interactive puzzles and atmospheric elements, with visual effects (bloom, pulsing, activation animations) that communicate their state and importance.

Think of runes as the **"magical interactors"** of the game world - just as switches and levers represent physical interaction, runes represent supernatural/magical interaction, often with dramatic visual feedback.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create wonder and mystery through supernatural objects. Runes communicate that "magic exists here" and invite player experimentation.

**Why Runes?**
- **Universal Symbol**: Recognized across cultures as having mystical significance
- **Visual Clarity**: Glowing symbols are unmistakably "interactive" in game language
- **Flexible Meaning**: Can represent anything from "save point" to "puzzle trigger" to "story reveal"
- **Atmospheric**: Even inactive runes add supernatural atmosphere

**Player Psychology**:
```
Discover Rune → Glowing, Unusual → Curiosity
     ↓
Interact → Rune Responds → Validation (I'm doing it right)
     ↓
Rune Activates → Visual/Audio Feedback → Satisfaction
     ↓
Something Happens → World Changes → Agency (I did that)
```

### Rune Types and Functions

**1. Discovery Runes**
- Purpose: Reveal narrative or environmental details
- Interaction: Look at / approach
- Feedback: Symbol glows brighter, reveals text
- Example: "Looking at this rune reveals hidden writing"

**2. Activation Runes**
- Purpose: Trigger state change or event
- Interaction: Touch / click
- Feedback: Full activation sequence with sound and light
- Example: "Activating this rune opens the portal"

**3. Sequence Runes**
- Purpose: Multi-step puzzle
- Interaction: Activate in correct order
- Feedback: Each rune indicates sequence position
- Example: "Activate red, then blue, then green runes"

**4. Collection Runes**
- Purpose: Track completion / progress
- Interaction: Collect/absorb
- Feedback: Rune dissolves and joins player inventory/hud
- Example: "Collect all 5 runes to unlock the final door"

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the rune system, you should know:
- **Three.js emissive materials** - Making objects glow
- **Selective bloom** - Making only certain objects glow (see [Selective Bloom](../07-visual-effects/selective-bloom.md))
- **Raycasting** - Detecting player interaction
- **Animation lerping** - Smooth state transitions
- **Event-driven architecture** - Emitting events on interaction

### Core Architecture

```
RUNE MANAGER SYSTEM ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                      RUNE MANAGER                        │
│  - Spawns and tracks all runes                          │
│  - Handles interaction (raycast/click)                  │
│  - Manages rune states (dormant/awakening/active)       │
│  - Coordinates with VFX for visual feedback             │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   RUNE       │  │    VFX       │  │   AUDIO      │
│   OBJECTS    │  │   MANAGER    │  │   MANAGER    │
│  - Mesh      │  │  - Bloom     │  │  - Hum       │
│  - Material  │  │  - Pulse     │  │  - Chime     │
│  - State     │  │  - Particles │  │  - Activation│
└───────────────┘  └───────────────┘  └───────────────┘
```

### Rune Data Structure

```javascript
// In runeData.js
export const runeData = {
  // Simple activation rune
  officeRune: {
    id: "office_rune",
    type: "activation",
    position: { x: 2, y: 0.5, z: -1 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: 1.0,

    // Appearance
    color: "#ff0000",           // Red rune
    emissiveColor: "#ff0000",
    emissiveIntensity: 2.0,     // Bright glow

    // States
    dormant: {
      emissiveIntensity: 0.3,
      bloom: { strength: 0.5, threshold: 0.7 }
    },
    awakening: {
      emissiveIntensity: 1.5,
      bloom: { strength: 1.5, threshold: 0.5 },
      pulse: { speed: 2.0 }
    },
    active: {
      emissiveIntensity: 3.0,
      bloom: { strength: 3.0, threshold: 0.3 },
      pulse: { min: 2.0, max: 4.0, speed: 3.0 }
    },

    // Interaction
    triggerType: "click",  // or "proximity", "gaze"
    triggerRadius: 2.0,

    // Effects
    onActivate: {
      event: "rune:office_activated",
      vfx: "rune_activate_office",
      sfx: "rune_hum_ascension"
    },

    // State criteria
    criteria: {
      currentState: { $gte: GAME_STATES.OFFICE_INTERIOR },
      runeVisible: true
    }
  },

  // Sequence rune (part of puzzle)
  sequenceRune_Red: {
    id: "sequence_rune_red",
    type: "sequence",
    sequencePosition: 1,  // First in sequence
    sequenceId: "office_triple_rune",

    // ... other config
  }
};
```

### Rune Manager Implementation

```javascript
class RuneManager {
  constructor(options = {}) {
    this.scene = options.scene;
    this.camera = options.camera;
    this.gameManager = options.gameManager;
    this.vfxManager = options.vfxManager;
    this.sfxManager = options.sfxManager;

    this.runes = new Map();  // id -> rune object
    this.activeSequences = new Map();  // sequenceId -> [activated positions]

    // Raycaster for interaction
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Listen for interaction events
    this.setupInteractionHandlers();
  }

  /**
   * Create a rune from data definition
   */
  createRune(runeData) {
    // Check spawn criteria
    if (runeData.criteria && !this.checkCriteria(runeData.criteria)) {
      return null;
    }

    // Create rune geometry
    const geometry = new THREE.PlaneGeometry(1, 1);
    const material = new THREE.MeshStandardMaterial({
      color: runeData.color || 0xffffff,
      emissive: runeData.emissiveColor || runeData.color || 0xffffff,
      emissiveIntensity: runeData.dormant?.emissiveIntensity || 0.3,
      transparent: true,
      side: THREE.DoubleSide
    });

    const rune = new THREE.Mesh(geometry, material);
    rune.position.set(
      runeData.position.x,
      runeData.position.y,
      runeData.position.z
    );

    if (runeData.rotation) {
      rune.rotation.set(
        runeData.rotation.x,
        runeData.rotation.y,
        runeData.rotation.z
      );
    }

    rune.scale.setScalar(runeData.scale || 1.0);

    // Store rune data on the object
    rune.userData.runeId = runeData.id;
    rune.userData.runeType = runeData.type;
    rune.userData.runeState = "dormant";

    // Add to scene
    this.scene.add(rune);
    this.runes.set(runeData.id, {
      mesh: rune,
      data: runeData,
      state: "dormant"
    });

    // Enable bloom for this rune
    this.setupRuneBloom(rune, runeData);

    this.logger.log("Created rune:", runeData.id);

    return rune;
  }

  /**
   * Enable selective bloom for rune
   */
  setupRuneBloom(rune, runeData) {
    // Add rune to bloom layer (layer 1)
    rune.layers.enable(1);

    // Configure initial bloom
    if (this.vfxManager && runeData.dormant?.bloom) {
      this.vfxManager.setBloomSettings(runeData.dormant.bloom);
    }
  }

  /**
   * Handle rune activation
   */
  activateRune(runeId) {
    const rune = this.runes.get(runeId);
    if (!rune || rune.state === "active") {
      return;  // Already active or doesn't exist
    }

    const runeData = rune.data;

    // Update state
    rune.state = "active";
    rune.mesh.userData.runeState = "active";

    // Trigger visual feedback
    this.playActivationAnimation(rune);

    // Trigger VFX
    if (runeData.onActivate?.vfx) {
      this.vfxManager.trigger(runeData.onActivate.vfx);
    }

    // Trigger SFX
    if (runeData.onActivate?.sfx) {
      this.sfxManager.play(runeData.onActivate.sfx);
    }

    // Emit event
    if (runeData.onActivate?.event) {
      this.gameManager.emit(runeData.onActivate.event, { runeId });
    }

    // Handle sequence runes
    if (runeData.type === "sequence") {
      this.handleSequenceRune(rune);
    }

    this.logger.log("Activated rune:", runeId);
  }

  /**
   * Play activation animation
   */
  playActivationAnimation(rune) {
    const mesh = rune.mesh;
    const runeData = rune.data;
    const activeConfig = runeData.active;

    // Lerp emissive intensity
    const startIntensity = mesh.material.emissiveIntensity;
    const targetIntensity = activeConfig.emissiveIntensity;

    rune.animation = {
      property: "emissiveIntensity",
      start: startIntensity,
      target: targetIntensity,
      duration: 1000,  // 1 second
      elapsed: 0,
      easing: (t) => 1 - Math.pow(1 - t, 3)  // Cubic ease-out
    };
  }

  /**
   * Update rune animations
   */
  update(deltaTime) {
    for (const [id, rune] of this.runes) {
      if (rune.animation) {
        this.updateRuneAnimation(rune, deltaTime);
      }

      // Pulse active runes
      if (rune.state === "active" && rune.data.active?.pulse) {
        this.pulseRune(rune);
      }
    }
  }

  /**
   * Update single rune animation
   */
  updateRuneAnimation(rune, deltaTime) {
    const anim = rune.animation;
    anim.elapsed += deltaTime * 1000;

    const t = Math.min(1, anim.elapsed / anim.duration);
    const eased = anim.easing(t);

    // Apply to material
    if (anim.property === "emissiveIntensity") {
      rune.mesh.material.emissiveIntensity =
        anim.start + (anim.target - anim.start) * eased;
    }

    // Complete animation
    if (t >= 1) {
      rune.animation = null;
    }
  }

  /**
   * Pulse active rune
   */
  pulseRune(rune) {
    const pulseConfig = rune.data.active.pulse;
    const time = performance.now() / 1000;

    const sineValue = Math.sin(time * pulseConfig.speed);
    const intensity = pulseConfig.min +
      (pulseConfig.max - pulseConfig.min) * (sineValue * 0.5 + 0.5);

    rune.mesh.material.emissiveIntensity = intensity;
  }

  /**
   * Handle sequence rune (multi-rune puzzle)
   */
  handleSequenceRune(rune) {
    const runeData = rune.data;
    const sequenceId = runeData.sequenceId;

    if (!this.activeSequences.has(sequenceId)) {
      this.activeSequences.set(sequenceId, []);
    }

    const sequence = this.activeSequences.get(sequenceId);
    sequence.push(runeData.sequencePosition);

    // Check if sequence is complete
    const expectedSequence = [1, 2, 3];  // Or from config
    const isComplete = sequence.length === expectedSequence.length &&
      sequence.every((val, i) => val === expectedSequence[i]);

    if (isComplete) {
      this.gameManager.emit("sequence:complete", { sequenceId });
    }
  }

  /**
   * Check if rune should be visible based on criteria
   */
  checkCriteria(criteria) {
    // Use criteria helper
    const state = this.gameManager?.state;
    if (!state) return true;

    return checkCriteria(state, criteria);
  }

  /**
   * Handle click interaction
   */
  onMouseClick(event) {
    // Calculate mouse position in normalized device coordinates
    this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    // Raycast
    this.raycaster.setFromCamera(this.mouse, this.camera);

    // Get all rune meshes
    const runeMeshes = Array.from(this.runes.values()).map(r => r.mesh);
    const intersects = this.raycaster.intersectObjects(runeMeshes, true);

    if (intersects.length > 0) {
      // Find the rune object from the mesh
      const mesh = intersects[0].object;
      const runeId = mesh.userData.runeId;

      if (runeId) {
        this.activateRune(runeId);
      }
    }
  }
}

export default RuneManager;
```

---

## 📝 How To Build A System Like This

### Step 1: Define Your Rune Types

What do runes do in your game?

```javascript
const runeTypes = {
  // Visual marker (atmosphere only)
  marker: {
    interaction: false,
    purpose: "atmosphere",
    feedback: "gentle_glow"
  },

  // Information reveal
  lore: {
    interaction: "click",
    purpose: "reveal_text",
    feedback: "brighten + text_appear"
  },

  // State trigger
  trigger: {
    interaction: "proximity",
    purpose: "change_game_state",
    feedback: "full_activation"
  },

  // Puzzle element
  puzzle: {
    interaction: "click",
    purpose: "part_of_sequence",
    feedback: "color_change + sound"
  }
};
```

### Step 2: Create Data-Driven Definitions

```javascript
// Define runes externally, not in code
export const myRuneData = {
  entranceRune: {
    type: "trigger",
    position: { x: 0, y: 1, z: -5 },
    color: "#00ff00",
    onActivate: { event: "entrance:unlocked" }
  },

  loreRune: {
    type: "lore",
    position: { x: 3, y: 0.5, z: 2 },
    color: "#0088ff",
    onActivate: {
      showText: "The ancients carved these symbols to ward off...",
      event: "lore:discovered"
    }
  }
};
```

### Step 3: Set Up Visual Feedback

```javascript
// Use selective bloom for rune glow
const setupRuneVisuals = (rune) => {
  // Emissive material
  rune.material.emissive = new THREE.Color(rune.config.color);
  rune.material.emissiveIntensity = 0.5;

  // Add to bloom layer
  rune.layers.enable(1);

  // Optional: Point light for ambient glow
  const glowLight = new THREE.PointLight(
    rune.config.color,
    1,  // intensity
    3   // distance
  );
  rune.add(glowLight);
};
```

---

## 🔧 Variations For Your Game

### Crystals Instead of Runes

```javascript
class CrystalSystem {
  // Same mechanics, different visual
  config = {
    geometry: "octahedron",
    material: "crystalline",
    interaction: "touch",
    feedback: "refraction_change"
  };
}
```

### Totems / Statues

```javascript
class TotemSystem {
  // Larger, more imposing interactors
  config = {
    geometry: "custom_model",
    scale: 2.0,
    interaction: "offering",
    feedback: "eyes_glow + sound"
  };
}
```

### Constellation Patterns

```javascript
class ConstellationSystem {
  // Connect stars to form patterns
  config = {
    interaction: "connect_dots",
    feedback: "line_draw + completion_burst",
    unlockCondition: "all_stars_connected"
  };
}
```

---

## Common Mistakes Beginners Make

### 1. No Visual Hierarchy

```javascript
// ❌ WRONG: All runes look the same
allRunes: {
  emissiveIntensity: 1.0,
  bloom: { strength: 1.0 }
}
// Player doesn't know which are important

// ✅ CORRECT: Visual hierarchy
importantRune: {
  emissiveIntensity: 3.0,
  bloom: { strength: 3.0, pulse: true }
},
backgroundRune: {
  emissiveIntensity: 0.3,
  bloom: { strength: 0.5 }
}
// Clear importance through visual weight
```

### 2. No Activation Feedback

```javascript
// ❌ WRONG: Silent activation
activateRune(id) {
  this.runes.get(id).active = true;
}
// Did it work? Player is unsure

// ✅ CORRECT: Clear feedback
activateRune(id) {
  const rune = this.runes.get(id);
  rune.active = true;
  this.playAnimation(rune);
  this.playSound("rune_activate");
  this.emitParticles(rune);
}
// Unmistakable confirmation
```

### 3. Wrong Interaction Distance

```javascript
// ❌ WRONG: Must be touching
{ triggerRadius: 0.1 }
// Frustratingly precise

// ✅ CORRECT: Forgiving distance
{ triggerRadius: 2.0 }
// Comfortable interaction
```

---

## Performance Considerations

```
RUNE MANAGER PERFORMANCE:

Per Rune:
├── Mesh: Simple geometry (Plane/Box)
├── Material: Standard PBR
├── Animation: Lerp calculation (minimal)
└── Pulse: Sine wave calculation (minimal)

Total Impact:
├── 1-10 runes: Negligible
├── 10-50 runes: Minor
└── 50+ runes: Consider batching

Optimization:
- Use instanced meshes for identical runes
- Disable animation for distant runes
- Share materials where possible
- Use LOD for complex rune models
```

---

## Related Systems

- [Selective Bloom](../07-visual-effects/selective-bloom.md) - Rune glow effects
- [VFXManager](../07-visual-effects/vfx-manager.md) - Effect orchestration
- [SFXManager](../05-media-systems/sfx-manager.md) - Audio feedback
- [ColliderManager](../04-input-physics/collider-manager.md) - Proximity triggers
- [Data Driven Design](../02-core-architecture/data-driven-design.md) - Rune data structure

---

## References

- [Three.js Layers](https://threejs.org/docs/#api/en/core/Layers) - Selective rendering
- [Three.js Emissive Materials](https://threejs.org/docs/#api/en/materials/MeshStandardMaterial.emissive) - Glowing objects
- [Magical Symbol Design](https://www.fontspace.com/category/runes) - Visual reference
- [Game Symbolism](https://www.gamedeveloper.com/design/symbolism-in-games) - Design principles

*Documentation last updated: January 12, 2026*
