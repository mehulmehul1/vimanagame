# Scene Case Study: The Phone Booth

## 🎬 Scene Overview

**Location**: Exterior alley, near the four-way intersection
**Narrative Context**: Player's first major interaction - a mysteriously ringing phone booth creates immediate intrigue
**Player Experience**: Tension, curiosity, anticipation - "What happens when I answer?"

The phone booth scene represents a masterclass in environmental storytelling and player guidance through audio-visual cues. It serves as the player's introduction to interactive objects in the game world and sets the tone for the mysterious, slightly off-kilter atmosphere that permeates the experience.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create tension and curiosity through an familiar object behaving unexpectedly.

**Why a Phone Booth?**
- **Archetypal Symbol**: Phone booths are deeply embedded in our cultural consciousness as places of important calls, emergencies, and supernatural encounters (horror trope)
- **Familiar Yet Wrong**: Everyone knows what a phone booth is, so when it rings unexpectedly in an empty space, it immediately feels "wrong"
- **Isolation**: A phone booth is literally an enclosed space - it creates a private moment in a public space, perfect for intimate, unsettling communication
- **Personal Address**: When the call is "for you," it breaks the fourth wall and creates direct engagement with the player

**Player Psychology**:
```
Familiar Object → Unexpected Event (Ringing) → Curiosity
     ↓
Approach → Audio Grows Louder → Tension Builds
     ↓
Interact → "This Call Is For You" → Unsettling Realization
     ↓
Pick Up → Physical Connection (Receiver in Hand) → Vulnerability
```

### Design Decisions

**1. Positioning for Discovery**
The phone booth is placed near the four-way intersection hub but slightly offset - visible enough to draw attention, not so obvious that it feels forced. This creates natural exploration rather than waypoint-following.

**2. Audio as Primary Cue**
The ringing sound is the primary attractor - it travels through the environment, guiding players by sound rather than visual markers. This respects player intelligence and creates organic discovery.

**3. Cord Physics for Tactile Realism**
The physics-based telephone cord serves multiple purposes:
- **Grounding**: Physical connection between player and object
- **Constraints**: Cord length naturally limits movement, creating subtle guidance
- **Realism**: Swinging, drooping cord adds life and tangibility

**4. Progressive Revelation**
The interaction unfolds in stages:
1. Distant ringing (curiosity)
2. Approach (tension)
3. Proximity trigger (anticipation)
4. Answer (connection)
5. Dialog delivery (narrative)
6. Drive-by interruption (escalation)

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
                    PLAZA (Spawn)
                         |
                         |
                    [FOUR-WAY INTERSECTION]
                         |
            ┌────────────┴────────────┐
            |                         |
         [ALLEY]                  [OTHER ZONES]
            |
         [PHONE BOOTH]
         ┌─────────┐
         │  ☎️     │ ← Positioned slightly off-center
         │  ████  │   Creates natural discovery
         │  ████  │   Not directly in path
         │  ████  │   Visible from intersection
         └─────────┘
            |
         [ALLEY CONTINUES]
```

**Sight Lines**: The phone booth is visible from the intersection but not directly in the main path, requiring player curiosity to discover it.

**Atmosphere**: Dimly lit, isolated from the main plaza, creating a sense of being "somewhere you shouldn't be."

### Player Path Design

1. **Spawn in Plaza** → Orient and explore
2. **Hear Ringing** → Natural audio attraction
3. **Spot Phone Booth** → Visual confirmation
4. **Approach** → Audio intensifies with proximity
5. **Interact** → Collider trigger activates dialog
6. **Answer** → Camera animation, receiver pickup
7. **Listen** → Dialog delivery while holding receiver
8. **Drive-by Event** → Car horn, shots fired, state progression

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the phone booth implementation, you should know:
- **Three.js Object3D hierarchy** - Parent-child relationships and scene graph
- **Three.js attach()** - Preserving world transform during reparenting
- **Rapier Physics joints** - Rope joints and chain simulations
- **Linear interpolation (lerp)** - Smooth value transitions
- **Quaternion slerp** - Smooth rotation interpolation
- **Kinematic vs Dynamic bodies** - Physics body types and their behaviors

### Core Systems Involved

```
PHONE BOOTH SYSTEM ARCHITECTURE:

┌─────────────────────────────────────────────────────────────┐
│                      PHONE BOOTH CONTROLLER                  │
│  - Manages receiver animation state                         │
│  - Coordinates with GameManager for state transitions       │
│  - Triggers audio and visual feedback                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  PHONE CORD   │  │  ANIMATION   │  │   AUDIO/VFX   │
│  - Physics    │  │  - Lerp      │  │  - Ringing    │
│  - Rope joints│  │  - Reparent  │  │  - Dialog     │
│  - Visual tube│  │  - Drop anim │  │  - State sync │
└───────────────┘  └───────────────┘  └───────────────┘
```

### PhoneBooth Controller Class

The `PhoneBooth` class (`../src/content/phonebooth.js`) orchestrates the entire phone booth interaction:

```javascript
class PhoneBooth {
  constructor(options = {}) {
    this.sceneManager = options.sceneManager;
    this.lightManager = options.lightManager;
    this.sfxManager = options.sfxManager;
    this.physicsManager = options.physicsManager;

    // Receiver animation state
    this.receiverLerp = null;      // Pickup animation
    this.receiverDropLerp = null;  // Drop animation
    this.receiver = null;          // THREE.Object3D reference
    this.phoneCord = null;         // PhoneCord instance

    // Configuration
    this.config = {
      receiverTargetPos: new THREE.Vector3(-0.3, 0, -0.3),
      receiverTargetRot: new THREE.Euler(-0.5, -0.5, -Math.PI / 2),
      receiverLerpDuration: 1.5,
      receiverLerpEase: (t) => 1 - Math.pow(1 - t, 3), // Cubic ease-out
      // Cord exists while state < OFFICE_INTERIOR
      cordCriteria: {
        currentState: { $lt: GAME_STATES.OFFICE_INTERIOR }
      }
    };
  }
}
```

### The Phone Cord System

The `PhoneCord` class (`../src/content/phoneCord.js`) is a reusable physics-based cord simulation:

```javascript
// Cord physics configuration
config = {
  cordSegments: 12,           // Number of chain links
  cordSegmentLength: 0.05,    // Length per segment (5cm)
  cordSegmentRadius: 0.002,   // Physics collider radius
  cordMass: 0.002,            // Mass per segment (2g)
  cordDamping: 8.0,           // Prevents wild swinging
  cordRigidSegments: 0,       // Fixed segments at attach point
  cordVisualRadius: 0.008,    // Visual tube radius
  cordCollisionGroup: 0x00040002  // Collides with environment only
}
```

**How the Cord Works**:

```
CORD PHYSICS STRUCTURE:

CordAttach (Kinematic Anchor)
    │
    │ Fixed Joint
    ▼
[Segment 0] [Segment 1] ... [Segment N] (Chain with Rope Joints)
    │
    │ Rope Joint
    ▼
Receiver Anchor (Kinematic - follows receiver mesh)
    │
    └──→ Receiver Mesh (Visual representation)

KEY CONCEPTS:
- Kinematic bodies: Follow their target, don't respond to forces
- Dynamic segments: React to gravity, swing naturally
- Rope joints: Limit max distance between segments
- Visual tube: Three.js TubeGeometry following physics points
```

### Receiver Reparenting Animation

The critical moment when player "answers" the phone:

```javascript
reparentReceiver() {
  // THREE.SceneUtils.attach preserves world position during reparenting
  this.receiver = this.sceneManager.reparentChild(
    "phonebooth",  // Current parent
    "Receiver",    // Child object name
    this.camera    // New parent
  );

  if (this.receiver) {
    // Start smooth lerp to target position
    this.startReceiverLerp();
  }
}

startReceiverLerp() {
  const startQuat = this.receiver.quaternion.clone();
  const targetQuat = new THREE.Quaternion().setFromEuler(
    this.config.receiverTargetRot
  );

  this.receiverLerp = {
    object: this.receiver,
    startPos: this.receiver.position.clone(),
    targetPos: this.config.receiverTargetPos,
    startQuat: startQuat,
    targetQuat: targetQuat,
    startScale: this.receiver.scale.clone(),
    targetScale: this.config.receiverTargetScale,
    duration: this.config.receiverLerpDuration,
    elapsed: 0,
  };
}
```

**Update Loop - Per-Frame Lerp**:

```javascript
updateReceiverLerp(dt) {
  if (!this.receiverLerp) return;

  this.receiverLerp.elapsed += dt;
  const t = Math.min(1, this.receiverLerp.elapsed / this.receiverLerp.duration);

  // Apply easing (cubic ease-out)
  const eased = this.config.receiverLerpEase(t);

  // Lerp position
  this.receiverLerp.object.position.lerpVectors(
    this.receiverLerp.startPos,
    this.receiverLerp.targetPos,
    eased
  );

  // Slerp rotation (quaternion for smooth interpolation)
  this.receiverLerp.object.quaternion.slerpQuaternions(
    this.receiverLerp.startQuat,
    this.receiverLerp.targetQuat,
    eased
  );

  // Complete animation
  if (t >= 1) {
    this.receiverLerp = null;
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Define the Emotional Goal

Before writing any code, answer:
- **What should the player feel?** (Curiosity, tension, unease)
- **What's the narrative purpose?** (First interaction, mystery setup)
- **What's the payoff?** (The call is "for you")

### Step 2: Choose Your Interaction Metaphor

Consider alternatives and why the phone booth works:
- **Mailbox**: Passive, one-way communication
- **Radio**: Broadcast, not personal
- **Phone booth**: Active, two-way, personal

The phone booth creates direct, personal communication - the call is "for YOU."

### Step 3: Position for Natural Discovery

```javascript
// Positioning considerations:
const positionConfig = {
  // Visible from hub but not in direct path
  offsetFromMainPath: true,

  // Audio range - should be hearable from intersection
  audioRange: 15,  // meters

  // Trigger collider size
  triggerRadius: 2,  // meters for interaction trigger

  // Sight lines
  visibleFrom: ["intersection", "plaza"],
  notVisibleFrom: ["distant_areas"]
};
```

### Step 4: Layer Audio for Atmosphere

```
AUDIO LAYERING:

Distance > 15m: Silence (no ring)
Distance 10-15m: Faint ring (curiosity)
Distance 5-10m: Clear ring (guidance)
Distance < 5m: Loud ring (tension)
In Trigger Zone: Ring stops → Voice begins

Implementation:
- Use SFX spatial audio for 3D positioning
- Attach sound to phonebooth object
- Set falloff for natural distance attenuation
```

### Step 5: Add Physical Feedback

The cord physics makes the interaction feel real:

```javascript
// When player answers:
1. Animate receiver moving to camera position
2. Attach receiver to camera (parent-child relationship)
3. Cord physics automatically follows (anchor updates)
4. Player movement now has physical constraint (cord length)
5. Cord swings naturally as player moves

This creates:
- Tactile connection to the object
- Subtle movement constraint
- Visual feedback (swinging cord)
```

### Step 6: Coordinate State Changes

```javascript
// Listen for state changes
gameManager.on("state:changed", (newState, oldState) => {
  // When leaving ANSWERED_PHONE, lock receiver position
  if (oldState.currentState === GAME_STATES.ANSWERED_PHONE) {
    this.stopReceiverLerp();
    this.receiverPositionLocked = true;
  }

  // When entering DRIVE_BY, drop receiver with physics
  if (newState.currentState === GAME_STATES.DRIVE_BY) {
    this.dropReceiverWithPhysics();
  }

  // Destroy cord after office transition
  if (!checkCriteria(newState, this.config.cordCriteria)) {
    this.phoneCord.destroy();
  }
});
```

---

## 🔧 Variations For Your Game

### Mystery Variation: Mailbox with Letter

```javascript
class MailboxInteraction {
  // Instead of ringing phone:
  // - Letter appears in mailbox when player approaches
  // - Opening mailbox triggers dialog
  // - Letter has physics (can be dropped, carried)
  // - Less immediate than phone, more passive discovery

  config = {
    triggerType: "proximity",  // Instead of audio
    interaction: "open_mailbox",
    pickupAnimation: "reach_and_grab",
    stateChange: "letter_acquired"
  };
}
```

### Horror Variation: TV That Turns On

```javascript
class HauntedTV {
  // TV turns on when player enters room
  // - Static noise builds tension
  // - Ghostly image appears
  // - No physical interaction needed
  // - Creates dread, not curiosity

  config = {
    triggerType: "zone_entry",
    visualEffect: "static",
    audioLoop: "white_noise",
    eventTiming: "after_delay"  // Build anticipation
  };
}
```

### Action Variation: Weapon Cache

```javascript
class WeaponPickup {
  - Weapon glows (bloom effect)
  - Audio cue (power-up sound)
  - Physical pickup animation
  - Immediate gameplay impact (new ability)

  config = {
    attractMode: "visual_bloom",
    interaction: "touch_to_acquire",
    feedback: "screen_flash + sound",
    gameplayEffect: "enable_shooting"
  };
}
```

### Puzzle Variation: Two-Handed Phone

```javascript
class TwoHandedPhoneInteraction {
  // Requires holding receiver in one hand
  // And dialing/pushing buttons with other hand
  // Creates physical puzzle constraint

  config = {
    primaryHand: "receiver",
    secondaryHand: "dial_numbers",
    constraint: "must_holding_receiver",
    successCondition: "correct_sequence_dialed"
  };
}
```

---

## Performance Considerations

```
PHONE BOOTH PERFORMANCE:

Cord Physics:
├── Segments: 12 (lightweight)
├── Joints: 13 (minimal CPU)
├── Update Rate: Every frame (required for smooth swing)
└── Impact: Negligible on desktop, minor on mobile

Visual Tube:
├── Recreated each frame (TubeGeometry)
├── Vertices: ~200 (depends on segments and radial segments)
├── Material: Standard (PBR)
└── Optimization: Cache geometry if cord is static

Audio:
├── Spatial audio: One source
├── Looping ring: Minimal overhead
└── Dialog: Streamed, not loaded entirely

Recommendations:
- Reduce cord segments on mobile (8 instead of 12)
- Consider lower-poly receiver model
- Use LOD for distant viewing
```

---

## Common Mistakes Beginners Make

### 1. Cord Too Short or Too Long

```javascript
// ❌ WRONG: Cord too restrictive
{ cordSegments: 6, cordSegmentLength: 0.03 }
// Total length: 18cm - player can barely move!

// ❌ WRONG: Cord too loose
{ cordSegments: 20, cordSegmentLength: 0.1 }
// Total length: 2m - cord drags on ground

// ✅ CORRECT: Balanced
{ cordSegments: 12, cordSegmentLength: 0.05 }
// Total length: 60cm - natural constraint without frustration
```

### 2. No Damping on Cord Physics

```javascript
// ❌ WRONG: Wild swinging cord
{ cordDamping: 0.1 }  // Too little damping
// Cord swings like a pendulum forever

// ✅ CORRECT: Natural settling
{ cordDamping: 8.0 }  // High damping
// Cord swings naturally then settles
```

### 3. Instant Reparenting Without Animation

```javascript
// ❌ WRONG: Jarring teleport
camera.attach(receiver);
// Receiver snaps to camera instantly

// ✅ CORRECT: Smooth animation
reparentReceiver();  // Preserves world position
startReceiverLerp();  // Smooth transition to target
// Natural, pleasing motion
```

### 4. Wrong Collision Groups

```javascript
// ❌ WRONG: Cord collides with player
{ cordCollisionGroup: 0x0003 }  // Collides with everything
// Cord pushes player around, frustrating

// ✅ CORRECT: Cord only collides with environment
{ cordCollisionGroup: 0x00040002 }  // Group 2, collides with Group 3 only
// Cord passes through player, hits walls
```

---

## Related Systems

- [PhysicsManager](../04-input-physics/physics-manager.md) - Physics simulation
- [ColliderManager](../04-input-physics/collider-manager.md) - Trigger zones
- [DialogManager](../05-media-systems/dialog-manager.md) - Dialog delivery
- [SFXManager](../05-media-systems/sfx-manager.md) - Spatial audio
- [AnimationManager](../06-animation/animation-manager.md) - Camera animations
- [GameState System](../02-core-architecture/game-state-system.md) - State management

---

## Source File Reference

**Primary Files**:
- `../src/content/phonebooth.js` - Phone booth controller (646 lines)
- `../src/content/phoneCord.js` - Physics-based cord simulation (634 lines)

**Key Classes**:
- `PhoneBooth` - Main interaction controller
- `PhoneCord` - Reusable cord physics module

**Dependencies**:
- Three.js (Object3D, Vector3, Quaternion, SceneUtils.attach)
- Rapier (RigidBodyDesc, JointData, ColliderDesc)
- GameManager (state store, event emitter)
- CriteriaHelper (state-based triggers)

---

## References

- [Three.js Object3D](https://threejs.org/docs/#api/en/core/Object3D) - Scene graph and hierarchy
- [Three.js attach()](https://threejs.org/docs/#api/en/core/Object3D.attach) - Preserving world transform during reparenting
- [Rapier Joints](https://rapier.rs/docs/user_guides/javascript/joints) - Physics joints and constraints
- [Quaternion Slerp](https://threejs.org/docs/#api/en/math/Quaternion.slerp) - Smooth rotation interpolation
- [Vector3 lerp](https://threejs.org/docs/#api/en/math/Vector3.lerp) - Linear interpolation

*Documentation last updated: January 12, 2026*
