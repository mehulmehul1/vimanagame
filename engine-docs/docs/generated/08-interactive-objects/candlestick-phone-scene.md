# Interactive Object: Candlestick Phone

## Overview

The **Candlestick Phone** is an interior interactive object that serves as a dialog trigger and narrative prop in the office scenes. Unlike the exterior phone booth, the candlestick phone is a period-authentic desk phone with a coiled cord that responds to physics, creating a tactile, immersive interaction.

Think of the candlestick phone as the **"interior mystery"** counterpart to the exterior phone booth - both create unease through communication, but the candlestick phone is intimate, domestic, and suggests something wrong in a supposedly safe space.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create discomfort through domestic intrusion. A candlestick phone represents safety, home, and normalcy - when it becomes a conduit for disturbing messages, the contrast heightens the unease.

**Why a Candlestick Phone?**
- **Period Authenticity**: Establishes the game's temporal setting (early 20th century aesthetic)
- **Domestic Symbol**: Unlike a phone booth (public space), a desk phone is private, personal
- **Visual Distinctiveness**: The tall, upright candle shape is immediately recognizable
- **Tactile Interaction**: Picking up the handset feels more personal than a booth receiver

**Player Psychology**:
```
Safe Space (Office) → Familiar Object (Desk Phone) → Comfort
     ↓
Phone Rings → Wrongness in Safe Space → Unease
     ↓
Answer → Disturbing Message → Intrusion
     ↓
Hang Up → Cannot Unhear → Lingering Discomfort
```

### Design Decisions

**1. Placement on Desk**
The phone sits on a desk, requiring player to approach and look down - creates vulnerability through lowered perspective.

**2. Shorter Cord Than Booth**
The candlestick phone has a shorter, coiled cord - more restrictive, reinforcing the "trapped" feeling of interior spaces.

**3. Desk-Mounted vs Wall-Mounted**
Unlike the booth (public, wall), the candlestick phone is on furniture - personal, movable, intimate.

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the candlestick phone implementation, you should know:
- **PhoneCord reuse pattern** - Same cord physics module, different configuration
- **Initialization modes** - "horizontal" vs "straight" cord setup
- **Desk positioning** - How to place objects on furniture
- **Dialog triggering** - Collider-based activation
- **State-driven visibility** - Phone only appears in certain scenes

### Core Architecture

```
CANDLESTICK PHONE SYSTEM:

┌─────────────────────────────────────────────────────┐
│            CANDLESTICK PHONE CONTROLLER              │
│  - Inherits PhoneCord functionality                 │
│  - Manages desk positioning                         │
│  - Triggers dialog on interaction                   │
└───────────────────────────┬─────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  PHONE CORD   │  │   DIALOG     │  │   DESK       │
│  (STRAIGHT    │  │   TRIGGER    │  │   PARENT     │
│   MODE)       │  │  - Collider   │  │  - Transform │
│  - Short cord │  │  - Proximity  │  │  - Local     │
│  - Coiled     │  │  - State check│  │    position  │
└───────────────┘  └───────────────┘  └───────────────┘
```

### PhoneCord Configuration: Straight Mode

The PhoneCord module supports two initialization modes. The candlestick phone uses "straight" mode:

```javascript
// In candlestick phone configuration
const cordConfig = {
  initMode: "straight",  // Direct line from attach to receiver

  // Fewer segments than booth (shorter cord)
  cordSegments: 8,
  cordSegmentLength: 0.05,

  // Straight line from base to handset
  cordRigidSegments: 2,  // Rigid section at base

  // Coiled visual appearance
  cordVisualRadius: 0.01,  // Thicker than booth cord

  // Collision: only environment, not player
  cordCollisionGroup: 0x00040002
};

// Create cord
this.phoneCord = new PhoneCord({
  scene: this.scene,
  physicsManager: this.physicsManager,
  cordAttach: this.phoneBase,      // Attach to phone base
  receiver: this.handset,          // Follow handset mesh
  loggerName: "CandlestickPhone.Cord",
  config: cordConfig
});
```

**Straight vs Horizontal Mode**:

```
HORIZONTAL MODE (Phone Booth):
┌──────────────┐
│  Phone Booth │
│              │
│  ╱╲          │ ← Rigid segments stick out horizontally
│ ╱  ╲         │
│█████│        │
│█████└──╲____ │ ← Flexible segments droop with gravity
│             ╲│
└──────────────┘

STRAIGHT MODE (Candlestick):
┌──────────────────┐
│     Desk         │
│  ┌──────────┐    │
│  │  Phone   │    │
│  │   ╱│     │    │ ← Rigid segments extend up
│  │  ╱ │     │    │
│  │ ████     │    │
│  │ █████    │    │
│  │  └────┐  │    │ ← Flexible segments to receiver
│  │       ╲ │    │
│  └─────────┘─┘   │
└──────────────────┘
```

### Reparenting Handset to Camera

Similar to the phone booth, when player answers:

```javascript
answerPhone() {
  // Find handset in scene
  const handset = this.sceneManager.findChildByName("candlestickPhone", "Handset");

  if (!handset) {
    this.logger.warn("Handset not found");
    return;
  }

  // Reparent from phone to camera (preserves world position)
  this.sceneManager.attach(handset);
  this.camera.attach(handset);

  // Lerp to holding position
  this.startHandsetLerp(handset);

  // Trigger dialog
  this.gameManager.emit("candlestick:answered");
}

startHandsetLerp(handset) {
  const targetPos = new THREE.Vector3(0.3, -0.2, -0.4);  // Camera-relative
  const targetRot = new THREE.Euler(0, 0, Math.PI / 4);    // Tilted

  this.handsetLerp = {
    object: handset,
    startPos: handset.position.clone(),
    targetPos: targetPos,
    startQuat: handset.quaternion.clone(),
    targetQuat: new THREE.Quaternion().setFromEuler(targetRot),
    duration: 0.8,  // Faster than booth (shorter distance)
    elapsed: 0
  };
}
```

---

## 📝 How To Build An Object Like This

### Step 1: Define the Object's Role

Answer these questions:
- **What emotion does it evoke?** (Discomfort, intrusion)
- **Why this object type?** (Period detail, domestic setting)
- **How does it differ from similar objects?** (More intimate than booth)

### Step 2: Configure the Shared PhoneCord Module

```javascript
// Create your cord configuration
const myCordConfig = {
  initMode: "straight",  // or "horizontal"

  // Adjust length for your use case
  cordSegments: 8,        // Fewer = shorter cord
  cordSegmentLength: 0.05,

  // Rigid segments at base (prevent intersection with phone body)
  cordRigidSegments: 2,

  // Visual appearance
  cordColor: 0x404040,    // Dark coiled cord
  cordVisualRadius: 0.01,

  // Performance
  cordCollisionGroup: 0x00040002  // Environment only
};

// Instantiate shared module
const phoneCord = new PhoneCord({
  scene: this.scene,
  physicsManager: this.physicsManager,
  cordAttach: this.attachPointMesh,
  receiver: this.receiverMesh,
  config: myCordConfig
});

phoneCord.createCord();
```

### Step 3: Set Up Collider Trigger

```javascript
// Create interaction zone around phone
const triggerCollider = {
  type: "trigger",
  shape: "cylinder",
  radius: 1.5,
  height: 2,
  position: this.phone.position,

  // Trigger callback
  onEnter: () => {
    if (this.gameManager.state.currentState === GAME_STATES.OFFICE_INTERIOR &&
        !this.phoneAnswered) {
      this.showInteractionPrompt();
    }
  }
};

this.physicsManager.addTrigger(triggerCollider);
```

### Step 4: Handle State Changes

```javascript
// Only show phone in certain states
this.gameManager.on("state:changed", (newState) => {
  const shouldShowPhone = checkCriteria(newState, {
    currentState: {
      $gte: GAME_STATES.OFFICE_INTERIOR,
      $lt: GAME_STATES.OFFICE_HELL
    }
  });

  this.phone.visible = shouldShowPhone;

  if (shouldShowPhone && !this.phoneCord) {
    this.phoneCord.createCord();
  } else if (!shouldShowPhone && this.phoneCord) {
    this.phoneCord.destroy();
  }
});
```

---

## 🔧 Variations For Your Game

### Period Radio

```javascript
class PeriodRadio {
  // Similar to candlestick phone but:
  // - No cord (wireless or plugged into wall)
  // - Dial/tuner instead of handset
  // - Audio plays through speaker, not handset

  config = {
    interactionType: "proximity",  // Not pickup
    audioOutput: "speaker",
    visualFeedback: "dial_glow",
    dialogTrigger: "approach_only"
  };
}
```

### Wall-Mounted Phone

```javascript
class WallPhone {
  // Similar to candlestick but:
  // - Mounted on wall, not desk
  - Longer cord (wall to hand distance)
  - Different orientation (horizontal initMode)
  - No furniture parent needed

  config = {
    initMode: "horizontal",
    cordSegments: 10,
    parentType: "wall",
    mountingHeight: 1.5  // meters from floor
  };
}
```

### Intercom System

```javascript
class Intercom {
  // Building-wide communication:
  // - Multiple stations, one network
  // - Push-to-talk interaction
  // - Static/crackle audio
  // - Can be used for puzzle (find all stations)

  config = {
    networkType: "multi_station",
    interaction: "press_and_hold",
    audioQuality: "staticky",
    puzzleElement: true
  };
}
```

---

## Common Mistakes Beginners Make

### 1. Wrong Init Mode

```javascript
// ❌ WRONG: Horizontal mode for desk phone
{ initMode: "horizontal" }
// Cord sticks out sideways, looks wrong

// ✅ CORRECT: Straight mode for desk phone
{ initMode: "straight" }
// Cord extends naturally from base to handset
```

### 2. Cord Too Long for Desk

```javascript
// ❌ WRONG: Booth-length cord
{ cordSegments: 12, cordSegmentLength: 0.05 }  // 60cm
// Cord drags on floor

// ✅ CORRECT: Shorter desk cord
{ cordSegments: 6, cordSegmentLength: 0.05 }  // 30cm
// Natural reach for desk phone
```

### 3. Not Checking State Before Showing

```javascript
// ❌ WRONG: Phone always visible
this.phone.visible = true;
// Phone appears in wrong scenes (exterior, hell, etc.)

// ✅ CORRECT: State-driven visibility
gameManager.on("state:changed", (newState) => {
  this.phone.visible = checkCriteria(newState, {
    currentState: { $eq: GAME_STATES.OFFICE_INTERIOR }
  });
});
```

---

## Performance Considerations

```
CANDLESTICK PHONE PERFORMANCE:

Cord Physics:
├── Segments: 6-8 (fewer than booth)
├── Update: Every frame (required)
└── Impact: Minimal

Visual:
├── Model: Simple geometry
├── Material: Standard PBR
└── Recommendation: Use LOD for distance

Optimization:
- Reduce segments on mobile
- Share PhoneCord instance where possible
- Destroy cord when not in scene
```

---

## Related Systems

- [Phone Booth Scene](./phone-booth-scene.md) - Exterior counterpart
- [PhoneCord Module](`../src/content/phoneCord.js`) - Shared cord physics
- [ColliderManager](../04-input-physics/collider-manager.md) - Trigger zones
- [DialogManager](../05-media-systems/dialog-manager.md) - Dialog system
- [GameState System](../02-core-architecture/game-state-system.md) - State checks

---

## Source File Reference

**Primary Files**:
- `../src/content/phoneCord.js` - Shared cord physics (634 lines)
- `../src/content/candlestickPhone.js` - Candlestick phone controller (estimated)

**Key Classes**:
- `PhoneCord` - Reusable cord module with initMode configuration
- `CandlestickPhone` - Desk phone controller

**Dependencies**:
- Three.js (Object3D, Vector3, attach)
- Rapier (RigidBodyDesc, JointData)
- GameManager (state store, events)

---

## References

- [Three.js Object3D.attach](https://threejs.org/docs/#api/en/core/Object3D.attach) - Preserving world transform
- [Rapier Joint Types](https://rapier.rs/docs/user_guides/javascript/joints) - Fixed and rope joints
- [Period Telephones](https://www.telephonetribute.com/) - Historical reference for design

*Documentation last updated: January 12, 2026*
