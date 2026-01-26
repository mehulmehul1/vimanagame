# Manager-Based Architecture Pattern - First Principles Guide

## Overview

The **Manager-Based Architecture** is a design pattern where each major system in the game is managed by a dedicated "Manager" class. Think of it like organizing a large company - instead of one person doing everything, you have specialized managers for different departments (HR, Finance, Operations, etc.).

In the Shadow Engine, each manager is responsible for one specific area:
- **GameManager** - Overall state and coordination
- **SceneManager** - 3D content loading/unloading
- **DialogManager** - Spoken dialogue
- **MusicManager** - Background music
- **InputManager** - Player input
- **PhysicsManager** - Physics simulation
- And many more...

## What You Need to Know First

Before understanding the manager pattern, you should know:
- **JavaScript classes** - How to create and use classes
- **Separation of concerns** - Different code for different purposes
- **Event-driven programming** - Reacting to events
- **Basic game loops** - Update, render, repeat
- **What a "singleton" is** - A class with only one instance

### Quick Refresher: Classes

```javascript
// A simple class
class Dog {
  constructor(name) {
    this.name = name;
  }

  bark() {
    console.log(`${this.name} says: Woof!`);
  }
}

const fido = new Dog("Fido");
fido.bark(); // "Fido says: Woof!"
```

---

## What Problem Does the Manager Pattern Solve?

### The Alternative: One Giant Class

Without managers, you might have one massive "Game" class:

```javascript
// ❌ BAD: Everything in one place
class Game {
  update() {
    this.updateInput();
    this.updatePhysics();
    this.updateDialog();
    this.updateMusic();
    this.updateVideo();
    this.updateGraphics();
    this.updateUI();
    this.updateAnimations();
    this.updateEffects();
    // ... 100+ more things
  }
}
```

**Problems with this approach:**
- **Impossible to understand** - Too much code in one place
- **Hard to modify** - Changing one thing might break something else
- **Can't reuse** - Can't use the music system in another project
- **Difficult to test** - Can't test one system in isolation

### The Manager Approach: Organized and Modular

```javascript
// ✅ GOOD: Each system has its own manager
class Game {
  constructor() {
    this.inputManager = new InputManager();
    this.physicsManager = new PhysicsManager();
    this.dialogManager = new DialogManager();
    this.musicManager = new MusicManager();
    // ... etc
  }

  update() {
    this.inputManager.update();
    this.physicsManager.update();
    this.dialogManager.update();
    this.musicManager.update();
    // Clear delegation!
  }
}
```

---

## Core Concepts of the Manager Pattern

### 1. Single Responsibility Principle

Each manager has **one job** and does it well:

| Manager | Responsibility |
|---------|----------------|
| GameManager | Store state, emit events, coordinate |
| SceneManager | Load/unload 3D objects, splats |
| DialogManager | Play dialog tracks, show captions |
| MusicManager | Play music, handle crossfades |
| InputManager | Capture keyboard, mouse, gamepad, touch |
| PhysicsManager | Run physics simulation |
| CharacterController | Handle player movement |
| VideoManager | Play video with alpha channel |
| SFXManager | Play sound effects with 3D positioning |

### 2. Managers Communicate via Events

Managers don't call each other directly. Instead, they **emit events** and **listen to events**:

```
┌─────────────┐                           ┌─────────────┐
│ GameManager │                           │ DialogMgr   │
│             │   emit("state:changed")   │             │
└──────┬──────┘ ───────────────────────▶ └──────┬──────┘
       │                                         │
       │                                   Listens for
       │                                   "state:changed"
       │                                         │
       ▼                                         ▼
┌─────────────┐                           ┌─────────────┐
│ MusicManager │                          │ VideoManager│
└──────┬──────┘                           └──────┬──────┘
       │                                         │
       │                                   Listens for
       │                                   "state:changed"
       ▼                                         ▼
    Plays music                              Plays video
```

**Why use events instead of direct calls?**
- **Loose coupling** - Managers don't need to know about each other
- **Flexible** - Easy to add new listeners
- **Debuggable** - Can log all events to see what's happening

### 3. Centralized State in GameManager

All game state lives in one place: `GameManager.state`

```
┌─────────────────────────────────────────────────────────┐
│                    GameManager.state                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │ currentState: 14                               │   │
│  │ controlEnabled: true                           │   │
│  │ isViewmasterEquipped: false                    │   │
│  │ currentZone: "office"                          │   │
│  │ ... (45+ state properties)                     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Any manager can READ this state                        │
│  Only GameManager can WRITE this state                 │
│  (via setState() method)                               │
└─────────────────────────────────────────────────────────┘
```

---

## Manager Lifecycle

### Initialization Phase

```javascript
// 1. Create all managers
const gameManager = new GameManager();
const sceneManager = new SceneManager();
const dialogManager = new DialogManager();
const musicManager = new MusicManager();
// ... etc

// 2. Initialize GameManager with references to all managers
await gameManager.initialize({
  sceneManager,
  dialogManager,
  musicManager,
  // ... all other managers
});

// 3. Each manager sets up event listeners
dialogManager.listenTo(gameManager);
musicManager.listenTo(gameManager);
// ... etc
```

### The Game Loop

```javascript
function gameLoop() {
  // 1. Update all managers
  inputManager.update();
  characterController.update();
  physicsManager.update();
  dialogManager.update();
  musicManager.update();
  videoManager.update();
  vfxManager.update();
  // ... etc

  // 2. Render the scene
  renderer.render(scene, camera);

  // 3. Request next frame
  requestAnimationFrame(gameLoop);
}
```

### Cleanup Phase

```javascript
// When the game is shutting down:
async function cleanup() {
  // 1. Stop all media
  dialogManager.dispose();
  musicManager.dispose();
  videoManager.dispose();

  // 2. Clean up physics
  physicsManager.dispose();

  // 3. Remove event listeners
  gameManager.removeAllListeners();

  // 4. Dispose of Three.js resources
  sceneManager.dispose();
}
```

---

## Anatomy of a Manager

### The Base Manager Pattern

All managers in the Shadow Engine follow this structure:

```javascript
class SomeManager {
  constructor() {
    // Configuration
    this.config = {};

    // State
    this.initialized = false;
    this.currentItems = [];

    // References
    this.gameManager = null;
    this.data = null;
  }

  // Initialization
  async initialize(gameManager, data) {
    this.gameManager = gameManager;
    this.data = data;
    this.initialized = true;

    // Set up event listeners
    this.listenTo(gameManager);
  }

  // Event subscription
  listenTo(gameManager) {
    gameManager.on("state:changed", this.onStateChanged.bind(this));
    gameManager.on("game:paused", this.onPause.bind(this));
    gameManager.on("game:resumed", this.onResume.bind(this));
  }

  // Event handlers
  onStateChanged(newState, oldState) {
    // React to state changes
    const matchingItems = this.data.filter(item =>
      matchesCriteria(item.criteria, newState)
    );
    this.playItems(matchingItems);
  }

  onPause() {
    // Pause this manager's activity
  }

  onResume() {
    // Resume this manager's activity
  }

  // Per-frame update
  update(deltaTime) {
    if (!this.initialized) return;
    // Update this manager's systems
  }

  // Cleanup
  dispose() {
    this.gameManager.off("state:changed", this.onStateChanged);
    this.currentItems = [];
    this.initialized = false;
  }
}
```

---

## Data Flow Between Managers

### Example: Player Picks Up an Object

```
1. InputManager detects player pressed "E" key
   │
   │ emit("input:action", { action: "interact" })
   ▼
2. CharacterController receives input
   │
   │ Check what's in front of player
   │ Raycast finds "Phone" object
   │
   │ emit("object:interacted", { object: "Phone" })
   ▼
3. GameManager receives interaction
   │
   │ Update state
   │ setState({ hasPhone: true, currentState: PHONE_ACQUIRED })
   ▼
4. GameManager emits "state:changed"
   │
   ├──▶ DialogManager receives event
   │        │
   │        │ Finds dialog matching PHONE_ACQUIRED state
   │        │ Plays: "You found the phone!"
   │
   ├──▶ MusicManager receives event
   │        │
   │        │ Crossfades to "mystery" music track
   │
   ├──▶ VideoManager receives event
   │        │
   │        │ Plays phone ringing video
   │
   └──▶ VFXManager receives event
            │
            │ Plays "glow" effect on phone object
```

---

## Types of Managers in the Shadow Engine

### 1. Core Managers (Essential)

| Manager | Purpose | Key Methods |
|---------|---------|-------------|
| GameManager | State + coordination | `setState()`, `getState()`, `on()` |
| SceneManager | 3D content loading | `loadObjectsForState()`, `loadSplat()` |
| InputManager | All input sources | `getAction()`, `isPressed()` |

### 2. Media Managers (Content Playback)

| Manager | Purpose | Key Methods |
|---------|---------|-------------|
| DialogManager | Spoken dialogue | `playDialog()`, `showCaptions()` |
| MusicManager | Background music | `playTrack()`, `crossfadeTo()` |
| VideoManager | Video with alpha | `playVideo()`, `stopVideo()` |
| SFXManager | Sound effects | `playSound()`, `playSpatial()` |

### 3. Gameplay Managers (Game Mechanics)

| Manager | Purpose | Key Methods |
|---------|---------|-------------|
| CharacterController | Player movement | `update()`, `setEnabled()` |
| PhysicsManager | Physics simulation | `createRigidBody()`, `step()` |
| ColliderManager | Trigger zones | `checkCollisions()` |

### 4. Visual Managers (Graphics)

| Manager | Purpose | Key Methods |
|---------|---------|-------------|
| VFXManager | Visual effects | `playEffect()`, `stopEffect()` |
| LightManager | Lighting | `setLightsForState()` |
| AnimationManager | Animations | `playAnimation()`, `stopAnimation()` |

### 5. UI Managers (User Interface)

| Manager | Purpose | Key Methods |
|---------|---------|-------------|
| UIManager | All UI screens | `showScreen()`, `hideScreen()` |
| LoadingScreen | Loading display | `setProgress()`, `show()` |

---

## Communication Patterns

### Pattern 1: State-Driven Communication

The most common pattern - managers react to state changes:

```javascript
// In DialogManager
listenTo(gameManager) {
  gameManager.on("state:changed", (newState) => {
    const dialogsToPlay = this.data.filter(dialog =>
      matchesCriteria(dialog.criteria, newState)
    );
    this.playDialogs(dialogsToPlay);
  });
}
```

### Pattern 2: Request-Response

Managers request something from another:

```javascript
// CharacterController asks PhysicsManager for collisions
const collisions = physicsManager.checkCollisions(playerCollider);
if (collisions.length > 0) {
  // Handle collision
}
```

### Pattern 3: Direct Notification

Managers emit specific events:

```javascript
// DialogManager emits when captions change
this.emit("caption:changed", { text: newCaption });

// UIManager listens
uiManager.on("caption:changed", ({ text }) => {
  this.captionElement.textContent = text;
});
```

---

## Common Mistakes Beginners Make

### 1. Creating Multiple Instances

```javascript
// ❌ WRONG: Multiple instances cause conflicts
const gameManager1 = new GameManager();
const gameManager2 = new GameManager();

// ✅ CORRECT: Use singleton pattern
const gameManager = GameManager.getInstance();
// Or: just create ONE instance and share it
```

### 2. Tight Coupling Between Managers

```javascript
// ❌ WRONG: DialogManager directly calls MusicManager
class DialogManager {
  playDialog(dialog) {
    musicManager.crossfadeTo("dialog-music"); // Tight coupling!
  }
}

// ✅ CORRECT: Use events
class DialogManager {
  playDialog(dialog) {
    this.emit("dialog:started", { dialog });
  }
}

// MusicManager listens
musicManager.on("dialog:started", () => {
  this.crossfadeTo("dialog-music");
});
```

### 3. Not Cleaning Up Event Listeners

```javascript
// ❌ WRONG: Memory leak!
class SomeManager {
  initialize(gameManager) {
    gameManager.on("state:changed", this.onChange);
    // Never removes listener!
  }
}

// ✅ CORRECT: Clean up in dispose()
class SomeManager {
  initialize(gameManager) {
    this.gameManager = gameManager;
    this.boundOnChange = this.onChange.bind(this);
    gameManager.on("state:changed", this.boundOnChange);
  }

  dispose() {
    this.gameManager.off("state:changed", this.boundOnChange);
  }
}
```

### 4. Managers Doing Too Much

```javascript
// ❌ WRONG: DialogManager also handles video playback
class DialogManager {
  playDialog(dialog) {
    this.audio.play();
    this.video.play(); // This should be VideoManager's job!
    this.captions.show();
  }
}

// ✅ CORRECT: Each manager handles one thing
class DialogManager {
  playDialog(dialog) {
    this.audio.play();
    this.captions.show();
  }
}
```

### 5. Circumventing GameManager for State

```javascript
// ❌ WRONG: Managers storing their own state
class DialogManager {
  constructor() {
    this.currentState = 0; // Don't do this!
  }
}

// ✅ CORRECT: All state in GameManager
class DialogManager {
  getState() {
    return this.gameManager.getState().currentState;
  }
}
```

---

## Benefits of the Manager Pattern

### 1. Easier to Understand

Each manager is a self-contained system. You can understand DialogManager without knowing how MusicManager works.

### 2. Easier to Test

```javascript
// Can test DialogManager in isolation
const mockGameManager = new EventEmitter();
const dialogManager = new DialogManager();
await dialogManager.initialize(mockGameManager, testData);

// Test it!
dialogManager.playDialog(testDialog);
assert(dialogManager.currentDialog === testDialog);
```

### 3. Easier to Extend

```javascript
// Want to add a new manager? Just create it!
const achievementManager = new AchievementManager();
await gameManager.initialize({
  // ... existing managers
  achievementManager
});
```

### 4. Reusable Components

Managers can be extracted and used in other projects:

```javascript
// The DialogManager could be used in ANY game!
import { DialogManager } from '@shadow-engine/dialog';
```

---

## Related Patterns

### Singleton Pattern

Most managers are singletons - only one instance exists:

```javascript
class GameManager {
  static instance = null;

  static getInstance() {
    if (!GameManager.instance) {
      GameManager.instance = new GameManager();
    }
    return GameManager.instance;
  }
}
```

### Observer Pattern (Event Emitter)

Managers use the Observer pattern via events:

```javascript
// GameManager is the "subject"
gameManager.emit("state:changed", newState);

// Other managers are "observers"
dialogManager.on("state:changed", handler);
musicManager.on("state:changed", handler);
```

### Data-Driven Design

Managers are configured by external data files:

```javascript
// dialogData.js defines what dialog exists
export const dialogTracks = {
  intro: { file: "intro.mp3", criteria: { currentState: 1 } }
};

// DialogManager just loads and plays
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SHADOW ENGINE                                  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        GameManager                              │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │              State: { currentState, controlEnabled }    │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │                                                                  │   │
│  │    emit("state:changed")   emit("game:paused")  emit(...)      │   │
│  └──────────────────────────┬───────────────────────┬───────────────┘   │
│                             │                       │                   │
│        ┌────────────────────┼───────────────────────┼─────────────┐     │
│        │                    │                       │             │     │
│        ▼                    ▼                       ▼             ▼     │
│  ┌───────────┐        ┌───────────┐           ┌───────────┐  ┌───────────┐
│  │   Media   │        │ Gameplay  │           │  Visual   │  │    UI     │
│  │ Managers  │        │ Managers  │           │ Managers  │  │ Managers  │
│  ├───────────┤        ├───────────┤           ├───────────┤  ├───────────┤
│  │ Dialog    │        │ Input     │           │ Scene     │  │ UI        │
│  │ Music     │        │ Character │           │ VFX       │  │ Loading   │
│  │ SFX       │        │ Physics   │           │ Light     │  │ Options   │
│  │ Video     │        │ Collider  │           │ Animation │  │ Captions  │
│  └───────────┘        └───────────┘           └───────────┘  └───────────┘
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   External Data     │
                   ├─────────────────────┤
                   │ dialogData.js       │
                   │ musicData.js        │
                   │ sceneData.js        │
                   │ vfxData.js          │
                   │ ...                 │
                   └─────────────────────┘
```

---

## Next Steps

Now that you understand the manager pattern:

- [GameManager Deep Dive](./game-manager-deep-dive.md) - The central coordinator
- [Game State System](./game-state-system.md) - How state is managed
- [Data-Driven Design](./data-driven-design.md) - How data files work
- [SceneManager](../03-scene-rendering/scene-manager.md) - 3D content management
- [DialogManager](../05-media-systems/dialog-manager.md) - Dialog system

---

## References

- [Design Patterns: Singleton](https://refactoring.guru/design-patterns/singleton) - Singleton pattern explanation
- [Design Patterns: Observer](https://refactoring.guru/design-patterns/observer) - Observer pattern explanation
- [Separation of Concerns Principle](https://en.wikipedia.org/wiki/Separation_of_concerns) - SoC principle
- [Single Responsibility Principle](https://en.wikipedia.org/wiki/Single-responsibility_principle) - SRP principle
- [Event Emitter Pattern](https://nodejs.dev/learn/the-event-emitter-pattern) - Event-driven architecture

*Documentation last updated: January 12, 2026*
