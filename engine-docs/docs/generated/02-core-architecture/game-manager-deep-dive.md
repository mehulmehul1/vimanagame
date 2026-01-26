# GameManager - Core Architecture Deep Dive

## Overview

The **GameManager** is the central brain of the Shadow Engine. It stores all game state, emits events that other systems react to, and coordinates between all the different managers (dialog, music, video, input, etc.).

Think of GameManager as the **central nervous system** of the game - everything flows through it, and it coordinates all the other systems.

## What You Need to Know First

Before understanding GameManager, you should know:
- **What "state" means** - The current condition or status of something
- **Event-driven programming** - A pattern where things happen in response to events
- **JavaScript objects** - Key-value pairs for storing data
- **Basic game loops** - Update, render, repeat

---

## Core Concepts

### The Event Emitter Pattern

GameManager uses an **event emitter pattern**. This is like a notification system:

```
┌─────────────┐         emit         ┌─────────────┐
│ GameManager │ ─────────────────────>│  Listener   │
│             │   "state:changed"     │             │
└─────────────┘                       └─────────────┘
       │                                     │
       │                                     ▼
       │                            Do something
       │                            in response
       │
       ▼
┌─────────────┐
│ Other System │
│ (Dialog,     │
│  Music, etc.)│
└─────────────┘
```

**How it works:**
1. GameManager's state changes
2. GameManager emits "state:changed" event
3. All subscribed systems receive the event
4. Each system reacts appropriately

### MongoDB-Style Criteria

The engine uses a clever system inspired by MongoDB database queries. Instead of writing complex `if` statements, you define **criteria** that match game state:

```javascript
// Instead of writing:
if (gameState.currentState === 5 || gameState.currentState === 6) {
  // Do something
}

// You write:
criteria: { currentState: { $in: [5, 6] } }
```

This makes data files much cleaner and more readable!

---

## GameManager Structure

### State Object

The GameManager stores all game state in a single object:

```javascript
{
  // Lifecycle
  isPlaying: false,
  isPaused: false,

  // Scene/World
  currentScene: null,

  // Narrative progression
  currentState: GAME_STATES.LOADING, // -1 to 44
  currentZone: "plaza",

  // Player control
  controlEnabled: false,

  // View-Master state
  isViewmasterEquipped: false,
  viewmasterManuallyRemoved: false,
  viewmasterInsanityIntensity: 0.0,
  viewmasterOverheatCount: 0,

  // Drawing game state
  drawingSuccessCount: 0,
  drawingFailureCount: 0,
  currentDrawingTarget: null,
  sawRune: false,
  runeSightings: 0,

  // Platform detection
  isIOS: false,
  isSafari: false,
  isMobile: false
}
```

### Game States (45 States)

The game has 45 narrative states representing the story progression:

```javascript
GAME_STATES = {
  LOADING: -1,                  // Initial loading
  START_SCREEN: 0,             // Main menu
  INTRO: 1,                    // Game begins
  TITLE_SEQUENCE: 2,           // Title cards
  // ... 40+ more states for narrative progression ...
  GAME_OVER: 44                // End of game
}
```

**Why numeric?** Numeric states allow comparisons:
- `{ currentState: { $gte: INTRO, $lt: DRIVE_BY } }` - Between INTRO and DRIVE_BY
- States can be checked against ranges
- Makes branching logic simpler

---

## Key Responsibilities

### 1. State Management

**Setting State:**
```javascript
gameManager.setState({
  currentState: GAME_STATES.OFFICE_INTERIOR,
  controlEnabled: true
});
```

**Getting State:**
```javascript
const state = gameManager.getState();
console.log(state.currentState); // 14
```

### 2. Event Emission

GameManager emits these key events:

| Event | When | Who Listens |
|-------|------|-------------|
| `state:changed` | Any state change | All managers |
| `character-controller:enabled` | Controls enabled | InputManager |
| `character-controller:disabled` | Controls disabled | InputManager |
| `game:paused` | Game paused | Various systems |
| `game:resumed` | Game resumed | Various systems |

**Subscribing to events:**
```javascript
gameManager.on("state:changed", (newState, oldState) => {
  console.log(`State changed from ${oldState.currentState} to ${newState.currentState}`);
  // React to state change
});
```

### 3. Scene Management

GameManager coordinates with SceneManager to load/unload objects:

```javascript
// Called automatically when state changes
async updateSceneForState(options = {}) {
  // Get objects that match current state criteria
  const objectsToLoad = getSceneObjectsForState(this.state, options);

  // Load new objects
  await this.sceneManager.loadObjectsForState(objectsToLoad);

  // Unload objects that don't match
  // ...
}
```

### 4. Debug Spawning

URL parameters can override initial state for testing:

```
?gameState=OFFICE_INTERIOR&dialogChoice2=EMPATH
```

This allows jumping to any point in the game for testing!

---

## The Criteria System

### Supported Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equals | `{ currentState: { $eq: 5 } }` |
| `$ne` | Not equals | `{ currentState: { $ne: 0 } }` |
| `$gt` | Greater than | `{ currentState: { $gt: 10 } }` |
| `$gte` | Greater or equal | `{ currentState: { $gte: 5 } }` |
| `$lt` | Less than | `{ currentState: { $lt: 20 } }` |
| `$lte` | Less or equal | `{ currentState: { $lte: 15 } }` |
| `$in` | In array | `{ currentState: { $in: [5, 6, 7] } }` |
| `$nin` | Not in array | `{ currentState: { $nin: [0, 1] } }` |
| `$mod` | Modulo | `{ count: { $mod: [2, 0] } }` (even numbers) |

### Real Examples from Shadow Engine

```javascript
// Dialog plays during intro sequence
dialogData: {
  intro: {
    criteria: { currentState: GAME_STATES.INTRO }
  }
}

// Music changes between scenes
musicData: {
  officeTrack: {
    criteria: { currentState: { $gte: OFFICE_INTERIOR, $lt: LIGHTS_OUT } }
  }
}

// Video plays for specific dialog choice
videoData: {
  edisonVideo: {
    criteria: { dialogChoice2: DIALOG_RESPONSE_TYPES.EDISON }
  }
}
```

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         GameManager                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     State Object                              │   │
│  │  - currentState: number                                       │   │
│  │  - controlEnabled: boolean                                   │   │
│  │  - currentZone: string                                        │   │
│  │  - isViewmasterEquipped: boolean                              │   │
│  │  - ... (45+ state properties)                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Event Emitters                              │   │
│  │  emit("state:changed", newState, oldState)                  │   │
│  │  emit("character-controller:enabled")                       │   │
│  │  emit("game:paused")                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Manager References                               │   │
│  │  - dialogManager                                             │   │
│  │  - musicManager                                              │   │
│  │  - sfxManager                                                │   │
│  │  - videoManager                                              │   │
│  │  - sceneManager                                              │   │
│  │  - characterController                                       │   │
│  │  - inputManager                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
         │                   │                   │
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ DialogManager│    │ MusicManager │    │VideoManager │
│             │    │             │    │             │
│ Listen to    │    │ Listen to    │    │ Listen to    │
│ state:changed│    │ state:changed│    │ state:changed│
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## How State Changes Flow

```
1. Something happens (player action, time passes, etc.)
              │
              ▼
2. Code calls: gameManager.setState({ currentState: NEW_STATE })
              │
              ▼
3. GameManager updates internal state
              │
              ▼
4. GameManager emits "state:changed" event
              │
              ├──▶ DialogManager receives event
              │        │
              │        ▼
              │   Check if any dialogs match new state
              │   Play matching dialogs
              │
              ├──▶ MusicManager receives event
              │        │
              │        ▼
              │   Check if any music matches new state
              │   Play matching music
              │
              ├──▶ VideoManager receives event
              │        │
              │        ▼
              │   Check if any videos match new state
              │   Play matching videos
              │
              ├──▶ SceneManager receives event
              │        │
              │        ▼
              │   Load/unload scene objects
              │
              └──▶ All other managers...
```

---

## Key Methods

### setState(newState)
Update game state and notify all listeners.

```javascript
gameManager.setState({
  currentState: GAME_STATES.OFFICE_INTERIOR,
  controlEnabled: true
});
```

### getState()
Get a copy of current state.

```javascript
const state = gameManager.getState();
console.log(state.currentState);
```

### on(event, callback)
Subscribe to an event.

```javascript
gameManager.on("state:changed", (newState, oldState) => {
  console.log("State changed!");
});
```

### off(event, callback)
Unsubscribe from an event.

```javascript
gameManager.off("state:changed", callback);
```

### initialize(managers)
Set up references to all other managers.

```javascript
await gameManager.initialize({
  dialogManager,
  musicManager,
  sfxManager,
  videoManager,
  // ... etc
});
```

---

## Common Mistakes Beginners Make

### 1. Mutating State Directly

❌ **WRONG:**
```javascript
gameManager.state.currentState = 5; // Won't emit events!
```

✅ **CORRECT:**
```javascript
gameManager.setState({ currentState: 5 }); // Emits "state:changed"
```

### 2. Forgetting to Subscribe

❌ **WRONG:**
```javascript
// Expecting music to play automatically
gameManager.setState({ currentState: GAME_STATES.INTRO });
// Music won't play if MusicManager isn't listening!
```

✅ **CORRECT:**
```javascript
// During initialization:
musicManager.listenTo(gameManager); // Subscribes to events

// Now state changes will trigger music
gameManager.setState({ currentState: GAME_STATES.INTRO });
```

### 3. Hard-Coding State Numbers

❌ **WRONG:**
```javascript
if (state.currentState === 14) { // What is 14?!
  // Do something
}
```

✅ **CORRECT:**
```javascript
if (state.currentState === GAME_STATES.OFFICE_INTERIOR) {
  // Clear and readable!
}
```

### 4. Not Using Criteria

❌ **WRONG:**
```javascript
// Complex if statements everywhere
if (state.currentState === 5 || state.currentState === 6 || state.currentState === 7) {
  playMusic();
}
```

✅ **CORRECT:**
```javascript
// In data file:
criteria: { currentState: { $in: [5, 6, 7] } }
```

---

## Data-Driven Design

The Shadow Engine is **data-driven** - content is defined in separate data files, not hardcoded:

| Data File | Contains |
|-----------|----------|
| `dialogData.js` | All dialog tracks with criteria |
| `musicData.js` | All music tracks with criteria |
| `sfxData.js` | All sound effects with criteria |
| `videoData.js` | All videos with criteria |
| `lightData.js` | All lights with criteria |
| `vfxData.js` | All visual effects with criteria |
| `sceneData.js` | All 3D objects with criteria |

**Example from dialogData.js:**
```javascript
export const dialogTracks = {
  intro: {
    file: "audio/dialog/intro.mp3",
    captions: [
      { text: "Hello...", start: 0, end: 2000 }
    ],
    criteria: { currentState: GAME_STATES.INTRO }
  }
};
```

When state changes to INTRO, DialogManager automatically plays this dialog!

---

## Debug Features

### URL Parameter Spawning

Skip to any game state via URL:

```
?gameState=OFFICE_INTERIOR
?dialogChoice2=EMPATH
```

### Debug Mode

GameManager detects debug spawn and:
- Preloads all matching assets
- Enables controls immediately
- Sets player position/rotation

---

## Performance Considerations

1. **State changes are fast** - Just copying an object and emitting events
2. **Deferred loading** - Large assets (splats) can be loaded later
3. **Criteria caching** - Criteria are evaluated once per state change
4. **Event efficiency** - Only relevant systems receive events

---

## Next Steps

Now that you understand the core architecture:

- [SceneManager Deep Dive](../03-scene-rendering/scene-manager.md) - How 3D content is loaded
- [Criteria Helper Guide](../02-core-architecture/criteria-system.md) - Criteria matching in detail
- [DialogManager](../05-media-systems/dialog-manager.md) - How dialog works
- [Data-Driven Design](../02-core-architecture/data-driven-design.md) - How content files work

---

## Source File Reference

- **Location:** `src/gameManager.js`
- **Key exports:** `GameManager` class
- **Dependencies:** Three.js, criteriaHelper, debugSpawner, various content controllers
- **Used by:** All managers in the engine

---

## References

- [MongoDB Query Operators](https://www.mongodb.com/docs/manual/reference/operator/query/) - Inspiration for criteria system
- [Event Emitter Pattern](https://nodejs.dev/learn/the-event-emitter-pattern) - Node.js event pattern
- [State Management Patterns](https://www.patterns.dev/posts/state-pattern/) - State management in JS

*Documentation last updated: January 2026*
