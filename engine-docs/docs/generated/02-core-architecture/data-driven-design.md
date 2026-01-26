# Data-Driven Design Pattern - First Principles Guide

## Overview

**Data-Driven Design** is a programming paradigm where program logic describes **what data to match** and **how to process it**, rather than defining a fixed sequence of steps.

In the Shadow Engine, this means:
- All game content (dialog, music, videos, scenes, etc.) is defined in separate **Data files**
- The engine code is generic - it works with ANY data that follows the pattern
- Adding new content doesn't require changing code - just add to the data files

This separation between **code** (how things work) and **data** (what exists) is fundamental to the engine's architecture.

## What You Need to Know First

Before understanding data-driven design, you should know:
- **JavaScript objects** - Key-value pairs for structured data
- **JavaScript modules** - Import/export for organizing code
- **Basic JSON-like structures** - Nested objects and arrays
- **The criteria system** - MongoDB-style query operators (covered in [Game State System](./game-state-system.md))

### Quick Example: Hard-Coded vs Data-Driven

```javascript
// ❌ HARD-CODED: Content mixed with code
function playOfficeMusic() {
  if (currentState === 14 || currentState === 15 ||
      currentState === 16 || currentState === 17) {
    const audio = new Audio("assets/music/office.mp3");
    audio.volume = 0.7;
    audio.loop = true;
    audio.play();
  }
}

// ✅ DATA-DRIVEN: Content separated from code
// In musicData.js:
export const musicTracks = {
  officeAmbience: {
    file: "assets/music/office.mp3",
    volume: 0.7,
    loop: true,
    criteria: { currentState: { $gte: 14, $lt: 18 } }
  }
};

// MusicManager is generic - works with ANY track data:
class MusicManager {
  initialize(gameManager, musicData) {
    this.musicData = musicData;
    gameManager.on("state:changed", (state) => {
      this.playMatchingTracks(state);
    });
  }
}
```

---

## Why Data-Driven Design?

### The Problem: Code and Content Mixed

When content is hard-coded into the game engine:

```
┌─────────────────────────────────────────────────────────────┐
│                    MONOLITHIC CODE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  function updateGame() {                                     │
│    // Game logic mixed with content                          │
│    if (state === 14) playMusic("office.mp3");                │
│    if (state === 14) playDialog("intro.mp3");                │
│    if (state === 15) playMusic("office.mp3");                │
│    if (state === 15) playDialog("hello.mp3");                │
│    // ... thousands of lines like this                       │
│  }                                                           │
│                                                              │
│  Problems:                                                   │
│  - Adding content requires code changes                      │
│  - Non-programmers can't add content                         │
│  - Code becomes huge and unmaintainable                      │
│  - Can't reuse code for other projects                       │
└─────────────────────────────────────────────────────────────┘
```

### The Solution: Separate Code from Data

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA-DRIVEN DESIGN                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐          ┌─────────────────┐           │
│  │   ENGINE CODE   │          │   DATA FILES    │           │
│  │   (generic)     │          │   (content)     │           │
│  ├─────────────────┤          ├─────────────────┤           │
│  │ MusicManager    │   reads  │ musicData.js    │           │
│  │ DialogManager   │   reads  │ dialogData.js   │           │
│  │ SceneManager    │   reads  │ sceneData.js    │           │
│  │ VFXManager      │   reads  │ vfxData.js      │           │
│  └─────────────────┘          └─────────────────┘           │
│         │                            │                       │
│         └────────────┬───────────────┘                       │
│                      ▼                                       │
│              Generic managers work with                       │
│              ANY data following the pattern                  │
│                                                              │
│  Benefits:                                                   │
│  - Add content without touching code                         │
│  - Designers can work in data files                          │
│  - Code stays clean and reusable                              │
│  - Easy to port to other projects                             │
└─────────────────────────────────────────────────────────────┘
```

---

## The Data File Structure

All data files in the Shadow Engine follow a consistent pattern:

```javascript
// The Pattern:
export const dataType = {
  itemName: {
    // Properties defining the content
    file: "path/to/file.ext",
    // ... other properties

    // When this content should be active
    criteria: { /* criteria object */ }
  }
};
```

### All Data Files in the Engine

| Data File | Contains | Managed By |
|-----------|----------|------------|
| `dialogData.js` | All dialog tracks with captions | DialogManager |
| `musicData.js` | All background music tracks | MusicManager |
| `sfxData.js` | All sound effects | SFXManager |
| `videoData.js` | All video clips with alpha | VideoManager |
| `sceneData.js` | All 3D objects and splats | SceneManager |
| `animationData.js` | Camera and object animations | AnimationManager |
| `lightData.js` | All lights and their properties | LightManager |
| `vfxData.js` | All visual effects | VFXManager |
| `interactiveObjectData.js` | All interactable objects | Various managers |

---

## Deep Dive: Each Data File

### 1. dialogData.js

Defines all spoken dialogue in the game.

```javascript
// dialogData.js
export const dialogTracks = {
  intro: {
    file: "audio/dialog/intro.mp3",
    volume: 1.0,
    captions: [
      { text: "Welcome...", start: 0, end: 2000 },
      { text: "I've been waiting.", start: 2000, end: 5000 }
    ],
    criteria: { currentState: GAME_STATES.INTRO }
  },

  officeWelcome: {
    file: "audio/dialog/office-welcome.mp3",
    captions: [
      { text: "Ah, you're here!", start: 0, end: 1500 },
      { text: "Come in, sit down.", start: 1500, end: 3000 }
    ],
    criteria: {
      currentState: { $gte: OFFICE_INTERIOR, $lt: LIGHTS_OUT },
      sawRune: false  // Only if player hasn't seen the rune
    }
  },

  runeEncounter: {
    file: "audio/dialog/rune-whisper.mp3",
    captions: [
      { text: "*eerie whispering*", start: 0, end: 4000 }
    ],
    criteria: {
      currentState: { $gte: RUNE_APPEARS },
      runeSightings: { $mod: [3, 0] }  // Every 3rd sighting
    }
  }
};
```

**Dialog Track Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `file` | string | Path to audio file |
| `volume` | number | 0.0 to 1.0 |
| `captions` | array | Array of caption objects |
| `criteria` | object | When this dialog plays |
| `priority` | number | (optional) Higher priority interrupts |

**Caption Object:**

| Property | Type | Description |
|----------|------|-------------|
| `text` | string | Caption text to display |
| `start` | number | Start time in milliseconds |
| `end` | number | End time in milliseconds |

### 2. musicData.js

Defines all background music tracks.

```javascript
// musicData.js
export const musicTracks = {
  mainMenu: {
    file: "audio/music/menu-theme.mp3",
    volume: 0.6,
    loop: true,
    fadeIn: 2000,  // milliseconds
    fadeOut: 3000,
    criteria: { currentState: GAME_STATES.START_SCREEN }
  },

  officeAmbience: {
    file: "audio/music/office-drone.mp3",
    volume: 0.4,
    loop: true,
    fadeIn: 5000,
    fadeOut: 3000,
    criteria: {
      currentState: { $gte: OFFICE_INTERIOR, $lt: LIGHTS_OUT }
    }
  },

  tensionMusic: {
    file: "audio/music/tension-builder.mp3",
    volume: 0.8,
    loop: true,
    fadeIn: 1000,
    fadeOut: 2000,
    criteria: {
      currentState: { $in: [RUNE_SIGHTING_1, RUNE_SIGHTING_2, RUNE_SIGHTING_3] }
    }
  },

  climaxMusic: {
    file: "audio/music/final-confrontation.mp3",
    volume: 1.0,
    loop: false,  // Play once
    fadeIn: 500,
    fadeOut: 0,
    criteria: { currentState: GAME_STATES.FINAL_BATTLE }
  }
};
```

**Music Track Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `file` | string | Path to audio file |
| `volume` | number | 0.0 to 1.0 |
| `loop` | boolean | Whether to loop the track |
| `fadeIn` | number | Fade-in duration (ms) |
| `fadeOut` | number | Fade-out duration (ms) |
| `criteria` | object | When this music plays |
| `crossfadeTo` | string | (optional) Next track to crossfade to |

### 3. sceneData.js

Defines all 3D content in the game.

```javascript
// sceneData.js
export const sceneObjects = {
  // Gaussian Splat scenes
  officeSplat: {
    type: "splat",
    file: "splats/office.sog",
    position: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
    criteria: {
      currentZone: "office",
      currentState: { $gte: OFFICE_INTERIOR, $lt: OFFICE_EXIT }
    }
  },

  plazaSplat: {
    type: "splat",
    file: "splats/plaza.sog",
    position: { x: 0, y: 0, z: 0 },
    criteria: { currentZone: "plaza" }
  },

  // GLTF Models
  phoneBooth: {
    type: "model",
    file: "models/phone-booth.glb",
    position: { x: 5, y: 0, z: -10 },
    rotation: { x: 0, y: Math.PI / 4, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
    criteria: { currentZone: "plaza" }
  },

  // Interactive objects
  officeDoor: {
    type: "interactive",
    id: "office-door",
    file: "models/door.glb",
    position: { x: 0, y: 0, z: 5 },
    interactionType: "click",
    onInteract: "enterOffice",
    criteria: {
      currentZone: "plaza",
      currentState: { $gte: PLAZA_ARRIVAL }
    }
  }
};
```

**Scene Object Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `type` | string | "splat", "model", "interactive", etc. |
| `file` | string | Path to asset file |
| `position` | object | { x, y, z } coordinates |
| `rotation` | object | (optional) { x, y, z } Euler rotation |
| `scale` | object | (optional) { x, y, z } scale |
| `criteria` | object | When object is visible/active |
| `interactionType` | string | (interactive) "click", "proximity", etc. |
| `onInteract` | string | (interactive) Action to take |

### 4. vfxData.js

Defines all visual effects.

```javascript
// vfxData.js
export const vfxEffects = {
  // Dissolve effect
  officeDissolve: {
    type: "dissolve",
    duration: 3000,
    target: "officeSplat",
    direction: "out",  // or "in"
    criteria: {
      currentState: OFFICE_EXIT,
      currentZone: "office"
    }
  },

  // Glitch effect
  viewmasterGlitch: {
    type: "glitch",
    intensity: 0.8,
    duration: 500,
    target: "screen",
    criteria: {
      isViewmasterEquipped: true,
      viewmasterInsanityIntensity: { $gt: 0.5 }
    }
  },

  // Desaturation
  runeDesaturation: {
    type: "desaturate",
    amount: 0.9,  // 0.0 to 1.0
    duration: 2000,
    criteria: {
      currentState: { $gte: RUNE_ENCOUNTER },
      sawRune: true
    }
  },

  // Bloom
  glowEffect: {
    type: "bloom",
    threshold: 0.7,
    strength: 1.5,
    radius: 0.5,
    targets: ["runeObject", "candleFlame"],
    criteria: { sawRune: true }
  }
};
```

**VFX Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `type` | string | "dissolve", "glitch", "desaturate", "bloom", etc. |
| `duration` | number | Effect duration in ms |
| `target` | string/array | Target object(s) |
| `intensity`/`amount` | number | Effect strength (0.0 to 1.0) |
| `criteria` | object | When effect triggers |

### 5. animationData.js

Defines all camera and object animations.

```javascript
// animationData.js
export const animations = {
  // Camera animations
  introCameraSweep: {
    type: "camera",
    duration: 8000,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 5 }, target: { x: 0, y: 1, z: 0 } },
      { time: 0.3, position: { x: 2, y: 1.6, z: 3 }, target: { x: 0, y: 1, z: 0 } },
      { time: 0.6, position: { x: -2, y: 1.6, z: 3 }, target: { x: 0, y: 1, z: 0 } },
      { time: 1.0, position: { x: 0, y: 1.6, z: 0 }, target: { x: 0, y: 1.5, z: -5 } }
    ],
    easing: "easeInOutCubic",
    criteria: { currentState: GAME_STATES.INTRO }
  },

  // Object animations
  phoneRinging: {
    type: "object",
    target: "phoneBoothPhone",
    duration: 500,
    loop: true,
    keyframes: [
      { time: 0, rotation: { x: 0, y: 0, z: 0 } },
      { time: 0.1, rotation: { x: 0.05, y: 0, z: 0 } },
      { time: 0.2, rotation: { x: -0.05, y: 0, z: 0 } },
      { time: 0.3, rotation: { x: 0, y: 0, z: 0 } }
    ],
    criteria: { currentState: PHONE_RINGING }
  }
};
```

**Animation Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `type` | string | "camera" or "object" |
| `target` | string | Object ID (for object animations) |
| `duration` | number | Total duration in ms |
| `loop` | boolean | Whether to loop |
| `keyframes` | array | Array of keyframe objects |
| `easing` | string | Easing function name |
| `criteria` | object | When animation plays |

### 6. videoData.js

Defines all video content.

```javascript
// videoData.js
export const videos = {
  edisonEnding: {
    file: "video/edison-ending.webm",
    position: { x: 0, y: 1, z: -2 },
    scale: 2,
    loop: false,
    alphaChannel: true,  // WebM with alpha
    criteria: { dialogChoice2: DIALOG_RESPONSE_TYPES.EDISON }
  },

  empathEnding: {
    file: "video/empath-ending.webm",
    position: { x: 0, y: 1, z: -2 },
    scale: 2,
    loop: false,
    alphaChannel: true,
    criteria: { dialogChoice2: DIALOG_RESPONSE_TYPES.EMPATH }
  }
};
```

**Video Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `file` | string | Path to WebM file |
| `position` | object | { x, y, z } world position |
| `scale` | number | Video scale multiplier |
| `loop` | boolean | Whether to loop |
| `alphaChannel` | boolean | Video has transparency |
| `criteria` | object | When video plays |

---

## How Managers Use Data Files

### The Generic Pattern

All managers follow this same pattern:

```javascript
class SomeManager {
  async initialize(gameManager, data) {
    this.gameManager = gameManager;
    this.data = data;  // Store the data

    // Listen for state changes
    gameManager.on("state:changed", this.onStateChanged.bind(this));
  }

  onStateChanged(newState, oldState) {
    // Find items matching the new state
    const matchingItems = this.getMatchingItems(newState);

    // Process each matching item
    matchingItems.forEach(item => {
      this.processItem(item);
    });
  }

  getMatchingItems(state) {
    // Convert object to array and filter by criteria
    return Object.values(this.data).filter(item => {
      return matchesCriteria(item.criteria, state);
    });
  }

  processItem(item) {
    // Subclasses implement this
    throw new Error("Subclasses must implement processItem");
  }
}
```

### Example: DialogManager

```javascript
class DialogManager extends SomeManager {
  processItem(dialog) {
    // Don't interrupt playing dialog
    if (this.currentDialog && dialog.priority <= this.currentPriority) {
      return;
    }

    // Play the dialog
    this.playDialog(dialog);
    this.showCaptions(dialog.captions);
  }

  playDialog(dialog) {
    this.audio = new Howl({
      src: [dialog.file],
      volume: dialog.volume || 1.0,
      onend: () => {
        this.currentDialog = null;
      }
    });

    this.audio.play();
    this.currentDialog = dialog;
  }
}
```

---

## Benefits of Data-Driven Design

### 1. Designers Can Work Independently

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKFLOW                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Game Designer:                                              │
│  1. Opens dialogData.js                                     │
│  2. Adds new dialog track:                                   │
│     newDialog: {                                            │
│       file: "new-line.mp3",                                 │
│       captions: [{ text: "Hello!", start: 0, end: 1000 }], │
│       criteria: { currentState: 20 }                        │
│     }                                                        │
│  3. Saves file                                              │
│  4. Runs game - dialog just works!                          │
│                                                              │
│  No programmer needed. No code changes required.             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Reusable Engine Code

```javascript
// The same MusicManager works for ANY game:

// Horror Game
const horrorMusicData = {
  spookyAmbience: { file: "spooky.mp3", criteria: { ... } }
};

// Racing Game
const racingMusicData = {
  highEnergy: { file: "rock.mp3", criteria: { ... } }
};

// Same manager, different data!
const horrorMusicManager = new MusicManager(horrorMusicData);
const racingMusicManager = new MusicManager(racingMusicData);
```

### 3. Easy Testing and Iteration

```javascript
// Want to test a specific dialog?
// Just temporarily modify its criteria:

testDialog: {
  file: "audio/dialog/test.mp3",
  criteria: { currentState: START_SCREEN }  // Override to play immediately
}

// No need to add special debug code!
```

### 4. Localization Friendly

```javascript
// dialogData.js
export const dialogTracks = {
  intro: {
    file: "audio/dialog/en/intro.mp3",  // English
    captions: [
      { text: "Welcome...", ... }
    ],
    criteria: { currentState: INTRO }
  }
};

// dialogData.es.js (Spanish)
export const dialogTracks = {
  intro: {
    file: "audio/dialog/es/intro.mp3",  // Spanish
    captions: [
      { text: "Bienvenido...", ... }
    ],
    criteria: { currentState: INTRO }
  }
};

// Just load different data file for different languages!
```

---

## Common Mistakes Beginners Make

### 1. Hard-Coding Content in Managers

```javascript
// ❌ WRONG: Content in code
class DialogManager {
  playIntroDialog() {
    this.play("audio/dialog/intro.mp3");  // Hard-coded!
  }
}

// ✅ CORRECT: Content in data
// dialogData.js
intro: {
  file: "audio/dialog/intro.mp3",
  criteria: { currentState: INTRO }
}
```

### 2. Forgetting Criteria

```javascript
// ❌ WRONG: When does this play?
{
  file: "music.mp3"
  // No criteria - never plays!
}

// ✅ CORRECT: Always include criteria
{
  file: "music.mp3",
  criteria: { currentState: SOME_STATE }
}
```

### 3. Inconsistent Data Structure

```javascript
// ❌ WRONG: Different structures
{
  track1: { file: "a.mp3", vol: 0.5 },    // "vol"
  track2: { file: "b.mp3", volume: 0.7 }  // "volume"
}

// ✅ CORRECT: Consistent naming
{
  track1: { file: "a.mp3", volume: 0.5 },
  track2: { file: "b.mp3", volume: 0.7 }
}
```

### 4. Not Using Meaningful Keys

```javascript
// ❌ WRONG: What is "track1"?
{
  track1: { file: "office.mp3", ... },
  track2: { file: "tension.mp3", ... }
}

// ✅ CORRECT: Descriptive keys
{
  officeAmbience: { file: "office.mp3", ... },
  tensionBuilder: { file: "tension.mp3", ... }
}
```

### 5. Over-Complicating Criteria

```javascript
// ❌ WRONG: Too complex
criteria: {
  $and: [
    { $or: [{ currentState: 1 }, { currentState: 2 }] },
    { $or: [{ sawRune: true }, { runeSightings: { $gt: 0 } }] }
  ]
}

// ✅ CORRECT: Simplify if possible
criteria: {
  currentState: { $in: [1, 2] },
  sawRune: true  // If this is what you really mean
}
```

---

## Creating Your Own Data Files

### Step 1: Define Your Content Type

What kind of content are you defining?

```
Example: Achievement system

What does each achievement need?
- id: Unique identifier
- name: Display name
- description: What the player did
- icon: Path to icon image
- criteria: When it unlocks
```

### Step 2: Create the Data File

```javascript
// achievementData.js
export const achievements = {
  firstSteps: {
    id: "first_steps",
    name: "First Steps",
    description: "Enter the office for the first time",
    icon: "icons/first-steps.png",
    criteria: {
      currentState: OFFICE_INTERIOR,
      hasEnteredOffice: false
    },
    onUnlock: {
      type: "notification",
      message: "Achievement Unlocked: First Steps!"
    }
  },

  runeSeeker: {
    id: "rune_seeker",
    name: "Rune Seeker",
    description: "Discover all 5 hidden runes",
    icon: "icons/rune-seeker.png",
    criteria: {
      runeSightings: { $gte: 5 }
    },
    onUnlock: {
      type: "unlock",
      content: "secret-ending"
    }
  }
};
```

### Step 3: Create a Manager (or Extend Existing)

```javascript
class AchievementManager {
  async initialize(gameManager, achievementData) {
    this.gameManager = gameManager;
    this.achievements = achievementData;
    this.unlocked = new Set();

    // Listen for state changes
    gameManager.on("state:changed", this.checkAchievements.bind(this));
  }

  checkAchievements(state) {
    Object.values(this.achievements).forEach(achievement => {
      if (this.unlocked.has(achievement.id)) return;

      if (matchesCriteria(achievement.criteria, state)) {
        this.unlock(achievement);
      }
    });
  }

  unlock(achievement) {
    this.unlocked.add(achievement.id);

    // Handle unlock action
    if (achievement.onUnlock.type === "notification") {
      this.showNotification(achievement.onUnlock.message);
    }
  }
}
```

### Step 4: Register the Manager

```javascript
// In main initialization
const achievementManager = new AchievementManager();
await achievementManager.initialize(gameManager, achievementData);
```

---

## Performance Considerations

### Data File Size

Large data files can slow down startup. Consider:

```javascript
// ❌ WRONG: One massive file
export const allGameData = {
  // 10,000+ items...
};

// ✅ CORRECT: Split into logical files
// dialogData.js - ~100 items
// musicData.js - ~20 items
// sceneData.js - ~200 items
```

### Criteria Complexity

Complex criteria are evaluated every state change:

```javascript
// ✅ FAST: Simple criteria
criteria: { currentState: 14 }

// ⚠️ SLOWER: Complex criteria
criteria: { $and: [{ $or: [...] }, { $or: [...] }] }
```

Consider caching results if needed.

---

## Next Steps

Now that you understand data-driven design:

- [Game State System](./game-state-system.md) - How criteria matching works
- [GameManager Deep Dive](./game-manager-deep-dive.md) - State management
- [Manager-Based Architecture](./manager-based-architecture-pattern.md) - How managers work
- [SceneManager](../03-scene-rendering/scene-manager.md) - Loading scene data
- [DialogManager](../05-media-systems/dialog-manager.md) - Dialog system example

---

## References

- [Data-Driven Programming (Wikipedia)](https://en.wikipedia.org/wiki/Data-driven_programming) - Programming paradigm overview
- [Data-Oriented Design](https://www.dataorienteddesign.com/dodbook/) - Book on DOD principles
- [Separation of Concerns](https://en.wikipedia.org/wiki/Separation_of_concerns) - Design principle
- [Configuration as Code](https://martinfowler.com/articles/configuration.html) - Martin Fowler on config

*Documentation last updated: January 12, 2026*
