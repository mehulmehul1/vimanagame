# Game State System & Criteria Helper - First Principles Guide

## Overview

The **Game State System** is how the Shadow Engine tracks and responds to everything happening in the game. It combines two powerful concepts:

1. **Centralized State Store** - All game data in one place
2. **Criteria Helper** - A MongoDB-style query language for matching state

This combination allows content (dialog, music, videos, etc.) to automatically play when certain game conditions are met - no complex `if` statements required!

## What You Need to Know First

Before understanding the state system, you should know:
- **JavaScript objects** - Key-value pairs for storing data
- **Comparison operators** - `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logical operators** - AND (`&&`), OR (`||`), NOT (`!`)
- **Arrays** - Lists of values like `[1, 2, 3]`
- **Basic database queries** (helpful but not required)

### Quick Refresher: Objects and Properties

```javascript
// A JavaScript object
const person = {
  name: "Alice",
  age: 30,
  isStudent: false,
  favoriteColors: ["blue", "green"]
};

// Accessing properties
person.name;        // "Alice"
person.age;         // 30
person.isStudent;   // false
person.favoriteColors; // ["blue", "green"]
```

---

## Part 1: The State Store

### What Is the State Store?

The state store is a **single JavaScript object** that holds ALL game state:

```javascript
// The complete game state
{
  // Lifecycle
  isPlaying: false,
  isPaused: false,

  // Narrative progression (which story moment we're in)
  currentState: 14,  // GAME_STATES.OFFICE_INTERIOR

  // Scene/World
  currentScene: null,
  currentZone: "office",

  // Player control
  controlEnabled: true,

  // View-Master (a game item)
  isViewmasterEquipped: false,
  viewmasterManuallyRemoved: false,
  viewmasterInsanityIntensity: 0.0,
  viewmasterOverheatCount: 0,

  // Drawing minigame
  drawingSuccessCount: 0,
  drawingFailureCount: 0,
  currentDrawingTarget: null,

  // Narrative flags
  sawRune: false,
  runeSightings: 0,

  // Player choices
  dialogChoice1: "EMPATH",  // Player's first dialog choice
  dialogChoice2: null,      // Player's second dialog choice

  // Platform detection
  isIOS: false,
  isSafari: false,
  isMobile: false
}
```

### Why a Single Object?

Having all state in one object provides:

1. **Single Source of Truth** - No confusion about where data lives
2. **Easy Debugging** - Can log the entire state at any time
3. **Simple Persistence** - Can save/load the whole state
4. **Predictable Updates** - All changes go through one method

### State Organization

The state is organized into categories:

```
┌─────────────────────────────────────────────────────────────┐
│                      GAME STATE                             │
├─────────────────────────────────────────────────────────────┤
│  LIFECYCLE          │  isPlaying, isPaused                  │
│  NARRATIVE          │  currentState, currentZone            │
│  PLAYER_CONTROL     │  controlEnabled                       │
│  ITEMS              │  isViewmasterEquipped, ...            │
│  MINIGAMES          │  drawingSuccessCount, ...             │
│  FLAGS              │  sawRune, runeSightings               │
│  CHOICES            │  dialogChoice1, dialogChoice2          │
│  PLATFORM           │  isIOS, isMobile, ...                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 2: Reading and Updating State

### Reading State

Use `gameManager.getState()` to get a **copy** of the current state:

```javascript
const state = gameManager.getState();

console.log(state.currentState);           // 14
console.log(state.controlEnabled);         // true
console.log(state.isViewmasterEquipped);   // false

// Can check conditions
if (state.currentState === GAME_STATES.OFFICE_INTERIOR) {
  console.log("Player is in the office!");
}
```

**Important:** `getState()` returns a **copy**, not the original. You can't modify state by changing the copy:

```javascript
const state = gameManager.getState();
state.currentState = 999;  // This does NOT change the actual state!
```

### Updating State

Use `gameManager.setState()` to update state:

```javascript
// Update single property
gameManager.setState({
  currentState: GAME_STATES.DRIVE_BY
});

// Update multiple properties
gameManager.setState({
  currentState: GAME_STATES.OFFICE_INTERIOR,
  controlEnabled: true,
  isViewmasterEquipped: true
});
```

**What happens when you call `setState()`:**

```
1. gameManager.setState({ currentState: 14 })
              │
              ▼
2. GameManager creates new state object
              │
              ▼
3. GameManager emits "state:changed" event
              │
              ├──▶ DialogManager receives event
              │        │
              │        ▼
              │   Checks if any dialog matches state 14
              │   Plays matching dialog
              │
              ├──▶ MusicManager receives event
              │        │
              │        ▼
              │   Checks if any music matches state 14
              │   Plays matching music
              │
              └──▶ All other managers do the same
```

---

## Part 3: The Criteria Helper

### What Is the Criteria Helper?

The **criteria helper** is a function that checks if a state object **matches** a set of criteria. It uses the same query syntax as MongoDB database queries.

**Basic example:**

```javascript
// Check if state matches criteria
const state = { currentState: 14, controlEnabled: true };
const criteria = { currentState: 14 };

matchesCriteria(criteria, state);  // true
```

### Why Use Criteria?

Without criteria, you'd write complex `if` statements:

```javascript
// ❌ WITHOUT criteria - complex and hard to read
function shouldPlayOfficeMusic(state) {
  if (state.currentState === 14 ||
      state.currentState === 15 ||
      state.currentState === 16 ||
      state.currentState === 17) {
    return true;
  }
  return false;
}
```

With criteria, it's declarative and clean:

```javascript
// ✅ WITH criteria - simple and clear
const officeMusic = {
  file: "office-music.mp3",
  criteria: { currentState: { $in: [14, 15, 16, 17] } }
};

// The criteria helper does the checking!
if (matchesCriteria(officeMusic.criteria, state)) {
  playMusic(officeMusic.file);
}
```

---

## Part 4: Criteria Operators

### Equality Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equals | `{ currentState: { $eq: 14 } }` |
| `$ne` | Not equals | `{ currentState: { $ne: 0 } }` |

**Examples:**

```javascript
// $eq - exact match (can be omitted, it's the default)
{ currentState: 14 }                    // Same as { currentState: { $eq: 14 } }
{ controlEnabled: true }                // Same as { controlEnabled: { $eq: true } }

// $ne - not equal
{ currentState: { $ne: 0 } }            // currentState is NOT 0
{ dialogChoice1: { $ne: null } }        // dialogChoice1 is NOT null
```

### Comparison Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$gt` | Greater than | `{ currentState: { $gt: 10 } }` |
| `$gte` | Greater or equal | `{ currentState: { $gte: 5 } }` |
| `$lt` | Less than | `{ currentState: { $lt: 20 } }` |
| `$lte` | Less or equal | `{ currentState: { $lte: 15 } }` |

**Examples:**

```javascript
// $gt - greater than
{ currentState: { $gt: 10 } }           // currentState > 10

// $gte - greater than or equal
{ currentState: { $gte: 5 } }            // currentState >= 5

// $lt - less than
{ currentState: { $lt: 20 } }            // currentState < 20

// $lte - less than or equal
{ currentState: { $lte: 15 } }           // currentState <= 15

// Combining for ranges
{ currentState: { $gte: 14, $lt: 20 } }  // 14 <= currentState < 20
```

**Real-world example from Shadow Engine:**

```javascript
// Music plays during office scenes (states 14-17)
const officeMusic = {
  file: "office-music.mp3",
  criteria: {
    currentState: { $gte: OFFICE_INTERIOR, $lt: LIGHTS_OUT }
  }
};
```

### Array Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$in` | In array | `{ currentState: { $in: [1, 2, 3] } }` |
| `$nin` | Not in array | `{ currentState: { $nin: [0, 1] } }` |

**Examples:**

```javascript
// $in - value is in the array
{ currentState: { $in: [5, 6, 7] } }     // currentState is 5, 6, or 7

// $nin - value is NOT in the array
{ currentState: { $nin: [0, 1] } }       // currentState is NOT 0 or 1

// Real example: dialog plays in multiple states
const introDialog = {
  file: "intro.mp3",
  criteria: { currentState: { $in: [INTRO, TITLE_SEQUENCE, TUTORIAL] } }
};
```

### Logical Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$and` | All conditions must be true | `{ $and: [{ currentState: 14 }, { controlEnabled: true }] }` |
| `$or` | At least one condition must be true | `{ $or: [{ currentState: 0 }, { currentState: 1 }] }` |
| `$not` | Condition must be false | `{ currentState: { $not: { $eq: 0 } } }` |

**Examples:**

```javascript
// $and - all must match
{ $and: [
  { currentState: 14 },
  { controlEnabled: true }
]}
// Same as shorthand: { currentState: 14, controlEnabled: true }

// $or - at least one must match
{ $or: [
  { currentState: 0 },
  { currentState: 1 }
]}
// currentState is 0 OR 1

// Complex example
{ $and: [
  { currentState: { $gte: 10 } },
  { $or: [
    { dialogChoice1: "EMPATH" },
    { dialogChoice1: "LOGIC" }
  ]}
]}
// currentState >= 10 AND (dialogChoice1 is EMPATH OR LOGIC)
```

### Modulo Operator

| Operator | Meaning | Example |
|----------|---------|---------|
| `$mod` | Modulo (remainder) | `{ count: { $mod: [2, 0] } }` |

**Example:**

```javascript
// $mod - divides and checks remainder
{ runeSightings: { $mod: [2, 0] } }      // runeSightings is even
{ count: { $mod: [3, 1] } }              // count % 3 equals 1
```

---

## Part 5: Real Examples from Shadow Engine

### Example 1: Dialog by State

```javascript
// From dialogData.js
export const dialogTracks = {
  intro: {
    file: "audio/dialog/intro.mp3",
    captions: [
      { text: "Welcome...", start: 0, end: 2000 },
      { text: "I've been waiting.", start: 2000, end: 5000 }
    ],
    criteria: { currentState: GAME_STATES.INTRO }
  },

  officeWelcome: {
    file: "audio/dialog/office-welcome.mp3",
    captions: [
      { text: "Ah, you're here!", start: 0, end: 1500 }
    ],
    criteria: {
      currentState: { $gte: OFFICE_INTERIOR, $lt: LIGHTS_OUT },
      sawRune: false
    }
  }
};
```

### Example 2: Music by State Range

```javascript
// From musicData.js
export const musicTracks = {
  officeAmbience: {
    file: "audio/music/office.mp3",
    volume: 0.7,
    loop: true,
    criteria: {
      currentState: { $gte: OFFICE_INTERIOR, $lt: LIGHTS_OUT }
    }
  },

  tensionMusic: {
    file: "audio/music/tension.mp3",
    volume: 0.9,
    loop: true,
    criteria: {
      currentState: { $in: [RUNE_SIGHTING_1, RUNE_SIGHTING_2, RUNE_SIGHTING_3] }
    }
  }
};
```

### Example 3: Video by Player Choice

```javascript
// From videoData.js
export const videos = {
  edisonEnding: {
    file: "video/edison-ending.webm",
    position: { x: 0, y: 1, z: -2 },
    criteria: { dialogChoice2: DIALOG_RESPONSE_TYPES.EDISON }
  },

  empathEnding: {
    file: "video/empath-ending.webm",
    position: { x: 0, y: 1, z: -2 },
    criteria: { dialogChoice2: DIALOG_RESPONSE_TYPES.EMPATH }
  }
};
```

### Example 4: Complex Criteria

```javascript
// VFX that only plays under specific conditions
export const vfxEffects = {
  runeGlow: {
    type: "glow",
    target: "runeObject",
    criteria: {
      $and: [
        { currentState: { $gte: RUNE_ENCOUNTER } },
        { sawRune: true },
        { runeSightings: { $gte: 3 } }
      ]
    }
  },

  viewmasterGlitch: {
    type: "glitch",
    criteria: {
      $and: [
        { isViewmasterEquipped: true },
        { viewmasterInsanityIntensity: { $gt: 0.5 } }
      ]
    }
  }
};
```

---

## Part 6: How Managers Use Criteria

### The Pattern

All media managers follow the same pattern:

```javascript
class SomeManager {
  async initialize(gameManager, data) {
    this.gameManager = gameManager;
    this.data = data;  // Array of items with criteria

    // Listen for state changes
    gameManager.on("state:changed", this.onStateChanged.bind(this));
  }

  onStateChanged(newState, oldState) {
    // Find all items matching the new state
    const matchingItems = this.data.filter(item => {
      return matchesCriteria(item.criteria, newState);
    });

    // Do something with matching items
    this.playItems(matchingItems);
  }
}
```

### Real Example: DialogManager

```javascript
class DialogManager {
  onStateChanged(newState, oldState) {
    // Find all dialog tracks matching new state
    const dialogToPlay = Object.values(this.dialogTracks)
      .filter(track => matchesCriteria(track.criteria, newState));

    // Play each matching dialog
    dialogToPlay.forEach(track => {
      this.playDialog(track);
    });
  }
}
```

---

## Part 7: Criteria Helper Implementation

Here's how the criteria helper is implemented:

```javascript
/**
 * Check if state matches criteria
 * @param {Object} criteria - The criteria to match
 * @param {Object} state - The state to check against
 * @returns {boolean} - True if state matches criteria
 */
function matchesCriteria(criteria, state) {
  // If no criteria, it matches
  if (!criteria) return true;

  // Check each condition
  for (const [key, value] of Object.entries(criteria)) {
    // Handle operators ($eq, $ne, $gt, etc.)
    if (key.startsWith('$')) {
      if (!evaluateOperator(key, value, state)) {
        return false;
      }
    }
    // Handle nested criteria
    else if (typeof value === 'object' && value !== null) {
      const stateValue = state[key];

      // Check if value is an operator object
      if (Object.keys(value).some(k => k.startsWith('$'))) {
        if (!matchesCriteria(value, { [key]: stateValue })) {
          return false;
        }
      }
      // Nested object comparison
      else if (!matchesCriteria(value, stateValue || {})) {
        return false;
      }
    }
    // Simple equality check
    else if (state[key] !== value) {
      return false;
    }
  }

  return true;
}

/**
 * Evaluate a single operator
 */
function evaluateOperator(operator, operand, state) {
  switch (operator) {
    case '$eq':
      return Object.values(state)[0] === operand;

    case '$ne':
      return Object.values(state)[0] !== operand;

    case '$gt':
      return Object.values(state)[0] > operand;

    case '$gte':
      return Object.values(state)[0] >= operand;

    case '$lt':
      return Object.values(state)[0] < operand;

    case '$lte':
      return Object.values(state)[0] <= operand;

    case '$in':
      return operand.includes(Object.values(state)[0]);

    case '$nin':
      return !operand.includes(Object.values(state)[0]);

    case '$and':
      return operand.every(condition =>
        matchesCriteria(condition, state)
      );

    case '$or':
      return operand.some(condition =>
        matchesCriteria(condition, state)
      );

    case '$mod':
      const [divisor, remainder] = operand;
      return Object.values(state)[0] % divisor === remainder;

    default:
      return false;
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Mutating State Directly

```javascript
// ❌ WRONG: Direct mutation doesn't emit events
gameManager.state.currentState = 14;

// ✅ CORRECT: Use setState
gameManager.setState({ currentState: 14 });
```

### 2. Forgetting Criteria Are Objects

```javascript
// ❌ WRONG: Criteria must be an object
criteria: currentState === 14

// ✅ CORRECT: Wrap in object
criteria: { currentState: 14 }
```

### 3. Not Handling Undefined State Values

```javascript
// ❌ WRONG: Will crash if dialogChoice2 is undefined
criteria: { dialogChoice2: { $ne: null } }

// ✅ CORRECT: Use $ne with safe check
criteria: { dialogChoice2: "EMPATH" }  // Only matches if explicitly set
```

### 4. Confusing $and Syntax

```javascript
// ❌ WRONG: $and needs an array
criteria: { $and: { currentState: 14, controlEnabled: true } }

// ✅ CORRECT: $and takes an array of conditions
criteria: { $and: [
  { currentState: 14 },
  { controlEnabled: true }
]}
```

### 5. Not Using Ranges

```javascript
// ❌ WRONG: Verbose listing
criteria: { currentState: { $in: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] } }

// ✅ CORRECT: Use range operators
criteria: { currentState: { $gte: 10, $lt: 20 } }
```

---

## Performance Considerations

### Criteria Evaluation Cost

Criteria are evaluated **every time state changes**. Keep them simple:

```javascript
// ✅ FAST: Simple comparisons
criteria: { currentState: 14 }

// ✅ FAST: Range check
criteria: { currentState: { $gte: 10, $lt: 20 } }

// ⚠️ SLOWER: Complex nested logic
criteria: { $and: [
  { $or: [{ currentState: 1 }, { currentState: 2 }] },
  { $or: [{ sawRune: true }, { runeSightings: { $gt: 0 } }] }
]}
```

### Caching Matching Items

Managers should cache which items match to avoid re-checking:

```javascript
class DialogManager {
  onStateChanged(newState) {
    const stateKey = JSON.stringify(newState);

    // Check cache
    if (this.matchCache[stateKey]) {
      return this.matchCache[stateKey];
    }

    // Find matches
    const matches = this.findMatches(newState);

    // Cache for next time
    this.matchCache[stateKey] = matches;

    return matches;
  }
}
```

---

## Next Steps

Now that you understand the state system:

- [GameManager Deep Dive](./game-manager-deep-dive.md) - The coordinator
- [Manager-Based Architecture](./manager-based-architecture-pattern.md) - How managers work
- [Data-Driven Design](./data-driven-design.md) - How data files use criteria
- [DialogManager](../05-media-systems/dialog-manager.md) - Dialog system
- [MusicManager](../05-media-systems/music-manager.md) - Music system

---

## References

- [MongoDB Query Operators](https://www.mongodb.com/docs/manual/reference/operator/query/) - Original inspiration
- [JavaScript Object Comparison](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/is) - Deep equality
- [State Management Patterns](https://www.patterns.dev/posts/state-pattern/) - State pattern in JS
- [Criteria Package](https://www.npmjs.com/package/criteria) - Similar criteria libraries

*Documentation last updated: January 12, 2026*
