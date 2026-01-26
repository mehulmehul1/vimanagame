# DialogManager - First Principles Guide

## Overview

The **DialogManager** handles all spoken dialogue and caption/subtitle display in the Shadow Engine. It plays audio dialog tracks synchronized with timed captions, using the Howler.js audio library for reliable cross-browser audio playback.

Think of DialogManager as the **voice and subtitle system** for your game - it ensures characters speak at the right moments and players can read along with captions.

## What You Need to Know First

Before understanding DialogManager, you should know:
- **Audio playback in browsers** - How Web Audio API works
- **Timing and synchronization** - Coordinating audio with text
- **Event-driven programming** - Reacting to state changes
- **Howler.js library** - The audio library used (verified v2.2.4+)

### Quick Refresher: Audio Timing

```javascript
// Audio files are measured in milliseconds (ms)
// 1000 ms = 1 second

const dialog = {
  file: "intro.mp3",
  captions: [
    { text: "Hello...", start: 0, end: 2000 },    // 0-2 seconds
    { text: "Welcome!", start: 2000, end: 4000 }  // 2-4 seconds
  ]
};

// Caption timing must match the audio perfectly!
```

---

## Part 1: Why Use a DialogManager?

### The Problem: Audio is Tricky in Browsers

Without a proper audio manager, you face:

```javascript
// ❌ WITHOUT DialogManager - Problems everywhere:

// Problem 1: Different browsers behave differently
const audio = new Audio("dialog.mp3");
audio.play(); // Might fail! No promise handling!

// Problem 2: Timing captions is hard
function showCaptions() {
  setTimeout(() => show("Hello"), 0);        // When does this actually show?
  setTimeout(() => show("Welcome!"), 2000);  // What if audio lags?
}

// Problem 3: No queue management
playDialog("intro.mp3");
playDialog("response.mp3");  // Both play at once!

// Problem 4: Can't interrupt or pause
// Once audio starts, it keeps going...
```

### The Solution: Dedicated Dialog System

```javascript
// ✅ WITH DialogManager - Clean and reliable:

// In dialogData.js:
intro: {
  file: "audio/dialog/intro.mp3",
  captions: [
    { text: "Hello...", start: 0, end: 2000 },
    { text: "Welcome!", start: 2000, end: 4000 }
  ],
  criteria: { currentState: GAME_STATES.INTRO }
}

// DialogManager handles:
// - Cross-browser audio (Howler.js)
// - Perfect caption synchronization
// - Queue and priority management
// - Pause/resume support
// - Volume and fade controls
```

---

## Part 2: Dialog Data Structure

### Basic Dialog Track

```javascript
// In dialogData.js
export const dialogTracks = {
  intro: {
    // Audio file path
    file: "audio/dialog/intro.mp3",

    // Volume (0.0 to 1.0)
    volume: 1.0,

    // Caption/subtitle data
    captions: [
      {
        text: "Welcome to the shadow realm...",
        start: 0,      // Start time in milliseconds
        end: 2500      // End time in milliseconds
      },
      {
        text: "I've been waiting for you.",
        start: 2500,
        end: 5000
      }
    ],

    // When this dialog plays
    criteria: { currentState: GAME_STATES.INTRO },

    // Optional: Priority for interrupting
    priority: 10
  }
};
```

### Dialog Track Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | string | Yes | Path to audio file |
| `volume` | number | No | Playback volume (0-1, default: 1.0) |
| `captions` | array | Yes | Array of caption objects |
| `criteria` | object | Yes | State criteria for playback |
| `priority` | number | No | Higher values interrupt current dialog |
| `loop` | boolean | No | Whether to loop the audio |
| `fadeIn` | number | No | Fade-in duration in ms |
| `fadeOut` | number | No | Fade-out duration in ms |

### Caption Object Properties

| Property | Type | Description |
|----------|------|-------------|
| `text` | string | The caption text to display |
| `start` | number | Start time in milliseconds |
| `end` | number | End time in milliseconds |
| `speaker` | string (optional) | Speaker name/label |

---

## Part 3: How DialogManager Works

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DialogManager                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Dialog Queue                              │   │
│  │  [Dialog 1] → [Dialog 2] → [Dialog 3]                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Current Playback State                       │   │
│  │  - currentDialog: reference                                 │   │
│  │  - currentHowl: Howl instance                              │   │
│  │  - currentTime: milliseconds                                │   │
│  │  - activeCaptions: array                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Caption Display                             │   │
│  │  - caption element positioning                             │   │
│  │  - text updates synchronized with audio                    │   │
│  │  - fade in/out transitions                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                            │
         │ listens to                 │ emits
         ▼                            ▼
┌─────────────────┐          ┌─────────────────┐
│  GameManager    │          │   UIManager     │
│  "state:changed"│          │   Caption UI     │
└─────────────────┘          └─────────────────┘
```

### Playback Flow

```
1. State changes (GameManager emits "state:changed")
              │
              ▼
2. DialogManager receives event
              │
              ▼
3. Find all dialog tracks matching new state
              │
              ▼
4. Sort by priority (highest first)
              │
              ▼
5. Queue matching dialogs
              │
              ▼
6. Play first dialog in queue
    │
    ├──▶ Load audio file (Howler.js)
    │
    ├──▶ Start audio playback
    │
    ├──▶ Begin caption synchronization loop
    │       │
    │       ├──▶ Each frame: check audio time
    │       │
    │       ├──▶ Update active captions
    │       │
    │       └──▶ Fade in/out as needed
    │
    └──▶ On complete: play next in queue
```

---

## Part 4: Caption Synchronization

### The Synchronization Loop

```javascript
class DialogManager {
  updateCaptionSync() {
    if (!this.currentDialog || !this.currentHowl) return;

    // Get current audio time in milliseconds
    const currentTime = this.currentHowl.seek() * 1000;

    // Find all captions that should be visible now
    const activeCaptions = this.currentDialog.captions.filter(
      caption => currentTime >= caption.start && currentTime < caption.end
    );

    // Update caption display
    this.displayCaptions(activeCaptions);
  }
}
```

### Why requestAnimationFrame?

DialogManager uses `requestAnimationFrame` for caption updates:

```javascript
// ✅ CORRECT: Uses game loop
function gameLoop() {
  dialogManager.updateCaptionSync();
  requestAnimationFrame(gameLoop);  // Sync with frame rate
}

// ❌ WRONG: Uses setInterval
setInterval(() => {
  dialogManager.updateCaptionSync();
}, 16);  // Can drift out of sync!
```

**Why?**
- Syncs with browser's paint cycle (60fps)
- Pauses automatically when tab is inactive
- Prevents timing drift
- More efficient than setInterval

---

## Part 5: Queue and Priority Management

### Dialog Queue Behavior

```javascript
// DialogManager plays ONE dialog at a time

// If dialog is already playing:
playDialog(newDialog) {
  if (this.isPlaying() && newDialog.priority <= this.currentPriority) {
    // Add to queue (plays after current finishes)
    this.queue.push(newDialog);
  } else if (newDialog.priority > this.currentPriority) {
    // Interrupt current dialog!
    this.interruptAndPlay(newDialog);
  } else {
    // Play immediately (nothing playing)
    this.playImmediately(newDialog);
  }
}
```

### Priority Example

```javascript
// In dialogData.js
export const dialogTracks = {
  // Low priority - ambient chatter
  ambientChatter: {
    file: "chatter.mp3",
    priority: 1,
    criteria: { currentZone: "office" }
  },

  // Medium priority - story dialog
  storyDialog: {
    file: "story.mp3",
    priority: 5,
    criteria: { currentState: STORY_BEAT_1 }
  },

  // High priority - critical information
  criticalWarning: {
    file: "warning.mp3",
    priority: 10,
    criteria: { dangerLevel: { $gt: 0.8 } }
  }
};

// Playback example:
// 1. ambientChatter is playing
// 2. storyDialog triggers → Interrupts ambient, plays story
// 3. criticalWarning triggers → Interrupts story, plays warning
```

---

## Part 6: Howler.js Integration

### Howler.js API Reference (Verified v2.2.4+)

Based on the official Howler.js documentation:

```javascript
import { Howl, Howler } from 'howler';

// Creating a sound instance
const sound = new Howl({
  src: ['dialog.mp3'],
  volume: 1.0,
  html5: false,  // Use Web Audio API (better timing)
  loop: false
});

// Basic methods
sound.play();        // Start playback
sound.pause();       // Pause playback
sound.stop();        // Stop and reset to beginning
sound.seek(2.5);     // Jump to 2.5 seconds

// Getting current position
const currentTime = sound.seek();  // Returns position in seconds

// Events
sound.on('play', () => console.log('Started playing'));
sound.on('end', () => console.log('Finished playing'));
sound.on('loaderror', (id, error) => console.error('Load failed!', error));

// Volume control
sound.volume(0.5);  // Set to 50%
sound.fade(1.0, 0.0, 1000);  // Fade out over 1 second
```

### DialogManager Wraps Howler

```javascript
class DialogManager {
  playDialog(dialog) {
    // Create Howl instance
    this.currentHowl = new Howl({
      src: [dialog.file],
      volume: dialog.volume || 1.0,
      html5: false,
      loop: dialog.loop || false,

      // Event handlers
      onplay: () => {
        this.isPlaying = true;
        this.startCaptionSync();
      },

      onend: () => {
        this.isPlaying = false;
        this.stopCaptionSync();
        this.playNextInQueue();
      },

      onloaderror: (id, error) => {
        console.error(`Failed to load dialog: ${dialog.file}`, error);
        this.playNextInQueue();  // Skip failed dialog
      }
    });

    // Apply fade in if specified
    if (dialog.fadeIn) {
      this.currentHowl.volume(0);  // Start silent
      this.currentHowl.play();
      this.currentHowl.fade(0, dialog.volume, dialog.fadeIn);
    } else {
      this.currentHowl.play();
    }
  }
}
```

---

## Part 7: Caption Display System

### Caption Element Structure

```html
<!-- In your HTML -->
<div id="caption-container" class="caption-container hidden">
  <div id="caption-text" class="caption-text"></div>
  <div id="caption-speaker" class="caption-speaker"></div>
</div>
```

### CSS Styling

```css
.caption-container {
  position: absolute;
  bottom: 10%;
  left: 50%;
  transform: translateX(-50%);
  text-align: center;
  pointer-events: none;
  transition: opacity 0.2s ease;
}

.caption-container.hidden {
  opacity: 0;
}

.caption-text {
  font-size: 1.5rem;
  color: white;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
  background: rgba(0, 0, 0, 0.5);
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
}

.caption-speaker {
  font-size: 1rem;
  color: #aaa;
  margin-bottom: 0.25rem;
}
```

### Displaying Captions

```javascript
class DialogManager {
  displayCaptions(captions) {
    if (!captions || captions.length === 0) {
      // Hide caption container
      this.captionContainer.classList.add('hidden');
      return;
    }

    // Show caption container
    this.captionContainer.classList.remove('hidden');

    // Update caption text (handle multiple captions)
    const captionText = captions
      .map(c => c.text)
      .join('\n');  // Multiple captions on separate lines

    this.captionTextElement.textContent = captionText;

    // Update speaker if present
    if (captions[0].speaker) {
      this.captionSpeakerElement.textContent = captions[0].speaker;
      this.captionSpeakerElement.style.display = 'block';
    } else {
      this.captionSpeakerElement.style.display = 'none';
    }
  }
}
```

---

## Part 8: Advanced Features

### Crossfading Between Dialogs

```javascript
// In dialogData.js
officeToHallway: {
  file: "audio/dialog/leaving-office.mp3",
  volume: 1.0,
  fadeOut: 2000,  // Fade out current dialog over 2 seconds
  captions: [
    { text: "I should keep moving...", start: 0, end: 3000 }
  ],
  criteria: {
    currentState: OFFICE_EXIT,
    inSpookyRoom: true
  }
}
```

### Dialog Chains (playNext)

```javascript
// DialogManager can automatically play the next dialog
// when one completes, based on state:

export const dialogTracks = {
  part1: {
    file: "part1.mp3",
    captions: [{ text: "First part...", start: 0, end: 2000 }],
    criteria: { currentState: CONVERSATION_START }
  },

  part2: {
    file: "part2.mp3",
    captions: [{ text: "Second part...", start: 0, end: 2000 }],
    criteria: { currentState: CONVERSATION_CONTINUE }
  }
};

// GameManager transitions between states,
// DialogManager plays each as the state changes
```

### Conditional Dialog Based on Choices

```javascript
// Different dialog based on player choice

export const dialogTracks = {
  // If player chose EMPATH path
  empathResponse: {
    file: "audio/dialog/empath-response.mp3",
    captions: [
      { text: "I understand how you feel.", start: 0, end: 3000 }
    ],
    criteria: {
      currentState: DIALOG_RESPONSE,
      dialogChoice1: "EMPATH"
    }
  },

  // If player chose LOGIC path
  logicResponse: {
    file: "audio/dialog/logic-response.mp3",
    captions: [
      { text: "That doesn't make sense.", start: 0, end: 2500 }
    ],
    criteria: {
      currentState: DIALOG_RESPONSE,
      dialogChoice1: "LOGIC"
    }
  }
};
```

### Interruptible vs Non-Interruptible

```javascript
// Some dialog shouldn't be interrupted!

export const dialogTracks = {
  // Can be interrupted (default)
  ambientChatter: {
    file: "chatter.mp3",
    priority: 1,
    interruptible: true,  // Can be cut off
    criteria: { currentZone: "office" }
  },

  // Cannot be interrupted
  criticalStory: {
    file: "critical.mp3",
    priority: 10,
    interruptible: false,  // Must play fully
    criteria: { currentState: STORY_CLIMAX }
  }
};
```

---

## Part 9: Pause and Resume

### Handling Game Pause

```javascript
class DialogManager {
  pause() {
    if (this.currentHowl && this.isPlaying) {
      this.wasPlaying = true;
      this.currentHowl.pause();
      this.stopCaptionSync();  // Stop checking captions
    }
  }

  resume() {
    if (this.currentHowl && this.wasPlaying) {
      this.currentHowl.play();
      this.wasPlaying = false;
      this.startCaptionSync();  // Resume caption checking
    }
  }
}

// GameManager calls these when pausing/resuming:
gameManager.on("game:paused", () => dialogManager.pause());
gameManager.on("game:resumed", () => dialogManager.resume());
```

---

## Common Mistakes Beginners Make

### 1. Caption Timing Doesn't Match Audio

```javascript
// ❌ WRONG: Timing is off
captions: [
  { text: "Hello", start: 0, end: 1 }  // 1ms? Way too short!
]

// ✅ CORRECT: Use milliseconds from actual audio
captions: [
  { text: "Hello", start: 0, end: 2500 }  // 2.5 seconds
]
```

### 2. Forgetting to Stop Caption Sync

```javascript
// ❌ WRONG: Caption sync keeps running!
onend: () => {
  this.isPlaying = false;
  // Forgot to stop update loop!
}

// ✅ CORRECT: Clean up properly
onend: () => {
  this.isPlaying = false;
  this.stopCaptionSync();  // Stop the loop!
}
```

### 3. Not Handling Load Errors

```javascript
// ❌ WRONG: No error handling
new Howl({
  src: [dialog.file],
  onplay: () => { /* ... */ }
  // What if file doesn't exist?
});

// ✅ CORRECT: Handle errors
new Howl({
  src: [dialog.file],
  onloaderror: (id, error) => {
    console.error(`Failed to load: ${dialog.file}`, error);
    // Continue to next dialog
    this.playNextInQueue();
  }
});
```

### 4. Using html5: true Unnecessarily

```javascript
// ❌ WRONG: Forces HTML5 Audio (worse timing)
new Howl({
  src: [dialog.file],
  html5: true  // Poor timing, no seek support
});

// ✅ CORRECT: Let Howler choose (default is Web Audio)
new Howl({
  src: [dialog.file],
  html5: false  // Web Audio API (better timing)
});
```

### 5. Not Considering Different Languages

```javascript
// ❌ WRONG: Hard-coded English
captions: [
  { text: "Hello!", start: 0, end: 1000 }
]

// ✅ CORRECT: Support localization
captions: [
  { text: getLocalizedText("hello"), start: 0, end: 1000 }
]

// Or separate data files:
// dialogData.en.js, dialogData.es.js, etc.
```

---

## Performance Considerations

### Audio File Format

Use formats that balance quality and size:

| Format | Browser Support | Quality/Size |
|--------|-----------------|--------------|
| MP3 | Best | Good compression |
| OGG | Good | Open, similar to MP3 |
| WAV | Poor | Uncompressed (large!) |

**Recommendation:** Use MP3 at 128-192 kbps for dialog.

### Preloading vs Streaming

```javascript
// For frequent dialog, preload:
const dialog = new Howl({
  src: ['frequent-line.mp3'],
  preload: true  // Load into memory immediately
});

// For rare/long dialog, let it stream:
const dialog = new Howl({
  src: ['rare-long-line.mp3'],
  preload: false  // Load when played
});
```

### Concurrency Limits

Howler.js has limits on simultaneous sounds:

```javascript
// Default: Can play multiple sounds at once
// But too many = performance issues

// Solution: Prioritize important dialog
if (Howler._howls.length > MAX_CONCURRENT) {
  // Stop lowest-priority dialog
  this.stopLowestPriority();
}
```

---

## 🎮 Game Design Perspective

### Creative Intent

Dialog serves multiple game design purposes:

1. **Exposition** - Delivering story information
2. **Character voice** - Revealing personality through speech patterns
3. **Pacing control** - Dialog can slow down or speed up moments
4. **Atmosphere** - Ambient chatter brings life to spaces

### Dialog Pacing Techniques

```javascript
// Technique 1: Staggered information
part1: {
  file: "hint1.mp3",
  captions: [{ text: "Something feels off...", start: 0, end: 3000 }],
  criteria: { currentState: AREA_ENTER }
},

part2: {
  file: "hint2.mp3",
  captions: [{ text: "I can't quite place it.", start: 0, end: 3000 }],
  criteria: { currentState: AREA_EXPLORE }
}

// Player gets info gradually as they explore
```

---

## 🎨 Level Design Perspective

### How To Design Dialog Like This

#### Step 1: Map Story Beats

```
Story Flow:
Start → Discovery → Tension → Reveal → Resolution

Each beat needs corresponding dialog.
```

#### Step 2: Write and Record

```
1. Write script with timing marks
2. Record voiceover
3. Export audio (MP3, 128kbps, mono is fine for speech)
4. Note timing in your data file
```

#### Step 3: Define Criteria

```javascript
// Each dialog plays exactly when needed
discoveryDialog: {
  file: "discovery.mp3",
  criteria: {
    currentState: DISCOVERY_MOMENT,
    sawRune: false  // Only if player hasn't seen it yet
  }
}
```

---

## Next Steps

Now that you understand DialogManager:

- [MusicManager](../05-media-systems/music-manager.md) - Background music system
- [SFXManager](../05-media-systems/sfx-manager.md) - Sound effects system
- [VideoManager](../05-media-systems/video-manager.md) - Video playback system
- [GameManager Deep Dive](../02-core-architecture/game-manager-deep-dive.md) - State coordination

---

## References

- [Howler.js Official Documentation](https://howlerjs.com/) - Verified v2.2.4+
- [Howler.js GitHub](https://github.com/goldfire/howler.js) - Source code and issues
- [Web Audio API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) - Underlying API
- [Audio Sprite Best Practices](https://howlerjs.com/#sprites) - Efficient audio management
- [Caption Accessibility Guidelines](https://www.w3.org/TR/webvtt1/) - WebVTT caption format

*Documentation last updated: January 12, 2026*
