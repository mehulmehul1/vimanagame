# MusicManager - First Principles Guide

## Overview

The **MusicManager** handles background music playback with smooth crossfade transitions in the Shadow Engine. It manages the emotional backdrop of your game - from subtle ambience to tension-building crescendos - ensuring that music changes are seamless rather than jarring.

Think of MusicManager as the **DJ for your game world** - just as a skilled DJ crossfades between songs to maintain the mood at a party, MusicManager smoothly transitions between music tracks based on what's happening in your game.

## What You Need to Know First

Before understanding MusicManager, you should know:
- **Howler.js library** - Web audio library (v2.2.4+) used for playback
- **Audio formats** - MP3, OGG, and web audio formats
- **Volume and fades** - Gradual volume changes for smooth transitions
- **Event-driven programming** - Reacting to game state changes
- **The manager pattern** - Single responsibility for music system

### Quick Refresher: Crossfading

```
CROSSFADE: Smoothly transitioning between two audio tracks

Track A Volume:  ████████████░░░░░░░░░░  fades out
Track B Volume:  ░░░░░░░░░░░░████████████  fades in

Time: ────────────────────────────────────▶
       0s      2s      4s      6s      8s
                └───────┘
             crossfade period
             (typically 2-5 seconds)

WITHOUT crossfade:  ████████│░░│░░░░░██████  ← Jarring cut!
WITH crossfade:     ███████████░░░░█████████  ← Smooth!
```

**Why crossfade matters:** In games, abrupt music changes break immersion. A smooth crossfade keeps players in the experience.

---

## Part 1: Why Use a Dedicated Music Manager?

### The Problem: Basic Audio is Limiting

Without a dedicated music system:

```javascript
// ❌ WITHOUT MusicManager - Jarring and manual
function playTensionMusic() {
  // Stop current music abruptly
  currentAudio.pause();

  // Start new music at full volume
  const tensionAudio = new Audio("music/tension.mp3");
  tensionAudio.volume = 1.0;
  tensionAudio.play();

  // Player notices the harsh transition!
}
```

**Problems:**
- Abrupt cuts between tracks
- No volume control
- Can't coordinate with game events
- Manual cleanup required
- No looping control

### The Solution: MusicManager

```javascript
// ✅ WITH MusicManager - Smooth and automatic
musicManager.crossfadeTo("tensionMusic", {
  duration: 3000,  // 3 second crossfade
  volume: 0.8
});

// Old track fades out, new track fades in simultaneously
// Player barely notices the transition!
```

**Benefits:**
- Seamless transitions between tracks
- Volume control with fades
- Automatic looping
- State-driven playback
- Memory-efficient audio management

---

## Part 2: Music Data Structure

### Basic Music Track Definition

```javascript
// In musicData.js
export const musicTracks = {
  mainMenu: {
    // Audio file path (MP3 recommended for compatibility)
    file: "audio/music/menu-theme.mp3",

    // Base volume (0.0 to 1.0)
    volume: 0.6,

    // Should this track loop?
    loop: true,

    // Fade in duration (milliseconds)
    fadeIn: 2000,

    // Fade out duration (milliseconds)
    fadeOut: 3000,

    // When this music plays
    criteria: { currentState: GAME_STATES.START_SCREEN }
  },

  officeAmbience: {
    file: "audio/music/office-drone.mp3",
    volume: 0.4,
    loop: true,
    fadeIn: 5000,  // Longer fade for atmosphere
    fadeOut: 3000,
    criteria: {
      currentState: { $gte: OFFICE_INTERIOR, $lt: LIGHTS_OUT }
    }
  },

  tensionBuilder: {
    file: "audio/music/tension-rise.mp3",
    volume: 0.8,
    loop: true,
    fadeIn: 1000,  // Quick fade for urgency
    fadeOut: 2000,
    criteria: {
      currentState: { $in: [RUNE_SIGHTING_1, RUNE_SIGHTING_2, RUNE_SIGHTING_3] }
    }
  }
};
```

### Music Track Properties Reference

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `file` | string | Yes | Path to audio file (MP3/OGG) |
| `volume` | number | No | Base volume (0.0-1.0, default: 1.0) |
| `loop` | boolean | No | Whether to loop (default: true) |
| `fadeIn` | number | No | Fade-in duration in ms |
| `fadeOut` | number | No | Fade-out duration in ms |
| `criteria` | object | Yes | State criteria for playback |
| `crossfadeTo` | string | No | Auto-transition to this track |
| `priority` | number | No | Higher priority overrides current |

---

## Part 3: How MusicManager Works

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MusicManager                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Active Tracks                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │   │
│  │  │   Current       │  │   Previous      │                   │   │
│  │  │   (fading out)  │  │   (cleaning up) │                   │   │
│  │  └─────────────────┘  └─────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Crossfade Controller                            │   │
│  │  - Manages volume transitions                               │   │
│  │  - Coordinates fade-in/fade-out timing                      │   │
│  │  - Ensures seamless transitions                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Howler.js Integration                         │   │
│  │  - Creates Howl instances for each track                    │   │
│  │  - Uses Web Audio API for smooth fades                      │   │
│  │  - Handles streaming for large files                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                            │
         │ listens to                 │ plays audio
         ▼                            ▼
┌─────────────────┐          ┌─────────────────┐
│  GameManager    │          │   Web Audio     │
│  "state:changed"│          │   (Speakers)     │
└─────────────────┘          └─────────────────┘
```

### Music Playback Flow

```
1. State changes (GameManager emits "state:changed")
              │
              ▼
2. MusicManager receives event
              │
              ▼
3. Find tracks matching new state
              │
              ▼
4. Determine playback action:
    │
    ├──▶ If different from current: CROSSFADE
    │       │
    │       ├── Fade out current track
    │       │   (over fadeOut duration)
    │       │
    │       └── Fade in new track
    │           (over fadeIn duration)
    │
    ├──▶ If same as current: CONTINUE
    │
    └──▶ If no match: FADE OUT
              │
              ▼
5. Update volume each frame during transition
              │
              ▼
6. Stop and dispose of faded-out tracks
```

---

## Part 4: Crossfade Implementation

### The Crossfade Algorithm

```javascript
class MusicManager {
  crossfadeTo(newTrackKey, duration = 3000) {
    const newTrack = this.data[newTrackKey];
    const currentHowl = this.currentHowl;

    // If nothing playing, just start new track
    if (!currentHowl) {
      this.playTrack(newTrackKey);
      return;
    }

    // Don't crossfade to same track
    if (this.currentTrackKey === newTrackKey) {
      return;
    }

    // Create new Howl instance (starts at volume 0)
    const newHowl = new Howl({
      src: [newTrack.file],
      volume: 0,  // Start silent
      loop: newTrack.loop !== false,
      html5: false  // Use Web Audio API for smooth fades
    });

    // Start playback
    newHowl.play();

    // Calculate fade step sizes
    const fadeSteps = 60;  // Update at 60fps
    const stepDuration = duration / fadeSteps;
    const currentVolume = this.currentTrack.volume || 1.0;
    const newVolume = newTrack.volume || 1.0;
    const fadeOutStep = currentVolume / fadeSteps;
    const fadeInStep = newVolume / fadeSteps;

    // Crossfade loop
    let step = 0;
    const fadeInterval = setInterval(() => {
      step++;

      // Fade out current
      const newCurrentVol = Math.max(0, currentHowl.volume() - fadeOutStep);
      currentHowl.volume(newCurrentVol);

      // Fade in new
      const newNewVol = Math.min(newVolume, newHowl.volume() + fadeInStep);
      newHowl.volume(newNewVol);

      // Check if crossfade complete
      if (step >= fadeSteps) {
        clearInterval(fadeInterval);
        currentHowl.stop();  // Stop old track completely
        this.currentHowl = newHowl;
        this.currentTrackKey = newTrackKey;
        this.currentTrack = newTrack;
      }
    }, stepDuration);
  }
}
```

### Using requestAnimationFrame for Smooth Fades

A better approach using the game loop:

```javascript
class MusicManager {
  crossfadeTo(newTrackKey, duration) {
    // ... setup code ...

    this.crossfadeState = {
      currentHowl,
      newHowl,
      startTime: performance.now(),
      duration,
      startVolumeCurrent: currentHowl.volume(),
      targetVolumeNew: newTrack.volume || 1.0
    };
  }

  update(deltaTime) {
    if (!this.crossfadeState) return;

    const elapsed = performance.now() - this.crossfadeState.startTime;
    const progress = Math.min(1, elapsed / this.crossfadeState.duration);

    // Apply smooth easing (ease-in-out)
    const eased = this.easeInOutCubic(progress);

    // Update volumes
    this.crossfadeState.currentHowl.volume(
      this.crossfadeState.startVolumeCurrent * (1 - eased)
    );
    this.crossfadeState.newHowl.volume(
      this.crossfadeState.targetVolumeNew * eased
    );

    // Clean up when done
    if (progress >= 1) {
      this.crossfadeState.currentHowl.stop();
      this.currentHowl = this.crossfadeState.newHowl;
      this.currentTrackKey = this.newTrackKey;
      this.crossfadeState = null;
    }
  }

  easeInOutCubic(t) {
    return t < 0.5
      ? 4 * t * t * t
      : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }
}
```

---

## Part 5: Audio File Formats and Optimization

### Format Comparison

| Format | Pros | Cons | Best For |
|--------|------|------|----------|
| **MP3** | Universal support | Patent issues (expired) | Main music tracks |
| **OGG** | Open source, good compression | Slightly less support | Web games |
| **AAC** | Excellent quality | Licensing for encoding | Apple ecosystems |
| **OPUS** | Best quality/size | Newer, less support | Future-proofing |

### Recommended Settings

```bash
# FFmpeg command for high-quality MP3
ffmpeg -i input.wav \
  -codec:a libmp3lame \
  -b:a 192k \
  -qscale:a 2 \
  output.mp3

# For looping tracks, add seamless loop info
ffmpeg -i input.wav \
  -codec:a libmp3lame \
  -b:a 192k \
  -metadata loop_start="0" \
  -metadata loop_end="45.5" \
  output.mp3
```

### Bitrate Guidelines

| Use Case | Bitrate | File Size (per minute) |
|----------|---------|----------------------|
| Ambience/Background | 128 kbps | ~1 MB |
| Standard Music | 192 kbps | ~1.5 MB |
| Featured/Cinematic | 256-320 kbps | ~2-2.5 MB |

---

## Part 6: State-Driven Music System

### How Music Responds to Game State

```javascript
// MusicManager listens to state changes
class MusicManager {
  listenTo(gameManager) {
    gameManager.on("state:changed", this.onStateChanged.bind(this));
  }

  onStateChanged(newState, oldState) {
    // Find all tracks matching new state
    const matchingTracks = this.getMatchingTracks(newState);

    if (matchingTracks.length === 0) {
      // No music for this state - fade out current
      this.fadeOutAll();
      return;
    }

    // Sort by priority (highest first)
    matchingTracks.sort((a, b) => (b.priority || 0) - (a.priority || 0));

    const bestMatch = matchingTracks[0];

    // Crossfade to the best match
    if (bestMatch.key !== this.currentTrackKey) {
      const duration = this.calculateCrossfadeDuration(
        this.currentTrack,
        bestMatch
      );
      this.crossfadeTo(bestMatch.key, duration);
    }
  }

  calculateCrossfadeDuration(fromTrack, toTrack) {
    // Use track-specific durations if defined
    if (toTrack.fadeIn && fromTrack?.fadeOut) {
      return Math.max(toTrack.fadeIn, fromTrack.fadeOut);
    }
    // Default based on mood transition
    return 3000;  // 3 seconds
  }
}
```

### Music State Examples

```javascript
export const musicTracks = {
  // Menu music - low key, welcoming
  menuTheme: {
    file: "audio/music/menu.mp3",
    volume: 0.5,
    loop: true,
    fadeIn: 2000,
    fadeOut: 2000,
    criteria: {
      currentState: { $in: [START_SCREEN, PAUSE_MENU, OPTIONS_MENU] }
    }
  },

  // Exploration - calm, mysterious
  explorationAmbience: {
    file: "audio/music/exploration.mp3",
    volume: 0.4,
    loop: true,
    fadeIn: 5000,
    fadeOut: 5000,
    priority: 1,
    criteria: {
      currentState: { $gte: EXPLORATION_START, $lt: COMBAT_START },
      isInCombat: false
    }
  },

  // Combat - intense, driving
  combatMusic: {
    file: "audio/music/combat.mp3",
    volume: 0.9,
    loop: true,
    fadeIn: 500,  // Quick fade for action
    fadeOut: 2000,
    priority: 10,  // High priority overrides exploration
    criteria: {
      isInCombat: true,
      enemyHealth: { $gt: 0 }
    }
  },

  // Victory - triumphant
  victoryFanfare: {
    file: "audio/music/victory.mp3",
    volume: 1.0,
    loop: false,  // Play once
    fadeIn: 500,
    fadeOut: 3000,
    priority: 5,
    criteria: {
      isInCombat: false,
      justWonCombat: true
    }
  },

  // Boss battle - epic
  bossBattle: {
    file: "audio/music/boss.mp3",
    volume: 1.0,
    loop: true,
    fadeIn: 200,
    fadeOut: 2000,
    priority: 20,  // Highest priority
    criteria: {
      currentState: BOSS_BATTLE,
      bossHealth: { $gt: 0 }
    }
  }
};
```

---

## Part 7: Advanced Music Features

### Layered Music (Dynamic Mixing)

```javascript
// For dynamic intensity changes
class MusicManager {
  playLayeredMusic(baseTrack, layers) {
    // Base layer always plays
    this.playTrack(baseTrack);

    // Add layers based on game state
    this.layers = layers.map(layer => ({
      howl: new Howl({
        src: [layer.file],
        volume: 0,
        loop: true
      }),
      condition: layer.condition
    }));

    // Start layers (silent)
    this.layers.forEach(layer => layer.howl.play());
  }

  updateLayers(gameState) {
    this.layers.forEach(layer => {
      const shouldPlay = layer.condition(gameState);
      const targetVolume = shouldPlay ? 1.0 : 0;
      // Smooth volume transition
      this.fadeTo(layer.howl, targetVolume, 1000);
    });
  }
}
```

### Stinger Sounds (Short Musical Phrases)

```javascript
export const musicStingers = {
  discovery: {
    file: "audio/music/stingers/discovery.mp3",
    volume: 0.8,
    criteria: {
      justFoundSecret: true
    }
  },

  jumpScare: {
    file: "audio/music/stingers/scare.mp3",
    volume: 1.0,
    criteria: {
      currentState: JUMP_SCARE_TRIGGER
    }
  }
};
```

### Music for Different Game Sections

```javascript
export const musicTracks = {
  // Hub area - safe, welcoming
  hubArea: {
    file: "audio/music/hub.mp3",
    volume: 0.5,
    loop: true,
    fadeIn: 3000,
    criteria: { currentZone: "hub" }
  },

  // Dangerous area - tense
  dangerZone: {
    file: "audio/music/danger.mp3",
    volume: 0.7,
    loop: true,
    fadeIn: 2000,
    criteria: { currentZone: "danger_zone" }
  },

  // Boss arena - epic
  bossArena: {
    file: "audio/music/boss-arena.mp3",
    volume: 1.0,
    loop: true,
    fadeIn: 500,
    criteria: { currentZone: "boss_arena" }
  }
};
```

---

## Part 8: Common Music Use Cases

### 1. Menu Music

```javascript
mainMenu: {
  file: "audio/music/menu-theme.mp3",
  volume: 0.6,
  loop: true,
  fadeIn: 2000,
  fadeOut: 2000,
  criteria: {
    currentState: { $in: [START_SCREEN, MAIN_MENU] }
  }
}
```

### 2. Dynamic Intensity

```javascript
// Three intensity levels
calmMusic: {
  file: "audio/music/calm.mp3",
  volume: 0.4,
  criteria: { intensity: { $lt: 0.3 } }
},

mediumMusic: {
  file: "audio/music/medium.mp3",
  volume: 0.6,
  criteria: { intensity: { $gte: 0.3, $lt: 0.7 } }
},

intenseMusic: {
  file: "audio/music/intense.mp3",
  volume: 0.9,
  criteria: { intensity: { $gte: 0.7 } }
}
```

### 3. Character Themes

```javascript
characterThemes: {
  edisonTheme: {
    file: "audio/music/themes/edison.mp3",
    volume: 0.5,
    criteria: {
      currentConversation: "edison",
      dialogTurn: "edison"
    }
  },

  empathTheme: {
    file: "audio/music/themes/empath.mp3",
    volume: 0.5,
    criteria: {
      currentConversation: "empath",
      dialogTurn: "empath"
    }
  }
}
```

### 4. Area-Specific Music

```javascript
areaMusic: {
  office: {
    file: "audio/music/areas/office.mp3",
    criteria: { currentZone: "office" }
  },

  plaza: {
    file: "audio/music/areas/plaza.mp3",
    criteria: { currentZone: "plaza" }
  },

  underground: {
    file: "audio/music/areas/underground.mp3",
    criteria: { currentZone: "underground" }
  }
}
```

---

## Common Mistakes Beginners Make

### 1. Hard Crossfade Cuts

```javascript
// ❌ WRONG: Abrupt transition
function switchTrack(newTrack) {
  currentAudio.stop();
  newAudio.play();
}

// ✅ CORRECT: Smooth crossfade
function switchTrack(newTrack) {
  crossfade(currentAudio, newAudio, 3000);
}
```

### 2. Wrong Audio Format

```javascript
// ❌ WRONG: WAV files are too large
file: "music/track.wav"  // 50 MB!

// ✅ CORRECT: Use compressed format
file: "music/track.mp3"  // 3 MB
```

### 3. Not Using Loop Points

```javascript
// ❌ WRONG: Loop causes audible pop
loop: true  // Jumps from end to beginning

// ✅ CORRECT: Prepare seamless loop
// Use audio editor to create loop points
loop: true,
loopStart: 5.2,  // seconds
loopEnd: 45.8
```

### 4. Ignoring Volume Balance

```javascript
// ❌ WRONG: All tracks same volume
calm: { volume: 1.0 },
intense: { volume: 1.0 }  // Too loud!

// ✅ CORRECT: Balance relative intensity
calm: { volume: 0.4 },
intense: { volume: 0.9 }
```

### 5. Forgetting to Clean Up

```javascript
// ❌ WRONG: Memory leak
playTrack(track) {
  const howl = new Howl({ src: [track.file] });
  howl.play();
  // Never disposes!
}

// ✅ CORRECT: Clean up when done
playTrack(track) {
  if (this.currentHowl) {
    this.currentHowl.unload();  // Free memory
  }
  this.currentHowl = new Howl({ src: [track.file] });
  this.currentHowl.play();
}
```

---

## Performance Considerations

### Memory Management

```javascript
// Preload frequently used tracks
const preloadTracks = ["menu", "exploration", "combat"];

// Lazy load rare tracks
const oneShotTracks = ["ending_a", "ending_b", "ending_c"];

// Unload when not needed
howl.unload();  // Frees decoded audio buffer
```

### Streaming vs Preload

```javascript
// For long tracks, stream
longTrack: {
  file: "audio/music/long-ambient.mp3",
  stream: true,  // Don't decode entirely
  preload: false
}

// For short frequently-used tracks, preload
shortStinger: {
  file: "audio/music/stinger.mp3",
  preload: true,
  stream: false
}
```

### Concurrent Track Limits

```javascript
// Only one main music track at a time
// But can layer:
// - 1 main track
// - 2-3 layer tracks
// - 1 stinger (one-shot)

// Limit: Max 5 concurrent music Howls
```

---

## 🎮 Game Design Perspective

### Music Functions in Games

1. **Atmosphere** - Establish mood and setting
2. **Pacing** - Control tension and release
3. **Feedback** - Confirm player actions
4. **Narrative** - Advance story emotionally
5. **Immersion** - Maintain player engagement

### Design Principles

```javascript
// Principle 1: Respect player attention
menuMusic: {
  file: "menu.mp3",
  volume: 0.5,
  // Lower volume for menus (player reads text)
}

// Principle 2: Match gameplay intensity
calm探索: {
  file: "calm.mp3",
  volume: 0.3,
  // Subtle during exploration
}

combat: {
  file: "combat.mp3",
  volume: 0.9,
  // Loud during action
}

// Principle 3: Seamless transitions
explorationToCombat: {
  file: "transition.mp3",
  fadeIn: 0,   // Instant start on trigger
  fadeOut: 5000,  // Long fade after
  // Don't interrupt action moment
}

// Principle 4: looping for ambience, linear for story
ambientLoop: {
  file: "ambient.mp3",
  loop: true,
  // Background atmosphere
}

storyMoment: {
  file: "story-music.mp3",
  loop: false,
  // Plays once for dramatic moment
}
```

### Emotional Progression

```javascript
// Music follows emotional arc
export const emotionalProgression = {
  // Beginning: Mystery
  mysteryTheme: {
    file: "audio/music/01-mystery.mp3",
    volume: 0.5,
    criteria: { progression: "start" }
  },

  // Middle: Tension builds
  tensionTheme: {
    file: "audio/music/02-tension.mp3",
    volume: 0.7,
    criteria: { progression: "mid" }
  },

  // Climax: Full intensity
  climaxTheme: {
    file: "audio/music/03-climax.mp3",
    volume: 1.0,
    criteria: { progression: "climax" }
  },

  // Resolution: Peaceful
  resolutionTheme: {
    file: "audio/music/04-resolution.mp3",
    volume: 0.6,
    criteria: { progression: "end" }
  }
};
```

---

## Next Steps

Now that you understand MusicManager:

- [DialogManager](./dialog-manager.md) - Spoken dialogue system
- [VideoManager](./video-manager.md) - Video playback with alpha
- [SFXManager](./sfx-manager.md) - Sound effects with spatial audio
- [Game State System](../02-core-architecture/game-state-system.md) - State-driven playback

---

## References

- [Howler.js Documentation](https://howlerjs.com/) - Official Howler.js docs (v2.2.4+)
- [Web Audio API (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) - Browser audio API
- [Audio fade techniques](https://www.html5rocks.com/en/tutorials/audio/scheduling/) - Precise audio scheduling
- [Game audio best practices](https://www.gamedeveloper.com/audio/game-audio-best-practices) - Industry standards

*Documentation last updated: January 12, 2026*
