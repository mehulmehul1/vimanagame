# Data File Formats - Shadow Engine

## Overview

The Shadow Engine uses a data-driven design where game content—scenes, dialog, animations, audio, effects, and interactive objects—is defined in separate `*Data.js` files. This separates content from code, making it easier to iterate on game design without touching engine logic, and allows non-programmers to edit game content.

Think of data files like the **"blueprints" and "recipes"**—like a restaurant separates recipes from cooking technique, the engine separates game content from rendering code, allowing designers to "write the menu" while engineers "build the kitchen."

---

## 📁 Data File Structure

```
DATA FILES ARCHITECTURE:

┌─────────────────────────────────────────────────────────┐
│                    CONTENT LAYER                         │
│  All game data defined in separate, focused files        │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  SCENEDATA    │  │  DIALOGDATA   │  │  ANIMATIONDATA│
│  - Zones      │  │  - Conversations│  │  - Camera     │
│  - Spawns     │  │  - Branching   │  │  - Objects    │
│  - Triggers   │  │  - Characters  │  │  - Keyframes   │
└───────────────┘  └───────────────┘  └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    ┌───────────────┐
                    │  AUDIODATA    │
                    │  - Music       │
                    │  - SFX         │
                    └───────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   VFXDATA     │  │ INTERACTIVE   │  │  COMBINED     │
│  - Effects     │  │ OBJECTDATA    │  │  GAME STATE   │
│  - Timelines   │  │  - Props      │  │  - Runtime     │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## SceneData.js

Defines all game zones, their contents, spawn points, and triggers.

### File Structure

```javascript
// SceneData.js
export const SceneData = {
  // Zone definitions
  zones: {
    plaza: {
      id: 'plaza',
      name: 'Central Plaza',
      position: { x: 0, y: 0, z: 0 },
      bounds: { min: { x: -50, y: 0, z: -50 }, max: { x: 50, y: 20, z: 50 } },
      splats: '/assets/splats/plaza.splat',
      models: {
        environment: '/assets/models/plaza.glb',
        props: ['/assets/models/bench.glb', '/assets/models/fountain.glb']
      },
      lighting: {
        ambient: { color: 0x404060, intensity: 0.5 },
        directional: {
          color: 0xffeedd,
          intensity: 1.0,
          position: { x: 10, y: 20, z: 10 },
          target: { x: 0, y: 0, z: 0 }
        }
      },
      spawns: [
        {
          id: 'player_start',
          position: { x: 0, y: 1, z: 10 },
          rotation: { x: 0, y: 0, z: 0 },
          camera: 'third_person'
        }
      ],
      triggers: [
        {
          id: 'plaza_to_alley',
          type: 'zone',
          bounds: { min: { x: -5, y: 0, z: 45 }, max: { x: 5, y: 5, z: 55 } },
          onEnter: 'load_zone',
          targetZone: 'alley',
          transition: 'fade'
        }
      ]
    },

    alley: {
      id: 'alley',
      name: 'Dark Alley',
      // ... similar structure
    }
  },

  // Connections between zones
  connections: [
    { from: 'plaza', to: 'alley', bidirectional: true },
    { from: 'plaza', to: 'street', bidirectional: true }
  ]
};
```

### Zone Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique zone identifier |
| `name` | string | Display name |
| `position` | Vector3 | Center position |
| `bounds` | Box3 | Zone boundaries |
| `splats` | string | Gaussian splat file |
| `models` | object | 3D model files |
| `lighting` | object | Light configuration |
| `spawns` | array | Player spawn points |
| `triggers` | array | Zone triggers |

### Creating a New Zone

```javascript
// Add to SceneData.js
zones: {
  my_custom_zone: {
    id: 'my_custom_zone',
    name: 'My Custom Area',
    position: { x: 100, y: 0, z: 100 },
    bounds: { min: { x: 50, y: 0, z: 50 }, max: { x: 150, y: 20, z: 150 } },
    splats: '/assets/splats/my_zone.splat',
    models: {
      environment: '/assets/models/my_zone.glb'
    },
    spawns: [
      {
        id: 'spawn_point',
        position: { x: 100, y: 1, z: 110 },
        rotation: { x: 0, y: Math.PI, z: 0 }
      }
    ]
  }
}
```

---

## DialogData.js

Defines all conversations, character lines, and player choices.

### File Structure

```javascript
// DialogData.js
export const DialogData = {
  conversations: {
    intro_conversation: {
      id: 'intro_conversation',
      nodes: {
        start: {
          speaker: 'narrator',
          text: 'You awaken in a strange place...',
          expression: 'neutral',
          voiceLine: 'audio/narrator/intro.mp3',
          nextNode: 'question',
          choices: null
        },

        question: {
          speaker: 'guide',
          text: 'Welcome, traveler. Are you ready to begin?',
          expression: 'friendly',
          choices: [
            {
              text: 'Yes, I\'m ready.',
              nextNode: 'accept_response',
              condition: null,  // Always shown
              consequences: ['eagerness:1']
            },
            {
              text: 'Where am I?',
              nextNode: 'confused_response',
              condition: null,
              consequences: ['confusion:1']
            },
            {
              text: '*Stay silent*',
              nextNode: 'silent_response',
              condition: null,
              consequences: ['caution:1']
            }
          ]
        },

        accept_response: {
          speaker: 'guide',
          text: 'Excellent. Let us begin your journey.',
          nextNode: null,  // End dialog
          choices: null,
          action: 'enable_movement'
        }
      }
    },

    phone_conversation: {
      // Phone booth dialog
      nodes: {
        ring_start: {
          speaker: 'phone',
          text: '*ringing*',
          nextNode: 'ring_continue',
          choices: null,
          action: 'play_ring_sound'
        },

        ring_continue: {
          speaker: 'unknown_voice',
          text: 'You\'ve been expected. Answer.',
          choices: [
            {
              text: 'Answer the phone',
              nextNode: 'phone_answered',
              condition: '!phone_answered'
            },
            {
              text: 'Walk away',
              nextNode: 'phone_ignored',
              condition: null
            }
          ]
        },

        phone_answered: {
          speaker: 'unknown_voice',
          text: 'I see you\'re finally awake. The shadows have been waiting.',
          choices: [
            {
              text: 'Who are you?',
              nextNode: 'reveal_identity',
              condition: 'curiosity>2'
            },
            {
              text: 'What do you want?',
              nextNode: 'explain_mission',
              condition: null
            }
          ]
        }
      }
    }
  },

  // Speaker definitions
  speakers: {
    narrator: {
      name: 'Narrator',
      color: '#888888',
      portrait: null,
      voice: 'narrator'
    },
    guide: {
      name: 'Mysterious Guide',
      color: '#00ff88',
      portrait: '/assets/portraits/guide.png',
      voice: 'guide'
    },
    unknown_voice: {
      name: '???',
      color: '#ff0066',
      portrait: null,
      voice: 'unknown'
    }
  }
};
```

### Dialog Properties

| Property | Type | Description |
|----------|------|-------------|
| `speaker` | string | Speaker ID (maps to speakers) |
| `text` | string | Dialog text |
| `expression` | string | Facial expression/state |
| `voiceLine` | string | Audio file path |
| `nextNode` | string | Next node ID (null = end) |
| `choices` | array | Available choices |
| `action` | string | Game action to trigger |

### Conditional Choices

```javascript
choices: [
  {
    text: 'Aggressive response',
    nextNode: 'aggressive_path',
    // Show only if aggression flag is set
    condition: 'aggression > 2'
  },
  {
    text: 'Diplomatic response',
    nextNode: 'diplomatic_path',
    // Show only NOT in combat mode
    condition: '!in_combat'
  }
]
```

### Creating New Dialog

```javascript
// Add to DialogData.js
my_conversation: {
  id: 'my_conversation',
  nodes: {
    start: {
      speaker: 'npc',
      text: 'Hello there, traveler!',
      choices: [
        {
          text: 'Greetings!',
          nextNode: 'friendly_response'
        },
        {
          text: 'Leave',
          nextNode: null  // End dialog
        }
      ]
    },

    friendly_response: {
      speaker: 'npc',
      text: 'A friendly soul! How delightful.',
      nextNode: null,
      consequences: ['friendship:+1']
    }
  }
}
```

---

## AnimationData.js

Defines camera movements, object animations, and cinematics.

### File Structure

```javascript
// AnimationData.js
export const AnimationData = {
  camera: {
    intro_sequence: {
      id: 'intro_sequence',
      type: 'camera',
      duration: 5.0,
      easing: 'easeInOutCubic',
      keyframes: [
        {
          time: 0.0,
          position: { x: 0, y: 20, z: 50 },
          target: { x: 0, y: 0, z: 0 },
          fov: 60
        },
        {
          time: 2.5,
          position: { x: 25, y: 15, z: 35 },
          target: { x: 0, y: 0, z: 0 },
          fov: 75
        },
        {
          time: 5.0,
          position: { x: 0, y: 2, z: 5 },
          target: { x: 0, y: 1.6, z: 0 },
          fov: 75
        }
      ]
    },

    dramatic_zoom: {
      id: 'dramatic_zoom',
      type: 'camera',
      duration: 2.0,
      easing: 'easeInOutQuad',
      keyframes: [
        {
          time: 0.0,
          position: { x: 0, y: 5, z: 20 },
          target: { x: 0, y: 0, z: 0 },
          fov: 60
        },
        {
          time: 2.0,
          position: { x: 0, y: 5, z: 10 },
          target: { x: 0, y: 0, z: 0 },
          fov: 30  // Dramatic zoom
        }
      ]
    }
  },

  objects: {
    door_open: {
      id: 'door_open',
      type: 'object',
      target: 'door_01',
      duration: 1.5,
      easing: 'easeOutQuad',
      keyframes: [
        {
          time: 0.0,
          position: { x: 0, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0 }
        },
        {
          time: 1.5,
          position: { x: 0, y: 0, z: 0 },
          rotation: { x: 0, y: Math.PI * 0.5, z: 0 }  // Swing open
        }
      ]
    },

    character_wave: {
      id: 'character_wave',
      type: 'object',
      target: 'npc_guide',
      duration: 2.0,
      easing: 'easeInOutSine',
      keyframes: [
        {
          time: 0.0,
          position: { x: 0, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0 }
        },
        {
          time: 0.5,
          position: { x: 0, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0 },
          action: 'arm_raise'  // Custom action
        },
        {
          time: 1.0,
          position: { x: 0, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0 },
          action: 'arm_wave'
        },
        {
          time: 2.0,
          position: { x: 0, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0 },
          action: 'arm_lower'
        }
      ]
    }
  }
};
```

### Animation Properties

| Property | Type | Description |
|----------|------|-------------|
| `type` | string | Animation type (camera, object) |
| `target` | string | Target object ID |
| `duration` | number | Animation length in seconds |
| `easing` | string | Easing function |
| `keyframes` | array | Keyframe definitions |

### Easing Functions

```javascript
// Available easing functions
const easing = {
  linear: 't => t',
  easeInQuad: 't => t * t',
  easeOutQuad: 't => t * (2 - t)',
  easeInOutQuad: 't => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t',
  easeInOutCubic: 't => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1',
  easeInOutSine: 't => -(Math.cos(Math.PI * t) - 1) / 2'
};
```

---

## AudioData.js

Defines music tracks, sound effects, and audio-reactive elements.

### File Structure

```javascript
// AudioData.js
export const AudioData = {
  music: {
    main_theme: {
      id: 'main_theme',
      path: '/assets/audio/music/main_theme.mp3',
      volume: 0.7,
      loop: true,
      fadeIn: 2.0,
      fadeOut: 2.0,
      tags: ['ambient', 'exploration'],
      layer: {
        base: '/assets/audio/music/main_theme_base.mp3',
        drums: '/assets/audio/music/main_theme_drums.mp3',
        melody: '/assets/audio/music/main_theme_melody.mp3'
      }
    },

    combat_music: {
      id: 'combat_music',
      path: '/assets/audio/music/combat.mp3',
      volume: 0.9,
      loop: true,
      tags: ['combat', 'intense']
    },

    club_scene: {
      id: 'club_scene',
      path: '/assets/audio/music/club.mp3',
      volume: 0.8,
      loop: true,
      bpm: 128,  // For audio-reactive effects
      tags: ['club', 'final'],
      reactive: {
        bass: { frequency: [60, 80], bands: [60, 120] },
        kick: { frequency: [0, 5], threshold: 0.8 }
      }
    }
  },

  sfx: {
    // UI Sounds
    ui_click: {
      id: 'ui_click',
      path: '/assets/audio/sfx/click.wav',
      volume: 0.3
    },

    ui_hover: {
      id: 'ui_hover',
      path: '/assets/audio/sfx/hover.wav',
      volume: 0.2
    },

    ui_confirm: {
      id: 'ui_confirm',
      path: '/assets/audio/sfx/confirm.wav',
      volume: 0.4
    },

    // Gameplay Sounds
    footstep: {
      id: 'footstep',
      path: '/assets/audio/sfx/footstep_grass.mp3',
      volume: 0.5,
      variations: [
        '/assets/audio/sfx/footstep_grass_01.mp3',
        '/assets/audio/sfx/footstep_grass_02.mp3',
        '/assets/audio/sfx/footstep_grass_03.mp3'
      ]
    },

    door_open: {
      id: 'door_open',
      path: '/assets/audio/sfx/door_open.wav',
      volume: 0.6,
      spatial: true,
      maxDistance: 10
    },

    phone_ring: {
      id: 'phone_ring',
      path: '/assets/audio/sfx/phone_ring.wav',
      volume: 0.8,
      loop: true,
      spatial: true,
      maxDistance: 20,
      attenuation: 0.5
    },

    // Impact Sounds
    impact_soft: {
      id: 'impact_soft',
      path: '/assets/audio/sfx/impact_soft.wav',
      volume: 0.5
    },

    impact_hard: {
      id: 'impact_hard',
      path: '/assets/audio/sfx/impact_hard.wav',
      volume: 1.0
    }
  },

  voice: {
    narrator_intro: {
      id: 'narrator_intro',
      path: '/assets/audio/voice/narrator_intro.mp3',
      volume: 1.0,
      subtitle: 'You awaken in a strange place...',
      speaker: 'narrator'
    },

    guide_greeting: {
      id: 'guide_greeting',
      path: '/assets/audio/voice/guide_greeting.mp3',
      volume: 1.0,
      subtitle: 'Welcome, traveler.',
      speaker: 'guide'
    }
  }
};
```

---

## VFXData.js

Defines visual effects, their properties, and trigger conditions.

### File Structure

```javascript
// VFXData.js
export const VFXData = {
  effects: {
    dissolve_to_black: {
      id: 'dissolve_to_black',
      type: 'dissolve',
      duration: 2.0,
      color: 0x000000,
      particleCount: 1000,
      particleSpread: 5,
      particleColor: 0x00ff88
    },

    glitch_severe: {
      id: 'glitch_severe',
      type: 'glitch',
      duration: 3.0,
      intensity: 1.0,
      chromatic: true,
      scanlines: true,
      noiseIntensity: 0.5
    },

    bloom_surge: {
      id: 'bloom_surge',
      type: 'bloom',
      duration: 0.5,
      threshold: 0.3,
      radius: 1.0,
      strength: 2.0,
      color: 0x00ff88
    },

    splat_morph: {
      id: 'splat_morph',
      type: 'splat_transform',
      duration: 5.0,
      from: 'normal',
      to: 'nightmare',
      parameters: {
        scale: 2.0,
        colorShift: { hue: 180, saturation: -50 }
      }
    }
  },

  timelines: {
    office_hell_transition: {
      id: 'office_hell_transition',
      effects: [
        {
          time: 0.0,
          effect: 'glitch_start',
          duration: 1.0
        },
        {
          time: 1.0,
          effect: 'splat_morph',
          duration: 3.0
        },
        {
          time: 4.0,
          effect: 'bloom_surge',
          duration: 0.5
        },
        {
          time: 4.5,
          effect: 'glitch_end',
          duration: 0.5
        }
      ]
    }
  }
};
```

---

## InteractiveObjectData.js

Defines all interactive objects and their behaviors.

### File Structure

```javascript
// InteractiveObjectData.js
export const InteractiveObjectData = {
  phone_booth: {
    id: 'phone_booth',
    type: 'interactive',
    model: '/assets/models/phonebooth.glb',
    position: { x: -15, y: 0, z: 30 },
    rotation: { x: 0, y: -Math.PI / 6, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
    collider: {
      type: 'box',
      bounds: { x: 2, y: 3, z: 2 },
      trigger: true
    },
    interactions: [
      {
        trigger: 'onEnter',
        action: 'playRing',
        sound: 'phone_ring',
        loop: true
      },
      {
        trigger: 'onClick',
        condition: '!phone_answered',
        action: 'startDialog',
        dialog: 'phone_conversation'
      },
      {
        trigger: 'onClick',
        condition: 'phone_answered',
        action: 'playSound',
        sound: 'dial_tone'
      }
    ]
  },

  radio: {
    id: 'radio',
    type: 'interactive',
    model: '/assets/models/radio.glb',
    position: { x: 20, y: 1, z: 10 },
    interactions: [
      {
        trigger: 'onProximity',
        radius: 5,
        action: 'playAudio',
        audio: 'radio_broadcast',
        volume: 0.5
      }
    ]
  },

  viewmaster: {
    id: 'viewmaster',
    type: 'interactive',
    model: '/assets/models/viewmaster.glb',
    interactions: [
      {
        trigger: 'onPickup',
        action: 'showView',
        view: 'viewmaster_nightmare'
      },
      {
        trigger: 'onUse',
        action: 'triggerVFX',
        effect: 'viewmaster_glitch'
      }
    ]
  },

  amplifier_cord: {
    id: 'amplifier_cord',
    type: 'physics_puzzle',
    model: '/assets/models/cord.glb',
    position: { x: 5, y: 1, z: -10 },
    physics: {
      type: 'rope',
      start: { x: 5, y: 1.5, z: -10 },
      end: { x: 15, y: 1.5, z: -10 },
      segments: 20
    },
    interactions: [
      {
        trigger: 'onDrag',
        action: 'checkConnection',
        target: 'amplifier',
        success: 'powerOn',
        failure: 'playSound',
        sound: 'cord_snap'
      }
    ]
  }
};
```

---

## Data File Best Practices

### 1. Keep Files Focused

```javascript
// ✅ GOOD: Each file has clear purpose
SceneData.js    // Spatial data
DialogData.js   // Conversations
AudioData.js    // Sounds

// ❌ BAD: Mixed purposes
GameData.js  // Contains everything
```

### 2. Use Consistent Naming

```javascript
// IDs use snake_case
my_custom_zone
phone_booth
intro_conversation

// Keys use camelCase in objects
position: { x: 0, y: 0, z: 0 }
```

### 3. Add Comments for Clarity

```javascript
zones: {
  plaza: {
    id: 'plaza',
    name: 'Central Plaza',
    // Main player spawn area
    // Safe zone, no enemies
    spawns: [ /* ... */ ]
  }
}
```

### 4. Organize for Readability

```javascript
// Group related items
const data = {
  // === ZONES ===
  zones: { /* ... */ },

  // === CONNECTIONS ===
  connections: [ /* ... */ ],

  // === SPAWNS ===
  spawns: [ /* ... */ ]
};
```

---

## Loading Data Files

The engine loads and merges all data files:

```javascript
// In GameManager or DataManager
import { SceneData } from '../data/SceneData.js';
import { DialogData } from '../data/DialogData.js';
import { AnimationData } from '../data/AnimationData.js';
import { AudioData } from '../data/AudioData.js';
import { VFXData } from '../data/VFXData.js';
import { InteractiveObjectData } from '../data/InteractiveObjectData.js';

// Merge into single game state
const GameData = {
  scenes: SceneData.zones,
  dialog: DialogData.conversations,
  animations: AnimationData,
  audio: AudioData,
  vfx: VFXData.effects,
  interactive: InteractiveObjectData
};
```

---

## Related Systems

- [API Reference](./api-reference.md) - How to use data files
- [Scene-by-Scene Case Studies](../14-scene-case-studies/) - How data creates scenes
- [Game State System](../02-core-architecture/game-state-system.md) - State management

---

## Source File Reference

**Data Files**:
- `../src/data/SceneData.js` - Scene/zone definitions
- `../src/data/DialogData.js` - Dialog trees
- `../src/data/AnimationData.js` - Animations
- `../src/data/AudioData.js` - Music and SFX
- `../src/data/VFXData.js` - Visual effects
- `../src/data/InteractiveObjectData.js` - Interactive props

---

## References

- [JSON Format](https://www.json.org/) - Data format
- [JavaScript Modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules) - Import/export

*Documentation last updated: January 12, 2026*
