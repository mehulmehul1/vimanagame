# Scene Case Study: Alley Sections

## 🎬 Scene Overview

**Location**: Multiple alley sections connecting exterior zones
**Narrative Context**: Transitional spaces that serve both as navigation corridors and atmospheric storytelling opportunities
**Player Experience**: Tension building, environmental curiosity, passage between known and unknown

The Alley Sections represent some of the most effective environmental storytelling in the game. These aren't just corridors to move through—they're carefully crafted transitional spaces that build atmosphere through visual progression, sound design, and environmental detail. Each alley has its own character while maintaining visual cohesion with the overall exterior environment.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create a sense of journey—alleys aren't just connections, they're experiences.

**Why Alleys Matter**:

```
TRANSITIONAL SPACE PSYCHOLOGY:

Safe Zone → [ALLEY] → New Discovery
   ↑             ↓            ↑
Comfort      Tension      Curiosity

ALLEYS AS:
├── Narrative Bridges: Connect plot beats
├── Pacing Tools: Control exploration speed
├── Atmosphere Builders: Establish mood before destination
├── Environmental Storytelling: Show, don't tell
└── Reward Corridors: Journey makes arrival meaningful
```

### Design Philosophy for Alleys

**1. Alleys Have Personality**

Each alley section has a distinct character:
- **Intro Alley**: Short, safe, introduces exploration
- **Long View Alley**: Extended sight lines to distant points of interest
- **Navigable Alley**: Full exploration space with details to discover
- **Dark Alley**: Tension-building, hints at danger

**2. Progressive Disclosure**

```
ALLEY REVEAL STRUCTURE:

Entry:
├── See alley entrance
├── Can't see end (mystery)
└── Audio hints at what's ahead

Progress:
├── Environmental details tell story
├── Pacing through obstacles/turns
└── Build anticipation

Destination:
├── Reach the interactive object/location
├── Payoff for exploration
└── New narrative beat
```

**3. Environmental Storytelling**

```
STORYTELLING THROUGH DETAIL:

Visual Elements:
├── Debris/graffiti → History of this place
├── Lighting changes → Time of day/mood
├── Physical obstructions → Past events
└── Architectural variety → Different eras/uses

Audio Elements:
├── Echoes → Space size perception
├── Distant sounds → What's ahead
├── Close sounds → Immediate surroundings
└── Silence → Tension building
```

---

## 🎨 Level Design Breakdown

### Alley Types and Their Purposes

```
                    ALLEY TYPOLOGY:

┌─────────────────────────────────────────────────────────┐
│                    INTRO ALLEY                          │
│  Purpose: Tutorial, safe exploration                    │
│  Length: Short (10-15m)                                 │
│  Width: Wide (3-4m)                                     │
│  Lighting: Bright, welcoming                            │
│  Obstacles: None                                        │
│  Destination: Phone booth (clearly visible)             │
│  Player Feeling: "This is how I explore"                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   LONG VIEW ALLEY                       │
│  Purpose: Build anticipation to destination             │
│  Length: Long (30-40m)                                  │
│  Width: Medium (2-3m)                                   │
│  Lighting: Dimming toward destination                    │
│  Obstacles: Minor (debris to navigate)                  │
│  Destination: Visible at end (tease)                    │
│  Player Feeling: "What's that way down there?"          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  NAVIGABLE ALLEY                        │
│  Purpose: Full exploration, discovery space             │
│  Length: Medium (20-25m)                                │
│  Width: Variable (2-5m)                                 │
│  Lighting: Pool/spot lighting                           │
│  Obstacles: Interactive (objects to examine)            │
│  Destination: Multiple possible                         │
│  Player Feeling: "I should look around in here"         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    DARK ALLEY                           │
│  Purpose: Tension, atmosphere, danger hinting           │
│  Length: Any                                            │
│  Width: Narrow (1.5-2m)                                 │
│  Lighting: Very dim, flickering                         │
│  Obstacles: Major (blockages requiring navigation)      │
│  Destination: Unknown until close                       │
│  Player Feeling: "I should be careful"                  │
└─────────────────────────────────────────────────────────┘
```

### Spatial Layout - Intro Alley (Phone Booth Path)

```
                    INTRO ALLEY LAYOUT:

   [INTERSECTION] → [ALLEY ENTRY] → [PHONE BOOTH] → [ALLEY END]

                    ╔════════════════════════════╗
                    ║     INTERSECTION HUB       ║
                    ║            ↓               ║
                    ║    ═══════╦══════         ║
                    ║           ║               ║
    ═════════════════╩═══════════╩═════════════════════
    │                                                │
    │              INTRO ALLEY                       │
    │  ┌──────────────────────────────────────┐     │
    │  │  Width: 3.5m (comfortable)           │     │
    │  │  Length: 12m (quick transit)        │     │
    │  │                                      │     │
    │  │  [ENTRY]                    [PHONE]  │     │
    │  │    ↓                            ↓     │     │
    │  │  Bright light          Ringing audio  │     │
    │  │  (from intersection)  getting louder │     │
    │  │                                      │     │
    │  │  Details:                             │     │
    │  │  - Street lights along one side      │     │
    │  │  - Chain link fence (left)           │     │
    │  │  - Brick wall (right)                │     │
    │  │  - Scattered debris                  │     │
    │  │  - Phone booth visible at end        │     │
    │  └──────────────────────────────────────┘     │
    │                                                │
    │              [ALLEY CONTINUES]                 │
    ════════════════════════════════════════════════

DESIGN NOTES:
- Phone booth visible from entry (clear destination)
- No turns (straight shot = easy navigation)
- Audio gradient (ringing gets louder as you approach)
- "Safe" alley (player's first exploration)
```

### Spatial Layout - Long View Alley

```
                   LONG VIEW ALLEY LAYOUT:

   [INTERSECTION] → [ALLEY ENTRY] → [MID SECTION] → [RADIO/DESTINATION]

                    ╔════════════════════════════╗
                    ║     INTERSECTION HUB       ║
                    ║            ↓               ║
                    ║    ═══════╦══════         ║
                    ║           ║               ║
    ═════════════════╩═══════════╩═════════════════════
    │                                                │
    │              LONG VIEW ALLEY                   │
    │  ┌─────────────────────────────────────────┐  │
    │  │  Width: 2.5m (narrower = more intimate)│  │
    │  │  Length: 35m (builds anticipation)     │  │
    │  │                                        │  │
    │  │  [ENTRY]      [TURN]   [TEASE]  [END] │  │
    │  │    ↓           ↓         ↓        ↓    │  │
    │  │  Bright      Shadow   Radio    Close │  │
    │  │  light      zone     glow     view   │  │
    │  │                                        │  │
    │  │  SECTIONS:                             │  │
    │  │  1. Entry zone (0-10m):                │  │
    │  │     - From intersection light          │  │
    │  │     - Street lamp                      │  │
    │  │     - Sense of leaving safety          │  │
    │  │                                        │  │
    │  │  2. Mid section (10-20m):              │  │
    │  │     - Slight turn (breaks sight line)  │  │
    │  │     - Overhead structure (darker)      │  │
    │  │     - Audio begins (faint radio)       │  │
    │  │                                        │  │
    │  │  3. Tease zone (20-30m):               │  │
    │  │     - Radio becomes visible (glow)     │  │
    │  │     - Audio clearly audible            │  │
    │  │     - Debris to navigate around        │  │
    │  │                                        │  │
    │  │  4. End zone (30-35m):                 │  │
    │  │     - Radio clearly visible            │  │
    │  │     - Interaction trigger              │  │
    │  │     - Dead end (or continue option)    │  │
    │  └─────────────────────────────────────────┘  │
    │                                                │
    ════════════════════════════════════════════════

DESIGN NOTES:
- Turn at midpoint breaks long straight line (adds interest)
- Progressive audio reveal (faint → clear)
- Visual tease (radio visible before reachable)
- Longer path = greater sense of discovery
```

### Progressive Atmosphere

```
ALLEY ATMOSPHERE PROGRESSION:

INTRO ALLEY (Safe Exploration):
├── Lighting: Bright, uniform
├── Audio: Intersection ambience fading, phone ringing rising
├── Width: Wide (no claustrophobia)
├── Details: Sparse (focus on destination)
└── Mood: Welcoming, tutorial-like

LONG VIEW ALLEY (Building Anticipation):
├── Lighting: Gradient (bright → dim → destination glow)
├── Audio: Layered (ambience → destination sound)
├── Width: Medium (slight intimacy)
├── Details: Increasing density toward destination
└── Mood: Journey, anticipation

NAVIGABLE ALLEY (Discovery Space):
├── Lighting: Pool lighting (bright spots, dark between)
├── Audio: Interactive sounds (objects can be examined)
├── Width: Variable (creates rhythm)
├── Details: Dense (worth exploring)
└── Mood: Curiosity, investigation

DARK ALLEY (Tension Building):
├── Lighting: Dim, flickering
├── Audio: Near-silence with distant ominous sounds
├── Width: Narrow (claustrophobic)
├── Details: Threatening or ominous
└── Mood: Caution, danger
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the alley implementation, you should know:
- **Splat Regions**: Portion of a larger splat file to render
- **Audio Attenuation Curves**: Volume falloff over distance
- **Trigger Chains**: Multiple triggers creating sequential events
- **Occlusion Culling**: Not rendering what's behind walls
- **Path Triggers**: Invisible volumes for zone detection

### Scene Data Structure

```javascript
// SceneData.js - Alley zone configurations
export const SCENES = {
  // Intro Alley (Phone Booth path)
  alley_intro_phone: {
    id: 'alley_intro_phone',
    name: 'Alley to Phone Booth',
    type: 'alley',
    subtype: 'intro',

    // Shared exterior splat, region defined
    splat: {
      file: '/assets/splats/exterior.ply',
      region: {
        minX: 10, maxX: 25,
        minY: 0, maxY: 8,
        minZ: -5, maxZ: 5
      }
    },

    // Alley characteristics
    alley: {
      length: 12,
      width: 3.5,
      hasTurn: false,
      destinationVisible: true,
      destinationId: 'phone_booth'
    },

    // Entry point from intersection
    entryPoint: {
      position: { x: 10, y: 1.7, z: 0 },
      rotation: { x: 0, y: 0, z: 0 }  // Facing east
    },

    // Exit to phone booth zone
    exitPoint: {
      position: { x: 25, y: 1.7, z: 0 },
      toZone: 'phone_booth'
    },

    // Audio gradient
    audio: {
      entrance: {
        sound: 'intersection_ambience',
        volume: 0.3,
        fadeOut: true
      },
      destination: {
        sound: 'ringing_phone',
        volumeAtEntry: 0.2,
        volumeAtExit: 0.6,
        position: { x: 27, y: 1, z: 0 }
      },
      ambience: 'alley_ambience'
    },

    // Lighting progression
    lighting: {
      type: 'gradient',
      startIntensity: 0.8,
      endIntensity: 0.5,
      color: 0xffeecc  // Warm street lights
    }
  },

  // Long View Alley (Radio path)
  alley_long_view_radio: {
    id: 'alley_long_view_radio',
    name: 'Alley to Radio',
    type: 'alley',
    subtype: 'long_view',

    splat: {
      file: '/assets/splats/exterior.ply',
      region: {
        minX: -25, maxX: -5,
        minY: 0, maxY: 8,
        minZ: -8, maxZ: 8
      }
    },

    alley: {
      length: 35,
      width: 2.5,
      hasTurn: true,
      turnPosition: { x: -15, y: 1, z: 0 },
      turnAngle: Math.PI / 8,  // Slight bend
      destinationVisible: false,  // Until turn
      destinationId: 'radio'
    },

    entryPoint: {
      position: { x: -5, y: 1.7, z: 0 },
      rotation: { x: 0, y: Math.PI, z: 0 }  // Facing west
    },

    exitPoint: {
      position: { x: -25, y: 1.7, z: 0 },
      toZone: 'radio'
    },

    // Sequential audio reveal
    audio: {
      zones: [
        {
          zRange: [0, 10],  // Entry section
          ambience: 'intersection_ambience',
          destinationVolume: 0.1
        },
        {
          zRange: [10, 20],  // Mid section
          ambience: 'alley_mid_ambience',
          destinationVolume: 0.3,
          trigger: 'mid_point_reveal'
        },
        {
          zRange: [20, 35],  // Destination section
          ambience: 'radio_proximity_ambience',
          destinationVolume: 0.6
        }
      ]
    },

    // Sectional lighting
    lighting: {
      type: 'sectional',
      sections: [
        { range: [0, 10], intensity: 0.7 },
        { range: [10, 20], intensity: 0.4, flicker: false },
        { range: [20, 35], intensity: 0.3, source: 'radio_glow' }
      ]
    }
  }
};
```

### Alley Manager Implementation

```javascript
// AlleyManager.js - Handles alley-specific logic
class AlleyManager {
  constructor(sceneManager, audioManager) {
    this.sceneManager = sceneManager;
    this.audioManager = audioManager;

    this.currentAlley = null;
    this.playerProgress = 0;  // 0-1 along alley
    this.triggersPassed = new Set();
  }

  enterAlley(alleyId) {
    const alley = SCENES[alleyId];
    this.currentAlley = alley;
    this.playerProgress = 0;
    this.triggersPassed.clear();

    // Set up audio gradient
    this.setupAudioGradient(alley);

    // Set up lighting progression
    this.setupLightingProgression(alley);

    // Register progress triggers
    this.setupProgressTriggers(alley);
  }

  setupAudioGradient(alley) {
    if (alley.audio.destination) {
      // Crossfade between entrance and destination audio
      this.audioManager.setupCrossFade(
        alley.audio.entrance.sound,
        alley.audio.destination.sound,
        alley.audio.destination.position
      );
    }

    // For long alleys with zones
    if (alley.audio.zones) {
      for (const zone of alley.audio.zones) {
        this.audioManager.registerAudioZone(zone);
      }
    }
  }

  setupLightingProgression(alley) {
    if (alley.lighting.type === 'gradient') {
      // Smooth interpolation from start to end
      this.sceneManager.setupLightingGradient(
        alley.lighting.startIntensity,
        alley.lighting.endIntensity,
        alley.length
      );
    } else if (alley.lighting.type === 'sectional') {
      // Step changes at specific points
      for (const section of alley.lighting.sections) {
        this.sceneManager.setupLightingSection(section);
      }
    }
  }

  setupProgressTriggers(alley) {
    // Triggers at various points along alley
    const triggerPoints = [
      { progress: 0.25, event: 'alley_quarter' },
      { progress: 0.5, event: 'alley_halfway' },
      { progress: 0.75, event: 'alley_three_quarters' }
    ];

    for (const trigger of triggerPoints) {
      this.sceneManager.registerProgressTrigger({
        position: this.getAlleyPositionAt(alley, trigger.progress),
        radius: 2,
        event: trigger.event,
        once: true,
        callback: () => this.onProgressTrigger(trigger.event)
      });
    }
  }

  update(playerPosition) {
    if (!this.currentAlley) return;

    // Calculate player progress along alley
    this.playerProgress = this.calculateProgress(
      playerPosition,
      this.currentAlley
    );

    // Update audio mix based on progress
    this.updateAudioMix(this.playerProgress);

    // Update lighting based on progress
    this.updateLighting(this.playerProgress);
  }

  calculateProgress(playerPos, alley) {
    const entry = alley.entryPoint.position;
    const exit = alley.exitPoint.position;

    const alleyVector = {
      x: exit.x - entry.x,
      z: exit.z - entry.z
    };
    const alleyLength = Math.sqrt(
      alleyVector.x * alleyVector.x +
      alleyVector.z * alleyVector.z
    );

    const playerVector = {
      x: playerPos.x - entry.x,
      z: playerPos.z - entry.z
    };

    // Project player position onto alley direction
    const progress = (
      playerVector.x * alleyVector.x +
      playerVector.z * alleyVector.z
    ) / (alleyLength * alleyLength);

    return Math.max(0, Math.min(1, progress));
  }

  updateAudioMix(progress) {
    const alley = this.currentAlley;

    if (alley.audio.destination) {
      // Interpolate between entrance and destination volumes
      const entranceVol = alley.audio.entrance.volume * (1 - progress);
      const destVol = this.lerp(
        alley.audio.destination.volumeAtEntry,
        alley.audio.destination.volumeAtExit,
        progress
      );

      this.audioManager.setCrossFadeMix(entranceVol, destVol);
    }
  }

  updateLighting(progress) {
    const alley = this.currentAlley;

    if (alley.lighting.type === 'gradient') {
      const intensity = this.lerp(
        alley.lighting.startIntensity,
        alley.lighting.endIntensity,
        progress
      );
      this.sceneManager.updateAlleyLighting(intensity);
    }
  }

  getAlleyPositionAt(alley, progress) {
    const entry = alley.entryPoint.position;
    const exit = alley.exitPoint.position;

    return {
      x: this.lerp(entry.x, exit.x, progress),
      y: entry.y,
      z: this.lerp(entry.z, exit.z, progress)
    };
  }

  lerp(a, b, t) {
    return a + (b - a) * t;
  }
}
```

### Environmental Detail Placement

```javascript
// AlleyDetailManager.js - Props and environmental storytelling
class AlleyDetailManager {
  constructor(sceneManager) {
    this.sceneManager = sceneManager;
  }

  populateAlley(alleyId) {
    const alley = SCENES[alleyId];

    // Add details based on alley type
    switch (alley.subtype) {
      case 'intro':
        this.addIntroDetails(alley);
        break;
      case 'long_view':
        this.addLongViewDetails(alley);
        break;
      case 'navigable':
        this.addNavigableDetails(alley);
        break;
      case 'dark':
        this.addDarkDetails(alley);
        break;
    }
  }

  addIntroDetails(alley) {
    // Sparse details for intro alley
    const details = [
      {
        type: 'street_light',
        position: { x: 15, y: 4, z: -2 },
        purpose: 'Lighting + visual interest'
      },
      {
        type: 'debris_small',
        position: { x: 18, y: 0, z: 1 },
        purpose: 'Slight texture to ground'
      },
      {
        type: 'graffiti',
        position: { x: 20, y: 2, z: -3.5 },
        surface: 'wall',
        purpose: 'Environmental storytelling'
      }
    ];

    for (const detail of details) {
      this.sceneManager.addDetail(detail);
    }
  }

  addLongViewDetails(alley) {
    // Progressive detail density
    const details = [
      // Entry section (sparse)
      { type: 'street_light', position: { x: -8, y: 4, z: 2 } },
      { type: 'trash_bag', position: { x: -10, y: 0, z: -1 } },

      // Mid section (increasing)
      { type: 'overhead_structure', position: { x: -15, y: 3, z: 0 } },
      { type: 'debris_pile', position: { x: -17, y: 0, z: 1.5 } },
      { type: 'graffiti_set', position: { x: -18, y: 1.5, z: -2.5 } },

      // Tease section (dense)
      { type: 'discarded_item', position: { x: -22, y: 0.5, z: 0 } },
      { type: 'broken_light', position: { x: -23, y: 3, z: 2 }, flicker: true },
      { type: 'footprints', position: { x: -24, y: 0, z: 0 }, leadTo: 'radio' }
    ];

    for (const detail of details) {
      this.sceneManager.addDetail(detail);
    }
  }

  addNavigableDetails(alley) {
    // Interactive details worth examining
    const details = [
      {
        type: 'examinable_object',
        position: { x: 0, y: 1, z: 0 },
        interaction: 'look_at',
        flavorText: 'Someone was here recently...'
      },
      {
        type: 'clue_object',
        position: { x: 5, y: 0.8, z: 2 },
        interaction: 'pick_up',
        givesItem: 'note_fragment'
      }
    ];

    for (const detail of details) {
      this.sceneManager.addInteractiveDetail(detail);
    }
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Define Alley Purpose

```
ALLEY DESIGN BRIEF:

1. What is the alley's narrative purpose?
   Intro: Tutorial, introduce exploration
   Long View: Build anticipation to destination
   Navigable: Discovery space, reward exploration
   Dark: Tension, threat hinting

2. Where does it connect?
   From: [previous zone]
   To: [destination zone]

3. What should the player feel?
   Emotional goal: [curiosity/tension/safety]

4. How long should the journey feel?
   Physical length: [meters]
   Pacing length: [perceived time based on details]

5. What's the destination payoff?
   Interactive object? Scene reveal? New zone?
```

### Step 2: Design the Physical Space

```javascript
// Alley space configuration:

const alleyConfig = {
  // Basic dimensions
  dimensions: {
    length: getLengthForType(),     // Based on alley type
    width: getWidthForType(),       // Based on desired feel
    height: 3,                      // Head clearance + room

    // Height variance for interest
    ceilingVariation: {
      enabled: true,
      min: 2.8,
      max: 4.5
    }
  },

  // Path characteristics
  path: {
    straight: type === 'intro',
    hasTurns: type === 'long_view' || type === 'navigable',
    turnCount: type === 'long_view' ? 1 : 2,
    turnSharpness: 'gentle'  // Or 'sharp' for tension
  },

  // Boundaries
  boundaries: {
    left: 'brick_wall',
    right: 'chain_link_fence',
    overhead: 'open_sky'  // Or 'overhead_structure' for dark alleys
  }
};

function getLengthForType() {
  const lengths = {
    intro: 10-15,
    long_view: 30-40,
    navigable: 20-25,
    dark: 15-25
  };
  return lengths[type];
}

function getWidthForType() {
  const widths = {
    intro: 3.5-4,      // Comfortable
    long_view: 2.5-3,  // Intimate
    navigable: 3-5,    // Variable for interest
    dark: 1.5-2        // Claustrophobic
  };
  return widths[type];
}
```

### Step 3: Plan Progressive Reveal

```javascript
// What the player sees/experiences at each point:

const progressiveReveal = {
  // At entry (0% progress)
  entry: {
    visible: ['alley_start', 'first_section'],
    audio: ['previous_zone_ambience', 'faint_destination_sound'],
    lighting: 'from_previous_zone',
    mood: 'transition'
  },

  // At quarter point (25% progress)
  quarter: {
    visible: ['alley_mid_section'],
    audio: ['alley_ambience', 'destination_getting_clearer'],
    lighting: 'transition',
    mood: 'curiosity',
    triggerEvent: 'first_reveal'
  },

  // At halfway (50% progress)
  halfway: {
    visible: ['more_details', 'destination_glimpse'],
    audio: ['clear_destination_audio'],
    lighting: 'destination_influenced',
    mood: 'anticipation',
    triggerEvent: 'halfway_reveal'
  },

  // At destination (100% progress)
  destination: {
    visible: ['full_destination', 'interact_prompt'],
    audio: ['destination_audio', 'interaction_ready'],
    lighting: 'destination_lit',
    mood: 'arrival',
    triggerEvent: 'arrive_destination'
  }
};
```

### Step 4: Layer Environmental Details

```javascript
// Detail placement strategy:

const detailStrategy = {
  // Density progression
  density: {
    entry: 'sparse',      // Don't overwhelm at start
    middle: 'medium',     // Build interest
    destination: 'dense'  // Reward exploration
  },

  // Detail types by alley type
  intro: [
    'street_light',          // Navigational aid
    'minimal_debris',        // Slight texture
    'clear_sight_lines'      // Don't block destination view
  ],

  long_view: [
    'progressive_lighting',  // Gradient from bright to dim
    'mid_point_obstacle',    // Break sight line, create reveal
    'tease_objects',         // Hints before full reveal
    'footprints_path'        // Lead toward destination
  ],

  navigable: [
    'examinable_props',      // Interactive objects
    'environmental_story',   // Clues about what happened
    'multiple_points_of_interest'  // Reasons to explore
  ],

  dark: [
    'flickering_lights',     // Unease
    'ominous_debris',        // Threat suggestion
    'sound_triggers',        // Audio scares
    'limited_visibility'     // Can't see far
  ]
};
```

### Step 5: Design Audio Journey

```javascript
// Audio progression through alley:

const audioJourney = {
  // Crossfade approach
  approach: 'crossfade',

  // Start: Previous zone ambience
  start: {
    primary: 'previous_zone_ambience',
    volume: 0.5,
    fadeOut: true,
    startPoint: 0.0,
    endPoint: 0.3  // Fades out by 30% progress
  },

  // Middle: Alley-specific ambience
  middle: {
    primary: 'alley_ambience',
    volume: 0.4,
    fadeIn: true,
    startPoint: 0.2,
    endPoint: 0.8
  },

  // End: Destination sound
  end: {
    primary: 'destination_sound',
    volumeAtStart: 0.1,   // Faint at beginning
    volumeAtEnd: 0.7,     // Clear at destination
    position: 'destination_object',
    startPoint: 0.0,
    endPoint: 1.0
  },

  // Special trigger sounds
  triggers: [
    { at: 0.5, sound: 'midway_event', once: true },
    { at: 0.8, sound: 'destination_approach', once: true }
  ]
};
```

---

## 🔧 Variations For Your Game

### Variation 1: Vertical Alley (Multi-Level)

```javascript
const verticalAlley = {
  // Instead of horizontal, goes up/down
  type: 'vertical',

  sections: [
    { level: 0, name: 'street_level' },
    { level: 1, name: 'fire_escape', access: 'stairs' },
    { level: 2, name: 'rooftop', access: 'ladder' }
  ],

  // Each level has different character
  atmosphere: {
    0: 'street_level',
    1: 'elevated_exposure',
    2: 'open_air'
  }
};
```

### Variation 2: Time-Shift Alley

```javascript
const timeShiftAlley = {
  // Alley changes as you progress
  type: 'time_shift',

  progression: {
    start: { era: 'modern', lighting: 'electric' },
    middle: { era: '1900s', lighting: 'gas' },
    end: { era: '1800s', lighting: 'candles' }
  },

  // Visual transformation
  transition: 'gradual_morph'
};
```

### Variation 3: Mirror Alley

```javascript
const mirrorAlley = {
  // Reflections show different reality
  type: 'psychological',

  feature: {
    mirrors: {
      count: 5,
      spacing: 'even',
      reflection: 'alternate_reality'
    }
  },

  // Real alley vs reflected alley differ
  reality: {
    visible: 'normal',
    reflected: 'distorted'
  }
};
```

---

## Performance Considerations

```
ALLEY PERFORMANCE OPTIMIZATION:

Splat Rendering:
├── Use region culling (only render current alley)
├── LOD for distant details
├── Don't render alleys player isn't in
└── Target: Smooth transition between zones

Audio:
├── Limit concurrent sounds (3-4 max)
├── Use shared ambience with volume mixing
├── Spatial audio only for nearby sources
└── Target: No audio popping during transitions

Lighting:
├── Pre-bake where possible
├── Use shared lights for similar sections
├── Limit real-time shadows in alleys
└── Target: Stable frame rate throughout

DETAILS:
├── Use instancing for repeated objects (debris)
├── Limit unique mesh count
├── Combine static geometry
└── Target: 60 FPS on target hardware
```

---

## Common Mistakes Beginners Make

### 1. Making Alleys Too Long

```javascript
// ❌ WRONG: 100m alley with no interest
{ length: 100, details: 'minimal' }
// Player gets bored, feels like walking simulator

// ✅ CORRECT: Appropriate length with content
{ length: 35, details: 'progressive', pacing: 'varied' }
```

### 2. No Visual Distinction

```javascript
// ❌ WRONG: All alleys look the same
// Player can't tell where they are

// ✅ CORRECT: Unique character per alley
{ visualTheme: 'unique_to_each' }
```

### 3. Destination Always Visible

```javascript
// ❌ WRONG: See everything from entry
// No mystery, no discovery

// ✅ CORRECT: Progressive reveal
{ revealStrategy: 'gradual', useTurns: true }
```

### 4. Dead Ends Without Reward

```javascript
// ❌ WRONG: Long alley with nothing at end
// Player feels time was wasted

// ✅ CORRECT: Destination has meaning
{ destinationReward: 'meaningful_interaction_or_discovery' }
```

---

## Related Systems

- [ZoneManager](../03-scene-rendering/zone-manager.md) - Zone transitions
- [SceneManager](../03-scene-rendering/scene-manager.md) - Scene rendering
- [SFXManager](../05-media-systems/sfx-manager.md) - Spatial audio
- [Four-Way Intersection](./four-way-intersection-scene.md) - Hub zone
- [Phone Booth Scene](../08-interactive-objects/phone-booth-scene.md) - Example destination

---

## Source File Reference

**Scene Data**:
- `content/SceneData.js` - Alley zone definitions

**Managers**:
- `managers/AlleyManager.js` - Alley-specific logic
- `managers/DetailManager.js` - Environmental props

**Assets**:
- `assets/splats/exterior.ply` - Exterior splat with alley regions
- `assets/audio/alley_ambience.mp3` - Alley atmosphere

---

## 🧠 Creative Process Summary

**From Concept to Alley Scene**:

```
1. DEFINE PURPOSE
   "What is this alley for in the narrative?"

2. CHOOSE TYPE
   "Intro/Long View/Navigable/Dark?"

3. DESIGN SPACE
   "Length, width, path characteristics"

4. PLAN REVEAL
   "What does player see/feel at each point?"

5. ADD DETAILS
   "Environmental storytelling through objects"

6. LAYER AUDIO
   "Audio journey from entry to destination"

7. TEST Pacing
   "Does it feel too long? Too short? Just right?"

8. REFINE
   "Adjust based on player feedback"
```

---

## References

- [Environmental Storytelling](https://www.youtube.com/watch?v=Fte_eO5ykqI) - Video essay
- [Level Design: Corridors](https://www.gamedeveloper.com/) - Article series
- [Audio Spatialization](https://webaudioapi.com/book/) - Technical reference
- [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Rendering tech

*Documentation last updated: January 12, 2026*
