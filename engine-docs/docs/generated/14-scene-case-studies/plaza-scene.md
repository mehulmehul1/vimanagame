# Scene Case Study: The Plaza

## 🎬 Scene Overview

**Location**: Game starting area, exterior plaza environment
**Narrative Context**: Player's first impression of the game world - awakening in an unfamiliar, slightly unsettling environment
**Player Experience**: Disorientation → Curiosity → Exploration → Discovery

The Plaza scene serves as the player's introduction to the game world. It's the "hello" moment - the first impression that establishes the visual style, atmospheric tone, and fundamental gameplay loop. Every element is designed to orient the player while creating immediate intrigue about where they are and what's happening.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create a sense of beautiful unease - an environment that's visually stunning yet feels slightly "off," encouraging exploration while hinting at deeper mystery.

**Design Philosophy for Starting Areas**:

```
PLAYER EMOTIONAL JOURNEY IN PLAZA:

Awakening (Confusion)
    ↓
Visual Orientation (Where am I?)
    ↓
Atmospheric Absorption (This looks amazing)
    ↓
Tactile Exploration (I can move around)
    ↓
Discovery (What's that sound? Where does that go?)
    ↓
Engagement (I want to explore more)
```

### Why This Design Works?

**1. Gaussian Splatting as First Impression**
- **Visual Impact**: Photorealistic environment immediately impresses
- **Familiarity**: Real-world objects feel recognizable and grounding
- **The Uncanny**: Perfect reality feels slightly wrong in a game context
- **Technical Showcase**: Demonstrates engine capabilities immediately

**2. Open but Guided Space**
- **No Walls**: Player feels free to explore
- **Natural Paths**: Environmental cues guide without forcing
- **Multiple Destinations**: Intersection offers choices, not corridors
- **Return Point**: Plaza becomes familiar hub for orientation

**3. Audio as Narrative Hook**
- **Environmental Sounds**: Wind, distant city noises create atmosphere
- **The Ringing Phone**: Audio cue that draws player to first interaction
- **Silence as Tool**: Quiet moments amplify tension when sounds occur

**4. Progressive Disclosure**
```
PLAZA REVEAL STRUCTURE:

Frame 1: Black → Fade In (visual awakening)
Frame 2: Player Spawn View (establishing shot)
Frame 3: Look Around (player agency)
Frame 4: Movement Tutorial (subtle guidance)
Frame 5: Audio Cue Discovery (ringing phone)
Frame 6: First Intersection (path choice)
```

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
                    PLAZA LAYOUT DIAGRAM:

    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║    [SPAWN POINT] ← Player spawns facing North    ║
    ║         ↓                                       ║
    ║    ═══════════                                 ║
    ║    ║          ║  Main open space encourages     ║
    ║    ║  PLAZA  ║  free movement and exploration   ║
    ║    ║  AREA  ║                                  ║
    ║    ║          ║  Environmental landmarks:       ║
    ║    ═════════════                              ║
    ║         ↓                                       ║
    ║    [FOUR-WAY INTERSECTION]                      ║
    ║         ↓                                       ║
    ║    ═════╦═════                                 ║
    ║         ║                                      ║
    ║    ┌────╫────┐                                ║
    ║    │    ║    │  Hub connects to all zones      ║
    ║  [N] [S] [E] [W]                              ║
    ║    │    ║    │  Each direction offers          ║
    ║    └────╫────┘  different discoveries          ║
    ║         ║                                      ║
    ║    ═════╩═════                                 ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝

KEY DESIGN ELEMENTS:

Spawn Point:
├── Positioned for optimal view of plaza
├── Facing toward intersection (suggests forward movement)
├── No immediate threats (safe exploration space)
└── Slightly elevated (slight downward angle = empowerment)

Plaza Area:
├── Open space (freedom of movement)
├── Environmental objects (benches, lights, debris)
├── Splat-captured reality (photorealistic detail)
└── Multiple sight lines (see various destinations)

Intersection:
├── Natural hub (all paths meet here)
├── Clear choice points (left, right, forward, back)
├── Landmarks visible (phone booth, radio, etc.)
└── Return orientation (plaza visible from all paths)
```

### Sight Lines and Player Flow

```
SIGHT LINE ANALYSIS:

From Spawn:
┌─────────────────────────────────────┐
│  ✓ Can see: Plaza main area        │
│  ✓ Can see: Beginning of alleys    │
│  ✓ Can see: Intersection hub       │
│  ✗ Cannot see: End destinations    │
│                                     │
│  PURPOSE: Show immediate options,   │
│           hide distant mysteries    │
└─────────────────────────────────────┘

From Intersection:
┌─────────────────────────────────────┐
│  ✓ Can see: Plaza (safe return)    │
│  ✓ Can see: Phone booth (alley)    │
│  ✓ Can see: Radio (other direction)│
│  ✓ Can see: Multiple path options  │
│                                     │
│  PURPOSE: Navigation hub,          │
│           always know where you are │
└─────────────────────────────────────┘

DESIGN PRINCIPLE:
"Always show where you've been,
hint at where you can go,
never show everything at once"
```

### Atmospheric Design

```
PLAZA ATMOSPHERE LAYERS:

Visual Layer:
├── Gaussian Splat: Photorealistic environment
├── Lighting: Moody, slightly dim (evening/overcast)
├── Colors: Desaturated, earth tones
├── Post-Processing: Subtle vignette focus
└── Effect: Beautiful but melancholy

Audio Layer:
├── Ambient: Wind, distant city sounds
├── Movement: Footstep sounds on different surfaces
├── Spatial: Audio from nearby destinations
└── Effect: Alive but lonely

Interaction Layer:
├── Freedom: Full movement immediately
├── No Barriers: Can walk anywhere
├── Subtle Guidance: Audio cues, not arrows
└── Effect: Empowered exploration

MOOD TARGET:
"This is a real place, but something
 isn't quite right here. I should
 look around, but I should also
 be careful."
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the plaza implementation, you should know:
- **Gaussian Splatting**: Point-cloud rendering technique for photorealistic scenes
- **Three.js Scene Graph**: Parent-child object relationships
- **First-Person Camera**: Camera controls for player view
- **Spawn Points**: Defined positions and orientations for player entry
- **Trigger Zones**: Invisible volumes that detect player presence
- **Audio Spatialization**: 3D positioned sound sources

### Scene Data Structure

```javascript
// SceneData.js - Plaza zone configuration
export const SCENES = {
  plaza: {
    id: 'plaza',
    name: 'Plaza',
    type: 'exterior',

    // Gaussian Splat file
    splat: {
      file: '/assets/splats/plaza.ply',
      // PLY format: Point cloud with position, color, covariance
      // Generated from photogrammetry or LIDAR scan
      maxPoints: 10000000,  // 10M splat points
      lodLevels: 3,         // Level of detail
      renderScale: 1.0      // Full resolution
    },

    // Player spawn configuration
    spawn: {
      position: { x: 0, y: 1.7, z: 5 },  // Eye height (1.7m)
      rotation: { x: 0, y: 0, z: 0 },    // Facing forward (-Z)
      // Player spawns at edge of plaza, looking toward center
    },

    // Environment
    environment: {
      lighting: {
        ambient: { color: 0x404060, intensity: 0.3 },
        directional: {
          color: 0xffeedd,
          intensity: 0.8,
          position: { x: 10, y: 20, z: 10 },
          castShadow: true
        }
      },
      fog: {
        color: 0x505060,
        near: 20,
        far: 100
      }
    },

    // Audio zones
    audio: {
      ambient: {
        sound: 'plaza_ambience',
        volume: 0.4,
        loop: true,
        spatial: false  // Non-positioned background
      }
    },

    // Connections to other zones
    connections: [
      { to: 'intersection', trigger: { x: 0, y: 1, z: 0, radius: 5 } },
      { to: 'alley_north', trigger: { x: 0, y: 1, z: -15, radius: 3 } },
      { to: 'alley_east', trigger: { x: 15, y: 1, z: 0, radius: 3 } },
      { to: 'alley_west', trigger: { x: -15, y: 1, z: 0, radius: 3 } }
    ]
  }
};
```

### Splat Loading and Rendering

```javascript
// SceneManager.js - Plaza initialization
class SceneManager {
  async loadPlazaScene() {
    // 1. Load Gaussian Splat
    const splatRenderer = new SplatRenderer();
    await splatRenderer.load('/assets/splats/plaza.ply');

    // 2. Configure splat rendering
    splatRenderer.configure({
      renderScale: game.options.renderScale,
      splatSize: game.options.splatSize,
      covariance: true,  // Use covariance for splat shape
      opacity: 1.0
    });

    // 3. Add to scene
    this.scene.add(splatRenderer.mesh);

    // 4. Set up environment
    this.setupLighting();
    this.setupPostProcessing();

    // 5. Position player at spawn
    this.spawnPlayer(SCENES.plaza.spawn);
  }

  spawnPlayer(spawnConfig) {
    const player = game.getManager('player');
    player.setPosition(spawnConfig.position);
    player.setRotation(spawnConfig.rotation);

    // Fade in from black
    game.getManager('vfx').trigger('fade_in', {
      duration: 2.0,
      color: 0x000000
    });
  }
}
```

### Zone Transition System

```javascript
// ZoneManager.js - Handling plaza connections
class ZoneManager {
  update(playerPosition) {
    // Check all zone triggers
    for (const connection of SCENES.plaza.connections) {
      if (this.isInTrigger(playerPosition, connection.trigger)) {
        this.transitionTo(connection.to);
      }
    }
  }

  async transitionTo(zoneId) {
    const targetZone = SCENES[zoned];

    // Fade out current zone
    await this.fadeOut(0.5);

    // Load new zone splat
    await this.loadZoneSplat(targetZone.splat.file);

    // Update player position
    this.movePlayerTo(targetZone.entryPoint);

    // Fade in new zone
    await this.fadeIn(0.5);

    // Update ambient audio
    game.getManager('audio').playAmbient(targetZone.audio.ambient);
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Define the Emotional Goal

Before any technical work, answer:

```
CREATIVE BRIEF QUESTIONS:

1. What should the player FEEL in this scene?
   Plaza: Beautiful unease, curiosity, disorientation

2. What is the scene's NARRATIVE purpose?
   Plaza: Establish world, introduce gameplay, create mystery

3. What should the player DO here?
   Plaza: Look around, move freely, discover paths forward

4. What does this scene TEACH the player?
   Plaza: Movement controls, 3D space, audio-visual quality

5. What's the PAYOFF for exploring?
   Plaza: Discovery of first interactive object (phone booth)
```

### Step 2: Capture or Create the Environment

```javascript
// Gaussian Splat acquisition options:

OPTION A: Photogrammetry
├── Tools: RealityCapture, Metashape, COLMAP
├── Process: Photos → 3D point cloud → PLY file
├── Best for: Real-world locations, existing structures
└── Plaza method: Likely photogrammetry

OPTION B: LIDAR Scan
├── Tools: iPhone Pro LiDAR, terrestrial LIDAR
├── Process: Laser scan → point cloud → PLY file
├── Best for: Accurate interior spaces
└── Higher quality, more expensive

OPTION C: Procedural/Modeled
├── Tools: Blender, Maya, traditional 3D
├── Process: Model → UV → Texture → Export
├── Best for: Stylized, controlled environments
└── Not used here (plaza is captured reality)
```

### Step 3: Design Player Entry

```javascript
// Spawn configuration checklist:

const spawnConfig = {
  // Position: Where does player appear?
  position: {
    x: 0,      // Center of space
    y: 1.7,    // Eye height (CRITICAL for first-person immersion)
    z: 5       // Edge of area (not center - allows looking inward)
  },

  // Rotation: What do they see first?
  rotation: {
    // Face toward main point of interest
    // Don't face directly at important interactables
    // Allow natural discovery through turning
  },

  // Initial state: What's happening?
  initialState: {
    movementEnabled: true,      // Immediate control
    audioFadeIn: true,          // Gradual audio
    visualFadeIn: true,         // Gradual visual
    showHint: false             // No hand-holding
  }
};

FIRST IMPRESSION RULE:
"The player's first view should be:
 - Visually interesting
 - Not overwhelming
 - Hinting at exploration possibilities
 - Free of immediate threats"
```

### Step 4: Create Navigation Paths

```javascript
// Path design principles:

const pathDesign = {
  // 1. Multiple options, not single corridor
  type: 'hub',
  branches: ['north', 'south', 'east', 'west'],

  // 2. Visual attraction to each path
  visualCues: {
    north: { visible: 'distant_building' },
    south: { visible: 'glowing_sign' },
    east: { visible: 'interesting_object' },
    west: { visible: 'open_space' }
  },

  // 3. Audio guidance (subtle, not explicit)
  audioCues: {
    north: { sound: 'distant_music', volume: 0.3 },
    east: { sound: 'ringing_phone', volume: 0.5 }
    // Let interesting sounds draw players naturally
  },

  // 4. Always-visible return point
  orientation: {
    landmark: 'plaza_center',
    alwaysVisible: true
  }
};

NAVIGATION PRINCIPLE:
"Players should choose their path through
 curiosity, not necessity. Each direction
 should feel equally valid."
```

### Step 5: Layer Atmosphere

```javascript
// Atmospheric layering:

const atmosphere = {
  // Visual layers
  visual: {
    // Time of day (lighting mood)
    timeOfDay: 'late_afternoon',  // Golden hour = beautiful melancholy

    // Weather/atmosphere
    fog: {
      enabled: true,
      density: 0.02,  // Subtle depth
      color: 0x607080  // Blue-gray overcast
    },

    // Post-processing
    postProcessing: {
      vignette: 0.3,      // Draw focus to center
      bloom: 0.1,         // Subtle glow on lights
      saturation: 0.85,   // Slightly desaturated
      contrast: 1.1       // Mild contrast boost
    }
  },

  // Audio layers
  audio: {
    // Base ambience
    ambient: 'plaza_wind',

    // Layered elements
    layers: [
      { sound: 'distant_traffic', volume: 0.2 },
      { sound: 'wind_chimes', volume: 0.1 },
      { sound: 'birds', volume: 0.15 }
    ],

    // Dynamic based on position
    proximity: {
      phone_booth: { sound: 'ringing', maxDist: 20 },
      radio: { sound: 'static', maxDist: 15 }
    }
  }
};
```

### Step 6: Test and Iterate

```javascript
// Playtesting checklist:

const testCriteria = {
  // First impressions
  first30Seconds: {
    playerUnderstandsWhereTheyAre: true,
    playerKnowsTheyCanMove: true,
    playerFeelsMotivatedToExplore: true,
    visualImpactIsStrong: true
  },

  // Navigation
  navigation: {
    playerCanFindAllPaths: true,
    playerCanReturnToPlaza: true,
    noPlayerGetsStuck: true,
    pathsAreVisuallyDistinct: true
  },

  // Atmosphere
  atmosphere: {
    moodIsAppropriate: true,
    audioEnhancesNotDistracts: true,
    frameRateIsAcceptable: true,
    noVisualGlitches: true
  }
};
```

---

## 🔧 Variations For Your Game

### Variation 1: Industrial Starting Area

```javascript
const industrialPlaza = {
  mood: 'gritty, abandoned',

  visualChanges: {
    // Rust, decay, harsher lighting
    postProcessing: {
      saturation: 0.6,   // More desaturated
      contrast: 1.3,     // Higher contrast
      vignette: 0.5      // Stronger vignette
    },

    // Different time of day
    timeOfDay: 'night',
    lighting: {
      // Harsh artificial lights
      type: 'point_lights',
      color: 0xff6600,   // Orange sodium lights
      castShadow: true
    }
  },

  // Different audio atmosphere
  audio: {
    ambient: 'industrial_hum',
    layers: ['dripping_water', 'metal_creaks', 'distant_machinery']
  }
};
```

### Variation 2: Fantasy Starting Glade

```javascript
const fantasyPlaza = {
  mood: 'magical, inviting',

  visualChanges: {
    // Vibrant colors, soft lighting
    postProcessing: {
      saturation: 1.2,   // Boosted saturation
      bloom: 0.4,        // Strong bloom on magical elements
      vignette: 0.1      // Minimal vignette
    },

    // Magical time
    timeOfDay: 'golden_hour',
    lighting: {
      type: 'directional',
      color: 0xffddaa,   // Warm golden light
      godRays: true
    }
  },

  // Magical audio
  audio: {
    ambient: 'forest_ambience',
    layers: ['magical_chimes', 'gentle_breeze', 'distant_waterfall']
  }
};
```

### Variation 3: Sci-Fi Space Dock

```javascript
const sciFiPlaza = {
  mood: 'cold, technological',

  visualChanges: {
    // Clean, cool tones
    postProcessing: {
      saturation: 0.9,
      bloom: 0.3,        // On holographic elements
      chromaticAberration: 0.05  // Subtle sci-fi feel
    },

    // Artificial environment
    timeOfDay: 'N/A',    // Interior space
    lighting: {
      type: 'area',
      color: 0xaaddff,   // Cool blue-white
      shadows: false     // No shadows (harsh light)
    }
  },

  // Tech audio
  audio: {
    ambient: 'space_station_hum',
    layers: ['computer_beeps', 'air_circulation', 'distant_announcements']
  }
};
```

---

## Performance Considerations

```
PLAZA PERFORMANCE OPTIMIZATION:

Splat Rendering:
├── Point Count: 10M (plaza.ply)
├── Render Scale: 1.0 (desktop), 0.5 (mobile)
├── LOD: Switch to lower detail at distance
├── Frustum Culling: Don't render off-screen splats
└── Target: 60 FPS desktop, 30 FPS mobile

Lighting:
├── Lights: Minimal (directional sun + ambient)
├── Shadows: Optional on mobile
├── Baked Lighting: Pre-computed when possible
└── Target: Single pass lighting

Post-Processing:
├── Vignette: Cheap (full-screen quad)
├── Bloom: Expensive (multi-pass)
├── Fog: Moderate (pixel shader)
└── Target: Quality slider in options

Audio:
├── Sources: 3-4 simultaneous maximum
├── Spatialization: Web Audio API
├── Streaming: Large files, not loaded entirely
└── Target: No audio hitching

RECOMMENDATION:
Profile on target hardware early.
Use PerformanceManager to auto-detect
appropriate quality settings.
```

---

## Common Mistakes Beginners Make

### 1. Player Spawns Facing a Wall

```javascript
// ❌ WRONG: Spawn facing nothing interesting
spawn: {
  position: { x: 0, y: 1.7, z: 0 },
  rotation: { x: 0, y: 0, z: 0 }  // Facing blank wall
}

// ✅ CORRECT: Spawn facing point of interest
spawn: {
  position: { x: 0, y: 1.7, z: 5 },
  rotation: { x: 0, y: Math.PI, z: 0 }  // Facing center
}
```

### 2. Too Many Options At Once

```javascript
// ❌ WRONG: 8 paths, overwhelming
paths: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

// ✅ CORRECT: 3-4 clear options
paths: ['forward', 'left', 'right']  // Player can understand choices
```

### 3. No Visual Return Point

```javascript
// ❌ WRONG: Can't see plaza from side areas
// Players get lost, don't know how to return

// ✅ CORRECT: Plaza always visible
// Use sight lines to keep reference visible
```

### 4. Audio Too Loud From Start

```javascript
// ❌ WRONG: Full volume ambience immediately
audio: { volume: 1.0 }  // Jarring, hurts ears

// ✅ CORRECT: Fade in gradually
audio: {
  volume: 0.4,
  fadeIn: 2.0  // Seconds to full volume
}
```

---

## Related Systems

- [SceneManager](../03-scene-rendering/scene-manager.md) - Scene loading and rendering
- [Gaussian Splatting Explained](../01-foundation/gaussian-splatting-explained.md) - Point cloud rendering
- [CharacterController](../04-input-physics/character-controller.md) - First-person movement
- [SFXManager](../05-media-systems/sfx-manager.md) - Spatial audio
- [VFXManager](../07-visual-effects/vfx-manager.md) - Post-processing
- [ZoneManager](../03-scene-rendering/zone-manager.md) - Scene transitions

---

## Source File Reference

**Scene Data**:
- `content/SceneData.js` - Plaza zone configuration and connections
- `content/AnimationData.js` - Camera animations for title sequence

**Managers**:
- `managers/SceneManager.js` - Scene loading and splat rendering
- `managers/ZoneManager.js` - Zone transitions and loading
- `managers/PlayerManager.js` - Spawn and position management

**Assets**:
- `assets/splats/plaza.ply` - Gaussian splat point cloud
- `assets/audio/plaza_ambience.mp3` - Environmental audio

---

## 🧠 Creative Process Summary

**From Concept to Plaza Scene**:

```
1. EMOTIONAL GOAL
   "Beautiful unease - real but wrong"

2. NARRATIVE PURPOSE
   "Establish world, introduce gameplay"

3. ENVIRONMENT ACQUISITION
   "Photogrammetry of real location"

4. PLAYER PLACEMENT
   "Spawn at edge, look inward, eye height"

5. PATH DESIGN
   "Hub with multiple options, not corridors"

6. ATMOSPHERE LAYERING
   "Lighting + fog + audio + post-processing"

7. TESTING
   "First impression check, navigation verify"

8. ITERATION
   "Adjust based on playtester feedback"
```

---

## References

- [Three.js Scene Graph](https://threejs.org/docs/#api/en/core/Object3D) - Object hierarchy
- [Gaussian Splatting Papers](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Research papers
- [Photogrammetry Guide](https://www.youtube.com/watch?v=7kI1限) - Tutorial series
- [Level Design Book](https://www.youtube.com/@gamedevlab) - Design principles
- [First-Person Camera Controls](https://threejs.org/docs/#examples/en/controls/FirstPersonControls) - Reference

*Documentation last updated: January 12, 2026*
