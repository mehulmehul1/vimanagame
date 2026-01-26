# Scene Case Study: Four-Way Intersection

## 🎬 Scene Overview

**Location**: Central hub connecting all exterior zones
**Narrative Context**: The crossroads - player's first meaningful navigation choice, moment of agency
**Player Experience**: Orientation → Choice → Discovery → Return

The Four-Way Intersection is the "heart" of the exterior environment. It's where all paths converge and diverge, serving as both a navigation landmark and a psychological anchor point. This scene demonstrates the art of hub design - creating a space that players naturally return to while always offering new directions to explore.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create a sense of meaningful agency - the player chooses where to go, not the game.

**Why a Hub Design Works**:

```
HUB PSYCHOLOGY:

Single Corridor:
├── Player feels guided/railroaded
├── No decision-making
├── Boring, predictable
└── "I'm being told where to go"

Hub Design:
├── Player feels empowered
├── Meaningful choices
├── Natural exploration
└── "I decide where to go"

THE INTERSECTION IS A HUB:
Not just a place you pass through,
but a place you return to.
```

### Design Decisions

**1. Central Reference Point**
- **Visible Landmarks**: Plaza to south, phone booth to east, radio to west
- **Return Orientation**: Player always knows where the hub is
- **Mental Map**: Intersection becomes the "center" of player's mental map
- **Safety**: Hub feels safe, areas beyond feel unknown

**2. Equal-Weighted Choices**
```
PATH BALANCING:

East Path (Phone Booth):
├── Audio cue: Ringing phone
├── Visual: Visible alley
├── Mystery: Why is it ringing?
└── Appeal: High (curiosity)

West Path (Radio):
├── Audio cue: Faint static/music
├── Visual: Different architecture
├── Mystery: What's the broadcast?
└── Appeal: Medium (ambient interest)

North Path (Alley):
├── No immediate audio cue
├── Visual: Darker, more ominous
├── Mystery: What's down there?
└── Appeal: Variable (brave explorers)

Return to Plaza:
├── Familiar, safe
├── Starting area
├── No mysteries (known)
└── Appeal: Low (already explored)

DESIGN: All paths feel valid, no "correct" choice
```

**3. Progressive Discovery**
```
DISCOVERY FLOW:

Loop 1: Plaza → Intersection → One Path Explore → Return to Intersection
Loop 2: Intersection → Different Path → New Discovery → Return
Loop 3: Intersection → Deeper Exploration → ...

Each return to intersection:
- Confirms orientation ("I know where I am")
- Offers new choice ("What now?")
- Reinforces exploration ("There's still more to find")
```

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
                    FOUR-WAY INTERSECTION LAYOUT:

               [NORTH: Long Alley / Darker]
                     ↑
                     │
                     │
    ═════════════════╦═════════════════
    │                │                 │
    │                │                 │
[W] │            [INTERSECTION]       │ [E]
    │           ┌─────────┐           │
    │           │    ╋    │           │
    │           │  (YOU)  │           │
E   │           └─────────┘           │
a   │                │                 │
s   │                │                 │
t   │                │                 │
    │                ↓                 │
    ══════════════════════════════════
                     │
                     │
                [SOUTH: Plaza / Spawn / Safe Return]

KEY DESIGN ELEMENTS:

Intersection Center:
├── Open space (can stop, look around, decide)
├── Visible landmarks in all directions
├── No obstacles blocking path choices
└── "Breathing room" for decision-making

Path Entry Points:
├── Clearly visible openings
├── Different visual character per path
├── Audio cues help distinguish
└── No ambiguity about "where does this go?"

Return Visibility:
├── Plaza visible from all path exits
├── Intersection visible from all zones
├── Player can always find their way back
└── Mental map: "I'm at the crossroads"
```

### Sight Lines and Navigation

```
SIGHT LINE ANALYSIS - FROM INTERSECTION:

Looking North:
┌─────────────────────────────────────┐
│  ✓ Can see: Alley entrance          │
│  ✓ Can see: Darker atmosphere       │
│  ✗ Cannot see: Alley end contents   │
│                                     │
│  PURPOSE: Hint at danger/mystery,   │
│           don't spoil discovery     │
└─────────────────────────────────────┘

Looking East:
┌─────────────────────────────────────┐
│  ✓ Can see: Phone booth shape       │
│  ✓ Can see: Alley leading to it     │
│  ✗ Cannot see: Phone booth details  │
│                                     │
│  PURPOSE: Audio (ringing) + visual  │
│           create clear attraction   │
└─────────────────────────────────────┘

Looking West:
┌─────────────────────────────────────┐
│  ✓ Can see: Different architecture  │
│  ✓ Can see: Radio glow (if active)  │
│  ✗ Cannot see: Radio interaction    │
│                                     │
│  PURPOSE: Alternative attraction,   │
│           secondary mystery         │
└─────────────────────────────────────┘

Looking South (Return):
┌─────────────────────────────────────┐
│  ✓ Can see: Plaza main area         │
│  ✓ Can see: Spawn point (familiar)  │
│  ✓ Can see: Safe space              │
│                                     │
│  PURPOSE: Always know the way home, │
│           psychological safety      │
└─────────────────────────────────────┘

NAVIGATION PRINCIPLE:
"From the intersection, you should see:
 - Where you came from (orientation)
 - Where you can go (choices)
 - Not what you'll find (mystery)"
```

### Player Flow Patterns

```
TYPICAL PLAYER FLOW:

Pattern A - Sequential Explorer:
┌─────────────────────────────────────────┐
│ Plaza → Intersection → Phone Booth     │
│ ↓                                      │
│ Return to Intersection                 │
│ ↓                                      │
│ Intersection → Radio                  │
│ ↓                                      │
│ Return to Intersection                 │
│ ↓                                      │
│ Intersection → North Alley            │
│ └─→ Deeper exploration...             │
└─────────────────────────────────────────┘

Pattern B - Thorough Explorer:
┌─────────────────────────────────────────┐
│ Plaza → Intersection → Quick look all  │
│ directions → Pick one → Deep dive      │
│ → Return → Pick another...             │
└─────────────────────────────────────────┘

Pattern C - Audio-Follower:
┌─────────────────────────────────────────┐
│ Follow ringing phone → Phone booth     │
│ → After dialog, hear something else    │
│ → Follow that sound → Radio            │
│ → etc.                                 │
└─────────────────────────────────────────┘

KEY INSIGHT:
All patterns use the intersection as
a waypoint/anchor. Hub design supports
multiple playstyles naturally.
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the intersection implementation, you should know:
- **Zone Triggers**: Invisible volumes that detect player entry/exit
- **Scene Streaming**: Loading/unloading content based on player position
- **Audio Attenuation**: Volume based on distance from source
- **Landmark Objects**: Reference points for player orientation
- **Pathfinding Data**: Connection graph for navigation

### Scene Data Structure

```javascript
// SceneData.js - Intersection zone configuration
export const SCENES = {
  intersection: {
    id: 'intersection',
    name: 'Four-Way Intersection',
    type: 'hub',

    // Part of larger splat, not separate file
    splat: {
      file: '/assets/splats/exterior.ply',
      region: {
        minX: -20, maxX: 20,
        minY: 0, maxY: 10,
        minZ: -20, maxZ: 20
      }
    },

    // Zone boundaries for trigger
    bounds: {
      center: { x: 0, y: 1, z: 0 },
      size: { x: 15, y: 5, z: 15 }
    },

    // Connections to all adjacent zones
    connections: [
      {
        to: 'plaza',
        direction: 'south',
        trigger: { x: 0, y: 1, z: 10, radius: 3 },
        entryPoint: { x: 0, y: 1.7, z: 8 },
        visible: true
      },
      {
        to: 'alley_east_phone',
        direction: 'east',
        trigger: { x: 10, y: 1, z: 0, radius: 3 },
        entryPoint: { x: 8, y: 1.7, z: 0 },
        audioAttractor: 'ringing_phone'
      },
      {
        to: 'alley_west_radio',
        direction: 'west',
        trigger: { x: -10, y: 1, z: 0, radius: 3 },
        entryPoint: { x: -8, y: 1.7, z: 0 },
        audioAttractor: 'radio_static'
      },
      {
        to: 'alley_north',
        direction: 'north',
        trigger: { x: 0, y: 1, z: -10, radius: 3 },
        entryPoint: { x: 0, y: 1.7, z: -8 },
        atmosphere: 'darker'
      }
    ],

    // Landmarks for orientation
    landmarks: [
      {
        id: 'plaza_obelisk',
        position: { x: 0, y: 0, z: 12 },
        description: 'South: Back to Plaza',
        alwaysVisible: true
      },
      {
        id: 'phone_booth_visible',
        position: { x: 15, y: 0, z: 0 },
        description: 'East: Ringing Phone',
        audioRange: 20
      }
    ],

    // Ambience
    audio: {
      ambient: {
        sound: 'intersection_ambience',
        volume: 0.3
      },
      positional: [
        { sound: 'ringing_phone', position: { x: 15, y: 1, z: 0 } },
        { sound: 'radio_static', position: { x: -12, y: 1, z: 0 } }
      ]
    }
  }
};
```

### Zone Manager Implementation

```javascript
// ZoneManager.js - Hub zone handling
class ZoneManager {
  constructor() {
    this.currentZone = 'plaza';
    this.previousZone = null;
    this.zoneHistory = [];

    // Preload nearby zones
    this.loadedZones = new Set(['plaza']);
    this.loadingZones = new Set();
  }

  update(playerPosition) {
    // Check all zone triggers
    for (const zone of Object.values(SCENES)) {
      if (this.isInZone(playerPosition, zone)) {
        if (this.currentZone !== zone.id) {
          this.enterZone(zone.id);
        }
      }

      // Check connections (zone transitions)
      for (const connection of zone.connections || []) {
        if (this.isInTrigger(playerPosition, connection.trigger)) {
          this.transitionTo(connection.to);
        }
      }
    }
  }

  enterZone(zoneId) {
    this.previousZone = this.currentZone;
    this.currentZone = zoneId;
    this.zoneHistory.push(zoneId);

    // Emit event for other systems
    game.emit('zone:entered', {
      zone: zoneId,
      from: this.previousZone,
      history: this.zoneHistory
    });

    // Update ambience
    this.updateAmbience(zoneId);

    // Preload adjacent zones
    this.preloadAdjacent(zoneId);
  }

  preloadAdjacent(zoneId) {
    const zone = SCENES[zoneId];
    if (!zone.connections) return;

    for (const connection of zone.connections) {
      const adjacentId = connection.to;
      if (!this.loadedZones.has(adjacentId) &&
          !this.loadingZones.has(adjacentId)) {
        this.loadZone(adjacentId);
      }
    }
  }

  async loadZone(zoneId) {
    if (this.loadedZones.has(zoneId)) return;
    this.loadingZones.add(zoneId);

    // Load zone assets
    await this.loadZoneAssets(zoneId);

    this.loadingZones.delete(zoneId);
    this.loadedZones.add(zoneId);
  }

  updateAmbience(zoneId) {
    const zone = SCENES[zoneId];
    const audio = game.getManager('audio');

    // Fade out old ambience
    audio.fadeAmbience(0.5);

    // Start new ambience
    setTimeout(() => {
      audio.playAmbient(zone.audio.ambient);

      // Set up positional audio
      if (zone.audio.positional) {
        for (const sound of zone.audio.positional) {
          audio.playPositional(sound.sound, sound.position);
        }
      }
    }, 500);
  }

  isInZone(position, zone) {
    const bounds = zone.bounds;
    return position.x >= bounds.center.x - bounds.size.x / 2 &&
           position.x <= bounds.center.x + bounds.size.x / 2 &&
           position.y >= bounds.center.y - bounds.size.y / 2 &&
           position.y <= bounds.center.y + bounds.size.y / 2 &&
           position.z >= bounds.center.z - bounds.size.z / 2 &&
           position.z <= bounds.center.z + bounds.size.z / 2;
  }

  isInTrigger(position, trigger) {
    const dx = position.x - trigger.x;
    const dy = position.y - trigger.y;
    const dz = position.z - trigger.z;
    return Math.sqrt(dx*dx + dy*dy + dz*dz) < trigger.radius;
  }
}
```

### Orientation System

```javascript
// OrientationManager.js - Help players know where they are
class OrientationManager {
  constructor() {
    this.landmarks = new Map();
    this.compassEnabled = true;
  }

  registerLandmark(id, data) {
    this.landmarks.set(id, {
      position: new THREE.Vector3(data.position.x, data.position.y, data.position.z),
      description: data.description,
      alwaysVisible: data.alwaysVisible || false,
      audioRange: data.audioRange || 0
    });
  }

  getPlayerDirections() {
    const player = game.getManager('player').getPosition();
    const playerForward = game.getManager('player').getForward();

    const directions = [];

    for (const [id, landmark] of this.landmarks) {
      const toLandmark = landmark.position.clone().sub(player);
      const distance = toLandmark.length();

      // Check if in range
      if (landmark.audioRange > 0 && distance > landmark.audioRange) {
        continue;
      }

      // Calculate angle
      toLandmark.normalize();
      const dot = playerForward.dot(toLandmark);
      const angle = Math.acos(dot) * (180 / Math.PI);

      // Determine direction
      let direction = '';
      if (angle < 30) direction = 'ahead';
      else if (angle > 150) direction = 'behind';
      else {
        const cross = new THREE.Vector3().crossVectors(playerForward, toLandmark);
        direction = cross.y > 0 ? 'left' : 'right';
      }

      directions.push({
        id,
        description: landmark.description,
        direction,
        distance: Math.round(distance)
      });
    }

    return directions;
  }

  // Optional: Show on-screen compass
  updateCompass() {
    if (!this.compassEnabled) return;

    const directions = this.getPlayerDirections();
    const ui = game.getManager('ui');

    ui.updateCompass(directions);
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Determine Hub Purpose

```
HUB DESIGN QUESTIONS:

1. What connects here?
   Intersection: 4 zones (Plaza, Phone Booth Alley, Radio Alley, North Alley)

2. What's the relationship between zones?
   Intersection: All are exterior, similar difficulty, equal importance

3. Should player always return here?
   Intersection: Yes, it's the central navigation anchor

4. What makes this memorable?
   Intersection: Open space, clear sight lines, audio cues from all directions

5. How do we prevent confusion?
   Intersection: Landmarks visible, consistent layout, audio distinguishes paths
```

### Step 2: Design the Physical Space

```javascript
// Hub space design principles:

const hubDesign = {
  // Size: Large enough to stop and look around
  size: {
    width: 15,   // Meters - room to turn, look, decide
    height: 5,   // Open vertical space
    depth: 15
  },

  // Layout: Central open space, paths radiating outward
  layout: 'radial',

  // Path characteristics
  paths: [
    {
      direction: 'north',
      width: 4,           // Wide enough to feel welcoming
      visualCharacter: 'darker',  // Hints at atmosphere
      audioCue: null,     // Mystery - no sound
      landmark: 'distant_building'
    },
    {
      direction: 'east',
      width: 3,
      visualCharacter: 'alley',
      audioCue: 'ringing_phone',  // Clear attraction
      landmark: 'phone_booth_shape'
    },
    {
      direction: 'west',
      width: 3,
      visualCharacter: 'archway',
      audioCue: 'radio_static',    // Subtle attraction
      landmark: 'radio_glow'
    },
    {
      direction: 'south',
      width: 5,           // Wider = familiar/safe
      visualCharacter: 'open',
      audioCue: null,     // Already explored
      landmark: 'plaza_obelisk'
    }
  ]
};
```

### Step 3: Create Visual Distinction

```javascript
// Make each path visually unique:

const pathDistinction = {
  // Use architecture
  architecture: {
    north: 'brick_walls',      // Industrial feel
    east: 'chain_link_fence',  // Urban decay
    west: 'stone_archway',     // Older, different era
    south: 'open_plaza'        // Familiar, safe
  },

  // Use lighting
  lighting: {
    north: { intensity: 0.5, color: 0x8090a0 },   // Dim, blue
    east: { intensity: 0.7, color: 0xffaa80 },    // Streetlight glow
    west: { intensity: 0.6, color: 0xffddaa },    // Warm interior spill
    south: { intensity: 0.9, color: 0xffffff }    // Bright, safe
  },

  // Use atmosphere
  atmosphere: {
    north: { fogDensity: 0.03 },     // Ominous
    east: { fogDensity: 0.015 },     // Normal
    west: { fogDensity: 0.02 },      // Slightly hazy
    south: { fogDensity: 0.01 }      // Clear
  }
};
```

### Step 4: Layer Audio Guidance

```javascript
// Audio as navigation aid:

const audioGuidance = {
  // Primary attractor (strongest cue)
  primary: {
    sound: 'ringing_phone',
    position: { x: 15, y: 1, z: 0 },
    volume: 0.8,
    maxDistance: 25,
    purpose: 'Clear destination for curious players'
  },

  // Secondary attractor
  secondary: {
    sound: 'radio_static',
    position: { x: -12, y: 1, z: 0 },
    volume: 0.4,
    maxDistance: 15,
    purpose: 'Subtle alternative for thorough explorers'
  },

  // Ambient differentiation
  ambient: {
    north: 'distant_wind',     // Eerie
    east: 'street_ambience',   // Urban
    west: 'interior_echo',     // Suggests interior
    south: 'plaza_breeze'      // Familiar
  }
};

AUDIO PRINCIPLE:
"Each path should have a unique audio
 signature. Players should be able to
 close their eyes and know where
 each path leads."
```

### Step 5: Implement Zone Transitions

```javascript
// Smooth zone transitions:

class ZoneTransition {
  async transitionTo(currentZone, targetZone, playerPos) {
    // 1. Determine transition type
    const transitionType = this.getTransitionType(currentZone, targetZone);

    // 2. Preload target zone assets
    await this.preloadZone(targetZone);

    // 3. Apply transition effect
    switch (transitionType) {
      case 'seamless':
        // No fade, zones share splat
        await this.seamlessTransition(currentZone, targetZone);
        break;

      case 'fade':
        // Quick fade for splat change
        await this.fadeTransition(currentZone, targetZone);
        break;

      case 'portal':
        // Doorway/transition object
        await this.portalTransition(currentZone, targetZone);
        break;
    }

    // 4. Update player position if needed
    const entryPoint = SCENES[targetZone].entryPoint;
    if (entryPoint) {
      game.getManager('player').setPosition(entryPoint);
    }

    // 5. Update ambience
    this.updateZoneAmbience(targetZone);
  }

  getTransitionType(from, to) {
    // If zones share a splat file, seamless
    if (SCENES[from].splat.file === SCENES[to].splat.file) {
      return 'seamless';
    }
    // If there's a physical portal
    if (this.hasPortal(from, to)) {
      return 'portal';
    }
    // Default: fade
    return 'fade';
  }

  async fadeTransition(from, to) {
    // Quick fade (0.3s) to avoid disrupting flow
    await game.getManager('vfx').trigger('fade_out', { duration: 0.3 });

    // Swap splats
    await game.getManager('scene').loadSplat(SCENES[to].splat.file);

    await game.getManager('vfx').trigger('fade_in', { duration: 0.3 });
  }
}
```

### Step 6: Add Orientation Aids

```javascript
// Subtle player guidance:

const orientationAids = {
  // Landmarks
  landmarks: [
    {
      position: { x: 0, y: 3, z: 12 },
      type: 'obelisk',
      purpose: 'Visual anchor pointing to plaza'
    },
    {
      position: { x: 12, y: 2, z: 0 },
      type: 'street_light',
      purpose: 'East path marker'
    }
  ],

  // Subtle lighting cues
  lightingCues: {
    safe: { intensity: 1.0, color: 0xffffcc },  // Plaza direction
    mysterious: { intensity: 0.6, color: 0x80a0cc }  // Unknown areas
  },

  // Optional: On-screen hints after wandering
  hints: {
    triggerAfter: 60,  // Seconds without finding new zone
    message: 'The intersection connects all areas...',
    cooldown: 120
  }
};
```

---

## 🔧 Variations For Your Game

### Variation 1: Circular Plaza Hub

```javascript
const circularHub = {
  layout: 'circular',

  paths: [
    { angle: 0, zone: 'north' },
    { angle: 72, zone: 'northeast' },
    { angle: 144, zone: 'southeast' },
    { angle: 216, zone: 'southwest' },
    { angle: 288, zone: 'northwest' }
  ],

  // Central feature instead of open space
  center: {
    type: 'fountain',
    interactive: true,
    purpose: 'Beautiful centerpiece, also landmark'
  }
};
```

### Variation 2: Multi-Level Hub

```javascript
const multiLevelHub = {
  levels: {
    ground: {
      paths: ['north', 'south', 'east', 'west'],
      atmosphere: 'street_level'
    },
    upper: {
      access: 'stairs_or_elevator',
      paths: ['rooftop_route'],
      atmosphere: 'elevated_vantage'
    },
    lower: {
      access: 'subway_entrance',
      paths: ['underground_route'],
      atmosphere: 'subway_tunnel'
    }
  }
};
```

### Variation 3: Dynamic Hub (Changes Over Time)

```javascript
const dynamicHub = {
  initialState: {
    availablePaths: ['south', 'east'],
    blockedPaths: ['north', 'west'],
    reason: 'construction/debris'
  },

  // After player completes some action:
  changedState: {
    availablePaths: ['south', 'east', 'north', 'west'],
    blockedPaths: [],
    reason: 'debris cleared by event'
  }
};
```

---

## Performance Considerations

```
HUB PERFORMANCE CONSIDERATIONS:

Zone Loading:
├── Preload adjacent zones when entering hub
├── Keep hub zone always loaded (central reference)
├── Unload distant zones to free memory
└── Target: Smooth transitions, no loading stutters

Audio Management:
├── Limit positional audio sources (4-6 max)
├── Use single shared ambience, modify with filters
├── Fade distant sounds to reduce CPU
└── Target: No audio glitches during zone change

Visual Complexity:
├── Hub is shared splat region (no reload needed)
├── Use LOD for distant landmarks
├── Cull objects behind player
└── Target: 60 FPS with all zones loaded

RECOMMENDATION:
The hub should be the "lightest" zone
performance-wise since it's always
potentially active/visible.
```

---

## Common Mistakes Beginners Make

### 1. Making Paths Too Similar

```javascript
// ❌ WRONG: All paths look the same
paths: [
  { direction: 'north', visual: 'brick_alley' },
  { direction: 'east', visual: 'brick_alley' },
  { direction: 'west', visual: 'brick_alley' }
]
// Player can't distinguish, gets confused

// ✅ CORRECT: Distinct visual character
paths: [
  { direction: 'north', visual: 'dark_tunnel' },
  { direction: 'east', visual: 'chain_fence_alley' },
  { direction: 'west', visual: 'stone_archway' }
]
```

### 2. No Return Visibility

```javascript
// ❌ WRONG: Can't see hub from side zones
// Players get lost, don't know how to return

// ✅ CORRECT: Hub always visible
// Use sight lines, lighting, or height differences
```

### 3. Overwhelming Player With Options

```javascript
// ❌ WRONG: 8 paths all at once
// Player freezes, doesn't know where to start

// ✅ CORRECT: 3-4 meaningful choices
// Each is distinct, player can make decision
```

### 4. No Audio Distinction

```javascript
// ❌ WRONG: Same ambience for all paths
// Player relies only on vision

// ✅ CORRECT: Unique audio per path
// Audio + Vision = Clearer navigation
```

---

## Related Systems

- [ZoneManager](../03-scene-rendering/zone-manager.md) - Zone transitions and loading
- [SceneManager](../03-scene-rendering/scene-manager.md) - Scene rendering
- [SFXManager](../05-media-systems/sfx-manager.md) - Spatial audio
- [Plaza Scene](./plaza-scene.md) - Connected safe zone
- [Phone Booth Scene](../08-interactive-objects/phone-booth-scene.md) - East path destination

---

## Source File Reference

**Scene Data**:
- `content/SceneData.js` - Hub zone and connection definitions

**Managers**:
- `managers/ZoneManager.js` - Zone detection and transitions
- `managers/OrientationManager.js` - Player orientation and landmarks

**Assets**:
- `assets/splats/exterior.ply` - Shared exterior splat
- `assets/audio/intersection_ambience.mp3` - Hub atmosphere

---

## 🧠 Creative Process Summary

**From Concept to Intersection Scene**:

```
1. NARRATIVE NEED
   "Player needs a central place to navigate from"

2. HUB DESIGN
   "Four-way intersection, all paths equal weight"

3. VISUAL DISTINCTION
   "Different architecture/lighting per path"

4. AUDIO GUIDANCE
   "Unique sound signature per direction"

5. ORIENTATION AIDS
   "Landmarks, sight lines, always-visible hub"

6. TEST AND REFINE
   "Do players know where they are?
    Can they find their way back?
    Do all paths feel worth exploring?"
```

---

## References

- [Level Design: Hub Worlds](https://www.youtube.com/watch?v=U2V8UoG5-gA) - Video essay
- [Game Space Navigation](https://www.gamasutra.com/blogs/HarveySmith/20190814/) - Article
- [Audio Positioning Guide](https://webaudioapi.com/book/) - Web Audio spatial reference
- [Zone Loading Patterns](https://blog.ourcade.co/posts/2021/open-world-streaming/) - Technical article

*Documentation last updated: January 12, 2026*
