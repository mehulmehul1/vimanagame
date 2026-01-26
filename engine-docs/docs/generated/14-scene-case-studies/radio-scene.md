# Scene Case Study: The Radio

## 🎬 Scene Overview

**Location**: Exterior alley or building exterior, west path from intersection
**Narrative Context**: Environmental storytelling through a broadcast—a radio playing news/reportage that reveals world details without direct exposition
**Player Experience: Curiosity → Discovery → Worldbuilding through passive listening

The Radio scene represents one of the most elegant forms of environmental storytelling—a passive audio experience that players can choose to engage with or ignore. Unlike the phone booth's direct interaction, the radio offers a broadcast that players can listen to at their leisure, picking up fragments of narrative that paint a picture of the world outside the player's immediate experience.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create atmospheric context—players learn about the world through overheard broadcast, feeling like they're tuning into something larger.

**Why a Radio Works**:

```
ENVIRONMENTAL STORYTELLING THROUGH AUDIO:

Direct Exposition (Boring):
├── "The year is 2024 and..."
├── Player is told information
├── Feels forced, game-y
└─→ "I'm being educated"

Overheard Broadcast (Engaging):
├── "...reports of unusual phenomena..."
├── Player discovers information
├── Feels natural, accidental
└─→ "I stumbled onto something real"

THE RADIO IS:
- A window to the wider world
- Passive (player chooses to listen)
- Atmospheric (adds to mood)
- Non-intrusive (can be ignored)
```

### Design Philosophy

**1. Passive Discovery**

```
PLAYER AGENCY IN DISCOVERY:

Phone Booth (Active):
├── Ringing demands attention
├── Player must interact to progress
├── Forced engagement
└─→ "I have to answer this"

Radio (Passive):
├── Audio plays regardless
├── Player can approach or leave
├── Optional engagement
└─→ "I'm curious, I'll listen"

BOTH APPROACHES VALID:
Active creates key narrative moments
Passive creates atmospheric depth
```

**2. Fragmented Narrative**

```
BROADCAST AS PUZZLE PIECES:

Full Report Would Be:
"Today at 3 PM, authorities confirmed
 reports of strange phenomena in the
 industrial district. Witnesses describe
 lights and sounds that defy explanation.
 Police are investigating..."

Fragmented Delivery (What Player Hears):
♪ [static] ...unusual phenomena... [static]
    ...industrial district... [static]
    ...lights and sounds... [static]
    ...defying explanation... [static] ♪

PLAYER EXPERIENCE:
"I'm piecing this together myself"
More engaging, more mysterious
Player feels smart for decoding
```

**3. Ambient Integration**

```

RADIO AS PART OF ENVIRONMENT:

Not Just:
- Audio plays in vacuum
- Player hears clearly everywhere

But:
- Audio attenuates with distance
- Becomes muffled around corners
- Competes with other ambient sounds
- Feels like real environmental element

RESULT:
Radio feels grounded in space
Player can choose proximity
World feels more real
```

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
                    RADIO SCENE LAYOUT:

    [FOUR-WAY INTERSECTION]
         ↓
    (West Path)
         ↓
    LONG ALLEY
         ↓
    ╔══════════════════════════════════════════════════╗
    ║         ALLEY WITH RADIO                           ║
    ║                                                     ║
    ║  [ENTRANCE]              [RADIO]        [END]       ║
    ║      ↓                     ↓              ↓          ║
    ║  ═════════╗             ╔══════╗      ════════       ║
    ║  │        │             │      │      │           ║
    ║  │        │             │ ☊️   │      │           ║
    ║  │ Dim    │             │      │      │ Dead end  ║
    ║  │ lit    │             │ Glow │      │ or turn  ║
    ║  │        │             │      │      │           ║
    ║  ═════════╝             ╚══════╝      ════════       ║
    ║                                                     ║
    ║  AUDIO GRADIENT:                                   ║
    ║  ════════════════════════════════════════════════  ║
    ║  20m: Barely audible (curiosity)                   ║
    ║  15m: Faint, words unclear (investigate)            ║
    ║  10m: Clear enough to understand (engage)           ║
    ║  5m: Full clarity, immersed (close listening)        ║
    ║                                                     ║
    ╚════════════════════════════════════════════════════╝

DESIGN NOTES:

Positioning:
├── Not directly in path (player can bypass)
├── At end of alley (encourages full traversal)
├── Visible glow attracts attention
└── Audio gradually increases (pulls player in)

Atmosphere:
├── Alley is dimmer than main path
├── Creates intimate listening space
├── Dead end or turn means player stops anyway
└── Natural place to pause and listen
```

### Audio Design

```

RADIO AUDIO STRUCTURE:

Layer 1 - The Broadcast Content:
├── News report format
├── Professional voice
├── Formal language
├── Specific information
└── Purpose: Convey narrative details

Layer 2 - The Radio Hardware:
├── Static/hiss between words
├── Slight distortion
├── Volume fluctuations
├── Occasional interference
└── Purpose: Make it feel like real radio

Layer 3 - Environmental Context:
├── Room reverb
├── Alley echo
├── Wind interference
├── Distance attenuation
└── Purpose: Ground audio in space

EXAMPLE SCRIPT:

♪ [static hiss]

ANNOUNCER: "We interrupt our scheduled
 programming for a special report..."

[static burst]

"...police are investigating reports
 of [static] ...unusual phenomena...

[static clears briefly]

...residents describe [static] ...lights that
 don't match natural patterns...

[longer static]

...authorities urge calm while...

[signal fades]

♪

PRINCIPLE:
Imperfection = authenticity.
Perfect audio feels fake.
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the radio implementation, you should know:
- **Spatial Audio**: 3D positioned sound sources with distance attenuation
- **Audio Loops**: Seamless looping of ambient broadcasts
- **Trigger Zones**: Detecting player proximity for audio level changes
- **Audio Streaming**: Playing long audio without loading entirely into memory
- **Occlusion**: How walls and corners affect sound propagation

### Scene Data Structure

```javascript
// SceneData.js - Radio configuration
export const SCENES = {
  radio_alley: {
    id: 'radio_alley',
    name: 'Alley with Radio',
    type: 'exterior',
    subtype: 'environmental_storytelling',

    // Radio object
    radio: {
      id: 'environmental_radio',
      type: 'prop',
      model: '/assets/models/radio.glb',
      position: { x: -15, y: 1, z: 0 },
      rotation: { x: 0, y: 0.3, z: 0 },

      // Visual appearance
      appearance: {
        glowing: true,
        glowColor: 0xffaa00,  // Warm amber glow
        glowIntensity: 0.5,
        antennaExtended: true
      },

      // Audio configuration
      audio: {
        broadcast: '/assets/audio/radio_broadcast.mp3',
        loop: true,
        baseVolume: 1.0,

        // Distance-based attenuation
        attenuation: {
          maxDistance: 25,  // Can hear from 25m away
          refDistance: 3,   // Full volume within 3m
          rolloffFactor: 1.5
        },

        // Occlusion (walls block sound)
        occlusion: {
          enabled: true,
          wallDamping: 0.7  // Reduces sound through walls
        },

        // Static layers (for realism)
        staticLayers: [
          {
            sound: '/assets/audio/radio_static.ogg',
            volume: 0.15,
            loop: true
          }
        ]
      },

      // Interaction (optional)
      interaction: {
        type: 'passive',  // No direct interaction needed
        canTurnOff: false,  // Or true if player can turn it off
        canChangeStation: false  // Or true for multiple broadcasts
      }
    },

    // Alley characteristics
    alley: {
      length: 20,
      width: 2.5,
      lighting: 'dim',
      atmosphere: 'intimate'
    }
  }
};
```

### Radio Prop Manager

```javascript
// RadioPropManager.js - Environmental radio prop
class RadioPropManager {
  constructor(sceneManager, audioManager) {
    this.scene = sceneManager;
    this.audio = audioManager;

    this.radios = new Map();
    this.activeBroadcasts = new Map();
  }

  createRadio(radioConfig) {
    // Create visual radio prop
    const radio = this.createRadioMesh(radioConfig);

    // Set up audio
    this.setupRadioAudio(radioConfig, radio);

    // Store reference
    this.radios.set(radioConfig.id, radio);

    return radio;
  }

  createRadioMesh(config) {
    // Load or create radio model
    let mesh;
    if (config.model) {
      mesh = this.scene.loadModel(config.model);
    } else {
      mesh = this.createDefaultRadio();
    }

    mesh.position.set(
      config.position.x,
      config.position.y,
      config.position.z
    );

    // Set rotation
    mesh.rotation.y = config.rotation.y;

    // Add glow effect
    if (config.appearance.glowing) {
      this.addRadioGlow(mesh, config.appearance.glowColor);
    }

    return mesh;
  }

  addRadioGlow(mesh, color) {
    // Create point light for glow
    const glow = new THREE.PointLight(color, 0.5, 3);
    glow.position.set(0, 0.5, 0);
    mesh.add(glow);

    // Add emissive material to radio face
    if (mesh.material) {
      mesh.material.emissive = new THREE.Color(color);
      mesh.material.emissiveIntensity = 0.3;
    }
  }

  createDefaultRadio() {
    // Create simple radio geometry
    const group = new THREE.Group();

    // Body
    const bodyGeo = new THREE.BoxGeometry(0.4, 0.25, 0.2);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x4a4a4a,
      roughness: 0.8
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    group.add(body);

    // Speaker grille
    const grilleGeo = new THREE.PlaneGeometry(0.15, 0.1);
    const grilleMat = new THREE.MeshStandardMaterial({
      color: 0x2a2a2a,
      roughness: 0.9
    });
    const grille = new THREE.Mesh(grilleGeo, grilleMat);
    grille.position.set(0, 0.05, 0.11);
    group.add(grille);

    // Antenna
    const antennaGeo = new THREE.CylinderGeometry(0.005, 0.005, 0.3);
    const antennaMat = new THREE.MeshStandardMaterial({
      color: 0x888888,
      roughness: 0.3,
      metalness: 0.8
    });
    const antenna = new THREE.Mesh(antennaGeo, antennaMat);
    antenna.position.set(0.15, 0.2, 0);
    antenna.rotation.x = -0.2;
    group.add(antenna);

    return group;
  }

  setupRadioAudio(config, radioMesh) {
    // Create spatial audio source
    const broadcast = this.audio.createSpatialSource({
      url: config.audio.broadcast,
      position: config.position,
      loop: config.audio.loop,
      volume: config.audio.baseVolume,
      attenuation: config.audio.attenuation,
      occlusion: config.audio.occlusion
    });

    // Add static layer
    if (config.audio.staticLayers) {
      for (const staticLayer of config.audio.staticLayers) {
        broadcast.addLayer(staticLayer.sound, {
          volume: staticLayer.volume,
          loop: true
        });
      }
    }

    // Start playing
    broadcast.play();

    // Store reference
    this.activeBroadcasts.set(config.id, broadcast);

    // Attach to mesh so it moves with it
    radioMesh.userData.audioSource = broadcast;
  }

  update(playerPosition, deltaTime) {
    // Update all radio audio based on player position
    for (const [id, broadcast] of this.activeBroadcasts) {
      const radio = this.radios.get(id);

      // Calculate distance
      const distance = playerPosition.distanceTo(radio.position);

      // Check occlusion (raycast for walls)
      const occlusion = this.calculateOcclusion(playerPosition, radio.position);

      // Update audio
      broadcast.updatePosition(radio.position);
      broadcast.updateOcclusion(occlusion);
    }
  }

  calculateOcclusion(listenerPos, sourcePos) {
    // Raycast from source to listener
    const direction = new THREE.Vector3()
      .subVectors(listenerPos, sourcePos)
      .normalize();

    const ray = new THREE.Ray(sourcePos, direction);
    const distance = sourcePos.distanceTo(listenerPos);

    const hit = this.scene.physics.raycast(ray, {
      maxDistance: distance,
      collisionGroups: 1  // Wall collision group
    });

    if (hit) {
      // Sound is occluded by wall
      return 1.0 - hit.fraction;  // More occlusion = less sound
    }

    return 0;  // No occlusion
  }
}
```

### Broadcast Content System

```javascript
// BroadcastContent.js - Manages radio broadcast scripts
class BroadcastContent {
  constructor() {
    this.broadcasts = new Map();
    this.currentBroadcast = null;
    this.playbackPosition = 0;
  }

  loadBroadcast(broadcastId) {
    // Load broadcast script
    const broadcast = {
      id: broadcastId,
      title: 'News Report - Anomalous Events',

      // Script with timing
      segments: [
        {
          startTime: 0,
          duration: 3,
          type: 'static',
          intensity: 0.5
        },
        {
          startTime: 3,
          duration: 8,
          type: 'speech',
          speaker: 'announcer',
          text: "We interrupt our scheduled programming for a special report. Authorities have confirmed unusual phenomena reported in the industrial district.",
          voice: 'professional_calm'
        },
        {
          startTime: 11,
          duration: 2,
          type: 'static',
          intensity: 0.7
        },
        {
          startTime: 13,
          duration: 10,
          type: 'speech',
          speaker: 'announcer',
          text: "Witnesses describe lights that don't match known patterns and sounds that seem to originate from underground. Police are urging calm while investigations continue.",
          voice: 'professional_calm'
        },
        {
          startTime: 23,
          duration: 5,
          type: 'static',
          intensity: 0.3
        }
      ],

      // Total loop duration
      loopDuration: 28,

      // Audio files
      audioFiles: {
        speech: '/assets/audio/radio_speech.ogg',
        static: '/assets/audio/radio_static.ogg'
      }
    };

    this.broadcasts.set(broadcastId, broadcast);
    return broadcast;
  }

  // Get text at current playback position
  getCurrentText(broadcastId, elapsedTime) {
    const broadcast = this.broadcasts.get(broadcastId);
    if (!broadcast) return null;

    // Loop time within broadcast duration
    const loopTime = elapsedTime % broadcast.loopDuration;

    // Find current segment
    for (const segment of broadcast.segments) {
      if (loopTime >= segment.startTime &&
          loopTime < segment.startTime + segment.duration) {
        if (segment.type === 'speech') {
          return segment.text;
        }
      }
    }

    return null;
  }

  // Get subtitles for display
  getSubtitle(broadcastId, elapsedTime) {
    const text = this.getCurrentText(broadcastId, elapsedTime);

    if (text) {
      // Calculate progress within current speech segment
      // for subtitle reveal animation
      return {
        text: text,
        progress: this.getTextProgress(broadcastId, elapsedTime)
      };
    }

    return null;
  }

  getTextProgress(broadcastId, elapsedTime) {
    const broadcast = this.broadcasts.get(broadcastId);
    const loopTime = elapsedTime % broadcast.loopDuration;

    for (const segment of broadcast.segments) {
      if (segment.type === 'speech' &&
          loopTime >= segment.startTime &&
          loopTime < segment.startTime + segment.duration) {
        const segmentProgress = (loopTime - segment.startTime) / segment.duration;
        return Math.min(1, segmentProgress * 2);  // Text reveals over half duration
      }
    }

    return 0;
  }
}
```

### Subtitle Display System

```javascript
// RadioSubtitleUI.js - Shows radio content as text
class RadioSubtitleUI {
  constructor() {
    this.container = null;
    this.currentText = null;
    this.isVisible = false;
  }

  show(proximity) {
    if (!this.container) {
      this.createContainer();
    }

    // Only show when player is close enough
    if (proximity < 15) {  // Within hearing range
      this.isVisible = true;
      this.container.style.opacity = this.getOpacityForDistance(proximity);
    } else {
      this.isVisible = false;
      this.container.style.opacity = 0;
    }
  }

  getOpacityForDistance(distance) {
    // Fade in as player approaches
    if (distance > 15) return 0;
    if (distance < 8) return 0.8;
    return 0.8 * (1 - (distance - 8) / 7);
  }

  createContainer() {
    this.container = document.createElement('div');
    this.container.className = 'radio-subtitles';
    this.container.style.cssText = `
      position: fixed;
      bottom: 20%;
      left: 50%;
      transform: translateX(-50%);
      max-width: 600px;
      text-align: center;
      color: rgba(255, 255, 255, 0.9);
      font-size: 18px;
      font-family: 'Courier New', monospace;
      text-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
      pointer-events: none;
      transition: opacity 0.3s ease;
      opacity: 0;
      z-index: 100;
    `;

    document.body.appendChild(this.container);
  }

  update(subtitleData) {
    if (!subtitleData || !this.isVisible) {
      if (this.container) {
        this.container.textContent = '';
      }
      return;
    }

    // Reveal text progressively
    const charCount = Math.floor(subtitleData.text.length * subtitleData.progress);
    const revealedText = subtitleData.text.substring(0, charCount);

    this.container.textContent = revealedText;
  }

  hide() {
    this.isVisible = false;
    if (this.container) {
      this.container.style.opacity = 0;
    }
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Define the Information Goal

```
RADIO CONTENT BRIEF:

1. What does player need to know?
    World context, recent events, atmosphere

2. What should player feel?
    Curious, like they're tuning into something real

3. How much information?
    Enough to intrigue, not so much it overwhelms

4. What's the tone?
    News report = professional, adds authenticity

5. How does this connect to main narrative?
    Hints at larger story without revealing all
```

### Step 2: Write the Broadcast

```javascript
// Broadcast script template:

const broadcastScript = {
  intro: [
    { text: "We interrupt...", duration: 3 },
    { static: 1 }
  ],

  content: [
    { text: "Key information 1", duration: 5 },
    { static: 0.5 },
    { text: "Key information 2", duration: 4 },
    { static: 0.5 },
    { text: "Intriguing hint...", duration: 6 }
  ],

  outro: [
    { static: 2 },
    { text: "Please stand by...", duration: 3 }
  ]
};
```

### Step 3: Design the Space

```javascript
// Positioning strategy:

const radioPlacement = {
  // Not in main path
  offsetFromMain: true,

  // At end of side path
  encouragesFullTraversal: true,

  // Visual attraction
  glowVisibleFrom: 20,  // meters

  // Audio attraction
  audibleFrom: 25,

  // Clear listening area
  nearbySpace: 'dead_end_or_niche'
};
```

---

## 🔧 Variations For Your Game

### Variation 1: Television

```javascript
const televisionProp = {
  // Instead of radio, use TV
  type: 'television',

  // Visual content instead of just audio
  hasVideo: true,
  videoContent: 'news_broadcast.webm',

  // Screen glow more visible
  lightEmits: true,

  // Can be turned off/on
  interactive: true
};
```

### Variation 2: Phone Call

```javascript
const phoneCallProp = {
  // Overheard phone conversation
  type: 'phone',

  // Two-sided conversation
  speakers: ['caller', 'receiver'],

  // More intimate, personal
  proximity: 'closer',

  // Different information revealed
  content: 'personal_rather_than_news'
};
```

### Variation 3: Multiple Stations

```javascript
const multiRadio = {
  // Player can tune between stations
  tunable: true,

  stations: [
    { id: 'news', content: 'broadcast_news' },
    { id: 'music', content: 'broadcast_music' },
    { id: 'static', content: 'broadcast_static' }
  ],

  // Each reveals different information
  narrative: 'fragmented_across_stations'
};
```

---

## Performance Considerations

```
RADIO PERFORMANCE:

Audio:
├── Streaming, not full load
├── Loop point handling
├── Spatial processing (moderate CPU)
└── Target: No audio glitches

Occlusion:
├── Raycast per frame (can be expensive)
├── Consider simplified line test
├── Or disable occlusion for mobile
└── Target: Acceptable without occlusion

Subtitles:
├── Simple text display (cheap)
├── Reveal calculation (minimal)
└── Target: Negligible impact
```

---

## Common Mistakes Beginners Make

### 1. Too Clear/Perfect Audio

```javascript
// ❌ WRONG: Perfect broadcast quality
// Breaks immersion, feels fake

// ✅ CORRECT: Static, interference, imperfections
// Feels like real radio broadcast
```

### 2. Information Dump

```javascript
// ❌ WRONG: All backstory in one broadcast
// Overwhelming, boring

// ✅ CORRECT: Fragments, hints, intrigue
// Player wants to learn more
```

### 3: No Spatial Context

```javascript
// ❌ WRONG: Same volume everywhere
// Radio feels disconnected from world

// ✅ CORRECT: Distance attenuation, occlusion
// Radio exists in 3D space
```

---

## Related Systems

- [SFXManager](../05-media-systems/sfx-manager.md) - Spatial audio
- [Phone Booth Scene](../08-interactive-objects/phone-booth-scene.md) - Active counterpart
- [Alley Sections](./alley-sections-scene.md) - Environmental context

---

## Source File Reference

**Scene Data**:
- `content/SceneData.js` - Radio zone configuration

**Managers**:
- `managers/RadioPropManager.js` - Radio prop control
- `managers/BroadcastContent.js` - Broadcast script management

**Assets**:
- `assets/models/radio.glb` - Radio prop model
- `assets/audio/radio_broadcast.mp3` - Broadcast audio

---

## References

- [Web Audio API](https://webaudioapi.com/book/) - Spatial audio reference
- [Environmental Storytelling](https://www.youtube.com/watch?v=Fte_eO5ykqI) - Video essay
- [Audio Attenuation](https://webaudioapi.com/book/chapter-4/) - Technical guide

*Documentation last updated: January 12, 2026*
