# Scene Case Study: Office Interior

## 🎬 Scene Overview

**Location**: Interior office building, accessed from exterior zones
**Narrative Context**: A transition from exterior exploration to interior mystery—a safe space that holds clues to the larger narrative
**Player Experience**: Relief → Curiosity → Investigation → Unease (pre-Office Hell transformation)

The Office Interior scene represents a crucial narrative and atmospheric shift in the game. After navigating the exterior alleys and encountering the phone booth's disturbing call, the office offers a moment of respite—a "safe" interior space that feels initially welcoming but holds its own mysteries. This scene demonstrates how to create interior spaces that balance exploration with narrative progression.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Create a sense of "false safety"—a space that feels comforting initially but hints at darker truths.

**Why This Design Matters**:

```
INTERIOR EMOTIONAL JOURNEY:

Exterior (Unknown/Tense)
    ↓
Enter Office (Relief/Safety)
    ↓
Explore Familiar Environment (Comfort)
    ↓
Discover Clues (Curiosity)
    ↓
Notice Something Wrong (Unease)
    ↓
[Transformation to Office Hell]

DESIGN PRINCIPLE:
"Safe spaces make the horror more effective
 when they transform. The comfort of the
 office makes its nightmare version
 more terrifying."
```

### Design Decisions

**1. Familiarity as Comfort**

The office is designed to feel recognizably real:
- **Everyday Objects**: Desks, chairs, computers, papers
- **Normal Layout**: Logical room organization
- **Lighting**: Warm, artificial office lighting
- **Sound**: HVAC hum, fluorescent buzz (comforting to some)

**2. Environmental Storytelling**

```
WHAT THE OFFICE TELLS US:

Visual Clues:
├── Papers scattered → Something happened here
├── Coffee mug, half-full → Sudden departure
├── Family photos → Real people worked here
├── Calendar, stuck on a date → Time is frozen
└── Personal items → Lives interrupted

Audio Clues:
├── HVAC hum → Normal office sound
├── Clock ticking → Time passes (or doesn't?)
├── Distant street sounds → Connection to exterior
└── Silence → No one else is here

Narrative Function:
├── This was a real place
├── Real people worked here
├── Something interrupted normalcy
├── Player is investigating what happened
```

**3. Progressive Unsettling**

```
FROM NORMAL TO WRONG:

Initial Entry (Normal):
├── Familiar office environment
├── Warm lighting
├── Organized space
└── Player feels: "I can explore safely"

Deeper Exploration (Questions):
├── Slight inconsistencies
├── Documents that don't quite make sense
├── Objects that seem slightly "off"
└── Player feels: "Something happened here"

Discovery (Unsettling):
├── Clear signs of disruption
├── Personal items with disturbing context
├── Hints at the transformation to come
└── Player feels: "This isn't right..."

Climax (Transformation):
├── Office Hell begins
├── Safe space becomes nightmare
└── Player feels: "Nowhere is safe"
```

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
                    OFFICE INTERIOR LAYOUT:

    ╔════════════════════════════════════════════════════╗
   ║                                                      ║
   ║  [ENTRANCE] → Small foyer with door to exterior      ║
   ║       ↓                                               ║
   ║  ╔═══════════════════════════════════════════════╗   ║
   ║  ║                   MAIN OFFICE                  ║   ║
   ║  ║                                                ║   ║
   ║  ║  [WINDOW WALL] ← View to exterior (safe)       ║   ║
   ║  ║                                                ║   ║
   ║  ║     ╔═══════════════╗                          ║   ║
   ║  ║     ║  DESK AREA    ║ ← Player's focus area   ║   ║
   ║  ║     ║               ║   - Computer (on)       ║   ║
   ║  ║     ║  [MONITOR]    ║   - Documents           ║   ║
   ║  ║     ║  [KEYBOARD]   ║   - Personal items      ║   ║
   ║  ║     ║  [PAPERS]     ║   - Viewmaster (hidden) ║   ║
   ║  ║     ╚═══════════════╝                          ║   ║
   ║  ║                                                ║   ║
   ║  ║  [SIDE WALL]                                  ║   ║
   ║  ║    - Filing cabinet (searchable)              ║   ║
   ║  ║    - Bookshelf (clues)                        ║   ║
   ║  ║    - Coat rack (empty)                        ║   ║
   ║  ║                                                ║   ║
   ║  ║  [BACK WALL]                                 ║   ║
   ║  ║    - Door to STORAGE (locked initially)       ║   ║
   ║  ║    - Clock (stuck or working?)                ║   ║
   ║  ║                                                ║   ║
   ║  ╚═══════════════════════════════════════════════╝   ║
    ║                                                      ║
    ╚════════════════════════════════════════════════════╝

KEY INTERACTIVE AREAS:

Desk Area (Primary Focus):
├── Computer: Can be accessed, shows documents
├── Documents: Clues about what happened
├── Viewmaster: Hidden, key to next narrative beat
├── Personal Items: Photos, mug (storytelling)
└── Drawers: Can be opened, contain items

Storage Room (Secondary):
├── Initially locked
├── Opens after certain criteria met
├── Contains additional clues
└── Potential path to other areas

Window Wall:
├── View to exterior (connection to known space)
├── Day/night cycle visible (or frozen?)
├── Provides orientation
└── Transformation visible from here (later)
```

### Player Path Flow

```
OFFICE EXPLORATION FLOW:

Entry:
┌─────────────────────────────────────────┐
│ Player enters from exterior              │
│ Door closes behind (can reopen)          │
│ Initial impression: Normal office        │
│ Action: Look around, get bearings        │
└─────────────────────────────────────────┘
            ↓
Primary Exploration:
┌─────────────────────────────────────────┐
│ Desk area attracts attention            │
│ Computer screen on (glowing)            │
│ Papers scattered (what happened?)       │
│ Action: Examine desk, read documents    │
└─────────────────────────────────────────┘
            ↓
Secondary Discovery:
┌─────────────────────────────────────────┐
│ Notice side areas                       │
│ Bookshelf, filing cabinet               │
│ Storage door (locked)                   │
│ Action: Search for clues, try door      │
└─────────────────────────────────────────┘
            ↓
Key Discovery:
┌─────────────────────────────────────────┐
│ Find Viewmaster (hidden or revealed)    │
│ This triggers next narrative event      │
│ Or leads toward Office Hell             │
│ Action: Pick up Viewmaster, use it      │
└─────────────────────────────────────────┘
            ↓
[Transformation Preparation]
            ↓
Office Hell (or next narrative beat)
```

### Atmosphere Design

```
OFFICE ATMOSPHERE LAYERS:

Visual Layer:
├── Warm artificial lighting (fluorescent)
├── Organized clutter (lived-in, not chaotic)
├── Realistic materials (wood, metal, fabric)
├── Familiar colors (beiges, grays, muted blues)
└── Effect: Comforting, recognizable, safe

Audio Layer:
├── HVAC hum (constant, reassuring)
├── Fluorescent buzz (subtle)
├── Clock ticking (time marker)
├── Distant exterior sounds (connection)
└── Effect: Normalcy, routine, safety

Interaction Layer:
├── Examine objects (desk, papers, etc.)
├── Open drawers (discovery)
├── Read computer (information)
├── Find Viewmaster (n progression)
└── Effect: Investigation, agency, discovery

MOOD TARGET:
"This is a normal place where something
 unusual happened. I'm safe for now,
 but I should figure out what went on
 here."
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the office interior implementation, you should know:
- **Interior Splat Rendering**: Gaussian splatting for indoor spaces
- **Bounding Volume Hierarchy**: Efficient indoor collision detection
- **Dynamic Lighting**: Interior light sources vs exterior
- **Interactive Objects**: Examine, pickup, and trigger systems
- **State-Based Content**: Objects that change based on game state

### Scene Data Structure

```javascript
// SceneData.js - Office interior configuration
export const SCENES = {
  office_interior: {
    id: 'office_interior',
    name: 'Office Interior',
    type: 'interior',

    // Interior Gaussian Splat
    splat: {
      file: '/assets/splats/office.ply',
      // Interior splats often need different settings
      settings: {
        renderScale: 1.0,
        splatSize: 1.2,  // Slightly larger for close-up viewing
        opacity: 1.0
      }
    },

    // Entry from exterior
    entryPoint: {
      position: { x: 0, y: 1.7, z: 4 },
      rotation: { x: 0, y: Math.PI, z: 0 },  // Face into room
      transition: 'fade',
      duration: 0.5
    },

    // Interior lighting
    lighting: {
      ambient: {
        color: 0xe8e8f0,
        intensity: 0.4
      },
      // Fluorescent ceiling lights
      ceilingLights: [
        {
          position: { x: 0, y: 2.8, z: 0 },
          color: 0xffffee,
          intensity: 0.8,
          castShadow: false,  // Too expensive for interior
          type: 'area'
        }
      ],
      // Monitor glow
      monitorLight: {
        position: { x: -0.5, y: 1.2, z: -1 },
        color: 0x88aaff,
        intensity: 0.3,
        flicker: false
      }
    },

    // Interactive objects
    interactables: [
      {
        id: 'office_desk',
        type: 'examine',
        position: { x: -0.5, y: 1, z: -1 },
        interactions: [
          { action: 'examine', result: 'desk_description' },
          { action: 'open_drawer', result: 'drawer_contents' }
        ]
      },
      {
        id: 'office_computer',
        type: 'use',
        position: { x: -0.5, y: 1.2, z: -1 },
        interactions: [
          { action: 'use', result: 'show_computer_screen' },
          { action: 'read_document', result: 'document_content' }
        ],
        state: {
          screenOn: true,
          currentDocument: 'welcome'
        }
      },
      {
        id: 'viewmaster',
        type: 'pickup',
        position: { x: 0.5, y: 0.95, z: -0.5 },
        visible: false,  // Hidden initially
        revealCriteria: {
          hasReadComputer: true,
          hasExaminedDesk: true
        },
        onPickup: {
          trigger: 'viewmaster_acquired',
          nextState: 'viewmaster_sequence'
        }
      },
      {
        id: 'storage_door',
        type: 'door',
        position: { x: 2, y: 1, z: -2 },
        locked: true,
        unlockCriteria: {
          hasKey: false,
          puzzleSolved: true
        }
      }
    ],

    // Audio
    audio: {
      ambience: 'office_ambience',  // HVAC hum
      volume: 0.3,
      layers: [
        { sound: 'fluorescent_hum', volume: 0.1, loop: true },
        { sound: 'clock_tick', volume: 0.05, loop: true, interval: 1.0 }
      ],
      positional: [
        {
          sound: 'computer_fan',
          position: { x: -0.5, y: 1, z: -1 },
          volume: 0.15,
          radius: 3
        }
      ]
    },

    // Clues and narrative content
    clues: [
      {
        id: 'desk_photos',
        location: 'office_desk',
        content: 'family_photos',
        mood: 'normal_people_worked_here'
      },
      {
        id: 'calendar',
        location: 'wall',
        content: 'frozen_date',
        mood: 'time_stopped'
      },
      {
        id: 'computer_document',
        location: 'office_computer',
        content: 'project_notes',
        mood: 'something_was_being_developed'
      }
    ]
  }
};
```

### Office Manager Implementation

```javascript
// OfficeManager.js - Interior-specific logic
class OfficeManager {
  constructor(sceneManager, interactionManager, gameState) {
    this.sceneManager = sceneManager;
    this.interaction = interactionManager;
    this.gameState = gameState;

    this.currentOffice = null;
    this.objectsExamined = new Set();
    this.discoveredClues = new Set();
  }

  enterOffice(officeId) {
    const office = SCENES[officeId];
    this.currentOffice = office;

    // Load office splat
    this.loadOfficeSplat(office);

    // Set up interior lighting
    this.setupLighting(office);

    // Register interactive objects
    this.registerInteractables(office);

    // Start office ambience
    this.startAmbience(office);

    // Check for reveal criteria (Viewmaster, etc.)
    this.checkRevealCriteria();
  }

  loadOfficeSplat(office) {
    this.sceneManager.loadSplat(office.splat.file, {
      // Indoor settings
      renderScale: office.splat.settings.renderScale,
      splatSize: office.splat.settings.splatSize,
      // Indoor scenes often need different opacity handling
      opacityMode: 'additive',
      // Close objects need more detail
      lodBias: -0.5
    });
  }

  setupLighting(office) {
    // Create interior lights
    for (const lightConfig of office.lighting.ceilingLights) {
      const light = new THREE.AreaLight(
        lightConfig.color,
        lightConfig.intensity
      );
      light.position.set(
        lightConfig.position.x,
        lightConfig.position.y,
        lightConfig.position.z
      );
      light.lookAt(0, 0, 0);
      this.sceneManager.addLight(light);
    }

    // Add monitor glow
    const monitorLight = new THREE.PointLight(
      office.lighting.monitorLight.color,
      office.lighting.monitorLight.intensity,
      3  // Radius
    );
    monitorLight.position.set(
      office.lighting.monitorLight.position.x,
      office.lighting.monitorLight.position.y,
      office.lighting.monitorLight.position.z
    );
    this.sceneManager.addLight(monitorLight);
  }

  registerInteractables(office) {
    for (const object of office.interactables) {
      this.interaction.register(object);

      // Set up reveal triggers
      if (object.revealCriteria) {
        this.setupRevealTrigger(object);
      }
    }
  }

  setupRevealTrigger(object) {
    // Watch for criteria to be met
    const checkReveal = () => {
      if (this.meetsCriteria(object.revealCriteria)) {
        this.revealObject(object.id);
      }
    };

    // Check after each interaction
    game.on('interaction:complete', checkReveal);
  }

  meetsCriteria(criteria) {
    for (const [key, value] of Object.entries(criteria)) {
      switch (key) {
        case 'hasReadComputer':
          if (this.objectsExamined.has('office_computer') !== value)
            return false;
          break;
        case 'hasExaminedDesk':
          if (this.objectsExamined.has('office_desk') !== value)
            return false;
          break;
      }
    }
    return true;
  }

  revealObject(objectId) {
    const object = this.currentOffice.interactables.find(o => o.id === objectId);
    if (!object || object.visible) return;

    // Reveal animation
    this.sceneManager.revealObject(objectId, {
      duration: 1.0,
      effect: 'fade_in',
      onComplete: () => {
        object.visible = true;
        game.emit('object:revealed', { id: objectId });

        // Add interaction prompt
        this.interaction.showPrompt(objectId, 'Press E to examine');
      }
    });
  }

  examineObject(objectId) {
    this.objectsExamined.add(objectId);

    // Get object data
    const object = this.currentOffice.interactables.find(o => o.id === objectId);
    if (!object) return;

    // Show examination UI
    this.showExaminationUI(object);

    // Check for new reveals
    this.checkRevealCriteria();

    // Track discovery
    game.emit('clue:discovered', {
      objectId,
      location: this.currentOffice.id
    });
  }

  showExaminationUI(object) {
    // Different UI based on object type
    switch (object.type) {
      case 'examine':
        this.interaction.showDescription(object.description);
        break;
      case 'use':
        this.interaction.showUseInterface(object);
        break;
      case 'pickup':
        this.interaction.showPickupPrompt(object);
        break;
    }
  }

  checkRevealCriteria() {
    for (const object of this.currentOffice.interactables) {
      if (object.revealCriteria && !object.visible) {
        if (this.meetsCriteria(object.revealCriteria)) {
          this.revealObject(object.id);
        }
      }
    }
  }

  startAmbience(office) {
    const audio = game.getManager('audio');

    // Main ambience
    audio.playAmbient(office.audio.ambience, {
      volume: office.audio.volume,
      fadeIn: 1.0
    });

    // Layered sounds
    for (const layer of office.audio.layers) {
      audio.playLayered(layer.sound, {
        volume: layer.volume,
        loop: layer.loop,
        interval: layer.interval
      });
    }

    // Positional sounds
    for (const sound of office.audio.positional) {
      audio.playPositional(sound.sound, sound.position, {
        volume: sound.volume,
        radius: sound.radius
      });
    }
  }

  exitOffice() {
    // Clean up office-specific state
    this.objectsExamined.clear();
    this.currentOffice = null;

    // Stop office ambience
    game.getManager('audio').fadeAmbience(0.5);
  }
}
```

### Computer Interface System

```javascript
// ComputerInterface.js - In-game computer UI
class ComputerInterface {
  constructor() {
    this.documents = new Map();
    this.currentDocument = null;
  }

  loadComputer(computerId) {
    const computer = SCENES.office_interior.interactables
      .find(o => o.id === computerId);

    // Load documents
    this.documents.set('welcome', {
      title: 'Welcome to Shadow Corp',
      content: `
        <h2>Shadow Corp Internal Network</h2>
        <p>Last login: October 14, 2024 - 9:32 AM</p>
        <hr>
        <p>Select a document to read:</p>
        <ul>
          <li onclick="loadDoc('project_notes')">Project Notes</li>
          <li onclick="loadDoc('journal')">Research Journal</li>
          <li onclick="loadDoc('email')">Recent Email</li>
        </ul>
      `
    });

    this.documents.set('project_notes', {
      title: 'Project Notes',
      content: `
        <h2>Project Gaussian</h2>
        <p>Status: Phase 3 Testing</p>
        <p>The reality capture system is working beyond expectations.
        Subjects report increasingly vivid experiences in the
        captured spaces.</p>
        <p><em>[Document continues...]</em></p>
        <p class="warning">Note: Subject #7 reported
        "bleeding through" effects. Monitoring required.</p>
      `
    });

    // Show initial document
    this.showDocument('welcome');
  }

  showDocument(docId) {
    const doc = this.documents.get(docId);
    this.currentDocument = doc;

    // Update UI
    const ui = game.getManager('ui');
    ui.showComputerInterface({
      title: doc.title,
      content: doc.content,
      onBack: () => this.showDocument('welcome'),
      onClose: () => this.close()
    });

    // Track that player has read this
    this.documents.get(docId).read = true;
    game.emit('computer:document_read', { docId });
  }

  close() {
    game.getManager('ui').hideComputerInterface();
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Define Interior Purpose

```
INTERIOR DESIGN BRIEF:

1. What is the narrative purpose?
    Office: Safe haven, clue discovery,
             narrative progression, contrast
             to exterior and later horror

2. What should the player feel?
    Office: Initially safe → curious → unsettled

3. What actions should the player take?
    Office: Explore, examine objects, read
             documents, find Viewmaster

4. How does this connect to other scenes?
    Office: Accessible from exterior,
             leads to Office Hell transformation

5. What's the key discovery/progression?
    Office: Viewmaster acquisition, computer
             documents reveal story
```

### Step 2: Design the Space Layout

```javascript
// Interior space configuration:

const interiorConfig = {
  // Room dimensions
  dimensions: {
    width: 6,    // Meters
    depth: 5,
    height: 3
  },

  // Zone layout
  zones: [
    {
      name: 'entry',
      size: { width: 2, depth: 1.5 },
      purpose: 'transition from exterior'
    },
    {
      name: 'main_area',
      size: { width: 4, depth: 3 },
      purpose: 'primary exploration'
    },
    {
      name: 'desk_area',
      size: { width: 1.5, depth: 1 },
      purpose: 'key interactions'
    }
  ],

  // Furniture placement
  furniture: [
    {
      type: 'desk',
      position: { x: -0.5, z: -1 },
      facing: 'south',
      contents: ['computer', 'documents', 'viewmaster']
    },
    {
      type: 'chair',
      position: { x: -0.5, z: 0 },
      facing: 'north'
    },
    {
      type: 'bookshelf',
      position: { x: -2.5, z: -2 },
      contents: ['books', 'clues']
    }
  ]
};
```

### Step 3: Create Environmental Storytelling

```javascript
// Storytelling through objects:

const storytellingObjects = [
  {
    object: 'coffee_mug',
    position: 'on_desk',
    state: 'half_full',
    story: 'Someone left suddenly, mid-activity'
  },
  {
    object: 'family_photo',
    position: 'on_desk',
    content: 'normal_family',
    story: 'Real people work here, have lives'
  },
  {
    object: 'calendar',
    position: 'on_wall',
    date: 'October 14, 2024',
    story: 'Time may be frozen at this moment'
  },
  {
    object: 'scattered_papers',
    position: 'on_floor',
    state: 'haphazard',
    story: 'Something disruptive happened'
  }
];

STORYTELLING PRINCIPLE:
"Every object should tell part of the story.
 The coffee mug isn't just a mug—it's
 evidence of sudden departure. The
 calendar isn't decor—it's a clue
 about time."
```

### Step 4: Design Interaction Flow

```javascript
// Player interaction sequence:

const interactionFlow = {
  // Initial entry
  entry: {
    playerSees: 'Normal office',
    canInteractWith: ['desk', 'computer', 'door'],
    goal: 'Encourage exploration'
  },

  // After examining desk
  deskExamined: {
    reveal: 'More details visible',
    newInteractions: ['drawers', 'photos'],
    progressNarrative: true
  },

  // After reading computer
  computerRead: {
    reveal: 'Viewmaster appears',
    newInteractions: ['viewmaster'],
    progressNarrative: true,
    trigger: 'next_narrative_beat'
  },

  // After Viewmaster pickup
  viewmasterAcquired: {
    triggerEvent: 'office_hell_preview',
    prepareFor: 'transformation'
  }
};
```

---

## 🔧 Variations For Your Game

### Variation 1: Abandoned Office

```javascript
const abandonedOffice = {
  // More decay, less order
  atmosphere: 'neglected',

  visualChanges: {
    dust: 'thick_layer',
    debris: 'scattered',
    lighting: 'dim_flickering',
    objects: 'overturned'
  },

  // Different story
  narrative: 'Abandoned years ago, not recently'
};
```

### Variation 2: High-Tech Lab

```javascript
const techLab = {
  // Instead of office, a lab
  atmosphere: 'clinical',

  visualChanges: {
    equipment: 'scientific',
    lighting: 'harsh_fluorescent',
    colors: 'white_blue_stainless'
  },

  // Different interactions
  interactions: [
    'analyze_samples',
    'read_data_logs',
    'use_equipment'
  ]
};
```

### Variation 3: Living Space

```javascript
const livingSpace = {
  // Home instead of office
  atmosphere: 'intimate',

  visualChanges: {
    furniture: 'residential',
    personalItems: 'abundant',
    lighting: 'warm_lamps'
  },

  // Story is personal, not professional
  narrative: 'Someone lived here, personal story'
};
```

---

## Performance Considerations

```
OFFICE INTERIOR PERFORMANCE:

Splat Rendering:
├── Interior splats often higher detail
├── Close viewing distance = need quality
├── Consider LOD for far objects
└── Target: Stable 60 FPS

Lighting:
├── Area lights are expensive
├── Use baked lighting where possible
├── Limit shadow casting lights
├── Use light probes for indirect
└── Target: 2-3 dynamic lights max

Interactive Objects:
├── Use instancing for repeated objects
├── Combine small static objects
├── Optimized collision shapes
└── Target: Smooth interaction response

RECOMMENDATION:
Test on minimum spec hardware.
Interior spaces are where
performance issues often appear.
```

---

## Common Mistakes Beginners Make

### 1. Too Many Interactables

```javascript
// ❌ WRONG: 20+ things to examine
// Player gets exhausted, loses focus

// ✅ CORRECT: 5-7 meaningful objects
// Each tells part of story, feels rewarding
```

### 2. No Visual Hierarchy

```javascript
// ❌ WRONG: Everything equally visible
// Player doesn't know where to look

// ✅ CORRECT: Clear focal points
// Desk → Computer → Key objects
```

### 3. Information Dump

```javascript
// ❌ WRONG: Long documents, walls of text
// Player skips, misses important info

// ✅ CORRECT: Bite-sized, scannable
// Key info highlighted, optional details
```

### 4. No Feedback

```javascript
// ❌ WRONG: Examining object does nothing visible
// Player unsure if interaction worked

// ✅ CORRECT: Clear response
// Sound, animation, UI feedback
```

---

## Related Systems

- [Interactive Object System](../05-interactive-objects/interactive-object-system.md) - Object interaction
- [DialogManager](../05-media-systems/dialog-manager.md) - Text display
- [Viewmaster Scene](../08-interactive-objects/viewmaster-scene.md) - Key discovery
- [Office Hell Scene](./office-hell-scene.md) - Transformation scene

---

## Source File Reference

**Scene Data**:
- `content/SceneData.js` - Office zone configuration
- `content/DialogData.js` - Computer documents and text

**Managers**:
- `managers/OfficeManager.js` - Interior-specific logic
- `managers/InteractionManager.js` - Object interactions

**Assets**:
- `assets/splats/office.ply` - Interior splat
- `assets/audio/office_ambience.mp3` - HVAC hum

---

## 🧠 Creative Process Summary

**From Concept to Office Scene**:

```
1. NARRATIVE NEED
   "Player needs a safe space to discover clues"

2. SPACE DESIGN
   "Normal office, familiar layout"

3. ENVIRONMENTAL STORYTELLING
   "Objects that tell a story"

4. INTERACTION DESIGN
   "Examine → Read → Discover → Progress"

5. ATMOSPHERE LAYERING
   "Warm light, normal sounds, then subtle unease"

6. PROGRESSION
   "Each discovery leads to next,
    culminating in Viewmaster"

7. TRANSFORMATION PREP
   "Safe space becomes nightmare,
    making transformation more effective"
```

---

## References

- [Interior Design for Games](https://www.youtube.com/watch?v=Tfv-vG8J5R8) - Video essay
- [Environmental Storytelling](https://www.gamedeveloper.com/design/) - Article series
- [Writing for Games](https://www.youtube.com/watch?v=K7YrfjSkiDo) - Narrative design
- [Gaussian Splatting Indoors](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Tech reference

*Documentation last updated: January 12, 2026*
