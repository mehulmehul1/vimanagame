# Music Room Asset Breakdown
## Archive of Voices - Culture Chamber

**Chamber:** Culture / Music
**Editorial Pillar:** Culture, Music, Wisdom, Philosophy
**Color Signature:** Amber (Primary), Gold, Orange (Secondary)
**Mood:** Warm, cultural, resonant, ethereal
**Duration:** ~60-90 seconds of gameplay

---

## OVERVIEW

The Archive of Voices is Vimana's communication hub—a resonating chamber where cultures stay connected across distance. Players learn to harmonize with the ship through a duet-based harp interaction with jelly creatures.

### Philosophy
> "The ship doesn't test you. It teaches you. When you play wrong, it doesn't fail you. It just sings again, more slowly, more clearly, until you can join in harmony. This is a duet—a shared moment of music-making."

### Visual Direction
- **Bioluminescent water pool** as the floor
- **Resonating chamber walls** that respond to sound
- **Six-string harp mechanism** grown from the chamber (not manufactured)
- **Jelly creatures** that emerge as musical messengers
- **Vortex portal** that intensifies with harmony
- **Warm amber/cyan glow** for cultural atmosphere

---

## ASSET LIST

### 1. ENVIRONMENT: Chamber Splat (CULTURE-001)

**Status:** In-Progress (Redesign)
**Priority:** CRITICAL
**Type:** Splat Environment
**Assigned To:** Tuhinanshu
**Estimated:** 5 days

#### Description
The entire chamber environment captured as a Gaussian Splat. This is the base visual layer that all other assets sit within.

#### Visual Requirements
- **Scale:** Large circular chamber, ~15m diameter
- **Floor:** Water surface (separate shader, below)
- **Walls:** Organic, grown architecture—curved surfaces, no hard edges
- **Ceiling:** Open to above with soft ambient light
- **Architecture Style:** Cyber-organic—membrane-like surfaces, bioluminescent veins
- **Color Palette:** Amber walls, cyan accents, warm glow from center
- **Atmosphere:** Misty, ethereal, slightly underwater feeling

#### Reference Keywords
- Bioluminescent cave
- Organic architecture
- Grown spacecraft interior
- Amber glow chamber
- Resonating space

#### Capture Notes
```
Camera Positions to Capture:
1. Entry view (from hub connection)
2. Center view (harp position)
3. Vortex view (back of chamber)
4. Detail shots (wall textures, organic details)

Quality Variants Needed:
- Max: 8-10M splats (desktop GPU)
- Desktop: 5-8M splats
- Laptop: 2-5M splats
- Mobile: 1-2M splats
```

#### Files
- **Concepts:** `docs/concepts/culture/music-room/`
- **Splat:** `public/assets/splats/culture/music-room/chamber.splat`
- **Existing:** `public/assets/splats/music-room-prototype.splat` (for reference)

---

### 2. PROP: Six-String Harp Mechanism (CULTURE-002)

**Status:** Backlog
**Priority:** CRITICAL
**Type:** GLB Model
**Assigned To:** Tuhinanshu
**Estimated:** 4 days

#### Description
Interactive musical instrument—six strings that players pluck to create notes. The harp should look "grown" from the chamber, not manufactured.

#### Visual Requirements
- **Style:** Organic, grown from chamber walls/floor
- **Strings:** 6 strings, each with unique color glow
  - String 1 (C4): Cyan glow
  - String 2 (D4): Blue glow
  - String 3 (E4): Green glow
  - String 4 (F4): Yellow glow
  - String 5 (G4): Orange glow
  - String 6 (A4): Red glow
- **Frame:** Curved, membrane-like connection points
- **Material:** Translucent, bioluminescent when active
- **Position:** Center of chamber, over water pool
- **Height:** ~2m tall, player-accessible

#### Interaction Notes
```
String States:
- Idle: Faint glow, visible but subtle
- Hover: Brighter glow, indicates interactivity
- Plucked: Bright flash + ripple effect
- Teaching: Jelly-creature demonstration shows which string

Technical:
- Raycast detection for each string
- Visual feedback on hover/pluck
- Sound trigger on pluck
```

#### Reference Keywords
- Organic harp
- Grown instrument
- Bioluminescent strings
- Musical membrane
- Alien instrument

#### Modeling Specs
- **Poly Count:** ~5,000-10,000 triangles
- **Textures:** 1024x1024 for materials
- **Export:** GLB with proper scale (1 unit = 1 meter)
- **Orientation:** Y-up, facing +Z (player view)

#### Files
- **Concepts:** `docs/concepts/culture/harp/`
- **Model:** `public/assets/models/culture/harp-mechanism.glb`
- **Textures:** `public/assets/textures/culture/harp/`

---

### 3. PROP: Player Standing Platform (CULTURE-003)

**Status:** Backlog
**Priority:** HIGH
**Type:** GLB Model
**Assigned To:** Tuhinanshu
**Estimated:** 2 days

#### Description
Circular platform where the player stands during the harp duet. Detaches and rides to the vortex on completion.

#### Visual Requirements
- **Shape:** Circular, ~2m diameter
- **Style:** Matching chamber architecture—organic edge
- **Material:** Translucent surface, bioluminescent rim
- **Animation:** Detaches and floats to vortex (5 seconds)

#### Animation Notes
```
Platform Animation Sequence:
1. Static during gameplay
2. On completion: Gentle lift-off
3. Float toward vortex (lerp position)
4. Fade into vortex
5. Player transitions to white flash
```

#### Reference Keywords
- Floating platform
- Organic disc
- Translucent floor
- Bioluminescent edge

#### Modeling Specs
- **Poly Count:** ~500-1,000 triangles
- **Textures:** 512x512
- **Export:** GLB

#### Files
- **Concepts:** `docs/concepts/culture/platform/`
- **Model:** `public/assets/models/culture/standing-platform.glb`

---

### 4. DESIGN: Jelly Creature Reference (CULTURE-004)

**Status:** Backlog
**Priority:** HIGH
**Type:** Design Reference
**Assigned To:** Tuhinanshu
**Estimated:** 2 days

#### Description
Jelly creatures are the "musical messengers" that teach players the harp sequences. This task creates visual design references for their procedural generation.

#### Visual Requirements
- **Body:** Translucent, jelly-like sphere
- **Surface:** Subtle pulsing animation
- **Bioluminescence:** Glows when teaching
- **Size:** ~0.5m diameter
- **Movement:** Emerges from water, arcs through air, submerges

#### Color Per Note
| Note | Color | RGB | Hex |
|------|-------|-----|-----|
| C4 | Cyan | (0, 255, 255) | #00FFFF |
| D4 | Blue | (0, 150, 255) | #0096FF |
| E4 | Green | (0, 255, 150) | #00FF96 |
| F4 | Yellow | (255, 255, 0) | #FFFF00 |
| G4 | Orange | (255, 150, 0) | #FF9600 |
| A4 | Red | (255, 50, 50) | #FF3232 |

#### Animation States
```
1. Submerged: Visible under water surface
2. Emergence: Jump out with water splash
3. Teaching: Float near target string, pulse glow
4. Submersion: Return to water gently
```

#### Reference Keywords
- Jellyfish
- Bioluminescent creature
- Translucent sphere
- Organic messenger
- Underwater alien life

#### Files
- **Concepts:** `docs/concepts/culture/jellies/`
- **Reference:** `docs/references/culture/jelly-creatures.md`

---

### 5. PROP: Nautilus Shell Collectible (CULTURE-005)

**Status:** Backlog
**Priority:** MEDIUM
**Type:** GLB Model
**Assigned To:** Tuhinanshu
**Estimated:** 2 days

#### Description
Procedural nautilus spiral shell that appears after vortex completion. Player clicks to collect, it flies to the UI slot.

#### Visual Requirements
- **Shape:** Nautilus spiral (golden ratio)
- **Material:** Iridescent—shifts color with view angle
- **Size:** ~0.2m diameter
- **Animation:**
  - 3-second materialize animation (fade in + scale)
  - Bobbing idle motion
  - 1.5-second fly-to-UI on collect

#### Iridescence Colors
- Pink/Purple at angle
- Cyan/Green at another
- Gold/Amber head-on

#### Reference Keywords
- Nautilus shell
- Golden ratio spiral
- Iridescent material
- Bioluminescent shell

#### Modeling Specs
- **Poly Count:** ~2,000-5,000 triangles
- **Textures:** 512x512 iridescence map
- **Export:** GLB

#### Files
- **Concepts:** `docs/concepts/culture/shell/`
- **Model:** `public/assets/models/culture/nautilus-shell.glb`

---

### 6. SHADER: Bioluminescent Water (CULTURE-006)

**Status:** Review (Already Implemented)
**Priority:** CRITICAL
**Type:** GLSL Shader
**Assigned To:** Mehul
**Estimated:** Done, needs polish with new assets

#### Description
Water surface shader with bioluminescent ripples that respond to the six harp strings individually.

#### Current Implementation
- **Files:** `src/shaders/water-vertex.glsl`, `src/shaders/water-fragment.glsl`
- **Class:** `src/entities/WaterMaterial.ts`
- **Story:** Completed in 1.1

#### Visual Requirements for Redesign
- Match new chamber splat aesthetics
- Amber/cyan color scheme
- Ripples from each string position
- Bioluminescent intensity based on duet progress

#### Files
- **Shaders:** `src/shaders/water-*.glsl`
- **Material:** `src/entities/WaterMaterial.ts`

---

### 7. SHADER: Vortex Portal (CULTURE-007)

**Status:** Review (Already Implemented)
**Priority:** CRITICAL
**Type:** GLSL Shader + Particles
**Assigned To:** Mehul
**Estimated:** Done, needs polish with new assets

#### Description
SDF torus vortex with 2000 particles flowing in spiral pattern. Intensifies based on duet progress (0-1).

#### Current Implementation
- **Files:** `src/shaders/vortex-*.glsl`
- **Classes:** `src/entities/VortexMaterial.ts`, `src/entities/VortexParticles.ts`
- **Story:** Completed in 1.1

#### Visual Requirements for Redesign
- Position: (0, 0.5, 2) in chamber coordinates
- Scale: ~3m diameter torus
- Colors: Cyan → Purple → White gradient
- Intensity: Dormant (0.0) → Full (3.0 emissive)

#### Files
- **Shaders:** `src/shaders/vortex-*.glsl`
- **Classes:** `src/entities/Vortex*.ts`

---

### 8. SHADER: Jelly Creature (CULTURE-008)

**Status:** Review (Already Implemented)
**Priority:** HIGH
**Type:** GLSL Shader
**Assigned To:** Mehul
**Estimated:** Done, needs visual review with design

#### Description
Enhanced shader for jelly creatures with teaching state, bioluminescence, and organic pulse.

#### Current Implementation
- **Files:** `src/shaders/jelly-*.glsl`
- **Class:** `src/entities/JellyCreature.ts`
- **Story:** Completed in 1.2

#### Visual Requirements
- Translucent body with subsurface scattering
- Bioluminescent glow (color per note)
- Pulse animation rate
- Teaching state: brighter glow

#### Files
- **Shaders:** `src/shaders/jelly-*.glsl`
- **Class:** `src/entities/JellyCreature.ts`

---

### 9. VFX: Post-Processing (CULTURE-009)

**Status:** Backlog
**Priority:** MEDIUM
**Type:** VFX / Post-Processing
**Assigned To:** Tuhinanshu
**Estimated:** 1 day

#### Description
Chamber-specific post-processing effects for ethereal atmosphere.

#### Effects List
- **Bloom:** Soft glow on bioluminescent elements
- **Chromatic Aberration:** Subtle, for dreamlike feel
- **Vignette:** Focus attention on center
- **Color Grading:** Warm amber tint

#### Reference
- **Comparables:** Abzû, Journey ethereal scenes

---

### 10. VFX: Camera Presets (CULTURE-010)

**Status:** Backlog
**Priority:** LOW
**Type:** Camera Animation
**Assigned To:** Tuhinanshu
**Estimated:** 1 day

#### Description
Defined camera positions and transitions for the chamber.

#### Shots
1. **Entry:** Hub → Chamber transition
2. **Harp View:** Default gameplay view
3. **Vortex Reveal:** Platform ride camera
4. **White Flash:** Final transition

---

### 11. AUDIO: Harp String Samples (CULTURE-011)

**Status:** Backlog
**Priority:** CRITICAL
**Type:** Audio
**Assigned To:** TBD
**Estimated:** 2 days

#### Description
Six individual harp string pluck samples for notes C4, D4, E4, F4, G4, A4.

#### Requirements
- **Format:** MP3/OGG (web-ready)
- **Duration:** ~2 seconds each with natural decay
- **Quality:** Pure tones with organic resonance
- **No reverb:** Added in-engine

#### Files
- **Location:** `public/assets/audio/culture/harp-strings/`
- **Naming:** `c4.mp3`, `d4.mp3`, `e4.mp3`, `f4.mp3`, `g4.mp3`, `a4.mp3`

---

### 12. AUDIO: Chamber Ambience (CULTURE-012)

**Status:** Backlog
**Priority:** MEDIUM
**Type:** Audio
**Assigned To:** TBD
**Estimated:** 2 days

#### Description
Background ambience for the chamber—echoing whispers, harmonic resonance, water sounds.

#### Requirements
- **Format:** MP3/OGG, looping
- **Duration:** ~30 seconds loop
- **Elements:**
  - Subtle water movement
  - Distant echoing whispers
  - Harmonic resonance drone
  - Ship pulse/hum

#### Files
- **Location:** `public/assets/audio/culture/ambience.mp3`

---

### 13. AUDIO: Feedback Sounds (CULTURE-013)

**Status:** Backlog
**Priority:** MEDIUM
**Type:** Audio
**Assigned To:** TBD
**Estimated:** 1 day

#### Description
Sound effects for player feedback—correct note, wrong note, completion.

#### Sounds Needed
| Sound | Description | Duration |
|-------|-------------|----------|
| Discordant | Wrong note - minor second dissonance | 0.5s |
| Harmony | Correct note - player + perfect fifth | 1s |
| Completion | Sequence finish - C major chord | 2s |
| Reminder | Gentle re-teaching prompt (E4) | 1s |

#### Files
- **Location:** `public/assets/audio/culture/feedback/`

---

## PIPELINE SEQUENCE

### Phase 1: Foundation (Week 1)
1. **[CULTURE-001]** Chamber Splat Environment
2. **[CULTURE-004]** Jelly Creature Design Reference
3. **[CULTURE-002]** Harp Mechanism Model

### Phase 2: Props (Week 2)
4. **[CULTURE-003]** Standing Platform Model
5. **[CULTURE-005]** Nautilus Shell Model

### Phase 3: Polish (Week 3)
6. **[CULTURE-009]** Post-Processing Effects
7. **[CULTURE-010]** Camera Presets
8. Shader reviews and adjustments
9. Audio integration

### Phase 4: Integration (Week 4)
10. All assets integrated and tested
11. Performance validation
12. Cross-browser testing

---

## REFERENCE MOOD BOARD

### Visual Style
- **Architecture:** Grown, organic, membrane-like
- **Lighting:** Bioluminescent, warm amber glow
- **Materials:** Translucent, iridescent, alive
- **Colors:** Amber (#FFB000), Cyan (#00FFFF), Gold (#FFD700)

### Comparables
- **Games:** Abzû (underwater), Journey (spiritual), Outer Wilds (mystery)
- **Films:** The Abyss (bio-luminescence), Avatar (organic tech)
- **Nature:** Bioluminescent caves, deep sea creatures, aurora borealis

### Keywords for AI Image Generation
```
"bioluminescent organic chamber"
"grown spacecraft interior amber glow"
"cyber-organic architecture membrane"
"resonating music chamber ethereal"
"underwater cave with light"
"jellyfish bioluminescence"
"alien musical instrument"
```

---

## FILE STRUCTURE

```
vimana/
├── docs/
│   ├── concepts/
│   │   └── culture/
│   │       ├── music-room/
│   │       ├── harp/
│   │       ├── jellies/
│   │       ├── platform/
│   │       └── shell/
│   └── references/
│       └── culture/
│           └── jelly-creatures.md
│
├── public/
│   └── assets/
│       ├── splats/
│       │   └── culture/
│       │       └── music-room/
│       │           └── chamber.splat
│       ├── models/
│       │   └── culture/
│       │       ├── harp-mechanism.glb
│       │       ├── standing-platform.glb
│       │       └── nautilus-shell.glb
│       ├── textures/
│       │   └── culture/
│       │       ├── harp/
│       │       └── shell/
│       └── audio/
│           └── culture/
│               ├── harp-strings/
│               ├── ambience.mp3
│               └── feedback/
│
└── src/
    └── shaders/
        ├── water-*.glsl
        ├── vortex-*.glsl
        └── jelly-*.glsl
```

---

## CHECKLIST FOR TUHINANSHU

### Before Starting
- [ ] Read this entire document
- [ ] Review existing prototype in `public/assets/splats/music-room-prototype.splat`
- [ ] Set up reference board (Miro/Pinterest)
- [ ] Create folder structure under `docs/concepts/culture/`

### During Production
- [ ] Use `asset-template.md` to track each asset
- [ ] Update `asset-pipeline.yaml` as you progress
- [ ] Save concept iterations with clear filenames
- [ ] Test models in Three.js viewer before export

### Before Handoff
- [ ] All GLB files import without errors
- [ ] Poly counts within targets
- [ ] Textures optimized (power-of-2 dimensions)
- [ ] Files named consistently
- [ ] Update asset-pipeline.yaml to "review" status

---

## NEXT STEPS

1. **Today:** Create reference board for chamber redesign
2. **Tomorrow:** Start chamber concept sketches
3. **This Week:** Complete chamber splat capture
4. **Next Week:** Model harp mechanism

---

**Last Updated:** 2026-01-26
**Document Owner:** Mehul
**Assigned To:** Tuhinanshu
