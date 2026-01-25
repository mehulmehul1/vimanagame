---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
inputDocuments: ["game-brief.md"]
documentCounts:
  briefs: 1
  research: 0
  brainstorming: 0
  projectDocs: 0
workflowType: 'gdd'
lastStep: 14
project_name: 'shadowvimana'
user_name: 'Mehul'
date: '2026-01-15'
game_type: 'adventure'
game_name: 'VIMANA'
needs_narrative: true
---

# VIMANA - Game Design Document

**Author:** Mehul
**Game Type:** Adventure
**Target Platform(s):** Web Browser (WebGPU/WebGL2)
**Status:** Complete

---

## Executive Summary

### Game Name

VIMANA

### Core Concept

VIMANA is a contemplative first-person web exploration experience set inside a bio-organic Vimana spacecraft—a living vessel "grown" rather than built, currently dormant in the depths. The experience serves as the immersive launch platform for VIMANA, a new international magazine at the intersection of Art, Nature, Culture, and Technology.

Players inhabit the perspective of a boy from Kolkata who touched a mysterious orb and vanished. Awakening inside the sleeping Vimana, the player's interactions gradually *wake up the ship*—bioluminescence flickers to life, membranes pulse with renewed energy, the vessel itself responds to presence.

The core loop is one of awakening: four story elements (Art, Nature, Culture, Technology) must be engaged through interaction, each bringing the Vimana closer to full consciousness. This is not puzzle-solving or challenge-based—it's a ritual of activation, a symbiotic dance between player and vessel.

The experience lasts 5-10 minutes and follows an emotional arc from confusion through curiosity, awakening, achievement, transcendence, and finally initiation. When the ship is fully awake, the Voyage Gate opens and the ascension begins—Vimana flies through water → sky → space → void, sealing the player's subscription to the first issue.

Built on the Shadow Engine with Gaussian Splatting, VIMANA delivers photorealistic web graphics directly in the browser, embodying the magazine's ethos: **technology grown from nature, not built upon it.**

### Game Type

**Type:** Adventure
**Framework:** This GDD uses the Adventure template with type-specific sections for narrative structure, environmental storytelling, player immersion, and exploration design.

---

## Target Platform(s)

### Primary Platform

**Web Browser (WebGPU/WebGL2)** — VIMANA is a browser-first experience designed to reach the widest possible audience without requiring downloads or installations.

#### Performance Tiers

| Tier | Target Device | Splat Count | Bitrate | Expected FPS |
|------|---------------|-------------|---------|--------------|
| `max` | Desktop with discrete GPU | 8-10M | High | 60+ |
| `desktop` | Standard desktop | 5-8M | Medium | 45-60 |
| `laptop` | Integrated GPU laptop | 2-5M | Low-Medium | 30-45 |
| `mobile` | Mobile/tablet | 1-2M | Low | 30 |

#### Platform Considerations

**WebGPU Adoption:**
- Primary rendering path via WebGPU (Chrome 113+, Edge 113+)
- Graceful fallback to WebGL2 for Safari and older browsers
- Progressive enhancement approach — core experience works everywhere

**iOS-Specific Considerations:**
- Safari WebGL2 has stricter memory limits (~500MB GPU allocation)
- Reduced splat count required for stability
- Touch-only controls (no mouse hover)
- Potential need for iOS-specific asset quality settings

**Progressive Loading Strategy:**
1. Initial scene loads with base quality (1-2M splats)
2. Device capability detected on first frame
3. Quality scales up automatically if hardware supports it
4. Streaming allows higher quality splats to load asynchronously

**TypeGPU Integration:**
TypeGPU (`@typegpu/three`) will be used for shader authoring within the Three.js TSL (Node Material) system. This provides:
- **Type-safe shader programming** — Catch shader errors at compile time, not runtime
- **Unit-testable GPU code** — Test shader logic like any other TypeScript
- **JavaScript control flow** — Use familiar JS patterns for GPU operations
- **WebGPU-first design** — Native WebGPU with Three.js integration

### Control Scheme

**Primary Input:** Drag/Look Navigation
- Desktop: Click and drag to look around
- Mobile: Touch and drag to look around
- No joystick or WASD — pure environmental exploration

**Why This Control Scheme:**
- Accessible to non-gamers (no game knowledge required)
- Matches the contemplative, observational nature of the experience
- Works seamlessly across touch and mouse inputs
- Reduces motion sickness (player controls pace of movement)

---

## Target Audience

### Demographics

| Segment | Description |
|---------|-------------|
| **Age** | 18-40 (primary core 25-35) |
| **Geography** | International, urban centers |
| **Cultural Orientation** | Culturally curious, globally-minded |
| **Gaming Experience** | Casual to mid-core gaming familiarity |
| **Tech Comfort** | Comfortable with web experiences |

### Player Motivations

**Primary Motivations:**
1. **Discovery** — Players who enjoy exploring and uncovering mysteries
2. **Initiation** — Feeling of being part of something exclusive and new
3. **Taste Curation** — Alignment with VIMANA magazine's Art/Nature/Culture/Tech intersection

**Psychographic Profile:**
- Values aesthetic experiences over challenge-based gameplay
- Appreciates art, design, and thoughtful curation
- Drawn to solarpunk/cyber-organic aesthetics
- Seeks meaning in digital experiences
- Magazine readers, art patrons, culture enthusiasts

### Session Design

**Target Session Length:** 5-10 minutes
- Short enough to share casually via link
- Long enough to create emotional impact
- Complete experience in one sitting (no save/load)

**Sharing Intent:**
- Designed to be shareable ("You have to see this")
- Voyeuristic appeal — watching someone else experience it is also enjoyable
- Each playthrough reveals new details for replay value

---

## Goals and Context

### Project Goals

| Goal Type | Goal | Success Criteria |
|-----------|------|------------------|
| **Business** | Drive waitlist signups for VIMANA magazine launch | 500 signups before first issue release |
| **Personal** | Master end-to-end game development with Gaussian Splatting + WebGPU | Complete full production pipeline; reusable framework for future games |
| **Technical** | Ship stable experience across all target platforms | 60fps on desktop, 30fps mobile on 98%+ of devices; critical bug-free release |
| **Recognition** | Establish technical and artistic credibility in web experiences | Awwwards Site of the Day nomination/win |

### Background and Rationale

**Why VIMANA Matters Now:**

VIMANA represents the convergence of several technological and cultural moments:

- **Technological Inflection Point:** Gaussian Splatting (3DGS) and WebGPU have reached browser maturity, making photorealistic web experiences possible for the first time. Being early to this technology creates a first-mover advantage.

- **Magazine Reimagined:** Traditional media launches are increasingly digital-first. VIMANA the magazine needs a launch experience that embodies its ethos—technology grown from nature, not built upon it. The experience *is* the brand statement.

- **Market Gap:** Web experiences are either functional (SaaS marketing sites) or superficial (scrolling animations). There's a middle ground of *immersive* web experiences that deliver emotional impact without requiring app installation.

- **Capability Building:** This project serves as the foundation for a broader capability—delivering any type of web-based game experience. The Shadow Engine + Gaussian Splatting pipeline becomes reusable IP.

**Why This Form:**

A contemplative first-person exploration was chosen because it:
- Maximizes visual impact of Gaussian Splatting
- Minimizes mechanical complexity (focus on polish, not systems)
- Aligns with magazine brand (curated, aesthetic, discovery-oriented)
- Creates shareability essential for organic growth

---

## Unique Selling Points (USPs)

### 1. Photorealistic Web-First Experience
**What:** VIMANA delivers console-quality visuals through a browser, no download required.
**Why Different:** Most web "experiences" are stylized or abstract. Gaussian Splatting enables photorealism that rivals native applications, creating genuine surprise and delight.
**Defensible:** Requires specialized 3DGS capture pipeline and technical expertise not commonly found in web dev.

### 2. Bio-Interactive Awakening Mechanic
**What:** Players don't explore a static environment—they *wake up* a living spacecraft through their presence and attention. The ship responds biologically (bioluminescence, pulse, movement) to player engagement.
**Why Different:** Standard environmental storytelling is passive. VIMANA creates a two-way emotional relationship—player and ship co-awakening.
**Defensible:** Specific narrative-art-technology integration that's hard to replicate without unified vision.

### 3. Four-Pillar Editorial Integration
**What:** The four activation elements (Art, Nature, Culture, Technology) map directly to the magazine's editorial pillars—making the experience a literal embodiment of the brand.
**Why Different:** Most marketing experiences feel disconnected from the product. VIMANA *is* the magazine's mission in interactive form.
**Defensible:** Only works because this is a magazine launch; the brand-product-experience alignment is unique to this context.

### 4. Solarpunk Cyber-Organic Aesthetic
**What:** A visual language where technology appears grown rather than built—membranes, bioluminescence, organic curves, no hard edges.
**Why Different:** Cyberpunk dominates sci-fi aesthetics. Solarpunk/cyber-organic is underserved and increasingly resonant.
**Defensible:** Requires specific art direction and asset pipeline; difficult to copy without authentic commitment to the aesthetic.

### Competitive Positioning

VIMANA occupies an uncrowded space:

| Category | VIMANA | Traditional Web Experiences | VR/Console Games |
|----------|--------|----------------------------|------------------|
| Access | One link, any device | One link, any device | Hardware required |
| Visual Fidelity | Photorealistic (3DGS) | Stylized/WebGL | Photorealistic |
| Session Length | 5-10 min | 1-2 min | Hours |
| Emotional Depth | Contemplative narrative | Superficial | Deep |
| Technical Novelty | WebGPU + 3DGS | Established | Established |

**Positioning Statement:** VIMANA is the most visually impressive web experience you can open in a browser—bridging the gap between casual web browsing and immersive entertainment.

---

## Core Gameplay

### Game Pillars

| Pillar | Description | How It Manifests |
|--------|-------------|------------------|
| **Contemplative Awakening** | The experience is a ritual of awakening, not a challenge to overcome. Player and ship co-awaken through presence and attention. | No puzzles, no enemies, no failure states. Pacing allows for observation. The ship responds to player presence biologically. |
| **Environmental Discovery** | Story is told through discovered details—the player learns by looking, finding, and interacting. Each interaction fills in understanding of both the Vimana world and the magazine. | Story fragments scattered throughout: artifacts, visual cues, audio diaries, ship responses. No paragraphs to read—look and understand. |
| **Dual Narrative Layer** | The experience simultaneously tells the Vimana story (ship, crew, origin) AND introduces the magazine (mission, content, team). Both narratives intertwine—understanding one illuminates the other. | The ship's history mirrors the magazine's mission. Crew stories reflect editorial pillars. By the end, player knows both worlds. |
| **Earned Transcendence** | The climax (Voyage Gate, ascension) must be earned through complete engagement, not granted. | Gate only opens after all four elements activated. The sequence from water → sky → space → void seals the initiation—player is now "part of Vimana." |

**Pillar Prioritization:** When pillars conflict, prioritize in this order:
1. **Contemplative Awakening** — Never sacrifice the core emotional experience
2. **Environmental Discovery** — Understanding emerges from interaction, not exposition
3. **Dual Narrative Layer** — Both stories (ship + magazine) should be clear by end
4. **Earned Transcendence** — The climax must feel deserved

### Core Gameplay Loop

**Loop Diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     DISCOVERY & AWAKENING LOOP                  │
└─────────────────────────────────────────────────────────────────┘

     Observe                    Discover                    Awaken
        ▼                          ▼                          ▼
   ┌─────────┐              ┌─────────┐              ┌─────────┐
   │ Look    │──────────────▶│ Find    │──────────────▶│ Ship    │
   │ around  │   Curiosity   │ fragment│   Interaction  │responds │
   │         │◀──────────────│(lore)   │◀──────────────│ + Story │
   └─────────┘   Discovery   └─────────┘   Insight     │unfolds  │
        │                          │              └─────────┘
        └──────────────────────────┼──────────────────────┘
                                   │
                                   ▼
                           ┌─────────────┐
                           │ World +     │
                           │ Brand under-│
                           │ standing    │
                           │ grows       │
                           └─────────────┘
                                   │
                                   ▼
                           Repeat (4 elements + discoveries)
```

**Core Loop Steps:**

1. **Observe** — Player looks around the darkened Vimana interior. Subtle cues suggest something more: an object, a symbol, a faint sound. Curiosity draws them forward.

2. **Discover** — Player focuses on/finds a narrative fragment along their path:
   - **Physical artifact** — Crew personal item, instrument, symbol
   - **Visual cue** — Projection, mural, interface-like surface
   - **Audio echo** — Voice, sound, music from the ship's memory
   - **Ship response** — The Vimana itself communicates through pulse, light, movement

3. **Understand** — Through the discovered fragment, player learns:
   - **Vimana World:** Ship's origin, crew members, their mission, why dormant
   - **Magazine Reality:** Editorial pillars, content types, team behind it, what to expect

4. **Awaken** — Each discovery + element activation strengthens the ship's awakening. More areas become accessible. The narrative builds toward understanding.

**Loop Timing:** Each discovery cycle takes 1-2 minutes. 4 main elements + scattered fragments = 5-10 minute total experience.

**Dual Narrative Examples:**

| Fiction Layer (Vimana Ship) | Magazine Layer (Vimana Brand) |
|----------------------------|-------------------------------|
| Ship was grown, not built | Magazine as organic intersection of disciplines |
| Crew was multidisciplinary — artists, scientists, philosophers | Vimana curates across Art, Nature, Culture, Technology |
| Ship went dormant after "great forgetting" | Magazine responds to era of fragmentation |
| Awakening = reconnection with purpose | Subscription = joining the mission |
| Voyage Gate leads to next chapter | First issue is the beginning |

**By End of Experience, Player Should Know:**

**Fictional Understanding:**
- What a Vimana is (bio-organic spacecraft, grown not built)
- Why this one is dormant (awaiting the right "pilot")
- Who the crew was (their specialties, personalities, glimpsed through artifacts)
- Where it's going (the voyage — what comes after awakening)

**Brand Understanding:**
- What Vimana magazine is (Art × Nature × Culture × Technology)
- Who creates it (the team — glimpsed through crew analogies)
- Why it exists now (response to fragmentation, hunger for integration)
- What to expect from first issue (the four pillars in action)

### Win/Loss Conditions

#### Victory Conditions

VIMANA has no traditional "win state"—success is completing the understanding journey:

| Milestone | Condition |
|-----------|-----------|
| **Narrative Fragments Found** | Player has discovered enough story fragments to understand both the Vimana world and the magazine mission |
| **All Four Elements Activated** | Art, Nature, Culture, Technology elements awakened |
| **Voyage Gate Opens** | Ship fully awake → Gate materializes |
| **Ascension & Initiation** | Player enters Gate → water→sky→space→void → subscription seal (now "part of Vimana") |

#### Failure Conditions

**No Failure States** — VIMANA is designed as an experience without failure:

- No time limits
- No way to "lose" or miss critical information permanently
- No game over
- Players can linger, re-explore, replay for new details

**Rationale:** Aligns with **Contemplative Awakening** pillar. The goal is understanding and emotional resonance, not challenge.

---

## Game Mechanics

### Primary Mechanics

VIMANA's mechanics are intentionally minimal—focusing on presence and observation rather than challenge or skill.

| Mechanic | Verb | What It Does | Serves Pillar |
|----------|------|--------------|---------------|
| **Gaze to Awaken** | Look | Holding view on interactable elements gradually activates them—like warming up rather than flipping a switch | Contemplative Awakening |
| **Observe & Discover** | Notice | Subtle visual/audio cues draw attention to narrative fragments throughout the ship | Environmental Discovery |
| **Ship Response** | Feel | The Vimana itself responds biologically to player presence—bioluminescence pulse, membrane movement, sound | Dual Narrative Layer |
| **Progressive Awakening** | Awaken | Each discovered element strengthens the ship's overall awakening; world becomes more vibrant and vocal | Earned Transcendence |

### Mechanic Details

**1. Gaze to Awaken (Primary Interaction)**

The core interaction—inspired by the Shadow Engine's Viewmaster Controller but reimagined for awakening instead of horror:

- **When:** Whenever player centers view on a story element or narrative fragment
- **How it works:**
  - Raycast detects gaze on interactable
  - Gaze duration builds "awakening intensity" (0-1)
  - No click or tap required—pure gaze-based interaction
  - Works identically on desktop (mouse look) and mobile (touch drag)
- **Progressive response:**
  - 0.0-0.3: **Noticed** — Element faintly glows, acknowledges presence
  - 0.3-0.6: **Awakening** — Pulse intensifies, sound emerges, information begins to reveal
  - 0.6-1.0: **Awakening** — Full brightness, narrative fragment plays, ship responds
  - 1.0: **Complete** — Element fully activated, contributes to global awakening
- **Feel:** Gradual, organic—like something waking up, not a machine turning on
- **Technical basis:** Adapted from Shadow Engine's `ViewmasterGazeTracker` with positive reinforcement

**2. Observe & Discover (Narrative)**

Story fragments are scattered throughout the Vimana, discoverable through looking:

- **Physical artifacts:** Crew items, instruments, symbols (captured as 3D splats)
- **Visual projections:** Membrane surfaces displaying imagery/data
- **Audio echoes:** Voice fragments, music, ship sounds (spatial 3D audio)
- **Ship memories:** Environmental sequences triggered by proximity

**3. Ship Response (Feedback Loop)**

The Vimana is alive—constantly responding to player presence:

- **Bioluminescence:** Global glow that intensifies with awakening level
- **Membrane pulse:** Subtle breathing movement throughout the ship
- **Audio response:** Ship hums, resonates, responds to gaze events
- **Progressive visibility:** As ship awakens, previously hidden areas reveal themselves

**4. Progressive Awakening (Meta-Progression)**

Global tracking of journey completion:

```javascript
// GameState tracking
awakeningLevel: 0.0-1.0        // Overall ship consciousness
elementsActivated: 0-4         // Art, Nature, Culture, Technology
currentElement: string         // Currently focused element
fragmentsDiscovered: number    // Narrative pieces found
voyageGateOpen: boolean        // True when all 4 elements complete
```

### Mechanic Interactions

```
Look Around → Notice Element → Gaze to Awaken
                    ↓
            Narrative Fragment Revealed
                    ↓
            Ship Responds (glow, sound, pulse)
                    ↓
            Awakening Level Increases
                    ↓
            More of Ship Becomes Visible
                    ↓
            Repeat until all 4 elements complete
                    ↓
            Voyage Gate Opens → Ascension Sequence
```

### Mechanic Progression

| Awakening Level | Ship State | What's Visible |
|-----------------|------------|----------------|
| 0.0-0.25 | Dormant | Dark, minimal bioluminescence, subtle cues |
| 0.25-0.5 | Stirring | First elements active, moderate glow, audio emerging |
| 0.5-0.75 | Waking | Multiple elements active, ship clearly responding |
| 0.75-1.0 | Awake | All elements active, Voyage Gate materializing |
| 1.0 | Ascended | Gate open, water→sky→space→void sequence begins |

---

## Controls and Input

### Control Scheme (Web Browser)

**Desktop (Mouse):**
| Action | Input | Behavior |
|--------|-------|----------|
| Look Around | Click + Drag | Camera rotates following mouse movement |
| Activate Element | Hold center view | Automatic gaze-based activation (2-3 sec) |
| Restart | R key | Reset to beginning |

**Mobile/Tablet (Touch):**
| Action | Input | Behavior |
|--------|-------|----------|
| Look Around | Touch + Drag | Camera rotates following finger movement |
| Activate Element | Hold center view | Automatic gaze-based activation |
| Restart | Pull-to-refresh | Reset to beginning |

### Input Feel

- **Damping:** Smooth camera with inertia—like floating in liquid, not air
- **Sensitivity:** Configurable, defaults to "slow and contemplative"
- **No snap transitions:** All movement continuous and fluid
- **Gaze tolerance:** Forgiving aim—generous hitboxes on interactables

### Accessibility Controls

| Feature | Implementation |
|---------|----------------|
| **Reduced Motion** | Disable camera drift and inertial effects |
| **Sensitivity Slider** | User-adjustable look speed |
| **Colorblind Modes** | Alternative palettes for bioluminescence |
| **Subtitles** | Optional text for any voice/audio elements |
| **One-Handed Mode** | Full experience playable with thumb-only touch |
| **Screen Reader** | ARIA labels on UI overlays (subscription form) |

---

## Cinematic Ascension Sequence

### The Final Sequence

When all four elements are activated and the Voyage Gate opens:

**Phase 1: The Gate Manifests**
- Bioluminescent energy coalesces at ship's forward area
- Portal forms—liquid light membrane
- Audio: Deep resonant tone, like a cosmic chord

**Phase 2: Water → Sky**
- Vimana begins to rise
- Visuals: Bubbles, water distortion, light refraction
- Breaking surface: Dramatic light burst, water droplets
- Audio: Underwater muffled → sharp clarity

**Phase 3: Sky → Space**
- Ascending through clouds
- Sky darkens, stars emerge
- Vimana accelerates upward
- Audio: Wind fades into silence

**Phase 4: Space → Void**
- Stars streak past (warp effect)
- Colors shift, reality bends
- Entering the void beyond space
- Audio: Cosmic tone, then silence

**Phase 5: Initiation**
- Void resolves to VIMANA magazine preview
- Subscription form appears
- Player is now "part of Vimana"
- First issue access granted

**Technical Implementation:**
- Camera animation system (from Shadow Engine's animation pipeline)
- Shader effects for each phase (water, clouds, stars, void)
- MusicManager crossfade through ascension soundtrack
- WebGPU post-processing for visual effects
- Duration: ~60-90 seconds
- Skippable: ESC key or tap to skip to subscription form

---

## Adventure Specific Design

### Exploration Mechanics

**World Structure: Hub-Based with Non-Linear Chambers**

VIMANA uses a central hub (the Vimana main chamber) that connects to four themed chambers. Players can explore chambers in any order, with the Voyage Gate at the center serving as the final unlockable area.

| Chamber | Editorial Pillar | Aesthetic Signature | Key Narrative Beat |
|---------|------------------|---------------------|-------------------|
| **Gallery of Forms** | Art | Floating crystalline sculptures, light-refracting membranes | Beauty as memory—what the Vimana's crew created |
| **Hydroponic Memory** | Nature | Living fungal networks, bioluminescent flora | Life as interconnected systems—the crew's stewardship |
| **Archive of Voices** | Culture | Bioluminescent water pool, six-string harp, vortex portal, jelly teachers | The ship teaches you its song—a duet, not a test |
| **Engine of Growth** | Technology | Pulsing bio-mechanical systems, growth vats | Technology grown from biology—the crew's innovation |

**Movement and Traversal:**
- First-person drag/look navigation only
- No player character movement through space—the Vimana moves around the player
- Camera transitions guide attention between chambers
- Organic camera paths hint without forcing

**Observation and Inspection:**
- Gaze-based interaction (holding center view on elements)
- No clicking or tapping required
- Discovery happens through looking, not hunting
- Generous hitboxes on interactables (60° cone of detection)

**Discovery Rewards:**
- **Narrative fragments:** Each chamber reveals 2-3 story pieces about crew/purpose
- **Visual expansion:** As ship awakens, hidden areas become visible
- **Audio progression:** New sound layers unlock with each chamber
- **Progress feedback:** Global bioluminescence intensifies with progress

**Pacing of Exploration:**
- No time pressure
- Players can linger in any chamber
- Gentle audiovisual cues suggest progression without forcing
- Average chamber exploration: 60-90 seconds

### Story Integration

**Story Delivery Methods:**

| Method | Usage | Examples |
|--------|-------|----------|
| **Environmental** | Primary | Architecture, props, visual details tell story |
| **Audio** | Secondary | Ambient sounds, musical themes, spatial audio echoes |
| **Visual Cues** | Guidance | Bioluminescence guides attention, light reveals narrative |
| **No Text/Dialogue** | Intentional | Story emerges through observation, not reading |

**Player Agency in Story:**
- **Linear narrative backbone:** Prologue → awakening → exploration → ascension always occurs
- **Non-linear discovery:** Chambers can be visited in any order
- **No branching outcomes:** Story always ends with subscription/initiation
- **Agency through pacing:** Player controls how quickly they discover and understand

**Story Pacing:**
```
Confusion (0:00-0:45)   - Prologue, awakening in darkness
Curiosity (0:45-2:00)   - First chamber discovery
Discovery (2:00-5:00)   - Exploring remaining chambers
Achievement (5:00-6:00) - All elements activated, Gate opens
Transcendence (6:00-7:00) - Ascension sequence
Initiation (7:00-8:00)   - Subscription seal
```

**Character Introduction:**
- No direct character encounters
- Crew revealed through artifacts and environmental storytelling
- Each chamber features a different crew member's specialty
- Player understands crew by the end, even without meeting them

**Climax Structure:**
- **Buildup:** Four chambers awakened → Voyage Gate manifests
- **Release:** Ascension sequence (water → sky → space → void)
- **Resolution:** Subscription form + "You are part of VIMANA now"

### Environmental Storytelling

**Visual Storytelling Techniques:**

| Technique | Application |
|-----------|-------------|
| **Lighting** | Bioluminescence intensity shows awakening state |
| **Color** | Each chamber has unique color signature (cyan, green, amber, violet) |
| **Scale** | Cathedral-like interiors evoke awe, intimate corners create discovery |
| **Movement** | Membrane pulse shows ship is alive, not static |
| **Audio** | Spatial positioning tells story through sound placement |

**Audio Atmosphere:**
- **No music initially** — only ambient ship sounds (hum, pulse, distant creaks)
- **Chamber themes** — each area has distinct audio signature
- **Progressive layering** — new audio elements added as ship awakens
- **Ascension score** — music emerges only during final sequence

**Readable Documents:**
- No text to read — narrative is visual/auditory only
- Story conveyed through:
  - Projected imagery on membrane surfaces
  - Physical artifacts (crew items, instruments)
  - Audio echoes (voice fragments, music clips)
  - Environmental details (growth patterns, wear marks)

**Environmental Clues:**
- **Dormant state** — darkness, minimal bioluminescence, stillness
- **Awakening** — light spreads, motion increases, sound emerges
- **History** — worn surfaces, personal items, evidence of past habitation
- **Purpose** — ship architecture reveals its function as a vessel of growth

**Show vs. Tell Balance:**
- **90% show, 10% tell** — nearly everything is shown, not told
- **No exposition** — no character explains anything directly
- **Ambiguity embraced** — some details left open to interpretation
- **Emotional truth over facts** — feeling matters more than literal explanation

### Puzzle Systems

**No Puzzles**

VIMANA is intentionally puzzle-free. The experience is about:
- **Observation, not problem-solving**
- **Presence, not challenge**
- **Discovery, not conquest**

**What Exists Instead:**
- Gaze-based activation (holding view on elements)
- Progressive unlocking (chambers reveal as ship awakens)
- Environmental guidance (subtle cues direct attention)
- Completion through engagement (all four chambers must be visited)

**Rationale:**
- Puzzles would interrupt contemplative flow
- Challenge-based gameplay conflicts with "awakening" theme
- Target audience may not be gamers
- Magazine launch should feel effortless, not frustrating

### Character Interaction

**No Direct NPC Interaction**

VIMANA has no living characters to interact with. The "characters" are:
- **The Vimana itself** — a living spacecraft that responds to presence
- **The absent crew** — revealed through artifacts and environmental storytelling

**Ship as Character:**
- **Responsive:** Bioluminescence, pulse, sound react to player
- **Alive:** Membrane movement, breathing rhythms, organic behaviors
- **Communicative:** Uses light and sound to guide and respond
- **Evolutionary:** Transforms from dormant to awakened through player interaction

**Crew Presence:**
- **Artifacts:** Personal items scattered throughout chambers
- **Audio echoes:** Voice fragments, music, recorded thoughts
- **Visual projections:** Membrane surfaces display memories/creations
- **Environmental details:** Living spaces show who was there

**No Dialogue System:**
- No spoken or written dialogue
- Story conveyed through environmental observation
- Audio elements are ambient/expressive, not conversational
- Player understanding emerges through synthesis, not direct communication

### Inventory and Items

**No Inventory System**

VIMANA has no inventory, collectibles, or items to carry. This is intentional:
- **No mechanical complexity** — focus on observation, not management
- **No collection pressure** — experience isn't about "getting everything"
- **No inventory UI** — clean, immersive visual experience
- **No replay requirement** — single complete narrative arc

**What Players "Collect":**
- **Understanding:** Narrative fragments build comprehension
- **Awakening:** Each chamber visited strengthens ship consciousness
- **Emotional resonance:** Moments of beauty, wonder, discovery
- **Brand connection:** Understanding of VIMANA magazine's mission

**Progression Gates:**
- **Voyage Gate:** Only opens after all four chambers visited
- **Ascension sequence:** Only triggers after Gate entered
- **Subscription form:** Only appears after ascension complete

---

## Progression and Balance

### Player Progression

**Progression Type: Narrative & Environmental**

VIMANA uses narrative progression rather than power/skill progression. The "growth" is:
- **Understanding:** Player learns more about Vimana world and magazine
- **Awakening:** Ship becomes more active and responsive
- **Revelation:** Hidden areas and details become visible
- **Emotional:** Journey from confusion to transcendence

**Progression Types Active:**

| Type | Implementation | Feel |
|------|----------------|------|
| **Narrative** | Story unfolds through chamber exploration | Discovering a mystery |
| **Content** | New chambers and areas unlock | Opening up the world |
| **Collection** | Story fragments and discoveries build understanding | Piecing together meaning |

**Progression Pacing:**

| Phase | Duration | What Unlocks |
|-------|----------|--------------|
| **Prologue** | 0:00-1:00 | Initial context, player as disappeared boy |
| **First Chamber** | 1:00-2:30 | Ship begins responding, first narrative fragments |
| **Mid-Exploration** | 2:30-5:00 | Multiple chambers active, ship clearly alive |
| **Final Chamber** | 5:00-6:00 | All elements active, Voyage Gate manifests |
| **Ascension** | 6:00-7:30 | Full ship activation, voyage begins |

**No Meta-Progression:**
- No items or stats to carry between sessions
- No skill development (player doesn't "get better")
- Each playthrough is self-contained
- Replay value comes from noticing new details, not progression

### Difficulty Curve

**Flat Difficulty — No Challenge Scaling**

VIMANA has no traditional difficulty. There is no:
- Skill testing
- Fail states
- Time limits
- Performance requirements
- Obstacles to overcome

**Difficulty Philosophy:**
- **Accessibility over challenge** — anyone should be able to complete
- **Experience over gameplay** — focus on being there, not beating it
- **Emotional resonance, not frustration** — feelings come from wonder, not victory

**Player-Side Challenge Factors:**

| Factor | Mitigation |
|--------|------------|
| **Navigation** | Gaze-based interaction, generous hitboxes, no way to get lost |
| **Understanding** | Environmental storytelling, multiple narrative layers, no required reading |
| **Technical** | Performance tiers, progressive loading, graceful degradation |
| **Accessibility** | Reduced motion options, colorblind modes, subtitle support |

**Challenge Sources Intentionally Absent:**
- No enemies or threats
- No time pressure
- No resource management
- No puzzles or riddles
- No skill gates

### Economy and Resources

**No Economy or Resource System**

VIMANA has no in-game economy, currency, or resources. This is by design:
- **Experience is not transactional** — no buying, selling, or trading
- **No accumulation** — focus on presence, not collection
- **No grinding** — single continuous narrative flow
- **No premium mechanics** — magazine subscription is real-world, not in-game

**What Players "Earn":**
- **Understanding:** Knowledge of Vimana world and magazine
- **Emotional experience:** Wonder, curiosity, transcendence
- **Brand connection:** Relationship with VIMANA magazine
- **Social currency:** Having seen something shareable

**Subscription Model:**
- **Not in-game purchase** — real-world magazine subscription
- **Earned access** — only presented after completing the experience
- **Initiation framing** — "joining" Vimana, not "buying" access
- **No in-game gating** — full experience available before subscription ask

---

## Level Design Framework

### Structure Type

**Hub-Based with Non-Linear Chambers**

VIMANA uses a central hub chamber that connects to four themed chambers. Players can explore chambers in any order, with the Voyage Gate at the center serving as the final unlockable area.

**Structure Advantages:**
- **Non-linear exploration:** Players choose chamber order
- **Clear progress tracking:** Visual indication of chambers visited
- **Central return point:** Hub provides orientation between chambers
- **Final unlock location:** Voyage Gate at center creates satisfying conclusion

### Level Types

| Chamber | Pillar | Aesthetic | Key Elements | Approx. Duration |
|---------|--------|-----------|--------------|------------------|
| **Prologue** | Narrative | Dark water, emerging orb | Kolkata boy disappearance | 45 sec |
| **Main Hub** | Central | Vimana interior, dormant ship | Entry, navigation, Voyage Gate | 30-60 sec |
| **Gallery of Forms** | Art | Crystalline sculptures, light refraction | Visual beauty, crew creations | 60-90 sec |
| **Hydroponic Memory** | Nature | Fungal networks, bioluminescent flora | Life systems, growth | 60-90 sec |
| **Archive of Voices** | Culture | Resonating chambers, artifacts | Wisdom, philosophy | 60-90 sec |
| **Engine of Growth** | Technology | Bio-mechanical systems, growth vats | Innovation, ship systems | 60-90 sec |
| **Ascension Space** | Finale | Portal, environmental transitions | Water→sky→space→void | 60-90 sec |

### Tutorial Integration

**No Traditional Tutorial**

VIMANA requires no explicit tutorial because:
- **Single interaction:** Gaze-to-awaken is intuitive
- **No complex systems:** No UI, inventory, or controls to learn
- **Visual guidance:** Bioluminescence draws attention to interactables
- **Forgiving interaction:** Generous hitboxes, no penalty for looking

**Onboarding Through Experience:**

| Moment | Teaching Method | What Player Learns |
|--------|-----------------|-------------------|
| **Opening darkness** | Restriction then reveal | Looking reveals content |
| **First glowing element** | Visual cue + response | Gaze activates the world |
| **First chamber entry** | Environmental guidance | Chambers contain narrative |
| **Progressive ship response** | Feedback loop | Presence matters |

**Accessibility Considerations:**
- **No prior gaming knowledge required**
- **No memorization of controls**
- **No reading necessary**
- **No time pressure to learn**

### Special Levels

**The Ascension Sequence (Cinematic Finale)**

Unlike other chambers, the ascension is a scripted cinematic sequence:
- **No player control** — passive viewing experience
- **Linear progression** — water → sky → space → void
- **Skippable** — ESC or tap to jump to subscription
- **Emotional climax** — music, visuals, motion build to transcendence

**Voyage Gate (Unlockable Hub)**

- **Initially sealed** — inaccessible until all four chambers visited
- **Visual transformation** — materializes progressively as chambers complete
- **Final trigger** — entering Gate launches ascension sequence

### Level Progression

**Unlock System: Chamber-Based Progress**

| Progress State | Unlocked Areas | Ship State | Visual Feedback |
|----------------|----------------|------------|-----------------|
| **0 Chambers** | Hub only | Dormant | Dark, minimal glow |
| **1 Chamber** | Hub + 1 chamber | Stirring | First pulse, faint glow |
| **2 Chambers** | Hub + 2 chambers | Waking | Moderate light, audio emerging |
| **3 Chambers** | Hub + 3 chambers | Nearly awake | Bright, responsive |
| **4 Chambers** | Hub + all chambers + Gate | Awake | Full bioluminescence, Gate open |
| **Gate Entered** | Ascension sequence | Ascended | Portal active, journey begins |

**Non-Linear Chamber Order:**
- Players can visit chambers in any order
- Each chamber is self-contained narratively
- Revisiting chambers allowed for discovery
- No dependencies between chambers

**Replayability:**
- **New details on replay:** Environmental storytelling has depth
- **Different chamber orders:** Varies emotional pacing
- **Shared experience:** Watching others play is enjoyable
- **No completion requirement:** Experience works even without finishing

**Save/Load:**
- **No save system** — experience designed for single sitting
- **Short duration (5-10 min)** — no break needed
- **Restart available** — R key or pull-to-refresh
- **State persistence** — browser refresh continues if closed mid-session

---

## Art and Audio Direction

### Art Style

**Solarpunk Cyber-Organic Aesthetic**

VIMANA's visual language embodies "technology grown from nature, not built upon it." This is achieved through:

**Core Visual Principles:**

| Principle | Implementation | Examples |
|-----------|----------------|----------|
| **No hard edges** | Everything curved, organic, flowing | Membranes, tendrils, grown surfaces |
| **Biological materials** | Grown, not manufactured | Fungal, coral-like, crystalline structures |
| **Internal light** | Bioluminescence as primary illumination | Glowing from within, no external light sources |
| **Living movement** | Subtle pulse, breathe, respond | Membrane undulation, light pulse, organic motion |

**Color Palette:**

| Color Type | Hex/Description | Usage | Emotional Association |
|------------|-----------------|-------|----------------------|
| **Deep Ocean Navy** | #0a1628, #0f2744 | Primary darkness, void spaces, shadows | Mystery, depth, unknown |
| **Spectral Cyan** | #00d4ff, #4ff0ff | Bioluminescence, energy flows, active elements | Wonder, technology, life |
| **Bioluminescent Accents** | #7fff7f, #ff7fff | Organic highlights, membrane glow | Vitality, transformation |
| **Void White** | #f8f8f8 → #ffffff | Ascension climax, fade to white | Transcendence, purity |

**Chamber Color Signatures:**

| Chamber | Primary Color | Secondary Colors | Mood |
|---------|--------------|------------------|------|
| **Gallery of Forms** | Cyan | White, silver | Aesthetic, cerebral |
| **Hydroponic Memory** | Green | Emerald, lime | Living, growing |
| **Archive of Voices** | Amber | Gold, orange | Warm, cultural |
| **Engine of Growth** | Violet | Purple, indigo | Technological, innovative |

**Visual Reference Points:**

- **Deep-sea biology:** Anglerfish bioluminescence, jellyfish transparency
- **Fungal networks:** Mycelial threads, mushroom gills, organic growth
- **Vedic architecture:** Palace geometry reimagined biologically
- **Solarpunk visions:** Technology in harmony with nature

**Lighting Design:**
- **Primary:** Bioluminescence emanates from surfaces themselves
- **Directional:** No visible light sources; light *is* the material
- **Dynamic:** Pulse, breathe, respond to player presence
- **Progressive:** Intensity increases with awakening level

**Camera and Perspective:**
- **Static FPS:** No camera shake or artificial movement
- **Gentle引导:** Organic camera paths hint without forcing
- **Vista moments:** Deliberate pauses for visual awe
- **Ascension:** Smooth, continuous motion through environments

**Photorealistic Rendering:**
- **Gaussian Splatting:** Primary rendering technique for environments
- **AI-Generated Assets:** Splat scenes via Marble Labs or equivalent
- **Hybrid Approach:** Traditional 3D for UI elements, interactables
- **Performance Tiers:** Quality variants for different device capabilities

### Audio and Music

**Audio Philosophy: Sound as Environment**

Audio is not a layer *on top* of the experience—it *is* the environment. Players should feel immersed in a living acoustic space.

**Audio Pillars:**

| Pillar | Description | Implementation |
|--------|-------------|----------------|
| **Presence** | The Vimana feels alive through ambient sound | Continuous ship hum, membrane stretch, biological sounds |
| **Guidance** | Audio cues draw attention without being directive | Spatial positioning of interactive elements |
| **Emotion** | Each chamber has emotional signature through soundscape | Distinct audio palette per chamber |
| **Climax** | Ascension sequence has dedicated musical score | Orchestral/electronic hybrid builds to transcendence |

**Sound Design Categories:**

| Category | Examples | Technical Approach |
|----------|----------|-------------------|
| **Ambient Chambers** | Hum, resonance, organic motion | Each chamber unique to its pillar |
| **Biological Sounds** | Breathing, pulse, membrane stretch | Layered, responsive to awakening level |
| **Interaction Feedback** | Subtle confirmations, state changes | Spatial audio positioned at interactable |
| **Ascension Score** | Orchestral/electronic hybrid | Builds to transcendent climax |
| **Silence** | Strategic pauses for impact | Not all moments need sound |

**Spatial Audio Implementation:**

- **Howler.js** for 3D spatial positioning
- **Reverb zones:** Each chamber has unique acoustic signature
- **Distance attenuation:** Sounds evolve as player explores
- **Crossfading:** Smooth transitions between zones
- **Awakening response:** Audio intensifies with ship consciousness

**Chamber Soundscapes:**

| Chamber | Audio Signature | Key Elements |
|---------|----------------|--------------|
| **Gallery of Forms** | Crystal chimes, pure tones | Bell-like, resonant, clear |
| **Hydroponic Memory** | Forest ambience, organic growth | Rustling, dripping, living |
| **Archive of Voices** | Echoing whispers, harmonic resonance | Human elements, layered voices |
| **Engine of Growth** | Mechanical pulse, energy flow | Rhythmic, powerful, ascending |
| **Ascension** | Full orchestral/electronic score | Building, transcendent, emotional |

**Voice Treatment:**

- **Optional narration:** Prologue/ascension may have brief voiceover
- **Style:** Documentary or poetic reading (not character acting)
- **Language:** English primary, potential for future localization
- **Subtitles:** Optional text display for accessibility

**Music Strategy:**

| Phase | Music | Purpose |
|-------|-------|---------|
| **Prologue** | Minimal ambient | Establish mystery |
| **Exploration** | No music | Environmental audio only |
| **Awakening** | Musical elements emerge | Ship becomes more "vocal" |
| **Ascension** | Full score | Emotional climax |
| **Initiation** | Resolution | Subscription completion |

**Audio Progression:**

```
0.0-0.25 Awakening: Pure ambient (ship sounds, minimal)
0.25-0.5 Awakening: Ambient + tonal elements (musicality emerges)
0.5-0.75 Awakening: Layered audio (complex, responsive)
0.75-1.0 Awakening: Full soundscape (rich, immersive)
1.0+ (Ascension): Music enters (emotional climax)
```

### Aesthetic Goals

**How Art and Audio Support Game Pillars:**

| Pillar | Art Support | Audio Support |
|--------|------------|---------------|
| **Contemplative Awakening** | Organic visuals feel alive, not mechanical | Living soundscapes respond to presence |
| **Environmental Discovery** | Visual cues guide without forcing | Spatial audio draws attention |
| **Dual Narrative Layer** | Ship aesthetics mirror magazine ethos | Chamber soundscapes reflect editorial pillars |
| **Earned Transcendence** | Visual crescendo in ascension sequence | Musical climax seals emotional journey |

**Emotional Arc Through Aesthetics:**

```
Confusion → Dark, quiet, minimal cues
Curiosity → First light and sound emerge
Discovery → Rich visuals and audio reveal depth
Achievement → Full brightness and resonance
Transcendence → Light and music crescendo
Initiation → Fade to white with complete resolution
```

---

## Technical Specifications

### Performance Requirements

**Frame Rate Targets:**

| Tier | Target | Minimum Acceptable | Resolution |
|------|--------|-------------------|------------|
| **max** | 60+ FPS | 45 FPS | 1080p+ |
| **desktop** | 60 FPS | 45 FPS | 1080p |
| **laptop** | 45-60 FPS | 30 FPS | 720p-1080p |
| **mobile** | 30 FPS | 24 FPS | 720p or device native |

**Performance Budgets:**

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Initial Load** | <5 seconds to first frame | Time to interactive |
| **Full Load** | <15 seconds for base experience | All base splats loaded |
| **Frame Time** | 16.6ms (60fps) / 33.3ms (30fps) | Performance API |
| **Memory** | <500MB on mobile Safari | Memory profiler |
| **GPU** | <80% utilization on target devices | Frame timing metrics |

**Load Time Requirements:**

- **Time to first frame:** <5 seconds on desktop, <8 seconds on mobile
- **Progressive loading:** Base scene loads first, higher quality streams in
- **Background streaming:** Non-visible splats load asynchronously
- **Optimization:** LOD variants, compression, aggressive culling

### Platform-Specific Details

**Web Browser (Primary Platform)**

**Supported Browsers:**

| Browser | Rendering Path | Min Version | Notes |
|---------|----------------|-------------|-------|
| Chrome/Edge | WebGPU (primary), WebGL2 (fallback) | 113+ | Primary target |
| Firefox | WebGL2 | 120+ | No WebGPU support yet |
| Safari | WebGL2 | 17+ | Memory-limited, no WebGPU |
| Opera | WebGPU (primary), WebGL2 (fallback) | 99+ | Chromium-based |
| Mobile Chrome | WebGPU (varies by device) | 113+ | Device-dependent |
| Mobile Safari | WebGL2 | iOS 17+ | Strict memory limits |

**WebGPU Implementation:**
- **Primary renderer:** WebGPU via Three.js WebGPURenderer
- **Fallback:** WebGL2 via Three.js WebGLRenderer
- **Feature detection:** Automatic capability detection on load
- **Graceful degradation:** Core experience works on all supported browsers

**iOS-Specific Considerations:**
- **Memory limits:** ~500MB GPU allocation hard cap
- **Splat count:** Mobile splat variants (1-2M splats max)
- **Touch-only:** No hover states, touch interactions primary
- **Audio:** Requires user gesture to start audio context
- **Performance:** Thermal throttling consideration

**Android-Specific Considerations:**
- **Browser fragmentation:** Chrome-based browsers preferred
- **WebGPU support:** Device and Chrome version dependent
- **Memory handling:** More permissive than iOS
- **Audio:** Spatial audio via Web Audio API

### Asset Requirements

**Splat Assets (Primary):**

| Asset Type | Quantity | Specifications |
|------------|----------|----------------|
| **Environment Splats** | 5-7 scenes | Prologue, Hub, 4 Chambers, Ascension |
| **Quality Variants** | 4 per scene | Max, Desktop, Laptop, Mobile |
| **File Sizes** | 5-50MB per scene per quality | Compressed SOG format |
| **Total Storage** | ~500MB-1GB | All variants compressed |

**Splat Production Pipeline:**

```
1. Concept Art
   ↓
2. Traditional 3D Blockout (Blender)
   - Rough forms, spatial relationships
   - Lighting studies
   ↓
3. AI Splat Generation (Marble Labs / Luma AI / Polycam)
   - Photo/video capture of physical maquettes OR
   - AI generation from concept art/text prompts
   ↓
4. Splat Processing
   - Quality variants (max/desktop/laptop/mobile)
   - LOD generation
   - SOG format export
   ↓
5. Integration into Shadow Engine
   - Zone loading setup
   - Trigger zone placement
   - Performance profiling
```

**Traditional 3D Assets:**

| Asset Type | Quantity | Purpose | Format |
|------------|----------|---------|--------|
| **UI Elements** | 5-10 models | Subscription form, settings | glTF/GLB |
| **Interactables** | 20-30 objects | Narrative artifacts | glTF/GLB |
| **VFX Elements** | 5-10 systems | Particle effects, shaders | Custom shaders |

**Audio Assets:**

| Asset Type | Quantity | Duration | Format |
|------------|----------|----------|--------|
| **Ambient Loops** | 5-7 tracks | 30-60 sec each | MP3/OGG |
| **Chamber Themes** | 4 tracks | 60-90 sec each | MP3/OGG |
| **Ascension Score** | 1 sequence | 60-90 sec | MP3/OGG |
| **SFX** | 30-50 sounds | 1-5 sec each | MP3/OGG |
| **Voice (optional)** | 2-4 clips | 10-20 sec each | MP3/OGG |

**Asset Storage and Delivery:**
- **CDN hosting:** Fast global delivery
- **Compression:** SOG compression, audio optimization
- **Lazy loading:** Non-visible assets load on demand
- **Caching:** Browser cache for repeat visits

### Technical Constraints

**Known Constraints:**

| Constraint | Impact | Mitigation |
|-----------|--------|------------|
| **Splat file size** | Large scenes = 40MB+ | LOD variants, progressive loading |
| **iOS memory limits** | Crashes on large scenes | Aggressive culling, mobile-specific counts |
| **WebGPU adoption** | Fallback to WebGL2 needed | Feature detection, graceful degradation |
| **No native haptics** | Limited mobile feedback | Visual + audio feedback only |
| **Battery drain** | GPU-intensive | Performance profiles, frame rate caps |

**Shadow Engine Dependencies:**
- **Three.js** (r160+) — 3D rendering framework
- **@sparkjsdev/spark** — Gaussian Splatting renderer (SOG format)
- **@typegpu/three** — Type-safe GPU shader programming
- **Howler.js** — Spatial audio and crossfades
- **Rapier** — Physics for trigger zones and interactables
- **Vite** — Build tool and dev server

**Browser API Requirements:**
- **WebGPU** or **WebGL2** for rendering
- **Web Audio API** for spatial audio
- **Pointer Events** for unified input handling
- **Intersection Observer** for lazy loading
- **RequestAnimationFrame** for render loop

---

## Development Epics

### Epic Overview

| # | Epic Name | Scope | Dependencies | Est. Stories |
|---|-----------|-------|--------------|--------------|
| 1 | Foundation & Core Systems | Engine setup, base mechanics | None | 8-10 |
| 2 | Splat Pipeline & Environments | Asset creation, scene integration | Foundation | 10-12 |
| 3 | Interactive Awakening | Gaze system, ship response | Foundation | 6-8 |
| 4 | Chamber Content | Four themed chambers | Splat Pipeline | 8-10 |
| 5 | Ascension Sequence | Cinematic finale | Chamber Content | 6-8 |
| 6 | Audio & Polish | Sound design, final polish | All previous | 8-10 |
| 7 | Subscription Integration | Magazine signup flow | Foundation | 4-6 |
| 8 | Performance & Launch | Optimization, testing | All previous | 6-8 |

### Recommended Sequence

**Phase 1: Foundation (Epics 1, 7)**
- Establish core technical base
- Implement basic interaction
- Set up subscription backend

**Phase 2: Content (Epics 2, 3, 4)**
- Build asset pipeline
- Create core mechanics
- Populate chambers

**Phase 3: Finale (Epic 5)**
- Implement ascension sequence
- Complete narrative arc

**Phase 4: Polish (Epics 6, 8)**
- Audio integration
- Performance optimization
- Launch preparation

### Vertical Slice

**The first playable milestone** includes:
- One complete chamber (Gallery of Forms)
- Core gaze interaction system
- Ship response feedback
- Basic audio atmosphere
- Performance target validation

**Vertical Slice Goals:**
- Prove Gaussian Splatting performance
- Validate gaze interaction feel
- Test emotional resonance
- Confirm production pipeline

---

## Epic Details

## Epic 1: Foundation & Core Systems

### Goal
Establish the technical foundation for VIMANA including engine setup, basic scene loading, input handling, and state management.

### Scope

**Includes:**
- Shadow Engine integration and configuration
- Three.js + SparkJS renderer setup
- WebGPU/WebGL2 fallback system
- Basic scene loading and camera control
- Input handling (mouse and touch)
- GameManager state machine
- Device detection and performance profiling
- Basic UI framework

**Excludes:**
- Gaze interaction system (Epic 3)
- Splat scene content (Epic 2)
- Audio implementation (Epic 6)

### Dependencies
None — this is the foundation epic

### Deliverable
A working base project with:
- Empty Vimana interior (placeholder splat)
- Working camera control (drag to look)
- Device-based performance tier selection
- Basic state management system

### Stories

1. As a developer, I can initialize the Shadow Engine with WebGPU/WebGL2 support so that the experience runs on target browsers
2. As a developer, I can detect device capabilities and select appropriate performance tier so that the experience runs smoothly
3. As a player, I can look around the environment with mouse/touch drag so that I can explore the space
4. As a developer, I can load and render SOG format splat files so that Gaussian Splatting content displays
5. As a developer, I can implement the GameManager state machine so that narrative progression is tracked
6. As a developer, I can create a basic UI overlay system so that subscription forms and settings can be displayed
7. As a player, I can restart the experience with R key/pull-to-refresh so that I can replay without page refresh
8. As a developer, I can set up build pipeline with Vite so that development is efficient

---

## Epic 2: Splat Pipeline & Environments

### Goal
Create the Gaussian Splatting asset production pipeline and generate all environment splats for the experience.

### Scope

**Includes:**
- Splat capture/generation workflow setup
- AI splat generation (Marble Labs/Luma AI)
- Quality variant creation (max, desktop, laptop, mobile)
- Prologue environment splat
- Main Hub interior splat
- Four chamber environment splats
- Ascension space splat
- SOG file optimization and compression

**Excludes:**
- Interactive elements within chambers (Epic 4)
- Chamber-specific content and artifacts

### Dependencies
Foundation & Core Systems (Epic 1) must be complete to test splat loading

### Deliverable
All environment splat assets ready for integration:
- 7 quality-tiered SOG files (5 environments × quality variants)
- Documented production pipeline
- Performance-tested assets

### Stories

1. As a developer, I can set up the AI splat generation pipeline (Marble Labs) so that I can create environment assets
2. As a developer, I can generate quality variants for each splat scene so that performance tiers work correctly
3. As a developer, I can create the Prologue environment splat so that the opening scene exists
4. As a developer, I can create the Main Hub interior splat so that the central navigation space exists
5. As a developer, I can create the Gallery of Forms splat so that the Art chamber environment exists
6. As a developer, I can create the Hydroponic Memory splat so that the Nature chamber environment exists
7. As a developer, I can create the Archive of Voices splat so that the Culture chamber environment exists
8. As a developer, I can create the Engine of Growth splat so that the Technology chamber environment exists
9. As a developer, I can create the Ascension Space splat so that the finale environment exists
10. As a developer, I can optimize splat file sizes while maintaining visual quality so that load times are acceptable

---

## Epic 3: Interactive Awakening

### Goal
Implement the core gaze-to-awaken interaction system and ship response feedback loop.

### Scope

**Includes:**
- Gaze detection raycasting system
- Awakening intensity tracking (0-1 progression)
- Interactive element hitbox setup
- Visual feedback (glow, pulse, brightness)
- Audio feedback (ship response sounds)
- Global awakening level tracking
- Element activation state management
- Progressive visibility system

**Excludes:**
- Chamber-specific interactive content (Epic 4)
- Narrative fragment content

### Dependencies
Foundation & Core Systems (Epic 1) must be complete

### Deliverable
Fully functional interaction system:
- Gaze on elements triggers awakening response
- Ship responds visually and audibly to player presence
- Global awakening tracks progress across all chambers

### Stories

1. As a player, I can activate elements by holding my gaze on them so that the ship awakens
2. As a player, I see visual feedback (glow, pulse) when gazing at interactables so that I know the system is responding
3. As a player, I hear audio feedback from the ship when I activate elements so that I feel the ship is alive
4. As a developer, I can configure hitboxes on interactive elements with generous detection zones so that interaction feels natural
5. As a player, I see progressive response from elements (noticed → awakening → complete) so that the awakening feels organic
6. As a developer, I can track global awakening level across all chambers so that the Voyage Gate unlocks correctly
7. As a player, I notice the ship becoming more active (brighter, more responsive) as I explore so that I feel progress

---

## Epic 4: Chamber Content

### Goal
Populate the four chambers with narrative fragments, interactive elements, and environmental storytelling.

### Scope

**Includes:**
- Narrative fragment placement in each chamber
- Interactive element creation (artifacts, projections, audio echoes)
- Chamber-specific visual theming
- Environmental storytelling elements
- Crew artifact models
- Trigger zone placement for narrative events
- Chamber completion tracking

**Excludes:**
- Final ascension sequence (Epic 5)
- Full audio implementation (Epic 6)

### Dependencies
Splat Pipeline & Environments (Epic 2) and Interactive Awakening (Epic 3) must be complete

### Deliverable
Four fully populated chambers:
- Each chamber has 2-3 narrative fragments
- Interactive elements respond to gaze
- Chamber completion tracked individually

### Stories

1. As a player, I can discover narrative fragments in the Gallery of Forms so that I learn about the Vimana's artistic legacy
2. As a player, I can discover narrative fragments in the Hydroponic Memory so that I learn about the Vimana's connection to nature
3. As a player, I can discover narrative fragments in the Archive of Voices so that I learn about the Vimana's cultural wisdom
4. As a player, I can discover narrative fragments in the Engine of Growth so that I learn about the Vimana's technological innovation
5. As a developer, I can place interactive artifacts (3D models) in chambers so that players have focal points for gaze interaction
6. As a player, I can see visual cues (bioluminescence) guiding me to narrative fragments so that I can find content naturally
7. As a developer, I can track which chambers have been completed so that the Voyage Gate unlocks appropriately
8. As a player, I can revisit chambers to find additional details so that replay value exists

---

## Epic 5: Ascension Sequence

### Goal
Implement the scripted cinematic ascension sequence (water → sky → space → void) and subscription reveal.

### Scope

**Includes:**
- Voyage Gate manifestation effect
- Camera animation system for ascension
- Environmental phase transitions (water, sky, space, void)
- Shader effects for each phase
- Ascension space environment
- Subscription form reveal
- Skip functionality (ESC/tap)
- Phase timing and pacing

**Excludes:**
- Ascension music score (Epic 6)

### Dependencies
Chamber Content (Epic 4) must be complete

### Deliverable
Complete ascension experience:
- Voyage Gate opens after all four chambers complete
- Entering Gate triggers cinematic sequence
- Subscription form appears at climax
- Experience can be skipped if desired

### Stories

1. As a player, I see the Voyage Gate manifest after completing all chambers so that I know the experience is culminating
2. As a player, I can enter the Voyage Gate to trigger the ascension so that the finale begins
3. As a player, I experience the water → sky → space → void sequence so that I feel transcendence
4. As a developer, I can implement camera animations for the ascension so that the sequence feels cinematic
5. As a developer, I can create shader effects for each ascension phase so that the environmental transitions are convincing
6. As a player, I am presented with the subscription form after the ascension so that I can join VIMANA
7. As a player, I can skip the ascension sequence with ESC or tap so that I'm not forced to watch it every time
8. As a developer, I can time the ascension to ~60-90 seconds so that the pacing feels right

---

## Epic 6: Audio & Polish

### Goal
Implement complete audio system and apply visual polish throughout the experience.

### Scope

**Includes:**
- Chamber-specific ambient soundscapes
- Ship response audio (pulse, hum, awakening sounds)
- Spatial audio positioning (Howler.js)
- Ascension music score
- Audio crossfading between zones
- Visual polish (particle effects, post-processing)
- UI refinement and animation
- Accessibility features (colorblind modes, reduced motion)

**Excludes:**
- Performance optimization (Epic 8)

### Dependencies
All previous epics should be substantially complete

### Deliverable
Fully polished experience with:
- Rich, responsive audio throughout
- Visual effects at target quality
- Accessibility features implemented

### Stories

1. As a player, I hear ambient soundscapes in each chamber so that each area has distinct atmosphere
2. As a player, I hear the ship respond to my presence with audio so that the Vimana feels alive
3. As a player, I experience spatial audio so that sound direction helps me locate interactive elements
4. As a player, I hear music during the ascension so that the climax has emotional impact
5. As a developer, I can implement audio crossfading between chambers so that transitions are smooth
6. As a player, I can enable reduced motion mode so that motion sensitivity is addressed
7. As a player, I can select colorblind modes so that bioluminescence cues remain visible
8. As a developer, I can add particle effects and post-processing so that the visual quality meets target

---

## Epic 7: Subscription Integration

### Goal
Implement the magazine subscription flow and backend integration.

### Scope

**Includes:**
- Subscription form UI design and implementation
- Email capture system
- Payment processor integration (Stripe)
- Base mini-app integration
- Confirmation email system
- Analytics integration (signup tracking)
- A/B testing framework for conversion optimization

**Excludes:**
- Subscription marketing copy (content task)

### Dependencies
Foundation & Core Systems (Epic 1) for UI framework

### Deliverable
Working subscription system:
- Beautiful, on-brand subscription form
- Stripe payment processing
- Email capture and confirmation
- Analytics tracking complete

### Stories

1. As a player, I can see a beautiful subscription form after the ascension so that I can join VIMANA
2. As a player, I can enter my email to subscribe so that I receive the first issue
3. As a developer, I can integrate Stripe payment processing so that subscriptions are handled securely
4. As a developer, I can send confirmation emails so that users know their subscription is confirmed
5. As a developer, I can integrate with the Base mini-app so that Web3 users can subscribe
6. As a developer, I can track signup events in analytics so that conversion can be measured
7. As a player, I receive immediate confirmation upon subscribing so that I know the transaction succeeded

---

## Epic 8: Performance & Launch

### Goal
Optimize performance across all target devices and prepare for public launch.

### Scope

**Includes:**
- Cross-device performance testing
- Memory optimization (especially iOS)
- Frame rate consistency improvements
- Load time optimization
- Progressive loading refinement
- Bug fixing and stability improvements
- Analytics and monitoring setup
- Launch preparation (press kit, sharing assets)

**Excludes:**
New feature development

### Dependencies
All previous epics complete

### Deliverable
Launch-ready experience:
- 60fps on desktop, 30fps on mobile targets
- <5 second initial load time
- <2% crash rate
- Press kit and launch assets ready

### Stories

1. As a developer, I can test the experience on desktop browsers so that performance targets are validated
2. As a developer, I can test on iOS Safari so that memory crashes are eliminated
3. As a developer, I can test on Android devices so that cross-platform compatibility is confirmed
4. As a developer, I can optimize splat loading so that initial load time is under 5 seconds
5. As a developer, I can implement frame rate monitoring so that performance issues are identified
6. As a developer, I can fix critical bugs so that the experience is stable for launch
7. As a developer, I can create press kit and sharing assets so that launch marketing is supported
8. As a developer, I can set up analytics and error monitoring so that post-launch issues are trackable

---

## Success Metrics

### Technical Metrics

**Key Technical KPIs:**

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Frame Rate (Desktop)** | 60 FPS consistent | Performance API monitoring |
| **Frame Rate (Mobile)** | 30 FPS consistent | Performance API monitoring |
| **Initial Load Time** | <5 seconds | Navigation Timing API |
| **Full Experience Load** | <15 seconds | Asset loading events |
| **Crash Rate** | <2% of sessions | Error tracking (Sentry) |
| **Mobile Safari Stability** | No memory crashes | Device testing farm |
| **WebGPU Adoption** | >60% of capable users | Analytics feature detection |
| **Browser Coverage** | 95%+ of visitors work | Analytics browser tracking |

**Performance Monitoring:**
- **Real user monitoring (RUM)** for production performance
- **Frame time metrics** captured during sessions
- **Memory profiling** on iOS devices
- **Load time breakdown** by asset type
- **Performance score** by device tier

### Gameplay Metrics

**Key Gameplay KPIs:**

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Completion Rate** | >40% reach Voyage Gate | Analytics progression events |
| **Average Session Length** | 6-8 minutes | Analytics session duration |
| **Chamber Distribution** | All chambers visited equally | Analytics chamber entry events |
| **Subscription Rate** | >15% of completers | Analytics signup events |
| **Share Rate** | >10% of visitors | Social share tracking |
| **Return Visits** | >20% of first-time visitors | Analytics unique visitors |
| **Mobile vs Desktop Split** | 40% mobile, 60% desktop | Analytics device tracking |

**Engagement Metrics:**
- **Gaze interaction count** per session
- **Narrative fragments discovered** per player
- **Time spent per chamber**
- **Ascension skip rate** (indicates pacing issues)
- **Replay rate** (percentage who start second session)

**Conversion Metrics:**
- **Landing page to experience start** conversion
- **Experience complete to signup** conversion
- **Email-only signup rate** (no payment)
- **Paid subscription rate**
- **Base mini-app conversion** (if applicable)

### Qualitative Success Criteria

**Indicators of Success:**
- Players describe the experience using pillar words ("awakening," "beautiful," "mysterious")
- Social media shares include positive commentary on visuals and uniqueness
- Press coverage mentions technical achievement and aesthetic innovation
- Streamers/YouTubers create content around the experience
- Players recommend to friends with "you have to see this"
- Reviews highlight the unique combination of art, nature, culture, technology

**Qualitative Tracking:**
- Social media sentiment analysis
- Review keyword analysis
- Press coverage tone and focus
- Community feedback themes
- Awwwards/jury feedback (if submitted)

### Metric Review Cadence

**Weekly during development:**
- Performance metrics by device tier
- Frame rate consistency
- Load time breakdown
- Bug crash reports

**Daily during launch week:**
- All technical KPIs
- All gameplay KPIs
- Conversion funnel analysis
- Error rates and types

**Monthly post-launch:**
- Long-term retention metrics
- Qualitative feedback synthesis
- Feature request analysis
- Competitive landscape review

---

## Out of Scope

### Explicitly Out of Scope for V1.0

**Features:**
- Multiplayer or social features within the experience
- Level editor or customization tools
- Mod support or user-generated content
- Save/load system (experience is designed as single-sitting)
- Replayability features (beyond inherent narrative depth)
- Achievement systems or badges
- Leaderboards or competition elements

**Content:**
- Additional game modes or variations
- DLC or expansion content
- Alternative endings or branching narratives
- Extended story beyond initial magazine launch
- Multiple language support (post-launch consideration)
- VR or AR versions (future exploration)

**Platforms:**
- Console ports (PS5, Xbox, Switch)
- Native mobile apps (iOS App Store, Google Play)
- Desktop applications (Steam, Epic Games Store)
- Web3 beyond Base integration

**Polish:**
- Full voice acting for narrative (optional narration only)
- Orchestral score beyond ascension sequence
- Motion capture for animations
- Procedural content generation
- Advanced graphics options beyond performance tiers

### Deferred to Post-Launch

**Potential V2.0 Features:**
- Extended narrative with additional chambers
- Interactive "crew member" encounters
- Deeper exploration of Vimana lore
- Magazine content integration within the experience
- Multiplayer shared exploration mode
- VR adaptation for full immersion
- Additional language localizations
- "Director's commentary" mode

**Technical Improvements:**
- Advanced WebGPU features as browser support improves
- AI-driven procedural content
- Real-time global illumination
- Haptic feedback for compatible devices
- AR mode for mobile devices

---

## Assumptions and Dependencies

### Key Assumptions

**Technical Assumptions:**
- WebGPU adoption will continue to grow (>70% of target audience within 6 months)
- Chrome/Edge will maintain WebGPU stability and performance
- iOS Safari will maintain WebGL2 support without regression
- Gaussian Splatting tools (Marble Labs, Luma AI) will remain available and improve
- Three.js and SparkJS will continue active development
- Device capabilities will improve (mobile GPU performance trajectory continues)

**Team Assumptions:**
- Solo developer can complete core development within 3-4 months
- Audio composer/sound designer can be contracted or existing skills sufficient
- Splat generation can be done via AI tools without dedicated 3D artist
- Web3/Base integration expertise can be acquired or learned
- QA testing can be done via device farm + personal testing

**Market Assumptions:**
- Target audience (18-40, culturally curious) will share unique web experiences
- Awwwards/design community will recognize technical achievement
- Magazine launch timing aligns with market readiness
- Subscription model is viable for this type of content
- Base/Coinbase user base provides meaningful early adopter segment

**Content Assumptions:**
- 5-10 minute experience length is appropriate for target audience
- Contemplative, non-challenging gameplay will resonate
- Magazine brand integration feels authentic, not forced
- Solarpunk/cyber-organic aesthetic has broad appeal
- Narrative complexity is appropriate without being overwhelming

### External Dependencies

**Third-Party Services:**
- **Stripe** — Payment processing for subscriptions
- **Base/Coinbase** — Web3 wallet integration
- **Email service** (SendGrid/Mailchimp) — Confirmation emails
- **Analytics** (Google Analytics, Plausible) — Usage tracking
- **Error monitoring** (Sentry) — Crash reporting
- **CDN** (Cloudflare, AWS CloudFront) — Asset delivery
- **Hosting** (Vercel, Netlify) — Web deployment

**Content Dependencies:**
- **Marble Labs / Luma AI** — AI splat generation tools
- **Music licensing** or original composition for ascension score
- **Voice talent** (optional) for prologue/ascension narration
- **Magazine content** — First issue must be ready for subscription fulfillment

**Platform Dependencies:**
- **Chrome WebGPU roadmap** — No regression in WebGPU support
- **Safari WebGL2** — No removal of current WebGL2 features
- **iOS memory limits** — No further reduction in GPU memory allocation
- **Base platform stability** — No breaking changes to Base SDK

**Dependency Mitigation:**
- **WebGPU fallback** — WebGL2 implementation ready as alternative
- **Asset flexibility** — Splat scenes can be regenerated with different tools
- **Platform diversity** — Multiple hosting/deployment options available
- **Content adaptability** — Narrative works without voiceover if needed

### Risk Factors

**High Risk:**
- **Performance failure on target devices** — Could result in negative user experience and brand damage
- **iOS memory crashes** — Mobile Safari crashes would exclude significant audience
- **Low completion rate** — Players dropping off before Voyage Gate would result in few conversions

**Medium Risk:**
- **Ambiguous brand connection** — Players not understanding the magazine connection
- **Splat quality limitations** — AI-generated splats not meeting visual quality targets
- **WebGPU adoption slower than expected** — Fallback to WebGL2 reduces visual quality

**Low Risk:**
- **Competitive products launching** — Unique positioning and technical barrier to entry
- **Magazine launch delays** — Experience remains relevant as standalone brand piece
- **Platform policy changes** — Web standards relatively stable, minimal policy risk

---

## Document Information

**Document:** VIMANA - Game Design Document
**Version:** 1.0
**Created:** 2026-01-15
**Author:** Mehul
**Status:** Complete

### Change Log

| Version | Date | Changes |
| ------- | ---- | ------- |
| 1.0 | 2026-01-15 | Initial GDD complete - all 14 steps finished |

---

## Next Steps

### Recommended Workflows

**1. Narrative Design Document** (Recommended for this game type)

Based on the Adventure game classification, a dedicated Narrative Design Document would benefit VIMANA. This would cover:
- Detailed story structure and beats
- Character profiles (the absent crew)
- World lore and Vimana history
- Environmental storytelling framework
- Dialogue/script for any voice elements

**Command:** Use the Narrative Design workflow next
**Input:** This GDD
**Output:** `narrative-design.md`

**2. Game Architecture**

Define the technical architecture, tech stack decisions, and system design for implementation.

**Command:** Use the Game Architecture workflow
**Input:** This GDD
**Output:** `architecture.md`

**3. Sprint Planning**

Set up the first development sprint based on the defined epics.

**Command:** Use the Sprint Planning workflow
**Input:** This GDD + epics
**Output:** `sprint-status.yaml`

### Immediate Actions

- [ ] Review GDD with any stakeholders
- [ ] Generate sample splat scenes to test pipeline
- [ ] Set up Shadow Engine project with WebGPU/WebGL2
- [ ] Create concept art for four chamber aesthetics
- [ ] Commission or create audio atmosphere samples
- [ ] Set up Stripe subscription backend
- [ ] Begin vertical slice development (Foundation + Splat Pipeline)

---

**GDD Workflow Complete**

All 14 steps completed. The VIMANA Game Design Document is ready for the next phase of development.
