---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
inputDocuments: []
documentCounts:
  brainstorming: 0
  research: 0
  notes: 0
workflowType: 'game-brief'
lastStep: 10
project_name: 'shadowvimana'
user_name: 'Mehul'
date: '2026-01-15'
game_name: 'VIMANA'
---

# Game Brief: VIMANA

**Date:** 2026-01-15
**Author:** Mehul
**Status:** Draft for GDD Development

---

## Executive Summary

{{executive_summary}}

---

## Game Vision

### Core Concept

An atmospheric first-person web exploration set inside a bio-organic Vimana spacecraft, serving as the immersive launch platform for VIMANA—a new international magazine at the intersection of Art, Nature, Culture, and Technology.

### Elevator Pitch

VIMANA is a contemplative web experience that introduces the world to a magazine unlike any other. Explore a living spacecraft grown from deep-sea biology and fungal intelligence—a metaphor for the publication's four pillars: Art, Nature, Culture, and Technology. Discover each chamber to unlock the magazine's worldview, then voyage beyond through a ritual ascension that seals your subscription to the first issue.

### Vision Statement

To create a launch experience that embodies VIMANA magazine's ethos: **technology grown from nature, not built upon it.** Players will discover a publication that challenges the cynical "tech vs nature" paradigm—one that treats biology as the ultimate industrial designer and asks: what if the most advanced technology wasn't manufactured, but *cultivated*? The experience celebrates Solarpunk optimism, positioning VIMANA as the first publication for a generation that believes humanity's future lies in growing with our planet, not escaping from it.

---

## Target Market

### Primary Audience

Culturally curious individuals aged 18-40 who seek out beautiful, shareable web experiences that evoke wonder—the "awww" crowd. They're drawn to content at the intersection of disciplines: avant-garde art, underground music, cutting-edge technology, and biomimetic design. They want to witness something unique, feel part of something grander, and anticipate being initiated into "top notch shit."

**Demographics:**
- Age: 18-40 (primary core 25-35)
- Platform agnostic: mobile and desktop parity
- Discovers via: Direct website, shared links, Base mini app

**Gaming Preferences:**
- Casual experience: natural story progression, no complicated mechanics
- Familiar with immersive web experiences (Awwwards, interactive art sites)
- Values atmosphere and aesthetic over challenge or skill

**Motivations:**
- **Discovery:** Witnessing something unique they haven't seen before
- **Initiation:** Anticipating membership in something exclusive and grand
- **Taste curation:** Being "in the know" about cutting-edge culture

### Secondary Audience

**1. Casual Social Browsers**
- Stumble via social shares, may not subscribe immediately
- Value the viral "wow" factor, share with their networks
- Entry point to the VIMANA brand

**2. Industry Professionals**
- Artists, designers, musicians, creative directors
- Seek to be featured, collaborate, or align with the brand
- Value the aesthetic and technical achievement

**3. Crypto/Base Community**
- Discover via the Base mini app integration
- Bridge between web3 culture and high-end editorial
- Tech-savvy early adopters

### Market Context

VIMANA launches at a moment when technology has finally caught up to the vision. AI-generated Gaussian splats (Marble Labs) and WebGPU maturity enable photorealistic web experiences that were impossible just two years ago.

**Proof of Concept:**
- **Shadow of the Czar** — demonstrates the engine's capability for atmospheric, narrative-driven web experiences
- **World Labs case studies** — validates appetite for AI-generated immersive environments
- **Awwwards winners** — established audience for beautiful, shareable web experiences

**Market Opportunity:**

There is no direct competition. VIMANA operates in a different league—creating a new category that merges high-end magazine publishing, immersive 3D web experiences, and Solarpunk/cyber-organic aesthetic philosophy. Traditional magazines (*Wired*, *Kinfolk*, *Aeon*) operate in 2D. Game studios focus on entertainment. VIMANA fuses editorial depth with immersive technology to launch a publication, not just sell a product.

---

## Game Fundamentals

### Core Gameplay Pillars

**1. Narrative Transformation**
The player isn't just "exploring a spaceship"—they *become* the boy from Kolkata who touched the orb and vanished. Through first-person POV, they experience what happened next. This creates emotional resonance: you're not a visitor, you're the protagonist who disappeared.

**2. Four-Pillar Discovery**
The Vimana contains four chambers, each embodying one of VIMANA magazine's domains: Art, Nature, Culture, Technology. Players must visit each chamber, experiencing its unique aesthetic and thematic content. Exploration is the core loop—organic movement, environmental storytelling, no puzzles or challenges.

**3. Earned Initiation**
The Voyage Gate (preorder CTA) only opens after all four chambers have been explored. This transforms the call-to-action from a sales pitch into a ritual achievement. You've witnessed the magazine's worldview—you've *earned* passage.

**4. Cinematic Ascension**
The climax is a scripted first-person journey: submerged in water, rising through clouds, ascending into space, then vanishing into white. This ritual finale seals the subscription and delivers an unforgettable emotional peak—the experience is the message.

### Primary Mechanics

**First-Person Atmospheric Navigation**
- Look-based movement (drag/look to navigate, similar to Shadow of the Czar)
- No joystick, no complex controls—pure immersion
- Organic camera paths guide players without railroading

**Environmental Interaction**
- Chambers reveal content as you explore them (no clicking required)
- Light, sound, and visual cues signal progress
- Trigger zones advance the narrative state automatically

**Progressive Unlocking**
- Voyage Gate remains sealed until all four modules are visited
- Visual feedback shows which chambers remain
- State tracking remembers exploration progress

**Scripted Sequences**
- Prologue: Global orb emergence, Kolkata boy's disappearance
- Awakening: First-person POV inside Vimana
- Ascension: Water → sky → space → void (finale)

### Player Experience Goals

| Goal | How It Feels | Implementation |
|------|--------------|----------------|
| **Wonder** | "I've never seen anything like this" | Photorealistic splat rendering, bioluminescent environments |
| **Curiosity** | "I want to see what's in that chamber" | Distinct visual identities for Art/Nature/Culture/Tech spaces |
| **Achievement** | "I earned this" | Voyage Gate only unlocks after full exploration |
| **Transformation** | "I became someone else for a moment" | Boy POV narrative framing, ascension ritual |
| **Initiation** | "I'm part of something beginning" | Vanishing into white = subscription confirmed |

**The Experience Arc:**
```
Confusion (Where am I?) → Curiosity (What is this place?) →
Discovery (Each chamber reveals a worldview) →
Achievement (All four explored, Gate opens) →
Transcendence (Ascension sequence) →
Initiation (Subscription sealed, return to reality changed)
```

---

## Scope and Constraints

### Target Platforms

**Primary: Web Browser (WebGPU/WebGL2)**

| Platform | Min Spec | Splat Limit | Notes |
|----------|----------|-------------|-------|
| **Desktop** | Discrete GPU (GTX 1660+) | 5-8M splats | Full-quality splats, all effects |
| **Laptop** | Integrated GPU (Intel Iris Xe+) | 2-5M splats | Medium quality splats |
| **Mobile** | Modern phone (iPhone 12+, equivalent Android) | 1-2M splats | Optimized splats, touch controls |

**Browser Support:**
- Chrome/Edge 120+ (WebGPU preferred)
- Safari 17+ (WebGL2 fallback)
- Firefox 120+ (WebGL2)
- **Mobile Safari**: Special handling for iOS memory constraints

**Accessibility:**
- Subtitle/caption support for prologue and ascension narration
- Reduced motion option for ascension sequence
- Screen reader announcements for state changes

### Development Timeline

**[FUTURE: To be defined in GDD planning phase]**

Marked for detailed planning:
- [ ] Asset creation pipeline (AI splat generation, traditional 3D modeling for hybrid elements)
- [ ] Content production (four module narratives, ascension script refinement)
- [ ] Audio production (ambient chamber soundscapes, ascension score)
- [ ] Performance optimization across device profiles
- [ ] Base mini app integration
- [ ] Subscription/payment backend integration

### Budget Considerations

**[FUTURE: To be defined in production planning]**

Key cost drivers identified:
- **Splat Capture/Generation**: AI-generated splats via Marble Labs or equivalent
- **Audio**: Original composition for chamber themes + ascension sequence
- **Voice Talent**: Narration for prologue and ascension (if voiced)
- **Web3 Integration**: Base smart contract deployment + mini app development
- **Performance Testing**: Device farm testing across target profiles

### Team Resources

**[FUTURE: To be defined based on availability]**

Core roles needed:
- **Creative Director**: Vision holder, aesthetic consistency
- **Technical Lead**: Shadow Engine implementation, performance optimization
- **3D Artist**: Hybrid content (splat capture support, traditional 3D for UI/interactables)
- **Audio Designer**: Soundscape composition, spatial audio
- **Web3 Developer**: Base integration, subscription smart contracts
- **QA/Tester**: Cross-device performance validation

### Technical Constraints

**Shadow Engine Capabilities:**
- Gaussian Splatting via @sparkjsdev/spark (SOG format)
- WebGPU + WebGL2 fallback rendering
- State-driven scene loading via GameManager event system
- Zone-based splat loading/unloading for memory management
- Howler.js for spatial audio and crossfades
- Rapier physics for trigger zones and interactables

**Known Constraints:**

| Constraint | Impact | Mitigation |
|-----------|--------|------------|
| **Splat file size** | High-quality scenes = 40MB+ | LOD variants, progressive loading |
| **iOS memory limits** | Crashes on large scenes | Aggressive culling, mobile-specific splat counts |
| **WebGPU adoption** | Fallback to WebGL2 needed | Detect capabilities, degrade gracefully |
| **No native haptics** | Limited feedback on mobile | Visual + audio feedback only |
| **Battery drain** | GPU-intensive | Performance profiles, frame rate caps |

**Content Constraints:**
- Short experience (5-10 minutes first playthrough)
- Single continuous flow (no save/load, no branching)
- One-time narrative (no replayability designed initially)
- Four modules must be visited in any order (non-linear exploration)
- Voyage Gate CTA gated behind 100% module completion

---

## Reference Framework

### Inspiration Games

**Shadow of the Czar** (Own Proof of Concept)
- Atmospheric first-person web experience using Gaussian Splatting
- Demonstrates Shadow Engine's capability for narrative-driven immersion
- Validates look-based navigation and state-driven progression
- *Key learning: Photorealistic web experiences are achievable today*

**World Labs / Marble Labs**
- AI-generated 3D environments from text/image prompts
- Demonstrates consumer appetite for explorable AI spaces
- Validates Gaussian Splatting as web-ready technology
- *Key learning: AI splat generation can create unique environments efficiently*

**Awwwards SOTD Winners**
- Beautiful, shareable web experiences that go viral
- "Awww" crowd values aesthetic over interaction complexity
- *Key learning: Visual spectacle drives social sharing*

**Journey** (thatgamecompany)
- Emotional wordless narrative through environmental storytelling
- Ascending into the stars as a transformative climax
- *Key learning: Ritual experiences create lasting emotional impact*

**Proteus** (Ed Key)
- Exploration without challenge, pure atmosphere
- First-person movement through generative environments
- *Key learning: Curiosity alone can drive engagement*

### Competitive Analysis

**Traditional Magazines (2D Digital)**
| Publication | Format | VIMANA's Edge |
|-------------|--------|---------------|
| *Wired* | 2D website, articles | Immersive 3D launch experience |
| *Kinfolk* | Minimalist design | Living, breathing environments |
| *Aeon* | Long-form essays | Experiential worldview introduction |

**Interactive/Web Experiences**
| Experience | Focus | VIMANA's Edge |
|------------|-------|---------------|
| NFB's *Bear 71* | Documentary web | Photorealistic 3D + magazine launch |
*The Wilderness Downtown* | Music video | Splatting + editorial depth |
*Galapagos* (Brendan Dawes) | Generative art | Narrative transformation + CTA |

**No Direct Competition:**
VIMANA creates a new category at the intersection of:
- High-end magazine publishing (editorial depth)
- Immersive 3D web experiences (Gaussian Splatting)
- Solarpunk/cyber-organic philosophy (aesthetic worldview)

### Key Differentiators

**1. Launch Experience as Editorial**
- Unlike magazines that announce with press releases, VIMANA *is* the announcement
- The experience embodies the magazine's worldview before a single article is read
- Players understand the publication's ethos through participation, not explanation

**2. Technology Grown from Nature**
- Cyber-organic aesthetic: Vimana is biological, not mechanical
- Challenges the "tech vs nature" paradigm
- Spaces feel cultivated, not manufactured
- *No other publication embraces this philosophy*

**3. Earned Call-to-Action**
- Voyage Gate opens only after full exploration
- Subscription becomes a ritual achievement, not a transaction
- Transforms "buy now" into "you are ready to voyage"

**4. Photorealistic Web**
- First magazine launch using Gaussian Splatting
- Visual quality indistinguishable from pre-rendered video
- Runs in browser with no app download
- *Competitors are years behind this capability*

**5. POV Transformation**
- Player *is* the protagonist who disappeared
- Creates empathy through embodiment
- Not a visitor experiencing an artifact—you *are* the story

**6. International + Timeless**
- Prologue spans global (orbs emerge worldwide)
- Boy from Kolkata centers the Global South
- Solarpunk future speaks to generation超越 Western tech-dystopia narratives

---

## Content Framework

### World and Setting

**The Vimana**

A bio-organic spacecraft of unknown origin, "grown" rather than built. Its architecture draws from:

- **Deep-sea biology**: Bioluminescent chambers, flowing organic forms
- **Fungal intelligence**: Mycelial networks, interconnected systems
- **Vedic mythology**: Vimana as ancient flying palace, reimagined through sci-fi lens

**Visual Language:**
| Element | Description |
|---------|-------------|
| **Materials** | No hard edges or metal surfaces. Grown surfaces, membrane walls, crystalline structures |
| **Lighting** | Bioluminescence as primary light source. Spectral cyan, deep ocean navy, glowing accents |
| **Scale** | Vast interior spaces that feel Cathedral-like. Human-scale intimacy within cosmic grandeur |
| **Movement** | The Vimana itself breathes—subtle organic motion in walls, light pulses, ambient shifts |

**The Four Chambers** (Story Progression Points)

*[To be detailed in GDD: Each chamber embodies one of VIMANA magazine's pillars]*

| Chamber | Pillar | Aesthetic Signature |
|---------|--------|---------------------|
| **Module 1** | Art | [TBD] |
| **Module 2** | Nature | [TBD] |
| **Module 3** | Culture | [TBD] |
| **Module 4** | Technology | [TBD] |

**The Voyage Gate**
- Central hub chamber, initially sealed
- Visually distinct: aperture, threshold, passage
- Opens only after all four modules visited
- Triggers ascension sequence when entered

**The Ascension Space**
- Transition chamber for finale
- Water → sky → space → void visual progression
- Vimana visible through viewport during ascent

### Narrative Approach

**Wordless Environmental Storytelling**

The experience has no dialogue or text. Story is conveyed through:

- **Visual progression**: Each chamber reveals its theme through environment, not exposition
- **Audio design**: Soundscapes communicate emotion and theme
- **Light cues**: Bioluminescent pulses guide attention
- **Spatial design**: Architecture itself tells the story

**Script Structure:**

| Phase | Description | Duration |
|-------|-------------|----------|
| **Prologue** | Orbs emerge globally. Boy in Kolkata touches orb, vanishes. | ~45 sec |
| **Awakening** | First-person POV inside Vimana. Confusion, curiosity. | ~60 sec |
| **Exploration** | Four chambers visited in any order. Each reveals a pillar. | ~4-6 min |
| **Unlocking** | All chambers visited. Voyage Gate opens. Ritual readiness. | ~30 sec |
| **Ascension** | Vimana flies. Water → sky → space → void. Subscription sealed. | ~60 sec |

**Total Experience: 5-10 minutes**

**Narrative Perspective:**
- **First-person POV**: Player IS the boy who disappeared
- **No breaking character**: Never shows the player's body or reflection
- **Embodiment over observation**: Player participates, doesn't watch

**Emotional Arc:**
```
Confusion → Curiosity → Discovery → Achievement → Transcendence → Initiation
```

### Content Volume

**[FUTURE: To be scoped in GDD]**

Initial content estimates:

| Content Type | Quantity | Notes |
|--------------|----------|-------|
| **Splat Environments** | 5-6 scenes | Prologue exterior, 4 chambers, ascension space |
| **Audio Tracks** | 8-12 | Ambient chamber themes, ascension score |
| **Voice Lines** | 0-2 | Optional narration for prologue/ascension |
| **3D Models** | Hybrid | Splat capture + traditional models for UI/interactables |
| **Narrative Scripts** | 1 | Core story (already drafted) |
| **Trigger Zones** | ~10-15 | State-driven progression points |

**Scope Constraints:**
- Single continuous flow, no branching
- No save/load system
- No inventory or collectibles
- One-time experience (no planned replayability)

---

## Art and Audio Direction

### Visual Style

**Color Palette**

| Color Type | Hex/Description | Usage |
|------------|-----------------|-------|
| **Deep Ocean Navy** | #0a1628, #0f2744 | Primary darkness, void spaces, shadows |
| **Spectral Cyan** | #00d4ff, #4ff0ff | Bioluminescence, energy flows, active elements |
| **Bioluminescent Accents** | #7fff7f, #ff7fff | Organic highlights, membrane glow |
| **Void White** | #f8f8f8 → #ffffff | Ascension climax, fade to white |

**Aesthetic Philosophy: Cyber-Organicism**

The defining principle: **Technology grown from nature, not built upon it.**

| Traditional Sci-Fi | VIMANA Cyber-Organic |
|--------------------|----------------------|
| Hard edges, sharp angles | Organic curves, flowing surfaces |
| Metal, carbon fiber | Grown membranes, crystalline structures |
| Visible bolts, seams | Seamless, biologically integrated |
| Mechanical movement | Breathing, pulsing, alive |
| Artificial lighting | Bioluminescent, internal glow |
| Cold, sterile | Warm, living, symbiotic |

**Visual Reference Points:**

- **Deep-sea creatures**: Anglerfish bioluminescence, jellyfish transparency
- **Fungal networks**: Mycelial threads, mushroom gills, organic growth patterns
- **Vedic architecture**: Palace geometry reimagined through biological lens
- **Solarpunk visions**: Technology in harmony with nature, not dominating it

**Lighting Design:**
- **Primary**: Bioluminescence emanates from surfaces themselves
- **Directional**: No visible light sources; light *is* the material
- **Dynamic**: Pulse, breathe, respond to player presence
- **Color Theory**: Cyan/blue for wonder, warm accents for life, white for transcendence

**Camera & Frame:**
- **Static FPS**: No camera shake or artificial movement
- **Gentle引导**: Organic camera paths hint without forcing
- **Vista moments**: Deliberate pauses for visual awe
- **Ascension**: Smooth, continuous motion through environments

### Audio Style

**Philosophy: Sound as Environment**

Audio is not a layer *on top* of the experience—it *is* the environment. Players should feel immersed in a living acoustic space.

**Audio Pillars:**

| Pillar | Description |
|--------|-------------|
| **Presence** | The Vimana feels alive through ambient sound |
| **Guidance** | Audio cues draw attention without being directive |
| **Emotion** | Each chamber has emotional signature through soundscape |
| **Climax** | Ascension sequence has dedicated musical score |

**Sound Design Categories:**

| Category | Examples | Notes |
|----------|----------|-------|
| **Ambient Chambers** | Hum, resonance, organic motion | Each chamber unique to its pillar |
| **Biological Sounds** | Breathing, pulse, membrane stretch | Vimana is alive |
| **Interaction Feedback** | Subtle confirmations, state changes | Light touches, not gamey |
| **Ascension Score** | Orchestral/electronic hybrid | Builds to transcendent climax |
| **Silence** | Strategic pauses for impact | Not all moments need sound |

**Spatial Audio:**
- **Howler.js spatial positioning**: Sounds emanate from actual 3D positions
- **Reverb zones**: Each chamber has unique acoustic signature
- **Distance attenuation**: Sounds evolve as player explores
- **Crossfading**: Smooth transitions between zones

**Voice Treatment:**
- **Optional narration**: Prologue/ascension may have voiceover
- **Style options**: Documentary narrator, poetic reading, or fully wordless
- **Language considerations**: International audience, potential for multiple languages

### Production Approach

**Visual Production Pipeline:**

```
1. Concept Art
   ↓
2. Traditional 3D Blockout (Blender/Maya)
   - Rough forms, spatial relationships
   - Lighting studies
   ↓
3. AI Splat Generation (Marble Labs / equivalent)
   - Photo/video capture of physical maquettes OR
   - AI generation from concept art/text prompts
   ↓
4. Splat Processing
   - Quality variants (max/desktop/laptop/mobile)
   - LOD generation
   ↓
5. Integration into Shadow Engine
   - SOG format export
   - Zone loading setup
   - Trigger zone placement
   ↓
6. Iteration
   - Performance profiling
   - Visual polish
```

**Audio Production Pipeline:**

```
1. Sound Design Brief
   - Per-chamber emotional targets
   - Reference tracks
   ↓
2. Field Recording (optional)
   - Organic textures for authenticity
   ↓
3. Composition/Design
   - Chamber-specific soundscapes
   - Ascension score
   ↓
4. Spatial Implementation
   - Howler.js positioning
   - Reverb zone tuning
   ↓
5. Cross-device Testing
   - Laptop speakers
   - Desktop audio
   - Mobile speakers
   - Headphones
```

**Hybrid Production Philosophy:**

VIMANA uses **both** Gaussian Splatting AND traditional 3D:

| Element | Production Method | Why |
|---------|-------------------|-----|
| **Environments** | AI Splat Generation | Photorealism, efficiency |
| **UI Elements** | Traditional 3D/2D | Crisp text, clear affordances |
| **Interactables** | Traditional 3D | Precise collision, game feel |
| **Particles/VFX** | Shader-based | Dynamic, performant |

**Tools & Technologies:**

| Task | Tool |
|------|------|
| **Splat Generation** | Marble Labs, Luma AI, Polycam |
| **Traditional 3D** | Blender, Maya |
| **Splat Processing** | SparkJS toolchain |
| **Audio** | Ableton Live, Reaper, Pro Tools |
| **Engine** | Shadow Engine (Three.js + SparkJS) |
| **Version Control** | Git |
| **Project Management** | [TBD in planning] |

**Quality Standards:**

- **60fps target** across all device profiles
- **Mobile Safari compatibility** rigorously tested
- **Progressive loading** for bandwidth-constrained users
- **Accessibility** (captions, reduced motion, screen reader)

---

## Risk Assessment

### Key Risks

**1. Performance Failure on Target Devices**

Gaussian Splatting is GPU-intensive. If the experience runs poorly on mid-range devices, users will bounce before reaching the Voyage Gate.

- **Impact**: High (core experience compromised)
- **Likelihood**: Medium (proven tech, but mobile Safari is unpredictable)

**2. User Doesn't Reach the CTA**

If players drop off before exploring all four chambers, the Voyage Gate never opens and no subscription occurs.

- **Impact**: Critical (no conversion)
- **Likelihood**: Medium (5-10 minute ask is substantial)

**3. Ambiguity About What VIMANA Is**

Users complete the experience but don't understand they just engaged with a magazine launch.

- **Impact**: High (brand awareness achieved, but no subscription)
- **Likelihood**: Medium (abstract experience requires careful signaling)

**4. AI Splat Quality Doesn't Meet Vision**

Generated splats look "AI-ish" or uncanny, breaking the photorealistic promise.

- **Impact**: Medium (aesthetic damage, but experience still functional)
- **Likelihood**: Low (Marble Labs/World Labs show strong results)

### Technical Challenges

**Mobile Safari Memory Limits**

iOS browsers aggressively tab-squeeze and memory-limit web experiences. Large SOG files can cause crashes.

- **Challenge**: Balancing visual quality with memory constraints
- **Mitigation**: Aggressive LOD variants, mobile-specific splat counts, early memory detection

**WebGPU Adoption Gap**

Not all browsers support WebGPU yet. Fallback to WebGL2 must be seamless.

- **Challenge**: Two rendering paths increase complexity
- **Mitigation**: Feature detection on load, graceful degradation, transparent fallback

**Splat File Loading Times**

40MB+ files take time to load, especially on mobile connections.

- **Challenge**: Perceived load time before experience begins
- **Mitigation**: Progressive loading, visual preloaders, background asset streaming

**Cross-Device Consistency**

Experience must feel equivalent on desktop (mouse) and mobile (touch).

- **Challenge**: Different input paradigms, screen sizes, performance profiles
- **Mitigation**: Responsive UI, device-specific tuning, extensive device farm testing

### Market Risks

**"Too Weird" for Mainstream Audience**

The abstract, wordless, contemplative nature may alienate users expecting traditional magazine marketing.

- **Risk**: Niche appeal limits total addressable audience
- **Counter**: Target audience isn't "everyone"—it's culturally curious early adopters

**Viral Fatigue**

Even if users share, the experience doesn't convert to sustained interest in the magazine itself.

- **Risk**: One-time wonder, no subscriber retention
- **Counter**: First issue content must deliver on the experience's promise

**Web3/Base Integration Backlash**

Some users may reject the crypto/Base integration as extractive or "grifty."

- **Risk**: Brand damage from association with speculative crypto culture
- **Counter**: Position Base as infrastructure, not speculative tool—focus on reader access, not speculation

**Timing Wrong for Magazine Launch**

Launch timing may coincide with economic downturn, media contraction, or attention saturation.

- **Risk**: Harder to convert free experience to paid subscription
- **Counter**: Magazine launch timing is flexible; experience can remain as evergreen brand introduction

### Mitigation Strategies

**Performance Mitigations:**

| Strategy | Implementation |
|----------|----------------|
| **Device Profiling** | Detect capabilities on load, serve appropriate splat quality |
| **Frame Rate Monitoring** | Dynamic quality adjustment if FPS drops below threshold |
| **Aggressive Culling** | Don't render off-screen splats; portal culling between zones |
| **Loading Management** | Progressive streaming, background asset fetch |

**Conversion Mitigations:**

| Strategy | Implementation |
|----------|----------------|
| **Progress Feedback** | Visual indicators showing chambers visited (X/4) |
| **Gate Preview** | Tease Voyage Gate early so players know what they're working toward |
| **Early Exit Options** | Allow "soft exit" with reminder to return, reducing frustration |
| **Post-Experience Landing** | Clear explainer page after ascension: "You just experienced VIMANA magazine" |

**Brand Clarity Mitigations:**

| Strategy | Implementation |
|----------|----------------|
| **Magazine Branding** | Subtle VIMANA logo placement throughout, not intrusive |
| **Post-Ascension Context** | Landing page explains what was experienced and invites subscription |
| **Social Preview Assets** | Share cards clearly communicate "magazine launch experience" |
| **Press Kit** | Journalist-facing materials explaining the concept clearly |

**Quality Mitigations:**

| Strategy | Implementation |
|----------|----------------|
| **Early Splat Tests** | Generate sample splats before full commitment |
| **Hybrid Fallback** | Traditional 3D for elements splats can't handle well |
| **Style Embrace** | If splats have distinctive AI "look," embrace it as aesthetic choice |
| **Iterative Testing** | Regular playtests across target device profiles |

---

## Success Criteria

### MVP Definition

**Minimum Viable Product = Launch-Ready Magazine Experience**

The VIMANA experience is MVP when it delivers:

| Component | MVP Requirement |
|-----------|-----------------|
| **Prologue** | Global orb emergence, Kolkata boy disappearance, transfer to first-person POV |
| **Four Chambers** | Four explorable modules (Art/Nature/Culture/Tech), visitable in any order |
| **Voyage Gate** | Sealed hub chamber that opens after all four chambers visited |
| **Ascension** | Vimana flies through water → sky → space → void sequence |
| **Subscription CTA** | Post-ascension landing with subscription option |
| **Performance** | 60fps on desktop, 30fps+ on target mobile devices |
| **Browser Support** | Chrome/Edge (WebGPU), Safari/Firefox (WebGL2 fallback) |
| **Base Integration** | Mini app functional, subscription flow complete |

**What MVP Does NOT Include:**
- Extended replayability features
- Multiple language support (can be post-launch)
- VR/AR variants (future consideration)
- Social sharing features beyond basic URL sharing

### Success Metrics

**[FUTURE: To be defined with marketing/business stakeholders]**

Preliminary metric categories:

| Metric Category | Examples |
|----------------|----------|
| **Acquisition** | Unique visitors, traffic sources, share rate |
| **Engagement** | Completion rate (% who reach ascension), time spent, chamber visit distribution |
| **Conversion** | Subscription rate, cost per acquisition |
| **Technical** | Frame rate distribution, crash rate by device/browser, load times |
| **Sentiment** | Social mentions, press coverage, Awwwards submission |

**Target Success Indicators:**
- **Completion Rate**: >40% reach the Voyage Gate (industry benchmark for narrative web experiences: ~20-30%)
- **Share Rate**: >10% of visitors share link (viral coefficient >1.0)
- **Press Coverage**: Features in 3+ design/tech publications (Dezeen, It's Nice That, Core77, etc.)

### Launch Goals

**[FUTURE: To be defined with marketing/business stakeholders]**

**Primary Launch Goal:**

Establish VIMANA magazine as a culturally significant publication through a launch experience that:

1. **Creates Cultural Impact**
   - "Have you seen that VIMANA thing?" becomes a topic in creative circles
   - Experience is shared on design Twitter, creative Discords, Awwwards community
   - Press coverage frames VIMANA as the future of magazine launches

2. **Builds Subscriber Base**
   - Convert experience completers into first-issue subscribers
   - Build email list for ongoing engagement
   - Establish pricing model and perceived value

3. **Validates Technology & Aesthetic**
   - Proves Gaussian Splatting as viable web publishing medium
   - Establishes cyber-organic aesthetic as brand signature
   - Creates reusable patterns for future VIMANA digital content

**Secondary Launch Goals:**

| Goal | Description |
|------|-------------|
| **Industry Recognition** | Awwwards SOTD, FWA, Webby nominations |
| **Technical Leadership** | Conference talks, case studies on the tech stack |
| **Community Building** | Discord/community formation around the magazine ethos |
| **Foundation for Issue 2** | Learn from analytics to improve next issue's experience |

**Launch Readiness Checklist:**

- [ ] All four chambers visually distinct and thematically coherent
- [ ] Ascension sequence emotionally satisfying
- [ ] Subscription flow functional and tested
- [ ] Base mini app integration working
- [ ] Performance targets met across device profiles
- [ ] Accessibility features implemented (captions, reduced motion)
- [ ] Press kit prepared
- [ ] Share preview assets created
- [ ] Analytics instrumentation complete
- [ ] Post-experience landing page designed

---

## Next Steps

### Immediate Actions

**1. Begin Game Design Document (GDD) Creation**

Use the completed Game Brief as input for the GDD workflow. The GDD will expand on:

- Detailed chamber designs (each module's unique aesthetic and narrative beats)
- State management architecture (GameManager flow for all narrative states)
- Scene organization (splat loading/unloading zones)
- Audio implementation details (specific tracks, timing, crossfade points)

**Command:** `/bmad:bmgd:workflows:gdd`

**2. Asset Planning & Splat Generation**

- Identify real-world locations or build physical maquettes for splat capture
- Test Marble Labs / Luma AI generation pipeline
- Establish quality targets for each device profile
- Create concept art for chamber interiors as splat generation guides

**3. Audio Design Brief**

- Commission or create ambient chamber themes
- Define ascension score emotional arc
- Identify voiceover requirements (if any)
- Select sound designer/composer

**4. Technical Architecture Specification**

- Define GameManager state flow (45 states as per Shadow Engine pattern)
- Map trigger zones to state transitions
- Specify Base integration points
- Plan performance optimization approach

### Research Needs

**[Items marked for future exploration]**

- [ ] Detailed age range breakdown (18-35 vs 40s) — preferences, engagement patterns
- [ ] Specific publications/communities target audience already loves
- [ ] Refinement of discovery channels (Base mini app, social, direct traffic)
- [ ] Subscription pricing research — what will the market bear?
- [ ] Competitor deep-dive: How do other magazines launch digitally?

### Open Questions

**[Questions requiring stakeholder input or future resolution]**

| Question | Stakeholder | Priority |
|----------|-------------|----------|
| What is the subscription price point? | Business/Mktg | High |
| Should prologue/ascension have voice narration? | Creative | Medium |
| What is the timeline for first issue release? | Editorial | High |
| Are there geographic restrictions on subscriptions? | Business/Legal | High |
| What happens after ascension? Return to start or exit? | Creative | Medium |
| Should the experience be replayable after subscription? | Product | Low |

---

## Appendices

### A. Research Summary

**Document Sources Consulted:**

1. **Shadow Engine Documentation**
   - Tech Stack Overview (Three.js, SparkJS, Rapier, Howler, Vite)
   - Gaussian Splatting Explained (SOG format, performance considerations)
   - GameManager Deep Dive (State-driven architecture, event system)
   - SceneManager Documentation (Zone-based loading/unloading)

2. **Miro Board Context** (board ID: uXjVJ494rik=)
   - Style guide: Deep ocean navy, spectral cyan, bioluminescent accents
   - Technical specifications: Scene controller, module system, voyage gate
   - Functional requirements: Four pillars, ascension sequence
   - Full narrative script (prologue → awakening → exploration → ascension)

3. **Market References**
   - Awwwards SOTD winners (shareable web experiences)
   - World Labs / Marble Labs (AI splat generation)
   - Shadow of the Czar (own proof of concept)

### B. Stakeholder Input

**Mehul (Project Owner)** — Provided:
- Initial vision correction: VIMANA is a magazine, this is its launch experience
- Miro board with style guide, technical specs, functional requirements
- Full narrative script with plot details
- Confirmation of cyber-organic aesthetic direction
- Clarification that Vimana itself flies in ascension sequence
- Instruction to develop four chambers as story progression points (detailed in GDD)

**Party Mode Agent Perspectives** (Step 4 Review):
- **Samus Shepard (Game Designer)**: Validated emotional pillars, questioned ascension agency (resolved: Vimana flies itself)
- **Sally (UX Designer)**: Emphasized emotional resonance, raised brand clarity considerations
- **Cloud Dragonborn (Game Architect)**: Confirmed solid architectural foundations, suggested guided ascension

### C. References

**Technical References:**
- [SparkJS.dev Documentation](https://sparkjs.dev/docs/) — Gaussian Splatting renderer
- [Three.js Documentation](https://threejs.org/docs/) — 3D graphics framework
- [Shadow Engine Docs](engine-docs/docs/generated/) — Internal engine documentation

**Design References:**
- Journey (thatgamecompany) — Ritual experience design
- Proteus (Ed Key) — Exploration without challenge
- Awwwards SOTD winners — Shareable aesthetic experiences
- NFB's Bear 71 — Documentary web experiences

**Aesthetic References:**
- Deep-sea biology — Bioluminescent environments
- Fungal intelligence — Mycelial networks
- Vedic mythology — Vimana as flying palace
- Solarpunk — Technology grown from nature

---

_This Game Brief serves as the foundational input for Game Design Document (GDD) creation._

_Next Steps: Use `/bmad:bmgd:workflows:gdd` to create detailed game design documentation._

---

**Game Brief Status: COMPLETE**

All 10 steps completed:
- [x] Step 1: Initialization
- [x] Step 2: Game Vision
- [x] Step 3: Target Market
- [x] Step 4: Game Fundamentals
- [x] Step 5: Scope & Constraints
- [x] Step 6: Reference Framework
- [x] Step 7: Content Framework
- [x] Step 8: Art & Audio Direction
- [x] Step 9: Risk Assessment
- [x] Step 10: Success Criteria

---

*Document generated: 2026-01-15*
*Project: VIMANA*
*Workflow: BMAD Game Brief v6.0.0-alpha.23*
