---
title: 'Music Room Proto Epic - Archive of Voices'
project: 'VIMANA'
date: '2026-01-24'
author: 'Mehul'
version: '1.0'
status: 'in-progress'
epic_number: 1
---

# Epic 1: Music Room Prototype - Archive of Voices

## Overview

**Epic Goal:** Implement the Archive of Voices chamber - the Culture/Music room of Vimana where players learn to harmonize with the ship through a duet-based harp interaction.

**Location:** Harp Room (Archive of Voices)
**Narrative Role:** "This is Vimana's communication hub—how cultures stay connected across distance."
**Player Experience:** The ship teaches you its song through jelly creatures, you join the duet, harmony opens the vortex

**Philosophy:**
> "The ship doesn't test you. It teaches you. When you play wrong, it doesn't fail you. It just sings again, more slowly, more clearly, until you can join in harmony. This is a duet—a shared moment of music-making."

**Duration Target:** ~6 minutes of gameplay

---

## Stories

### Story 1.1: Visual Foundation - Water & Vortex Shaders

**Status:** ready-for-review
**Estimated:** Day 1
**Dependencies:** None

**Description:**
Apply enhanced visual shaders to the water surface and create the SDF torus vortex with particle system. This establishes the visual foundation for the entire chamber.

**Acceptance Criteria:**
- [ ] Water shader applied to ArenaFloor with bioluminescent resonance
- [ ] Water ripples respond to all 6 string frequencies individually
- [ ] SDF torus vortex mesh created at proper position (0, 0.5, 2)
- [ ] Vortex particle system with 2000 particles (LOD-enabled)
- [ ] Particles flow in spiral pattern through torus
- [ ] Glow intensifies based on activation level (0-1)
- [ ] 60 FPS maintained with all systems active

**Technical Implementation:**
- EnhancedWaterMaterial with 6-string frequency uniforms
- VortexParticles class with LOD system (50%/25%/0% skip rates)
- SDF torus shader with activation-based animation
- Bioluminescent color response to harp vibrations

---

### Story 1.2: Jelly Creatures - Musical Messengers

**Status:** ready-for-review
**Estimated:** Day 2
**Dependencies:** Story 1.1

**Description:**
Create procedural jelly creatures that emerge from the water to demonstrate the ship's song. These are "musical messengers" that teach the player which strings to play.

**Acceptance Criteria:**
- [ ] JellyCreature class with enhanced shader material
- [ ] Organic pulsing animation (different rate per note)
- [ ] Bioluminescent glow that intensifies when teaching
- [ ] Jump-out animation from water surface
- [ ] Target string ripple visualization
- [ ] Smooth emergence and submersion

**Technical Implementation:**
- Procedural jelly geometry (sphere-based with displacement)
- Enhanced jelly shader with teaching state uniforms
- Jump animation with arc trajectory
- Water ripple effect on target string

---

### Story 1.3: Gentle Feedback System

**Status:** ready-for-review
**Estimated:** Day 3
**Dependencies:** Story 1.2

**Description:**
Implement the gentle feedback system for wrong notes - camera shake + discordant tone + patient re-teaching. No failure states, only guidance.

**Acceptance Criteria:**
- [ ] GentleFeedback class with camera shake
- [ ] GentleAudioFeedback with discordant/reminder sounds
- [ ] Shake intensity: 0.5 for wrong note, 0.3 for premature play
- [ ] Discordant tone uses minor second dissonance
- [ ] Patient reminder plays after wrong note (gentle E4)
- [ ] No reset - player stays on same note

**Technical Implementation:**
- Camera shake with decay (500ms duration)
- Web Audio API oscillators for sounds
- Gain envelopes for smooth attack/release
- No state machine reset on wrong input

---

### Story 1.4: Duet Mechanics - Patient Teaching

**Status:** ready-for-review
**Estimated:** Day 4
**Dependencies:** Story 1.3

**Description:**
Implement the core duet mechanic where jelly creatures teach note sequences and the player joins in harmony. Three teaching sequences with patient re-teaching on mistakes.

**Acceptance Criteria:**
- [ ] PatientJellyManager class with three sequences
- [ ] Sequence 1: C, D, E (strings 0, 1, 2)
- [ ] Sequence 2: F, G, A (strings 3, 4, 5)
- [ ] Sequence 3: E, G, D (strings 2, 4, 1)
- [ ] Jelly demonstrates → player plays → harmony chord or gentle correction
- [ ] No failure - jelly reappears to demonstrate again
- [ ] Harmony chord plays on correct note (player + perfect fifth)

**Technical Implementation:**
- Teaching sequence state machine
- 2-second demonstration phase per note
- Harmony chord synthesis (player freq × 1.5)
- Completion chord (C major) on sequence finish
- DuetProgressTracker for harmony scoring

---

### Story 1.5: Vortex Activation

**Status:** ready-for-review
**Estimated:** Day 4 (paired with 1.4)
**Dependencies:** Story 1.4

**Description:**
Connect duet progress to vortex activation. As harmony increases, the vortex becomes more defined and the ship's "voice" grows stronger.

**Acceptance Criteria:**
- [ ] Vortex activation follows duet progress (0-1)
- [ ] Vortex emissive intensity scales: 0 → 3.0
- [ ] Water harmonicResonance uniform updates
- [ ] Particle spin speed increases with activation
- [ ] Core white light intensifies at full activation
- [ ] Platform detaches and rides to vortex on completion

**Technical Implementation:**
- Duet progress to vortex material uniforms
- Platform animation with lerp to vortex position
- 5-second platform ride animation
- Transition to shell spawn on arrival

---

### Story 1.6: Shell Collection System

**Status:** ready-for-review
**Estimated:** Day 5
**Dependencies:** Story 1.5

**Description:**
Create the procedural SDF shell collectible that appears after vortex completion. 3-second materialize animation, click-to-collect, fly-to-UI animation.

**Acceptance Criteria:**
- [ ] SDFShellMaterial with nautilus spiral SDF
- [ ] 3-second appear animation with easing
- [ ] Iridescence shifts with view angle
- [ ] Raycast click detection
- [ ] 1.5-second fly-to-UI animation with shrink
- [ ] Shell spawns at (0, 1.0, 1.0) in front of player
- [ ] Bobbing idle animation

**Technical Implementation:**
- Nautilus spiral SDF formula
- Simplex noise for dissolve effect
- Raycaster for mouse interaction
- Smoothstep easing for animations
- Canvas-drawn shell icons for UI

---

### Story 1.7: UI Overlay System

**Status:** ready-for-review
**Estimated:** Day 6
**Dependencies:** Story 1.6

**Description:**
Create the 4-slot UI overlay showing collected shells. Progress tracking across the 4 rooms with animated slot filling.

**Acceptance Criteria:**
- [ ] ShellUIOverlay with 4 slots at top-left
- [ ] Canvas-drawn nautilus spiral icons
- [ ] Slot fill animation with glow effect
- [ ] Gentle pulse animation on filled slots
- [ ] Shell count tracking (0-4)
- [ ] Global window.shellUI registration

**Technical Implementation:**
- Fixed position CSS overlay (z-index: 1000)
- Canvas 2D API for shell icon drawing
- CSS transitions for slot animations
- Event system integration with shell system

---

### Story 1.8: White Flash Ending

**Status:** ready-for-review
**Estimated:** Day 7
**Dependencies:** Story 1.7

**Description:**
Implement the vortex engulf ending - shader-based white flash with spiral pattern that fades to pure white. No 3D white room needed.

**Acceptance Criteria:**
- [ ] WhiteFlashEnding class with full-screen quad
- [ ] 8-second total duration
- [ ] Phase 1 (0-60%): Multi-layered spiral intensifies
- [ ] Phase 2 (40-100%): Fade to white
- [ ] Color transition: cyan → purple → white
- [ ] Vignette effect for focus
- [ ] Smooth alpha fade at end
- [ ] Scene completion callback

**Technical Implementation:**
- Full-screen quad shader with spiral SDF
- Multiple rotational symmetry layers
- Progress-based color mixing
- Camera-parented mesh for proper positioning
- Cleanup method for memory management

---

### Story 1.9: Performance & Polish

**Status:** in-dev
**Estimated:** Day 7 (paired with 1.8)
**Dependencies:** Story 1.8

**Description:**
Final performance optimization, shader preloading, mobile fallbacks, and overall polish pass.

**Acceptance Criteria:**
- [ ] Async shader loading system implemented
- [ ] 60 FPS on desktop, 30 FPS mobile
- [ ] Memory cleanup on scene transitions
- [ ] Device capability detection
- [ ] Performance monitor integration
- [ ] Mobile particle count fallback (500 max)
- [ ] All shaders compiled in < 5 seconds

**Technical Implementation:**
- ShaderLoader class with promise-based compilation
- VortexParticles destroy() method
- PatientJellyManager destroy() method
- DeviceCapabilities detection
- PerformanceMonitor with FPS tracking
- LOD system active on low-end devices

---

## Epic Retrospective

**Status:** optional
**To be completed after:** All stories reach "done" status

**Retrospective Questions:**
- What worked well in the duet-based teaching mechanic?
- Were players able to understand the jelly demonstrations?
- Was the gentle feedback system appropriately non-punitive?
- Did the vortex activation feel earned?
- How was the shell collection satisfaction?
- Did the white flash ending provide transcendent closure?
- Any performance issues on target devices?
- What would improve for future chambers?

---

## Success Metrics

**Visual Quality:**
- [ ] Water responds to all 6 strings individually
- [ ] Vortex particles flow smoothly in spiral
- [ ] Shell iridescence shifts with view angle
- [ ] White flash has smooth spiral-to-white transition

**Gameplay Feel:**
- [ ] Wrong note feels gentle, not punitive
- [ ] Jelly demonstrations are clear
- [ ] No reset - player stays on same note
- [ ] Harmony buildup feels rewarding
- [ ] Shell collection is satisfying
- [ ] Ending feels transcendent

**Performance:**
- [ ] 60 FPS desktop, 30 FPS mobile
- [ ] Shader compilation < 5 seconds
- [ ] Memory stable across device tiers
- [ ] Smooth state transitions
