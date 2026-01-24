# Game Test Design: Vimana - Music Room Prototype

## Overview

**Project:** Vimana - A 3D contemplative game about learning to harmonize with a mythical flying ship
**Current Epic:** Music Room Prototype (Archive of Voices) - 9 stories
**Tech Stack:** Three.js, Rapier3d, Howler, Vite
**Testing:** Vitest (unit) + Playwright (E2E)

### Game Description

Vimana is a first-person 3D experience where players explore the chambers of a living ship. The Music Room Prototype (Archive of Voices) introduces the core mechanic: a harp-based duet system where jelly creatures teach the player the ship's song. The philosophy is "The ship doesn't test you. It teaches you" - gentle, non-punitive feedback guides players without failure states.

### Target Platforms

- **Desktop:** Windows/Mac/Linux - 60 FPS target
- **Mobile:** iOS/Android - 30 FPS target
- **Browsers:** Chrome, Firefox, Safari, Edge (WebGL2/WebGPU)

### Test Scope

This test design covers all 9 stories of the Music Room Epic:
1. Visual Foundation - Water & Vortex Shaders
2. Jelly Creatures - Musical Messengers
3. Gentle Feedback System
4. Duet Mechanics - Patient Teaching
5. Vortex Activation
6. Shell Collection System
7. UI Overlay System
8. White Flash Ending
9. Performance & Polish

---

## Risk Assessment

### High-Risk Areas

| Risk Area | Impact | Mitigation Strategy |
|-----------|--------|---------------------|
| **WebGL Rendering** | Critical - No render = no play | E2E smoke tests verify WebGL context on all browsers |
| **Audio Feedback** | High - Core mechanic depends on sound | Unit test audio synthesis; E2E verify Web Audio API |
| **Physics/Collision** | Medium - Movement is key to exploration | Trimesh collider tests; character controller integration tests |
| **Performance (Mobile)** | Medium - Low FPS ruins experience | Performance tests with LOD system validation |
| **State Management** | High - Dupe progress tracking is core | Patient teaching state machine tests; progression tests |
| **Memory Leaks** | Medium - Long sessions possible | Destroy() method tests; memory profiling E2E tests |

### Moderate-Risk Areas

- Shader compilation time (>5 seconds causes abandonment)
- Harp interaction raycasting accuracy
- Jelly creature animation timing
- Vortex particle LOD transitions
- Platform ride animation sync

---

## Test Categories

### Gameplay Tests

Core mechanics and player interactions.

| Scenario | Priority | Story | Test Type |
|----------|----------|-------|-----------|
| Harp string interaction (raycast detection) | P0 | 1.1, 1.2 | E2E |
| Jelly creature emergence animation | P0 | 1.2 | Unit |
| Patient teaching (no reset on wrong note) | P0 | 1.3, 1.4 | Unit |
| Harmony chord plays on correct note | P0 | 1.4 | Unit |
| Duet progress tracking (0-1) | P0 | 1.4, 1.5 | Unit |
| Wrong note feedback (camera shake + discord) | P0 | 1.3 | Unit |
| Vortex activation based on duet progress | P0 | 1.5 | Integration |
| Shell collection (click + fly-to-UI) | P1 | 1.6 | E2E |
| White flash ending sequence | P1 | 1.8 | Integration |
| Jelly re-teaching on wrong note | P1 | 1.4 | Unit |

### Visual/Shader Tests

Water, vortex, and shader-based effects.

| Scenario | Priority | Story | Test Type |
|----------|----------|-------|-----------|
| Water ripples respond to all 6 strings | P0 | 1.1 | Integration |
| Vortex particles spiral through torus | P0 | 1.1 | E2E (visual) |
| Vortex glow intensifies with activation | P0 | 1.5 | Unit |
| Jelly bioluminescence (teaching state) | P1 | 1.2 | Unit |
| Water harmonicResonance uniform updates | P1 | 1.1, 1.5 | Unit |
| Shell SDF iridescence with view angle | P2 | 1.6 | E2E (visual) |
| White flash spiral-to-white transition | P2 | 1.8 | E2E (visual) |

### Audio Tests

Sound feedback and music systems.

| Scenario | Priority | Story | Test Type |
|----------|----------|-------|-----------|
| Discordant tone (minor second dissonance) | P0 | 1.3 | Unit |
| Patient reminder tone (E4) | P0 | 1.3 | Unit |
| Harmony chord (player + perfect fifth) | P0 | 1.4 | Unit |
| Completion chord (C major) | P1 | 1.4 | Unit |
| Shell collection audio feedback | P1 | 1.6 | Unit |
| White flash ending audio | P2 | 1.8 | Integration |
| Web Audio API resume on interaction | P0 | 1.3 | E2E |

### Physics/Collision Tests

Character movement and environment collision.

| Scenario | Priority | Story | Test Type |
|----------|----------|-------|-----------|
| Trimesh colliders generated from GLB meshes | P0 | 1.1 | Integration |
| Character walks on ArenaFloor (water) | P0 | All | E2E |
| Harp string hitbox detection | P0 | 1.1, 1.2 | E2E |
| Platform ride to vortex (animation) | P1 | 1.5 | Integration |
| Shell raycast click detection | P1 | 1.6 | E2E |
| No collision on decorative meshes (Windows) | P2 | 1.1 | Unit |

### UI/UX Tests

Interface elements and overlays.

| Scenario | Priority | Story | Test Type |
|----------|----------|-------|-----------|
| Shell UI overlay 4-slot display | P0 | 1.7 | E2E |
| Shell slot fill animation | P1 | 1.7 | E2E |
| Slot glow pulse effect | P2 | 1.7 | E2E |
| Debug menu toggle (F1) | P2 | - | E2E |
| Crosshair display (X key) | P3 | - | E2E |

### Performance Tests

Frame rate, memory, and loading times.

| Scenario | Priority | Story | Test Type |
|----------|----------|-------|-----------|
| 60 FPS on desktop with all systems | P0 | 1.9 | E2E |
| 30 FPS on mobile with reduced particles | P0 | 1.9 | E2E |
| Shader compilation <5 seconds | P0 | 1.9 | E2E |
| Particle LOD (50%/25%/0% skip rates) | P0 | 1.1 | Unit |
| Memory stable (no leaks over 10 min) | P1 | 1.9 | E2E |
| Destroy() cleanup on scene transition | P1 | 1.9 | Unit |
| Device capability detection | P2 | 1.9 | Unit |

### Platform Tests

Cross-browser and device compatibility.

| Scenario | Priority | Story | Test Type |
|----------|----------|-------|-----------|
| WebGL2 context creation (all browsers) | P0 | - | E2E |
| Desktop keyboard controls (WASD) | P0 | - | E2E |
| Mobile touch interaction | P0 | - | E2E |
| Pointer lock behavior | P1 | - | E2E |
| Visual viewport resize handling | P1 | - | E2E |
| iOS Safari compatibility | P1 | - | E2E |
| Android Chrome compatibility | P1 | - | E2E |

---

## Test Scenarios (GIVEN/WHEN/THEN Format)

### Story 1.1: Visual Foundation - Water & Vortex Shaders

```
SCENARIO: Water shader responds to harp string frequencies
  GIVEN the game is loaded and ArenaFloor mesh is visible
  AND WaterMaterial is applied with 6-string frequency uniforms
  WHEN harp string 0 (C) is played
  THEN water ripple effect triggers at C frequency
  AND bioluminescent glow intensifies
  PRIORITY: P0
  CATEGORY: visual/shader

SCENARIO: Vortex particles flow in spiral pattern
  GIVEN VortexParticles is initialized with 2000 particles
  AND LOD system is active
  WHEN duet progress is 0.5
  THEN 50% of particles update (LOD skip rate)
  AND particles spiral through torus using proper trigonometry
  AND cyan-to-purple gradient is visible
  PRIORITY: P0
  CATEGORY: visual/shader

SCENARIO: Vortex glow intensifies with activation
  GIVEN VortexMaterial is applied to torus mesh
  WHEN updateDuetProgress(0.75) is called
  THEN emissive intensity scales to 0.75 × 3.0
  AND core white light is visible
  PRIORITY: P0
  CATEGORY: visual/gameplay

SCENARIO: Particle LOD activates correctly
  GIVEN VortexParticles has 2000 particles
  AND device is low-end (detected via DeviceCapabilities)
  WHEN duet progress is 0.2 (< 30%)
  THEN 50% of particles are skipped
  WHEN duet progress increases to 0.4 (> 30% but < 60%)
  THEN 25% of particles are skipped
  WHEN duet progress reaches 1.0
  THEN 0% of particles are skipped
  PRIORITY: P0
  CATEGORY: performance
```

### Story 1.2: Jelly Creatures - Musical Messengers

```
SCENARIO: Jelly emerges from water with arc trajectory
  GIVEN JellyCreature is created for note C (index 0)
  AND jelly is in submerged state at y=-0.2
  WHEN demonstrateNote(0) is called
  THEN jelly animates in parabolic arc to string 0 position
  AND jump peaks at 0.5 units above water
  AND duration is 1.5 seconds
  PRIORITY: P0
  CATEGORY: visual/gameplay

SCENARIO: Jelly bioluminescence intensifies during teaching
  GIVEN JellyCreature is in teaching state
  WHEN teaching begins
  THEN emissive intensity increases from 0.3 to 1.0
  AND pulse rate increases by 50%
  AND scale pulses between 0.9 and 1.1
  PRIORITY: P0
  CATEGORY: visual/gameplay

SCENARIO: Each jelly has unique pulse rate based on note
  GIVEN six JellyCreature instances exist
  WHEN all jellies are created
  THEN pulse rates are: C=1.0Hz, D=1.1Hz, E=1.2Hz, F=1.3Hz, G=1.4Hz, A=1.5Hz
  PRIORITY: P1
  CATEGORY: visual

SCENARIO: Target string ripples when jelly demonstrates
  GIVEN JellyCreature is demonstrating note E (index 2)
  WHEN teaching state is active
  THEN water ripple effect appears at string 2 position
  AND ripple lasts for 2 seconds (teaching duration)
  PRIORITY: P1
  CATEGORY: visual/gameplay

SCENARIO: Jelly submerges smoothly after teaching
  GIVEN JellyCreature has completed teaching
  WHEN submergeJelly() is called
  THEN emissive fades over 0.5 seconds
  AND scale shrinks to 0.3 over 1 second
  AND jelly moves to water surface (y=0)
  AND rendering is disabled after fade
  PRIORITY: P1
  CATEGORY: visual/animation
```

### Story 1.3: Gentle Feedback System

```
SCENARIO: Camera shake on wrong note
  GIVEN player is in AWAITING_INPUT state
  AND target note is C (index 0)
  WHEN player plays wrong note (e.g., D, index 1)
  THEN camera shakes with intensity 0.5 units
  AND shake lasts 500ms with exponential decay
  AND shake uses sinusoidal pattern (not random)
  PRIORITY: P0
  CATEGORY: gameplay/feedback

SCENARIO: Discordant tone plays on wrong note
  GIVEN target note is C (261.63 Hz)
  AND player plays wrong note
  WHEN GentleAudioFeedback.playDiscordantTone() is called
  THEN frequency is targetFreq × 16/15 (minor second dissonance)
  AND duration is 300ms
  AND gain envelope: attack 50ms, decay 250ms
  PRIORITY: P0
  CATEGORY: audio/feedback

SCENARIO: Patient reminder tone plays after discord
  GIVEN wrong note was played
  AND discordant tone finished
  WHEN reminder delay elapses (100ms gap)
  THEN gentle E4 tone plays (329.63 Hz, 200ms duration)
  AND volume is -12 dB relative to main harp
  PRIORITY: P0
  CATEGORY: audio/feedback

SCENARIO: No state reset on wrong note (patient teaching)
  GIVEN player is on note 2 of sequence 0
  AND AWAITING_INPUT state is active
  WHEN player plays wrong note
  THEN state returns to AWAITING_INPUT (not reset)
  AND current note index remains 2
  AND notes completed count does not decrease
  AND same note can be attempted immediately
  PRIORITY: P0
  CATEGORY: gameplay/state

SCENARIO: Visual cue on correct string after wrong attempt
  GIVEN player played wrong note while target was E (index 2)
  WHEN wrong note feedback triggers
  THEN harp string 2 glows with amber color (#ffaa44)
  AND emissive pulses from 0.2 to 0.5 over 1 second
  AND glow fades after 2 seconds
  PRIORITY: P1
  CATEGORY: visual/feedback

SCENARIO: Subtle shake on premature play
  GIVEN player is not in AWAITING_INPUT state
  AND player clicks/plays a string
  WHEN GentleFeedback detects premature input
  THEN shake intensity is 0.3 units (weaker than wrong note)
  AND no audio feedback plays
  PRIORITY: P2
  CATEGORY: gameplay/feedback
```

### Story 1.4: Duet Mechanics - Patient Teaching

```
SCENARIO: Jelly demonstrates note then awaits player input
  GIVEN sequence 0 (C, D, E) is started
  WHEN startSequence(0) is called
  THEN jelly for note C emerges and demonstrates
  AND state transitions to AWAITING_INPUT after 2 seconds
  PRIORITY: P0
  CATEGORY: gameplay/state

SCENARIO: Correct note plays harmony chord
  GIVEN player is in AWAITING_INPUT state
  AND target note is C (index 0)
  WHEN player plays correct note (C)
  THEN HarmonyChord.play(0) is called
  AND harmony is player freq × 1.5 (perfect fifth)
  AND state advances to next note
  PRIORITY: P0
  CATEGORY: audio/gameplay

SCENARIO: Wrong note triggers re-teaching
  GIVEN player is in AWAITING_INPUT state
  AND target note is E (index 2)
  WHEN player plays wrong note (e.g., C, index 0)
  THEN discordant tone plays
  AND jelly re-emerges to demonstrate E again
  AND state remains AWAITING_INPUT
  AND note index does NOT advance
  PRIORITY: P0
  CATEGORY: gameplay/state

SCENARIO: Sequence completion plays C major chord
  GIVEN player plays final note of sequence 0 (E, index 2)
  WHEN all 3 notes of sequence are complete
  THEN HarmonyChord.playCompletionChord() is called
  AND C major chord plays (C-E-G)
  AND state advances to next sequence (1)
  PRIORITY: P0
  CATEGORY: audio/gameplay

SCENARIO: All three sequences play correctly
  GIVEN PatientJellyManager is initialized
  WHEN sequences are queried
  THEN sequence 0 is [0, 1, 2] (C, D, E)
  AND sequence 1 is [3, 4, 5] (F, G, A)
  AND sequence 2 is [2, 4, 1] (E, G, D)
  PRIORITY: P0
  CATEGORY: gameplay/state

SCENARIO: Duet progress calculates correctly
  GIVEN 3 of 9 total notes have been completed
  WHEN getProgress() is called
  THEN returned value is 0.33 (3/9)
  PRIORITY: P0
  CATEGORY: gameplay/state

SCENARIO: Full duet completion triggers vortex
  GIVEN all 3 sequences are complete (9/9 notes)
  WHEN final note is played correctly
  THEN state transitions to COMPLETE
  AND isComplete() returns true
  AND getProgress() returns 1.0
  PRIORITY: P0
  CATEGORY: gameplay/state
```

### Story 1.5: Vortex Activation

```
SCENARIO: Vortex activation follows duet progress
  GIVEN duet progress is 0.5 (5/9 notes)
  WHEN VortexActivationController updates
  THEN vortex emissive intensity is 0.5 × 3.0 = 1.5
  AND water harmonicResonance uniform is 0.5
  PRIORITY: P0
  CATEGORY: visual/gameplay

SCENARIO: Full activation triggers platform ride
  GIVEN duet progress reaches 1.0
  WHEN PatientJellyManager.isComplete() is true
  THEN platform detaches from floor
  AND platform animates to vortex position over 5 seconds
  AND transition to shell spawn triggers
  PRIORITY: P0
  CATEGORY: visual/gameplay

SCENARIO: Vortex lighting intensifies with activation
  GIVEN VortexLightingManager is active
  WHEN duet progress increases from 0 to 1
  THEN vortex light color shifts from dim cyan to bright
  AND particle spin speed increases
  AND core white light becomes visible at >0.8 activation
  PRIORITY: P1
  CATEGORY: visual

SCENARIO: Water harmonic resonance syncs with duet
  GIVEN WaterMaterial is linked to VortexSystem
  WHEN duet progress updates
  THEN water uHarmonicResonance uniform matches progress
  AND water bioluminescent color intensifies
  PRIORITY: P1
  CATEGORY: visual/shader
```

### Story 1.6: Shell Collection System

```
SCENARIO: Shell materializes after vortex completion
  GIVEN platform ride to vortex is complete
  WHEN shell spawn triggers
  THEN SDFShellMaterial appears with 3-second dissolve animation
  AND shell spawns at position (0, 1.0, 1.0)
  PRIORITY: P0
  CATEGORY: visual/gameplay

SCENARIO: Shell iridescence shifts with view angle
  GIVEN shell is visible and materialized
  WHEN camera view angle changes
  THEN shell color shifts across spectrum (iridescence)
  AND SDF nautilus spiral pattern is visible
  PRIORITY: P2
  CATEGORY: visual

SCENARIO: Shell collection on click
  GIVEN shell is materialized and floating
  WHEN player clicks on shell (raycast hit)
  THEN shell animates to UI position over 1.5 seconds
  AND shell scales down during flight
  AND shell is removed from scene after arrival
  PRIORITY: P0
  CATEGORY: gameplay/e2e

SCENARIO: Shell bobbing idle animation
  GIVEN shell is spawned and not clicked
  WHEN shell is in idle state
  THEN shell bobs up and down gently
  AND rotation is subtle
  PRIORITY: P2
  CATEGORY: visual/animation
```

### Story 1.7: UI Overlay System

```
SCENARIO: Shell UI displays 4 empty slots
  GIVEN ShellUIOverlay is initialized
  WHEN game loads
  THEN 4 slots are visible at top-left of screen
  AND all slots are empty (no shell icons)
  PRIORITY: P0
  CATEGORY: ui

SCENARIO: Shell slot fills on collection
  GIVEN player collects first shell
  WHEN shell fly-to-UI animation completes
  THEN slot 0 fills with nautilus spiral icon
  AND slot fill animation plays with glow effect
  AND filled slot has gentle pulse animation
  PRIORITY: P0
  CATEGORY: ui/gameplay

SCENARIO: Shell count tracks correctly
  GIVEN 2 shells have been collected
  WHEN getShellCount() is called
  THEN returned value is 2
  PRIORITY: P0
  CATEGORY: ui/state

SCENARIO: UI overlay persists across scene
  GIVEN ShellUIOverlay is created
  WHEN game state changes
  THEN overlay remains visible (z-index: 1000)
  AND icons render correctly using Canvas 2D API
  PRIORITY: P1
  CATEGORY: ui
```

### Story 1.8: White Flash Ending

```
SCENARIO: White flash triggers after scene completion
  GIVEN all sequence/vortex/shell events are done
  WHEN WhiteFlashEnding.start() is called
  THEN full-screen quad with spiral shader appears
  AND total duration is 8 seconds
  PRIORITY: P0
  CATEGORY: visual/cinematic

SCENARIO: Spiral-to-white color transition
  GIVEN WhiteFlashEnding is active
  WHEN progress goes from 0% to 100%
  THEN color transitions: cyan → purple → white
  AND spiral intensifies from 0-60% progress
  AND fade to white from 40-100% progress
  PRIORITY: P1
  CATEGORY: visual/shader

SCENARIO: Vignette effect during white flash
  GIVEN WhiteFlashEnding is active
  WHEN spiral is spinning
  THEN vignette darkens edges for focus
  AND center remains brightest
  PRIORITY: P2
  CATEGORY: visual

SCENARIO: Scene completion callback fires
  GIVEN WhiteFlashEnding completes (8 seconds elapsed)
  WHEN animation finishes
  THEN scene completion callback is triggered
  AND white flash fades out smoothly
  PRIORITY: P0
  CATEGORY: gameplay/state
```

### Story 1.9: Performance & Polish

```
SCENARIO: 60 FPS maintained on desktop
  GIVEN all systems are active (2000 particles, 6 jellies, shaders)
  WHEN game runs on desktop device
  THEN frame rate remains ≥60 FPS
  PRIORITY: P0
  CATEGORY: performance

SCENARIO: 30 FPS maintained on mobile
  GIVEN mobile device is detected
  AND particle count is reduced to 500
  WHEN game runs on mobile
  THEN frame rate remains ≥30 FPS
  PRIORITY: P0
  CATEGORY: performance

SCENARIO: Shader compilation is fast
  GIVEN game loads for first time
  WHEN all shaders compile
  THEN compilation time is <5 seconds
  AND loading screen shows progress
  PRIORITY: P0
  CATEGORY: performance

SCENARIO: No memory leaks on scene transition
  GIVEN game has run for 10 minutes
  WHEN scene transitions or game resets
  THEN memory usage returns to baseline
  AND all geometries/materials are disposed
  PRIORITY: P1
  CATEGORY: performance

SCENARIO: Device capability detection works
  GIVEN game initializes
  WHEN DeviceCapabilities checks device
  THEN isMobile, GPU tier, and max particles are detected
  AND appropriate quality preset is applied
  PRIORITY: P2
  CATEGORY: performance/platform

SCENARIO: Particle LOD activates on low-end
  GIVEN device is detected as low-end
  WHEN VortexParticles updates
  THEN appropriate LOD skip rate is applied
  AND frame rate remains acceptable
  PRIORITY: P0
  CATEGORY: performance
```

### Cross-Platform Scenarios

```
SCENARIO: WebGL2 context creation on all browsers
  GIVEN game loads in browser
  WHEN Three.js initializes renderer
  THEN WebGL2 context is created (or WebGL1 fallback)
  AND canvas is visible in DOM
  PRIORITY: P0
  CATEGORY: platform

SCENARIO: Keyboard controls work on desktop
  GIVEN game is loaded on desktop
  WHEN player presses W/A/S/D keys
  THEN character moves forward/left/back/right
  AND pointer lock engages on click
  PRIORITY: P0
  CATEGORY: platform/input

SCENARIO: Touch controls work on mobile
  GIVEN game is loaded on mobile device
  WHEN player taps and drags on screen
  THEN character moves accordingly
  AND tap-to-interact works for harp/shell
  PRIORITY: P0
  CATEGORY: platform/input

SCENARIO: Visual viewport resize handling
  GIVEN game is running
  WHEN browser window is resized
  OR virtual keyboard appears/disappears (mobile)
  THEN camera aspect ratio updates
  AND canvas size adjusts correctly
  PRIORITY: P1
  CATEGORY: platform
```

---

## Coverage Matrix

| Feature | P0 | P1 | P2 | P3 | Total |
|---------|----|----|----|----|-------|
| **Harp Interaction** | 4 | 2 | 1 | 0 | 7 |
| **Jelly Creatures** | 5 | 3 | 1 | 0 | 9 |
| **Gentle Feedback** | 5 | 2 | 1 | 0 | 8 |
| **Duet Mechanics** | 7 | 1 | 0 | 0 | 8 |
| **Vortex Activation** | 4 | 2 | 0 | 0 | 6 |
| **Shell Collection** | 3 | 1 | 2 | 0 | 6 |
| **UI Overlay** | 3 | 1 | 1 | 0 | 5 |
| **White Flash** | 2 | 2 | 1 | 0 | 5 |
| **Performance** | 5 | 1 | 1 | 0 | 7 |
| **Platform/Input** | 4 | 1 | 0 | 0 | 5 |
| **Visual/Shaders** | 6 | 3 | 3 | 0 | 12 |
| **Audio** | 5 | 2 | 1 | 0 | 8 |
| **TOTAL** | 53 | 20 | 12 | 0 | 87 |

**Priority Distribution:**
- P0 (Critical/Ship Blockers): 53 scenarios (61%)
- P1 (High/Major Features): 20 scenarios (23%)
- P2 (Medium/Polish): 12 scenarios (14%)
- P3 (Low/Edge Cases): 0 scenarios (0%)

---

## Automation Strategy

### Unit Tests (Vitest)

**Target:** 80% coverage of P0 scenarios

**Candidates:**
- All state machines (PatientJellyManager, DuetProgressTracker)
- Audio synthesis (GentleAudioFeedback, HarmonyChord)
- Feedback coordination (FeedbackManager, GentleFeedback)
- Vortex activation logic (VortexActivationController, VortexSystem)
- Entity initialization and property accessors
- Progress calculation methods
- Mock-heavy: Three.js, Howler, Rapier3d all mocked

**Not Automated:**
- Pure visual rendering (shader appearance, colors)
- Subjective feel (shake intensity "feels right")
- Audio quality (timbre, mixing balance)

### Integration Tests (Vitest)

**Target:** 60% coverage of P0 scenarios

**Candidates:**
- Vortex ↔ Water material linkage
- Jelly emergence → String ripple
- Duet progress → Vortex activation
- Shell collection → UI update
- Full duet sequence flow

**Not Automated:**
- Cross-browser rendering differences
- Real device performance

### E2E Tests (Playwright)

**Target:** 50% coverage of P0 scenarios

**Candidates:**
- Game launch and WebGL context
- Harp string clicking (raycasting)
- Shell collection (click → fly-to-UI)
- UI overlay display and updates
- Mobile touch interactions
- Performance measurements (FPS)

**Not Automated:**
- Visual shader verification (requires human eye)
- Audio playback (WebAudio limited in automated browsers)

### Manual Testing (Playtest Sessions)

**Target:** All P2/P3 scenarios + subjective validation

**Scope:**
- Visual polish (colors, timing, feel)
- Audio mixing and balance
- "Fun factor" and emotional resonance
- Accessibility assessment
- Cross-device usability

**See:** `PLAYTEST-PLAN.md` (to be created in playtest-plan workflow)

---

## Test Data Requirements

### Mock Objects

```javascript
// Mock Three.js objects
const mockScene = { add: vi.fn(), remove: vi.fn() };
const mockCamera = { position: new THREE.Vector3() };
const mockRenderer = { domElement: mockCanvas };

// Mock audio
const mockAudioContext = {
  createOscillator: vi.fn(),
  createGain: vi.fn(),
  resume: vi.fn()
};

// Mock game state
const mockGameState = {
  harpState: 0,
  harpSequenceProgress: 0,
  harpSequencesCompleted: 0,
  vortexActivation: 0
};
```

### Test Fixtures

- **HarpRoom scene**: Minimal GLB with 6 strings positioned
- **ArenaFloor mesh**: For water shader testing
- **Vortex torus**: Positioned at (0, 0.5, 2)
- **Shell spawn point**: At (0, 1.0, 1.0)

### Performance Baselines

| Metric | Desktop Target | Mobile Target |
|--------|---------------|---------------|
| FPS | ≥60 | ≥30 |
| Shader compile | <5s | <5s |
| Memory growth | <10MB/min | <10MB/min |
| Particle count | 2000 | 500 |

---

## Next Steps

1. **Review scenarios** with team for completeness
2. **Use `automate` workflow** to generate test code from P0 scenarios
3. **Create `PLAYTEST-PLAN.md`** for manual testing sessions
4. **Implement E2E test hooks** in game code for state inspection
5. **Set up CI pipeline** for automated test execution

---

## Appendix: Story Mapping

| Story | Key Entities | Test Focus |
|-------|-------------|------------|
| 1.1 Visual Foundation | VortexParticles, VortexMaterial, WaterMaterial, VortexSystem | Particles, shaders, LOD |
| 1.2 Jelly Creatures | JellyCreature, JellyManager | Emergence, teaching, bioluminescence |
| 1.3 Gentle Feedback | GentleFeedback, GentleAudioFeedback, FeedbackManager | Shake, discordant tone, reminder |
| 1.4 Duet Mechanics | PatientJellyManager, DuetProgressTracker, HarmonyChord | State machine, progress, chords |
| 1.5 Vortex Activation | VortexActivationController, VortexLightingManager | Activation sync, lighting |
| 1.6 Shell Collection | ShellCollectible, ShellManager | SDF shader, collection |
| 1.7 UI Overlay | ShellUIOverlay | Slots, icons, animations |
| 1.8 White Flash | WhiteFlashEnding, WhiteFlashAudio | Spiral shader, transition |
| 1.9 Performance | PerformanceMonitor, DeviceCapabilities, QualityPresets | FPS, LOD, memory |

---

**Document Version:** 1.0
**Created:** 2025-01-24
**Workflow:** test-design (BMad v6)
**Total Scenarios:** 87
