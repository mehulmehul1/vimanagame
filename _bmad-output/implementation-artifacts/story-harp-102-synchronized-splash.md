# STORY-HARP-102: Synchronized Splash System

**Epic**: `HARP-ENHANCEMENT` - Harp Minigame Design Alignment
**Story ID**: `STORY-HARP-102`
**Points**: `2`
**Status**: `Ready for Dev`
**Owner**: `TBD`
**Related:** Sprint Change Proposal 2026-01-29
**Depends On:** STORY-HARP-101

---

## User Story

As a **player**, I want **all jellyfish to splash down together after the demonstration**, so that **I have a clear "your turn" signal to begin playing the phrase**.

---

## Overview

Implement the synchronized splash effect where all jellyfish that demonstrated the phrase fall back into the water simultaneously. This creates a unified visual and audio cue that signals the transition from Vimana's turn to player's turn.

**Reference:** `vimana_harp_minigame_design.md` - Phase 2: "Synchronized Landing (Turn Signal)"

---

## Background

**Design Intent:**
> "All jellyfish fall back into the water at the SAME TIME. A single unified splash occurs. The splash is not part of the sequence."
>
> "Clearly signals: 'The Vimana's phrase is complete. Your turn.'"

**Current State:**
- Individual jellyfish submerge one at a time
- No unified "turn signal" effect
- Unclear when player should respond

---

## Technical Specification

### Splash Trigger Method

Add to `PatientJellyManager.ts`:

```typescript
/**
 * Trigger synchronized splash - all jellies submerge together
 *
 * This is the TURN SIGNAL that clearly communicates:
 * "The Vimana's phrase is complete. Your turn."
 *
 * @param completedSequence - Array of note indices that were demonstrated
 */
private triggerSynchronizedSplash(completedSequence: number[]): void {
    this.state = DuetState.TURN_SIGNAL;

    // Submerge all active jellies simultaneously
    for (const noteIndex of completedSequence) {
        this.jellyManager.submergeJelly(noteIndex);
    }

    // Visual: All jellies descend together
    // Audio: Single unified splash (not multiple splashes)

    // Play unified splash sound
    this.harmonyChord.playSplashSound();

    // Trigger visual splash effect at water surface
    if (this.callbacks.onSynchronizedSplash) {
        this.callbacks.onSynchronizedSplash(completedSequence);
    }

    // Transition to awaiting player response
    const SPLASH_DURATION = 1000; // 1 second for splash animation
    setTimeout(() => {
        this.state = DuetState.AWAITING_PHRASE_RESPONSE;
        this.currentNoteIndex = 0;  // Reset to start of phrase for player

        if (this.callbacks.onTurnBegins) {
            this.callbacks.onTurnBegins();
        }
    }, SPLASH_DURATION);
}
```

### JellyManager Submerge Method

Add to `JellyManager.ts`:

```typescript
/**
 * Submerge a specific jelly (make it descend into water)
 *
 * @param stringIndex - Which jelly to submerge (0-5)
 */
public submergeJelly(stringIndex: number): void {
    const jelly = this.jellies[stringIndex];
    if (!jelly) return;

    // Trigger descent animation
    jelly.submerge();

    // Animation: Smooth descent to water surface over ~500ms
    // Jelly position goes from airborne → at/below water surface
    // Visual: Jelly becomes semi-transparent as it enters water
}
```

### HarmonyChord Splash Sound

Add to `HarmonyChord.ts`:

```typescript
/**
 * Play unified splash sound for synchronized landing
 *
 * Design note: This should be a SINGLE unified sound,
 * not multiple splash sounds layered.
 */
public playSplashSound(): void {
    const context = this.audioContext;

    // Create a gentle, unified splash sound
    const splash = this.createSplashBuffer();

    const source = context.createBufferSource();
    source.buffer = splash;

    const gainNode = context.createGain();
    gainNode.gain.setValueAtTime(0.3, context.currentTime); // Gentle volume
    gainNode.gain.exponentialDecayTo(0.01, context.currentTime + 0.8);

    source.connect(gainNode);
    gainNode.connect(this.masterGain);

    source.start();
}

/**
 * Generate splash sound buffer
 */
private createSplashBuffer(): AudioBuffer {
    const sampleRate = this.audioContext.sampleRate;
    const duration = 0.8;
    const buffer = this.audioContext.createBuffer(1, sampleRate * duration, sampleRate);
    const data = buffer.getChannelData(0);

    // Pink noise + lowpass filter = water splash
    for (let i = 0; i < data.length; i++) {
        const t = i / sampleRate;
        // Envelope: quick attack, exponential decay
        const envelope = Math.exp(-t * 5);
        // Noise with lowpass characteristic
        data[i] = (Math.random() * 2 - 1) * envelope * 0.3;
    }

    return buffer;
}
```

### Visual Splash Effect

Create new file `src/entities/SynchronizedSplashEffect.ts`:

```typescript
/**
 * Visual effect for synchronized jellyfish splash
 *
 * Creates expanding ripple rings from all jelly positions
 * simultaneously, emphasizing the "unified" nature of the turn signal.
 */
export class SynchronizedSplashEffect {
    private ripples: SplashRipple[] = [];
    private active: boolean = false;

    /**
     * Trigger synchronized splash effect
     *
     * @param positions - World positions of all jellies splashing
     */
    trigger(positions: THREE.Vector3[]): void {
        this.active = true;
        this.ripples = [];

        // Create ripple at each jelly position
        for (const pos of positions) {
            this.ripples.push({
                position: pos.clone(),
                radius: 0.1,
                maxRadius: 2.5,
                alpha: 1.0,
                age: 0
            });
        }

        console.log(`[SynchronizedSplash] Triggered with ${positions.length} ripples`);
    }

    update(deltaTime: number): void {
        if (!this.active) return;

        for (const ripple of this.ripples) {
            ripple.age += deltaTime;
            ripple.radius = THREE.MathUtils.lerp(0.1, ripple.maxRadius, ripple.age / 0.8);
            ripple.alpha = 1.0 - (ripple.age / 0.8);
        }

        // Remove finished ripples
        this.ripples = this.ripples.filter(r => r.age < 0.8);

        if (this.ripples.length === 0) {
            this.active = false;
        }
    }

    render(renderer: THREE.WebGLRenderer): void {
        // Render expanding ring quads at each ripple position
        // Use instanced rendering for efficiency
    }
}

interface SplashRipple {
    position: THREE.Vector3;
    radius: number;
    maxRadius: number;
    alpha: number;
    age: number;
}
```

---

## Implementation Tasks

1. **[METHOD]** Add `triggerSynchronizedSplash()` to PatientJellyManager
2. **[METHOD]** Add `submergeJelly()` to JellyManager
3. **[AUDIO]** Add `playSplashSound()` to HarmonyChord
4. **[VISUAL]** Create SynchronizedSplashEffect class
5. **[TIMING]** Set splash duration to 1 second before player turn
6. **[CALLBACK]** Add onSynchronizedSplash and onTurnBegins callbacks
7. **[STATE]** Add TURN_SIGNAL state to DuetState enum

---

## File Structure

```
src/entities/
├── PatientJellyManager.ts (add triggerSynchronizedSplash)
├── JellyManager.ts (add submergeJelly)
├── SynchronizedSplashEffect.ts (new file)
└── Jelly.ts (add submerge method)

src/audio/
└── HarmonyChord.ts (add playSplashSound)
```

---

## Sequence Diagram

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│ PatientJelly│         │  JellyManager│         │ HarmonyChord│
└──────┬──────┘         └──────┬───────┘         └──────┬──────┘
       │                       │                        │
       │ triggerSynchronizedSplash(sequence)          │
       ├───────────────────────┐│                       │
       │                       ││                       │
       │    submergeJelly(0)   ││                       │
       ├──────────────────────>││                       │
       │    submergeJelly(1)   ││                       │
       ├──────────────────────>││                       │
       │    submergeJelly(2)   ││                       │
       ├──────────────────────>││                       │
       │                       ││                       │
       │    playSplashSound()  ││                       │
       ├──────────────────────────────────────────────>│
       │                       ││                       │
       │    <SPLASH VISUAL>    ││                       │
       ├───────────────────────────────────────────────┤
       │                       ││                       │
       │    [1 second passes]  ││                       │
       │                       ││                       │
       │    state = AWAITING_PHRASE_RESPONSE           │
       │                       ││                       │
       │    onTurnBegins()     ││                       │
       ├───────────────────────┴───────────────────────┤
       │                       │                        │
```

---

## Acceptance Criteria

- [ ] All jellies submerge simultaneously (not one-by-one)
- [ ] Submerge animation completes in ~500ms
- [ ] Single unified splash sound plays (not multiple overlapping splashes)
- [ ] Visual ripple effect appears at all jelly positions
- [ ] State transitions to AWAITING_PHRASE_RESPONSE after splash
- [ ] Total turn signal duration is ~1 second
- [ ] Player can begin playing after signal completes
- [ ] Callback fires for UI/story integration

---

## Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| STORY-HARP-101 | Phrase-First Mode | Required |
| JellyManager.ts | Existing | ✅ Ready |
| HarmonyChord.ts | Existing | ✅ Ready |
| PatientJellyManager.ts | Existing | ✅ Ready |

---

## Testing

**Manual Test Cases:**
1. Complete phrase demonstration → all jellies splash together
2. Count splash sounds → exactly 1 sound (not 3 separate splashes)
3. Time the splash → ~1 second total duration
4. After splash → player can play harp immediately
5. Visual ripples → appear at all jelly positions

**Edge Cases:**
- Splash triggered with 0 jellies (should not crash)
- Splash triggered during player input (should be ignored)
- Multiple splash triggers in quick succession

---

## Audio Specification

| Parameter | Value | Description |
|-----------|-------|-------------|
| Duration | 0.8s | Length of splash sound |
| Attack | 0.05s | Quick attack for realistic splash |
| Decay | 0.75s | Exponential decay |
| Volume | 0.3 | Gentle, not overwhelming |
| Filter | Lowpass | Pink noise → water-like |
| Frequency | <2kHz | Low-frequency emphasis |

---

## Visual Specification

| Parameter | Value | Description |
|-----------|-------|-------------|
| Ripple Count | 3 | One per jelly in sequence |
| Start Radius | 0.1 | Small ring at splash point |
| Max Radius | 2.5 | Expands to this size |
| Duration | 0.8s | Matches audio |
| Color | Cyan/White | Bioluminescent splash |
| Alpha | 1.0 → 0 | Fade out |

---

## Notes

**Design Philosophy:**
- The splash is NOT part of the musical sequence
- It is a clear SIGNAL, not gameplay content
- All movement synchronized emphasizes "togetherness"
- Single unified sound = one entity acting, not many

**Performance:**
- 3 simultaneous ripple meshes
- Each: 64 vertices
- Total: <200 vertices, negligible impact

---

**Sources:**
- `vimana_harp_minigame_design.md` (Synchronized Landing section)
- Sprint Change Proposal 2026-01-29
- Audio design principles for water sounds
