# STORY-HARP-101: Phrase-First Teaching Mode

**Epic**: `HARP-ENHANCEMENT` - Harp Minigame Design Alignment
**Story ID**: `STORY-HARP-101`
**Points**: `5`
**Status**: `Ready for Dev`
**Owner**: `TBD`
**Related**: Sprint Change Proposal 2026-01-29

---

## User Story

As a **player**, I want **the Vimana to demonstrate the full musical phrase before I respond**, so that **I experience a true musical duet where I must remember and replay the melody**.

---

## Overview

Implement phrase-first teaching mode where all jellyfish emerge sequentially to demonstrate the full phrase, then wait for the player to remember and replay the entire sequence. This aligns the implementation with the original design specification.

**Reference:** `vimana_harp_minigame_design.md` - Phase 2: The Vimana Plays (Jellyfish Phase)

---

## Background

**Current State:**
- Jellyfish teach one note at a time
- Player plays each note immediately
- No memory component to the interaction

**Desired State:**
- All jellyfish for the phrase emerge together
- Player watches full demonstration
- Player must remember and replay the phrase
- Creates a "call and response" duet experience

---

## Technical Specification

### State Machine Extensions

Add to `PatientJellyManager.ts`:

```typescript
export enum DuetState {
    // Existing states
    IDLE = 'IDLE',
    DEMONSTRATING = 'DEMONSTRATING',
    AWAITING_INPUT = 'AWAITING_INPUT',
    PLAYING_HARMONY = 'PLAYING_HARMONY',
    COMPLETE = 'COMPLETE',

    // NEW: Phrase-first mode states
    PHRASE_DEMONSTRATION = 'PHRASE_DEMONSTRATION',  // All jellies showing sequence
    TURN_SIGNAL = 'TURN_SIGNAL',                    // Synchronized splash
    AWAITING_PHRASE_RESPONSE = 'AWAITING_PHRASE_RESPONSE'  // Player's turn
}

export interface TeachingModeConfig {
    mode: 'phrase-first' | 'note-by-note';
    showAllJellies: boolean;
    synchronizedSplash: boolean;
}
```

### Phrase Demonstration Flow

```typescript
/**
 * Start phrase-first demonstration (all jellies show sequence)
 */
public startPhraseFirstSequence(sequenceIndex: number): void {
    this.state = DuetState.PHRASE_DEMONSTRATION;
    this.currentSequence = sequenceIndex;
    this.currentNoteIndex = 0;
    this.teachingMode = 'phrase-first';

    const sequence = TEACHING_SEQUENCES[sequenceIndex];

    // Spawn ALL jellies for the sequence at once
    for (let i = 0; i < sequence.length; i++) {
        const noteIndex = sequence[i];

        // Spawn jelly at target string position
        this.jellyManager.spawnJelly(noteIndex);

        // Stagger their emergence slightly for visual clarity
        setTimeout(() => {
            this.jellyManager.beginTeaching(noteIndex);
            this.harmonyChord.playDemonstrationNote(noteIndex, 1.5);

            // Trigger visual indicator
            if (this.callbacks.onNoteDemonstrated) {
                this.callbacks.onNoteDemonstrated(noteIndex, i);
            }
        }, i * 800); // 800ms between each jelly
    }

    // After all notes demonstrated, trigger synchronized splash
    const totalDemoTime = sequence.length * 800 + 1500;
    setTimeout(() => {
        this.triggerSynchronizedSplash(sequence);
    }, totalDemoTime);
}
```

### Configuration Integration

```typescript
export class PatientJellyManager {
    private teachingMode: 'phrase-first' | 'note-by-note' = 'phrase-first';
    private config: TeachingModeConfig = {
        mode: 'phrase-first',
        showAllJellies: true,
        synchronizedSplash: true
    };

    /**
     * Set teaching mode
     */
    public setTeachingMode(mode: 'phrase-first' | 'note-by-note'): void {
        this.teachingMode = mode;
        this.config.mode = mode;
        this.config.showAllJellies = (mode === 'phrase-first');
        this.config.synchronizedSplash = (mode === 'phrase-first');
    }
}
```

---

## Implementation Tasks

1. **[STATE]** Add PHRASE_DEMONSTRATION, TURN_SIGNAL, AWAITING_PHRASE_RESPONSE to DuetState enum
2. **[METHOD]** Implement `startPhraseFirstSequence()` method
3. **[CONFIG]** Add TeachingModeConfig interface and setTeachingMode() method
4. **[JELLY]** Update JellyManager to handle multiple active jellies simultaneously
5. **[TIMING]** Implement staggered emergence (800ms intervals between jellies)
6. **[CALLBACK]** Add onNoteDemonstrated callback for each note in sequence
7. **[FALLBACK]** Ensure existing note-by-note mode remains functional

---

## File Structure

```
src/entities/PatientJellyManager.ts
├── DuetState enum (extended)
├── TeachingModeConfig interface (new)
├── startPhraseFirstSequence() method (new)
├── setTeachingMode() method (new)
└── Existing methods (preserved)
```

---

## Integration Points

**HarpRoom.ts changes:**
```typescript
export class HarpRoom {
    private teachingMode: 'phrase-first' | 'note-by-note' = 'phrase-first';

    private setupManagers(): void {
        // ... existing code ...

        // Set teaching mode on jelly manager
        this.patientJellyManager.setTeachingMode(this.teachingMode);

        // Updated callback for phrase demonstration
        this.patientJellyManager.setCallbacks({
            // ... existing callbacks ...
            onNoteDemonstrated: (noteIndex: number, sequenceIndex: number) => {
                console.log(`Note ${noteIndex} demonstrated (#${sequenceIndex + 1} in phrase)`);
                this.showSequenceIndicator(noteIndex, sequenceIndex);
            }
        });
    }
}
```

---

## Acceptance Criteria

- [ ] New DuetState values added without breaking existing states
- [ ] `startPhraseFirstSequence()` spawns all jellies for current sequence
- [ ] Jellies emerge with 800ms stagger for visual clarity
- [ ] Each jelly plays its demonstration note during emergence
- [ ] Full sequence completes before turn signal
- [ ] `setTeachingMode()` switches between phrase-first and note-by-note
- [ ] Existing note-by-note mode remains functional
- [ ] No memory leaks from multiple active jellies

---

## Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| PatientJellyManager.ts | Existing code | ✅ Implemented |
| JellyManager.ts | Existing code | ✅ Implemented |
| HarmonyChord.ts | Existing code | ✅ Implemented |
| STORY-HARP-102 | Synchronized Splash | Required (next story) |

---

## Testing

**Manual Test Cases:**
1. Set mode to 'phrase-first', start sequence → all jellies spawn
2. Count spawned jellies → equals sequence length (3 for first sequence)
3. Time between jellies → ~800ms stagger
4. Each jelly plays note → correct pitch for string
5. Switch to 'note-by-note' → original behavior restored

**Edge Cases:**
- Starting new sequence while one is in progress
- Switching modes during active sequence
- Maximum sequence length handling

---

## Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `teachingMode` | 'phrase-first' | 'phrase-first' \| 'note-by-note' | Active teaching mode |
| `jellyStaggerMs` | 800 | 500-1500 | Delay between jelly emergence |
| `demoNoteDuration` | 1.5 | 1.0-3.0 | Seconds each note plays |

---

## Notes

**Design Philosophy:**
- Preserves existing functionality (no breaking changes)
- Phrase-first mode becomes the design-aligned default
- Note-by-note remains available as easier alternative

**Performance:**
- Multiple active jellies: ~3 max (sequence length)
- Each jelly: ~500 vertices
- Total overhead: <2000 vertices, negligible impact

---

**Sources:**
- `vimana_harp_minigame_design.md` (Phase 2)
- Sprint Change Proposal 2026-01-29
- Existing `PatientJellyManager.ts` implementation
