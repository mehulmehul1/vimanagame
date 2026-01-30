# STORY-HARP-103: Phrase Response Handling

**Epic**: `HARP-ENHANCEMENT` - Harp Minigame Design Alignment
**Story ID**: `STORY-HARP-103`
**Points**: `3`
**Status**: `Ready for Dev`
**Owner:** `TBD`
**Related:** Sprint Change Proposal 2026-01-29
**Depends On:** STORY-HARP-101, STORY-HARP-102

---

## User Story

As a **player**, I want **to play back the full musical phrase from memory**, so that **I complete the musical duet by matching the Vimana's melody**.

---

## Overview

Implement player response handling for phrase-first mode. After the synchronized splash signals the player's turn, the player must remember and replay the entire sequence. Correct completion advances to the next sequence; wrong notes trigger patient redemonstration.

**Reference:** `vimana_harp_minigame_design.md` - Phase 3: Player Response

---

## Background

**Design Intent:**
> "The player must reproduce: The same note sequence in the same order"
> "What Does NOT Matter: Timing precision, Speed of playing, Holding notes"
> "Players are allowed to take their time"

**Current State:**
- Note-by-note mode validates each note immediately
- No phrase-level memory component

**Desired State:**
- Player plays full sequence from memory
- Each note validated against expected position in phrase
- Full phrase completion triggers progression

---

## Technical Specification

### Player Input Handler

Modify `PatientJellyManager.ts`:

```typescript
/**
 * Handle player input (harp string played)
 * Works for both phrase-first and note-by-note modes
 */
public handlePlayerInput(playedNoteIndex: number): void {
    // Log for debugging
    console.log(`[PatientJellyManager] handlePlayerInput: note=${playedNoteIndex}, state=${this.state}`);

    // Ignore if complete
    if (this.state === DuetState.COMPLETE) {
        console.log('[PatientJellyManager] Ignoring input - duet complete');
        return;
    }

    // PHRASE-FIRST MODE: Validate against current expected note in sequence
    if (this.state === DuetState.AWAITING_PHRASE_RESPONSE) {
        this.handlePhraseFirstInput(playedNoteIndex);
        return;
    }

    // NOTE-BY-NOTE MODE: Existing logic (unchanged)
    if (this.state === DuetState.AWAITING_INPUT) {
        this.handleNoteByNoteInput(playedNoteIndex);
        return;
    }

    // Allow playing during demonstration (skip to player's turn)
    if (this.state === DuetState.PHRASE_DEMONSTRATION) {
        console.log('[PatientJellyManager] Player played during phrase demo, ending demo');
        this.endDemonstrationEarly();
        this.state = DuetState.AWAITING_PHRASE_RESPONSE;
        this.currentNoteIndex = 0;
        this.handlePhraseFirstInput(playedNoteIndex);
    }
}

/**
 * Handle input during phrase-first response phase
 */
private handlePhraseFirstInput(playedNoteIndex: number): void {
    this.progressTracker.recordAttempt();

    const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

    console.log(`[Phrase-First] Expecting note ${targetNote}, played ${playedNoteIndex} (${this.currentNoteIndex + 1}/${TEACHING_SEQUENCES[this.currentSequence].length})`);

    if (playedNoteIndex === targetNote) {
        this.handleCorrectNoteInPhrase();
    } else {
        this.handleWrongNoteInPhrase(targetNote, playedNoteIndex);
    }
}

/**
 * Handle correct note within phrase response
 */
private handleCorrectNoteInPhrase(): void {
    const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

    // Play individual note confirmation (subtle feedback)
    this.harmonyChord.playNoteConfirmation(targetNote);

    // Trigger visual feedback
    this.feedbackManager.triggerCorrectNote(targetNote);

    // Callback
    if (this.callbacks.onNoteComplete) {
        this.callbacks.onNoteComplete(this.currentSequence, this.currentNoteIndex);
    }

    // Advance to next note in phrase
    this.currentNoteIndex++;
    const currentSequence = TEACHING_SEQUENCES[this.currentSequence];

    if (this.currentNoteIndex >= currentSequence.length) {
        // Whole phrase completed correctly!
        this.completePhrase();
    } else {
        console.log(`[Phrase-First] Correct! ${this.currentNoteIndex}/${currentSequence.length} complete. Next note expected.`);
    }
}

/**
 * Handle wrong note during phrase response
 */
private handleWrongNoteInPhrase(targetNote: number, playedNote: number): void {
    console.log(`[Phrase-First] Wrong note! Expected ${targetNote}, played ${playedNote}`);

    // Gentle feedback (shake + discordant + highlight)
    this.feedbackManager.triggerWrongNote(targetNote);

    // Callback
    if (this.callbacks.onWrongNote) {
        this.callbacks.onWrongNote(targetNote, playedNote);
    }

    // IMPORTANT: Patient teaching - redemonstrate the full phrase
    // No punishment, no progress lost - gentle retry
    setTimeout(() => {
        console.log('[Phrase-First] Redemonstrating phrase...');
        this.startPhraseFirstSequence(this.currentSequence);
    }, this.config.wrongNoteDelay * 1000);
}

/**
 * Phrase completed successfully
 */
private completePhrase(): void {
    console.log(`[Phrase-First] Sequence ${this.currentSequence} complete!`);

    // Play full harmony chord (ship joins in)
    this.harmonyChord.playCompletionChord();

    // Mark progress
    this.progressTracker.markSequenceComplete(this.currentSequence);

    // Trigger visual celebration
    this.feedbackManager.triggerSequenceComplete(this.currentSequence);

    // Callback
    if (this.callbacks.onSequenceComplete) {
        this.callbacks.onSequenceComplete(this.currentSequence);
    }

    // Move to next sequence or complete duet
    if (this.currentSequence >= TEACHING_SEQUENCES.length - 1) {
        this.completeDuet();
    } else {
        setTimeout(() => {
            this.currentSequence++;
            this.startPhraseFirstSequence(this.currentSequence);
        }, this.config.sequenceDelay * 1000);
    }
}

/**
 * Complete entire duet
 */
private completeDuet(): void {
    this.state = DuetState.COMPLETE;
    console.log('[Phrase-First] DUET COMPLETE!');

    if (this.callbacks.onDuetComplete) {
        this.callbacks.onDuetComplete();
    }
}

/**
 * Existing note-by-note handler (preserved, renamed)
 */
private handleNoteByNoteInput(playedNoteIndex: number): void {
    // [existing code from current handlePlayerInput]
    // Preserved for note-by-note fallback mode
}
```

---

## Implementation Tasks

1. **[REFACTOR]** Rename existing `handlePlayerInput()` logic to `handleNoteByNoteInput()`
2. **[METHOD]** Implement `handlePhraseFirstInput()` method
3. **[METHOD]** Implement `handleCorrectNoteInPhrase()` method
4. **[METHOD]** Implement `handleWrongNoteInPhrase()` method
5. **[METHOD]** Implement `completePhrase()` method
6. **[LOGGING]** Add detailed console logging for phrase progression
7. **[FALLBACK]** Ensure note-by-note mode continues to work

---

## State Transition Diagram

```
                    ┌─────────────────────────────────────────┐
                    │         AWAITING_PHRASE_RESPONSE         │
                    │     (Player's turn - remember phrase)    │
                    └─────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              Correct Note         Wrong Note          All Notes
                    │                    │                  Complete
                    ▼                    ▼                    ▼
            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
            │ Next Note    │    │ Redemonstrate │    │ Phrase       │
            │ in Sequence  │    │ Full Phrase   │    │ Complete     │
            └──────────────┘    └──────────────┘    └──────────────┘
                    │                    │                    │
                    │                    └──>──────────────────┤
                    │                                         │
                    ▼                                         ▼
            [Continue Playing]                     [Next Sequence /
                                                     Duet Complete]
```

---

## File Structure

```
src/entities/PatientJellyManager.ts
├── handlePlayerInput() (modified - routes to mode-specific handler)
├── handlePhraseFirstInput() (new)
├── handleNoteByNoteInput() (renamed from existing logic)
├── handleCorrectNoteInPhrase() (new)
├── handleWrongNoteInPhrase() (new)
├── completePhrase() (new)
└── endDemonstrationEarly() (new helper)
```

---

## Integration Points

**HarpRoom.ts callback updates:**
```typescript
this.patientJellyManager.setCallbacks({
    // ... existing callbacks ...

    onNoteComplete: (seq, note) => {
        console.log(`Note ${note + 1} of sequence ${seq} complete`);
        // Visual feedback for individual note in phrase
        this.noteVisualizer?.showNote(note, 0.3);
    },

    onSequenceComplete: (seq) => {
        console.log(`Full sequence ${seq} complete!`);
        // Celebration effect
        this.noteVisualizer?.showChord([0, 2, 4], 2.0); // C major visual
        this.updateGameStateOnSequenceComplete();
    },

    onWrongNote: (target, played) => {
        console.log(`Wrong note: played ${played}, target ${target}`);
        this.triggerWrongNote(target);
    }
});
```

---

## Acceptance Criteria

- [ ] `handlePlayerInput()` routes to correct handler based on state
- [ ] Correct notes in sequence advance `currentNoteIndex`
- [ ] Wrong note triggers redemonstration of full phrase
- [ ] Completing full phrase triggers `onSequenceComplete`
- [ ] All 3 sequences complete triggers `onDuetComplete`
- [ ] Note-by-note mode remains functional via `handleNoteByNoteInput()`
- [ ] Console logging shows progression through phrase
- [ ] No memory leaks on redemonstration

---

## Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| STORY-HARP-101 | Phrase-First Mode | Required |
| STORY-HARP-102 | Synchronized Splash | Required |
| FeedbackManager.ts | Existing | ✅ Ready |
| HarmonyChord.ts | Existing | ✅ Ready |
| ProgressTracker.ts | Existing | ✅ Ready |

---

## Testing

**Manual Test Cases:**

1. **Correct Sequence:**
   - Demo: C-D-E → Splash → Play: C-D-E → Sequence Complete

2. **Wrong Note - Redemonstration:**
   - Demo: C-D-E → Splash → Play: C-F → Redemonstrate C-D-E

3. **Note-by-Note Fallback:**
   - Set mode to 'note-by-note' → Original behavior

4. **Early Playing During Demo:**
   - Demo starts → Play immediately → Demo ends, player's turn begins

5. **Full Duet Completion:**
   - Complete all 3 sequences → Duet complete event fires

**Edge Cases:**
- Playing wrong note multiple times
- Playing rapidly during phrase response
- Starting new sequence while one is active

---

## Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `wrongNoteDelay` | 1.0 | 0.5-3.0 | Seconds before redemonstration |
| `sequenceDelay` | 2.0 | 1.0-5.0 | Seconds between sequences |
| `wrongNoteAction` | 'redemonstrate' | 'redemonstrate' \| 'continue' | What happens on wrong note |

---

## Debug Logging

Expected console output during phrase-first play:

```
[PatientJellyManager] handlePlayerInput: note=0, state=AWAITING_PHRASE_RESPONSE
[Phrase-First] Expecting note 0, played 0 (1/3)
[Phrase-First] Correct! 1/3 complete. Next note expected.

[PatientJellyManager] handlePlayerInput: note=1, state=AWAITING_PHRASE_RESPONSE
[Phrase-First] Expecting note 1, played 1 (2/3)
[Phrase-First] Correct! 2/3 complete. Next note expected.

[PatientJellyManager] handlePlayerInput: note=2, state=AWAITING_PHRASE_RESPONSE
[Phrase-First] Expecting note 2, played 2 (3/3)
[Phrase-First] Sequence 0 complete!
```

Wrong note scenario:
```
[PatientJellyManager] handlePlayerInput: note=3, state=AWAITING_PHRASE_RESPONSE
[Phrase-First] Expecting note 0, played 3 (1/3)
[Phrase-First] Wrong note! Expected 0, played 3
[Phrase-First] Redemonstrating phrase...
```

---

## Notes

**Design Philosophy:**
- No punishment for wrong notes - only gentle redemonstration
- Player can take their time - no timing pressure
- Progress through phrases feels like learning, not passing/failing
- Each correct note in phrase provides subtle confirmation feedback

**Memory Considerations:**
- Sequences are 3 notes each (manageable memory load)
- Visual feedback helps track progress (1/3, 2/3, 3/3)
- Redemonstration is immediate and clear on wrong note

---

**Sources:**
- `vimana_harp_minigame_design.md` (Player Response section)
- Sprint Change Proposal 2026-01-29
- Existing `PatientJellyManager.ts` implementation
