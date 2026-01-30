# Sprint Change Proposal: Harp Minigame Implementation Alignment

**Date:** 2026-01-29
**Project:** shadowvimana (VIMANA)
**Status:** Ready for Review
**Scope:** Minor
**Classification:** Implementation Adjustment

---

## 1. Issue Summary

### Problem Statement

The harp music interaction implementation differs from the original design specification in `_bmad-output/vimana_harp_minigame_design.md`. The current code implements a **note-by-note teaching system** while the design specifies a **phrase-first memory duet**.

### Discovery Context

- **When:** 2026-01-29 during code review comparing design to implementation
- **How:** Comparison between design doc and `src/scenes/HarpRoom.ts`, `src/entities/PatientJellyManager.ts`
- **Evidence:** Concrete implementation differences documented below

### Key Deviations

| Design Specification | Current Implementation | Impact |
|---------------------|----------------------|--------|
| **Full sequence shown first** - All jellyfish jump out, player watches entire phrase, then player responds | **Note-by-note teaching** - One jellyfish appears, teaches one note, player plays that note, then next jelly appears | Reduces memory challenge significantly |
| **Synchronized splash turn signal** - All jellyfish fall back together to signal player's turn | **Individual submergence** - Each jellyfish submerges after its individual note | Missing clear "your turn" signal |
| **Memory test: 3-4 note phrase** - Player must remember and reproduce sequence | **Simplified: One note at a time** - Player only needs to match immediate note | Easier than designed experience |

---

## 2. Impact Analysis

### Epic Impact

| Epic | Status | Impact | Details |
|------|--------|--------|---------|
| EPIC-1: Music Room Prototype | Done | None | Stories completed, feature functional |
| EPIC-2: WaterBall Fluid | Done | None | No interaction |
| EPIC-4: WebGPU Migration | In-Progress | None | No interaction |
| Future Epics | Planned | None | No dependencies |

### Story Impact

| Story | Status | Change Required |
|-------|--------|-----------------|
| STORY-005: Harp-to-Water Interaction | Ready for Dev | None - water physics unaffected |
| PatientJellyManager.ts | Implemented | New sub-story needed for phrase-first mode |
| HarpRoom.ts | Implemented | Integration adjustments needed |

### Artifact Conflicts

| Artifact | Conflict | Resolution Needed |
|----------|----------|-------------------|
| `vimana_harp_minigame_design.md` | ⚠️ YES | Implementation does not match design |
| GDD (Chamber: Archive of Voices) | ⚠️ YES | Describes "duet, not a test" - current easier mode may not achieve this |
| Architecture | ✅ No conflicts | Code structure is sound |
| UI/UX | ⚠️ Minor | Visual feedback system exists but may need updates |

### Technical Impact

- **Code Changes Required:** Medium
- **Breaking Changes:** None - additive changes only
- **Performance Impact:** Negligible
- **Testing Required:** Medium - new flow path to validate

---

## 3. Recommended Approach: Direct Adjustment

### Selected Option: Option 1 - Direct Adjustment

**Rationale:**
1. **Code foundation is solid** - PatientJellyManager is well-structured and extensible
2. **No rollback needed** - Current implementation works, just needs enhancement
3. **Incremental delivery** - Can add phrase-first mode as enhancement
4. **Low risk** - Changes are additive, don't break existing functionality
5. **Maintains timeline** - Fits within existing sprint structure

### Effort Estimate

| Component | Estimate | Complexity |
|-----------|----------|------------|
| Phrase demonstration state | 4 hours | Medium |
| Synchronized splash system | 2 hours | Low |
| Player response phase handling | 4 hours | Medium |
| Visual feedback updates | 2 hours | Low |
| Testing and validation | 3 hours | Medium |
| **Total** | **15 hours** | **Medium** |

### Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Breaking existing flow | Low | Add as new mode, keep current as fallback |
| Performance impact | Low | State machine already efficient |
| UX confusion | Medium | Clear visual distinction between modes |

---

## 4. Detailed Change Proposals

### 4.1 PatientJellyManager.ts Changes

**File:** `src/entities/PatientJellyManager.ts`

**OLD (Current):**
```typescript
export enum DuetState {
    IDLE = 'IDLE',
    DEMONSTRATING = 'DEMONSTRATING',
    AWAITING_INPUT = 'AWAITING_INPUT',
    PLAYING_HARMONY = 'PLAYING_HARMONY',
    COMPLETE = 'COMPLETE'
}
```

**NEW:**
```typescript
export enum DuetState {
    IDLE = 'IDLE',
    // Phrase-first mode states
    PHRASE_DEMONSTRATION = 'PHRASE_DEMONSTRATION',  // All jellies show sequence
    TURN_SIGNAL = 'TURN_SIGNAL',                    // Synchronized splash
    AWAITING_PHRASE_RESPONSE = 'AWAITING_PHRASE_RESPONSE', // Player repeats full phrase
    // Note-by-note mode (existing, kept as fallback)
    DEMONSTRATING = 'DEMONSTRATING',
    AWAITING_INPUT = 'AWAITING_INPUT',
    // Shared states
    PLAYING_HARMONY = 'PLAYING_HARMONY',
    COMPLETE = 'COMPLETE'
}

export interface TeachingModeConfig {
    mode: 'phrase-first' | 'note-by-note';  // NEW
    showAllJellies: boolean;                // NEW
    synchronizedSplash: boolean;            // NEW
}
```

**Rationale:** Add phrase-first mode while preserving existing note-by-note as fallback option.

---

### 4.2 Demonstration Flow Changes

**NEW Method in PatientJellyManager:**

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
        this.jellyManager.spawnJelly(noteIndex);

        // Stagger their emergence slightly for visual clarity
        setTimeout(() => {
            this.jellyManager.beginTeaching(noteIndex);
            this.harmonyChord.playDemonstrationNote(noteIndex, 1.5);
        }, i * 800); // 800ms between each jelly
    }

    // After all notes demonstrated, trigger synchronized splash
    const totalDemoTime = sequence.length * 800 + 1500;
    setTimeout(() => {
        this.triggerSynchronizedSplash(sequence);
    }, totalDemoTime);
}

/**
 * Trigger synchronized splash - all jellies submerge together
 */
private triggerSynchronizedSplash(completedSequence: number[]): void {
    this.state = DuetState.TURN_SIGNAL;

    // Submerge all active jellies simultaneously
    for (const noteIndex of completedSequence) {
        this.jellyManager.submergeJelly(noteIndex);
    }

    // Play unified splash sound
    this.harmonyChord.playSplashSound();

    // Transition to awaiting player response
    setTimeout(() => {
        this.state = DuetState.AWAITING_PHRASE_RESPONSE;
        this.currentNoteIndex = 0;  // Reset to start of phrase for player
    }, 1000);
}
```

**Rationale:** Implements the designed "show full phrase first" behavior with synchronized splash as turn signal.

---

### 4.3 Player Response Handling

**MODIFIED Method in PatientJellyManager:**

```typescript
/**
 * Handle player input (works for both modes)
 */
public handlePlayerInput(playedNoteIndex: number): void {
    if (this.state === DuetState.COMPLETE) return;

    // PHRASE-FIRST MODE: Check against current expected note in sequence
    if (this.state === DuetState.AWAITING_PHRASE_RESPONSE) {
        const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

        if (playedNoteIndex === targetNote) {
            // Correct note in sequence
            this.handleCorrectNoteInPhrase();
        } else {
            // Wrong note - must restart phrase
            this.handleWrongNoteInPhrase(targetNote, playedNoteIndex);
        }
        return;
    }

    // NOTE-BY-NOTE MODE: Existing logic (unchanged)
    if (this.state === DuetState.AWAITING_INPUT) {
        // [existing code unchanged]
    }
}

/**
 * Handle correct note within phrase response
 */
private handleCorrectNoteInPhrase(): void {
    // Play individual note confirmation
    this.harmonyChord.playNoteConfirmation(this.currentNoteIndex);

    this.currentNoteIndex++;
    const currentSequence = TEACHING_SEQUENCES[this.currentSequence];

    if (this.currentNoteIndex >= currentSequence.length) {
        // Whole phrase completed correctly!
        this.completePhrase();
    }
}

/**
 * Handle wrong note during phrase response
 */
private handleWrongNoteInPhrase(targetNote: number, playedNote: number): void {
    // Gentle feedback
    this.feedbackManager.triggerWrongNote(targetNote);

    // Redemonstrate the full phrase (patient teaching)
    setTimeout(() => {
        this.startPhraseFirstSequence(this.currentSequence);
    }, 1000);
}

/**
 * Phrase completed successfully
 */
private completePhrase(): void {
    // Play full harmony chord (ship joins in)
    this.harmonyChord.playCompletionChord();

    // Mark progress
    this.progressTracker.markSequenceComplete(this.currentSequence);

    // Callback
    if (this.callbacks.onSequenceComplete) {
        this.callbacks.onSequenceComplete(this.currentSequence);
    }

    // Move to next sequence or complete duet
    if (this.currentSequence >= TEACHING_SEQUENCES.length - 1) {
        this.state = DuetState.COMPLETE;
        if (this.callbacks.onDuetComplete) {
            this.callbacks.onDuetComplete();
        }
    } else {
        setTimeout(() => {
            this.currentSequence++;
            this.startPhraseFirstSequence(this.currentSequence);
        }, 2000);
    }
}
```

**Rationale:** Handles phrase-first response where player must remember and replay full sequence.

---

### 4.4 HarpRoom.ts Integration

**File:** `src/scenes/HarpRoom.ts`

**CHANGE:**
```typescript
// NEW: Configure teaching mode
private teachingMode: 'phrase-first' | 'note-by-note' = 'phrase-first';

/**
 * Start the duet with configured teaching mode
 */
private startDuet(): void {
    if (this.teachingMode === 'phrase-first') {
        this.patientJellyManager.startPhraseFirstSequence(0);
    } else {
        this.patientJellyManager.start();  // Existing note-by-note
    }
}
```

**Rationale:** Allows easy switching between modes via configuration.

---

### 4.5 Visual Feedback Updates

**NEW: Enhanced Visual Cues for Phrase Mode**

```typescript
// In onDemonstrationStart for phrase mode
if (this.teachingMode === 'phrase-first') {
    // Show all note positions simultaneously with indicators
    for (let i = 0; i < sequence.length; i++) {
        const noteIndex = sequence[i];
        this.noteVisualizer.showNote(noteIndex, 1.5);

        // Show sequence number above each jelly
        this.jellyLabels.showLabel(noteIndex, `${i + 1}`);
    }
}

// In synchronized splash
if (this.state === DuetState.TURN_SIGNAL) {
    // Visual ripple emanating from all jellies simultaneously
    this.triggerSynchronizedSplashEffect();
}
```

**Rationale:** Visual indicators help player understand sequence order during phrase demonstration.

---

### 4.6 Design Document Update

**File:** `_bmad-output/vimana_harp_minigame_design.md`

**ADD section:**

```markdown
## Implementation Notes

### Teaching Modes

The harp minigame supports two teaching modes:

1. **Phrase-First Mode (Default)** - Matches original design:
   - All jellyfish demonstrate sequence first
   - Synchronized splash signals player's turn
   - Player must remember and replay full phrase
   - Tests memory of melodic sequences

2. **Note-by-Note Mode (Fallback)** - Current implementation:
   - One jellyfish at a time teaches individual notes
   - Player plays each note immediately
   - Easier, more accessible experience
   - Better for very young players or accessibility

**Configuration:** Set `teachingMode` in HarpRoom.ts
```

**Rationale:** Documents both modes as valid options, phrase-first as design-aligned default.

---

## 5. Implementation Handoff

### Change Scope Classification: **MINOR**

**Justification:**
- Changes are additive, not breaking
- Existing functionality preserved
- No architectural changes
- Can be implemented directly by development team

### Handoff Recipients

| Role | Responsibility | Deliverables |
|------|----------------|--------------|
| **Development Team** | Implement proposed code changes | Updated PatientJellyManager.ts, HarpRoom.ts |
| **QA/Testing** | Validate phrase-first mode works as designed | Test coverage for new flow |
| **Product Owner** | Review and approve UX feel | Playtest feedback |

### Success Criteria

- [ ] Phrase-first mode demonstrates full sequence before player responds
- [ ] Synchronized splash provides clear "your turn" signal
- [ ] Player can replay full phrase from memory
- [ ] Wrong note triggers full phrase redemonstration
- [ ] Note-by-note mode remains functional as fallback
- [ ] Visual indicators show sequence order clearly
- [ ] Performance remains <1ms per frame overhead

### Implementation Sequence

```
1. Update PatientJellyManager.ts state machine (2h)
2. Implement phrase-first demonstration flow (3h)
3. Add synchronized splash system (2h)
4. Update player response handling (3h)
5. Add visual feedback enhancements (2h)
6. Testing and validation (3h)
7. Documentation updates (1h)
---
Total: 16 hours (2 developer-days)
```

---

## 6. Approval and Next Steps

### Required Approvals

- [ ] Product Owner: UX approach acceptable?
- [ ] Tech Lead: Implementation approach sound?
- [ ] **User (Mehul): Proceed with implementation?**

### After Approval

1. Create story files for implementation:
   - STORY-HARP-001: Phrase-First Teaching Mode
   - STORY-HARP-002: Synchronized Splash System
   - STORY-HARP-003: Phrase Response Handling

2. Update sprint-status.yaml with new stories

3. Assign to development team

4. Track implementation through existing sprint workflow

---

## Appendix: Decision Matrix

| Factor | Phrase-First (Design) | Note-by-Note (Current) | Hybrid Approach |
|--------|---------------------|----------------------|-----------------|
| Design Alignment | ✅ 100% | ⚠️ 60% | ✅ 100% |
| Implementation Effort | Medium | ✅ Complete | Medium |
| Player Difficulty | Medium (designed) | Easy | Configurable |
| Code Complexity | Medium | ✅ Simple | Medium |
| Testing Required | Medium | ✅ Done | Medium |
| Risk | Low | ✅ None | Low |

**Recommendation:** Hybrid approach - implement phrase-first as default, keep note-by-note as option.

---

**Document Version:** 1.0
**Last Updated:** 2026-01-29
**Next Review:** After approval
