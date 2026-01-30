# EPIC-005: Harp Minigame Design Alignment

**Epic ID**: `EPIC-005`
**Status**: `Ready for Dev`
**Points**: `12`
**Sprint**: `Enhancement - Harp Phrase-First Mode`
**Created**: `2026-01-29`
**Author**: `Mehul`

---

## Overview

**Epic Goal:** Align the harp music interaction implementation with the original design specification by implementing phrase-first teaching mode.

**Context:** Code review revealed that `PatientJellyManager.ts` implements note-by-note teaching (one jelly at a time) while the design doc (`vimana_harp_minigame_design.md`) specifies phrase-first teaching (all jellies demonstrate, then player responds from memory).

**Design Philosophy:**
> "The ship teaches you, it doesn't test you. A call-and-response duet where the Vimana demonstrates a full musical phrase, then the player remembers and replays it."

**Duration Target:** 2 developer days (~16 hours)

---

## Success Definition

**Done When:**
- [ ] Phrase-first mode demonstrates full sequence before player responds
- [ ] Synchronized splash provides clear "your turn" signal
- [ ] Player can replay full phrase from memory
- [ ] Wrong note triggers patient redemonstration (not punishment)
- [ ] Note-by-note mode remains functional as fallback
- [ ] Visual sequence indicators (①, ②, ③...) show order
- [ ] Implementation matches design document specification

---

## Stories

| ID | Title | Points | Status | Owner |
|----|-------|--------|--------|-------|
| STORY-HARP-101 | Phrase-First Teaching Mode | 5 | Ready for Dev | TBD |
| STORY-HARP-102 | Synchronized Splash System | 2 | Ready for Dev | TBD |
| STORY-HARP-103 | Phrase Response Handling | 3 | Ready for Dev | TBD |
| STORY-HARP-104 | Visual Sequence Indicators | 2 | Ready for Dev | TBD |

**Total Points: 12**

**Story Dependencies:**
```
HARP-101 (Foundation) ─┬─> HARP-102 (Splash)
                       ├─> HARP-103 (Response)
                       └─> HARP-104 (Visuals)
```

---

## Technical References

### Source Material
- `vimana_harp_minigame_design.md` - Original design specification
- `sprint-change-proposal-harp-2026-01-29.md` - Full analysis and rationale

### Key Files
| File | Purpose |
|------|---------|
| `src/entities/PatientJellyManager.ts` | Core teaching logic to extend |
| `src/entities/JellyManager.ts` | Jellyfish spawning and animation |
| `src/entities/JellyLabelManager.ts` | Visual indicator system |
| `src/audio/HarmonyChord.ts` | Splash sound and feedback |
| `src/scenes/HarpRoom.ts` | Integration point |

---

## Dependencies

### Required
- ✅ EPIC-1 (Music Room Prototype) - Done, provides foundation
- ✅ EPIC-2 (WaterBall Fluid) - Done, harp-water interaction exists

### Blocked By
- None - ready to implement

### Blocking
- None - enhancement work only

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PHRASE-FIRST TEACHING FLOW                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHRASE DEMONSTRATION (Vimana's Turn)                    PLAYER'S TURN   │
│  ┌─────────────────────────────────────────────────────┐    ┌─────────┐│
│  │ 1. Jelly ① emerges → plays note C                   │    │         ││
│  │ 2. Jelly ② emerges → plays note D                   │    │  Player ││
│  │ 3. Jelly ③ emerges → plays note E                   │    │ remembers││
│  │    └────────────────────────────────────────────────┤    │  & plays ││
│  │ 4. ALL jellies splash down together (TURN SIGNAL)   │───│  phrase  ││
│  │                                                      │    │         ││
│  └─────────────────────────────────────────────────────┘    └─────────┘│
│                                                                 │
│  CORRECT: Full harmony chord (ship joins in)                     │
│  WRONG: Gentle redemonstration (patient, not punishing)           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Changes:**
1. New states: `PHRASE_DEMONSTRATION`, `TURN_SIGNAL`, `AWAITING_PHRASE_RESPONSE`
2. Multi-jelly spawning (not sequential single-jelly)
3. Synchronized splash animation
4. Phrase-level validation (not note-by-note)

---

## File Structure

```
src/entities/
├── PatientJellyManager.ts           # Extended with phrase-first logic
│   ├── DuetState enum (3 new states)
│   ├── startPhraseFirstSequence()
│   ├── triggerSynchronizedSplash()
│   ├── handlePhraseFirstInput()
│   └── setTeachingMode()
│
├── JellyManager.ts                  # Add submergeJelly() method
│
├── JellyLabelManager.ts             # Enhanced with sequence labels
│   ├── showSequenceLabel()
│   ├── hideAll()
│   └── Number texture rendering
│
└── SynchronizedSplashEffect.ts      # NEW: Visual splash effect

src/audio/
└── HarmonyChord.ts                  # Add playSplashSound()

src/scenes/
└── HarpRoom.ts                      # Integration with teaching mode config
```

---

## Epic Acceptance Criteria

### Functional
- [ ] Phrase-first mode: All jellies demonstrate before player responds
- [ ] Synchronized splash: Single unified splash, not multiple
- [ ] Player response: Must replay full sequence from memory
- [ ] Wrong note: Triggers full redemonstration (patient retry)
- [ ] Completion: All 3 sequences (C-D-E, F-G-A, E-G-D) playable
- [ ] Fallback: Note-by-note mode still works via config

### Visual
- [ ] Number indicators (①, ②, ③...) above jellies during demo
- [ ] Labels bounce in with animation
- [ ] Labels fade out on player's turn
- [ ] Splash creates expanding ripple effects
- [ ] All visual feedback is bioluminescent/cyan theme

### Audio
- [ ] Each jelly plays its note during demonstration
- [ ] Single unified splash sound (not overlapping splashes)
- [ ] Harmony chord on phrase completion
- [ ] Discordant feedback on wrong note

### Performance
- [ ] No frame rate degradation (<1ms overhead)
- [ ] No memory leaks from multiple jellies
- [ ] Clean state transitions

### Debug
- [ ] `window.debugVimana.teachingMode` switcher
- [ ] Console logging for phrase progression
- [ ] Visual debug view of jelly positions

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing note-by-note mode | Medium | Keep as fallback, add mode switcher config |
| Performance from multiple active jellies | Low | Max 3 jellies, simple geometry |
| Player confusion about turn signal | Medium | Clear splash + visual ripples |
| Wrong note feeling punitive | Low | Immediate redemonstration, no progress loss |

---

## Timeline Notes

**Story Sequence (Must implement in order):**
1. **STORY-HARP-101** (Foundation) - New states, spawn logic
2. **STORY-HARP-102** (Splash) - Depends on 101 for state machine
3. **STORY-HARP-103** (Response) - Depends on 101 for input routing
4. **STORY-HARP-104** (Visuals) - Can parallel with 102/103

**Estimated effort:**
- STORY-HARP-101: 5 hours
- STORY-HARP-102: 3 hours
- STORY-HARP-103: 4 hours
- STORY-HARP-104: 3 hours
- **Total: ~15 hours (2 days)**

---

## Epic Retrospective

**Status:** To be completed after epic is done

**Retrospective Questions:**
- Did phrase-first mode match player expectations?
- Was synchronized splash clear as turn signal?
- Did visual indicators help memory or clutter the view?
- Should note-by-note mode remain or be removed?
- Any unexpected edge cases in phrase validation?

---

## Design Alignment Checklist

Original design requirements vs implementation:

| Design Requirement | Story Coverage | Status |
|-------------------|----------------|--------|
| All jellies emerge for phrase | HARP-101 | ✅ Planned |
| Order shows sequence | HARP-104 | ✅ Planned |
| Synchronized splash = turn signal | HARP-102 | ✅ Planned |
| Player remembers and replays | HARP-103 | ✅ Planned |
| Gentle feedback on wrong note | HARP-103 | ✅ Planned |
| No failure states | HARP-103 | ✅ Planned |
| Ship "joins in" with harmony | HARP-103 | ✅ Planned |

---

**Sources:**
- [vimana_harp_minigame_design.md](../vimana_harp_minigame_design.md)
- [sprint-change-proposal-harp-2026-01-29.md](../sprint-change-proposal-harp-2026-01-29.md)
- Existing `PatientJellyManager.ts` implementation
