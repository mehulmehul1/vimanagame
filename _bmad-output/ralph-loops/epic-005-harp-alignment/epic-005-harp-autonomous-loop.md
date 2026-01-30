# Ralph Loop Prompt - EPIC-005 Harp Design Alignment (Autonomous)

## Context

You are an autonomous development loop executing **EPIC-005: Harp Minigame Design Alignment** for the VIMANA project.

**Agent Persona:** @C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana\.claude\commands\bmad\bmgd\agents\game-dev.md

**Epic File:** `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\_bmad-output\ralph-loops\epic-005-harp-alignment\epic-005-harp-design-alignment.md`

**Implementation Artifacts:** `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\_bmad-output\implementation-artifacts\story-harp-*.md`

---

## Mission

Execute all 4 stories of EPIC-005 autonomously until complete. Do not stop for user input unless a critical decision point is reached.

---

## Stories to Execute (Dependency Order)

### Phase 1: Foundation
- **Story harp-101**: Phrase-First Teaching Mode (No dependencies - START HERE)

### Phase 2: Features (After harp-101, can run in parallel)
- **Story harp-102**: Synchronized Splash System
- **Story harp-103**: Phrase Response Handling
- **Story harp-104**: Visual Sequence Indicators

---

## Autonomous Execution Protocol

### For Each Story:

1. **READ** the implementation artifact
   - Example: `story-harp-101-phrase-first-teaching.md`

2. **LOAD** current codebase state
   - Read `src/entities/PatientJellyManager.ts`
   - Read `src/entities/JellyManager.ts`
   - Read `src/audio/HarmonyChord.ts`
   - Understand current architecture

3. **IMPLEMENT** following acceptance criteria
   - Add new states to DuetState enum
   - Create new methods as specified
   - Modify existing files preserving compatibility
   - Follow code patterns from Dev Notes sections

4. **TEST** each implementation
   - Run `npm run build` to check TypeScript
   - Run `npm run dev` to verify no runtime errors
   - Check console for warnings

5. **COMMIT** after each story completion
   - Commit message format: `feat: STORY-HARP-10X - [Story Name]`
   - Include co-author tag

6. **TRACK** progress
   - Mark completed tasks in progress.txt
   - Update prd.json to set `passes: true`

7. **CONTINUE** to next story
   - Do not pause for approval unless critical error
   - Log completion and proceed

---

## Critical Decision Points (Require Human Input)

PAUSE and ask user if:
1. **Breaking changes detected** - If modifying public APIs unexpectedly
2. **Major architectural deviation** - If implementation differs significantly from spec
3. **Compilation failure** - If TypeScript errors cannot be resolved
4. **Performance regression** - If FPS drops below targets

**DO NOT PAUSE FOR:**
- Code formatting/style
- Minor refactoring within same approach
- Test writing
- Comments/documentation

---

## Project Context

**Working Directory:** `C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana`

**Key Files:**
- `src/entities/PatientJellyManager.ts` - Core teaching logic (to extend)
- `src/entities/JellyManager.ts` - Jellyfish spawning (to extend)
- `src/entities/JellyLabelManager.ts` - Label system (to enhance)
- `src/audio/HarmonyChord.ts` - Audio feedback (to extend)
- `src/scenes/HarpRoom.ts` - Integration point

**Design Reference:** `vimana_harp_minigame_design.md`
- Phrase-first demonstration: All jellies show sequence
- Synchronized splash: Turn signal for player
- Player responds: Full phrase from memory
- Gentle feedback: Redemonstration on wrong note

---

## Execution Checklist

Use this checklist to track progress. Mark each as completed:

### Foundation Phase
- [ ] Story harp-101: Phrase-First Teaching Mode
  - [ ] Add PHRASE_DEMONSTRATION, TURN_SIGNAL, AWAITING_PHRASE_RESPONSE to DuetState
  - [ ] Implement startPhraseFirstSequence() method
  - [ ] Add TeachingModeConfig interface
  - [ ] Implement setTeachingMode() method
  - [ ] Update JellyManager for multi-jelly spawning
  - [ ] Add 800ms stagger between jelly emergence
  - [ ] Test: Existing note-by-note mode still works

### Features Phase
- [ ] Story harp-102: Synchronized Splash System
  - [ ] Implement triggerSynchronizedSplash() method
  - [ ] Add submergeJelly() to JellyManager
  - [ ] Add playSplashSound() to HarmonyChord
  - [ ] Create SynchronizedSplashEffect class
  - [ ] Test: Single splash sound (not multiple)
  - [ ] Test: All jellies submerge together
- [ ] Story harp-103: Phrase Response Handling
  - [ ] Implement handlePhraseFirstInput() method
  - [ ] Implement handleCorrectNoteInPhrase() method
  - [ ] Implement handleWrongNoteInPhrase() method
  - [ ] Implement completePhrase() method
  - [ ] Refactor: Rename existing logic to handleNoteByNoteInput()
  - [ ] Test: Phrase completion triggers next sequence
  - [ ] Test: Wrong note triggers redemonstration
- [ ] Story harp-104: Visual Sequence Indicators
  - [ ] Enhance JellyLabelManager with sequence labels
  - [ ] Implement showSequenceLabel() method
  - [ ] Create circled number textures (①, ②, ③)
  - [ ] Add bounce-in animation
  - [ ] Add fade-out animation
  - [ ] Implement hideAll() method
  - [ ] Test: Labels always face camera (billboarding)

---

## Git Commit Pattern

After each story, commit with:

```
feat: STORY-HARP-10X - [Story Name]

- [Key change 1]
- [Key change 2]
- [Key change 3]

Closes #[story-number-in-tracking]

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

---

## Error Handling Strategy

1. **TypeScript Errors**
   - Fix immediately if clear
   - Log and continue if minor type mismatch can be resolved later
   - PAUSE if blocking compilation

2. **Runtime Errors**
   - Check browser console
   - Verify state transitions
   - Check callback wiring
   - PAUSE if cannot diagnose

3. **Import/Export Errors**
   - Verify file paths
   - Check module resolution
   - Fix before proceeding

---

## Success Criteria

**EPIC COMPLETE when:**
- All 4 stories implemented
- All TypeScript compiles without errors
- `npm run dev` runs successfully
- Phrase-first mode demonstrates full sequence
- Synchronized splash provides clear turn signal
- Player can replay full phrase from memory
- Wrong note triggers patient redemonstration
- Visual sequence indicators (①, ②, ③) visible
- Note-by-note mode still functional
- All commits pushed to branch

---

## Final Output

When complete, generate:
1. Summary of all changes
2. Files created/modified list
3. Known issues or limitations
4. Next steps for Epic 6 or continuation

---

## Start Command

Begin execution immediately with:
1. Check current git branch
2. Read `src/entities/PatientJellyManager.ts` to understand current state
3. Start with Story harp-101 (Phrase-First Teaching Mode)
4. Execute autonomously until complete

**GO NOW - DO NOT WAIT FOR USER INPUT**
