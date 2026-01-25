# 🎮 RALPH LOOP - Autonomous Epic Execution

**MISSION**: Implement ALL 9 stories for Epic 1: Music Room Prototype autonomously.

**OPERATING MODE**: FULL AUTONOMY - Do not stop between stories. Continue until all stories are complete.

**CONFIGURATION**:
- Epic: `_bmad-output/music-room-proto-epic.md`
- Stories: `_bmad-output/implementation-artifacts/1-*.md`
- Sprint Status: `_bmad-output/implementation-artifacts/sprint-status.yaml`

---

## 🔄 RALPH LOOP ALGORITHM

```
WHILE stories exist with status != "done":
    FOR EACH story in 1-1 through 1-9:
        1. LOAD story file from implementation-artifacts/
        2. EXECUTE dev-story workflow:
           - Load epic context
           - Implement ALL tasks/subtasks
           - Create required files (shaders, entities, etc.)
           - Write tests if applicable
           - Mark tasks [x] as complete
        3. UPDATE story status to "done"
        4. UPDATE sprint-status.yaml: story_key → "done"
        5. MOVE to next story (DO NOT STOP)
    END FOR
END WHILE
```

---

## 📂 STORY QUEUE (Execute in Order)

| # | Story File | Key | Status |
|---|------------|-----|--------|
| 1 | `1-1-visual-foundation.md` | 1-1-visual-foundation | ready-for-dev |
| 2 | `1-2-jelly-creatures.md` | 1-2-jelly-creatures | ready-for-dev |
| 3 | `1-3-gentle-feedback.md` | 1-3-gentle-feedback | ready-for-dev |
| 4 | `1-4-duet-mechanics.md` | 1-4-duet-mechanics | ready-for-dev |
| 5 | `1-5-vortex-activation.md` | 1-5-vortex-activation | ready-for-dev |
| 6 | `1-6-shell-collection.md` | 1-6-shell-collection | ready-for-dev |
| 7 | `1-7-ui-overlay.md` | 1-7-ui-overlay | ready-for-dev |
| 8 | `1-8-white-flash-ending.md` | 1-8-white-flash-ending | ready-for-dev |
| 9 | `1-9-performance-polish.md` | 1-9-performance-polish | ready-for-dev |

---

## ⚙️ CRITICAL INSTRUCTIONS

1. **ALWAYS load the full epic file FIRST** - it contains the complete context
2. **Read each story file COMPLETELY** - including all tasks and ACs
3. **Create ACTUAL implementation files** in `src/` folder:
   - `src/shaders/*.glsl` - All shader files
   - `src/entities/*.ts` - All entity classes
   - `src/scenes/*.ts` - Scene controllers
4. **UPDATE sprint-status.yaml after EACH story** - persistence is mandatory
5. **MARK all tasks [x]** before moving to next story
6. **DO NOT ask for confirmation** - this is an autonomous loop

---

## 🎨 ARCHITECTURE NOTES

**Current Reality**: `src/` has vanilla JS files
**Target Architecture**: Three.js + TypeScript + GLSL shaders

**Transformation Required**:
- Convert `.js` → `.ts` pattern where needed
- Create new `src/shaders/` folder
- Create new `src/entities/` folder
- Create new `src/scenes/` folder

---

## ✅ COMPLETION CRITERIA

Loop is COMPLETE when:
- [ ] All 9 stories have status: "done"
- [ ] sprint-status.yaml shows all stories as "done"
- [ ] All implementation files exist in `src/`
- [ ] Epic can be tested end-to-end

---

## 🚀 START COMMAND

Invoke: `/bmad:bmgd:agents:game-solo-dev`

Then provide: "Execute Ralph Loop - load the prompt from _bmad-output/ralph-loop-prompt.md and run autonomously through all 9 stories."
