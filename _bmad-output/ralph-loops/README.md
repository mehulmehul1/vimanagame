# Ralph Loops - Autonomous Execution for Vimana

All Ralph loops live here. No Ralph files should be in the project root.

## What is Ralph?

**Ralph** = Bash loop that runs Claude Code repeatedly until all tasks are complete.

Each loop:
1. Reads a PRD (workflows or stories)
2. Picks highest priority incomplete item
3. Runs Claude Code to execute it
4. Validates and commits if passing
5. Updates progress and repeats

---

## Directory Structure

```
ralph-loops/
├── README.md                          ← This file
│
├── gametest/                          ← QA/Testing workflows (6 workflows)
│   ├── ralph.sh
│   ├── CLAUDE.md
│   ├── prd.json
│   └── progress.txt
│
├── epic-001-music-room/               ← Music Room Prototype (9 stories)
│   ├── ralph.sh
│   ├── CLAUDE.md
│   ├── prd.json
│   └── progress.txt
│
├── epic-002-waterball/                ← WaterBall Fluid System (6 stories) ✅
│   ├── ralph.sh
│   ├── CLAUDE.md
│   ├── prd.json
│   └── progress.txt
│
├── epic-004-webgpu/                   ← WebGPU Migration (8 stories) 🔄
│   ├── ralph.sh
│   ├── CLAUDE.md
│   ├── prd.json
│   └── progress.txt
│
└── archive/                           ← Previous runs archived by branch/date
```

---

## Running Ralph Loops

**From the project directory** (`C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana`):

```bash
# Run gametest workflows
bash ../_bmad-output/ralph-loops/gametest/ralph.sh

# Run epic-001 stories
bash ../_bmad-output/ralph-loops/epic-001-music-room/ralph.sh

# Run epic-002 stories (COMPLETE)
bash ../_bmad-output/ralph-loops/epic-002-waterball/ralph.sh

# Run epic-004 stories (WebGPU Migration)
bash ../_bmad-output/ralph-loops/epic-004-webgpu/ralph.sh

# Custom max iterations
bash ../_bmad-output/ralph-loops/epic-004-webgpu/ralph.sh 50
```

---

## Loop Status

| Loop | Status | Items | Completed |
|------|--------|-------|-----------|
| **gametest** | ✅ COMPLETE | 6 workflows | 6/6 |
| **epic-001-music-room** | 🔄 Ready | 9 stories | 0/9 |
| **epic-002-waterball** | ✅ COMPLETE | 6 stories | 6/6 |
| **epic-004-webgpu** | 🔄 Ready | 8 stories | 0/8 |

---

## Creating a New Ralph Loop

1. Create directory: `ralph-loops/[loop-name]/`
2. Copy a template from an existing loop
3. Modify `prd.json` with your workflows/stories
4. Modify `CLAUDE.md` with instructions
5. Update this README

---

## File Descriptions

| File | Purpose |
|------|---------|
| `ralph.sh` | Bash loop script - reads PRD, runs Claude, checks for COMPLETE signal |
| `CLAUDE.md` | Agent instructions - tells Claude what to do each iteration |
| `prd.json` | Workflows or stories with completion state (`passes: true/false`) |
| `progress.txt` | Learnings log - patterns discovered, gotchas, useful context |
| `archive/` | Previous runs - auto-archived when branch changes |

---

## Stop Condition

Ralph loops exit when they see:
```
<promise>COMPLETE</promise>
```

Claude outputs this when ALL items in `prd.json` have `passes: true`.

---

**Maintained By**: Development Team
**Last Updated**: 2026-01-26
