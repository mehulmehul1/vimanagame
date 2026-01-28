# Ralph Gametest Workflows - Setup and Execution Guide

**Project**: Vimana
**Date**: 2026-01-24
**Purpose**: Run all 6 BMAD gametest workflows autonomously using Ralph

---

## 📋 What Is This?

This setup uses **Ralph** (from https://github.com/snarktank/ralph) to autonomously execute all 6 BMAD gametest workflows on the Vimana project.

**Ralph** = Bash loop that runs Claude Code repeatedly until all tasks are complete.

---

## 🚀 Quick Start

### Prerequisites

1. **Claude Code installed**:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Claude Code authenticated**:
   ```bash
   claude --auth
   ```

3. **jq installed** (for JSON parsing):
   ```bash
   # Windows (via Git Bash or WSL)
   # Already available in Git Bash
   ```

### Running Ralph

```bash
cd C:\Users\mehul\OneDrive\Desktop\Studio\PROJECTS\shadowczarengine\vimana
bash ralph-gametest.sh
```

Or with custom max iterations:

```bash
bash ralph-gametest.sh 50
```

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| `ralph-gametest.sh` | Bash loop script (from GitHub Ralph) |
| `CLAUDE-GAMETEST.md` | Ralph instructions for gametest workflows |
| `prd-gametest.json` | 6 gametest workflows as user stories |
| `progress-gametest.txt` | Learnings log (appended each iteration) |

---

## 🔄 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ralph Loop (ralph-gametest.sh)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FOR each iteration (max 20 by default):                       │
│                                                                  │
│    1. Read prd-gametest.json                                   │
│    2. Pick highest priority workflow where passes: false       │
│    3. Run Claude Code with CLAUDE-GAMETEST.md as prompt        │
│    4. Claude Code executes ONE workflow                         │
│    5. Validates deliverables                                   │
│    6. Runs quality checks (tests, typecheck)                  │
│    7. Commits if validation passes                             │
│    8. Updates prd-gametest.json (passes: true)                │
│    9. Appends learnings to progress-gametest.txt               │
│                                                                  │
│    IF all workflows pass:                                       │
│      → Output <promise>COMPLETE</promise>                      │
│      → Exit successfully                                        │
│                                                                  │
│    ELSE:                                                        │
│      → Continue to next iteration                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 The 6 Workflows (In Priority Order)

| Priority | Workflow ID | Title | Description |
|----------|-------------|-------|-------------|
| 1 | test-framework | Game Test Framework Setup | Initialize/verify test framework, create docs |
| 2 | test-design | Game Test Design | Create test scenarios for all 9 stories |
| 3 | automate | Automated Test Generation | Generate unit + E2E tests |
| 4 | playtest-plan | Playtest Planning | Create playtesting sessions |
| 5 | performance | Performance Testing Strategy | FPS, memory, loading tests |
| 6 | test-review | Test Suite Review | Validate all tests, identify gaps |

---

## 🎯 Expected Output

After Ralph completes all 6 workflows, you'll have:

```
vimana/tests/
├── README.md                    ← Framework documentation
├── TEST-DESIGN.md               ← Test scenarios (GIVEN/WHEN/THEN)
├── AUTOMATION-SUMMARY.md        ← Generated test summary
├── PLAYTEST-PLAN.md             ← Playtesting sessions
├── PERFORMANCE-PLAN.md          ← Performance strategy
├── TEST-REVIEW.md               ← Test quality review
├── unit/
│   ├── managers.test.ts
│   ├── entities.test.ts
│   └── utils.test.ts
├── e2e/
│   ├── harp-interaction.spec.ts
│   ├── jelly-creatures.spec.ts
│   ├── vortex.spec.ts
│   └── smoke.spec.ts
└── performance/
    ├── particles.test.ts
    └── memory.test.ts
```

---

## 🔍 Monitoring Progress

### Check Current Status

```bash
# See which workflows are done
cat prd-gametest.json | jq '.userStories[] | {id, title, passes}'
```

### See Learnings

```bash
# See what Ralph has learned from previous iterations
cat progress-gametest.txt
```

### Check Git History

```bash
# See commits Ralph has made
git log --oneline -10
```

---

## ⚙️ Troubleshooting

### Ralph Stops Early

- Check `progress-gametest.txt` for errors
- Check `prd-gametest.json` to see which workflows failed
- Fix issues manually, then re-run Ralph

### Workflow Fails Validation

- Ralph will retry the workflow in the next iteration
- Check the workflow's `checklist.md` for what's required
- Ralph learns from failures and adjusts

### Tests Fail

- Ralph will NOT commit broken code
- Ralph will retry the workflow
- Check test output for what's failing

---

## 🛑 Stopping Ralph

```bash
# Press Ctrl+C to stop Ralph
# Ralph will complete current iteration before stopping
```

---

## 📊 After Completion

When all 6 workflows complete:

1. **Review deliverables**: Check tests/ folder
2. **Run all tests**: `npm run test && npm run test:e2e`
3. **Check coverage**: Verify all features covered
4. **Read progress-gametest.txt**: See what Ralph learned

---

## 🔗 Related Files

- **Music Room Epic**: `../_bmad-output/music-room-proto-epic.md`
- **BMAD Workflows**: `../_bmad/bmgd/workflows/gametest/`
- **Current Ralph Progress**: `progress.json`, `status.json`

---

## 💡 Key Differences from Standard Ralph

1. **Custom PRD file**: Uses `prd-gametest.json` instead of `prd.json`
2. **Custom prompt**: Uses `CLAUDE-GAMETEST.md` instead of `CLAUDE.md`
3. **Custom progress**: Uses `progress-gametest.txt` instead of `progress.txt`
4. **Workflow-based**: Each "user story" is a BMAD gametest workflow
5. **QA-focused**: Ralph acts as QA agent, not dev agent

---

*Generated by Clawdbot Second Brain*
*Date: 2026-01-24*
*Ralph Setup for Vimana Gametest Workflows*
