# EPIC-XXX: [Epic Title]

**Epic ID**: `EPIC-XXX`
**Status**: `Draft` | `Ready for Dev` | `In Progress` | `Done`
**Points**: `TBD`
**Sprint**: `Phase X - [Feature Area]`
**Created**: `YYYY-MM-DD`
**Author**: `Name`

---

## Overview

**Epic Goal:** [One-sentence summary of what this epic achieves]

**Context:** Why are we doing this? What problem does it solve?

**Philosophy/Design Principles:**
> "Any relevant design philosophy or guiding principles"

**Duration Target:** X weeks of gameplay / X weeks of development

---

## Success Definition

**Done When:**
- [ ] Key success criteria 1
- [ ] Key success criteria 2
- [ ] Key success criteria 3

---

## Stories

| ID | Title | Points | Status | Owner |
|----|-------|--------|--------|-------|
| STORY-XXX-001 | [Story Title] | TBD | Draft | TBD |
| STORY-XXX-002 | [Story Title] | TBD | Draft | TBD |
| STORY-XXX-003 | [Story Title] | TBD | Draft | TBD |

**Total Points: TBD**

---

## Technical References

### Source Material
- [Link](URL) - Description
- [Link](URL) - Description

### Key Files
| File | Purpose |
|------|---------|
| `path/to/file` | Description |

---

## Dependencies

### Required
- Requirement 1
- Requirement 2

### Blocked By
- Potential blocker (if any)

### Blocking
- What this epic blocks (if anything)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         [EPIC NAME] ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUTS                      PROCESS                    OUTPUT          │
│  ┌──────────┐              ┌─────────────┐          ┌──────────────┐   │
│  │ Input 1  │─────────────││             │─────────│  Output 1    │   │
│  │          │              ││   System    │          │              │   │
│  └──────────┘              ││             │          └──────────────┘   │
│                             └─────────────┘                              │
│  ┌──────────┐                                                           │
│  │ Input 2  │─────────────→                                            │
│  └──────────┘              ...                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
src/systems/[feature]/
├── [MainSystem].ts                  # Main orchestrator
├── compute/                          # Compute shaders (if applicable)
│   └── *.wgsl
├── render/                           # Rendering (if applicable)
│   └── *.wgsl
└── types.ts                          # Type definitions
```

---

## Epic Acceptance Criteria

- [ ] All stories marked as complete
- [ ] Visual/functional requirement 1
- [ ] Visual/functional requirement 2
- [ ] Performance requirement (60fps, etc.)
- [ ] Debug views available: `window.debugVimana.[feature].*`
- [ ] No errors in console

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Risk 1 | High/Medium/Low | Mitigation strategy |
| Risk 2 | High/Medium/Low | Mitigation strategy |

---

## Timeline Notes

- **STORY-XXX-001** is the foundation - blocks other stories
- **STORY-XXX-002, STORY-XXX-003** can develop in parallel
- Estimated effort: **X weeks**

---

## Epic Retrospective

**Status:** To be completed after epic is done

**Retrospective Questions:**
- What went well?
- What could be improved?
- Any unexpected issues?
- Lessons learned for future epics?

---

**Sources:**
- [Reference 1](URL)
- [Reference 2](URL)
