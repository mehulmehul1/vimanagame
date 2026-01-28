# Planning Artifacts - VIMANA Project

This directory contains all planning artifacts for the VIMANA project, organized by type.

## Directory Structure

```
planning-artifacts/
├── README.md                    # This file
├── INDEX.md                     # Quick reference of all artifacts
├── epics/                       # Epic definitions (large features spanning multiple stories)
│   ├── EPIC-001-music-room-proto.md
│   ├── EPIC-002-waterball-fluid-system.md
│   └── EPIC-XXX-[name].md       # Future epics
├── stories/                     # Story breakdowns (implementation units)
│   ├── STORY-001-*.md
│   ├── STORY-002-*.md
│   └── STORY-XXX-*.md
├── features/                    # Future feature requests (not yet prioritized)
│   ├── FEATURE-XXX-*.md
│   └── backlog.md
├── bugs/                        # Bug reports and fix tracking
│   ├── BUG-XXX-*.md
│   └── known-issues.md
└── templates/                   # Templates for creating new artifacts
    ├── epic-template.md
    ├── story-template.md
    ├── feature-template.md
    └── bug-template.md
```

## Organization Principles

### Epics (EPIC-XXX)
- **Purpose**: Large feature work that spans multiple stories/sprints
- **Scope**: 2-6 weeks of work, 20+ story points
- **Contents**: Overview, story list, dependencies, acceptance criteria, risks
- **Status Draft → Ready for Dev → In Progress → Done**

### Stories (STORY-XXX)
- **Purpose**: Implementation units that can be completed in 1-5 days
- **Scope**: 1-13 story points (following Fibonacci sequence: 1, 2, 3, 5, 8, 13)
- **Contents**: User story format, technical specs, acceptance criteria
- **Parent**: Each story belongs to exactly one epic
- **Status**: Draft → Ready for Dev → In Dev → Review → Done

### Features (FEATURE-XXX)
- **Purpose**: Future feature requests not yet prioritized
- **Scope**: Any size, from small enhancements to large system changes
- **Contents**: Description, rationale, rough effort estimate
- **Status**: Backlog → Prioritized → (becomes Epic/Story)

### Bugs (BUG-XXX)
- **Purpose**: Bug reports and fix tracking
- **Scope**: Any size, categorized by severity
- **Contents**: Description, reproduction steps, severity, fix status
- **Status**: Open → In Progress → Fixed → Verified

## Current Epics

| ID | Title | Status | Points | Link |
|----|-------|--------|--------|------|
| EPIC-001 | Music Room Prototype - Archive of Voices | ready-for-review | TBD | [EPIC-001](./epics/EPIC-001-music-room-proto.md) |
| EPIC-002 | WaterBall Fluid Simulation System | Ready for Dev | 29 | [EPIC-002](./epics/EPIC-002-waterball-fluid-system.md) |

## Story Numbering

Stories are numbered sequentially across all epics:

```
EPIC-001: Stories 1.1-1.9 (implementation-artifacts/1-*)
EPIC-002: Stories 002-001 to 002-006 (planning-artifacts/stories/story-00X-*.md)
```

**Note**: EPIC-001 stories use the old naming convention (1-1, 1-2, etc.) and live in `implementation-artifacts/`. Future epics should use the unified `STORY-XXX` format.

## Adding New Artifacts

1. **New Epic**: Copy `templates/epic-template.md`, fill in details, save as `epics/EPIC-XXX-[name].md`
2. **New Story**: Copy `templates/story-template.md`, fill in details, save as `stories/STORY-XXX-[name].md`
3. **New Feature**: Copy `templates/feature-template.md`, save as `features/FEATURE-XXX-[name].md`
4. **New Bug**: Copy `templates/bug-template.md`, save as `bugs/BUG-XXX-[name].md`

Update `INDEX.md` when adding new artifacts.

## Related Directories

- `../implementation-artifacts/` - Detailed implementation notes for EPIC-001 stories
- `../gdd.md` - Game Design Document
- `../narrative-design.md` - Narrative design docs
- `../game-brief.md` - Project overview

---

**Last Updated**: 2026-01-26
**Maintained By**: Development Team
