# PRP: Scene Flow Navigator - Story-Centric Redesign

**Project**: Shadow Czar Engine Editor
**Component**: Scene Flow Navigator Panel
**Status**: Planning
**Priority**: High
**Author**: Ralph Workflow
**Created**: 2025-01-17

---

## 1. Executive Summary

Transform the Scene Flow Navigator from a linear state viewer into a **story-centric timeline editor** for authoring narrative flow. The physical world (zones, splats, objects) is built in the 3D viewport; this panel focuses purely on narrative progression.

---

## 2. Current State Analysis

### Existing Implementation
- **Location**: `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx` (421 lines)
- **Technology**: ReactFlow for node graph visualization
- **Data Source**: `editor/data/gameStateData.ts` (mirrors `src/gameData.js`)

### Current Features
| Feature | Status | Notes |
|---------|--------|-------|
| Display GAME_STATES as nodes | ✅ Working | Linear left-to-right layout |
| Color coding by category | ✅ Working | 13 categories defined |
| Double-click to jump to state | ✅ Working | Uses URL parameter reload |
| Current state highlight | ✅ Working | Visual indicator |
| Keyboard navigation | ✅ Working | Arrow keys + Enter |
| MiniMap | ✅ Working | Overview of flow |
| Auto-layout | ✅ Working | By state value |

### Issues & Limitations
| Issue | Impact | Priority |
|-------|--------|----------|
| Linear-only view (no branching) | Can't show dialog choices | High |
| No state editing capabilities | Read-only display | High |
| Hardcoded state data | Can't add/remove states | High |
| No transition logic visibility | Can't see what triggers next state | High |
| No zone assignment | Can't link states to physical locations | Medium |
| No content attachment | Can't link videos/dialogs/music | Medium |
| URL reload for jumps | Disruptive editor experience | Low |

---

## 3. Requirements

### 3.1 Core Philosophy
```
┌─────────────────────────────────────────────────────────────────┐
│                     DESIGN PRINCIPLE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  3D VIEWPORT                   SCENE FLOW NAVIGATOR             │
│  (Physical World)               (Narrative Flow)                │
│                                                                  │
│  • Zone layout                  • Story timeline                │
│  • Splat placement              • State transitions             │
│  • GLTF models                  • Trigger conditions            │
│  • Colliders                    • Content links                 │
│  • Lights                       • Branching paths               │
│  • Object transforms            • Dialog choices                │
│                                                                  │
│  Created in Blender/Coded       Story Authoring Tool            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Functional Requirements

#### FR1: Story Timeline View
- Display all GAME_STATES as a timeline from left to right
- Show branching paths for dialog choices
- Visual grouping by story acts (Act 1, Act 2, Act 3, etc.)
- Collapsible act sections for overview

#### FR2: State Inspector Panel
- When a state node is selected, show inspector panel
- Display and edit: zone assignment, player position, criteria
- Show attached content (videos, dialogs, music)
- List outgoing transitions with trigger conditions

#### FR3: State Editing
- Add new state (auto-assigns next number)
- Delete state (with confirmation)
- Rename state (updates label)
- Duplicate state with content

#### FR4: Transition Management
- Visualize all possible transitions from a state
- Add transition: Source State → Target State
- Define trigger condition (onComplete, onChoice, onProximity, etc.)
- Label transitions (e.g., "Player answers phone")

#### FR5: Content Attachment
- Attach video to state (browse VideoManager assets)
- Attach dialog tree to state
- Attach music/sfx to state
- Preview content from inspector

#### FR6: Zone Assignment
- Assign which zone a state occurs in
- Show zone name on state node
- Filter states by zone
- Jump to zone in 3D viewport

#### FR7: Player Position Capture
- "Capture from Viewport" button grabs current player position
- Store position per state for debug spawn
- Visual indicator if position is set

#### FR8: Jump to State (Testing)
- Click "Test State" button
- Sets gameState URL parameter
- Smooth reload (better than current window.location)

### 3.3 Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Performance | <100ms to render 50+ states |
| Responsiveness | Handle 100+ states without lag |
| Data Sync | Changes immediately available to game |
| Undo/Redo | Full undo stack for all operations |
| Persistence | Auto-save to project file |

---

## 4. Design Specifications

### 4.1 UI Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SCENE FLOW NAVIGATOR                                    [_][□][X]      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Act 1: Introduction                      [▼]                    │   │
│  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐                 │   │
│  │  │INTRO │───▶│TITLE │───▶│CAT   │───▶│PHONE │                  │   │
│  │  │  1   │    │  2   │    │  4   │    │  6   │                 │   │
│  │  └──────┘    └──────┘    └──────┘    └──────┘                 │   │
│  │      │                        │            │                   │   │
│  │      └────────────────────────┴────────────┼───────┐           │   │
│  │                                                │       │           │   │
│  │  Act 2: The Journey              [▼]          ▼       ▼           │   │
│  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐ ┌──────┐       │   │
│  │  │DRIVE │───▶│ENTER │───▶│VIEW  │    │CHOICE ││CHOICE│       │   │
│  │  │  10  │    │  13  │    │  17  │    │  26   ││  26  │       │   │
│  │  └──────┘    └──────┘    └──────┘    └──────┘ └──────┘       │   │
│  │                                              │       │           │   │
│  │  Act 3: Resolution                  [▼]      └───────┴───────┐   │   │
│  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐  │   │
│  │  │WAKE  │───▶│SAVE  │───▶│OUTRO │    │OUTRO │    │OUTRO │  │   │
│  │  │  32  │    │  34  │    │  38  │    │  39   │    │  40   │  │   │
│  │  └──────┘    └──────┘    └──────┘    └──────┘    └──────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  STATE INSPECTOR                                    [Clear]     │   │
│  │  ┌────────────────────────────────────────────────────────────┐│   │
│  │  │ State: PHONE_BOOTH_RINGING (6)                            ││   │
│  │  │──────────────────────────────────────────────────────────││   │
│  │  │ Zone: [Plaza ▼]                              [Jump Zone] ││   │
│  │  │                                                          ││   │
│  │  │ Player Position:                                        ││   │
│  │  │   x: -8.45  y: 1.6  z: 3.10            [Capture View]  ││   │
│  │  │                                                          ││   │
│  │  │ Entry Criteria:                                         ││   │
│  │  │   ☑ currentState >= 5                                    ││   │
│  │  │                                                          ││   │
│  │  │ Content:                                                 ││   │
│  │  │   📹 Video: phone-booth-ringing.mp4         [Play]      ││   │
│  │  │   💬 Dialog: phone_conversation           [Edit]       ││   │
│  │  │   🎵 Music: tense_ambience.mp3            [Preview]    ││   │
│  │  │                                                          ││   │
│  │  │ Transitions OUT:                                        ││   │
│  │  │   ┌─────────────────────────────────────────────────┐  ││   │
│  │  │   │ On: Player answers phone                        │  ││   │
│  │  │   │ ───────────────────────────────────────────────▶│  ││   │
│  │  │   │                                                  │  ││   │
│  │  │   │ Target: ANSWERED_PHONE (7)                      │  ││   │
│  │  │   │ [Edit Trigger] [Delete]                          │  ││   │
│  │  │   └─────────────────────────────────────────────────┘  ││   │
│  │  │   ┌─────────────────────────────────────────────────┐  ││   │
│  │  │   │ On: Player walks away                            │  ││   │
│  │  │   │ ───────────────────────────────────────────────▶│  ││   │
│  │  │   │                                                  │  ││   │
│  │  │   │ Target: CAT_DIALOG_CHOICE_2 (23)                │  ││   │
│  │  │   │ [Edit Trigger] [Delete]                          │  ││   │
│  │  │   └─────────────────────────────────────────────────┘  ││   │
│  │  │   [+ Add Transition]                                  ││   │
│  │  │                                                          ││   │
│  │  │ [Test State] [Duplicate] [Delete State]                ││   │
│  │  └────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Toolbar: [Add State] [Undo] [Redo] [Auto Layout] [Export JSON]        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Node Design

```
┌─────────────────────────────────┐
│  6   ◈                          │  ← State value + criteria badge
│  PHONE BOOTH RINGING            │  ← State name
│  ───────────────────────        │
│  Zone: Plaza                    │  ← Zone assignment
│  📹 💬 🎵                       │  ← Content indicators
│                                 │
│  [Test]                         │  ← Quick action
└─────────────────────────────────┘
     │
     └──▶ Transitions shown as labeled edges
```

### 4.3 Edge (Transition) Design

```
     PHONE_BOOTH_RINGING
           │
           │ "Player answers"
           │─────────────────────────────▶ ANSWERED_PHONE
           │
           │ "Walks away"
           └─────────────────────────────▶ CAT_DIALOG_CHOICE_2
```

### 4.4 Act Grouping

States organized into collapsible acts:
- **Act 1**: Introduction (LOADING through CAT_DIALOG_CHOICE)
- **Act 2**: The Journey (NEAR_RADIO through POST_CURSOR)
- **Act 3**: Resolution (OUTRO through GAME_OVER)

---

## 5. Data Structures

### 5.1 Enhanced State Definition

```typescript
interface StoryState {
  id: string;                    // e.g., "PHONE_BOOTH_RINGING"
  value: number;                 // e.g., 6
  label: string;                 // e.g., "Phone Booth Ringing"
  act: number;                   // 1, 2, 3...
  category: string;              // Matches STATE_CATEGORIES
  color: string;                 // For UI

  // Physical context
  zone: string | null;           // e.g., "plaza"
  playerPosition?: {             // For debug spawn
    x: number;
    y: number;
    z: number;
    rotation?: { x: number; y: number; z: number };
  };

  // Content attachments
  content: {
    video?: string;              // Path to video file
    dialog?: string;             // Dialog tree ID
    music?: string;              // Music track ID
    sfx?: string[];              // SFX IDs
    cameraAnimation?: string;    // Camera animation ID
  };

  // Entry criteria (when this state becomes active)
  criteria?: {
    currentState?: { $gte?: number; $lt?: number; $in?: number[] };
    dialogChoice?: number;       // For branching on choices
    customCondition?: string;    // JavaScript expression
  };

  // Outgoing transitions
  transitions: Transition[];

  // Metadata
  description?: string;
  notes?: string;
}

interface Transition {
  id: string;                    // Unique transition ID
  from: string;                  // Source state ID
  to: string;                    // Target state ID
  trigger: TriggerType;
  label: string;                 // Human-readable description

  // Trigger-specific data
  condition?: string;            // JavaScript expression
  dialogChoice?: number;         // Which choice leads here
  timeout?: number;              // ms before auto-transition
  proximity?: {                  // Position-based trigger
    x: number; y: number; z: number;
    radius: number;
  };
}

type TriggerType =
  | "onComplete"       // Video/dialog finishes
  | "onChoice"         // Player selects dialog option
  | "onTimeout"        // Timer expires
  | "onProximity"      // Player enters area
  | "onInteract"       // Player clicks/hovers object
  | "onState"          // Another state reaches
  | "custom";          // Custom JavaScript condition
```

### 5.2 Story Data File Format

```json
// editor/data/storyData.json
{
  "states": {
    "PHONE_BOOTH_RINGING": {
      "value": 6,
      "label": "Phone Booth Ringing",
      "act": 1,
      "category": "Phone Booth Scene",
      "color": "#a855f7",
      "zone": "plaza",
      "playerPosition": { "x": -8.45, "y": 1.6, "z": 3.1 },
      "content": {
        "video": "/content/video/phone-ringing.mp4",
        "dialog": "phone_conversation",
        "music": "tense_ambience"
      },
      "criteria": {
        "currentState": { "$gte": 5 }
      },
      "transitions": [
        {
          "id": "trans_phone_answer",
          "from": "PHONE_BOOTH_RINGING",
          "to": "ANSWERED_PHONE",
          "trigger": "onInteract",
          "label": "Player answers phone"
        },
        {
          "id": "trans_phone_walk_away",
          "from": "PHONE_BOOTH_RINGING",
          "to": "CAT_DIALOG_CHOICE_2",
          "trigger": "onProximity",
          "label": "Walks away",
          "proximity": {
            "x": 0, "y": 0, "z": 10,
            "radius": 5
          }
        }
      ],
      "description": "The phone booth rings as player approaches"
    }
  },
  "acts": {
    "1": {
      "name": "Act 1: Introduction",
      "color": "#3b82f6",
      "states": ["START_SCREEN", "INTRO", "TITLE_SEQUENCE", "CAT_DIALOG_CHOICE"]
    },
    "2": {
      "name": "Act 2: The Journey",
      "color": "#22c55e",
      "states": ["NEAR_RADIO", "PHONE_BOOTH_RINGING", "DRIVE_BY", "VIEWMASTER"]
    },
    "3": {
      "name": "Act 3: Resolution",
      "color": "#6366f1",
      "states": ["WAKING_UP", "OUTRO", "GAME_OVER"]
    }
  }
}
```

---

## 6. Implementation Plan

### Phase 1: Data Layer Foundation
**Goal**: Create editable story data structure

| Task | File | Estimate |
|------|------|----------|
| Create `StoryStateManager` class | `editor/managers/StoryStateManager.ts` | 4h |
| Create `storyData.json` from existing states | `editor/data/storyData.json` | 2h |
| Add TypeScript interfaces for story data | `editor/types/story.ts` | 1h |
| Create story data import/export utilities | `editor/utils/storyDataIO.ts` | 2h |

### Phase 2: Enhanced Node Component
**Goal**: Richer node display with zone and content indicators

| Task | File | Estimate |
|------|------|----------|
| Redesign SceneNode with new data fields | `editor/components/SceneNode.tsx` | 4h |
| Add content indicator badges | `editor/components/SceneNode.tsx` | 2h |
| Add zone label display | `editor/components/SceneNode.tsx` | 1h |
| Update SceneNode styles | `editor/components/SceneNode.css` | 2h |

### Phase 3: State Inspector Panel
**Goal**: Edit all state properties from UI

| Task | File | Estimate |
|------|------|----------|
| Create StateInspector component | `editor/components/StateInspector.tsx` | 6h |
| Add zone selector dropdown | `editor/components/StateInspector.tsx` | 2h |
| Add player position editor | `editor/components/StateInspector.tsx` | 3h |
| Add content attachment UI | `editor/components/StateInspector.tsx` | 4h |
| Add criteria builder | `editor/components/StateInspector.tsx` | 4h |

### Phase 4: Transition Management
**Goal**: Create and edit state transitions

| Task | File | Estimate |
|------|------|----------|
| Create TransitionBuilder component | `editor/components/TransitionBuilder.tsx` | 4h |
| Add trigger type selector | `editor/components/TransitionBuilder.tsx` | 2h |
| Add condition editor for each trigger type | `editor/components/TransitionBuilder.tsx` | 6h |
| Implement transition edge labels | `editor/components/TransitionEdge.tsx` | 3h |
| Add delete/edit transition UI | `editor/components/StateInspector.tsx` | 2h |

### Phase 5: State CRUD Operations
**Goal**: Create, read, update, delete states

| Task | File | Estimate |
|------|------|----------|
| Implement add state action | `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx` | 3h |
| Implement delete state action | `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx` | 2h |
| Implement duplicate state action | `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx` | 2h |
| Add confirmation dialogs | `editor/components/ConfirmDialog.tsx` | 2h |
| Implement undo/redo stack | `editor/managers/UndoManager.ts` | 4h |

### Phase 6: Act Grouping & Layout
**Goal**: Organize states into collapsible acts

| Task | File | Estimate |
|------|------|----------|
| Create ActGroup node wrapper | `editor/components/ActGroup.tsx` | 4h |
| Implement collapsible act sections | `editor/components/ActGroup.tsx` | 2h |
| Add auto-layout by act | `editor/utils/storyLayout.ts` | 3h |
| Add act colors and styling | `editor/components/ActGroup.css` | 2h |

### Phase 7: Testing Integration
**Goal**: Jump to states for testing

| Task | File | Estimate |
|------|------|----------|
| Implement smooth state jump | `editor/utils/stateJumper.ts` | 2h |
| Add "Test State" button | `editor/components/StateInspector.tsx` | 1h |
| Add preview mode overlay | `editor/components/TestOverlay.tsx` | 2h |

### Phase 8: Persistence & Export
**Goal**: Save and load story data

| Task | File | Estimate |
|------|------|----------|
| Implement auto-save | `editor/managers/StoryStateManager.ts` | 2h |
| Add export to JSON | `editor/utils/storyDataIO.ts` | 1h |
| Add import from JSON | `editor/utils/storyDataIO.ts` | 1h |
| Add validation on import | `editor/utils/storyValidator.ts` | 2h |

### Phase 9: Integration with Game
**Goal**: Sync editor data with game code

| Task | File | Estimate |
|------|------|----------|
| Generate gameData.js from storyData.json | `scripts/generateGameState.js` | 4h |
| Generate sceneData criteria from story | `scripts/generateSceneCriteria.js` | 4h |
| Add build script integration | `package.json` | 1h |

---

## 7. File Structure

```
editor/
├── panels/
│   └── SceneFlowNavigator/
│       ├── SceneFlowNavigator.tsx          (Main panel - enhance)
│       ├── SceneFlowNavigator.css          (Styles - enhance)
│       └── SceneFlowNavigator.test.tsx     (Tests - new)
├── components/
│   ├── SceneNode.tsx                       (Enhance)
│   ├── SceneNode.css                       (Enhance)
│   ├── StateInspector.tsx                  (NEW)
│   ├── StateInspector.css                  (NEW)
│   ├── TransitionBuilder.tsx               (NEW)
│   ├── TransitionBuilder.css               (NEW)
│   ├── TransitionEdge.tsx                  (NEW)
│   ├── TransitionEdge.css                  (NEW)
│   ├── ActGroup.tsx                        (NEW)
│   ├── ActGroup.css                        (NEW)
│   ├── ConfirmDialog.tsx                   (NEW)
│   └── TestOverlay.tsx                     (NEW)
├── managers/
│   ├── StoryStateManager.ts                (NEW)
│   └── UndoManager.ts                      (NEW)
├── data/
│   ├── gameStateData.ts                    (Deprecate - move to storyData)
│   └── storyData.json                      (NEW - source of truth)
├── types/
│   └── story.ts                            (NEW - TypeScript interfaces)
├── utils/
│   ├── storyDataIO.ts                      (NEW)
│   ├── storyLayout.ts                      (NEW)
│   ├── storyValidator.ts                   (NEW)
│   └── stateJumper.ts                      (NEW)
└── assets/
    └── icons/
        ├── video.svg                       (NEW)
        ├── dialog.svg                      (NEW)
        ├── music.svg                       (NEW)
        └── zone.svg                        (NEW)

scripts/
├── generateGameState.js                    (NEW - codegen)
└── generateSceneCriteria.js                (NEW - codegen)
```

---

## 8. Technical Considerations

### 8.1 ReactFlow Limitations
- ReactFlow is great for graphs but limited for hierarchical grouping
- **Solution**: Use ReactFlow's `group` nodes for Act containers
- Custom edge components for labeled transitions

### 8.2 Data Synchronization
- Story data exists in two places: editor (JSON) and game (JavaScript)
- **Solution**: Build script generates game code from story data
- Editor is source of truth; game code is generated

### 8.3 State Jumping
- Current implementation uses `window.location.href` (disruptive)
- **Solution**: Use postMessage communication between editor and game iframe
- Or: Use state management with hot-reload capability

### 8.4 Performance with Large Stories
- 50+ states could slow down ReactFlow
- **Solution**: Virtualization for very large stories
- Collapsed act groups don't render children

### 8.5 Undo/Redo
- Need to track all mutations to story data
- **Solution**: Command pattern with UndoManager
- Serialize undo stack to localStorage for persistence

---

## 9. Open Questions

| Question | Options | Decision Needed |
|----------|---------|-----------------|
| Should story data sync bi-directionally with game? | One-way (editor→game), Two-way sync | One-way; game code is generated |
| How to handle dialog branching visualization? | Separate nodes per branch, Edge labels | Edge labels + separate nodes for choices |
| Should we support multiple projects? | Single project, Multi-project | Single project for now |
| File format for story data? | JSON, YAML, Custom | JSON for tooling support |

---

## 10. Success Criteria

- [ ] Can add, edit, and delete story states
- [ ] Can define transitions between states with triggers
- [ ] Can assign zones to states
- [ ] Can capture player position from viewport
- [ ] Can attach content (video, dialog, music) to states
- [ ] Can test jump to any state
- [ ] Changes persist to storyData.json
- [ ] Game code can be generated from story data
- [ ] UI handles 50+ states without lag
- [ ] Undo/redo works for all operations

---

## 11. Dependencies

### External
- `reactflow` - Already installed
- `zustand` - Recommended for state management
- `immer` - Recommended for immutable updates

### Internal
- SceneManager - For zone and object data
- GameManager - For state definitions (as source)
- Dialog data - For dialog tree references
- Video data - For video asset references

---

## 12. Timeline

| Phase | Tasks | Estimate |
|-------|-------|----------|
| Phase 1: Data Layer | 4 tasks | 9h |
| Phase 2: Enhanced Node | 4 tasks | 9h |
| Phase 3: State Inspector | 5 tasks | 19h |
| Phase 4: Transitions | 5 tasks | 17h |
| Phase 5: State CRUD | 5 tasks | 13h |
| Phase 6: Act Grouping | 4 tasks | 11h |
| Phase 7: Testing | 3 tasks | 5h |
| Phase 8: Persistence | 4 tasks | 6h |
| Phase 9: Integration | 3 tasks | 9h |
| **TOTAL** | **37 tasks** | **~98 hours** |

---

## 13. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| ReactFlow doesn't support complex grouping | Medium | Custom act group component |
| Story data gets out of sync with game | High | Build script enforcement |
| Performance issues with many states | Medium | Lazy loading, virtualization |
| Undo/redo complexity | Low | Use proven library (Undo.js) |

---

## 14. Next Steps

1. **Review this PRP** with team
2. **Answer open questions** (Section 9)
3. **Create detailed tickets** for Phase 1 tasks
4. **Set up story data structure** - begin Phase 1
5. **Implement StateInspector** - highest value feature

---

**Document Version**: 1.0
**Last Updated**: 2025-01-17
