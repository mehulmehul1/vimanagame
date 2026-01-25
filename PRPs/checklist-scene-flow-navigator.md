# PRP Checklist: Scene Flow Navigator - Story-Centric Redesign

**Project**: Shadow Czar Engine Editor
**Component**: Scene Flow Navigator Panel
**PRP Document**: `editor/prp-scene-flow-navigator-story-centric.md`
**Status**: In Progress
**Created**: 2025-01-17

---

## PHASE 1: Data Layer Foundation

### Task 1: Create TypeScript Interfaces for Story Data
STATUS [DONE]
CREATE `editor/types/story.ts`:
  - DEFINE `TriggerType` union type: "onComplete" | "onChoice" | "onTimeout" | "onProximity" | "onInteract" | "onState" | "custom"
  - DEFINE `PlayerPosition` interface with x, y, z, optional rotation
  - DEFINE `StateContent` interface with optional video, dialog, music, sfx, cameraAnimation
  - DEFINE `EntryCriteria` interface with currentState criteria ($gte, $lt, $in), dialogChoice, customCondition
  - DEFINE `Transition` interface with id, from, to, trigger, label, condition, dialogChoice, timeout, proximity
  - DEFINE `StoryState` interface with id, value, label, act, category, color, zone, playerPosition, content, criteria, transitions, description, notes
  - DEFINE `Act` interface with name, color, states array
  - DEFINE `StoryData` interface with states Record, acts Record
  - EXPORT all interfaces and types
  - CREATE unit tests in `editor/types/story.test.ts`

### Task 2: Create StoryStateManager Class
STATUS [DONE]
CREATE `editor/managers/StoryStateManager.ts`:
  - DEFINE class `StoryDataManager` with:
    - `private data: StoryData`
    - `private listeners: Set<(data: StoryData) => void>`
    - `private undoStack: StoryAction[]`
    - `private redoStack: StoryAction[]`
  - IMPLEMENT `async load(path: string): Promise<StoryData>` method
  - IMPLEMENT `save(path: string): Promise<void>` method
  - IMPLEMENT `getState(id: string): StoryState | undefined` method
  - IMPLEMENT `addState(state: StoryState): void` method
  - IMPLEMENT `updateState(id: string, updates: Partial<StoryState>): void` method
  - IMPLEMENT `deleteState(id: string): void` method
  - IMPLEMENT `duplicateState(id: string, newId: string, newLabel: string): StoryState` method
  - IMPLEMENT `addTransition(fromId: string, toId: string, transition: Omit<Transition, 'id'>): Transition` method
  - IMPLEMENT `deleteTransition(transitionId: string): void` method
  - IMPLEMENT `subscribe(callback: (data: StoryData) => void): () => void` method
  - IMPLEMENT `undo(): StoryAction | undefined` method
  - IMPLEMENT `redo(): StoryAction | undefined` method
  - IMPLEMENT `getNextStateValue(): number` method (for auto-increment)
  - CREATE unit tests in `editor/managers/StoryStateManager.test.ts`

### Task 3: Create Initial storyData.json from Existing Game States
STATUS [DONE]
CREATE `editor/data/storyData.json`:
  - INGEST from `src/gameData.js` GAME_STATES enum
  - INGEST from `editor/data/gameStateData.ts` STATE_CATEGORIES
  - MIGRATE all 44 states to new StoryState format
  - ASSIGN acts: Act 1 (LOADING through CAT_DIALOG_CHOICE), Act 2 (NEAR_RADIO through POST_CURSOR), Act 3 (OUTRO through GAME_OVER)
  - PRESERVE existing color codes from STATE_CATEGORIES
  - INITIALIZE empty content {} for all states
  - INITIALIZE empty transitions [] for all states
  - SET zone to null initially (to be filled later)
  - CREATE acts object with 3 acts

### Task 4: Create Story Data Import/Export Utilities
STATUS [DONE]
CREATE `editor/utils/storyDataIO.ts`:
  - IMPLEMENT `async loadStoryData(path: string): Promise<StoryData>` function
  - IMPLEMENT `async saveStoryData(data: StoryData, path: string): Promise<void>` function
  - IMPLEMENT `validateStoryData(data: unknown): data is StoryData` function with runtime type checking
  - IMPLEMENT `exportToJSON(data: StoryData): string` function with pretty formatting
  - IMPLEMENT `importFromJSON(json: string): StoryData` function with validation
  - IMPLEMENT `generateBackupPath(originalPath: string): string` function with timestamp
  - CREATE unit tests in `editor/utils/storyDataIO.test.ts`

---

## PHASE 2: Enhanced Node Component

### Task 5: Redesign SceneNode Component with New Data Fields
STATUS [DONE]
MODIFY `editor/components/SceneNode.tsx`:
  - UPDATE `SceneNodeData` interface to EXTEND `Partial<StoryState>`
  - ADD `zone?: string` field display
  - ADD `hasVideo: boolean`, `hasDialog: boolean`, `hasMusic: boolean` flags
  - ADD `transitionsCount: number` for visual indicator
  - PRESERVE existing `onJumpToState` callback
  - MODIFY component to ACCEPT new data structure
  - UPDATE TypeScript imports to INCLUDE types from `types/story.ts`
  - KEEP existing Handle components for ReactFlow connections

### Task 6: Add Content Indicator Badges to SceneNode
STATUS [DONE]
MODIFY `editor/components/SceneNode.tsx`:
  - ADD content badge section in JSX:
    - CREATE `<VideoBadge />` component when `hasVideo === true`
    - CREATE `<DialogBadge />` component when `hasDialog === true`
    - CREATE `<MusicBadge />` component when `hasMusic === true`
  - IMPLEMENT badge icons (SVG or emoji: 📹 💬 🎵)
  - ADD tooltips on hover showing content file names
  - PRESERVE existing criteria badge (◈)

### Task 7: Add Zone Label Display to SceneNode
STATUS [DONE]
MODIFY `editor/components/SceneNode.tsx`:
  - ADD zone display row below state label
  - SHOW zone name when `zone !== null`
  - USE truncated zone name if longer than 12 characters
  - ADD zone color indicator dot
  - IMPLEMENT click-to-filter functionality (optional for now)
  - PRESERVE existing layout structure

### Task 8: Update SceneNode Styles
STATUS [DONE]
MODIFY `editor/components/SceneNode.css`:
  - ADD `.scene-node-zone` class styles
  - ADD `.scene-node-content-badges` class styles
  - ADD `.scene-node-badge` class for individual content badges
  - ADD hover effects for content badges
  - MODIFY padding to accommodate new elements
  - PRESERVE existing `.scene-node.current` and `.scene-node.selected` styles
  - ENSURE responsive width (max-width: 180px)

---

## PHASE 3: State Inspector Panel

### Task 9: Create StateInspector Component Structure
STATUS [DONE]
CREATE `editor/components/StateInspector.tsx`:
  - DEFINE props interface: `{ state: StoryState | null; onUpdate: (id: string, updates: Partial<StoryState>) => void; onDelete: (id: string) => void; onDuplicate: (id: string) => void; availableZones: ZoneInfo[] }`
  - IMPLEMENT null state handling ("Select a state to inspect")
  - CREATE panel header with state name and value badge
  - CREATE collapsible sections: Zone, Position, Content, Criteria, Transitions
  - IMPLEMENT loading state indicator
  - CREATE empty state placeholder illustration

### Task 10: Add Zone Selector Dropdown to StateInspector
STATUS [DONE]
MODIFY `editor/components/StateInspector.tsx`:
  - CREATE `ZoneSelector` subcomponent
  - IMPLEMENT dropdown with `availableZones` prop
  - ADD "No Zone" option (null value)
  - IMPLEMENT onChange handler calling `onUpdate(id, { zone: value })`
  - ADD "Jump to Zone" button next to selector
  - DISPLAY zone color indicator in dropdown items
  - CREATE zone selection tests

### Task 11: Add Player Position Editor to StateInspector
STATUS [DONE]
MODIFY `editor/components/StateInspector.tsx`:
  - CREATE `PositionEditor` subcomponent
  - IMPLEMENT X, Y, Z input fields (number type, step 0.01)
  - ADD optional rotation fields (X, Y, Z in degrees)
  - IMPLEMENT "Capture from Viewport" button
    - INVOKE postMessage to game iframe
    - OR CALL game manager API if available
  - ADD "Clear Position" button
  - IMPLEMENT debounced update on input change
  - CREATE position editor tests

### Task 12: Add Content Attachment UI to StateInspector
STATUS [DONE]
MODIFY `editor/components/StateInspector.tsx`:
  - CREATE `ContentAttachment` subcomponent
  - IMPLEMENT video selector with dropdown
  - IMPLEMENT dialog selector with dropdown
  - IMPLEMENT music selector with dropdown
  - ADD file picker buttons for each content type
  - ADD preview/play buttons next to each content
  - IMPLEMENT "Remove" button for each content field
  - DISPLAY content duration/size metadata
  - CREATE content attachment tests

### Task 13: Add Entry Criteria Builder to StateInspector
STATUS [DONE]
MODIFY `editor/components/StateInspector.tsx`:
  - CREATE `CriteriaBuilder` subcomponent
  - IMPLEMENT currentState criteria editor:
    - ADD operator dropdown ($gte, $lt, $eq, $in)
    - ADD value input field
  - IMPLEMENT dialogChoice criteria editor
  - IMPLEMENT customCondition text area for JavaScript expressions
  - ADD "Add Condition" button
  - IMPLEMENT condition validation
  - DISPLAY human-readable criteria summary
  - CREATE criteria builder tests

### Task 14: Create TransitionBuilder Component
STATUS [DONE]
CREATE `editor/components/TransitionBuilder.tsx`:
  - DEFINE props interface: `{ fromState: string; availableTargets: string[]; onAdd: (transition: Omit<Transition, 'id'>) => void; }`
  - IMPLEMENT target state selector dropdown
  - IMPLEMENT trigger type selector (onComplete, onChoice, onTimeout, onProximity, onInteract, onState, custom)
  - IMPLEMENT label text input
  - IMPLEMENT conditional fields based on trigger type:
    - `onTimeout`: SHOW duration input (ms)
    - `onProximity`: SHOW x, y, z, radius inputs
    - `onChoice`: SHOW choice number input
    - `custom`: SHOW code editor area
  - IMPLEMENT "Add Transition" button with validation
  - CREATE transition builder tests

### Task 15: Add Transition List to StateInspector
STATUS [DONE]
MODIFY `editor/components/StateInspector.tsx`:
  - CREATE `TransitionList` subcomponent
  - ITERATE through `state.transitions` array
  - DISPLAY each transition with:
    - Target state name (clickable link)
    - Trigger type badge
    - Label text
    - Edit and Delete buttons
  - IMPLEMENT edit mode (opens TransitionBuilder with existing data)
  - IMPLEMENT delete confirmation
  - ADD "Add Transition" button at bottom of list
  - CREATE transition list tests

### Task 16: Add State Inspector Styles
STATUS [DONE]
CREATE `editor/components/StateInspector.css`:
  - DEFINE `.state-inspector` container styles
  - DEFINE `.state-inspector-header` with title and actions
  - DEFINE `.inspector-section` collapsible sections
  - DEFINE `.inspector-field` label/input pairs
  - DEFINE `.inspector-button` action buttons
  - DEFINE `.transition-item` list item styles
  - ADD transition animations for section expand/collapse
  - IMPLEMENT scrollable content area (max-height with overflow-y)

---

## PHASE 4: Integration with SceneFlowNavigator

### Task 17: Integrate StateInspector with SceneFlowNavigator
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - IMPORT `StoryStateManager` from `managers/StoryStateManager`
  - IMPORT `StateInspector` from `components/StateInspector`
  - IMPORT `StoryState` type from `types/story.ts`
  - ADD state manager instance (useRef or singleton)
  - ADD `selectedState` state variable
  - ADD side panel/split pane layout for StateInspector
  - IMPLEMENT onNodeClick handler to set selectedState
  - PASS state updates to StoryStateManager
  - IMPLEMENT refresh after state changes
  - PRESERVE existing ReactFlow functionality

### Task 18: Update Node Generation to Use StoryStateManager
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - MODIFY `generateNodes()` to CALL `storyStateManager.data.states`
  - REPLACE hardcoded GAME_STATES with dynamic story data
  - MAP StoryState to ReactFlow Node:
    - id = state.id
    - data.zone = state.zone
    - data.hasVideo = !!state.content.video
    - data.hasDialog = !!state.content.dialog
    - data.hasMusic = !!state.content.music
  - IMPLEMENT position calculation by act grouping
  - PRESERVE existing auto-layout behavior

### Task 19: Update Edge Generation to Show Transitions
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - MODIFY `generateEdges()` to ITERATE through each state's transitions
  - CREATE edge for each transition in state.transitions array
  - USE transition.label as edge label
  - COLOR code edges by trigger type:
    - onComplete: gray
    - onChoice: blue
    - onProximity: green
    - onInteract: purple
  - ADD animated prop for primary transitions
  - IMPLEMENT curved edges for multiple transitions between same states

---

## PHASE 5: State CRUD Operations

### Task 20: Implement Add State Action
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - ADD "Add State" button to toolbar
  - IMPLEMENT click handler:
    - CALL `storyManager.getNextStateValue()` for new value
    - PROMPT user for state name/label
    - CREATE new StoryState with defaults
    - CALL `storyManager.addState(newState)`
    - REFRESH node display
  - IMPLEMENT auto-position new node near selected node or center
  - ADD keyboard shortcut (Ctrl+N / Cmd+N)

### Task 21: Implement Delete State Action
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - ADD "Delete State" button (enabled only when state selected)
  - IMPLEMENT click handler:
    - SHOW confirmation dialog
    - CHECK for incoming transitions (warn if breaking links)
    - CALL `storyManager.deleteState(stateId)`
    - REFRESH node display
    - CLEAR selectedState
  - ADD keyboard shortcut (Delete key)

### Task 22: Implement Duplicate State Action
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - ADD "Duplicate State" button (enabled only when state selected)
  - IMPLEMENT click handler:
    - CALL `storyManager.duplicateState(sourceId, newId, newLabel)`
    - INCREMENT state value automatically
    - COPY transitions (with adjusted from/to)
    - COPY content references
    - REFRESH node display
    - SELECT new duplicated state
  - ADD keyboard shortcut (Ctrl+D / Cmd+D)

### Task 23: Create ConfirmDialog Component
STATUS [DONE]
CREATE `editor/components/ConfirmDialog.tsx`:
  - DEFINE props: `{ open: boolean; title: string; message: string; onConfirm: () => void; onCancel: () => void; }`
  - IMPLEMENT modal overlay with backdrop
  - DISPLAY title and message
  - IMPLEMENT Confirm and Cancel buttons
  - ADD keyboard handlers (Enter=Confirm, Escape=Cancel)
  - IMPLEMENT focus management (trap focus in modal)
  - CREATE dialog tests

### Task 24: Implement UndoManager
STATUS [DONE]
CREATE `editor/managers/UndoManager.ts`:
  - DEFINE class with `undoStack: Action[]`, `redoStack: Action[]`
  - IMPLEMENT `executeAction(action: Action): void` method
  - IMPLEMENT `undo(): Action | undefined` method
  - IMPLEMENT `redo(): Action | undefined` method
  - IMPLEMENT `canUndo(): boolean` getter
  - IMPLEMENT `canRedo(): boolean` getter
  - IMPLEMENT `clear(): void` method
  - ADD max stack size limit (default 100)
  - PERSIST undo stack to localStorage (optional)
  - CREATE UndoManager tests

---

## PHASE 6: Toolbar Actions

### Task 25: Add Toolbar to SceneFlowNavigator
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - ADD toolbar panel above ReactFlow canvas
  - IMPLEMENT toolbar buttons: Add State, Undo, Redo, Auto Layout, Export, Import
  - USE icon buttons (SVG or emoji) with tooltips
  - IMPLEMENT disabled states for buttons
  - ADD keyboard shortcut hints in tooltips
  - PRESERVE existing help panel

### Task 26: Implement Auto Layout Function
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - CREATE `autoLayout()` function
  - IMPLEMENT dagre layout algorithm or custom positioning:
    - GROUP states by act
    - POSITION acts horizontally with spacing
    - POSITION states within act vertically
    - CALCULATE edge routing to minimize overlaps
  - ADD "Auto Layout" button handler
  - ANIMATE transition to new positions
  - CREATE layout utility tests

### Task 27: Implement Export/Import Functions
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - ADD "Export JSON" button
  - IMPLEMENT click handler:
    - CALL `storyDataIO.exportToJSON(storyManager.data)`
    - CREATE download blob with timestamp filename
    - TRIGGER browser download
  - ADD "Import JSON" button
  - IMPLEMENT file input handler:
    - READ file content
    - CALL `storyDataIO.importFromJSON(content)`
    - VALIDATE imported data
    - REFRESH display on success
    - SHOW error on failure

---

## PHASE 7: Testing Integration

### Task 28: Implement State Jump Functionality
STATUS [DONE]
MODIFY `editor/utils/stateJumper.ts` (CREATE):
  - IMPLEMENT `jumpToState(stateName: string, stateValue: number): void` function
  - USE postMessage API for iframe communication
  - OR UPDATE URL parameter with `gameState={stateName}`
  - IMPLEMENT smooth reload (preserve viewport if possible)
  - ADD visual feedback during transition
  - CREATE state jumper tests

### Task 29: Add Test State Button to StateInspector
STATUS [DONE]
MODIFY `editor/components/StateInspector.tsx`:
  - ADD "Test State" button to inspector header
  - IMPLEMENT click handler:
    - CALL `stateJumper.jumpToState(state.id, state.value)`
    - SHOW loading indicator during transition
  - ADD keyboard shortcut (Ctrl+Enter / Cmd+Enter)
  - PRESERVE state for quick iteration

### Task 30: Create TestOverlay Component
STATUS [DONE]
CREATE `editor/components/TestOverlay.tsx`:
  - DEFINE overlay to show during state testing
  - DISPLAY current state name and value
  - ADD "Return to Editor" button
  - IMPLEMENT escape hatch (Escape key)
  - SHOW visual indicator when in test mode
  - CREATE overlay tests

---

## PHASE 8: Polish & UX

### Task 31: Add Loading States
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - ADD loading state variable
  - SHOW loading spinner during data load
  - DISPLAY skeleton nodes during async operations
  - IMPLEMENT error boundary for graceful failures
  - ADD retry button on load failure

### Task 32: Add Keyboard Shortcuts Help
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - UPDATE help panel with new shortcuts:
    - Ctrl+N: New State
    - Ctrl+D: Duplicate State
    - Delete: Delete State
    - Ctrl+Z: Undo
    - Ctrl+Y: Redo
    - Ctrl+S: Save
    - Ctrl+Enter: Test State
  - IMPLEMENT keyboard event handlers
  - PREVENT conflicts with existing ReactFlow shortcuts

### Task 33: Implement Auto-Save
STATUS [DONE]
MODIFY `editor/managers/StoryStateManager.ts`:
  - ADD auto-save timer (debounce 5 seconds)
  - IMPLEMENT `saveOnChange()` listener
  - SHOW "Saving..." indicator
  - DISPLAY "Last saved: HH:MM:SS" timestamp
  - HANDLE save failures gracefully
  - ADD manual save button

### Task 34: Add Search/Filter Functionality
STATUS [DONE]
MODIFY `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx`:
  - ADD search input to toolbar
  - IMPLEMENT filter by state name
  - IMPLEMENT filter by zone
  - IMPLEMENT filter by content type
  - HIGHLIGHT matching nodes
  - DIM non-matching nodes
  - ADD clear search button

### Task 35: Improve Edge Visualization
STATUS [DONE]
MODIFY `editor/components/TransitionEdge.tsx` (CREATE):
  - CREATE custom edge component for transitions
  - DISPLAY edge label with trigger type
  - ADD color coding by trigger type
  - IMPLEMENT hover tooltip with full details
  - ADD edge thickness variation (primary = thicker)
  - IMPLEMENT animated dashed lines for active transitions

---

## PHASE 9: Code Generation (Game Integration)

### Task 36: Create Game State Code Generator
STATUS [ ]
CREATE `scripts/generateGameState.js`:
  - READ `editor/data/storyData.json`
  - GENERATE `src/gameData.js` GAME_STATES enum
  - GENERATE `src/gameData.js` DIALOG_RESPONSE_TYPES enum
  - GENERATE `src/gameData.js` startScreen object
  - PRESERVE existing comments and structure
  - ADD code generation timestamp header
  - IMPLEMENT CLI: `node scripts/generateGameState.js`

### Task 37: Create Scene Criteria Generator
STATUS [ ]
CREATE `scripts/generateSceneCriteria.js`:
  - READ `editor/data/storyData.json`
  - EXTRACT zone and criteria from each state
  - GENERATE scene object entries with criteria
  - OUTPUT valid JavaScript for sceneData.js
  - PRESERVE existing manual entries
  - IMPLEMENT dry-run mode (--check flag)
  - IMPLEMENT CLI: `node scripts/generateSceneCriteria.js`

### Task 38: Add Build Script Integration
STATUS [ ]
MODIFY `package.json`:
  - ADD `"generate:story"` script calling both generators
  - ADD `"generate:story:watch"` script for development
  - ADD pre-build hook to run generators
  - UPDATE documentation in README
  - TEST build process end-to-end

---

## VALIDATION & TESTING

### Task 39: Create Integration Tests
STATUS [ ]
CREATE `editor/panels/SceneFlowNavigator/SceneFlowNavigator.test.tsx`:
  - TEST full component rendering
  - TEST node selection and inspection
  - TEST state addition
  - TEST state deletion
  - TEST state updates
  - TEST undo/redo flow
  - TEST import/export
  - USE React Testing Library
  - ACHIEVE >80% code coverage

### Task 40: End-to-End User Flow Test
STATUS [ ]
CREATE `editor/e2e/storyEditor.e2e.ts`:
  - TEST: Create new state from scratch
  - TEST: Assign zone and capture position
  - TEST: Attach content (video, dialog, music)
  - TEST: Create transition between states
  - TEST: Duplicate state with content
  - TEST: Export and import story data
  - TEST: Generate game code from story
  - VERIFY all generated code compiles
  - VERIFY game runs with new state

---

## COMPLETION CHECKLIST

### Documentation
STATUS [ ]
CREATE `editor/docs/SCENE_FLOW_EDITOR.md`:
  - DOCUMENT user interface
  - DOCUMENT keyboard shortcuts
  - DOCUMENT file formats (storyData.json schema)
  - DOCUMENT integration with game code
  - CREATE troubleshooting guide

### Deployment
STATUS [ ]
VERIFY build process:
  - RUN `npm run build` successfully
  - VERIFY no TypeScript errors
  - VERIFY no ESLint warnings
  - TEST generated game code in actual game
  - VERIFY state jumping works in production

---

## SUMMARY

- **Total Tasks**: 40
- **Estimated Time**: ~100 hours
- **Phases**: 9

**PREREQUISITES**:
- Node.js 18+
- React 18+
- TypeScript 5+
- ReactFlow 11+

**DEPENDENCIES**:
- Existing: ReactFlow, Three.js
- New: immer (for immutable updates), zustand (optional for state management)

**RISKS**:
- ReactFlow grouping limitations → Use custom act containers
- Data sync issues → Enforce generated code pattern
- Performance at scale → Implement virtualization if needed
