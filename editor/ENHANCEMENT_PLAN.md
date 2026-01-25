# Shadow Web Editor - Enhancement Plan

## Executive Summary

The editor has core functionality working (viewport, gizmos, inspector, hierarchy), but several panels need significant enhancements to be practical for game development. This plan outlines specific improvements needed.

---

## 1. Scene Flow Navigator Enhancements

### Current State
- Uses ReactFlow to display game states as nodes in a linear flow
- Has state categories and double-click to jump functionality
- Shows current state highlight

### Issues Identified
- Displays linear flow from state to state, not showing the actual branching/conditional nature
- Doesn't match the functionality of `sceneSelectorMenu.js` which provides:
  - Category-based organization for quick navigation
  - Direct state jumping via URL parameter
  - Current state highlighting
  - Compact list view for quick access

### Proposed Enhancements

#### 1.1 Add List View Mode (Like sceneSelectorMenu.js)
```typescript
interface SceneSelectorViewProps {
  categories: Record<string, { states: string[]; color: string; description: string }>;
  currentStateValue: number;
  onJumpToState: (stateName: string) => void;
}
```

**Implementation:**
- Add a toggle button to switch between Flow View and List View
- List view should mirror `STATE_CATEGORIES` from `sceneSelectorMenu.js`:
  ```javascript
  const STATE_CATEGORIES = {
    "Intro & Title": ["LOADING", "START_SCREEN", "INTRO", ...],
    "Exterior - Plaza": ["CAT_DIALOG_CHOICE", "NEAR_RADIO"],
    "Phone Booth Scene": ["PHONE_BOOTH_RINGING", "ANSWERED_PHONE", ...],
    // ... etc
  };
  ```

#### 1.2 Show State Criteria on Nodes
- Load criteria from `animationCameraData.js` and `animationObjectData.js`
- Display which conditions trigger each state on the node
- Show branching paths when multiple states can follow from one state

#### 1.3 Add State Transition View
- Show which events trigger state changes
- Display dialog choices that lead to different states
- Visualize conditional branches (e.g., DIALOG_RESPONSE_TYPES)

---

## 2. Timeline Panel Enhancements

### Current State
- Has timeline scrubbing, play/pause controls
- Shows camera tracks (position, lookAt, FOV)
- Can add keyframes manually
- **Empty keyframes** - no existing animation data loaded

### Issues Identified
- No keyframes loaded from existing animation data
- Cannot edit existing animations from `animationCameraData.js`
- No visualization of object animations from `animationObjectData.js`
- Timeline doesn't show actual animation durations from the data

### Proposed Enhancements

#### 2.1 Load Existing Camera Animations
```typescript
interface CameraAnimationData {
  id: string;
  type: 'animation' | 'lookat' | 'moveTo' | 'fade';
  path?: string; // For jsonAnimation type
  position?: { x: number; y: number; z: number };
  positions?: Array<{ x: number; y: number; z: number }>; // For sequence lookat
  duration?: number;
  transitionTime?: number;
  lookAtHoldDuration?: number;
  zoomOptions?: { zoomFactor: number; holdDuration: number; ... };
  criteria?: { currentState: number | { $gte: number; $lt: number } };
}
```

**Implementation:**
- Add button to "Load from animationCameraData.js"
- Parse camera animations and create tracks with keyframes
- For `jsonAnimation` type: load the JSON file and extract position keyframes
- For `lookat` type: create lookAt keyframes with position and timing
- Show criteria as labels on the timeline

#### 2.2 Load Existing Object Animations
```typescript
interface ObjectAnimationData {
  id: string;
  targetObjectId: string;
  childMeshName?: string;
  duration: number;
  properties: {
    position?: { from?: {x,y,z}; to: {x,y,z} | Array<{x,y,z}> };
    rotation?: { from?: {x,y,z}; to: {x,y,z} | Array<{x,y,z}> };
    scale?: { from?: number | {x,y,z}; to: number | {x,y,z} };
    opacity?: { from: number; to: number };
  };
  easing: string;
  criteria?: { currentState: number | { $gte: number; $lt: number } };
}
```

**Implementation:**
- When an object is selected, show its animations from `animationObjectData.js`
- Create tracks for position, rotation, scale, opacity
- Parse keyframe arrays from the `to` property
- Display easing type on keyframes

#### 2.3 Animation Timeline View
- Group animations by game state (show currentState transitions)
- Add "State Track" showing when each state is active
- Allow editing of delay, duration, priority values
- Show animation chains (playNext property)

---

## 3. Node Graph Panel Enhancements

### Current State
- Has node templates (events, actions, conditions, variables)
- Can drag nodes and connect them
- Generates JavaScript code
- **Not connected to actual game state/events**

### Issues Identified
- Generated code doesn't integrate with existing game systems
- No way to visualize existing game logic
- Nodes generate generic code, not game-specific logic
- Not integrated with the criteria/event system

### Proposed Enhancements

#### 3.1 Visual Scripting for Game Events
Connect to existing game events and state changes:

```typescript
interface GameEventNode {
  type: 'game-event';
  event: 'state:changed' | 'ui:shown' | 'video:play' | 'shadow:amplifications' | ...;
  condition?: CriteriaNode;
  actions: ActionNode[];
}

interface CriteriaNode {
  type: 'criteria';
  expression: string; // e.g., "currentState >= GAME_STATES.PHONE_BOOTH_RINGING"
  // Visual representation of criteria helper logic
}
```

#### 3.2 State Transition Graph
- Visualize state flow from `animationCameraData.js`
- Show `onComplete` callbacks as edges to new states
- Display `playNext` animation chains
- Allow editing state transitions visually

#### 3.3 Dialog Tree Editor
- Visualize dialog choices from `dialogData.js`
- Show dialog response types (EMPATH, PSYCHOLOGIST, LAWFUL, etc.)
- Allow editing dialog flow visually
- Connect dialog choices to resulting state changes

#### 3.4 Integration with Game Data
- Load existing event handlers as nodes
- Show video play events (`video:play:shadowAmplifications`)
- Display shadow speak events (`shadow:speaks`)
- Allow attaching new actions to existing events

---

## 4. Shader Editor Panel Enhancements

### Current State
- Has shader presets (Lambert, Phong, Rim Light, Toon, Hologram)
- Live preview on sphere
- Can edit vertex/fragment shaders
- Can export as JSON

### Issues Identified
- No connection to scene objects
- Cannot apply shaders to selected objects
- Shaders are self-contained, not integrated with game materials
- No way to edit materials on actual scene objects

### Proposed Enhancements

#### 4.1 Material Editor Integration
```typescript
interface MaterialEditorProps {
  selectedObject: THREE.Object3D | null;
  onMaterialUpdate: (uuid: string, material: THREE.Material) => void;
}
```

**Implementation:**
- When an object is selected, show its current material
- Allow editing material properties (color, roughness, metalness)
- Show shader uniforms if using ShaderMaterial
- Apply changes in real-time to the selected object

#### 4.2 Shader Preset Application
- Add "Apply to Selected Object" button
- Convert preset shaders to Three.js materials
- Support both MeshStandardMaterial and ShaderMaterial
- Handle texture inputs (if any)

#### 4.3 Material Library
- Save custom shaders as presets
- Load shaders from game data
- Share shaders between objects
- Export material definitions for use in `sceneData.js`

---

## 5. Panel UX Enhancements

### Current State
- Fixed panel layout
- No drag/resize capability
- No minimize functionality
- Panels switch via toolbar buttons (F4-F8)

### Issues Identified
- Cannot customize panel layout
- Cannot resize panels to fit workflow
- Panels take full center area when active
- No way to view multiple panels simultaneously

### Proposed Enhancements

#### 5.1 Draggable Panels
Use `react-grid-layout` or `react-resizable`:

```bash
npm install react-grid-layout
npm install --save-dev @types/react-grid-layout
```

```typescript
import GridLayout, { Layout } from 'react-grid-layout';

const EditorLayout: React.FC = () => {
  const [layout, setLayout] = useState<Layout[]>([
    { i: 'viewport', x: 0, y: 0, w: 8, h: 12, minW: 4, minH: 6 },
    { i: 'hierarchy', x: 0, y: 0, w: 2, h: 12, minW: 2, minH: 4 },
    { i: 'inspector', x: 10, y: 0, w: 2, h: 12, minW: 2, minH: 4 },
    { i: 'timeline', x: 0, y: 12, w: 8, h: 4, minW: 4, minH: 2 },
    { i: 'console', x: 0, y: 16, w: 12, h: 3, minW: 4, minH: 2 },
  ]);

  return (
    <GridLayout
      layout={layout}
      onLayoutChange={setLayout}
      cols={12}
      rowHeight={30}
      width={1200}
      draggableHandle=".panel-header"
    >
      {/* Panels */}
    </GridLayout>
  );
};
```

#### 5.2 Minimizable Panels
- Add minimize/maximize button to each panel header
- Collapse to just header bar
- Preserve state across sessions (localStorage)

#### 5.3 Tabbed Panels
- Allow multiple panels in the same area with tabs
- Drag tabs between areas
- Remember last active tab

#### 5.4 Saved Layouts
- Save panel layout presets
- "Animation" layout: Timeline + Node Graph large
- "Scene" layout: Scene Flow + Hierarchy
- "Debug" layout: Console + Inspector
- Save custom layouts to localStorage

---

## 6. Integration Enhancements

### 6.1 Game State Bridge
Create a bridge to communicate between editor and game:

```typescript
// editor/core/GameStateBridge.ts
class GameStateBridge {
  // Load game data files
  async loadAnimationData(): Promise<void>;
  async loadObjectData(): Promise<void>;
  async loadDialogData(): Promise<void>;

  // Subscribe to game state changes
  onStateChange(callback: (state: number) => void): void;

  // Apply changes from editor to game
  applyKeyframeEdit(trackId: string, keyframe: Keyframe): void;
  applyStateTransition(from: number, to: number): void;

  // Export editor changes back to data files
  exportToDataFiles(): void;
}
```

### 6.2 Hot Reload Support
- Watch for changes to animation data files
- Reload timeline when files change
- Preserve unsaved editor changes

### 6.3 Undo/Redo System
- Track all edits to keyframes, nodes, materials
- Ctrl+Z / Ctrl+Y support
- Per-panel undo stacks

---

## 7. Implementation Priority

### Phase 1: Core Usability (High Priority)
1. **Timeline - Load existing keyframes** from animationCameraData.js
2. **Scene Flow - Add list view** matching sceneSelectorMenu.js
3. **Panel UX - Add drag/resize** using react-grid-layout

### Phase 2: Integration (Medium Priority)
4. **Node Graph - Connect to game events** and state transitions
5. **Shader Editor - Apply to selected objects**
6. **Timeline - Object animations** from animationObjectData.js

### Phase 3: Polish (Low Priority)
7. **Undo/Redo system**
8. **Saved layouts**
9. **Hot reload support**

---

## 8. File Changes Summary

### New Files Needed
- `editor/core/GameStateBridge.ts` - Game data integration
- `editor/components/ResizablePanel.tsx` - Resizable panel wrapper
- `editor/data/animationCameraParser.ts` - Parse camera animations
- `editor/data/animationObjectParser.ts` - Parse object animations

### Modified Files
- `editor/panels/SceneFlowNavigator/SceneFlowNavigator.tsx` - Add list view
- `editor/panels/Timeline/Timeline.tsx` - Load existing animations
- `editor/panels/NodeGraph/NodeGraph.tsx` - Connect to game events
- `editor/panels/ShaderEditor/ShaderEditor.tsx` - Apply to objects
- `editor/App.tsx` - Implement drag/resize layout
- `editor/styles/editor.css` - Panel layout styles

---

## 9. Technical Considerations

### Performance
- Lazy load large animation data files
- Debounce keyframe edits
- Virtualize long node lists

### Compatibility
- Maintain backward compatibility with existing game data format
- Don't break existing animation system
- Support both editor and runtime workflows

### User Experience
- Preserve editor state when switching panels
- Show loading indicators for large data
- Provide clear feedback when applying changes
