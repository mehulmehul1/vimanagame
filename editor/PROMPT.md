# Shadow Web Editor - Complete Implementation

## Goal
Complete the Shadow Web Editor implementation from 56% to 100%. All core systems (selection, gizmos, inspector) are working. Complete remaining Phases 5-7 from fix_plan.md.

## Current Status (Phase 1-4: ✅ COMPLETE)
- ✅ 4-panel layout (Hierarchy, Viewport, Inspector, AssetBrowser, SceneFlowNavigator)
- ✅ Object selection with raycasting
- ✅ Transform gizmos (Translate G, Rotate R, Scale S)
- ✅ Inspector with property editing (two-way sync)
- ✅ Undo/Redo system
- ✅ Play/Edit mode toggle
- ✅ Hot-reload scene changes
- ✅ Auto-save with crash recovery
- ✅ Criteria Builder UI
- ✅ Scene Flow Navigator (ReactFlow-based visual state editor)

## Remaining Work (Phases 5-7)

### Phase 5: Performance Optimization (6 tasks)

**TASK 25: Priority Queue Loading**
- Create `PriorityLoadManager.ts` in `core/`
- Sort objects by `priority` property before loading
- Load visible-in-frustum objects first (using frustum culling check)
- Show loading progress indicator in Viewport
- Cancel loading when switching scenes

**TASK 26: LOD System for Splats**
- Create `LODManager.ts` in `core/`
- Distance-based quality for Gaussian splats
- LOD levels: Near (16k), Medium (35k), Far (preview quality)
- Smooth transitions between LOD levels
- Per-object LOD configuration via userData

**TASK 27: Frustum Culling**
- Create `FrustumCullingManager.ts` in `core/`
- Cull objects outside camera frustum
- Per-object culling radius via `userData.cullingRadius`
- Debug overlay showing culled objects (toggle with F8)
- `userData.excludeFromCulling` flag to always render

**TASK 28: Quality Settings**
- Create `QualitySettings.ts` in `core/`
- Pixel ratio selector: Low (0.5), Medium (1), High (2)
- Editor renders at pixelRatio: 1 by default
- High DPI mode for screenshots only
- UI in menubar to select quality

**TASK 29: Virtualize Hierarchy**
- Install: `npm install react-virtuoso`
- Refactor `panels/Hierarchy/Hierarchy.tsx` to use Virtuoso
- Only render visible nodes + 10 buffer
- Support 1000+ objects smoothly

**TASK 30: Thumbnail Cache System**
- Create `ThumbnailCache.ts` in `utils/`
- Generate thumbnails on asset import
- Save to `editor/.cache/` folder as PNG
- Index cache on editor load
- Invalidate cache when asset changes (mtime check)

### Phase 6: Timeline & Animation (4 tasks)

**TASK 31: Timeline Panel Integration**
- Install: `npm install @theatre/core @theatre/react`
- Create `panels/Timeline/Timeline.tsx`
- Theatre.js integration
- Timeline scrubbing with play/pause controls
- Time display (current/total)

**TASK 32: Camera Animation Tracks**
- Add keyframes for camera position
- Add keyframes for camera lookAt target
- FOV keyframes
- Easing curve selection (linear, ease-in, ease-out, ease-in-out)
- Export to `animationCameraData.js`

**TASK 33: Object Animation Tracks**
- Position/Rotation/Scale keyframes for objects
- Property animation (visibility, opacity)
- Multiple object tracks
- Export to `animationObjectData.js`

**TASK 34: Keyframe Editing UI**
- Visual keyframe representation on timeline
- Drag to move keyframes
- Copy/paste keyframes
- Delete keyframes (Del key)
- Multi-keyframe selection (Shift+Click)

### Phase 7: Polish & Features (7 tasks)

**TASK 35: Node Graph for Visual Scripting**
- Create `panels/NodeGraph/NodeGraph.tsx`
- Use ReactFlow for node graph UI
- Event nodes: `state:changed`, `onEnter`, `onExit`, `onClick`
- Action nodes: `setState`, `showDialog`, `playSound`, `loadScene`
- Condition nodes: criteria builder (AND/OR logic)
- Code generation to JavaScript (`src/generated/sceneLogic.js`)

**TASK 36: TSL Shader Editor**
- Create `panels/ShaderEditor/ShaderEditor.tsx`
- Monaco editor for TSL code input
- Live preview on sphere in mini-viewport
- Export to material format
- Preset shaders library

**TASK 37: Console/Profiler Panel**
- Create `panels/Console/Console.tsx`
- Log viewer with filtering (info, warn, error)
- Performance stats (FPS, memory, draw calls)
- Splat count display
- Network requests log
- Manager status indicators

**TASK 38: Keyboard Shortcuts System**
- Create `core/KeyboardShortcutManager.ts`
- Global shortcuts: Ctrl+S (save), Ctrl+Z (undo), Ctrl+Shift+Z (redo), Ctrl+D (duplicate), Del (delete), F2 (rename)
- Context-sensitive shortcuts (G/R/S for gizmos when object selected)
- Shortcut reference panel (F1)
- Configurable via user preferences

**TASK 39: User Preferences**
- Create `core/UserPreferences.ts`
- Settings storage in localStorage
- Preferences UI panel (Edit → Preferences)
- Options: theme (dark/light), panel layout, auto-save interval, default gizmo mode
- Import/export preferences

**TASK 40: Toolbar and MenuBar**
- Enhance existing menubar with dropdown menus:
  - File: New, Open, Save, Export, Exit
  - Edit: Undo, Redo, Cut, Copy, Paste, Duplicate, Delete
  - View: Panel toggles, Fullscreen, Stats overlay
  - Help: Documentation, Shortcuts, About
- Toolbar with quick actions below menubar

**TASK 41: Documentation**
- Create `docs/editor-user-guide.md`
- `docs/editor-api-reference.md`
- Example scenes in `examples/` folder
- In-editor tooltips for UI elements

## Success Criteria
- All Phases 5-7 tasks completed
- Editor runs smoothly with 1000+ objects
- Timeline and animation fully functional
- Visual scripting node graph works
- User can export complete scenes
- Type check passes: `npx tsc --noEmit`
- Build succeeds: `npm run build`

## Development Commands
```bash
cd editor
npm run dev          # Start dev server at http://localhost:3001/
npm run build        # Production build
npx tsc --noEmit     # Type check only
```

## Technical Notes
- React 18 + TypeScript + Vite
- Three.js for 3D rendering
- ReactFlow for node graphs (Scene Flow Navigator already uses it)
- Theatre.js for timeline animation
- Do NOT modify `src/` directory (runtime code)
- All editor code in `editor/` directory

## Priority Order
1. **Phase 5** (Performance) - Do first for smooth workflow
2. **Phase 7** (Core polish) - MenuBar, Console, Shortcuts, Preferences
3. **Phase 6** (Animation) - Timeline/keyframes (can be deferred if time)

## Exit When
- All tasks in fix_plan.md marked as [x] complete
- Editor builds without errors
- All features functional in browser test
