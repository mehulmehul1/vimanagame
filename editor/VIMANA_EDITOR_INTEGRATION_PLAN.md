# Vimana + Shadow Web Editor Integration Plan

## Executive Summary

**YES, we can open the vimana game in the Shadow Web Editor!**

The vimana game uses the same tech stack as the editor (Three.js + Spark.js + Vite), and the editor already supports loading GLB/GLTF models and sceneData.js files. This integration will enable visual editing of the music room scene, harp string tuning, water shader presets, vortex settings, physics colliders, and intro video.

---

## Why It Will Work

| Component | Vimana Has | Editor Supports | Status |
|-----------|-----------|-----------------|--------|
| **GLB Models** | ✅ music_room.glb | ✅ GLTFLoader + SceneLoader | 🟢 READY |
| **Gaussian Splats** | ✅ (via Spark.js) | ✅ SparkRenderer integration | 🟢 READY |
| **Three.js Scene** | ✅ | ✅ Full Three.js support | 🟢 READY |
| **sceneData.js** | ✅ | ✅ DataManager reads/writes | 🟢 READY |
| **Lighting** | ✅ lightData.js | ✅ LightManager + Inspector | 🟢 READY |
| **Transform** | ✅ position/rotation/scale | ✅ Gizmos (G/R/S) | 🟢 READY |
| **Physics** | ✅ Rapier colliders | ⬜ Need PhysicsPanel | 🟡 TODO |
| **Video** | ✅ Intro video | ⬜ Need VideoPanel | 🟡 TODO |

---

## 5-Phase Integration Plan

### Phase 1: Quick Load (Immediate) - 2-4 hours ⚡

Load vimana scene into editor with zero code changes:

| Task | Description | File |
|------|-------------|------|
| 1.1 | Create `editor/data/vimanaConfig.ts` with paths to vimana assets | `editor/data/vimanaConfig.ts` |
| 1.2 | Add "Open Project" dropdown to MenuBar | `editor/components/MenuBar.tsx` |
| 1.3 | Test loading music_room.glb in editor | `editor/core/SceneLoader.ts` |

**Outcome**: Editor can load and display vimana music room with all objects visible in Hierarchy

---

### Phase 2: Vimana-Specific Panels - 1-2 days 🎨

Create panels tailored for vimana development:

| Panel | Features | Priority |
|-------|----------|----------|
| **VideoPanel** | Preview intro video with alpha, edit start/duration/loop | HIGH |
| **PhysicsPanel** | Visualize/edit Rapier colliders (boxes, capsules) | HIGH |
| **MusicRoomPanel** | Harp string editor, water shader presets, vortex editor | MEDIUM |

**VideoPanel Features:**
- Video preview with alpha overlay
- Start time, duration, loop settings
- Click-to-start overlay toggle
- Export to videoData.js

**PhysicsPanel Features:**
- Visualize collider shapes in 3D viewport
- Edit position, size, rotation
- Add/remove colliders
- Export to colliderData.js

**MusicRoomPanel Features:**
- Harp string frequency editor (6 strings)
- Water shader presets (Calm, Ripple, Bioluminescent, Resonance)
- Vortex SDF torus uniform editor
- Particle system controls

---

### Phase 3: Harp Room Tools - 2-3 days 🎵

Implement vimana-specific tools for the music room:

| Tool | Description |
|------|-------------|
| **Water Shader Presets** | Calm, Ripple, Bioluminescent, Resonance with real-time uniform editing |
| **Harp String Editor** | 6 strings, frequency tuning, vibration preview, Timeline integration |
| **Vortex Editor** | SDF torus uniforms, particle controls, activation level slider |

---

### Phase 4: Gameplay Integration - 3-4 days 🎮

Connect editor to vimana gameplay:

| Feature | Description |
|---------|-------------|
| **SceneFlowNavigator - Vimana Mode** | Replace Shadow Czar states with vimana states (LOADING → VIDEO_INTRO → MUSIC_ROOM) |
| **Timeline - Harp Animation** | Harp string animation tracks, keyframe string plucks, water ripple animations |
| **Play Mode Integration** | Editor "Play Mode" → vimana gameplay, test harp interaction, physics, video intro |

---

### Phase 5: Two-Way Sync - 1-2 days 🔄

Enable seamless editing workflow:

```
Vimana Game          Shadow Web Editor
     │                       │
     │  1. Load scene   ◄────┼────┐
     │  2. Edit objects         │
     │  3. Save scene    ─────►┼────┤
     │                       │    ▼
     │  4. Refresh     ◄──────┼── Run game
     │                       │
```

**Implementation:**
- FileWatcher watches vimana/src/sceneData.js
- Auto-reload on save
- "Test in Game" button opens vimana in new tab

---

## Task List

17 total tasks (9 vimana integration + 8 shadow editor enhancements)

### Vimana Integration Tasks (9 tasks)

| ID | Priority | Task | Hours |
|----|----------|------|-------|
| vimana-editor-1 | HIGH | Create vimana project config | 1 |
| vimana-editor-2 | HIGH | Add "Open Project" dropdown | 2 |
| vimana-editor-3 | HIGH | Test vimana scene load | 1 |
| vimana-editor-4 | HIGH | Create VideoPanel | 4 |
| vimana-editor-5 | HIGH | Create PhysicsPanel | 6 |
| vimana-editor-6 | MEDIUM | Create MusicRoomPanel | 6 |
| vimana-editor-7 | MEDIUM | Update SceneFlowNavigator | 3 |
| vimana-editor-8 | MEDIUM | Add Timeline harp tracks | 4 |
| vimana-editor-9 | LOW | Two-way sync with FileWatcher | 2 |

**Total: ~29 hours**

### Shadow Editor Enhancements (8 tasks)

| ID | Priority | Task | Hours |
|----|----------|------|-------|
| shadow-editor-1 | HIGH | SceneFlowNavigator - Add List View | 4 |
| shadow-editor-2 | HIGH | Timeline - Load camera animations | 6 |
| shadow-editor-3 | HIGH | Timeline - Load object animations | 5 |
| shadow-editor-7 | HIGH | GameStateBridge - Integration layer | 5 |
| shadow-editor-4 | MEDIUM | NodeGraph - Connect to events | 8 |
| shadow-editor-5 | MEDIUM | ShaderEditor - Apply to objects | 4 |
| shadow-editor-6 | LOW | Panel UX - Drag/resize layout | 6 |
| shadow-editor-8 | LOW | Undo/Redo - Per-panel stacks | 4 |

**Total: ~42 hours**

---

## Project Structure

```
shadowczarengine/
├── editor/                          # Shadow Web Editor (98% complete)
│   ├── data/
│   │   └── vimanaConfig.ts          # NEW: Vimana project config
│   ├── panels/
│   │   ├── VideoPanel/              # NEW: Video editing panel
│   │   │   ├── VideoPanel.tsx
│   │   │   ├── VideoPanel.css
│   │   │   └── index.ts
│   │   ├── PhysicsPanel/            # NEW: Physics collider editor
│   │   │   ├── PhysicsPanel.tsx
│   │   │   ├── PhysicsPanel.css
│   │   │   └── index.ts
│   │   └── MusicRoomPanel/          # NEW: Harp & water editor
│   │       ├── MusicRoomPanel.tsx
│   │       ├── MusicRoomPanel.css
│   │       └── index.ts
│   └── components/
│       └── MenuBar.tsx              # UPDATE: Add project selector
└── vimana/                          # Vimana game (existing)
    ├── src/
    │   ├── sceneData.js             # EDIT: Loaded by editor
    │   ├── colliderData.js          # EDIT: Editable in PhysicsPanel
    │   ├── lightData.js             # EDIT: Editable in Inspector
    │   ├── videoData.js             # EDIT: Editable in VideoPanel
    │   └── gameData.js              # READ: SceneFlowNavigator states
    └── public/
        └── assets/
            └── models/
                └── music_room.glb   # LOAD: Displayed in editor
```

---

## Tech Stack

- **Editor**: React + TypeScript + PCUI + Three.js + Spark.js + ReactFlow
- **Vimana**: Three.js + Spark.js + Rapier physics + Vite
- **Common**: Both use Three.js r180, Spark.js 0.1.10, Vite 7.x

---

## Next Steps

### Immediate (Phase 1)
1. ✅ Analysis complete
2. ⬜ Create `editor/data/vimanaConfig.ts`
3. ⬜ Add project selector to MenuBar
4. ⬜ Test loading music_room.glb

### This Week (Phase 2)
- Create VideoPanel
- Create PhysicsPanel
- Test basic editing workflow

### Next Week (Phase 3-4)
- MusicRoomPanel with harp editor
- SceneFlowNavigator vimana mode
- Timeline harp tracks

---

## Conclusion

The vimana game **can be opened in the Shadow Web Editor** with minimal code changes. The integration will enable:

1. **Visual scene editing** - Edit music room objects in 3D viewport
2. **Harp string tuning** - Visual editor for 6 strings with real-time preview
3. **Water shader presets** - Quick presets for different moods
4. **Physics collider editing** - Visual colliders with box/capsule shapes
5. **Intro video editing** - Preview and edit video with alpha
6. **Animation timeline** - Keyframe harp plucks and water ripples

**Estimated time to Phase 1 completion**: 2-4 hours
**Estimated time to full integration**: ~10-14 days

---

*Last updated: 2026-01-23*
