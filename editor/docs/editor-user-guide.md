# Shadow Web Editor - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Working with Scenes](#working-with-scenes)
4. [Object Manipulation](#object-manipulation)
5. [Animation Timeline](#animation-timeline)
6. [Performance Optimization](#performance-optimization)
7. [Keyboard Shortcuts](#keyboard-shortcuts)
8. [Preferences](#preferences)

---

## Getting Started

### Installation

The Shadow Web Editor is a web-based 3D editor for the Shadow Czar Engine. To run the editor:

```bash
cd editor
npm run dev
```

The editor will open at `http://localhost:3001/`

### First Launch

On first launch, the editor will:
- Initialize the 3D viewport
- Load any existing scene data
- Set up default panels (Hierarchy, Viewport, Inspector)

---

## Interface Overview

### Main Layout

The editor consists of four main panels:

```
+------------------+------------------------------+------------------+
|   Hierarchy      |          Viewport             |    Inspector     |
|   (20% width)    |          (50% width)          |   (30% width)    |
+------------------+------------------------------+------------------+
|                    Menu Bar / Status Bar                        |
+---------------------------------------------------------------+
```

### Panels

1. **Hierarchy (Left)** - Scene tree view showing all objects
2. **Viewport (Center)** - 3D rendering area with camera controls
3. **Inspector (Right)** - Property editor for selected objects
4. **Timeline/Console (Bottom)** - Animation tools and debug info (toggleable)

---

## Working with Scenes

### Loading Scenes

Use the **Load Shadow Czar Scene** button in the menu bar to load a scene. The editor will:
- Parse `gameData.js` for scene objects
- Load Gaussian splats, GLTF models, and primitives
- Apply transforms from the scene data

### Scene Flow Navigator

Press **F4** or click the **Scene Flow** button to open the visual state editor:
- Double-click a state node to jump to that scene
- Use arrow keys to navigate between states
- Press Enter on a state to load it

### Saving Scenes

Press **Ctrl+S** to save changes to `sceneData.js`. Auto-save is enabled by default every 2 minutes.

---

## Object Manipulation

### Selection

- **Click** - Select object
- **Ctrl+Click** - Multi-select
- **Escape** - Deselect all

### Transform Gizmos

Select an object and press:
- **G** - Translate mode (move)
- **R** - Rotate mode
- **S** - Scale mode

Use the colored handles to transform:
- **Red** - X axis
- **Green** - Y axis
- **Blue** - Z axis

### Hierarchy Operations

- **F2** or **Double-click** - Rename object
- **Eye icon** - Toggle visibility
- **Lock icon** - Toggle lock state
- **Drag** - Reorder objects in hierarchy

---

## Animation Timeline

### Opening the Timeline

Press **Ctrl+3** or select from the View menu.

### Creating Animations

1. Select an object or camera
2. Click **+ Track** to add an animation track
3. Move the playhead to desired time
4. Press **K** or click **+ Keyframe** to add a keyframe
5. Adjust object properties at different times
6. Press **Space** to play

### Timeline Controls

- **Space** - Play/Pause
- **Home** - Jump to start
- **End** - Jump to end
- **K** - Add keyframe at current time
- **Delete** - Remove selected keyframe

### Easing Options

Select a keyframe and choose easing:
- Linear - Constant speed
- Ease In - Accelerate
- Ease Out - Decelerate
- Ease In-Out - Accelerate then decelerate

---

## Performance Optimization

### Quality Settings

Access via **View → Quality**:
- **Low** - 0.5x pixel ratio, no shadows
- **Medium** - 1x pixel ratio, shadows enabled
- **High** - 2x pixel ratio, full quality

### Frustum Culling

Objects outside the camera view are automatically culled (F8 to toggle debug overlay).

### LOD System

Gaussian splats automatically adjust quality based on distance:
- Near (0-20 units) - Full quality
- Medium (20-50 units) - 60% quality
- Far (50+ units) - Preview quality

---

## Keyboard Shortcuts

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save scene |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `Ctrl+D` | Duplicate selection |
| `Delete` | Delete selection |
| `F1` | Show shortcuts reference |
| `F2` | Rename selected |
| `F4` | Toggle Scene Flow Navigator |
| `F8` | Toggle frustum culling debug |
| `Ctrl+P` | Toggle play/edit mode |

### Viewport Shortcuts

| Shortcut | Action |
|----------|--------|
| `G` | Translate gizmo |
| `R` | Rotate gizmo |
| `S` | Scale gizmo |
| `F` | Focus on selection |
| `Shift+F` | Frame selection |

### Panel Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+1` | Toggle Hierarchy |
| `Ctrl+2` | Toggle Inspector |
| `Ctrl+3` | Toggle Timeline |
| `Ctrl+4` | Toggle Console |

---

## Preferences

Access preferences via **Edit → Preferences**:

### Appearance
- **Theme** - Dark or light mode
- **Font Size** - Small, medium, or large
- **Accent Color** - UI highlight color

### Editor Behavior
- **Auto-save Interval** - Minutes between saves
- **Default Gizmo Mode** - Initial transform tool
- **Show Grid/Axes** - Viewport helpers

### Performance
- **Quality Level** - Render quality preset
- **Enable Frustum Culling** - Hide off-screen objects
- **Enable LOD** - Distance-based quality

### Import/Export

Export your preferences to share settings between machines:
- **Export** - Saves to JSON file
- **Import** - Load settings from file

---

## Tips & Tricks

1. **Performance** - Use frustum culling and LOD for large scenes
2. **Workflow** - Learn keyboard shortcuts for faster editing
3. **Backup** - Enable "Create backup on save" in preferences
4. **Debugging** - Use Console panel (Ctrl+4) to view logs and performance
5. **Animation** - Set keyframes on whole seconds for easier editing

---

## Troubleshooting

### Scene not loading

1. Check browser console for errors (F12)
2. Verify `gameData.js` exists and is valid
3. Check file paths for splat/model assets

### Poor performance

1. Lower quality setting in View menu
2. Enable frustum culling and LOD
3. Reduce number of visible splats

### Gizmo not showing

1. Make sure an object is selected
2. Check that object is not locked
3. Try cycling gizmo modes (G/R/S)

---

For more information, see the [API Reference](./editor-api-reference.md).
