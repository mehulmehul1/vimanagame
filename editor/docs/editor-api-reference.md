# Shadow Web Editor - API Reference

## Core Managers

The Shadow Web Editor is built around a set of singleton managers that handle different aspects of the editor.

---

## EditorManager

The core manager for the editor. Handles Three.js scene, camera, renderer, and basic operations.

### Methods

#### `getInstance(): EditorManager`
Get the singleton instance.

#### `initialize(container: HTMLElement): Promise<void>`
Initialize the editor with a DOM container.

#### `createPrimitive(type: 'box' | 'sphere' | 'plane' | 'cone' | 'cylinder'): THREE.Object3D`
Create a primitive object and add it to the scene.

#### `deleteObject(object: THREE.Object3D): void`
Delete an object from the scene.

#### `selectObject(object: THREE.Object3D | null): void`
Select an object (emits `selectionChanged` event).

#### `enterPlayMode(): void`
Switch to play mode (disables editor controls).

#### `exitPlayMode(): void`
Switch to edit mode (enables editor controls).

### Events

| Event | Data | Description |
|-------|------|-------------|
| `initialized` | - | Editor has been initialized |
| `selectionChanged` | `{ previous, current }` | Selection has changed |
| `objectCreated` | `{ object, type }` | Object was created |
| `objectDeleted` | `{ object }` | Object was deleted |
| `playModeEntered` | - | Entered play mode |
| `playModeExited` | - | Exited play mode |
| `resize` | `{ width, height }` | Viewport was resized |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `scene` | `THREE.Scene` | Three.js scene |
| `camera` | `THREE.PerspectiveCamera` | Scene camera |
| `renderer` | `THREE.WebGLRenderer` | WebGL renderer |
| `sparkRenderer` | `SparkRenderer \| null` | Gaussian splat renderer |
| `orbitControls` | `OrbitControls \| null` | Camera controls |
| `isPlaying` | `boolean` | Play mode state |
| `selectedObject` | `THREE.Object3D \| null` | Currently selected object |

---

## SceneLoader

Handles loading of scene objects from `sceneData.js`.

### Methods

#### `getInstance(): SceneLoader`
Get the singleton instance.

#### `loadSceneObject(objectData: SceneObjectData): Promise<THREE.Object3D \| null>`
Load a single scene object.

#### `unloadSceneObject(objectId: string): void`
Unload a scene object.

#### `unloadAll(): void`
Unload all scene objects.

#### `getObject(objectId: string): THREE.Object3D \| undefined`
Get a loaded object by ID.

### Events

| Event | Data | Description |
|-------|------|-------------|
| `objectLoaded` | `{ id, object }` | Object was loaded |
| `objectLoadError` | `{ id, error }` | Object failed to load |
| `objectUnloaded` | `{ id }` | Object was unloaded |

---

## SelectionManager

Manages multi-object selection.

### Methods

#### `getInstance(): SelectionManager`
Get the singleton instance.

#### `select(object: THREE.Object3D): void`
Select a single object (clears previous selection).

#### `addToSelection(object: THREE.Object3D): void`
Add object to selection (multi-select).

#### `removeFromSelection(object: THREE.Object3D): void`
Remove object from selection.

#### `clearSelection(): void`
Clear all selections.

#### `getSelection(): THREE.Object3D[]`
Get array of selected objects.

#### `getSelectedObjects(): Set<THREE.Object3D>`
Get Set of selected objects.

---

## TransformGizmoManager

Manages transform gizmos (translate, rotate, scale).

### Methods

#### `getInstance(): TransformGizmoManager`
Get the singleton instance.

#### `setMode(mode: 'translate' | 'rotate' | 'scale'): void`
Set the gizmo mode.

#### `getMode(): 'translate' | 'rotate' | 'scale'`
Get current gizmo mode.

#### `setEnabled(enabled: boolean): void`
Enable/disable gizmo rendering.

---

## UndoRedoManager

Implements undo/redo functionality using the command pattern.

### Methods

#### `getInstance(): UndoRedoManager`
Get the singleton instance.

#### `executeCommand(command: Command): void`
Execute a command (adds to undo stack).

#### `undo(): void`
Undo last command.

#### `redo(): void`
Redo last undone command.

#### `canUndo(): boolean`
Check if undo is available.

#### `canRedo(): boolean`
Check if redo is available.

#### `clear(): void`
Clear undo/redo history.

---

## DataManager

Handles reading/writing `sceneData.js`.

### Methods

#### `getInstance(): DataManager`
Get the singleton instance.

#### `loadSceneData(): Promise<SceneDataFormat>`
Load scene data from `gameData.js`.

#### `saveSceneData(sceneData: SceneDataFormat): Promise<void>`
Save scene data to `sceneData.js`.

#### `getSceneData(): SceneDataFormat \| null`
Get current scene data.

### Events

| Event | Data | Description |
|-------|------|-------------|
| `dataLoaded` | `SceneDataFormat` | Scene data was loaded |
| `dataSaved` | `SceneDataFormat` | Scene data was saved |
| `dataChanged` | - | Scene data was modified |

---

## AutoSaveManager

Handles automatic scene saving with crash recovery.

### Methods

#### `getInstance(): AutoSaveManager`
Get the singleton instance.

#### `enable(intervalMinutes: number): void`
Enable auto-save with specified interval.

#### `disable(): void`
Disable auto-save.

#### `saveNow(): void`
Trigger an immediate save.

#### `hasRecoveryData(): boolean`
Check if crash recovery data exists.

#### `recover(): Promise<SceneDataFormat \| null>`
Recover from crash if data exists.

---

## PriorityLoadManager (Phase 5)

Manages priority-based scene loading.

### Methods

#### `getInstance(): PriorityLoadManager`
Get the singleton instance.

#### `loadSceneWithPriority(sceneData: SceneDataFormat, options?): Promise<void>`
Load scene with priority ordering.

#### `cancelLoading(): void`
Cancel current loading operation.

#### `getLoadProgress(): { loaded, total, current }`
Get loading progress.

---

## LODManager (Phase 5)

Manages Level of Detail for Gaussian splats.

### Methods

#### `getInstance(): LODManager`
Get the singleton instance.

#### `start(): void`
Start LOD updates.

#### `stop(): void`
Stop LOD updates.

#### `registerObject(object: THREE.Object3D, config?): void`
Register object for LOD.

#### `unregisterObject(object: THREE.Object3D): void`
Unregister object from LOD.

#### `forceLODLevel(object: THREE.Object3D, level: number): void`
Force specific LOD level.

---

## FrustumCullingManager (Phase 5)

Manages frustum culling for performance.

### Methods

#### `getInstance(): FrustumCullingManager`
Get the singleton instance.

#### `start(): void`
Start culling updates.

#### `stop(): void`
Stop culling updates.

#### `registerObject(object: THREE.Object3D, config?): void`
Register object for culling.

#### `toggleDebugOverlay(): void`
Toggle debug overlay (F8).

#### `getStats(): { total, culled, visible }`
Get culling statistics.

---

## QualitySettings (Phase 5)

Manages rendering quality settings.

### Methods

#### `getInstance(): QualitySettings`
Get the singleton instance.

#### `setQualityLevel(level: 'low' | 'medium' | 'high'): void`
Set quality preset.

#### `setPixelRatio(ratio: number): void`
Set custom pixel ratio.

#### `takeScreenshot(): string`
Capture high-quality screenshot.

---

## ThumbnailCache (Phase 5)

Manages thumbnail generation and caching.

### Methods

#### `getInstance(): ThumbnailCache`
Get the singleton instance.

#### `getThumbnail(assetPath: string, assetMtime?): Promise<string \| null>`
Get cached thumbnail.

#### `generateThumbnail(assetPath, assetType, assetData?): Promise<string>`
Generate new thumbnail.

#### `invalidate(assetPath: string): void`
Invalidate cached thumbnail.

---

## KeyboardShortcutManager (Phase 7)

Manages global keyboard shortcuts.

### Methods

#### `getInstance(): KeyboardShortcutManager`
Get the singleton instance.

#### `register(shortcut: KeyboardShortcut): void`
Register a keyboard shortcut.

#### `unregister(id: string): void`
Unregister a shortcut.

#### `setContext(context): void`
Set active context.

#### `getShortcutsByCategory(): ShortcutCategory[]`
Get shortcuts organized by category.

---

## UserPreferences (Phase 7)

Manages user settings with localStorage persistence.

### Methods

#### `getInstance(): UserPreferences`
Get the singleton instance.

#### `get<K>(key: K): UserPreferenceSchema[K]`
Get a preference value.

#### `set<K>(key: K, value): void`
Set a preference value.

#### `exportToFile(): void`
Export preferences to file.

#### `importFromFile(file: File): Promise<boolean>`
Import preferences from file.

---

## Types

### SceneObjectData

```typescript
interface SceneObjectData {
    id: string;
    type: 'splat' | 'gltf' | 'primitive';
    path?: string;
    description?: string;
    position: { x: number; y: number; z: number };
    rotation: { x: number; y: number; z: number };
    scale: { x: number; y: number; z: number } | number;
    priority?: number;
    preload?: boolean;
    criteria?: Record<string, any>;
    options?: {
        visible?: boolean;
        debugMaterial?: boolean;
        gizmo?: boolean;
        // ... more options
    };
}
```

### SceneDataFormat

```typescript
type SceneDataFormat = Record<string, SceneObjectData>;
```

---

## Usage Examples

### Creating a Custom Panel

```typescript
import React, { useEffect } from 'react';
import EditorManager from '../core/EditorManager';

const MyPanel: React.FC = () => {
    const [editorManager] = useState(() => EditorManager.getInstance());

    useEffect(() => {
        const handleSelectionChange = (data: any) => {
            console.log('Selection changed:', data.current);
        };

        editorManager.on('selectionChanged', handleSelectionChange);

        return () => {
            editorManager.off('selectionChanged', handleSelectionChange);
        };
    }, [editorManager]);

    return (
        <div className="my-panel">
            {/* Panel content */}
        </div>
    );
};
```

### Creating a Custom Command

```typescript
import UndoRedoManager from '../core/UndoRedoManager';

class MoveObjectCommand {
    constructor(
        private object: THREE.Object3D,
        private newPosition: THREE.Vector3,
        private oldPosition: THREE.Vector3
    ) {}

    execute(): void {
        this.object.position.copy(this.newPosition);
    }

    undo(): void {
        this.object.position.copy(this.oldPosition);
    }
}

// Usage
const command = new MoveObjectCommand(
    selectedObject,
    newPosition,
    oldPosition
);
UndoRedoManager.getInstance().executeCommand(command);
```

---

For more information, see the [User Guide](./editor-user-guide.md).
