# Shadow Engine Editor - Complete Feature Specification

> A Unity/Unreal-inspired web-based game engine editor for Shadow Czar Engine
> Tech Stack: WebGPU, Spark.js, Three.js, React Fiber, TypeScript

---

## Table of Contents

1. [Editor Overview](#1-editor-overview)
2. [Core Architecture](#2-core-architecture)
3. [Panel Specifications](#3-panel-specifications)
4. [Data Format Specifications](#4-data-format-specifications)
5. [Asset Pipeline](#5-asset-pipeline)
6. [Visual Scripting System](#6-visual-scripting-system)
7. [Build & Deploy](#7-build--deploy)
8. [Collaboration Features](#8-collaboration-features)
9. [Performance & Profiling](#9-performance--profiling)
10. [Roadmap](#10-roadmap)

---

## 1. Editor Overview

### Editor Layout

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  🎮 SHADOW ENGINE EDITER                                    MyProject 🔽 □ □ │
├──────────┬──────────────────────────────────────────┬───────────────────────────────┤
│          │                                          │                               │
│  ASSETS  │         SCENE VIEWPORT                  │        INSPECTOR             │
│          │  ┌────────────────────────────────┐    │                               │
│ 📦 Models │  │                          │    │  🎯 Selected Object           │
│ 🌐 Splats │  │                          │    │  ┌───────────────────────┐   │
│ 🎵 Audio  │  │      3D / 2D VIEW         │    │  │ Transform              │   │
│ 🖼️ Images │  │                          │    │  │ Position X Y Z        │   │
│ 🎬 Videos │  │                          │    │  │ Rotation X Y Z        │   │
│ 📜 Scripts │  │      Gizmo Overlay        │    │  │ Scale    X Y Z        │   │
│ 🎨 Materials│                           │    │  └───────────────────────┘   │
│ 📁 Scenes  │  └────────────────────────────────┘    │  ┌───────────────────────┐   │
│ 📁 Prefabs │                                          │  │ Material ▼             │   │
│          │  ┌────────────────────────────────┐    │  │ Base Color   [██]     │   │
│ ───────── │  │ HIERARCHY / OUTLINER        │    │  │ Metalness   [───]     │   │
│          │  │ 🌐 plaza                      │    │  │ Roughness   [───]     │   │
│ FAVORITES │  │   ┊ 📦 phonebooth              │    │  └───────────────────────┘   │
│ ⭐ plaza  │  │   ┊ 💡 officeLight             │    │  ┌───────────────────────┐   │
│ ⭐ office │  │                              │    │  │ Components ▼          │   │
│          │  │ 💡 POINT LIGHTS (3)         │    │  │ ☑ PhysicsBody          │   │
│ TOOLBOX  │  │ 📦 GLTF OBJECTS (15)        │    │  │ ☑ TriggerZone         │   │
│          │  │ 🎬 VIDEOS (2)                │    │  │ ☑ AudioSource          │   │
│ 🔷 Cube   │  └────────────────────────────────┘    │  │ ☑ AnimationController   │   │
│ ⚪ Sphere │                                          │  │ [+ Add Component]      │   │
│ 💡 Point  │  ┌────────────────────────────────┐    │  └───────────────────────┘   │
│ 🔦 Spot   │  │ VISUAL SCRIPT / LOGIC        │    │                               │
│          │  │ ┌─── onStateChanged ─────┐   │    │  ┌───────────────────────┐   │
│          │  │ │  STATE_CHANGED ───────┼──►│    │  │ Console ▼              │   │
│          │  │ └────────────────────────┘   │    │  │ No errors or warnings │   │
│          │  │                              │    │  └───────────────────────┘   │
│          │  └────────────────────────────────┘    │                               │
└──────────┴──────────────────────────────────────────┴───────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ ▶ Play | ⏸ Pause | ⏹ Stop | 💾 Save | 🔃 Build | 📦 Export | ⚙ Settings | ? Help │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Toolbar Icons & Tools

| Tool | Icon | Shortcut | Description |
|------|------|----------|-------------|
| Select | ↖ | Q | Select objects in scene |
| Move | ✥ | G | Translate objects (Gizmo) |
| Rotate | ↻ | R | Rotate objects |
| Scale | ⤡ | S | Scale objects |
| Pan | ✋ | Middle | Pan viewport |
| Orbit | 🔄 | Alt+Left | Rotate camera around selection |
| Zoom | 🔍 | Scroll / ± | Zoom in/out |
| Snap | 🧲 | Ctrl | Grid snapping toggle |

---

## 2. Core Architecture

### 2.1 Editor State Management

```typescript
interface EditorState {
  // Project
  projectId: string;
  projectName: string;
  projectPath: string;
  lastSaved: Date;

  // Scene
  currentSceneId: string;
  openScenes: string[];
  unsavedChanges: boolean;

  // Selection
  selectedObjects: string[];
  activeObject: string | null;

  // Edit Mode
  playMode: boolean;
  paused: boolean;
  editMode: "scene" | "visual" | "asset";

  // Viewport
  viewportMode: "3d" | "2d" | "game";
  cameraState: CameraState;

  // Tools
  activeTool: "select" | "move" | "rotate" | "scale" | "pan" | "orbit";
  snapEnabled: boolean;
  snapValue: number;

  // History
  undoStack: Command[];
  redoStack: Command[];
}
```

### 2.2 Manager Integration

```typescript
class EditorManager {
  // Core managers from engine
  gameManager: GameManager;
  sceneManager: SceneManager;
  zoneManager: ZoneManager;
  lightManager: LightManager;
  inputManager: InputManager;
  physicsManager: PhysicsManager;
  animationManager: AnimationManager;
  dialogManager: DialogManager;
  musicManager: MusicManager;
  sfxManager: SFXManager;
  videoManager: VideoManager;
  vfxManager: VFXManager;
  uiManager: UIManager;

  // Editor-specific managers
  assetDatabase: AssetDatabase;
  projectManager: ProjectManager;
  buildManager: BuildManager;
  consoleManager: ConsoleManager;
  collaborationManager: CollaborationManager;
  profilerManager: ProfilerManager;
}
```

### 2.3 Event System

```typescript
interface EditorEvents {
  // Object events
  "object:selected": (objects: SceneObject[]) => void;
  "object:created": (object: SceneObject) => void;
  "object:deleted": (objectId: string) => void;
  "object:changed": (objectId: string, change: ObjectChange) => void;
  "object:parented": (childId: string, parentId: string) => void;
  "object:unparented": (childId: string) => void;

  // Scene events
  "scene:loaded": (sceneId: string) => void;
  "scene:saved": (sceneId: string) => void;
  "scene:changed": () => void;

  // Asset events
  "asset:imported": (asset: Asset) => void;
  "asset:deleted": (assetId: string) => void;

  // Editor events
  "editor:play": () => void;
  "editor:pause": () => void;
  "editor:stop": () => void;
  "editor:build": () => void;

  // Component events
  "component:added": (objectId: string, component: Component) => void;
  "component:removed": (objectId: string, componentType: string) => void;
  "component:changed": (objectId: string, component: Component) => void;
}
```

---

## 3. Panel Specifications

### 3.1 Asset Browser Panel

```typescript
interface AssetBrowserPanel {
  // Tabs
  tabs: ("models" | "splats" | "audio" | "images" | "videos" | "materials" | "scripts" | "scenes" | "prefabs")[];

  // Features
  searchQuery: string;
  filterType: string;
  viewMode: "grid" | "list" | "tree";
  previewEnabled: boolean;

  // Drag & Drop
  canDragTo: ("viewport" | "hierarchy" | "inspector")[];

  // Import
  importDialog: ImportDialog;

  // Asset Metadata
  getAssetInfo(assetId: string): AssetInfo;
}
```

**Features:**
- Thumbnail preview generation
- Asset tagging and favorites
- Recent assets
- Collection folders
- Bulk import
- Asset reimporting
- Dependency tracking

### 3.2 Hierarchy / Outliner Panel

```typescript
interface HierarchyPanel {
  // Tree Structure
  rootNode: SceneNode;
  displayMode: "hierarchy" | "flat" | "type";

  // Features
  searchQuery: string;
  filterType: string;
  sortBy: "name" | "type" | "layer";

  // Node Types
  nodeTypes: {
    "folder": SceneFolder;
    "scene": SceneRoot;
    "splat": SplatObject;
    "gltf": GLTFObject;
    "light": LightObject;
    "camera": CameraObject;
    "empty": EmptyObject;
    "video": VideoObject;
    "trigger": TriggerZone;
    "spawn": SpawnPoint;
  };

  // Operations
  createObject(type: string, parent?: string): string;
  deleteObject(objectId: string): void;
  duplicateObject(objectId: string): string;
  reparent(objectId: string, newParentId: string): void;
  renameObject(objectId: string, newName: string): void;
}
```

**Features:**
- Drag to reparent
- Multi-select with Shift/Ctrl
- Favorite/pin frequently used objects
- Show/hide in scene visibility
- Lock/unlock objects
- Layer assignment
- Search by name/type/tag
- Fold/unfold branches

### 3.3 Inspector Panel

```typescript
interface InspectorPanel {
  // Sections
  sections: InspectorSection[];

  // Context-sensitive
  getSections(selection: SceneObject[]): InspectorSection[];
}

type InspectorSection =
  | TransformSection
  | MaterialSection
  | LightingSection
  | PhysicsSection
  | AudioSection
  | AnimationSection
  | VideoSection
  | TriggerSection
  | ScriptSection
  | ComponentSection
  | PrefabSection;
```

#### Transform Section

```typescript
interface TransformSection {
  position: Vector3;
  rotation: { euler: Euler; quaternion: Quaternion; order: RotationOrder };
  scale: Vector3;

  // Options
  worldSpace: boolean;
  uniformScale: boolean;

  // Reset
  resetPosition(): void;
  resetRotation(): void;
  resetScale(): void;
}
```

#### Material Section

```typescript
interface MaterialSection {
  // Base
  baseColor: Color;
  metallic: number;
  roughness: number;
  opacity: number;
  transparent: boolean;
  doubleSided: boolean;

  // Advanced
  emissive: Color;
  emissiveIntensity: number;
  normalMap: Texture;
  roughnessMap: Texture;
  metalnessMap: Texture;
  heightMap: Texture;
  occlusionMap: Texture;

  // Spark-specific (for splats)
  envMap: EnvironmentMap;
  envMapIntensity: number;
  splatMaterial: SplatMaterialPreset;
}
```

#### Physics Section

```typescript
interface PhysicsSection {
  // Rigidbody
  type: "static" | "dynamic" | "kinematic";
  mass: number;
  linearDamping: number;
  angularDamping: number;
  gravityScale: number;

  // Colliders
  colliders: Collider[];
  colliderType: "box" | "sphere" | "capsule" | "mesh" | "heightfield";

  // Constraints
  constraints: Constraint[];
}

interface Collider {
  type: "box" | "sphere" | "capsule" | "mesh";
  center: Vector3;
  size: Vector3;
  isTrigger: boolean;
}
```

#### Animation Section

```typescript
interface AnimationSection {
  // Clips
  clips: AnimationClip[];
  defaultClip: string;

  // Playback
  loop: boolean;
  autoplay: boolean;
  playAutomatically: boolean;

  // Blending
  blendMode: "blend" | "additive" | "mix";
  blendWeight: number;

  // Root Motion
  applyRootMotion: boolean;
  rootMotionBone: string;

  // Timeline
  timelineVisible: boolean;
  currentTime: number;
}
```

#### Script Section

```typescript
interface ScriptSection {
  // Attached Scripts
  scripts: ScriptReference[];

  // Script Properties
  properties: ScriptProperty[];

  // Events
  onStart: ScriptEvent[];
  onUpdate: ScriptEvent[];
  onCollision: ScriptEvent[];
  onTrigger: ScriptEvent[];
}

interface ScriptReference {
  id: string;
  name: string;
  enabled: boolean;
  order: number;
}
```

#### Video Section (from your engine)

```typescript
interface VideoSection {
  // Video Asset
  videoFile: string;
  loop: boolean;
  autoplay: boolean;
  muted: boolean;

  // Display
  position: Vector3;
  rotation: Euler;
  scale: Vector3;
  width: number;
  height: number;

  // Effects
  bloom: number;
  chromatic: number;

  // State Triggers
  criteria: Criteria;
}
```

### 3.4 Visual Scripting Panel

```typescript
interface VisualScriptingPanel {
  // Graph Editor
  graph: ScriptGraph;

  // Node Types
  nodeCategories: {
    "events": EventNode[];
    "flow": FlowNode[];
    "logic": LogicNode[];
    "math": MathNode[];
    "comparison": ComparisonNode[];
    "state": StateNode[];
    "audio": AudioNode[];
    "visual": VisualNode[];
    "scene": SceneNode[];
    "player": PlayerNode[];
    "custom": CustomNode[];
  };

  // Graph Operations
  createNode(type: string, position: Vector2): ScriptNode;
  deleteNode(nodeId: string): void;
  connectNodes(sourceId: string, targetId: string, port: string): void;
  disconnectNodes(connectionId: string): void;
}
```

**Node Examples:**

```typescript
// Event Nodes
interface OnStateChangedNode {
  type: "onStateChanged";
  output: "trigger";
  state: GameState;
}

interface OnCollisionEnterNode {
  type: "onCollisionEnter";
  output: "trigger";
  collider: string;
}

interface OnKeyDownNode {
  type: "onKeyDown";
  output: "trigger";
  key: string;
}

// Flow Nodes
interface IfNode {
  type: "if";
  inputs: { condition: boolean; true: any; false: any; };
  outputs: { result: any };
}

interface LoopNode {
  type: "loop";
  inputs: { condition: boolean; body: any };
  outputs: { output: any };
}

interface SequenceNode {
  type: "sequence";
  inputs: { steps: any[] };
  outputs: { complete: boolean };
}

// State Nodes
interface SetStateNode {
  type: "setState";
  inputs: { state: GameState; value: any };
}

interface GetStateNode {
  type: "getState";
  inputs: { state: GameState };
  outputs: { value: any };
}

// Scene Nodes
interface PlayDialogNode {
  type: "playDialog";
  inputs: { dialogId: string };
  outputs: { onComplete: boolean };
}

interface PlayMusicNode {
  type: "playMusic";
  inputs: { musicId: string; volume: number; fadeDuration: number };
}

interface SetLightColorNode {
  type: "setLightColor";
  inputs: { lightId: string; color: Color };
}

// Player Nodes
interface MovePlayerNode {
  type: "movePlayer";
  inputs: { position: Vector3; lookAt: Vector3; duration: number };
  outputs: { onComplete: boolean };
}

interface EnableControlNode {
  type: "enableControl";
  inputs: { enabled: boolean };
}
```

### 3.5 Console Panel

```typescript
interface ConsolePanel {
  // Log Levels
  filters: ("log" | "info" | "warn" "error" | "debug")[];

  // Features
  clearOnPlay: boolean;
  collapseDuplicates: boolean;
  showTimestamp: boolean;
  stackTrace: boolean;

  // Actions
  clear(): void;
  copy(): void;
  export(): void;
}
```

### 3.6 Profiler Panel

```typescript
interface ProfilerPanel {
  // Metrics
  categories: ProfilerCategory[];

  // Recording
  isRecording: boolean;
  frameData: FrameSample[];

  // Views
  currentView: "timeline" | "hierarchy" | "memory";
}

type ProfilerCategory =
  | "rendering"
  | "scripts"
  | "physics"
  | "animation"
  | "audio"
  | "garbage collection"
  | "system";

interface FrameSample {
  frame: number;
  time: number;
  fps: number;
  memory: number;
  categories: Record<ProfilerCategory, number>;
}
```

---

## 4. Data Format Specifications

### 4.1 Scene File Format (.scene.json)

```json
{
  "version": "1.0",
  "name": "OfficeScene",
  "description": "The main office interior",
  "settings": {
    "gravity": { "x": 0, "y": -9.81, "z": 0 },
    "ambientLight": { "r": 0.1, "g": 0.1, "b": 0.15 },
    "environment": "/environments/office.hdr"
  },
  "hierarchy": [
    {
      "id": "office_root",
      "name": "Office",
      "type": "folder",
      "children": [
        {
          "id": "office_interior",
          "name": "OfficeInterior",
          "type": "splat",
          "asset": "/splats/office.sog",
          "transform": {
            "position": { "x": 5.36, "y": 0.83, "z": 78.39 },
            "rotation": { "x": -3.1416, "y": 1.0358, "z": -3.1416 },
            "scale": { "x": 1, "y": 1, "z": 1 }
          },
          "layers": ["interior"],
          "enabled": true
        },
        {
          "id": "phonebooth",
          "name": "PhoneBooth",
          "type": "gltf",
          "asset": "/models/phonebooth.glb",
          "transform": {
            "position": { "x": 0, "y": 0, "z": 5 },
            "rotation": { "x": 0, "y": 0, "z": 0 },
            "scale": { "x": 1, "y": 1, "z": 1 }
          },
          "components": [
            {
              "type": "PhysicsCollider",
              "shape": "box",
              "size": { "x": 1, "y": 2, "z": 1 },
              "isTrigger": false
            },
            {
              "type": "Interactable",
              "prompt": "Press E to use phone"
            }
          ],
          "children": [
            {
              "id": "phone_handset",
              "name": "Handset",
              "type": "gltf",
              "asset": "/models/phone_handset.glb"
            }
          ]
        },
        {
          "id": "office_light",
          "name": "OfficeLight",
          "type": "light",
          "lightType": "point",
          "color": { "r": 1, "g": 0.9, "b": 0.7, "a": 1 },
          "intensity": 1.5,
          "range": 15,
          "transform": {
            "position": { "x": 0, "y": 3, "z": 0 }
          },
          "audioReactive": {
            "enabled": true,
            "frequencyBands": [0, 3]
          }
        },
        {
          "id": "dialog_trigger",
          "name": "EnterOfficeDialog",
          "type": "trigger",
          "shape": "box",
          "size": { "x": 5, "y": 3, "z": 5 },
          "transform": {
            "position": { "x": 0, "y": 1, "z": 0 }
          },
          "components": [
            {
              "type": "TriggerZone",
              "zone": "office",
              "once": true,
              "onEnter": {
                "type": "visual",
                "nodes": [
                  {
                    "type": "setState",
                    "state": "OFFICE_INTERIOR"
                  },
                  {
                    "type": "playDialog",
                    "dialog": "dialog_office_welcome"
                  }
                ]
              }
            }
          ]
        },
        {
          "id": "ambient_audio",
          "name": "AmbientAudio",
          "type": "audio",
          "asset": "/audio/office_ambience.mp3",
          "loop": true,
          "volume": 0.3,
          "spatial": {
            "enabled": true,
            "position": { "x": 0, "y": 0, "z": 0 },
            "rolloff": 2,
            "maxDistance": 20
          },
          "criteria": {
            "currentState": { "$gte": "OFFICE_INTERIOR", "$lt": "LIGHTS_OUT" }
          }
        }
      ]
    }
  ]
}
```

### 4.2 Prefab File Format (.prefab.json)

```json
{
  "version": "1.0",
  "name": "PhoneBooth",
  "description": "Interactive phone booth prop",
  "icon": "/prefabs/phonebooth.png",
  "hierarchy": [
    {
      "id": "phonebooth",
      "name": "PhoneBooth",
      "type": "gltf",
      "asset": "/models/phonebooth.glb",
      "transform": {
        "position": { "x": 0, "y": 0, "z": 0 },
        "rotation": { "x": 0, "y": 0, "z": 0 },
        "scale": { "x": 1, "y": 1, "z": 1 }
      },
      "components": [
        {
          "type": "PhysicsCollider",
          "shape": "box",
          "size": { "x": 1, "y": 2.5, "z": 1 }
        },
        {
          "type": "Interactable",
          "prompt": "Press E to answer",
          "icon": "📞"
        },
        {
          "type": "StateMachine",
          "states": {
            "idle": {
              "animations": ["idle"],
              "transitions": [
                { "to": "ringing", "trigger": "onStateEnter", "condition": { "state": "PHONE_BOOTH_RINGING" } }
              ]
            },
            "ringing": {
              "animations": ["ringing"],
              "sound": "/audio/phone_ring.mp3",
              "transitions": [
                { "to": "idle", "trigger": "onAnswer" }
              ]
            }
          }
        }
      ],
      "children": [
        {
          "id": "handset",
          "name": "Handset",
          "type": "gltf",
          "asset": "/models/phone_handset.glb"
        },
        {
          "id": "cord",
          "name": "Cord",
          "type": "physics-chain",
          "asset": "/models/phone_cord.glb"
        }
      ]
    }
  ]
}
```

### 4.3 Material Definition (.material.json)

```json
{
  "version": "1.0",
  "name": "OfficeWallMaterial",
  "type": "standard",
  "properties": {
    "baseColor": { "r": 0.8, "g": 0.7, "b": 0.6, "a": 1.0 },
    "metallic": 0.0,
    "roughness": 0.8,
    "emissive": { "r": 0, "g": 0, "b": 0, "a": 1 },
    "normalMap": "/textures/office_wall_normal.png",
    "roughnessMap": "/textures/office_wall_roughness.png"
  },
  "splatMaterial": {
    "envMapIntensity": 1.0,
    "contactShadow": {
      "enabled": true,
      "size": { "x": 0.5, "y": 0.5 },
      "opacity": 0.4
    }
  }
}
```

### 4.4 Visual Script (.script.json)

```json
{
  "version": "1.0",
  "name": "OnPlayerEnterOffice",
  "description": "Triggered when player enters the office",
  "nodes": [
    {
      "id": "node_1",
      "type": "onTriggerEnter",
      "position": { "x": 100, "y": 100 },
      "data": { "trigger": "office_trigger", "object": "player" }
    },
    {
      "id": "node_2",
      "type": "setState",
      "position": { "x": 300, "y": 100 },
      "data": { "state": "OFFICE_INTERIOR" }
    },
    {
      "id": "node_3",
      "type": "playDialog",
      "position": { "x": 300, "y": 200 },
      "data": { "dialog": "dialog_office_welcome" }
    },
    {
      "id": "node_4",
      "type": "playMusic",
      "position": { "x": 300, "y": 300 },
      "data": { "music": "music_office_ambience", "volume": 0.6, "fade": 2.0 }
    },
    {
      "id": "node_5",
      "type": "enableControl",
      "position": { "x": 300, "y": 400 },
      "data": { "enabled": true }
    }
  ],
  "connections": [
    { "from": "node_1", "to": "node_2", "output": "onEnter" },
    { "from": "node_2", "to": "node_3", "output": "onComplete" },
    { "from": "node_3", "to": "node_4", "output": "onComplete" },
    { "from": "node_4", "to": "node_5", "output": "onComplete" }
  ]
}
```

### 4.5 Project File (.project.json)

```json
{
  "version": "1.0",
  "name": "ShadowCzarGame",
  "description": "A psychological horror game",
  "settings": {
    "rendering": {
      "antiAliasing": "FXAA",
      "shadows": true,
      "shadowQuality": "high",
      "bloom": true,
      "postProcessing": true
    },
    "physics": {
      "gravity": { "x": 0, "y": -9.81, "z": 0 },
      "subSteps": 8,
      "solverIterations": 10
    },
    "audio": {
      "masterVolume": 1.0,
      "musicVolume": 0.7,
      "sfxVolume": 0.8,
      "spatialAudio": true
    },
    "performance": {
      "targetFPS": 60,
      "vSync": true,
      "profile": "balanced"
    }
  },
  "buildSettings": {
    "target": "web",
    "compression": true,
    "minify": true,
    "sourceMaps": true,
    "bundleSize": "auto"
  },
  "scenes": ["scenes/intro.scene.json", "scenes/plaza.scene.json"],
  "startupScene": "scenes/intro.scene.json"
}
```

---

## 5. Asset Pipeline

### 5.1 Asset Import Dialog

```typescript
interface AssetImportDialog {
  // Supported Formats
  supportedFormats: {
    models: [".glb", ".gltf", ".fbx", ".obj", ".sog"];
    textures: [".png", ".jpg", ".webp", ".hdr", ".exr"];
    audio: [".mp3", ".wav", ".ogg", ".aac"];
    video: [".webm", ".mp4"];
    splats: [".sog"];
  };

  // Import Options
  importOptions: {
    models: {
      generateNormals: boolean;
      generateTangents: boolean;
      compressTextures: boolean;
      embedAnimations: boolean;
    };
    textures: {
      maxSize: number;
      generateMipmaps: boolean;
      compression: "none" | "bc7" | "etc2";
    };
    audio: {
      normalize: boolean;
      compression: "mp3" | "aac" | "opus";
      targetSampleRate: number;
    };
  };
}
```

### 5.2 Asset Processing Pipeline

```
Source Assets
     │
     ▼
[Import Dialog]
     │
     ├─► Models → [Validator] → [Converter] → [Optimizer] → [Asset Database]
     │                                         │
     │                                         ├─► Generate LODs
     │                                         ├─► Compress textures
     │                                         └─► Generate thumbnails
     │
     ├─► Splats → [Validator] → [Copier] → [Asset Database]
     │                                    │
     │                                    └─► Generate thumbnail
     │
     ├─► Audio → [Validator] → [Encoder] → [Asset Database]
     │                                  │
     │                                  ├─► Generate waveform
     │                                  └─► Normalize volume
     │
     ├─► Video → [Validator] → [Transcoder] → [Asset Database]
     │                                   │
     │                                   ├─► Generate poster
     │                                   └─► Optimize for web
     │
     └─► Images → [Validator] → [Processor] → [Asset Database]
                                     │
                                     ├─► Generate mipmaps
                                     ├─► Compress (WebP)
                                     └─► Resize variants
```

### 5.3 Asset Database

```typescript
interface AssetDatabase {
  // Storage
  assets: Map<string, Asset>;
  collections: Map<string, string[]>;

  // Queries
  findByType(type: AssetType): Asset[];
  findByTag(tag: string): Asset[];
  search(query: string): Asset[];
  getDependencies(assetId: string): Asset[];

  // Operations
  import(file: File, options?: ImportOptions): Promise<Asset>;
  delete(assetId: string): void;
  duplicate(assetId: string): Asset;
  move(assetId: string, newCollection: string): void;

  // Metadata
  addTag(assetId: string, tag: string): void;
  removeTag(assetId: string, tag: string): void;
  setFavorite(assetId: string, favorite: boolean): void;
}

interface Asset {
  id: string;
  name: string;
  type: AssetType;
  path: string;
  thumbnail: string;

  // Metadata
  tags: string[];
  favorite: boolean;
  collections: string[];

  // Dependencies
  dependencies: string[];

  // Properties (type-specific)
  properties: AssetProperties;

  // Import settings
  importSettings: any;
}

type AssetType =
  | "model"
  | "splat"
  | "texture"
  | "material"
  | "audio"
  | "video"
  | "animation"
  | "script"
  | "scene"
  | "prefab";
```

---

## 6. Visual Scripting System

### 6.1 Node Categories

#### Event Nodes
- `onStateChanged` - Triggered when game state changes
- `onCollisionEnter` - Triggered when entering collider
- `onCollisionExit` - Triggered when exiting collider
- `onTriggerEnter` - Triggered when entering trigger zone
- `onKeyDown` - Triggered when key pressed
- `onKeyUp` - Triggered when key released
- `onMouseDown` - Triggered when mouse clicked
- `onGamepadInput` - Triggered by gamepad buttons
- `onTouchStart` - Triggered on touch begin
- `onAnimationComplete` - Triggered when animation finishes
- `onAudioComplete` - Triggered when audio finishes
- `onVideoComplete` - Triggered when video finishes
- `onTimeline` - Triggered at specific timeline time
- `onCustomEvent` - Triggered by custom event

#### Flow Nodes
- `if` - Conditional branching
- `else` - Alternative branch
- `switch` - Multi-way branching
- `loop` - Repeated execution
- `for` - Counted iteration
- `forEach` - Iterate over array
- `while` - Conditional iteration
- `break` - Exit loop
- `continue` - Skip to next iteration
- `sequence` - Execute in order
- `parallel` - Execute simultaneously
- `race` - First to finish wins
- `branch` - Multi-path flow
- `merge` - Wait for all inputs

#### Logic Nodes
- `compare` - Compare two values (=, ≠, <, ≤, >, ≥)
- `and` - Logical AND
- `or` - Logical OR
- `not` - Logical NOT
- `boolean` - Boolean value
- `select` - Choose between values
- `gate` - Conditional pass-through
- `delay` - Wait for duration
- `cooldown` - Rate limit execution
- `debounce` - Wait for pause
- `throttle` - Rate limit
- `once` - Execute only once
- `toggle` - Flip flop state

#### Math Nodes
- `add`, `subtract`, `multiply`, `divide`
- `mod`, `pow`, `sqrt`
- `min`, `max`, `clamp`
- `abs`, `floor`, `ceil`, `round`
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- `atan2` - Angle from components
- `lerp` - Linear interpolation
- `smoothstep` - Smooth interpolation
- `distance` - Distance between points
- `angle` - Angle between directions
- `dot`, `cross` - Vector operations
- `random` - Random number in range

#### State Nodes
- `getState` - Get game state value
- `setState` - Set game state
- `hasState` - Check state flag
- `transitionState` - Animate state change

#### Scene Nodes
- `createObject` - Instantiate object in scene
- `destroyObject` - Remove from scene
- `findObject` - Find object by name/tag
- `getParent` - Get object parent
- `setParent` - Reparent object
- `getChild` - Get child object
- `getChildren` - Get all children
- `setPosition` - Set object position
- `getPosition` - Get object position
- `setRotation` - Set object rotation
- `getRotation` - Get object rotation
- `setScale` - Set object scale
- `setVisible` - Show/hide object
- `setActive` - Enable/disable object

#### Audio Nodes
- `playMusic` - Play background music
- `stopMusic` - Stop music
- `playSFX` - Play sound effect
- `stopSFX` - Stop sound effect
- `setMusicVolume` - Set music volume
- `setMasterVolume` - Set master volume
- `fadeMusic` - Fade music over time
- `playDialog` - Play dialog track
- `stopDialog` - Stop dialog

#### Visual Nodes
- `setBloom` - Set bloom intensity
- `setVignette` - Set vignette intensity
- `setGlitch` - Set glitch effect
- `setDesaturate` - Set desaturation
- `triggerDissolve` - Start dissolve effect
- `setPostProcessing` - Enable/disable effect

#### Player Nodes
- `movePlayer` - Move player to position
- `getPlayerPosition` - Get player position
- `enableControl` - Enable player control
- `disableControl` - Disable player control
- `setViewmasterEquipped` - Equip/unequip viewmaster
- `getIsViewmasterEquipped` - Check if equipped
- `addDrawingSuccess` - Increment drawing counter
- `getDrawingSuccessCount` - Get drawing count
- `setRuneSighting` - Set rune sightings

#### Video Nodes
- `playVideo` - Play video
- `stopVideo` - Stop video
- `pauseVideo` - Pause video
- `setVideoPosition` - Set video transform
- `getVideoPlaying` - Check if video playing

#### Variable Nodes
- `setVariable` - Set variable value
- `getVariable` - Get variable value
- `incrementVariable` - Add to variable
- `compareVariable` - Compare variable

### 6.2 Visual Script Editor UI

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  VISUAL SCRIPT: OnPlayerEnterOffice                                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  [🟢 onTriggerEnter]    [📦 setState]    [💬 playDialog]    [🎵 playMusic]       │
│        office_trigger        ───────► OFFICE   ───────► welcome    ───────► ambience    │
│            player                 _INTERIOR         _               _            │
│           │││              ───►                │││             │││          │
│           │││              ───►                │││             │││          │
│           └─┴─┘              ──►                └─┴─┘           └─┴─┘          │
│                                                                                             │
│  Node Library:                                                                                │
│  ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐        │
│  │ Events     │ Flow       │ Logic      │ State      │ Audio      │ Visual     │        │
│  │ onState..  │ if         │ compare    │ getState  │ playMusic  │ setBloom   │        │
│  │ onCollis.. │ loop       │ and        │ setState  │ playDialog │ setVignette │        │
│  │ onKeyDown.. │ sequence   │ or         │            │ stopMusic  │            │        │
│  └────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘        │
│                                                                                             │
│  [Save] [Close]                                                                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Variable System

```typescript
interface VariableSystem {
  // Global Variables
  globals: Map<string, Variable>;

  // Local Variables (per script)
  locals: Map<string, Map<string, Variable>>;

  // Player Variables
  player: Map<string, Variable>;

  // Object Variables
  objects: Map<string, Map<string, Variable>>;
}

interface Variable {
  id: string;
  name: string;
  type: "boolean" | "number" | "string" | "vector3" | "color";
  value: any;
  persistent: boolean; // Save to disk
  synchronized: boolean; // Sync across network (multiplayer)
  readOnly: boolean;
}
```

---

## 7. Build & Deploy

### 7.1 Build Configuration

```typescript
interface BuildConfig {
  // Target Platform
  platform: "web" | "web-mobile" | "web-desktop";

  // Build Mode
  mode: "development" | "production" | "editor";

  // Optimizations
  optimizations: {
    minify: boolean;
    treeShaking: boolean;
    codeSplitting: boolean;
    compression: boolean;
    sourceMaps: boolean;
  };

  // Asset Processing
  assets: {
    compressTextures: boolean;
    compressAudio: boolean;
    generateLODs: boolean;
    bundleAssets: boolean;
  };

  // Output
  output: {
    directory: string;
    publicPath: string;
    filenameFormat: string;
  };
}
```

### 7.2 Build Process

```
[Build Start]
     │
     ▼
[Validate Project]
     │  ├─► Check all references
     │  ├─► Validate scenes
     │  ├─► Check scripts
     │  └─► Verify assets
     │
     ▼
[Compile Scripts]
     │  ├─► TypeScript → JavaScript
     │  ├─► Visual Scripts → JavaScript
     │  └─► Shaders → GLSL
     │
     ▼
[Process Assets]
     │  ├─► Models: Optimize, compress
     │  ├─► Textures: Compress, generate mipmaps
     │  ├─► Audio: Normalize, compress
     │  ├──► Video: Transcode to WebM
     │  └─► Splats: Copy as-is
     │
     ▼
[Bundle Code]
     │  ├─► Vendor chunk (Three.js, etc.)
     │  ├─► Engine chunk (core systems)
     │  ├─► Game data chunk (scenes, scripts)
     │  └─── Content chunks (lazy-loaded)
     │
     ▼
[Generate WebManifest]
     │  ├─► PWA manifest
     │  ├─── Service worker
     │  └─► Icons and splash screens
     │
     ▼
[Optimize]
     │  ├─► Tree shaking
     │  ├─► Code splitting
     │  ├─► Minification
     │  └──► Compression
     │
     ▼
[Output]
     ├─► dist/index.html
     ├─► dist/assets/*.js
     ├─► dist/assets/*.css
     ├─► dist/assets/*.json
     └─► dist/assets/*.wasm
```

### 7.3 One-Click Deploy

```typescript
interface DeployTarget {
  name: string;
  type: "vercel" | "netlify" | "firebase" | "github-pages" | "custom";

  // Configuration
  config: {
    domain?: string;
    buildCommand?: string;
    outputDirectory?: string;
    environmentVariables?: Record<string, string>;
    headers?: Record<string, string>;
    redirects?: Record<string, string>;
  };

  // Deploy
  deploy(): Promise<DeployResult>;
}

interface DeployResult {
  url: string;
  success: boolean;
  error?: string;
}
```

---

## 8. Collaboration Features

### 8.1 Version Control Integration

```typescript
interface VersionControlIntegration {
  // Git Operations
  status: GitStatus;
  branch: string;
  commits: GitCommit[];

  // Operations
  commit(message: string, files?: string[]): Promise<void>;
  pull(): Promise<void>;
  push(): Promise<void>;
  createBranch(name: string): Promise<void>;
  mergeBranch(source: string): Promise<void>;

  // Diff View
  showDiff(file: string): DiffResult;
  revertChanges(file: string): void;
}
```

### 8.2 Multiplayer Collaboration

```typescript
interface MultiplayerCollab {
  // Session
  sessionId: string;
  users: Collaborator[];
  isActive: boolean;

  // Operations
  joinSession(sessionId: string): void;
  leaveSession(): void;

  // Real-time
  onObjectChanged: (objectId: string, change: ObjectChange) => void;
  onSelectionChanged: (userId: string, selection: string[]) => void;
  onCursorMoved: (userId: string, position: Vector2) => void;
  onChatMessage: (userId: string, message: string) => void;

  // Presence
  showUserCursors: boolean;
  showUserNames: boolean;
}

interface Collaborator {
  id: string;
  name: string;
  color: string;
  cursor: { x: number; y: number };
  selection: string[];
}
```

---

## 9. Performance & Profiling

### 9.1 Profiler Panel

```typescript
interface ProfilerPanel {
  // Recording
  isRecording: boolean;
  frameData: FrameData[];

  // Metrics
  categories: ProfilerCategory[];

  // Views
  views: ProfilerView[];
}

interface FrameData {
  frameNumber: number;
  timestamp: number;
  fps: number;
  ms: number; // Frame time in milliseconds
  memory: MemorySample;

  // Breakdown by category
  rendering: number;  // GPU time
  scripts: number;     // Script execution
  physics: number;     // Physics simulation
  animation: number;   // Animation updates
  audio: number;       // Audio processing
  gc: number;          // Garbage collection
  other: number;       // Overhead

  // Detailed samples
  samples: ProfilerSample[];
}

interface ProfilerSample {
  name: string;
  category: string;
  depth: number;
  ms: number;        // Time spent
  selfMs: number;    // Self time (excluding children)
  calls: number;     // Number of calls
}
```

### 9.2 Memory Profiler

```typescript
interface MemoryProfiler {
  // Current state
  current: MemoryState;

  // Timeline
  history: MemorySample[];

  // Breakdown
  byType: Record<string, number>;
  byObject: Record<string, number>;

  // Operations
  takeSnapshot(): MemorySnapshot;
  compareSnapshots(before: string, after: string): MemoryDiff;

  // Leak detection
  detectPotentialLeaks(): PotentialLeak[];
}

interface MemoryState {
  used: number;        // Bytes in use
  total: number;       // Total heap size
  limit: number;       // Heap size limit

  // Breakdown
  textures: number;
  geometries: number;
  materials: number;
  renderTargets: number;
  audio: number;
  other: number;
}
```

### 9.3 Network Profiler

```typescript
interface NetworkProfiler {
  // Requests
  requests: NetworkRequest[];

  // Bandwidth
  currentBandwidth: number;
  totalLoaded: number;
  totalRequests: number;

  // Filtering
  filterByType: NetworkRequest[];
  filterByUrl: string;
}

interface NetworkRequest {
  url: string;
  type: "xhr" | "fetch" | "script" | "stylesheet" | "image" | "media" | "other";
  size: number;
  duration: number;
  status: number;
  cached: boolean;
  timing: RequestTiming;
}
```

---

## 10. Roadmap

### Phase 1: Foundation (Current)
- ✅ Basic scene editing
- ✅ Object manipulation (gizmos)
- ✅ Property inspector
- ✅ Asset browser (basic)
- ✅ Scene save/load
- ✅ Undo/redo
- ✅ Hierarchy panel

### Phase 2: Content Creation
- ⏳ Prefab system
- ⏳ Material editor
- ⏳ Animation timeline
- ⏳ Particle effects editor
- ⏳ Post-processing editor
- ⏳ Lighting editor

### Phase 3: Logic & Scripting
- ⏳ Visual scripting node editor
- ⏳ Variable system
- ⏳ Event triggers
- ⏳ Component system
- ⏳ Script editor integration

### Phase 4: Polish
- ⏳ Build system
- ⏳ Profiler tools
- ⏳ Memory profiler
- ⏳ Network profiler
- ⏳ Performance recommendations

### Phase 5: Collaboration
- ⏳ Version control (Git integration)
- ⏳ Multiplayer editing
- ⏳ Real-time collaboration
- ⏳ Comments and annotations
- ⏳ Change history

### Phase 6: Publishing
- ⏳ One-click deploy
- ⏳ CDN integration
- ⏳ Custom domain
- ⏳ Analytics integration
- ⏳ Monetization features

---

## Appendix A: Keyboard Shortcuts Reference

### Global Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl+E` | Toggle Editor |
| `Ctrl+S` | Save Scene |
| `Ctrl+Shift+S` | Save Project |
| `Ctrl+O` | Open Scene |
| `Ctrl+N` | New Scene |
| `Ctrl+D` | Duplicate |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Ctrl+C` | Copy |
| `Ctrl+V` | Paste |
| `Ctrl+X` | Cut |
| `Delete` | Delete |
| `F5` | Play |
| `F6` | Pause |
| `F7` | Stop |
| `Ctrl+B` | Build |
| `Ctrl+R` | Refresh |
| `Ctrl+Shift+F` | Search in scene |

### Tool Shortcuts
| Shortcut | Tool |
|----------|------|
| `Q` | Select |
| `G` | Move |
| `R` | Rotate |
| `S` | Scale |
| `W` | Translate (alt for Move) |
| `E` | Rotate (alt for Rotate) |
| `T` | Scale (alt for Scale) |
| `F` | Focus on Selection |
| `Del` | Delete |
| `Shift+D` | Duplicate |

### Viewport Shortcuts
| Shortcut | Action |
|----------|--------|
| `Right Mouse` | Orbit |
| `Middle Mouse` / `Alt+Left` | Pan |
| `Scroll` / `±` | Zoom |
| `F` | Frame Selection |
| `Z` | Zoom to Object |
| `/` | Toggle Grid |

---

## Appendix B: File Structure

```
shadowczarengine/
├── src/
│   ├── editor/                          # Editor application
│   │   ├── ShadowEditor.js            # Main editor
│   │   ├── panels/
│   │   │   ├── AssetBrowserPanel.js
│   │   │   ├── HierarchyPanel.js
│   │   │   ├── InspectorPanel.js
│   │   │   ├── VisualScriptPanel.js
│   │   │   ├── ConsolePanel.js
│   │   │   ├── ProfilerPanel.js
│   │   │   └── BuildPanel.js
│   │   ├── tools/
│   │   │   ├── SelectTool.js
│   │   │   ├── MoveTool.js
│   │   │   ├── RotateTool.js
│   │   │   ├── ScaleTool.js
│   │   │   └── GizmoTool.js
│   │   ├── commands/
│   │   │   ├── Command.js
│   │   │   ├── CreateObjectCommand.js
│   │   │   ├── DeleteObjectCommand.js
│   │   │   ├── MoveObjectCommand.js
│   │   │   └── SetPropertyCommand.js
│   │   ├── components/
│   │   │   ├── Component.js
│   │   │   ├── TransformComponent.js
│   │   │   ├── PhysicsComponent.js
│   │   │   ├── AudioComponent.js
│   │   │   └── AnimationComponent.js
│   │   ├── serialization/
│   │   │   ├── SceneSerializer.js
│   │   │   ├── ProjectSerializer.js
│   │   │   └── PrefabSerializer.js
│   │   ├── undo/
│   │   │   └── UndoManager.js
│   │   └── ui/
│   │       └── EditorUI.js
│   │
│   ├── core/                             # Engine core
│   │   ├── GameManager.js
│   │   ├── SceneManager.js
│   │   ├── ZoneManager.js
│   │   ├── LightManager.js
│   │   ├── InputManager.js
│   │   ├── PhysicsManager.js
│   │   ├── AnimationManager.js
│   │   ├── DialogManager.js
│   │   ├── MusicManager.js
│   │   ├── SFXManager.js
│   │   ├── VFXManager.js
│   │   └── UIManager.js
│   │
│   ├── data/                             # Game data
│   │   ├── scenes/                      # Scene files (.scene.json)
│   │   ├── prefabs/                     # Prefab files (.prefab.json)
│   │   ├── materials/                   # Material files (.material.json)
│   │   ├── scripts/                     # Visual scripts (.script.json)
│   │   ├── audio/                       # Audio files
│   │   ├── models/                      # 3D models
│   │   ├── splats/                      # Gaussian splats
│   │   └── textures/                    # Texture files
│   │
│   └── main.js                          # Entry point
│
├── public/                              # Static assets
├── index.html
├── vite.config.js
├── package.json
└── tsconfig.json
```

---

## Summary

The complete **Shadow Engine Editor** would include:

1. **Scene Management** - Create, load, save scenes with full object hierarchy
2. **Asset Pipeline** - Import, process, organize all game assets
3. **Object Manipulation** - Full transform editing with gizmos
4. **Component System** - Attach behaviors to objects visually
5. **Visual Scripting** - Node-based logic creation without coding
6. **Material Editor** - Create and edit materials visually
7. **Animation Tools** - Timeline-based animation editing
8. **Particle Systems** - Visual particle effect creation
9. **Post-Processing** - Configure bloom, glitch, and other effects
10. **Audio Tools** - Set up spatial audio and dialog
11. **Physics Setup** - Configure colliders and rigid bodies
12. **Video Integration** - Place and configure video planes
13. **Play/Edit Mode** - Test changes instantly without rebuilding
14. **Profiler** - Identify performance bottlenecks
15. **Build System** - One-click deployment
16. **Collaboration** - Work with others in real-time

This creates a **Unity-like experience entirely in the browser** for building games like Shadow Czar!
