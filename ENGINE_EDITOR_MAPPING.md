# Editor-to-Engine Data Mapping

This document shows exactly how the visual editor maps to your existing manager-based architecture.

## Overview

Your engine's power comes from **managers listening to state changes** and checking **criteria** to decide what to do. The editor visualizes this relationship and generates the data files that managers consume.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         THE EDITOR-ENGINE LOOP                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────┐      generates      ┌──────────────────────────┐     │
│   │   VISUAL    │ ──────────────────> │   DATA FILES            │     │
│   │   EDITOR    │                    │   (JSON/JS modules)      │     │
│   └─────────────┘                    └──────────────────────────┘     │
│                                                  │                     │
│                                                  │ managers read       │
│                                                  ▼                     │
│   ┌─────────────┐      listen to       ┌──────────────────────────┐     │
│   │   GAME      │ <─────────────────── │   MANAGERS              │     │
│   │   MANAGER   │    state changes    │   (consume data files)  │     │
│   └─────────────┘                    └──────────────────────────┘     │
│        │                                                                      │
│        │ emits                                                               │
│        ▼                                                                      │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    ALL OTHER MANAGERS                           │   │
│   │  • SceneManager     • DialogManager    • VideoManager           │   │
│   │  • AnimationManager • MusicManager     • SFXManager             │   │
│   │  • LightManager     • VFXManager      • ColliderManager         │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Complete Mapping Table

| Editor Panel | Data File Generated | Manager Consuming | Runtime Effect |
|-------------|-------------------|-------------------|----------------|
| **Scene Orchestrator** | `gameData.js` (GAME_STATES enum) | GameManager | State enum values |
| **Scene Orchestrator** | `sceneData.js` (criteria entries) | SceneManager | Load/show objects |
| **Scene Set Editor** | `sceneData.js` (splat/gltf objects) | SceneManager | Render 3D content |
| **Interactable Objects** | `sceneData.js` (interactive objects) | GameManager | Interaction triggers |
| **Dialog Editor** | `dialogData.js` | DialogManager | Play dialog + captions |
| **Dialog Choice Editor** | `dialogChoiceData.js` | DialogManager + GameManager | Show choices, handle selection |
| **Video Timeline** | `videoData.js` | VideoManager | Play WebM videos |
| **Music Timeline** | `musicData.js` | MusicManager | Crossfade background music |
| **SFX Timeline** | `sfxData.js` | SFXManager | Play spatial sound effects |
| **Camera Animation** | `animationCameraData.js` | AnimationManager | Animate camera position |
| **Object Animation** | `animationObjectData.js` | AnimationManager | Animate object transform |
| **Light Editor** | `lightData.js` | LightManager | Enable/disable lights |
| **Trigger Zone Editor** | `colliderData.js` | ColliderManager | Detect enter/exit |
| **VFX Timeline** | `vfxData.js` | VFXManager | Trigger post-processing |
| **Objective Editor** | `gameData.js` (objectives) | UIManager | Show goal UI |

## Detailed Examples

### 1. Creating A New Scene State

**Editor Action:**
```
Scene Orchestrator → Add New Scene → Name: "RADIO_ROOM"
```

**Generated in `gameData.js`:**
```javascript
export const GAME_STATES = {
  // ... existing states
  RADIO_ROOM: 45,  // Auto-assigned next ID
};
```

**Generated in `sceneData.js`:**
```javascript
export const sceneObjects = [
  // Radio room splats and models
  {
    id: "radio_room_splat",
    type: "splat",
    url: "assets/splats/radio_room.sog",
    criteria: { currentState: GAME_STATES.RADIO_ROOM }
  },
];
```

**Runtime Flow:**
```
GameManager.setState({ currentState: GAME_STATES.RADIO_ROOM })
    ↓ (emits state:changed event)
SceneManager hears event
    ↓ (checks criteria)
"Does currentState === RADIO_ROOM match my criteria?" → YES
    ↓
SceneManager loads radio_room.sog
```

---

### 2. Dialog Choreography

**Editor Action:**
```
Dialog Editor → Add Node → Speaker: "CAT" → Text: "That's just an old box."
```

**Generated in `dialogData.js`:**
```javascript
export const dialogTracks = {
  cat_about_radio: {
    id: "cat_about_radio",
    criteria: {
      currentState: GAME_STATES.RADIO_ROOM,
      flags: { radio_approached: true }
    },
    dialog: [
      {
        speaker: "CAT",
        text: "That's just an old box.",
        audio: "assets/audio/cat_radio_01.wav",
        duration: 3.2
      }
    ]
  }
};
```

**Runtime Flow:**
```
Player approaches radio
    ↓
ColliderManager detects zone enter
    ↓
GameManager sets flag: { radio_approached: true }
    ↓ (still in RADIO_ROOM state)
DialogManager checks criteria:
  "Is currentState === RADIO_ROOM AND flags.radio_approached === true?" → YES
    ↓
DialogManager plays cat_about_radio track
```

---

### 3. Camera Animation

**Editor Action:**
```
Timeline Editor → Camera Track → Add Keyframe
  Time: 0s → Position: {0, 1.6, 0}, LookAt: "radio_01"
  Time: 3s → Position: {1, 1.6, 2}, LookAt: "radio_01"
```

**Generated in `animationCameraData.js`:**
```javascript
export const cameraAnimations = {
  radio_closeup: {
    id: "radio_closeup",
    criteria: { currentState: GAME_STATES.RADIO_ROOM },
    playNext: GAME_STATES.POST_RADIO,
    keyframes: [
      { time: 0, position: { x: 0, y: 1.6, z: 0 }, lookAt: "radio_01" },
      { time: 3000, position: { x: 1, y: 1.6, z: 2 }, lookAt: "radio_01" }
    ]
  }
};
```

**Runtime Flow:**
```
State changes to RADIO_ROOM
    ↓
AnimationManager checks criteria
    ↓ (matches)
AnimationManager takes camera control from CharacterController
    ↓
AnimationManager interpolates keyframes over 3 seconds
    ↓
At 3s, calls gameManager.setState(GAME_STATES.POST_RADIO)
```

---

### 4. Dialog Choice with Branching

**Editor Action:**
```
Dialog Choice Editor → Add Choice
  Prompt: "What will you do with the radio?"
  Options:
    - "Use it" → Leads to: PHONE_BOOTH_RINGING
    - "Leave it" → Leads to: OUTRO_CAT
```

**Generated in `dialogChoiceData.js`:**
```javascript
export const dialogChoices = [
  {
    id: "radio_choice",
    criteria: {
      currentState: GAME_STATES.RADIO_ROOM,
      flags: { cat_explained_radio: true }
    },
    prompt: "What will you do with the radio?",
    choices: [
      {
        text: "Use it",
        resultState: GAME_STATES.PHONE_BOOTH_RINGING,
        setFlags: { chose_radio: true }
      },
      {
        text: "Leave it",
        resultState: GAME_STATES.OUTRO_CAT,
        setFlags: { ignored_radio: true }
      }
    ]
  }
];
```

**Runtime Flow:**
```
DialogManager finishes cat explanation
    ↓
DialogChoiceUI checks dialogChoiceData for matching criteria
    ↓ (matches radio_choice)
DialogChoiceUI shows prompt with two buttons
    ↓
Player clicks "Use it"
    ↓
GameManager.setState({
  currentState: GAME_STATES.PHONE_BOOTH_RINGING,
  flags: { chose_radio: true, cat_explained_radio: true }
})
    ↓ (state change triggers all managers)
SceneManager loads phone booth scene
DialogManager plays ringing audio
AnimationManager starts camera drift
```

---

### 5. Trigger Zone with Proximity Check

**Editor Action:**
```
Trigger Zone Editor → Create Sphere Zone
  Position: {2, 0, 1}, Radius: 2.5
  Trigger: On Enter
  Actions: Show Dialog, Set Flag
```

**Generated in `colliderData.js`:**
```javascript
export const colliders = [
  {
    id: "radio_trigger_zone",
    type: "sphere",
    position: { x: 2, y: 0, z: 1 },
    radius: 2.5,
    trigger: "enter",
    once: false,
    onEnter: {
      showDialog: "cat_approach_radio",
      setFlags: { near_radio: true }
    },
    onExit: {
      setFlags: { near_radio: false }
    },
    criteria: { currentState: GAME_STATES.RADIO_ROOM }
  }
];
```

**Runtime Flow:**
```
Each frame, PhysicsManager updates character position
    ↓
ColliderManager checks all colliders with matching criteria
    ↓
"Is player inside radio_trigger_zone AND currentState === RADIO_ROOM?"
    ↓ (YES - entered this frame)
ColliderManager fires onEnter callbacks
    ↓
GameManager sets flag: near_radio = true
DialogManager checks criteria for cat_approach_radio
    ↓ (criteria include { flags: { near_radio: true } })
DialogManager plays the dialog
```

---

### 6. Multi-Manager Choreographed Sequence

**Editor Action:**
```
Timeline Editor → Create Sequence "radio_power_on"
  0.0s: Camera cut to closeup
  0.5s: VFX bloom increase
  1.0s: Audio reactive light pulse
  1.5s: SFX static sound
  2.0s: Dialog line plays
```

**Generated Data Files:**

**`animationCameraData.js`:**
```javascript
{
  id: "radio_power_on_cam",
  criteria: { customTrigger: "radio_power_on" },
  keyframes: [
    { time: 0, shot: "closeup", target: "radio_01" }
  ]
}
```

**`vfxData.js`:**
```javascript
{
  id: "radio_bloom",
  type: "selectiveBloom",
  criteria: { customTrigger: "radio_power_on" },
  delay: 500,
  strength: 2.0
}
```

**`lightData.js`:**
```javascript
{
  id: "radio_light",
  type: "point",
  position: { x: 2, y: 1, z: 1 },
  audioReactive: true,
  criteria: { customTrigger: "radio_power_on" },
  delay: 1000
}
```

**`sfxData.js`:**
```javascript
{
  id: "radio_static",
  url: "assets/audio/radio_static.wav",
  spatial: true,
  position: { x: 2, y: 1, z: 1 },
  criteria: { customTrigger: "radio_power_on" },
  delay: 1500
}
```

**`dialogData.js`:**
```javascript
{
  id: "radio_power_on_dialog",
  criteria: { customTrigger: "radio_power_on" },
  delay: 2000,
  dialog: [{ speaker: "RADIO", text: "*static*", audio: "..." }]
}
```

**Runtime Flow:**
```
GameManager.emit("radio_power_on")
    ↓
All managers receive event
    ↓
AnimationManager: "Do I have animations for radio_power_on?" → YES (delay 0ms)
    ↓ (waits)
    ↓ (at 0ms) Camera cuts to closeup
VFXManager: "Do I have effects for radio_power_on?" → YES (delay 500ms)
    ↓ (waits 500ms)
    ↓ (at 500ms) Bloom increases
LightManager: "Do I have lights for radio_power_on?" → YES (delay 1000ms)
    ↓ (waits 500ms more)
    ↓ (at 1000ms) Light pulses to audio
SFXManager: "Do I have sounds for radio_power_on?" → YES (delay 1500ms)
    ↓ (waits 500ms more)
    ↓ (at 1500ms) Static plays
DialogManager: "Do I have dialog for radio_power_on?" → YES (delay 2000ms)
    ↓ (waits 500ms more)
    ↓ (at 2000ms) "*static*" displays
```

---

## Criteria System in Editor

The visual editor's condition builder generates the exact criteria format your `criteriaHelper.js` expects:

**Visual Editor:**
```
┌─────────────────────────────────┐
│  State is RADIO_ROOM            │  ← Simple equality
│  AND flag met_cat is true       │  ← Flag check
│  OR player is near radio        │  ← Proximity
└─────────────────────────────────┘
```

**Generated Criteria:**
```javascript
{
  $or: [
    {
      currentState: GAME_STATES.RADIO_ROOM,
      flags: { met_cat: true }
    },
    {
      player_position: {
        $near: { x: 2, y: 0, z: 1, radius: 2.5 }
      }
    }
  ]
}
```

**How it works:**
1. Editor visualizes criteria as blocks
2. Each block maps to a criteriaHelper operator
3. Generated JSON matches exact format
4. All managers use `criteriaHelper.match(criteria, gameState)` to check

---

## The Editor's "Save" Action

When user clicks Save in editor:

```
1. Gather all scene data (objects, transforms, interactables)
2. Gather all choreography (timelines, keyframes)
3. Gather all dialog (nodes, choices, branches)
4. Gather all triggers (zones, conditions)
5. Generate/update data files:
   ├─ gameData.js (GAME_STATES enum, initial state)
   ├─ sceneData.js (splat/gltf objects with criteria)
   ├─ dialogData.js (dialog tracks with criteria)
   ├─ dialogChoiceData.js (choice prompts)
   ├─ videoData.js (video definitions)
   ├─ musicData.js (music tracks)
   ├─ sfxData.js (sound effects)
   ├─ animationCameraData.js (camera animations)
   ├─ animationObjectData.js (object animations)
   ├─ lightData.js (light definitions)
   ├─ colliderData.js (trigger colliders)
   └─ vfxData.js (VFX definitions)
6. Vite HMR refreshes the modules
7. Game managers immediately see new data
8. Test in editor without full rebuild
```

---

## File: Data Structure Reference

### sceneData.js Entry
```javascript
{
  id: "object_id",
  type: "splat" | "gltf" | "collider",
  url: "assets/path/to/file.sog",
  position: { x: 0, y: 0, z: 0 },
  rotation: { x: 0, y: 0, z: 0 },
  scale: { x: 1, y: 1, z: 1 },
  visible: true,
  criteria: { currentState: GAME_STATES.SOME_STATE },
  // For interactables:
  interactable: {
    type: "inspect" | "collect" | "trigger",
    distance: 2.0,
    once: true,
    onInteract: { showDialog: "some_dialog" }
  }
}
```

### dialogData.js Entry
```javascript
{
  id: "dialog_id",
  criteria: { currentState: GAME_STATES.SOME_STATE },
  dialog: [
    {
      speaker: "CHARACTER_NAME",
      text: "Dialog line here",
      audio: "assets/audio/line.wav",
      duration: 3.5,
      caption: true
    }
  ],
  playNext: GAME_STATES.NEXT_STATE,  // Optional auto-advance
  delay: 0  // Delay before starting (ms)
}
```

### animationCameraData.js Entry
```javascript
{
  id: "animation_id",
  criteria: { currentState: GAME_STATES.SOME_STATE },
  keyframes: [
    {
      time: 0,
      position: { x: 0, y: 1.6, z: 0 },
      rotation: { x: 0, y: 0, z: 0 },
      fov: 75,
      ease: "easeInOut"
    },
    // ... more keyframes
  ],
  duration: 3000,
  playNext: GAME_STATES.NEXT_STATE
}
```

---

## Summary

The visual editor doesn't replace your architecture—it **makes it visible**:

| What You Have | What Editor Provides |
|--------------|---------------------|
| Manager-based system | Visual node graph of manager responses |
| Criteria matching | Drag-and-drop condition builder |
| Data files | Live-editable JSON with preview |
| State-driven | Visual state machine diagram |
| Timeline data | Keyframe editor with scrubbing |
| Console debugging | In-editor play mode with overlays |

Your existing managers don't need to change. They continue to:
1. Listen for `state:changed` events
2. Check their data files for matching criteria
3. Execute when criteria match

The editor just makes creating those data files visual and iterative.
