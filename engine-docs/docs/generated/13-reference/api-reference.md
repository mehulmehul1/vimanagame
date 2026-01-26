# API Reference - Shadow Engine

## Overview

This API Reference provides a complete overview of all managers, their methods, and how to use them. Each manager in the Shadow Engine serves a specific purpose—from rendering graphics to playing audio—and understanding their APIs is essential for building games with this engine.

Think of this API Reference as the **"engineer's manual"**—like a car manual lists every button, switch, and dial, this reference documents every class, method, and property you can use when building with the Shadow Engine.

---

## 📑 Quick Reference

### Core Managers

| Manager | Purpose | Key Methods |
|---------|---------|-------------|
| **GameManager** | Central coordination | `init()`, `start()`, `emit()`, `on()` |
| **SceneManager** | 3D scene rendering | `loadScene()`, `addObject()`, `removeObject()` |
| **InputManager** | Player input | `on()`, `getAxis()`, `isPressed()` |
| **PhysicsManager** | Physics simulation | `createBody()`, `createCollider()`, `step()` |
| **AudioManager** | Sound & music | `playMusic()`, `playSFX()`, `stop()` |
| **DialogManager** | Conversations | `startDialog()`, `showChoice()` |
| **AnimationManager** | Animations | `play()`, `stop()`, `chain()` |
| **VFXManager** | Visual effects | `trigger()`, `createEffect()` |

---

## GameManager

The central coordinator for the entire game. Manages initialization, game loop, event system, and state tracking.

### Constructor

```javascript
const game = new GameManager(options);
```

**Options:**
- `container: HTMLElement` - DOM element to render into
- `debug: boolean` - Enable debug mode
- `logger: Object` - Custom logger (default: console)

### Methods

#### `async init()`

Initialize the game engine.

```javascript
await game.init();
```

#### `async start()`

Start the game loop.

```javascript
await game.start();
```

#### `pause()`

Pause the game.

```javascript
game.pause();
```

#### `resume()`

Resume from pause.

```javascript
game.resume();
```

#### `on(eventName, callback)`

Subscribe to an event.

```javascript
game.on('player:jump', () => {
  console.log('Player jumped!');
});
```

#### `emit(eventName, data)`

Emit an event to all subscribers.

```javascript
game.emit('game:over', { score: 1000 });
```

#### `getState(key)`

Get a value from the game state store.

```javascript
const playerHealth = game.getState('player.health');
```

#### `setState(key, value)`

Set a value in the game state store.

```javascript
game.setState('player.health', 100);
```

#### `getManager(name)`

Get a reference to a manager.

```javascript
const audio = game.getManager('audio');
const physics = game.getManager('physics');
```

### Events

| Event | When Emitted | Data |
|-------|--------------|------|
| `game:initialized` | After init() completes | `{}` |
| `game:started` | After start() completes | `{}` |
| `game:paused` | When game is paused | `{}` |
| `game:resumed` | When game is resumed | `{}` |
| `game:over` | When game ends | `{ score, time }` |

---

## SceneManager

Manages the 3D scene, including Gaussian splats, GLTF models, lighting, and rendering.

### Constructor

```javascript
const sceneManager = new SceneManager({
  gameManager,
  container
});
```

### Methods

#### `async loadScene(sceneConfig)`

Load a scene configuration.

```javascript
await sceneManager.loadScene({
  id: 'plaza',
  splats: ['/assets/splats/plaza.splat'],
  models: ['/assets/models/environment.glb'],
  lighting: { ambient: 0.4, directional: 0.8 }
});
```

#### `addObject(object, options)`

Add an object to the scene.

```javascript
const mesh = new THREE.Mesh(geometry, material);
sceneManager.addObject(mesh, {
  position: new THREE.Vector3(0, 0, 0),
  rotation: new THREE.Euler(0, Math.PI, 0),
  scale: new THREE.Vector3(1, 1, 1)
});
```

#### `removeObject(object)`

Remove an object from the scene.

```javascript
sceneManager.removeObject(mesh);
```

#### `getCamera()`

Get the active camera.

```javascript
const camera = sceneManager.getCamera();
```

#### `setCamera(camera)`

Set the active camera.

```javascript
sceneManager.setCamera(myCamera);
```

#### `getScene()`

Get the Three.js scene object.

```javascript
const scene = sceneManager.getScene();
```

#### `getRenderer()`

Get the Three.js renderer.

```javascript
const renderer = sceneManager.getRenderer();
```

---

## InputManager

Handles all player input: keyboard, mouse, gamepad, and touch.

### Constructor

```javascript
const input = new InputManager({
  gameManager
});
```

### Methods

#### `on(action, callback)`

Bind an action to a callback.

```javascript
input.on('jump', () => player.jump());
input.on('shoot', () => weapon.fire());
```

#### `off(action, callback)`

Unbind an action.

```javascript
input.off('jump', jumpHandler);
```

#### `getAxis(axis)`

Get analog axis value (gamepad stick, mouse delta).

```javascript
const horizontal = input.getAxis('moveX');  // -1 to 1
const vertical = input.getAxis('moveY');     // -1 to 1
```

#### `isPressed(action)`

Check if an action is currently pressed.

```javascript
if (input.isPressed('jump')) {
  player.jump();
}
```

#### `wasPressed(action)`

Check if an action was pressed this frame.

```javascript
if (input.wasPressed('interact')) {
  interactWithObject();
}
```

#### `wasReleased(action)`

Check if an action was released this frame.

```javascript
if (input.wasReleased('fire')) {
  weapon.stopFiring();
}
```

#### `getPointerPosition()`

Get current pointer (mouse/touch) position.

```javascript
const { x, y } = input.getPointerPosition();
```

#### `getPointerDelta()`

Get pointer movement since last frame.

```javascript
const { x, y } = input.getPointerDelta();
```

### Default Actions

| Action | Default Bindings |
|--------|------------------|
| `forward` | W, Arrow Up |
| `backward` | S, Arrow Down |
| `left` | A, Arrow Left |
| `right` | D, Arrow Right |
| `jump` | Space |
| `interact` | E |
| `run` | Shift |
| `pause` | Escape |

---

## PhysicsManager

Manages the physics simulation using Rapier.

### Constructor

```javascript
const physics = new PhysicsManager({
  gameManager,
  gravity: { x: 0, y: -9.81, z: 0 }
});
```

### Methods

#### `createBody(options)`

Create a physics body.

```javascript
const body = physics.createBody({
  type: 'dynamic',  // 'dynamic', 'static', 'kinematic'
  position: { x: 0, y: 0, z: 0 },
  rotation: { x: 0, y: 0, z: 0 },
  mass: 1,
  linearDamping: 0.1,
  angularDamping: 0.1
});
```

#### `createCollider(body, options)`

Create a collider for a body.

```javascript
physics.createCollider(body, {
  shape: 'box',  // 'box', 'sphere', 'capsule', 'trimesh'
  halfExtents: { x: 1, y: 1, z: 1 },
  position: { x: 0, y: 0, z: 0 },
  isSensor: false
});
```

#### `removeBody(body)`

Remove a body from simulation.

```javascript
physics.removeBody(body);
```

#### `step(deltaTime)`

Step the physics simulation.

```javascript
physics.step(deltaTime);
```

#### `raycast(origin, direction, maxDistance)`

Cast a ray and return hit info.

```javascript
const hit = physics.raycast(
  { x: 0, y: 2, z: 0 },
  { x: 0, y: -1, z: 0 },
  100
);

if (hit) {
  console.log('Hit:', hit.body, hit.position);
}
```

---

## AudioManager

Handles music, sound effects, and spatial audio.

### Constructor

```javascript
const audio = new AudioManager({
  gameManager
});
```

### Methods

#### `async playMusic(trackId, options)`

Play background music.

```javascript
await audio.playMusic('main_theme', {
  volume: 0.7,
  loop: true,
  fade: 1.0  // Fade in duration
});
```

#### `stopMusic(fadeDuration)`

Stop the current music.

```javascript
audio.stopMusic(2.0);  // 2 second fade out
```

#### `playSFX(soundId, options)`

Play a sound effect.

```javascript
audio.playSFX('footstep', {
  volume: 0.5,
  position: new THREE.Vector3(0, 0, 0)
});
```

#### `playSFXAt(soundId, position, options)`

Play spatial 3D sound.

```javascript
audio.playSFXAt('explosion', new THREE.Vector3(10, 0, 5), {
  volume: 1.0,
  maxDistance: 50,
  rolloff: 1
});
```

#### `setMasterVolume(volume)`

Set master volume (0-1).

```javascript
audio.setMasterVolume(0.8);
```

#### `setMusicVolume(volume)`

Set music volume.

```javascript
audio.setMusicVolume(0.6);
```

#### `setSFXVolume(volume)`

Set SFX volume.

```javascript
audio.setSFXVolume(0.8);
```

---

## DialogManager

Manages dialog display and player choices.

### Constructor

```javascript
const dialog = new DialogManager({
  gameManager,
  uiManager
});
```

### Methods

#### `startDialog(dialogId, startNode)`

Begin a dialog sequence.

```javascript
dialog.startDialog('intro_conversation', 'start');
```

#### `showDialog(config)`

Show a dialog choice popup.

```javascript
dialog.showDialog({
  title: 'Warning',
  message: 'Do you want to continue?',
  choices: [
    { label: 'Yes', value: true },
    { label: 'No', value: false }
  ],
  callback: (result) => {
    if (result) {
      // User chose Yes
    }
  }
});
```

#### `hideDialog()`

Hide the current dialog.

```javascript
dialog.hideDialog();
```

#### `setFlag(flag, value)`

Set a dialog flag (for branching).

```javascript
dialog.setFlag('met_character', true);
```

#### `hasFlag(flag)`

Check if a flag exists.

```javascript
if (dialog.hasFlag('completed_quest')) {
  // Show different dialog
}
```

---

## AnimationManager

Manages camera and object animations.

### Constructor

```javascript
const animation = new AnimationManager({
  gameManager,
  sceneManager
});
```

### Methods

#### `play(animationId, options)`

Play an animation.

```javascript
animation.play('camera_pan', {
  duration: 2.0,
  easing: 'easeInOut',
  onComplete: () => {
    console.log('Animation complete');
  }
});
```

#### `stop(animationId)`

Stop a playing animation.

```javascript
animation.stop('camera_pan');
```

#### `chain(animations)`

Play animations in sequence.

```javascript
animation.chain([
  { id: 'camera_intro', duration: 2 },
  { id: 'camera_main', duration: 5 },
  { id: 'camera_outro', duration: 2 }
]);
```

#### `pause(animationId)`

Pause an animation (can be resumed).

```javascript
animation.pause('camera_pan');
```

#### `resume(animationId)`

Resume a paused animation.

```javascript
animation.resume('camera_pan');
```

---

## VFXManager

Manages visual effects like dissolve, bloom, glitch.

### Constructor

```javascript
const vfx = new VFXManager({
  gameManager,
  sceneManager
});
```

### Methods

#### `trigger(effectId, options)`

Trigger a visual effect.

```javascript
vfx.trigger('dissolve', {
  target: mesh,
  duration: 2.0,
  color: 0x00ff88
});
```

#### `createEffect(effectId, options)`

Create a reusable effect instance.

```javascript
const dissolve = vfx.createEffect('dissolve', {
  target: mesh
});

// Later:
dissolve.start();
```

---

## CharacterController

Handles player movement and first-person controls.

### Constructor

```javascript
const player = new CharacterController({
  gameManager,
  camera,
  inputManager,
  physicsManager
});
```

### Methods

#### `update(deltaTime)`

Update player controller (called every frame).

```javascript
player.update(deltaTime);
```

#### `setPosition(position)`

Set player position.

```javascript
player.setPosition(new THREE.Vector3(0, 1, 0));
```

#### `getPosition()`

Get current player position.

```javascript
const pos = player.getPosition();
```

#### `setRotation(yaw)`

Set player rotation (yaw angle).

```javascript
player.setRotation(Math.PI);
```

#### `setSpeed(speed)`

Set movement speed.

```javascript
player.setSpeed(5.0);  // units per second
```

#### `jump(force)`

Make the player jump.

```javascript
player.jump(5.0);
```

#### `isGrounded()`

Check if player is on the ground.

```javascript
if (player.isGrounded()) {
  player.jump();
}
```

---

## Usage Examples

### Basic Game Setup

```javascript
// Initialize game
const game = new GameManager({
  container: document.body
});

await game.init();

// Get managers
const scene = game.getManager('scene');
const input = game.getManager('input');
const audio = game.getManager('audio');
const physics = game.getManager('physics');

// Setup input
input.on('jump', () => player.jump());

// Start game
await game.start();
```

### Loading a Scene

```javascript
const scene = game.getManager('scene');

await scene.loadScene({
  id: 'level1',
  splats: ['/assets/splats/level1.splat'],
  models: ['/assets/models/level1.glb'],
  ambientLight: 0x404040,
  directionalLight: {
    color: 0xffffff,
    intensity: 1.0,
    position: new THREE.Vector3(5, 10, 5)
  }
});
```

### Playing Audio

```javascript
const audio = game.getManager('audio');

// Background music
await audio.playMusic('ambient_music', {
  volume: 0.5,
  loop: true
});

// Sound effects
audio.playSFX('door_open', { volume: 0.8 });

// Spatial sound
audio.playSFXAt('explosion', new THREE.Vector3(10, 0, 5), {
  volume: 1.0,
  maxDistance: 50
});
```

### Event-Driven Architecture

```javascript
// Subscribe to events
game.on('player:died', () => {
  audio.playSFX('death_sound');
  vfx.trigger('dissolve', { target: playerMesh });
});

game.on('level:complete', (data) => {
  game.setState('score', data.score);
  audio.playMusic('victory_theme');
});

// Emit events
game.emit('player:died', { cause: 'fell' });
game.emit('level:complete', { score: 1000, time: 120 });
```

---

## Common Patterns

### Singleton Access

```javascript
// Most managers are singletons accessed through GameManager
const game = GameManager.getInstance();
const scene = game.getManager('scene');
const audio = game.getManager('audio');
```

### Async Initialization

```javascript
// Many operations are async (loading assets, etc.)
await game.init();
await scene.loadScene(config);
await audio.playMusic('theme');
```

### Event Listeners

```javascript
// Always clean up listeners
function onPlayerJump() {
  audio.playSFX('jump');
}

game.on('player:jump', onPlayerJump);

// Later:
game.off('player:jump', onPlayerJump);
```

---

## Type Definitions

### Vector3

```javascript
interface Vector3 {
  x: number;
  y: number;
  z: number;

  add(v: Vector3): Vector3;
  sub(v: Vector3): Vector3;
  multiplyScalar(s: number): Vector3;
  clone(): Vector3;
}
```

### Quaternion

```javascript
interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;

  setFromEuler(euler: Euler): Quaternion;
  clone(): Quaternion;
}
```

### Event Callback

```typescript
type EventCallback<T = any> = (data: T) => void;
```

---

## Related Systems

All managers are interconnected:

- [GameManager](../02-core-architecture/game-manager-deep-dive.md) - Central coordination
- [InputManager](../04-input-physics/input-manager.md) - Input handling
- [PhysicsManager](../04-input-physics/physics-manager.md) - Physics simulation

---

## Source File Reference

**Manager Locations:**
- `../src/managers/GameManager.js` - Core game manager
- `../src/managers/SceneManager.js` - Scene rendering
- `../src/managers/InputManager.js` - Input handling
- `../src/managers/PhysicsManager.js` - Physics
- `../src/managers/AudioManager.js` - Audio
- `../src/managers/DialogManager.js` - Dialog system
- `../src/managers/AnimationManager.js` - Animations
- `../src/managers/VFXManager.js` - Visual effects
- `../src/character/CharacterController.js` - Player controller

---

## References

- [Three.js API Reference](https://threejs.org/docs/) - 3D library
- [Rapier Documentation](https://rapier.rs/docs/) - Physics engine
- [Howler.js API](https://howlerjs.com/api) - Audio library
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) - Native audio

*Documentation last updated: January 12, 2026*
