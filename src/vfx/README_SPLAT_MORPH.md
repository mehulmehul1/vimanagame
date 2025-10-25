# Splat Morph Effect

## Overview

The Splat Morph effect creates a dramatic transition between two Gaussian splat scenes. During the transition, the source splat scatters into a cloud of particles, then reforms into the target splat.

## Implementation

This effect uses SparkJS's `dyno` shader system to manipulate individual splat particles in real-time.

## Files Created/Modified

### New Files

- **`src/vfx/splatMorph.js`** - Main effect class that extends VFXManager
  - Gets references to both splats from SceneManager
  - Applies dyno shader modifiers to both splats
  - Manages the animation timeline

### Modified Files

- **`src/vfxData.js`** - Added `splatMorphEffects` configuration

  - `hellTransition` effect triggered at `GAME_STATES.VIEWMASTER_HELL`
  - Configurable parameters: speed, duration, scatter radius

- **`src/vfxManager.js`** - Integrated into VFXSystemManager

  - Import and initialize SplatMorphEffect
  - Connect to game manager for state-driven behavior
  - Update in animation loop

- **`src/main.js`** - Updated initialization

  - Pass `sceneManager` to `vfxManager.initialize()`

- **`src/sceneData.js`** - Added office-hell.sog splat object
  - Configured to load at `GAME_STATES.VIEWMASTER_HELL`
  - Positioned to overlap with interior splat for seamless morph

## How It Works

### State-Based Triggering

1. **Interior splat** (`interior-nan-2.sog`) is loaded by SceneManager at `POST_DRIVE_BY` state
2. **Office hell splat** (`office-hell.sog`) is loaded by SceneManager at `VIEWMASTER_HELL` state
3. **Morph effect** initializes when first entering `VIEWMASTER_HELL` state
4. On initialization:
   - Gets reference to existing interior splat from SceneManager
   - Waits for office-hell splat to finish loading from SceneManager
   - Applies dyno modifiers to both splats to control their visibility and animation

### Animation Sequence

The morph happens in two phases:

1. **Scatter Phase (first 50% of transition)**

   - Source splat particles move towards random positions in a disk pattern
   - Particles shrink to 20% of original size
   - Alpha fades to 50%

2. **Reform Phase (second 50% of transition)**
   - Target splat particles emerge from scatter positions
   - Particles grow back to full size
   - Alpha fades in from 50% to 100%

### Parameters (configurable in vfxData.js)

```javascript
{
  speedMultiplier: 1.0,                        // Animation speed multiplier
  staySeconds: 1.5,                           // Time to stay on current splat before transitioning
  transitionSeconds: 2.0,                     // Duration of the morph transition
  randomRadius: 8.0,                          // Radius of scatter cloud (in world units)
  scatterCenter: { x: -5.14, y: 3.05, z: 84.66 }, // Center point of scatter cloud
  trigger: "start",                           // "start", "pause", or "reset"
}
```

## Usage in Game

### Current Configuration

When the player reaches `VIEWMASTER_HELL` state:

- The interior office scene morphs into the "hell" office scene
- Transition takes 2 seconds (configurable)
- Effect stays on target splat after transition completes

### Customizing the Effect

To adjust timing or behavior, edit `src/vfxData.js`:

```javascript
export const splatMorphEffects = {
  hellTransition: {
    id: "hellTransition",
    parameters: {
      speedMultiplier: 1.0, // Make faster/slower
      staySeconds: 1.5, // Time before transition starts
      transitionSeconds: 2.0, // Transition duration
      randomRadius: 8.0, // Scatter cloud size (larger = more spread out)
      scatterCenter: { x: -5.14, y: 3.05, z: 84.66 }, // Where particles scatter to
      trigger: "start",
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.VIEWMASTER_HELL, // When to trigger
      },
    },
    priority: 10,
  },
};
```

## Technical Details

### Shader Implementation

The effect uses SparkJS's `dyno` system to create a custom splat modifier:

```javascript
// Each splat particle's position and alpha are calculated based on:
// - Current animation time
// - Which mesh it belongs to (source or target)
// - Random scatter position (deterministic based on particle index)
// - Easing function for smooth transitions
```

### Performance

- Modifies splat data on GPU (very efficient)
- No CPU overhead during animation
- Both splats are rendered simultaneously with alpha blending

### Memory Management

- Both splats are managed by SceneManager based on game state criteria
- Interior splat stays loaded during VIEWMASTER_HELL (needed for morph transition)
- Office hell splat loads at VIEWMASTER_HELL and stays loaded afterward
- Morph effect only manages the dyno modifiers, not the splat lifecycle
- When disposed, effect removes modifiers but leaves splats in scene

## Extending This Effect

To create additional morph effects:

1. Add both splat objects to `sceneData.js`:

```javascript
sourceSplat: {
  id: "sourceSplat",
  type: "splat",
  path: "/source.sog",
  // ... position, rotation, scale, criteria
},
targetSplat: {
  id: "targetSplat",
  type: "splat",
  path: "/target.sog",
  // ... position, rotation, scale, criteria
},
```

2. Add a new effect entry in `vfxData.js`:

```javascript
anotherTransition: {
  id: "anotherTransition",
  parameters: { /* ... */ },
  criteria: { currentState: GAME_STATES.SOME_STATE },
  priority: 20,
}
```

3. To morph between different splats, modify `splatMorph.js`:
   - Change the source/target splat IDs in `onFirstEnable()` and `_waitForTargetSplat()`
   - Ensure both splats have overlapping positions for seamless morph

## Troubleshooting

### Effect doesn't trigger

- Check that `VIEWMASTER_HELL` state is being reached
- Verify interior splat is loaded before effect triggers
- Check browser console for error messages

### Visual artifacts

- Adjust `randomRadius` parameter (larger = more dispersed cloud)
- Modify `transitionSeconds` for smoother/faster transition
- Check that both splats have similar position/rotation/scale

### Performance issues

- Reduce `speedMultiplier` (paradoxically, faster animations can be cheaper)
- Ensure only 2 splats are involved (remove old splats from scene)

## Credits

Based on SparkJS morph effect example from SparkJS.dev
