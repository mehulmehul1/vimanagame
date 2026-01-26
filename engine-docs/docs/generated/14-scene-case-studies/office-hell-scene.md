# Scene Case Study: Office Hell

## 🎬 Scene Overview

**Location**: Transformed version of the Office Interior
**Narrative Context**: The nightmare version—a reality breakdown where the familiar office becomes surreal, threatening, and deeply unsettling
**Player Experience**: Shock → Disorientation → Fear → Desperation

The Office Hell scene is one of the most impactful moments in the game—the transformation of a "safe" space into a nightmare realm. This scene demonstrates how to use environmental transformation, visual distortion, and psychological horror to create a profound shift in player experience. The familiar office becomes unrecognizable, teaching players that no place is truly safe.

---

## 🎮 Game Design Perspective

### Creative Intent

**Emotional Goal**: Destroy the player's sense of safety—make the familiar threatening and the reliable unstable.

**Why This Transformation Matters**:

```
THE HORROR OF FAMILIARITY CORRUPTED:

Safe Office → Office Hell
    ↓
Player thought: "This is safe"
Player now thinks: "Nowhere is safe"

Familiar Objects → Distorted Versions
    ↓
Desk → Writhing mass
Chair → Broken, wrong proportions
Computer → Glitching, impossible

PSYCHOLOGICAL IMPACT:
Things that should be safe aren't.
Familiarity becomes weapon against player.
The foundation of reality feels unstable.
```

### Design Philosophy

**1. The Transformation Moment**

```javascript
// The precise moment of transformation:

TRANSFORM_SEQUENCE:

1. SUBTLE WRONGNESS (Pre-transform)
   ├─ Lights flicker once (unusual)
   ├─ Sound distorts briefly (audio glitch)
   ├─ Color shift subtle (saturations desaturate)
   └─ Player feels: "Did something just happen?"

2. BUILD-UP (Escalating)
   ├─ Flickering increases
   ├─ Objects begin to shift position
   ├─ Audio becomes distorted
   ├─ Shadows lengthen unnaturally
   └─ Player feels: "Something is very wrong"

3. SNAP (The transformation)
   ├─ Reality breaks
   ├─ Splat morphs to nightmare version
   ├─ Lighting shifts to hellish colors
   ├─ Audio becomes chaotic
   ├─ Player movement affected
   └─ Player feels: "I need to get out of here"

4. NEW REALITY (Post-transform)
   ├─ Office is unrecognizable
   ├─ Physics may be altered
   ├─ Navigation becomes puzzle
   └─ Player feels: "Where am I? How do I escape?"
```

**2. Visual Horror Techniques**

```
DISTORTION METHODS:

Geometry Distortion:
├── Stretch proportions (too tall/thin)
├── Bend straight lines (curves where there were none)
├── Melt objects (drip, sag)
└── Break physics (float, collapse)

Color Distortion:
├── Shift palette (wrong colors for familiar objects)
├── High contrast (harsh, violent)
├── Unnatural glows (things that shouldn't glow)
└── Desaturate then tint (remove warmth, add cold/blood)

Motion Distortion:
├── Things move when they shouldn't
├── Jitter, shake (unstable reality)
├── Flow like liquid (solid becomes fluid)
└── Pulse/breathe (objects feel alive in wrong way)

COMBINATION:
The more distortions layered,
the more profound the horror.
```

**3. Audio Horror**

```
SOUND DESIGN FOR NIGHTMARE:

Pre-Transform:
├── Normal office ambience
├── Sudden interruption
├── Brief distortion
└── Return to "normal" (but is it?)

Transform Moment:
├── Reality-tearing sound
├── All sounds muffled then amplified
├── Tinnitus-like ringing
└── Dropped into new ambience

Post-Transform:
├── Wrong version of office sounds
├── Distorted HVAC (groaning, not humming)
├── Impossible acoustics (echo where there shouldn't be)
├── Whispers, voices (not real?)
└── Musical elements (droning, dissonant)

PRINCIPLE:
Sound should feel like the environment
is alive and hostile, not just "scary music"
```

---

## 🎨 Level Design Breakdown

### Transformation Progression

```
                    OFFICE HELL TRANSFORMATION:

BEFORE (Normal Office):
┌─────────────────────────────────────────────────────────┐
│ Layout: Familiar, logical                               │
│ Lighting: Warm fluorescent                              │
│ Colors: Normal office colors                            │
│ Objects: In correct positions                           │
│ Physics: Normal                                         │
│ Audio: HVAC hum, quiet                                  │
│ Player Feeling: Safe, comfortable                       │
└─────────────────────────────────────────────────────────┘
                          ↓
                    [TRIGGER EVENT]
                    (Viewmaster use
                     or time-based
                     or state change)
                          ↓
DURING TRANSITION (5-10 seconds):
┌─────────────────────────────────────────────────────────┐
│ 0-2s: Subtle glitches begin                              │
│   - Single light flicker                                 │
│   - Audio briefly distorts                               │
│   - Color shifts slightly                               │
│                                                          │
│ 2-5s: Escalation                                        │
│   - More flickering, spreading                          │
│   - Objects begin to move/displace                      │
│   - Audio becomes chaotic                               │
│   - Splat begins to morph                               │
│                                                          │
│ 5-8s: Reality breaks                                    │
│   - SNAP moment                                         │
│   - Full splat swap                                     │
│   - Lighting changes completely                         │
│   - Audio drops into hell ambience                      │
│                                                          │
│ 8-10s: New reality stabilizes                           │
│   - Office Hell fully loaded                            │
│   - Player controls affected                            │
│   - Navigation changes                                  │
└─────────────────────────────────────────────────────────┘
                          ↓
AFTER (Office Hell):
┌─────────────────────────────────────────────────────────┐
│ Layout: Distorted, nonsensical                          │
│ Lighting: Red/blood, harsh shadows                      │
│ Colors: Wrong, oversaturated or desaturated             │
│ Objects: Melting, floating, wrong proportions           │
│ Physics: Altered (some float, some are heavy)           │
│ Audio: Groaning, whispers, droning                      │
│ Player Feeling: Terrified, desperate, disoriented       │
└─────────────────────────────────────────────────────────┘
```

### Spatial Layout - Office Hell

```
                    OFFICE HELL LAYOUT:

    ╔════════════════════════════════════════════════════╗
   ║                     [DISTORTED ENTRANCE]            ║
   ║          Door drips, frame bent, wrong scale         ║
   ║       ↓                                             ║
   ║  ╔═══════════════════════════════════════════════╗  ║
   ║  ║         HELLISH OFFICE SPACE                   ║  ║
   ║  ║                                                ║  ║
   ║  ║  [Window Wall] ← Now shows impossible view     ║  ║
   ║  ║    - Blood red sky outside                      ║  ║
   ║  ║    - Or completely different location          ║  ║
   ║  ║    - Or swirling void                          ║  ║
   ║  ║                                                ║  ║
   ║  ║     ╔═══════════════╗  (DISTORTED)              ║  ║
   ║  ║     ║   DESK        ║ ← Has melted, spread      ║  ║
   ║  ║     ║  (WRITHING)   ║   - Wood flows like wax  ║  ║
   ║  ║     ║               ║   - Objects float above  ║  ║
   ║  ║     ║  [MONITOR]    ║   - Monitor shows glitch  ║  ║
   ║  ║     ║  (FACES?)     ║   - Screen may have face  ║  ║
   ║  ║     ╚═══════════════╝                          ║  ║
   ║  ║                                                ║  ║
   ║  ║  [Walls] ← Breathing, pulsing                   ║  ║
   ║  ║    - Texture moves (not static)                ║  ║
   ║  ║    - Bleeding/oozing substances                 ║  ║
   ║  ║    - Wrong geometry (non-Euclidean hints)      ║  ║
   ║  ║                                                ║  ║
   ║  ║  [Ceiling] ← Lowering? Or infinite height?     ║  ║
   ║  ║    - Lights flicker chaotically                 ║  ║
   ║  ║    - Some float down slowly                    ║  ║
   ║  ║    - Shadows move independently                ║  ║
   ║  ║                                                ║  ║
   ║  ║  [Floor] ← Not flat anymore                     ║  ║
   ║  ║    - Tilts, slopes (movement affected)          ║  ║
   ║  ║    - Some areas have no friction                ║  ║
   ║  ║    - Others are "sticky" (movement slowed)      ║  ║
   ║  ║                                                ║  ║
   ║  ╚═══════════════════════════════════════════════╝  ║
    ║                                                      ║
    ╚════════════════════════════════════════════════════╝

NEW NAVIGATION CHALLENGES:

Altered Physics:
├── Some objects float (grab for platforms?)
├── Some areas have altered gravity
├── Movement speed varies by location
└── Jump height may be affected

Geometry Changes:
├── Paths that existed may be blocked
├── New paths may have opened (impossible geometry)
├── Doorways may lead elsewhere
└── Space may be larger/smaller than before

Exit Puzzle:
├── Original door may not work
├── New exit method required
├── May need to solve "nightmare puzzle"
└── May require surviving until timer ends
```

### Visual Distortion Examples

```
SPECIFIC DISTORTION TECHNIQUES:

The Desk:
├── Surface ripples like water
├── Objects on desk sink slowly
├── Wood grain moves (not static texture)
├── Edges drip onto floor
└── Effect: Solid becomes fluid

The Monitor:
├── Screen shows impossible images
├── Pixels drift apart (not cohesive)
├── Face may appear in static
├── Screen extends beyond frame (wrong geometry)
└── Effect: Technology becomes alive/wrong

The Chair:
├── Stretched too tall (unnatural proportions)
├── Spine bent like it's hurt
├── Wheels may have turned into something else
├── Rocks/breathes when not touched
└── Effect: Furniture is in pain

The Room:
├── Walls breathe in and out
├── Corners don't meet (wrong angles)
├── Distance is distorted (far things look close)
├── May be larger inside than outside
└── Effect: Space itself is hostile

COMBINED EFFECT:
Player's understanding of reality
is systematically broken down.
Nothing can be trusted.
```

---

## 🛠️ Technical Implementation

### What You Need to Know First

Before understanding the Office Hell implementation, you should know:
- **Splat Morphing**: Transitioning between two Gaussian splat captures
- **Post-Processing Effects**: Glitch, chromatic aberration, color shifts
- **Audio Distortion**: Real-time audio manipulation and effects
- **Material Shaders**: Custom shaders for "living" surfaces
- **Physics Alteration**: Changing gravity and collision properties

### Scene Data Structure

```javascript
// SceneData.js - Office Hell configuration
export const SCENES = {
  office_hell: {
    id: 'office_hell',
    name: 'Office Hell',
    type: 'interior_nightmare',

    // Nightmare version splat
    splat: {
      file: '/assets/splats/office_hell.ply',
      // Different rendering settings
      settings: {
        renderScale: 1.0,
        splatSize: 1.5,  // Larger for more "presence"
        opacity: 0.9,    // Slightly transparent = ghostly
        // Shader modifications for horror
        distortion: {
          enabled: true,
          type: 'pulse_breathe',  // Surface movement
          intensity: 0.3,
          speed: 0.5
        }
      }
    },

    // Transformation settings
    transformation: {
      from: 'office_interior',
      duration: 8.0,
      stages: [
        { time: 0.0, effect: 'flicker_start' },
        { time: 2.0, effect: 'object_displace' },
        { time: 4.0, effect: 'reality_tear' },
        { time: 5.0, effect: 'splat_swap' },
        { time: 6.0, effect: 'lighting_shift' },
        { time: 7.0, effect: 'stabilize' }
      ]
    },

    // Hellish lighting
    lighting: {
      ambient: {
        color: 0x331111,  // Blood red ambient
        intensity: 0.2
      },
      // Chaotic point lights
      chaosLights: [
        {
          position: { x: 0, y: 2.5, z: 0 },
          color: 0xff0000,
          intensity: 1.5,
          flicker: true,
          flickerSpeed: 10,  // Very fast
          flickerIntensity: 0.8
        },
        {
          position: { x: -2, y: 1, z: -1 },
          color: 0xff3300,
          intensity: 0.8,
          pulse: true,
          pulseSpeed: 2
        }
      ],
      // Volumetric fog (more atmosphere)
      fog: {
        color: 0x220000,
        density: 0.05,
        animated: true
      }
    },

    // Altered physics
    physics: {
      gravity: { x: 0, y: -5, z: 0 },  // Reduced gravity
      // Some objects float
      floatingObjects: ['debris_01', 'debris_02', 'paper_cluster'],
      // Some areas have no friction
      lowFrictionZones: [
        { center: { x: 0, z: 0 }, radius: 2 }
      ],
      // Some areas push player
      forceFields: [
        {
          center: { x: 1, y: 1, z: -1 },
          force: { x: 2, y: 0, z: 0 },
          radius: 1.5
        }
      ]
    },

    // Horror audio
    audio: {
      main: 'office_hell_ambience',
      volume: 0.6,
      layers: [
        { sound: 'reality_tear_loop', volume: 0.3 },
        { sound: 'whispers', volume: 0.15, random: true },
        { sound: 'heartbeat', volume: 0.2, syncToPlayer: true },
        { sound: 'drone_low', volume: 0.4 }
      ],
      distortion: {
        enabled: true,
        bitcrush: 0.3,
        pitchShift: -0.2,
        reverb: 0.8
      }
    },

    // Post-processing effects
    postProcessing: {
      chromaticAberration: 0.02,
      filmGrain: 0.15,
      vignette: 0.5,
      glitch: {
        enabled: true,
        intensity: 0.3,
        frequency: 0.1
      },
      colorGrading: {
        saturation: 0.5,
        contrast: 1.3,
        redChannel: 1.2,
        blueChannel: 0.8
      }
    }
  }
};
```

### Transformation Manager

```javascript
// TransformationManager.js - Handles scene transformation
class TransformationManager {
  constructor(sceneManager, audioManager, vfxManager) {
    this.scene = sceneManager;
    this.audio = audioManager;
    this.vfx = vfxManager;

    this.activeTransform = null;
    this.transformTimer = 0;
  }

  async transform(fromScene, toScene) {
    const config = toScene.transformation;
    this.activeTransform = {
      from: fromScene,
      to: toScene,
      config: config,
      currentStage: 0,
      timer: 0
    };

    // Begin transformation sequence
    for (const stage of config.stages) {
      await this.executeStage(stage);
      await this.delay(stage.time - this.transformTimer);
      this.transformTimer = stage.time;
    }

    // Transformation complete
    game.emit('transformation:complete', {
      from: fromScene.id,
      to: toScene.id
    });
  }

  async executeStage(stage) {
    switch (stage.effect) {
      case 'flicker_start':
        this.startFlicker();
        break;

      case 'object_displace':
        this.displaceObjects();
        break;

      case 'reality_tear':
        await this.realityTear();
        break;

      case 'splat_swap':
        await this.swapSplat();
        break;

      case 'lighting_shift':
        this.shiftLighting();
        break;

      case 'stabilize':
        this.stabilize();
        break;
    }
  }

  startFlicker() {
    // Begin with subtle light flicker
    this.vfx.trigger('light_flicker', {
      intensity: 0.2,
      frequency: 2,
      targets: 'all_lights'
    });

    // Audio glitch
    this.audio.playOneShot('transform_glitch_01', { volume: 0.4 });
  }

  displaceObjects() {
    // Objects begin to shift position
    const objects = this.scene.getInteractableObjects();

    for (const obj of objects) {
      // Random displacement
      const displacement = {
        x: (Math.random() - 0.5) * 0.1,
        y: (Math.random() - 0.5) * 0.05,
        z: (Math.random() - 0.5) * 0.1
      };

      // Animate to new position
      this.scene.animateObject(obj.id, {
        position: displacement,
        duration: 2.0,
        easing: 'easeInOutElastic'
      });
    }

    // Increased flicker
    this.vfx.updateEffect('light_flicker', {
      intensity: 0.4,
      frequency: 5
    });
  }

  async realityTear() {
    // The "snap" moment - reality breaks
    game.emit('reality:tearing');

    // Screen effects
    this.vfx.trigger('screen_shake', {
      intensity: 0.8,
      duration: 0.5
    });

    this.vfx.trigger('flash', {
      color: 0xff0000,
      duration: 0.2
    });

    // Audio crescendo
    this.audio.playOneShot('reality_tear', {
      volume: 1.0,
      fadeIn: 0.3
    });

    // Wait for effect
    await this.delay(500);

    // Mute all audio briefly
    this.audio.setMasterVolume(0);
  }

  async swapSplat() {
    // Swap the splat from normal to hell version
    const fromSplat = this.activeTransform.from.splat;
    const toSplat = this.activeTransform.to.splat;

    // Fade out old splat
    await this.scene.fadeSplat(0, 0.5);

    // Swap
    await this.scene.loadSplat(toSplat.file, toSplat.settings);

    // Fade in new splat
    await this.scene.fadeSplat(1, 0.5);

    // Restore audio with new ambience
    this.audio.setMasterVolume(1);
    this.audio.playAmbient(this.activeTransform.to.audio.main, {
      volume: this.activeTransform.to.audio.volume,
      fadeIn: 1.0
    });
  }

  shiftLighting() {
    // Remove old lights, add hell lights
    this.scene.clearLights();

    const toScene = this.activeTransform.to;
    for (const lightConfig of toScene.lighting.chaosLights) {
      this.scene.addChaosLight(lightConfig);
    }

    // Update ambient
    this.scene.setAmbientLight(toScene.lighting.ambient);

    // Add fog
    this.scene.setFog(toScene.lighting.fog);
  }

  stabilize() {
    // Effects settle into new normal
    this.vfx.stopEffect('light_flicker');

    // Enable post-processing
    this.vfx.enablePostProcessing(
      this.activeTransform.to.postProcessing
    );

    // Apply physics changes
    this.scene.setPhysics(
      this.activeTransform.to.physics
    );

    // Player is now in Office Hell
    game.emit('player:in_hell');
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

### Hell Material Shaders

```javascript
// HellMaterials.js - Custom shaders for horror effects
class HellMaterials {
  constructor() {
    this.materials = new Map();
  }

  // Breathing/pulsing surface material
  createBreathingMaterial(baseMaterial) {
    return new THREE.ShaderMaterial({
      uniforms: {
        baseTexture: { value: baseMaterial.map },
        time: { value: 0 },
        distortionIntensity: { value: 0.3 },
        distortionSpeed: { value: 0.5 },
        colorShift: { value: new THREE.Color(0xff3333) }
      },
      vertexShader: `
        uniform float time;
        uniform float distortionIntensity;
        uniform float distortionSpeed;

        varying vec2 vUv;
        varying float vDisplacement;

        // Simplex noise function
        vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

        float snoise(vec2 v) {
          const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                           -0.577350269189626, 0.024390243902439);
          vec2 i  = floor(v + dot(v, C.yy));
          vec2 x0 = v - i + dot(i, C.xx);
          vec2 i1;
          i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
          vec4 x12 = x0.xyxy + C.xxzz;
          x12.xy -= i1;
          i = mod289(i);
          vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
            + i.x + vec3(0.0, i1.x, 1.0));
          vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
            dot(x12.zw,x12.zw)), 0.0);
          m = m*m;
          m = m*m;
          vec3 x = 2.0 * fract(p * C.www) - 1.0;
          vec3 h = abs(x) - 0.5;
          vec3 ox = floor(x + 0.5);
          vec3 a0 = x - ox;
          m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
          vec3 g;
          g.x  = a0.x  * x0.x  + h.x  * x0.y;
          g.yz = a0.yz * vec2(x12.xz) + h.yz * vec2(x12.yw);
          return 130.0 * dot(m, g);
        }

        void main() {
          vUv = uv;
          float noise = snoise(position.xy * 0.5 + time * distortionSpeed);
          vDisplacement = noise * distortionIntensity;
          vec3 newPosition = position + normal * vDisplacement;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D baseTexture;
        uniform vec3 colorShift;
        varying vec2 vUv;
        varying float vDisplacement;

        void main() {
          vec4 baseColor = texture2D(baseTexture, vUv);
          // Mix with red based on displacement
          vec3 finalColor = mix(baseColor.rgb, colorShift, vDisplacement * 0.5);
          gl_FragColor = vec4(finalColor, baseColor.a);
        }
      `
    });
  }

  // Dripping/melting material
  createMeltingMaterial(baseMaterial) {
    return new THREE.ShaderMaterial({
      uniforms: {
        baseTexture: { value: baseMaterial.map },
        time: { value: 0 },
        dripSpeed: { value: 0.2 },
        dripAmount: { value: 0.5 }
      },
      vertexShader: `
        uniform float time;
        uniform float dripSpeed;
        uniform float dripAmount;

        varying vec2 vUv;
        varying float vDrip;

        void main() {
          vUv = uv;
          // Create drip effect based on Y position and time
          float drip = sin(position.y * 10.0 - time * dripSpeed) * 0.5 + 0.5;
          vDrip = drip * dripAmount * (1.0 - position.y);
          vec3 newPosition = position;
          newPosition.y -= vDrip;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D baseTexture;
        varying vec2 vUv;
        varying float vDrip;

        void main() {
          vec4 baseColor = texture2D(baseTexture, vUv);
          // Darken where dripping
          float darkening = vDrip * 0.5;
          gl_FragColor = vec4(baseColor.rgb * (1.0 - darkening), baseColor.a);
        }
      `
    });
  }

  update(time) {
    for (const [id, material] of this.materials) {
      if (material.uniforms.time) {
        material.uniforms.time.value = time;
      }
    }
  }
}
```

---

## 📝 How To Build A Scene Like This

### Step 1: Define the Horror Concept

```
HORROR DESIGN BRIEF:

1. What safe space are we corrupting?
    Office: Familiar, comfortable, recently explored

2. What makes it frightening?
    - Familiarity becomes threat
    - Physics break down
    - Reality becomes unstable
    - No escape apparent

3. What's the core emotion?
    Office Hell: Terror, disorientation, desperation

4. How does player escape (if at all)?
    - May be temporary (lasts for set time)
    - May require puzzle solution
    - May be narrative event (survive until end)

5. What's the narrative purpose?
    - Show reality is unstable
    - Raise stakes (nowhere safe)
    - Prepare player for more horror
```

### Step 2: Design the Transformation

```javascript
// Transformation timeline:

const transformationTimeline = {
  preEvent: {
    duration: 5,  // Seconds of "normal"
    playerAction: 'using_viewmaster',  // What triggers it
    subtleHints: [
      { at: -3, hint: 'light_flicker_once' },
      { at: -1, hint: 'audio_glitch_brief' }
    ]
  },

  buildUp: {
    duration: 3,
    effects: [
      { at: 0, effect: 'flicker_begin' },
      { at: 1, effect: 'object_shift' },
      { at: 2, effect: 'audio_distort_increase' }
    ]
  },

  snapMoment: {
    duration: 0.5,
    effects: [
      'screen_flash_red',
      'audio_silence_then_scream',
      'screen_shake_heavy',
      'reality_tear_sound'
    ]
  },

  postTransform: {
    duration: 2,
    effects: [
      'new_splat_fade_in',
      'hell_lighting_fade_in',
      'hell_ambience_fade_in'
    ]
  },

  stabilize: {
    after: 5.5,  // Total time from start
    playerState: 'in_hell',
    newRules: 'explain_navigation_changes'
  }
};
```

### Step 3: Create Distorted Assets

```javascript
// Asset distortion strategies:

const assetDistortion = {
  // Splat capture
  splat: {
    // Options:
    // 1. Capture separate "hell" version
    // 2. Procedurally distort normal splat
    // 3. Blend between two captures

    method: 'separate_capture',
    // Capture same space with:
    // - Red lighting
    // - Objects moved/melted
    // - Different atmosphere
  },

  // Materials
  materials: {
    // Use custom shaders for:
    // - Breathing surfaces
    // - Melting/dripping
    // - Color shifting
    // - Displacement maps
  },

  // Audio
  audio: {
    // Layers of horror:
    // - Base ambience (drone, groans)
    // - Random elements (whispers, pops)
    // - Player-synced (heartbeat)
    // - One-shots (tear, scream)
  }
};
```

### Step 4: Design Altered Gameplay

```javascript
// How player experience changes:

const alteredGameplay = {
  movement: {
    // Some areas slower
    slowZones: [
      { center: { x: 0, z: 0 }, radius: 2, speedMultiplier: 0.5 }
    ],

    // Some areas have no friction
    slipperyZones: [
      { center: { x: 1, z: -1 }, radius: 1.5 }
    ],

    // Reduced gravity
    gravity: 0.6,

    // Movement may drift
    driftEnabled: true
  },

  vision: {
    // Post-processing effects
    chromaticAberration: 0.02,
    vignette: 0.6,
    filmGrain: 0.2,

    // Occasional vision block
    staticOverlays: {
      enabled: true,
      frequency: 'random',
      duration: 0.2
    }
  },

  interaction: {
    // Most interactions disabled
    availableInteractions: ['exit_trigger_only'],

    // Some objects may be "grabbed" for platforms
    grabbableObjects: ['floating_debris']
  }
};
```

---

## 🔧 Variations For Your Game

### Variation 1: Temporary Nightmare

```javascript
const temporaryNightmare = {
  // Hell version only lasts for set time
  duration: 60,  // Seconds

  onEnd: {
    transition: 'fade_to_normal',
    playerState: 'traumatized_but_safe',
    permanentChange: 'subtle_wrongness_remains'
  }
};
```

### Variation 2: Recursive Hell

```javascript
const recursiveHell = {
  // Each time you enter, it's worse
  visits: [
    { visit: 1, severity: 'mild' },
    { visit: 2, severity: 'moderate' },
    { visit: 3, severity: 'severe' },
    { visit: 4, severity: 'extreme' }
  ]
};
```

### Variation 3: Puzzle Hell

```javascript
const puzzleHell = {
  // Player must solve puzzle to escape
  escapeCondition: {
    type: 'puzzle',
    puzzle: 'collect_fragments',
    fragments: 5,
    scattered: true,
    grabCondition: 'survive_hazards'
  }
};
```

---

## Performance Considerations

```
OFFICE HELL PERFORMANCE:

Splat Rendering:
├── Hell splat may be higher density
├── Shader effects add GPU load
├── Consider LOD for horror distance
└── Target: Accept 45 FPS (atmosphere > smooth)

Post-Processing:
├── Multiple effects are expensive
├── Chromatic aberration + grain + vignette
├── Consider quality settings
└── Target: Quality slider in options

Audio:
├── Many layers + distortion
├── Real-time effects are CPU intensive
├── Pre-render where possible
└── Target: No audio crackling

Physics:
├── Altered physics still need calculation
├── Floating objects add overhead
├── Force fields require per-frame checks
└── Target: Stable 30 FPS minimum

RECOMMENDATION:
Hell scenes are performance-heavy.
Optimize heavily, test on
minimum spec hardware.
```

---

## Common Mistakes Beginners Make

### 1. Transforming Too Abruptly

```javascript
// ❌ WRONG: Instant snap to hell
// Player is confused, not scared

// ✅ CORRECT: Build-up over several seconds
// Tension rises, then snap = more impact
```

### 2. Too Many Effects

```javascript
// ❌ WRONG: Every horror effect at once
// Player becomes numb, overwhelmed

// ✅ CORRECT: Layer effects progressively
// Each new effect adds to unease
```

### 3. No Clear Exit/Goal

```javascript
// ❌ WRONG: Player doesn't know what to do
// Frustration, not fear

// ✅ CORRECT: Clear objective
// "Find the exit," "Survive for 60 seconds," etc.
```

### 4: Transforming Back Too Easily

```javascript
// ❌ WRONG: Hell ends immediately, no consequences
// Player feels cheated, tension evaporates

// ✅ CORRECT: Transformation has lasting impact
// Things remain wrong, player is changed
```

---

## Related Systems

- [Office Interior Scene](./office-interior-scene.md) - Pre-transform version
- [VFXManager](../07-visual-effects/vfx-manager.md) - Visual effects
- [SFXManager](../05-media-systems/sfx-manager.md) - Audio effects
- [Post-Processing Effects](../07-visual-effects/glitch-post-processing.md) - Glitch effects
- [Dissolve Effect](../07-visual-effects/dissolve-effect.md) - Transition effects

---

## Source File Reference

**Scene Data**:
- `content/SceneData.js` - Office Hell configuration
- `content/AnimationData.js` - Transformation animations

**Managers**:
- `managers/TransformationManager.js` - Scene transitions
- `managers/HellMaterials.js` - Horror shader materials

**Assets**:
- `assets/splats/office_hell.ply` - Nightmare splat
- `assets/audio/office_hell_ambience.mp3` - Horror soundscape

---

## 🧠 Creative Process Summary

**From Concept to Office Hell**:

```
1. HORROR CONCEPT
   "Corrupt the safe space"

2. TRANSFORMATION DESIGN
   "Build tension, snap to new reality"

3. VISUAL DISTORTION
   "Everything becomes wrong version
    of itself"

4. AUDIO HORROR
   "Soundscape of nightmare"

5. GAMEPLAY ALTERATION
   "Change rules, make survival challenge"

6. ESCAPE CONDITION
   "Clear goal amid chaos"

7. LASTING IMPACT
   "Player is changed by experience"
```

---

## References

- [Silent Hill Design](https://www.youtube.com/watch?v=M4skP6bN_Ks) - Video essay on horror
- [P.T. Analysis](https://www.youtube.com/watch?v=U2V8UoG5-gA) - Psychological horror breakdown
- [Shader Programming](https://www.shadertoy.com/) - Shader examples
- [Audio Horror Design](https://www.youtube.com/watch?v=Q83eK9aY3bY) - Sound design tutorial

*Documentation last updated: January 12, 2026*
