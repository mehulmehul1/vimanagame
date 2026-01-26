# Viewmaster Scene - Psychological Horror Mechanic

**Scene Case Study #14**

---

## What You Need to Know First

- **Gaussian Splatting Basics** (See: *Rendering System*)
- **Interactive Object System** (See: *Interactive Objects*)
- **Insanity/Sanity Systems** (See: *Game Mechanics*)
- **Post-Processing Effects** (See: *VFXManager*)

---

## Scene Overview

| Property | Value |
|----------|-------|
| **Location** | Office interior, on desk |
| **Narrative Context** | Psychological horror device - shows glimpses of "what's really there" |
| **Player Experience** | Discovery → Examination → Disturbing Images → Sanity Impact → Visual Aftereffects |
| **Atmosphere** | Unsettling, invasive, reality-questioning |
| **Technical Focus** | Stereoscopic image viewer, sanity mechanics, visual distortion effects |

### The Scene's Purpose

The Viewmaster is a **psychological horror device** that introduces the concept that **reality is not what it seems**. Unlike other interactive objects that provide information or advance the plot, the Viewmaster's purpose is to:

1. **Undermine player confidence** in what they're seeing
2. **Introduce sanity mechanics** - repeated viewing has consequences
3. **Foreshadow truth** - images hint at the true nature of the world
4. **Create lingering effects** - viewing changes the game's visual presentation

This is a **horror through information** mechanic - showing the player something they weren't meant to see.

---

## 🎮 Game Design Perspective

### Creative Intent

**Why a Viewmaster? Why not a photo album, video tape, or vision?**

| Medium | Horror Effect |
|---------|---------------|
| **Photo Album** | Static, controlled viewing - player decides when to stop |
| **Video Tape** | Linear narrative - player is passive observer |
| **Vision/Dream** | External force - happens TO the player |
| **Viewmaster** | **Voluntary violation - player CHOOSES to see, but can't unsee** |

The Viewmaster creates **complicity in horror**. The player must physically pick it up and look through it to see the disturbing images. The horror comes from:

1. **Choice** - "I shouldn't look, but I want to"
2. **Intimacy** - It's just for you, through a small lens
3. **Permanence** - Once seen, can't be forgotten
4. **Escalation** - Each viewing shows more disturbing content

### Design Philosophy

**The Viewmaster as Insanity Meter:**

```
First viewing: Normal(ish) scenes, slightly off
         ↓
Second viewing: Same scenes, but something is wrong
         ↓
Third viewing: Disturbing details revealed
         ↓
Fourth+ viewing: The truth - entities, watching, threat
         ↓
Cumulative effect: Player's screen begins to show these things
```

The Viewmaster is **training the player to see** - and then making them see it everywhere.

### Mood Building

The Viewmaster contributes to horror through:

1. **Innocence Subverted** - Toy-like object, but shows horror
2. **Violation of Privacy** - Looking into private, disturbing moments
3. **Gaslighting** - "Did I really see that? Or am I losing it?"
4. **Creeping Dread** - Effects continue after putting it down

### Player Psychology

| Psychological Effect | How the Viewmaster Achieves It |
|---------------------|-------------------------------|
| **Curiosity** | What's on these slides? |
| **Violation** | I'm seeing something private/forbidden |
| **Doubt** | Is the game showing me this, or is my character insane? |
| **Paranoia** - If Viewmaster shows entities, are they here NOW? |
| **Helplessness** - I can't stop seeing what I've seen |

---

## 🎨 Level Design Breakdown

### Spatial Layout

```
┌─────────────────────────────────────────────┐
│                                             │
│   [Office Interior - Desk Area]             │
│                                             │
│         ┌─────────────────────┐              │
│         │                     │              │
│         │   [DESK]            │              │
│         │                     │              │
│         │  ┌───────┐          │              │
│         │  │ VIEW- │          │              │
│         │  │ MASTER│          │              │
│         │  └───────┘          │              │
│         │                     │              │
│         │  [Computer] [Chair] │              │
│         └─────────────────────┘              │
│                                             │
│   Player approaches desk from standing area  │
│   Viewmaster visible from ~5m away          │
│   Desk height: ~0.75m, eye-level interaction │
└─────────────────────────────────────────────┘
```

### Player Path

```
1. Player enters office, explores
   ↓
2. Notices Viewmaster on desk
   ↓
3. "Press E to examine" prompt
   ↓
4. Player presses E - first-person examination
   ↓
5. Viewmaster lifts to player's face (viewport animates)
   ↓
6. Stereoscopic image visible through circular viewport
   ↓
7. Player uses scroll wheel or triggers to advance slides
   ↓
8. Each slide shows increasingly disturbing content
   ↓
9. After X slides viewed, player lowers Viewmaster
   ↓
10. Visual aftereffects begin (hallucinations, distortions)
   ↓
11. Repeat viewings unlock more slides, worse effects
```

### Atmosphere Layers

| Layer | Elements |
|-------|----------|
| **Visual** | Retro toy aesthetic, circular viewport, stereoscopic depth |
| **Audio** | Clicking advance mechanism, muffled ambient, heartbeat on disturbing slides |
| **Lighting** | Darkens room when viewing, spotlight effect through viewport |
| **Interaction** | Smooth lifting animation, responsive slide advance, tangible feeling of use |

---

## Technical Implementation

### Viewmaster Data Structure

```javascript
export const VIEWMASTER = {
  id: 'viewmaster_office',
  type: 'interactive_prop',

  // Splat/Model data
  splat: {
    file: '/assets/splats/viewmaster.ply',
    maxPoints: 800000,
    renderScale: 1.0
  },

  // World placement
  transform: {
    position: { x: 2.3, y: 0.78, z: -1.5 },  // On desk surface
    rotation: { x: 0, y: 45, z: 0 },
    scale: { x: 1, y: 1, z: 1 }
  },

  // Interaction configuration
  interaction: {
    type: 'viewmaster',
    prompt: 'Examine Viewmaster',
    promptKey: 'e',
    maxDistance: 2.0
  },

  // Viewport configuration (what player sees)
  viewport: {
    shape: 'circle',
    radius: 0.15,  // Normalized screen space
    vignetteIntensity: 0.8,
    edgeDarkening: 0.9,
    stereoscopic: true,
    eyeSeparation: 0.02
  },

  // Slides configuration
  slides: {
    // Slide reel 1: Available from first viewing
    reel1: {
      id: 'reel_normal',
      unlockTrigger: null,  // Available immediately
      slides: [
        {
          id: 'slide_1_1',
          imageLeft: '/assets/textures/viewmaster/r1_s1_left.png',
          imageRight: '/assets/textures/viewmaster/r1_s1_right.png',
          caption: '',
          audio: null,
          duration: 0,
          disturbanceLevel: 1,
          description: 'Seemingly normal office scene'
        },
        {
          id: 'slide_1_2',
          imageLeft: '/assets/textures/viewmaster/r1_s2_left.png',
          imageRight: '/assets/textures/viewmaster/r1_s2_right.png',
          caption: '',
          audio: null,
          duration: 0,
          disturbanceLevel: 2,
          description: 'Office from different angle, something in corner'
        },
        {
          id: 'slide_1_3',
          imageLeft: '/assets/textures/viewmaster/r1_s3_left.png',
          imageRight: '/assets/textures/viewmaster/r1_s3_right.png',
          caption: '',
          audio: '/assets/audio/viewmaster/whisper.ogg',
          duration: 2000,
          disturbanceLevel: 3,
          description: 'Figure standing in corner, facing wall'
        }
      ]
    },

    // Slide reel 2: Unlocks after second viewing
    reel2: {
      id: 'reel_disturbing',
      unlockTrigger: 'viewmaster_second_use',
      slides: [
        {
          id: 'slide_2_1',
          imageLeft: '/assets/textures/viewmaster/r2_s1_left.png',
          imageRight: '/assets/textures/viewmaster/r2_s1_right.png',
          caption: '',
          audio: null,
          duration: 0,
          disturbanceLevel: 4,
          description: 'Same office, but walls are wrong'
        },
        {
          id: 'slide_2_2',
          imageLeft: '/assets/textures/viewmaster/r2_s2_left.png',
          imageRight: '/assets/textures/viewmaster/r2_s2_right.png',
          caption: '',
          audio: '/assets/audio/viewmaster/heartbeat.ogg',
          duration: 3000,
          disturbanceLevel: 5,
          description: 'Multiple figures, all facing player'
        }
      ]
    },

    // Slide reel 3: Unlocks after third viewing - the truth
    reel3: {
      id: 'reel_truth',
      unlockTrigger: 'viewmaster_third_use',
      slides: [
        {
          id: 'slide_3_1',
          imageLeft: '/assets/textures/viewmaster/r3_s1_left.png',
          imageRight: '/assets/textures/viewmaster/r3_s1_right.png',
          caption: '',
          audio: '/assets/audio/viewmaster/static_burst.ogg',
          duration: 500,
          disturbanceLevel: 8,
          description: 'Entity face, inches from lens'
        }
      ]
    }
  },

  // Sanity impact
  sanity: {
    impactPerSlide: 5,  // Sanity points lost per slide viewed
    thresholdForEffects: 20,  // Total sanity loss before visual effects start
    maxImpact: 50  // Maximum sanity loss from Viewmaster
  },

  // Visual aftereffects (insanity manifestations)
  aftereffects: {
    enabled: true,

    // Effects that trigger based on sanity loss
    effects: [
      {
        threshold: 20,
        effect: 'peripheral_figures',
        description: 'Shadowy figures at edge of vision'
      },
      {
        threshold: 30,
        effect: 'face_distortion',
        description: 'Brief glimpses of distorted faces'
      },
      {
        threshold: 40,
        effect: 'reality_glitch',
        description: 'World flickers to Viewmaster vision'
      }
    ]
  },

  // Audio configuration
  audio: {
    pickup: '/assets/audio/viewmaster/pickup.ogg',
    putdown: '/assets/audio/viewmaster/putdown.ogg',
    advance: '/assets/audio/viewmaster/advance.ogg',  // Click sound
    heartbeat: '/assets/audio/viewmaster/heartbeat.ogg',
    whispers: '/assets/audio/viewmaster/whispers.ogg'
  },

  // Animation configuration
  animation: {
    liftDuration: 600,  // ms to bring to face
    lowerDuration: 400,  // ms to put down
    viewportFadeIn: 200,
    viewportFadeOut: 150
  }
};
```

### Viewmaster Interaction Manager

```javascript
/**
 * Manages Viewmaster interactions and sanity effects
 */
class ViewmasterManager {
  constructor(scene) {
    this.scene = scene;
    this.gameManager = scene.gameManager;
    this.viewmasters = new Map();
  }

  /**
   * Initialize a Viewmaster
   */
  async loadViewmaster(config) {
    const viewmaster = {
      id: config.id,
      config: config,
      mesh: null,
      state: 'idle',
      currentReel: null,
      currentSlide: 0,
      viewingCount: 0,
      sanityImpact: 0
    };

    // Load Viewmaster mesh
    viewmaster.mesh = await this.loadViewmasterMesh(config);

    // Preload slide images
    await this.preloadSlides(config.slides);

    // Register interaction
    this.scene.interaction.register({
      id: config.id,
      type: config.interaction.type,
      prompt: config.interaction.prompt,
      key: config.interaction.promptKey,
      maxDistance: config.interaction.maxDistance,
      onInteract: () => this.onInteract(config.id)
    });

    this.viewmasters.set(config.id, viewmaster);
    return viewmaster;
  }

  /**
   * Load Viewmaster visual mesh
   */
  async loadViewmasterMesh(splatConfig) {
    const mesh = await this.scene.splatLoader.load(splatConfig.file);

    mesh.position.set(
      splatConfig.transform.position.x,
      splatConfig.transform.position.y,
      splatConfig.transform.position.z
    );

    mesh.rotation.set(
      THREE.MathUtils.degToRad(splatConfig.transform.rotation.x),
      THREE.MathUtils.degToRad(splatConfig.transform.rotation.y),
      THREE.MathUtils.degToRad(splatConfig.transform.rotation.z)
    );

    return mesh;
  }

  /**
   * Preload slide images for faster viewing
   */
  async preloadSlides(slidesConfig) {
    const loader = new THREE.TextureLoader();
    const loadedSlides = new Map();

    for (const [reelId, reel] of Object.entries(slidesConfig)) {
      loadedSlides.set(reelId, {
        config: reel,
        textures: []
      });

      for (const slide of reel.slides) {
        const leftTexture = await loader.loadAsync(slide.imageLeft);
        const rightTexture = await loader.loadAsync(slide.imageRight);

        loadedSlides.get(reelId).textures.push({
          left: leftTexture,
          right: rightTexture,
          config: slide
        });
      }
    }

    this.loadedSlides = loadedSlides;
  }

  /**
   * Handle player interacting with Viewmaster
   */
  async onInteract(viewmasterId) {
    const viewmaster = this.viewmasters.get(viewmasterId);
    if (!viewmaster || viewmaster.state !== 'idle') return;

    viewmaster.state = 'lifting';

    // Play pickup sound
    this.scene.audio.playOneShot(viewmaster.config.audio.pickup);

    // Determine which reel to use
    const reelId = this.selectReel(viewmaster);
    viewmaster.currentReel = reelId;
    viewmaster.currentSlide = 0;

    // Animate lifting to face
    await this.animateLift(viewmaster);

    // Start viewing mode
    this.startViewing(viewmaster, reelId);

    // Increment viewing count
    viewmaster.viewingCount++;

    // Trigger unlock for next reel
    this.checkReelUnlocks(viewmaster);
  }

  /**
   * Select appropriate slide reel based on viewing count
   */
  selectReel(viewmaster) {
    const viewingCount = viewmaster.viewingCount;

    if (viewingCount >= 2) {
      return 'reel3';  // Truth reel
    } else if (viewingCount === 1) {
      return 'reel2';  // Disturbing reel
    } else {
      return 'reel1';  // Normal reel
    }
  }

  /**
   * Check if new reels should be unlocked
   */
  checkReelUnlocks(viewmaster) {
    const count = viewmaster.viewingCount;

    if (count === 1) {
      this.gameManager.setCriteria('viewmaster_second_use', true);
    } else if (count === 2) {
      this.gameManager.setCriteria('viewmaster_third_use', true);
    }
  }

  /**
   * Animate Viewmaster lifting to player's face
   */
  async animateLift(viewmaster) {
    const duration = viewmaster.config.animation.liftDuration;
    const startTime = performance.now();

    const config = viewmaster.config;

    return new Promise((resolve) => {
      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = this.easeOutCubic(progress);

        // In actual implementation, this would animate the Viewmaster mesh
        // lifting from desk to player's face position

        if (progress >= 1) {
          viewmaster.state = 'viewing';
          resolve();
        } else {
          requestAnimationFrame(animate);
        }
      };

      animate();
    });
  }

  /**
   * Start viewing mode - show first slide
   */
  startViewing(viewmaster, reelId) {
    // Enable Viewmaster post-processing effect
    this.scene.vfx.enableEffect('viewmaster_viewport', {
      shape: viewmaster.config.viewport.shape,
      radius: viewmaster.config.viewport.radius,
      vignetteIntensity: viewmaster.config.viewport.vignetteIntensity,
      stereoscopic: viewmaster.config.viewport.stereoscopic
    });

    // Show first slide
    this.showSlide(viewmaster, reelId, 0);

    // Setup input for advancing slides
    this.setupViewingInput(viewmaster, reelId);
  }

  /**
   * Display a specific slide
   */
  showSlide(viewmaster, reelId, slideIndex) {
    const reel = this.loadedSlides.get(reelId);
    if (!reel || slideIndex >= reel.textures.length) {
      // End of reel - lower Viewmaster
      this.endViewing(viewmaster);
      return;
    }

    const slide = reel.textures[slideIndex];
    viewmaster.currentSlide = slideIndex;

    // Display slide in viewport
    this.scene.vfx.setViewmasterSlide({
      left: slide.left,
      right: slide.right,
      stereoscopic: viewmaster.config.viewport.stereoscopic
    });

    // Play slide-specific audio
    if (slide.config.audio) {
      this.scene.audio.playOneShot(slide.config.audio);
    }

    // Apply sanity impact
    this.applySanityImpact(viewmaster, slide.config);
  }

  /**
   * Apply sanity impact from viewing disturbing content
   */
  applySanityImpact(viewmaster, slideConfig) {
    const impact = slideConfig.disturbanceLevel * viewmaster.config.sanity.impactPerSlide;
    viewmaster.sanityImpact += impact;

    // Get current player sanity
    const currentSanity = this.gameManager.getState('player.sanity') || 100;
    const newSanity = Math.max(0, currentSanity - impact);

    this.gameManager.setState('player.sanity', newSanity);

    // Check for aftereffect thresholds
    this.checkAftereffects(viewmaster);
  }

  /**
   * Check if sanity-based effects should trigger
   */
  checkAftereffects(viewmaster) {
    const totalImpact = viewmaster.sanityImpact;

    for (const effect of viewmaster.config.aftereffects.effects) {
      if (totalImpact >= effect.threshold && !effect.triggered) {
        effect.triggered = true;
        this.triggerAftereffect(effect);
      }
    }
  }

  /**
   * Trigger an insanity aftereffect
   */
  triggerAftereffect(effectConfig) {
    switch (effectConfig.effect) {
      case 'peripheral_figures':
        this.scene.vfx.enableEffect('peripheral_figures', {
          intensity: 0.3,
          frequency: 0.1
        });
        break;

      case 'face_distortion':
        this.scene.vfx.enableEffect('face_distortion', {
          duration: 2000,
          intensity: 0.5
        });
        break;

      case 'reality_glitch':
        this.scene.vfx.enableEffect('reality_glitch', {
          frequency: 0.05,
          intensity: 0.7
        });
        break;
    }

    console.log(`Aftereffect triggered: ${effectConfig.effect}`);
  }

  /**
   * Setup input controls while viewing
   */
  setupViewingInput(viewmaster, reelId) {
    const onAdvance = () => {
      if (viewmaster.state !== 'viewing') return;

      // Play advance click sound
      this.scene.audio.playOneShot(viewmaster.config.audio.advance);

      // Show next slide
      this.showSlide(viewmaster, reelId, viewmaster.currentSlide + 1);
    };

    const onCancel = () => {
      if (viewmaster.state !== 'viewing') return;
      this.endViewing(viewmaster);
    };

    // Register input handlers
    viewmaster.inputHandlers = {
      advance: onAdvance,
      cancel: onCancel
    };

    // Mouse click or right trigger to advance
    this.scene.input.on('mouseDown', onAdvance);
    this.scene.input.on('gamepadButton0', onAdvance);

    // Escape or right trigger to exit
    this.scene.input.on('keyDown:escape', onCancel);
    this.scene.input.on('gamepadButton1', onCancel);
  }

  /**
   * End viewing mode - lower Viewmaster
   */
  async endViewing(viewmaster) {
    // Unregister input handlers
    if (viewmaster.inputHandlers) {
      this.scene.input.off('mouseDown', viewmaster.inputHandlers.advance);
      this.scene.input.off('gamepadButton0', viewmaster.inputHandlers.advance);
      this.scene.input.off('keyDown:escape', viewmaster.inputHandlers.cancel);
      this.scene.input.off('gamepadButton1', viewmaster.inputHandlers.cancel);
      delete viewmaster.inputHandlers;
    }

    viewmaster.state = 'lowering';

    // Fade out viewport
    await this.fadeViewportOut(viewmaster);

    // Disable Viewmaster effect
    this.scene.vfx.disableEffect('viewmaster_viewport');

    // Animate lowering
    await this.animateLower(viewmaster);

    // Play putdown sound
    this.scene.audio.playOneShot(viewmaster.config.audio.putdown);

    viewmaster.state = 'idle';
  }

  /**
   * Fade out Viewmaster viewport
   */
  async fadeViewportOut(viewmaster) {
    const duration = viewmaster.config.animation.viewportFadeOut;
    const startTime = performance.now();
    const startOpacity = 1;

    return new Promise((resolve) => {
      const fade = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const opacity = startOpacity * (1 - progress);

        this.scene.vfx.setViewmasterOpacity(opacity);

        if (progress >= 1) {
          resolve();
        } else {
          requestAnimationFrame(fade);
        }
      };

      fade();
    });
  }

  /**
   * Animate Viewmaster lowering back to desk
   */
  async animateLower(viewmaster) {
    const duration = viewmaster.config.animation.lowerDuration;
    const startTime = performance.now();

    return new Promise((resolve) => {
      const animate = () => {
        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = this.easeInCubic(progress);

        // Animate mesh back to desk position

        if (progress >= 1) {
          resolve();
        } else {
          requestAnimationFrame(animate);
        }
      };

      animate();
    });
  }

  /**
   * Update Viewmaster effects
   */
  update(deltaTime) {
    // Update any active insanity effects
    for (const viewmaster of this.viewmasters.values()) {
      if (viewmaster.sanityImpact > 0) {
        this.updateInsanityEffects(viewmaster, deltaTime);
      }
    }
  }

  /**
   * Update active insanity effects
   */
  updateInsanityEffects(viewmaster, deltaTime) {
    const totalImpact = viewmaster.sanityImpact;

    // Peripheral figures effect
    if (totalImpact >= 20) {
      const intensity = Math.min((totalImpact - 20) / 20, 1) * 0.5;
      // Update peripheral figure visibility
    }

    // Reality glitch - increases with higher sanity loss
    if (totalImpact >= 40) {
      const glitchChance = (totalImpact - 40) / 60;  // 0 to 1
      if (Math.random() < glitchChance * 0.01) {
        this.scene.vfx.trigger('momentary_glitch', {
          duration: 100 + Math.random() * 200
        });
      }
    }
  }

  /**
   * Easing functions
   */
  easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  easeInCubic(t) {
    return t * t * t;
  }
}
```

### Viewmaster Viewport Effect (Shader)

```javascript
/**
 * Post-processing effect for Viewmaster viewport
 */
class ViewmasterViewportEffect {
  constructor() {
    this.uniforms = {
      tDiffuse: { value: null },
      slideImageLeft: { value: null },
      slideImageRight: { value: null },
      viewportRadius: { value: 0.15 },
      vignetteIntensity: { value: 0.8 },
      edgeDarkening: { value: 0.9 },
      opacity: { value: 0 },
      eyeSeparation: { value: 0.02 },
      time: { value: 0 }
    };

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: `
        varying vec2 vUv;

        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform sampler2D slideImageLeft;
        uniform sampler2D slideImageRight;
        uniform float viewportRadius;
        uniform float vignetteIntensity;
        uniform float edgeDarkening;
        uniform float opacity;
        uniform float eyeSeparation;
        uniform float time;

        varying vec2 vUv;

        void main() {
          vec2 center = vec2(0.5, 0.5);
          float dist = distance(vUv, center);

          // Circular viewport mask
          float viewportMask = smoothstep(
            viewportRadius + 0.02,
            viewportRadius - 0.02,
            dist
          );

          // Vignette effect
          float vignette = 1.0 - (dist / viewportRadius) * vignetteIntensity;
          vignette = clamp(vignette, 0.0, 1.0);

          // Eye offset for stereoscopic effect
          float eyeOffset = (vUv.x < 0.5) ? -eyeSeparation : eyeSeparation;

          vec4 worldColor = texture2D(tDiffuse, vUv);

          vec4 slideColor;
          if (opacity > 0.0) {
            vec2 slideUv = vUv;
            slideUv.x += eyeOffset;
            vec2 eyeSlide = (vUv.x < 0.5) ?
              texture2D(slideImageLeft, slideUv).rgb :
              texture2D(slideImageRight, slideUv).rgb;

            slideColor = vec4(eyeSlide, 1.0);
          } else {
            slideColor = vec4(0.0);
          }

          // Darken edges of viewport
          float edge = 1.0 - smoothstep(
            viewportRadius - 0.05,
            viewportRadius,
            dist
          );
          vec3 edgeDarkeningVec = vec3(1.0 - edge * edgeDarkening);

          // Combine world and slide
          vec3 finalColor = mix(worldColor.rgb, slideColor.rgb, opacity * viewportMask);
          finalColor *= vignette;
          finalColor *= edgeDarkeningVec;

          // Darken outside viewport when viewing
          float outsideDarkening = mix(1.0, 0.1, opacity);
          if (viewportMask < 0.5) {
            finalColor *= outsideDarkening;
          }

          gl_FragColor = vec4(finalColor, 1.0);
        }
      `
    });
  }

  setSlide(leftTexture, rightTexture) {
    this.uniforms.slideImageLeft.value = leftTexture;
    this.uniforms.slideImageRight.value = rightTexture;
  }

  setOpacity(value) {
    this.uniforms.opacity.value = value;
  }

  setTime(value) {
    this.uniforms.time.value = value;
  }
}
```

---

## How To Build A Scene Like This

### Step 1: Define the Horror Mechanic

What fear does this device invoke?

```javascript
const horrorType = {
  viewmaster: {
    primary: 'reality questioning',
    secondary: 'complicity in horror',
    method: 'showing things player cannot unsee',
    persistence: 'effects continue after interaction'
  }
};
```

### Step 2: Design the Progression

```javascript
const progression = {
  viewing1: 'Mildly unsettling',
  viewing2: 'Disturbing',
  viewing3: 'Terrifying truth',
  cumulative: 'Permanent sanity loss and effects'
};
```

### Step 3: Create Visual Assets

```javascript
const assetCreation = {
  images: 'Stereoscopic pairs (slightly offset)',

  technique: 'Render 3D scene from two eye positions',

  content: 'Start normal, progressively add disturbing elements',

  format: 'High contrast for clear viewing through dark viewport'
};
```

### Step 4: Implement Viewport Effect

```javascript
const viewportEffect = {
  mask: 'Circular viewport',

  vignette: 'Darken edges for depth',

  stereoscopy: 'Different image for each eye',

  darkening: 'Darken world outside viewport to focus attention'
};
```

### Step 5: Add Consequences

```javascript
const consequences = {
  immediate: 'Sanity decrease per slide viewed',

  cumulative: 'Visual effects that persist in the world',

  progression: 'More disturbing slides unlock over time',

  permanence: 'Cannot undo sanity loss'
};
```

### Step 6: Test and Iterate

**Playtest Questions:**
1. Are the slides disturbing or just confusing?
2. Do the aftereffects enhance or annoy?
3. Is the player motivated to look again?
4. Does the horror stay with the player?

---

## Variations For Your Game

### Variation 1: Puzzle Clues

```javascript
const puzzleVersion = {
  slides: 'Contain puzzle clues and codes',

  usage: 'Required for progression',

  tone: 'Mysterious rather than horror',

  replay: 'Can reference slides anytime'
};
```

### Variation 2: Story Retelling

```javascript
const storyVersion = {
  slides: 'Narrative sequence showing past events',

  purpose: 'Environmental storytelling',

  effect: 'No sanity penalty, just information'
};
```

### Variation 3: Future Visions

```javascript
const propheticVersion = {
  slides: 'Show future events or warnings',

  mechanic: 'Player sees what will happen',

  consequences: 'Can change actions based on visions'
};
```

### Variation 4: Multiple Devices

```javascript
const multiDeviceVersion = {
  devices: ['Viewmaster', 'Kaleidoscope', 'Stereoscope'],

  each: 'Shows different aspect of truth',

  combination: 'Using all reveals complete picture'
};
```

---

## Performance Considerations

### Texture Optimization

```javascript
const optimization = {
  slideResolution: '1024x1024 per eye is sufficient',

  compression: 'Use compressed texture formats',

  streaming: 'Load next slide while viewing current',

  disposal: 'Unload viewed slides when not needed'
};
```

---

## Common Mistakes Beginners Make

### Mistake 1: Too Subtle

```javascript
// BAD: Player doesn't notice the horror
const badSubtlety = {
  slide1: 'Completely normal office',
  slide2: 'Office with tiny shadow',
  slide3: 'Same but slightly darker'
};

// GOOD: Clear escalation
const goodEscalation = {
  slide1: 'Normal office with subtle wrongness',
  slide2: 'Figure clearly visible',
  slide3: 'Entity faces player, unmistakable threat'
};
```

### Mistake 2: No Consequences

```javascript
// BAD: Viewing has no impact
const badConsequences = {
  effect: 'None, player just sees images'
};

// GOOD: Lasting impact
const goodConsequences = {
  effect: 'Sanity decrease, persistent visual effects',
  cumulative: 'Each viewing makes things worse'
};
```

### Mistake 3: Poor Viewport Design

```javascript
// BAD: Can't see clearly, no sense of depth
const badViewport = {
  shape: 'square',
  stereoscopic: false,
  vignette: 'none'
};

// GOOD: Immersive viewing experience
const goodViewport = {
  shape: 'circular',
  stereoscopic: true,
  vignette: 'strong edge darkening'
};
```

---

## Related Systems

- **Insanity System** - For managing sanity and its effects
- **Post-Processing** - For viewport and distortion effects
- **Dialog System** - For alternative storytelling methods
- **Other Interactive Objects** - For comparison of interaction types

---

## References

- **Shadow Engine Documentation**: `docs/`
- **Post-Processing Effects**: See *VFXManager*
- **Psychological Horror**: See *Game Design Principles*

---

**RALPH_STATUS:**
- **Status**: Viewmaster Scene documentation complete
- **Files Created**: `docs/generated/14-scene-case-studies/viewmaster-scene.md`
- **Related Documentation**: All Phase 14 scene case studies
- **Next**: Candlestick Phone Scene documentation
