import { dyno } from "@sparkjsdev/spark";
import { VFXManager } from "../vfxManager.js";
import { Logger } from "../utils/logger.js";
import { ProceduralAudio } from "./proceduralAudio.js";

/**
 * SplatMorphEffect - Morphs between two splat scenes
 *
 * Creates a dramatic transition where one splat scatters into particles
 * and reforms into a different splat scene.
 *
 * Usage:
 * - Extends VFXManager for automatic state-driven behavior
 * - Configured via vfxData.js
 * - Transitions triggered by game state changes
 */
export class SplatMorphEffect extends VFXManager {
  constructor(scene, sceneManager) {
    super("SplatMorph", true);
    this.scene = scene;
    this.sceneManager = sceneManager;

    // Splat meshes
    this.sourceMesh = null; // interior-nan-2.sog (already loaded)
    this.targetMesh = null; // office-hell.sog (we'll load this)
    this.meshes = []; // [source, target]

    // Animation state
    this.time = dyno.dynoFloat(0.0);
    this.isTransitioning = false;
    this.isPaused = true; // Start paused until triggered

    // Parameters (set by vfxData.js via applyEffect)
    this.parameters = {};

    // Procedural audio
    this.audio = new ProceduralAudio({
      name: "SplatMorphAudio",
      baseFrequency: 155.56 * 2, // Eb3
      volume: 0.4,
      baseOscType: "sawtooth",
      filterType: "lowpass",
      filterFreq: 2000,
      filterQ: 5,
      distortionAmount: 20,
      delayTime: 0.15,
      delayFeedback: 0.4,
      lfoFreq: 0.3,
      lfoDepth: 50,
      fadeInTime: 0.5, // Very fast fade in
      fadeOutTime: 0.25, // Fast fade out
    });
    this.lastPhase = 0;
    this.lastVelocity = 0;

    // Dyno dynamic uniforms (initialized with placeholder values, updated by applyEffect)
    this.numObjectsDyn = dyno.dynoInt(2); // Always 2 meshes for this effect
    this.stayDyn = dyno.dynoFloat(0);
    this.transDyn = dyno.dynoFloat(0);
    this.radiusDyn = dyno.dynoFloat(0);
    this.scatterCenterDyn = {
      x: dyno.dynoFloat(0),
      y: dyno.dynoFloat(0),
      z: dyno.dynoFloat(0),
    };
    // Wipe mode parameters
    this.transitionModeDyn = dyno.dynoInt(0); // 0 = scatter, 1 = wipe
    this.wipeDirectionDyn = dyno.dynoInt(0); // 0 = bottom-to-top, 1 = top-to-bottom
    this.wipeSoftnessDyn = dyno.dynoFloat(0.1); // Wipe edge softness

    this.logger.log("SplatMorphEffect created");
  }

  /**
   * Hook: Called the first time an effect matches
   * Get splats from sceneManager and setup the morph
   */
  async onFirstEnable(effect, state) {
    this.logger.log("First enable - waiting for splats and setting up morph");

    try {
      // Get the source splat (interior-nan-2.sog) from sceneManager
      this.sourceMesh = this.sceneManager.getObject("interior");
      if (!this.sourceMesh) {
        this.logger.error("Source splat 'interior' not found in sceneManager");
        return;
      }
      this.logger.log("Found source splat: interior-nan-2.sog");

      // Wait for target splat (office-hell.sog) to be loaded by sceneManager
      await this._waitForTargetSplat();

      // Setup morph modifiers on both splats
      this._setupMorphModifiers();

      // Initialize procedural audio
      await this.audio.initialize();

      // Apply the effect parameters
      this.applyEffect(effect, state);
    } catch (error) {
      this.logger.error("Failed to initialize splat morph:", error);
    }
  }

  /**
   * Wait for target splat (office-hell.sog) to be loaded by sceneManager
   * @private
   */
  async _waitForTargetSplat() {
    this.logger.log("Waiting for target splat: office-hell.sog");

    // Check if already loaded
    this.targetMesh = this.sceneManager.getObject("officeHell");
    if (this.targetMesh) {
      this.logger.log("Target splat already loaded");
      this.meshes = [this.sourceMesh, this.targetMesh];
      return;
    }

    // Wait for it to load (sceneManager loads it based on state criteria)
    this.logger.log("Target splat not loaded yet, waiting...");
    const maxWaitTime = 10000; // 10 seconds timeout
    const startTime = Date.now();

    while (!this.targetMesh && Date.now() - startTime < maxWaitTime) {
      // Check if it's loading
      if (this.sceneManager.isLoading("officeHell")) {
        this.logger.log("  officeHell is loading...");
        // Wait for the loading promise
        try {
          await this.sceneManager.loadingPromises.get("officeHell");
          this.targetMesh = this.sceneManager.getObject("officeHell");
          break;
        } catch (error) {
          this.logger.error("Error waiting for officeHell to load:", error);
          throw error;
        }
      }

      // Check again if it's now available
      this.targetMesh = this.sceneManager.getObject("officeHell");
      if (this.targetMesh) {
        break;
      }

      // Wait a bit before checking again
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    if (!this.targetMesh) {
      throw new Error(
        "Timeout waiting for officeHell splat to load from sceneManager"
      );
    }

    this.logger.log("Target splat loaded by sceneManager");
    this.meshes = [this.sourceMesh, this.targetMesh];
  }

  /**
   * Create the morph dyno shader
   * @private
   */
  _createMorphDyno() {
    return new dyno.Dyno({
      inTypes: {
        gsplat: dyno.Gsplat,
        gt: "float",
        objectIndex: "int",
        stay: "float",
        trans: "float",
        numObjects: "int",
        randomRadius: "float",
        offsetY: "float",
        scatterCenterX: "float",
        scatterCenterY: "float",
        scatterCenterZ: "float",
        transitionMode: "int", // 0 = scatter, 1 = wipe
        wipeDirection: "int", // 0 = bottom-to-top, 1 = top-to-bottom
        wipeSoftness: "float",
      },
      outTypes: { gsplat: dyno.Gsplat },
      globals: () => [
        dyno.unindent(`
          vec3 hash3(int n) {
            float x = float(n);
            return fract(sin(vec3(x, x + 1.0, x + 2.0)) * 43758.5453123);
          }
          float ease(float x) { return x*x*(3.0 - 2.0*x); }
          vec3 randPos(int splatIndex, float radius, vec3 center) {
            // Uniform sphere sampling around center point
            vec3 h = hash3(splatIndex);
            float theta = 6.28318530718 * h.x; // Azimuthal angle
            float phi = acos(2.0 * h.y - 1.0);  // Polar angle
            float r = radius * pow(h.z, 0.333); // Cubic root for uniform distribution
            
            // Convert spherical to Cartesian and offset by center
            return center + vec3(
              r * sin(phi) * cos(theta),
              r * sin(phi) * sin(theta),
              r * cos(phi)
            );
          }
        `),
      ],
      statements: ({ inputs, outputs }) =>
        dyno.unindentLines(`
          ${outputs.gsplat} = ${inputs.gsplat};
          float stay = ${inputs.stay};
          float trans = ${inputs.trans};
          float w = ${inputs.gt};
          
          // Clamp transition to happen once (no cycling)
          bool inTrans = w > stay && w < (stay + trans);
          bool afterTrans = w >= (stay + trans);
          
          float uPhase = inTrans ? clamp((w - stay) / trans, 0.0, 1.0) : 0.0;
          if (afterTrans) uPhase = 1.0;
          
          int idx = ${inputs.objectIndex};
          int mode = ${inputs.transitionMode};
          
          vec3 origScale = ${inputs.gsplat}.scales;
          vec3 small = ${inputs.gsplat}.scales * 0.2;
          float alpha = 0.0;
          vec3 pos = ${inputs.gsplat}.center;

          // Mode 1: Wipe transition
          if (mode == 1) {
            // Get Y coordinate of splat (for bottom-to-top or top-to-bottom wipe)
            float splatY = ${inputs.gsplat}.center.y;
            
            // Estimate scene bounds (adjust these based on your scene)
            float minY = -5.0;  // Approximate floor
            float maxY = 10.0;  // Approximate ceiling
            
            // Normalize Y to 0-1 range
            float normalizedY = clamp((splatY - minY) / (maxY - minY), 0.0, 1.0);
            
            // Flip for top-to-bottom (wipeDirection = 1)
            if (${inputs.wipeDirection} == 1) {
              normalizedY = 1.0 - normalizedY;
            }
            
            // Calculate wipe threshold with softness
            float softness = ${inputs.wipeSoftness};
            float wipeMask = smoothstep(
              uPhase - softness,
              uPhase + softness,
              normalizedY
            );
            
            // Source splat (idx 0): fades out as wipe progresses
            if (idx == 0) {
              if (!inTrans && !afterTrans) {
                // Before transition: show source
                alpha = 1.0;
                ${outputs.gsplat}.scales = origScale;
              } else {
                // During/after transition: fade out based on wipe
                alpha = 1.0 - wipeMask;
                ${outputs.gsplat}.scales = mix(origScale, small, wipeMask);
              }
            }
            // Target splat (idx 1): fades in as wipe progresses
            else if (idx == 1) {
              if (!inTrans) {
                // Before transition: hide target
                alpha = 0.0;
                ${outputs.gsplat}.scales = small;
              } else {
                // During/after transition: fade in based on wipe
                alpha = wipeMask;
                ${outputs.gsplat}.scales = mix(small, origScale, wipeMask);
              }
            }
          }
          // Mode 0: Scatter transition (original behavior)
          else {
            bool phaseScatter = uPhase < 0.5;
            float s = phaseScatter ? (uPhase / 0.5) : ((uPhase - 0.5) / 0.5);

            vec3 scatterCenter = vec3(${inputs.scatterCenterX}, ${inputs.scatterCenterY}, ${inputs.scatterCenterZ});
            vec3 rp = randPos(int(${inputs.gsplat}.index), ${inputs.randomRadius}, scatterCenter);
            rp.y -= ${inputs.offsetY};
            vec3 rpMid = mix(${inputs.gsplat}.center, rp, 0.7);

            // Source splat (idx 0)
            if (idx == 0) {
              if (!inTrans && !afterTrans) {
                // Before transition: show source
                alpha = 1.0;
                pos = ${inputs.gsplat}.center;
                ${outputs.gsplat}.scales = origScale;
              } else if (inTrans && phaseScatter) {
                // First half of transition: scatter source
                alpha = 1.0 - ease(s) * 0.5;
                pos = mix(${inputs.gsplat}.center, rpMid, ease(s));
                ${outputs.gsplat}.scales = mix(origScale, small, ease(s));
              } else {
                // Second half and after: hide source
                alpha = 0.0;
                pos = rpMid;
                ${outputs.gsplat}.scales = small;
              }
            }
            // Target splat (idx 1)
            else if (idx == 1) {
              if (!inTrans) {
                // Before transition: hide target
                alpha = 0.0;
                pos = rpMid;
                ${outputs.gsplat}.scales = small;
              } else if (phaseScatter) {
                // First half of transition: keep hidden
                alpha = 0.0;
                pos = rpMid;
                ${outputs.gsplat}.scales = small;
              } else {
                // Second half and after: show target
                alpha = max(ease(s), 0.5);
                pos = mix(rpMid, ${inputs.gsplat}.center, ease(s));
                ${outputs.gsplat}.scales = mix(small, origScale, ease(s));
              }
              if (afterTrans) {
                // After transition: fully show target
                alpha = 1.0;
                pos = ${inputs.gsplat}.center;
                ${outputs.gsplat}.scales = origScale;
              }
            }

            pos.y += ${inputs.offsetY};
          }

          ${outputs.gsplat}.center = pos;
          ${outputs.gsplat}.rgba.a = ${inputs.gsplat}.rgba.a * alpha;
        `),
    });
  }

  /**
   * Get morph modifier for a specific splat
   * @private
   */
  _getMorphModifier(
    gt,
    idx,
    stay,
    trans,
    numObjects,
    randomRadius,
    offsetY,
    scatterCenter,
    transitionMode,
    wipeDirection,
    wipeSoftness
  ) {
    const dyn = this._createMorphDyno();
    return dyno.dynoBlock(
      { gsplat: dyno.Gsplat },
      { gsplat: dyno.Gsplat },
      ({ gsplat }) => ({
        gsplat: dyn.apply({
          gsplat,
          gt,
          objectIndex: idx,
          stay,
          trans,
          numObjects,
          randomRadius,
          offsetY,
          scatterCenterX: scatterCenter.x,
          scatterCenterY: scatterCenter.y,
          scatterCenterZ: scatterCenter.z,
          transitionMode,
          wipeDirection,
          wipeSoftness,
        }).gsplat,
      })
    );
  }

  /**
   * Setup morph modifiers on both splat meshes
   * @private
   */
  _setupMorphModifiers() {
    if (!this.sourceMesh || !this.targetMesh) {
      this.logger.error("Cannot setup modifiers - missing meshes");
      return;
    }

    const offsetYValues = [dyno.dynoFloat(0.0), dyno.dynoFloat(0.0)];

    this.meshes.forEach((mesh, i) => {
      mesh.worldModifier = this._getMorphModifier(
        this.time,
        dyno.dynoInt(i),
        this.stayDyn,
        this.transDyn,
        this.numObjectsDyn,
        this.radiusDyn,
        offsetYValues[i],
        this.scatterCenterDyn,
        this.transitionModeDyn,
        this.wipeDirectionDyn,
        this.wipeSoftnessDyn
      );
      mesh.updateGenerator();
    });

    this.logger.log("Morph modifiers applied to both splats");
  }

  /**
   * Hook: Apply effect parameters
   */
  applyEffect(effect, state) {
    const params = effect.parameters || {};

    // Update parameters from vfxData.js
    this.parameters = { ...params };

    // Update dyno uniforms
    if (params.staySeconds !== undefined) {
      this.stayDyn.value = params.staySeconds;
    }
    if (params.transitionSeconds !== undefined) {
      this.transDyn.value = params.transitionSeconds;
    }
    if (params.randomRadius !== undefined) {
      this.radiusDyn.value = params.randomRadius;
    }

    // Update scatter center if provided
    if (params.scatterCenter) {
      this.scatterCenterDyn.x.value = params.scatterCenter.x;
      this.scatterCenterDyn.y.value = params.scatterCenter.y;
      this.scatterCenterDyn.z.value = params.scatterCenter.z;
    }

    // Update transition mode (scatter vs wipe)
    if (params.mode !== undefined) {
      const mode = params.mode === "wipe" ? 1 : 0;
      this.transitionModeDyn.value = mode;
      this.logger.log(`Transition mode: ${params.mode} (${mode})`);
    }

    // Update wipe direction (bottom-to-top vs top-to-bottom)
    if (params.wipeDirection !== undefined) {
      const direction = params.wipeDirection === "top-to-bottom" ? 1 : 0;
      this.wipeDirectionDyn.value = direction;
      this.logger.log(`Wipe direction: ${params.wipeDirection} (${direction})`);
    }

    // Update wipe softness
    if (params.wipeSoftness !== undefined) {
      this.wipeSoftnessDyn.value = params.wipeSoftness;
    }

    // Control animation state
    if (params.trigger === "start") {
      this.logger.log("Starting morph transition");
      this.isPaused = false;
      this.isTransitioning = true;
      this.time.value = 0.0; // Reset animation
      this.lastPhase = 0;
      this.lastVelocity = 0;
      if (!params.suppressAudio) {
        this.audio.start();
      }
    } else if (params.trigger === "pause") {
      this.isPaused = true;
      this.isTransitioning = false;
      this.audio.stop();
    } else if (params.trigger === "reset") {
      this.time.value = 0.0;
      this.isPaused = true;
      this.isTransitioning = false;
      this.audio.stop();
    }

    this.logger.log(`Applied effect: ${effect.id}`, this.parameters);
  }

  /**
   * Hook: Called when no effect matches
   */
  onNoEffect(state) {
    // Pause the animation when leaving the state
    this.isPaused = true;
    this.isTransitioning = false;
    this.audio.stop();

    // Remove world modifiers from both splats to prevent them staying in broken state
    if (this.sourceMesh) {
      this.sourceMesh.worldModifier = null;
      this.sourceMesh.updateGenerator();
      this.sourceMesh.updateVersion();
    }
    if (this.targetMesh) {
      this.targetMesh.worldModifier = null;
      this.targetMesh.updateGenerator();
      this.targetMesh.updateVersion();
    }

    this.logger.log("No effect - removed worldModifiers and paused morph");
  }

  /**
   * Update animation
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    if (this.isPaused || !this.isTransitioning) return;
    if (!this.sourceMesh || !this.targetMesh) return;

    // Update time
    const speedMultiplier = this.parameters.speedMultiplier || 1.0;
    this.time.value += dt * speedMultiplier;

    // Calculate phase for audio synchronization
    const w = this.time.value;
    const stay = this.parameters.staySeconds || 0;
    const trans = this.parameters.transitionSeconds || 1;

    // Calculate phase (0 to 1 during transition)
    let uPhase = 0;
    const inTrans = w > stay && w < stay + trans;
    const afterTrans = w >= stay + trans;

    if (inTrans) {
      uPhase = Math.max(0, Math.min(1, (w - stay) / trans));
    } else if (afterTrans) {
      uPhase = 1.0;
    }

    // Stop audio after transition completes
    if (afterTrans && this.audio.isPlaying) {
      this.audio.stop();
      this.logger.log("Transition complete - stopping audio");
    }

    // Determine if we're scattering or reforming
    const isScattering = uPhase < 0.5;

    // Calculate velocity from phase using derivative of easing function
    // ease(x) = x*x*(3 - 2*x)
    // ease'(x) = 6*x - 6*x*x
    let s; // normalized phase within current half (0 to 1)
    if (isScattering) {
      s = uPhase / 0.5; // 0 to 1 during first half
    } else {
      s = (uPhase - 0.5) / 0.5; // 0 to 1 during second half
    }

    // Velocity is the derivative of the easing function
    const rawVelocity = 6 * s - 6 * s * s;
    // Normalize to 0-1 range (derivative peak is 1.5)
    const velocity = Math.max(0, Math.min(1, rawVelocity / 1.5));

    // Calculate acceleration (change in velocity)
    const dt_safe = Math.max(dt, 0.001); // Prevent division by zero
    const acceleration = (velocity - this.lastVelocity) / dt_safe;
    // Normalize acceleration to roughly -1 to 1 range
    const normalizedAccel = Math.max(-1, Math.min(1, acceleration * 0.1));

    // Update audio with morph state
    // Map particle velocity to filter frequency (higher velocity = brighter)
    const minFreq = 200;
    const maxFreq = 8000;
    const targetFilterFreq = minFreq + (maxFreq - minFreq) * velocity;

    // Map acceleration to filter resonance (more acceleration = more dramatic)
    const minQ = 1;
    const maxQ = 15;
    const accelIntensity = Math.abs(normalizedAccel);
    const targetQ = minQ + (maxQ - minQ) * accelIntensity;

    // Map phase to pitch bend
    let pitchMultiplier;
    if (isScattering) {
      pitchMultiplier = 1.0 + uPhase * 0.5; // Rise from 1.0 to 1.5
    } else {
      if (uPhase < 0.5) {
        pitchMultiplier = 1.0 - uPhase * 0.4; // 1.0 -> 0.8
      } else {
        pitchMultiplier = 0.8 + (uPhase - 0.5) * 0.4; // 0.8 -> 1.0
      }
    }

    // Map acceleration to delay feedback (more chaos when accelerating)
    const minFeedback = 0.2;
    const maxFeedback = 0.65;
    const targetFeedback =
      minFeedback + (maxFeedback - minFeedback) * accelIntensity;

    // Map velocity to stereo panning (particles moving = sound moves)
    const panAmount = Math.sin(uPhase * Math.PI * 4) * velocity * 0.7;

    // Map acceleration to volume (peaks during high acceleration)
    const minVolume = 0.2;
    const maxVolume = 0.4;
    const targetVolume =
      minVolume + (maxVolume - minVolume) * (0.5 + accelIntensity * 0.5);

    this.audio.updateParams({
      filterFreq: targetFilterFreq,
      filterQ: targetQ,
      pitchMultiplier,
      delayFeedback: targetFeedback,
      pan: panAmount,
      volume: targetVolume,
      transitionTime: 0.05,
    });

    // Store for next frame
    this.lastPhase = uPhase;
    this.lastVelocity = velocity;

    // Update both meshes to apply dyno modifiers
    this.meshes.forEach((mesh) => {
      mesh.updateVersion();
    });
  }

  /**
   * Cleanup
   */
  dispose() {
    this.logger.log("Disposing splat morph effect");

    // Clean up audio
    if (this.audio) {
      this.audio.dispose();
      this.audio = null;
    }

    // Note: Both splats are managed by sceneManager
    // We just clean up our references and remove the modifiers

    // Remove world modifiers from both splats
    if (this.sourceMesh) {
      this.sourceMesh.worldModifier = null;
      this.sourceMesh.updateGenerator();
    }
    if (this.targetMesh) {
      this.targetMesh.worldModifier = null;
      this.targetMesh.updateGenerator();
    }

    this.sourceMesh = null;
    this.targetMesh = null;
    this.meshes = [];
  }
}

export default SplatMorphEffect;
