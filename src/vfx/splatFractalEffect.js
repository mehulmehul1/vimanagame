import { dyno } from "@sparkjsdev/spark";
import { VFXManager } from "../vfxManager.js";
import { Logger } from "../utils/logger.js";
import { ProceduralAudio } from "./proceduralAudio.js";

/**
 * SplatFractalEffect - Applies fractal shader effects to splat meshes
 *
 * Creates various visual effects including:
 * - Electronic: Fractal patterns with head movement
 * - Deep Meditation: Breathing animation with complex fractals
 * - Waves: Sine wave distortions
 * - Disintegrate: Particle disintegration
 * - Flare: Light flare gathering effect
 *
 * Usage:
 * - Extends VFXManager for automatic state-driven behavior
 * - Configured via vfxData.js
 * - Can be applied to specific splat meshes or all splats
 */
export class SplatFractalEffect extends VFXManager {
  constructor(scene, sceneManager) {
    super("SplatFractalEffect", false);
    this.scene = scene;
    this.sceneManager = sceneManager;

    // Target splat meshes
    this.targetMeshes = [];
    this.targetMeshIds = [];

    // Animation time
    this.animateT = dyno.dynoFloat(0);

    // Effect parameters (set by vfxData.js via applyEffect)
    this.parameters = {};

    // Intensity ramping
    this.currentIntensity = 0;
    this.targetIntensity = 0;
    this.rampStartTime = 0;
    this.rampDuration = 0;
    this.isRamping = false;
    this.rampOutDuration = 0;
    this.isRampingOut = false;
    this.rampOutStartIntensity = 0;

    // Effect type mapping
    this.effectTypeMap = {
      electronic: 1,
      deepMeditation: 2,
      waves: 3,
      flare: 4,
      disintegrate: 5,
    };

    // Procedural audio that builds tension leading to morph
    this.audio = new ProceduralAudio({
      name: "SplatFractalAudio",
      baseFrequency: 155.56, // Eb3 - same key as morph
      volume: 0.25, // Slightly quieter than morph
      baseOscType: "sine", // Start smooth
      subOscType: "sine",
      modOscType: "sine",
      filterType: "lowpass",
      filterFreq: 800, // Start low/dark
      filterQ: 2,
      distortionAmount: 8, // Moderate distortion
      delayTime: 0.2,
      delayFeedback: 0.35,
      lfoFreq: 0.2, // Slow modulation
      lfoDepth: 40,
      fadeInTime: 0.1,
      fadeOutTime: 0.15,
    });

    this.logger.log("SplatFractalEffect created");
  }

  /**
   * Hook: Called the first time an effect matches
   */
  async onFirstEnable(effect, state) {
    this.logger.log("First enable - setting up fractal effect");

    try {
      // Initialize procedural audio
      await this.audio.initialize();

      // Ensure arrays are initialized
      if (!Array.isArray(this.targetMeshes)) {
        this.targetMeshes = [];
      }
      if (!Array.isArray(this.targetMeshIds)) {
        this.targetMeshIds = [];
      }

      // Get target mesh IDs from effect parameters
      const targetIds = effect.parameters?.targetMeshIds || [];
      
      // Ensure targetIds is an array
      if (!Array.isArray(targetIds)) {
        this.logger.error("targetMeshIds must be an array", targetIds);
        return;
      }

      if (targetIds.length === 0) {
        this.logger.warn("No target mesh IDs specified");
        return;
      }

      // Check if we actually need meshes for this effect
      const intensity = effect.parameters?.intensity ?? 0;
      const rampDuration = effect.parameters?.rampDuration ?? 0;
      const needsMeshes = intensity > 0 || rampDuration > 0;

      if (!needsMeshes) {
        // Effect has zero intensity and no ramp - no meshes needed
        this.logger.log(
          "Effect has zero intensity and no ramp - skipping mesh loading"
        );
        // Store target IDs for potential future use, but don't load meshes
        this.targetMeshIds = Array.isArray(targetIds) ? [...targetIds] : [];
        // Apply effect directly (will be a no-op but sets up state correctly)
        this.applyEffect(effect, state);
        return;
      }

      // Wait for and get meshes from sceneManager
      for (const meshId of targetIds) {
        // Validate meshId
        if (typeof meshId !== 'string') {
          this.logger.warn(`Invalid meshId type: ${typeof meshId}, skipping`, meshId);
          continue;
        }

        // Check if mesh already exists before waiting
        let mesh = this.sceneManager.getObject(meshId);
        
        if (!mesh) {
          // Only wait for mesh if it's actually being loaded
          if (this.sceneManager.isLoading(meshId)) {
            mesh = await this._waitForMesh(meshId);
          } else {
            // Mesh isn't loaded and isn't loading - skip it
            this.logger.log(
              `Mesh "${meshId}" not available and not loading - skipping (may not be loaded for current state)`
            );
            continue;
          }
        }

        if (mesh) {
          this.targetMeshes.push(mesh);
          this.targetMeshIds.push(meshId);
          this.logger.log(`Found target mesh: ${meshId}`);
        } else {
          this.logger.warn(`Target mesh "${meshId}" not found or timed out`);
        }
      }

      if (this.targetMeshes.length === 0) {
        this.logger.warn(
          "No target meshes available - effect may not apply until meshes are loaded"
        );
        // Store target IDs for potential future use (create new array to avoid reference issues)
        this.targetMeshIds = Array.isArray(targetIds) ? [...targetIds] : [];
        // Still apply effect (will be a no-op but sets up state correctly)
        this.applyEffect(effect, state);
        return;
      }

      // Apply the effect
      this.applyEffect(effect, state);

      // Start audio when effect starts
      if (this.audio && !this.audio.isPlaying) {
        this.audio.start();
      }
    } catch (error) {
      this.logger.error("Failed to initialize fractal effect:", error);
    }
  }

  /**
   * Wait for a mesh to be loaded by sceneManager
   * @private
   */
  async _waitForMesh(meshId) {
    this.logger.log(`Waiting for mesh: ${meshId}`);

    // Check if already loaded
    let mesh = this.sceneManager.getObject(meshId);
    if (mesh) {
      this.logger.log(`Mesh ${meshId} already loaded`);
      return mesh;
    }

    // Wait for it to load
    const maxWaitTime = 10000; // 10 seconds timeout
    const startTime = Date.now();

    while (!mesh && Date.now() - startTime < maxWaitTime) {
      // Check if it's loading
      if (this.sceneManager.isLoading(meshId)) {
        this.logger.log(`  ${meshId} is loading...`);
        try {
          await this.sceneManager.loadingPromises.get(meshId);
          mesh = this.sceneManager.getObject(meshId);
          break;
        } catch (error) {
          this.logger.error(`Error waiting for ${meshId} to load:`, error);
          return null;
        }
      }

      // Check again if it's now available
      mesh = this.sceneManager.getObject(meshId);
      if (mesh) {
        break;
      }

      // Wait a bit before checking again
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    if (!mesh) {
      this.logger.error(`Timeout waiting for ${meshId} to load`);
      return null;
    }

    this.logger.log(`Mesh ${meshId} loaded successfully`);
    return mesh;
  }

  /**
   * Create the fractal shader dyno
   * @private
   */
  _createFractalDyno() {
    return new dyno.Dyno({
      inTypes: {
        gsplat: dyno.Gsplat,
        t: "float",
        effectType: "int",
        intensity: "float",
      },
      outTypes: { gsplat: dyno.Gsplat },
      globals: () => [
        dyno.unindent(`
          vec3 hash(vec3 p) {
            return fract(sin(p*123.456)*123.456);
          }

          mat2 rot(float a) {
            float s = sin(a), c = cos(a);
            return mat2(c, -s, s, c);
          }

          vec3 headMovement(vec3 pos, float t) {
            pos.xy *= rot(smoothstep(-1., -2., pos.y) * .2 * sin(t*2.));
            return pos;
          }

          vec3 breathAnimation(vec3 pos, float t) {
            float b = sin(t*1.5);
            pos.yz *= rot(smoothstep(-1., -3., pos.y) * .15 * -b);
            pos.z += .3;
            pos.y += 1.2;
            pos *= 1. + exp(-3. * length(pos)) * b;
            pos.z -= .3;
            pos.y -= 1.2;
            return pos;
          }

          vec4 fractal1(vec3 pos, float t, float intensity) {
            float m = 100.;
            vec3 p = pos * .1;
            p.y += .5;
            for (int i = 0; i < 8; i++) {
              p = abs(p) / clamp(abs(p.x * p.y), 0.3, 3.) - 1.;
              p.xy *= rot(radians(90.));
              if (i > 1) m = min(m, length(p.xy) + step(.3, fract(p.z * .5 + t * .5 + float(i) * .2)));
            }
            m = step(m, 0.5) * 1.3 * intensity;
            return vec4(-pos.y * .3, 0.5, 0.7, .3) * intensity + m;
          }

          vec4 fractal2(vec3 center, vec3 scales, vec4 rgba, float t, float intensity) {
            vec3 pos = center;
            float splatSize = length(scales);
            float pattern = exp(-50. * splatSize);
            vec3 p = pos * .65;
            pos.y += 2.;
            float c = 0.;
            float l, l2 = length(p);
            float m = 100.;
            
            for (int i = 0; i < 10; i++) {
              p.xyz = abs(p.xyz) / dot(p.xyz, p.xyz) - .8;
              l = length(p.xyz);
              c += exp(-1. * abs(l - l2) * (1. + sin(t * 1.5 + pos.y)));
              l2 = length(p.xyz);
              m = min(m, length(p.xyz));
            }
            
            c = smoothstep(0.3, 0.5, m + sin(t * 1.5 + pos.y * .5)) + c * .1;              
            return vec4(vec3(length(rgba.rgb)) * vec3(c, c*c, c*c*c) * intensity, 
                      rgba.a * exp(-20. * splatSize) * m * intensity);
          }

          vec4 sin3D(vec3 p, float t) {
            // Add noise to position for more variation
            vec3 noisePos = p + hash(p) * 0.5;
            
            // Multiple frequency layers for more interesting patterns
            float m1 = exp(-2. * length(sin(noisePos * 5. + t * 3.))) * 5.;
            float m2 = exp(-3. * length(sin(noisePos * 8. + t * 2. + hash(p).x * 6.28))) * 3.;
            float m3 = exp(-4. * length(sin(noisePos * 12. + t * 1.5 + hash(p).y * 6.28))) * 2.;
            
            float m = m1 + m2 * 0.5 + m3 * 0.3;
            return vec4(m) + .3;
          }

          vec4 disintegrate(vec3 pos, float t, float intensity) {
            vec3 p = pos + (hash(pos) * 2. - 1.) * intensity;
            float tt = smoothstep(-1., 0.5, -sin(t + -pos.y * .5));  
            p.xz *= rot(tt * 2. + p.y * 2. * tt);
            return vec4(mix(p, pos, tt), tt);
          }
          
          vec4 flare(vec3 pos, float t) {
            vec3 p = vec3(0., -1.5, 0.);
            float tt = smoothstep(-1., .5, sin(t + hash(pos).x));  
            tt = tt * tt;              
            p.x += sin(t * 2.) * tt;
            p.z += sin(t * 2.) * tt;
            p.y += sin(t) * tt;
            return vec4(mix(pos, p, tt), tt);
          }
        `),
      ],
      statements: ({ inputs, outputs }) =>
        dyno.unindentLines(`
          ${outputs.gsplat} = ${inputs.gsplat};
          
          vec3 localPos = ${inputs.gsplat}.center;
          vec3 splatScales = ${inputs.gsplat}.scales;
          vec4 splatColor = ${inputs.gsplat}.rgba;
          
          if (${inputs.effectType} == 1) {
            ${outputs.gsplat}.center = headMovement(localPos, ${inputs.t});
            vec4 effect1 = fractal1(localPos, ${inputs.t}, ${inputs.intensity});
            ${outputs.gsplat}.rgba.rgba = mix(splatColor, splatColor*effect1, ${inputs.intensity});
          } 
          else if (${inputs.effectType} == 2) {
            vec4 effectColor = fractal2(localPos, splatScales, splatColor, ${inputs.t}, ${inputs.intensity});
            ${outputs.gsplat}.rgba.rgba = mix(splatColor, effectColor, ${inputs.intensity});
            ${outputs.gsplat}.center = breathAnimation(localPos, ${inputs.t});
          } 
          else if (${inputs.effectType} == 3) {
            vec4 effect = sin3D(localPos, ${inputs.t});
            ${outputs.gsplat}.rgba.rgba = mix(splatColor, splatColor*effect, ${inputs.intensity});
            vec3 pos = localPos;
            pos.y += 1.;
            pos *= (1. + effect.x * .05 * ${inputs.intensity});
            pos.y -= 1.;
            ${outputs.gsplat}.center = pos;
          } 
          else if (${inputs.effectType} == 5) {
            vec4 e = disintegrate(localPos, ${inputs.t}, ${inputs.intensity});
            ${outputs.gsplat}.center = e.xyz;
            ${outputs.gsplat}.scales = mix(vec3(.01, .01, .01), ${inputs.gsplat}.scales, e.w);
          } 
          else if (${inputs.effectType} == 4) {
            vec4 e = flare(localPos, ${inputs.t});
            ${outputs.gsplat}.center = e.xyz;
            ${outputs.gsplat}.rgba.rgb = mix(splatColor.rgb, vec3(1.), abs(e.w));
            ${outputs.gsplat}.rgba.a = mix(splatColor.a, 0.3, abs(e.w));
          }
        `),
    });
  }

  /**
   * Get fractal modifier for a splat
   * @private
   */
  _getFractalModifier(effectType, intensity) {
    const dyn = this._createFractalDyno();
    return dyno.dynoBlock(
      { gsplat: dyno.Gsplat },
      { gsplat: dyno.Gsplat },
      ({ gsplat }) => ({
        gsplat: dyn.apply({
          gsplat,
          t: this.animateT,
          effectType: dyno.dynoInt(effectType),
          intensity: dyno.dynoFloat(intensity),
        }).gsplat,
      })
    );
  }

  /**
   * Try to load meshes if they're not already loaded but are needed
   * @private
   */
  async _tryLoadMeshes(targetIds) {
    if (!targetIds || !Array.isArray(targetIds) || targetIds.length === 0) return;

    // Ensure arrays are properly initialized
    if (!Array.isArray(this.targetMeshes)) {
      this.targetMeshes = [];
    }
    if (!Array.isArray(this.targetMeshIds)) {
      this.targetMeshIds = [];
    }

    for (const meshId of targetIds) {
      // Validate meshId
      if (typeof meshId !== 'string') {
        this.logger.warn(`Invalid meshId type in _tryLoadMeshes: ${typeof meshId}, skipping`, meshId);
        continue;
      }

      // Skip if already loaded
      if (this.targetMeshes.some((m, i) => this.targetMeshIds[i] === meshId)) {
        continue;
      }

      // Try to get mesh (might be available now)
      let mesh = this.sceneManager.getObject(meshId);
      
      if (!mesh && this.sceneManager.isLoading(meshId)) {
        // Mesh is loading, wait for it
        mesh = await this._waitForMesh(meshId);
      }

      if (mesh) {
        this.targetMeshes.push(mesh);
        const index = this.targetMeshIds.indexOf(meshId);
        if (index === -1) {
          this.targetMeshIds.push(meshId);
        }
        this.logger.log(`Loaded mesh: ${meshId}`);
      }
    }
  }

  /**
   * Hook: Apply effect parameters
   */
  applyEffect(effect, state) {
    const params = effect.parameters || {};

    // Update parameters from vfxData.js
    this.parameters = { ...params };

    const effectType = this.effectTypeMap[params.effectType] || 3; // Default to waves
    const targetIntensity =
      params.intensity !== undefined ? params.intensity : 0.8;
    const rampDuration =
      params.rampDuration !== undefined ? params.rampDuration : 0;
    const rampOutDuration =
      params.rampOutDuration !== undefined ? params.rampOutDuration : 0;

    this.logger.log(
      `Applying fractal effect: ${params.effectType} (type=${effectType}), target intensity=${targetIntensity}, ramp=${rampDuration}s, rampOut=${rampOutDuration}s`
    );

    // Update target mesh IDs if provided
    const targetIds = params.targetMeshIds || this.targetMeshIds || [];
    
    // Ensure arrays are properly initialized
    if (!Array.isArray(this.targetMeshes)) {
      this.targetMeshes = [];
    }
    if (!Array.isArray(this.targetMeshIds)) {
      this.targetMeshIds = [];
    }
    
    // Update targetMeshIds if provided (create a new array to avoid reference issues)
    if (Array.isArray(params.targetMeshIds) && params.targetMeshIds.length > 0) {
      this.targetMeshIds = [...params.targetMeshIds];
    } else if (Array.isArray(targetIds) && targetIds.length > 0 && this.targetMeshIds.length === 0) {
      this.targetMeshIds = [...targetIds];
    }

    // If we need meshes but don't have them, try to load them asynchronously
    if ((targetIntensity > 0 || rampDuration > 0) && this.targetMeshes.length === 0 && targetIds.length > 0) {
      this.logger.log("Meshes needed but not loaded - attempting to load asynchronously");
      // Fire and forget - meshes will be picked up when they become available
      this._tryLoadMeshes(targetIds).then(() => {
        // When meshes are loaded, apply the effect if still needed
        if (this.currentIntensity > 0 || this.isRamping) {
          const effectType = this.effectTypeMap[this.parameters.effectType] || 3;
          this.targetMeshes.forEach((mesh) => {
            mesh.objectModifier = this._getFractalModifier(
              effectType,
              this.currentIntensity
            );
            mesh.updateGenerator();
          });
          this.logger.log(`Applied effect to ${this.targetMeshes.length} mesh(es) after async load`);
        }
      }).catch((error) => {
        this.logger.error("Failed to load meshes asynchronously:", error);
      });
    }

    // Store ramp-out duration for when effect ends
    this.rampOutDuration = rampOutDuration;

    // Cancel any ongoing ramp-out
    this.isRampingOut = false;

    // Setup intensity ramping
    if (rampDuration > 0) {
      this.currentIntensity = 0;
      this.targetIntensity = targetIntensity;
      this.rampStartTime = this.animateT.value;
      this.rampDuration = rampDuration;
      this.isRamping = true;
      this.logger.log(
        `Starting intensity ramp from 0 to ${targetIntensity} over ${rampDuration}s`
      );
    } else {
      this.currentIntensity = targetIntensity;
      this.targetIntensity = targetIntensity;
      this.isRamping = false;
    }

    // Apply modifier to all target meshes (or remove if intensity is 0)
    if (this.currentIntensity <= 0) {
      // Remove modifiers when intensity is 0
      this.targetMeshes.forEach((mesh) => {
        if (mesh.objectModifier) {
          mesh.objectModifier = null;
          mesh.updateGenerator();
        }
      });
      this.logger.log(`Removed fractal modifiers (intensity is 0)`);
    } else {
      // Apply modifier with current intensity
      if (this.targetMeshes.length === 0) {
        this.logger.warn(
          "Cannot apply effect - no target meshes available. Effect will apply when meshes are loaded."
        );
      } else {
        this.targetMeshes.forEach((mesh) => {
          mesh.objectModifier = this._getFractalModifier(
            effectType,
            this.currentIntensity
          );
          mesh.updateGenerator();
        });
        this.logger.log(`Applied effect to ${this.targetMeshes.length} mesh(es)`);
      }
    }
  }

  /**
   * Hook: Called when no effect matches
   */
  onNoEffect(state) {
    // Stop audio
    if (this.audio && this.audio.isPlaying) {
      this.audio.stop();
    }

    // If we have a ramp-out duration, start ramping out
    if (this.rampOutDuration > 0 && this.currentIntensity > 0) {
      this.isRamping = false;
      this.isRampingOut = true;
      this.rampStartTime = this.animateT.value;
      this.rampOutStartIntensity = this.currentIntensity;
      this.targetIntensity = 0;
      this.logger.log(
        `Starting ramp-out from ${this.currentIntensity} to 0 over ${this.rampOutDuration}s`
      );
    } else {
      // No ramp-out, remove modifiers immediately
      this.targetMeshes.forEach((mesh) => {
        if (mesh.objectModifier) {
          mesh.objectModifier = null;
          mesh.updateGenerator();
        }
      });
      this.logger.log("No effect - removed fractal modifiers immediately");
    }
  }

  /**
   * Update animation
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    if (this.targetMeshes.length === 0) return;

    // Update animation time
    this.animateT.value += dt;

    // Handle intensity ramping (ramp in)
    if (this.isRamping) {
      const elapsed = this.animateT.value - this.rampStartTime;
      const progress = Math.min(elapsed / this.rampDuration, 1.0);

      // Smooth easing function
      const easedProgress = progress * progress * (3 - 2 * progress);

      this.currentIntensity = easedProgress * this.targetIntensity;

      if (progress >= 1.0) {
        this.isRamping = false;
        this.currentIntensity = this.targetIntensity;
        this.logger.log(`Intensity ramp complete at ${this.targetIntensity}`);

        // Fade out audio as we reach peak - morph audio will take over
        if (this.audio && this.audio.isPlaying) {
          this.audio.stop();
        }
      }

      // Update modifiers with new intensity
      const effectType = this.effectTypeMap[this.parameters.effectType] || 3;
      this.targetMeshes.forEach((mesh) => {
        mesh.objectModifier = this._getFractalModifier(
          effectType,
          this.currentIntensity
        );
        mesh.updateGenerator();
      });

      // Sync audio with visual intensity ramp
      this._updateAudio(easedProgress);
    }

    // Handle intensity ramping out
    if (this.isRampingOut) {
      const elapsed = this.animateT.value - this.rampStartTime;
      const progress = Math.min(elapsed / this.rampOutDuration, 1.0);

      // Smooth easing function (same as ramp in)
      const easedProgress = progress * progress * (3 - 2 * progress);

      // Ramp from start intensity to 0
      this.currentIntensity = this.rampOutStartIntensity * (1.0 - easedProgress);

      // Update modifiers with new intensity
      const effectType = this.effectTypeMap[this.parameters.effectType] || 3;
      this.targetMeshes.forEach((mesh) => {
        mesh.objectModifier = this._getFractalModifier(
          effectType,
          this.currentIntensity
        );
        mesh.updateGenerator();
      });

      // When ramp-out complete, remove modifiers
      if (progress >= 1.0) {
        this.isRampingOut = false;
        this.currentIntensity = 0;
        this.targetMeshes.forEach((mesh) => {
          if (mesh.objectModifier) {
            mesh.objectModifier = null;
            mesh.updateGenerator();
          }
        });
        this.logger.log("Ramp-out complete - removed fractal modifiers");
      }
    }

    // Update all target meshes
    this.targetMeshes.forEach((mesh) => {
      mesh.updateVersion();
    });
  }

  setExternalIntensity(intensity = 0) {
    const clamped = Math.max(0, intensity); // Only clamp minimum, allow high intensities
    this.isRamping = false;
    this.isRampingOut = false;
    this.currentIntensity = clamped;
    this.targetIntensity = clamped;

    const effectType = this.effectTypeMap[this.parameters.effectType] || 3;
    this.targetMeshes.forEach((mesh) => {
      mesh.objectModifier = this._getFractalModifier(effectType, clamped);
      mesh.updateGenerator();
    });
  }

  /**
   * Update audio parameters based on intensity ramp
   * @private
   */
  _updateAudio(progress) {
    if (!this.audio || !this.audio.isPlaying) return;

    // Map progress (0-1) to audio parameters
    // Build tension as intensity increases

    // Filter frequency: start low (800 Hz), rise to bright (2500 Hz)
    const minFreq = 800;
    const maxFreq = 2500;
    const targetFilterFreq = minFreq + (maxFreq - minFreq) * progress;

    // Filter Q (resonance): increase for more tension
    const minQ = 2;
    const maxQ = 8;
    const targetQ = minQ + (maxQ - minQ) * progress;

    // Pitch rises slightly as tension builds
    const pitchMultiplier = 1.0 + progress * 0.3; // Up to 30% higher

    // Volume follows intensity closely
    const maxVolume = this.audio.config.volume;
    const targetVolume = maxVolume * Math.sqrt(progress); // Square root for quicker perception

    // Delay feedback increases for more chaos
    const minFeedback = 0.3;
    const maxFeedback = 0.5;
    const targetFeedback = minFeedback + (maxFeedback - minFeedback) * progress;

    // Pan sweeps as intensity builds
    const panAmount = Math.sin(progress * Math.PI * 2) * progress * 0.6;

    this.audio.updateParams({
      filterFreq: targetFilterFreq,
      filterQ: targetQ,
      pitchMultiplier,
      volume: targetVolume,
      delayFeedback: targetFeedback,
      pan: panAmount,
      transitionTime: 0.1,
    });
  }

  /**
   * Cleanup
   */
  dispose() {
    this.logger.log("Disposing fractal effect");

    // Clean up audio
    if (this.audio) {
      this.audio.dispose();
      this.audio = null;
    }

    // Remove modifiers from all meshes
    this.targetMeshes.forEach((mesh) => {
      if (mesh.objectModifier) {
        mesh.objectModifier = null;
        mesh.updateGenerator();
      }
    });

    this.targetMeshes = [];
    this.targetMeshIds = [];
  }
}

export default SplatFractalEffect;
