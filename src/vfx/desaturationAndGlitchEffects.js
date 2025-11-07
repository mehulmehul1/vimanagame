import * as THREE from "three";
import { FullScreenQuad } from "three/addons/postprocessing/Pass.js";
import { VFXManager } from "../vfxManager.js";
import { ProceduralAudio } from "./proceduralAudio.js";

/**
 * DesaturationAndGlitchEffects - Combined post-processing shader
 * Applies both desaturation and glitch effects in a single pass for better performance
 *
 * Maintains separate controls, triggers, and timing for both effects
 */
export class DesaturationAndGlitchEffects extends VFXManager {
  constructor(renderer) {
    super("DesaturationAndGlitchEffects", false);

    this.renderer = renderer;
    this.enabled = false;
    this._vfxDataModule = null; // Cache for vfxData module
    this._currentDesatEffectId = null;
    this._currentGlitchEffectId = null;
    this._pendingDesatEffect = null;
    this._pendingGlitchEffect = null;
    this._desatDelayTimeout = null;
    this._glitchDelayTimeout = null;

    // Desaturation properties
    this.desatEnabled = false;
    this.desatProgress = 0; // 0 = color, 1 = grayscale
    this.desatAnimating = false;
    this.desatAnimationTarget = 0;
    this.desatAnimationDuration = 5.0;
    this.desatAnimationSpeed = 0;
    this.desatCurrentState = 0;
    this.desatTransitionMode = "bleed"; // 'fade', 'bleed', or 'wipe'
    this.desatBleedScale = 3;
    this.desatBleedSoftness = 0.2;
    this.desatWipeDirection = "bottom-to-top";
    this.desatWipeSoftness = 0.15;
    this.desatEnableAudio = false;
    this.desatLastProgress = 0;

    // Glitch properties
    this.glitchEnabled = false;
    this.glitchIntensity = 0.08;
    this.glitchGoWild = false;
    this.glitchEnableAudio = false;
    this.glitchCurF = 0;
    this.glitchRandX = 120;

    // Desaturation procedural audio
    this.desatAudio = new ProceduralAudio({
      name: "DesaturationAudio",
      baseFrequency: 155.56,
      volume: 0.1,
      baseOscType: "sine",
      subOscType: "sine",
      modOscType: "triangle",
      filterType: "lowpass",
      filterFreq: 2000,
      filterQ: 1.2,
      distortionAmount: 3,
      delayTime: 0.25,
      delayFeedback: 0.2,
      lfoFreq: 0.4,
      lfoDepth: 15,
      fadeInTime: 0,
      fadeOutTime: 1.0,
      fadeInCurve: "exponential",
      fadeOutCurve: "exponential",
      enableSweep: true,
      sweepBaseFreq: 3500,
      sweepRange: 1200,
      sweepRate: 3.0,
      sweepGain: 0.2,
    });

    // Glitch procedural audio
    this.glitchAudio = new ProceduralAudio({
      name: "GlitchAudio",
      baseFrequency: 800.0,
      volume: 0.3,
      baseOscType: "square",
      subOscType: "sawtooth",
      modOscType: "sawtooth",
      filterType: "bandpass",
      filterFreq: 5000,
      filterQ: 5,
      distortionAmount: 50,
      delayTime: 0.02,
      delayFeedback: 0.8,
      lfoFreq: 20.0,
      lfoDepth: 2000,
      fadeInTime: 0.05,
      fadeOutTime: 0.1,
      fadeInCurve: "linear",
      fadeOutCurve: "exponential",
      enableSweep: true,
      sweepBaseFreq: 6000,
      sweepRange: 4000,
      sweepRate: 25.0,
      sweepGain: 0.4,
    });

    // Generate displacement texture for glitch
    this._heightMap = this._generateHeightmap(64);

    // Combined shader uniforms
    this.shader = {
      uniforms: {
        tDiffuse: { value: null },
        tDisp: { value: this._heightMap },
        // Desaturation uniforms
        desatAmount: { value: 0.0 },
        animProgress: { value: 0.0 },
        transitionMode: { value: 1.0 },
        bleedScale: { value: this.desatBleedScale },
        bleedSoftness: { value: this.desatBleedSoftness },
        wipeDirection: { value: 0.0 },
        wipeSoftness: { value: this.desatWipeSoftness },
        // Glitch uniforms
        glitchSeed: { value: 0.0 },
        glitchByp: { value: 1 },
        glitchAmount: { value: 0.0 },
        glitchAngle: { value: 0.0 },
        glitchSeedX: { value: 0.0 },
        glitchSeedY: { value: 0.0 },
        glitchDistortionX: { value: 0.5 },
        glitchDistortionY: { value: 0.6 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform sampler2D tDisp;
        uniform float desatAmount;
        uniform float animProgress;
        uniform float transitionMode;
        uniform float bleedScale;
        uniform float bleedSoftness;
        uniform float wipeDirection;
        uniform float wipeSoftness;
        uniform float glitchSeed;
        uniform int glitchByp;
        uniform float glitchAmount;
        uniform float glitchAngle;
        uniform float glitchSeedX;
        uniform float glitchSeedY;
        uniform float glitchDistortionX;
        uniform float glitchDistortionY;
        varying vec2 vUv;
        
        // Constants
        const vec3 LUMA_WEIGHTS = vec3(0.299, 0.587, 0.114);
        const int NOISE_OCTAVES = 4;
        const float NOISE_INITIAL_AMPLITUDE = 0.5;
        const float NOISE_FREQUENCY_MULT = 2.0;
        const float NOISE_AMPLITUDE_MULT = 0.5;
        
        // Noise functions for desaturation
        float noise(vec2 st) {
          return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
        }
        
        float smoothNoise(vec2 st) {
          vec2 i = floor(st);
          vec2 f = fract(st);
          vec2 u = f * f * (3.0 - 2.0 * f);
          float a = noise(i);
          float b = noise(i + vec2(1.0, 0.0));
          float c = noise(i + vec2(0.0, 1.0));
          float d = noise(i + vec2(1.0, 1.0));
          return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
        }
        
        float fractalNoise(vec2 st) {
          float value = 0.0;
          float amplitude = NOISE_INITIAL_AMPLITUDE;
          float frequency = 1.0;
          for(int i = 0; i < NOISE_OCTAVES; i++) {
            value += amplitude * smoothNoise(st * frequency);
            frequency *= NOISE_FREQUENCY_MULT;
            amplitude *= NOISE_AMPLITUDE_MULT;
          }
          return value;
        }
        
        // Glitch rand function
        float rand(vec2 co) {
          return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
        }
        
        void main() {
          vec2 p = vUv;
          
          // Apply glitch distortion first (if enabled)
          if (glitchByp == 0) {
            vec4 normal = texture2D(tDisp, p * glitchSeed * glitchSeed);
            p.x += normal.x * glitchDistortionX;
            p.y += normal.y * glitchDistortionY;
            vec2 offset = glitchAmount * vec2(cos(glitchAngle), sin(glitchAngle));
            p += offset;
          }
          
          // Sample color (potentially distorted by glitch)
          vec4 color = texture2D(tDiffuse, p);
          
          // Apply desaturation
          float gray = dot(color.rgb, LUMA_WEIGHTS);
          float desatAmountFinal = desatAmount;
          
          // Mode 2: Wipe transition
          if (transitionMode > 1.5) {
            float pixelY = vUv.y;
            if (wipeDirection > 0.5) {
              pixelY = 1.0 - pixelY;
            }
            float wipeEdge = animProgress * (1.0 + wipeSoftness);
            float wipeMask = 1.0 - smoothstep(
              wipeEdge - wipeSoftness,
              wipeEdge,
              pixelY
            );
            desatAmountFinal = wipeMask;
          }
          // Mode 1: Bleed transition
          else if (transitionMode > 0.5) {
            float noiseValue = fractalNoise(vUv * bleedScale);
            float edgeMargin = bleedSoftness * 0.5;
            float pixelThreshold = edgeMargin + noiseValue * (1.0 - bleedSoftness);
            float bleedMask = smoothstep(
              pixelThreshold - edgeMargin,
              pixelThreshold + edgeMargin,
              animProgress
            );
            desatAmountFinal = bleedMask;
          }
          
          // Lerp between color and grayscale
          vec3 result = mix(color.rgb, vec3(gray), desatAmountFinal);
          
          // Apply glitch color shifts (if enabled)
          if (glitchByp == 0) {
            vec2 block = vec2(floor(p.x * 10.0) / 10.0, floor(p.y * 10.0) / 10.0);
            result.r += rand(block) * glitchSeedX;
            result.g += rand(block + 1.0) * glitchSeedX;
            result.b += rand(block + 2.0) * glitchSeedX;
            result += mix(-glitchAmount, glitchAmount, rand(block + glitchSeed)) * glitchSeedY;
          }
          
          gl_FragColor = vec4(result, color.a);
        }
      `,
    };

    // Create render target
    const size = this.renderer.getSize(new THREE.Vector2());
    this.renderTarget = new THREE.WebGLRenderTarget(size.x, size.y, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    });

    // Create post-processing material
    this.postMaterial = new THREE.ShaderMaterial(this.shader);
    this.fullScreenQuad = new FullScreenQuad(this.postMaterial);
  }

  /**
   * Generate displacement texture for glitch effect
   * @private
   */
  _generateHeightmap(dt_size) {
    const data_arr = new Float32Array(dt_size * dt_size);
    const length = dt_size * dt_size;

    for (let i = 0; i < length; i++) {
      const val = THREE.MathUtils.randFloat(0, 1);
      data_arr[i] = val;
    }

    const texture = new THREE.DataTexture(
      data_arr,
      dt_size,
      dt_size,
      THREE.RedFormat,
      THREE.FloatType
    );
    texture.needsUpdate = true;
    return texture;
  }

  /**
   * Generate random trigger interval for glitch
   * @private
   */
  _generateGlitchTrigger() {
    this.glitchRandX = THREE.MathUtils.randInt(120, 240);
  }

  /**
   * Enable/disable the combined effect
   */
  enable(enabled = true) {
    this.enabled = enabled;
  }

  disable() {
    this.enabled = false;
  }

  /**
   * Override: Called when first effect matches
   * Handles both desaturation and glitch effect types
   */
  async onFirstEnable(effect, state) {
    const effectType = this._getEffectType(effect);
    
    if (effectType === "desaturation") {
      this.logger.log("Enabling desaturation effect for first time");
      const params = effect.parameters || {};
      this.desatEnableAudio = params.enableAudio === true;
      
      if (!this.desatEnableAudio && this.desatAudio && this.desatAudio.isPlaying) {
        this.desatAudio.stop();
      }
      
      this.desatEnabled = true;
      this.enable(true);
      
      if (this.desatEnableAudio) {
        await this.desatAudio.initialize();
        if (this.desatCurrentState < 0.5 && !this.desatAudio.isPlaying) {
          this.desatAudio.start();
        }
      }
    } else if (effectType === "glitch") {
      this.logger.log("Enabling glitch effect for first time");
      const params = effect.parameters || {};
      this.glitchEnableAudio = params.enableAudio === true;
      
      if (!this.glitchEnableAudio && this.glitchAudio && this.glitchAudio.isPlaying) {
        this.glitchAudio.stop();
      }
      
      this.glitchEnabled = true;
      this.enable(true);
      
      if (this.glitchEnableAudio) {
        await this.glitchAudio.initialize();
      }
    }
    
    this.applyEffect(effect, state);
  }

  /**
   * Override: Handle when no effect matches
   * Only disable if both effects are inactive
   */
  onNoEffect(state) {
    // Individual effects handle their own disable logic in applyEffect
    // This is called per-effect-type, so we don't need to disable here
  }

  /**
   * Override: Set game manager - cache vfxData module
   */
  setGameManager(gameManager, vfxTypeOrGetter) {
    // Import and cache vfxData module
    import("../vfxData.js").then((module) => {
      this._vfxDataModule = module;
    });
    
    // Call parent setGameManager
    super.setGameManager(gameManager, vfxTypeOrGetter);
  }

  /**
   * Override: Update for state - check both effect types independently
   */
  updateForState(state) {
    if (!state) return;

    // Wait for vfxData module to load
    if (!this._vfxDataModule) {
      import("../vfxData.js").then((module) => {
        this._vfxDataModule = module;
        this._updateBothEffects(state);
      });
      return;
    }

    this._updateBothEffects(state);
  }

  /**
   * Update both desaturation and glitch effects
   * @private
   */
  _updateBothEffects(state) {
    const desatEffect = this._vfxDataModule.getVfxEffectForState("desaturation", state);
    const glitchEffect = this._vfxDataModule.getVfxEffectForState("glitch", state);

    // Handle desaturation effect
    if (desatEffect) {
      const isNewDesatEffect = this._currentDesatEffectId !== desatEffect.id;
      const isPendingDesatEffect = this._pendingDesatEffect && this._pendingDesatEffect.id === desatEffect.id;
      const isFirstDesatEnable = !this.desatEnabled;
      
      // Skip if this is the same effect already applied or pending
      if (!isNewDesatEffect && !isPendingDesatEffect && !isFirstDesatEnable) {
        // Effect already active, do nothing
      } else {
        // Cancel any pending delayed effect if switching to a different effect
        if (this._pendingDesatEffect && this._pendingDesatEffect.id !== desatEffect.id) {
          this._cancelDesatDelay();
        }
        
        // Handle first enable
        if (isFirstDesatEnable) {
          this.desatEnabled = true;
          this._hasEverBeenEnabled = true;
          this.onFirstEnable(desatEffect, state);
        }
        
        // Check for delay
        const delay = desatEffect.delay || 0;
        
        if (delay > 0 && !isPendingDesatEffect) {
          // Effect should be delayed
          this.logger.log(`Desaturation effect ${desatEffect.id} will be applied in ${delay} seconds`);
          this._pendingDesatEffect = desatEffect;
          
          this._desatDelayTimeout = setTimeout(() => {
            this.logger.log(`Applying delayed desaturation effect: ${desatEffect.id}`);
            this._currentDesatEffectId = desatEffect.id;
            this._pendingDesatEffect = null;
            this._desatDelayTimeout = null;
            this.applyEffect(desatEffect, state);
          }, delay * 1000);
        } else if (!isPendingDesatEffect) {
          // No delay, apply immediately
          if (isNewDesatEffect) {
            this._currentDesatEffectId = desatEffect.id;
            this.logger.log(`Applying desaturation effect: ${desatEffect.id}`);
            this.applyEffect(desatEffect, state);
          }
        }
      }
    } else {
      // No desaturation effect - cancel delay and disable desaturation
      this._cancelDesatDelay();
      if (this.desatEnabled) {
        this.desatEnabled = false;
        this._currentDesatEffectId = null;
        if (this.desatEnableAudio && this.desatAudio && this.desatAudio.isPlaying) {
          this.desatAudio.stop();
        }
      }
    }

    // Handle glitch effect
    if (glitchEffect) {
      const isNewGlitchEffect = this._currentGlitchEffectId !== glitchEffect.id;
      const isPendingGlitchEffect = this._pendingGlitchEffect && this._pendingGlitchEffect.id === glitchEffect.id;
      const isFirstGlitchEnable = !this.glitchEnabled;
      
      // Skip if this is the same effect already applied or pending
      if (!isNewGlitchEffect && !isPendingGlitchEffect && !isFirstGlitchEnable) {
        // Effect already active, do nothing
      } else {
        // Cancel any pending delayed effect if switching to a different effect
        if (this._pendingGlitchEffect && this._pendingGlitchEffect.id !== glitchEffect.id) {
          this._cancelGlitchDelay();
        }
        
        // Handle first enable
        if (isFirstGlitchEnable) {
          this.glitchEnabled = true;
          this._hasEverBeenEnabled = true;
          this.onFirstEnable(glitchEffect, state);
        }
        
        // Check for delay
        const delay = glitchEffect.delay || 0;
        
        if (delay > 0 && !isPendingGlitchEffect) {
          // Effect should be delayed
          this.logger.log(`Glitch effect ${glitchEffect.id} will be applied in ${delay} seconds`);
          this._pendingGlitchEffect = glitchEffect;
          
          this._glitchDelayTimeout = setTimeout(() => {
            this.logger.log(`Applying delayed glitch effect: ${glitchEffect.id}`);
            this._currentGlitchEffectId = glitchEffect.id;
            this._pendingGlitchEffect = null;
            this._glitchDelayTimeout = null;
            this.applyEffect(glitchEffect, state);
          }, delay * 1000);
        } else if (!isPendingGlitchEffect) {
          // No delay, apply immediately
          if (isNewGlitchEffect) {
            this._currentGlitchEffectId = glitchEffect.id;
            this.logger.log(`Applying glitch effect: ${glitchEffect.id}`);
            this.applyEffect(glitchEffect, state);
          }
        }
      }
    } else {
      // No glitch effect - cancel delay and disable glitch
      this._cancelGlitchDelay();
      if (this.glitchEnabled) {
        this.glitchEnabled = false;
        this._currentGlitchEffectId = null;
        this.setGlitchIntensity(0.0);
        if (this.glitchEnableAudio && this.glitchAudio && this.glitchAudio.isPlaying) {
          this.glitchAudio.stop();
        }
      }
    }

    // Enable combined effect if either sub-effect is active
    this.enabled = this.desatEnabled || this.glitchEnabled;
  }

  /**
   * Cancel pending desaturation delay
   * @private
   */
  _cancelDesatDelay() {
    if (this._desatDelayTimeout) {
      clearTimeout(this._desatDelayTimeout);
      this._desatDelayTimeout = null;
      if (this._pendingDesatEffect) {
        this.logger.log(`Cancelled delayed desaturation effect: ${this._pendingDesatEffect.id}`);
      }
      this._pendingDesatEffect = null;
    }
  }

  /**
   * Cancel pending glitch delay
   * @private
   */
  _cancelGlitchDelay() {
    if (this._glitchDelayTimeout) {
      clearTimeout(this._glitchDelayTimeout);
      this._glitchDelayTimeout = null;
      if (this._pendingGlitchEffect) {
        this.logger.log(`Cancelled delayed glitch effect: ${this._pendingGlitchEffect.id}`);
      }
      this._pendingGlitchEffect = null;
    }
  }

  /**
   * Override: Apply effect from game state
   * Handles both desaturation and glitch effect types
   */
  applyEffect(effect, state) {
    const effectType = this._getEffectType(effect);
    const params = effect.parameters || {};

    if (effectType === "desaturation") {
      this.desatEnableAudio = params.enableAudio === true;
      
      if (!this.desatEnableAudio && this.desatAudio && this.desatAudio.isPlaying) {
        this.desatAudio.stop();
      }
      if (this.desatEnableAudio && !this.desatAudio.audioContext) {
        this.desatAudio.initialize().catch((err) => {
          this.logger.warn(`Failed to initialize desaturation audio: ${err}`);
        });
      }

      if (state?.currentState === 32) {
        this.logger.log(
          `[DEBUG] Desaturation applyEffect: ${effect.id}, target: ${params.target}, isViewmasterEquipped: ${state?.isViewmasterEquipped}`
        );
      }

      if (params.duration !== undefined) {
        this.desatAnimationDuration = params.duration;
      }

      const options = {
        mode: params.mode || "bleed",
        direction: params.direction || params.wipeDirection || "bottom-to-top",
        softness: params.softness !== undefined ? params.softness : params.wipeSoftness,
        suppressAudio: params.suppressAudio || false,
      };

      this.desatAnimateTo(params.target, options);
    } else if (effectType === "glitch") {
      this.glitchEnableAudio = params.enableAudio === true;
      
      if (!this.glitchEnableAudio && this.glitchAudio && this.glitchAudio.isPlaying) {
        this.glitchAudio.stop();
      }
      if (this.glitchEnableAudio && !this.glitchAudio.audioContext) {
        this.glitchAudio.initialize().catch((err) => {
          this.logger.warn(`Failed to initialize glitch audio: ${err}`);
        });
      }

      if (params.intensity !== undefined) {
        this.setGlitchIntensity(params.intensity);
      }

      if (params.goWild !== undefined) {
        this.setGlitchWild(params.goWild);
      }
    }
  }

  /**
   * Override: Handle when no effect matches for a specific type
   * This is called per-effect-type, so we handle each separately
   */
  onNoEffect(state) {
    // Don't disable the combined effect here - individual effects handle their own state
    // The effect stays enabled as long as at least one sub-effect is active
  }


  /**
   * Determine effect type from effect data
   * @private
   */
  _getEffectType(effect) {
    // Check if it's a desaturation effect by looking for desaturation-specific parameters
    if (effect.parameters && (effect.parameters.target !== undefined || effect.parameters.mode !== undefined)) {
      return "desaturation";
    }
    // Check if it's a glitch effect by looking for glitch-specific parameters
    if (effect.parameters && (effect.parameters.intensity !== undefined || effect.parameters.goWild !== undefined)) {
      return "glitch";
    }
    // Default based on effect ID pattern
    if (effect.id && effect.id.toLowerCase().includes("glitch")) {
      return "glitch";
    }
    return "desaturation";
  }

  // Desaturation methods
  setDesatAmount(amount) {
    this.desatProgress = THREE.MathUtils.clamp(amount, 0, 1);
    this.desatCurrentState = this.desatProgress;
    this.desatAnimating = false;

    if (this.desatEnableAudio) {
      if (this.desatCurrentState < 0.5 && !this.desatAudio.isPlaying) {
        this.desatAudio.start();
      } else if (this.desatCurrentState >= 0.5 && this.desatAudio.isPlaying) {
        this.desatAudio.stop();
      }
    }
  }

  desatAnimateTo(targetAmount, options = {}) {
    if (!this.enabled) {
      this.logger.warn("Effect is disabled. Call enable() first.");
      return;
    }

    this.desatAnimationTarget = THREE.MathUtils.clamp(targetAmount, 0, 1);

    if (this.desatAnimationDuration <= 0) {
      this.desatProgress = this.desatAnimationTarget;
      this.desatCurrentState = this.desatAnimationTarget;
      this.desatAnimating = false;
      this.desatLastProgress = this.desatProgress;

      const mode = options.mode || "bleed";
      this.desatTransitionMode = mode;
      this._updateDesatShaderMode(mode, options);
      return;
    }

    this.desatAnimationSpeed = 1.0 / this.desatAnimationDuration;
    this.desatAnimating = true;

    if (this.desatEnableAudio && !options.suppressAudio) {
      if (targetAmount < 0.5 && !this.desatAudio.isPlaying) {
        this.desatAudio.start();
      }
      if (this.desatCurrentState < 0.5 && !this.desatAudio.isPlaying) {
        this.desatAudio.start();
      }
    }

    this.desatLastProgress = this.desatProgress;

    const mode = options.mode || "bleed";
    this.desatTransitionMode = mode;
    this._updateDesatShaderMode(mode, options);
  }

  _updateDesatShaderMode(mode, options) {
    if (mode === "wipe") {
      this.postMaterial.uniforms.transitionMode.value = 2.0;
      const direction = options.direction || this.desatWipeDirection;
      this.desatWipeDirection = direction;
      this.postMaterial.uniforms.wipeDirection.value = direction === "top-to-bottom" ? 1.0 : 0.0;
      const softness = options.softness !== undefined ? options.softness : this.desatWipeSoftness;
      this.desatWipeSoftness = softness;
      this.postMaterial.uniforms.wipeSoftness.value = softness;
    } else if (mode === "bleed") {
      this.postMaterial.uniforms.transitionMode.value = 1.0;
      this.postMaterial.uniforms.bleedScale.value = this.desatBleedScale;
      this.postMaterial.uniforms.bleedSoftness.value = this.desatBleedSoftness;
    } else {
      this.postMaterial.uniforms.transitionMode.value = 0.0;
    }
  }

  // Glitch methods
  setGlitchIntensity(intensity) {
    this.glitchIntensity = THREE.MathUtils.clamp(intensity, 0.0, 1.0);
  }

  setGlitchWild(wild = false) {
    this.glitchGoWild = wild;
  }

  /**
   * Update animation (call each frame)
   */
  update(deltaTime) {
    // Update desaturation animation
    if (this.desatAnimating) {
      const distance = this.desatAnimationTarget - this.desatProgress;
      const step = Math.sign(distance) * this.desatAnimationSpeed * deltaTime;

      if (Math.abs(distance) <= Math.abs(step)) {
        this.desatProgress = this.desatAnimationTarget;
        this.desatAnimating = false;
        this.desatCurrentState = this.desatAnimationTarget;

        if (this.desatEnableAudio) {
          if (this.desatAnimationTarget >= 0.5 && this.desatAudio.isPlaying) {
            this.desatAudio.stop();
          } else if (this.desatAnimationTarget < 0.5 && !this.desatAudio.isPlaying) {
            this.desatAudio.start();
          }
        }
      } else {
        this.desatProgress += step;
        this.desatCurrentState = this.desatProgress;

        if (this.desatEnableAudio) {
          this._updateDesatAudio(deltaTime);
        }
      }
    } else if (this.desatEnabled && this.desatEnableAudio) {
      if (this.desatCurrentState < 0.5 && !this.desatAudio.isPlaying) {
        this.desatAudio.start();
      } else if (this.desatCurrentState >= 0.5 && this.desatAudio.isPlaying) {
        this.desatAudio.stop();
      }
    }

    // Update glitch animation
    if (!this.enabled || !this._hasEverBeenEnabled) {
      return;
    }

    if (this.glitchIntensity <= 0.001) {
      this.postMaterial.uniforms.glitchByp.value = 1;
      this.postMaterial.uniforms.glitchAmount.value = 0.0;
      this.postMaterial.uniforms.glitchSeedX.value = 0.0;
      this.postMaterial.uniforms.glitchSeedY.value = 0.0;
      
      if (this.glitchEnableAudio && this.glitchAudio && this.glitchAudio.isPlaying) {
        this.glitchAudio.stop();
      }
      return;
    }

    if (this.glitchEnableAudio && this.glitchAudio && !this.glitchAudio.isPlaying && this.glitchAudio.audioContext) {
      this.glitchAudio.start();
    }

    if (this.glitchEnableAudio && this.glitchAudio && this.glitchAudio.isPlaying) {
      const audioVolume = 0.3 * this.glitchIntensity;
      const sweepRate = this.glitchGoWild ? 40.0 : 25.0;
      const lfoFreq = this.glitchGoWild ? 30.0 : 20.0;
      const filterFreq = this.glitchGoWild ? 7000 : 5000;
      
      this.glitchAudio.updateParams({
        volume: audioVolume,
        sweepRate: sweepRate,
        lfoFreq: lfoFreq,
        filterFreq: filterFreq,
        transitionTime: 0.05,
      });
    }

    this.postMaterial.uniforms.glitchSeed.value = Math.random();
    this.postMaterial.uniforms.glitchByp.value = 0;

    if (this.glitchCurF % this.glitchRandX === 0 || this.glitchGoWild) {
      this.postMaterial.uniforms.glitchAmount.value = Math.random() / 30;
      this.postMaterial.uniforms.glitchAngle.value = THREE.MathUtils.randFloat(-Math.PI, Math.PI);
      this.postMaterial.uniforms.glitchSeedX.value = THREE.MathUtils.randFloat(-1, 1);
      this.postMaterial.uniforms.glitchSeedY.value = THREE.MathUtils.randFloat(-1, 1);
      this.postMaterial.uniforms.glitchDistortionX.value = THREE.MathUtils.randFloat(0, 1);
      this.postMaterial.uniforms.glitchDistortionY.value = THREE.MathUtils.randFloat(0, 1);
      this.glitchCurF = 0;
      this._generateGlitchTrigger();
    } else if (this.glitchCurF % this.glitchRandX < this.glitchRandX / 5) {
      this.postMaterial.uniforms.glitchAmount.value = Math.random() / 90;
      this.postMaterial.uniforms.glitchAngle.value = THREE.MathUtils.randFloat(-Math.PI, Math.PI);
      this.postMaterial.uniforms.glitchDistortionX.value = THREE.MathUtils.randFloat(0, 1);
      this.postMaterial.uniforms.glitchDistortionY.value = THREE.MathUtils.randFloat(0, 1);
      this.postMaterial.uniforms.glitchSeedX.value = THREE.MathUtils.randFloat(-0.3, 0.3);
      this.postMaterial.uniforms.glitchSeedY.value = THREE.MathUtils.randFloat(-0.3, 0.3);
    } else if (!this.glitchGoWild) {
      this.postMaterial.uniforms.glitchByp.value = 1;
    }

    this.postMaterial.uniforms.glitchAmount.value *= this.glitchIntensity;
    this.postMaterial.uniforms.glitchSeedX.value *= this.glitchIntensity;
    this.postMaterial.uniforms.glitchSeedY.value *= this.glitchIntensity;

    this.glitchCurF++;

    // Update shader uniforms
    this.postMaterial.uniforms.desatAmount.value = this.desatCurrentState;
    this.postMaterial.uniforms.animProgress.value = this.desatProgress;
  }

  /**
   * Update desaturation audio parameters
   * @private
   */
  _updateDesatAudio(deltaTime) {
    const velocity = Math.abs(this.desatProgress - this.desatLastProgress) / Math.max(deltaTime, 0.001);
    const normalizedVelocity = Math.min(velocity * 2, 1);

    const minFreq = 300;
    const maxFreq = 4000;
    const targetFilterFreq = maxFreq - (maxFreq - minFreq) * this.desatProgress;

    const minQ = 2;
    const maxQ = 10;
    const targetQ = minQ + (maxQ - minQ) * normalizedVelocity;

    const pitchMultiplier = 1.0 + normalizedVelocity * 0.2;
    const panAmount = Math.sin(this.desatProgress * Math.PI) * 0.5;

    const colorAmount = 1.0 - this.desatProgress;
    const minVolume = 0.0;
    const maxVolume = this.desatAudio.config.volume;
    const baseCurve = colorAmount * colorAmount;
    let baseVolume = minVolume + (maxVolume - minVolume) * baseCurve;
    const velocityBoost = normalizedVelocity * maxVolume * 0.8;
    const targetVolume = Math.min(baseVolume + velocityBoost, maxVolume * 1.2);

    const desatAmount = this.desatProgress;
    const sweepAmount = desatAmount * desatAmount;

    this.desatAudio.updateParams({
      filterFreq: targetFilterFreq,
      filterQ: targetQ,
      pitchMultiplier,
      pan: panAmount,
      volume: targetVolume,
      sweepAmount: sweepAmount,
      transitionTime: 0.1,
    });

    this.desatLastProgress = this.desatProgress;
  }

  /**
   * Resize the render target
   */
  setSize(width, height) {
    this.renderTarget.setSize(width, height);
    if (this._intermediateTarget) {
      this._intermediateTarget.setSize(width, height);
    }
  }

  /**
   * Render the scene with combined effects
   */
  render(scene, camera) {
    if (this.enabled && this._hasEverBeenEnabled) {
      this.renderer.setRenderTarget(this.renderTarget);
      this.renderer.render(scene, camera);

      this.postMaterial.uniforms.tDiffuse.value = this.renderTarget.texture;
      this.postMaterial.uniforms.desatAmount.value = this.desatCurrentState;
      this.postMaterial.uniforms.animProgress.value = this.desatProgress;

      this.renderer.setRenderTarget(null);
      this.fullScreenQuad.render(this.renderer);
    } else {
      this.renderer.render(scene, camera);
    }
  }

  /**
   * Dispose of resources
   */
  dispose() {
    this._cancelDesatDelay();
    this._cancelGlitchDelay();
    this.renderTarget.dispose();
    if (this._intermediateTarget) {
      this._intermediateTarget.dispose();
    }
    this.fullScreenQuad.dispose();
    this.postMaterial.dispose();
    this._heightMap.dispose();
    if (this.desatEnableAudio && this.desatAudio) {
      this.desatAudio.dispose();
      this.desatAudio = null;
    }
    if (this.glitchEnableAudio && this.glitchAudio) {
      this.glitchAudio.dispose();
      this.glitchAudio = null;
    }
  }
}

export default DesaturationAndGlitchEffects;

