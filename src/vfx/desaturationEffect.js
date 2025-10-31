import * as THREE from "three";
import { VFXManager } from "../vfxManager.js";
import { ProceduralAudio } from "./proceduralAudio.js";

/**
 * DesaturationEffect - Post-processing shader for animating color to grayscale
 *
 * Usage:
 *   const desat = new DesaturationEffect(renderer);
 *   desat.enable(); // Enable for color scenes
 *   desat.animateToGrayscale(); // Animate to B&W
 *   desat.render(scene, camera); // Call instead of renderer.render()
 */
export class DesaturationEffect extends VFXManager {
  constructor(renderer) {
    super("DesaturationEffect", false);

    this.renderer = renderer;
    this.enabled = false;
    this.progress = 0; // 0 = color, 1 = grayscale
    this.animating = false;
    this.animationTarget = 0;
    this.animationDuration = 5.0; // Single duration value for all transitions
    this.animationSpeed = 0;
    this.currentState = 0; // Track current state: 0 = color, 1 = grayscale

    // Transition properties
    this.transitionMode = "bleed"; // 'fade', 'bleed', or 'wipe'
    this.bleedScale = 3; // Scale of noise pattern
    this.bleedSoftness = 0.2; // Softness of bleed edges
    this.wipeDirection = "bottom-to-top"; // 'bottom-to-top' or 'top-to-bottom'
    this.wipeSoftness = 0.15; // Softness of wipe edge

    // Audio control
    this.enableAudio = false; // Set to true to enable procedural audio

    // Procedural audio (lower volume than splat morph)
    this.audio = new ProceduralAudio({
      name: "DesaturationAudio",
      baseFrequency: 155.56, // Eb3
      volume: 0.1, // Quieter for more subtlety
      baseOscType: "sine", // Pure sine for smooth tone
      subOscType: "sine", // Pure sine sub for warmth
      modOscType: "triangle", // Triangle wave for softer harmonics
      filterType: "lowpass",
      filterFreq: 2000, // Sweet spot for warmth
      filterQ: 1.2, // Low Q = gentle, non-resonant
      distortionAmount: 3, // Minimal distortion = cleaner
      delayTime: 0.25, // Longer = more spacious/reverb-like
      delayFeedback: 0.2, // Moderate feedback
      lfoFreq: 0.4, // Slow, gentle modulation
      lfoDepth: 15, // Subtle depth
      fadeInTime: 0, // Gentle fade in
      fadeOutTime: 1.0, // Long fade out
      fadeInCurve: "exponential", // Natural-sounding fade
      fadeOutCurve: "exponential", // Natural-sounding fade
      // Radio tuning sweep effect (high-frequency oscillating)
      enableSweep: true,
      sweepBaseFreq: 3500, // High frequency center
      sweepRange: 1200, // Sweep range in Hz
      sweepRate: 3.0, // 3 sweeps per second
      sweepGain: 0.2, // Sweep volume multiplier
    });
    this.lastProgress = 0;

    // Create desaturation shader
    this.shader = {
      uniforms: {
        tDiffuse: { value: null },
        desatAmount: { value: 0.0 },
        animProgress: { value: 0.0 },
        transitionMode: { value: 1.0 }, // 0=fade, 1=bleed, 2=wipe
        bleedScale: { value: this.bleedScale },
        bleedSoftness: { value: this.bleedSoftness },
        wipeDirection: { value: 0.0 }, // 0=bottom-to-top, 1=top-to-bottom
        wipeSoftness: { value: this.wipeSoftness },
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
        uniform float desatAmount;
        uniform float animProgress;
        uniform float transitionMode;
        uniform float bleedScale;
        uniform float bleedSoftness;
        uniform float wipeDirection;
        uniform float wipeSoftness;
        varying vec2 vUv;
        
        // Constants
        const vec3 LUMA_WEIGHTS = vec3(0.299, 0.587, 0.114);
        const int NOISE_OCTAVES = 4;
        const float NOISE_INITIAL_AMPLITUDE = 0.5;
        const float NOISE_FREQUENCY_MULT = 2.0;
        const float NOISE_AMPLITUDE_MULT = 0.5;
        
        // Simple 2D noise function
        float noise(vec2 st) {
          return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
        }
        
        // Smooth noise (interpolated)
        float smoothNoise(vec2 st) {
          vec2 i = floor(st);
          vec2 f = fract(st);
          
          // Four corners
          float a = noise(i);
          float b = noise(i + vec2(1.0, 0.0));
          float c = noise(i + vec2(0.0, 1.0));
          float d = noise(i + vec2(1.0, 1.0));
          
          // Smooth Hermite interpolation
          vec2 u = f * f * (3.0 - 2.0 * f);
          
          return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
        }
        
        // Fractal noise (multiple octaves)
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
        
        void main() {
          vec4 color = texture2D(tDiffuse, vUv);
          
          // Calculate luminance (proper grayscale conversion)
          float gray = dot(color.rgb, LUMA_WEIGHTS);
          
          float desatAmountFinal = desatAmount;
          
          // Mode 2: Wipe transition (vertical sweep)
          if (transitionMode > 1.5) {
            // Get Y coordinate of pixel (0 = bottom, 1 = top in UV space)
            float pixelY = vUv.y;
            
            // Flip for top-to-bottom wipe
            if (wipeDirection > 0.5) {
              pixelY = 1.0 - pixelY;
            }
            
            // Remap animProgress to account for softness so wipe completes at boundaries
            // When animProgress = 1.0, we want the wipe edge to be past the top (at 1.0 + wipeSoftness)
            float wipeEdge = animProgress * (1.0 + wipeSoftness);
            
            // Calculate wipe mask with smoothstep for soft edge
            // Invert so grayscale is LEFT BEHIND the wipe (below), not ahead (above)
            float wipeMask = 1.0 - smoothstep(
              wipeEdge - wipeSoftness,
              wipeEdge,
              pixelY
            );
            
            desatAmountFinal = wipeMask;
          }
          // Mode 1: Bleed transition (noise-based)
          else if (transitionMode > 0.5) {
            // Generate noise pattern [0, 1]
            float noiseValue = fractalNoise(vUv * bleedScale);
            
            // Map noise to threshold range that uses full progress [0, 1]
            // Early pixels (noise=0) transition at start, late pixels (noise=1) at end
            float edgeMargin = bleedSoftness * 0.5;
            float pixelThreshold = edgeMargin + noiseValue * (1.0 - bleedSoftness);
            
            // Smooth transition around each pixel's threshold
            float bleedMask = smoothstep(
              pixelThreshold - edgeMargin,
              pixelThreshold + edgeMargin,
              animProgress
            );
            
            desatAmountFinal = bleedMask;
          }
          // Mode 0: Fade transition (simple linear)
          else {
            desatAmountFinal = desatAmount;
          }
          
          // Lerp between color and grayscale
          vec3 result = mix(color.rgb, vec3(gray), desatAmountFinal);
          
          gl_FragColor = vec4(result, color.a);
        }
      `,
    };

    // Create render target for post-processing
    const size = this.renderer.getSize(new THREE.Vector2());
    this.renderTarget = new THREE.WebGLRenderTarget(size.x, size.y, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    });

    // Create post-processing scene and camera
    this.postScene = new THREE.Scene();
    this.postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.postMaterial = new THREE.ShaderMaterial(this.shader);
    this.postQuad = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      this.postMaterial
    );
    this.postScene.add(this.postQuad);
  }

  /**
   * Enable/disable the desaturation effect
   * @param {boolean} enabled
   */
  enable(enabled = true) {
    this.enabled = enabled;
  }

  /**
   * Disable the desaturation effect
   */
  disable() {
    this.enabled = false;
  }

  /**
   * Override: Called when first effect matches - enable the desaturation system
   * @param {Object} effect - Effect data from vfxData.js
   * @param {Object} state - Current game state
   */
  async onFirstEnable(effect, state) {
    this.logger.log("Enabling desaturation effect for first time");
    this.enable(true);
    // Initialize audio if enabled
    if (this.enableAudio) {
      await this.audio.initialize();
    }
    // Apply the initial effect
    this.applyEffect(effect, state);

    // Start audio if we're in color mode (progress < 0.5)
    if (this.enableAudio && this.currentState < 0.5 && !this.audio.isPlaying) {
      this.audio.start();
    }
  }

  /**
   * Override: Apply effect from game state
   * @param {Object} effect - Effect data from vfxData.js
   * @param {Object} state - Current game state
   */
  applyEffect(effect, state) {
    const params = effect.parameters || {};

    // Update animation duration if specified
    if (params.duration !== undefined) {
      this.animationDuration = params.duration;
    }

    // Animate to the target amount with specified options
    const options = {
      mode: params.mode || "bleed",
      direction: params.direction || params.wipeDirection || "bottom-to-top",
      softness:
        params.softness !== undefined ? params.softness : params.wipeSoftness,
      suppressAudio: params.suppressAudio || false,
    };

    this.animateTo(params.target, options);
  }

  /**
   * Override: Handle when no effect matches state (disable desaturation)
   * @param {Object} state - Current game state
   */
  onNoEffect(state) {
    this.logger.log("No desaturation effect needed - disabling");
    // Stop audio when effect is disabled
    if (this.enableAudio && this.audio.isPlaying) {
      this.audio.stop();
    }
    // Optionally fade to color before disabling
    // this.animateTo(0.0, { mode: "fade", duration: 1.0 });
    // For now, just keep last state but could disable if desired
  }

  /**
   * Set desaturation amount instantly (no animation)
   * @param {number} amount - 0 = color, 1 = grayscale
   */
  setAmount(amount) {
    this.progress = THREE.MathUtils.clamp(amount, 0, 1);
    this.currentState = this.progress;
    this.animating = false;

    // Update audio state: on when in color (< 0.5), off when grayscale (>= 0.5)
    if (this.enableAudio) {
      if (this.currentState < 0.5 && !this.audio.isPlaying) {
        this.audio.start();
      } else if (this.currentState >= 0.5 && this.audio.isPlaying) {
        this.audio.stop();
      }
    }
  }

  /**
   * Animate to a target desaturation amount
   * @param {number} targetAmount - 0 = color, 1 = grayscale
   * @param {Object} options - Transition settings:
   *   - mode: 'fade' | 'bleed' (default: 'bleed')
   */
  animateTo(targetAmount, options = {}) {
    if (!this.enabled) {
      this.logger.warn("Effect is disabled. Call enable() first.");
      return;
    }

    this.animationTarget = THREE.MathUtils.clamp(targetAmount, 0, 1);
    
    // Handle instant transitions (duration = 0)
    if (this.animationDuration <= 0) {
      this.progress = this.animationTarget;
      this.currentState = this.animationTarget;
      this.animating = false;
      this.lastProgress = this.progress;
      
      // Configure transition mode even for instant transitions
      const mode = options.mode || "bleed";
      this.transitionMode = mode;
      if (mode === "wipe") {
        this.postMaterial.uniforms.transitionMode.value = 2.0;
        const direction = options.direction || this.wipeDirection;
        this.wipeDirection = direction;
        this.postMaterial.uniforms.wipeDirection.value =
          direction === "top-to-bottom" ? 1.0 : 0.0;
        const softness =
          options.softness !== undefined ? options.softness : this.wipeSoftness;
        this.wipeSoftness = softness;
        this.postMaterial.uniforms.wipeSoftness.value = softness;
      } else if (mode === "bleed") {
        this.postMaterial.uniforms.transitionMode.value = 1.0;
        this.postMaterial.uniforms.bleedScale.value = this.bleedScale;
        this.postMaterial.uniforms.bleedSoftness.value = this.bleedSoftness;
      } else {
        this.postMaterial.uniforms.transitionMode.value = 0.0;
      }
      return;
    }
    
    this.animationSpeed = 1.0 / this.animationDuration;
    this.animating = true;

    // Audio logic: sound is on when in color (progress close to 0)
    if (this.enableAudio && !options.suppressAudio) {
      // Start audio if we're in or transitioning to color
      if (targetAmount < 0.5 && !this.audio.isPlaying) {
        this.audio.start();
      }
      // Also start if currently in color and animating
      if (this.currentState < 0.5 && !this.audio.isPlaying) {
        this.audio.start();
      }
    }

    this.lastProgress = this.progress;

    // Configure transition mode (default to noise-based bleed)
    const mode = options.mode || "bleed";
    this.transitionMode = mode;

    if (mode === "wipe") {
      // Wipe mode
      this.postMaterial.uniforms.transitionMode.value = 2.0;
      const direction = options.direction || this.wipeDirection;
      this.wipeDirection = direction;
      this.postMaterial.uniforms.wipeDirection.value =
        direction === "top-to-bottom" ? 1.0 : 0.0;
      const softness =
        options.softness !== undefined ? options.softness : this.wipeSoftness;
      this.wipeSoftness = softness;
      this.postMaterial.uniforms.wipeSoftness.value = softness;
    } else if (mode === "bleed") {
      // Bleed mode
      this.postMaterial.uniforms.transitionMode.value = 1.0;
      this.postMaterial.uniforms.bleedScale.value = this.bleedScale;
      this.postMaterial.uniforms.bleedSoftness.value = this.bleedSoftness;
    } else {
      // Fade mode
      this.postMaterial.uniforms.transitionMode.value = 0.0;
    }
  }

  /**
   * Animate to grayscale
   * @param {Object} options - Transition settings (see animateTo)
   */
  animateToGrayscale(options = {}) {
    this.animateTo(1.0, options);
  }

  /**
   * Animate to color
   * @param {Object} options - Transition settings (see animateTo)
   */
  animateToColor(options = {}) {
    this.animateTo(0.0, options);
  }

  /**
   * Toggle between color and grayscale
   * @param {Object} options - Transition settings (see animateTo)
   */
  toggle(options = {}) {
    const target = this.currentState < 0.5 ? 1.0 : 0.0;
    this.animateTo(target, options);
  }

  /**
   * Update animation (call each frame)
   * @param {number} deltaTime - Time since last frame in seconds
   */
  update(deltaTime) {
    if (!this.animating) {
      // Even when not animating, manage audio state based on current progress
      // Audio should be on when in color (progress close to 0)
      if (this.enableAudio) {
        if (this.currentState < 0.5 && !this.audio.isPlaying) {
          this.audio.start();
        } else if (this.currentState >= 0.5 && this.audio.isPlaying) {
          this.audio.stop();
        }
      }
      return;
    }

    // Move toward target
    const distance = this.animationTarget - this.progress;
    const step = Math.sign(distance) * this.animationSpeed * deltaTime;

    if (Math.abs(distance) <= Math.abs(step)) {
      // Reached target
      this.progress = this.animationTarget;
      this.animating = false;

      // Update current state when animation completes
      this.currentState = this.animationTarget;

      // Stop audio if we've reached grayscale (progress >= 0.5)
      if (this.enableAudio) {
        if (this.animationTarget >= 0.5 && this.audio.isPlaying) {
          this.audio.stop();
        }
        // Start audio if we've reached color (progress < 0.5)
        else if (this.animationTarget < 0.5 && !this.audio.isPlaying) {
          this.audio.start();
        }
      }
    } else {
      // Continue animating
      this.progress += step;
      this.currentState = this.progress;

      // Update audio parameters based on progress and velocity
      if (this.enableAudio) {
        this._updateAudio(deltaTime);
      }
    }
  }

  /**
   * Update audio parameters based on desaturation progress
   * @private
   */
  _updateAudio(deltaTime) {
    // Calculate velocity (rate of change of progress)
    const velocity =
      Math.abs(this.progress - this.lastProgress) / Math.max(deltaTime, 0.001);
    // Normalize to roughly 0-1 range
    const normalizedVelocity = Math.min(velocity * 2, 1);

    // Map progress to filter frequency
    // More color (progress close to 0): higher frequency (brighter)
    // More grayscale (progress close to 1): lower frequency (darker)
    const minFreq = 300;
    const maxFreq = 4000;
    const targetFilterFreq = maxFreq - (maxFreq - minFreq) * this.progress;

    // Map velocity to filter resonance (faster = more resonant)
    const minQ = 2;
    const maxQ = 10;
    const targetQ = minQ + (maxQ - minQ) * normalizedVelocity;

    // Map velocity to pitch (subtle pitch variation with speed)
    const pitchMultiplier = 1.0 + normalizedVelocity * 0.2;

    // Map progress to stereo pan (slow sweep)
    const panAmount = Math.sin(this.progress * Math.PI) * 0.5;

    // Volume inversely related to progress (high when color, low when grayscale)
    // Full color (progress = 0): max volume
    // Full grayscale (progress = 1): silent
    const colorAmount = 1.0 - this.progress; // 1 when color, 0 when grayscale
    const minVolume = 0.0;
    const maxVolume = this.audio.config.volume;

    // Base volume uses a gentle squared curve
    const baseCurve = colorAmount * colorAmount;
    let baseVolume = minVolume + (maxVolume - minVolume) * baseCurve;

    // Add a LARGE boost during active transitions (high velocity)
    // This makes the audio immediately perceptible when transition starts
    const velocityBoost = normalizedVelocity * maxVolume * 0.8; // Up to 80% boost

    // Combine base volume with velocity boost, but cap at max
    const targetVolume = Math.min(baseVolume + velocityBoost, maxVolume * 1.2);

    // Sweep amount increases as desaturation increases (radio tuning effect)
    // More grayscale = more sweep. Use squared curve for smooth ramp
    const desatAmount = this.progress; // 0 = color, 1 = grayscale
    const sweepAmount = desatAmount * desatAmount; // Squared for smooth buildup

    this.audio.updateParams({
      filterFreq: targetFilterFreq,
      filterQ: targetQ,
      pitchMultiplier,
      pan: panAmount,
      volume: targetVolume,
      sweepAmount: sweepAmount, // Control radio tuning sweep
      transitionTime: 0.1,
    });

    this.lastProgress = this.progress;
  }

  /**
   * Resize the render target (call when window resizes)
   * @param {number} width
   * @param {number} height
   */
  setSize(width, height) {
    this.renderTarget.setSize(width, height);
  }

  /**
   * Render the scene with desaturation effect
   * @param {THREE.Scene} scene
   * @param {THREE.Camera} camera
   */
  render(scene, camera) {
    // Once enabled, always render through post-processing to avoid switching
    // between rendering paths (which can cause visual pops)
    // Only skip post-processing if effect has never been enabled
    if (this.enabled && this._hasEverBeenEnabled) {
      // Render scene to texture
      this.renderer.setRenderTarget(this.renderTarget);
      this.renderer.render(scene, camera);

      // Apply desaturation shader
      this.postMaterial.uniforms.tDiffuse.value = this.renderTarget.texture;

      // Pass animation progress to drive transitions
      this.postMaterial.uniforms.desatAmount.value = this.currentState;
      this.postMaterial.uniforms.animProgress.value = this.progress;

      // Render to screen
      this.renderer.setRenderTarget(null);
      this.renderer.render(this.postScene, this.postCamera);
    } else {
      // Effect not yet enabled, render normally
      this.renderer.render(scene, camera);
    }
  }

  /**
   * Dispose of resources
   */
  dispose() {
    this.renderTarget.dispose();
    this.postMaterial.dispose();
    if (this.enableAudio && this.audio) {
      this.audio.dispose();
      this.audio = null;
    }
  }
}

export default DesaturationEffect;
