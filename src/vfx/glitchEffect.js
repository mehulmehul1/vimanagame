import * as THREE from "three";
import { VFXManager } from "../vfxManager.js";
import { DigitalGlitch } from "./shaders/DigitalGlitch.js";
import { ProceduralAudio } from "./proceduralAudio.js";

/**
 * GlitchEffect - Post-processing shader for digital glitch effect
 *
 * Usage:
 *   const glitch = new GlitchEffect(renderer);
 *   glitch.enable();
 *   glitch.setIntensity(0.08);
 *   glitch.render(scene, camera);
 */
export class GlitchEffect extends VFXManager {
  constructor(renderer) {
    super("GlitchEffect", false);

    this.renderer = renderer;
    this.enabled = false;
    this.intensity = 0.08;
    this.goWild = false;
    
    // Audio control
    this.enableAudio = false; // Set to true to enable procedural audio
    
    // Glitch timing
    this._curF = 0;
    this._randX = 120;

    // Procedural audio - rapidly glitching static
    this.audio = new ProceduralAudio({
      name: "GlitchAudio",
      baseFrequency: 800.0, // Higher frequency for static
      volume: 0.3,
      baseOscType: "square", // Square wave for harsh static
      subOscType: "sawtooth", // Sawtooth for edge
      modOscType: "sawtooth", // Aggressive modulation
      filterType: "bandpass",
      filterFreq: 5000, // High frequency band for static
      filterQ: 5, // High Q for sharp static bursts
      distortionAmount: 50, // Heavy distortion for chaotic static
      delayTime: 0.02, // Very short delay for rapid glitches
      delayFeedback: 0.8, // High feedback for chaotic tails
      lfoFreq: 20.0, // Very fast oscillation - 20 Hz
      lfoDepth: 2000, // Deep frequency modulation for rapid glitches
      fadeInTime: 0.05, // Very quick fade in
      fadeOutTime: 0.1,
      fadeInCurve: "linear",
      fadeOutCurve: "exponential",
      enableSweep: true,
      sweepBaseFreq: 6000,
      sweepRange: 4000,
      sweepRate: 25.0, // Very fast sweeps - 25 per second for rapid glitching
      sweepGain: 0.4,
    });

    // Create glitch shader material
    this.shader = {
      uniforms: THREE.UniformsUtils.clone(DigitalGlitch.uniforms),
      vertexShader: DigitalGlitch.vertexShader,
      fragmentShader: DigitalGlitch.fragmentShader,
    };

    // Generate displacement texture (height map)
    this._heightMap = this._generateHeightmap(64);
    this.shader.uniforms.tDisp.value = this._heightMap;

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
   * Generate displacement texture (height map) for glitch effect
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
   * Generate random trigger interval
   * @private
   */
  _generateTrigger() {
    this._randX = THREE.MathUtils.randInt(120, 240);
  }

  /**
   * Enable/disable the glitch effect
   * @param {boolean} enabled
   */
  enable(enabled = true) {
    this.enabled = enabled;
  }

  /**
   * Disable the glitch effect
   */
  disable() {
    this.enabled = false;
  }

  /**
   * Set glitch intensity
   * @param {number} intensity - Intensity value (0.0 to 1.0)
   */
  setIntensity(intensity) {
    this.intensity = THREE.MathUtils.clamp(intensity, 0.0, 1.0);
  }

  /**
   * Set wild mode (continuously intense glitching)
   * @param {boolean} wild
   */
  setWild(wild = false) {
    this.goWild = wild;
  }

  /**
   * Override: Called when first effect matches - enable the glitch system
   * @param {Object} effect - Effect data from vfxData.js
   * @param {Object} state - Current game state
   */
  async onFirstEnable(effect, state) {
    this.logger.log("Enabling glitch effect for first time");
    
    // Check if audio should be enabled from effect parameters - default to false if not specified
    const params = effect.parameters || {};
    this.enableAudio = params.enableAudio === true;
    
    // Stop any currently playing audio if disabled
    if (!this.enableAudio && this.audio && this.audio.isPlaying) {
      this.audio.stop();
    }
    
    // Initialize procedural audio only if enabled
    if (this.enableAudio) {
      await this.audio.initialize();
    }
    
    this.enable(true);
    this.applyEffect(effect, state);
    
    // Only start audio if intensity is actually above threshold
    // Audio will start automatically in update() when intensity > 0.001
  }

  /**
   * Override: Apply effect from game state
   * @param {Object} effect - Effect data from vfxData.js
   * @param {Object} state - Current game state
   */
  applyEffect(effect, state) {
    const params = effect.parameters || {};
    
    // Update enableAudio from effect parameters - default to false if not specified
    this.enableAudio = params.enableAudio === true;
    
    // If audio is being disabled, stop it
    if (!this.enableAudio && this.audio && this.audio.isPlaying) {
      this.audio.stop();
    }
    // If audio is being enabled and context doesn't exist, initialize it
    if (this.enableAudio && !this.audio.audioContext) {
      this.audio.initialize().catch((err) => {
        this.logger.warn(`Failed to initialize glitch audio: ${err}`);
      });
    }

    if (params.intensity !== undefined) {
      this.setIntensity(params.intensity);
    }

    if (params.goWild !== undefined) {
      this.setWild(params.goWild);
    }
  }

  /**
   * Override: Handle when no effect matches state (disable glitch)
   * @param {Object} state - Current game state
   */
  onNoEffect(state) {
    this.logger.log("No glitch effect needed - disabling");
    
    // Stop audio when glitch effect is disabled (only if audio is enabled)
    if (this.enableAudio && this.audio && this.audio.isPlaying) {
      this.audio.stop();
    }
  }

  /**
   * Update glitch animation (call each frame)
   * @param {number} deltaTime - Time since last frame in seconds
   */
  update(deltaTime) {
    if (!this.enabled || !this._hasEverBeenEnabled) {
      return;
    }

    // If intensity is effectively zero, bypass glitch entirely
    if (this.intensity <= 0.001) {
      this.shader.uniforms.byp.value = 1;
      this.shader.uniforms.amount.value = 0.0;
      this.shader.uniforms.seed_x.value = 0.0;
      this.shader.uniforms.seed_y.value = 0.0;
      
      // Stop audio when intensity is zero (only if audio is enabled)
      if (this.enableAudio && this.audio && this.audio.isPlaying) {
        this.audio.stop();
      }
      return;
    }

    // Start audio if not playing and intensity is above threshold (only if audio is enabled)
    if (this.enableAudio && this.audio && !this.audio.isPlaying && this.audio.audioContext) {
      this.audio.start();
    }

    // Update audio to match glitch intensity and wildness (only if audio is enabled)
    if (this.enableAudio && this.audio && this.audio.isPlaying) {
      // Volume scales with intensity
      const audioVolume = 0.3 * this.intensity;
      
      // When goWild is true, make it even more chaotic
      const sweepRate = this.goWild ? 40.0 : 25.0;
      const lfoFreq = this.goWild ? 30.0 : 20.0;
      const filterFreq = this.goWild ? 7000 : 5000;
      
      this.audio.updateParams({
        volume: audioVolume,
        sweepRate: sweepRate,
        lfoFreq: lfoFreq,
        filterFreq: filterFreq,
        transitionTime: 0.05, // Rapid updates for glitchy feel
      });
    }

    this.shader.uniforms.tDiffuse.value = null; // Will be set in render
    this.shader.uniforms.seed.value = Math.random();
    this.shader.uniforms.byp.value = 0;

    if (this._curF % this._randX === 0 || this.goWild) {
      // Major glitch
      this.shader.uniforms.amount.value = Math.random() / 30;
      this.shader.uniforms.angle.value = THREE.MathUtils.randFloat(
        -Math.PI,
        Math.PI
      );
      this.shader.uniforms.seed_x.value = THREE.MathUtils.randFloat(-1, 1);
      this.shader.uniforms.seed_y.value = THREE.MathUtils.randFloat(-1, 1);
      this.shader.uniforms.distortion_x.value = THREE.MathUtils.randFloat(0, 1);
      this.shader.uniforms.distortion_y.value = THREE.MathUtils.randFloat(0, 1);
      this._curF = 0;
      this._generateTrigger();
    } else if (this._curF % this._randX < this._randX / 5) {
      // Minor glitch
      this.shader.uniforms.amount.value = Math.random() / 90;
      this.shader.uniforms.angle.value = THREE.MathUtils.randFloat(
        -Math.PI,
        Math.PI
      );
      this.shader.uniforms.distortion_x.value = THREE.MathUtils.randFloat(0, 1);
      this.shader.uniforms.distortion_y.value = THREE.MathUtils.randFloat(0, 1);
      this.shader.uniforms.seed_x.value = THREE.MathUtils.randFloat(-0.3, 0.3);
      this.shader.uniforms.seed_y.value = THREE.MathUtils.randFloat(-0.3, 0.3);
    } else if (!this.goWild) {
      // No glitch
      this.shader.uniforms.byp.value = 1;
    }

    // Scale intensity
    this.shader.uniforms.amount.value *= this.intensity;
    this.shader.uniforms.seed_x.value *= this.intensity;
    this.shader.uniforms.seed_y.value *= this.intensity;

    this._curF++;
  }

  /**
   * Cleanup
   */
  dispose() {
    if (this.audio) {
      this.audio.dispose();
      this.audio = null;
    }
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
   * Render the scene with glitch effect
   * This should be called after the base scene is rendered to a texture
   * @param {THREE.Texture|null} inputTexture - Input texture (required)
   * @param {THREE.Scene} scene - Scene to render (if inputTexture is null)
   * @param {THREE.Camera} camera - Camera to use (if inputTexture is null)
   */
  render(inputTexture, scene = null, camera = null) {
    if (!this.enabled || !this._hasEverBeenEnabled) {
      // If not enabled, pass through input texture or render scene directly
      if (inputTexture && inputTexture instanceof THREE.Texture) {
        // Render input texture to screen (pass through)
        this.postMaterial.uniforms.tDiffuse.value = inputTexture;
        this.postMaterial.uniforms.byp.value = 1; // Bypass glitch
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.postScene, this.postCamera);
        this.postMaterial.uniforms.byp.value = 0; // Reset
        return;
      }
      if (scene && camera) {
        this.renderer.render(scene, camera);
      }
      return;
    }

    let sourceTexture = inputTexture;

    // If no input texture provided, render scene first
    if (!sourceTexture && scene && camera) {
      this.renderer.setRenderTarget(this.renderTarget);
      this.renderer.render(scene, camera);
      sourceTexture = this.renderTarget.texture;
    }

    // Apply glitch shader
    if (sourceTexture) {
      this.postMaterial.uniforms.tDiffuse.value = sourceTexture;

      // Render to screen
      this.renderer.setRenderTarget(null);
      this.renderer.render(this.postScene, this.postCamera);
    }
  }

  /**
   * Dispose of resources
   */
  dispose() {
    this.renderTarget.dispose();
    this.postMaterial.dispose();
    this._heightMap.dispose();
  }
}

export default GlitchEffect;

