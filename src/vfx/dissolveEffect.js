/**
 * Dissolve Effect
 *
 * State-driven dissolve effect for GLTF models using Perlin noise.
 * Extends VFXManager for automatic game state integration.
 *
 * Features:
 * - Multi-material support (traverses GLTF and modifies all materials)
 * - Particle emission from dissolving edges
 * - Colored edge glow
 * - Automatic or manual progress control
 *
 * Usage:
 *   const dissolve = new DissolveEffect(scene, sceneManager, renderer);
 *   dissolve.setGameManager(gameManager, "dissolve");
 *   dissolve.update(deltaTime);
 */

import * as THREE from "three";
import { VFXManager } from "../vfxManager.js";
import { Logger } from "../utils/logger.js";
import { perlinNoise } from "./shaders/perlinNoise.glsl.js";
import {
  vertexGlobal,
  vertexMain,
  fragmentGlobal,
  fragmentMain,
} from "./shaders/dissolveShader.glsl.js";
import {
  particleVertexShader,
  particleFragmentShader,
} from "./shaders/dissolveParticle.glsl.js";
import { setupUniforms, setupShaderSnippets } from "../utils/shaderHelper.js";
import { DissolveParticleSystem } from "./dissolveParticleSystem.js";
import * as BufferGeometryUtils from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { ProceduralAudio } from "./proceduralAudio.js";
import { getSceneObjectsForState } from "../sceneData.js";

export class DissolveEffect extends VFXManager {
  constructor(scene, sceneManager, renderer) {
    super("DissolveEffect", true); // Enable debug logging
    this.logger = new Logger("DissolveEffect", false);

    this.scene = scene;
    this.sceneManager = sceneManager;
    this.renderer = renderer;

    // Track active dissolves: objectId -> dissolve state
    this.activeDissolves = new Map();

    // Load particle texture
    this.textureLoader = new THREE.TextureLoader();
    this.particleTexture = null;
    this.textureLoader.load("/images/particle.png", (texture) => {
      this.particleTexture = texture;
      this.logger.log("Particle texture loaded");
    });

    // Animation state
    this.autoAnimate = false;
    this.animationDirection = 1; // 1 = forward, -1 = reverse
    this.animationSpeed = 1.5;

    // Audio control
    this.enableAudio = false;
    this.suppressAudio = false; // Can be set per-effect to temporarily disable audio

    // Procedural audio - wishy-washy oscillating static effect
    this.audio = new ProceduralAudio({
      name: "DissolveAudio",
      baseFrequency: 220.0, // A3 - higher for brightness
      volume: 0.12,
      baseOscType: "triangle",
      subOscType: "sine",
      modOscType: "sine",
      filterType: "bandpass",
      filterFreq: 4000,
      filterQ: 3,
      distortionAmount: 12,
      delayTime: 0.08,
      delayFeedback: 0.5,
      lfoFreq: 10.0, // Fast oscillation - 10 Hz
      lfoDepth: 80,
      fadeInTime: 0.2,
      fadeOutTime: 0.3,
      fadeInCurve: "exponential",
      fadeOutCurve: "exponential",
      enableSweep: true,
      sweepBaseFreq: 5000,
      sweepRange: 2000,
      sweepRate: 12.0, // Fast sweeps - 12 per second
      sweepGain: 0.3,
    });
    this.lastProgress = 0;

    this.logger.log("DissolveEffect initialized");
  }

  /**
   * Override: Called when first effect matches - initialize audio
   * @param {Object} effect - Effect data from vfxData.js
   * @param {Object} state - Current game state
   */
  async onFirstEnable(effect, state) {
    this.logger.log("[DissolveEffect] Enabling dissolve effect for first time");
    // Don't call applyEffect here - let the normal updateForState flow handle it
    // This prevents double-calling applyEffect which can cause Safari issues
  }

  /**
   * Apply dissolve effect based on effect parameters
   * @param {Object} effect - Effect data from vfxData.js
   * @param {Object} state - Current game state
   */
  async applyEffect(effect, state) {
    const params = effect.parameters || {};

    this.logger.log(
      `🔥 [DissolveEffect] Applying effect: ${effect.id} (state: ${state?.currentState})`,
      params
    );

    // Safari detection for debugging
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    if (isSafari) {
      this.logger.log(`[Safari] applyEffect called for ${effect.id}`);
    }

    try {
      // Enable/disable audio based on effect parameters and initialize if needed
      // Safari autoplay fix: Don't block effect application if audio can't initialize
      if (params.enableAudio !== undefined) {
        const wasEnabled = this.enableAudio;
        this.enableAudio = params.enableAudio;

        // Initialize audio if it's being enabled and context doesn't exist
        // Use non-blocking approach - don't await, let it happen in background
        if (this.enableAudio && !this.audio.audioContext) {
          this.audio.initialize().catch((error) => {
            // Safari autoplay policy: Audio context may fail to initialize without user gesture
            // This is OK - visual effect should still work
            this.logger.warn(
              `[DissolveEffect] Audio initialization failed (may need user gesture):`,
              error
            );
          });
          this.logger.log("Audio initialization started (non-blocking)");
        }

        // If audio context exists but is suspended, try to resume it (non-blocking)
        if (
          this.enableAudio &&
          this.audio.audioContext &&
          this.audio.audioContext.state === "suspended"
        ) {
          this.audio.audioContext.resume().catch((error) => {
            // Safari autoplay policy: Resume may fail without user gesture
            // This is OK - visual effect should still work
            this.logger.warn(
              `[DissolveEffect] Audio context resume failed (may need user gesture):`,
              error
            );
          });
          this.logger.log("Audio context resume attempted (non-blocking)");
        }
      }

      // Handle suppressAudio parameter (for reverse transitions)
      if (params.suppressAudio !== undefined) {
        this.suppressAudio = params.suppressAudio;
        // If suppressing audio and it's currently playing, stop it
        if (
          this.suppressAudio &&
          this.audio.audioContext &&
          this.audio.isPlaying
        ) {
          this.audio.stop();
        }
      }

      // Extract parameters
      const targetObjectIds = params.targetObjectIds || [];
      const progress = params.progress !== undefined ? params.progress : 0;
      const targetProgress =
        params.targetProgress !== undefined ? params.targetProgress : 14.0;
      const autoAnimate = params.autoAnimate || false;
      const transitionDuration =
        params.transitionDuration !== undefined ? params.transitionDuration : 0;
      const mode = params.mode || "noise"; // "noise" or "wipe"
      const wipeDirection = params.wipeDirection || "bottom-to-top";
      const wipeSoftness =
        params.wipeSoftness !== undefined ? params.wipeSoftness : 0.15;
      const edgeColor1 = params.edgeColor1 || "#4d9bff";
      const edgeColor2 = params.edgeColor2 || "#0733ff";
      const particleColor = params.particleColor || "#4d9bff";
      const frequency =
        params.frequency !== undefined ? params.frequency : 0.45;
      const edgeWidth = params.edgeWidth !== undefined ? params.edgeWidth : 0.8;
      const particleIntensity =
        params.particleIntensity !== undefined ? params.particleIntensity : 1.0;

      // Particle system parameters
      const particleSize =
        params.particleSize !== undefined ? params.particleSize : 35.0;
      const particleDecimation =
        params.particleDecimation !== undefined
          ? params.particleDecimation
          : 10;
      const particleDispersion =
        params.particleDispersion !== undefined
          ? params.particleDispersion
          : 8.0;
      const particleVelocitySpread =
        params.particleVelocitySpread !== undefined
          ? params.particleVelocitySpread
          : 0.15;

      // Store animation settings
      this.autoAnimate = autoAnimate;
      this.targetProgress = targetProgress;
      this.transitionDuration = transitionDuration;

      // Setup dissolve for each target object
      // Use async iteration to wait for objects that might be loading
      // Safari-compatible: use traditional for loop instead of for...of with continue
      this.logger.log(
        `[DissolveEffect] Processing ${
          targetObjectIds.length
        } target objects: [${targetObjectIds.join(", ")}]`
      );

      for (let i = 0; i < targetObjectIds.length; i++) {
        const objectId = targetObjectIds[i];
        this.logger.log(
          `[DissolveEffect] Processing object: ${objectId} (${i + 1}/${
            targetObjectIds.length
          })`
        );
        const existingDissolve = this.activeDissolves.get(objectId);

        // Check if mode changed - if so, need to recreate
        const modeChanged = existingDissolve && existingDissolve.mode !== mode;
        this.logger.log(
          `[DissolveEffect] ${objectId}: existingDissolve=${!!existingDissolve}, modeChanged=${modeChanged}`
        );

        // Check if object still exists in scene manager
        let sceneObject = this.sceneManager.getObject(objectId);
        this.logger.log(
          `[DissolveEffect] ${objectId}: sceneObject found=${!!sceneObject}`
        );

        // If not found, wait for it (handles race condition where updateSceneForState hasn't started)
        if (!sceneObject) {
          this.logger.log(
            `[DissolveEffect] ${objectId}: Object not found, waiting for it...`
          );
          try {
            sceneObject = await this._waitForObject(objectId);
            this.logger.log(
              `[DissolveEffect] ${objectId}: _waitForObject returned: ${!!sceneObject}`
            );
          } catch (error) {
            this.logger.error(
              `[DissolveEffect] Error waiting for object ${objectId}:`,
              error
            );
            this.logger.error(`[DissolveEffect] Error stack:`, error?.stack);
            sceneObject = null;
          }
          if (!sceneObject) {
            // Object not found and not loading - remove any existing dissolve
            this.logger.warn(
              `[DissolveEffect] ${objectId}: Object still not found after waiting, skipping`
            );
            if (existingDissolve) {
              this.logger.log(
                `[DissolveEffect] ${objectId}: Object not found and not loading, removing dissolve`
              );
              this.removeDissolve(objectId);
            }
            // Skip this object - use if/else instead of continue for Safari compatibility
          } else {
            // Object found, proceed with setup
            this._setupDissolveForObject(
              objectId,
              sceneObject,
              existingDissolve,
              modeChanged,
              mode,
              progress,
              targetProgress,
              transitionDuration,
              wipeDirection,
              wipeSoftness,
              edgeColor1,
              edgeColor2,
              particleColor,
              frequency,
              edgeWidth,
              particleIntensity,
              particleSize,
              particleDecimation,
              particleDispersion,
              particleVelocitySpread
            );
          }
        } else {
          // Object found immediately, proceed with setup
          this._setupDissolveForObject(
            objectId,
            sceneObject,
            existingDissolve,
            modeChanged,
            mode,
            progress,
            targetProgress,
            transitionDuration,
            wipeDirection,
            wipeSoftness,
            edgeColor1,
            edgeColor2,
            particleColor,
            frequency,
            edgeWidth,
            particleIntensity,
            particleSize,
            particleDecimation,
            particleDispersion,
            particleVelocitySpread
          );
        }
      }

      // Remove dissolves for objects no longer in target list
      const activeIds = Array.from(this.activeDissolves.keys());
      activeIds.forEach((objectId) => {
        if (!targetObjectIds.includes(objectId)) {
          this.logger.log(
            `[DissolveEffect] Removing dissolve for ${objectId} (no longer in target list)`
          );
          this.removeDissolve(objectId);
        }
      });

      this.logger.log(
        `[DissolveEffect] applyEffect completed for ${effect.id}`
      );
    } catch (error) {
      this.logger.error(
        `[DissolveEffect] Fatal error in applyEffect for ${effect.id}:`,
        error
      );
      this.logger.error(`[DissolveEffect] Error stack:`, error?.stack);
      throw error; // Re-throw so VFXManager can catch it
    }
  }

  /**
   * Helper method to setup dissolve for a single object (Safari-compatible refactor)
   * @private
   */
  _setupDissolveForObject(
    objectId,
    sceneObject,
    existingDissolve,
    modeChanged,
    mode,
    progress,
    targetProgress,
    transitionDuration,
    wipeDirection,
    wipeSoftness,
    edgeColor1,
    edgeColor2,
    particleColor,
    frequency,
    edgeWidth,
    particleIntensity,
    particleSize,
    particleDecimation,
    particleDispersion,
    particleVelocitySpread
  ) {
    if (!existingDissolve || modeChanged) {
      // Update existing dissolve in place if mode changed
      if (modeChanged) {
        this.logger.log(
          `Mode changed for ${objectId}, updating in place (${existingDissolve.mode} -> ${mode})`
        );
        this.logger.log(
          `New progress: ${progress} -> ${targetProgress}, duration: ${transitionDuration}`
        );

        // Update uniforms and state in place without recreating materials
        if (existingDissolve && existingDissolve.dissolveUniforms) {
          if (sceneObject) {
            const bbox = new THREE.Box3().setFromObject(sceneObject);
            const color1 = new THREE.Color(edgeColor1);
            const color2 = new THREE.Color(edgeColor2);

            // Update all uniforms in place
            existingDissolve.dissolveUniforms.uProgress.value = progress;
            existingDissolve.dissolveUniforms.uDissolveMode.value =
              mode === "wipe" ? 1.0 : 0.0;
            existingDissolve.dissolveUniforms.uWipeDirection.value =
              wipeDirection === "bottom-to-top" ? 0.0 : 1.0;
            existingDissolve.dissolveUniforms.uWipeSoftness.value =
              wipeSoftness;
            existingDissolve.dissolveUniforms.uWipeBounds.value.set(
              bbox.min.y,
              bbox.max.y
            );
            existingDissolve.dissolveUniforms.uEdgeColor1.value.set(
              color1.r,
              color1.g,
              color1.b
            );
            existingDissolve.dissolveUniforms.uEdgeColor2.value.set(
              color2.r,
              color2.g,
              color2.b
            );
            existingDissolve.dissolveUniforms.uFreq.value = frequency;
            existingDissolve.dissolveUniforms.uEdge.value = edgeWidth;

            // Update state
            existingDissolve.progress = progress;
            existingDissolve.targetProgress = targetProgress;
            existingDissolve.initialProgress = progress;
            existingDissolve.transitionStartTime = Date.now();
            existingDissolve.transitionDuration = transitionDuration;
            existingDissolve.mode = mode;

            // Remove old particles (wipe mode doesn't need them)
            if (existingDissolve.particleMesh) {
              this.scene.remove(existingDissolve.particleMesh);
              existingDissolve.particleMesh.geometry.dispose();
              existingDissolve.particleMesh.material.dispose();
              existingDissolve.particleMesh = null;
              existingDissolve.particleSystem = null;
            }

            this.logger.log(
              `✓ Updated in place: progress=${progress}, bounds=${bbox.min.y.toFixed(
                2
              )}-${bbox.max.y.toFixed(2)}`
            );
          }
        }
        return; // Don't recreate
      }

      this.logger.log(`[DissolveEffect] ${objectId}: Calling setupDissolve`);
      try {
        this.setupDissolve(
          objectId,
          progress,
          targetProgress,
          mode,
          wipeDirection,
          wipeSoftness,
          edgeColor1,
          edgeColor2,
          particleColor,
          frequency,
          edgeWidth,
          particleIntensity,
          particleSize,
          particleDecimation,
          particleDispersion,
          particleVelocitySpread
        );
        this.logger.log(
          `[DissolveEffect] ${objectId}: setupDissolve completed`
        );
      } catch (error) {
        this.logger.error(
          `[DissolveEffect] ${objectId}: Error in setupDissolve:`,
          error
        );
        this.logger.error(
          `[DissolveEffect] ${objectId}: Error stack:`,
          error?.stack
        );
      }
    } else {
      // Update existing dissolve parameters (same mode)
      this.logger.log(
        `[DissolveEffect] ${objectId}: Updating existing dissolve parameters`
      );
      try {
        this.updateDissolveParams(
          existingDissolve,
          targetProgress,
          edgeColor1,
          edgeColor2,
          particleColor,
          frequency,
          edgeWidth
        );
        this.logger.log(
          `[DissolveEffect] ${objectId}: updateDissolveParams completed`
        );
      } catch (error) {
        this.logger.error(
          `[DissolveEffect] ${objectId}: Error in updateDissolveParams:`,
          error
        );
        this.logger.error(
          `[DissolveEffect] ${objectId}: Error stack:`,
          error?.stack
        );
      }
    }
  }

  /**
   * Setup dissolve effect for a specific object
   * @param {string} objectId - Scene object ID
   * @param {number} progress - Initial progress value
   * @param {string} edgeColor1 - Primary edge color (hex)
   * @param {string} edgeColor2 - Secondary edge color (hex)
   * @param {string} particleColor - Particle color (hex)
   * @param {number} frequency - Noise frequency
   * @param {number} edgeWidth - Edge glow width
   */
  setupDissolve(
    objectId,
    progress,
    targetProgress,
    mode = "noise",
    wipeDirection = "bottom-to-top",
    wipeSoftness = 0.15,
    edgeColor1,
    edgeColor2,
    particleColor,
    frequency,
    edgeWidth,
    particleIntensity = 1.0,
    particleSize = 35.0,
    particleDecimation = 10,
    particleDispersion = 8.0,
    particleVelocitySpread = 0.15
  ) {
    this.logger.log(`Setting up dissolve for object: ${objectId}`);

    // Get object from scene manager
    const sceneObject = this.sceneManager.getObject(objectId);
    if (!sceneObject) {
      this.logger.error(`Object not found: ${objectId}`);
      // Remove any existing dissolve if object doesn't exist
      if (this.activeDissolves.has(objectId)) {
        this.removeDissolve(objectId);
      }
      return;
    }

    // Find and disable contact shadow if it exists
    // SceneManager stores contact shadows in a Map by objectId
    let contactShadow = null;
    if (this.sceneManager && this.sceneManager.contactShadows) {
      contactShadow = this.sceneManager.contactShadows.get(objectId);
    }

    if (contactShadow) {
      if (typeof contactShadow.disable === "function") {
        this.logger.log(`Disabling contact shadow for ${objectId}`);
        contactShadow.disable();
      } else {
        this.logger.warn(
          `Contact shadow found for ${objectId} but no disable() method`
        );
      }
    } else {
      this.logger.log(`No contact shadow found for ${objectId}`);
    }

    // Convert colors
    const color1 = new THREE.Color(edgeColor1);
    const color2 = new THREE.Color(edgeColor2);
    const pColor = new THREE.Color(particleColor);

    // Calculate world-space bounding box for wipe effect
    const bbox = new THREE.Box3().setFromObject(sceneObject);
    const minY = bbox.min.y;
    const maxY = bbox.max.y;

    this.logger.log(
      `Object bounds: Y range ${minY.toFixed(2)} to ${maxY.toFixed(2)}`
    );

    // Create shared uniforms for dissolve shader
    const dissolveUniforms = {
      uEdgeColor1: { value: new THREE.Vector3(color1.r, color1.g, color1.b) },
      uEdgeColor2: { value: new THREE.Vector3(color2.r, color2.g, color2.b) },
      uFreq: { value: frequency },
      uAmp: { value: 16.0 },
      uProgress: { value: progress },
      uEdge: { value: edgeWidth },
      uDissolveMode: { value: mode === "wipe" ? 1.0 : 0.0 },
      uWipeDirection: { value: wipeDirection === "bottom-to-top" ? 0.0 : 1.0 },
      uWipeSoftness: { value: wipeSoftness },
      uWipeBounds: { value: new THREE.Vector2(minY, maxY) },
    };

    // Helper to check if mesh is part of contact shadow
    const isContactShadowMesh = (mesh) => {
      // Check if mesh or any parent is a contact shadow
      let current = mesh;
      let depth = 0;
      while (current && depth < 10) {
        // Limit depth to avoid infinite loops
        if (
          current.userData.isContactShadow ||
          current.name === "contactShadow" ||
          current.name.includes("ContactShadow") ||
          current.name.toLowerCase().includes("shadow")
        ) {
          this.logger.log(
            `Skipping contact shadow mesh: ${mesh.name || "unnamed"} (parent: ${
              current.name
            })`
          );
          return true;
        }
        current = current.parent;
        depth++;
      }
      return false;
    };

    // Apply shader to all materials in the object
    const modifiedMaterials = [];
    let materialMeshCount = 0;
    const meshesToProcess = [];

    // Collect meshes from root object
    sceneObject.traverse((child) => {
      if (child.isMesh && child.material) {
        meshesToProcess.push(child);
      }
    });

    // Process all collected meshes
    for (const child of meshesToProcess) {
      materialMeshCount++;
      // Skip contact shadow meshes by checking parent hierarchy
      if (isContactShadowMesh(child)) {
        this.logger.log(
          `Skipping contact shadow mesh for materials: ${
            child.name || "unnamed"
          }`
        );
        continue;
      }

      const materials = Array.isArray(child.material)
        ? child.material
        : [child.material];

      materials.forEach((mat) => {
        // Store original onBeforeCompile if not already stored
        if (!mat.userData.originalOnBeforeCompile) {
          mat.userData.originalOnBeforeCompile = mat.onBeforeCompile;
        }

        const originalOnBeforeCompile = mat.userData.originalOnBeforeCompile;

        mat.onBeforeCompile = (shader) => {
          // Call original if exists
          if (originalOnBeforeCompile) {
            originalOnBeforeCompile(shader);
          }

          // Inject dissolve shader
          setupUniforms(shader, dissolveUniforms);
          setupShaderSnippets(
            shader,
            vertexGlobal,
            vertexMain,
            perlinNoise + fragmentGlobal,
            fragmentMain
          );
        };

        mat.userData.dissolveModified = true;
        mat.userData.dissolveUniforms = dissolveUniforms;

        // Force material version bump to invalidate Three.js shader cache
        mat.version++;
        mat.needsUpdate = true;

        if (!modifiedMaterials.includes(mat)) {
          modifiedMaterials.push(mat);
        }
      });
    }

    this.logger.log(
      `Applied dissolve shader to ${modifiedMaterials.length} material(s) from ${materialMeshCount} mesh(es) for ${objectId}`
    );

    // Merge geometries from all meshes for particle system
    // With useContainer: true, meshes are nested inside the GLTF model child
    const geometries = [];
    let meshCount = 0;
    sceneObject.traverse((child) => {
      if (child.isMesh && child.geometry) {
        meshCount++;
        // Skip contact shadow meshes by checking parent hierarchy
        if (isContactShadowMesh(child)) {
          this.logger.log(
            `Skipping contact shadow mesh: ${child.name || "unnamed"}`
          );
          return;
        }

        const geo = child.geometry.clone();
        child.updateMatrixWorld(true);
        geo.applyMatrix4(child.matrixWorld);
        geometries.push(geo);
        this.logger.log(
          `Found mesh geometry: ${child.name || "unnamed"} (${
            geo.attributes.position.count
          } vertices)`
        );
      }
    });

    this.logger.log(
      `Traverse found ${meshCount} total meshes, ${geometries.length} valid geometries for ${objectId}`
    );

    if (geometries.length === 0) {
      // Remove any existing dissolve since we can't set it up
      if (this.activeDissolves.has(objectId)) {
        this.removeDissolve(objectId);
      }
      return;
    }

    const mergedGeo =
      geometries.length > 1
        ? BufferGeometryUtils.mergeGeometries(geometries)
        : geometries[0];

    // Decimate geometry to reduce particle count dramatically
    // Only use every Nth vertex for particles (e.g., every 10th = 90% reduction)
    const decimationFactor = particleDecimation; // Higher = fewer particles
    const originalPositions = mergedGeo.getAttribute("position");
    const decimatedCount = Math.floor(
      originalPositions.count / decimationFactor
    );
    const decimatedPositions = new Float32Array(decimatedCount * 3);

    for (let i = 0; i < decimatedCount; i++) {
      const sourceIndex = i * decimationFactor;
      decimatedPositions[i * 3 + 0] = originalPositions.getX(sourceIndex);
      decimatedPositions[i * 3 + 1] = originalPositions.getY(sourceIndex);
      decimatedPositions[i * 3 + 2] = originalPositions.getZ(sourceIndex);
    }

    const decimatedGeo = new THREE.BufferGeometry();
    decimatedGeo.setAttribute(
      "position",
      new THREE.BufferAttribute(decimatedPositions, 3)
    );

    this.logger.log(
      `Decimated particles: ${originalPositions.count} -> ${decimatedCount} (${decimationFactor}x reduction)`
    );

    // Create particle system with decimated geometry and custom parameters
    // Skip if particle size is 0 or decimation is too high (wipe mode doesn't need particles)
    const particleSystem =
      particleSize > 0 && decimationFactor < 100
        ? new DissolveParticleSystem(
            decimatedGeo,
            particleDispersion,
            particleVelocitySpread
          )
        : null;

    // Create particle uniforms
    const particleUniforms = {
      uPixelDensity: { value: this.renderer.getPixelRatio() },
      uBaseSize: { value: particleSize },
      uFreq: dissolveUniforms.uFreq,
      uAmp: dissolveUniforms.uAmp,
      uEdge: dissolveUniforms.uEdge,
      uColor: {
        value: new THREE.Vector3(
          pColor.r * particleIntensity,
          pColor.g * particleIntensity,
          pColor.b * particleIntensity
        ),
      },
      uProgress: dissolveUniforms.uProgress,
      uParticleTexture: { value: this.particleTexture },
    };

    // Create particle material
    const particleMaterial = new THREE.ShaderMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      uniforms: particleUniforms,
      vertexShader: particleVertexShader,
      fragmentShader: particleFragmentShader,
      depthWrite: false,
      depthTest: true, // Enable depth test so renderOrder works correctly
    });

    // Create particle mesh (only if we have a particle system)
    let particleMesh = null;
    if (particleSystem) {
      // NOTE: Geometry already has world transforms baked in, so don't apply parent transform again
      particleMesh = new THREE.Points(decimatedGeo, particleMaterial);
      // Don't copy position/rotation/scale - geometry is already in world space

      // Set high render order to render on top of splat scenes (SparkRenderer is 9998)
      particleMesh.renderOrder = 9999;

      this.scene.add(particleMesh);
    }

    // Store dissolve state
    this.activeDissolves.set(objectId, {
      object: sceneObject,
      dissolveUniforms,
      particleSystem,
      particleMesh,
      particleUniforms,
      modifiedMaterials,
      progress,
      targetProgress,
      initialProgress: progress, // Store starting point
      transitionStartTime: Date.now(), // Track when transition started
      transitionDuration: this.transitionDuration, // Store per-dissolve duration
      contactShadow, // Store reference to re-enable later
      contactShadowEnabled: false, // Track enabled state (starts disabled since we disable at setup)
      mode, // Store mode for change detection
    });

    this.logger.log(
      `Dissolve setup complete for ${objectId} (${modifiedMaterials.length} materials, ${geometries.length} meshes)`
    );
  }

  /**
   * Update dissolve parameters for existing dissolve
   * @param {Object} dissolveState - Dissolve state object
   * @param {number} targetProgress - New target progress
   * @param {string} edgeColor1 - Primary edge color (hex)
   * @param {string} edgeColor2 - Secondary edge color (hex)
   * @param {string} particleColor - Particle color (hex)
   * @param {number} frequency - Noise frequency
   * @param {number} edgeWidth - Edge glow width
   */
  updateDissolveParams(
    dissolveState,
    targetProgress,
    edgeColor1,
    edgeColor2,
    particleColor,
    frequency,
    edgeWidth
  ) {
    const color1 = new THREE.Color(edgeColor1);
    const color2 = new THREE.Color(edgeColor2);
    const pColor = new THREE.Color(particleColor);

    dissolveState.dissolveUniforms.uEdgeColor1.value.set(
      color1.r,
      color1.g,
      color1.b
    );
    dissolveState.dissolveUniforms.uEdgeColor2.value.set(
      color2.r,
      color2.g,
      color2.b
    );
    dissolveState.dissolveUniforms.uFreq.value = frequency;
    dissolveState.dissolveUniforms.uEdge.value = edgeWidth;

    // Update target progress and reset transition timing
    if (dissolveState.targetProgress !== targetProgress) {
      this.logger.log(
        `Updating target progress for dissolve: ${dissolveState.progress} -> ${targetProgress}`
      );
      dissolveState.targetProgress = targetProgress;
      dissolveState.initialProgress = dissolveState.progress;
      dissolveState.transitionStartTime = Date.now();
      dissolveState.transitionDuration = this.transitionDuration;
    }

    if (dissolveState.particleUniforms) {
      dissolveState.particleUniforms.uColor.value.set(
        pColor.r,
        pColor.g,
        pColor.b
      );
    }
  }

  /**
   * Remove dissolve effect from an object
   * @param {string} objectId - Scene object ID
   */
  removeDissolve(objectId) {
    const dissolveState = this.activeDissolves.get(objectId);
    if (!dissolveState) return;

    this.logger.log(`Removing dissolve from object: ${objectId}`);

    // Re-enable contact shadow if it was disabled
    if (
      dissolveState.contactShadow &&
      typeof dissolveState.contactShadow.enable === "function"
    ) {
      this.logger.log(`Re-enabling contact shadow for ${objectId}`);
      dissolveState.contactShadow.enable();
    }

    // Remove particle mesh (if exists)
    if (dissolveState.particleMesh) {
      this.scene.remove(dissolveState.particleMesh);
      dissolveState.particleMesh.geometry.dispose();
      dissolveState.particleMesh.material.dispose();
    }

    // Reset materials - restore original state
    dissolveState.modifiedMaterials.forEach((mat) => {
      // Restore original onBeforeCompile
      const originalOnBeforeCompile = mat.userData.originalOnBeforeCompile;
      if (originalOnBeforeCompile) {
        mat.onBeforeCompile = originalOnBeforeCompile;
      } else {
        mat.onBeforeCompile = () => {};
      }

      // Clear user data
      mat.userData.dissolveModified = false;
      mat.userData.dissolveUniforms = null;
      mat.userData.originalOnBeforeCompile = null;

      // Force material version bump to invalidate Three.js shader cache
      mat.version++;

      // Force material to recompile with original shader
      mat.needsUpdate = true;
    });

    // Clear all references to allow object to be garbage collected
    dissolveState.object = null;
    dissolveState.modifiedMaterials = null;
    dissolveState.dissolveUniforms = null;
    dissolveState.particleSystem = null;
    dissolveState.particleUniforms = null;
    dissolveState.contactShadow = null;

    this.activeDissolves.delete(objectId);
  }

  /**
   * Update dissolve animation
   * @param {number} deltaTime - Time since last frame in seconds
   */
  update(deltaTime) {
    if (this.activeDissolves.size === 0) {
      if (this.enableAudio && this.audio.audioContext && this.audio.isPlaying) {
        this.audio.stop();
      }
      return;
    }

    // Clean up dissolves for objects that no longer exist in scene manager
    const activeIds = Array.from(this.activeDissolves.keys());
    activeIds.forEach((objectId) => {
      const sceneObject = this.sceneManager.getObject(objectId);
      if (!sceneObject) {
        this.logger.log(
          `Object ${objectId} no longer exists, removing dissolve`
        );
        this.removeDissolve(objectId);
      }
    });

    if (this.activeDissolves.size === 0) {
      if (this.enableAudio && this.audio.audioContext && this.audio.isPlaying) {
        this.audio.stop();
      }
      return;
    }

    let anyTransitioning = false;
    let maxAbsProgress = 0;

    this.activeDissolves.forEach((dissolveState, objectId) => {
      // Animate towards target progress
      const currentProgress = dissolveState.progress;
      const targetProgress =
        dissolveState.targetProgress !== undefined
          ? dissolveState.targetProgress
          : this.targetProgress;

      if (currentProgress !== targetProgress) {
        let newProgress = currentProgress;

        if (this.autoAnimate) {
          // Auto-animate bounce-back behavior (for demo/testing)
          if (this.animationDirection === 1) {
            newProgress += 0.15; // Fixed increment
          } else {
            newProgress -= 0.15;
          }

          // Bounce at limits
          if (newProgress >= 14.0) {
            this.animationDirection = -1;
            newProgress = 14.0;
          } else if (newProgress <= -14.0) {
            this.animationDirection = 1;
            newProgress = -14.0;
          }
        } else {
          // One-way transition towards target (time-based)
          const duration =
            dissolveState.transitionDuration || this.transitionDuration || 0;
          if (duration > 0) {
            const elapsed =
              (Date.now() - dissolveState.transitionStartTime) / 1000;
            const t = Math.min(elapsed / duration, 1.0);
            newProgress =
              dissolveState.initialProgress +
              (targetProgress - dissolveState.initialProgress) * t;

            // Log animation progress occasionally
            if (Math.random() < 0.02) {
              this.logger.log(
                `⏱️ Animating ${objectId}: ${currentProgress.toFixed(
                  2
                )} -> ${targetProgress.toFixed(2)} (t=${t.toFixed(
                  2
                )}, duration=${duration}s)`
              );
            }
          } else {
            // Instant transition if no duration specified
            newProgress = targetProgress;
          }
        }

        dissolveState.progress = newProgress;
        dissolveState.dissolveUniforms.uProgress.value = newProgress;
      }

      // Update particle system visibility and animation
      if (dissolveState.particleSystem && dissolveState.particleMesh) {
        // Only show/update particles during the dissolve transition (not at extremes)
        const inTransition = currentProgress > -14.0 && currentProgress < 14.0;
        dissolveState.particleMesh.visible = inTransition;

        // Only update particle animation if visible
        if (inTransition) {
          dissolveState.particleSystem.updateAttributesValues();
        }
      }

      // Re-enable contact shadow when object becomes fully visible again (wipe-in complete)
      if (dissolveState.contactShadow && currentProgress <= -14.0) {
        if (!dissolveState.contactShadowEnabled) {
          this.logger.log(
            `Wipe-in complete, re-enabling contact shadow for ${objectId}`
          );
          dissolveState.contactShadow.enable();
          dissolveState.contactShadowEnabled = true; // Mark as enabled
        }
      }
      // Disable contact shadow when dissolve transition starts (fade out early)
      else if (dissolveState.contactShadow && currentProgress > -14.0) {
        if (dissolveState.contactShadowEnabled) {
          this.logger.log(
            `Dissolve started, disabling contact shadow for ${objectId} (progress: ${currentProgress.toFixed(
              2
            )})`
          );
          dissolveState.contactShadow.disable();
          dissolveState.contactShadowEnabled = false; // Mark as disabled
        }
      }

      // Track if any dissolve is actively transitioning
      const inTransition = currentProgress > -14.0 && currentProgress < 14.0;
      if (inTransition) {
        anyTransitioning = true;
        maxAbsProgress = Math.max(maxAbsProgress, Math.abs(currentProgress));
      }
    });

    // Audio: Play during active transitions, update based on progress
    // Check both enableAudio and !suppressAudio (suppressAudio is for reverse transitions)
    if (this.enableAudio && !this.suppressAudio) {
      // Ensure audio context is initialized and valid
      if (!this.audio.audioContext) {
        // Audio was stopped/closed, need to re-initialize
        this.audio
          .initialize()
          .then(() => {
            this.logger.log("Audio re-initialized in update loop");
          })
          .catch((err) => {
            this.logger.warn(`Failed to re-initialize audio: ${err}`);
          });
      } else if (this.audio.audioContext.state === "suspended") {
        // Resume suspended audio context (browser may suspend it after inactivity)
        this.audio.audioContext
          .resume()
          .then(() => {
            this.logger.log("Audio context resumed in update loop");
          })
          .catch((err) => {
            this.logger.warn(`Failed to resume audio context: ${err}`);
          });
      }

      if (
        this.audio.audioContext &&
        this.audio.audioContext.state === "running" &&
        anyTransitioning
      ) {
        if (!this.audio.isPlaying && !this.audio.pendingStart) {
          // start() is now async - handle it non-blocking
          this.audio.start().catch((error) => {
            this.logger.warn("Failed to start audio in update loop:", error);
          });
          this.logger.log("Audio start attempted in update loop");
        }
        this._updateAudio(maxAbsProgress, deltaTime);
      } else if (this.audio.isPlaying && !anyTransitioning) {
        this.audio.stop();
      }
    } else if (this.audio.audioContext && this.audio.isPlaying) {
      // If audio is suppressed or disabled, stop it if playing
      this.audio.stop();
    }
  }

  /**
   * Update audio parameters based on dissolve progress
   * @private
   */
  _updateAudio(absProgress, deltaTime) {
    const normalizedProgress = Math.min(absProgress / 14.0, 1.0);

    const velocity =
      Math.abs(normalizedProgress - this.lastProgress) /
      Math.max(deltaTime, 0.001);
    const normalizedVelocity = Math.min(velocity * 2, 1);

    const minFreq = 3000;
    const maxFreq = 6000;
    const targetFilterFreq = minFreq + (maxFreq - minFreq) * normalizedProgress;

    const minQ = 2;
    const maxQ = 6;
    const targetQ = minQ + (maxQ - minQ) * normalizedVelocity;

    const pitchMultiplier = 1.0 + normalizedVelocity * 0.15;

    const panAmount = Math.sin(normalizedProgress * Math.PI * 2) * 0.4;

    // Keep volume around 0.3 of max with minimal variation
    const baseVolumeLevel = 0.3; // Stay around 30% of max
    const variationAmount = 0.15; // ±15% variation
    const variation =
      Math.sin(normalizedProgress * Math.PI * 4) * variationAmount;
    const velocityVariation = normalizedVelocity * 0.1; // Small boost during fast changes
    const targetVolume =
      this.audio.config.volume *
      (baseVolumeLevel + variation + velocityVariation);

    const sweepAmount = normalizedProgress * normalizedProgress;

    this.audio.updateParams({
      filterFreq: targetFilterFreq,
      filterQ: targetQ,
      pitchMultiplier,
      pan: panAmount,
      volume: targetVolume,
      sweepAmount: sweepAmount,
      transitionTime: 0.1,
    });

    this.lastProgress = normalizedProgress;
  }

  /**
   * Wait for an object to be loaded by sceneManager
   * Handles race condition where updateSceneForState hasn't started loading yet
   * @private
   */
  async _waitForObject(objectId) {
    this.logger.log(
      `[DissolveEffect] _waitForObject: Waiting for object: ${objectId}`
    );

    // Check if already loaded
    let object = this.sceneManager.getObject(objectId);
    if (object) {
      this.logger.log(
        `[DissolveEffect] _waitForObject: Object ${objectId} already loaded`
      );
      return object;
    }

    // Check if object should be loaded for current state
    // This helps handle race conditions where updateSceneForState hasn't started yet
    let shouldBeLoaded = false;
    if (this.gameManager) {
      const currentState = this.gameManager.getState();
      const objectsForState = getSceneObjectsForState(currentState);
      shouldBeLoaded = objectsForState.some((obj) => obj.id === objectId);
      this.logger.log(
        `[DissolveEffect] _waitForObject: ${objectId} shouldBeLoaded=${shouldBeLoaded} (state: ${currentState?.currentState})`
      );
    } else {
      this.logger.warn(
        `[DissolveEffect] _waitForObject: gameManager not available`
      );
    }

    // Wait for it to load
    const maxWaitTime = 10000; // 10 seconds timeout
    const startTime = Date.now();
    let hasCheckedLoading = false;

    while (!object && Date.now() - startTime < maxWaitTime) {
      // Check if it's loading
      if (this.sceneManager.isLoading(objectId)) {
        hasCheckedLoading = true;
        this.logger.log(
          `[DissolveEffect] _waitForObject: ${objectId} is loading...`
        );
        try {
          const loadingPromise =
            this.sceneManager.loadingPromises.get(objectId);
          if (loadingPromise) {
            this.logger.log(
              `[DissolveEffect] _waitForObject: ${objectId} Found loading promise, awaiting...`
            );
            await loadingPromise;
            object = this.sceneManager.getObject(objectId);
            this.logger.log(
              `[DissolveEffect] _waitForObject: ${objectId} Loading promise resolved, object=${!!object}`
            );
            break;
          } else {
            // Promise not found in map (shouldn't happen if isLoading is true, but handle it)
            this.logger.warn(
              `[DissolveEffect] _waitForObject: Loading promise not found for ${objectId} despite isLoading=true`
            );
          }
        } catch (error) {
          this.logger.error(
            `[DissolveEffect] _waitForObject: Error waiting for ${objectId} to load:`,
            error
          );
          this.logger.error(
            `[DissolveEffect] _waitForObject: Error stack:`,
            error?.stack
          );
          return null;
        }
      }

      // If object should be loaded but isn't loading yet, wait a bit for updateSceneForState to start
      // This handles the race condition where state changed but scene loading hasn't started
      if (shouldBeLoaded && !hasCheckedLoading) {
        const elapsed = Date.now() - startTime;
        // Give updateSceneForState up to 500ms to start loading
        if (elapsed < 500) {
          this.logger.log(
            `  ${objectId} should be loaded for current state, waiting for scene manager to start loading...`
          );
          await new Promise((resolve) => setTimeout(resolve, 100));
          continue;
        } else {
          hasCheckedLoading = true;
          // After 500ms, if it's still not loading, it might not be loaded for this state
          // or there's an issue - continue waiting anyway in case it starts
        }
      }

      // Check again if it's now available
      object = this.sceneManager.getObject(objectId);
      if (object) {
        break;
      }

      // Wait a bit before checking again
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    if (!object) {
      this.logger.error(`Timeout waiting for ${objectId} to load`);
      return null;
    }

    this.logger.log(`Object ${objectId} loaded successfully`);
    return object;
  }

  /**
   * Called when no effect matches (cleanup)
   * @param {Object} state - Current game state
   */
  onNoEffect(state) {
    this.logger.log(
      `❌ No effect matches (state: ${state?.currentState}) - cleaning up dissolves`
    );

    // Stop audio
    if (this.audio.audioContext && this.audio.isPlaying) {
      this.audio.stop();
    }

    // Reset flags so next effect can enable audio fresh
    this.enableAudio = false;
    this.suppressAudio = false;

    // Remove all active dissolves
    const activeIds = Array.from(this.activeDissolves.keys());
    activeIds.forEach((objectId) => {
      this.removeDissolve(objectId);
    });
  }

  /**
   * Dispose of all resources
   */
  dispose() {
    // Remove all dissolves
    const activeIds = Array.from(this.activeDissolves.keys());
    activeIds.forEach((objectId) => {
      this.removeDissolve(objectId);
    });

    if (this.particleTexture) {
      this.particleTexture.dispose();
    }

    if (this.audio && this.audio.audioContext) {
      this.audio.dispose();
      this.audio = null;
    }
  }
}

export default DissolveEffect;
