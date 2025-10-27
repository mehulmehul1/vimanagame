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

export class DissolveEffect extends VFXManager {
  constructor(scene, sceneManager, renderer) {
    super("DissolveEffect", true);

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

    this.logger.log("DissolveEffect initialized");
  }

  /**
   * Apply dissolve effect based on effect parameters
   * @param {Object} effect - Effect data from vfxData.js
   * @param {Object} state - Current game state
   */
  applyEffect(effect, state) {
    const params = effect.parameters || {};

    this.logger.log(
      `🔥 Applying effect: ${effect.id} (state: ${state?.currentState})`,
      params
    );

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
    const frequency = params.frequency !== undefined ? params.frequency : 0.45;
    const edgeWidth = params.edgeWidth !== undefined ? params.edgeWidth : 0.8;
    const particleIntensity =
      params.particleIntensity !== undefined ? params.particleIntensity : 1.0;

    // Particle system parameters
    const particleSize =
      params.particleSize !== undefined ? params.particleSize : 35.0;
    const particleDecimation =
      params.particleDecimation !== undefined ? params.particleDecimation : 10;
    const particleDispersion =
      params.particleDispersion !== undefined ? params.particleDispersion : 8.0;
    const particleVelocitySpread =
      params.particleVelocitySpread !== undefined
        ? params.particleVelocitySpread
        : 0.15;

    // Store animation settings
    this.autoAnimate = autoAnimate;
    this.targetProgress = targetProgress;
    this.transitionDuration = transitionDuration;

    // Setup dissolve for each target object
    targetObjectIds.forEach((objectId) => {
      const existingDissolve = this.activeDissolves.get(objectId);

      // Check if mode changed - if so, need to recreate
      const modeChanged = existingDissolve && existingDissolve.mode !== mode;

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
            const sceneObject = this.sceneManager.getObject(objectId);
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
      } else {
        // Update existing dissolve parameters (same mode)
        this.updateDissolveParams(
          existingDissolve,
          targetProgress,
          edgeColor1,
          edgeColor2,
          particleColor,
          frequency,
          edgeWidth
        );
      }
    });

    // Remove dissolves for objects no longer in target list
    const activeIds = Array.from(this.activeDissolves.keys());
    activeIds.forEach((objectId) => {
      if (!targetObjectIds.includes(objectId)) {
        this.removeDissolve(objectId);
      }
    });
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
      // Contact shadows use renderOrder 10000
      if (mesh.renderOrder >= 10000) {
        this.logger.log(
          `Skipping mesh with renderOrder ${mesh.renderOrder}: ${
            mesh.name || "unnamed"
          }`
        );
        return true;
      }

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
    sceneObject.traverse((child) => {
      if (child.isMesh && child.material) {
        // Skip contact shadow meshes by checking parent hierarchy
        if (isContactShadowMesh(child)) {
          return;
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
    });

    // Merge geometries from all meshes for particle system
    const geometries = [];
    sceneObject.traverse((child) => {
      if (child.isMesh && child.geometry) {
        // Skip contact shadow meshes by checking parent hierarchy
        if (isContactShadowMesh(child)) {
          return;
        }

        const geo = child.geometry.clone();
        child.updateMatrixWorld(true);
        geo.applyMatrix4(child.matrixWorld);
        geometries.push(geo);
      }
    });

    if (geometries.length === 0) {
      this.logger.warn(`No geometries found for object: ${objectId}`);
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

    this.activeDissolves.delete(objectId);
  }

  /**
   * Update dissolve animation
   * @param {number} deltaTime - Time since last frame in seconds
   */
  update(deltaTime) {
    if (this.activeDissolves.size === 0) return;

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
      // Disable contact shadow when object is fully dissolved
      else if (dissolveState.contactShadow && currentProgress >= 14.0) {
        if (dissolveState.contactShadowEnabled) {
          this.logger.log(
            `Dissolve complete, disabling contact shadow for ${objectId}`
          );
          dissolveState.contactShadow.disable();
          dissolveState.contactShadowEnabled = false; // Mark as disabled
        }
      }
    });
  }

  /**
   * Called when no effect matches (cleanup)
   * @param {Object} state - Current game state
   */
  onNoEffect(state) {
    this.logger.log(
      `❌ No effect matches (state: ${state?.currentState}) - cleaning up dissolves`
    );

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
  }
}

export default DissolveEffect;
