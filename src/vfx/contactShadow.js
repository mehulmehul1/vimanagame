import * as THREE from "three";
import { Logger } from "../utils/logger.js";
import { HorizontalBlurShader } from "three/examples/jsm/shaders/HorizontalBlurShader.js";
import { VerticalBlurShader } from "three/examples/jsm/shaders/VerticalBlurShader.js";

// Create module-level logger
const logger = new Logger("ContactShadow", false);

/**
 * ContactShadow
 *
 * Creates contact shadows underneath 3D objects using depth rendering.
 * More performant and simpler than traditional shadow maps - no lights required!
 *
 * Based on the three.js contact shadows example.
 *
 * Performance: Each shadow renders 3 times per update (depth + 2 blur passes).
 * Use isStatic:true for objects that never move, or updateFrequency to throttle updates.
 *
 * Usage:
 * import { ContactShadow } from './vfx/contactShadow.js';
 *
 * // Static object (renders once, never updates):
 * const staticShadow = new ContactShadow(renderer, scene, parentObject, {
 *   size: { x: 0.5, y: 0.5 },
 *   blur: 3.5,
 *   darkness: 1.5,
 *   opacity: 0.5,
 *   fadeDuration: 0.3,  // Fade in/out duration (seconds)
 *   isStatic: true  // Render once and stop
 * });
 *
 * // Animated object (updates every frame):
 * const dynamicShadow = new ContactShadow(renderer, scene, parentObject, {
 *   updateFrequency: 1,  // Every frame
 *   trackMesh: "CarMesh"
 * });
 *
 * // In animation loop (call update before render):
 * contactShadow.update(deltaTime);  // For fade animations
 * contactShadow.render();
 *
 * // Enable/disable with fade:
 * contactShadow.enable();  // Fade in
 * contactShadow.disable(); // Fade out
 *
 * // For static objects that need to re-render after moving:
 * staticShadow.requestUpdate();
 */

/**
 * ContactShadow class - renders contact shadows for an object using depth rendering
 */
export class ContactShadow {
  constructor(renderer, scene, parentObject, config = {}) {
    this.renderer = renderer;
    this.scene = scene;
    this.parentObject = parentObject;

    const {
      size = { x: 0.5, y: 0.5 },
      offset = { x: 0, y: -0.05, z: 0 },
      blur = 3.5,
      darkness = 1.5,
      opacity = 0.5,
      renderTargetSize = 512,
      cameraHeight = 0.25,
      name = "contactShadow",
      debug = false,
      trackMesh = null, // Optional: name of specific child mesh to track (for animated models)
      updateFrequency = 3, // Update every N frames (1 = every frame, higher = better performance)
      isStatic = false, // If true, render once and never again (for static objects)
      fadeDuration = 0.3, // Fade in/out duration in seconds
    } = config;

    this.config = {
      size,
      offset,
      blur,
      darkness,
      opacity,
      cameraHeight,
      debug,
      isStatic,
    };

    logger.log(
      `Creating contact shadow "${name}" with size=(${size.x}, ${
        size.y
      }), blur=${blur}, darkness=${darkness}, ${
        isStatic
          ? "static (render once)"
          : `updateFrequency=${updateFrequency} (${
              updateFrequency === 1
                ? "every frame"
                : `every ${updateFrequency} frames`
            })`
      }`
    );

    // Create the shadow group
    this.shadowGroup = new THREE.Group();
    this.shadowGroup.position.set(offset.x, offset.y, offset.z);
    this.shadowGroup.name = name;

    // Add to parent
    parentObject.add(this.shadowGroup);

    // Keep shadow horizontal but preserve Y-axis rotation (heading)
    // Get parent's world rotation and extract only Y component
    const parentWorldQuaternion = new THREE.Quaternion();
    parentObject.getWorldQuaternion(parentWorldQuaternion);

    // Convert to euler to extract individual axes
    const worldEuler = new THREE.Euler();
    worldEuler.setFromQuaternion(parentWorldQuaternion, "YXZ");

    // Create a quaternion with only Y rotation
    const yOnlyQuaternion = new THREE.Quaternion();
    yOnlyQuaternion.setFromEuler(new THREE.Euler(0, worldEuler.y, 0, "YXZ"));

    // Get parent's world quaternion and invert it, then apply Y-only rotation
    const inverseParent = parentWorldQuaternion.clone().invert();
    this.shadowGroup.quaternion.copy(inverseParent).multiply(yOnlyQuaternion);

    logger.log(
      `Shadow horizontal with Y rotation: ${worldEuler.y.toFixed(2)} rad`
    );

    // Render targets for depth and blur
    this.renderTarget = new THREE.WebGLRenderTarget(
      renderTargetSize,
      renderTargetSize
    );
    this.renderTarget.texture.generateMipmaps = false;

    this.renderTargetBlur = new THREE.WebGLRenderTarget(
      renderTargetSize,
      renderTargetSize
    );
    this.renderTargetBlur.texture.generateMipmaps = false;

    // Create plane geometry (face up)
    const planeGeometry = new THREE.PlaneGeometry(size.x, size.y).rotateX(
      Math.PI / 2
    );

    // Main shadow plane (displays the rendered shadow texture)
    const planeMaterial = new THREE.MeshBasicMaterial({
      map: this.renderTarget.texture,
      opacity: debug ? 1.0 : opacity, // Full opacity in debug to see what's rendered
      transparent: true,
      depthWrite: false,
    });
    this.plane = new THREE.Mesh(planeGeometry, planeMaterial);
    this.plane.renderOrder = 10000; // Render after everything, including splats
    this.plane.scale.y = -1; // Flip Y from texture
    this.shadowGroup.add(this.plane);

    logger.log(
      `Shadow plane created at local position (${offset.x}, ${offset.y}, ${offset.z}), renderOrder: ${this.plane.renderOrder}`
    );

    // Blur plane (used for blur passes)
    this.blurPlane = new THREE.Mesh(planeGeometry);
    this.blurPlane.visible = false;
    this.shadowGroup.add(this.blurPlane);

    // Orthographic camera looking down
    this.shadowCamera = new THREE.OrthographicCamera(
      -size.x / 2,
      size.x / 2,
      size.y / 2,
      -size.y / 2,
      0,
      cameraHeight
    );
    this.shadowCamera.rotation.x = Math.PI / 2;
    this.shadowGroup.add(this.shadowCamera);

    // Camera helper for debugging
    if (debug) {
      this.cameraHelper = new THREE.CameraHelper(this.shadowCamera);
      scene.add(this.cameraHelper);
    }

    // Depth material (goes from transparent to black based on depth)
    this.depthMaterial = new THREE.MeshDepthMaterial();
    this.depthMaterial.userData.darkness = { value: darkness };
    this.depthMaterial.onBeforeCompile = (shader) => {
      shader.uniforms.darkness = this.depthMaterial.userData.darkness;
      shader.fragmentShader = /* glsl */ `
        uniform float darkness;
        ${shader.fragmentShader.replace(
          "gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );",
          "gl_FragColor = vec4( vec3( 0.0 ), ( 1.0 - fragCoordZ ) * darkness );"
        )}
      `;
    };
    this.depthMaterial.depthTest = false;
    this.depthMaterial.depthWrite = false;

    // Blur materials
    this.horizontalBlurMaterial = new THREE.ShaderMaterial(
      HorizontalBlurShader
    );
    this.horizontalBlurMaterial.depthTest = false;

    this.verticalBlurMaterial = new THREE.ShaderMaterial(VerticalBlurShader);
    this.verticalBlurMaterial.depthTest = false;

    logger.log(`Contact shadow "${name}" created successfully`);

    // Initialize enabled state (default to true)
    this.enabled = true;

    // Fade animation properties
    this.fadeDuration = config.fadeDuration || 0.3; // Default 0.3 seconds
    this.fadeState = "visible"; // "visible", "fading-in", "fading-out", "hidden"
    this.fadeProgress = 1.0; // 0 = fully transparent, 1 = fully opaque
    this.targetOpacity = opacity; // Store original opacity

    // Performance optimization: track frame updates
    this.frameCounter = 0;
    this.updateFrequency = updateFrequency; // Update every N frames (1 = every frame, higher = better perf)
    this.needsUpdate = true; // Force first render
    this.hasRenderedOnce = false; // Track if static shadow has rendered

    // Store the actual model to track (for containers with useContainer: true)
    if (trackMesh) {
      // Search for specific mesh by name in the hierarchy
      let found = null;
      parentObject.traverse((child) => {
        if (child.name === trackMesh && !found) {
          found = child;
        }
      });
      this.trackedObject = found || parentObject;

      if (found) {
        logger.log(`Found and tracking mesh: "${trackMesh}"`);
      } else {
        logger.warn(`Mesh "${trackMesh}" not found, tracking parent instead`);
      }
    } else {
      // Find the first non-shadow child (the actual GLTF model)
      this.trackedObject =
        parentObject.children.find((child) => child !== this.shadowGroup) ||
        parentObject;
    }

    logger.log(
      `Tracking object: "${this.trackedObject.name || "unnamed"}" (type: ${
        this.trackedObject.type
      })`
    );
    logger.log(`Parent has ${parentObject.children.length} children`);
  }

  /**
   * Update fade animation
   * Call this in your animation loop before render()
   * @param {number} deltaTime - Time elapsed since last frame in seconds
   */
  update(deltaTime) {
    // Update fade animation
    if (this.fadeState === "fading-in") {
      this.fadeProgress += deltaTime / this.fadeDuration;
      if (this.fadeProgress >= 1.0) {
        this.fadeProgress = 1.0;
        this.fadeState = "visible";
        logger.log(`Shadow "${this.shadowGroup.name}" fade in complete`);
      }
      // Update plane opacity
      this.plane.material.opacity = this.targetOpacity * this.fadeProgress;
    } else if (this.fadeState === "fading-out") {
      this.fadeProgress -= deltaTime / this.fadeDuration;
      if (this.fadeProgress <= 0.0) {
        this.fadeProgress = 0.0;
        this.fadeState = "hidden";
        this.plane.visible = false;
        logger.log(`Shadow "${this.shadowGroup.name}" fade out complete`);
      }
      // Update plane opacity
      this.plane.material.opacity = this.targetOpacity * this.fadeProgress;
    }
  }

  /**
   * Enable the shadow with fade in
   */
  enable() {
    if (this.enabled && this.fadeState !== "hidden") return;

    this.enabled = true;
    this.fadeState = "fading-in";
    this.fadeProgress = 0.0;
    this.plane.material.opacity = 0.0;
    this.plane.visible = true;
    logger.log(`Shadow "${this.shadowGroup.name}" fading in`);
  }

  /**
   * Disable the shadow with fade out
   */
  disable() {
    if (!this.enabled && this.fadeState === "hidden") return;

    this.enabled = false;
    this.fadeState = "fading-out";
    this.fadeProgress = 1.0;
    logger.log(`Shadow "${this.shadowGroup.name}" fading out`);
  }

  /**
   * Blur the shadow texture
   * @param {number} amount - Blur amount
   * @private
   */
  blurShadow(amount) {
    this.blurPlane.visible = true;

    // Horizontal blur pass
    this.blurPlane.material = this.horizontalBlurMaterial;
    this.blurPlane.material.uniforms.tDiffuse.value = this.renderTarget.texture;
    this.horizontalBlurMaterial.uniforms.h.value = (amount * 1) / 256;

    this.renderer.setRenderTarget(this.renderTargetBlur);
    this.renderer.render(this.blurPlane, this.shadowCamera);

    // Vertical blur pass
    this.blurPlane.material = this.verticalBlurMaterial;
    this.blurPlane.material.uniforms.tDiffuse.value =
      this.renderTargetBlur.texture;
    this.verticalBlurMaterial.uniforms.v.value = (amount * 1) / 256;

    this.renderer.setRenderTarget(this.renderTarget);
    this.renderer.render(this.blurPlane, this.shadowCamera);

    this.blurPlane.visible = false;
  }

  /**
   * Render the contact shadow
   * Call this in your animation loop (after update())
   */
  render() {
    // Skip rendering if hidden or disabled (but allow fade-out state to continue showing)
    if (
      this.fadeState === "hidden" ||
      (!this.enabled && this.fadeState !== "fading-out")
    ) {
      this.plane.visible = false;
      return;
    }

    // Performance optimization: static shadows render once and never again
    if (this.config.isStatic && this.hasRenderedOnce) {
      // Keep plane visible with cached texture, but skip all rendering
      this.plane.visible = true;
      return;
    }

    // Performance optimization: throttle updates for non-static shadows
    if (!this.config.isStatic) {
      this.frameCounter++;
      if (!this.needsUpdate && this.frameCounter % this.updateFrequency !== 0) {
        // Skip rendering this frame, but keep plane visible with cached texture
        this.plane.visible = true;
        return;
      }
    }

    // Reset needsUpdate flag after rendering
    this.needsUpdate = false;

    // Show plane if enabled
    this.plane.visible = true;

    // Update position to follow the tracked object (handles animated objects)
    if (this.trackedObject !== this.parentObject) {
      // Get the tracked object's position in parent's local space
      const trackedLocalPos = this.parentObject.worldToLocal(
        this.trackedObject.getWorldPosition(new THREE.Vector3())
      );

      // Apply offset
      this.shadowGroup.position.set(
        trackedLocalPos.x + this.config.offset.x,
        trackedLocalPos.y + this.config.offset.y,
        trackedLocalPos.z + this.config.offset.z
      );
    }

    // Update rotation every frame to maintain Y-axis alignment
    // Track the actual object being animated, not just the parent
    const trackedWorldQuaternion = new THREE.Quaternion();
    this.trackedObject.getWorldQuaternion(trackedWorldQuaternion);

    const worldEuler = new THREE.Euler();
    worldEuler.setFromQuaternion(trackedWorldQuaternion, "YXZ");

    const yOnlyQuaternion = new THREE.Quaternion();
    yOnlyQuaternion.setFromEuler(new THREE.Euler(0, worldEuler.y, 0, "YXZ"));

    // Get parent's world quaternion for proper local transformation
    const parentWorldQuaternion = new THREE.Quaternion();
    this.parentObject.getWorldQuaternion(parentWorldQuaternion);
    const inverseParent = parentWorldQuaternion.clone().invert();
    this.shadowGroup.quaternion.copy(inverseParent).multiply(yOnlyQuaternion);

    // Save scene state
    const initialBackground = this.scene.background;
    const initialClearAlpha = this.renderer.getClearAlpha();
    const initialRenderTarget = this.renderer.getRenderTarget();

    // Hide camera helper during render
    if (this.cameraHelper) {
      this.cameraHelper.visible = false;
    }

    // Hide the shadow plane itself during depth render
    this.plane.visible = false;

    // Remove background and apply depth material
    this.scene.background = null;
    this.scene.overrideMaterial = this.depthMaterial;
    this.renderer.setClearAlpha(0);

    // Render depth to render target
    this.renderer.setRenderTarget(this.renderTarget);
    this.renderer.render(this.scene, this.shadowCamera);

    // Reset override material
    this.scene.overrideMaterial = null;

    // Show camera helper and plane again
    if (this.cameraHelper) {
      this.cameraHelper.visible = true;
    }
    this.plane.visible = true;

    // Apply blur
    this.blurShadow(this.config.blur);
    this.blurShadow(this.config.blur * 0.4); // Second pass to reduce artifacts

    // Restore scene state
    this.renderer.setRenderTarget(initialRenderTarget);
    this.renderer.setClearAlpha(initialClearAlpha);
    this.scene.background = initialBackground;

    // Mark as rendered for static shadows
    if (this.config.isStatic) {
      this.hasRenderedOnce = true;
    }
  }

  /**
   * Request an immediate update on next render
   * Useful when the object moves or changes
   * For static shadows, this resets the "rendered once" flag
   */
  requestUpdate() {
    this.needsUpdate = true;
    if (this.config.isStatic) {
      this.hasRenderedOnce = false;
    }
  }

  /**
   * Update shadow properties
   * @param {Object} updates - Properties to update
   */
  updateProperties(updates = {}) {
    let changed = false;

    if (updates.opacity !== undefined) {
      this.targetOpacity = updates.opacity;
      this.config.opacity = updates.opacity;
      // Update current opacity based on fade state
      if (this.fadeState === "visible") {
        this.plane.material.opacity = updates.opacity;
      } else {
        this.plane.material.opacity = updates.opacity * this.fadeProgress;
      }
      changed = true;
    }

    if (updates.darkness !== undefined) {
      this.depthMaterial.userData.darkness.value = updates.darkness;
      this.config.darkness = updates.darkness;
      changed = true;
    }

    if (updates.blur !== undefined) {
      this.config.blur = updates.blur;
      changed = true;
    }

    if (updates.updateFrequency !== undefined) {
      this.updateFrequency = updates.updateFrequency;
      changed = true;
    }

    if (updates.fadeDuration !== undefined) {
      this.fadeDuration = updates.fadeDuration;
      changed = true;
    }

    if (changed) {
      this.needsUpdate = true; // Request re-render when properties change
      logger.log(`Updated contact shadow properties`);
    }
  }

  /**
   * Dispose and cleanup
   */
  dispose() {
    logger.log(`Disposing contact shadow "${this.shadowGroup.name}"`);

    // Remove from parent
    if (this.shadowGroup.parent) {
      this.shadowGroup.parent.remove(this.shadowGroup);
    }

    // Dispose render targets
    this.renderTarget.dispose();
    this.renderTargetBlur.dispose();

    // Dispose geometries and materials
    this.plane.geometry.dispose();
    this.plane.material.dispose();
    this.blurPlane.geometry.dispose();
    this.depthMaterial.dispose();
    this.horizontalBlurMaterial.dispose();
    this.verticalBlurMaterial.dispose();

    // Remove camera helper
    if (this.cameraHelper) {
      this.scene.remove(this.cameraHelper);
    }

    logger.log(`Contact shadow disposed`);
  }
}

/**
 * Helper function to create a contact shadow
 * @param {THREE.WebGLRenderer} renderer - The renderer
 * @param {THREE.Scene} scene - The scene
 * @param {THREE.Object3D} parentObject - The parent object
 * @param {Object} config - Configuration options
 * @returns {ContactShadow}
 */
export function createContactShadow(renderer, scene, parentObject, config) {
  return new ContactShadow(renderer, scene, parentObject, config);
}
