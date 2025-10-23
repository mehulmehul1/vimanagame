import * as THREE from "three";
import { checkCriteria } from "./utils/criteriaHelper.js";
import { Logger } from "./utils/logger.js";

/**
 * ColliderManager - Manages trigger colliders and intersection detection
 *
 * Features:
 * - Creates trigger colliders from collider data below
 * - Detects when character enters/exits colliders
 * - Sets game state through GameManager
 * - Supports box, sphere, and capsule shapes
 * - Handles one-time triggers and enable/disable states
 * - Supports state-based activation criteria
 */

class ColliderManager {
  constructor(
    physicsManager,
    gameManager,
    colliderData = [],
    scene = null,
    sceneManager = null,
    gizmoManager = null
  ) {
    this.physicsManager = physicsManager;
    this.gameManager = gameManager;
    this.scene = scene;
    this.sceneManager = sceneManager;
    this.gizmoManager = gizmoManager; // For registering debug meshes
    this.logger = new Logger("ColliderManager", false);
    this.colliders = [];
    this.debugMeshes = new Map(); // Map of collider id -> debug mesh
    this.activeColliders = new Set(); // Track which colliders the character is currently inside
    this.triggeredOnce = new Set(); // Track which "once" colliders have been triggered

    // Check for gizmo-enabled colliders and set global flag
    this.checkForGizmoColliders(colliderData);

    // Initialize all colliders
    this.initializeColliders(colliderData);

    // Initialize debug visualization meshes (if enabled via URL param OR for gizmo-enabled colliders)
    this.initializeDebugMeshes();
  }

  /**
   * Check for gizmo-enabled colliders and set global flag
   * @param {Array} colliderData - Array of collider definitions
   */
  checkForGizmoColliders(colliderData) {
    const hasGizmo = colliderData.some(
      (collider) => collider && collider.gizmo === true
    );

    if (hasGizmo) {
      try {
        if (
          window?.gameManager &&
          typeof window.gameManager.setState === "function"
        ) {
          window.gameManager.setState({ hasGizmoInData: true });
          this.logger.log(
            "Set hasGizmoInData=true due to gizmo-enabled colliders"
          );
        }
      } catch (e) {
        this.logger.error("Failed to set hasGizmoInData:", e);
      }
    }
  }

  /**
   * Initialize colliders from data
   * @param {Array} colliderData - Array of collider definitions
   */
  initializeColliders(colliderData) {
    colliderData.forEach((data) => {
      const collider = this.createCollider(data);
      if (collider) {
        this.colliders.push({
          id: data.id,
          data: data,
          collider: collider,
          handle: collider.handle,
          enabled: data.enabled !== false,
        });
      }
    });

    this.logger.log(`Initialized ${this.colliders.length} colliders`);
  }

  /**
   * Create a physics collider from data
   * @param {Object} data - Collider data
   * @returns {Object} Rapier collider
   */
  createCollider(data) {
    const { type, position, rotation, dimensions } = data;

    // Convert rotation from degrees to quaternion
    const euler = new THREE.Euler(
      THREE.MathUtils.degToRad(rotation.x),
      THREE.MathUtils.degToRad(rotation.y),
      THREE.MathUtils.degToRad(rotation.z)
    );
    const quat = new THREE.Quaternion().setFromEuler(euler);

    let colliderDesc;

    // Create appropriate collider shape
    switch (type) {
      case "box":
        colliderDesc = this.physicsManager.createSensorBox(
          dimensions.x,
          dimensions.y,
          dimensions.z
        );
        break;

      case "sphere":
        colliderDesc = this.physicsManager.createSensorSphere(
          dimensions.radius
        );
        break;

      case "capsule":
        colliderDesc = this.physicsManager.createSensorCapsule(
          dimensions.halfHeight,
          dimensions.radius
        );
        break;

      default:
        this.logger.warn(`Unknown collider type "${type}" for ${data.id}`);
        return null;
    }

    if (!colliderDesc) return null;

    // Set position and rotation on the descriptor
    colliderDesc.setTranslation(position.x, position.y, position.z);
    colliderDesc.setRotation({ x: quat.x, y: quat.y, z: quat.z, w: quat.w });

    // Create the actual collider from the descriptor
    return this.physicsManager.createColliderFromDesc(colliderDesc);
  }

  /**
   * Check for intersections with character and trigger events
   * @param {Object} characterBody - Rapier rigid body of the character
   */
  update(characterBody) {
    // Update debug mesh visibility based on activation conditions
    this.updateDebugMeshVisibility();

    // Get character collider
    const characterCollider = characterBody.collider(0);

    this.colliders.forEach(({ id, data, collider, enabled }) => {
      if (!enabled) return;
      if (data.once && this.triggeredOnce.has(id)) return;

      // Check activation conditions based on game state
      if (!this.checkActivationConditions(data)) return;

      // Check intersection
      const isIntersecting = this.physicsManager.checkIntersection(
        characterCollider,
        collider
      );

      const wasActive = this.activeColliders.has(id);

      if (isIntersecting && !wasActive) {
        // Character just entered
        this.activeColliders.add(id);
        this.onEnter(id, data);

        if (data.once) {
          this.triggeredOnce.add(id);
          // Clean up the collider after a short delay to allow events to complete
          setTimeout(() => {
            this.removeCollider(id);
          }, 100);
        }
      } else if (!isIntersecting && wasActive) {
        // Character just exited
        this.activeColliders.delete(id);
        this.onExit(id, data);
      }
    });
  }

  /**
   * Check if collider's activation conditions are met
   * @param {Object} data - Collider data
   * @returns {boolean} True if collider should be active
   */
  checkActivationConditions(data) {
    const gameState = this.gameManager.getState();

    // Check criteria (supports operators like $gte, $lt, etc.)
    if (data.criteria) {
      return checkCriteria(gameState, data.criteria);
    }

    return true;
  }

  /**
   * Handle character entering a collider
   * @param {string} id - Collider ID
   * @param {Object} data - Collider data
   */
  onEnter(id, data) {
    this.logger.log(`Entered "${id}"`);

    // Set state if defined
    if (data.setStateOnEnter) {
      this.gameManager.setState(data.setStateOnEnter);
    }
  }

  /**
   * Handle character exiting a collider
   * @param {string} id - Collider ID
   * @param {Object} data - Collider data
   */
  onExit(id, data) {
    this.logger.log(`Exited "${id}"`);

    // Set state if defined
    if (data.setStateOnExit) {
      this.gameManager.setState(data.setStateOnExit);
    }
  }

  /**
   * Enable a collider by ID
   * @param {string} id - Collider ID
   */
  enableCollider(id) {
    const collider = this.colliders.find((c) => c.id === id);
    if (collider) {
      collider.enabled = true;
      this.logger.log(`Enabled "${id}"`);
    }
  }

  /**
   * Disable a collider by ID
   * @param {string} id - Collider ID
   */
  disableCollider(id) {
    const collider = this.colliders.find((c) => c.id === id);
    if (collider) {
      collider.enabled = false;
      // Remove from active if it was active
      this.activeColliders.delete(id);
      this.logger.log(`Disabled "${id}"`);
    }
  }

  /**
   * Reset a "once" collider so it can trigger again
   * @param {string} id - Collider ID
   */
  resetOnceCollider(id) {
    this.triggeredOnce.delete(id);
    this.logger.log(`Reset once-trigger for "${id}"`);
  }

  /**
   * Check if character is currently inside a collider
   * @param {string} id - Collider ID
   * @returns {boolean}
   */
  isInCollider(id) {
    return this.activeColliders.has(id);
  }

  /**
   * Get all currently active colliders
   * @returns {Array<string>} Array of collider IDs
   */
  getActiveColliders() {
    return Array.from(this.activeColliders);
  }

  /**
   * Add debug mesh for a collider
   * @param {string} id - Collider ID
   * @param {THREE.Mesh} mesh - Debug mesh
   */
  addDebugMesh(id, mesh) {
    this.debugMeshes.set(id, mesh);
  }

  /**
   * Update debug mesh visibility based on activation conditions
   */
  updateDebugMeshVisibility() {
    this.colliders.forEach(({ id, data, enabled }) => {
      const mesh = this.debugMeshes.get(id);
      if (!mesh) return;

      // Gizmo colliders are always visible for authoring
      if (data.gizmo) {
        mesh.visible = enabled;
        return;
      }

      // Hide if disabled or activation conditions not met
      const isActive = enabled && this.checkActivationConditions(data);
      mesh.visible = isActive;
    });
  }

  /**
   * Initialize debug visualization meshes for colliders
   */
  initializeDebugMeshes() {
    const showColliders =
      this.gameManager.getURLParam("showColliders") === "true";
    const hasGizmoManager = !!this.gizmoManager;
    const hasAnyGizmoColliders = this.colliders.some(({ data }) => data.gizmo);

    if (!showColliders && !hasAnyGizmoColliders) {
      this.logger.log(
        "Collider debug visualization disabled (add ?showColliders=true to URL to enable)"
      );
      return;
    }

    if (showColliders) {
      this.logger.log(
        "Collider debug visualization enabled (showColliders=true)"
      );
    }

    this.colliders.forEach(({ id, data, enabled }) => {
      // Create debug mesh if URL param is set OR if this specific collider has gizmo flag
      const shouldCreateDebugMesh = showColliders || data.gizmo;
      if (!enabled || !shouldCreateDebugMesh) return;

      let geometry;
      switch (data.type) {
        case "box":
          geometry = new THREE.BoxGeometry(
            data.dimensions.x * 2,
            data.dimensions.y * 2,
            data.dimensions.z * 2
          );
          break;
        case "sphere":
          geometry = new THREE.SphereGeometry(data.dimensions.radius, 16, 16);
          break;
        case "capsule":
          geometry = new THREE.CapsuleGeometry(
            data.dimensions.radius,
            data.dimensions.halfHeight * 2,
            8,
            16
          );
          break;
      }

      if (geometry) {
        // Wireframe debug material
        const material = new THREE.MeshBasicMaterial({
          color: data.gizmo ? 0xff0000 : 0x00ff00, // Red for gizmo colliders, green for debug
          wireframe: true,
          wireframeLinewidth: 2,
        });
        const mesh = new THREE.Mesh(geometry, material);

        // Apply position
        mesh.position.set(data.position.x, data.position.y, data.position.z);

        // Apply rotation (convert degrees to radians)
        mesh.rotation.set(
          THREE.MathUtils.degToRad(data.rotation.x),
          THREE.MathUtils.degToRad(data.rotation.y),
          THREE.MathUtils.degToRad(data.rotation.z)
        );

        this.scene.add(mesh);
        this.debugMeshes.set(id, mesh);

        // Register with gizmo manager if this collider has gizmo flag
        if (data.gizmo && this.gizmoManager) {
          this.gizmoManager.registerObject(mesh, id, "collider");
          this.logger.log(`Registered gizmo collider: ${id}`);
        }

        if (showColliders) {
          this.logger.log(`Added debug mesh for collider: ${id}`);
        } else if (data.gizmo) {
          this.logger.log(
            `Added gizmo debug mesh for collider: ${id} (red wireframe)`
          );
        }
      }
    });
  }

  /**
   * Remove and clean up a collider completely
   * @param {string} id - Collider ID
   */
  removeCollider(id) {
    const colliderIndex = this.colliders.findIndex((c) => c.id === id);
    if (colliderIndex === -1) return;

    const { collider } = this.colliders[colliderIndex];

    // Remove from physics world
    if (collider && this.physicsManager.world) {
      this.physicsManager.world.removeCollider(collider);
    }

    // Remove debug mesh from scene
    if (this.debugMeshes.has(id)) {
      const mesh = this.debugMeshes.get(id);
      if (this.scene && mesh) {
        this.scene.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
      }
      this.debugMeshes.delete(id);
    }

    // Remove from colliders array
    this.colliders.splice(colliderIndex, 1);

    // Clean up tracking sets
    this.activeColliders.delete(id);
    this.triggeredOnce.delete(id);

    this.logger.log(`Removed and cleaned up collider "${id}"`);
  }
}

export default ColliderManager;
