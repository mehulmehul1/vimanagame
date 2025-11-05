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
    this.logger = new Logger("ColliderManager", true); // Enable logging for debugging
    this.colliders = [];
    this.debugMeshes = new Map(); // Map of collider id -> debug mesh
    this.activeColliders = new Set(); // Track which colliders the character is currently inside
    this.triggeredOnce = new Set(); // Track which "once" colliders have been triggered

    // Store original THREE.Mesh objects for zone colliders (for raycasting point-in-mesh checks)
    this.zoneMeshes = new Map(); // Map of zone id -> THREE.Mesh

    // Camera probe for zone detection during START_SCREEN and camera animations
    this.cameraProbeBody = null;
    this.cameraProbeCollider = null;
    this.camera = null; // Will be set by setCamera()

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
   * Register a trimesh trigger collider from a loaded mesh
   * @param {string} id - Collider ID (e.g., "zone-introAlley")
   * @param {THREE.Mesh} mesh - THREE.js mesh object
   * @param {Object} colliderData - Collider data with setStateOnEnter, criteria, etc.
   * @returns {boolean} True if successfully registered
   */
  registerTrimeshTriggerCollider(id, mesh, colliderData) {
    if (!this.physicsManager || !mesh) {
      this.logger.warn(
        `Cannot register trimesh trigger collider "${id}" - missing physics manager or mesh`
      );
      return false;
    }

    const RAPIER = this.physicsManager.RAPIER;
    const world = this.physicsManager.world;

    // Extract geometry from mesh (vertices are already in world space, meshes at origin)
    const geometryData = this.physicsManager.extractGeometryFromMesh(mesh);
    if (!geometryData) {
      this.logger.error(
        `Failed to extract geometry for trimesh trigger collider "${id}"`
      );
      return false;
    }

    const { vertices, indices } = geometryData;

    // Create sensor trimesh collider descriptor
    // Don't set collision groups explicitly - use Rapier's default (same as regular box/sphere colliders)
    const colliderDesc = this.physicsManager.createSensorTrimesh(
      vertices,
      indices
    );

    // Trimesh colliders are already in world space, so create a fixed body at origin
    const bodyDesc = RAPIER.RigidBodyDesc.fixed()
      .setTranslation(0, 0, 0)
      .setRotation({ x: 0, y: 0, z: 0, w: 1 });
    const body = world.createRigidBody(bodyDesc);

    // Create the collider and attach to the body
    const collider = world.createCollider(colliderDesc, body);

    if (!collider) {
      this.logger.error(`Failed to create trimesh trigger collider "${id}"`);
      return false;
    }

    // Register with ColliderManager
    this.colliders.push({
      id,
      data: colliderData,
      collider,
      enabled: colliderData.enabled !== false, // Default to enabled
    });

    // Store mesh reference for zone colliders (for raycasting point-in-mesh checks)
    if (id.startsWith("zone-")) {
      this.zoneMeshes.set(id, mesh);
    }

    // Create debug mesh for visualization (if enabled via URL param)
    const urlParams = new URLSearchParams(window.location.search);
    const showColliders = urlParams.get("showColliders") === "true";
    if (showColliders && this.scene) {
      // Clone the mesh geometry for the debug mesh
      const debugGeometry = mesh.geometry.clone();

      // Create wireframe debug material
      const debugMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ff00, // Green for zone colliders
        wireframe: true,
        wireframeLinewidth: 2,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide,
        depthTest: true,
        depthWrite: false,
      });

      const debugMesh = new THREE.Mesh(debugGeometry, debugMaterial);

      // Apply mesh's world transform (since vertices are already in world space, mesh is at origin)
      // But we need to apply the mesh's local transform if it has one
      if (mesh.parent) {
        mesh.parent.updateWorldMatrix(true, false);
        debugMesh.applyMatrix4(mesh.parent.matrixWorld);
      }

      // Set high render order so it renders on top
      debugMesh.renderOrder = 9999;

      this.scene.add(debugMesh);
      this.debugMeshes.set(id, debugMesh);

      this.logger.log(`Created debug mesh for zone collider "${id}"`);
    }

    this.logger.log(
      `Registered trimesh trigger collider "${id}" from mesh "${mesh.name}" (${
        vertices.length / 3
      } vertices, ${indices.length / 3} triangles)`
    );

    // Log first few vertices for debugging zone colliders
    if (id.startsWith("zone-")) {
      const zoneName = id.replace("zone-", "");
      this.logger.log(
        `Zone collider "${zoneName}" registered - first vertex: (${vertices[0].toFixed(
          2
        )}, ${vertices[1].toFixed(2)}, ${vertices[2].toFixed(2)})`
      );
    }

    return true;
  }

  /**
   * Set camera reference for camera-based zone detection
   * @param {THREE.Camera} camera - Camera object
   */
  setCamera(camera) {
    this.camera = camera;
    this._createCameraProbe();
  }

  /**
   * Create a camera probe body for zone detection during camera animations
   * @private
   */
  _createCameraProbe() {
    if (!this.physicsManager || this.cameraProbeBody) return;

    const RAPIER = this.physicsManager.RAPIER;
    const world = this.physicsManager.world;

    // Create a kinematic body at origin (will be updated each frame)
    const bodyDesc =
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(0, 0, 0);
    this.cameraProbeBody = world.createRigidBody(bodyDesc);

    // Create a small sphere sensor collider (radius 0.2m) for the camera probe
    // Increased radius to improve trimesh intersection detection
    // Don't set collision groups explicitly - use Rapier's default (same as regular colliders)
    const colliderDesc = RAPIER.ColliderDesc.ball(0.2).setSensor(true);
    this.cameraProbeCollider = world.createCollider(
      colliderDesc,
      this.cameraProbeBody
    );

    this.logger.log("Created camera probe for zone detection");

    // Log probe details for debugging
    if (this.cameraProbeCollider) {
      const shape = this.cameraProbeCollider.shape;
      this.logger.log(
        `Camera probe collider created: type=${shape.type}, radius=${
          shape.radius || "N/A"
        }`
      );
    }
  }

  /**
   * Check for intersections with character or camera and trigger events
   * @param {Object} characterBody - Rapier rigid body of the character (optional if using camera)
   * @param {boolean} useCamera - If true, use camera position instead of character position
   */
  update(characterBody = null, useCamera = false) {
    // Update debug mesh visibility based on activation conditions
    this.updateDebugMeshVisibility();

    // Determine which collider to use for intersection checks
    let probeCollider = null;

    if (useCamera && this.camera && this.cameraProbeBody) {
      // Camera probe position is updated in main.js before physics step
      // Just use the probe collider for intersection checks
      probeCollider = this.cameraProbeCollider;
    } else if (characterBody) {
      // Use character collider
      probeCollider = characterBody.collider(0);
    } else {
      // No valid probe available
      return;
    }

    // Check intersections (physics step should have happened before this)
    // Remove noisy logging - only log when zones actually change

    this.colliders.forEach(({ id, data, collider, enabled }) => {
      if (!enabled) return;
      if (data.once && this.triggeredOnce.has(id)) return;

      // Check activation conditions based on game state
      if (!this.checkActivationConditions(data)) return;

      // Check intersection
      // For zone colliders, use Three.js raycasting for point-in-mesh check
      // For other colliders, use physics intersection check
      let isIntersecting = false;

      if (id.startsWith("zone-") && this.zoneMeshes.has(id)) {
        // Use raycasting for zone colliders
        const zoneMesh = this.zoneMeshes.get(id);
        const probePos =
          useCamera && this.cameraProbeBody
            ? this.cameraProbeBody.translation()
            : characterBody
            ? characterBody.translation()
            : null;

        if (probePos) {
          isIntersecting = this.checkPointInMesh(
            zoneMesh,
            new THREE.Vector3(probePos.x, probePos.y, probePos.z)
          );
        }
      } else {
        // Use physics intersection for regular colliders
        isIntersecting = this.physicsManager.checkIntersection(
          probeCollider,
          collider
        );
      }

      const wasActive = this.activeColliders.has(id);

      // Zone colliders are handled silently - no logging here to avoid spam
      // ZoneManager will log when actual zone changes occur

      if (isIntersecting && !wasActive) {
        // Character/camera just entered
        this.activeColliders.add(id);

        // For zone colliders, notify ZoneManager that this zone is now active
        if (id.startsWith("zone-")) {
          const zoneName = id.replace("zone-", "");
          if (this.gameManager && this.gameManager.zoneManager) {
            this.gameManager.zoneManager.addActiveZone(zoneName);
          }
        }

        this.onEnter(id, data);

        if (data.once) {
          this.triggeredOnce.add(id);
          // Clean up the collider after a short delay to allow events to complete
          setTimeout(() => {
            this.removeCollider(id);
          }, 100);
        }
      } else if (!isIntersecting && wasActive) {
        // Character/camera just exited
        this.activeColliders.delete(id);

        // For zone colliders, notify ZoneManager that this zone is no longer active
        if (id.startsWith("zone-")) {
          const zoneName = id.replace("zone-", "");
          if (this.gameManager && this.gameManager.zoneManager) {
            this.gameManager.zoneManager.removeActiveZone(zoneName);
          }
        }

        this.onExit(id, data);
      }
    });
  }

  /**
   * Check if a point is inside a mesh using Three.js raycasting
   * Uses the odd/even intersection rule: cast a ray and count intersections
   * @param {THREE.Mesh} mesh - THREE.js mesh to test against
   * @param {THREE.Vector3} point - Point to test
   * @returns {boolean} True if point is inside the mesh
   */
  /**
   * Check if a point is inside a mesh using raycasting against the geometry
   * Since vertices are in world space, we can use the geometry directly
   * @param {THREE.Mesh} mesh - THREE.js mesh to test against
   * @param {THREE.Vector3} point - Point to test (in world space)
   * @returns {boolean} True if point is inside the mesh
   */
  checkPointInMesh(mesh, point) {
    const geometry = mesh.geometry;
    if (!geometry) return false;

    const positionAttribute = geometry.attributes.position;
    if (!positionAttribute) return false;

    // Get indices (either from index buffer or sequential)
    const indexAttribute = geometry.index;
    const indices = indexAttribute ? indexAttribute.array : null;

    // Cast a ray from the point along positive X-axis
    const rayOrigin = point.clone();
    const rayDirection = new THREE.Vector3(1, 0, 0);

    let intersectionCount = 0;

    // Iterate through all triangles
    const vertexCount = positionAttribute.count;
    const triangleCount = indices ? indices.length / 3 : vertexCount / 3;

    for (let i = 0; i < triangleCount; i++) {
      let v0, v1, v2;

      if (indices) {
        const i0 = indices[i * 3];
        const i1 = indices[i * 3 + 1];
        const i2 = indices[i * 3 + 2];
        v0 = new THREE.Vector3(
          positionAttribute.getX(i0),
          positionAttribute.getY(i0),
          positionAttribute.getZ(i0)
        );
        v1 = new THREE.Vector3(
          positionAttribute.getX(i1),
          positionAttribute.getY(i1),
          positionAttribute.getZ(i1)
        );
        v2 = new THREE.Vector3(
          positionAttribute.getX(i2),
          positionAttribute.getY(i2),
          positionAttribute.getZ(i2)
        );
      } else {
        v0 = new THREE.Vector3(
          positionAttribute.getX(i * 3),
          positionAttribute.getY(i * 3),
          positionAttribute.getZ(i * 3)
        );
        v1 = new THREE.Vector3(
          positionAttribute.getX(i * 3 + 1),
          positionAttribute.getY(i * 3 + 1),
          positionAttribute.getZ(i * 3 + 1)
        );
        v2 = new THREE.Vector3(
          positionAttribute.getX(i * 3 + 2),
          positionAttribute.getY(i * 3 + 2),
          positionAttribute.getZ(i * 3 + 2)
        );
      }

      // Check ray-triangle intersection using Möller-Trumbore algorithm
      const edge1 = new THREE.Vector3().subVectors(v1, v0);
      const edge2 = new THREE.Vector3().subVectors(v2, v0);
      const h = new THREE.Vector3().crossVectors(rayDirection, edge2);
      const a = edge1.dot(h);

      // Ray is parallel to triangle
      if (Math.abs(a) < 1e-8) continue;

      const f = 1.0 / a;
      const s = new THREE.Vector3().subVectors(rayOrigin, v0);
      const u = f * s.dot(h);

      if (u < 0.0 || u > 1.0) continue;

      const q = new THREE.Vector3().crossVectors(s, edge1);
      const v = f * rayDirection.dot(q);

      if (v < 0.0 || u + v > 1.0) continue;

      const t = f * edge2.dot(q);

      // Intersection found (t > 0 means ray goes forward)
      if (t > 1e-8) {
        intersectionCount++;
      }
    }

    // Odd number of intersections = inside, even number = outside
    return intersectionCount % 2 === 1;
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
    // Suppress logging for zone colliders - ZoneManager handles logging
    if (!id.startsWith("zone-")) {
      this.logger.log(`Entered "${id}"`);
    }

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
    // Suppress logging for zone colliders - ZoneManager handles logging
    if (!id.startsWith("zone-")) {
      this.logger.log(`Exited "${id}"`);
    }

    // For zone colliders, don't clear zone on exit - let the next zone set it
    // This prevents rapid zone switching when zones overlap
    if (id.startsWith("zone-")) {
      // Zone colliders - don't clear zone on exit, let next zone handle it
      return;
    }

    // Set state if defined (for non-zone colliders)
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

    // Remove zone mesh reference if it exists
    if (this.zoneMeshes.has(id)) {
      this.zoneMeshes.delete(id);
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
