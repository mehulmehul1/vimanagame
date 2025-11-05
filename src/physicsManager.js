const RAPIER = await import("@dimforge/rapier3d");
import * as THREE from "three";
import { Logger } from "./utils/logger.js";

class PhysicsManager {
  constructor() {
    this.RAPIER = RAPIER; // Store RAPIER reference for external use
    this.gravity = { x: 0.0, y: -9.81, z: 0.0 };
    this.world = new RAPIER.World(this.gravity);
    this.trimeshColliders = new Map(); // Map of id -> { collider, body }
    this.logger = new Logger("PhysicsManager", false);
    //this.createFloor();
  }

  // createFloor() {
  //   const floorDesc = RAPIER.RigidBodyDesc.fixed().setTranslation(0, 0, 0);
  //   const floor = this.world.createRigidBody(floorDesc);
  //   const floorColliderDesc = RAPIER.ColliderDesc.cuboid(
  //     1000,
  //     0.1,
  //     1000
  //   ).setFriction(1.0);
  //   this.world.createCollider(floorColliderDesc, floor);
  //   return floor;
  // }

  createCharacter(
    position = { x: 0, y: 0, z: 0 },
    rotation = { x: 0, y: 0, z: 0 }
  ) {
    // Convert Euler angles in DEGREES to quaternion
    const euler = new THREE.Euler(
      THREE.MathUtils.degToRad(rotation.x),
      THREE.MathUtils.degToRad(rotation.y),
      THREE.MathUtils.degToRad(rotation.z)
    );
    const quat = new THREE.Quaternion().setFromEuler(euler);

    const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(position.x, position.y, position.z)
      .setRotation({ x: quat.x, y: quat.y, z: quat.z, w: quat.w })
      .setLinearDamping(0.2)
      .lockRotations(); // Lock all rotations so capsule stays upright
    const body = this.world.createRigidBody(bodyDesc);

    // Character capsule collider
    // halfHeight=0.6, radius=0.3
    // Total height = 2*halfHeight + 2*radius = 1.8m
    const colliderDesc = RAPIER.ColliderDesc.capsule(0.6, 0.3)
      .setFriction(0.9)
      .setMass(60);
    this.world.createCollider(colliderDesc, body);
    return body;
  }

  /**
   * Create a sensor box collider descriptor (trigger, no physics interaction)
   * @param {number} hx - Half-extent X
   * @param {number} hy - Half-extent Y
   * @param {number} hz - Half-extent Z
   * @returns {Object} Collider descriptor
   */
  createSensorBox(hx, hy, hz) {
    return RAPIER.ColliderDesc.cuboid(hx, hy, hz).setSensor(true);
  }

  /**
   * Create a sensor sphere collider descriptor (trigger, no physics interaction)
   * @param {number} radius - Sphere radius
   * @returns {Object} Collider descriptor
   */
  createSensorSphere(radius) {
    return RAPIER.ColliderDesc.ball(radius).setSensor(true);
  }

  /**
   * Create a sensor trimesh collider descriptor (trigger, no physics interaction)
   * @param {Float32Array} vertices - Vertex positions (world space)
   * @param {Uint32Array} indices - Triangle indices
   * @returns {Object} Rapier ColliderDesc
   */
  createSensorTrimesh(vertices, indices) {
    return this.RAPIER.ColliderDesc.trimesh(vertices, indices).setSensor(true);
  }

  /**
   * Create a sensor capsule collider descriptor (trigger, no physics interaction)
   * @param {number} halfHeight - Half height of the cylindrical part
   * @param {number} radius - Capsule radius
   * @returns {Object} Collider descriptor
   */
  createSensorCapsule(halfHeight, radius) {
    return RAPIER.ColliderDesc.capsule(halfHeight, radius).setSensor(true);
  }

  /**
   * Create a collider from a descriptor
   * @param {Object} colliderDesc - Collider descriptor
   * @returns {Object} Collider
   */
  createColliderFromDesc(colliderDesc) {
    return this.world.createCollider(colliderDesc);
  }

  /**
   * Check if two colliders are intersecting
   * @param {Object} collider1 - First collider
   * @param {Object} collider2 - Second collider
   * @returns {boolean} True if intersecting
   */
  checkIntersection(collider1, collider2) {
    return this.world.intersectionPair(collider1, collider2);
  }

  /**
   * Check if a point is inside a trimesh collider
   * @param {Object} trimeshCollider - Rapier trimesh collider
   * @param {Object} point - Point to check {x, y, z} or Rapier Vector3
   * @returns {boolean} True if point is inside the trimesh
   */
  checkPointInTrimesh(trimeshCollider, point) {
    // Use intersectionPair with a temporary sphere sensor at the point
    // This is more reliable than intersectionPair for trimesh sensors
    // The radius needs to be large enough to intersect with the trimesh even if slightly off
    
    const pointVec = point.x !== undefined 
      ? new this.RAPIER.Vector3(point.x, point.y, point.z)
      : point;
    
    // Create a temporary sphere sensor at the point with a reasonable radius
    const tempBodyDesc = this.RAPIER.RigidBodyDesc.kinematicPositionBased()
      .setTranslation(pointVec.x, pointVec.y, pointVec.z);
    const tempBody = this.world.createRigidBody(tempBodyDesc);
    const tempColliderDesc = this.RAPIER.ColliderDesc.ball(0.5).setSensor(true); // Larger radius for better detection
    tempColliderDesc.setCollisionGroups(0xffff0000); // Match probe collision groups
    const tempCollider = this.world.createCollider(tempColliderDesc, tempBody);
    
    // Check intersection - this should work even without a physics step
    const intersects = this.world.intersectionPair(tempCollider, trimeshCollider);
    
    // Clean up immediately
    this.world.removeCollider(tempCollider, false);
    this.world.removeRigidBody(tempBody);
    
    return intersects;
  }

  /**
   * Extract geometry data from a THREE.js object (traverses all meshes)
   * @param {THREE.Object3D} object - THREE.js object containing meshes
   * @returns {Object} { vertices: Float32Array, indices: Uint32Array } or null if no geometry found
   */
  extractGeometryFromObject(object) {
    const allVertices = [];
    const allIndices = [];
    let vertexOffset = 0;

    object.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const geometry = child.geometry;

        // Get position attribute
        const positionAttribute = geometry.attributes.position;
        if (!positionAttribute) return;

        // Apply the mesh's world transform to vertices
        child.updateWorldMatrix(true, false); // Update world matrix with parent transforms
        const worldMatrix = child.matrixWorld;

        // Extract vertices and apply world transform
        const vertex = new THREE.Vector3();
        for (let i = 0; i < positionAttribute.count; i++) {
          vertex.fromBufferAttribute(positionAttribute, i);
          vertex.applyMatrix4(worldMatrix);
          allVertices.push(vertex.x, vertex.y, vertex.z);
        }

        // Extract indices
        if (geometry.index) {
          const indexArray = geometry.index.array;
          for (let i = 0; i < indexArray.length; i++) {
            allIndices.push(indexArray[i] + vertexOffset);
          }
        } else {
          // No index buffer, generate sequential indices
          for (let i = 0; i < positionAttribute.count; i++) {
            allIndices.push(i + vertexOffset);
          }
        }

        vertexOffset += positionAttribute.count;
      }
    });

    if (allVertices.length === 0 || allIndices.length === 0) {
      this.logger.warn("No geometry data found in object");
      return null;
    }

    this.logger.log(
      `Extracted ${allVertices.length / 3} vertices and ${
        allIndices.length / 3
      } triangles from object`
    );

    return {
      vertices: new Float32Array(allVertices),
      indices: new Uint32Array(allIndices),
    };
  }

  /**
   * Extract geometry data from a single mesh (for trigger colliders)
   * Uses vertices as-is since they're already in world space (mesh at origin, vertices offset)
   * @param {THREE.Mesh} mesh - THREE.js mesh object
   * @returns {Object} { vertices: Float32Array, indices: Uint32Array } or null if no geometry found
   */
  extractGeometryFromMesh(mesh) {
    if (!mesh || !mesh.isMesh || !mesh.geometry) {
      return null;
    }

    const geometry = mesh.geometry;
    const positionAttribute = geometry.attributes.position;
    if (!positionAttribute) {
      return null;
    }

    // Extract vertices as-is (they're already in world space)
    const vertices = [];
    for (let i = 0; i < positionAttribute.count; i++) {
      vertices.push(
        positionAttribute.getX(i),
        positionAttribute.getY(i),
        positionAttribute.getZ(i)
      );
    }

    // Extract indices
    const indices = [];
    if (geometry.index) {
      const indexArray = geometry.index.array;
      for (let i = 0; i < indexArray.length; i++) {
        indices.push(indexArray[i]);
      }
    } else {
      // No index buffer, generate sequential indices
      for (let i = 0; i < positionAttribute.count; i++) {
        indices.push(i);
      }
    }

    if (vertices.length === 0 || indices.length === 0) {
      return null;
    }

    return {
      vertices: new Float32Array(vertices),
      indices: new Uint32Array(indices),
    };
  }

  /**
   * Create a static trimesh collider from a THREE.js object
   * @param {string} id - Unique identifier for this collider
   * @param {THREE.Object3D} object - THREE.js object containing mesh geometry
   * @param {Object} position - World position {x, y, z} (not used - geometry is already in world space)
   * @param {Object} rotation - Quaternion rotation {x, y, z, w} (not used - geometry is already in world space)
   * @returns {Object} { collider, body } or null if failed
   */
  createTrimeshCollider(
    id,
    object,
    position = { x: 0, y: 0, z: 0 },
    rotation = { x: 0, y: 0, z: 0, w: 1 }
  ) {
    // Extract geometry from the object (with world transforms already applied)
    const geometryData = this.extractGeometryFromObject(object);
    if (!geometryData) {
      this.logger.error(
        `Failed to extract geometry for trimesh collider "${id}"`
      );
      return null;
    }

    const { vertices, indices } = geometryData;

    // Create a fixed (static) rigid body at origin with no rotation
    // The geometry vertices are already in world space, so we don't need to transform the body
    const bodyDesc = RAPIER.RigidBodyDesc.fixed()
      .setTranslation(0, 0, 0)
      .setRotation({ x: 0, y: 0, z: 0, w: 1 });
    const body = this.world.createRigidBody(bodyDesc);

    // Create trimesh collider descriptor with proper friction
    // Belongs to group 3 (environment) so cords can collide with it
    // Collides with all groups (0xffff) so character controller still works
    const colliderDesc = RAPIER.ColliderDesc.trimesh(vertices, indices)
      .setFriction(0.7)
      .setRestitution(0.0) // No bounciness
      .setCollisionGroups(0xffff0008); // Belongs to group 3 (0x0008), collides with all groups (0xffff)

    // Create the collider and attach to the body
    const collider = this.world.createCollider(colliderDesc, body);

    // Store reference
    this.trimeshColliders.set(id, { collider, body });

    this.logger.log(
      `Created trimesh collider "${id}" with ${
        vertices.length / 3
      } vertices and ${indices.length / 3} triangles`
    );

    // Debug: Log first few vertices to verify position
    this.logger.log(
      `  First vertex: (${vertices[0].toFixed(2)}, ${vertices[1].toFixed(
        2
      )}, ${vertices[2].toFixed(2)})`
    );
    this.logger.log(`  Collider friction: ${collider.friction()}`);
    this.logger.log(`  Collider is sensor: ${collider.isSensor()}`);
    this.logger.log(`  Body type: ${body.bodyType()}`);

    return { collider, body };
  }

  /**
   * Remove a trimesh collider by ID
   * @param {string} id - Collider ID
   * @returns {boolean} True if removed, false if not found
   */
  removeTrimeshCollider(id) {
    const colliderData = this.trimeshColliders.get(id);
    if (!colliderData) {
      return false;
    }

    const { collider, body } = colliderData;

    // Remove collider and body from world
    if (collider) {
      this.world.removeCollider(collider, false);
    }
    if (body) {
      this.world.removeRigidBody(body);
    }

    this.trimeshColliders.delete(id);
    this.logger.log(`Removed trimesh collider "${id}"`);

    return true;
  }

  /**
   * Check if a trimesh collider exists
   * @param {string} id - Collider ID
   * @returns {boolean}
   */
  hasTrimeshCollider(id) {
    return this.trimeshColliders.has(id);
  }

  /**
   * Cast a ray downward to find the floor height at a given X/Z position
   * @param {number} x - World X coordinate
   * @param {number} z - World Z coordinate
   * @param {number} startY - Y coordinate to start the ray from (default: 100)
   * @param {number} maxDistance - Maximum ray distance (default: 200)
   * @returns {number|null} Floor Y coordinate if hit, null otherwise
   */
  getFloorHeightAt(x, z, startY = 100, maxDistance = 200) {
    // Create ray pointing downward
    const rayOrigin = { x, y: startY, z };
    const rayDirection = { x: 0, y: -1, z: 0 }; // Straight down
    const ray = new this.RAPIER.Ray(rayOrigin, rayDirection);

    // Cast ray and get first hit
    const hit = this.world.castRay(ray, maxDistance, true); // solid = true (only hit solid colliders)

    if (hit) {
      // Calculate hit point Y coordinate
      const hitY = startY + hit.toi * rayDirection.y; // toi = time of impact
      this.logger.log(
        `Floor raycast at (${x.toFixed(2)}, ${z.toFixed(
          2
        )}): hit at Y=${hitY.toFixed(2)}`
      );
      return hitY;
    }

    this.logger.warn(
      `Floor raycast at (${x.toFixed(2)}, ${z.toFixed(
        2
      )}): no hit within ${maxDistance}m`
    );
    return null;
  }

  step() {
    this.world.step();
  }
}

export default PhysicsManager;
