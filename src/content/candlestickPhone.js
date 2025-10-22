import * as THREE from "three";
import { Logger } from "../utils/logger.js";
import PhoneCord from "./phoneCord.js";

/**
 * CandlestickPhone - Manages candlestick phone interactions and cord physics
 *
 * Features:
 * - Physics-based telephone cord simulation
 * - Receiver management
 * - Future: Interaction system for picking up receiver
 *
 * Usage:
 * const candlestickPhone = new CandlestickPhone({
 *   sceneManager,
 *   physicsManager,
 *   scene,
 *   camera,
 * });
 * candlestickPhone.initialize();
 */
class CandlestickPhone {
  constructor(options = {}) {
    this.sceneManager = options.sceneManager;
    this.physicsManager = options.physicsManager;
    this.scene = options.scene;
    this.camera = options.camera;
    this.logger = new Logger("CandlestickPhone", false);

    // Phone components
    this.cordAttach = null; // Where the cord attaches to the phone base
    this.receiver = null; // The handheld receiver
    this.colliderMesh = null; // Reference to the Collider mesh

    // Phone cord (reusable module)
    this.phoneCord = null; // PhoneCord instance

    // Physics
    this.tableCollider = null; // Static collider for table beneath phone (cube)
    this.tableRigidBody = null; // Rigid body for table collider
    this.meshCollider = null; // Trimesh collider from Collider mesh
    this.meshColliderBody = null; // Rigid body for trimesh collider
    this.lastPhonePosition = new THREE.Vector3(); // Track phone position for updates

    // Configuration
    this.config = {
      // Cord configuration (darker/different appearance from phonebooth)
      cordConfig: {
        cordSegments: 10,
        cordSegmentLength: 0.06,
        cordSegmentRadius: 0.05, // Larger collider for reliable collision detection (15mm)
        cordMass: 0.001,
        cordDamping: 5.0,
        cordAngularDamping: 5.0,
        cordDroopAmount: 0,
        cordRigidSegments: 0, // All segments dynamic - using baked transforms for initial pose
        cordColor: 0x2a2a2a, // Darker grey for vintage look
        cordVisualRadius: 0.008,
        cordMetalness: 0.4,
        cordRoughness: 0.7,
        initMode: "straight",
        // Collision groups: Belongs to group 2, collides ONLY with group 4 (table cube/trimesh)
        // Format: 0xMMMMGGGG where MMMM = collision mask, GGGG = membership groups
        cordCollisionGroup: 0x00080002, // Belongs to group 2 (0x0002), collides with group 4 (0x0008)
        initialSegmentTransforms: [
          {
            index: 0,
            position: {
              x: -4.76614236831665,
              y: 1.1566543579101562,
              z: 85.10692596435547,
            },
            rotation: {
              x: 0.01608729548752308,
              y: 0.0039041037671267986,
              z: -0.6943904161453247,
              w: 0.7194080352783203,
            },
          },
          {
            index: 1,
            position: {
              x: -4.842679500579834,
              y: 1.1704081296920776,
              z: 85.14257049560547,
            },
            rotation: {
              x: -0.26038578152656555,
              y: -0.6332693099975586,
              z: -0.7261661887168884,
              w: -0.06206420436501503,
            },
          },
          {
            index: 2,
            position: {
              x: -4.890603542327881,
              y: 1.1704081296920776,
              z: 85.21875,
            },
            rotation: {
              x: 0.5373536944389343,
              y: -0.2495744377374649,
              z: 0.3767964243888855,
              w: -0.7120309472084045,
            },
          },
          {
            index: 3,
            position: {
              x: -4.979955673217773,
              y: 1.1704081296920776,
              z: 85.20796966552734,
            },
            rotation: {
              x: 0.3558013141155243,
              y: -0.1129140853881836,
              z: -0.914277970790863,
              w: -0.15732675790786743,
            },
          },
          {
            index: 4,
            position: {
              x: -4.987236022949219,
              y: 1.1704081296920776,
              z: 85.29729461669922,
            },
            rotation: {
              x: 0.38533830642700195,
              y: -0.8555383682250977,
              z: 0.08497804403305054,
              w: 0.3351823687553406,
            },
          },
          {
            index: 5,
            position: {
              x: -4.899610996246338,
              y: 1.1704082489013672,
              z: 85.2767562866211,
            },
            rotation: {
              x: -0.5119867324829102,
              y: 0.6058070063591003,
              z: -0.5236726999282837,
              w: -0.31086090207099915,
            },
          },
          {
            index: 6,
            position: {
              x: -4.809802532196045,
              y: 1.1704081296920776,
              z: 85.2710189819336,
            },
            rotation: {
              x: -0.7087559103965759,
              y: 0.6893279552459717,
              z: -0.08311052620410919,
              w: -0.12483887374401093,
            },
          },
          {
            index: 7,
            position: {
              x: -4.720593452453613,
              y: 1.1704086065292358,
              z: 85.2591781616211,
            },
            rotation: {
              x: 0.19807924330234528,
              y: -0.15645527839660645,
              z: -0.05735204368829727,
              w: 0.9659178256988525,
            },
          },
          {
            index: 8,
            position: {
              x: -4.634779453277588,
              y: 1.180083990097046,
              z: 85.2403335571289,
            },
            rotation: {
              x: 0.6636924743652344,
              y: 0.08627700805664062,
              z: 0.7408108115196228,
              w: -0.057167429476976395,
            },
          },
          {
            index: 9,
            position: {
              x: -4.622823238372803,
              y: 1.2620922327041626,
              z: 85.20442199707031,
            },
            rotation: {
              x: -0.23143023252487183,
              y: -0.7361793518066406,
              z: -0.3078458607196808,
              w: 0.5565167665481567,
            },
          },
        ],
      },
      // Table configuration
      tableSize: { x: 1, y: 0.1, z: 1 }, // Half-extents (full size: 0.8x0.04x0.8m)
      tableOffset: -0.1, // Offset below cord attach point
    };
  }

  /**
   * Initialize the candlestick phone
   * Sets up the phone cord and finds necessary meshes
   */
  initialize(gameManager = null) {
    if (!this.sceneManager) {
      this.logger.warn("No SceneManager provided");
      return;
    }

    this.gameManager = gameManager;

    // Find the CordAttach and Receiver meshes
    this.cordAttach = this.sceneManager.findChildByName(
      "candlestickPhone",
      "CordAttach"
    );
    this.receiver = this.sceneManager.findChildByName(
      "candlestickPhone",
      "Receiver"
    );

    if (!this.cordAttach) {
      this.logger.warn("CordAttach mesh not found in candlestickPhone model");
    }

    if (!this.receiver) {
      this.logger.warn("Receiver mesh not found in candlestickPhone model");
    }

    // Find the Collider mesh and make it invisible
    this.colliderMesh = this.sceneManager.findChildByName(
      "candlestickPhone",
      "Collider"
    );
    if (this.colliderMesh) {
      this.colliderMesh.visible = false;
      this.logger.log("Made Collider mesh invisible");

      // Create trimesh collider from the Collider mesh
      this.createColliderTrimesh();

      // Store initial phone position
      const phoneObject = this.sceneManager.getObject("candlestickPhone");
      if (phoneObject) {
        phoneObject.getWorldPosition(this.lastPhonePosition);
      }
    } else {
      this.logger.warn("Collider mesh not found in candlestickPhone model");
    }

    // Create the phone cord if all components are present
    if (this.cordAttach && this.receiver && this.physicsManager) {
      this.phoneCord = new PhoneCord({
        scene: this.scene,
        physicsManager: this.physicsManager,
        cordAttach: this.cordAttach,
        receiver: this.receiver,
        gameManager: this.gameManager, // Pass gameManager (no auto-destroy configured)
        loggerName: "CandlestickPhone.Cord",
        config: this.config.cordConfig,
      });

      const success = this.phoneCord.createCord();
      if (success) {
        this.logger.log("Phone cord created successfully");
      } else {
        this.logger.warn("Failed to create phone cord");
      }
    } else {
      this.logger.warn(
        "Cannot create phone cord - missing CordAttach, Receiver, or PhysicsManager"
      );
    }

    // Create table collider beneath the phone
    this.createTableCollider();

    this.logger.log("Initialized");
  }

  /**
   * Create a static cube collider beneath the phone to act as a table
   */
  createTableCollider() {
    if (!this.physicsManager || !this.sceneManager) {
      this.logger.warn(
        "Cannot create table collider - missing physics manager"
      );
      return;
    }

    // Get the phone's world position from the scene
    const phoneObject = this.sceneManager.getObject("candlestickPhone");
    if (!phoneObject) {
      this.logger.warn(
        "Cannot create table collider - candlestickPhone object not found"
      );
      return;
    }

    // Use CordAttach position if available (more accurate), otherwise use phone base
    let referencePos = new THREE.Vector3();
    if (this.cordAttach) {
      this.cordAttach.getWorldPosition(referencePos);
      this.logger.log(
        "Using CordAttach position as reference for table placement"
      );
    } else {
      phoneObject.getWorldPosition(referencePos);
      this.logger.log(
        "Using phone base position as reference for table placement"
      );
    }

    // Calculate table position (well below the cord attach point or phone base)
    const tablePos = {
      x: referencePos.x,
      y: referencePos.y + this.config.tableOffset,
      z: referencePos.z,
    };

    const RAPIER = this.physicsManager.RAPIER;
    const world = this.physicsManager.world;

    // Create static rigid body for the table
    const bodyDesc = RAPIER.RigidBodyDesc.fixed().setTranslation(
      tablePos.x,
      tablePos.y,
      tablePos.z
    );
    this.tableRigidBody = world.createRigidBody(bodyDesc);

    // Create cuboid collider
    // Collision groups: Belongs to group 4, collides ONLY with group 2 (phone cord)
    // This prevents the table from interfering with character or environment
    const colliderDesc = RAPIER.ColliderDesc.cuboid(
      this.config.tableSize.x,
      this.config.tableSize.y,
      this.config.tableSize.z
    )
      .setFriction(0.7)
      .setRestitution(0.0) // No bounciness
      .setCollisionGroups(0x00020008); // Belongs to group 4 (0x0008), collides with group 2 (0x0002)

    this.tableCollider = world.createCollider(
      colliderDesc,
      this.tableRigidBody
    );

    const tableTop = tablePos.y + this.config.tableSize.y;
    const tableBottom = tablePos.y - this.config.tableSize.y;

    this.logger.log(
      `Created table collider at position (${tablePos.x.toFixed(
        2
      )}, ${tablePos.y.toFixed(2)}, ${tablePos.z.toFixed(2)})`
    );
    this.logger.log(
      `Table size: ${(this.config.tableSize.x * 2).toFixed(2)}m x ${(
        this.config.tableSize.y * 2
      ).toFixed(2)}m x ${(this.config.tableSize.z * 2).toFixed(2)}m`
    );
    this.logger.log(
      `Table Y range: ${tableBottom.toFixed(2)}m (bottom) to ${tableTop.toFixed(
        2
      )}m (top)`
    );
    this.logger.log(`Reference point Y: ${referencePos.y.toFixed(2)}m`);
    this.logger.log(
      `Clearance below reference: ${(referencePos.y - tableTop).toFixed(2)}m`
    );
  }

  /**
   * Create a trimesh collider from the Collider mesh
   */
  createColliderTrimesh() {
    if (!this.physicsManager || !this.colliderMesh) {
      this.logger.warn(
        "Cannot create trimesh collider - missing physics manager or collider mesh"
      );
      return;
    }

    const RAPIER = this.physicsManager.RAPIER;
    const world = this.physicsManager.world;

    // Extract geometry from the collider mesh (in WORLD space)
    const geometryData = this.physicsManager.extractGeometryFromObject(
      this.colliderMesh
    );
    if (!geometryData) {
      this.logger.error("Failed to extract geometry from Collider mesh");
      return;
    }

    const { vertices, indices } = geometryData;

    // Create a kinematic rigid body at origin
    // Note: vertices are in world space, so body stays at origin
    const bodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased()
      .setTranslation(0, 0, 0)
      .setRotation({ x: 0, y: 0, z: 0, w: 1 });
    this.meshColliderBody = world.createRigidBody(bodyDesc);

    // Create trimesh collider with collision groups matching the table cube
    // Belongs to group 4, collides ONLY with group 2 (phone cord)
    const colliderDesc = RAPIER.ColliderDesc.trimesh(vertices, indices)
      .setFriction(0.7)
      .setRestitution(0.0)
      .setCollisionGroups(0x00020008); // Belongs to group 4, collides with group 2

    this.meshCollider = world.createCollider(
      colliderDesc,
      this.meshColliderBody
    );

    // Debug: Check collision groups
    const collisionGroups = this.meshCollider.collisionGroups();
    this.logger.log(
      `Created trimesh collider from Collider mesh with ${
        vertices.length / 3
      } vertices and ${
        indices.length / 3
      } triangles (world-space geometry, body at origin)`
    );
    this.logger.log(
      `Trimesh collision groups: 0x${collisionGroups
        .toString(16)
        .padStart(8, "0")}`
    );
    this.logger.log(
      `  Belongs to groups: 0x${(collisionGroups & 0xffff)
        .toString(16)
        .padStart(4, "0")}`
    );
    this.logger.log(
      `  Collides with groups: 0x${((collisionGroups >> 16) & 0xffff)
        .toString(16)
        .padStart(4, "0")}`
    );
  }

  /**
   * Update method - call in animation loop
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    // Update the phone cord (handles kinematic anchor and visual line)
    if (this.phoneCord) {
      this.phoneCord.update();
    }

    // Update mesh collider position to follow phone
    // Note: Geometry is in world space (baked at initial phone position)
    // Body position acts as an offset, so we calculate delta from initial position
    if (this.meshColliderBody && this.sceneManager) {
      const phoneObject = this.sceneManager.getObject("candlestickPhone");
      if (phoneObject) {
        const currentPos = new THREE.Vector3();
        phoneObject.getWorldPosition(currentPos);

        // Calculate offset from initial position
        const offset = new THREE.Vector3().subVectors(
          currentPos,
          this.lastPhonePosition
        );

        // Set body position to the offset (geometry is already at initial world position)
        this.meshColliderBody.setTranslation(
          {
            x: offset.x,
            y: offset.y,
            z: offset.z,
          },
          true
        );

        // Debug logging every 60 frames
        if (!this._updateFrameCount) this._updateFrameCount = 0;
        this._updateFrameCount++;
        if (this._updateFrameCount % 60 === 0) {
          this.logger.log(
            `Collider update: phone=(${currentPos.x.toFixed(
              2
            )}, ${currentPos.y.toFixed(2)}, ${currentPos.z.toFixed(
              2
            )}), offset=(${offset.x.toFixed(2)}, ${offset.y.toFixed(
              2
            )}, ${offset.z.toFixed(2)})`
          );
        }
      }
    }
  }

  /**
   * Get receiver object
   * @returns {THREE.Object3D|null}
   */
  getReceiver() {
    return this.receiver;
  }

  /**
   * Get cord attach point
   * @returns {THREE.Object3D|null}
   */
  getCordAttach() {
    return this.cordAttach;
  }

  /**
   * Get phone cord instance
   * @returns {PhoneCord|null}
   */
  getPhoneCord() {
    return this.phoneCord;
  }

  /**
   * Clean up resources
   */
  destroy() {
    // Destroy the phone cord
    if (this.phoneCord) {
      this.phoneCord.destroy();
      this.phoneCord = null;
    }

    // Remove table collider
    if (this.physicsManager && this.tableRigidBody) {
      const world = this.physicsManager.world;
      world.removeRigidBody(this.tableRigidBody);
      this.tableRigidBody = null;
      this.tableCollider = null;
      this.logger.log("Removed table collider");
    }

    // Remove mesh trimesh collider
    if (this.physicsManager && this.meshColliderBody) {
      const world = this.physicsManager.world;
      if (this.meshCollider) {
        world.removeCollider(this.meshCollider, false);
        this.meshCollider = null;
      }
      world.removeRigidBody(this.meshColliderBody);
      this.meshColliderBody = null;
      this.logger.log("Removed mesh trimesh collider");
    }

    this.receiver = null;
    this.cordAttach = null;
    this.logger.log("Destroyed");
  }
}

export default CandlestickPhone;
