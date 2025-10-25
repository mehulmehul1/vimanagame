import * as THREE from "three";
import { Logger } from "../utils/logger.js";
import { GAME_STATES } from "../gameData.js";
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
    this.phoneObject = null; // Root object for the candlestick phone
    this.cordAttach = null; // Where the cord attaches to the phone base
    this.receiver = null; // The handheld receiver
    this.phoneBody = null; // Main phone body (base + stem)
    this.phoneGroup = null; // Blender top-level group (e.g., "Phone Parent Empty")
    this.colliderMesh = null; // Reference to the Collider mesh

    // Phone cord (reusable module)
    this.phoneCord = null; // PhoneCord instance

    // Physics
    this.tableCollider = null; // Static collider for table beneath phone (cube)
    this.tableRigidBody = null; // Rigid body for table collider
    this.meshCollider = null; // Trimesh collider from Collider mesh
    this.meshColliderBody = null; // Rigid body for trimesh collider
    this.lastPhonePosition = new THREE.Vector3(); // Track phone position for updates

    // Animation state
    this.receiverLerp = null;
    this.phoneBodyLerp = null;
    this.isHeldToCamera = false; // When true, enforce target local transforms
    this.receiverHeldScale = null; // Preserve original receiver scale when held

    // Original world transforms (captured at initialize)
    this.receiverOriginalWorldPos = null;
    this.receiverOriginalWorldQuat = null;
    this.receiverOriginalScale = null;
    this.phoneOriginalWorldPos = null;
    this.phoneOriginalWorldQuat = null;
    this.phoneOriginalScale = null;

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
      // Lerp targets when answering office phone
      receiverTargetPos: new THREE.Vector3(-0.4, 0.05, -0.5),
      receiverTargetRot: new THREE.Euler(-0.5, -0.4, -Math.PI / 2),
      receiverForwardRoll: Math.PI, // additional roll around camera forward (local Z)
      receiverTargetScale: new THREE.Vector3(1.0, 1.0, 1.0),
      phoneBodyTargetPos: new THREE.Vector3(0, -0.5, -0.5), // centered, just under camera
      phoneBodyTargetRot: new THREE.Euler(0.15, 0, 0), // slight up-tilt toward camera
      phoneBodyTargetScale: new THREE.Vector3(1.0, 1.0, 1.0),
      lerpDuration: 1.2,
      lerpEase: (t) => 1 - Math.pow(1 - t, 3), // cubic ease-out
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

    // React to state changes (e.g., OFFICE_PHONE_ANSWERED)
    if (this.gameManager) {
      this.gameManager.on("state:changed", (newState) => {
        if (newState.currentState === GAME_STATES.OFFICE_PHONE_ANSWERED) {
          this.handleOfficePhoneAnswered();
        }
        if (newState.currentState === GAME_STATES.PRE_VIEWMASTER) {
          this.putDownToOriginal();
        }
      });
    }

    // Cache phone root object (used for fallback lerp/parenting)
    this.phoneObject = this.sceneManager.getObject("candlestickPhone");

    // Find the CordAttach and Receiver meshes
    this.cordAttach = this.sceneManager.findChildByName(
      "candlestickPhone",
      "CordAttach"
    );
    this.receiver = this.sceneManager.findChildByName(
      "candlestickPhone",
      "Receiver"
    );
    this.phoneBody = this.sceneManager.findChildByName(
      "candlestickPhone",
      "PhoneBody"
    );
    // Try to find a higher-level group first (matches Blender screenshot)
    this.phoneGroup = this.sceneManager.findChildByName(
      "candlestickPhone",
      "Phone_Parent_Empty"
    );

    if (!this.cordAttach) {
      this.logger.warn("CordAttach mesh not found in candlestickPhone model");
    }

    if (!this.receiver) {
      this.logger.warn("Receiver mesh not found in candlestickPhone model");
    }
    if (!this.phoneBody) {
      this.logger.warn("PhoneBody mesh not found in candlestickPhone model");
    }
    if (!this.phoneGroup) {
      this.logger.log(
        "Phone group 'Phone Parent Empty' not found; will use body/root fallback"
      );
    }

    // Capture original world transforms (for putting it back down)
    if (this.receiver) {
      this.receiverOriginalWorldPos = new THREE.Vector3();
      this.receiverOriginalWorldQuat = new THREE.Quaternion();
      this.receiver.getWorldPosition(this.receiverOriginalWorldPos);
      this.receiver.getWorldQuaternion(this.receiverOriginalWorldQuat);
      this.receiverOriginalScale = this.receiver.scale.clone();
    }
    if (this.phoneGroup || this.phoneBody || this.phoneObject) {
      const obj = this.phoneGroup || this.phoneBody || this.phoneObject;
      const wp = new THREE.Vector3();
      const wq = new THREE.Quaternion();
      obj.getWorldPosition(wp);
      obj.getWorldQuaternion(wq);
      this.phoneOriginalWorldPos = wp.clone();
      this.phoneOriginalWorldQuat = wq.clone();
      this.phoneOriginalScale = obj.scale.clone();
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
   * Handle office phone answered: reparent body and receiver to camera and start lerps
   */
  handleOfficePhoneAnswered() {
    if (!this.sceneManager || !this.camera) {
      this.logger.warn("Cannot handle office phone answer - missing managers");
      return;
    }

    // Reparent receiver to the camera (preserve world transform)
    if (this.receiver && this.receiver.parent !== this.camera) {
      this.receiver = this.sceneManager.reparentChild(
        "candlestickPhone",
        "Receiver",
        this.camera
      );
    }

    // Reparent phone group/body to the camera (preserve world transform)
    if (this.phoneGroup && this.phoneGroup.parent !== this.camera) {
      this.camera.attach(this.phoneGroup);
    } else if (this.phoneBody && this.phoneBody.parent !== this.camera) {
      this.phoneBody = this.sceneManager.reparentChild(
        "candlestickPhone",
        "PhoneBody",
        this.camera
      );
    } else if (this.phoneObject && this.phoneObject.parent !== this.camera) {
      // Final fallback: attach entire phone to camera
      this.camera.attach(this.phoneObject);
    }

    // Start lerps
    if (this.receiver) this.startReceiverLerp();
    if (this.phoneGroup) this.startPhoneGroupLerp();
    else if (this.phoneBody) this.startPhoneBodyLerp();
    else if (this.phoneObject) this.startPhoneObjectLerp();

    // Disable local colliders while held to avoid clipping/pushback
    this.removeHeldPhysics();

    // Mark as held to camera so we can enforce target pose
    this.isHeldToCamera = true;
  }

  /** Start receiver lerp to target relative to camera */
  startReceiverLerp() {
    if (!this.receiver) return;
    const startQuat = this.receiver.quaternion.clone();
    // Base target from configured Euler
    const baseQuat = new THREE.Quaternion().setFromEuler(
      this.config.receiverTargetRot
    );
    // Apply extra roll around camera forward axis
    const rollQuat = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 0, 1),
      this.config.receiverForwardRoll || 0
    );
    const targetQuat = baseQuat.multiply(rollQuat);
    // Capture current scale so we don't accidentally resize the receiver
    this.receiverHeldScale = this.receiver.scale.clone();
    this.receiverLerp = {
      object: this.receiver,
      startPos: this.receiver.position.clone(),
      targetPos: this.config.receiverTargetPos,
      startQuat,
      targetQuat,
      startScale: this.receiver.scale.clone(),
      targetScale: this.receiver.scale.clone(), // keep same scale
      duration: this.config.lerpDuration,
      elapsed: 0,
    };
  }

  /**
   * Put the phone back down on the table: detach from camera and lerp
   * both receiver and body/group back to their original world transforms.
   */
  putDownToOriginal() {
    if (!this.sceneManager) return;

    // Stop held enforcement
    this.isHeldToCamera = false;

    // Detach receiver and body/group back to scene root while preserving world transform
    if (this.receiver && this.receiver.parent === this.camera) {
      this.scene.attach(this.receiver);
    }
    const bodyObj = this.phoneGroup || this.phoneBody || this.phoneObject;
    if (bodyObj && bodyObj.parent === this.camera) {
      this.scene.attach(bodyObj);
    }

    // Recreate local lerps that move from current local to target world
    if (
      this.receiver &&
      this.receiverOriginalWorldPos &&
      this.receiverOriginalWorldQuat
    ) {
      // Convert world targets to current parent's local space
      const parent = this.receiver.parent || this.scene;
      const invParent = new THREE.Matrix4().copy(parent.matrixWorld).invert();

      const targetPosWorld = this.receiverOriginalWorldPos.clone();
      const targetQuatWorld = this.receiverOriginalWorldQuat.clone();
      const targetPosLocal = targetPosWorld.clone().applyMatrix4(invParent);
      const targetQuatLocal = targetQuatWorld.clone();

      this.receiverLerp = {
        object: this.receiver,
        startPos: this.receiver.position.clone(),
        targetPos: targetPosLocal,
        startQuat: this.receiver.quaternion.clone(),
        targetQuat: targetQuatLocal,
        startScale: this.receiver.scale.clone(),
        targetScale: this.receiverOriginalScale
          ? this.receiverOriginalScale.clone()
          : this.receiver.scale.clone(),
        duration: this.config.lerpDuration,
        elapsed: 0,
      };
    }

    if (bodyObj && this.phoneOriginalWorldPos && this.phoneOriginalWorldQuat) {
      const parent = bodyObj.parent || this.scene;
      const invParent = new THREE.Matrix4().copy(parent.matrixWorld).invert();

      const targetPosWorld = this.phoneOriginalWorldPos.clone();
      const targetQuatWorld = this.phoneOriginalWorldQuat.clone();
      const targetPosLocal = targetPosWorld.clone().applyMatrix4(invParent);
      const targetQuatLocal = targetQuatWorld.clone();

      this.phoneBodyLerp = {
        object: bodyObj,
        startPos: bodyObj.position.clone(),
        targetPos: targetPosLocal,
        startQuat: bodyObj.quaternion.clone(),
        targetQuat: targetQuatLocal,
        startScale: bodyObj.scale.clone(),
        targetScale: this.phoneOriginalScale
          ? this.phoneOriginalScale.clone()
          : bodyObj.scale.clone(),
        duration: this.config.lerpDuration,
        elapsed: 0,
      };
    }

    // Optionally restore local physics now that it's back down
    // (leave disabled unless you want cord collisions with the table again)
  }

  /** Start phone body lerp to target relative to camera */
  startPhoneBodyLerp() {
    if (!this.phoneBody) return;
    const startQuat = this.phoneBody.quaternion.clone();
    const targetQuat = new THREE.Quaternion().setFromEuler(
      this.config.phoneBodyTargetRot
    );
    this.phoneBodyLerp = {
      object: this.phoneBody,
      startPos: this.phoneBody.position.clone(),
      targetPos: this.config.phoneBodyTargetPos,
      startQuat,
      targetQuat,
      startScale: this.phoneBody.scale.clone(),
      targetScale: this.config.phoneBodyTargetScale,
      duration: this.config.lerpDuration,
      elapsed: 0,
    };
  }

  /** Start phone group lerp to target relative to camera */
  startPhoneGroupLerp() {
    if (!this.phoneGroup) return;
    const startQuat = this.phoneGroup.quaternion.clone();
    const targetQuat = new THREE.Quaternion().setFromEuler(
      this.config.phoneBodyTargetRot
    );
    this.phoneBodyLerp = {
      object: this.phoneGroup,
      startPos: this.phoneGroup.position.clone(),
      targetPos: this.config.phoneBodyTargetPos,
      startQuat,
      targetQuat,
      startScale: this.phoneGroup.scale.clone(),
      targetScale: this.config.phoneBodyTargetScale,
      duration: this.config.lerpDuration,
      elapsed: 0,
    };
  }

  /** Start whole-phone fallback lerp */
  startPhoneObjectLerp() {
    if (!this.phoneObject) return;
    const startQuat = this.phoneObject.quaternion.clone();
    const targetQuat = new THREE.Quaternion().setFromEuler(
      this.config.phoneBodyTargetRot
    );
    this.phoneBodyLerp = {
      object: this.phoneObject,
      startPos: this.phoneObject.position.clone(),
      targetPos: this.config.phoneBodyTargetPos,
      startQuat,
      targetQuat,
      startScale: this.phoneObject.scale.clone(),
      targetScale: this.config.phoneBodyTargetScale,
      duration: this.config.lerpDuration,
      elapsed: 0,
    };
  }

  /** Remove table + mesh colliders when phone is brought to camera */
  removeHeldPhysics() {
    if (!this.physicsManager) return;
    const world = this.physicsManager.world;
    if (this.tableRigidBody) {
      world.removeRigidBody(this.tableRigidBody);
      this.tableRigidBody = null;
      this.tableCollider = null;
    }
    if (this.meshColliderBody) {
      if (this.meshCollider) {
        world.removeCollider(this.meshCollider, false);
        this.meshCollider = null;
      }
      world.removeRigidBody(this.meshColliderBody);
      this.meshColliderBody = null;
    }
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

    // Update receiver lerp
    if (this.receiverLerp) {
      this.receiverLerp.elapsed += dt;
      const t = Math.min(
        1,
        this.receiverLerp.elapsed / this.receiverLerp.duration
      );
      const eased = this.config.lerpEase(t);
      this.receiverLerp.object.position.lerpVectors(
        this.receiverLerp.startPos,
        this.receiverLerp.targetPos,
        eased
      );
      this.receiverLerp.object.quaternion.slerpQuaternions(
        this.receiverLerp.startQuat,
        this.receiverLerp.targetQuat,
        eased
      );
      this.receiverLerp.object.scale.lerpVectors(
        this.receiverLerp.startScale,
        this.receiverLerp.targetScale,
        eased
      );
      if (t >= 1) this.receiverLerp = null;
    }

    // Update phone body lerp
    if (this.phoneBodyLerp) {
      this.phoneBodyLerp.elapsed += dt;
      const t = Math.min(
        1,
        this.phoneBodyLerp.elapsed / this.phoneBodyLerp.duration
      );
      const eased = this.config.lerpEase(t);
      this.phoneBodyLerp.object.position.lerpVectors(
        this.phoneBodyLerp.startPos,
        this.phoneBodyLerp.targetPos,
        eased
      );
      this.phoneBodyLerp.object.quaternion.slerpQuaternions(
        this.phoneBodyLerp.startQuat,
        this.phoneBodyLerp.targetQuat,
        eased
      );
      this.phoneBodyLerp.object.scale.lerpVectors(
        this.phoneBodyLerp.startScale,
        this.phoneBodyLerp.targetScale,
        eased
      );
      if (t >= 1) this.phoneBodyLerp = null;
    }

    // If held to camera, enforce target local transforms (prevents other systems from drifting it)
    if (this.isHeldToCamera) {
      if (
        this.receiver &&
        !this.receiverLerp &&
        this.receiver.parent === this.camera
      ) {
        this.receiver.position.copy(this.config.receiverTargetPos);
        const baseQuat = new THREE.Quaternion().setFromEuler(
          this.config.receiverTargetRot
        );
        const rollQuat = new THREE.Quaternion().setFromAxisAngle(
          new THREE.Vector3(0, 0, 1),
          this.config.receiverForwardRoll || 0
        );
        this.receiver.quaternion.copy(baseQuat.multiply(rollQuat));
        if (this.receiverHeldScale) {
          this.receiver.scale.copy(this.receiverHeldScale);
        }
      }

      const bodyObj = this.phoneBody || this.phoneObject;
      if (bodyObj && !this.phoneBodyLerp && bodyObj.parent === this.camera) {
        bodyObj.position.copy(this.config.phoneBodyTargetPos);
        bodyObj.quaternion.setFromEuler(this.config.phoneBodyTargetRot);
        bodyObj.scale.copy(this.config.phoneBodyTargetScale);
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
