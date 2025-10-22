import * as THREE from "three";
import { GAME_STATES } from "../gameData.js";
import { Logger } from "../utils/logger.js";
import { checkCriteria } from "../utils/criteriaHelper.js";
import PhoneCord from "./phoneCord.js";

/**
 * PhoneBooth - Manages phonebooth-specific interactions and animations
 *
 * Features:
 * - Receiver reparenting and lerp animation
 * - Physics-based telephone cord simulation
 * - Audio-reactive light integration
 * - Phone booth state management
 * - Animation callbacks
 */
class PhoneBooth {
  constructor(options = {}) {
    this.sceneManager = options.sceneManager;
    this.lightManager = options.lightManager;
    this.sfxManager = options.sfxManager;
    this.physicsManager = options.physicsManager;
    this.scene = options.scene;
    this.camera = options.camera;
    this.characterController = options.characterController; // Reference to character controller
    this.logger = new Logger("PhoneBooth", false);

    // Receiver animation state
    this.receiverLerp = null;
    this.receiverDropLerp = null; // Lerp for dropping animation
    this.receiver = null;
    this.receiverOriginalWorldPos = null; // Original world position on load
    this.receiverOriginalWorldRot = null; // Original world rotation on load
    this.cordAttach = null;
    this.receiverPositionLocked = false;
    this.lockedReceiverPos = null;
    this.lockedReceiverRot = null;

    // Phone cord (reusable module)
    this.phoneCord = null; // PhoneCord instance
    this.receiverRigidBody = null; // Dynamic rigid body for dropped receiver
    this.receiverCollider = null; // Collider for dropped receiver

    // Configuration
    this.config = {
      receiverTargetPos: new THREE.Vector3(-0.3, 0, -0.3), // Position relative to camera
      receiverTargetRot: new THREE.Euler(-0.5, -0.5, -Math.PI / 2),
      receiverTargetScale: new THREE.Vector3(1.0, 1.0, 1.0), // Scale when held (1.0 = original size)
      receiverLerpDuration: 1.5,
      receiverLerpEase: (t) => 1 - Math.pow(1 - t, 3), // Cubic ease-out

      // Cord existence criteria - cord should exist while state < OFFICE_INTERIOR
      cordCriteria: {
        currentState: { $lt: GAME_STATES.OFFICE_INTERIOR }, // state < 13
      },

      // Receiver physics configuration (for when dropped)
      receiverColliderHeight: 0.215, // Height of cylindrical collider in meters
      receiverColliderRadius: 0.05, // Radius of cylindrical collider in meters
      receiverMass: 0.15, // Mass in kg (phone receivers are ~150g)
      receiverDamping: 0.8, // Linear damping
      receiverAngularDamping: 1.0, // Angular damping
    };
  }

  /**
   * Initialize the phonebooth
   * Sets up event listeners and creates the phone cord
   */
  initialize(gameManager = null) {
    if (!this.sceneManager) {
      this.logger.warn("No SceneManager provided");
      return;
    }

    this.gameManager = gameManager;

    // Listen for animation finished events
    this.sceneManager.on("animation:finished", (animId) => {
      if (animId === "phonebooth-ring") {
        this.handleAnimationFinished();
      }
    });

    // Listen for game state changes
    if (this.gameManager) {
      this.gameManager.on("state:changed", (newState, oldState) => {
        // Check if phone cord should be destroyed based on criteria
        if (this.phoneCord && this.config.cordCriteria) {
          const criteriaMet = checkCriteria(newState, this.config.cordCriteria);
          if (!criteriaMet) {
            this.logger.log("🗑️ Cord criteria no longer met, destroying cord");
            this.phoneCord.destroy();
            this.phoneCord = null;
          }
        }

        // When leaving ANSWERED_PHONE state
        if (
          oldState.currentState === GAME_STATES.ANSWERED_PHONE &&
          newState.currentState !== GAME_STATES.ANSWERED_PHONE
        ) {
          // Stop receiver lerp at current position
          this.stopReceiverLerp();

          // If receiver is attached to camera, lock its local position
          if (this.receiver && this.receiver.parent === this.camera) {
            this.logger.log("Locking receiver position relative to camera");
            // Store the current local position to prevent any transform updates
            const lockedPos = this.receiver.position.clone();
            const lockedRot = this.receiver.rotation.clone();

            // Set a flag to prevent position updates
            this.receiverPositionLocked = true;
            this.lockedReceiverPos = lockedPos;
            this.lockedReceiverRot = lockedRot;
          }
        }

        // When entering DRIVE_BY state, drop the receiver with physics
        if (newState.currentState === GAME_STATES.DRIVE_BY) {
          this.dropReceiverWithPhysics();
        }
      });
    }

    // Find the CordAttach and Receiver meshes
    this.cordAttach = this.sceneManager.findChildByName(
      "phonebooth",
      "CordAttach"
    );
    this.receiver = this.sceneManager.findChildByName("phonebooth", "Receiver");

    // Store receiver's original world position and rotation for drop animation
    if (this.receiver) {
      this.receiverOriginalWorldPos = new THREE.Vector3();
      this.receiverOriginalWorldRot = new THREE.Quaternion();
      this.receiver.getWorldPosition(this.receiverOriginalWorldPos);
      this.receiver.getWorldQuaternion(this.receiverOriginalWorldRot);
      this.logger.log(
        "Stored receiver original position:",
        this.receiverOriginalWorldPos.toArray()
      );

      // If starting in DRIVE_BY or later, position receiver in dropped state
      if (
        this.gameManager &&
        this.gameManager.state &&
        this.gameManager.state.currentState >= GAME_STATES.DRIVE_BY
      ) {
        this.logger.log(
          "Starting in DRIVE_BY or later state, positioning receiver in dropped state"
        );
        this.initializeReceiverInDroppedState();
      }
    }

    if (this.cordAttach && this.receiver && this.physicsManager) {
      // Create the phone cord using the PhoneCord module
      this.phoneCord = new PhoneCord({
        scene: this.scene,
        physicsManager: this.physicsManager,
        cordAttach: this.cordAttach,
        receiver: this.receiver,
        loggerName: "PhoneBooth.Cord",
      });
      this.phoneCord.createCord();
    } else {
      this.logger.warn(
        "Cannot create phone cord - missing CordAttach, Receiver, or PhysicsManager"
      );
    }

    this.logger.log("Initialized");
  }

  /**
   * Handle phonebooth animation finished
   * Called when the phone booth ring animation completes
   */
  handleAnimationFinished() {
    this.logger.log("Ring animation finished, reparenting receiver");

    // Reparent receiver to camera - the cord will follow automatically
    this.reparentReceiver();
  }

  /**
   * Reparent the receiver from the phone booth to the camera
   * Preserves world position and smoothly lerps to target position
   */
  reparentReceiver() {
    if (!this.sceneManager || !this.camera) {
      this.logger.warn("Cannot reparent receiver - missing managers");
      return;
    }

    // Reparent the "Receiver" mesh from phonebooth to camera
    // This preserves world position using THREE.js attach()
    this.receiver = this.sceneManager.reparentChild(
      "phonebooth",
      "Receiver",
      this.camera
    );

    if (this.receiver) {
      // Log receiver transform info for debugging
      const worldPos = new THREE.Vector3();
      this.receiver.getWorldPosition(worldPos);

      this.logger.log("Receiver successfully attached to camera");
      this.logger.log("  Local position:", this.receiver.position.toArray());
      this.logger.log(
        "  Local rotation:",
        this.receiver.rotation.toArray().slice(0, 3)
      );
      this.logger.log("  Local scale:", this.receiver.scale.toArray());
      this.logger.log("  World position:", worldPos.toArray());
      this.logger.log("  Parent:", this.receiver.parent?.type || "none");

      // Disable character physics collisions to prevent cord from pushing character
      if (this.characterController) {
        this.characterController.disablePhysicsCollisions();
      }

      // Start lerp animation to move receiver to target position
      this.startReceiverLerp();
    } else {
      this.logger.warn("Failed to attach receiver to camera");
    }
  }

  /**
   * Start the receiver lerp animation
   * Smoothly moves receiver from its current position to the target position
   */
  startReceiverLerp() {
    if (!this.receiver) {
      this.logger.warn("Cannot start lerp - no receiver");
      return;
    }

    // Convert Euler to Quaternion for smooth interpolation
    const startQuat = this.receiver.quaternion.clone();
    const targetQuat = new THREE.Quaternion().setFromEuler(
      this.config.receiverTargetRot
    );

    this.receiverLerp = {
      object: this.receiver,
      startPos: this.receiver.position.clone(),
      targetPos: this.config.receiverTargetPos,
      startQuat: startQuat,
      targetQuat: targetQuat,
      startScale: this.receiver.scale.clone(),
      targetScale: this.config.receiverTargetScale,
      duration: this.config.receiverLerpDuration,
      elapsed: 0,
    };

    this.logger.log("Starting receiver lerp animation");
  }

  /**
   * Stop the receiver lerp animation
   * Keeps receiver attached to camera and physics following it
   */
  stopReceiverLerp() {
    if (this.receiverLerp) {
      this.logger.log("Stopping receiver lerp");
      // Stop the lerp at its current position
      this.receiverLerp = null;
    }
  }

  /**
   * Start the receiver drop animation
   * Moves receiver to hang below the rigid cord section
   */
  startReceiverDropLerp() {
    if (
      !this.receiver ||
      !this.receiverOriginalWorldPos ||
      !this.receiverOriginalWorldRot ||
      !this.cordAttach ||
      !this.phoneCord
    ) {
      this.logger.warn(
        "Cannot start drop lerp - no receiver or original position"
      );
      return;
    }

    // Current state (where the receiver is now in world space, detached from camera)
    const startPos = this.receiver.position.clone();
    const startQuat = this.receiver.quaternion.clone();

    // Target: Below the end of the rigid cord section
    const cordAttachPos = new THREE.Vector3();
    this.cordAttach.getWorldPosition(cordAttachPos);

    const targetPos = cordAttachPos.clone();
    // Position at end of rigid section (use phoneCord config)
    targetPos.x +=
      this.phoneCord.config.cordSegmentLength *
      this.phoneCord.config.cordRigidSegments;
    targetPos.y -= this.phoneCord.config.cordSegmentLength * 2; // Match curve
    targetPos.y -= 0.3; // Hang below the rigid section

    // Flip the original rotation upside down (180 degrees on X axis)
    const targetQuat = this.receiverOriginalWorldRot.clone();
    const flipRotation = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0),
      Math.PI // 180 degrees
    );
    targetQuat.multiply(flipRotation);

    this.receiverDropLerp = {
      object: this.receiver,
      startPos: startPos,
      targetPos: targetPos,
      startQuat: startQuat,
      targetQuat: targetQuat,
      duration: 0.8, // 0.8 seconds for the drop
      elapsed: 0,
    };

    this.logger.log(
      "Starting receiver drop animation (below rigid cord section)"
    );
    this.logger.log("  Target position:", targetPos.toArray());
  }

  /**
   * Update receiver drop lerp animation
   * @param {number} dt - Delta time in seconds
   */
  updateReceiverDropLerp(dt) {
    if (!this.receiverDropLerp) return;

    this.receiverDropLerp.elapsed += dt;
    const t = Math.min(
      1,
      this.receiverDropLerp.elapsed / this.receiverDropLerp.duration
    );

    // Apply easing (cubic ease-in for falling motion)
    const eased = t * t * t;

    // Lerp position
    this.receiverDropLerp.object.position.lerpVectors(
      this.receiverDropLerp.startPos,
      this.receiverDropLerp.targetPos,
      eased
    );

    // Lerp rotation (using quaternion slerp for smooth interpolation)
    this.receiverDropLerp.object.quaternion.slerpQuaternions(
      this.receiverDropLerp.startQuat,
      this.receiverDropLerp.targetQuat,
      eased
    );

    // Complete animation
    if (t >= 1) {
      this.logger.log("Receiver drop animation complete");
      this.receiverDropLerp = null;
    }
  }

  /**
   * Drop the receiver with physics
   * Detaches from camera and adds a dynamic rigid body so it falls and hangs by the cord
   */
  dropReceiverWithPhysics() {
    if (!this.receiver || !this.physicsManager || !this.phoneCord) {
      this.logger.warn(
        "Cannot drop receiver - missing receiver, physics manager, or phone cord"
      );
      return;
    }

    this.logger.log("Detaching receiver from camera (no physics)");

    // Unlock position so we can move it
    this.receiverPositionLocked = false;
    this.lockedReceiverPos = null;
    this.lockedReceiverRot = null;

    // Get current world position and rotation before reparenting
    const worldPos = new THREE.Vector3();
    const worldQuat = new THREE.Quaternion();
    this.receiver.getWorldPosition(worldPos);
    this.receiver.getWorldQuaternion(worldQuat);

    this.logger.log(
      "Receiver current parent:",
      this.receiver.parent?.name || "none"
    );
    this.logger.log(
      "Receiver world position before detach:",
      worldPos.toArray()
    );

    // Detach from camera and add to scene using THREE.js attach method
    // This preserves world transform
    this.scene.attach(this.receiver);

    this.logger.log(
      "Receiver detached from camera, new parent:",
      this.receiver.parent?.name || "none"
    );
    this.logger.log(
      "Receiver world position after detach:",
      this.receiver.position.toArray()
    );

    // Start drop animation: move 1m down and flip upside down
    this.startReceiverDropLerp();

    // Re-enable character physics collisions now that receiver is dropped
    if (this.characterController) {
      this.characterController.enablePhysicsCollisions();
    }
  }

  /**
   * Update receiver lerp animation
   * @param {number} dt - Delta time in seconds
   */
  updateReceiverLerp(dt) {
    if (!this.receiverLerp) return;

    this.receiverLerp.elapsed += dt;
    const t = Math.min(
      1,
      this.receiverLerp.elapsed / this.receiverLerp.duration
    );

    // Apply easing
    const eased = this.config.receiverLerpEase(t);

    // Lerp position
    this.receiverLerp.object.position.lerpVectors(
      this.receiverLerp.startPos,
      this.receiverLerp.targetPos,
      eased
    );

    // Lerp rotation (using quaternion slerp for smooth interpolation)
    this.receiverLerp.object.quaternion.slerpQuaternions(
      this.receiverLerp.startQuat,
      this.receiverLerp.targetQuat,
      eased
    );

    // Lerp scale
    this.receiverLerp.object.scale.lerpVectors(
      this.receiverLerp.startScale,
      this.receiverLerp.targetScale,
      eased
    );

    // Complete animation
    if (t >= 1) {
      this.logger.log("Receiver lerp animation complete");
      this.receiverLerp = null;
    }
  }

  /**
   * Update method - call in animation loop
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    this.updateReceiverLerp(dt);
    this.updateReceiverDropLerp(dt);

    // If receiver position is locked, enforce it (prevents animation system from moving it)
    if (
      this.receiverPositionLocked &&
      this.receiver &&
      this.lockedReceiverPos
    ) {
      this.receiver.position.copy(this.lockedReceiverPos);
      this.receiver.rotation.copy(this.lockedReceiverRot);
    }

    // If receiver has its own physics body, sync mesh with physics
    if (this.receiverRigidBody && this.receiver) {
      const translation = this.receiverRigidBody.translation();
      const rotation = this.receiverRigidBody.rotation();

      this.receiver.position.set(translation.x, translation.y, translation.z);
      this.receiver.quaternion.set(
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w
      );
    }

    // Update the phone cord (handles kinematic anchor and visual line)
    if (this.phoneCord) {
      this.phoneCord.update();
    }
  }

  /**
   * Set receiver target position
   * @param {THREE.Vector3} position - Target position relative to camera
   */
  setReceiverTargetPosition(position) {
    this.config.receiverTargetPos.copy(position);
  }

  /**
   * Set receiver target rotation
   * @param {THREE.Euler} rotation - Target rotation relative to camera (Euler angles)
   */
  setReceiverTargetRotation(rotation) {
    this.config.receiverTargetRot.copy(rotation);
  }

  /**
   * Set receiver target scale
   * @param {THREE.Vector3} scale - Target scale (1.0 = original size)
   */
  setReceiverTargetScale(scale) {
    this.config.receiverTargetScale.copy(scale);
  }

  /**
   * Set receiver lerp duration
   * @param {number} duration - Duration in seconds
   */
  setReceiverLerpDuration(duration) {
    this.config.receiverLerpDuration = duration;
  }

  /**
   * Get receiver object
   * @returns {THREE.Object3D|null}
   */
  getReceiver() {
    return this.receiver;
  }

  /**
   * Check if receiver is attached to camera
   * @returns {boolean}
   */
  isReceiverAttached() {
    return this.receiver !== null && this.receiver.parent === this.camera;
  }

  /**
   * Initialize receiver in dropped state (for debug spawning into DRIVE_BY or later)
   * Positions the receiver in its final dangling position without animation
   */
  initializeReceiverInDroppedState() {
    if (
      !this.receiver ||
      !this.receiverOriginalWorldPos ||
      !this.receiverOriginalWorldRot ||
      !this.cordAttach ||
      !this.phoneCord
    ) {
      this.logger.warn(
        "Cannot initialize dropped state - missing receiver or original position"
      );
      return;
    }

    // Detach from phonebooth and add to scene
    this.scene.attach(this.receiver);

    // Position below the end of the rigid cord section
    const cordAttachPos = new THREE.Vector3();
    this.cordAttach.getWorldPosition(cordAttachPos);

    const droppedPos = cordAttachPos.clone();
    // Position at end of rigid section (use phoneCord config)
    droppedPos.x +=
      this.phoneCord.config.cordSegmentLength *
      this.phoneCord.config.cordRigidSegments;
    droppedPos.y -= this.phoneCord.config.cordSegmentLength * 2; // Match curve
    droppedPos.y -= 0.3; // Hang below the rigid section

    // Flip the original rotation upside down (180 degrees on X axis)
    const droppedQuat = this.receiverOriginalWorldRot.clone();
    const flipRotation = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0),
      Math.PI
    );
    droppedQuat.multiply(flipRotation);

    // Apply transform
    this.receiver.position.copy(droppedPos);
    this.receiver.quaternion.copy(droppedQuat);

    this.logger.log(
      "Receiver initialized in dropped state below rigid cord at:",
      droppedPos.toArray()
    );

    // Re-enable character physics collisions (they would have been disabled when held)
    if (this.characterController) {
      this.characterController.enablePhysicsCollisions();
    }
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

    if (this.receiver && this.receiver.parent) {
      this.receiver.parent.remove(this.receiver);
    }
    this.receiver = null;
    this.receiverLerp = null;
    this.receiverDropLerp = null;
    this.cordAttach = null;
  }
}

export default PhoneBooth;
