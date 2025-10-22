import * as THREE from "three";
import { Logger } from "../utils/logger.js";

/**
 * PhoneCord - Manages physics-based telephone cord simulation
 *
 * Features:
 * - Physics-based telephone cord with chain segments
 * - Kinematic anchor that follows the receiver
 * - Visual tube rendering along the cord
 * - Configurable rigid segments at attach point
 * - Natural droop and curve
 *
 * Usage:
 * const phoneCord = new PhoneCord({
 *   scene,
 *   physicsManager,
 *   cordAttach: cordAttachMesh,
 *   receiver: receiverMesh,
 * });
 */
class PhoneCord {
  constructor(options = {}) {
    this.scene = options.scene;
    this.physicsManager = options.physicsManager;
    this.cordAttach = options.cordAttach; // THREE.Object3D where cord attaches (phone booth end)
    this.receiver = options.receiver; // THREE.Object3D for receiver (handheld end)
    this.logger = new Logger(options.loggerName || "PhoneCord", true);

    // Phone cord physics simulation
    this.cordLinks = []; // Array of { rigidBody, mesh, joint }
    this.cordLineMesh = null; // Visual line representation
    this.cordAttachAnchor = null; // Kinematic body that follows the cord attach point
    this.receiverAnchor = null; // Kinematic body that follows the receiver

    // Configuration
    this.config = {
      // Cord configuration
      cordSegments: 12, // Number of links in the chain
      cordSegmentLength: 0.05, // Length of each segment (longer for slack)
      cordSegmentRadius: 0.002, // Radius of each segment (very slender)
      cordMass: 0.002, // Mass of each segment (lighter for natural droop)
      cordDamping: 8.0, // Linear damping (high to prevent wild movement)
      cordAngularDamping: 8.0, // Angular damping (high to prevent spinning)
      cordDroopAmount: 2, // How much the cord droops in the middle (0 = straight, 1+ = more droop)
      cordRigidSegments: 1, // Number of initial segments that use fixed joints (rigid at phone booth end)

      // Initialization mode: "horizontal" or "straight"
      // "horizontal": Extends along X with Y droop (for phonebooth)
      // "straight": Direct line from attach to receiver (for candlestick phone on table)
      initMode: "horizontal",

      // Visual configuration
      cordColor: 0x808080, // Grey color
      cordVisualRadius: 0.008, // Radius of visual tube (thicker than physics collider)
      cordMetalness: 0.3,
      cordRoughness: 0.8,

      // Collision groups: 0x00040002
      // - Belongs to group 2 (0x0002) - Phone cord/receiver
      // - Collides with group 3 (0x0004) - Environment only
      // - Does NOT collide with group 1 (character controller)
      cordCollisionGroup: 0x00040002,

      // Override with user options
      ...options.config,
    };
  }

  /**
   * Create the physics-based phone cord
   * Creates a chain of rigid bodies connected by spherical joints
   */
  createCord() {
    if (!this.physicsManager || !this.cordAttach || !this.receiver) {
      this.logger.warn("Cannot create cord - missing components");
      return false;
    }

    const world = this.physicsManager.world;
    const RAPIER = this.physicsManager.RAPIER;

    // Get world positions of cord attachment points
    const cordAttachPos = new THREE.Vector3();
    const receiverPos = new THREE.Vector3();
    this.cordAttach.getWorldPosition(cordAttachPos);
    this.receiver.getWorldPosition(receiverPos);

    // Calculate cord direction
    const cordDirection = new THREE.Vector3()
      .subVectors(receiverPos, cordAttachPos)
      .normalize();
    const segmentLength = this.config.cordSegmentLength;

    // Calculate total cord length (longer than straight distance for natural droop)
    const straightDistance = cordAttachPos.distanceTo(receiverPos);
    const totalCordLength = this.config.cordSegments * segmentLength;
    const slackFactor = totalCordLength / straightDistance;

    this.logger.log(
      "Creating phone cord with",
      this.config.cordSegments,
      "segments"
    );
    this.logger.log(
      "  Slack factor:",
      slackFactor.toFixed(2),
      "(>1 means cord will droop)"
    );
    this.logger.log(
      "  CordAttach position (phone booth):",
      cordAttachPos.toArray()
    );
    this.logger.log("  Receiver position:", receiverPos.toArray());
    this.logger.log(
      "  Rigid segments:",
      this.config.cordRigidSegments,
      "at attach point end"
    );

    // Create cord segments with initial droop/curve
    for (let i = 0; i < this.config.cordSegments; i++) {
      let pos = new THREE.Vector3();

      // Choose initialization mode
      if (this.config.initMode === "straight") {
        // STRAIGHT MODE: Direct line from attach to receiver (for candlestick phone)
        // No Y droop - prevents segments from spawning inside table geometry
        const t = (i + 0.5) / this.config.cordSegments;
        pos.lerpVectors(cordAttachPos, receiverPos, t);

        if (i < this.config.cordRigidSegments) {
          this.logger.log(
            `  Segment ${i} (RIGID, straight mode): position`,
            pos.toArray()
          );
        }
      } else {
        // HORIZONTAL MODE: X extension with Y droop (for phonebooth)
        // For the FIRST rigid segments (at phone booth), position them to stick out with smooth curve
        if (i < this.config.cordRigidSegments) {
          // Stick out from phone booth with gradual downward curve
          const rigidStep = (i + 1) / this.config.cordRigidSegments;
          pos.copy(cordAttachPos);
          // Extend horizontally outward from phone booth (positive X = right)
          pos.x += rigidStep * segmentLength * this.config.cordRigidSegments; // Extend right
          // Add smooth downward curve (quadratic easing)
          const curveFactor = rigidStep * rigidStep; // Accelerating curve downward
          pos.y -= curveFactor * segmentLength * 2; // Gradual drop
          this.logger.log(
            `  Segment ${i} (RIGID, horizontal mode): position`,
            pos.toArray()
          );
        } else {
          // For flexible segments, calculate position from end of rigid section to receiver
          const flexibleSegmentIndex = i - this.config.cordRigidSegments;
          const totalFlexibleSegments =
            this.config.cordSegments - this.config.cordRigidSegments;
          const flexT = (flexibleSegmentIndex + 0.5) / totalFlexibleSegments;

          // Start position is end of rigid section (with curve)
          const rigidEndPos = new THREE.Vector3().copy(cordAttachPos);
          rigidEndPos.x += segmentLength * this.config.cordRigidSegments; // Full horizontal extension
          rigidEndPos.y -= segmentLength * 2; // Match the curve drop at the end

          // Lerp from end of rigid section to receiver
          pos.lerpVectors(rigidEndPos, receiverPos, flexT);

          // Add vertical droop in the middle (parabolic curve) for natural hanging
          // Maximum droop at flexT=0.5 (middle of flexible cord)
          const droopCurve = Math.sin(flexT * Math.PI); // 0 at ends, 1 at middle
          const droopOffset = droopCurve * this.config.cordDroopAmount;
          pos.y -= droopOffset; // Pull down by droop amount
        }
      }

      // Create rigid body for this segment
      // For rigid segments at phone booth, make them KINEMATIC (unaffected by gravity)
      let rigidBodyDesc;
      if (i < this.config.cordRigidSegments) {
        rigidBodyDesc =
          RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(
            pos.x,
            pos.y,
            pos.z
          );
        this.logger.log(`  Segment ${i} is KINEMATIC (won't fall)`);
      } else {
        rigidBodyDesc = RAPIER.RigidBodyDesc.dynamic()
          .setTranslation(pos.x, pos.y, pos.z)
          .setLinearDamping(this.config.cordDamping)
          .setAngularDamping(this.config.cordAngularDamping);
      }

      const rigidBody = world.createRigidBody(rigidBodyDesc);

      // Create collider (small sphere)
      const colliderDesc = RAPIER.ColliderDesc.ball(
        this.config.cordSegmentRadius
      )
        .setMass(this.config.cordMass)
        .setCollisionGroups(this.config.cordCollisionGroup);

      const collider = world.createCollider(colliderDesc, rigidBody);

      // Debug: Log collision groups for first segment
      if (i === 0) {
        const groups = collider.collisionGroups();
        this.logger.log(
          `  Segment 0 collision groups: 0x${groups
            .toString(16)
            .padStart(8, "0")}`
        );
        this.logger.log(
          `    Belongs to: 0x${(groups & 0xffff).toString(16).padStart(4, "0")}`
        );
        this.logger.log(
          `    Collides with: 0x${((groups >> 16) & 0xffff)
            .toString(16)
            .padStart(4, "0")}`
        );
      }

      // No visual mesh per segment - we'll use the line renderer instead
      const mesh = null;

      // Create joint to previous segment or anchor point
      let joint = null;
      if (i === 0) {
        // First segment - attach to CordAttach with a FIXED joint (rigid at phone booth)
        // We'll create a "virtual" kinematic body at the cord attach point
        const anchorBodyDesc =
          RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(
            cordAttachPos.x,
            cordAttachPos.y,
            cordAttachPos.z
          );

        // Rotate anchor body to point horizontally right (same as rigid segments)
        const anchorRotation = new THREE.Quaternion().setFromAxisAngle(
          new THREE.Vector3(0, 0, 1),
          -Math.PI / 2 // -90 degrees (pointing right)
        );
        anchorBodyDesc.setRotation({
          x: anchorRotation.x,
          y: anchorRotation.y,
          z: anchorRotation.z,
          w: anchorRotation.w,
        });

        const anchorBody = world.createRigidBody(anchorBodyDesc);

        // Store anchor body for updating in animation loop
        this.cordAttachAnchor = anchorBody;

        // Create FIXED joint for first segment (kinematic segments don't need special joint config)
        const params = RAPIER.JointData.fixed(
          { x: 0, y: 0, z: 0 }, // Anchor on the fixed point
          { w: 1.0, x: 0, y: 0, z: 0 }, // Rotation at anchor
          { x: 0, y: 0, z: 0 }, // Anchor on the segment
          { w: 1.0, x: 0, y: 0, z: 0 } // Rotation at segment
        );

        joint = world.createImpulseJoint(params, anchorBody, rigidBody, true);

        // Store anchor body reference
        this.cordLinks.push({
          rigidBody: anchorBody,
          mesh: null,
          joint: null,
          isAnchor: true,
          isCordAttachAnchor: true,
        });
      } else if (i < this.config.cordRigidSegments) {
        // First N segments after anchor - use FIXED joints (kinematic segments stay in place)
        const prevLink = this.cordLinks[this.cordLinks.length - 1];

        const params = RAPIER.JointData.fixed(
          { x: 0, y: 0, z: 0 }, // Anchor on previous segment
          { w: 1.0, x: 0, y: 0, z: 0 }, // Rotation at previous
          { x: 0, y: 0, z: 0 }, // Anchor on current segment
          { w: 1.0, x: 0, y: 0, z: 0 } // Rotation at current
        );

        joint = world.createImpulseJoint(
          params,
          prevLink.rigidBody,
          rigidBody,
          true
        );
      } else {
        // Remaining segments - use rope joints (strict max length)
        const prevLink = this.cordLinks[this.cordLinks.length - 1];

        // Use a rope joint with generous slack to prevent constraint fighting
        const params = RAPIER.JointData.rope(
          segmentLength * 1.5, // Max length (50% slack for flexibility)
          { x: 0, y: 0, z: 0 }, // Center of previous segment
          { x: 0, y: 0, z: 0 } // Center of current segment
        );

        joint = world.createImpulseJoint(
          params,
          prevLink.rigidBody,
          rigidBody,
          true
        );
      }

      this.cordLinks.push({
        rigidBody,
        mesh,
        joint,
        isAnchor: false,
      });
    }

    // Create receiver anchor (kinematic body that will follow the receiver)
    const receiverAnchorDesc =
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(
        receiverPos.x,
        receiverPos.y,
        receiverPos.z
      );
    this.receiverAnchor = world.createRigidBody(receiverAnchorDesc);

    // Attach last segment to receiver anchor with rope joint (flexible at receiver)
    const lastLink = this.cordLinks[this.cordLinks.length - 1];
    const lastJointParams = RAPIER.JointData.rope(
      segmentLength * 1.5, // Max length (50% slack for flexibility)
      { x: 0, y: 0, z: 0 }, // Last segment center
      { x: 0, y: 0, z: 0 } // Receiver anchor center
    );
    const lastJoint = world.createImpulseJoint(
      lastJointParams,
      lastLink.rigidBody,
      this.receiverAnchor,
      true
    );

    // Store reference to last joint
    this.cordLinks.push({
      rigidBody: this.receiverAnchor,
      mesh: null,
      joint: lastJoint,
      isAnchor: true,
      isReceiverAnchor: true,
    });

    // Create visual line to represent the cord
    this.createCordLine();

    this.logger.log("Phone cord created successfully");
    return true;
  }

  /**
   * Create a visual line mesh for the phone cord
   */
  createCordLine() {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array((this.config.cordSegments + 2) * 3);
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    // Use TubeGeometry for a thicker, 3D cord
    const material = new THREE.MeshStandardMaterial({
      color: this.config.cordColor,
      metalness: this.config.cordMetalness,
      roughness: this.config.cordRoughness,
      wireframe: false, // Ensure solid mesh, not wireframe
    });

    // We'll update this to use a tube in the update method
    // For now, create a basic mesh that we'll replace with tube geometry
    this.cordLineMesh = new THREE.Mesh(geometry, material);
    this.cordLineMesh.renderOrder = 1; // Render after most objects
    this.scene.add(this.cordLineMesh);
  }

  /**
   * Update the visual line to match physics simulation
   */
  updateCordLine() {
    if (!this.cordLineMesh || !this.cordAttach || !this.receiver) return;

    // Collect all points along the cord
    const points = [];

    // Start point (CordAttach)
    const cordAttachPos = new THREE.Vector3();
    this.cordAttach.getWorldPosition(cordAttachPos);
    points.push(cordAttachPos.clone());

    // Cord segments (skip the anchor, start from actual segments)
    for (let i = 1; i < this.cordLinks.length; i++) {
      const link = this.cordLinks[i];
      if (link.isAnchor) continue;

      const translation = link.rigidBody.translation();
      points.push(
        new THREE.Vector3(translation.x, translation.y, translation.z)
      );
    }

    // End point (Receiver)
    const receiverPos = new THREE.Vector3();
    this.receiver.getWorldPosition(receiverPos);
    points.push(receiverPos.clone());

    // Create a smooth curve through the points
    const curve = new THREE.CatmullRomCurve3(points);

    // Create tube geometry along the curve
    const tubeGeometry = new THREE.TubeGeometry(
      curve,
      points.length * 2, // segments
      this.config.cordVisualRadius, // radius (thicker than the physics collider)
      8, // radial segments
      false // not closed
    );

    // Replace the old geometry
    if (this.cordLineMesh.geometry) {
      this.cordLineMesh.geometry.dispose();
    }
    this.cordLineMesh.geometry = tubeGeometry;
  }

  /**
   * Update method - call in animation loop
   * Updates the kinematic anchors to follow the cordAttach and receiver
   */
  update() {
    // Update kinematic anchor to follow the cord attach point
    if (this.cordAttachAnchor && this.cordAttach) {
      const cordAttachPos = new THREE.Vector3();
      this.cordAttach.getWorldPosition(cordAttachPos);
      this.cordAttachAnchor.setTranslation(
        { x: cordAttachPos.x, y: cordAttachPos.y, z: cordAttachPos.z },
        true
      );
    }

    // Update kinematic anchor to follow the receiver
    if (this.receiverAnchor && this.receiver) {
      const receiverPos = new THREE.Vector3();
      this.receiver.getWorldPosition(receiverPos);
      this.receiverAnchor.setTranslation(
        { x: receiverPos.x, y: receiverPos.y, z: receiverPos.z },
        true
      );
    }

    // Update the visual cord
    if (this.cordLinks.length > 0) {
      this.updateCordLine();
    }
  }

  /**
   * Destroy the phone cord physics and visuals
   */
  destroy() {
    if (!this.physicsManager) return;

    const world = this.physicsManager.world;

    // Remove all cord links
    for (const link of this.cordLinks) {
      if (link.joint) {
        world.removeImpulseJoint(link.joint, true);
      }
      if (link.rigidBody) {
        world.removeRigidBody(link.rigidBody);
      }
      if (link.mesh) {
        this.scene.remove(link.mesh);
        link.mesh.geometry.dispose();
        link.mesh.material.dispose();
      }
    }

    this.cordLinks = [];
    this.cordAttachAnchor = null;
    this.receiverAnchor = null;

    // Remove line mesh
    if (this.cordLineMesh) {
      this.scene.remove(this.cordLineMesh);
      this.cordLineMesh.geometry.dispose();
      this.cordLineMesh.material.dispose();
      this.cordLineMesh = null;
    }

    this.logger.log("Phone cord destroyed");
  }
}

export default PhoneCord;
