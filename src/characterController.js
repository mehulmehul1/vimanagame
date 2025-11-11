import * as THREE from "three";
import { Howl } from "howler";
import { Logger } from "./utils/logger.js";
import { GAME_STATES } from "./gameData.js";

class CharacterController {
  constructor(
    character,
    camera,
    renderer,
    inputManager,
    sfxManager = null,
    sparkRenderer = null,
    idleHelper = null,
    initialRotation = null
  ) {
    this.character = character;
    this.camera = camera;
    this.renderer = renderer;
    this.inputManager = inputManager;
    this.sfxManager = sfxManager;
    this.sparkRenderer = sparkRenderer;
    this.idleHelper = idleHelper;
    this.logger = new Logger("CharacterController", true);

    // Camera rotation (use provided initial rotation or default to -180 degrees)
    const defaultYaw = THREE.MathUtils.degToRad(-180);
    this.logger.log("CharacterController initialRotation:", initialRotation);
    this.yaw = initialRotation
      ? THREE.MathUtils.degToRad(initialRotation.y)
      : defaultYaw;
    this.pitch = initialRotation
      ? THREE.MathUtils.degToRad(initialRotation.x || 0)
      : 0;
    this.targetYaw = this.yaw;
    this.targetPitch = this.pitch;
    this.logger.log(
      `CharacterController initial yaw: ${THREE.MathUtils.radToDeg(
        this.yaw
      )}°, pitch: ${THREE.MathUtils.radToDeg(this.pitch)}°`
    );

    // Set camera rotation immediately so it's correct from the start
    if (initialRotation) {
      const euler = new THREE.Euler(this.pitch, this.yaw, 0, "YXZ");
      this.camera.quaternion.setFromEuler(euler);
    }

    // Body rotation - separate from camera, only updates during movement
    this.bodyYaw = this.yaw; // Body faces same direction as initial camera

    // Camera look-at system
    this.isLookingAt = false;
    this.lookAtTarget = null;
    this.lookAtPositionFunction = null; // Function to get dynamic lookat position
    this.lookAtDuration = 0;
    this.lookAtProgress = 0;
    this.lookAtStartQuat = new THREE.Quaternion();
    this.lookAtEndQuat = new THREE.Quaternion();
    this.lookAtOnComplete = null;
    this.lookAtDisabledInput = false;
    this.inputDisabled = false;
    this.cameraSyncDisabled = false; // When true, camera won't sync to physics body
    this.frozenCameraBasePosition = new THREE.Vector3(); // Base position when camera is frozen
    this.lookAtReturnToOriginalView = false;
    this.lookAtReturnDuration = 0;
    this.lookAtReturning = false;
    this.lookAtHolding = false;
    this.lookAtHoldTimer = 0;
    this.lookAtHoldDuration = 0;

    // Character move-to system
    this.isMovingTo = false;
    this.moveToStartPos = new THREE.Vector3();
    this.moveToTargetPos = new THREE.Vector3();
    this.moveToStartYaw = 0;
    this.moveToTargetYaw = 0;
    this.moveToStartPitch = 0;
    this.moveToTargetPitch = 0;
    this.moveToDuration = 0;
    this.moveToProgress = 0;
    this.moveToOnComplete = null;
    this.moveToInputControl = null;
    this.moveToRestoreInput = true; // Default to true

    // Depth of Field system
    this.dofEnabled = true; // Can be controlled externally
    this.baseApertureSize = 0.01; // Base aperture from options menu
    this.baseFocalDistance = 6.0; // Base focal distance from options menu
    this.currentFocalDistance = this.baseFocalDistance;
    this.currentApertureSize = this.baseApertureSize;
    this.targetFocalDistance = this.baseFocalDistance;
    this.targetApertureSize = this.baseApertureSize;
    this.dofTransitioning = false;
    this.lookAtDofActive = false; // Track if we're in look-at DoF mode
    this.dofHoldTimer = 0; // Time to hold DoF after look-at completes
    this.dofHoldDuration = 2.0; // Hold DoF for 2 seconds
    this.dofTransitionStartProgress = 0.8; // Start DoF transition at 80% of look-at animation
    this.dofTransitionDuration = 2; // How long the DoF transition takes in seconds
    this.dofTransitionProgress = 0; // Current progress of DoF transition (0 to 1)
    this.returnTransitionDuration = null; // Override transition duration during return-to-original

    // FOV Zoom system (synced with DoF)
    this.baseFov = null; // Will be set from camera's initial FOV
    this.currentFov = null;
    this.targetFov = null;
    this.startFov = null; // Captured at start of each transition
    this.zoomTransitioning = false;
    this.zoomTransitionProgress = 0; // Separate progress for zoom
    this.zoomFactor = 1.5; // 15% zoom
    this.lookAtZoomActive = false;

    // Headbob state
    this.headbobTime = 0;
    this.headbobIntensity = 0;
    this.idleHeadbobTime = 0;
    this.headbobEnabled = true;

    // Viewmaster insanity buildup - smoothed for gradual ramp-down
    this.insanityIntensitySmoothed = 0.0; // Current smoothed value (0.0 to 1.0)
    this.insanityRampDownSpeed = 0.5; // How fast it ramps down per second (0.0 to 1.0 over ~3 seconds)

    // Idle glance system
    this.glanceEnabled = true; // Enable idle look-around behavior
    this.glanceState = null; // null, 'glancing', 'returning'
    this.glanceProgress = 0; // 0 to 1
    this.glanceDuration = 5.0; // Duration of one glance animation (set randomly per glance)
    this.glanceTimer = 0; // Time until next glance (0 = start immediately when idle)
    this.wasIdleAllowed = false; // Track previous idle state for edge detection
    this.glanceStartYaw = 0;
    this.glanceTargetYaw = 0;
    this.glanceStartPitch = 0;
    this.glanceTargetPitch = 0;
    this.glanceStartRoll = 0;
    this.glanceTargetRoll = 0;
    this.currentRoll = 0; // Current head tilt
    this.isLerpingRollToZero = false; // Flag for smooth roll transition after blended animations
    this.rollLerpSpeed = 2.0; // Speed at which roll lerps back to 0 (radians per second)
    this.gizmoRollSpeed = 0.1; // Speed for manual roll adjustment in gizmo mode (radians per second)

    // Audio
    this.audioListener = new THREE.AudioListener();
    this.camera.add(this.audioListener);
    this.footstepSound = null;
    this.isPlayingFootsteps = false;

    // First-person body model
    // Master toggle: set to true to enable loading/animating first-person body, false to disable all body code
    this.enableFirstPersonBody = false;
    this.bodyModel = null; // The loaded GLTF scene
    this.bodyModelGroup = null; // Container for the body model (from scene manager)
    this.bodyAnimationMixer = null; // Animation mixer for the body
    this.walkAnimation = null; // The walk animation action
    this.idleAnimation = null; // The idle animation action
    this.currentBodyAnimation = null; // Track which animation is currently active
    this.sceneManager = null; // Will be set via setSceneManager()

    // Neck rotation limits (in radians)
    this.neckRotationLimit = Math.PI / 6; // 30 degrees max neck rotation (more sensitive)

    // Character container - holds the body, positioned/rotated with character
    // Camera remains separate but follows this container
    this.characterContainer = new THREE.Group();
    // Body will be positioned at (0, bodyOffsetY, 0) relative to container

    // Settings
    this.baseSpeed = 2.5;
    this.minSpeed = 0.5; // Minimum movement speed
    this.maxSpeed = 10.0; // Maximum movement speed
    this.speedIncrement = 0.5; // Speed adjustment increment
    this.sprintMultiplier = 1.75; // Reduced from 2.0 (30% slower sprint)
    this.cameraHeight = 0.9; // Camera offset from capsule center (eye height)
    this.cameraSmoothingFactor = 0.15;

    // Flight mode (for gizmo debug mode)
    this.flightMode = false;
    this.flightSpeed = 5.0; // Speed for vertical flight
    this.verticalInput = 0; // Q/E input for up/down

    // Reusable Vector3 objects to avoid allocations per frame
    this._tempForward = new THREE.Vector3();
    this._tempRight = new THREE.Vector3();
    this._desired = new THREE.Vector3();
    this._up = new THREE.Vector3(0, 1, 0); // World up vector (reusable)
    this._verticalVelocity = new THREE.Vector3(); // For flight mode
    this._cameraOffset = new THREE.Vector3(); // For camera positioning
    this._camFollow = new THREE.Vector3(); // For camera follow position
    this._forward = new THREE.Vector3(); // For getForwardRightVectors
    this._right = new THREE.Vector3(); // For getForwardRightVectors
    this._yAxis = new THREE.Vector3(0, 1, 0); // Y axis for rotations

    // Initialize FOV from camera
    this.baseFov = this.camera.fov;
    this.currentFov = this.baseFov;
    this.targetFov = this.baseFov;

    // Point frustum checking (only during CURSOR gameplay)
    this.pointFrustumCheckTimer = 0;
    this.pointFrustumCheckInterval = 1.0 / 3.0; // Check every 1/3 second
    this.runeSightings = 0; // Track number of rune sightings (0, 1, 2, etc.)
    this.lastSightedRuneLabel = null; // Track which rune was last sighted to prevent duplicate triggers
    this.lastViewmasterEquipped = false; // Track previous viewmaster state to detect when it's just equipped
    this.lastViewmasterTransitioning = false; // Track previous transitioning state
    this.lastFractalIntensity = 0; // Track previous fractal intensity to detect when it starts
    // Cached objects for frustum checks (reuse to avoid allocations)
    this._frustumCheckFrustum = new THREE.Frustum();
    this._frustumCheckMatrix = new THREE.Matrix4();
    this._frustumCheckVector = new THREE.Vector3();

    this.loadFootstepAudio();
  }

  /**
   * Set the scene manager reference (called after initialization)
   * @param {SceneManager} sceneManager - The scene manager instance
   */
  setSceneManager(sceneManager) {
    this.sceneManager = sceneManager;

    // Add character container to scene
    if (this.sceneManager.scene && this.characterContainer) {
      this.sceneManager.scene.add(this.characterContainer);
      this.logger.log("Character container added to scene");
    }

    // Try to attach the body model if it's already loaded
    this.attachFirstPersonBody();
  }

  /**
   * Set the physics manager reference (called after initialization)
   * @param {PhysicsManager} physicsManager - The physics manager instance
   */
  setPhysicsManager(physicsManager) {
    this.physicsManager = physicsManager;
    this.logger.log("Physics manager reference set");
  }

  /**
   * Attach the first-person body model from scene manager to camera
   * Called automatically when scene manager is set, or can be called manually
   */
  attachFirstPersonBody() {
    if (!this.enableFirstPersonBody) {
      return; // Body system disabled
    }

    if (!this.sceneManager) {
      this.logger.warn("Scene manager not set yet");
      return;
    }

    // Get the body model from scene manager
    const bodyObject = this.sceneManager.getObject("firstPersonBody");

    if (!bodyObject) {
      this.logger.warn("First-person body not loaded yet (will retry)");
      // Try again in a moment (model might still be loading)
      setTimeout(() => this.attachFirstPersonBody(), 100);
      return;
    }

    this.logger.log("Attaching first-person body to camera");
    this.logger.log("  - Body object:", bodyObject);
    this.logger.log("  - Body visible:", bodyObject.visible);
    this.logger.log("  - Body world position:", bodyObject.position);

    // The bodyObject is the container group from scene manager
    this.bodyModelGroup = bodyObject;

    // Find the actual model inside the container
    this.bodyModel = null;
    bodyObject.traverse((child) => {
      // Find the root mesh group (first non-container object with children)
      if (!this.bodyModel && child !== bodyObject && child.type === "Group") {
        this.bodyModel = child;
      }
    });

    if (!this.bodyModel) {
      // Fallback: if no group found, the bodyObject itself might be the model
      this.bodyModel = bodyObject;
    }

    // Hide head meshes for first-person view
    let minX = Infinity,
      maxX = -Infinity;
    let minY = Infinity,
      maxY = -Infinity;
    let minZ = Infinity,
      maxZ = -Infinity;
    let headMeshFound = false;

    this.bodyModel.traverse((child) => {
      if (child.isMesh) {
        this.logger.log(`  - Mesh: ${child.name}`);

        // Try to automatically hide the head mesh
        // Common naming patterns for head meshes
        const lowerName = child.name.toLowerCase();
        if (
          lowerName.includes("head") ||
          lowerName.includes("hair") ||
          lowerName.includes("face") ||
          lowerName.includes("skull")
        ) {
          child.visible = false;
          headMeshFound = true;
          this.logger.log(`    (Hidden - head mesh)`);
        }

        // Calculate bounding box to understand model dimensions
        if (child.geometry) {
          child.geometry.computeBoundingBox();
          const box = child.geometry.boundingBox;
          if (box) {
            minX = Math.min(minX, box.min.x);
            maxX = Math.max(maxX, box.max.x);
            minY = Math.min(minY, box.min.y);
            maxY = Math.max(maxY, box.max.y);
            minZ = Math.min(minZ, box.min.z);
            maxZ = Math.max(maxZ, box.max.z);
          }
        }
      }
    });

    const width = maxX - minX;
    const height = maxY - minY;
    const depth = maxZ - minZ;

    this.logger.log(`  - Body Model 3D Bounds:`);
    this.logger.log(
      `    X: ${minX.toFixed(2)} to ${maxX.toFixed(2)} (width: ${width.toFixed(
        2
      )})`
    );
    this.logger.log(
      `    Y: ${minY.toFixed(2)} to ${maxY.toFixed(
        2
      )} (height: ${height.toFixed(2)})`
    );
    this.logger.log(
      `    Z: ${minZ.toFixed(2)} to ${maxZ.toFixed(2)} (depth: ${depth.toFixed(
        2
      )})`
    );
    if (headMeshFound) {
      this.logger.log(`  - Head mesh(es) automatically hidden`);
    }

    // Parent body to character container
    // Body will be offset below the container origin
    this.bodyModelGroup.position.set(0, -0.85, 0.165); // Below character center
    this.bodyModelGroup.rotation.set(0, Math.PI, 0); // Rotate 180 degrees to face forward
    this.characterContainer.add(this.bodyModelGroup);

    this.logger.log(
      "Body parented to character container at offset (0, -0.8, 0)"
    );

    // Setup animations
    this.setupBodyAnimations();
  }

  /**
   * Setup animations for the first-person body model
   * @private
   */
  setupBodyAnimations() {
    if (!this.enableFirstPersonBody) return; // Body system disabled

    if (!this.bodyModel || !this.bodyModelGroup) return;

    // Check if SceneManager already created a mixer for this object
    const existingMixer =
      this.sceneManager?.animationMixers.get("firstPersonBody");

    this.logger.log("Setting up body animations");
    this.logger.log(
      "  - Existing mixer from SceneManager:",
      existingMixer ? "YES" : "NO"
    );

    if (existingMixer) {
      // Use SceneManager's existing mixer - get the action for manual control
      this.bodyAnimationMixer = existingMixer;
      const actions = Array.from(
        this.sceneManager.animationActions.entries()
      ).filter(([id]) => id.startsWith("firstPersonBody"));

      this.logger.log("  - Found actions from SceneManager:", actions.length);

      // Log all available actions for debugging
      this.logger.log("  - Available actions:");
      actions.forEach(([id, action]) => {
        this.logger.log(`    * "${id}"`);
      });

      // Find walk and idle animations
      for (const [id, action] of actions) {
        if (id.includes("walk")) {
          this.walkAnimation = action;
          this.logger.log(`  - Using walk animation: "${id}"`);
        } else if (id.includes("idle")) {
          this.idleAnimation = action;
          this.logger.log(`  - Using idle animation: "${id}"`);
        }
      }

      if (this.walkAnimation && this.idleAnimation) {
        // Setup walk animation
        this.walkAnimation.loop = THREE.LoopRepeat;
        this.walkAnimation.clampWhenFinished = false;
        this.walkAnimation.setEffectiveWeight(0); // Start with no weight
        this.walkAnimation.play();

        // Setup idle animation
        this.idleAnimation.loop = THREE.LoopRepeat;
        this.idleAnimation.clampWhenFinished = false;
        this.idleAnimation.setEffectiveWeight(1.0); // Start with full weight (idle by default)
        this.idleAnimation.play();

        this.currentBodyAnimation = this.idleAnimation;

        this.logger.log(
          `Setup animation blending:`,
          "Walk:",
          this.walkAnimation.isRunning(),
          "Idle:",
          this.idleAnimation.isRunning()
        );
      } else {
        this.logger.warn(
          "Could not find all animations (walk/idle). Found:",
          actions.map(([id]) => id)
        );
      }
    } else {
      // No sceneData animations - find them in the loaded GLTF
      this.logger.log(
        "  - No SceneManager mixer, searching for animations in GLTF..."
      );
      let animations = null;
      this.bodyModel.traverse((node) => {
        if (node.animations?.length > 0) animations = node.animations;
      });

      if (!animations?.length) {
        this.logger.warn("No animations found in body model");
        return;
      }

      this.logger.log(`  - Found ${animations.length} animation(s) in GLTF`);

      // Create our own mixer for manual control
      this.bodyAnimationMixer = new THREE.AnimationMixer(this.bodyModel);
      const walkClip = animations[0];
      this.walkAnimation = this.bodyAnimationMixer.clipAction(walkClip);
      this.walkAnimation.loop = THREE.LoopRepeat;
      this.walkAnimation.play();
      this.walkAnimation.paused = false; // Start playing immediately for testing
      this.walkAnimation.timeScale = 1.0;

      this.logger.log("Created animation mixer for body:", walkClip.name);
    }
  }

  /**
   * Set the idle helper reference (called after initialization)
   * @param {IdleHelper} idleHelper - The idle helper instance
   */
  setIdleHelper(idleHelper) {
    this.idleHelper = idleHelper;
  }

  /**
   * Set game manager and register event listeners
   * @param {GameManager} gameManager - The game manager instance
   */
  setGameManager(gameManager) {
    this.gameManager = gameManager;

    // Listen for camera:lookat events
    this.gameManager.on("camera:lookat", (data) => {
      // Log when lookat event is received
      if (data.id === "runeLookat") {
        this.logger.warn(
          `Received camera:lookat event for rune, control enabled: ${this.gameManager.isControlEnabled()}`
        );
      }

      // Check if control is enabled
      if (!this.gameManager.isControlEnabled()) {
        if (data.id === "runeLookat") {
          this.logger.warn("Rune lookat blocked - control not enabled");
        }
        return;
      }

      // Support dynamic position function for tracking moving objects
      let targetPos;
      if (typeof data.position === "function") {
        // Function returns position - evaluate it now and store function for updates
        const pos = data.position(this.gameManager);
        targetPos = new THREE.Vector3(pos.x, pos.y, pos.z);
        this.lookAtPositionFunction = data.position; // Store function for continuous updates

        // If this is the letter lookat, listen for animation finish to complete the lookat
        if (data.id === "letterLookat" && this.gameManager.sceneManager) {
          let finishHandler = null;
          let timeoutId = null;

          // Handler for animation finish event
          finishHandler = (animId) => {
            if (animId === "letter-anim") {
              // Letter animation finished - complete the lookat and restore controls
              this._completeDynamicLookat();
              if (finishHandler && this.gameManager.sceneManager) {
                // Remove listener manually since sceneManager doesn't have off()
                const listeners =
                  this.gameManager.sceneManager.eventListeners?.[
                    "animation:finished"
                  ];
                if (listeners) {
                  const index = listeners.indexOf(finishHandler);
                  if (index > -1) {
                    listeners.splice(index, 1);
                  }
                }
              }
              if (timeoutId) {
                clearTimeout(timeoutId);
                timeoutId = null;
              }
            }
          };

          // Listen for animation finish
          this.gameManager.sceneManager.on("animation:finished", finishHandler);

          // Fallback timeout: complete lookat after animation duration
          // Animation is ~5 seconds at 0.5x speed = 10 seconds, add 1 second buffer
          this._letterLookatTimeout = setTimeout(() => {
            this._completeDynamicLookat();
            if (finishHandler && this.gameManager.sceneManager) {
              // Remove listener manually since sceneManager doesn't have off()
              const listeners =
                this.gameManager.sceneManager.eventListeners?.[
                  "animation:finished"
                ];
              if (listeners) {
                const index = listeners.indexOf(finishHandler);
                if (index > -1) {
                  listeners.splice(index, 1);
                }
              }
            }
            this._letterLookatTimeout = null;
          }, 11000);
          timeoutId = this._letterLookatTimeout;
        }
      } else {
        // Static position
        targetPos = new THREE.Vector3(
          data.position.x,
          data.position.y,
          data.position.z
        );
        this.lookAtPositionFunction = null; // Clear any previous function
      }

      const onComplete = data.onComplete || null;
      const enableZoom =
        data.enableZoom !== undefined ? data.enableZoom : false;
      const zoomOptions = data.zoomOptions || {};
      const returnToOriginalView = data.returnToOriginalView || false;
      const returnDuration = data.returnDuration || data.duration;
      const holdDuration = data.holdDuration || 0;
      const restoreInput =
        data.restoreInput !== undefined ? data.restoreInput : true; // Default to true

      // Store restoreInput for use when completing dynamic lookat
      this.lookAtRestoreInput = restoreInput;

      this.lookAt(
        targetPos,
        data.duration,
        onComplete,
        enableZoom,
        zoomOptions,
        true, // Always disable input during lookat
        returnToOriginalView,
        returnDuration,
        holdDuration
      );
    });

    // Listen for character:moveto events
    this.gameManager.on("character:moveto", (data) => {
      // Check if control is enabled
      if (!this.gameManager.isControlEnabled()) return;

      let targetY = data.position.y;

      // Validate Y coordinate
      if (targetY === undefined || !isFinite(targetY)) {
        const currentPos = this.character.translation();
        targetY = currentPos.y;
        this.logger.warn(
          `MoveTo: Invalid Y coordinate provided, using current Y=${targetY.toFixed(
            2
          )}`
        );
      }

      const targetPos = new THREE.Vector3(
        data.position.x,
        targetY,
        data.position.z
      );

      // Parse rotation - lookat takes precedence over rotation
      let targetRotation = null;
      if (data.lookat) {
        // Calculate rotation to look at the target position from the destination
        const lookAtPos = new THREE.Vector3(
          data.lookat.x,
          data.lookat.y,
          data.lookat.z
        );

        // Calculate direction from target position to lookat position
        const direction = new THREE.Vector3()
          .subVectors(lookAtPos, targetPos)
          .normalize();

        // Calculate yaw (horizontal rotation) - negate because camera faces -Z
        const yaw = Math.atan2(-direction.x, -direction.z);

        // Calculate pitch (vertical rotation) - negate to match camera convention
        const horizontalDistance = Math.sqrt(
          direction.x * direction.x + direction.z * direction.z
        );
        const pitch = -Math.atan2(direction.y, horizontalDistance);

        targetRotation = { yaw, pitch };

        this.logger.log(
          `MoveTo with lookat: target rotation yaw=${THREE.MathUtils.radToDeg(
            yaw
          ).toFixed(1)}° pitch=${THREE.MathUtils.radToDeg(pitch).toFixed(1)}°`
        );
      } else if (data.rotation) {
        targetRotation = {
          yaw: data.rotation.yaw,
          pitch: data.rotation.pitch || 0,
        };
      }

      // Parse input control settings (what to disable: movement, rotation, or both)
      const inputControl = data.inputControl || {
        disableMovement: true,
        disableRotation: true,
      };

      // Parse restoreInput setting (whether to restore input on completion)
      const restoreInput =
        data.restoreInput !== undefined ? data.restoreInput : true;

      const onComplete = data.onComplete || null;

      this.moveTo(
        targetPos,
        targetRotation,
        data.duration,
        onComplete,
        inputControl,
        restoreInput
      );
    });

    this.logger.log("Event listeners registered");
  }

  loadFootstepAudio() {
    // Load footstep audio using Howler.js
    this.footstepSound = new Howl({
      src: ["./audio/sfx/gravel-steps.mp3"],
      loop: true,
      volume: 0.7,
      preload: true,
      onload: () => {
        this.logger.log("Footstep audio loaded successfully");
      },
      onloaderror: (id, error) => {
        this.logger.warn("Failed to load footstep audio:", error);
      },
    });

    // Register with SFX manager if available
    if (this.sfxManager) {
      this.sfxManager.registerSound("footsteps", this.footstepSound, 0.2);
    }
  }

  /**
   * Start camera look-at sequence
   * @param {THREE.Vector3} targetPosition - World position to look at
   * @param {number} duration - Time to complete the look-at in seconds
   * @param {Function} onComplete - Optional callback when complete
   * @param {boolean} enableZoom - Whether to enable zoom/DoF effects (default: false)
   * @param {Object} zoomOptions - Zoom/DoF configuration options
   * @param {boolean} disableInput - Whether to disable input during lookAt (default: true)
   * @param {boolean} returnToOriginalView - If true, return to original view before restoring control (default: false)
   * @param {number} returnDuration - Duration of the return animation in seconds (default: same as duration)
   * @param {number} holdDuration - How long to hold at target before returning or completing (default: 0)
   */
  lookAt(
    targetPosition,
    duration = 1.0,
    onComplete = null,
    enableZoom = false,
    zoomOptions = {},
    disableInput = true,
    returnToOriginalView = false,
    returnDuration = null,
    holdDuration = 0
  ) {
    this.isLookingAt = true;
    this.lookAtTarget = targetPosition.clone();
    this.lookAtDuration = duration;
    this.lookAtProgress = 0;
    this.lookAtOnComplete = onComplete;
    this.lookAtDisabledInput = disableInput; // Store whether we disabled input
    this.lookAtReturnToOriginalView = returnToOriginalView;
    this.lookAtReturnDuration = returnDuration || duration;
    this.lookAtReturning = false;
    this.lookAtHolding = false;
    this.lookAtHoldTimer = 0;

    // Parse zoom options with defaults
    const {
      zoomFactor = 1.5, // 1.5x zoom (FOV reduction)
      minAperture = 0.15, // Minimum aperture (less blur at distance)
      maxAperture = 0.35, // Maximum aperture (more blur close-up)
      transitionStart = 0.8, // When to start DoF transition (0-1)
      transitionDuration = 2.0, // How long the DoF transition takes
      holdDuration: zoomHoldDuration = 2.0, // How long to hold DoF after look-at completes (or before return if returnToOriginal)
      disableDoF = false, // If true, disable DoF effect (only use zoom)
    } = zoomOptions;

    // Use the passed holdDuration parameter, or fallback to zoom holdDuration for backwards compatibility
    // If returning to original view, hold before returning; otherwise hold is at the end
    if (returnToOriginalView) {
      this.lookAtHoldDuration =
        holdDuration || (enableZoom ? zoomHoldDuration : 0);
    } else {
      this.lookAtHoldDuration = holdDuration || 0;
    }

    // Store zoom config for this look-at
    this.currentZoomConfig = {
      zoomFactor,
      minAperture,
      maxAperture,
      transitionStart,
      transitionDuration,
      holdDuration: zoomHoldDuration,
      disableDoF,
    };

    // Only disable input if requested (e.g., if not being managed by moveTo)
    if (disableInput) {
      this.inputDisabled = true;
      this.inputManager.disable();
    }

    // Store current camera orientation as quaternion
    this.lookAtStartQuat.setFromEuler(
      new THREE.Euler(this.pitch, this.yaw, 0, "YXZ")
    );

    // Calculate target orientation
    const direction = new THREE.Vector3().subVectors(
      targetPosition,
      this.camera.position
    );
    const dirLen = direction.length();
    let targetYaw = this.yaw;
    let targetPitch = this.pitch;
    if (dirLen < 1e-6) {
      this.logger.warn(
        `lookAt: target coincides with camera (len=${dirLen.toFixed(
          6
        )}), keeping current rotation`
      );
    } else {
      direction.divideScalar(dirLen);
      const horiz = Math.sqrt(
        direction.x * direction.x + direction.z * direction.z
      );
      targetYaw = Math.atan2(-direction.x, -direction.z);
      targetPitch = Math.atan2(direction.y, horiz);
      if (!isFinite(targetYaw) || !isFinite(targetPitch)) {
        this.logger.warn(
          `lookAt: computed non-finite rotation (yaw=${targetYaw}, pitch=${targetPitch}), falling back to current`
        );
        targetYaw = this.yaw;
        targetPitch = this.pitch;
      }
    }

    // Create target quaternion from validated euler angles
    this.lookAtEndQuat.setFromEuler(
      new THREE.Euler(targetPitch, targetYaw, 0, "YXZ")
    );

    // Ensure we take the shorter rotation path
    // If dot product is negative, negate the target quaternion
    if (this.lookAtStartQuat.dot(this.lookAtEndQuat) < 0) {
      this.lookAtEndQuat.x *= -1;
      this.lookAtEndQuat.y *= -1;
      this.lookAtEndQuat.z *= -1;
      this.lookAtEndQuat.w *= -1;
    }

    // Calculate DoF values based on distance to target (but don't start transition yet)
    // Skip DoF if disableDoF is set in zoomOptions
    if (
      enableZoom &&
      this.sparkRenderer &&
      this.dofEnabled &&
      !this.currentZoomConfig?.disableDoF
    ) {
      const distance = this.camera.position.distanceTo(targetPosition);

      // Calculate target DoF settings
      const targetFocalDistance = distance;

      // Calculate aperture size for dramatic DoF effect using configured values
      // Larger aperture = more blur, smaller = less blur
      const { minAperture, maxAperture } = this.currentZoomConfig;

      // Scale aperture based on distance (closer = more DoF, further = less)
      // Clamp distance between 2 and 20 meters for sensible scaling
      const normalizedDistance = Math.max(2, Math.min(20, distance));
      const apertureScale = 1 - (normalizedDistance - 2) / 18; // 1.0 at 2m, 0.0 at 20m
      const targetApertureSize =
        minAperture + (maxAperture - minAperture) * apertureScale;

      // Store target values but don't start transition yet
      // Transition will start based on look-at progress
      this.targetFocalDistance = targetFocalDistance;
      this.targetApertureSize = targetApertureSize;
      this.lookAtDofActive = true;
      this.dofHoldTimer = 0; // Reset hold timer
      this.dofTransitionProgress = 0; // Reset progress

      this.logger.log(
        `DoF ready - Distance: ${distance.toFixed(
          2
        )}m, Aperture: ${targetApertureSize.toFixed(
          3
        )} (min: ${minAperture.toFixed(3)}, max: ${maxAperture.toFixed(
          3
        )}) (will transition at ${(
          this.currentZoomConfig.transitionStart * 100
        ).toFixed(0)}% over ${this.currentZoomConfig.transitionDuration}s)`
      );
    }

    // Set up zoom transition (synced with DoF)
    if (enableZoom) {
      this.startFov = this.currentFov; // Capture current FOV as start
      this.targetFov = this.baseFov / this.currentZoomConfig.zoomFactor; // Zoom in by reducing FOV
      this.lookAtZoomActive = true;
      this.zoomTransitioning = true; // Start zoom transition
      this.zoomTransitionProgress = 0; // Reset zoom progress

      this.logger.log(
        `Looking at target over ${duration}s (zoom: ${this.baseFov.toFixed(
          1
        )}° → ${this.targetFov.toFixed(
          1
        )}° [${this.currentZoomConfig.zoomFactor.toFixed(2)}x])`
      );
    } else {
      // If zoom is not enabled, immediately clear any existing zoom/DoF from previous lookat
      // This prevents conflicts when transitioning from a zoom lookat to a non-zoom lookat
      if (this.lookAtZoomActive || this.lookAtDofActive) {
        // Cancel hold timer and immediately start resetting
        this.dofHoldTimer = 0;

        // Reset DoF if it was active
        if (this.lookAtDofActive && this.sparkRenderer) {
          this.targetFocalDistance = this.baseFocalDistance;
          this.targetApertureSize = this.baseApertureSize;
          this.lookAtDofActive = false;
          this.dofTransitioning = true;
          this.dofTransitionProgress = 0;
        }

        // Reset zoom if it was active
        if (this.lookAtZoomActive) {
          this.startFov = this.currentFov; // Capture current FOV as start
          this.targetFov = this.baseFov;
          this.lookAtZoomActive = false;
          this.zoomTransitioning = true;
          this.zoomTransitionProgress = 0;
        }
      }
      this.logger.log(`Looking at target over ${duration}s (no zoom)`);
    }
  }

  /**
   * Cancel the look-at and restore player control
   * @param {boolean} updateYawPitch - If true, update yaw/pitch to match current camera orientation
   */
  /**
   * Complete a dynamic lookat and restore controls
   * @private
   */
  _completeDynamicLookat() {
    // Clear the position function and complete the lookat
    this.lookAtPositionFunction = null;
    this.isLookingAt = false;
    this.lookAtReturning = false;
    this.lookAtHolding = false;

    // Restore input based on restoreInput setting
    const restoreInput = this.lookAtRestoreInput;
    if (restoreInput !== false) {
      // Default is true (restore both), or can be object with movement/rotation flags
      const restoreMovement =
        restoreInput === true ||
        (restoreInput && restoreInput.movement !== false);
      const restoreRotation =
        restoreInput === true ||
        (restoreInput && restoreInput.rotation !== false);

      if (this.lookAtDisabledInput && (restoreMovement || restoreRotation)) {
        this.inputDisabled = false;
        this.inputManager.enable();
      }
    }
    // Reset glance state
    this.glanceState = null;
    this.glanceTimer = 0;
    this.wasIdleAllowed = false;
    this.currentRoll = 0;
    // Update yaw/pitch to match current camera orientation
    const euler = new THREE.Euler().setFromQuaternion(
      this.camera.quaternion,
      "YXZ"
    );
    this.yaw = euler.y;
    this.pitch = euler.x;
    this.targetYaw = this.yaw;
    this.targetPitch = this.pitch;
  }

  cancelLookAt(updateYawPitch = true) {
    if (!this.isLookingAt) return;

    this.isLookingAt = false;

    // Re-enable input manager only if we disabled it
    if (this.lookAtDisabledInput) {
      this.inputDisabled = false;
      this.inputManager.enable();
    }

    // Reset glance state
    this.glanceState = null;
    this.glanceTimer = 0;
    this.wasIdleAllowed = false;
    this.currentRoll = 0; // Reset head tilt

    if (updateYawPitch) {
      // Update yaw and pitch to match current camera orientation
      const euler = new THREE.Euler().setFromQuaternion(
        this.camera.quaternion,
        "YXZ"
      );
      this.yaw = euler.y;
      this.pitch = euler.x;
      this.targetYaw = this.yaw;
      this.targetPitch = this.pitch;
    }

    // Return DoF to base if it was active
    if (this.sparkRenderer && this.lookAtDofActive) {
      this.targetFocalDistance = this.baseFocalDistance;
      this.targetApertureSize = this.baseApertureSize;
      this.lookAtDofActive = false;
      this.dofTransitioning = true;
      this.dofTransitionProgress = 0; // Reset for return transition
      this.logger.log(`Returning DoF to base (cancelled)`);
    }

    // Return zoom to base if it was active
    if (this.lookAtZoomActive) {
      this.startFov = this.currentFov; // Capture current FOV as start
      this.targetFov = this.baseFov;
      this.lookAtZoomActive = false;
      this.zoomTransitioning = true;
      this.zoomTransitionProgress = 0; // Reset for return transition
      this.logger.log(`Returning zoom to base (cancelled)`);
    }

    this.logger.log("Look-at cancelled, control restored");
  }

  /**
   * Start character move-to sequence
   * Smoothly moves character to target position and rotation
   * @param {THREE.Vector3} targetPosition - World position to move to
   * @param {Object} targetRotation - Target rotation {yaw: radians, pitch: radians} (optional)
   * @param {number} duration - Time to complete the move in seconds
   * @param {Function} onComplete - Optional callback when complete
   * @param {Object} inputControl - Control what input to disable {disableMovement: true/false, disableRotation: true/false}
   * @param {boolean} restoreInput - Whether to restore input controls on completion (default: true)
   */
  moveTo(
    targetPosition,
    targetRotation = null,
    duration = 2.0,
    onComplete = null,
    inputControl = { disableMovement: true, disableRotation: true },
    restoreInput = true
  ) {
    this.isMovingTo = true;
    this.moveToDuration = duration;
    this.moveToProgress = 0;
    this.moveToOnComplete = onComplete;
    this.moveToInputControl = inputControl; // Store for restoration later
    this.moveToRestoreInput = restoreInput; // Store whether to restore input on completion

    // Disable input based on inputControl settings
    // Only set inputDisabled if BOTH movement and rotation are disabled
    if (inputControl.disableMovement && inputControl.disableRotation) {
      // Disable everything
      this.inputDisabled = true;
      this.inputManager.disable();
    } else if (inputControl.disableMovement) {
      // Disable only movement, keep rotation
      // Don't set inputDisabled - let inputManager handle selective blocking
      this.inputManager.disableMovement();
    } else if (inputControl.disableRotation) {
      // Disable only rotation, keep movement
      // Don't set inputDisabled - let inputManager handle selective blocking
      this.inputManager.disableRotation();
    }
    // If neither is disabled, don't change anything

    // Store current position (get from physics body)
    const currentPos = this.character.translation();
    this.moveToStartPos.set(currentPos.x, currentPos.y, currentPos.z);
    this.moveToTargetPos.copy(targetPosition);

    // Store current rotation
    this.moveToStartYaw = isFinite(this.yaw) ? this.yaw : 0;
    this.moveToStartPitch = isFinite(this.pitch) ? this.pitch : 0;
    if (!isFinite(this.yaw) || !isFinite(this.pitch)) {
      this.logger.warn(
        `MoveTo: Current yaw/pitch invalid (yaw=${this.yaw}, pitch=${this.pitch}), resetting to (0,0)`
      );
      this.yaw = 0;
      this.pitch = 0;
      this.targetYaw = 0;
      this.targetPitch = 0;
    }

    // Set target rotation (if provided, otherwise keep current)
    if (targetRotation) {
      this.moveToTargetYaw =
        targetRotation.yaw !== undefined && isFinite(targetRotation.yaw)
          ? targetRotation.yaw
          : this.moveToStartYaw;
      this.moveToTargetPitch =
        targetRotation.pitch !== undefined && isFinite(targetRotation.pitch)
          ? targetRotation.pitch
          : this.moveToStartPitch;
    } else {
      this.moveToTargetYaw = this.moveToStartYaw;
      this.moveToTargetPitch = this.moveToStartPitch;
    }
    if (!isFinite(this.moveToTargetYaw) || !isFinite(this.moveToTargetPitch)) {
      this.logger.warn(
        `MoveTo: Target yaw/pitch invalid (yaw=${this.moveToTargetYaw}, pitch=${this.moveToTargetPitch}), using (0,0)`
      );
      this.moveToTargetYaw = 0;
      this.moveToTargetPitch = 0;
    }

    // Normalize the yaw difference to ensure shortest rotation path
    // This prevents "unwinding" when yaw has accumulated over time
    let yawDiff = this.moveToTargetYaw - this.moveToStartYaw;
    yawDiff = Math.atan2(Math.sin(yawDiff), Math.cos(yawDiff)); // Normalize to [-π, π]
    this.moveToTargetYaw = this.moveToStartYaw + yawDiff; // Adjust target to shortest path

    this.logger.log(
      `Moving to position (${targetPosition.x.toFixed(
        2
      )}, ${targetPosition.y.toFixed(2)}, ${targetPosition.z.toFixed(
        2
      )}) over ${duration}s`
    );
  }

  /**
   * Reset character to upright position and orientation
   * Useful for resetting after animations that leave character in unusual positions
   */
  resetToUpright() {
    const currentPos = this.character.translation();

    const bodyY = 2.05;

    this.character.setTranslation(
      {
        x: currentPos.x,
        y: bodyY,
        z: currentPos.z,
      },
      true
    );

    const bodyQuat = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(0, 0, 0, "YXZ")
    );
    this.character.setRotation(
      { x: bodyQuat.x, y: bodyQuat.y, z: bodyQuat.z, w: bodyQuat.w },
      true
    );

    this.character.setLinvel({ x: 0, y: 0, z: 0 }, true);
    this.character.setAngvel({ x: 0, y: 0, z: 0 }, true);

    this.camera.position.set(
      currentPos.x,
      bodyY + this.cameraHeight,
      currentPos.z
    );

    this.yaw = 0;
    this.pitch = 0;
    this.targetYaw = 0;
    this.targetPitch = 0;
    this.bodyYaw = 0;
    this.currentRoll = 0; // Reset head tilt/roll from animations

    const euler = new THREE.Euler(0, 0, 0, "YXZ");
    this.camera.quaternion.setFromEuler(euler);

    this.cameraSyncDisabled = false;
    this.inputDisabled = false;
    if (this.inputManager) {
      this.inputManager.enable();
    }

    this.logger.log(`Character reset to upright at Y=${bodyY.toFixed(2)}`);
  }

  /**
   * Cancel the move-to and restore player control
   */
  cancelMoveTo() {
    if (!this.isMovingTo) return;

    this.isMovingTo = false;

    // Restore input based on what was disabled
    if (this.moveToInputControl) {
      if (
        this.moveToInputControl.disableMovement &&
        this.moveToInputControl.disableRotation
      ) {
        // Both were disabled, enable everything
        this.inputDisabled = false;
        this.inputManager.enable();
        this.logger.log("Move-to cancelled, restored full input");
      } else if (this.moveToInputControl.disableMovement) {
        // Only movement was disabled, restore it
        this.inputManager.enableMovement();
        this.logger.log("Move-to cancelled, restored movement input");
      } else if (this.moveToInputControl.disableRotation) {
        // Only rotation was disabled, restore it
        this.inputManager.enableRotation();
        this.logger.log("Move-to cancelled, restored rotation input");
      }
      this.moveToInputControl = null;
    } else {
      // Fallback: enable everything if inputControl wasn't stored
      this.inputDisabled = false;
      this.inputManager.enable();
      this.logger.log("Move-to cancelled, control restored (fallback)");
    }
  }

  getForwardRightVectors() {
    // Reuse vectors instead of creating new ones
    this._forward
      .set(0, 0, -1)
      .applyAxisAngle(this._yAxis, this.yaw)
      .setY(0)
      .normalize();
    this._right.crossVectors(this._forward, this._yAxis);
    return { forward: this._forward, right: this._right };
  }

  /**
   * Disable all character input (movement + rotation)
   * Convenience method for other systems (dialogs, cutscenes, etc.)
   */
  disableInput() {
    this.inputDisabled = true;
    this.inputManager.disable();
  }

  /**
   * Enable all character input (movement + rotation)
   * Convenience method for other systems
   * @param {boolean} syncFromCamera - If true, sync yaw/pitch from camera quaternion (default: true)
   */
  enableInput(syncFromCamera = true) {
    this.inputDisabled = false;
    this.cameraSyncDisabled = false; // Re-enable camera sync when input is enabled

    // Sync yaw/pitch from camera quaternion to prevent snapping when taking control
    if (syncFromCamera) {
      const euler = new THREE.Euler().setFromQuaternion(
        this.camera.quaternion,
        "YXZ"
      );
      this.yaw = euler.y;
      this.pitch = euler.x;
      this.targetYaw = this.yaw;
      this.targetPitch = this.pitch;
      this.bodyYaw = this.yaw; // Sync body yaw too
    }

    this.inputManager.enable();
  }

  /**
   * Disable camera syncing to physics body (leaves camera frozen in place)
   * Use this for animations that should leave the camera where it lands
   */
  disableCameraSync() {
    this.cameraSyncDisabled = true;
    // Store current camera position as base for idle breathing
    this.frozenCameraBasePosition.copy(this.camera.position);
  }

  /**
   * Re-enable camera syncing to physics body
   */
  enableCameraSync() {
    this.cameraSyncDisabled = false;
  }

  /**
   * Disable only movement input (rotation still works)
   */
  disableMovement() {
    this.inputManager.disableMovement();
  }

  /**
   * Enable movement input
   */
  enableMovement() {
    this.inputManager.enableMovement();
  }

  /**
   * Disable only rotation input (movement still works)
   */
  disableRotation() {
    this.inputManager.disableRotation();
  }

  /**
   * Enable rotation input
   */
  enableRotation() {
    this.inputManager.enableRotation();
  }

  /**
   * Increase movement speed (for gizmo mode)
   * @param {number} increment - Amount to increase by (default: uses speedIncrement)
   */
  increaseSpeed(increment = null) {
    const inc = increment !== null ? increment : this.speedIncrement;
    this.baseSpeed = Math.min(this.maxSpeed, this.baseSpeed + inc);
    this.logger.log(`Movement speed increased to ${this.baseSpeed.toFixed(2)}`);
  }

  /**
   * Decrease movement speed (for gizmo mode)
   * @param {number} increment - Amount to decrease by (default: uses speedIncrement)
   */
  decreaseSpeed(increment = null) {
    const inc = increment !== null ? increment : this.speedIncrement;
    this.baseSpeed = Math.max(this.minSpeed, this.baseSpeed - inc);
    this.logger.log(`Movement speed decreased to ${this.baseSpeed.toFixed(2)}`);
  }

  /**
   * Get current movement speed
   * @returns {number} Current base speed
   */
  getSpeed() {
    return this.baseSpeed;
  }

  /**
   * Adjust camera roll (rotation around forward axis) for gizmo mode
   * @param {number} delta - Amount to adjust roll by (radians)
   */
  adjustRoll(delta) {
    this.currentRoll += delta;
    // Clamp roll to reasonable range (-90 to 90 degrees)
    const maxRoll = Math.PI / 2; // 90 degrees
    this.currentRoll = Math.max(-maxRoll, Math.min(maxRoll, this.currentRoll));
    // Prevent auto-lerp to zero when manually adjusting
    this.isLerpingRollToZero = false;
    // Also cancel any active glance animations that might interfere
    if (this.glanceState !== null) {
      this.glanceState = null;
      this.glanceTimer = 5.0 + Math.random() * 3.0;
    }
  }

  /**
   * Get current camera roll
   * @returns {number} Current roll in radians
   */
  getRoll() {
    return this.currentRoll;
  }

  /**
   * Enable flight mode (for gizmo debug mode)
   * Disables gravity and allows free movement in all directions
   */
  enableFlightMode() {
    if (this.flightMode) return;

    this.flightMode = true;
    this.logger.log("Flight mode enabled (Q=down, E=up)");

    // Set up keyboard listener for Q/E vertical movement
    this.onFlightKeyDown = this.handleFlightKeyDown.bind(this);
    this.onFlightKeyUp = this.handleFlightKeyUp.bind(this);
    window.addEventListener("keydown", this.onFlightKeyDown);
    window.addEventListener("keyup", this.onFlightKeyUp);
  }

  /**
   * Disable flight mode and restore normal physics
   */
  disableFlightMode() {
    if (!this.flightMode) return;

    this.flightMode = false;
    this.verticalInput = 0;
    this.logger.log("Flight mode disabled");

    // Remove keyboard listeners
    if (this.onFlightKeyDown) {
      window.removeEventListener("keydown", this.onFlightKeyDown);
    }
    if (this.onFlightKeyUp) {
      window.removeEventListener("keyup", this.onFlightKeyUp);
    }
  }

  /**
   * Handle key down for flight controls
   */
  handleFlightKeyDown(event) {
    // Ignore if typing in an input
    if (
      document.activeElement.tagName === "INPUT" ||
      document.activeElement.tagName === "TEXTAREA"
    ) {
      return;
    }

    switch (event.key.toLowerCase()) {
      case "q":
        this.verticalInput = -1; // Down
        break;
      case "e":
        this.verticalInput = 1; // Up
        break;
    }
  }

  /**
   * Handle key up for flight controls
   */
  handleFlightKeyUp(event) {
    switch (event.key.toLowerCase()) {
      case "q":
      case "e":
        this.verticalInput = 0;
        break;
    }
  }

  /**
   * Disable character controller physics collisions
   * Useful when you want the character to not be pushed by physics objects
   */
  disablePhysicsCollisions() {
    if (!this.character) return;

    // Get all colliders attached to the character controller
    const numColliders = this.character.numColliders();
    for (let i = 0; i < numColliders; i++) {
      const collider = this.character.collider(i);
      if (collider) {
        // Set collision groups to only collide with environment (group 3)
        // This prevents phone cord (group 2) from pushing the character
        collider.setCollisionGroups(0x00040001); // Group 1, only collides with group 3 (environment)
      }
    }
    this.logger.log("Physics collisions limited to environment only");
  }

  /**
   * Enable character controller physics collisions (restore default)
   * Restores normal collision behavior with all physics objects
   */
  enablePhysicsCollisions() {
    if (!this.character) return;

    // Get all colliders attached to the character controller
    const numColliders = this.character.numColliders();
    for (let i = 0; i < numColliders; i++) {
      const collider = this.character.collider(i);
      if (collider) {
        // Restore default collision groups (collides with everything)
        collider.setCollisionGroups(0xffffffff); // Collide with all groups
      }
    }
    this.logger.log("Physics collisions restored to default");
  }

  /**
   * Debug helper: Adjust first-person body position
   * Can be called from browser console:
   * window.characterController.adjustBodyPosition(0, -1.4, -0.3)
   * window.characterController.adjustBodyRotation(0, Math.PI, 0)
   * window.characterController.adjustBodyScale(1.0)
   */
  adjustBodyPosition(x, y, z) {
    if (this.bodyModelGroup) {
      this.bodyModelGroup.position.set(x, y, z);
      this.logger.log(
        `Body position (relative to character container) set to: ${x}, ${y}, ${z}`
      );
    } else {
      this.logger.warn("Body model not loaded yet");
    }
  }

  adjustBodyRotation(x, y, z) {
    if (this.bodyModelGroup) {
      this.bodyModelGroup.rotation.set(x, y, z);
      this.logger.log(`Body rotation set to: ${x}, ${y}, ${z}`);
    } else {
      this.logger.warn("Body model not loaded yet");
    }
  }

  adjustBodyScale(scale) {
    if (this.bodyModelGroup) {
      this.bodyModelGroup.scale.setScalar(scale);
      this.logger.log(`Body scale set to: ${scale}`);
    } else {
      this.logger.warn("Body model not loaded yet");
    }
  }

  /**
   * Toggle body model visibility (for debugging)
   * window.characterController.toggleBodyVisibility()
   */
  toggleBodyVisibility() {
    if (this.bodyModelGroup) {
      this.bodyModelGroup.visible = !this.bodyModelGroup.visible;
      this.logger.log(
        `Body model visibility: ${this.bodyModelGroup.visible ? "ON" : "OFF"}`
      );
    } else {
      this.logger.warn("Body model not loaded yet");
    }
  }

  /**
   * Check animation state (for debugging)
   * window.characterController.checkAnimationState()
   */
  checkAnimationState() {
    this.logger.log("=== Body Animation State ===");
    this.logger.log("Body model group:", this.bodyModelGroup ? "YES" : "NO");
    this.logger.log("Body model:", this.bodyModel ? "YES" : "NO");
    this.logger.log("Animation mixer:", this.bodyAnimationMixer ? "YES" : "NO");
    this.logger.log("Walk animation:", this.walkAnimation ? "YES" : "NO");

    if (this.walkAnimation) {
      this.logger.log("  - Is running:", this.walkAnimation.isRunning());
      this.logger.log("  - Is paused:", this.walkAnimation.paused);
      this.logger.log("  - Time scale:", this.walkAnimation.timeScale);
      this.logger.log("  - Time:", this.walkAnimation.time);
    }

    if (this.sceneManager) {
      const mixer = this.sceneManager.animationMixers.get("firstPersonBody");
      this.logger.log(
        "SceneManager has mixer for firstPersonBody:",
        mixer ? "YES" : "NO"
      );

      const actions = Array.from(
        this.sceneManager.animationActions.entries()
      ).filter(([id]) => id.startsWith("firstPersonBody"));
      this.logger.log("SceneManager actions:", actions.length);
      actions.forEach(([id, action]) => {
        this.logger.log(
          `  - ${id}: running=${action.isRunning()}, paused=${action.paused}`
        );
      });
    }
  }

  /**
   * Enable wireframe mode on body model (for debugging)
   * window.characterController.setBodyWireframe(true)
   */
  setBodyWireframe(enabled) {
    if (this.bodyModel) {
      this.bodyModel.traverse((child) => {
        if (child.isMesh && child.material) {
          child.material.wireframe = enabled;
        }
      });
      this.logger.log(`Body wireframe: ${enabled ? "ON" : "OFF"}`);
    } else {
      this.logger.warn("Body model not loaded yet");
    }
  }

  /**
   * Log body model 3D bounds (for debugging)
   * window.characterController.logBodyBounds()
   */
  logBodyBounds() {
    if (!this.bodyModel) {
      this.logger.warn("Body model not loaded yet");
      return;
    }

    let minX = Infinity,
      maxX = -Infinity;
    let minY = Infinity,
      maxY = -Infinity;
    let minZ = Infinity,
      maxZ = -Infinity;
    let meshCount = 0;

    this.bodyModel.traverse((child) => {
      if (child.isMesh && child.geometry) {
        meshCount++;
        child.geometry.computeBoundingBox();
        const box = child.geometry.boundingBox;
        if (box) {
          minX = Math.min(minX, box.min.x);
          maxX = Math.max(maxX, box.max.x);
          minY = Math.min(minY, box.min.y);
          maxY = Math.max(maxY, box.max.y);
          minZ = Math.min(minZ, box.min.z);
          maxZ = Math.max(maxZ, box.max.z);
        }
      }
    });

    const width = maxX - minX;
    const height = maxY - minY;
    const depth = maxZ - minZ;

    this.logger.log("=== Body Model 3D Bounds ===");
    this.logger.log(`Meshes analyzed: ${meshCount}`);
    this.logger.log(
      `X: ${minX.toFixed(2)} to ${maxX.toFixed(2)} (width: ${width.toFixed(2)})`
    );
    this.logger.log(
      `Y: ${minY.toFixed(2)} to ${maxY.toFixed(2)} (height: ${height.toFixed(
        2
      )})`
    );
    this.logger.log(
      `Z: ${minZ.toFixed(2)} to ${maxZ.toFixed(2)} (depth: ${depth.toFixed(2)})`
    );
    this.logger.log(
      `\nBody offset in character container: (${this.bodyModelGroup.position.x.toFixed(
        2
      )}, ${this.bodyModelGroup.position.y.toFixed(
        2
      )}, ${this.bodyModelGroup.position.z.toFixed(2)})`
    );

    // Calculate where the bottom and top of the body are relative to the character
    const bottomY = minY + this.bodyModelGroup.position.y;
    const topY = maxY + this.bodyModelGroup.position.y;
    this.logger.log(`\nRelative to character container:`);
    this.logger.log(`  Bottom: y = ${bottomY.toFixed(2)}`);
    this.logger.log(`  Top: y = ${topY.toFixed(2)}`);
    this.logger.log(`  Height: ${height.toFixed(2)}`);
  }

  /**
   * Get position relative to player with offset applied in player's local space
   * @param {THREE.Vector3|Object} offset - Offset {x: left/right, y: up/down, z: forward/back} relative to player
   * @returns {Object} World position {x, y, z}
   */
  getPosition(offset = { x: 0, y: 0, z: 0 }) {
    const playerPos = this.character.translation();

    // Create offset vector in player's local space
    // x = left(-)/right(+), y = down(-)/up(+), z = back(-)/forward(+)
    const localOffset = new THREE.Vector3(offset.x, offset.y, offset.z);

    // Rotate offset by player's yaw to get world space offset
    localOffset.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw);

    return {
      x: playerPos.x + localOffset.x,
      y: playerPos.y + localOffset.y,
      z: playerPos.z + localOffset.z,
    };
  }

  /**
   * Debug helper: Log current player position and rotation
   * Can be called from browser console: window.characterController.logPosition()
   */
  logPosition() {
    const physicsPos = this.character.translation();
    const yawDeg = THREE.MathUtils.radToDeg(this.yaw);
    const pitchDeg = THREE.MathUtils.radToDeg(this.pitch);
    const bodyYawDeg = THREE.MathUtils.radToDeg(this.bodyYaw);

    this.logger.log("=== Physics Body (Capsule Collider Center) ===");
    this.logger.log(
      `Position: { x: ${physicsPos.x.toFixed(2)}, y: ${physicsPos.y.toFixed(
        2
      )}, z: ${physicsPos.z.toFixed(2)} }`
    );
    this.logger.log(
      `Rotation: { yaw: ${yawDeg.toFixed(2)}°, pitch: ${pitchDeg.toFixed(2)}° }`
    );

    // Character container (holds the body, rotates separately from camera)
    if (this.characterContainer) {
      this.logger.log("\n=== Character Container ===");
      const cp = this.characterContainer.position;
      const cr = this.characterContainer.rotation;
      this.logger.log(
        `Position: { x: ${cp.x.toFixed(2)}, y: ${cp.y.toFixed(
          2
        )}, z: ${cp.z.toFixed(2)} }`
      );
      this.logger.log(
        `Body Yaw: ${bodyYawDeg.toFixed(2)}° (rotation.y: ${cr.y.toFixed(
          2
        )} rad)`
      );
    }

    // Camera position
    this.logger.log("\n=== Camera ===");
    const camPos = this.camera.position;
    this.logger.log(
      `Position: { x: ${camPos.x.toFixed(2)}, y: ${camPos.y.toFixed(
        2
      )}, z: ${camPos.z.toFixed(2)} }`
    );
    this.logger.log(
      `Rotation: { yaw: ${yawDeg.toFixed(2)}°, pitch: ${pitchDeg.toFixed(2)}° }`
    );
    this.logger.log(
      `Height offset from physics center: ${this.cameraHeight.toFixed(2)}`
    );

    // Also log body model position if loaded
    if (this.bodyModelGroup) {
      this.logger.log(
        "\n=== First-Person Body Model (local offset in container) ==="
      );
      const bp = this.bodyModelGroup.position;
      const br = this.bodyModelGroup.rotation;
      this.logger.log(
        `Local Position: { x: ${bp.x.toFixed(2)}, y: ${bp.y.toFixed(
          2
        )}, z: ${bp.z.toFixed(2)} }`
      );
      this.logger.log(
        `Local Rotation: { x: ${br.x.toFixed(2)}, y: ${br.y.toFixed(
          2
        )}, z: ${br.z.toFixed(2)} }`
      );
    }

    return {
      position: { x: physicsPos.x, y: physicsPos.y, z: physicsPos.z },
      rotation: { yaw: this.yaw, pitch: this.pitch },
    };
  }

  /**
   * Check if a given position is visible in the camera frustum
   * @param {THREE.Vector3} position - World position to check
   * @returns {boolean} True if position is visible in camera frustum
   */
  isPositionInFrustum(position) {
    // Reuse cached frustum and matrix to avoid allocations
    const projectionMatrix = this.camera.projectionMatrix;
    const viewMatrix = this.camera.matrixWorldInverse;

    // Combine matrices to get the frustum (reuse cached matrix)
    this._frustumCheckMatrix.multiplyMatrices(projectionMatrix, viewMatrix);
    this._frustumCheckFrustum.setFromProjectionMatrix(this._frustumCheckMatrix);

    // Check if position is in frustum
    return this._frustumCheckFrustum.containsPoint(position);
  }

  /**
   * Check if a world position is within 10% margin of screen edges
   * @param {THREE.Vector3} worldPosition - World position to check
   * @returns {boolean} True if position is within 10% margin of screen edges
   */
  isPositionNearScreenEdge(worldPosition) {
    // Project world position to normalized device coordinates (-1 to 1)
    // Reuse cached vector to avoid allocation
    this._frustumCheckVector.copy(worldPosition);
    this._frustumCheckVector.project(this.camera);

    // Convert to screen coordinates (0 to width/height)
    const width = this.renderer.domElement.width;
    const height = this.renderer.domElement.height;
    const x = ((this._frustumCheckVector.x + 1) / 2) * width;
    const y = ((-this._frustumCheckVector.y + 1) / 2) * height;

    // Check if within 10% margin of any edge
    const margin = 0.1; // 10%
    const marginX = width * margin;
    const marginY = height * margin;

    return (
      x < marginX || x > width - marginX || y < marginY || y > height - marginY
    );
  }

  /**
   * Check if the current goal rune is within camera frustum and near screen edge
   * Only runs during CURSOR/CURSOR_FINAL gameplay states
   * Requires viewmaster to be equipped and animation finished (insanity progress bar started)
   * @param {number} dt - Delta time
   */
  checkPointFrustum(dt) {
    // Only check during CURSOR gameplay
    if (!this.gameManager) return;

    const gameState = this.gameManager.getState();
    const currentState = gameState?.currentState;
    const isCursorGameplay =
      currentState === GAME_STATES.CURSOR ||
      currentState === GAME_STATES.CURSOR_FINAL;

    if (!isCursorGameplay) {
      // Reset timers when not in CURSOR gameplay
      this.pointFrustumCheckTimer = 0;
      this.runeSightings = 0;
      this.lastSightedRuneLabel = null;
      this.lastViewmasterEquipped = false;
      this.lastViewmasterTransitioning = false;
      this.lastFractalIntensity = 0;
      return;
    }

    // Check if viewmaster is equipped
    const isViewmasterEquipped = gameState?.isViewmasterEquipped || false;

    // Update tracking state even when not equipped (to detect transitions)
    const wasJustEquipped =
      !this.lastViewmasterEquipped && isViewmasterEquipped;
    this.lastViewmasterEquipped = isViewmasterEquipped;

    if (!isViewmasterEquipped) {
      this.lastViewmasterTransitioning = false;
      return;
    }

    // Check if viewmaster animation is finished and insanity progress bar has started
    const viewmasterController = window.viewmasterController;
    if (!viewmasterController) {
      return;
    }

    // Animation must be finished (not transitioning)
    if (viewmasterController.isTransitioning) {
      this.lastViewmasterTransitioning = true;
      return;
    }

    // Insanity progress bar must have started (fractal intensity > minimum)
    const FRACTAL_MIN_INTENSITY = 0.02;
    const fractalIntensity = viewmasterController.currentFractalIntensity || 0;

    // Check if animation just finished (transition from transitioning to not transitioning)
    // This means all conditions are now met for the first time
    const animationJustFinished =
      this.lastViewmasterTransitioning && !viewmasterController.isTransitioning;

    // Check if fractal intensity just started (was below threshold, now above)
    const fractalJustStarted =
      (this.lastFractalIntensity || 0) < FRACTAL_MIN_INTENSITY &&
      fractalIntensity >= FRACTAL_MIN_INTENSITY;

    // Also check if viewmaster was just equipped (in case it was equipped while already in correct state)
    const shouldCheckImmediately =
      wasJustEquipped || animationJustFinished || fractalJustStarted;

    // Update fractal intensity tracking
    this.lastFractalIntensity = fractalIntensity;

    if (fractalIntensity < FRACTAL_MIN_INTENSITY) {
      return;
    }

    // Update tracking state
    this.lastViewmasterTransitioning = viewmasterController.isTransitioning;

    // If we should check immediately, force a check this frame
    if (shouldCheckImmediately) {
      this.pointFrustumCheckTimer = this.pointFrustumCheckInterval; // Force check now
      this.logger.warn(
        `Immediate frustum check triggered: wasJustEquipped=${wasJustEquipped}, animationJustFinished=${animationJustFinished}, fractalJustStarted=${fractalJustStarted}, fractalIntensity=${fractalIntensity.toFixed(
          3
        )}`
      );
    }

    // Don't trigger if sickness/woozy animation has started
    // Check if glitch sequence is active (max intensity reached)
    if (viewmasterController.glitchSequenceActive) {
      return;
    }

    // Check if woozy animation is playing (sickness effect)
    const animationManager = window.cameraAnimationManager;
    if (
      animationManager &&
      animationManager.isPlaying &&
      animationManager.currentAnimationData?.id === "woozy"
    ) {
      return;
    }

    // Update check timer (but skip if we need to check immediately)
    if (!shouldCheckImmediately) {
      this.pointFrustumCheckTimer += dt;
    }

    // Check immediately if viewmaster was just equipped or animation just finished
    // Otherwise, only check every 1/3 second for performance
    const shouldCheckNow =
      shouldCheckImmediately ||
      this.pointFrustumCheckTimer >= this.pointFrustumCheckInterval;

    if (shouldCheckNow) {
      this.pointFrustumCheckTimer = 0;

      // Get the current goal rune from drawing manager
      const drawingManager = window.drawingManager;
      if (!drawingManager || !drawingManager.currentGoalRune) {
        return;
      }

      const rune = drawingManager.currentGoalRune;

      // Get the world position of the rune mesh
      if (!rune || !rune.mesh) {
        return;
      }

      // Reuse cached vector to avoid allocation
      rune.mesh.getWorldPosition(this._frustumCheckVector);

      // Check if point is in frustum and NOT near screen edge (in interior/center)
      const isInFrustum = this.isPositionInFrustum(this._frustumCheckVector);
      const isNearEdge = this.isPositionNearScreenEdge(
        this._frustumCheckVector
      );

      // Debug logging
      if (this.pointFrustumCheckTimer === 0) {
        // Only log once per check interval to avoid spam
        this.logger.warn(
          `Frustum check: inFrustum=${isInFrustum}, nearEdge=${isNearEdge}, rune=${drawingManager.targetLabel}`
        );
      }

      if (isInFrustum && !isNearEdge) {
        // Don't trigger if a lookat is already in progress
        if (this.isLookingAt) {
          return;
        }

        // Check if runeLookat animation is already playing (prevent retriggering)
        const animationManager = window.cameraAnimationManager;
        if (
          animationManager &&
          animationManager.characterController?.isLookingAt
        ) {
          // Check if sawRune is already true (animation might be playing)
          const gameState = this.gameManager?.getState();
          if (gameState?.sawRune) {
            this.logger.warn(
              `Skipping - runeLookat already triggered (sawRune: true, isLookingAt: true)`
            );
            return;
          }
        }

        // Check if we've already triggered lookat for this rune
        // Allow multiple sightings (one per rune)
        const currentRuneLabel = drawingManager.targetLabel;
        const lastSightedRune = this.lastSightedRuneLabel;

        // Only skip if this exact rune was already sighted
        if (currentRuneLabel && currentRuneLabel === lastSightedRune) {
          this.logger.warn(
            `Skipping - already sighted rune: ${currentRuneLabel}`
          );
          return; // Already sighted this rune
        }

        // Trigger camera lookat animation (will override player input)
        this.logger.warn(
          `Rune sighted: ${currentRuneLabel} (sighting #${
            this.runeSightings + 1
          }) at position (${this._frustumCheckVector.x.toFixed(
            2
          )}, ${this._frustumCheckVector.y.toFixed(
            2
          )}, ${this._frustumCheckVector.z.toFixed(2)})`
        );
        this.triggerRuneLookat(
          this._frustumCheckVector.clone(),
          currentRuneLabel
        );
        this.lastSightedRuneLabel = currentRuneLabel; // Track which rune we sighted
      }
    }
  }

  /**
   * Trigger camera lookat animation to rune position
   * Sets game state - animation will pick it up automatically
   * @param {THREE.Vector3} runePosition - World position of the rune
   * @param {string} runeLabel - Label of the rune being sighted
   */
  triggerRuneLookat(runePosition, runeLabel) {
    if (!this.gameManager) return;

    // Increment rune sightings counter
    this.runeSightings++;

    // Set gameState flag - animation will listen for this state change
    this.gameManager.setState({
      sawRune: true,
      runeSightings: this.runeSightings,
    });

    this.logger.warn(
      `Rune sighted #${this.runeSightings} for rune: ${
        runeLabel || "unknown"
      } - setting sawRune: true (animation will trigger automatically)`
    );
  }

  /**
   * Set DoF enabled state (called by options menu)
   * @param {boolean} enabled - Whether DoF is enabled
   */
  setDofEnabled(enabled) {
    this.dofEnabled = enabled;
    this.logger.log(`DoF ${enabled ? "enabled" : "disabled"}`);
  }

  /**
   * Update depth of field parameters
   * @param {number} dt - Delta time
   */
  updateDepthOfField(dt) {
    // Handle DoF/zoom hold timer (runs even if DOF is disabled, for zoom-only lookats)
    if (this.dofHoldTimer > 0) {
      this.dofHoldTimer -= dt;
      if (this.dofHoldTimer <= 0) {
        // Hold period over, start transitioning back to base
        this.dofHoldTimer = 0;

        const wasDofActive = this.lookAtDofActive;
        const wasZoomActive = this.lookAtZoomActive;

        // Return DoF to base if it was active
        if (this.lookAtDofActive) {
          this.targetFocalDistance = this.baseFocalDistance;
          this.targetApertureSize = this.baseApertureSize;
          this.lookAtDofActive = false;
          this.dofTransitioning = true;
          this.dofTransitionProgress = 0; // Reset for return transition
        }

        // Return zoom to base if it was active
        if (this.lookAtZoomActive) {
          this.startFov = this.currentFov; // Capture current FOV as start
          this.targetFov = this.baseFov;
          this.lookAtZoomActive = false;
          this.zoomTransitioning = true;
          this.zoomTransitionProgress = 0; // Reset for return transition
        }
      }
    }

    if (!this.sparkRenderer || !this.dofEnabled) return;

    if (!this.dofTransitioning) return;

    // Update transition progress based on configured duration
    // Use returnTransitionDuration during return-to-original if set
    const transitionDuration =
      this.returnTransitionDuration ||
      this.currentZoomConfig?.transitionDuration ||
      this.dofTransitionDuration;
    this.dofTransitionProgress += dt / transitionDuration;
    const t = Math.min(1.0, this.dofTransitionProgress);

    // Use ease-out for smooth transition
    const eased = 1 - Math.pow(1 - t, 3);

    // Calculate start values (where we're transitioning from)
    const startFocalDistance = this.lookAtDofActive
      ? this.baseFocalDistance
      : this.currentFocalDistance;
    const startApertureSize = this.lookAtDofActive
      ? this.baseApertureSize
      : this.currentApertureSize;

    // Interpolate values
    this.currentFocalDistance =
      startFocalDistance +
      (this.targetFocalDistance - startFocalDistance) * eased;
    this.currentApertureSize =
      startApertureSize + (this.targetApertureSize - startApertureSize) * eased;

    // Update spark renderer
    const apertureAngle =
      2 *
      Math.atan((0.5 * this.currentApertureSize) / this.currentFocalDistance);
    this.sparkRenderer.apertureAngle = apertureAngle;
    this.sparkRenderer.focalDistance = this.currentFocalDistance;

    // Check if transition is complete
    if (t >= 1.0) {
      this.currentFocalDistance = this.targetFocalDistance;
      this.currentApertureSize = this.targetApertureSize;
      this.dofTransitioning = false;
      this.dofTransitionProgress = 0;
    }
  }

  /**
   * Update FOV zoom (synced with DoF transitions)
   * @param {number} dt - Delta time
   */
  updateZoom(dt) {
    if (!this.zoomTransitioning) return;

    // Update zoom transition progress (using configured duration, same as DoF)
    // Use returnTransitionDuration during return-to-original if set
    const transitionDuration =
      this.returnTransitionDuration ||
      this.currentZoomConfig?.transitionDuration ||
      this.dofTransitionDuration;

    if (!transitionDuration || transitionDuration <= 0) {
      this.zoomTransitionProgress += dt / 1.0;
    } else {
      this.zoomTransitionProgress += dt / transitionDuration;
    }
    const t = Math.min(1.0, this.zoomTransitionProgress);

    // Use ease-out for smooth transition (matching DoF)
    const eased = 1 - Math.pow(1 - t, 3);

    // Interpolate FOV from captured start to target
    this.currentFov = this.startFov + (this.targetFov - this.startFov) * eased;

    // Update camera FOV
    this.camera.fov = this.currentFov;
    this.camera.updateProjectionMatrix();

    // Check if transition is complete
    if (t >= 1.0) {
      this.currentFov = this.targetFov;
      this.camera.fov = this.currentFov;
      this.camera.updateProjectionMatrix();
      this.zoomTransitioning = false;
      this.zoomTransitionProgress = 0;

      // If lookat is complete and zoom just finished, restore input if needed
      if (
        !this.isLookingAt &&
        this.lookAtRestoreInput &&
        this.lookAtDisabledInput
      ) {
        const restoreInput = this.lookAtRestoreInput;
        if (restoreInput !== false) {
          const restoreMovement =
            restoreInput === true ||
            (restoreInput && restoreInput.movement !== false);
          const restoreRotation =
            restoreInput === true ||
            (restoreInput && restoreInput.rotation !== false);

          if (restoreMovement || restoreRotation) {
            this.inputDisabled = false;
            this.inputManager.enable();
            this.lookAtDisabledInput = false;
            this.logger.warn("Input restored after zoom transition completed");
          }
        }
      }
    }
  }

  /**
   * Update smoothed insanity intensity (call this from update() with dt)
   * @param {number} dt - Delta time in seconds
   */
  updateInsanityIntensity(dt) {
    // Get target intensity from fractal effect
    let targetIntensity = 0.0;
    if (window.vfxManager?.effects?.splatFractal) {
      const fractalEffect = window.vfxManager.effects.splatFractal;
      const currentIntensity = fractalEffect.currentIntensity || 0;

      // Fractal intensity ranges from 0.02 (FRACTAL_MIN_INTENSITY) to 10.0 (FRACTAL_MAX_INTENSITY)
      // Normalize to 0.0-1.0 range
      const MIN_INTENSITY = 0.02;
      const MAX_INTENSITY = 10.0;
      targetIntensity = Math.max(
        0,
        Math.min(
          1,
          (currentIntensity - MIN_INTENSITY) / (MAX_INTENSITY - MIN_INTENSITY)
        )
      );
    }

    // Smooth the intensity - ramp up quickly, ramp down slowly
    if (targetIntensity > this.insanityIntensitySmoothed) {
      // Ramp up instantly to match target (when viewmaster goes on)
      this.insanityIntensitySmoothed = targetIntensity;
    } else {
      // Ramp down gradually when viewmaster is removed (over ~3 seconds)
      const rampDownAmount = this.insanityRampDownSpeed * dt;
      this.insanityIntensitySmoothed = Math.max(
        0,
        this.insanityIntensitySmoothed - rampDownAmount
      );
    }
  }

  /**
   * Get normalized viewmaster insanity intensity (0.0 to 1.0)
   * Exposed for audio systems and other effects
   * Returns smoothed value that gradually ramps down when viewmaster is removed
   * @returns {number} Normalized intensity from 0.0 (calm) to 1.0 (maximum insanity)
   */
  getViewmasterInsanityIntensity() {
    return this.insanityIntensitySmoothed;
  }

  calculateIdleHeadbob() {
    // Base gentle breathing/idle animation
    const baseVerticalAmp = 0.006; // Increased from 0.002 for more noticeable breathing
    const baseHorizontalAmp = 0.003; // Increased from 0.001 for more noticeable breathing
    const idleFrequency = 0.75; // Normal breathing rate (Hz)

    const verticalBob =
      Math.sin(this.idleHeadbobTime * idleFrequency * Math.PI * 2) *
      baseVerticalAmp;
    const horizontalBob =
      Math.sin(this.idleHeadbobTime * idleFrequency * Math.PI) *
      baseHorizontalAmp;

    return {
      vertical: verticalBob,
      horizontal: horizontalBob,
    };
  }

  /**
   * Start a glance animation (look left/right and slightly up/down with head tilt)
   */
  startGlance() {
    this.glanceState = "glancing";
    this.glanceProgress = 0;
    this.glanceStartYaw = this.yaw;
    this.glanceStartPitch = this.pitch;
    this.glanceStartRoll = this.currentRoll;

    // Random duration for this glance (3-7 seconds)
    this.glanceDuration = 3.0 + Math.random() * 4.0;

    // Random horizontal direction and angle
    const horizontalDir = Math.random() > 0.5 ? 1 : -1;
    const glanceAngle = (Math.random() * 0.3 + 0.2) * horizontalDir; // 0.2 to 0.5 radians (~11 to 29 degrees)
    this.glanceTargetYaw = this.yaw + glanceAngle;

    // Random vertical angle (slight up or down)
    const verticalAngle = Math.random() * 0.15 - 0.075; // -0.075 to 0.075 radians (~-4 to 4 degrees)
    this.glanceTargetPitch = this.pitch + verticalAngle;

    // Subtle head tilt (roll) that follows the horizontal direction
    // Tilt slightly in the direction of the glance for natural head movement
    const rollAngle = (Math.random() * 0.3 + 0.04) * horizontalDir; // 0.04 to 0.12 radians (~2 to 7 degrees)
    this.glanceTargetRoll = rollAngle;

    // Clamp pitch to valid range
    this.glanceTargetPitch = Math.max(
      -Math.PI / 3,
      Math.min(Math.PI / 2 - 0.01, this.glanceTargetPitch)
    );
  }

  /**
   * Update idle glance system
   * @param {number} dt - Delta time
   * @param {boolean} isMoving - Whether the player is moving
   */
  updateIdleGlance(dt, isMoving) {
    // Skip if glance system is disabled or no idle helper
    if (!this.glanceEnabled || !this.idleHelper) return;

    // Use IdleHelper to check if idle behaviors should be allowed
    const shouldAllowIdle = this.idleHelper.shouldAllowIdleBehavior();

    // Detect when idle state becomes allowed (edge trigger)
    if (shouldAllowIdle && !this.wasIdleAllowed) {
      // Just became idle - start glance immediately
      this.glanceTimer = 0;
    }

    // Update previous idle state
    this.wasIdleAllowed = shouldAllowIdle;

    // Reset and stop glance if not allowed or if moving
    if (!shouldAllowIdle || isMoving || this.inputDisabled) {
      this.glanceState = null;
      this.glanceTimer = 0; // Reset timer for next idle period
      // Only reset roll if we're not in gizmo mode (gizmo mode might be manually adjusting roll)
      // Check if we're in gizmo mode by checking if flight mode is enabled
      if (!this.flightMode) {
        this.currentRoll = 0; // Reset head tilt
      }
      return;
    }

    // Update glance animation if glancing
    if (this.glanceState === "glancing") {
      this.glanceProgress += dt / this.glanceDuration;

      if (this.glanceProgress >= 1.0) {
        // Glance complete - start return to start
        this.glanceProgress = 0;
        this.glanceState = "returning";
      } else {
        // Animate glance with ease-in-out
        const t = this.glanceProgress;
        const easedT = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

        // Interpolate yaw, pitch, and roll
        this.targetYaw =
          this.glanceStartYaw +
          (this.glanceTargetYaw - this.glanceStartYaw) * easedT;
        this.targetPitch =
          this.glanceStartPitch +
          (this.glanceTargetPitch - this.glanceStartPitch) * easedT;
        this.currentRoll =
          this.glanceStartRoll +
          (this.glanceTargetRoll - this.glanceStartRoll) * easedT;
      }
    } else if (this.glanceState === "returning") {
      // Return to start position at the same speed
      this.glanceProgress += dt / this.glanceDuration;

      if (this.glanceProgress >= 1.0) {
        // Return complete - reset state
        this.glanceProgress = 1.0;
        this.glanceState = null;
        this.glanceTimer = 5.0 + Math.random() * 3.0; // Next glance in 5-8 seconds

        // Ensure we're back at start
        this.targetYaw = this.glanceStartYaw;
        this.targetPitch = this.glanceStartPitch;
        this.currentRoll = 0; // Return to no tilt
      } else {
        // Animate return with ease-in-out
        const t = this.glanceProgress;
        const easedT = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

        // Interpolate back to start
        this.targetYaw =
          this.glanceTargetYaw +
          (this.glanceStartYaw - this.glanceTargetYaw) * easedT;
        this.targetPitch =
          this.glanceTargetPitch +
          (this.glanceStartPitch - this.glanceTargetPitch) * easedT;
        this.currentRoll =
          this.glanceTargetRoll + (0 - this.glanceTargetRoll) * easedT; // Return to 0 tilt
      }
    } else {
      // Count down to next glance (only when idle is allowed)
      this.glanceTimer -= dt;

      if (this.glanceTimer <= 0) {
        this.startGlance();
      }
    }
  }

  calculateHeadbob(isSprinting) {
    // Different parameters for walking vs sprinting
    const walkFrequency = 2.2; // Steps per second
    const sprintFrequency = 2.6; // Reduced from 3.5 for less rapid bobbing
    const walkVerticalAmp = 0.01; // Vertical bobbing amplitude (reduced from 0.04)
    const sprintVerticalAmp = 0.02; // Reduced from 0.08
    const walkHorizontalAmp = 0.0075; // Horizontal swaying amplitude (reduced from 0.03)
    const sprintHorizontalAmp = 0.03; // Reduced from 0.06

    const frequency = isSprinting ? sprintFrequency : walkFrequency;
    const verticalAmp = isSprinting ? sprintVerticalAmp : walkVerticalAmp;
    const horizontalAmp = isSprinting ? sprintHorizontalAmp : walkHorizontalAmp;

    // Vertical bob: double frequency for realistic step pattern
    const verticalBob =
      Math.sin(this.headbobTime * frequency * Math.PI * 2) * verticalAmp;

    // Horizontal sway: half frequency for subtle side-to-side motion
    const horizontalBob =
      Math.sin(this.headbobTime * frequency * Math.PI) * horizontalAmp;

    // Apply intensity smoothing
    return {
      vertical: verticalBob * this.headbobIntensity,
      horizontal: horizontalBob * this.headbobIntensity,
    };
  }

  update(dt) {
    // Update depth of field (always active during transitions)
    this.updateDepthOfField(dt);

    // Update zoom (synced with DoF, always active during transitions)
    this.updateZoom(dt);

    // Check point frustum visibility (only during CURSOR gameplay)
    this.checkPointFrustum(dt);

    // Handle camera look-at sequence
    if (this.isLookingAt) {
      // Debug: log if we have a position function but aren't tracking
      // Handle holding phase (pause before returning to original)
      if (this.lookAtHolding) {
        this.lookAtHoldTimer += dt;

        if (this.lookAtHoldTimer >= this.lookAtHoldDuration) {
          // Hold complete, start return phase
          this.lookAtHolding = false;
          this.lookAtReturning = true;
          this.lookAtProgress = 0;
          this.logger.log(
            `Hold complete, starting return to original view (${this.lookAtReturnDuration}s)`
          );

          // Start DoF/zoom reset immediately (so they return to normal as camera returns)
          if (this.lookAtDofActive || this.lookAtZoomActive) {
            this.dofHoldTimer = 0;

            const wasDofActive = this.lookAtDofActive;
            const wasZoomActive = this.lookAtZoomActive;

            // Reset DOF if it was active
            if (this.sparkRenderer && this.lookAtDofActive) {
              this.targetFocalDistance = this.baseFocalDistance;
              this.targetApertureSize = this.baseApertureSize;
              this.lookAtDofActive = false;
              this.dofTransitioning = true;
              this.dofTransitionProgress = 0; // Reset for return transition
            }

            // Reset zoom if it was active
            if (this.lookAtZoomActive) {
              this.startFov = this.currentFov; // Capture current FOV for smooth transition
              this.targetFov = this.baseFov;
              this.lookAtZoomActive = false;
              this.zoomTransitioning = true;
              this.zoomTransitionProgress = 0; // Reset zoom progress
            }

            // Override transition duration to match return duration
            this.returnTransitionDuration = this.lookAtReturnDuration;

            this.logger.log(
              `Starting ${wasDofActive ? "DoF" : ""}${
                wasDofActive && wasZoomActive ? "/" : ""
              }${wasZoomActive ? "zoom" : ""} reset during return (${
                this.lookAtReturnDuration
              }s)`
            );
          }

          // Swap start and end quaternions for return journey
          const temp = this.lookAtStartQuat.clone();
          this.lookAtStartQuat.copy(this.lookAtEndQuat);
          this.lookAtEndQuat.copy(temp);
        }
        // Stay at target orientation while holding (no need to update quaternion)
        // But still allow dynamic tracking if we have a position function
        if (!this.lookAtPositionFunction) {
          return; // Skip rest of lookat update during hold (unless dynamic tracking)
        }
      }

      // Update lookat target if we have a dynamic position function (for tracking moving objects)
      // During initial transition, use normal interpolation. After transition completes, use direct tracking.
      if (this.lookAtPositionFunction && !this.lookAtReturning) {
        const pos = this.lookAtPositionFunction(this.gameManager);
        this.lookAtTarget.set(pos.x, pos.y, pos.z);

        // Calculate direction to target
        const direction = new THREE.Vector3().subVectors(
          this.lookAtTarget,
          this.camera.position
        );
        const dirLen = direction.length();

        // If target is too close to camera (e.g., letter reparented to camera), complete lookat
        if (dirLen < 0.01) {
          // Target is at camera position - complete the lookat
          this._completeDynamicLookat();
          return;
        }

        if (dirLen >= 0.0001) {
          direction.divideScalar(dirLen);
          const horiz = Math.sqrt(
            direction.x * direction.x + direction.z * direction.z
          );
          const targetYaw = Math.atan2(-direction.x, -direction.z);
          const targetPitch = Math.atan2(direction.y, horiz);

          if (isFinite(targetYaw) && isFinite(targetPitch)) {
            // Recalculate end quaternion for current target position (important for moving targets)
            const newEndQuat = new THREE.Quaternion().setFromEuler(
              new THREE.Euler(targetPitch, targetYaw, 0, "YXZ")
            );

            // Only use direct tracking AFTER the initial transition is complete
            // During transition (progress < 1.0), update end quaternion but use normal interpolation
            if (this.lookAtProgress >= 1.0) {
              // For dynamic tracking after transition, directly set camera rotation (smoothly interpolate for smoothness)
              // Use slerp from current rotation to target for smooth tracking
              const trackingSpeed = 0.15; // Higher = faster tracking, lower = smoother
              this.camera.quaternion.slerp(newEndQuat, trackingSpeed);
              this.camera.quaternion.normalize();

              // Update yaw/pitch for smooth handoff
              const euler = new THREE.Euler().setFromQuaternion(
                this.camera.quaternion,
                "YXZ"
              );
              this.yaw = euler.y;
              this.pitch = euler.x;
              this.targetYaw = this.yaw;
              this.targetPitch = this.pitch;

              // Skip the normal interpolation below for dynamic tracking (only after transition)
              return;
            } else {
              // During transition: update end quaternion so interpolation targets current position
              this.lookAtEndQuat.copy(newEndQuat);
              // Ensure quaternions are on same hemisphere
              if (this.lookAtStartQuat.dot(this.lookAtEndQuat) < 0) {
                this.lookAtEndQuat.x *= -1;
                this.lookAtEndQuat.y *= -1;
                this.lookAtEndQuat.z *= -1;
                this.lookAtEndQuat.w *= -1;
              }
              // Fall through to normal interpolation below
            }
          }
        }
      }

      // Use appropriate duration based on current phase
      const currentDuration = this.lookAtReturning
        ? this.lookAtReturnDuration
        : this.lookAtDuration;
      this.lookAtProgress += dt / currentDuration;

      // Start DoF and zoom transitions when we reach the configured threshold (only during initial lookat, not return)
      if (!this.lookAtReturning) {
        const transitionStart =
          this.currentZoomConfig?.transitionStart ||
          this.dofTransitionStartProgress;
        if (
          (this.lookAtDofActive || this.lookAtZoomActive) &&
          !this.dofTransitioning &&
          !this.zoomTransitioning &&
          this.lookAtProgress >= transitionStart
        ) {
          // Start DOF transition if DOF is active
          if (this.lookAtDofActive) {
            this.dofTransitioning = true;
          }
          // Start zoom transition if zoom is active
          if (this.lookAtZoomActive) {
            this.startFov = this.currentFov; // Capture current FOV as start for zoom
            this.zoomTransitioning = true;
          }
          this.logger.log(
            `Starting ${this.lookAtDofActive ? "DoF" : ""}${
              this.lookAtDofActive && this.lookAtZoomActive ? " and " : ""
            }${this.lookAtZoomActive ? "zoom" : ""} transition${
              this.lookAtDofActive && this.lookAtZoomActive ? "s" : ""
            } at ${(this.lookAtProgress * 100).toFixed(0)}% (threshold: ${(
              transitionStart * 100
            ).toFixed(0)}%)`
          );
        }
      }

      if (this.lookAtProgress >= 1.0) {
        this.lookAtProgress = 1.0;

        // For dynamic tracking with returnToOriginalView: false, complete the lookat after transition
        // but continue tracking for smooth camera movement
        // For returnToOriginalView: true, keep tracking during the hold phase before returning
        if (
          this.lookAtPositionFunction &&
          !this.lookAtReturning &&
          this.lookAtReturnToOriginalView
        ) {
          // Keep tracking - don't complete the lookat yet (will complete after hold/return)
          // The dynamic tracking code above will continue to update the camera
        } else if (this.lookAtReturnToOriginalView && !this.lookAtReturning) {
          // Check if we should hold before returning
          if (this.lookAtHoldDuration > 0) {
            // Start holding phase
            this.lookAtHolding = true;
            this.lookAtHoldTimer = 0;
            this.logger.log(
              `Holding at target for ${this.lookAtHoldDuration}s before returning`
            );
          } else {
            // No hold, start return immediately
            this.lookAtReturning = true;
            this.lookAtProgress = 0;
            this.logger.log(
              `Starting return to original view (${this.lookAtReturnDuration}s)`
            );

            // Start DoF/zoom reset immediately (so they return to normal as camera returns)
            if (this.lookAtDofActive || this.lookAtZoomActive) {
              this.dofHoldTimer = 0;

              const wasDofActive = this.lookAtDofActive;
              const wasZoomActive = this.lookAtZoomActive;

              // Reset DOF if it was active
              if (this.sparkRenderer && this.lookAtDofActive) {
                this.targetFocalDistance = this.baseFocalDistance;
                this.targetApertureSize = this.baseApertureSize;
                this.lookAtDofActive = false;
                this.dofTransitioning = true;
                this.dofTransitionProgress = 0; // Reset for return transition
              }

              // Reset zoom if it was active
              if (this.lookAtZoomActive) {
                this.startFov = this.currentFov; // Capture current FOV for smooth transition
                this.targetFov = this.baseFov;
                this.lookAtZoomActive = false;
                this.zoomTransitioning = true;
                this.zoomTransitionProgress = 0; // Reset zoom progress
              }

              // Override transition duration to match return duration
              this.returnTransitionDuration = this.lookAtReturnDuration;

              this.logger.log(
                `Starting ${wasDofActive ? "DoF" : ""}${
                  wasDofActive && wasZoomActive ? "/" : ""
                }${wasZoomActive ? "zoom" : ""} reset during return (${
                  this.lookAtReturnDuration
                }s)`
              );
            }

            // Swap start and end quaternions for return journey
            const temp = this.lookAtStartQuat.clone();
            this.lookAtStartQuat.copy(this.lookAtEndQuat);
            this.lookAtEndQuat.copy(temp);
          }
        } else if (
          !this.lookAtPositionFunction ||
          !this.lookAtReturnToOriginalView
        ) {
          // Look-at complete (or return complete)
          // Complete if: no dynamic tracking, OR returnToOriginalView is false (complete after transition even with dynamic tracking)
          this.isLookingAt = false;
          this.lookAtReturning = false;
          this.lookAtHolding = false;
          this.returnTransitionDuration = null; // Clear return transition override

          // Reset glance state
          this.glanceState = null;
          this.glanceTimer = 0;
          this.wasIdleAllowed = false;
          this.currentRoll = 0; // Reset head tilt

          // Start DoF/zoom reset timer (only if NOT returning to original view)
          // If returnToOriginalView was true, the reset already happened during the return phase
          if (
            !this.lookAtReturnToOriginalView &&
            (this.lookAtDofActive || this.lookAtZoomActive)
          ) {
            const resetDuration =
              this.currentZoomConfig?.holdDuration || this.dofHoldDuration;
            this.dofHoldTimer = resetDuration;
          }

          // Call completion callback if provided (should handle input restoration)
          if (this.lookAtOnComplete) {
            this.lookAtOnComplete();
            this.lookAtOnComplete = null;
          } else {
            // If no onComplete callback, restore input directly
            // This handles cases where zoom is still transitioning but lookat is complete
            if (this.lookAtRestoreInput && this.lookAtDisabledInput) {
              const restoreInput = this.lookAtRestoreInput;
              if (restoreInput !== false) {
                const restoreMovement =
                  restoreInput === true ||
                  (restoreInput && restoreInput.movement !== false);
                const restoreRotation =
                  restoreInput === true ||
                  (restoreInput && restoreInput.rotation !== false);

                if (restoreMovement || restoreRotation) {
                  // Only restore if zoom/DoF transitions are also complete
                  if (!this.zoomTransitioning && !this.dofTransitioning) {
                    this.inputDisabled = false;
                    this.inputManager.enable();
                    this.lookAtDisabledInput = false;
                    this.logger.warn(
                      "Input restored after lookat completed (no callback)"
                    );
                  }
                }
              }
            }
          }
        }
      }

      // Slerp between start and end quaternions
      const t = Math.min(this.lookAtProgress, 1.0);
      // Apply easing for smoother motion
      const easedT = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      const currentQuat = new THREE.Quaternion();
      currentQuat.slerpQuaternions(
        this.lookAtStartQuat,
        this.lookAtEndQuat,
        easedT
      );

      // Apply quaternion to camera
      this.camera.quaternion.copy(currentQuat);

      // Update yaw/pitch for smooth handoff to normal control
      const euler = new THREE.Euler().setFromQuaternion(currentQuat, "YXZ");
      this.yaw = euler.y;
      this.pitch = euler.x;
      this.targetYaw = this.yaw;
      this.targetPitch = this.pitch;
    }

    // Handle character move-to sequence
    if (this.isMovingTo) {
      this.moveToProgress += dt / this.moveToDuration;

      if (this.moveToProgress >= 1.0) {
        // Move complete
        this.moveToProgress = 1.0;
        this.isMovingTo = false;

        // Set final position and rotation
        this.character.setTranslation(
          {
            x: this.moveToTargetPos.x,
            y: this.moveToTargetPos.y,
            z: this.moveToTargetPos.z,
          },
          true
        );
        this.yaw = this.moveToTargetYaw;
        this.pitch = this.moveToTargetPitch;
        this.targetYaw = this.yaw;
        this.targetPitch = this.pitch;
        this.bodyYaw = this.yaw; // Update body yaw to match

        // Restore input based on what was disabled (only if restoreInput is true)
        if (this.moveToRestoreInput && this.moveToInputControl) {
          if (
            this.moveToInputControl.disableMovement &&
            this.moveToInputControl.disableRotation
          ) {
            // Both were disabled, enable everything
            this.inputDisabled = false;
            this.inputManager.enable();
            this.logger.log("Move-to complete, restored full input");
          } else if (this.moveToInputControl.disableMovement) {
            // Only movement was disabled, restore it
            this.inputManager.enableMovement();
            this.logger.log("Move-to complete, restored movement input");
          } else if (this.moveToInputControl.disableRotation) {
            // Only rotation was disabled, restore it
            this.inputManager.enableRotation();
            this.logger.log("Move-to complete, restored rotation input");
          }
          this.moveToInputControl = null;
          this.moveToRestoreInput = true; // Reset to default
        } else if (!this.moveToRestoreInput) {
          // Input restoration disabled - keep controls disabled
          this.logger.log(
            "Move-to complete, input NOT restored (restoreInput: false)"
          );
          this.moveToInputControl = null;
          this.moveToRestoreInput = true; // Reset to default for next moveTo
        }

        // Call completion callback if provided
        if (this.moveToOnComplete) {
          this.moveToOnComplete();
          this.moveToOnComplete = null;
        }

        this.logger.log("Move-to complete");
      } else {
        // Interpolate position and rotation
        const t = Math.min(this.moveToProgress, 1.0);
        // Apply ease-in-out for smooth motion
        const easedT = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

        // Lerp position
        const currentPos = new THREE.Vector3();
        currentPos.lerpVectors(
          this.moveToStartPos,
          this.moveToTargetPos,
          easedT
        );

        // Update physics body position
        this.character.setTranslation(
          { x: currentPos.x, y: currentPos.y, z: currentPos.z },
          true
        );

        // Interpolate rotation
        this.yaw =
          this.moveToStartYaw +
          (this.moveToTargetYaw - this.moveToStartYaw) * easedT;
        this.pitch =
          this.moveToStartPitch +
          (this.moveToTargetPitch - this.moveToStartPitch) * easedT;
        if (!isFinite(this.yaw) || !isFinite(this.pitch)) {
          this.logger.error(
            `MoveTo: interpolation produced NaN (start=(${this.moveToStartYaw}, ${this.moveToStartPitch}), target=(${this.moveToTargetYaw}, ${this.moveToTargetPitch}), t=${easedT})`
          );
          this.yaw = 0;
          this.pitch = 0;
        }

        this.targetYaw = this.yaw;
        this.targetPitch = this.pitch;

        // Also update body yaw during scripted movement
        this.bodyYaw = this.yaw;
      }
    }

    if (!this.isLookingAt && !this.inputDisabled) {
      // Normal camera control
      // Get camera input from input manager
      const cameraInput = this.inputManager.getCameraInput(dt);

      // Check if there's any manual camera input and cancel glance if so
      const hasManualInput =
        Math.abs(cameraInput.x) > 0.0001 || Math.abs(cameraInput.y) > 0.0001;
      if (hasManualInput && this.glanceState !== null) {
        // Cancel the glance and reset to current position
        this.glanceState = null;
        this.glanceTimer = 5.0 + Math.random() * 3.0; // Next glance in 5-8 seconds
        this.currentRoll = 0; // Reset head tilt immediately
        this.logger.log("Manual camera input detected, cancelling glance");
      }

      // Apply camera rotation differently based on input source
      if (cameraInput.hasGamepad) {
        // Gamepad: Apply directly to yaw/pitch for immediate response
        // (no target/smoothing to avoid "chasing" behavior)
        this.yaw -= cameraInput.x;
        this.pitch -= cameraInput.y;
        this.pitch = Math.max(
          -Math.PI / 3,
          Math.min(Math.PI / 2 - 0.01, this.pitch)
        );

        // Keep targets in sync
        this.targetYaw = this.yaw;
        this.targetPitch = this.pitch;
      } else if (hasManualInput) {
        // Mouse: Apply to targets with smoothing for precise control (only if there's input)
        // Validate cameraInput before applying
        const inputX = isFinite(cameraInput.x) ? cameraInput.x : 0;
        const inputY = isFinite(cameraInput.y) ? cameraInput.y : 0;

        // Ensure targets are finite before modifying
        if (!isFinite(this.targetYaw)) this.targetYaw = this.yaw;
        if (!isFinite(this.targetPitch)) this.targetPitch = this.pitch;

        this.targetYaw -= inputX;
        this.targetPitch -= inputY;
        this.targetPitch = Math.max(
          -Math.PI / 3,
          Math.min(Math.PI / 2 - 0.01, this.targetPitch)
        );

        // Ensure targets remain finite
        if (!isFinite(this.targetYaw)) this.targetYaw = this.yaw;
        if (!isFinite(this.targetPitch)) this.targetPitch = this.pitch;

        // Smooth camera rotation to reduce jitter
        // Ensure yaw/pitch are finite before smoothing
        if (!isFinite(this.yaw)) this.yaw = this.targetYaw;
        if (!isFinite(this.pitch)) this.pitch = this.targetPitch;

        this.yaw += (this.targetYaw - this.yaw) * this.cameraSmoothingFactor;
        this.pitch +=
          (this.targetPitch - this.pitch) * this.cameraSmoothingFactor;

        // Ensure smoothed values remain finite
        if (!isFinite(this.yaw)) this.yaw = this.targetYaw;
        if (!isFinite(this.pitch)) this.pitch = this.targetPitch;
      } else {
        // No manual input, still apply smoothing if there's a difference
        // Ensure all values are finite
        if (!isFinite(this.yaw)) this.yaw = 0;
        if (!isFinite(this.pitch)) this.pitch = 0;
        if (!isFinite(this.targetYaw)) this.targetYaw = this.yaw;
        if (!isFinite(this.targetPitch)) this.targetPitch = this.pitch;

        this.yaw += (this.targetYaw - this.yaw) * this.cameraSmoothingFactor;
        this.pitch +=
          (this.targetPitch - this.pitch) * this.cameraSmoothingFactor;

        // Ensure smoothed values remain finite
        if (!isFinite(this.yaw)) this.yaw = this.targetYaw;
        if (!isFinite(this.pitch)) this.pitch = this.targetPitch;
      }

      // Reset frame input after processing
      this.inputManager.resetFrameInput();
    }

    // Input -> desired velocity in XZ plane (disabled during look-at)
    let isMoving = false;
    let isSprinting = false;
    if (!this.inputDisabled) {
      const { forward, right } = this.getForwardRightVectors();
      const movementInput = this.inputManager.getMovementInput();

      // Check if touch joystick is active - if so, use speed multiplier for smooth speed control
      const touchSpeedMultiplier = this.inputManager.getTouchSpeedMultiplier();
      const isTouchActive = touchSpeedMultiplier > 0;

      const desired = this._desired;

      if (isTouchActive) {
        // Touch controls: speed ramps from 0 to sprint speed based on stick distance
        // At center (0) = 0 speed, at max distance (1) = sprint speed
        isSprinting = touchSpeedMultiplier >= 0.95; // Consider it sprinting when near max
        const maxSpeed = this.baseSpeed * this.sprintMultiplier;
        const moveSpeed = maxSpeed * touchSpeedMultiplier;

        // Apply movement input (y is forward/back, x is left/right)
        // Reuse temp vectors instead of cloning
        desired.copy(
          this._tempForward.copy(forward).multiplyScalar(movementInput.y)
        );
        desired.add(
          this._tempRight.copy(right).multiplyScalar(movementInput.x)
        );

        isMoving = desired.lengthSq() > 1e-6;
        if (isMoving) {
          desired.normalize().multiplyScalar(moveSpeed);
        } else {
          // No movement input but touch is active - don't move
          desired.set(0, 0, 0);
        }
      } else {
        // Normal keyboard/gamepad controls
        isSprinting = this.inputManager.isSprinting();
        const moveSpeed = isSprinting
          ? this.baseSpeed * this.sprintMultiplier
          : this.baseSpeed;

        // Apply movement input (y is forward/back, x is left/right)
        // Reuse temp vectors instead of cloning
        desired.copy(
          this._tempForward.copy(forward).multiplyScalar(movementInput.y)
        );
        desired.add(
          this._tempRight.copy(right).multiplyScalar(movementInput.x)
        );

        isMoving = desired.lengthSq() > 1e-6;
        if (isMoving) {
          desired.normalize().multiplyScalar(moveSpeed);
        }
      }

      // When moving, body gradually aligns with camera for smooth turning
      const angleDiff = this.yaw - this.bodyYaw;
      const normalizedDiff = Math.atan2(
        Math.sin(angleDiff),
        Math.cos(angleDiff)
      );
      // Smoother rotation while moving (0.15)
      this.bodyYaw += normalizedDiff * 0.15;

      // Always check neck limit (whether moving or not)
      let cameraBodyDiff = this.yaw - this.bodyYaw;
      cameraBodyDiff = Math.atan2(
        Math.sin(cameraBodyDiff),
        Math.cos(cameraBodyDiff)
      );

      // If not moving and beyond neck limit, turn body to catch up
      // Use moderate speed so body turns smoothly
      if (!isMoving && Math.abs(cameraBodyDiff) > this.neckRotationLimit) {
        this.bodyYaw += cameraBodyDiff * 0.06; // Balanced speed for smooth rotation
      }

      // Apply velocity
      const linvel = this.character.linvel();

      if (this.flightMode) {
        // Flight mode: full 3D movement with Q/E for vertical
        // Reuse vector instead of creating new one
        this._verticalVelocity
          .copy(this._up)
          .multiplyScalar(this.verticalInput * this.flightSpeed);

        // If not moving (or movement disabled), only apply vertical velocity
        if (!isMoving || !this.inputManager.isMovementEnabled()) {
          this.character.setLinvel(
            { x: 0, y: this._verticalVelocity.y, z: 0 },
            true
          );
        } else {
          this.character.setLinvel(
            {
              x: desired.x,
              y: this._verticalVelocity.y,
              z: desired.z,
            },
            true
          );
        }
      } else {
        // Normal mode: preserve Y velocity (gravity)
        // If not moving (or movement disabled), explicitly clear horizontal velocity
        if (!isMoving || !this.inputManager.isMovementEnabled()) {
          this.character.setLinvel({ x: 0, y: linvel.y, z: 0 }, true);
        } else {
          this.character.setLinvel(
            { x: desired.x, y: linvel.y, z: desired.z },
            true
          );
        }
      }
    } else {
      // Stop movement when input is disabled
      const linvel = this.character.linvel();
      if (this.flightMode) {
        // In flight mode, stop all movement including vertical
        this.character.setLinvel({ x: 0, y: 0, z: 0 }, true);
      } else {
        // Normal mode: preserve gravity
        this.character.setLinvel({ x: 0, y: linvel.y, z: 0 }, true);
      }
    }

    // Update idle glance system (before headbob so it can affect targetYaw)
    this.updateIdleGlance(dt, isMoving);

    // Update viewmaster insanity intensity (smoothed for gradual ramp-down)
    this.updateInsanityIntensity(dt);

    // Update headbob state
    const targetIntensity = this.headbobEnabled ? (isMoving ? 1.0 : 0.0) : 0.0;
    this.headbobIntensity += (targetIntensity - this.headbobIntensity) * 0.15; // Smooth transition

    // Always update idle headbob time for breathing animation
    this.idleHeadbobTime += dt;

    if (isMoving && this.headbobEnabled) {
      this.headbobTime += dt; // Accumulate time only when moving
    }

    // Update first-person body animation with blending
    if (
      this.enableFirstPersonBody &&
      this.bodyAnimationMixer &&
      this.walkAnimation &&
      this.idleAnimation
    ) {
      // Don't update the mixer if it belongs to SceneManager (it updates its own mixers)
      // Only update if we created our own mixer
      const isOurMixer =
        !this.sceneManager?.animationMixers.has("firstPersonBody");
      if (isOurMixer) {
        this.bodyAnimationMixer.update(dt);
      }

      // Blend between walk/idle animations
      const blendSpeed = 0.1; // How fast to blend between animations

      if (isMoving) {
        // Moving: Blend to walk animation
        const targetTimeScale = isSprinting ? 1.5 : 1.0;

        // Walk weight up, idle down
        const currentWalkWeight = this.walkAnimation.getEffectiveWeight();
        const newWalkWeight = Math.min(1.0, currentWalkWeight + blendSpeed);
        this.walkAnimation.setEffectiveWeight(newWalkWeight);
        this.idleAnimation.setEffectiveWeight(1.0 - newWalkWeight);

        // Adjust walk animation speed
        this.walkAnimation.timeScale = targetTimeScale;
      } else {
        // Idle: Blend to idle animation
        const currentIdleWeight = this.idleAnimation.getEffectiveWeight();
        const newIdleWeight = Math.min(1.0, currentIdleWeight + blendSpeed);
        this.idleAnimation.setEffectiveWeight(newIdleWeight);
        this.walkAnimation.setEffectiveWeight(1.0 - newIdleWeight);
      }
    } else if (
      this.enableFirstPersonBody &&
      this.bodyModelGroup &&
      !this.bodyAnimationMixer
    ) {
      // Log once that we don't have an animation mixer
      if (!this.loggedNoAnimation) {
        this.logger.warn("No animation mixer found for body");
        this.loggedNoAnimation = true;
      }
    }

    // Update footstep audio
    if (this.footstepSound) {
      // Resume audio context if it's suspended (browser autoplay policy)
      if (this.audioListener.context.state === "suspended") {
        this.audioListener.context.resume();
      }

      if (isMoving && !this.isPlayingFootsteps) {
        this.footstepSound.play();
        this.isPlayingFootsteps = true;
      } else if (!isMoving && this.isPlayingFootsteps) {
        this.footstepSound.stop();
        this.isPlayingFootsteps = false;
      }

      // Adjust playback rate based on sprint
      if (this.isPlayingFootsteps) {
        const playbackRate = isSprinting ? 1.5 : 1.0;
        this.footstepSound.rate(playbackRate);
      }
    }

    // Get physics body position (needed for character container update below)
    const p = this.character.translation();

    // Sync camera to physics body position (unless camera sync is disabled)
    if (!this.cameraSyncDisabled) {
      // Calculate headbob offset (movement + idle)
      const movementHeadbob = this.headbobEnabled
        ? this.calculateHeadbob(isSprinting)
        : { vertical: 0, horizontal: 0 };
      const idleHeadbob = this.headbobEnabled
        ? this.calculateIdleHeadbob()
        : { vertical: 0, horizontal: 0 };
      const { forward: fwd, right: rgt } = this.getForwardRightVectors();

      // Camera follow: position slightly behind and above the character with headbob
      // Reuse vectors instead of creating new ones
      this._cameraOffset.set(0, this.cameraHeight, 0);
      this._camFollow.set(p.x, p.y, p.z).add(this._cameraOffset);

      // Apply combined headbob: vertical (Y) and horizontal (side-to-side relative to view direction)
      this._camFollow.y += movementHeadbob.vertical + idleHeadbob.vertical;
      this._camFollow.add(
        this._tempRight
          .copy(rgt)
          .multiplyScalar(movementHeadbob.horizontal + idleHeadbob.horizontal)
      );

      this.camera.position.copy(this._camFollow);
    } else {
      // Camera sync is disabled (frozen state), but still apply idle breathing
      if (this.headbobEnabled) {
        const idleHeadbob = this.calculateIdleHeadbob();
        const { right: rgt } = this.getForwardRightVectors();

        // Apply idle breathing relative to frozen base position (amplified 3x for labored breathing)
        this.camera.position.copy(this.frozenCameraBasePosition);
        this.camera.position.y += idleHeadbob.vertical * 3.0;
        this.camera.position.add(
          this._tempRight.copy(rgt).multiplyScalar(idleHeadbob.horizontal * 3.0)
        );
      }
    }

    // Update character container position and rotation
    // Body is a child of this container, so it follows automatically
    // Use bodyYaw (not this.yaw) so idle glances don't rotate the body
    if (this.characterContainer) {
      this.characterContainer.position.set(p.x, p.y, p.z);
      this.characterContainer.rotation.set(0, this.bodyYaw, 0);
    }

    // Build look direction from yaw/pitch (only when not in look-at mode)
    if (!this.isLookingAt) {
      // Smoothly lerp roll back to zero after blended animations end
      if (this.isLerpingRollToZero) {
        const rollDelta = this.rollLerpSpeed * dt;
        if (Math.abs(this.currentRoll) <= rollDelta) {
          this.currentRoll = 0;
          this.isLerpingRollToZero = false;
        } else {
          this.currentRoll -= Math.sign(this.currentRoll) * rollDelta;
        }
      }

      // Validate pitch/yaw before using them (can become invalid during blending)
      if (!isFinite(this.pitch)) this.pitch = 0;
      if (!isFinite(this.yaw)) this.yaw = 0;
      if (!isFinite(this.currentRoll)) this.currentRoll = 0;

      // Validate camera position before lookAt
      const pos = this.camera.position;
      if (!isFinite(pos.x) || !isFinite(pos.y) || !isFinite(pos.z)) {
        // Skip lookAt if position is invalid (animation manager will handle it)
        return;
      }

      // Ensure camera quaternion is valid before lookAt (can be corrupted by animations)
      const q = this.camera.quaternion;
      if (
        !isFinite(q.x) ||
        !isFinite(q.y) ||
        !isFinite(q.z) ||
        !isFinite(q.w)
      ) {
        // Reconstruct from yaw/pitch if quaternion is invalid
        this.camera.quaternion.setFromEuler(
          new THREE.Euler(this.pitch, this.yaw, this.currentRoll, "YXZ")
        );
      } else {
        this.camera.quaternion.normalize();
      }

      // Apply camera rotation with roll
      // Use setFromEuler directly instead of lookAt to preserve roll
      this.camera.quaternion.setFromEuler(
        new THREE.Euler(this.pitch, this.yaw, this.currentRoll, "YXZ")
      );

      // Validate and normalize after setting quaternion
      const resultQ = this.camera.quaternion;
      if (
        !isFinite(resultQ.x) ||
        !isFinite(resultQ.y) ||
        !isFinite(resultQ.z) ||
        !isFinite(resultQ.w)
      ) {
        // Reconstruct from yaw/pitch/roll if quaternion is invalid
        this.camera.quaternion.setFromEuler(
          new THREE.Euler(this.pitch, this.yaw, this.currentRoll, "YXZ")
        );
      } else {
        this.camera.quaternion.normalize();
      }
    }
  }
}

export default CharacterController;
