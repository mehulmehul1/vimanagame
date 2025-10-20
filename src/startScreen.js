import * as THREE from "three";
import { createParticleText } from "./titleText.js";
import { TitleSequence } from "./titleSequence.js";
import { GAME_STATES } from "./gameData.js";
import "./styles/startScreen.css";

/**
 * StartScreen - Manages the intro camera animation and start button
 */
export class StartScreen {
  constructor(camera, scene, options = {}) {
    this.camera = camera;
    this.scene = scene;
    this.isActive = true;
    this.hasStarted = false;
    this.transitionProgress = 0;
    this.uiManager = options.uiManager || null;
    this.sceneManager = options.sceneManager || null;
    this.dialogManager = options.dialogManager || null;

    // Additional state
    this.introStartTriggered = false;
    this.titleSequence = null;
    this.title = null;
    this.byline = null;

    // Animation tracking
    this.coneCurveObject = null;
    this.coneAnimatedMesh = null;
    this.isLoadingAnimation = false;
    this.animationCompleteTime = null; // Set when START is clicked
    this.animationDirection = 1; // 1 for forward, -1 for reverse
    this.animationSpeed = 1; // Speed multiplier for animation
    this.cameraSpinProgress = 0; // 0 to 1 for 180-degree spin when reversing
    this.cameraSpinDuration = 1.0; // 1 second for spin
    this.isSpinning = false; // True when camera is spinning around
    this.lerpToSpawnDuration = 8.0; // Duration for lerp from animation end to spawn (set dynamically)

    // Unified path approach
    this.unifiedPath = null; // CatmullRomCurve3 that combines animation + path to spawn
    this.unifiedPathProgress = 0; // 0 to 1 along the unified path
    this.unifiedPathDuration = 0; // Total time to traverse the unified path
    this.isFollowingUnifiedPath = false; // True when following the combined path
    this.initialLookDirection = null; // Camera's look direction when unified path starts
    this.pathInitialTangent = null; // Path's tangent at start
    this.rotationTransitionTime = 0; // Elapsed time for rotation transition
    this.previousTangent = null; // For smoothing tangent changes
    this.smoothedTangent = null; // Smoothed tangent to reduce sudden rotations

    // Pre-start animation smoothing
    this.smoothedForward = null; // Smoothed forward vector for pre-start GLB animation

    // GLB animation start position (0 to 1, where 0 is start, 1 is end)
    this.glbAnimationStartProgress =
      options.glbAnimationStartProgress !== undefined
        ? options.glbAnimationStartProgress
        : 0; // Default to start of animation

    // Camera tilt/roll for unsteady flight effect
    this.tiltTime = Math.random() * 100; // Randomize start time
    this.tiltSpeed1 = 0.3 + Math.random() * 0.2; // Base tilt speed (0.3-0.5)
    this.tiltSpeed2 = 0.5 + Math.random() * 0.3; // Secondary tilt speed (0.5-0.8)
    this.tiltAmount = 0.08; // Max tilt in radians (~4.5 degrees)

    // Circle animation settings (fallback if GLB doesn't load)
    this.circleCenter = options.circleCenter || new THREE.Vector3(0, 0, 0);
    this.circleRadius = options.circleRadius || 15;
    this.circleHeight = options.circleHeight || 10;
    this.circleSpeed = options.circleSpeed || 0.3;
    this.circleTime = 0;

    // Target position (where camera should end up)
    this.targetPosition =
      options.targetPosition || new THREE.Vector3(10, 1.6, 15);
    this.targetRotation = options.targetRotation || {
      yaw: THREE.MathUtils.degToRad(-210),
      pitch: 0,
    };

    // Transition settings
    this.transitionDuration = options.transitionDuration || 2.0; // seconds

    // Store initial camera state for transition
    this.startPosition = new THREE.Vector3();
    this.startLookAt = new THREE.Vector3();

    // Create start button
    this.createStartButton();

    // Create title text particles
    const { title, byline } = this.createTitleText();
    this.title = title;
    this.byline = byline;

    // Load the camera curve animation
    this.loadCameraAnimation();
  }

  /**
   * Load the camera curve animation from ConeCurve.glb
   */
  async loadCameraAnimation() {
    if (!this.sceneManager) {
      console.warn(
        "StartScreen: No sceneManager provided, using fallback circle animation"
      );
      return;
    }

    this.isLoadingAnimation = true;

    try {
      // Get the coneCurve object (should be loaded by gameManager)
      this.coneCurveObject = this.sceneManager.getObject("coneCurve");

      if (!this.coneCurveObject) {
        console.warn(
          "StartScreen: coneCurve object not loaded yet, will retry"
        );
        // Retry after a short delay
        setTimeout(() => this.loadCameraAnimation(), 100);
        return;
      }

      // Find the "Cone" object within the loaded GLTF
      // Note: It doesn't have to be a mesh, just an Object3D that's being animated
      this.coneCurveObject.traverse((child) => {
        if (child.name === "Cone") {
          this.coneAnimatedMesh = child;
          console.log("StartScreen: Found Cone object, type:", child.type);
        }

        // Hide all meshes - we only need this for camera tracking
        if (child.isMesh) {
          child.visible = false;
        }
      });

      if (!this.coneAnimatedMesh) {
        console.warn(
          "StartScreen: Could not find 'Cone' mesh in ConeCurve.glb, using fallback"
        );
        this.isLoadingAnimation = false;
        return;
      }

      console.log("StartScreen: Camera animation loaded successfully");
      console.log(
        "StartScreen: Found animated mesh:",
        this.coneAnimatedMesh.name
      );

      // Manually trigger the animation to play
      if (this.sceneManager) {
        console.log("StartScreen: Manually playing coneCurve-anim");
        this.sceneManager.playAnimation("coneCurve-anim");

        // Set initial animation time based on glbAnimationStartProgress
        if (this.glbAnimationStartProgress > 0) {
          const action =
            this.sceneManager.animationActions.get("coneCurve-anim");
          if (action) {
            const clip = action.getClip();
            const duration = clip.duration;
            const startTime = duration * this.glbAnimationStartProgress;
            action.time = startTime;

            // Force the mixer to update immediately so the mesh transform reflects the new time
            const mixer = this.sceneManager.animationMixers.get("coneCurve");
            if (mixer) {
              mixer.update(0); // Update with 0 dt to evaluate current time without advancing
            }

            console.log(
              `StartScreen: Set animation start time to ${startTime.toFixed(
                2
              )}s (${(this.glbAnimationStartProgress * 100).toFixed(
                1
              )}% of ${duration.toFixed(2)}s)`
            );
          }
        }
      }

      this.isLoadingAnimation = false;
    } catch (error) {
      console.error("StartScreen: Error loading camera animation:", error);
      this.isLoadingAnimation = false;
    }
  }

  /**
   * Create the start button overlay
   */
  createStartButton() {
    // Create overlay container
    this.overlay = document.createElement("div");
    this.overlay.id = "intro-overlay";

    // Create tagline
    this.tagline = document.createElement("div");
    this.tagline.className = "intro-tagline";
    this.tagline.innerHTML = `In this town<br>it's hard to stray far from...`;

    // Create start button
    this.startButton = document.createElement("button");
    this.startButton.className = "intro-button";
    this.startButton.textContent = "START";

    // Click handler for start button
    this.startButton.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent click from reaching canvas
      this.startGame();

      // Request pointer lock when game starts (input will be disabled until control is enabled)
      const canvas = document.querySelector("canvas");
      if (canvas && canvas.requestPointerLock) {
        canvas.requestPointerLock();
      }
    });

    // Create options button
    this.optionsButton = document.createElement("button");
    this.optionsButton.className = "intro-button";
    this.optionsButton.textContent = "OPTIONS";

    // Click handler for options button
    this.optionsButton.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent click from reaching canvas
      if (this.uiManager) {
        this.uiManager.show("options-menu");
      }
    });

    this.overlay.appendChild(this.tagline);
    this.overlay.appendChild(this.startButton);
    this.overlay.appendChild(this.optionsButton);
    document.body.appendChild(this.overlay);

    // Register with UI manager if available
    if (this.uiManager) {
      this.uiManager.registerElement(
        "intro-screen",
        this.overlay,
        "MAIN_MENU",
        {
          blocksInput: true,
          pausesGame: false, // Game hasn't started yet
        }
      );
    }
  }

  /**
   * Create title text particles for the start screen
   */
  createTitleText() {
    // Create a separate scene for title text particles to render on top
    this.textScene = new THREE.Scene();
    // Optimized near/far planes for text particles at z: -10 with disperseDistance: 5
    // This provides much better depth precision and reduces jittering
    this.textCamera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1.0, // Near plane: text can be as close as ~5 units (10 - 5)
      20.0 // Far plane: text can be as far as ~15 units (10 + 5)
    );

    // Create first title text (particle-based)
    const textData1 = createParticleText(this.textScene, {
      text: "THE SHADOW\nof the Czar",
      font: "LePorsche",
      fontSize: 120,
      color: new THREE.Color(0xffffff), // White
      position: { x: 0, y: 0, z: -2.5 }, // Base position for animation
      scale: 2.5 / 80, // Increased from 1.0/80 for larger text
      animate: true,
      particleDensity: 0.5, // Higher density for better text quality
    });
    this.textScene.remove(textData1.mesh);
    this.textCamera.add(textData1.mesh);
    this.textScene.add(this.textCamera);
    textData1.mesh.userData.baseScale = 1.0 / 80;
    textData1.mesh.visible = false; // Hide initially

    // Create a wrapper object that includes particles
    const title = {
      mesh: textData1.mesh,
      particles: textData1.particles,
      update: textData1.update,
    };

    // Create second title text (positioned below first)
    const textData2 = createParticleText(this.textScene, {
      text: "by JAMES C. KANE",
      font: "LePorsche",
      fontSize: 30,
      color: new THREE.Color(0xffffff), // White
      position: { x: 0, y: -0.5, z: -2 }, // Base position lower for animation
      scale: 2.5 / 80, // Increased from 1.0/80 to match first text
      animate: true,
      particleDensity: 0.5, // Higher density for better text quality
    });
    this.textScene.remove(textData2.mesh);
    this.textCamera.add(textData2.mesh);
    textData2.mesh.userData.baseScale = 1.0 / 80;
    textData2.mesh.visible = false; // Hide initially

    // Create a wrapper object that includes particles
    const byline = {
      mesh: textData2.mesh,
      particles: textData2.particles,
      update: textData2.update,
    };

    return { title, byline };
  }

  /**
   * Start the game - create unified path from current position to spawn
   */
  startGame() {
    if (this.hasStarted) return;

    this.hasStarted = true;

    // Create unified path: sample animation curve + extend to spawn
    if (this.sceneManager && this.coneAnimatedMesh) {
      const action = this.sceneManager.animationActions.get("coneCurve-anim");
      if (action) {
        const clip = action.getClip();
        const duration = clip.duration;
        const currentTime = action.time;
        const progress = currentTime / duration;

        console.log(
          `StartScreen: Creating unified path from ${(progress * 100).toFixed(
            1
          )}% of animation`
        );

        // Determine which end of animation to head to
        const goingToStart = progress < 0.5;
        const targetTime = goingToStart ? 0 : duration;

        this.animationDirection = goingToStart ? -1 : 1;
        this.isSpinning = goingToStart;
        this.cameraSpinProgress = 0;

        // Sample points from animation curve
        const pathPoints = [];
        const numSamples = 20; // Number of points to sample from animation

        // Sample from current position to target (start or end)
        for (let i = 0; i <= numSamples; i++) {
          const t = i / numSamples;
          const sampleTime = THREE.MathUtils.lerp(currentTime, targetTime, t);

          // Set animation to this time and get position
          action.time = sampleTime;
          const mixer = this.sceneManager.animationMixers.get("coneCurve");
          if (mixer) {
            mixer.update(0); // Update to apply the time change
          }

          // Get world position at this time
          const worldPos = new THREE.Vector3();
          this.coneAnimatedMesh.getWorldPosition(worldPos);
          pathPoints.push(worldPos.clone());
        }

        // Add points extending to player spawn
        const lastAnimPoint = pathPoints[pathPoints.length - 1];
        const spawnPoint = this.targetPosition;

        // Add intermediate points for smooth curve to spawn
        const numSpawnPoints = 8;
        for (let i = 1; i <= numSpawnPoints; i++) {
          const t = i / numSpawnPoints;
          const point = new THREE.Vector3().lerpVectors(
            lastAnimPoint,
            spawnPoint,
            t
          );
          pathPoints.push(point);
        }

        // Create smooth Catmull-Rom curve through all points
        // Use 'chordal' type which reduces sharp turns at control points
        this.unifiedPath = new THREE.CatmullRomCurve3(
          pathPoints,
          false,
          "chordal"
        );
        this.unifiedPathProgress = 0;

        // Set duration based on intro dialog length
        let dialogDuration = 8.0; // Default fallback
        if (this.dialogManager) {
          const introDuration = this.dialogManager.getDialogDuration("intro");
          if (introDuration > 0) {
            dialogDuration = introDuration;
            console.log(
              `StartScreen: Using intro dialog duration: ${dialogDuration}s`
            );
          } else {
            console.warn(
              `StartScreen: Could not get intro dialog duration, using fallback: ${dialogDuration}s`
            );
          }
        }

        this.unifiedPathDuration = dialogDuration; // Total time for entire journey
        this.isFollowingUnifiedPath = true;

        // Capture current camera look direction
        const currentLook = new THREE.Vector3(0, 0, -1);
        currentLook.applyQuaternion(this.camera.quaternion);
        this.initialLookDirection = currentLook.clone();

        // Get initial tangent of the unified path
        this.pathInitialTangent = this.unifiedPath.getTangentAt(0).normalize();

        // Reset rotation transition time and tangent smoothing
        this.rotationTransitionTime = 0;
        this.previousTangent = this.pathInitialTangent.clone();
        // Initialize smoothed tangent to current look direction to match Phase 1 start
        this.smoothedTangent = currentLook.clone();

        // Set spin duration if reversing
        if (this.isSpinning) {
          this.cameraSpinDuration = 1.0;
        }

        // Pause the original animation
        action.paused = true;

        // Flip high-level state from startScreen -> titleSequence
        if (this.uiManager && this.uiManager.gameManager) {
          this.uiManager.gameManager.setState({
            currentState: GAME_STATES.TITLE_SEQUENCE,
          });
        }

        console.log(
          `StartScreen: Created unified path with ${pathPoints.length} points over ${this.unifiedPathDuration}s`
        );
      }
    }

    // Immediately fade out start menu
    this.overlay.style.opacity = "0";
    this.overlay.style.transition = "opacity 0.15s ease";
    setTimeout(() => {
      this.overlay.style.display = "none";
      if (this.uiManager) {
        this.uiManager.hide("intro-screen");
      }
    }, 150);
  }

  /**
   * Calculate camera roll/tilt for unsteady flight effect
   * @returns {number} - Roll angle in radians
   */
  calculateCameraTilt() {
    // Use two sine waves at different frequencies for natural-looking wobble
    const tilt1 = Math.sin(this.tiltTime * this.tiltSpeed1) * this.tiltAmount;
    const tilt2 =
      Math.sin(this.tiltTime * this.tiltSpeed2) * this.tiltAmount * 0.5;
    return tilt1 + tilt2;
  }

  /**
   * Update camera position for circling or unified path
   * @param {number} dt - Delta time in seconds
   * @returns {boolean} - True if still active, false if complete
   */
  update(dt) {
    // Update tilt time
    this.tiltTime += dt;
    // Sync text camera with main camera
    if (this.textCamera) {
      this.textCamera.position.copy(this.camera.position);
      this.textCamera.quaternion.copy(this.camera.quaternion);
      this.textCamera.aspect = this.camera.aspect;
      this.textCamera.updateProjectionMatrix();
    }

    // Follow unified path after START is clicked
    if (this.isFollowingUnifiedPath && this.unifiedPath) {
      // Advance along the path
      this.unifiedPathProgress += dt / this.unifiedPathDuration;
      this.rotationTransitionTime += dt;

      if (this.unifiedPathProgress >= 1.0) {
        // Reached the end - finish
        this.isActive = false;
        this.cleanup();
        return false;
      }

      // Get position on the curve
      const t = Math.min(1.0, this.unifiedPathProgress);
      const position = this.unifiedPath.getPointAt(t);
      this.camera.position.copy(position);

      // Get tangent (forward direction) on the curve
      const rawTangent = this.unifiedPath.getTangentAt(t).normalize();

      // Smooth the tangent to reduce sudden direction changes (like at the junction)
      // Use a very low smoothing factor (0.04) for extremely gradual changes
      const smoothingFactor = 0.04;
      this.smoothedTangent.lerp(rawTangent, smoothingFactor);
      this.smoothedTangent.normalize();

      const tangent = this.smoothedTangent.clone();

      // Calculate final look direction at spawn
      const finalLookDirection = new THREE.Vector3(0, 0, -1).applyEuler(
        new THREE.Euler(
          this.targetRotation.pitch,
          this.targetRotation.yaw,
          0,
          "YXZ"
        )
      );

      let lookDirection = tangent.clone();

      // Phase 1: Transition from initial look direction to path tangent (first 3 seconds)
      const initialTransitionDuration = 3.0;
      const blendZoneDuration = 0.5; // 0.5 second blend after Phase 1

      if (this.rotationTransitionTime < initialTransitionDuration) {
        // Smoothly rotate from initial direction to path tangent
        const transitionProgress =
          this.rotationTransitionTime / initialTransitionDuration; // 0 to 1
        const eased = 1 - Math.pow(1 - transitionProgress, 3); // Ease out cubic

        // Decompose direction vectors into yaw and pitch for controlled rotation
        // Initial look direction
        const initialYaw = Math.atan2(
          this.initialLookDirection.x,
          this.initialLookDirection.z
        );
        const initialPitch = Math.asin(-this.initialLookDirection.y);

        // Use raw tangent for Phase 1 to avoid smoothing lag
        const tangentYaw = Math.atan2(rawTangent.x, rawTangent.z);
        const tangentPitch = Math.asin(-rawTangent.y);

        // Interpolate yaw and pitch separately
        // Handle yaw wrap-around (shortest rotation path)
        let yawDiff = tangentYaw - initialYaw;
        if (yawDiff > Math.PI) yawDiff -= 2 * Math.PI;
        if (yawDiff < -Math.PI) yawDiff += 2 * Math.PI;

        const interpolatedYaw = initialYaw + yawDiff * eased;
        const interpolatedPitch =
          initialPitch + (tangentPitch - initialPitch) * eased;

        // Reconstruct look direction from interpolated angles
        lookDirection = new THREE.Vector3(
          Math.sin(interpolatedYaw) * Math.cos(interpolatedPitch),
          -Math.sin(interpolatedPitch),
          Math.cos(interpolatedYaw) * Math.cos(interpolatedPitch)
        ).normalize();
      }
      // Blend zone: Smooth transition from Phase 1 to Phase 2
      else if (
        this.rotationTransitionTime <
        initialTransitionDuration + blendZoneDuration
      ) {
        const blendProgress =
          (this.rotationTransitionTime - initialTransitionDuration) /
          blendZoneDuration;
        const blendEased = blendProgress * blendProgress; // Ease in quadratic

        // Calculate the Phase 1 final direction (at t=1.0)
        const initialYaw = Math.atan2(
          this.initialLookDirection.x,
          this.initialLookDirection.z
        );
        const initialPitch = Math.asin(-this.initialLookDirection.y);
        const tangentYaw = Math.atan2(tangent.x, tangent.z);
        const tangentPitch = Math.asin(-tangent.y);

        let yawDiff = tangentYaw - initialYaw;
        if (yawDiff > Math.PI) yawDiff -= 2 * Math.PI;
        if (yawDiff < -Math.PI) yawDiff += 2 * Math.PI;

        const phase1FinalYaw = initialYaw + yawDiff;
        const phase1FinalPitch = tangentPitch;

        const phase1Final = new THREE.Vector3(
          Math.sin(phase1FinalYaw) * Math.cos(phase1FinalPitch),
          -Math.sin(phase1FinalPitch),
          Math.cos(phase1FinalYaw) * Math.cos(phase1FinalPitch)
        ).normalize();

        // Blend from Phase 1 final to current tangent
        lookDirection = new THREE.Vector3()
          .lerpVectors(phase1Final, tangent, blendEased)
          .normalize();
      }
      // Phase 2: Follow tangent for most of journey
      else if (t < 0.6) {
        lookDirection = tangent;
      }
      // Phase 3: Rotate from tangent to final spawn orientation (last 40%)
      else {
        const finalRotProgress = (t - 0.6) / 0.4; // 0 to 1 over last 40%
        const eased = 1 - Math.pow(1 - finalRotProgress, 4); // Ease out quartic (slower)

        // Decompose direction vectors into yaw and pitch for controlled rotation
        // Current tangent direction
        const tangentYaw = Math.atan2(tangent.x, tangent.z);
        const tangentPitch = Math.asin(-tangent.y);

        // Final spawn direction
        const finalYaw = Math.atan2(finalLookDirection.x, finalLookDirection.z);
        const finalPitch = Math.asin(-finalLookDirection.y);

        // Interpolate yaw and pitch separately
        // Handle yaw wrap-around (go the long way around)
        let yawDiff = finalYaw - tangentYaw;
        if (yawDiff > Math.PI) yawDiff -= 2 * Math.PI;
        if (yawDiff < -Math.PI) yawDiff += 2 * Math.PI;

        // Invert to rotate the opposite direction (the long way)
        yawDiff = yawDiff > 0 ? yawDiff - 2 * Math.PI : yawDiff + 2 * Math.PI;

        const interpolatedYaw = tangentYaw + yawDiff * eased;
        const interpolatedPitch =
          tangentPitch + (finalPitch - tangentPitch) * eased;

        // Reconstruct look direction from interpolated angles
        lookDirection = new THREE.Vector3(
          Math.sin(interpolatedYaw) * Math.cos(interpolatedPitch),
          -Math.sin(interpolatedPitch),
          Math.cos(interpolatedYaw) * Math.cos(interpolatedPitch)
        ).normalize();
      }

      // Look in the calculated direction
      const lookTarget = new THREE.Vector3();
      lookTarget.copy(position).add(lookDirection.multiplyScalar(5));
      this.camera.lookAt(lookTarget);

      // Apply camera tilt/roll for unsteady flight effect
      // Lerp tilt to 0 during Phase 3 (final rotation to player view)
      let tiltMultiplier = 1.0;
      if (t >= 0.6) {
        // Reduce tilt during final 40% of journey
        const fadeProgress = (t - 0.6) / 0.4; // 0 to 1 over last 40%
        tiltMultiplier = 1.0 - fadeProgress; // 1.0 to 0
      }

      const rollAngle = this.calculateCameraTilt() * tiltMultiplier;
      const rollQuat = new THREE.Quaternion();
      rollQuat.setFromAxisAngle(lookDirection, rollAngle);
      this.camera.quaternion.multiply(rollQuat);

      return true;
    }

    // Pre-start: follow original GLB animation
    if (!this.hasStarted) {
      if (this.coneAnimatedMesh) {
        // Get the world position and rotation of the animated cone
        const worldPosition = new THREE.Vector3();
        const worldQuaternion = new THREE.Quaternion();
        const worldScale = new THREE.Vector3();

        this.coneAnimatedMesh.getWorldPosition(worldPosition);
        this.coneAnimatedMesh.getWorldQuaternion(worldQuaternion);
        this.coneAnimatedMesh.getWorldScale(worldScale);

        // Debug: Log position occasionally
        if (Math.random() < 0.01) {
          console.log("StartScreen: Cone position:", worldPosition.toArray());
          if (this.sceneManager) {
            const isPlaying =
              this.sceneManager.isAnimationPlaying("coneCurve-anim");
            console.log("StartScreen: Animation playing?", isPlaying);
          }
        }

        // Set camera to match the cone's position
        this.camera.position.copy(worldPosition);

        // Calculate the forward direction from the cone's rotation
        // Use the cone's local -Z axis as forward (standard Three.js convention)
        const rawForward = new THREE.Vector3(0, 0, -1);
        rawForward.applyQuaternion(worldQuaternion);

        // Apply camera spin if reversing (180-degree rotation)
        if (this.animationDirection === -1) {
          let spinAngle;

          if (this.cameraSpinProgress < 1.0) {
            // Initial spin - ease the spin progress for smooth rotation
            const eased =
              this.cameraSpinProgress < 0.5
                ? 2 * this.cameraSpinProgress * this.cameraSpinProgress
                : 1 - Math.pow(-2 * this.cameraSpinProgress + 2, 2) / 2;
            spinAngle = eased * Math.PI; // 0 to PI radians
          } else {
            // After spin completes, keep the 180-degree rotation for entire journey
            spinAngle = Math.PI;
          }

          // Rotate forward vector around Y axis
          const spinQuat = new THREE.Quaternion();
          spinQuat.setFromAxisAngle(new THREE.Vector3(0, 1, 0), spinAngle);
          rawForward.applyQuaternion(spinQuat);
        }

        // Initialize smoothed forward if not set
        if (!this.smoothedForward) {
          this.smoothedForward = rawForward.clone();
        }

        // Smooth the forward direction to reduce sudden rotation changes
        // Use a very low smoothing factor (0.04) for extremely gradual changes
        const smoothingFactor = 0.04;
        this.smoothedForward.lerp(rawForward, smoothingFactor);
        this.smoothedForward.normalize();

        // Calculate a look-at target point using smoothed forward
        const lookTarget = new THREE.Vector3();
        lookTarget.copy(worldPosition).add(this.smoothedForward);

        // Use lookAt to orient the camera (this handles all rotation axes correctly)
        this.camera.lookAt(lookTarget);

        // Apply camera tilt/roll for unsteady flight effect
        const rollAngle = this.calculateCameraTilt();
        const rollQuat = new THREE.Quaternion();
        rollQuat.setFromAxisAngle(this.smoothedForward, rollAngle);
        this.camera.quaternion.multiply(rollQuat);

        return true;
      } else {
        // Fallback: Circle animation
        this.circleTime += dt * this.circleSpeed;

        const x =
          this.circleCenter.x + Math.cos(this.circleTime) * this.circleRadius;
        const z =
          this.circleCenter.z + Math.sin(this.circleTime) * this.circleRadius;
        const y = this.circleHeight;

        this.camera.position.set(x, y, z);

        // Calculate forward direction (tangent to circle)
        const forwardX = -Math.sin(this.circleTime);
        const forwardZ = Math.cos(this.circleTime);

        const lookTarget = new THREE.Vector3(x + forwardX, y, z + forwardZ);
        this.camera.lookAt(lookTarget);

        // Apply camera tilt/roll for unsteady flight effect
        const rollAngle = this.calculateCameraTilt();
        const forward = new THREE.Vector3(forwardX, 0, forwardZ).normalize();
        const rollQuat = new THREE.Quaternion();
        rollQuat.setFromAxisAngle(forward, rollAngle);
        this.camera.quaternion.multiply(rollQuat);

        return true;
      }
    }

    return true;
  }

  /**
   * Check if intro is complete
   */
  isComplete() {
    return !this.isActive;
  }

  /**
   * Clean up resources
   */
  cleanup() {
    if (this.overlay && this.overlay.parentNode) {
      this.overlay.parentNode.removeChild(this.overlay);
    }

    // Hide or remove the camera curve object
    if (this.coneCurveObject && this.sceneManager) {
      this.scene.remove(this.coneCurveObject);
      console.log("StartScreen: Removed camera curve object");
    }
  }

  /**
   * Monitor start screen for start button click and trigger title sequence
   */
  checkIntroStart(sfxManager, gameManager) {
    if (!this.hasStarted) return; // Skip if start button not clicked

    if (this.hasStarted && !this.introStartTriggered) {
      this.introStartTriggered = true;

      // Ensure ambiance is on and attempt playback on first interaction
      if (sfxManager && !sfxManager.isPlaying("city-ambiance")) {
        sfxManager.play("city-ambiance");
      }

      // Make text visible before starting sequence
      this.title.mesh.visible = true;
      this.byline.mesh.visible = true;

      this.titleSequence = new TitleSequence([this.title, this.byline], {
        introDuration: 4.0,
        staggerDelay: 3.0,
        holdDuration: 4.0,
        outroDuration: 2.0,
        disperseDistance: 5.0,
        onComplete: () => {
          console.log("Title sequence complete");
          gameManager.setState({
            currentState: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
          });
        },
      });

      // Update game state - intro is ending, transitioning to gameplay
      gameManager.setState({
        currentState: GAME_STATES.TITLE_SEQUENCE,
      });
    }
  }

  /**
   * Get the title sequence
   */
  getTitleSequence() {
    return this.titleSequence;
  }

  /**
   * Get the text scene and camera for separate rendering
   */
  getTextRenderInfo() {
    return {
      scene: this.textScene,
      camera: this.textCamera,
    };
  }
}
