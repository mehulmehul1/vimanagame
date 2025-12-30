/**
 * startScreen.js - MAIN MENU AND INTRO SEQUENCE
 * =============================================================================
 *
 * ROLE: Manages the start screen UI, title sequence animation, and transition
 * into gameplay. Handles START button, options access, and audio unlocking.
 *
 * KEY RESPONSIBILITIES:
 * - Display start menu with START/OPTIONS buttons
 * - Run title sequence animation (particle or image-based)
 * - Unlock audio contexts on first interaction
 * - Handle gamepad navigation for menu
 * - Transition to INTRO state on START
 * - Manage animated background elements
 *
 * TITLE SEQUENCE:
 * Two modes: particle-based text or fallback image sequence.
 * Typewriter effect with keystroke sounds.
 *
 * =============================================================================
 */

import * as THREE from "three";
import { Howler } from "howler";
import { unlockAllAudioContexts } from "../vfx/proceduralAudio.js";
import { createParticleText, createParticleImage } from "../vfx/titleText.js";
import { TitleSequence } from "../vfx/titleSequence.js";
import { ImageTitleSequence } from "./imageTitleSequence.js";
import { GAME_STATES } from "../gameData.js";
import { GamepadMenuNavigation } from "./gamepadMenuNavigation.js";
import { Logger } from "../utils/logger.js";
import "../styles/startScreen.css";
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
    this.sfxManager = options.sfxManager || null;
    this.inputManager = options.inputManager || null;

    // Logger for debug messages
    this.logger = new Logger("StartScreen", false);

    // Additional state
    this.introStartTriggered = false;
    this.titleSequence = null;
    this.title = null;
    this.byline = null;
    this.keystrokeIndex = 0; // Track which keystroke sound to play next (0-3)
    this.useImageBackup = true; // Enable image backup system instead of particles

    // Gamepad navigation helper
    this.gamepadNav = new GamepadMenuNavigation({
      inputManager: this.inputManager,
      sfxManager: this.sfxManager,
      onNavigateUp: () => this.navigateMenu(-1),
      onNavigateDown: () => this.navigateMenu(1),
      onConfirm: () => this.confirmMenuSelection(),
    });

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
    this.smoothedTiltAxis = null; // Smoothed tilt axis for unified path tilt

    // Procedural path phase timing
    this.phase3StartT = 0.6; // Start of final turn to player view (as fraction of path), default 60%
    this.phase3StartDirection = null; // Camera direction when Phase 3 starts (captured once)
    this.phase3Started = false; // Track if Phase 3 has started
    this.initialTiltMultiplier = 1.0; // Tilt multiplier when START is clicked (captured to avoid snap)

    // Speed transition tracking (GLB -> unified path)
    this.initialAnimSpeed = 0; // Units per second at START (from GLB anim)
    this.targetPathSpeed = 0; // Units per second to traverse unified path over dialog duration

    // GLB animation start position (0 to 1, where 0 is start, 1 is end)
    this.glbAnimationStartProgress =
      options.glbAnimationStartProgress !== undefined
        ? options.glbAnimationStartProgress
        : 0; // Default to start of animation

    // Camera tilt/roll for unsteady flight effect
    this.tiltTime = Math.random() * 100; // Randomize start time
    this.tiltSpeed1 = 0.3 + Math.random() * 0.2; // Base tilt speed (0.3-0.5)
    this.tiltSpeed2 = 0.5 + Math.random() * 0.3; // Secondary tilt speed (0.5-0.8)
    this.tiltAmount = 0.12; // Max tilt in radians (~4.5 degrees)

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

    // Create title text (particles or image backup)
    if (this.useImageBackup) {
      const { title, byline } = this.createTitleImages();
      this.title = title;
      this.byline = byline;
    } else {
      const { title, byline } = this.createTitleText();
      this.title = title;
      this.byline = byline;
    }

    // Load the camera curve animation
    this.loadCameraAnimation();
  }

  /**
   * Load the camera curve animation from ConeCurve.glb
   */
  async loadCameraAnimation() {
    if (!this.sceneManager) {
      this.logger.warn(
        "No sceneManager provided, using fallback circle animation"
      );
      return;
    }

    this.isLoadingAnimation = true;

    try {
      // Get the coneCurve object (should be loaded by gameManager)
      this.coneCurveObject = this.sceneManager.getObject("coneCurve");

      if (!this.coneCurveObject) {
        this.logger.warn("coneCurve object not loaded yet, will retry");
        // Retry after a short delay
        setTimeout(() => this.loadCameraAnimation(), 100);
        return;
      }

      // Find the "Cone" object within the loaded GLTF
      // Note: It doesn't have to be a mesh, just an Object3D that's being animated
      this.coneCurveObject.traverse((child) => {
        if (child.name === "Cone") {
          this.coneAnimatedMesh = child;
          this.logger.log("Found Cone object, type:", child.type);
        }

        // Hide all meshes - we only need this for camera tracking
        if (child.isMesh) {
          child.visible = false;
        }
      });

      if (!this.coneAnimatedMesh) {
        this.logger.warn(
          "Could not find 'Cone' mesh in ConeCurve.glb, using fallback"
        );
        this.isLoadingAnimation = false;
        return;
      }

      this.logger.log("Camera animation loaded successfully");
      this.logger.log("Found animated mesh:", this.coneAnimatedMesh.name);

      // Manually trigger the animation to play
      if (this.sceneManager) {
        this.logger.log("Manually playing coneCurve-anim");
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

            this.logger.log(
              `Set animation start time to ${startTime.toFixed(2)}s (${(
                this.glbAnimationStartProgress * 100
              ).toFixed(1)}% of ${duration.toFixed(2)}s)`
            );
          }
        }
      }

      this.isLoadingAnimation = false;

      // Enable zone detection now that camera animation is loaded and positioned
      if (
        this.uiManager &&
        this.uiManager.gameManager &&
        this.uiManager.gameManager.zoneManager
      ) {
        this.uiManager.gameManager.zoneManager.enableZoneDetection();
      }
    } catch (error) {
      this.logger.error("Error loading camera animation:", error);
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

    // Create tagline with emblem image
    this.tagline = document.createElement("div");
    this.tagline.className = "intro-tagline";
    this.tagline.innerHTML = `<img src="/images/CliffCole_Emblem.svg" alt="From the Files of Confidential" />`;

    // Create start button
    this.startButton = document.createElement("button");
    this.startButton.className = "intro-button";
    this.startButton.textContent = "START";

    // Click handler for start button
    this.startButton.addEventListener("click", (e) => {
      // Audio/video unlocking is handled by global unlockAudioOnInteraction handler
      // (in capture phase, so it fires before this handler even if stopPropagation is called)
      // No need to unlock here - it's already done by the global handler

      e.stopPropagation(); // Prevent click from reaching canvas

      // Play typewriter return sound on click
      if (this.sfxManager) {
        this.sfxManager.play("typewriter-return");
      }

      this.startGame();

      // Request pointer lock when game starts (input will be disabled until control is enabled)
      // Don't request on mobile devices
      const isMobile =
        this.uiManager?.gameManager?.getState?.()?.isMobile || false;
      if (!isMobile) {
        const canvas = document.querySelector("canvas");
        if (canvas && canvas.requestPointerLock) {
          canvas.requestPointerLock();
        }
      }
    });

    // Mouse hover handler for start button
    this.startButton.addEventListener("mouseenter", () => {
      this.selectedButtonIndex = 0;
      this.updateButtonSelection();

      // Play typewriter keystroke sound on hover
      if (this.sfxManager) {
        const soundId = `typewriter-keystroke-0${this.keystrokeIndex}`;
        this.sfxManager.play(soundId);
        // Cycle through 0-3
        this.keystrokeIndex = (this.keystrokeIndex + 1) % 4;
      }
    });

    // Create options button
    this.optionsButton = document.createElement("button");
    this.optionsButton.className = "intro-button";
    this.optionsButton.textContent = "OPTIONS";

    // Click handler for options button
    this.optionsButton.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent click from reaching canvas

      // Play typewriter return sound on click
      if (this.sfxManager) {
        this.sfxManager.play("typewriter-return");
      }

      if (this.uiManager) {
        this.uiManager.show("options-menu");
      }
    });

    // Mouse hover handler for options button
    this.optionsButton.addEventListener("mouseenter", () => {
      this.selectedButtonIndex = 1;
      this.updateButtonSelection();

      // Play typewriter keystroke sound on hover
      if (this.sfxManager) {
        const soundId = `typewriter-keystroke-0${this.keystrokeIndex}`;
        this.sfxManager.play(soundId);
        // Cycle through 0-3
        this.keystrokeIndex = (this.keystrokeIndex + 1) % 4;
      }
    });

    this.overlay.appendChild(this.tagline);
    this.overlay.appendChild(this.startButton);
    this.overlay.appendChild(this.optionsButton);
    document.body.appendChild(this.overlay);

    // Start with transparent content and fade in over 1 second
    this.overlay.style.opacity = "0";
    this.overlay.style.transition = "opacity 1s ease-in";

    // Trigger fade-in after a brief delay to ensure DOM is ready
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        this.overlay.style.opacity = "1";
      });
    });

    // Setup menu navigation
    this.menuButtons = [this.startButton, this.optionsButton];
    this.selectedButtonIndex = 0; // Start button is selected by default
    this.updateButtonSelection();

    // Add keyboard navigation
    this.keydownHandler = (e) => {
      if (!this.overlay || this.overlay.style.display === "none") return;

      if (e.key === "ArrowDown" || e.key === "s" || e.key === "S") {
        e.preventDefault();
        this.selectedButtonIndex =
          (this.selectedButtonIndex + 1) % this.menuButtons.length;
        this.updateButtonSelection();

        // Play typewriter keystroke sound
        if (this.sfxManager) {
          const soundId = `typewriter-keystroke-0${this.keystrokeIndex}`;
          this.sfxManager.play(soundId);
          // Cycle through 0-3
          this.keystrokeIndex = (this.keystrokeIndex + 1) % 4;
        }
      } else if (e.key === "ArrowUp" || e.key === "w" || e.key === "W") {
        e.preventDefault();
        this.selectedButtonIndex =
          (this.selectedButtonIndex - 1 + this.menuButtons.length) %
          this.menuButtons.length;
        this.updateButtonSelection();

        // Play typewriter keystroke sound
        if (this.sfxManager) {
          const soundId = `typewriter-keystroke-0${this.keystrokeIndex}`;
          this.sfxManager.play(soundId);
          // Cycle through 0-3
          this.keystrokeIndex = (this.keystrokeIndex + 1) % 4;
        }
      } else if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();

        // Play typewriter return sound
        if (this.sfxManager) {
          this.sfxManager.play("typewriter-return");
        }

        this.menuButtons[this.selectedButtonIndex].click();
      }
    };
    document.addEventListener("keydown", this.keydownHandler);

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
   * Update button selection visual state
   */
  updateButtonSelection() {
    this.menuButtons.forEach((button, index) => {
      if (index === this.selectedButtonIndex) {
        button.classList.add("selected");
      } else {
        button.classList.remove("selected");
      }
    });
  }

  /**
   * Create simple HTML image elements for title backup system
   */
  createTitleImages() {
    // Create container for title images
    this.titleImageContainer = document.createElement("div");
    this.titleImageContainer.id = "title-images";
    this.titleImageContainer.className = "title-images-container";
    document.body.appendChild(this.titleImageContainer);

    // Create title image (50% screen width)
    const titleImg = document.createElement("img");
    titleImg.src = "/images/Czar_MainTitle.png";
    titleImg.className = "title-image";
    titleImg.style.width = "50%";
    titleImg.style.opacity = "0";
    this.titleImageContainer.appendChild(titleImg);

    // Create byline image (30% screen width)
    const bylineImg = document.createElement("img");
    bylineImg.src = "/images/JamesCKane.png";
    bylineImg.className = "byline-image";
    bylineImg.style.width = "30%";
    bylineImg.style.opacity = "0";
    this.titleImageContainer.appendChild(bylineImg);

    const title = {
      element: titleImg,
      fadeIn: false,
      fadeOut: false,
      opacity: 0,
    };

    const byline = {
      element: bylineImg,
      fadeIn: false,
      fadeOut: false,
      opacity: 0,
    };

    return { title, byline };
  }

  /**
   * Create title text particles for the start screen
   */
  createTitleText() {
    // Check if mobile device
    const isMobile =
      this.uiManager?.gameManager?.getState?.()?.isMobile || false;

    //
    // Adjust z positions for mobile (move further from camera)
    const titleZ = isMobile ? -3.5 : -2.25;
    const bylineZ = isMobile ? -3.5 : -2.25;
    // Reduce particle count on mobile for better performance
    const titleParticleCount = isMobile ? 3000 : 6000;
    const bylineParticleCount = isMobile ? 600 : 1200;

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

    // Create first title as image-based particles from PNG with alpha masking
    const imageData1 = createParticleImage(this.textScene, {
      imageUrl: "/images/Czar_MainTitle.png",
      position: { x: 0, y: 0, z: titleZ },
      scale: 0.03125,
      animate: true,
      maxParticles: titleParticleCount,
      alphaThreshold: 0.1, // keep semi-opaque pixels, discard near-fully transparent
    });
    this.textScene.remove(imageData1.mesh);
    this.textCamera.add(imageData1.mesh);
    this.textScene.add(this.textCamera);
    imageData1.mesh.userData.baseScale = 0.0125;
    imageData1.mesh.visible = false; // Hide initially

    // Create a wrapper object that includes particles
    const title = {
      mesh: imageData1.mesh,
      particles: imageData1.particles,
      update: imageData1.update,
      pointSize: 0.28, // Balanced for definition without being blown out
    };

    // Create byline as image-based particles below title
    const imageData2 = createParticleImage(this.textScene, {
      imageUrl: "/images/JamesCKane.png",
      position: { x: 0, y: -0.8, z: bylineZ },
      scale: 0.0225,
      animate: true,
      maxParticles: bylineParticleCount,
      alphaThreshold: 0.1,
    });
    this.textScene.remove(imageData2.mesh);
    this.textCamera.add(imageData2.mesh);
    imageData2.mesh.userData.baseScale = 0.0125;
    imageData2.mesh.visible = false; // Hide initially

    // Create a wrapper object that includes particles
    const byline = {
      mesh: imageData2.mesh,
      particles: imageData2.particles,
      update: imageData2.update,
      pointSize: 0.28, // Balanced for definition without being blown out
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

        this.logger.log(
          `Creating unified path from ${(progress * 100).toFixed(
            1
          )}% of animation`
        );

        // Determine which end of animation to head to
        const goingToStart = progress < 0.5;
        const targetTime = goingToStart ? 0 : duration;

        this.animationDirection = goingToStart ? -1 : 1;
        this.isSpinning = goingToStart;
        this.cameraSpinProgress = 0;

        // Estimate current speed on the GLB animation path (sample two close frames)
        const speedSampleDt = 0.016; // ~16ms
        const time1 = Math.max(0, currentTime - speedSampleDt);
        const time2 = currentTime;

        action.time = time1;
        const mixer = this.sceneManager.animationMixers.get("coneCurve");
        if (mixer) mixer.update(0);
        const pos1 = new THREE.Vector3();
        this.coneAnimatedMesh.getWorldPosition(pos1);

        action.time = time2;
        if (mixer) mixer.update(0);
        const pos2 = new THREE.Vector3();
        this.coneAnimatedMesh.getWorldPosition(pos2);

        const distanceTraveled = pos1.distanceTo(pos2);
        const actualTimeDiff = time2 - time1;
        const rawSpeed =
          actualTimeDiff > 0 ? distanceTraveled / actualTimeDiff : 0;
        const timeScale = action.timeScale || 1.0; // account for slow playback
        this.initialAnimSpeed = rawSpeed * timeScale;

        this.logger.log(
          `GLB speed ≈ ${this.initialAnimSpeed.toFixed(
            2
          )} u/s (raw ${rawSpeed.toFixed(2)}, ts ${timeScale})`
        );

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
        // Default fallback: sum of caption durations (2.0 + 3.5 + 2.5 + 2.5 + 2.0 + 2.0 + 2.5 = 17.0)
        let dialogDuration = 17.0;
        if (this.dialogManager) {
          const introDuration = this.dialogManager.getDialogDuration("intro");
          if (introDuration > 0) {
            dialogDuration = introDuration;
            this.logger.log(`Using intro dialog duration: ${dialogDuration}s`);
          } else {
            // getDialogDuration() now handles fallback to caption duration
            // for iOS prefetched audio, but if it still returns null, use caption-based fallback
            this.logger.warn(
              `Could not get intro dialog duration from audio, using caption-based fallback: ${dialogDuration}s`
            );
          }
        }

        this.unifiedPathDuration = dialogDuration; // Total time for entire journey

        // Compute target path speed so we can ease from GLB speed to this speed
        const pathLength = this.unifiedPath.getLength();
        this.targetPathSpeed =
          pathLength > 0 ? pathLength / dialogDuration : 1.0;
        this.logger.log(
          `Target path speed ≈ ${this.targetPathSpeed.toFixed(
            2
          )} u/s (length ${pathLength.toFixed(2)}, dur ${dialogDuration.toFixed(
            2
          )})`
        );
        this.isFollowingUnifiedPath = true;

        // Reset Phase 3 tracking for new path
        this.phase3Started = false;
        this.phase3StartDirection = null;

        // Capture current camera look direction
        const currentLook = new THREE.Vector3(0, 0, -1);
        currentLook.applyQuaternion(this.camera.quaternion);
        this.initialLookDirection = currentLook.clone();

        // Capture current tilt multiplier to avoid snapping when transitioning to unified path
        // Calculate based on current spin state (same logic as pre-start phase)
        if (this.isSpinning && this.cameraSpinProgress < 1.0) {
          // Fade in tilt as spin progresses (ease in cubic)
          this.initialTiltMultiplier =
            this.cameraSpinProgress *
            this.cameraSpinProgress *
            this.cameraSpinProgress;
        } else {
          // Full tilt if not spinning
          this.initialTiltMultiplier = 1.0;
        }

        // Get initial tangent of the unified path
        this.pathInitialTangent = this.unifiedPath.getTangentAt(0).normalize();

        // Reset rotation transition time and tangent smoothing
        this.rotationTransitionTime = 0;
        this.previousTangent = this.pathInitialTangent.clone();
        // Initialize smoothed tangent to current look direction to match Phase 1 start
        this.smoothedTangent = currentLook.clone();

        // Set when Phase 3 (turn to player view) begins based on GLB progress at START
        // Farther from origin (closer to 0.5) → later turn (80%); closer to origin → earlier (50%)
        // Map distanceFromOrigin in [0..0.5] to phase start in [0.5..0.8]
        const distanceFromOrigin = goingToStart ? progress : 1.0 - progress; // 0..0.5
        const tMin = 0.5;
        const tMax = 0.75;
        const normalized = Math.min(0.5, Math.max(0, distanceFromOrigin)) / 0.5; // 0..1
        this.phase3StartT = tMin + (tMax - tMin) * normalized;

        // Set spin duration if reversing
        if (this.isSpinning) {
          this.cameraSpinDuration = 1.0;
        }

        // Pause the original animation
        action.paused = true;

        // Flip high-level state from startScreen -> titleSequence
        if (this.uiManager && this.uiManager.gameManager) {
          this.uiManager.gameManager.setState({
            currentState: GAME_STATES.INTRO,
          });
        }

        this.logger.log(
          `Created unified path with ${pathPoints.length} points over ${this.unifiedPathDuration}s`
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
   * Navigate menu in specified direction
   * @param {number} direction - -1 for up, 1 for down
   */
  navigateMenu(direction) {
    this.selectedButtonIndex =
      (this.selectedButtonIndex + direction + this.menuButtons.length) %
      this.menuButtons.length;
    this.updateButtonSelection();
  }

  /**
   * Confirm current menu selection
   */
  confirmMenuSelection() {
    if (this.menuButtons && this.menuButtons[this.selectedButtonIndex]) {
      this.menuButtons[this.selectedButtonIndex].click();
    }
  }

  /**
   * Update camera position for circling or unified path
   * @param {number} dt - Delta time in seconds
   * @returns {boolean} - True if still active, false if complete
   */
  update(dt) {
    // Handle gamepad menu navigation (only when overlay is visible)
    if (
      !this.hasStarted &&
      this.overlay &&
      this.overlay.style.display !== "none"
    ) {
      this.gamepadNav.update(dt);
    }

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
      // Advance along the path using speed easing from GLB speed -> target path speed
      const speedRampDuration = 1.5; // seconds to complete acceleration blend
      const rampT = Math.min(
        1.0,
        this.rotationTransitionTime / speedRampDuration
      );
      const rampEased = rampT * rampT; // ease-in quadratic
      const currentSpeed = THREE.MathUtils.lerp(
        this.initialAnimSpeed || 0,
        this.targetPathSpeed || 1.0,
        rampEased
      );
      const pathLength = this.unifiedPath.getLength();
      const progressDelta =
        pathLength > 0
          ? (currentSpeed * dt) / pathLength
          : dt / this.unifiedPathDuration;
      this.unifiedPathProgress += progressDelta;
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
        // Smoothly rotate from initial direction to smoothed tangent (look-ahead behavior)
        const transitionProgress =
          this.rotationTransitionTime / initialTransitionDuration; // 0 to 1
        const eased = 1 - Math.pow(1 - transitionProgress, 3); // Ease out cubic

        // Use spherical interpolation (slerp) for perfectly smooth rotation
        // smoothedTangent is already smoothly updated, giving natural look-ahead behavior
        const targetDirection = this.smoothedTangent.clone().normalize();
        const startDirection = this.initialLookDirection.clone().normalize();

        // Ensure vectors are normalized
        if (startDirection.lengthSq() < 0.001) {
          startDirection.set(0, 0, -1);
        }
        if (targetDirection.lengthSq() < 0.001) {
          targetDirection.copy(startDirection);
        }

        // Use quaternion-based spherical linear interpolation for smooth rotation
        // Convert direction vectors to quaternions, slerp, then convert back
        const startQuat = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 0, -1),
          startDirection
        );
        const targetQuat = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 0, -1),
          targetDirection
        );
        const slerpedQuat = new THREE.Quaternion().slerpQuaternions(
          startQuat,
          targetQuat,
          eased
        );
        lookDirection = new THREE.Vector3(0, 0, -1)
          .applyQuaternion(slerpedQuat)
          .normalize();
      }
      // Blend zone: Smooth transition from Phase 1 to Phase 2
      else if (
        this.rotationTransitionTime <
        initialTransitionDuration + blendZoneDuration
      ) {
        // Phase 1 now ends at smoothedTangent, which matches tangent
        // Use tangent directly for seamless continuation
        lookDirection = tangent.clone();
      }
      // Phase 2: Follow tangent for most of journey
      else if (t < this.phase3StartT) {
        lookDirection = tangent;
      }
      // Phase 3: Rotate from tangent to final spawn orientation (final portion)
      else {
        // Capture starting direction when Phase 3 begins (once, for stability)
        if (!this.phase3Started) {
          this.phase3StartDirection = tangent.clone().normalize();
          this.phase3Started = true;
        }

        const remaining = Math.max(0.001, 1.0 - this.phase3StartT);
        const finalRotProgress = (t - this.phase3StartT) / remaining; // 0..1 over last portion
        const eased = 1 - Math.pow(1 - finalRotProgress, 4); // Ease out quartic (slower)

        // Use quaternion-based spherical interpolation for smooth, consistent rotation
        // This avoids angle wrap-around issues and ensures correct rotation direction
        const startDirection = this.phase3StartDirection.clone().normalize();
        const targetDirection = finalLookDirection.clone().normalize();

        // Ensure vectors are normalized
        if (startDirection.lengthSq() < 0.001) {
          startDirection.set(0, 0, -1);
        }
        if (targetDirection.lengthSq() < 0.001) {
          targetDirection.set(0, 0, -1);
        }

        // Use quaternion-based spherical linear interpolation
        // Ensure rotation always goes toward -X, never through +X
        const startQuat = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 0, -1),
          startDirection
        );
        const targetQuat = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 0, -1),
          targetDirection
        );

        // Safety check: if quaternions are invalid, fall back to simple lerp
        const startQuatValid = !isNaN(startQuat.w) && isFinite(startQuat.w);
        const targetQuatValid = !isNaN(targetQuat.w) && isFinite(targetQuat.w);

        if (!startQuatValid || !targetQuatValid) {
          // Fall back to simple vector lerp if quaternion creation failed
          lookDirection = new THREE.Vector3()
            .lerpVectors(startDirection, targetDirection, eased)
            .normalize();
          return true;
        }

        // Check if shortest path would pass through +X by sampling intermediate directions
        // We want to ensure rotation always stays toward -X (x should stay negative or near zero)
        let wouldPassThroughPositiveX = false;
        for (let i = 0.1; i < 0.9; i += 0.1) {
          const testQuat = new THREE.Quaternion().slerpQuaternions(
            startQuat,
            targetQuat,
            i
          );
          const testDir = new THREE.Vector3(0, 0, -1)
            .applyQuaternion(testQuat)
            .normalize();
          // If X becomes positive during rotation, we need to go the other way
          if (testDir.x > 0.01) {
            wouldPassThroughPositiveX = true;
            break;
          }
        }

        // If shortest path goes through +X, calculate rotation that goes the other way
        // We'll create a quaternion that represents the same target rotation but via the long path
        let finalTargetQuat = targetQuat;
        if (wouldPassThroughPositiveX) {
          // Check if start and target are too close (near-identity rotation)
          const dot = startDirection.dot(targetDirection);
          if (Math.abs(dot) > 0.999) {
            // Vectors are nearly parallel, skip long path calculation
            finalTargetQuat = targetQuat;
          } else {
            // Calculate relative rotation from start to target: target = start * relative
            // So relative = start.invert() * target
            const startQuatInv = startQuat.clone().invert();
            const relativeQuat = new THREE.Quaternion().multiplyQuaternions(
              startQuatInv,
              targetQuat
            );

            // Extract axis and angle manually from quaternion
            // For quaternion q = (x, y, z, w): angle = 2*acos(|w|), axis = (x,y,z)/sin(angle/2)
            // Normalize quaternion first to ensure accuracy
            const normalizedQuat = relativeQuat.clone().normalize();
            const w = Math.min(1, Math.max(-1, Math.abs(normalizedQuat.w)));
            const angle = 2 * Math.acos(w);

            // Safety check: if angle is invalid or too small, fall back to regular path
            const isValidAngle =
              !isNaN(angle) &&
              isFinite(angle) &&
              angle > 0.001 &&
              angle < Math.PI * 1.9;

            if (isValidAngle) {
              // Calculate axis: (x, y, z) / sin(angle/2)
              const sinHalfAngle = Math.sin(angle / 2);
              if (sinHalfAngle > 0.001) {
                const axis = new THREE.Vector3(
                  normalizedQuat.x / sinHalfAngle,
                  normalizedQuat.y / sinHalfAngle,
                  normalizedQuat.z / sinHalfAngle
                ).normalize();

                // Validate axis
                const isValidAxis =
                  axis.lengthSq() > 0.001 &&
                  !isNaN(axis.x) &&
                  !isNaN(axis.y) &&
                  !isNaN(axis.z) &&
                  isFinite(axis.x) &&
                  isFinite(axis.y) &&
                  isFinite(axis.z);

                if (isValidAxis) {
                  // Try both rotation directions and pick the one that avoids +X
                  // Option 1: Shortest path (already tested, goes through +X)
                  // Option 2: Long way - negate axis and use complementary angle
                  const reversedAxis = axis.clone().negate();
                  const longWayAngle = 2 * Math.PI - angle;
                  const longWayQuat = new THREE.Quaternion().setFromAxisAngle(
                    reversedAxis,
                    longWayAngle
                  );
                  const longWayTargetQuat =
                    new THREE.Quaternion().multiplyQuaternions(
                      startQuat,
                      longWayQuat
                    );

                  // Test if long way avoids +X (should always stay toward -X)
                  let longWayAvoidsPositiveX = true;
                  for (let i = 0.1; i < 0.9; i += 0.1) {
                    const testQuat = new THREE.Quaternion().slerpQuaternions(
                      startQuat,
                      longWayTargetQuat,
                      i
                    );
                    const testDir = new THREE.Vector3(0, 0, -1)
                      .applyQuaternion(testQuat)
                      .normalize();
                    if (testDir.x > 0.01) {
                      longWayAvoidsPositiveX = false;
                      break;
                    }
                  }

                  // Use long way if it avoids +X, otherwise fall back to regular (shouldn't happen)
                  if (longWayAvoidsPositiveX) {
                    finalTargetQuat = longWayTargetQuat;
                  }
                }
              }
            }
            // If invalid, fall back to regular targetQuat (shouldn't happen, but safety net)
          }
        }

        // Final safety check: ensure finalTargetQuat is valid before slerping
        const finalQuatValid =
          !isNaN(finalTargetQuat.w) &&
          isFinite(finalTargetQuat.w) &&
          !isNaN(finalTargetQuat.x) &&
          !isNaN(finalTargetQuat.y) &&
          !isNaN(finalTargetQuat.z);
        if (!finalQuatValid) {
          // Fall back to simple lerp if quaternion is invalid
          lookDirection = new THREE.Vector3()
            .lerpVectors(startDirection, targetDirection, eased)
            .normalize();
          return true;
        }

        const slerpedQuat = new THREE.Quaternion().slerpQuaternions(
          startQuat,
          finalTargetQuat,
          eased
        );
        lookDirection = new THREE.Vector3(0, 0, -1)
          .applyQuaternion(slerpedQuat)
          .normalize();

        // Safety check: if lookDirection is invalid, fall back to lerping between start and target
        if (
          !lookDirection ||
          lookDirection.lengthSq() < 0.001 ||
          isNaN(lookDirection.x) ||
          isNaN(lookDirection.y) ||
          isNaN(lookDirection.z)
        ) {
          lookDirection = new THREE.Vector3()
            .lerpVectors(startDirection, targetDirection, eased)
            .normalize();
        }
      }

      // Look in the calculated direction
      const lookTarget = new THREE.Vector3();
      lookTarget.copy(position).add(lookDirection.multiplyScalar(5));
      this.camera.lookAt(lookTarget);

      // Apply camera tilt/roll for unsteady flight effect (smoothed axis to avoid overdramatic tilt)
      if (!this.smoothedTiltAxis) {
        this.smoothedTiltAxis = lookDirection.clone();
      }
      const tiltAxisSmoothing = 0.08; // slightly faster than pre-start 0.04 to stay responsive
      this.smoothedTiltAxis.lerp(lookDirection, tiltAxisSmoothing);
      this.smoothedTiltAxis.normalize();
      // Smoothly fade tilt in during Phase 1, keep at full during Phase 2, fade out during Phase 3
      let tiltMultiplier = 1.0;

      // Fade in during Phase 1 (smooth transition from initial tilt to full tilt)
      if (this.rotationTransitionTime < initialTransitionDuration) {
        const fadeInProgress =
          this.rotationTransitionTime / initialTransitionDuration; // 0 to 1
        // Ease in cubic for smooth fade-in, lerp from initial tilt to 1.0
        const eased = fadeInProgress * fadeInProgress * fadeInProgress;
        tiltMultiplier = THREE.MathUtils.lerp(
          this.initialTiltMultiplier,
          1.0,
          eased
        );
      }
      // Full tilt during Phase 2 (already 1.0)
      // Fade out during Phase 3
      else if (t >= this.phase3StartT) {
        const remaining = Math.max(0.001, 1.0 - this.phase3StartT);
        const fadeProgress = (t - this.phase3StartT) / remaining; // 0..1
        tiltMultiplier = 1.0 - fadeProgress; // 1.0 to 0
      }

      const rollAngle = this.calculateCameraTilt() * tiltMultiplier;
      const rollQuat = new THREE.Quaternion();
      rollQuat.setFromAxisAngle(this.smoothedTiltAxis, rollAngle);
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
          this.logger.log("Cone position:", worldPosition.toArray());
          if (this.sceneManager) {
            const isPlaying =
              this.sceneManager.isAnimationPlaying("coneCurve-anim");
            this.logger.log("Animation playing?", isPlaying);
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
          // Increment spin progress
          if (this.cameraSpinProgress < 1.0) {
            this.cameraSpinProgress += dt / this.cameraSpinDuration;
            this.cameraSpinProgress = Math.min(1.0, this.cameraSpinProgress);
          }

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
        // Smoothly fade tilt in during spin to avoid competing rotations
        let tiltMultiplier = 1.0;
        const isSpinning =
          this.animationDirection === -1 && this.cameraSpinProgress < 1.0;
        if (isSpinning) {
          // Fade in tilt as spin progresses (ease in cubic)
          tiltMultiplier =
            this.cameraSpinProgress *
            this.cameraSpinProgress *
            this.cameraSpinProgress;
        }

        const rollAngle = this.calculateCameraTilt() * tiltMultiplier;
        const rollQuat = new THREE.Quaternion();
        rollQuat.setFromAxisAngle(this.smoothedForward, rollAngle);
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

    // Remove keyboard navigation event listener
    if (this.keydownHandler) {
      document.removeEventListener("keydown", this.keydownHandler);
      this.keydownHandler = null;
    }

    // Hide or remove the camera curve object
    if (this.coneCurveObject && this.sceneManager) {
      this.scene.remove(this.coneCurveObject);
      this.logger.log("Removed camera curve object");
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

      // Defer title sequence until INTRO_COMPLETE game state
      const startTitleSequence = () => {
        if (this.useImageBackup) {
          // Use simple image fade-in sequence
          this.titleSequence = new ImageTitleSequence(
            [this.title, this.byline],
            {
              introDuration: 1.0,
              staggerDelay: 3.0,
              holdDuration: 2.0,
              outroDuration: 1.0,
              onComplete: () => {
                this.logger.log("Title sequence complete");
                // Clean up image elements
                if (
                  this.titleImageContainer &&
                  this.titleImageContainer.parentNode
                ) {
                  this.titleImageContainer.parentNode.removeChild(
                    this.titleImageContainer
                  );
                  this.titleImageContainer = null;
                }
                this.titleSequence = null;
                gameManager.setState({
                  currentState: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
                });
              },
            }
          );
        } else {
          // Create sequence first so we can prime particle buffers before showing
          this.titleSequence = new TitleSequence([this.title, this.byline], {
            introDuration: 3.0,
            staggerDelay: 2.0,
            holdDuration: 3.0,
            outroDuration: 2.0,
            disperseDistance: 5.0,
            basePointSize: 0.56, // Doubled from 0.28 to compensate for reduced density
            onComplete: () => {
              this.logger.log("Title sequence complete");
              // Clean up particles from scene to free memory and stop updates
              if (this.title && this.title.mesh) {
                if (this.title.mesh.parent) {
                  this.title.mesh.parent.remove(this.title.mesh);
                }
                this.title.mesh.geometry.dispose();
                this.title.mesh.material.dispose();
                this.title.mesh = null;
              }
              if (this.byline && this.byline.mesh) {
                if (this.byline.mesh.parent) {
                  this.byline.mesh.parent.remove(this.byline.mesh);
                }
                this.byline.mesh.geometry.dispose();
                this.byline.mesh.material.dispose();
                this.byline.mesh = null;
              }
              this.titleSequence = null; // Clear reference to stop updates
              gameManager.setState({
                currentState: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
              });
            },
          });

          // Prime one update tick to ensure initial opacities/positions are set (avoids flash)
          if (
            this.titleSequence &&
            typeof this.titleSequence.update === "function"
          ) {
            this.titleSequence.update(0);
          }

          // Now make text visible
          this.title.mesh.visible = true;
          this.byline.mesh.visible = true;
        }

        // Update game state - intro is ending, transitioning to gameplay
        gameManager.setState({
          currentState: GAME_STATES.TITLE_SEQUENCE,
        });
      };

      // Drive by game state: wait for TITLE_SEQUENCE then trigger
      const gm = this.uiManager && this.uiManager.gameManager;
      if (gm && typeof gm.on === "function") {
        const onStateChanged = (newState, oldState) => {
          if (
            newState &&
            newState.currentState === GAME_STATES.TITLE_SEQUENCE
          ) {
            if (typeof gm.off === "function") {
              gm.off("state:changed", onStateChanged);
            }
            startTitleSequence();
          }
        };
        gm.on("state:changed", onStateChanged);
      } else {
        // Fallback if gameManager not available
        startTitleSequence();
      }
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
