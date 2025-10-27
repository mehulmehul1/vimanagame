import * as THREE from "three";
import { getCameraAnimationsForState } from "./animationCameraData.js";
import { objectAnimations } from "./animationObjectData.js";
import { Logger } from "./utils/logger.js";
import { checkCriteria } from "./utils/criteriaHelper.js";

/**
 * AnimationManager - Manages playback of camera and object animations
 *
 * Features:
 * - Load and play head-pose animations from JSON
 * - State-driven playback with criteria support
 * - Smooth handoff to/from character controller
 * - Local-space playback (applies deltas relative to starting pose)
 * - playOnce tracking per animation
 * - Configurable input restoration
 * - Object animation support (position, rotation, scale, opacity)
 */
class AnimationManager {
  constructor(camera, characterController, gameManager, options = {}) {
    // Debug logging
    this.logger = new Logger("AnimationManager", false);

    this.camera = camera;
    this.characterController = characterController;
    this.gameManager = gameManager;
    this.loadingScreen = options.loadingScreen || null; // For progress tracking
    this.physicsManager = options.physicsManager || null; // For floor height calculations
    this.sceneManager = options.sceneManager || null; // For object animations

    // Playback state
    this.isPlaying = false;
    this.currentAnimation = null;
    this.currentAnimationData = null;
    this.elapsed = 0;
    this.frameIdx = 1;
    this.onComplete = null;
    this.scaleY = 0.8; // Y-axis scale for current animation
    this.playbackRate = 1.0; // Playback speed multiplier for current animation

    // Pre-animation slerp state (to reset pitch to 0 while keeping yaw)
    this.isPreSlerping = false;
    this.preSlerpStartQuat = new THREE.Quaternion();
    this.preSlerpTargetQuat = new THREE.Quaternion(); // Will be set to zero pitch + current yaw
    this.preSlerpElapsed = 0;
    this.preSlerpDuration = 0.3; // 300ms quick slerp to level horizon

    // Deferred animation loading
    this.deferredAnimations = new Map(); // Map of animId -> { id, path, data }

    // Base pose (where animation starts from)
    this.baseQuat = new THREE.Quaternion();
    this.basePos = new THREE.Vector3();

    // Temp objects for interpolation
    this._interpDelta = new THREE.Quaternion();
    this._interpPos = new THREE.Vector3();
    this._rotatedPos = new THREE.Vector3();

    // Animation library
    this.animations = new Map();

    // Tracking for playOnce
    this.playedAnimations = new Set();

    // Track last state to detect changes
    this.lastState = null;

    // Delayed playback support
    this.pendingAnimations = new Map(); // Map of animId -> { animData, timer, delay }

    // Delayed input restoration (for lookat animations with zoom)
    this.pendingInputRestore = null; // { timer: 0, delay: number } or null

    // Lookat sequence state
    this.activeSequence = null; // { animData, currentIndex, isWaitingForNext } or null

    // Object animation state
    this.activeObjectAnimations = new Map(); // Map of animId -> { animData, elapsed, direction, sceneManager } or null

    // Post-animation settle-up to ensure clearance above floor
    this.isSettlingUp = false;
    this.settleStartY = 0;
    this.settleTargetY = 0;
    this.settleElapsed = 0;
    this.settleDuration = 0.3; // seconds
    this._pendingComplete = null;
    this._pendingAnimData = null; // Preserve animData during settling
    this.minCharacterCenterY = 0.9; // minimum body center Y to be above physics floor

    // Fade effect state (screen whiteout/blackout)
    this.isFading = false;
    this.fadeElapsed = 0;
    this.fadeData = null; // Current fade animation data
    this.fadeCube = null; // Mesh for fade effect
    this.fadeOnComplete = null;

    // Event listeners will be set up in initialize()
    this.logger.log(
      "AnimationManager constructed (call initialize() to set up event listeners)"
    );
  }

  /**
   * Initialize event listeners and check initial state
   * Call this AFTER CharacterController has registered its event listeners
   */
  initialize() {
    if (!this.gameManager) {
      this.logger.warn("Cannot initialize, no gameManager");
      return;
    }

    // Listen for state changes
    this.gameManager.on("state:changed", (newState, oldState) => {
      this.logger.log(
        `[AnimationManager] state:changed event - oldState.currentState: ${oldState.currentState}, newState.currentState: ${newState.currentState}`
      );
      this.onStateChanged(newState, oldState);
    });

    // Listen for camera:animation events
    this.gameManager.on("camera:animation", async (data) => {
      const { animation, onComplete } = data;
      this.logger.log(`AnimationManager: Playing animation: ${animation}`);

      // Load animation if not already loaded
      if (!this.getAnimationNames().includes(animation)) {
        const ok = await this.loadAnimation(animation, animation);
        if (!ok) {
          this.logger.warn(`Failed to load animation: ${animation}`);
          if (onComplete) onComplete(false);
          return;
        }
      }

      // Play animation
      this.play(animation, () => {
        if (this.debug) this.logger.log(`Animation complete: ${animation}`);
        if (onComplete) onComplete(true);
      });
    });

    // Check initial state for animations AFTER first render frame
    // This ensures input is fully initialized before any animation tries to disable it
    requestAnimationFrame(() => {
      const initialState = this.gameManager.getState();
      this.logger.log(
        `[AnimationManager] Checking initial state for animations:`,
        initialState
      );
      this.onStateChanged(initialState, {}); // Pass empty oldState so currentState will be different
    });

    this.logger.log("AnimationManager initialized with event listeners");
  }

  /**
   * Load animations from data
   * @param {Object} animationData - Camera animation data object
   * @returns {Promise<void>}
   */
  async loadAnimationsFromData(animationData) {
    const animations = Object.values(animationData);
    // Only load JSON animations (jsonAnimation or animation type with path), skip lookats and moveTos
    const animationsToLoad = animations.filter(
      (anim) =>
        (anim.type === "jsonAnimation" || anim.type === "animation") &&
        anim.path
    );

    // Separate preload and deferred animations
    const preloadAnimations = [];
    const deferredAnimations = [];

    for (const anim of animationsToLoad) {
      const preload = anim.preload !== false; // Default to true
      if (preload) {
        preloadAnimations.push(anim);
      } else {
        deferredAnimations.push(anim);
        this.deferredAnimations.set(anim.id, anim);
        this.logger.log(
          `AnimationManager: Deferred loading for animation "${anim.id}"`
        );
      }
    }

    // Load only preload animations during loading screen
    const loadPromises = preloadAnimations.map(
      (anim) => this.loadAnimation(anim.id, anim.path, true) // Pass preload flag
    );
    await Promise.all(loadPromises);

    const nonJsonCount = animations.length - animationsToLoad.length;
    if (this.debug)
      this.logger.log(
        `Loaded ${preloadAnimations.length} JSON animations from data (${deferredAnimations.length} deferred, ${nonJsonCount} lookats/moveTos)`
      );
  }

  /**
   * Load deferred animations (called after loading screen)
   */
  async loadDeferredAnimations() {
    if (this.debug)
      this.logger.log(
        `Loading ${this.deferredAnimations.size} deferred animations`
      );
    const loadPromises = [];
    for (const [id, anim] of this.deferredAnimations) {
      loadPromises.push(this.loadAnimation(id, anim.path, false));
    }
    await Promise.all(loadPromises);
    this.deferredAnimations.clear();
  }

  /**
   * Handle game state changes
   * @param {Object} newState - New game state
   * @param {Object} oldState - Previous game state
   */
  onStateChanged(newState, oldState = {}) {
    // Check if any state properties have changed
    const stateChanged = Object.keys(newState).some(
      (key) => newState[key] !== oldState[key]
    );

    if (!stateChanged) {
      this.logger.log(
        `[AnimationManager] No state changes detected, skipping animation check`
      );
      return;
    }

    // Log what changed for debugging
    const changedKeys = Object.keys(newState).filter(
      (key) => newState[key] !== oldState[key]
    );
    this.logger.log(
      `[AnimationManager] State changed (${changedKeys.join(
        ", "
      )}), checking for animations...`
    );

    // Get all animations that should play for this state (pass playedAnimations for playOnce filtering)
    const animations = getCameraAnimationsForState(
      newState,
      this.playedAnimations
    );

    // Also check object animations
    const objectAnims = Object.values(objectAnimations)
      .filter((anim) => {
        if (!anim.criteria) return false;
        const matches = checkCriteria(newState, anim.criteria);
        if (matches && anim.playOnce && this.playedAnimations.has(anim.id)) {
          return false;
        }
        return matches;
      })
      .sort((a, b) => (b.priority || 0) - (a.priority || 0));

    // Combine all animations
    const allAnimations = [...animations, ...objectAnims];

    if (!allAnimations || allAnimations.length === 0) {
      this.logger.log(`AnimationManager: No animations match current state`);
      return;
    }

    if (this.debug)
      this.logger.log(
        `Found ${allAnimations.length} animation(s) for state:`,
        allAnimations.map((a) => a.id)
      );

    // Process each matching animation
    for (const animData of allAnimations) {
      // Fade animations can play alongside other animations
      const isFadeAnimation = animData.type === "fade";

      // Don't interrupt currently playing animation or pre-slerp phase (unless it's a fade)
      if (!isFadeAnimation && (this.isPlaying || this.isPreSlerping)) {
        this.logger.log(
          `AnimationManager: Animation already playing or pre-slerping, skipping non-fade animation '${animData.id}'`
        );
        continue;
      }

      // Check if animation has a delay
      const delay = animData.delay || 0;

      if (delay > 0) {
        // Schedule delayed playback
        this.scheduleDelayedAnimation(animData, delay);
      } else {
        // Play immediately (playOnce check already handled in getCameraAnimationsForState)
        this.logger.log(
          `AnimationManager: State changed, playing '${animData.id}'`
        );
        this.playFromData(animData);
      }
    }
  }

  /**
   * Schedule an animation to play after a delay
   * @param {Object} animData - Animation data to schedule
   * @param {number} delay - Delay in seconds
   * @private
   */
  scheduleDelayedAnimation(animData, delay) {
    if (this.debug)
      this.logger.log(
        `Scheduling animation "${animData.id}" with ${delay}s delay`
      );

    this.pendingAnimations.set(animData.id, {
      animData,
      timer: 0,
      delay,
    });
  }

  /**
   * Cancel a pending delayed animation
   * @param {string} animId - Animation ID to cancel
   */
  cancelDelayedAnimation(animId) {
    if (this.pendingAnimations.has(animId)) {
      this.logger.log(
        `AnimationManager: Cancelled delayed animation "${animId}"`
      );
      this.pendingAnimations.delete(animId);
    }
  }

  /**
   * Cancel all pending delayed animations
   */
  cancelAllDelayedAnimations() {
    if (this.pendingAnimations.size > 0) {
      this.logger.log(
        `AnimationManager: Cancelling ${this.pendingAnimations.size} pending animation(s)`
      );
      this.pendingAnimations.clear();
    }
  }

  /**
   * Check if an animation is pending (scheduled with delay)
   * @param {string} animId - Animation ID to check
   * @returns {boolean}
   */
  isAnimationPending(animId) {
    return this.pendingAnimations.has(animId);
  }

  /**
   * Check if any animations are pending
   * @returns {boolean}
   */
  hasAnimationsPending() {
    return this.pendingAnimations.size > 0;
  }

  /**
   * Play an animation from data config
   * @param {Object} animData - Animation data from cameraAnimationData.js
   * @returns {boolean} Success
   */
  playFromData(animData) {
    // Handle fade type
    if (animData.type === "fade") {
      this.playFade(animData);
      return true;
    }

    // Handle lookat type
    if (animData.type === "lookat") {
      this.playLookat(animData);
      return true;
    }

    // Handle moveTo type
    if (animData.type === "moveTo") {
      this.playMoveTo(animData);
      return true;
    }

    // Handle objectAnimation type
    if (animData.type === "objectAnimation") {
      this.playObjectAnimation(animData);
      return true;
    }

    // Handle jsonAnimation or animation type (default)
    if (animData.type === "jsonAnimation" || animData.type === "animation") {
      const success = this.play(
        animData.id,
        () => {
          // Mark as played if playOnce
          if (animData.playOnce) {
            this.playedAnimations.add(animData.id);
          }
          // Call user-defined onComplete callback if present
          if (animData.onComplete) {
            animData.onComplete(this.gameManager);
          }
        },
        animData
      );
      return success;
    }

    this.logger.warn(`Unknown animation type "${animData.type}"`);
    return false;
  }

  /**
   * Create or get the fade cube mesh
   * @private
   */
  _getOrCreateFadeCube() {
    if (!this.fadeCube) {
      // Create a cube geometry that surrounds the camera
      const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
      const material = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0,
        side: THREE.BackSide, // Render inside of cube
        depthTest: false,
        depthWrite: false,
      });

      this.fadeCube = new THREE.Mesh(geometry, material);
      this.fadeCube.renderOrder = 9999; // Render last (in front of everything)

      // Add to camera so it follows camera transform
      this.camera.add(this.fadeCube);
      this.fadeCube.position.set(0, 0, -0.25); // Slightly in front of camera
    }

    return this.fadeCube;
  }

  /**
   * Play a fade effect from data config
   * @param {Object} fadeData - Fade data from cameraAnimationData.js
   */
  playFade(fadeData) {
    if (this.debug) this.logger.log(`Playing fade '${fadeData.id}'`);

    // Mark as played if playOnce
    if (fadeData.playOnce) {
      this.playedAnimations.add(fadeData.id);
    }

    // Store fade data
    this.isFading = true;
    this.fadeElapsed = 0;
    this.fadeData = {
      color: fadeData.color || { r: 1, g: 1, b: 1 }, // Default white
      fadeInTime: fadeData.fadeInTime || 0.1,
      holdTime: fadeData.holdTime || 0,
      fadeOutTime: fadeData.fadeOutTime || 1.0,
      maxOpacity: fadeData.maxOpacity !== undefined ? fadeData.maxOpacity : 1.0,
    };
    this.fadeOnComplete = fadeData.onComplete || null;

    // Get or create fade cube
    const cube = this._getOrCreateFadeCube();

    // Set color
    const color = this.fadeData.color;
    cube.material.color.setRGB(color.r, color.g, color.b);
    cube.material.opacity = 0;

    if (this.debug)
      this.logger.log(
        `Fade '${fadeData.id}' - in:${this.fadeData.fadeInTime}s hold:${this.fadeData.holdTime}s out:${this.fadeData.fadeOutTime}s`
      );
  }

  /**
   * Play a lookat from data config
   * @param {Object} lookAtData - Lookat data from cameraAnimationData.js
   */
  playLookat(lookAtData) {
    if (this.debug) this.logger.log(`Playing lookat '${lookAtData.id}'`);

    // Mark as played if playOnce
    if (lookAtData.playOnce) {
      this.playedAnimations.add(lookAtData.id);
    }

    // Resolve position if it's a function (support dynamic positioning)
    const resolvedData = { ...lookAtData };
    if (typeof lookAtData.position === "function") {
      resolvedData.position = lookAtData.position(this.gameManager);
      if (this.debug) {
        this.logger.log(
          `Resolved lookat position for '${lookAtData.id}':`,
          resolvedData.position
        );
      }
    }

    // Check if this is a sequence (positions array) or single position
    const isSequence = Array.isArray(resolvedData.positions);

    if (isSequence) {
      // Start sequence at first position
      this.activeSequence = {
        animData: resolvedData,
        currentIndex: 0,
        isWaitingForNext: false,
      };
      this._playLookatAtIndex(resolvedData, 0);
      return;
    }

    // Single position - use existing logic
    this._playSingleLookat(resolvedData);
  }

  /**
   * Play a single lookat (non-sequence)
   * @param {Object} lookAtData - Lookat data
   * @private
   */
  _playSingleLookat(lookAtData) {
    // Support both new (transitionTime) and old (duration) property names for backwards compatibility
    const transitionTime =
      lookAtData.transitionTime || lookAtData.duration || 2.0;
    const returnTransitionTime =
      lookAtData.returnTransitionTime ||
      lookAtData.returnDuration ||
      transitionTime;

    // Get lookat hold duration (separate from zoom hold duration)
    const lookAtHoldDuration = lookAtData.lookAtHoldDuration || 0;

    // Check if input should be restored (default: true for backwards compatibility)
    const shouldRestoreInput =
      lookAtData.restoreInput !== undefined ? lookAtData.restoreInput : true;

    // Determine if we need to delay input restoration after onComplete fires
    // Note: holdDuration is handled by characterController internally (transition → hold → return → onComplete)
    // We only need to delay for zoom effects that happen after the lookat sequence completes
    const needsDelayedRestore =
      shouldRestoreInput && lookAtData.enableZoom && lookAtData.zoomOptions;

    // Store user-defined onComplete callback
    const userOnComplete = lookAtData.onComplete;

    let onComplete = null;

    if (needsDelayedRestore) {
      const zoomTransitionDuration =
        lookAtData.zoomOptions?.transitionDuration || 0;

      let delayAfterLookat;
      if (lookAtData.returnToOriginalView) {
        // When returning to original view, zoom/DoF resets during the return transition
        // So we need to wait for the return transition to complete
        delayAfterLookat = returnTransitionTime;
        this.logger.log(
          `AnimationManager: Lookat '${lookAtData.id}' has zoom with return. ` +
            `Will restore control ${delayAfterLookat.toFixed(
              2
            )}s after lookat completes ` +
            `(return transition: ${returnTransitionTime}s)`
        );
      } else {
        // When not returning to original view, wait for zoom-out transition
        delayAfterLookat = zoomTransitionDuration;
        this.logger.log(
          `AnimationManager: Lookat '${lookAtData.id}' has zoom without return. ` +
            `Will restore control ${delayAfterLookat.toFixed(
              2
            )}s after lookat completes ` +
            `(zoom-out: ${zoomTransitionDuration}s)`
        );
      }

      // Provide onComplete that schedules delayed restoration and calls user callback
      onComplete = () => {
        this.pendingInputRestore = {
          timer: 0,
          delay: delayAfterLookat,
        };
        // Call user-defined callback if provided
        if (userOnComplete) {
          userOnComplete(this.gameManager);
        }
      };
    } else if (shouldRestoreInput) {
      // Immediate restoration when lookat completes (only if restoreInput is true and no zoom)
      onComplete = () => {
        if (this.characterController) {
          this.characterController.enableInput();
          if (this.debug)
            this.logger.log(
              `Lookat '${lookAtData.id}' complete, input restored`
            );
        }
        // Call user-defined callback if provided
        if (userOnComplete) {
          userOnComplete(this.gameManager);
        }
      };
    } else {
      // Don't restore input - just call user callback if provided
      onComplete = () => {
        this.logger.log(
          `AnimationManager: Lookat '${lookAtData.id}' complete, input NOT restored (manual restoration required)`
        );
        // Call user-defined callback if provided
        if (userOnComplete) {
          userOnComplete(this.gameManager);
        }
      };
    }

    // Emit lookat event through gameManager
    // Note: This doesn't block isPlaying - lookats can happen during JSON animations
    if (this.gameManager) {
      this.gameManager.emit("camera:lookat", {
        position: lookAtData.position,
        duration: transitionTime,
        holdDuration: lookAtHoldDuration,
        onComplete: onComplete,
        returnToOriginalView: lookAtData.returnToOriginalView || false,
        returnDuration: returnTransitionTime,
        enableZoom: lookAtData.enableZoom || false,
        zoomOptions: lookAtData.zoomOptions || {},
        colliderId: `camera-data-${lookAtData.id}`,
      });
    } else {
      this.logger.warn(`Cannot play lookat '${lookAtData.id}', no gameManager`);
    }
  }

  /**
   * Play a lookat at a specific index in a sequence
   * @param {Object} lookAtData - Lookat sequence data
   * @param {number} index - Index of position to look at
   * @private
   */
  _playLookatAtIndex(lookAtData, index) {
    if (
      !Array.isArray(lookAtData.positions) ||
      index >= lookAtData.positions.length
    ) {
      this.logger.warn(
        `Invalid sequence index ${index} for '${lookAtData.id}'`
      );
      return;
    }

    const position = lookAtData.positions[index];

    // Get settings for this position (merge defaults with per-position overrides)
    const sequenceSettings = lookAtData.sequenceSettings?.[index] || {};

    // Build effective settings for this position
    const transitionTime =
      sequenceSettings.transitionTime ?? lookAtData.transitionTime ?? 2.0;
    const lookAtHoldDuration =
      sequenceSettings.lookAtHoldDuration ?? lookAtData.lookAtHoldDuration ?? 0;
    const enableZoom =
      sequenceSettings.enableZoom ?? lookAtData.enableZoom ?? false;

    // Merge zoom options (per-position overrides take precedence)
    let zoomOptions = null;
    if (enableZoom) {
      zoomOptions = {
        ...(lookAtData.zoomOptions || {}),
        ...(sequenceSettings.zoomOptions || {}),
      };
    }

    // Determine if this is the last position in sequence
    const isLastPosition = index === lookAtData.positions.length - 1;
    const shouldLoop = lookAtData.loop ?? false;

    // Only return to original view if it's the last position and not looping
    const returnToOriginalView =
      isLastPosition &&
      !shouldLoop &&
      (lookAtData.returnToOriginalView ?? false);
    const returnTransitionTime =
      lookAtData.returnTransitionTime ?? transitionTime;

    // Check if input should be restored (only on last position if not looping)
    const shouldRestoreInput =
      isLastPosition &&
      !shouldLoop &&
      (lookAtData.restoreInput !== undefined ? lookAtData.restoreInput : true);

    // Determine if we need to delay input restoration
    const needsDelayedRestore = shouldRestoreInput && enableZoom && zoomOptions;

    // Store user-defined onComplete callback (only call on final position)
    const userOnComplete =
      isLastPosition && !shouldLoop ? lookAtData.onComplete : null;

    // Helper to progress to next position in sequence after waiting for hold duration
    const progressSequence = () => {
      // For sequences, wait for hold duration before progressing to next position
      // The characterController's onComplete fires after transition, not after hold
      if (!isLastPosition || shouldLoop) {
        const waitTime = lookAtHoldDuration || 0;
        if (waitTime > 0) {
          this.logger.log(
            `AnimationManager: Waiting ${waitTime.toFixed(
              2
            )}s before next position in sequence`
          );
          setTimeout(() => {
            this._onSequencePositionComplete(lookAtData, index);
          }, waitTime * 1000);
        } else {
          this._onSequencePositionComplete(lookAtData, index);
        }
      } else {
        // Final position or end of loop
        this._onSequencePositionComplete(lookAtData, index);
      }
    };

    let onComplete = null;

    if (needsDelayedRestore) {
      const zoomTransitionDuration = zoomOptions?.transitionDuration || 0;

      let delayAfterLookat;
      if (returnToOriginalView) {
        delayAfterLookat = returnTransitionTime;
        this.logger.log(
          `AnimationManager: Lookat '${lookAtData.id}' [${index}] has zoom with return. ` +
            `Will restore control ${delayAfterLookat.toFixed(
              2
            )}s after lookat completes`
        );
      } else {
        delayAfterLookat = zoomTransitionDuration;
        this.logger.log(
          `AnimationManager: Lookat '${lookAtData.id}' [${index}] has zoom without return. ` +
            `Will restore control ${delayAfterLookat.toFixed(
              2
            )}s after lookat completes`
        );
      }

      onComplete = () => {
        this.pendingInputRestore = {
          timer: 0,
          delay: delayAfterLookat,
        };
        if (userOnComplete) {
          userOnComplete(this.gameManager);
        }
        progressSequence();
      };
    } else if (shouldRestoreInput) {
      // Immediate restoration when lookat completes
      onComplete = () => {
        if (this.characterController) {
          this.characterController.enableInput();
          if (this.debug)
            this.logger.log(
              `Lookat '${lookAtData.id}' [${index}] complete, input restored`
            );
        }
        if (userOnComplete) {
          userOnComplete(this.gameManager);
        }
        progressSequence();
      };
    } else {
      // Don't restore input - just progress sequence
      onComplete = () => {
        if (isLastPosition && !shouldLoop) {
          this.logger.log(
            `AnimationManager: Lookat '${lookAtData.id}' complete, input NOT restored (manual restoration required)`
          );
        }
        if (userOnComplete) {
          userOnComplete(this.gameManager);
        }
        progressSequence();
      };
    }

    this.logger.log(
      `Playing lookat sequence '${lookAtData.id}' position ${index + 1}/${
        lookAtData.positions.length
      }`,
      `transition: ${transitionTime}s, hold: ${lookAtHoldDuration}s, return: ${returnToOriginalView}`,
      position
    );

    // Emit lookat event through gameManager
    if (this.gameManager) {
      this.gameManager.emit("camera:lookat", {
        position: position,
        duration: transitionTime,
        holdDuration: lookAtHoldDuration,
        onComplete: onComplete,
        returnToOriginalView: returnToOriginalView,
        returnDuration: returnTransitionTime,
        enableZoom: enableZoom,
        zoomOptions: zoomOptions || {},
        colliderId: `camera-data-${lookAtData.id}-${index}`,
      });
    } else {
      this.logger.warn(`Cannot play lookat '${lookAtData.id}', no gameManager`);
    }
  }

  /**
   * Handle completion of a position in a lookat sequence
   * @param {Object} lookAtData - Lookat sequence data
   * @param {number} completedIndex - Index of the position that just completed
   * @private
   */
  _onSequencePositionComplete(lookAtData, completedIndex) {
    if (
      !this.activeSequence ||
      this.activeSequence.animData.id !== lookAtData.id
    ) {
      // Sequence was cancelled or replaced
      return;
    }

    const nextIndex = completedIndex + 1;
    const shouldLoop = lookAtData.loop ?? false;

    // Check if we should continue to next position
    if (nextIndex < lookAtData.positions.length) {
      // More positions in sequence
      this.activeSequence.currentIndex = nextIndex;
      this._playLookatAtIndex(lookAtData, nextIndex);
    } else if (shouldLoop) {
      // Loop back to start
      this.activeSequence.currentIndex = 0;
      if (this.debug) {
        this.logger.log(`Looping sequence '${lookAtData.id}' back to start`);
      }
      this._playLookatAtIndex(lookAtData, 0);
    } else {
      // Sequence complete
      if (this.debug) {
        this.logger.log(`Sequence '${lookAtData.id}' complete`);
      }
      this.activeSequence = null;
    }
  }

  /**
   * Stop the current lookat sequence
   */
  stopSequence() {
    if (this.activeSequence) {
      this.logger.log(`Stopping sequence '${this.activeSequence.animData.id}'`);
      this.activeSequence = null;
    }
  }

  /**
   * Play an object animation from data config
   * @param {Object} animData - Object animation data from cameraAnimationData.js
   */
  playObjectAnimation(animData) {
    if (this.debug) this.logger.log(`Playing objectAnimation '${animData.id}'`);

    // Mark as played if playOnce
    if (animData.playOnce) {
      this.playedAnimations.add(animData.id);
    }

    // Check if we have sceneManager
    if (!this.sceneManager) {
      this.logger.warn(
        `Cannot play objectAnimation '${animData.id}', no sceneManager`
      );
      return;
    }

    // Get target object from scene
    let targetObject = this.sceneManager.getObject(animData.targetObjectId);
    if (!targetObject) {
      this.logger.warn(
        `Cannot play objectAnimation '${animData.id}', object '${animData.targetObjectId}' not found`
      );
      return;
    }

    // If childMeshName is specified, find that child mesh
    if (animData.childMeshName) {
      let childMesh = null;
      targetObject.traverse((child) => {
        if (child.name === animData.childMeshName) {
          childMesh = child;
        }
      });

      if (!childMesh) {
        this.logger.warn(
          `Cannot play objectAnimation '${animData.id}', child mesh '${animData.childMeshName}' not found in '${animData.targetObjectId}'`
        );
        return;
      }

      targetObject = childMesh;
      if (this.debug) {
        this.logger.log(
          `Found child mesh '${animData.childMeshName}' in '${animData.targetObjectId}'`
        );
      }
    }

    // Parse properties and set up animation state
    const properties = animData.properties || {};
    const duration = animData.duration || 1.0;

    // Handle reparenting to camera (like phone receiver)
    let originalParent = null;
    let worldPosition = null;
    let worldQuaternion = null;
    let worldScale = null;

    if (animData.reparentToCamera && this.camera) {
      // Store original parent
      originalParent = targetObject.parent;

      // Store world transform before reparenting
      worldPosition = new THREE.Vector3();
      worldQuaternion = new THREE.Quaternion();
      worldScale = new THREE.Vector3();
      targetObject.getWorldPosition(worldPosition);
      targetObject.getWorldQuaternion(worldQuaternion);
      targetObject.getWorldScale(worldScale);

      // Remove from current parent
      if (originalParent) {
        originalParent.remove(targetObject);
      }

      // Add to camera
      this.camera.add(targetObject);

      // Convert world transform to camera-local space
      const cameraWorldInverse = new THREE.Matrix4()
        .copy(this.camera.matrixWorld)
        .invert();
      const localMatrix = new THREE.Matrix4()
        .compose(worldPosition, worldQuaternion, worldScale)
        .premultiply(cameraWorldInverse);

      // Apply local transform
      localMatrix.decompose(
        targetObject.position,
        targetObject.quaternion,
        targetObject.scale
      );

      this.logger.log(
        `Reparented '${
          animData.targetObjectId
        }' to camera. Local pos: (${targetObject.position.x.toFixed(
          2
        )}, ${targetObject.position.y.toFixed(
          2
        )}, ${targetObject.position.z.toFixed(2)})`
      );
    }

    // Store animation state
    this.activeObjectAnimations.set(animData.id, {
      animData,
      targetObject,
      elapsed: 0,
      duration,
      direction: 1, // 1 for forward, -1 for reverse (yoyo or reverseOnCriteria)
      loopCount: 0,
      hasReversed: false, // Track if reverseOnCriteria has triggered
      properties: this._parseObjectAnimationProperties(
        targetObject,
        properties,
        animData.reparentToCamera
      ),
      originalParent, // Store for cleanup
      reparentedToCamera: !!animData.reparentToCamera,
    });

    this.logger.log(
      `Started objectAnimation '${animData.id}' on '${
        animData.targetObjectId
      }' (duration: ${duration.toFixed(2)}s)`
    );
  }

  /**
   * Parse and prepare animation properties
   * @param {THREE.Object3D} targetObject - Target object
   * @param {Object} properties - Animation properties config
   * @param {boolean} isReparentedToCamera - If true, object is in camera-local space
   * @returns {Object} Parsed properties with from/to values
   * @private
   */
  _parseObjectAnimationProperties(
    targetObject,
    properties,
    isReparentedToCamera = false
  ) {
    const parsed = {};

    // Position
    if (properties.position) {
      const from = properties.position.from || {
        x: targetObject.position.x,
        y: targetObject.position.y,
        z: targetObject.position.z,
      };

      let to = properties.position.to || from;

      // Handle dynamic camera-relative positioning (only for non-reparented objects)
      if (to === "CAMERA_FRONT" && this.camera && !isReparentedToCamera) {
        // Calculate position in front of camera
        const cameraPos = this.camera.position;
        const cameraDir = new THREE.Vector3();
        this.camera.getWorldDirection(cameraDir);

        // Position object 30cm in front of camera, slightly below eye level
        const offset = new THREE.Vector3(0, -0.1, -0.3);
        const worldOffset = offset.applyQuaternion(this.camera.quaternion);

        to = {
          x: cameraPos.x + worldOffset.x,
          y: cameraPos.y + worldOffset.y,
          z: cameraPos.z + worldOffset.z,
        };

        this.logger.log(
          `Dynamic camera-front position calculated: (${to.x.toFixed(
            2
          )}, ${to.y.toFixed(2)}, ${to.z.toFixed(2)})`
        );
      }

      // Check if to is an array of keyframes
      const isKeyframes = Array.isArray(to);
      if (isKeyframes) {
        // Build keyframe array with from as first keyframe
        const keyframes = [from, ...to];
        parsed.position = { keyframes, isKeyframes: true };
      } else {
        parsed.position = { from, to, isKeyframes: false };
      }
    }

    // Rotation - use quaternions for reparented objects, euler angles otherwise
    if (properties.rotation) {
      const to = properties.rotation.to;
      const isKeyframes = Array.isArray(to);

      if (isReparentedToCamera) {
        // Use quaternions for smooth interpolation when attached to camera
        const fromQuat = targetObject.quaternion.clone();

        if (isKeyframes) {
          // Convert keyframe eulers to quaternions
          const keyframes = [
            fromQuat,
            ...to.map((euler) =>
              new THREE.Quaternion().setFromEuler(
                new THREE.Euler(euler.x, euler.y, euler.z, "XYZ")
              )
            ),
          ];
          parsed.rotation = {
            keyframes,
            useQuaternion: true,
            isKeyframes: true,
          };
        } else {
          // Convert single target euler to quaternion
          const toEuler = to || { x: 0, y: 0, z: 0 };
          const toQuat = new THREE.Quaternion().setFromEuler(
            new THREE.Euler(toEuler.x, toEuler.y, toEuler.z, "XYZ")
          );
          parsed.rotation = {
            from: fromQuat,
            to: toQuat,
            useQuaternion: true,
            isKeyframes: false,
          };
        }
      } else {
        // Use euler angles for world-space objects
        const from = properties.rotation.from || {
          x: targetObject.rotation.x,
          y: targetObject.rotation.y,
          z: targetObject.rotation.z,
        };

        if (isKeyframes) {
          const keyframes = [from, ...to];
          parsed.rotation = {
            keyframes,
            useQuaternion: false,
            isKeyframes: true,
          };
        } else {
          // Allow partial rotation specs - preserve axes not specified in "to"
          const toValue = to || from;
          const toWithDefaults = {
            x: toValue.x !== undefined ? toValue.x : from.x,
            y: toValue.y !== undefined ? toValue.y : from.y,
            z: toValue.z !== undefined ? toValue.z : from.z,
          };
          parsed.rotation = {
            from,
            to: toWithDefaults,
            useQuaternion: false,
            isKeyframes: false,
          };
        }
      }
    }

    // Scale (can be number or {x, y, z})
    if (properties.scale) {
      let from, to;
      if (typeof properties.scale.from === "number") {
        from = {
          x: properties.scale.from,
          y: properties.scale.from,
          z: properties.scale.from,
        };
      } else if (properties.scale.from) {
        from = properties.scale.from;
      } else {
        from = {
          x: targetObject.scale.x,
          y: targetObject.scale.y,
          z: targetObject.scale.z,
        };
      }

      if (typeof properties.scale.to === "number") {
        to = {
          x: properties.scale.to,
          y: properties.scale.to,
          z: properties.scale.to,
        };
      } else if (properties.scale.to) {
        to = properties.scale.to;
      } else {
        to = from;
      }

      parsed.scale = { from, to };
    }

    // Opacity
    if (properties.opacity) {
      const from =
        properties.opacity.from !== undefined
          ? properties.opacity.from
          : this._getObjectOpacity(targetObject);
      const to =
        properties.opacity.to !== undefined ? properties.opacity.to : from;
      parsed.opacity = { from, to };
    }

    return parsed;
  }

  /**
   * Get current opacity of an object's materials
   * @param {THREE.Object3D} obj - Object to check
   * @returns {number} Opacity value (0-1)
   * @private
   */
  _getObjectOpacity(obj) {
    // Try to get opacity from first material found
    let opacity = 1.0;
    obj.traverse((child) => {
      if (child.material && opacity === 1.0) {
        if (Array.isArray(child.material)) {
          opacity = child.material[0].opacity || 1.0;
        } else {
          opacity = child.material.opacity || 1.0;
        }
      }
    });
    return opacity;
  }

  /**
   * Set opacity for all materials in an object
   * @param {THREE.Object3D} obj - Object to modify
   * @param {number} opacity - Opacity value (0-1)
   * @private
   */
  _setObjectOpacity(obj, opacity) {
    obj.traverse((child) => {
      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach((mat) => {
            mat.transparent = opacity < 1.0;
            mat.opacity = opacity;
            mat.needsUpdate = true;
          });
        } else {
          child.material.transparent = opacity < 1.0;
          child.material.opacity = opacity;
          child.material.needsUpdate = true;
        }
      }
    });
  }

  /**
   * Get the two keyframes and local interpolation value for a given global time
   * @param {Array} keyframes - Array of keyframe values
   * @param {number} globalT - Global interpolation value 0-1
   * @returns {Object} { from, to, localT }
   * @private
   */
  _getKeyframeSegment(keyframes, globalT) {
    if (keyframes.length < 2) {
      return { from: keyframes[0], to: keyframes[0], localT: 0 };
    }

    // Calculate which segment we're in
    const numSegments = keyframes.length - 1;
    const segmentLength = 1.0 / numSegments;
    const segmentIndex = Math.min(
      Math.floor(globalT / segmentLength),
      numSegments - 1
    );

    // Calculate local t within this segment (0-1)
    const segmentStart = segmentIndex * segmentLength;
    const localT = (globalT - segmentStart) / segmentLength;

    return {
      from: keyframes[segmentIndex],
      to: keyframes[segmentIndex + 1],
      localT: localT,
    };
  }

  /**
   * Get easing function by name
   * @param {string} name - Easing function name
   * @returns {Function} Easing function
   * @private
   */
  _getEasingFunction(name) {
    const easingFunctions = {
      linear: (t) => t,
      easeInQuad: (t) => t * t,
      easeOutQuad: (t) => t * (2 - t),
      easeInOutQuad: (t) =>
        t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2,
      easeInCubic: (t) => t * t * t,
      easeOutCubic: (t) => 1 - Math.pow(1 - t, 3),
      easeInOutCubic: (t) =>
        t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
      easeInOutElastic: (t) => {
        const c5 = (2 * Math.PI) / 4.5;
        return t === 0
          ? 0
          : t === 1
          ? 1
          : t < 0.5
          ? -(Math.pow(2, 20 * t - 10) * Math.sin((20 * t - 11.125) * c5)) / 2
          : (Math.pow(2, -20 * t + 10) * Math.sin((20 * t - 11.125) * c5)) / 2 +
            1;
      },
    };

    return easingFunctions[name] || easingFunctions.linear;
  }

  /**
   * Stop an object animation
   * @param {string} animId - Animation ID to stop
   */
  stopObjectAnimation(animId) {
    if (this.activeObjectAnimations.has(animId)) {
      this.logger.log(`Stopping objectAnimation '${animId}'`);
      this.activeObjectAnimations.delete(animId);
    }
  }

  /**
   * Stop all object animations
   */
  stopAllObjectAnimations() {
    if (this.activeObjectAnimations.size > 0) {
      this.logger.log(
        `Stopping ${this.activeObjectAnimations.size} object animation(s)`
      );
      this.activeObjectAnimations.clear();
    }
  }

  /**
   * Play a moveTo from data config
   * @param {Object} moveToData - MoveTo data from cameraAnimationData.js
   */
  playMoveTo(moveToData) {
    if (this.debug) this.logger.log(`Playing moveTo '${moveToData.id}'`);

    // Mark as played if playOnce
    if (moveToData.playOnce) {
      this.playedAnimations.add(moveToData.id);
    }

    // Support both new (transitionTime) and old (duration) property names for backwards compatibility
    const transitionTime =
      moveToData.transitionTime || moveToData.duration || 2.0;

    // Calculate final position with auto-height if enabled
    let finalPosition = { ...moveToData.position };

    if (moveToData.autoHeight && this.physicsManager) {
      // Raycast down to find floor height at the target X/Z coordinates
      const floorY = this.physicsManager.getFloorHeightAt(
        moveToData.position.x,
        moveToData.position.z,
        100, // Start ray from Y=100
        200 // Max ray distance
      );

      if (floorY !== null) {
        // Calculate body center Y (character center is at floor + half height)
        // Character capsule: halfHeight=0.6, radius=0.3, so center is at floor + 0.9
        const characterCenterY = 0.9; // From physicsManager.createCharacter: halfHeight=0.6 + radius=0.3
        const cameraHeight = this.characterController?.cameraHeight ?? 1.6;

        // Final Y = floor + character center + camera offset
        finalPosition.y = floorY + characterCenterY + cameraHeight;

        this.logger.log(
          `MoveTo '${
            moveToData.id
          }': Auto-calculated Y=${finalPosition.y.toFixed(2)} ` +
            `(floor=${floorY.toFixed(
              2
            )} + center=${characterCenterY} + camera=${cameraHeight})`
        );
      } else {
        this.logger.warn(
          `MoveTo '${
            moveToData.id
          }': Failed to find floor at (${moveToData.position.x.toFixed(2)}, ` +
            `${moveToData.position.z.toFixed(
              2
            )}), using provided Y=${moveToData.position.y.toFixed(2)}`
        );
      }
    }

    // Emit moveTo event through gameManager
    // Note: This doesn't block isPlaying - moveTos can happen during JSON animations
    if (this.gameManager) {
      this.gameManager.emit("character:moveto", {
        position: finalPosition,
        rotation: moveToData.rotation || null,
        lookat: moveToData.lookat || null, // Pass lookat position if provided
        duration: transitionTime, // Still use 'duration' for the event for compatibility with characterController
        inputControl: moveToData.inputControl || {
          disableMovement: true,
          disableRotation: true,
        },
        restoreInput:
          moveToData.restoreInput !== undefined
            ? moveToData.restoreInput
            : true, // Default to true for backwards compatibility
        onComplete: moveToData.onComplete || null,
      });
    } else {
      this.logger.warn(`Cannot play moveTo '${moveToData.id}', no gameManager`);
    }
  }

  /**
   * Load an animation from JSON
   * @param {string} name - Animation identifier
   * @param {string} url - Path to JSON file
   * @param {boolean} isPreload - Whether this is a preload (for loading screen tracking)
   * @returns {Promise<boolean>} Success
   */
  async loadAnimation(name, url, isPreload = false) {
    // Register with loading screen if preloading
    if (this.loadingScreen && isPreload) {
      this.loadingScreen.registerTask(`camera_anim_${name}`, 1);
    }

    try {
      let src = url || "";
      if (!/^(\/|https?:)/i.test(src)) {
        src = "/json/" + (src.endsWith(".json") ? src : src + ".json");
      }
      const res = await fetch(src);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const raw = Array.isArray(data.frames) ? data.frames : [];
      if (raw.length === 0) {
        this.logger.warn(`No frames in '${src}'`);
        if (this.loadingScreen && isPreload) {
          this.loadingScreen.completeTask(`camera_anim_${name}`);
        }
        return false;
      }

      // Build quaternions and compute deltas relative to first frame
      const quats = raw.map(
        (f) => new THREE.Quaternion(f.q[0], f.q[1], f.q[2], f.q[3])
      );
      const q0Inv = quats[0].clone().invert();
      const p0 = Array.isArray(raw[0].p)
        ? new THREE.Vector3(raw[0].p[0], raw[0].p[1], raw[0].p[2])
        : new THREE.Vector3(0, 0, 0);

      const t0 = typeof raw[0].t === "number" ? raw[0].t : 0;
      // TODO: Update this scale to match final world scale
      const positionScale = 2.0; // Scale factor for world coordinates
      const frames = raw.map((f, i) => {
        const t = (typeof f.t === "number" ? f.t : 0) - t0;
        const qd = q0Inv.clone().multiply(quats[i]);
        let pd = null;
        if (Array.isArray(f.p)) {
          const p = new THREE.Vector3(f.p[0], f.p[1], f.p[2]);
          const dpWorld = p.sub(p0).multiplyScalar(positionScale);
          pd = dpWorld.applyQuaternion(q0Inv.clone());
        }
        return { t, qd, pd };
      });

      const duration = frames[frames.length - 1].t;
      this.animations.set(name, { frames, duration });
      this.logger.log(
        `AnimationManager: Loaded '${name}' (${
          frames.length
        } frames, ${duration.toFixed(2)}s)`
      );

      // Mark as complete in loading screen if preloading
      if (this.loadingScreen && isPreload) {
        this.loadingScreen.completeTask(`camera_anim_${name}`);
      }

      return true;
    } catch (e) {
      this.logger.warn(`Failed to load '${name}':`, e);
      // Mark as complete even on error so loading screen can proceed
      if (this.loadingScreen && isPreload) {
        this.loadingScreen.completeTask(`camera_anim_${name}`);
      }
      return false;
    }
  }

  /**
   * Play an animation
   * @param {string} name - Animation identifier
   * @param {Function} onComplete - Optional completion callback
   * @param {Object} animData - Optional animation data config
   * @returns {boolean} Success
   */
  play(name, onComplete = null, animData = null) {
    const anim = this.animations.get(name);
    if (!anim) {
      this.logger.warn(`Animation '${name}' not found`);
      return false;
    }

    // Disable character controller immediately
    if (this.characterController) {
      this.characterController.disableInput();
    }

    // Start pre-slerp phase to reset pitch to 0 while preserving yaw
    // This ensures consistent vertical results regardless of initial look direction
    this.preSlerpStartQuat.copy(this.camera.quaternion);

    // Extract current yaw, set pitch to 0
    const euler = new THREE.Euler().setFromQuaternion(
      this.camera.quaternion,
      "YXZ"
    );
    const targetEuler = new THREE.Euler(0, euler.y, 0, "YXZ"); // pitch=0, keep yaw, roll=0
    this.preSlerpTargetQuat.setFromEuler(targetEuler);

    this.preSlerpElapsed = 0;
    this.isPreSlerping = true;

    // Store animation data for when pre-slerp completes
    this.currentAnimation = anim;
    this.currentAnimationData = animData;
    this.onComplete = onComplete;
    this.isPlaying = false; // Not playing actual animation yet

    if (this.debug)
      this.logger.log(
        `Pre-slerping to zero pitch (keeping yaw) before playing '${name}'`
      );
    return true;
  }

  /**
   * Start the actual animation playback (called after pre-slerp completes)
   * @private
   */
  _startAnimationPlayback() {
    // Capture current camera pose as base (should now be at zero pitch)
    this.baseQuat.copy(this.camera.quaternion);
    this.basePos.copy(this.camera.position);

    // Reset playback state
    this.elapsed = 0;
    this.frameIdx = 1;
    this.isPlaying = true;

    // Get Y-axis scale from animation data (default to 1.0)
    this.scaleY = this.currentAnimationData?.scaleY ?? 1.0;

    // Get playback rate from animation data (default to 1.0)
    this.playbackRate = this.currentAnimationData?.playbackRate ?? 1.0;

    if (this.debug)
      this.logger.log(
        `Starting animation playback from level horizon (pitch=0)${
          this.scaleY !== 1.0 ? ` with Y-scale ${this.scaleY}` : ""
        }${this.playbackRate !== 1.0 ? ` at ${this.playbackRate}x speed` : ""}`
      );
  }

  /**
   * Stop current animation
   * @param {boolean} syncController - If true, sync controller yaw/pitch to camera (can be overridden by animData)
   * @param {boolean} restoreInput - If true, restore input controls (can be overridden by animData)
   */
  stop(syncController = true, restoreInput = true) {
    // Stop pre-slerp if active
    if (this.isPreSlerping) {
      this.isPreSlerping = false;
      this.currentAnimation = null;
      this.currentAnimationData = null;
      if (this.characterController && restoreInput) {
        this.characterController.enableInput();
      }
      return;
    }

    if (!this.isPlaying) return;

    // Use animData config if available
    if (this.currentAnimationData) {
      syncController =
        this.currentAnimationData.syncController !== undefined
          ? this.currentAnimationData.syncController
          : syncController;
      restoreInput =
        this.currentAnimationData.restoreInput !== undefined
          ? this.currentAnimationData.restoreInput
          : restoreInput;
    }

    this.isPlaying = false;
    this.currentAnimation = null;
    this.currentAnimationData = null;

    // Update physics body position to match camera's final position
    if (this.characterController && this.characterController.character) {
      const character = this.characterController.character;

      // Calculate physics body position (camera position minus camera height offset)
      const cameraHeight = this.characterController.cameraHeight || 1.6;
      const bodyPosition = {
        x: this.camera.position.x,
        y: this.camera.position.y - cameraHeight,
        z: this.camera.position.z,
      };

      // Update physics body position
      character.setTranslation(bodyPosition, true);

      // Update physics body rotation to match camera's yaw (ignore pitch for capsule)
      const euler = new THREE.Euler().setFromQuaternion(
        this.camera.quaternion,
        "YXZ"
      );
      const bodyQuat = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(0, euler.y, 0, "YXZ")
      );
      character.setRotation(
        { x: bodyQuat.x, y: bodyQuat.y, z: bodyQuat.z, w: bodyQuat.w },
        true
      );

      // Reset velocity to prevent any residual movement
      character.setLinvel({ x: 0, y: 0, z: 0 }, true);
      character.setAngvel({ x: 0, y: 0, z: 0 }, true);
    }

    // Restore character controller
    if (this.characterController) {
      if (syncController) {
        const euler = new THREE.Euler().setFromQuaternion(
          this.camera.quaternion,
          "YXZ"
        );
        this.characterController.yaw = euler.y;
        this.characterController.pitch = euler.x;
        this.characterController.targetYaw = this.characterController.yaw;
        this.characterController.targetPitch = this.characterController.pitch;
      }

      // Only restore input if configured
      if (restoreInput) {
        this.characterController.enableInput();
        this.logger.log("AnimationManager: Stopped, input restored");
      } else {
        // If not restoring input, disable camera sync to leave camera frozen
        this.characterController.disableCameraSync();
        this.logger.log(
          "AnimationManager: Stopped, input NOT restored, camera frozen (manual restoration required)"
        );
      }
    }
  }

  /**
   * Update animation (call every frame)
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    // Handle pre-slerp phase (reset camera to neutral orientation before animation)
    if (this.isPreSlerping) {
      this.preSlerpElapsed += dt;
      const t = Math.min(1, this.preSlerpElapsed / this.preSlerpDuration);

      // Use eased interpolation for smoother transition
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // easeInOutQuad

      // Slerp camera rotation to neutral
      this.camera.quaternion
        .copy(this.preSlerpStartQuat)
        .slerp(this.preSlerpTargetQuat, eased);

      // Check if slerp is complete
      if (t >= 1) {
        this.isPreSlerping = false;
        this._startAnimationPlayback();
      }
      return; // Don't process other updates during pre-slerp
    }

    // Update fade effect
    if (this.isFading && this.fadeData) {
      this.fadeElapsed += dt;
      const cube = this.fadeCube;

      if (cube) {
        const { fadeInTime, holdTime, fadeOutTime, maxOpacity } = this.fadeData;
        const fadeInEnd = fadeInTime;
        const holdEnd = fadeInEnd + holdTime;
        const fadeOutEnd = holdEnd + fadeOutTime;

        let opacity = 0;

        if (this.fadeElapsed < fadeInEnd) {
          // Fade in phase
          const t = this.fadeElapsed / fadeInTime;
          opacity = t * maxOpacity;
        } else if (this.fadeElapsed < holdEnd) {
          // Hold phase
          opacity = maxOpacity;
        } else if (this.fadeElapsed < fadeOutEnd) {
          // Fade out phase
          const t = (this.fadeElapsed - holdEnd) / fadeOutTime;
          opacity = (1 - t) * maxOpacity;
        } else {
          // Complete
          opacity = 0;
          this.isFading = false;

          const callback = this.fadeOnComplete;
          this.fadeData = null;
          this.fadeOnComplete = null;

          if (callback) {
            callback();
          }

          this.logger.log(`Fade complete`);
        }

        cube.material.opacity = opacity;
      }
    }

    // Update pending delayed animations
    if (this.pendingAnimations.size > 0) {
      for (const [animId, pending] of this.pendingAnimations) {
        pending.timer += dt;

        // Check if delay has elapsed
        const isFadeAnimation = pending.animData.type === "fade";
        const canPlay =
          pending.timer >= pending.delay &&
          (isFadeAnimation || (!this.isPlaying && !this.isPreSlerping));

        if (canPlay) {
          if (this.debug)
            this.logger.log(
              `Playing delayed animation "${animId}"${
                isFadeAnimation ? " (fade)" : ""
              }`
            );
          this.pendingAnimations.delete(animId);
          this.playFromData(pending.animData);
          break; // Only play one animation per frame
        }
      }
    }

    // Update pending input restoration (for lookat animations with zoom)
    if (this.pendingInputRestore) {
      this.pendingInputRestore.timer += dt;

      if (this.pendingInputRestore.timer >= this.pendingInputRestore.delay) {
        // Restore input after delay
        if (this.characterController) {
          this.characterController.enableInput();
          if (this.debug)
            this.logger.log(
              `Restored control after zoom completion (${this.pendingInputRestore.delay.toFixed(
                2
              )}s delay)`
            );
        }
        this.pendingInputRestore = null;
      }
    }

    // Update active object animations
    if (this.activeObjectAnimations.size > 0) {
      const completedAnimations = [];

      for (const [animId, animState] of this.activeObjectAnimations) {
        const { animData, targetObject, duration, direction, properties } =
          animState;

        // Check if we should reverse based on criteria
        if (
          animData.reverseOnCriteria &&
          !animState.hasReversed &&
          this.gameManager
        ) {
          const currentState = this.gameManager.getState();
          const shouldReverse = checkCriteria(
            currentState,
            animData.reverseOnCriteria
          );

          if (shouldReverse && animState.direction === 1) {
            // Reverse the animation direction
            animState.direction = -1;
            animState.hasReversed = true;
            if (this.debug) {
              this.logger.log(
                `Reversing objectAnimation '${animId}' due to criteria match`
              );
            }
          }
        }

        // Update elapsed time
        animState.elapsed += dt * animState.direction;

        // Calculate normalized time (0-1)
        let t = Math.max(0, Math.min(1, animState.elapsed / duration));

        // Apply easing
        const easingName = animData.easing || "linear";
        const easingFunc = this._getEasingFunction(easingName);
        const easedT = easingFunc(t);

        // Apply property animations
        if (properties.position) {
          const { isKeyframes } = properties.position;
          if (isKeyframes) {
            // Keyframe animation
            const { keyframes } = properties.position;
            const { from, to, localT } = this._getKeyframeSegment(
              keyframes,
              easedT
            );
            targetObject.position.x = from.x + (to.x - from.x) * localT;
            targetObject.position.y = from.y + (to.y - from.y) * localT;
            targetObject.position.z = from.z + (to.z - from.z) * localT;
          } else {
            // Simple two-point animation
            const { from, to } = properties.position;
            targetObject.position.x = from.x + (to.x - from.x) * easedT;
            targetObject.position.y = from.y + (to.y - from.y) * easedT;
            targetObject.position.z = from.z + (to.z - from.z) * easedT;
          }
        }

        if (properties.rotation) {
          const { useQuaternion, isKeyframes } = properties.rotation;
          if (isKeyframes) {
            // Keyframe animation
            const { keyframes } = properties.rotation;
            const { from, to, localT } = this._getKeyframeSegment(
              keyframes,
              easedT
            );
            if (useQuaternion) {
              targetObject.quaternion.slerpQuaternions(from, to, localT);
            } else {
              targetObject.rotation.x = from.x + (to.x - from.x) * localT;
              targetObject.rotation.y = from.y + (to.y - from.y) * localT;
              targetObject.rotation.z = from.z + (to.z - from.z) * localT;
            }
          } else {
            // Simple two-point animation
            const { from, to } = properties.rotation;
            if (useQuaternion) {
              // Use quaternion slerp for smooth rotation (reparented objects)
              targetObject.quaternion.slerpQuaternions(from, to, easedT);
            } else {
              // Use euler lerp for world-space objects
              targetObject.rotation.x = from.x + (to.x - from.x) * easedT;
              targetObject.rotation.y = from.y + (to.y - from.y) * easedT;
              targetObject.rotation.z = from.z + (to.z - from.z) * easedT;
            }
          }
        }

        if (properties.scale) {
          const { from, to } = properties.scale;
          targetObject.scale.x = from.x + (to.x - from.x) * easedT;
          targetObject.scale.y = from.y + (to.y - from.y) * easedT;
          targetObject.scale.z = from.z + (to.z - from.z) * easedT;
        }

        if (properties.opacity) {
          const { from, to } = properties.opacity;
          const opacity = from + (to - from) * easedT;
          this._setObjectOpacity(targetObject, opacity);
        }

        // Check if animation is complete
        const isComplete =
          (direction > 0 && animState.elapsed >= duration) ||
          (direction < 0 && animState.elapsed <= 0);

        if (isComplete) {
          const shouldLoop = animData.loop || false;
          const shouldYoyo = animData.yoyo || false;

          if (shouldLoop) {
            if (shouldYoyo) {
              // Reverse direction for yoyo
              animState.direction *= -1;
              // Clamp elapsed to stay within bounds
              if (direction > 0) {
                animState.elapsed = duration;
              } else {
                animState.elapsed = 0;
              }
            } else {
              // Reset to start for normal loop
              animState.elapsed = 0;
            }
            animState.loopCount++;
          } else {
            // Animation complete, mark for removal
            completedAnimations.push(animId);

            // Call onComplete callback
            if (animData.onComplete) {
              animData.onComplete(this.gameManager);
            }

            if (this.debug) {
              this.logger.log(`objectAnimation '${animId}' complete`);
            }
          }
        }
      }

      // Remove completed animations
      for (const animId of completedAnimations) {
        this.activeObjectAnimations.delete(animId);
      }
    }

    // Handle settle-up phase (runs after animation frames complete)
    if (this.isSettlingUp) {
      this.settleElapsed += dt;
      const t = Math.min(1, this.settleElapsed / this.settleDuration);
      const eased = 1 - Math.pow(1 - t, 3);
      this.camera.position.y =
        this.settleStartY + (this.settleTargetY - this.settleStartY) * eased;
      if (t >= 1) {
        const callback = this._pendingComplete;
        const animData = this._pendingAnimData; // Preserve animation data settings
        this._pendingComplete = null;
        this._pendingAnimData = null;
        this.isSettlingUp = false;
        // Complete by stopping - use animation data settings if available
        if (animData) {
          this.currentAnimationData = animData;
          this.stop(true, true); // Will be overridden by animData
        } else {
          this.stop(true, true);
        }
        if (callback) callback();
      }
      return;
    }

    if (!this.isPlaying || !this.currentAnimation) return;

    // Apply playback rate to delta time
    this.elapsed += dt * this.playbackRate;
    const { frames, duration } = this.currentAnimation;

    // Check if animation is complete
    if (this.elapsed >= duration) {
      const last = frames[frames.length - 1];
      // Apply final pose
      this.camera.quaternion.copy(this.baseQuat).multiply(last.qd);
      if (last.pd) {
        this._interpPos.copy(last.pd);
        // Apply Y-axis scale if set
        if (this.scaleY !== 1.0) {
          this._interpPos.y *= this.scaleY;
        }
        this._rotatedPos.copy(this._interpPos).applyQuaternion(this.baseQuat);
        this.camera.position.copy(this.basePos).add(this._rotatedPos);
      }

      // Check if we should restore input (use animData if available)
      const shouldRestoreInput =
        this.currentAnimationData?.restoreInput !== undefined
          ? this.currentAnimationData.restoreInput
          : true;

      // Only ensure clearance if we're about to restore control
      // If restoreInput is false, leave camera exactly where animation landed it
      if (shouldRestoreInput) {
        const cameraHeight = this.characterController?.cameraHeight ?? 1.6;
        const bodyCenterY = this.camera.position.y - cameraHeight;
        if (bodyCenterY < this.minCharacterCenterY) {
          // Lerp camera up by the shortfall before restoring input
          const deltaUp = this.minCharacterCenterY - bodyCenterY;
          this.isSettlingUp = true;
          this.settleStartY = this.camera.position.y;
          this.settleTargetY = this.camera.position.y + deltaUp;
          this.settleElapsed = 0;
          this._pendingComplete = this.onComplete;
          this._pendingAnimData = this.currentAnimationData; // Preserve for settling phase
          return;
        }
      }

      // Complete
      const callback = this.onComplete;
      this.stop(true, true); // Use defaults, will be overridden by animData if present
      if (callback) callback();
      return;
    }

    // Advance frame cursor
    while (
      this.frameIdx < frames.length &&
      frames[this.frameIdx].t < this.elapsed
    ) {
      this.frameIdx++;
    }

    // Interpolate between frames
    const a = frames[Math.max(0, this.frameIdx - 1)];
    const b = frames[Math.min(frames.length - 1, this.frameIdx)];
    const span = Math.max(1e-6, b.t - a.t);
    const s = Math.min(1, Math.max(0, (this.elapsed - a.t) / span));

    // Apply rotation delta
    this._interpDelta.copy(a.qd).slerp(b.qd, s);
    this.camera.quaternion.copy(this.baseQuat).multiply(this._interpDelta);

    // Apply position delta
    if (a.pd && b.pd) {
      this._interpPos.copy(a.pd).lerp(b.pd, s);
      // Apply Y-axis scale if set
      if (this.scaleY !== 1.0) {
        this._interpPos.y *= this.scaleY;
      }
      this._rotatedPos.copy(this._interpPos).applyQuaternion(this.baseQuat);
      this.camera.position.copy(this.basePos).add(this._rotatedPos);
    }
  }

  /**
   * Check if an animation is currently playing (includes pre-slerp phase)
   * @returns {boolean}
   */
  get playing() {
    return this.isPlaying || this.isPreSlerping;
  }

  /**
   * Get list of loaded animation names
   * @returns {string[]}
   */
  getAnimationNames() {
    return Array.from(this.animations.keys());
  }
}

export default AnimationManager;
