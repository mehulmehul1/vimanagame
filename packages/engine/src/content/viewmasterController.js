/**
 * viewmasterController.js - VIEWMASTER EQUIP/UNEQUIP MECHANIC
 * =============================================================================
 *
 * ROLE: Controls the viewmaster headset mechanic including equip/unequip
 * animations, fractal VFX intensity, and overheat timeout.
 *
 * KEY RESPONSIBILITIES:
 * - Toggle viewmaster equip state via V key or button
 * - Animate equip/unequip object animations
 * - Track fractal intensity and overheat
 * - Trigger VFX (fractal distortion) when equipped
 * - Auto-equip during SHADOW_AMPLIFICATIONS state
 * - Handle timeout and max intensity transitions
 *
 * =============================================================================
 */

import * as THREE from "three";
import { objectAnimations } from "../animationObjectData.js";
import { GAME_STATES } from "../gameData.js";
import { Logger } from "../utils/logger.js";

const TOGGLE_ON_ID = "viewmasterToggleOn";
const TOGGLE_OFF_ID = "viewmasterToggleOff";
const FRACTAL_TIMEOUT_SECONDS = 14;
const FRACTAL_MIN_INTENSITY = 0.02;
const FRACTAL_MAX_INTENSITY = 10.0; // World completely dissolves at the end
const VFX_DELAY_RATIO = 0.7;

export default class ViewmasterController {
  constructor({
    gameManager,
    animationManager,
    sceneManager,
    vfxManager = null,
  }) {
    this.gameManager = gameManager;
    this.animationManager = animationManager;
    this.sceneManager = sceneManager;
    this.vfxManager = vfxManager || window?.vfxManager || null;

    this.logger = new Logger("ViewmasterController", false);

    this.isTransitioning = false;
    this.isToggleEnabled = false;
    this.transitionTimeout = null;
    this.initialPoseApplied = false;
    this.initialPoseTimer = null;
    this.pendingVfxTimeout = null;
    this.isEquipped = false;
    this.maskPlane = null;
    this.fractalTimer = 0;
    this.timeoutTriggered = false;
    this.currentFractalIntensity = 0;
    this._splatFractal = null;
    this.maxIntensityReached = false; // Track when max intensity is reached for SHADOW_AMPLIFICATIONS
    this.isAutoEquipping = false; // Track when auto-equipping for SHADOW_AMPLIFICATIONS
    this.glitchSequenceActive = false; // Track if glitch sequence is running
    this.glitchSequenceTimeout = null; // Timeout for glitch sequence
    this.equipRetryTimeout = null; // Timeout for retrying equip when object not loaded

    this.handleStateChanged = this.handleStateChanged.bind(this);
    this.handleKeyDown = this.handleKeyDown.bind(this);
  }

  initialize() {
    if (!this.gameManager || !this.animationManager) {
      this.logger.warn("Missing dependencies, skipping initialization");
      return;
    }

    this.gameManager.on("state:changed", this.handleStateChanged);
    window.addEventListener("keydown", this.handleKeyDown, false);
    this.setupButtonClickHandler();

    this.handleStateChanged(this.gameManager.getState());
    this.applyInitialAttachmentIfNeeded(this.gameManager.getState());
    this.applyFractalIntensity(0);
    this.setupProgressBarStyles();
    this.logger.log("Initialized");
  }

  setupProgressBarStyles() {
    const styleId = "viewmaster-progress-bar-styles";
    if (document.getElementById(styleId)) return;

    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
      #space-bar-hint.progress-mode {
        animation: none !important;
      }

      #space-bar-hint.progress-mode .progress-container {
        position: relative;
        width: 100%;
        height: 100%;
      }

      #space-bar-hint.progress-mode .base-image {
        filter: grayscale(100%) brightness(0.5);
      }

      #space-bar-hint.progress-mode .progress-fill {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        clip-path: inset(0 var(--progress-percent, 100%) 0 0);
        transition: clip-path 0.1s linear;
        pointer-events: none;
      }

      #space-bar-hint.progress-mode .progress-fill img {
        width: 100%;
        height: auto;
        display: block;
        filter: brightness(1);
      }
    `;
    document.head.appendChild(style);
  }

  updateSpectreScopeProgress(progress) {
    const spaceBarHint = document.getElementById("space-bar-hint");
    if (!spaceBarHint) return;

    const progressPercent = Math.max(0, Math.min(100, progress * 100));
    spaceBarHint.style.setProperty("--progress-percent", `${progressPercent}%`);

    const isEquipped = this.isEquipped;
    const gameState = this.gameManager?.getState?.();
    const currentState = gameState?.currentState;

    // Only show button between CURSOR and POST_CURSOR states
    if (
      currentState === undefined ||
      currentState < GAME_STATES.CURSOR ||
      currentState >= GAME_STATES.POST_CURSOR
    ) {
      spaceBarHint.style.opacity = "0";
      spaceBarHint.style.pointerEvents = "none";
      return;
    }

    if (isEquipped) {
      spaceBarHint.classList.add("progress-mode");
      spaceBarHint.classList.remove("animating");

      // Keep button clickable so user can take it off
      spaceBarHint.style.pointerEvents = "all";

      let container = spaceBarHint.querySelector(".progress-container");
      if (!container) {
        // Only create container and images ONCE, the first time
        const img = spaceBarHint.querySelector(
          "img:not(.progress-container img)"
        );
        if (img) {
          // Store reference to original image BEFORE modifying
          if (!spaceBarHint._originalImage) {
            spaceBarHint._originalImage = img;
          }

          const originalSrc = img.src;
          const originalAlt = img.alt || "Press Space to submit";

          container = document.createElement("div");
          container.className = "progress-container";

          // Create images ONCE and store references
          if (!spaceBarHint._baseImage) {
            spaceBarHint._baseImage = document.createElement("img");
            spaceBarHint._baseImage.src = originalSrc;
            spaceBarHint._baseImage.alt = originalAlt;
            spaceBarHint._baseImage.className = "base-image";
          }

          if (!spaceBarHint._progressFill) {
            spaceBarHint._progressFill = document.createElement("div");
            spaceBarHint._progressFill.className = "progress-fill";
            if (!spaceBarHint._whiteImage) {
              spaceBarHint._whiteImage = document.createElement("img");
              spaceBarHint._whiteImage.src = originalSrc;
              spaceBarHint._whiteImage.alt = originalAlt;
            }
            spaceBarHint._progressFill.appendChild(spaceBarHint._whiteImage);
          }

          container.appendChild(spaceBarHint._baseImage);
          container.appendChild(spaceBarHint._progressFill);

          // Hide original image instead of removing it
          img.style.display = "none";
          spaceBarHint.appendChild(container);
        }
      } else {
        // Container exists, just show it
        container.style.display = "";
        // Hide original image
        if (spaceBarHint._originalImage) {
          spaceBarHint._originalImage.style.display = "none";
        }
      }

      const progressFill = spaceBarHint.querySelector(".progress-fill");
      if (progressFill) {
        progressFill.style.clipPath = `inset(0 ${progressPercent}% 0 0)`;
      }

      spaceBarHint.style.opacity = "1";
    } else {
      // Only restore original image structure if we're actually switching from progress mode
      if (spaceBarHint.classList.contains("progress-mode")) {
        spaceBarHint.classList.remove("progress-mode");

        // Re-enable button clicks (mobile)
        if (spaceBarHint.classList.contains("mobile-button")) {
          spaceBarHint.style.pointerEvents = "all";
        }

        const container = spaceBarHint.querySelector(".progress-container");
        if (container) {
          // Hide container instead of removing it (preserves image elements)
          container.style.display = "none";

          // Show the original image again (it was never removed, just hidden)
          if (spaceBarHint._originalImage) {
            spaceBarHint._originalImage.style.display = "";
          }
        }
      }

      // When viewmaster is removed and toggle is enabled, start flashing to indicate it can be put on again
      const isMobile = gameState?.isMobile === true;
      const isIOS = gameState?.isIOS === true;
      const isDrawingActive = window.drawingManager?.isActive === true;

      if (this.isToggleEnabled && !isDrawingActive) {
        // Viewmaster toggle is enabled - show flashing hint to indicate it can be put on again
        if (isMobile || isIOS) {
          // Mobile: just show the button
          spaceBarHint.style.opacity = "1";
          spaceBarHint.style.pointerEvents = "all";
        } else {
          // Desktop: start flashing animation
          spaceBarHint.style.opacity = "1";
          spaceBarHint.style.pointerEvents = "all";
          spaceBarHint.classList.add("animating");

          // Set up animation loop (remove old listener first)
          const oldHandler = spaceBarHint._viewmasterFlashHandler;
          if (oldHandler) {
            spaceBarHint.removeEventListener("animationend", oldHandler);
          }

          const handler = () => {
            if (!this.isEquipped && this.isToggleEnabled && !isDrawingActive) {
              setTimeout(() => {
                if (
                  !this.isEquipped &&
                  this.isToggleEnabled &&
                  !isDrawingActive
                ) {
                  spaceBarHint.classList.remove("animating");
                  void spaceBarHint.offsetWidth;
                  spaceBarHint.classList.add("animating");
                }
              }, 2000);
            }
          };

          spaceBarHint._viewmasterFlashHandler = handler;
          spaceBarHint.addEventListener("animationend", handler);
        }
      } else if ((isMobile || isIOS) && !isDrawingActive) {
        // Hide button when toggle not enabled and drawing game not active
        spaceBarHint.style.opacity = "0";
      }
    }
  }

  handleStateChanged(newState, oldState) {
    const enabled =
      newState &&
      newState.currentState !== undefined &&
      newState.currentState >= GAME_STATES.SHADOW_AMPLIFICATIONS &&
      newState.currentState < GAME_STATES.POST_CURSOR;

    this.isToggleEnabled = enabled;

    // At POST_CURSOR or later, return viewmaster to LIGHTS_OUT position and disable toggle
    if (
      newState &&
      newState.currentState !== undefined &&
      newState.currentState >= GAME_STATES.POST_CURSOR
    ) {
      // Force unequip if equipped
      if (newState?.isViewmasterEquipped) {
        this.gameManager.setState({ isViewmasterEquipped: false });
      }

      // Return viewmaster to LIGHTS_OUT position
      this.returnToLightsOutPosition();
    }

    if (!enabled && newState?.isViewmasterEquipped) {
      this.gameManager.setState({ isViewmasterEquipped: false });
    }

    // Auto-equip viewmaster when entering SHADOW_AMPLIFICATIONS state
    // Also handle initialization case (oldState is undefined/null) when already in SHADOW_AMPLIFICATIONS
    const isEnteringShadowAmplifications =
      newState?.currentState === GAME_STATES.SHADOW_AMPLIFICATIONS &&
      (oldState === undefined ||
        oldState === null ||
        oldState?.currentState !== GAME_STATES.SHADOW_AMPLIFICATIONS);

    const isInitialization = oldState === undefined || oldState === null;

    if (
      isEnteringShadowAmplifications &&
      !this.isEquipped &&
      !this.isTransitioning &&
      !this.isAutoEquipping &&
      !newState?.viewmasterManuallyRemoved
    ) {
      this.logger.log(
        `SHADOW_AMPLIFICATIONS state detected (currentState: ${newState?.currentState}), auto-equipping viewmaster`
      );
      this.logger.log(
        `Current equipped state: ${this.isEquipped}, transitioning: ${
          this.isTransitioning
        }, autoEquipping: ${this.isAutoEquipping}, oldState: ${
          isInitialization ? "undefined/null (init)" : oldState.currentState
        }`
      );

      // If isViewmasterEquipped isn't already set in state, set it immediately for VFX
      if (!newState?.isViewmasterEquipped) {
        this.gameManager.setState({
          isViewmasterEquipped: true,
          viewmasterManuallyRemoved: false,
          viewmasterOverheatDialogIndex: null,
        });
      }

      // During initialization, delay slightly to ensure scene objects are loaded
      if (isInitialization) {
        setTimeout(() => {
          this.equipForShadowAmplifications();
        }, 100);
      } else {
        this.equipForShadowAmplifications();
      }
    }

    const previousEquipped = this.isEquipped;
    // Only update from state if not auto-equipping (to avoid overwriting during transition)
    if (!this.isAutoEquipping) {
      this.isEquipped = !!newState?.isViewmasterEquipped;
    }

    if (this.isEquipped !== previousEquipped) {
      // Don't play toggle animations at POST_CURSOR or later
      if (
        newState &&
        newState.currentState !== undefined &&
        newState.currentState >= GAME_STATES.POST_CURSOR
      ) {
        return;
      }

      // Play animation when state changes (but don't call toggle() which might set state again)
      if (!this.isTransitioning && this.isToggleEnabled) {
        if (!this.sceneManager?.hasObject("viewmaster")) {
          this.logger.warn("Viewmaster object not loaded yet");
        } else {
          const animationId = this.isEquipped ? TOGGLE_ON_ID : TOGGLE_OFF_ID;
          const animation = objectAnimations[animationId];

          if (animation) {
            this.isTransitioning = true;
            this.animationManager.playObjectAnimation(animation);
            this.ensureMaskPlane();
            this.updateMaskPlane(true);

            const durationMs = Math.max(animation.duration || 1, 1) * 1000;
            clearTimeout(this.transitionTimeout);
            this.transitionTimeout = setTimeout(() => {
              this.isTransitioning = false;
              this.updateMaskPlane(this.isEquipped);

              if (!this.isEquipped && this.timeoutTriggered) {
                this.timeoutTriggered = false;
              }

              if (!this.isEquipped && this.isToggleEnabled) {
                this.updateSpectreScopeProgress(0);
              }
            }, durationMs + 200);

            if (this.isEquipped) {
              this.stopFractalRamp();
              const vfxDelayMs = Math.max(durationMs * VFX_DELAY_RATIO, 0);
              if (this.pendingVfxTimeout) {
                clearTimeout(this.pendingVfxTimeout);
                this.pendingVfxTimeout = null;
              }
              this.pendingVfxTimeout = setTimeout(() => {
                this.startFractalRamp();
                this.pendingVfxTimeout = null;
              }, vfxDelayMs);
            } else {
              this.stopFractalRamp();
            }
          }
        }
      }

      if (this.isEquipped) {
        // Reset max intensity flag when equipping
        if (newState?.currentState === GAME_STATES.SHADOW_AMPLIFICATIONS) {
          this.maxIntensityReached = false;
        }
      } else {
        this.maxIntensityReached = false;
      }
    }

    // Show button on mobile when viewmaster toggle is enabled (even if not equipped yet)
    const spaceBarHint = document.getElementById("space-bar-hint");
    if (spaceBarHint && spaceBarHint.classList.contains("mobile-button")) {
      const isMobile = newState?.isMobile === true;
      const isIOS = newState?.isIOS === true;
      const isDrawingActive = window.drawingManager?.isActive === true;

      if (
        (isMobile || isIOS) &&
        (this.isToggleEnabled || this.isEquipped || isDrawingActive)
      ) {
        if (!this.isEquipped && !isDrawingActive) {
          // Show button when toggle is enabled but not equipped (for toggling on)
          spaceBarHint.style.opacity = "1";
          spaceBarHint.style.pointerEvents = "all";
        }
      }
    }

    // Reset max intensity flag and glitch sequence when leaving SHADOW_AMPLIFICATIONS
    if (newState?.currentState !== GAME_STATES.SHADOW_AMPLIFICATIONS) {
      if (this.maxIntensityReached) {
        this.maxIntensityReached = false;
      }
      if (this.glitchSequenceActive) {
        this.glitchSequenceActive = false;
        clearTimeout(this.glitchSequenceTimeout);
        this.glitchSequenceTimeout = null;
        // Clear glitch flag if we're leaving the state
        if (newState?.currentState !== GAME_STATES.CAT_SAVE) {
          this.gameManager.setState({ glitchIntense: false });
        }
      }
    }

    this.applyInitialAttachmentIfNeeded(newState);
  }

  setupButtonClickHandler() {
    const spaceBarHint = document.getElementById("space-bar-hint");
    if (!spaceBarHint) {
      setTimeout(() => this.setupButtonClickHandler(), 100);
      return;
    }

    // Prevent duplicate listeners
    if (spaceBarHint._viewmasterClickHandler) {
      return;
    }

    const handleButtonClick = (e) => {
      if (!this.isToggleEnabled) return;

      const activeElement = document.activeElement;
      if (activeElement) {
        const tag = activeElement.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") {
          return;
        }
      }

      e.preventDefault();
      e.stopPropagation();

      const currentState = this.gameManager.getState();
      const isDrawingActive = window.drawingManager?.isActive === true;

      if (isDrawingActive) {
        return;
      }

      const newEquippedState = !currentState.isViewmasterEquipped;
      this.gameManager.setState({
        isViewmasterEquipped: newEquippedState,
        viewmasterManuallyRemoved: false,
      });
    };

    spaceBarHint._viewmasterClickHandler = handleButtonClick;
    spaceBarHint.addEventListener("click", handleButtonClick);
    spaceBarHint.addEventListener("touchend", (e) => {
      handleButtonClick(e);
    });
  }

  handleKeyDown(event) {
    if (!this.isToggleEnabled) return;
    if (event.repeat) return;
    if (event.code !== "Space" && event.key !== " ") return;

    const activeElement = document.activeElement;
    if (activeElement) {
      const tag = activeElement.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") {
        return;
      }
    }

    // Toggle the state - handleStateChanged will call toggle() to play animation
    const currentState = this.gameManager.getState();
    const newEquippedState = !currentState.isViewmasterEquipped;
    // Reset manually removed flag when user presses spacebar to toggle
    this.gameManager.setState({
      isViewmasterEquipped: newEquippedState,
      viewmasterManuallyRemoved: false,
    });
  }

  applyInitialAttachmentIfNeeded(state) {
    if (this.initialPoseApplied) return;
    if (!state || state.currentState === undefined) return;
    if (state.currentState < GAME_STATES.CURSOR) return;

    if (!this.sceneManager?.hasObject("viewmaster")) {
      if (!this.initialPoseTimer) {
        this.initialPoseTimer = setTimeout(() => {
          this.initialPoseTimer = null;
          this.applyInitialAttachmentIfNeeded(this.gameManager.getState());
        }, 250);
      }
      return;
    }

    const animation = objectAnimations[TOGGLE_OFF_ID];
    if (!animation) {
      this.initialPoseApplied = true;
      this.logger.warn("Toggle-off animation missing; skipping initial pose");
      return;
    }

    this.initialPoseApplied = true;
    this.isEquipped = false;
    this.gameManager.setState({
      isViewmasterEquipped: false,
      viewmasterManuallyRemoved: false,
      viewmasterOverheatDialogIndex: null, // Reset dialog index
    });

    this.isTransitioning = true;
    this.animationManager.playObjectAnimation(animation);
    this.ensureMaskPlane();
    this.updateMaskPlane(false);
    this.stopFractalRamp();

    const durationMs = Math.max(animation.duration || 1, 1) * 1000;
    clearTimeout(this.transitionTimeout);
    this.transitionTimeout = setTimeout(() => {
      this.isTransitioning = false;
    }, durationMs + 200);
  }

  toggle() {
    if (this.isTransitioning) return;

    if (!this.isToggleEnabled) {
      this.logger.log("Viewmaster toggle not enabled yet");
      return;
    }

    if (!this.sceneManager?.hasObject("viewmaster")) {
      this.logger.warn("Viewmaster object not loaded yet");
      return;
    }

    // Use current isEquipped state (may have been set via state change)
    const nextEquipped = this.isEquipped;
    const animationId = nextEquipped ? TOGGLE_ON_ID : TOGGLE_OFF_ID;
    const animation = objectAnimations[animationId];

    if (!animation) {
      this.logger.warn(`Animation '${animationId}' not found`);
      return;
    }

    this.isTransitioning = true;
    // Don't set isEquipped here - it's already set from state
    this.animationManager.playObjectAnimation(animation);
    this.ensureMaskPlane();
    this.updateMaskPlane(true);

    const durationMs = Math.max(animation.duration || 1, 1) * 1000;
    clearTimeout(this.transitionTimeout);
    this.transitionTimeout = setTimeout(() => {
      this.isTransitioning = false;
      this.updateMaskPlane(this.isEquipped);

      // Reset timeout trigger after takeoff completes
      if (!this.isEquipped && this.timeoutTriggered) {
        this.timeoutTriggered = false;
      }

      // Reset progress bar and start flashing when viewmaster is removed
      if (!this.isEquipped && this.isToggleEnabled) {
        this.updateSpectreScopeProgress(0);
      }
    }, durationMs + 200);

    if (this.pendingVfxTimeout) {
      clearTimeout(this.pendingVfxTimeout);
      this.pendingVfxTimeout = null;
    }

    if (nextEquipped) {
      this.stopFractalRamp();
      const vfxDelayMs = Math.max(durationMs * VFX_DELAY_RATIO, 0);
      this.pendingVfxTimeout = setTimeout(() => {
        // State is already set, just start the fractal ramp
        this.startFractalRamp();
        this.pendingVfxTimeout = null;
      }, vfxDelayMs);
    } else {
      // Manually removed (not a timeout)
      // State is already set, just stop the fractal ramp
      this.stopFractalRamp();
    }
  }

  ensureMaskPlane() {
    if (this.maskPlane || !this.sceneManager) return;

    const viewmaster = this.sceneManager.getObject("viewmaster");
    if (!viewmaster) return;

    const geometry = new THREE.PlaneGeometry(0.6, 0.35);
    const material = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.0,
      depthWrite: true,
      depthTest: true,
      colorWrite: false,
      side: THREE.DoubleSide,
    });

    this.maskPlane = new THREE.Mesh(geometry, material);
    this.maskPlane.position.set(0, 0, -0.1);
    this.maskPlane.renderOrder = 10050;
    this.maskPlane.visible = false;

    viewmaster.add(this.maskPlane);
  }

  updateMaskPlane(active) {
    if (!this.maskPlane) return;
    this.maskPlane.visible = !!active;
  }

  getFractalEffect() {
    if (
      this._splatFractal &&
      typeof this._splatFractal.setExternalIntensity === "function"
    ) {
      return this._splatFractal;
    }

    const effect =
      this.vfxManager?.effects?.splatFractal ||
      window?.vfxManager?.effects?.splatFractal ||
      null;

    if (effect && typeof effect.setExternalIntensity === "function") {
      this._splatFractal = effect;
      return effect;
    }

    return null;
  }

  applyFractalIntensity(intensity) {
    const clamped = Math.max(0, intensity); // Only clamp minimum, let it go as high as needed
    this.currentFractalIntensity = clamped;

    const effect = this.getFractalEffect();
    if (effect) {
      effect.setExternalIntensity(clamped);
    }
  }

  startFractalRamp() {
    this.fractalTimer = 0;
    this.timeoutTriggered = false;
    this.applyFractalIntensity(FRACTAL_MIN_INTENSITY);
    // Reset progress bar to 0 when starting the ramp (headset is now fully on)
    this.updateSpectreScopeProgress(0);
  }

  stopFractalRamp() {
    this.fractalTimer = 0;
    this.timeoutTriggered = false;
    this.applyFractalIntensity(0);
  }

  equipForShadowAmplifications() {
    if (!this.sceneManager?.hasObject("viewmaster")) {
      this.logger.warn("Viewmaster object not loaded yet, retrying...");
      // Retry after a short delay (similar to applyInitialAttachmentIfNeeded)
      if (!this.equipRetryTimeout) {
        this.equipRetryTimeout = setTimeout(() => {
          this.equipRetryTimeout = null;
          this.equipForShadowAmplifications();
        }, 250);
      }
      return;
    }

    const animation = objectAnimations[TOGGLE_ON_ID];
    if (!animation) {
      this.logger.warn(`Animation '${TOGGLE_ON_ID}' not found`);
      return;
    }

    // Clear any pending retry timeout
    if (this.equipRetryTimeout) {
      clearTimeout(this.equipRetryTimeout);
      this.equipRetryTimeout = null;
    }

    this.isAutoEquipping = true;
    this.isTransitioning = true;
    this.isEquipped = true;
    // State is already set to isViewmasterEquipped: true by handleStateChanged pre-emptively
    this.animationManager.playObjectAnimation(animation);
    this.ensureMaskPlane();
    this.updateMaskPlane(true);

    const durationMs = Math.max(animation.duration || 1, 1) * 1000;
    clearTimeout(this.transitionTimeout);
    this.transitionTimeout = setTimeout(() => {
      this.isTransitioning = false;
      this.updateMaskPlane(this.isEquipped);
    }, durationMs + 200);

    if (this.pendingVfxTimeout) {
      clearTimeout(this.pendingVfxTimeout);
      this.pendingVfxTimeout = null;
    }

    const vfxDelayMs = Math.max(durationMs * VFX_DELAY_RATIO, 0);
    this.pendingVfxTimeout = setTimeout(() => {
      this.startFractalRamp();
      this.isAutoEquipping = false;
      this.pendingVfxTimeout = null;
    }, vfxDelayMs);
  }

  returnToLightsOutPosition() {
    if (!this.sceneManager?.hasObject("viewmaster")) {
      this.logger.warn("Viewmaster object not loaded yet, retrying...");
      setTimeout(() => {
        this.returnToLightsOutPosition();
      }, 250);
      return;
    }

    const animation = objectAnimations["viewmasterLightsOutPosition"];
    if (!animation) {
      this.logger.warn("viewmasterLightsOutPosition animation not found");
      return;
    }

    // Stop any ongoing animations
    this.stopFractalRamp();
    this.isEquipped = false;

    // Play the animation to return to LIGHTS_OUT position
    this.animationManager.playObjectAnimation(animation);
    this.updateMaskPlane(false);

    this.logger.log("Returning viewmaster to LIGHTS_OUT position");
  }

  forceTakeoff() {
    if (this.timeoutTriggered) return;
    this.timeoutTriggered = true;

    if (this.isTransitioning || !this.isEquipped) return;

    this.logger.log("Fractal intensity threshold reached, forcing takeoff");
    if (this.pendingVfxTimeout) {
      clearTimeout(this.pendingVfxTimeout);
      this.pendingVfxTimeout = null;
    }

    // Mark as manually removed (it's a timeout, user didn't click)
    // This prevents auto-equip logic from re-equipping it
    this.gameManager.setState({
      isViewmasterEquipped: false,
      viewmasterManuallyRemoved: true,
    });

    // Ensure progress bar resets and flashing starts after toggle completes
    setTimeout(() => {
      if (!this.isEquipped) {
        this.updateSpectreScopeProgress(0);
      }
    }, 100);
  }

  update(dt = 0.016) {
    // Refresh effect reference if it wasn't available earlier
    if (!this._splatFractal) {
      this.getFractalEffect();
    }

    const currentState = this.gameManager?.getState();
    const isShadowAmplifications =
      currentState?.currentState === GAME_STATES.SHADOW_AMPLIFICATIONS;

    // Continue fractal ramp while equipped OR during forced takeoff transition
    const shouldRampFractal =
      this.isEquipped || (this.timeoutTriggered && this.isTransitioning);

    if (shouldRampFractal) {
      // Only ramp if VFX effect is available
      if (this.getFractalEffect()) {
        // Check if rune lookat is active (pause intensity buildup during rune lock-on)
        const isRuneLookatActive =
          this.animationManager?.isPlaying &&
          this.animationManager?.currentAnimationData?.id === "runeLookat";

        // Only start/continue fractal timer and progress bar when transition is complete
        // and rune lookat is not active
        if (!this.isTransitioning && !isRuneLookatActive) {
          this.fractalTimer += dt;
        }
        const progress = Math.min(
          this.fractalTimer / FRACTAL_TIMEOUT_SECONDS,
          1
        );
        // Exponential ease-in: starts very slow, kicks up dramatically at the end
        const eased = Math.pow(progress, 4);
        const intensity =
          FRACTAL_MIN_INTENSITY +
          eased * (FRACTAL_MAX_INTENSITY - FRACTAL_MIN_INTENSITY);
        this.applyFractalIntensity(intensity);
        // Only update progress bar when transition is complete (headset fully on)
        if (!this.isTransitioning) {
          this.updateSpectreScopeProgress(progress);
        }

        if (progress >= 1 && this.isEquipped) {
          // Special handling for SHADOW_AMPLIFICATIONS: glitch sequence then transition
          if (isShadowAmplifications) {
            if (!this.maxIntensityReached && !this.glitchSequenceActive) {
              this.maxIntensityReached = true;
              this.glitchSequenceActive = true;
              this.logger.log(
                "Max fractal intensity reached in SHADOW_AMPLIFICATIONS, starting glitch sequence"
              );

              // Stop fractal audio immediately
              const fractalEffect = this.getFractalEffect();
              if (
                fractalEffect &&
                fractalEffect.audio &&
                fractalEffect.audio.isPlaying
              ) {
                fractalEffect.audio.stop();
                this.logger.log("Stopped fractal audio at max intensity");
              }

              // Trigger heavy glitch effect (glitch audio will start automatically)
              this.gameManager.setState({ glitchIntense: true });

              // After 3 seconds, take off viewmaster and transition
              clearTimeout(this.glitchSequenceTimeout);
              this.glitchSequenceTimeout = setTimeout(() => {
                this.logger.log(
                  "Glitch sequence complete, taking off viewmaster and transitioning to CAT_SAVE"
                );

                // Take off viewmaster and transition atomically
                this.isEquipped = false;
                this.applyFractalIntensity(0);
                this.gameManager.setState({
                  glitchIntense: false,
                  isViewmasterEquipped: false,
                  viewmasterManuallyRemoved: false,
                  viewmasterOverheatDialogIndex: null,
                  currentState: GAME_STATES.CAT_SAVE,
                });

                // Reset flags
                this.glitchSequenceActive = false;
                this.maxIntensityReached = false;
                this.glitchSequenceTimeout = null;

                // Sever the amplifier cord connection
                if (this.gameManager.amplifierCord) {
                  const phoneCord =
                    this.gameManager.amplifierCord.getPhoneCord();
                  if (phoneCord && typeof phoneCord.sever === "function") {
                    phoneCord.sever();
                    this.logger.log(
                      "Cord connection severed, cord falling to floor"
                    );
                  }
                }

                // Play takeoff animation
                const toggleOffAnimation = objectAnimations[TOGGLE_OFF_ID];
                if (toggleOffAnimation) {
                  this.isTransitioning = true;
                  this.animationManager.playObjectAnimation(toggleOffAnimation);
                  this.updateMaskPlane(false);

                  const durationMs =
                    Math.max(toggleOffAnimation.duration || 1, 1) * 1000;
                  clearTimeout(this.transitionTimeout);
                  this.transitionTimeout = setTimeout(() => {
                    this.isTransitioning = false;
                  }, durationMs + 200);
                }
              }, 3000); // 3 seconds
            }
            // Keep intensity at max during glitch sequence (already applied above)
          } else {
            // Normal behavior: force takeoff
            this.forceTakeoff();
          }
        }
      }
    } else if (this.currentFractalIntensity > 0 && !this.isTransitioning) {
      this.applyFractalIntensity(0);
    }

    // Update progress bar visibility and flashing state
    if (!this.isEquipped) {
      this.updateSpectreScopeProgress(0);
    }
  }
}
