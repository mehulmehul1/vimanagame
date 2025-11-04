import * as THREE from "three";
import { objectAnimations } from "../animationObjectData.js";
import { GAME_STATES } from "../gameData.js";
import { Logger } from "../utils/logger.js";

const TOGGLE_ON_ID = "viewmasterToggleOn";
const TOGGLE_OFF_ID = "viewmasterToggleOff";
const FRACTAL_TIMEOUT_SECONDS = 10;
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

    this.handleStateChanged(this.gameManager.getState());
    this.applyInitialAttachmentIfNeeded(this.gameManager.getState());
    this.applyFractalIntensity(0);
    this.logger.log("Initialized");
  }

  handleStateChanged(newState, oldState) {
    const enabled =
      newState &&
      newState.currentState !== undefined &&
      newState.currentState >= GAME_STATES.SHADOW_AMPLIFICATIONS;

    this.isToggleEnabled = enabled;

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
      !this.isAutoEquipping
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
      if (this.isEquipped) {
        this.startFractalRamp();
        // Reset max intensity flag when equipping
        if (newState?.currentState === GAME_STATES.SHADOW_AMPLIFICATIONS) {
          this.maxIntensityReached = false;
        }
      } else {
        this.stopFractalRamp();
        this.maxIntensityReached = false;
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

    this.toggle();
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

    if (!this.sceneManager?.hasObject("viewmaster")) {
      this.logger.warn("Viewmaster object not loaded yet");
      return;
    }

    const nextEquipped = !this.isEquipped;
    const animationId = nextEquipped ? TOGGLE_ON_ID : TOGGLE_OFF_ID;
    const animation = objectAnimations[animationId];

    if (!animation) {
      this.logger.warn(`Animation '${animationId}' not found`);
      return;
    }

    this.isTransitioning = true;
    this.isEquipped = nextEquipped;
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
    }, durationMs + 200);

    if (this.pendingVfxTimeout) {
      clearTimeout(this.pendingVfxTimeout);
      this.pendingVfxTimeout = null;
    }

    if (nextEquipped) {
      this.stopFractalRamp();
      const vfxDelayMs = Math.max(durationMs * VFX_DELAY_RATIO, 0);
      this.pendingVfxTimeout = setTimeout(() => {
        this.gameManager.setState({
          isViewmasterEquipped: true,
          viewmasterManuallyRemoved: false, // Reset when putting back on
          viewmasterOverheatDialogIndex: null, // Reset dialog index when putting back on
        });
        this.startFractalRamp();
        this.pendingVfxTimeout = null;
      }, vfxDelayMs);
    } else {
      // Manually removed (not a timeout)
      const currentState = this.gameManager.getState();
      // Preserve viewmasterOverheatDialogIndex if removed due to timeout/overheating
      // Only reset it if manually removed (not due to timeout)
      const shouldPreserveDialogIndex =
        this.timeoutTriggered &&
        currentState?.viewmasterOverheatDialogIndex !== null &&
        currentState?.viewmasterOverheatDialogIndex !== undefined;

      this.gameManager.setState({
        isViewmasterEquipped: false,
        viewmasterManuallyRemoved: !this.timeoutTriggered,
        // Only reset dialog index if manually removed (preserve it for overheating dialogs/animations)
        viewmasterOverheatDialogIndex: shouldPreserveDialogIndex
          ? currentState.viewmasterOverheatDialogIndex
          : null,
      });
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

  forceTakeoff() {
    if (this.timeoutTriggered) return;
    this.timeoutTriggered = true;

    if (this.isTransitioning || !this.isEquipped) return;

    this.logger.log("Fractal intensity threshold reached, forcing takeoff");
    if (this.pendingVfxTimeout) {
      clearTimeout(this.pendingVfxTimeout);
      this.pendingVfxTimeout = null;
    }

    // Mark as NOT manually removed (it's a timeout)
    this.gameManager.setState({ viewmasterManuallyRemoved: false });
    this.toggle();
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
        this.fractalTimer += dt;
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
  }
}
