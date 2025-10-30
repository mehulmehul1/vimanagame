import * as THREE from "three";
import { objectAnimations } from "../animationObjectData.js";
import { GAME_STATES } from "../gameData.js";
import { Logger } from "../utils/logger.js";

const TOGGLE_ON_ID = "viewmasterToggleOn";
const TOGGLE_OFF_ID = "viewmasterToggleOff";
const VFX_DELAY_RATIO = 0.7;

export default class ViewmasterController {
  constructor({ gameManager, animationManager, sceneManager }) {
    this.gameManager = gameManager;
    this.animationManager = animationManager;
    this.sceneManager = sceneManager;

    this.logger = new Logger("ViewmasterController", false);

    this.isTransitioning = false;
    this.isToggleEnabled = false;
    this.transitionTimeout = null;
    this.initialPoseApplied = false;
    this.initialPoseTimer = null;
    this.pendingStateTimeout = null;
    this.isEquipped = false;
    this.maskPlane = null;

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
    this.logger.log("Initialized");
  }

  handleStateChanged(newState) {
    const enabled =
      newState &&
      newState.currentState !== undefined &&
      newState.currentState >= GAME_STATES.CURSOR;

    this.isToggleEnabled = enabled;

    if (!enabled && newState?.isViewmasterEquipped) {
      this.gameManager.setState({ isViewmasterEquipped: false });
    }

    this.isEquipped = !!newState?.isViewmasterEquipped;

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
    this.gameManager.setState({ isViewmasterEquipped: false });

    this.isTransitioning = true;
    this.animationManager.playObjectAnimation(animation);
    this.ensureMaskPlane();
    this.updateMaskPlane(false);

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
    if (this.pendingStateTimeout) {
      clearTimeout(this.pendingStateTimeout);
      this.pendingStateTimeout = null;
    }
    this.animationManager.playObjectAnimation(animation);
    this.ensureMaskPlane();
    this.updateMaskPlane(true);

    const durationMs = Math.max(animation.duration || 1, 1) * 1000;
    clearTimeout(this.transitionTimeout);
    this.transitionTimeout = setTimeout(() => {
      this.isTransitioning = false;
      this.updateMaskPlane(this.isEquipped);
    }, durationMs + 200);

    if (nextEquipped) {
      const vfxDelay = Math.max(durationMs * VFX_DELAY_RATIO, 0);
      this.pendingStateTimeout = setTimeout(() => {
        this.gameManager.setState({ isViewmasterEquipped: true });
        this.pendingStateTimeout = null;
      }, vfxDelay);
    } else {
      this.gameManager.setState({ isViewmasterEquipped: false });
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
}
