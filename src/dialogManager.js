import { Howl } from "howler";
import * as THREE from "three";
import { GAME_STATES } from "./gameData.js";

/**
 * DialogManager - Handles dialog audio playback with synchronized captions
 *
 * Features:
 * - Play dialog audio files
 * - Display synchronized captions (HTML)
 * - Event-based triggering
 * - Queue multiple dialog sequences
 * - Callback support for dialog completion
 */

class DialogManager {
  constructor(options = {}) {
    this.scene = options.scene || null;
    this.camera = options.camera || null;
    this.sfxManager = options.sfxManager || null;
    this.gameManager = options.gameManager || null;
    this.dialogChoiceUI = options.dialogChoiceUI || null;
    this.loadingScreen = options.loadingScreen || null; // For progress tracking

    // Caption display (HTML)
    this.captionElement = options.captionElement || this.createCaptionElement();

    // Apply custom caption styling if provided, otherwise use defaults
    if (options.captionStyle) {
      this.setCaptionStyle(options.captionStyle);
    } else {
      this.applyDefaultCaptionStyle();
    }

    this.baseVolume = options.audioVolume || 0.8;
    this.audioVolume = this.baseVolume;
    this.currentDialog = null;
    this.currentAudio = null;
    this.captionQueue = [];
    this.captionIndex = 0;
    this.captionTimer = 0;
    this.isPlaying = false;
    this.onCompleteCallback = null;
    this.captionsEnabled = true; // Default to captions enabled

    // Per-dialog progress-based state triggers
    this.currentProgressTriggers = [];
    this.progressTriggersFired = new Set();

    // Delayed playback support
    this.pendingDialogs = new Map(); // Map of dialogId -> { dialogData, onComplete, timer, delay }

    // Preloading support
    this.preloadedAudio = new Map(); // Map of dialogId -> Howl instance
    this.deferredDialogs = new Map(); // Map of dialogId -> dialog data for later loading

    // Update volume based on SFX manager if available
    if (this.sfxManager) {
      this.audioVolume = this.baseVolume * this.sfxManager.getMasterVolume();
    }

    // Event listeners
    this.eventListeners = {
      "dialog:play": [],
      "dialog:stop": [],
      "dialog:complete": [],
      "dialog:caption": [],
    };

    // Set up state change listener if gameManager is provided
    if (this.gameManager) {
      this.setupStateListener();
    }
  }

  /**
   * Preload dialog audio files
   * @param {Object} dialogsData - Dialog data object (from dialogData.js)
   */
  preloadDialogs(dialogsData) {
    if (!dialogsData) return;

    Object.values(dialogsData).forEach((dialog) => {
      if (!dialog.audio) return; // Skip dialogs without audio

      const preload = dialog.preload !== false; // Default to true

      // If preload is false, defer loading
      if (!preload) {
        this.deferredDialogs.set(dialog.id, dialog);
        console.log(
          `DialogManager: Deferred loading for dialog "${dialog.id}"`
        );
        return;
      }

      // Register with loading screen if available and preloading
      if (this.loadingScreen && preload) {
        this.loadingScreen.registerTask(`dialog_${dialog.id}`, 1);
      }

      // Preload the audio
      const howl = new Howl({
        src: [dialog.audio],
        volume: this.audioVolume,
        preload: true,
        onload: () => {
          console.log(`DialogManager: Preloaded dialog "${dialog.id}"`);
          if (this.loadingScreen && preload) {
            this.loadingScreen.completeTask(`dialog_${dialog.id}`);
          }
        },
        onloaderror: (id, error) => {
          console.error(
            `DialogManager: Failed to preload dialog "${dialog.id}":`,
            error
          );
          if (this.loadingScreen && preload) {
            this.loadingScreen.completeTask(`dialog_${dialog.id}`);
          }
        },
        onend: () => {
          this.handleDialogComplete();
        },
      });

      this.preloadedAudio.set(dialog.id, howl);
    });
  }

  /**
   * Load deferred dialogs (called after loading screen)
   */
  loadDeferredDialogs() {
    console.log(
      `DialogManager: Loading ${this.deferredDialogs.size} deferred dialogs`
    );
    for (const [id, dialog] of this.deferredDialogs) {
      if (!dialog.audio) continue;

      const howl = new Howl({
        src: [dialog.audio],
        volume: this.audioVolume,
        preload: true,
        onload: () => {
          console.log(`DialogManager: Loaded deferred dialog "${dialog.id}"`);
        },
        onloaderror: (id, error) => {
          console.error(
            `DialogManager: Failed to load deferred dialog "${dialog.id}":`,
            error
          );
        },
        onend: () => {
          this.handleDialogComplete();
        },
      });

      this.preloadedAudio.set(dialog.id, howl);
    }
    this.deferredDialogs.clear();
  }

  /**
   * Set game manager and register event listeners
   * @param {GameManager} gameManager - The game manager instance
   */
  setGameManager(gameManager) {
    this.gameManager = gameManager;
    this.setupStateListener();
  }

  /**
   * Set up state change listener for auto-playing dialogs
   */
  setupStateListener() {
    if (!this.gameManager) return;

    // Track played dialogs for "once" functionality
    this.playedDialogs = new Set();

    // Import getDialogsForState
    import("./dialogData.js").then(({ getDialogsForState }) => {
      // Listen for state changes
      this.gameManager.on("state:changed", (newState, oldState) => {
        const matchingDialogs = getDialogsForState(
          newState,
          this.playedDialogs
        );

        // If there are matching dialogs for the new state
        if (matchingDialogs.length > 0) {
          const dialog = matchingDialogs[0];

          // Cancel any pending dialogs if we have a higher priority one
          if (this.hasDialogsPending()) {
            console.log(
              `DialogManager: Canceling pending dialogs for new dialog "${dialog.id}"`
            );
            this.cancelAllDelayedDialogs();
          }

          console.log(`DialogManager: Auto-playing dialog "${dialog.id}"`);

          // Track that this dialog has been played
          this.playedDialogs.add(dialog.id);

          // Emit event for tracking
          this.gameManager.emit("dialog:trigger", dialog.id, dialog);

          // Play the dialog
          this.playDialog(dialog, (completedDialog) => {
            this.gameManager.emit("dialog:finished", completedDialog);
          });
        }
      });

      console.log("DialogManager: Event listeners registered");
    });
  }

  /**
   * Create default caption element if none provided
   */
  createCaptionElement() {
    const caption = document.createElement("div");
    caption.id = "dialog-caption";
    document.body.appendChild(caption);
    return caption;
  }

  /**
   * Play a dialog sequence (cancels any currently playing dialog)
   * @param {Object} dialogData - Dialog data object with audio and captions
   * @param {Function} onComplete - Optional callback when dialog finishes
   */
  playDialog(dialogData, onComplete = null) {
    // Cancel any currently playing dialog
    if (this.isPlaying) {
      console.log(
        `DialogManager: Canceling current dialog "${this.currentDialog?.id}" for new dialog "${dialogData.id}"`
      );
      this.stopDialog();
    }

    // Cancel any pending delayed dialog with the same ID
    if (this.pendingDialogs.has(dialogData.id)) {
      this.cancelDelayedDialog(dialogData.id);
    }

    // Check if this dialog has a delay
    const delay = dialogData.delay || 0;

    if (delay > 0) {
      // Schedule delayed playback
      this.scheduleDelayedDialog(dialogData, onComplete, delay);
      return;
    }

    // Play immediately
    this._playDialogImmediate(dialogData, onComplete);
  }

  /**
   * Schedule a dialog to play after a delay
   * @param {Object} dialogData - Dialog data object
   * @param {Function} onComplete - Optional callback
   * @param {number} delay - Delay in seconds
   * @private
   */
  scheduleDelayedDialog(dialogData, onComplete, delay) {
    console.log(
      `DialogManager: Scheduling dialog "${dialogData.id}" with ${delay}s delay`
    );

    this.pendingDialogs.set(dialogData.id, {
      dialogData,
      onComplete,
      timer: 0,
      delay,
    });
  }

  /**
   * Cancel a pending delayed dialog
   * @param {string} dialogId - Dialog ID to cancel
   */
  cancelDelayedDialog(dialogId) {
    if (this.pendingDialogs.has(dialogId)) {
      console.log(`DialogManager: Cancelled delayed dialog "${dialogId}"`);
      this.pendingDialogs.delete(dialogId);
    }
  }

  /**
   * Cancel all pending delayed dialogs
   */
  cancelAllDelayedDialogs() {
    if (this.pendingDialogs.size > 0) {
      console.log(
        `DialogManager: Cancelling ${this.pendingDialogs.size} pending dialog(s)`
      );
      this.pendingDialogs.clear();
    }
  }

  /**
   * Immediately play a dialog (internal method)
   * @param {Object} dialogData - Dialog data object
   * @param {Function} onComplete - Optional callback
   * @private
   */
  async _playDialogImmediate(dialogData, onComplete) {
    this.currentDialog = dialogData;
    this.onCompleteCallback = onComplete;
    this.captionQueue = dialogData.captions || [];
    this.captionIndex = 0;
    this.captionTimer = 0;
    this.isPlaying = true;
    // Normalize and store progress-based state triggers for this dialog
    this.currentProgressTriggers = [];
    this.progressTriggersFired.clear();
    if (dialogData) {
      // Single trigger form
      if (
        dialogData.progressStateTrigger &&
        typeof dialogData.progressStateTrigger === "object"
      ) {
        const t = dialogData.progressStateTrigger;
        if (typeof t.progress === "number" && t.state !== undefined) {
          this.currentProgressTriggers.push({
            progress: Math.max(0, Math.min(1, t.progress)),
            state: t.state,
          });
        }
      }
      // Array form
      if (Array.isArray(dialogData.progressStateTriggers)) {
        dialogData.progressStateTriggers.forEach((t) => {
          if (t && typeof t.progress === "number" && t.state !== undefined) {
            this.currentProgressTriggers.push({
              progress: Math.max(0, Math.min(1, t.progress)),
              state: t.state,
            });
          }
        });
      }
      // Sort triggers ascending by progress
      this.currentProgressTriggers.sort((a, b) => a.progress - b.progress);
    }

    // Load and play audio
    if (dialogData.audio) {
      // Check if audio was preloaded or loaded from deferred
      if (this.preloadedAudio.has(dialogData.id)) {
        console.log(
          `DialogManager: Using preloaded audio for "${dialogData.id}"`
        );
        this.currentAudio = this.preloadedAudio.get(dialogData.id);
        this.currentAudio.volume(this.audioVolume);
        this.currentAudio.play();
      } else {
        // Check if this was a deferred dialog that hasn't been loaded yet
        if (this.deferredDialogs.has(dialogData.id)) {
          console.log(
            `DialogManager: Loading deferred dialog "${dialogData.id}" on-demand`
          );
          const deferredDialog = this.deferredDialogs.get(dialogData.id);

          this.currentAudio = new Howl({
            src: [deferredDialog.audio],
            volume: this.audioVolume,
            preload: true, // Explicitly load now
            onend: () => {
              this.handleDialogComplete();
            },
            onloaderror: (id, error) => {
              console.error("DialogManager: Failed to load audio", error);
              this.handleDialogComplete();
            },
          });

          // Remove from deferred map and add to preloaded
          this.deferredDialogs.delete(dialogData.id);
          this.preloadedAudio.set(dialogData.id, this.currentAudio);
        } else {
          // Fallback: Load on-demand from dialogData
          console.log(
            `DialogManager: Loading audio on-demand for "${dialogData.id}"`
          );
          this.currentAudio = new Howl({
            src: [dialogData.audio],
            volume: this.audioVolume,
            preload: true, // Explicitly load now
            onend: () => {
              this.handleDialogComplete();
            },
            onloaderror: (id, error) => {
              console.error("DialogManager: Failed to load audio", error);
              this.handleDialogComplete();
            },
          });
        }

        // Wait for the audio to load using Howler's event system (only if not already loaded)
        if (this.currentAudio.state && this.currentAudio.state() !== "loaded") {
          await new Promise((resolve) => {
            this.currentAudio.once("load", () => {
              console.log(
                `DialogManager: Loaded on-demand dialog "${dialogData.id}"`
              );
              resolve();
            });
            this.currentAudio.once("loaderror", (id, error) => {
              console.error(
                `DialogManager: Failed to load on-demand dialog "${dialogData.id}":`,
                error
              );
              resolve(); // Resolve anyway to prevent hanging
            });
          });
        }

        this.currentAudio.play();
      }
    }

    // Start first caption if available
    if (this.captionQueue.length > 0) {
      this.showCaption(this.captionQueue[0]);
    }

    this.emit("dialog:play", dialogData);
  }

  /**
   * Stop current dialog
   */
  stopDialog() {
    if (this.currentAudio) {
      this.currentAudio.stop();
      this.currentAudio.unload();
      this.currentAudio = null;
    }

    this.hideCaption();
    this.isPlaying = false;
    this.currentDialog = null;
    this.captionQueue = [];
    this.captionIndex = 0;
    this.captionTimer = 0;
    this.currentProgressTriggers = [];
    this.progressTriggersFired.clear();

    this.emit("dialog:stop");
  }

  /**
   * Show a caption
   * @param {Object} caption - Caption object with text and duration
   */
  showCaption(caption) {
    // Only show caption if captions are enabled
    if (this.captionsEnabled) {
      this.captionElement.textContent = caption.text;
    }

    this.captionTimer = 0;
    this.emit("dialog:caption", caption);
  }

  /**
   * Hide caption
   */
  hideCaption() {
    this.captionElement.textContent = "";
  }

  /**
   * Handle dialog completion
   */
  handleDialogComplete() {
    this.hideCaption();
    this.isPlaying = false;

    const completedDialog = this.currentDialog;
    this.currentDialog = null;
    this.currentAudio = null;
    this.currentProgressTriggers = [];
    this.progressTriggersFired.clear();

    // Check if this dialog should trigger choices
    if (completedDialog && this.dialogChoiceUI) {
      // Import dialogChoiceData dynamically to check for choices
      import("./dialogChoiceData.js").then((module) => {
        const choiceConfig = module.getChoiceForDialog(completedDialog.id);

        if (choiceConfig) {
          console.log(
            `DialogManager: Showing choices for dialog "${completedDialog.id}"`
          );
          const choiceData = module.buildChoiceData(choiceConfig);
          this.dialogChoiceUI.showChoices(choiceData);
        } else {
          // No choices, call onComplete if available
          this.handleOnComplete(completedDialog);
        }
      });
    } else {
      // No choice UI, call onComplete if available
      this.handleOnComplete(completedDialog);
    }

    this.emit("dialog:complete", completedDialog);

    if (this.onCompleteCallback) {
      this.onCompleteCallback(completedDialog);
      this.onCompleteCallback = null;
    }
  }

  /**
   * Handle onComplete callback for dialog
   * @param {Object} dialog - Completed dialog
   */
  handleOnComplete(dialog) {
    if (dialog && dialog.onComplete && this.gameManager) {
      if (typeof dialog.onComplete === "function") {
        try {
          console.log(
            `DialogManager: Calling onComplete for dialog "${dialog.id}"`
          );
          dialog.onComplete(this.gameManager);
        } catch (error) {
          console.error(
            `DialogManager: Error in onComplete for dialog "${dialog.id}":`,
            error
          );
        }
      }
    }
  }

  /**
   * Update method - call in animation loop
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    // Update pending delayed dialogs
    if (this.pendingDialogs.size > 0) {
      for (const [dialogId, pending] of this.pendingDialogs) {
        pending.timer += dt;

        // Check if delay has elapsed and no dialog is currently playing
        if (pending.timer >= pending.delay && !this.isPlaying) {
          console.log(`DialogManager: Playing delayed dialog "${dialogId}"`);
          this.pendingDialogs.delete(dialogId);
          this._playDialogImmediate(pending.dialogData, pending.onComplete);
          break; // Only play one dialog per frame
        }
      }
    }

    // Progress-based state triggers for current dialog (configured in dialogData)
    if (
      this.isPlaying &&
      this.currentAudio &&
      this.gameManager &&
      this.currentProgressTriggers.length > 0
    ) {
      try {
        const dur = this.currentAudio.duration
          ? this.currentAudio.duration()
          : 0;
        const pos = this.currentAudio.seek ? this.currentAudio.seek() : 0;
        if (dur > 0) {
          const prog = pos / dur; // 0..1
          for (let i = 0; i < this.currentProgressTriggers.length; i++) {
            const trig = this.currentProgressTriggers[i];
            if (prog >= trig.progress && !this.progressTriggersFired.has(i)) {
              // Fire state change
              if (typeof this.gameManager.setState === "function") {
                this.gameManager.setState({ currentState: trig.state });
              }
              this.progressTriggersFired.add(i);
            }
          }
        }
      } catch (e) {
        // ignore seek() while not loaded
      }
    }

    // Update current dialog captions
    if (!this.isPlaying || this.captionQueue.length === 0) {
      return;
    }

    this.captionTimer += dt;

    const currentCaption = this.captionQueue[this.captionIndex];
    if (currentCaption && this.captionTimer >= currentCaption.duration) {
      // Move to next caption
      this.captionIndex++;

      if (this.captionIndex < this.captionQueue.length) {
        this.showCaption(this.captionQueue[this.captionIndex]);
      } else {
        // No more captions
        this.hideCaption();

        // If there's no audio (caption-only dialog), complete the dialog now
        if (!this.currentAudio) {
          this.handleDialogComplete();
        }
      }
    }
  }

  /**
   * Set caption styling
   * @param {Object} styles - CSS style object
   */
  setCaptionStyle(styles) {
    Object.assign(this.captionElement.style, styles);
  }

  /**
   * Apply default caption styling
   */
  applyDefaultCaptionStyle() {
    this.setCaptionStyle({
      fontFamily: "PXCountryTypewriter, Arial, sans-serif",
      fontSize: "32px",
      background: "transparent",
      padding: "20px 40px",
      color: "#ffffff",
      textShadow: "2px 2px 8px rgba(0, 0, 0, 0.9), 0 0 20px rgba(0, 0, 0, 0.7)",
      maxWidth: "90%",
      lineHeight: "1.4",
    });
  }

  /**
   * Set audio volume
   * @param {number} volume - Volume level (0-1)
   */
  setVolume(volume) {
    const clamped = Math.max(0, Math.min(1, volume));
    this.baseVolume = clamped;
    this.updateVolume();
  }

  /**
   * Update volume based on SFX manager
   */
  updateVolume() {
    if (this.sfxManager) {
      this.audioVolume = this.baseVolume * this.sfxManager.getMasterVolume();
    } else {
      this.audioVolume = this.baseVolume;
    }

    if (this.currentAudio) {
      this.currentAudio.volume(this.audioVolume);
    }
  }

  /**
   * Set whether captions are enabled
   * @param {boolean} enabled - Whether to show captions
   */
  setCaptionsEnabled(enabled) {
    this.captionsEnabled = enabled;

    // If disabling captions and currently showing one, hide it
    if (!enabled) {
      this.hideCaption();
    }
    // If enabling captions and currently playing dialog, show current caption
    else if (enabled && this.isPlaying && this.captionQueue.length > 0) {
      const currentCaption = this.captionQueue[this.captionIndex];
      if (currentCaption) {
        this.captionElement.textContent = currentCaption.text;
      }
    }
  }

  /**
   * Check if dialog is currently playing
   * @returns {boolean}
   */
  isDialogPlaying() {
    return this.isPlaying;
  }

  /**
   * Check if a dialog is pending (scheduled with delay)
   * @param {string} dialogId - Dialog ID to check
   * @returns {boolean}
   */
  isDialogPending(dialogId) {
    return this.pendingDialogs.has(dialogId);
  }

  /**
   * Check if any dialog is pending
   * @returns {boolean}
   */
  hasDialogsPending() {
    return this.pendingDialogs.size > 0;
  }

  /**
   * Get the duration of a dialog audio file
   * @param {string} dialogId - Dialog ID to get duration for
   * @returns {number} Duration in seconds, or 0 if not available
   */
  getDialogDuration(dialogId) {
    // Check if audio is preloaded
    if (this.preloadedAudio.has(dialogId)) {
      const howl = this.preloadedAudio.get(dialogId);
      const duration = howl.duration();
      console.log(`DialogManager: Dialog "${dialogId}" duration: ${duration}s`);
      return duration || 0;
    }

    // Check if it's a deferred dialog
    if (this.deferredDialogs.has(dialogId)) {
      console.warn(
        `DialogManager: Dialog "${dialogId}" is deferred and not yet loaded, cannot get duration`
      );
      return 0;
    }

    console.warn(`DialogManager: Dialog "${dialogId}" not found`);
    return 0;
  }

  /**
   * Get the total duration of a dialog including captions
   * @param {Object} dialogData - Dialog data object
   * @returns {number} Total duration in seconds
   */
  getTotalDialogDuration(dialogData) {
    // If there's audio, use audio duration
    if (dialogData.audio && this.preloadedAudio.has(dialogData.id)) {
      return this.getDialogDuration(dialogData.id);
    }

    // Otherwise, sum caption durations
    if (dialogData.captions && dialogData.captions.length > 0) {
      const captionDuration = dialogData.captions.reduce(
        (total, caption) => total + (caption.duration || 0),
        0
      );
      return captionDuration;
    }

    return 0;
  }

  /**
   * Add event listener
   * @param {string} event - Event name
   * @param {function} callback - Callback function
   */
  on(event, callback) {
    if (this.eventListeners[event]) {
      this.eventListeners[event].push(callback);
    }
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {function} callback - Callback function
   */
  off(event, callback) {
    if (this.eventListeners[event]) {
      const index = this.eventListeners[event].indexOf(callback);
      if (index > -1) {
        this.eventListeners[event].splice(index, 1);
      }
    }
  }

  /**
   * Emit an event
   * @param {string} event - Event name
   * @param {...any} args - Arguments to pass to callbacks
   */
  emit(event, ...args) {
    if (this.eventListeners[event]) {
      this.eventListeners[event].forEach((callback) => callback(...args));
    }
  }

  /**
   * Clean up resources
   */
  destroy() {
    this.stopDialog();

    // Clear pending dialogs
    this.pendingDialogs.clear();

    if (this.captionElement && this.captionElement.parentNode) {
      this.captionElement.parentNode.removeChild(this.captionElement);
    }

    this.eventListeners = {};
  }
}

export default DialogManager;
