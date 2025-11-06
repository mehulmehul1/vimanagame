import { Howl, Howler } from "howler";
import { checkCriteria } from "./utils/criteriaHelper.js";
import { Logger } from "./utils/logger.js";

/**
 * SFXManager - Manages all sound effects with master volume control
 *
 * Features:
 * - Centralized SFX volume control using Howler.js
 * - Register/unregister individual sound effects
 * - Master volume that scales all SFX
 * - Individual sound volume relative to master
 * - Support for spatial/positional audio
 */

class SFXManager {
  constructor(options = {}) {
    this.masterVolume = options.masterVolume || 0.5;
    this.loadingScreen = options.loadingScreen || null; // For progress tracking
    this.sounds = new Map(); // Map of id -> {howl, baseVolume}
    this.dialogManager = null; // Will be set externally
    this.lightManager = options.lightManager || null; // LightManager for reactive lights
    this.gameManager = null;

    // Logger for debug messages
    this.logger = new Logger("SFXManager", false);

    // Track sounds that have been played once (for playOnce functionality)
    this.playedSounds = new Set();

    // Store deferred sounds for later loading
    this.deferredSounds = new Map(); // Store sound data for later loading

    // Delayed playback support
    this.pendingSounds = new Map(); // Map of soundId -> { soundId, timer, delay }

    // Loop delay support
    this.loopDelays = new Map(); // Map of soundId -> { delay, timer, active }
    this.loopingSounds = new Map(); // Map of soundId -> Howler sound instance ID

    // Set global Howler volume (we'll manage individual sounds separately)
    Howler.volume(1.0);
  }

  /**
   * Set game manager and register event listeners
   * @param {GameManager} gameManager - The game manager instance
   */
  setGameManager(gameManager) {
    this.gameManager = gameManager;

    // State change handler
    const handleStateChange = (newState, oldState) => {
      // Check all sounds with criteria and play/stop based on current state
      this.updateSoundsForState(newState);
    };

    // Listen for state changes
    this.gameManager.on("state:changed", handleStateChange);

    // Handle initial state
    const currentState = this.gameManager.getState();
    handleStateChange(currentState, null);

    this.logger.log("Event listeners registered and initial state handled");
  }

  /**
   * Update all sounds based on current game state
   * Checks criteria for each sound and plays/stops accordingly
   * @param {Object} state - Current game state
   */
  updateSoundsForState(state) {
    if (!state || !this._data) return;

    // Check both registered sounds and deferred sounds
    const allSoundIds = new Set([
      ...this.sounds.keys(),
      ...this.deferredSounds.keys(),
    ]);

    for (const id of allSoundIds) {
      const def = this._data[id];
      if (!def || !def.criteria) continue;

      const matchesCriteria = checkCriteria(state, def.criteria);
      const isPlaying = this.isPlaying(id);
      const hasPlayedOnce = this.playedSounds.has(id);
      const isPending = this.pendingSounds.has(id);
      const isDeferred = this.deferredSounds.has(id);

      // If criteria matches and sound is not playing
      if (matchesCriteria && !isPlaying && !isPending) {
        // Check playOnce - skip if already played
        if (def.playOnce && hasPlayedOnce) {
          continue;
        }

        // If sound is deferred, load it on-demand first
        if (isDeferred) {
          this.logger.log(`Loading deferred sound "${id}" on-demand for state ${state.currentState}`);
          // play() will handle loading deferred sounds
        }

        // Check if this sound has a delay
        const delay = def.delay || 0;

        if (delay > 0) {
          // Schedule delayed playback
          this.scheduleDelayedSound(id, delay);
        } else {
          // Play immediately (will load deferred sounds automatically)
          try {
            this.play(id);
            if (def.playOnce) {
              this.playedSounds.add(id);
            }
          } catch (e) {
            // Ignore autoplay errors, user gesture will trigger later
            this.logger.warn(`Failed to play sound "${id}":`, e);
          }
        }
      }
      // If criteria doesn't match and sound is playing or pending, stop/cancel it
      else if (!matchesCriteria) {
        if (isPlaying) {
          this.stop(id);
        }
        if (isPending) {
          this.cancelDelayedSound(id);
        }
      }
    }
  }

  /**
   * Schedule a sound to play after a delay
   * @param {string} soundId - Sound ID to schedule
   * @param {number} delay - Delay in seconds
   * @private
   */
  scheduleDelayedSound(soundId, delay) {
    this.logger.log(`Scheduling sound "${soundId}" with ${delay}s delay`);

    this.pendingSounds.set(soundId, {
      soundId,
      timer: 0,
      delay,
    });
  }

  /**
   * Cancel a pending delayed sound
   * @param {string} soundId - Sound ID to cancel
   */
  cancelDelayedSound(soundId) {
    if (this.pendingSounds.has(soundId)) {
      this.logger.log(`Cancelled delayed sound "${soundId}"`);
      this.pendingSounds.delete(soundId);
    }
  }

  /**
   * Cancel all pending delayed sounds
   */
  cancelAllDelayedSounds() {
    if (this.pendingSounds.size > 0) {
      this.logger.log(`Cancelling ${this.pendingSounds.size} pending sound(s)`);
      this.pendingSounds.clear();
    }
  }

  /**
   * Check if a sound is pending (scheduled with delay)
   * @param {string} soundId - Sound ID to check
   * @returns {boolean}
   */
  isSoundPending(soundId) {
    return this.pendingSounds.has(soundId);
  }

  /**
   * Check if any sounds are pending
   * @returns {boolean}
   */
  hasSoundsPending() {
    return this.pendingSounds.size > 0;
  }

  /**
   * Register a sound effect
   * @param {string} id - Unique identifier for this sound
   * @param {Howl|Object} howl - Howler.js Howl instance or object with setVolume method
   * @param {number} baseVolume - Base volume for this sound (0-1), defaults to 1.0
   */
  registerSound(id, howl, baseVolume = 1.0) {
    this.sounds.set(id, {
      howl,
      baseVolume,
      isProxy:
        typeof howl.volume !== "function" &&
        typeof howl.setVolume === "function",
    });

    // Apply current master volume
    this.updateSoundVolume(id);

    this.logger.log(`Registered sound "${id}" with base volume ${baseVolume}`);
  }

  /**
   * Unregister a sound effect
   * @param {string} id - Sound identifier
   */
  unregisterSound(id) {
    this.sounds.delete(id);
  }

  /**
   * Set master SFX volume (affects all sounds)
   * @param {number} volume - Master volume (0-1)
   */
  setMasterVolume(volume) {
    this.masterVolume = Math.max(0, Math.min(1, volume));

    // Update all registered sounds
    for (const [id] of this.sounds) {
      this.updateSoundVolume(id);
    }

    // Update dialog volume if dialog manager is registered
    if (this.dialogManager && this.dialogManager.updateVolume) {
      this.dialogManager.updateVolume();
    }
  }

  /**
   * Register dialog manager to be controlled by SFX volume
   * @param {DialogManager} dialogManager - Dialog manager instance
   */
  registerDialogManager(dialogManager) {
    this.dialogManager = dialogManager;
  }

  /**
   * Bulk-register sounds from a data object (e.g., sfxData.js)
   * @param {Record<string, any>} soundsData - Map of id -> sound descriptor
   */
  async registerSoundsFromData(soundsData) {
    if (!soundsData) return;
    // Keep a reference to the raw data definitions for state-driven rules
    this._data = soundsData;

    // In debug mode, check which sounds match the debug state and force them to preload
    let debugState = null;
    const matchingSoundIds = new Set();
    const { isDebugSpawnActive, getDebugSpawnState } = await import(
      "./utils/debugSpawner.js"
    );
    const { checkCriteria } = await import("./utils/criteriaHelper.js");

    if (isDebugSpawnActive()) {
      debugState = getDebugSpawnState();
      if (debugState) {
        // Check which sounds match the debug state
        Object.values(soundsData).forEach((sound) => {
          if (sound.criteria && checkCriteria(debugState, sound.criteria)) {
            matchingSoundIds.add(sound.id);
          }
        });
        if (matchingSoundIds.size > 0) {
          this.logger.log(
            `[Debug] Forcing preload for ${matchingSoundIds.size} matching SFX (state: ${debugState.currentState}): ${Array.from(matchingSoundIds).join(", ")}`
          );
        }
      }
    }

    Object.values(soundsData).forEach((sound) => {
      // In debug mode, force preload if this sound matches the debug state
      const shouldPreload =
        matchingSoundIds.has(sound.id)
          ? true
          : sound.preload !== undefined
          ? sound.preload
          : true; // Default to true for backwards compatibility

      // If preload is false, defer loading (file won't be fetched until after loading screen)
      if (!shouldPreload) {
        this.deferredSounds.set(sound.id, sound);
        this.logger.log(`Deferred loading for sound "${sound.id}"`);
        return;
      }

      // Register with loading screen if available and preloading
      if (this.loadingScreen && shouldPreload) {
        this.loadingScreen.registerTask(`sfx_${sound.id}`, 1);
      }

      // Check if this sound has a loop delay - if so, we'll manage looping manually
      const hasLoopDelay = sound.loop && sound.loopDelay > 0;

      const howl = new Howl({
        src: sound.src,
        loop: hasLoopDelay ? false : sound.loop, // Disable native loop if using loopDelay
        volume: sound.volume,
        ...(sound.rate !== undefined && { rate: sound.rate }),
        preload: shouldPreload,
        onload: () => {
          this.logger.log(`Loaded sound "${sound.id}"`);
          if (this.loadingScreen && shouldPreload) {
            this.loadingScreen.completeTask(`sfx_${sound.id}`);
          }
        },
        onloaderror: (id, error) => {
          this.logger.error(`Failed to load sound "${sound.id}":`, error);
          if (this.loadingScreen && shouldPreload) {
            this.loadingScreen.completeTask(`sfx_${sound.id}`);
          }
        },
        onend: hasLoopDelay ? () => this._handleLoopEnd(sound.id) : undefined,
      });

      // Apply spatial attributes after creation
      if (sound.spatial) {
        if (sound.position) {
          howl.pos(sound.position.x, sound.position.y, sound.position.z);
        }
        if (sound.pannerAttr) howl.pannerAttr(sound.pannerAttr);
      }

      this.registerSound(sound.id, howl, sound.volume ?? 1.0);

      // Register loop delay if present
      if (hasLoopDelay) {
        this.loopDelays.set(sound.id, {
          delay: sound.loopDelay,
          timer: 0,
          active: false,
        });
        this.logger.log(
          `Registered loop delay of ${sound.loopDelay}s for sound "${sound.id}"`
        );
      }

      // Request audio-reactive light creation from lightManager if configured
      if (
        sound.reactiveLight &&
        sound.reactiveLight.enabled &&
        this.lightManager
      ) {
        // Apply offset to sound position for reactive light
        const lightConfig = { ...sound.reactiveLight };
        if (sound.position && lightConfig.position) {
          lightConfig.position = {
            x: sound.position.x + (lightConfig.position.x || 0),
            y: sound.position.y + (lightConfig.position.y || 0),
            z: sound.position.z + (lightConfig.position.z || 0),
          };
        }

        this.lightManager.createReactiveLight(sound.id, howl, lightConfig);
      }
    });
  }

  /**
   * Load deferred sounds (called after loading screen)
   */
  loadDeferredSounds() {
    this.logger.log(`Loading ${this.deferredSounds.size} deferred sounds`);
    for (const [id, sound] of this.deferredSounds) {
      // Check if this sound has a loop delay - if so, we'll manage looping manually
      const hasLoopDelay = sound.loop && sound.loopDelay > 0;

      const howl = new Howl({
        src: sound.src,
        loop: hasLoopDelay ? false : sound.loop, // Disable native loop if using loopDelay
        volume: sound.volume,
        ...(sound.rate !== undefined && { rate: sound.rate }),
        preload: true, // Load now
        onload: () => {
          this.logger.log(`Loaded deferred sound "${sound.id}"`);
        },
        onloaderror: (id, error) => {
          this.logger.error(
            `Failed to load deferred sound "${sound.id}":`,
            error
          );
        },
        onend: hasLoopDelay ? () => this._handleLoopEnd(sound.id) : undefined,
      });

      // Apply spatial attributes after creation
      if (sound.spatial) {
        if (sound.position) {
          howl.pos(sound.position.x, sound.position.y, sound.position.z);
        }
        if (sound.pannerAttr) howl.pannerAttr(sound.pannerAttr);
      }

      this.registerSound(sound.id, howl, sound.volume ?? 1.0);

      // Register loop delay if present
      if (hasLoopDelay) {
        this.loopDelays.set(sound.id, {
          delay: sound.loopDelay,
          timer: 0,
          active: false,
        });
        this.logger.log(
          `Registered loop delay of ${sound.loopDelay}s for deferred sound "${sound.id}"`
        );
      }

      // Request audio-reactive light creation from lightManager if configured
      if (
        sound.reactiveLight &&
        sound.reactiveLight.enabled &&
        this.lightManager
      ) {
        // Apply offset to sound position for reactive light
        const lightConfig = { ...sound.reactiveLight };
        if (sound.position && lightConfig.position) {
          lightConfig.position = {
            x: sound.position.x + (lightConfig.position.x || 0),
            y: sound.position.y + (lightConfig.position.y || 0),
            z: sound.position.z + (lightConfig.position.z || 0),
          };
        }

        this.lightManager.createReactiveLight(sound.id, howl, lightConfig);
      }
    }
    this.deferredSounds.clear();
  }

  /**
   * Get current master volume
   * @returns {number}
   */
  getMasterVolume() {
    return this.masterVolume;
  }

  /**
   * Update a specific sound's volume based on master and base volumes
   * @param {string} id - Sound identifier
   */
  updateSoundVolume(id) {
    const soundData = this.sounds.get(id);
    if (!soundData) return;

    const { howl, baseVolume, isProxy } = soundData;
    const finalVolume = baseVolume * this.masterVolume;

    if (howl) {
      if (isProxy) {
        // Legacy proxy object with setVolume method (e.g., breathing system)
        howl.setVolume(finalVolume);
      } else {
        // Howler.js Howl instance
        howl.volume(finalVolume);
      }
    }
  }

  /**
   * Set base volume for a specific sound (will be scaled by master)
   * @param {string} id - Sound identifier
   * @param {number} baseVolume - Base volume (0-1)
   */
  setSoundBaseVolume(id, baseVolume) {
    const soundData = this.sounds.get(id);
    if (!soundData) return;

    soundData.baseVolume = Math.max(0, Math.min(1, baseVolume));
    this.updateSoundVolume(id);
  }

  /**
   * Get a registered sound (Howl instance)
   * @param {string} id - Sound identifier
   * @returns {Howl|null}
   */
  getSound(id) {
    const soundData = this.sounds.get(id);
    return soundData ? soundData.howl : null;
  }

  /**
   * Play a sound by ID
   * @param {string} id - Sound identifier
   * @returns {number|null|Promise<number|null>} Sound ID from Howler (for stopping specific instances)
   */
  async play(id) {
    // Check if sound is deferred and needs to be loaded on-demand
    if (!this.sounds.has(id) && this.deferredSounds.has(id)) {
      this.logger.log(`Loading deferred sound "${id}" on-demand`);
      const sound = this.deferredSounds.get(id);

      // Check if this sound has a loop delay - if so, we'll manage looping manually
      const hasLoopDelay = sound.loop && sound.loopDelay > 0;

      const howl = new Howl({
        src: sound.src,
        loop: hasLoopDelay ? false : sound.loop, // Disable native loop if using loopDelay
        volume: sound.volume,
        ...(sound.rate !== undefined && { rate: sound.rate }),
        preload: true, // Load now
        onend: hasLoopDelay ? () => this._handleLoopEnd(sound.id) : undefined,
      });

      // Wait for the sound to load using Howler's event system
      await new Promise((resolve) => {
        howl.once("load", () => {
          this.logger.log(`Loaded on-demand sound "${id}"`);
          resolve();
        });
        howl.once("loaderror", (loadId, error) => {
          this.logger.error(`Failed to load on-demand sound "${id}":`, error);
          resolve(); // Resolve anyway to prevent hanging
        });
      });

      // Apply spatial attributes after loading
      if (sound.spatial) {
        if (sound.position) {
          howl.pos(sound.position.x, sound.position.y, sound.position.z);
        }
        if (sound.pannerAttr) howl.pannerAttr(sound.pannerAttr);
      }

      this.registerSound(sound.id, howl, sound.volume ?? 1.0);

      // Register loop delay if present
      if (hasLoopDelay) {
        this.loopDelays.set(sound.id, {
          delay: sound.loopDelay,
          timer: 0,
          active: false,
        });
        this.logger.log(
          `Registered loop delay of ${sound.loopDelay}s for on-demand sound "${sound.id}"`
        );
      }

      // Request audio-reactive light creation from lightManager if configured
      if (
        sound.reactiveLight &&
        sound.reactiveLight.enabled &&
        this.lightManager
      ) {
        const lightConfig = { ...sound.reactiveLight };
        if (sound.position && lightConfig.position) {
          lightConfig.position = {
            x: sound.position.x + (lightConfig.position.x || 0),
            y: sound.position.y + (lightConfig.position.y || 0),
            z: sound.position.z + (lightConfig.position.z || 0),
          };
        }
        this.lightManager.createReactiveLight(sound.id, howl, lightConfig);
      }

      this.deferredSounds.delete(id);
    }

    const soundData = this.sounds.get(id);
    if (soundData && soundData.howl) {
      if (soundData.isProxy) {
        // Proxy objects don't have play method
        this.logger.warn(`Cannot play proxy object "${id}"`);
        return null;
      }
      const instanceId = soundData.howl.play();

      // Track the instance if this sound has a loop delay
      if (this.loopDelays.has(id)) {
        this.loopingSounds.set(id, instanceId);
      }

      return instanceId;
    }
    return null;
  }

  /**
   * Stop a sound by ID
   * @param {string} id - Sound identifier
   * @param {number} soundId - Optional: specific sound instance ID from play()
   */
  stop(id, soundId = null) {
    const soundData = this.sounds.get(id);
    if (soundData && soundData.howl) {
      if (soundData.isProxy) {
        // Proxy objects don't have stop method
        this.logger.warn(`Cannot stop proxy object "${id}"`);
        return;
      }
      if (soundId !== null) {
        soundData.howl.stop(soundId);
      } else {
        soundData.howl.stop();
      }

      // Clean up loop delay state
      if (this.loopDelays.has(id)) {
        const loopData = this.loopDelays.get(id);
        loopData.active = false;
        loopData.timer = 0;
      }
      this.loopingSounds.delete(id);
    }
  }

  /**
   * Stop all sounds
   */
  stopAll() {
    for (const [id, soundData] of this.sounds) {
      if (soundData.howl) {
        soundData.howl.stop();
      }
    }

    // Clean up all loop delay states
    for (const [id, loopData] of this.loopDelays) {
      loopData.active = false;
      loopData.timer = 0;
    }
    this.loopingSounds.clear();
  }

  /**
   * Handle the end of a sound with loop delay
   * @param {string} soundId - Sound identifier
   * @private
   */
  _handleLoopEnd(soundId) {
    const loopData = this.loopDelays.get(soundId);
    if (!loopData) return;

    // Start the delay timer
    loopData.active = true;
    loopData.timer = 0;
    this.logger.log(
      `Starting loop delay for "${soundId}" (${loopData.delay}s)`
    );
  }

  /**
   * Check if a sound is currently playing
   * @param {string} id - Sound identifier
   * @returns {boolean}
   */
  isPlaying(id) {
    const soundData = this.sounds.get(id);
    if (soundData && soundData.howl && !soundData.isProxy) {
      return soundData.howl.playing();
    }
    return false;
  }

  /**
   * Fade a sound's volume
   * @param {string} id - Sound identifier
   * @param {number} from - Starting volume (0-1)
   * @param {number} to - Target volume (0-1)
   * @param {number} duration - Duration in milliseconds
   * @param {number} soundId - Optional: specific sound instance ID
   */
  fade(id, from, to, duration, soundId = null) {
    const soundData = this.sounds.get(id);
    if (soundData && soundData.howl) {
      if (soundData.isProxy) {
        // Proxy objects don't have fade method
        this.logger.warn(`Cannot fade proxy object "${id}"`);
        return;
      }
      const fromScaled = from * this.masterVolume;
      const toScaled = to * this.masterVolume;

      if (soundId !== null) {
        soundData.howl.fade(fromScaled, toScaled, duration, soundId);
      } else {
        soundData.howl.fade(fromScaled, toScaled, duration);
      }
    }
  }

  /**
   * Get all registered sound IDs
   * @returns {Array<string>}
   */
  getSoundIds() {
    return Array.from(this.sounds.keys());
  }

  /**
   * Update method - call in animation loop to process delayed sounds
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    // Update pending delayed sounds
    if (this.pendingSounds.size > 0) {
      for (const [soundId, pending] of this.pendingSounds) {
        pending.timer += dt;

        // Check if delay has elapsed
        if (pending.timer >= pending.delay) {
          this.logger.log(`Playing delayed sound "${soundId}"`);
          this.pendingSounds.delete(soundId);

          // Get the sound definition to check playOnce
          const def = this._data?.[soundId];

          try {
            this.play(soundId);
            if (def?.playOnce) {
              this.playedSounds.add(soundId);
            }
          } catch (e) {
            // Ignore autoplay errors
            this.logger.warn(`Failed to play delayed sound "${soundId}"`, e);
          }
          break; // Only play one sound per frame
        }
      }
    }

    // Update loop delays
    if (this.loopDelays.size > 0) {
      for (const [soundId, loopData] of this.loopDelays) {
        if (!loopData.active) continue;

        loopData.timer += dt;

        // Check if loop delay has elapsed
        if (loopData.timer >= loopData.delay) {
          this.logger.log(`Replaying sound "${soundId}" after loop delay`);
          loopData.active = false;
          loopData.timer = 0;

          try {
            this.play(soundId);
          } catch (e) {
            this.logger.warn(
              `Failed to replay sound "${soundId}" after loop delay`,
              e
            );
          }
          break; // Only play one loop per frame
        }
      }
    }
  }

  /**
   * Clean up all sounds
   */
  destroy() {
    this.stopAll();

    // Clear pending sounds
    this.pendingSounds.clear();

    // Clear loop delay data
    this.loopDelays.clear();
    this.loopingSounds.clear();

    // Clean up sounds
    for (const [id, soundData] of this.sounds) {
      if (soundData.howl && !soundData.isProxy) {
        soundData.howl.unload();
      }
    }
    this.sounds.clear();
  }
}

export default SFXManager;
