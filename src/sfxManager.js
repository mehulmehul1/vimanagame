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

    // Store prefetched audio files (blobs) for iOS to avoid Howl pool exhaustion
    this.prefetchedAudio = new Map(); // Map of id -> { blob, blobUrl, soundData, size }

    // Prefetch budget system (iOS)
    this.prefetchBudgetMax = 25 * 1024 * 1024; // 25MB in bytes
    this.prefetchBudgetUsed = 0; // Current budget usage in bytes
    this.prefetchQueue = []; // Queue of sounds waiting to be prefetched (sorted by priority)

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
          this.logger.log(
            `Loading deferred sound "${id}" on-demand for state ${state.currentState}`
          );
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
            `[Debug] Forcing preload for ${
              matchingSoundIds.size
            } matching SFX (state: ${debugState.currentState}): ${Array.from(
              matchingSoundIds
            ).join(", ")}`
          );
        }
      }
    }

    // Check if we're on iOS - iOS has strict audio pool limits
    // gameManager might not be set yet, so check window.gameManager or state directly
    const isIOS =
      this.gameManager?.getState()?.isIOS ||
      (typeof window !== "undefined" &&
        window.gameManager?.getState()?.isIOS) ||
      false;

    Object.values(soundsData).forEach((sound) => {
      // In debug mode, force preload if this sound matches the debug state
      let shouldPreload = matchingSoundIds.has(sound.id)
        ? true
        : sound.preload !== undefined
        ? sound.preload
        : true; // Default to true for backwards compatibility

      // On iOS, add to prefetch queue instead of immediately prefetching
      if (isIOS && shouldPreload) {
        // Calculate priority based on earliest game state where criteria matches
        const priority = this._calculateSoundPriority(sound);
        this.prefetchQueue.push({ sound, priority });
        // Sort queue by priority (lower priority number = earlier in game = higher priority)
        this.prefetchQueue.sort((a, b) => a.priority - b.priority);
        this.logger.log(
          `iOS: Added sound "${sound.id}" to prefetch queue (priority: ${priority})`
        );
        // Process queue asynchronously (don't block)
        this._processPrefetchQueue();
        return; // Don't create Howl instance yet
      }

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
   * Calculate priority for a sound based on earliest game state where criteria matches
   * Lower number = earlier in game = higher priority
   * @param {Object} sound - Sound data
   * @returns {number} Priority value (lower = higher priority)
   * @private
   */
  _calculateSoundPriority(sound) {
    if (!sound.criteria) {
      // No criteria = always available, low priority
      return 9999;
    }

    // Import GAME_STATES to check priority
    // We'll use a simple heuristic: check if criteria references currentState
    const criteria = sound.criteria;

    // If criteria has currentState, use that value as priority
    if (criteria.currentState !== undefined) {
      if (typeof criteria.currentState === "number") {
        return criteria.currentState;
      } else if (criteria.currentState.$gte !== undefined) {
        // Use the minimum state value
        return criteria.currentState.$gte;
      } else if (
        criteria.currentState.$in !== undefined &&
        Array.isArray(criteria.currentState.$in)
      ) {
        // Use the minimum state from the array
        return Math.min(...criteria.currentState.$in);
      } else if (criteria.currentState.$eq !== undefined) {
        return criteria.currentState.$eq;
      }
    }

    // Default: medium priority
    return 50;
  }

  /**
   * Process prefetch queue, prefetching sounds when budget allows
   * @private
   */
  async _processPrefetchQueue() {
    // Don't process if already processing or no queue items
    if (this._processingPrefetchQueue || this.prefetchQueue.length === 0) {
      return;
    }

    this._processingPrefetchQueue = true;

    // Get current game state to check if criteria have passed
    const currentState = this.gameManager?.getState() || {};
    const { couldCriteriaStillMatch } = await import("./utils/criteriaHelper.js");

    while (this.prefetchQueue.length > 0) {
      // Check if we have budget available
      const availableBudget = this.prefetchBudgetMax - this.prefetchBudgetUsed;
      if (availableBudget <= 0) {
        // No budget available, wait for assets to be cleared
        this.logger.log(
          `Prefetch budget exhausted (${(
            this.prefetchBudgetUsed /
            1024 /
            1024
          ).toFixed(2)}MB / ${(this.prefetchBudgetMax / 1024 / 1024).toFixed(
            2
          )}MB), waiting for assets to clear`
        );
        break;
      }

      // Get next item from queue
      const { sound } = this.prefetchQueue.shift();

      // Skip if already prefetched or registered
      if (this.prefetchedAudio.has(sound.id) || this.sounds.has(sound.id)) {
        continue;
      }

      // Skip if criteria have already passed (e.g., in debug spawn mode)
      if (sound.criteria && !couldCriteriaStillMatch(currentState, sound.criteria)) {
        this.logger.log(
          `Skipping prefetch for sound "${sound.id}" - criteria have already passed (currentState: ${currentState.currentState})`
        );
        continue;
      }

      // Prefetch the sound
      await this._prefetchSound(sound);
    }

    this._processingPrefetchQueue = false;
  }

  /**
   * Prefetch an audio file using fetch() (doesn't use Howl audio pool)
   * @param {Object} sound - Sound data
   * @private
   */
  async _prefetchSound(sound) {
    const audioSrc = Array.isArray(sound.src) ? sound.src[0] : sound.src;

    // Register with loading screen if available
    if (this.loadingScreen) {
      this.loadingScreen.registerTask(`sfx_${sound.id}`, 1);
    }

    try {
      const response = await fetch(audioSrc);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      // Check Content-Length header for size estimate
      const contentLength = response.headers.get("Content-Length");
      const estimatedSize = contentLength ? parseInt(contentLength, 10) : null;

      const blob = await response.blob();
      const actualSize = blob.size;

      // Check if we have budget for this file
      const availableBudget = this.prefetchBudgetMax - this.prefetchBudgetUsed;
      if (actualSize > availableBudget) {
        this.logger.warn(
          `Sound "${sound.id}" (${(actualSize / 1024 / 1024).toFixed(
            2
          )}MB) exceeds available budget (${(
            availableBudget /
            1024 /
            1024
          ).toFixed(2)}MB), deferring`
        );
        // Fall back to deferred loading
        this.deferredSounds.set(sound.id, sound);
        if (this.loadingScreen) {
          this.loadingScreen.completeTask(`sfx_${sound.id}`);
        }
        return;
      }

      const blobUrl = URL.createObjectURL(blob);

      // Store prefetched data with size
      this.prefetchedAudio.set(sound.id, {
        blob,
        blobUrl,
        soundData: sound,
        size: actualSize,
      });

      // Update budget
      this.prefetchBudgetUsed += actualSize;

      this.logger.log(
        `Prefetched sound "${sound.id}" (${(actualSize / 1024 / 1024).toFixed(
          2
        )}MB, budget: ${(this.prefetchBudgetUsed / 1024 / 1024).toFixed(
          2
        )}MB / ${(this.prefetchBudgetMax / 1024 / 1024).toFixed(2)}MB)`
      );

      if (this.loadingScreen) {
        this.loadingScreen.completeTask(`sfx_${sound.id}`);
      }

      // Process more items from queue if budget allows
      this._processPrefetchQueue();
    } catch (error) {
      this.logger.error(`Failed to prefetch sound "${sound.id}":`, error);
      if (this.loadingScreen) {
        this.loadingScreen.completeTask(`sfx_${sound.id}`);
      }
      // Fall back to deferred loading
      this.deferredSounds.set(sound.id, sound);
    }
  }

  /**
   * Create Howl instance from prefetched blob
   * @param {string} id - Sound ID
   * @param {Object} prefetched - Prefetched data { blob, blobUrl, soundData }
   * @returns {Promise<Howl>} Promise that resolves when Howl is loaded
   * @private
   */
  _createHowlFromPrefetched(id, prefetched) {
    const { blobUrl, soundData } = prefetched;
    const hasLoopDelay = soundData.loop && soundData.loopDelay > 0;

    return new Promise((resolve, reject) => {
      const howl = new Howl({
        src: [blobUrl], // Use blob URL instead of original src
        loop: hasLoopDelay ? false : soundData.loop,
        volume: soundData.volume,
        ...(soundData.rate !== undefined && { rate: soundData.rate }),
        preload: true,
        onload: () => {
          this.logger.log(`Howl loaded from prefetched blob for sound "${id}"`);

          // Free budget when converting blob to Howl
          const prefetched = this.prefetchedAudio.get(id);
          if (prefetched && prefetched.size) {
            this.prefetchBudgetUsed -= prefetched.size;
            this.logger.log(
              `Freed ${(prefetched.size / 1024 / 1024).toFixed(
                2
              )}MB from budget (now: ${(
                this.prefetchBudgetUsed /
                1024 /
                1024
              ).toFixed(2)}MB / ${(
                this.prefetchBudgetMax /
                1024 /
                1024
              ).toFixed(2)}MB)`
            );
            // Revoke blob URL to free memory
            URL.revokeObjectURL(prefetched.blobUrl);
          }
          this.prefetchedAudio.delete(id);

          // Process more items from queue now that budget is available
          this._processPrefetchQueue();

          // Apply spatial attributes after creation
          if (soundData.spatial) {
            if (soundData.position) {
              howl.pos(
                soundData.position.x,
                soundData.position.y,
                soundData.position.z
              );
            }
            if (soundData.pannerAttr) howl.pannerAttr(soundData.pannerAttr);
          }

          this.registerSound(id, howl, soundData.volume ?? 1.0);

          // Register loop delay if present
          if (hasLoopDelay) {
            this.loopDelays.set(id, {
              delay: soundData.loopDelay,
              timer: 0,
              active: false,
            });
            this.logger.log(
              `Registered loop delay of ${soundData.loopDelay}s for sound "${id}"`
            );
          }

          // Request audio-reactive light creation from lightManager if configured
          if (
            soundData.reactiveLight &&
            soundData.reactiveLight.enabled &&
            this.lightManager
          ) {
            const lightConfig = { ...soundData.reactiveLight };
            if (soundData.position && lightConfig.position) {
              lightConfig.position = {
                x: soundData.position.x + (lightConfig.position.x || 0),
                y: soundData.position.y + (lightConfig.position.y || 0),
                z: soundData.position.z + (lightConfig.position.z || 0),
              };
            }
            this.lightManager.createReactiveLight(id, howl, lightConfig);
          }

          resolve(howl);
        },
        onloaderror: (loadId, error) => {
          this.logger.error(
            `Failed to load Howl from prefetched blob for sound "${id}":`,
            error
          );
          reject(error);
        },
        onend: hasLoopDelay
          ? () => this._handleLoopEnd(soundData.id)
          : undefined,
      });
    });
  }

  /**
   * Load deferred sounds (called after loading screen)
   */
  async loadDeferredSounds() {
    const isIOS = this.gameManager?.getState()?.isIOS || false;
    const currentState = this.gameManager?.getState() || {};
    const { couldCriteriaStillMatch } = await import("./utils/criteriaHelper.js");
    
    this.logger.log(
      `Loading ${this.deferredSounds.size} deferred sounds${
        isIOS ? " (iOS: sequential loading)" : ""
      }`
    );

    // First, create Howl instances from prefetched audio (iOS)
    if (this.prefetchedAudio.size > 0) {
      this.logger.log(
        `Creating Howl instances from ${this.prefetchedAudio.size} prefetched sounds`
      );
      for (const [id, prefetched] of this.prefetchedAudio) {
        // Skip if criteria have already passed
        if (prefetched.soundData?.criteria && !couldCriteriaStillMatch(currentState, prefetched.soundData.criteria)) {
          this.logger.log(
            `Skipping prefetched sound "${id}" - criteria have already passed (currentState: ${currentState.currentState})`
          );
          // Free budget
          if (prefetched && prefetched.size) {
            this.prefetchBudgetUsed -= prefetched.size;
            URL.revokeObjectURL(prefetched.blobUrl);
          }
          this.prefetchedAudio.delete(id);
          continue;
        }
        
        try {
          await this._createHowlFromPrefetched(id, prefetched);
          // Budget is freed in onload callback
          // Small delay between Howl creations on iOS
          if (isIOS) {
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
        } catch (error) {
          this.logger.error(
            `Failed to create Howl from prefetched sound "${id}":`,
            error
          );
          // Free budget on error
          if (prefetched && prefetched.size) {
            this.prefetchBudgetUsed -= prefetched.size;
            URL.revokeObjectURL(prefetched.blobUrl);
          }
          // Fall back to deferred loading
          this.deferredSounds.set(id, prefetched.soundData);
        }
      }
      // Clear prefetched map (budget already freed in onload callbacks)
      this.prefetchedAudio.clear();

      // Process queue now that budget is available
      this._processPrefetchQueue();
    }

    // Filter deferred sounds to skip those whose criteria have passed
    const soundsToLoad = [];
    for (const [id, sound] of this.deferredSounds) {
      if (sound.criteria && !couldCriteriaStillMatch(currentState, sound.criteria)) {
        this.logger.log(
          `Skipping deferred sound "${id}" - criteria have already passed (currentState: ${currentState.currentState})`
        );
        continue;
      }
      soundsToLoad.push([id, sound]);
    }

    // On iOS, load sounds sequentially with delays to avoid exhausting audio pool
    if (isIOS) {
      for (const [id, sound] of soundsToLoad) {
        await this._loadDeferredSound(id, sound);
        // Small delay between loads to prevent audio pool exhaustion
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
    } else {
      // On other platforms, load in parallel (faster)
      for (const [id, sound] of soundsToLoad) {
        this._loadDeferredSound(id, sound);
      }
    }
    this.deferredSounds.clear();
  }

  /**
   * Load a single deferred sound
   * @param {string} id - Sound ID
   * @param {Object} sound - Sound data
   * @private
   */
  _loadDeferredSound(id, sound) {
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
    // Check if sound is prefetched (iOS) - create Howl from blob
    if (!this.sounds.has(id) && this.prefetchedAudio.has(id)) {
      this.logger.log(`Creating Howl from prefetched blob for sound "${id}"`);
      const prefetched = this.prefetchedAudio.get(id);
      try {
        await this._createHowlFromPrefetched(id, prefetched);
        // Budget is freed in onload callback of _createHowlFromPrefetched
      } catch (error) {
        this.logger.error(
          `Failed to create Howl from prefetched sound "${id}":`,
          error
        );
        // Free budget even on error
        if (prefetched && prefetched.size) {
          this.prefetchBudgetUsed -= prefetched.size;
          URL.revokeObjectURL(prefetched.blobUrl);
        }
        this.prefetchedAudio.delete(id);
        // Process queue now that budget is available
        this._processPrefetchQueue();
        // Fall back to deferred loading
        if (this.deferredSounds.has(id)) {
          // Will be handled by deferred loading below
        } else {
          this.deferredSounds.set(id, prefetched.soundData);
        }
      }
    }

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
