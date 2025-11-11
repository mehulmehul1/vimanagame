import { Howl } from "howler";
import { Logger } from "./utils/logger.js";
import { checkCriteria } from "./utils/criteriaHelper.js";
import { VIEWMASTER_OVERHEAT_THRESHOLD } from "./dialogData.js";

/**
 * DialogManager - Manages dialog audio playback and synchronized captions
 *
 * Supports:
 * - Multiple concurrent dialogs playing simultaneously
 * - Unified caption queue that interleaves captions from all active dialogs chronologically
 * - Audio tracks from multiple dialogs playing at the same time (not stopped when new dialog starts)
 * - Video-synced captions
 * - State-based auto-play
 * - Dialog chaining via playNext
 */
class DialogManager {
  constructor(options = {}) {
    this.sfxManager = options.sfxManager;
    this.gameManager = options.gameManager;
    this.videoManager = options.videoManager;
    this.loadingScreen = options.loadingScreen;

    // Create caption element
    this.captionElement = options.captionElement || this.createCaptionElement();
    if (!options.captionElement) {
      document.body.appendChild(this.captionElement);
    }

    // Apply caption styling
    if (options.captionStyle) {
      this.setCaptionStyle(options.captionStyle);
    } else {
      this.applyDefaultCaptionStyle();
    }

    this.baseVolume = options.audioVolume || 0.8;
    this.audioVolume = this.baseVolume;

    // Support multiple concurrent dialogs
    this.activeDialogs = new Map(); // dialogId -> {dialogData, audio, videoId, videoStartTime, startTime, onComplete, progressTriggers, progressTriggersFired}

    // Unified caption queue that merges captions from all active dialogs
    this.unifiedCaptionQueue = []; // Sorted array of {caption, dialogId, absoluteStartTime, absoluteEndTime}
    this.currentCaptionIndex = 0;
    this.captionTimer = 0;
    this.queueNeedsRebuild = false; // Flag to track if queue needs rebuilding

    this.isPlaying = false; // True if any dialog is active
    this.captionsEnabled = true; // Default to captions enabled

    // Fade out support (per-dialog fade out can be added later if needed)
    this.isFadingOut = false;
    this.fadeOutDuration = 0;
    this.fadeOutTimer = 0;
    this.fadeOutStartVolume = 0;

    // Logger for debug messages
    this.logger = new Logger("DialogManager", false);

    // Delayed playback support
    this.pendingDialogs = new Map(); // Map of dialogId -> { dialogData, onComplete, timer, delay }

    // Preloading support
    this.preloadedAudio = new Map(); // Map of dialogId -> Howl instance
    this.deferredDialogs = new Map(); // Map of dialogId -> dialog data for later loading
    this.prefetchedAudio = new Map(); // Map of dialogId -> { blob, blobUrl, dialogData, size } for iOS

    // Prefetch budget system (iOS) - separate from SFXManager
    // Note: DialogManager has its own budget pool separate from SFXManager
    this.prefetchBudgetMax = 25 * 1024 * 1024; // 25MB in bytes
    this.prefetchBudgetUsed = 0; // Current budget usage in bytes
    this.prefetchQueue = []; // Queue of dialogs waiting to be prefetched (sorted by priority)

    // Cache getDialogsForState import for continuous evaluation
    this._getDialogsForState = null;
    this._dialogTracks = null;
    import("./dialogData.js").then((module) => {
      this._getDialogsForState = module.getDialogsForState;
      this._dialogTracks = module.dialogTracks;
    });

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
   * Update absolute times for video-synced captions without full rebuild
   * This is much faster than rebuilding the entire queue every frame
   * @private
   */
  _updateVideoCaptionTimes() {
    // Only update entries that need it (video-synced captions)
    for (let i = 0; i < this.unifiedCaptionQueue.length; i++) {
      const entry = this.unifiedCaptionQueue[i];
      if (entry.sourceType === "video") {
        const dialogInfo = this.activeDialogs.get(entry.dialogId);
        if (dialogInfo && dialogInfo.videoId && this.videoManager) {
          const videoPlayer = this.videoManager.getVideoPlayer(
            dialogInfo.videoId
          );
          if (videoPlayer && videoPlayer.video && videoPlayer.isPlaying) {
            // Use stored videoStartTime - DO NOT recalculate it here
            // videoStartTime should be set when dialog is first triggered and never change
            let videoStartTime = dialogInfo.videoStartTime;
            if (!videoStartTime || videoStartTime === null) {
              // Fallback: calculate if somehow missing, but log a warning
              videoStartTime =
                Date.now() - videoPlayer.video.currentTime * 1000;
              dialogInfo.videoStartTime = videoStartTime;
              this.logger.warn(
                `videoStartTime was null for "${
                  entry.dialogId
                }" - recalculated to ${new Date(
                  videoStartTime
                ).toLocaleTimeString()} (this should have been set when dialog triggered!)`
              );
            }

            // Recalculate absolute times based on stored videoStartTime and relative time
            // relativeTime never changes (it's the caption's position in the video)
            // videoStartTime should never change (it's when the video started)
            entry.absoluteStartTime =
              videoStartTime + entry.relativeTime * 1000;
            entry.absoluteEndTime =
              entry.absoluteStartTime + (entry.caption.duration || 3.0) * 1000;
          }
        }
      }
    }

    // Re-sort queue after updating times (but this is cheaper than full rebuild)
    this.unifiedCaptionQueue.sort(
      (a, b) => a.absoluteStartTime - b.absoluteStartTime
    );
  }

  /**
   * Rebuild unified caption queue from all active dialogs
   * Captions are sorted chronologically by their absolute start time
   * Only called when dialogs are added/removed, not every frame
   * @private
   */
  _rebuildUnifiedCaptionQueue() {
    this.unifiedCaptionQueue = [];

    for (const [dialogId, dialogInfo] of this.activeDialogs.entries()) {
      const { dialogData, startTime, videoId, videoStartTime } = dialogInfo;
      const captions = dialogData.captions || [];

      if (videoId && this.videoManager) {
        // Video-synced: use video currentTime for timing
        // Captions have startTime relative to video start (0s = video start)
        // If startTime is not provided, calculate it sequentially from previous caption
        const videoPlayer = this.videoManager.getVideoPlayer(videoId);
        if (videoPlayer && videoPlayer.video && videoPlayer.isPlaying) {
          // Use stored videoStartTime (set when dialog was triggered in _playDialogImmediate)
          // This is the absolute time when the video actually started playing (after delay)
          // It should never change - if it's null, calculate it once and store it
          let calculatedVideoStartTime = videoStartTime;
          if (!calculatedVideoStartTime || calculatedVideoStartTime === null) {
            // Calculate when video actually started: now minus current playback time
            // When video first starts (after delay), currentTime = 0, so this = Date.now()
            // If video has been playing, currentTime represents elapsed time since video started
            // So: videoStartTime = now - elapsedTime = time when video actually started
            const videoCurrentTime = videoPlayer.video.currentTime;
            calculatedVideoStartTime = Date.now() - videoCurrentTime * 1000;

            // Store it in dialogInfo so it persists across rebuilds
            dialogInfo.videoStartTime = calculatedVideoStartTime;

            this.logger.log(
              `Calculated videoStartTime for "${dialogId}" (video "${videoId}"): video currentTime=${videoCurrentTime.toFixed(
                2
              )}s, videoStartTime=${new Date(
                calculatedVideoStartTime
              ).toLocaleTimeString()}`
            );
          }

          const videoReferenceTime = calculatedVideoStartTime;

          let cumulativeVideoTime = 0;
          for (const caption of captions) {
            // If startTime is explicitly provided, use it; otherwise calculate sequentially
            let captionStartTime;
            if (caption.startTime !== undefined) {
              captionStartTime = caption.startTime;
            } else {
              // No startTime provided - use cumulative time (sequential from previous)
              captionStartTime = cumulativeVideoTime;
            }

            const captionEndTime = captionStartTime + (caption.duration || 3.0);

            // Always update cumulative for next caption (whether or not this one had explicit startTime)
            // This ensures sequential captions continue from where the previous one ended
            cumulativeVideoTime = captionEndTime;

            // Calculate absolute times: videoStartTime + relative caption time
            // videoStartTime is when the video actually started playing (after delay)
            // captionStartTime is relative to video start (0s = video start)
            const absoluteStartTime =
              videoReferenceTime + captionStartTime * 1000;
            const absoluteEndTime =
              absoluteStartTime + (caption.duration || 3.0) * 1000;

            const currentTime = Date.now();
            const isAlreadyExpired = currentTime >= absoluteEndTime;
            const isCurrentlyActive =
              currentTime >= absoluteStartTime && currentTime < absoluteEndTime;

            // Only log if caption is currently active or if it's the first time adding
            if (isCurrentlyActive || !isAlreadyExpired) {
              this.logger.log(
                `Adding caption "${caption.text.substring(
                  0,
                  30
                )}..." for "${dialogId}" - relative: ${captionStartTime.toFixed(
                  2
                )}s, absolute: ${new Date(
                  absoluteStartTime
                ).toLocaleTimeString()} - ${new Date(
                  absoluteEndTime
                ).toLocaleTimeString()}${
                  isAlreadyExpired
                    ? " (already expired)"
                    : isCurrentlyActive
                    ? " (currently active)"
                    : ""
                }`
              );
            }

            // Always add caption to queue, even if expired
            // The display logic will handle showing/hiding based on timing
            this.unifiedCaptionQueue.push({
              caption,
              dialogId,
              absoluteStartTime,
              absoluteEndTime,
              sourceType: "video",
              relativeTime: captionStartTime,
            });
          }
        }
      } else if (dialogInfo.audio) {
        // Audio-synced: use audio position + startTime
        let audioPosition = 0;
        try {
          if (dialogInfo.audio.seek) {
            audioPosition = dialogInfo.audio.seek() || 0;
          }
        } catch (e) {
          // Audio might not be loaded yet
        }

        let cumulativeTime = 0;
        for (const caption of captions) {
          const captionDuration = caption.duration || 3.0;
          const absoluteStartTime = startTime + cumulativeTime * 1000;
          const absoluteEndTime = absoluteStartTime + captionDuration * 1000;

          // Only add captions that are current or future relative to audio
          if (audioPosition <= cumulativeTime + captionDuration) {
            this.unifiedCaptionQueue.push({
              caption,
              dialogId,
              absoluteStartTime,
              absoluteEndTime,
              sourceType: "audio",
              relativeTime: cumulativeTime,
            });
          }

          cumulativeTime += captionDuration;
        }
      } else {
        // Caption-only dialog (no audio, no videoId): use sequential timing from startTime
        let cumulativeTime = 0;
        for (const caption of captions) {
          const captionDuration = caption.duration || 3.0;
          const absoluteStartTime = startTime + cumulativeTime * 1000;
          const absoluteEndTime = absoluteStartTime + captionDuration * 1000;

          this.unifiedCaptionQueue.push({
            caption,
            dialogId,
            absoluteStartTime,
            absoluteEndTime,
            sourceType: "caption-only",
            relativeTime: cumulativeTime,
          });

          cumulativeTime += captionDuration;
        }
      }
    }

    // Sort by absolute start time
    this.unifiedCaptionQueue.sort(
      (a, b) => a.absoluteStartTime - b.absoluteStartTime
    );

    // Reset caption index to find current position
    // Find the first caption that should be shown now (active or upcoming)
    const currentTime = Date.now();
    this.currentCaptionIndex = this.unifiedCaptionQueue.length; // Default: beyond queue

    for (let i = 0; i < this.unifiedCaptionQueue.length; i++) {
      const caption = this.unifiedCaptionQueue[i];
      // Point to first caption that is currently active
      // A caption is active if current time is between its start and end times
      // This catches captions that started in the past but are still within their display duration
      const isCurrentlyActive =
        currentTime >= caption.absoluteStartTime &&
        currentTime < caption.absoluteEndTime;

      if (isCurrentlyActive) {
        this.currentCaptionIndex = i;
        this.showCaption(caption.caption);
        this.logger.log(
          `Showing active caption "${caption.caption.text.substring(
            0,
            30
          )}..." (${caption.dialogId}) at queue rebuild`
        );
        break;
      }
    }

    // If no active caption was found, hide the caption (in case it was showing from a removed dialog)
    if (this.currentCaptionIndex >= this.unifiedCaptionQueue.length) {
      this.hideCaption();
    }
  }

  /**
   * Preload dialog audio files
   * @param {Object} dialogsData - Dialog data object (from dialogData.js)
   */
  async preloadDialogs(dialogsData) {
    if (!dialogsData) return;

    // In debug mode, check which dialogs match the debug state and force them to preload
    let debugState = null;
    const matchingDialogIds = new Set();
    const { isDebugSpawnActive, getDebugSpawnState } = await import(
      "./utils/debugSpawner.js"
    );
    const { getDialogsForState } = await import("./dialogData.js");

    if (isDebugSpawnActive()) {
      debugState = getDebugSpawnState();
      if (debugState) {
        const matchingDialogs = getDialogsForState(debugState, new Set());
        matchingDialogs.forEach((dialog) => {
          matchingDialogIds.add(dialog.id);
        });
        if (matchingDialogIds.size > 0) {
          this.logger.log(
            `[Debug] Forcing preload for ${
              matchingDialogIds.size
            } matching dialogs (state: ${
              debugState.currentState
            }): ${Array.from(matchingDialogIds).join(", ")}`
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

    Object.values(dialogsData).forEach((dialog) => {
      if (!dialog.audio) return; // Skip dialogs without audio

      // In debug mode, force preload if this dialog matches the debug state
      let shouldPreload = matchingDialogIds.has(dialog.id)
        ? true
        : dialog.preload !== undefined
        ? dialog.preload
        : true; // Default to true for backwards compatibility

      // On iOS, add to prefetch queue instead of immediately prefetching
      if (isIOS && shouldPreload) {
        // Calculate priority based on earliest game state where criteria matches
        const priority = this._calculateDialogPriority(dialog);
        this.prefetchQueue.push({ dialog, priority });
        // Sort queue by priority (lower priority number = earlier in game = higher priority)
        this.prefetchQueue.sort((a, b) => a.priority - b.priority);
        this.logger.log(
          `iOS: Added dialog "${dialog.id}" to prefetch queue (priority: ${priority})`
        );
        // Process queue asynchronously (don't block)
        this._processPrefetchQueue();
        return; // Don't create Howl instance yet
      }

      // If preload is false, defer loading (file won't be fetched until after loading screen)
      if (!shouldPreload) {
        this.deferredDialogs.set(dialog.id, dialog);
        this.logger.log(`Deferred loading for dialog "${dialog.id}"`);
        return;
      }

      // Register with loading screen if available and preloading
      if (this.loadingScreen && shouldPreload) {
        this.loadingScreen.registerTask(`dialog_${dialog.id}`, 1);
      }

      // Preload the audio
      const howl = new Howl({
        src: [dialog.audio],
        volume: this.audioVolume,
        preload: true,
        onload: () => {
          this.logger.log(`Loaded dialog "${dialog.id}"`);
          if (this.loadingScreen && shouldPreload) {
            this.loadingScreen.completeTask(`dialog_${dialog.id}`);
          }
        },
        onloaderror: (id, error) => {
          this.logger.error(`Failed to load dialog "${dialog.id}":`, error);
          if (this.loadingScreen && shouldPreload) {
            this.loadingScreen.completeTask(`dialog_${dialog.id}`);
          }
        },
      });

      this.preloadedAudio.set(dialog.id, howl);
      this.logger.log(`Preloaded dialog "${dialog.id}"`);
    });
  }

  /**
   * Calculate priority for a dialog based on earliest game state where criteria matches
   * Lower number = earlier in game = higher priority
   * @param {Object} dialog - Dialog data
   * @returns {number} Priority value (lower = higher priority)
   * @private
   */
  _calculateDialogPriority(dialog) {
    if (!dialog.criteria) {
      // No criteria = always available, low priority
      return 9999;
    }

    const criteria = dialog.criteria;

    // If criteria has currentState, use that value as priority
    if (criteria.currentState !== undefined) {
      if (typeof criteria.currentState === "number") {
        return criteria.currentState;
      } else if (criteria.currentState.$gte !== undefined) {
        return criteria.currentState.$gte;
      } else if (
        criteria.currentState.$in !== undefined &&
        Array.isArray(criteria.currentState.$in)
      ) {
        return Math.min(...criteria.currentState.$in);
      } else if (criteria.currentState.$eq !== undefined) {
        return criteria.currentState.$eq;
      }
    }

    // Default: medium priority
    return 50;
  }

  /**
   * Process prefetch queue, prefetching dialogs when budget allows
   * @private
   */
  async _processPrefetchQueue() {
    // Don't process if already processing or no queue items
    if (this._processingPrefetchQueue || this.prefetchQueue.length === 0) {
      return;
    }

    this._processingPrefetchQueue = true;

    while (this.prefetchQueue.length > 0) {
      // Check if we have budget available
      const availableBudget = this.prefetchBudgetMax - this.prefetchBudgetUsed;
      if (availableBudget <= 0) {
        this.logger.log(
          `Dialog prefetch budget exhausted (${(
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
      const { dialog } = this.prefetchQueue.shift();

      // Skip if already prefetched or registered
      if (
        this.prefetchedAudio.has(dialog.id) ||
        this.preloadedAudio.has(dialog.id)
      ) {
        continue;
      }

      // Prefetch the dialog
      await this._prefetchDialog(dialog);
    }

    this._processingPrefetchQueue = false;
  }

  /**
   * Prefetch a dialog audio file using fetch() (doesn't use Howl audio pool)
   * @param {Object} dialog - Dialog data
   * @private
   */
  async _prefetchDialog(dialog) {
    if (!dialog.audio) return;

    // Register with loading screen if available
    if (this.loadingScreen) {
      this.loadingScreen.registerTask(`dialog_${dialog.id}`, 1);
    }

    try {
      const response = await fetch(dialog.audio);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const blob = await response.blob();
      const actualSize = blob.size;

      // Check if we have budget for this file
      const availableBudget = this.prefetchBudgetMax - this.prefetchBudgetUsed;
      if (actualSize > availableBudget) {
        this.logger.warn(
          `Dialog "${dialog.id}" (${(actualSize / 1024 / 1024).toFixed(
            2
          )}MB) exceeds available budget (${(
            availableBudget /
            1024 /
            1024
          ).toFixed(2)}MB), deferring`
        );
        // Fall back to deferred loading
        this.deferredDialogs.set(dialog.id, dialog);
        if (this.loadingScreen) {
          this.loadingScreen.completeTask(`dialog_${dialog.id}`);
        }
        return;
      }

      const blobUrl = URL.createObjectURL(blob);

      // Store prefetched data with size
      this.prefetchedAudio.set(dialog.id, {
        blob,
        blobUrl,
        dialogData: dialog,
        size: actualSize,
      });

      // Update budget
      this.prefetchBudgetUsed += actualSize;

      this.logger.log(
        `Prefetched dialog "${dialog.id}" (${(actualSize / 1024 / 1024).toFixed(
          2
        )}MB, budget: ${(this.prefetchBudgetUsed / 1024 / 1024).toFixed(
          2
        )}MB / ${(this.prefetchBudgetMax / 1024 / 1024).toFixed(2)}MB)`
      );

      if (this.loadingScreen) {
        this.loadingScreen.completeTask(`dialog_${dialog.id}`);
      }

      // Process more items from queue if budget allows
      this._processPrefetchQueue();
    } catch (error) {
      this.logger.error(`Failed to prefetch dialog "${dialog.id}":`, error);
      if (this.loadingScreen) {
        this.loadingScreen.completeTask(`dialog_${dialog.id}`);
      }
      // Fall back to deferred loading
      this.deferredDialogs.set(dialog.id, dialog);
    }
  }

  /**
   * Create Howl instance from prefetched blob
   * @param {string} dialogId - Dialog ID
   * @param {Object} prefetched - Prefetched data { blob, blobUrl, dialogData, size }
   * @returns {Promise<Howl>} Promise that resolves when Howl is loaded
   * @private
   */
  _createHowlFromPrefetched(dialogId, prefetched) {
    const { blobUrl } = prefetched;

    return new Promise((resolve, reject) => {
      const howl = new Howl({
        src: [blobUrl], // Use blob URL instead of original src
        volume: this.audioVolume,
        preload: true,
        onload: () => {
          this.logger.log(
            `Howl loaded from prefetched blob for dialog "${dialogId}"`
          );

          // Free budget when converting blob to Howl
          if (prefetched && prefetched.size) {
            this.prefetchBudgetUsed -= prefetched.size;
            this.logger.log(
              `Freed ${(prefetched.size / 1024 / 1024).toFixed(
                2
              )}MB from dialog budget (now: ${(
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
          this.prefetchedAudio.delete(dialogId);

          // Process more items from queue now that budget is available
          this._processPrefetchQueue();

          this.preloadedAudio.set(dialogId, howl);
          resolve(howl);
        },
        onloaderror: (loadId, error) => {
          this.logger.error(
            `Failed to load Howl from prefetched blob for dialog "${dialogId}":`,
            error
          );
          // Free budget on error
          if (prefetched && prefetched.size) {
            this.prefetchBudgetUsed -= prefetched.size;
            URL.revokeObjectURL(prefetched.blobUrl);
          }
          this.prefetchedAudio.delete(dialogId);
          this._processPrefetchQueue();
          reject(error);
        },
      });
    });
  }

  /**
   * Load deferred dialogs (called after loading screen)
   */
  async loadDeferredDialogs() {
    const isIOS = this.gameManager?.getState()?.isIOS || false;

    // First, create Howl instances from prefetched audio (iOS)
    if (this.prefetchedAudio.size > 0) {
      this.logger.log(
        `Creating Howl instances from ${this.prefetchedAudio.size} prefetched dialogs`
      );
      for (const [dialogId, prefetched] of this.prefetchedAudio) {
        try {
          await this._createHowlFromPrefetched(dialogId, prefetched);
          // Budget is freed in onload callback
          // Small delay between Howl creations on iOS
          if (isIOS) {
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
        } catch (error) {
          this.logger.error(
            `Failed to create Howl from prefetched dialog "${dialogId}":`,
            error
          );
          // Free budget on error
          if (prefetched && prefetched.size) {
            this.prefetchBudgetUsed -= prefetched.size;
            URL.revokeObjectURL(prefetched.blobUrl);
          }
          // Fall back to deferred loading
          this.deferredDialogs.set(dialogId, prefetched.dialogData);
        }
      }
      // Clear prefetched map (budget already freed in onload callbacks)
      this.prefetchedAudio.clear();

      // Process queue now that budget is available
      this._processPrefetchQueue();
    }

    if (this.deferredDialogs.size === 0) {
      return;
    }

    this.logger.log(
      `Loading ${this.deferredDialogs.size} deferred dialogs${
        isIOS ? " (iOS: sequential loading)" : ""
      }`
    );

    // On iOS, load dialogs sequentially with delays to avoid exhausting audio pool
    if (isIOS) {
      for (const [dialogId, dialog] of this.deferredDialogs) {
        if (!dialog.audio) continue; // Skip dialogs without audio
        await this._loadDeferredDialog(dialogId, dialog);
        // Small delay between loads to prevent audio pool exhaustion
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
    } else {
      // On other platforms, load in parallel (faster)
      this.deferredDialogs.forEach((dialog, dialogId) => {
        if (!dialog.audio) return; // Skip dialogs without audio
        this._loadDeferredDialog(dialogId, dialog);
      });
    }
    this.deferredDialogs.clear();
  }

  /**
   * Load a single deferred dialog
   * @param {string} dialogId - Dialog ID
   * @param {Object} dialog - Dialog data
   * @private
   */
  _loadDeferredDialog(dialogId, dialog) {
    if (!dialog.audio) return; // Skip dialogs without audio

    const howl = new Howl({
      src: [dialog.audio],
      volume: this.audioVolume,
      preload: true,
      onload: () => {
        this.logger.log(`Loaded deferred dialog "${dialogId}"`);
      },
      onloaderror: (id, error) => {
        this.logger.error(
          `Failed to load deferred dialog "${dialogId}":`,
          error
        );
      },
    });

    this.preloadedAudio.set(dialogId, howl);
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
    if (!this.gameManager) {
      this.logger.warn(
        "Cannot setup state listener - gameManager not available"
      );
      return;
    }

    this.logger.log("Setting up state listener...");

    // Track played dialogs for "once" functionality
    this.playedDialogs = new Set();

    // Store a reference to check when dialog data is loaded
    this._dialogDataLoaded = false;
    this._pendingStateChecks = [];

    // Register listeners for video play events to trigger video-synced dialogs immediately
    // This ensures dialogs trigger as soon as videos start (e.g., via playNext)
    // rather than waiting for update loop detection
    import("./dialogData.js").then(
      ({ dialogTracks, getDialogsForState, VIEWMASTER_OVERHEAT_THRESHOLD }) => {
        this.logger.log("Dialog data loaded, registering state listener");
        this._dialogDataLoaded = true;
        // Cache for continuous checking in update()
        this._getDialogsForState = getDialogsForState;

        // Register event listeners for each video-synced dialog
        // Also register for counterpart videos (base <-> Safari)
        const registeredEvents = new Set();
        for (const dialog of Object.values(dialogTracks)) {
          if (dialog.videoId && dialog.autoPlay) {
            const videoIdsToRegister = this._getCounterpartVideoIds(
              dialog.videoId
            );
            for (const vidId of videoIdsToRegister) {
              const eventName = `video:play:${vidId}`;
              if (!registeredEvents.has(eventName)) {
                this.gameManager.on(eventName, () => {
                  this._checkAndTriggerVideoDialog(vidId);
                });
                registeredEvents.add(eventName);
                this.logger.log(
                  `Registered video play listener for dialog "${dialog.id}" on event "${eventName}"`
                );
              }
            }
          }
        }

        // Listen for state changes
        const stateChangeHandler = (newState, oldState) => {
          // Only process if currentState actually changed (not just property updates)
          if (oldState?.currentState === newState?.currentState) {
            // Skip property-only updates (e.g., viewmasterInsanityIntensity changes)
            this.logger.log(
              `Skipping state change handler - currentState unchanged (${newState?.currentState}), only properties updated`
            );
            return;
          }

          // Debug: Log when overheat dialog index changes
          if (
            newState.viewmasterOverheatDialogIndex !==
            oldState?.viewmasterOverheatDialogIndex
          ) {
            this.logger.log(
              `Viewmaster overheat dialog index changed: ${oldState?.viewmasterOverheatDialogIndex} -> ${newState.viewmasterOverheatDialogIndex}, intensity: ${newState.viewmasterInsanityIntensity}`
            );
          }

          const matchingDialogs = getDialogsForState(
            newState,
            this.playedDialogs
          );

          this.logger.log(
            `State changed: ${oldState?.currentState} -> ${
              newState.currentState
            }, found ${
              matchingDialogs.length
            } matching dialog(s): ${matchingDialogs
              .map((d) => d.id)
              .join(", ")}`
          );

          // Debug: Check for overheat dialogs specifically
          if (
            newState.viewmasterInsanityIntensity >=
              VIEWMASTER_OVERHEAT_THRESHOLD &&
            newState.viewmasterOverheatDialogIndex !== null &&
            newState.viewmasterOverheatDialogIndex !== undefined
          ) {
            const overheatDialogs = matchingDialogs.filter(
              (d) => d.id === "coleUghGimmeASec" || d.id === "coleUghItsTooMuch"
            );
            if (overheatDialogs.length > 0) {
              this.logger.log(
                `State change: Found ${
                  overheatDialogs.length
                } matching overheat dialog(s): ${overheatDialogs
                  .map((d) => d.id)
                  .join(", ")}`
              );
            } else {
              this.logger.log(
                `State change: Overheat criteria met but no dialogs matched. Index: ${newState.viewmasterOverheatDialogIndex}, Intensity: ${newState.viewmasterInsanityIntensity}, isViewmasterEquipped: ${newState.isViewmasterEquipped}`
              );
            }
          }

          // If there are matching dialogs for the new state
          if (matchingDialogs.length > 0) {
            // Process dialogs in priority order, play the first valid one
            // (Sequential behavior: highest priority dialog plays, interrupts current if needed)
            for (const dialog of matchingDialogs) {
              // Skip if already played and marked as "once" (check this FIRST)
              if (
                dialog.once &&
                this.playedDialogs &&
                this.playedDialogs.has(dialog.id)
              ) {
                this.logger.log(
                  `Skipping dialog "${dialog.id}" - already played (once: true)`
                );
                continue;
              }

              // Skip video-synced dialogs (handled in update loop when video is actually playing)
              // Video-synced dialogs should ONLY play when their video is playing, not from state changes
              if (dialog.videoId) {
                continue;
              }

              // Skip if this is already active
              if (this.activeDialogs.has(dialog.id)) {
                this.logger.log(
                  `Skipping dialog "${dialog.id}" - already active`
                );
                continue;
              }

              // Skip if already pending
              if (this.pendingDialogs.has(dialog.id)) {
                this.logger.log(
                  `Skipping dialog "${dialog.id}" - already pending`
                );
                continue;
              }

              // All dialogs can interrupt/run concurrently (add to active dialogs, don't stop existing)
              this.logger.log(
                `Auto-playing dialog "${dialog.id}" from state change`
              );

              // Only track in playedDialogs if marked as "once" (allows replay for once: false)
              if (dialog.once) {
                this.playedDialogs.add(dialog.id);
              }

              // Emit event for tracking
              this.gameManager.emit("dialog:trigger", dialog.id, dialog);

              // Play the dialog (will be scheduled with its delay if specified)
              this.playDialog(dialog, (completedDialog) => {
                this.gameManager.emit("dialog:finished", completedDialog);
              });

              // Break after playing first valid dialog (sequential behavior for state changes)
              break;
            }
          }
        };

        // Register the state change handler
        this.gameManager.on("state:changed", stateChangeHandler);

        this.logger.log("Event listeners registered");

        // Check initial state in case we're already in a state that should trigger dialogs
        const currentState = this.gameManager.getState();
        if (currentState) {
          this.logger.log(
            `Checking initial state: ${currentState.currentState}, isViewmasterEquipped: ${currentState.isViewmasterEquipped}`
          );
          const matchingDialogs = getDialogsForState(
            currentState,
            this.playedDialogs
          );
          this.logger.log(
            `Initial state check: found ${
              matchingDialogs.length
            } matching dialog(s): ${matchingDialogs
              .map((d) => d.id)
              .join(", ")}`
          );

          // Process matching dialogs for initial state
          if (matchingDialogs.length > 0) {
            for (const dialog of matchingDialogs) {
              // Skip if already played and marked as "once"
              if (
                dialog.once &&
                this.playedDialogs &&
                this.playedDialogs.has(dialog.id)
              ) {
                this.logger.log(
                  `Skipping initial dialog "${dialog.id}" - already played (once: true)`
                );
                continue;
              }

              // Skip video-synced dialogs (handled in update loop)
              if (dialog.videoId) {
                continue;
              }

              // Skip if this is already active
              if (this.activeDialogs.has(dialog.id)) {
                this.logger.log(
                  `Skipping initial dialog "${dialog.id}" - already active`
                );
                continue;
              }

              // Skip if already pending
              if (this.pendingDialogs.has(dialog.id)) {
                this.logger.log(
                  `Skipping initial dialog "${dialog.id}" - already pending`
                );
                continue;
              }

              this.logger.log(
                `Auto-playing dialog "${dialog.id}" from initial state check`
              );

              // Only track in playedDialogs if marked as "once"
              if (dialog.once) {
                this.playedDialogs.add(dialog.id);
              }

              // Emit event for tracking
              this.gameManager.emit("dialog:trigger", dialog.id, dialog);

              // Play the dialog (will be scheduled with its delay if specified)
              this.playDialog(dialog, (completedDialog) => {
                this.gameManager.emit("dialog:finished", completedDialog);
              });

              // Break after playing first valid dialog (sequential behavior)
              break;
            }
          }
        }

        // Register the state change handler
        this.gameManager.on("state:changed", stateChangeHandler);

        // Process any pending state checks that happened before dialog data loaded
        while (this._pendingStateChecks.length > 0) {
          const { newState, oldState } = this._pendingStateChecks.shift();
          this.logger.log(
            "Processing pending state check that occurred before dialog data loaded"
          );
          stateChangeHandler(newState, oldState);
        }
      }
    );

    // Also register a temporary listener BEFORE async import completes
    // This will catch state changes that happen during initialization
    const tempStateHandler = (newState, oldState) => {
      // Only process if currentState actually changed
      if (oldState?.currentState === newState?.currentState) {
        return;
      }

      if (!this._dialogDataLoaded) {
        // Store for processing after dialog data loads
        this._pendingStateChecks.push({ newState, oldState });
        this.logger.log(
          `State changed to ${newState?.currentState} before dialog data loaded, queuing for later processing`
        );
      }
    };

    this.gameManager.on("state:changed", tempStateHandler);
  }

  /**
   * Create default caption element if none provided
   */
  createCaptionElement() {
    const caption = document.createElement("div");
    caption.id = "dialog-caption";
    caption.className = "dialog-caption";
    return caption;
  }

  /**
   * Play a dialog
   * @param {Object} dialogData - Dialog data object
   * @param {Function} onComplete - Optional callback when dialog completes
   */
  async playDialog(dialogData, onComplete = null) {
    if (!dialogData) {
      this.logger.warn("playDialog called with null/undefined dialogData");
      return;
    }

    const delay = dialogData.delay || 0;

    if (delay > 0) {
      this.scheduleDelayedDialog(dialogData, onComplete, delay);
    } else {
      this._playDialogImmediate(dialogData, onComplete);
    }
  }

  /**
   * Schedule a dialog to play after a delay
   * @param {Object} dialogData - Dialog data object
   * @param {Function} onComplete - Optional callback
   * @param {number} delay - Delay in seconds
   */
  scheduleDelayedDialog(dialogData, onComplete, delay) {
    const dialogId = dialogData.id;
    this.pendingDialogs.set(dialogId, {
      dialogData,
      onComplete,
      timer: 0,
      delay,
    });
  }

  /**
   * Cancel a delayed dialog
   * @param {string} dialogId - Dialog ID to cancel
   */
  cancelDelayedDialog(dialogId) {
    if (this.pendingDialogs.has(dialogId)) {
      this.pendingDialogs.delete(dialogId);
    }
  }

  /**
   * Cancel all pending delayed dialogs
   */
  cancelAllDelayedDialogs() {
    if (this.pendingDialogs.size > 0) {
      this.logger.log(
        `Cancelling ${this.pendingDialogs.size} pending dialog(s)`
      );
      this.pendingDialogs.clear();
    }
  }

  /**
   * Immediately play a dialog (internal method)
   * Adds dialog to activeDialogs instead of replacing currentDialog
   * Audio plays concurrently with other dialogs
   * @param {Object} dialogData - Dialog data object
   * @param {Function} onComplete - Optional callback
   * @private
   */
  async _playDialogImmediate(dialogData, onComplete = null) {
    const dialogId = dialogData.id;
    const dialogStartTime = Date.now();

    // Combine both callbacks: call dialogData.onComplete first, then the passed onComplete
    // This allows both the dialog's own onComplete and any external tracking to work together
    const finalOnComplete = (gameManager) => {
      // Call dialogData.onComplete first if it exists
      if (
        dialogData.onComplete &&
        typeof dialogData.onComplete === "function"
      ) {
        dialogData.onComplete(gameManager);
      }
      // Then call the passed onComplete if provided
      if (onComplete && typeof onComplete === "function") {
        onComplete(dialogData);
      }
    };

    // Only set finalOnComplete if at least one callback exists
    const hasOnComplete =
      (dialogData.onComplete && typeof dialogData.onComplete === "function") ||
      (onComplete && typeof onComplete === "function");
    const finalOnCompleteToUse = hasOnComplete ? finalOnComplete : null;

    // Store progress-based state triggers for this dialog
    const progressTriggers = [];
    const progressTriggersFired = new Set();
    if (dialogData) {
      // Single trigger form
      if (
        dialogData.progressStateTrigger &&
        typeof dialogData.progressStateTrigger === "object"
      ) {
        const t = dialogData.progressStateTrigger;
        if (typeof t.progress === "number" && t.state !== undefined) {
          progressTriggers.push({
            progress: Math.max(0, Math.min(1, t.progress)),
            state: t.state,
          });
        }
      }
      // Array form
      if (Array.isArray(dialogData.progressStateTriggers)) {
        dialogData.progressStateTriggers.forEach((t) => {
          if (t && typeof t.progress === "number" && t.state !== undefined) {
            progressTriggers.push({
              progress: Math.max(0, Math.min(1, t.progress)),
              state: t.state,
            });
          }
        });
      }
      // Sort triggers ascending by progress
      progressTriggers.sort((a, b) => a.progress - b.progress);
    }

    let audio = null;
    let videoId = null;
    let videoStartTime = null;

    // Handle video-synced captions
    if (dialogData.videoId) {
      videoId = dialogData.videoId;

      // Get video player and calculate videoStartTime
      // This should only be called when video is already playing (triggered by update loop)
      if (this.videoManager) {
        const videoPlayer = this.videoManager.getVideoPlayer(videoId);
        if (videoPlayer && videoPlayer.video && videoPlayer.isPlaying) {
          // Calculate when video actually started: now - current playback time
          // When video starts playing (after delay), currentTime = 0, so videoStartTime = Date.now()
          videoStartTime = Date.now() - videoPlayer.video.currentTime * 1000;
          this.logger.log(
            `Syncing captions to video "${videoId}" at video time ${videoPlayer.video.currentTime.toFixed(
              2
            )}s (videoStartTime: ${new Date(
              videoStartTime
            ).toLocaleTimeString()})`
          );
        } else {
          this.logger.warn(
            `Video player not found or not playing for "${videoId}"`
          );
        }
      } else {
        this.logger.warn(
          `VideoManager not available for video-synced dialog "${dialogId}"`
        );
      }
    }
    // Load and play audio (only if not video-synced)
    else if (dialogData.audio) {
      // Check if audio was preloaded or loaded from deferred
      // Check if dialog is prefetched (iOS) - create Howl from blob
      if (this.prefetchedAudio.has(dialogId)) {
        this.logger.log(
          `Creating Howl from prefetched blob for dialog "${dialogId}"`
        );
        const prefetched = this.prefetchedAudio.get(dialogId);
        try {
          audio = await this._createHowlFromPrefetched(dialogId, prefetched);
          // Budget is freed in onload callback of _createHowlFromPrefetched
        } catch (error) {
          this.logger.error(
            `Failed to create Howl from prefetched dialog "${dialogId}":`,
            error
          );
          // Free budget even on error
          if (prefetched && prefetched.size) {
            this.prefetchBudgetUsed -= prefetched.size;
            URL.revokeObjectURL(prefetched.blobUrl);
          }
          this.prefetchedAudio.delete(dialogId);
          // Process queue now that budget is available
          this._processPrefetchQueue();
          // Fall back to deferred loading
          this.deferredDialogs.set(dialogId, prefetched.dialogData);
        }
      }

      if (!audio && this.preloadedAudio.has(dialogId)) {
        this.logger.log(`Using preloaded audio for "${dialogId}"`);
        audio = this.preloadedAudio.get(dialogId);

        // Set onend callback for preloaded audio (it might not have one)
        audio.once("end", () => {
          this.logger.log(`Preloaded audio for "${dialogId}" ended`);
          this._handleDialogComplete(dialogId);
        });

        audio.volume(this.audioVolume);
        const playId = audio.play();
        if (!playId) {
          this.logger.error(
            `Failed to start playback for "${dialogId}" - Howl returned no sound ID`
          );
        } else {
          this.logger.log(
            `Playing preloaded audio for "${dialogId}" (sound ID: ${playId})`
          );
        }
      } else {
        // Check if this was a deferred dialog that hasn't been loaded yet
        if (this.deferredDialogs.has(dialogId)) {
          this.logger.log(`Loading deferred dialog "${dialogId}" on-demand`);
          const deferredDialog = this.deferredDialogs.get(dialogId);

          audio = new Howl({
            src: [deferredDialog.audio],
            volume: this.audioVolume,
            preload: true, // Explicitly load now
            onend: () => {
              this._handleDialogComplete(dialogId);
            },
            onloaderror: (id, error) => {
              this.logger.error(
                `Failed to load audio for "${dialogId}":`,
                error
              );
              this._handleDialogComplete(dialogId);
            },
          });

          // Remove from deferred map and add to preloaded
          this.deferredDialogs.delete(dialogId);
          this.preloadedAudio.set(dialogId, audio);
        } else {
          // Fallback: Load on-demand from dialogData
          this.logger.log(`Loading audio on-demand for "${dialogId}"`);
          audio = new Howl({
            src: [dialogData.audio],
            volume: this.audioVolume,
            preload: true, // Explicitly load now
            onend: () => {
              this._handleDialogComplete(dialogId);
            },
            onloaderror: (id, error) => {
              this.logger.error(
                `Failed to load audio for "${dialogId}":`,
                error
              );
              this._handleDialogComplete(dialogId);
            },
            onplayerror: (id, error) => {
              this.logger.error(
                `Failed to play audio for "${dialogId}":`,
                error
              );
              this._handleDialogComplete(dialogId);
            },
          });
        }

        // Wait for the audio to load using Howler's event system
        const audioState = audio.state ? audio.state() : "unloaded";
        this.logger.log(`Audio state for "${dialogId}": ${audioState}`);

        if (audioState !== "loaded") {
          this.logger.log(`Waiting for audio to load for "${dialogId}"...`);
          await new Promise((resolve) => {
            const timeout = setTimeout(() => {
              this.logger.error(`Audio load timeout for "${dialogId}"`);
              resolve();
            }, 10000); // 10 second timeout

            audio.once("load", () => {
              clearTimeout(timeout);
              this.logger.log(`Loaded on-demand dialog "${dialogId}"`);
              resolve();
            });
            audio.once("loaderror", (id, error) => {
              clearTimeout(timeout);
              this.logger.error(
                `Failed to load on-demand dialog "${dialogId}":`,
                error,
                `Audio path: ${dialogData.audio}`
              );
              resolve(); // Resolve anyway to prevent hanging
            });
          });
        }

        const playId = audio.play();
        if (!playId) {
          this.logger.error(
            `Failed to start playback for "${dialogId}" - Howl returned no sound ID. Audio state: ${
              audio.state ? audio.state() : "unknown"
            }`
          );
        } else {
          this.logger.log(
            `Playing audio for "${dialogId}" (sound ID: ${playId})`
          );
        }
      }
    }

    // For video-synced dialogs, calculate videoStartTime: when video actually started playing
    // This accounts for delays - the video's currentTime starts at 0 when playback begins (after delay)
    // CRITICAL: This MUST be calculated when the dialog is first triggered, not during queue rebuilds
    // The videoStartTime represents when the video STARTED, which never changes, even if video keeps playing
    if (videoId && this.videoManager && !videoStartTime) {
      const videoPlayer = this.videoManager.getVideoPlayer(videoId);
      if (videoPlayer && videoPlayer.video && videoPlayer.isPlaying) {
        // videoStartTime = when the video actually started playing (after any delay)
        // currentTime represents elapsed time since video started
        // So: videoStartTime = now - elapsedTime = absolute time when video started
        const videoCurrentTime = videoPlayer.video.currentTime;
        videoStartTime = Date.now() - videoCurrentTime * 1000;
        this.logger.log(
          `Set videoStartTime for "${dialogId}" at video time ${videoCurrentTime.toFixed(
            2
          )}s (absolute: ${new Date(videoStartTime).toLocaleTimeString()})`
        );

        // Check if any captions will be missed due to late triggering
        if (dialogData.captions && videoCurrentTime > 0) {
          const firstCaptionWithStartTime = dialogData.captions.find(
            (c) => c.startTime !== undefined
          );
          if (firstCaptionWithStartTime) {
            const firstCaptionEndTime =
              firstCaptionWithStartTime.startTime +
              (firstCaptionWithStartTime.duration || 3.0);
            if (videoCurrentTime >= firstCaptionWithStartTime.startTime) {
              if (videoCurrentTime < firstCaptionEndTime) {
                this.logger.warn(
                  `⚠️ Video "${videoId}" is at ${videoCurrentTime.toFixed(
                    2
                  )}s, first caption "${firstCaptionWithStartTime.text.substring(
                    0,
                    30
                  )}..." starts at ${
                    firstCaptionWithStartTime.startTime
                  }s (ends ${firstCaptionEndTime.toFixed(
                    1
                  )}s). Caption SHOULD be active now!`
                );
              } else {
                this.logger.warn(
                  `⚠️ Video "${videoId}" has advanced ${videoCurrentTime.toFixed(
                    2
                  )}s, first caption "${firstCaptionWithStartTime.text.substring(
                    0,
                    30
                  )}..." (${
                    firstCaptionWithStartTime.startTime
                  }s-${firstCaptionEndTime.toFixed(1)}s) has already expired!`
                );
              }
            }
          }
        }
      } else {
        this.logger.warn(
          `Cannot set videoStartTime for "${dialogId}" - video player not available or not playing`
        );
      }
    }

    // Add dialog to active dialogs (don't stop existing dialogs)
    this.activeDialogs.set(dialogId, {
      dialogData,
      audio,
      videoId,
      videoStartTime,
      startTime: dialogStartTime,
      onComplete: finalOnCompleteToUse,
      progressTriggers,
      progressTriggersFired,
    });

    this.isPlaying = true; // At least one dialog is active

    // Rebuild queue IMMEDIATELY so captions are available right away
    // This is critical for video-synced captions where timing is precise
    if (this.activeDialogs.size > 0) {
      this._rebuildUnifiedCaptionQueue();
      this.queueNeedsRebuild = false;
    } else {
      // Mark for rebuild in next update if no dialogs yet
      this.queueNeedsRebuild = true;
    }

    this.emit("dialog:play", dialogData);
  }

  /**
   * Stop a specific dialog (or all dialogs if dialogId not specified)
   * @param {string|null} dialogId - Dialog ID to stop (null = stop all)
   * @param {number} fadeOutDuration - Duration in seconds for fade out (0 = immediate stop)
   */
  stopDialog(dialogId = null, fadeOutDuration = 0) {
    if (dialogId) {
      // Stop specific dialog
      const dialogInfo = this.activeDialogs.get(dialogId);
      if (!dialogInfo) return;

      if (fadeOutDuration > 0 && dialogInfo.audio) {
        // Fade out this dialog's audio
        const startVolume = this.audioVolume;
        const fadeOutStartTime = Date.now();

        const fadeOut = () => {
          const elapsed = (Date.now() - fadeOutStartTime) / 1000;
          const progress = Math.min(elapsed / fadeOutDuration, 1.0);
          const currentVolume = startVolume * (1.0 - progress);

          if (dialogInfo.audio && dialogInfo.audio.volume) {
            dialogInfo.audio.volume(currentVolume);
          }

          if (progress >= 1.0) {
            // Fade out complete, stop dialog
            if (dialogInfo.audio) {
              dialogInfo.audio.stop();
            }
            this._removeDialog(dialogId);
          } else if (this.activeDialogs.has(dialogId)) {
            requestAnimationFrame(fadeOut);
          }
        };
        fadeOut();
      } else {
        // Stop immediately
        if (dialogInfo.audio) {
          dialogInfo.audio.stop();
        }
        this._removeDialog(dialogId);
      }
    } else {
      // Stop all dialogs
      for (const [id, dialogInfo] of this.activeDialogs.entries()) {
        if (dialogInfo.audio) {
          dialogInfo.audio.stop();
        }
      }
      this.activeDialogs.clear();
      this.unifiedCaptionQueue = [];
      this.currentCaptionIndex = 0;
      this.isPlaying = false;
      this.hideCaption();
      this.emit("dialog:stop");
    }
  }

  /**
   * Remove a dialog from active dialogs
   * @private
   */
  _removeDialog(dialogId) {
    const dialogInfo = this.activeDialogs.get(dialogId);
    if (!dialogInfo) return;

    // Remove from active dialogs FIRST (before onComplete to avoid conflicts)
    this.activeDialogs.delete(dialogId);

    // Mark queue for rebuild (will happen in next update)
    this.queueNeedsRebuild = true;

    // Check if any dialogs are still active
    if (this.activeDialogs.size === 0) {
      this.isPlaying = false;
      this.hideCaption();
      this.emit("dialog:stop");
    } else {
      this.emit("dialog:complete", dialogInfo.dialogData);
    }

    // Call onComplete callback AFTER removal - this may trigger state changes
    // The state:changed listener will then check for new matching dialogs
    if (dialogInfo.onComplete) {
      this.logger.log(`Calling onComplete callback for dialog "${dialogId}"`);
      if (this.gameManager) {
        try {
          dialogInfo.onComplete(this.gameManager);
          this.logger.log(`onComplete callback for "${dialogId}" finished`);
        } catch (error) {
          this.logger.error(
            `Error in onComplete callback for "${dialogId}":`,
            error
          );
        }
      } else {
        // Fallback: pass dialogData if gameManager not available
        try {
          dialogInfo.onComplete(dialogInfo.dialogData);
        } catch (error) {
          this.logger.error(
            `Error in onComplete callback for "${dialogId}":`,
            error
          );
        }
      }
    }

    // Handle playNext chaining
    if (dialogInfo.dialogData.playNext) {
      this._handlePlayNext(dialogInfo.dialogData);
    }
  }

  /**
   * Handle dialog completion for a specific dialog
   * @private
   */
  _handleDialogComplete(dialogId) {
    this.logger.log(`Dialog "${dialogId}" completed, removing...`);
    this._removeDialog(dialogId);
  }

  /**
   * Handle playNext chaining
   * @private
   */
  async _handlePlayNext(completedDialogData) {
    if (!completedDialogData || !completedDialogData.playNext) {
      if (!completedDialogData) {
        this.logger.warn(
          "_handlePlayNext called with null/undefined dialogData"
        );
      } else {
        this.logger.log(
          `Dialog "${completedDialogData.id}" has no playNext property`
        );
      }
      return;
    }

    this.logger.log(
      `Chaining to next dialog from "${completedDialogData.id}" -> "${completedDialogData.playNext}"`
    );

    // Resolve the next dialog (could be an object or string ID)
    let nextDialog;
    if (typeof completedDialogData.playNext === "string") {
      this.logger.log(
        `Resolving dialog by ID: "${completedDialogData.playNext}"`
      );
      nextDialog = await this.resolveDialogById(completedDialogData.playNext);
      if (!nextDialog) {
        this.logger.error(
          `Failed to resolve dialog "${completedDialogData.playNext}" from "${completedDialogData.id}"`
        );
      } else {
        this.logger.log(
          `Resolved dialog "${completedDialogData.playNext}": found ${nextDialog.id}`
        );
      }
    } else {
      nextDialog = completedDialogData.playNext;
      this.logger.log(
        `Using direct dialog object: ${nextDialog.id || "unknown"}`
      );
    }

    if (!nextDialog) {
      this.logger.warn(
        `playNext dialog not found for "${completedDialogData.id}": ${completedDialogData.playNext}`
      );
      return;
    }

    this.logger.log(`Processing playNext for "${nextDialog.id}"`);

    // Check if dialog should play based on criteria
    // Note: If criteria is not defined, allow the dialog to play (it's explicitly chained)
    if (this.gameManager && nextDialog.criteria) {
      const currentState = this.gameManager.getState();
      const playCriteria = nextDialog.criteria;

      if (!checkCriteria(currentState, playCriteria)) {
        this.logger.log(
          `Next dialog "${nextDialog.id}" criteria not met, skipping playNext`
        );
        return;
      }
      this.logger.log(`Next dialog "${nextDialog.id}" criteria met`);
    } else {
      this.logger.log(
        `Next dialog "${nextDialog.id}" has no criteria - allowing playNext chain`
      );
    }

    // Check if dialog was already played once (if once flag is set)
    if (
      nextDialog.once &&
      this.playedDialogs &&
      this.playedDialogs.has(nextDialog.id)
    ) {
      this.logger.log(
        `Next dialog "${nextDialog.id}" already played once, skipping playNext`
      );
      return;
    }

    // Mark the next dialog as played if it has "once" flag
    // Do this BEFORE playing so it doesn't get triggered again
    if (nextDialog.once && this.playedDialogs) {
      this.playedDialogs.add(nextDialog.id);
      this.logger.log(`Marked chained dialog "${nextDialog.id}" as played`);
    }

    // Check if next dialog has a delay
    const delay = nextDialog.delay || 0;
    if (delay > 0) {
      this.logger.log(`Chaining to "${nextDialog.id}" with ${delay}s delay`);
      setTimeout(() => {
        this.logger.log(`Playing delayed chained dialog "${nextDialog.id}"`);
        this.playDialog(nextDialog);
      }, delay * 1000);
    } else {
      // Play immediately
      this.logger.log(`Playing chained dialog "${nextDialog.id}" immediately`);
      this.playDialog(nextDialog);
    }
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

    // Emit named event globally if specified
    if (caption.emitEvent && this.gameManager) {
      this.gameManager.emit(caption.emitEvent, caption);
      this.logger.log(
        `Emitted global event "${caption.emitEvent}" for caption: ${caption.text}`
      );
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
   * Resolve a dialog by ID from the dialogData
   * @param {string} dialogId - Dialog ID to resolve
   * @returns {Promise<Object|null>} Dialog object or null if not found
   */
  async resolveDialogById(dialogId) {
    // Try to use cached dialogTracks first
    if (this._dialogTracks && this._dialogTracks[dialogId]) {
      return this._dialogTracks[dialogId];
    }

    // Otherwise import it
    const { dialogTracks } = await import("./dialogData.js");

    // Cache it for future use
    if (!this._dialogTracks) {
      this._dialogTracks = dialogTracks;
    }

    const dialog = dialogTracks[dialogId];
    if (!dialog) {
      this.logger.warn(`Dialog "${dialogId}" not found in dialogTracks`);
    }
    return dialog || null;
  }

  /**
   * Get counterpart video ID (base <-> Safari)
   * @param {string} videoId - Video ID
   * @returns {Array<string>} Array of video IDs to check (original + counterpart)
   * @private
   */
  _getCounterpartVideoIds(videoId) {
    const videoIds = [videoId];
    if (videoId.endsWith("Safari")) {
      // If Safari video, also check base version
      const baseId = videoId.replace("Safari", "");
      videoIds.push(baseId);
    } else {
      // If base video, also check Safari version
      const safariId = videoId + "Safari";
      videoIds.push(safariId);
    }
    return videoIds;
  }

  /**
   * Check and trigger video-synced dialog for a specific video
   * Called immediately when video:play event is emitted
   * @param {string} videoId - Video ID
   * @private
   */
  _checkAndTriggerVideoDialog(videoId) {
    if (!this.videoManager || !this._dialogTracks || this.isFadingOut) {
      return;
    }

    // Check for dialogs matching this videoId or its counterpart
    const videoIdsToCheck = this._getCounterpartVideoIds(videoId);
    const matchingDialogs = Object.values(this._dialogTracks).filter(
      (d) => d.videoId && videoIdsToCheck.includes(d.videoId) && d.autoPlay
    );

    if (matchingDialogs.length === 0) return;

    // Verify video is actually playing (delay may still be pending)
    const videoPlayer = this.videoManager.getVideoPlayer(videoId);
    if (!videoPlayer) {
      return;
    }

    const hasPendingDelay =
      this.videoManager.pendingDelays &&
      this.videoManager.pendingDelays.has(videoId);

    const isVideoPlaying =
      !hasPendingDelay &&
      videoPlayer.isPlaying &&
      videoPlayer.video &&
      !videoPlayer.video.paused &&
      !videoPlayer.video.ended &&
      videoPlayer.video.readyState >= 2 &&
      videoPlayer.video.currentTime >= 0;

    if (!isVideoPlaying) {
      return;
    }

    const videoCurrentTime = videoPlayer.video.currentTime;

    // Trigger all matching dialogs
    for (const dialog of matchingDialogs) {
      // Skip if already played and marked as "once"
      if (
        dialog.once &&
        this.playedDialogs &&
        this.playedDialogs.has(dialog.id)
      ) {
        continue;
      }

      // Skip if already active
      if (this.activeDialogs.has(dialog.id)) {
        continue;
      }

      // Skip if already pending
      if (this.pendingDialogs.has(dialog.id)) {
        continue;
      }

      this.logger.log(
        `Video-synced dialog "${
          dialog.id
        }" triggered via event (video "${videoId}" at ${videoCurrentTime.toFixed(
          2
        )}s)`
      );

      if (videoCurrentTime > 1.0) {
        this.logger.warn(
          `⚠️ Video "${videoId}" has advanced ${videoCurrentTime.toFixed(
            2
          )}s before dialog "${
            dialog.id
          }" triggered. Early captions may be missed!`
        );
      }

      if (dialog.once && this.playedDialogs) {
        this.playedDialogs.add(dialog.id);
      }

      // Update dialog's videoId to match the actually playing video for caption syncing
      const dialogToPlay = { ...dialog };
      if (dialog.videoId !== videoId) {
        // Dialog is for counterpart video, but we're playing this one
        // Update videoId so captions sync to the actual playing video
        dialogToPlay.videoId = videoId;
      }
      this.playDialog(dialogToPlay);
    }
  }

  /**
   * Update method - call in animation loop
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    // Fallback: Check for video-synced dialogs in update loop (if event-based trigger missed)
    // Note: Video-synced dialogs can play even if another dialog is playing
    if (
      this.gameManager &&
      this.videoManager &&
      !this.isFadingOut &&
      this._dialogTracks
    ) {
      // Check all dialogs for video-synced ones
      for (const dialog of Object.values(this._dialogTracks)) {
        // Only check video-synced dialogs with autoPlay
        if (!dialog.videoId || !dialog.autoPlay) continue;

        // Skip if already played and marked as "once"
        if (
          dialog.once &&
          this.playedDialogs &&
          this.playedDialogs.has(dialog.id)
        ) {
          continue;
        }

        // Skip if already active
        if (this.activeDialogs.has(dialog.id)) {
          continue;
        }

        // Skip if already pending
        if (this.pendingDialogs.has(dialog.id)) {
          continue;
        }

        // Check if the video (or its counterpart) is actually playing
        const videoIdsToCheck = this._getCounterpartVideoIds(dialog.videoId);
        let videoPlayer = null;
        let playingVideoId = null;
        for (const vidId of videoIdsToCheck) {
          const player = this.videoManager.getVideoPlayer(vidId);
          if (player) {
            const hasPendingDelay =
              this.videoManager.pendingDelays &&
              this.videoManager.pendingDelays.has(vidId);

            const isVideoPlaying =
              !hasPendingDelay &&
              player.isPlaying &&
              player.video &&
              !player.video.paused &&
              !player.video.ended &&
              player.video.readyState >= 2 &&
              player.video.currentTime >= 0;

            if (isVideoPlaying) {
              videoPlayer = player;
              playingVideoId = vidId;
              break;
            }
          }
        }
        if (!videoPlayer || !playingVideoId) {
          continue;
        }

        const videoCurrentTime = videoPlayer.video.currentTime;
        this.logger.log(
          `Video-synced dialog "${
            dialog.id
          }" triggered via update loop (video "${playingVideoId}" at ${videoCurrentTime.toFixed(
            2
          )}s)`
        );

        if (videoCurrentTime > 1.0) {
          this.logger.warn(
            `⚠️ Video "${playingVideoId}" has advanced ${videoCurrentTime.toFixed(
              2
            )}s before dialog "${
              dialog.id
            }" triggered. Early captions may be missed!`
          );
        }

        if (dialog.once && this.playedDialogs) {
          this.playedDialogs.add(dialog.id);
        }

        // Update dialog's videoId to match the actually playing video for caption syncing
        const dialogToPlay = { ...dialog };
        if (dialog.videoId !== playingVideoId) {
          // Dialog is for counterpart video, but we're playing this one
          // Update videoId so captions sync to the actual playing video
          dialogToPlay.videoId = playingVideoId;
        }
        this.playDialog(dialogToPlay);
        break;
      }
    }

    // Check for auto-play dialogs (can interrupt current dialog)
    if (this.gameManager && !this.isFadingOut && this._getDialogsForState) {
      const currentState = this.gameManager.getState();
      const matchingDialogs = this._getDialogsForState(
        currentState,
        this.playedDialogs || new Set()
      );

      // Filter out video-synced dialogs (already handled above)
      // Video-synced dialogs should ONLY play when their video is playing, not from state changes
      const nonVideoDialogs = matchingDialogs.filter((d) => !d.videoId);

      // Debug: Log when we're checking for these specific dialogs
      if (
        currentState.viewmasterInsanityIntensity >=
          VIEWMASTER_OVERHEAT_THRESHOLD &&
        currentState.viewmasterOverheatDialogIndex !== null &&
        currentState.viewmasterOverheatDialogIndex !== undefined
      ) {
        const targetDialogs = nonVideoDialogs.filter(
          (d) => d.id === "coleUghGimmeASec" || d.id === "coleUghItsTooMuch"
        );
        if (targetDialogs.length > 0 && !this.isPlaying) {
          this.logger.log(
            `Found matching overheat dialog(s): ${targetDialogs
              .map((d) => d.id)
              .join(", ")}`
          );
        }
      }

      // Play matching dialogs that aren't already active or pending
      for (const dialog of nonVideoDialogs) {
        // Skip if already played and marked as "once" (check this FIRST)
        if (
          dialog.once &&
          this.playedDialogs &&
          this.playedDialogs.has(dialog.id)
        ) {
          // Don't log - this is expected behavior for "once" dialogs
          continue;
        }

        // Skip if already active
        if (this.activeDialogs.has(dialog.id)) {
          continue;
        }

        // Skip if already pending
        if (this.pendingDialogs.has(dialog.id)) {
          continue;
        }

        // All dialogs can run concurrently (add to active dialogs, don't stop existing)
        this.logger.log(`Auto-playing dialog "${dialog.id}"`);

        // Only track in playedDialogs if marked as "once" (allows replay for once: false)
        if (dialog.once && this.playedDialogs) {
          this.playedDialogs.add(dialog.id);
        }

        // Play the dialog (will be scheduled with its delay if specified)
        // This adds to activeDialogs, doesn't stop existing dialogs
        this.playDialog(dialog);
        break; // Only play one dialog per frame
      }
    }

    // Update pending delayed dialogs
    if (this.pendingDialogs.size > 0) {
      for (const [dialogId, pending] of this.pendingDialogs) {
        pending.timer += dt;

        // Check if delay has elapsed
        if (pending.timer >= pending.delay) {
          this.logger.log(`Playing delayed dialog "${dialogId}"`);
          this.pendingDialogs.delete(dialogId);
          this._playDialogImmediate(pending.dialogData, pending.onComplete);
          break; // Only play one dialog per frame
        }
      }
    }

    // Progress-based state triggers for all active dialogs
    if (this.gameManager && this.activeDialogs.size > 0) {
      for (const [dialogId, dialogInfo] of this.activeDialogs.entries()) {
        if (!dialogInfo.audio || dialogInfo.progressTriggers.length === 0) {
          continue;
        }

        try {
          const dur = dialogInfo.audio.duration
            ? dialogInfo.audio.duration()
            : 0;
          const pos = dialogInfo.audio.seek ? dialogInfo.audio.seek() : 0;
          if (dur > 0) {
            const prog = pos / dur; // 0..1
            for (let i = 0; i < dialogInfo.progressTriggers.length; i++) {
              const trig = dialogInfo.progressTriggers[i];
              if (
                prog >= trig.progress &&
                !dialogInfo.progressTriggersFired.has(i)
              ) {
                // Fire state change
                if (typeof this.gameManager.setState === "function") {
                  this.gameManager.setState({ currentState: trig.state });
                }
                dialogInfo.progressTriggersFired.add(i);
              }
            }
          }
        } catch (e) {
          // ignore seek() while not loaded
        }
      }
    }

    // Rebuild unified caption queue only when needed (dialogs added/removed)
    // For video-synced captions, update absolute times incrementally
    if (this.queueNeedsRebuild && this.activeDialogs.size > 0) {
      this._rebuildUnifiedCaptionQueue();
      this.queueNeedsRebuild = false;
    } else if (this.activeDialogs.size > 0) {
      // Update absolute times for video-synced captions without full rebuild
      this._updateVideoCaptionTimes();
    }

    // Update caption display based on unified queue
    if (this.activeDialogs.size > 0) {
      const currentTime = Date.now();

      // Find which caption should be shown now
      // Check ALL captions in the queue - don't stop at first gap
      let activeCaptionEntry = null;
      let activeCaptionIndex = -1;

      for (let i = 0; i < this.unifiedCaptionQueue.length; i++) {
        const captionEntry = this.unifiedCaptionQueue[i];

        // Check if this caption should be active now
        // A caption is active if current time is between its start and end
        // This catches captions that started in the past (e.g., if dialog triggered late via playNext)
        // but are still within their display duration
        const isCurrentlyActive =
          currentTime >= captionEntry.absoluteStartTime &&
          currentTime < captionEntry.absoluteEndTime;

        if (isCurrentlyActive) {
          // This caption should be shown
          activeCaptionEntry = captionEntry;
          activeCaptionIndex = i;
          break; // First match is correct (queue is sorted by time)
        }
      }

      // Show the active caption if found
      if (activeCaptionEntry) {
        if (this.currentCaptionIndex !== activeCaptionIndex) {
          this.currentCaptionIndex = activeCaptionIndex;
          this.showCaption(activeCaptionEntry.caption);
        }
      } else {
        // No active caption found - check if current caption has actually expired
        if (
          this.currentCaptionIndex >= 0 &&
          this.currentCaptionIndex < this.unifiedCaptionQueue.length
        ) {
          const currentCaption =
            this.unifiedCaptionQueue[this.currentCaptionIndex];

          // Only hide if the current caption has actually expired
          // Don't hide just because we're in a gap - there might be another caption coming
          if (currentTime >= currentCaption.absoluteEndTime) {
            // Current caption expired - check if we're past ALL captions
            const lastCaption =
              this.unifiedCaptionQueue[this.unifiedCaptionQueue.length - 1];
            if (currentTime >= lastCaption.absoluteEndTime) {
              // Past last caption entirely - hide it
              this.currentCaptionIndex = this.unifiedCaptionQueue.length;
              this.hideCaption();
            } else {
              // Between captions - current expired, but more may come
              // Don't update index yet, just hide - will find next caption next frame
              this.hideCaption();
            }
          }
          // If current caption hasn't expired, keep showing it (even if in a gap from another dialog)
        } else {
          // currentCaptionIndex is beyond queue or queue is empty - hide caption
          if (this.currentCaptionIndex < this.unifiedCaptionQueue.length) {
            this.currentCaptionIndex = this.unifiedCaptionQueue.length;
          }
          this.hideCaption();
        }
      }

      // Check if any caption-only dialogs have finished all their captions
      for (const [dialogId, dialogInfo] of this.activeDialogs.entries()) {
        // Check caption-only dialogs (no audio, no videoId)
        if (!dialogInfo.audio && !dialogInfo.videoId) {
          // Find the last caption for this dialog
          let lastCaption = null;
          for (let i = this.unifiedCaptionQueue.length - 1; i >= 0; i--) {
            if (this.unifiedCaptionQueue[i].dialogId === dialogId) {
              lastCaption = this.unifiedCaptionQueue[i];
              break;
            }
          }

          // If all captions have expired, complete the dialog
          if (lastCaption && currentTime >= lastCaption.absoluteEndTime) {
            this.logger.log(
              `Caption-only dialog "${dialogId}" completed (all captions finished)`
            );
            this._handleDialogComplete(dialogId);
          }
        }
      }

      // Check if we've passed all captions and no dialogs are active
      if (this.currentCaptionIndex >= this.unifiedCaptionQueue.length) {
        // Check if any dialogs are still active
        let allComplete = true;
        for (const [dialogId, dialogInfo] of this.activeDialogs.entries()) {
          if (dialogInfo.audio) {
            try {
              if (dialogInfo.audio.playing && dialogInfo.audio.playing()) {
                allComplete = false;
                break;
              }
            } catch (e) {
              // Audio might not be loaded
            }
          } else if (dialogInfo.videoId && this.videoManager) {
            const videoPlayer = this.videoManager.getVideoPlayer(
              dialogInfo.videoId
            );
            if (videoPlayer && videoPlayer.isPlaying) {
              allComplete = false;
              break;
            }
          }
        }

        // If all dialogs are complete, hide caption (but don't remove dialogs yet - they'll be removed on audio/video end)
        if (allComplete) {
          // Captions are already hidden, dialogs will be cleaned up on audio/video end
        }
      }
    } else {
      // No active dialogs
      if (this.currentCaptionIndex < this.unifiedCaptionQueue.length) {
        this.hideCaption();
        this.currentCaptionIndex = this.unifiedCaptionQueue.length;
      }
    }

    // Handle fade out
    if (this.isFadingOut) {
      this.fadeOutTimer += dt;
      const progress = Math.min(this.fadeOutTimer / this.fadeOutDuration, 1.0);
      const currentVolume = this.fadeOutStartVolume * (1.0 - progress);

      // Fade out all active audio
      for (const [dialogId, dialogInfo] of this.activeDialogs.entries()) {
        if (dialogInfo.audio && dialogInfo.audio.volume) {
          dialogInfo.audio.volume(currentVolume);
        }
      }

      // When fade out completes, stop all dialogs
      if (progress >= 1.0) {
        this.isFadingOut = false;
        this.fadeOutTimer = 0;
        this.fadeOutDuration = 0;
        this.stopDialog(); // Stop all dialogs
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
   * Note: The CSS file (dialog.css) already defines the styling, including the custom font.
   * We only set properties that aren't defined in CSS or need to be dynamic.
   */
  applyDefaultCaptionStyle() {
    // Only set properties that aren't already in dialog.css
    // The font-family "PXCountryTypewriter" is set in dialog.css, so we don't override it here
    // Most styling is handled by the CSS file, which is imported in main.js
  }

  /**
   * Set dialog volume
   * @param {number} volume - Volume level (0.0-1.0)
   */
  setVolume(volume) {
    this.baseVolume = Math.max(0, Math.min(1, volume));
    this.updateVolume();
  }

  /**
   * Update volume for all active dialogs
   */
  updateVolume() {
    this.audioVolume = this.baseVolume;
    if (this.sfxManager) {
      this.audioVolume *= this.sfxManager.getMasterVolume();
    }

    // Update volume for all active audio
    for (const [dialogId, dialogInfo] of this.activeDialogs.entries()) {
      if (dialogInfo.audio && dialogInfo.audio.volume) {
        dialogInfo.audio.volume(this.audioVolume);
      }
    }

    // Update volume for preloaded audio
    this.preloadedAudio.forEach((audio) => {
      if (audio.volume) {
        audio.volume(this.audioVolume);
      }
    });
  }

  /**
   * Set captions enabled/disabled
   * @param {boolean} enabled - Whether captions should be shown
   */
  setCaptionsEnabled(enabled) {
    this.captionsEnabled = enabled;
    if (!enabled) {
      this.hideCaption();
    }
  }

  /**
   * Check if any dialog is currently playing
   * @returns {boolean}
   */
  isDialogPlaying() {
    return this.isPlaying && this.activeDialogs.size > 0;
  }

  /**
   * Check if a specific dialog is pending (scheduled with delay)
   * @param {string} dialogId - Dialog ID to check
   * @returns {boolean}
   */
  isDialogPending(dialogId) {
    return this.pendingDialogs.has(dialogId);
  }

  /**
   * Check if any dialogs are pending
   * @returns {boolean}
   */
  hasDialogsPending() {
    return this.pendingDialogs.size > 0;
  }

  /**
   * Get dialog duration (for audio-based dialogs)
   * @param {string} dialogId - Dialog ID
   * @returns {number|null} Duration in seconds or null if not found/not loaded
   */
  getDialogDuration(dialogId) {
    const dialogInfo = this.activeDialogs.get(dialogId);
    if (dialogInfo && dialogInfo.audio) {
      try {
        const duration = dialogInfo.audio.duration
          ? dialogInfo.audio.duration()
          : null;
        if (duration && duration > 0) {
          return duration;
        }
      } catch (e) {
        // Audio might not be loaded yet
      }
    }

    // Check preloaded audio
    if (this.preloadedAudio.has(dialogId)) {
      const audio = this.preloadedAudio.get(dialogId);
      try {
        const duration = audio.duration ? audio.duration() : null;
        if (duration && duration > 0) {
          return duration;
        }
      } catch (e) {
        // Audio might not be loaded yet
      }
    }

    // On iOS, audio might be prefetched as blob but not yet converted to Howl
    // Fall back to caption-based duration calculation as fallback
    if (
      this.prefetchedAudio.has(dialogId) ||
      this.deferredDialogs.has(dialogId)
    ) {
      const dialogData = this.prefetchedAudio.has(dialogId)
        ? this.prefetchedAudio.get(dialogId).dialogData
        : this.deferredDialogs.get(dialogId);

      if (dialogData) {
        const captionDuration = this.getTotalDialogDuration(dialogData);
        if (captionDuration > 0) {
          this.logger.log(
            `Using caption-based duration (${captionDuration.toFixed(
              2
            )}s) for prefetched/deferred dialog "${dialogId}"`
          );
          return captionDuration;
        }
      }
    }

    // Try to get duration from dialog data if available (for iOS prefetched case)
    if (this._dialogTracks && this._dialogTracks[dialogId]) {
      const dialogData = this._dialogTracks[dialogId];
      const captionDuration = this.getTotalDialogDuration(dialogData);
      if (captionDuration > 0) {
        this.logger.log(
          `Using caption-based duration (${captionDuration.toFixed(
            2
          )}s) for dialog "${dialogId}" from dialogTracks`
        );
        return captionDuration;
      }
    }

    return null;
  }

  /**
   * Get total dialog duration including all captions
   * @param {Object} dialogData - Dialog data object
   * @returns {number} Total duration in seconds
   */
  getTotalDialogDuration(dialogData) {
    if (!dialogData) return 0;

    // For video-synced dialogs, use last caption end time
    if (dialogData.videoId && dialogData.captions) {
      let maxEndTime = 0;
      for (const caption of dialogData.captions) {
        const startTime = caption.startTime || 0;
        const endTime = startTime + (caption.duration || 3.0);
        maxEndTime = Math.max(maxEndTime, endTime);
      }
      return maxEndTime;
    }

    // For audio-based dialogs, sum caption durations
    if (dialogData.captions) {
      return dialogData.captions.reduce(
        (total, caption) => total + (caption.duration || 3.0),
        0
      );
    }

    return 0;
  }

  /**
   * Register event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    if (!this.eventListeners[event]) {
      this.eventListeners[event] = [];
    }
    this.eventListeners[event].push(callback);
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  off(event, callback) {
    if (!this.eventListeners[event]) {
      return;
    }
    const index = this.eventListeners[event].indexOf(callback);
    if (index !== -1) {
      this.eventListeners[event].splice(index, 1);
    }
  }

  /**
   * Emit event
   * @param {string} event - Event name
   * @param {...any} args - Event arguments
   */
  emit(event, ...args) {
    if (!this.eventListeners[event]) {
      return;
    }
    this.eventListeners[event].forEach((callback) => {
      try {
        callback(...args);
      } catch (error) {
        this.logger.error(`Error in event listener for "${event}":`, error);
      }
    });
  }

  /**
   * Clean up
   */
  destroy() {
    // Stop all dialogs
    this.stopDialog();

    // Clean up preloaded audio
    this.preloadedAudio.forEach((audio) => {
      audio.unload();
    });
    this.preloadedAudio.clear();

    // Remove caption element
    if (this.captionElement && this.captionElement.parentNode) {
      this.captionElement.parentNode.removeChild(this.captionElement);
    }
  }
}

export default DialogManager;
