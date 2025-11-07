import { Logger } from "./utils/logger.js";
import { GAME_STATES } from "./gameData.js";
import { getSceneObjectsForState } from "./sceneData.js";
import { isDebugSpawnActive } from "./utils/debugSpawner.js";

/**
 * ZoneManager - Manages loading/unloading of exterior splat zones based on player location
 *
 * Zone mapping:
 * - AlleyIntro: loads AlleyIntro + AlleyNavigable + FourWay
 * - AlleyNavigable: loads AlleyNavigable + AlleyLongView + FourWay + AlleyIntro
 * - FourWay: loads FourWay + AlleyNavigable + ThreeWay + Plaza
 * - ThreeWay: loads ThreeWay + FourWay + ThreeWay2
 * - ThreeWay2: loads ThreeWay2 + ThreeWay + Plaza
 * - Plaza: loads Plaza + ThreeWay2 + FourWay
 *
 * Only active when currentState < OFFICE_INTERIOR
 */

class ZoneManager {
  constructor(gameManager, sceneManager) {
    this.gameManager = gameManager;
    this.sceneManager = sceneManager;
    this.logger = new Logger("ZoneManager", true); // Enable logging

    // Desktop zone to splat mapping
    this.zoneToSplatsDesktop = {
      alleyIntro: ["alleyIntro", "alleyNavigable"],
      alleyNavigable: [
        "alleyNavigable",
        "alleyLongView",
        "fourWay",
        "alleyIntro",
      ],
      fourWay: ["fourWay", "alleyNavigable", "threeWay", "plaza"],
      threeWay: ["threeWay", "fourWay", "threeWay2"],
      threeWay2: ["threeWay2", "threeWay", "plaza"],
      plaza: ["plaza", "threeWay2", "fourWay"],
    };

    // Mobile zone to splat mapping (alleyNavigable merged with alleyIntro)
    this.zoneToSplatsMobile = {
      alleyIntro: ["alleyIntro", "fourWay"], // alleyNavigable merged into alleyIntro
      alleyNavigable: ["alleyIntro", "alleyLongView", "fourWay"], // Map to alleyIntro since merged
      fourWay: ["fourWay", "alleyIntro", "threeWay", "plaza"], // No alleyNavigable
      threeWay: ["threeWay", "fourWay", "threeWay2"],
      threeWay2: ["threeWay2", "threeWay", "plaza"],
      plaza: ["plaza", "threeWay2", "fourWay"],
    };

    // Current zone mapping (will be set based on isMobile)
    // Initialize with desktop mapping, will be updated on first state change
    this.zoneToSplats = this.zoneToSplatsDesktop;

    // Track currently loaded splats
    this.loadedSplats = new Set();

    // Current zone (null when not in exterior)
    this.currentZone = null;

    // Track if we've moved from initial state (prevents false positives during init)
    this.hasMovedFromInitialState = false;
    this.zoneDetectionTimeout = null; // Store timeout reference

    // Callback to enable zone detection (called by StartScreen when camera animation is ready)
    this.enableZoneDetectionCallback = null;

    // Only active when in exterior (before OFFICE_INTERIOR)
    this.isActive = false;

    // Zone change debouncing to prevent rapid switching
    this.pendingZoneChange = null;
    this.zoneChangeTimeout = null;
    this.zoneChangeDebounceDelay = 200; // 200ms debounce delay (increased for stability)

    // Track which zones are currently active (multiple can be active at boundaries)
    this.activeZones = new Set(); // Set of zone names that are currently intersecting

    // Listen for state changes
    if (this.gameManager) {
      this.gameManager.on("state:changed", (newState, oldState) => {
        this.handleStateChange(newState, oldState);
      });

      // Handle initial state
      this.handleStateChange(this.gameManager.getState(), null);
    }

    this.logger.log("ZoneManager initialized");
  }

  /**
   * Get the appropriate zone mapping based on isMobile state
   * @returns {Object} Zone to splats mapping
   */
  getZoneMapping() {
    if (!this.gameManager) {
      return this.zoneToSplatsDesktop; // Default to desktop
    }
    const state = this.gameManager.getState();
    const isMobile = state?.isMobile === true;
    return isMobile ? this.zoneToSplatsMobile : this.zoneToSplatsDesktop;
  }

  /**
   * Update zone mapping if isMobile state changed
   * @param {Object} newState - New game state
   * @param {Object} oldState - Previous game state
   */
  updateZoneMappingIfNeeded(newState, oldState) {
    if (!newState || !oldState) return;

    const wasMobile = oldState.isMobile === true;
    const isMobile = newState.isMobile === true;

    if (wasMobile !== isMobile) {
      this.zoneToSplats = this.getZoneMapping();
      this.logger.log(
        `Zone mapping updated for ${isMobile ? "mobile" : "desktop"} platform`
      );

      // If we have a current zone, reload it with the new mapping
      if (this.currentZone) {
        this.setZone(this.currentZone).catch((error) => {
          this.logger.error(
            "Error reloading zone after mapping change:",
            error
          );
        });
      }
    }
  }

  /**
   * Enable zone detection (called by StartScreen when camera animation is ready)
   * But still wait 10 seconds after plaza is set before actually enabling
   */
  enableZoneDetection() {
    // Don't enable immediately - wait for the timeout set when plaza is initialized
    // The timeout will set hasMovedFromInitialState after 10 seconds
    this.logger.log(
      "enableZoneDetection() called - will activate after 10 second delay from plaza setup"
    );
  }

  /**
   * Set callback to enable zone detection (called by StartScreen when camera animation is ready)
   */
  setEnableZoneDetectionCallback(callback) {
    this.enableZoneDetectionCallback = callback;
  }

  /**
   * Handle game state changes
   * @param {Object} newState - New game state
   * @param {Object} oldState - Previous game state
   */
  handleStateChange(newState, oldState) {
    if (!newState) return;

    // Update zone mapping based on current isMobile state (or if it changed)
    if (!oldState) {
      // Initial state - set mapping based on current isMobile
      this.zoneToSplats = this.getZoneMapping();
    } else {
      // Update zone mapping if isMobile changed
      this.updateZoneMappingIfNeeded(newState, oldState);
    }

    const wasActive = this.isActive;
    this.isActive = newState.currentState < GAME_STATES.ENTERING_OFFICE;

    // If we transitioned out of exterior (reached ENTERING_OFFICE or later), unload all exterior splats
    // From ENTERING_OFFICE onwards, scene objects load/unload purely based on criteria
    if (wasActive && !this.isActive) {
      this.logger.log(
        `Exiting exterior area (state ${newState.currentState} >= ENTERING_OFFICE) - unloading all exterior splats. Scene objects will now load/unload based on criteria.`
      );
      // Clear currentZone FIRST so unloadAllExteriorSplats doesn't block on safety check
      this.currentZone = null;
      // Clear any pending zone changes
      if (this.zoneChangeTimeout) {
        clearTimeout(this.zoneChangeTimeout);
        this.zoneChangeTimeout = null;
        this.pendingZoneChange = null;
      }
      // Clear active zones
      this.activeZones.clear();
      // Now unload all splats (safety check won't trigger since currentZone is null)
      this.unloadAllExteriorSplats();
      return;
    }

    // If debug spawn is active, enable zone detection immediately and skip default plaza setup
    // Collision detection will determine the correct zone from the start
    if (isDebugSpawnActive() && !this.hasMovedFromInitialState) {
      this.hasMovedFromInitialState = true;
      this.logger.log(
        "Zone detection enabled immediately (debug spawn active) - collision detection will determine zones"
      );
    }

    // If currentZone is set in state but doesn't match our currentZone, sync it
    // This handles hard-coded initial zones and state changes from other sources
    // Do this BEFORE the transition checks so it works even during activation
    // BUT: Don't sync zones from state until zone detection is enabled (prevents false positives during init)
    // EXCEPT: Allow setting initial "plaza" zone ONCE when activating (but NOT during debug spawn)
    // CRITICAL: Once zone detection is enabled via collision detection, ZoneManager is the source of truth
    // and we should NOT sync from state anymore (zone colliders don't set state, they call addActiveZone)
    if (
      newState.currentZone &&
      newState.currentZone !== this.currentZone &&
      this.isActive
    ) {
      // CRITICAL: Only sync from state BEFORE zone detection is enabled
      // Once zone detection is enabled, collision detection via addActiveZone/removeActiveZone is the source of truth
      if (!this.hasMovedFromInitialState) {
        // Zone detection not enabled yet - allow initial plaza setup (but NOT during debug spawn)
        if (
          newState.currentZone === "plaza" &&
          this.currentZone === null &&
          !isDebugSpawnActive()
        ) {
          // Only allow initial plaza setup before zone detection is enabled (normal gameplay only)
          if (!this.activeZones.has(newState.currentZone)) {
            this.activeZones.add(newState.currentZone);
          }
          this.setZone(newState.currentZone).catch((error) => {
            this.logger.error("Error setting initial zone:", error);
          });

          // Delay zone detection by 10 seconds after plaza is set to let everything settle
          // Clear any existing timeout first
          if (this.zoneDetectionTimeout) {
            clearTimeout(this.zoneDetectionTimeout);
          }
          this.zoneDetectionTimeout = setTimeout(() => {
            if (!this.hasMovedFromInitialState) {
              this.hasMovedFromInitialState = true;
              this.logger.log(
                "Zone detection enabled after 10 second delay (plaza set)"
              );
            }
          }, 10000);
        }
        // Otherwise, ignore state changes until zone detection is enabled
      } else {
        // Zone detection is enabled - ZoneManager is source of truth, don't sync from state
        // Zone colliders call addActiveZone/removeActiveZone directly, not setState
        this.logger.log(
          `[Zone] Ignoring state.currentZone="${newState.currentZone}" - zone detection is enabled, collision detection is source of truth`
        );
      }
    }

    // If we transitioned into exterior, wait for collision detection to set active zones
    if (!wasActive && this.isActive) {
      this.logger.log(
        `ZoneManager activated (currentState: ${
          newState.currentState
        }, < OFFICE_INTERIOR: ${
          newState.currentState < GAME_STATES.OFFICE_INTERIOR
        }) - waiting for zone collision detection`
      );
      // Initial zone setup is handled above - no need to duplicate here
      return;
    }

    // Zone changes are now handled entirely by collision detection (addActiveZone/removeActiveZone)
    // But we also sync with state changes for hard-coded initial zones
  }

  /**
   * Check if camera has moved significantly from spawn to enable zone detection
   * @returns {boolean} True if camera has moved at least 10 units from origin
   */
  _hasCameraMovedFromSpawn() {
    if (!this.gameManager || !this.gameManager.sceneManager) return false;
    // Check if camera exists and has moved significantly
    // We'll check this via collision detection results - if we get a legitimate zone detection
    // after plaza, we know camera is positioned
    return this.hasMovedFromInitialState;
  }

  /**
   * Set the current zone and load/unload splats accordingly
   * @param {string} zone - Zone name (e.g., "alleyIntro", "alleyNavigable", "fourWay")
   */
  async setZone(zone) {
    if (!this.isActive) {
      this.logger.warn(
        `Cannot set zone "${zone}" - ZoneManager is not active (must be before OFFICE_INTERIOR)`
      );
      return;
    }

    // CRITICAL: Don't allow zone changes until zone detection is enabled
    // This prevents false positives during initialization
    if (!this.hasMovedFromInitialState) {
      // Only allow setting the initial zone (plaza) from state when currentZone is null
      // Once plaza is set, block ALL other zone changes until enableZoneDetection() is called
      if (zone !== "plaza" || this.currentZone !== null) {
        this.logger.log(
          `[BLOCKED] Zone change to "${zone}" - zone detection not enabled yet (currentZone: ${this.currentZone}, hasMovedFromInitialState: ${this.hasMovedFromInitialState})`
        );
        return;
      }
    }

    if (!zone) {
      // Zone cleared - unload all splats
      // Clear currentZone FIRST so unloadAllExteriorSplats doesn't block on safety check
      this.currentZone = null;
      this.logger.log("Zone cleared - unloading all exterior splats");
      this.unloadAllExteriorSplats();
      return;
    }

    // Early return if already in this zone (prevents redundant work)
    if (zone === this.currentZone) {
      return;
    }

    // REMOVED: Code that auto-enabled zone detection when switching from plaza
    // Zone detection should ONLY be enabled via enableZoneDetection() called by StartScreen
    // This prevents false positives during initialization

    const oldZone = this.currentZone;
    this.currentZone = zone;

    // Get current zone mapping (may have changed if isMobile state changed)
    const zoneMapping = this.getZoneMapping();

    // Determine which splats should be loaded for this zone
    const requiredSplats = zoneMapping[zone] || [];

    this.logger.log(`Zone changed: ${oldZone || "none"} -> ${zone}`);
    this.logger.log(
      `Required splats for zone "${zone}": ${requiredSplats.join(", ")}`
    );
    this.logger.log(
      `Currently loaded splats: ${
        Array.from(this.loadedSplats).join(", ") || "none"
      }`
    );

    // First, verify which splats are already loaded and in scene - these should NOT be touched
    const alreadyLoadedAndInScene = new Set();
    for (const splatId of requiredSplats) {
      if (this.loadedSplats.has(splatId)) {
        const object = this.sceneManager.getObject(splatId);
        if (object && object.parent === this.sceneManager.scene) {
          // Already loaded and in scene - mark it so we don't touch it
          alreadyLoadedAndInScene.add(splatId);
        }
      }
    }

    // Determine which splats to unload (loaded but not required for new zone)
    const splatsToUnload = [];
    for (const splatId of this.loadedSplats) {
      // Don't unload if it's required for the new zone
      if (requiredSplats.includes(splatId)) {
        continue;
      }
      splatsToUnload.push(splatId);
    }

    // Determine which splats to load (required but not loaded AND not already in scene)
    const splatsToLoad = [];
    for (const splatId of requiredSplats) {
      // Skip if already loaded and in scene
      if (alreadyLoadedAndInScene.has(splatId)) {
        continue;
      }
      // Check if already loaded but not in scene - restore it immediately
      if (this.loadedSplats.has(splatId)) {
        const object = this.sceneManager.getObject(splatId);
        if (object) {
          // Object exists but might not be in scene - ensure it's in scene
          if (!object.parent || object.parent !== this.sceneManager.scene) {
            // Not in scene - add it
            this.sceneManager.addObjectToScene(splatId);
            continue;
          }
          // Object is in scene but wasn't caught by alreadyLoadedAndInScene check
          // (maybe parent changed between checks) - just track it
          this.loadedSplats.add(splatId); // Ensure it's tracked
          continue;
        }
        // Object doesn't exist even though it's in loadedSplats - remove from tracking and load fresh
        this.loadedSplats.delete(splatId);
      }
      // Not loaded - need to load it
      splatsToLoad.push(splatId);
    }

    // Log what we're about to do
    if (splatsToUnload.length > 0) {
      this.logger.log(
        `Unloading ${splatsToUnload.length} splat(s): ${splatsToUnload.join(
          ", "
        )}`
      );
    }
    if (splatsToLoad.length > 0) {
      this.logger.log(
        `Loading ${splatsToLoad.length} splat(s): ${splatsToLoad.join(", ")}`
      );
    }

    // IMPORTANT: Unload old splats FIRST before loading new ones
    // This ensures we don't have too many splats in memory/scene at once
    for (const splatId of splatsToUnload) {
      this.unloadSplat(splatId);
    }

    // Then load new splats after cleanup is complete
    for (const splatId of splatsToLoad) {
      await this.loadSplat(splatId);
    }

    // Log what's currently loaded for this zone
    const finalLoaded = Array.from(this.loadedSplats).filter((splatId) =>
      requiredSplats.includes(splatId)
    );
    this.logger.log(
      `Zone configured - Loaded splats: ${finalLoaded.join(", ")}`
    );
  }

  /**
   * Load a splat if it's not already loaded
   * @param {string} splatId - Splat ID (e.g., "alleyIntro", "alleyNavigable", "alleyLongView", "fourWay")
   */
  async loadSplat(splatId) {
    if (!this.sceneManager) {
      this.logger.error("Cannot load splat - sceneManager not available");
      return;
    }

    // First check if object is already loaded AND in scene - if so, nothing to do
    if (this.loadedSplats.has(splatId)) {
      const object = this.sceneManager.getObject(splatId);
      if (object && object.parent === this.sceneManager.scene) {
        // Already loaded and in scene - nothing to do
        return;
      }
    }

    // Get object from sceneData using criteria system to get correct mobile/desktop version
    if (!this.gameManager) {
      this.logger.error("Cannot load splat - gameManager not available");
      return;
    }

    const currentState = this.gameManager.getState();
    // Get all matching objects (both preload: true and preload: false)
    // This ensures we get the correct mobile/desktop version based on criteria
    const matchingObjects = getSceneObjectsForState(currentState);

    // Find the object with matching ID that also matches criteria
    const objectData = matchingObjects.find((obj) => obj.id === splatId);

    if (!objectData) {
      this.logger.warn(
        `Splat "${splatId}" not found in sceneData or doesn't match current criteria (isMobile: ${currentState?.isMobile})`
      );
      return;
    }

    // Check if already loaded in sceneManager but not in scene
    if (this.sceneManager.hasObject(splatId)) {
      // Object is loaded but not in scene - add it
      if (this.sceneManager.objectsNotInScene.has(splatId)) {
        this.sceneManager.addObjectToScene(splatId);
        this.loadedSplats.add(splatId);
        return;
      }
      // Object already in scene but not tracked - add to tracking
      const object = this.sceneManager.getObject(splatId);
      if (object && object.parent === this.sceneManager.scene) {
        this.loadedSplats.add(splatId);
        return;
      }
    }

    // Load the splat
    try {
      await this.sceneManager.loadObject(objectData, false); // false = add to scene immediately
      this.loadedSplats.add(splatId);
    } catch (error) {
      this.logger.error(`Failed to load splat "${splatId}":`, error);
    }
  }

  /**
   * Unload a splat (remove from scene but keep loaded)
   * @param {string} splatId - Splat ID
   */
  unloadSplat(splatId) {
    if (!this.sceneManager) {
      this.logger.error("Cannot unload splat - sceneManager not available");
      return;
    }

    if (!this.loadedSplats.has(splatId)) {
      return; // Already unloaded
    }

    // Safety check: Don't unload if it's required for the current zone
    if (this.currentZone) {
      const zoneMapping = this.getZoneMapping();
      const requiredSplats = zoneMapping[this.currentZone] || [];
      if (requiredSplats.includes(splatId)) {
        this.logger.warn(
          `Attempted to unload required splat "${splatId}" for zone "${this.currentZone}" - skipping`
        );
        return;
      }
    }

    const object = this.sceneManager.getObject(splatId);
    if (object && object.parent === this.sceneManager.scene) {
      // Remove from scene
      this.sceneManager.scene.remove(object);
      this.sceneManager.objectsNotInScene.add(splatId);
    }
    this.loadedSplats.delete(splatId);
  }

  /**
   * Unload all exterior splats (when entering office)
   */
  unloadAllExteriorSplats() {
    const splatsToUnload = Array.from(this.loadedSplats);
    for (const splatId of splatsToUnload) {
      this.unloadSplat(splatId);
    }
    this.logger.log("All exterior splats unloaded");
  }

  /**
   * Add a zone to the active zones set (called by ColliderManager when entering)
   * @param {string} zoneName - Zone name
   */
  addActiveZone(zoneName) {
    if (!this.isActive) return;

    // CRITICAL: Completely disable collision-based zone detection until explicitly enabled
    // This prevents false positives during initialization when camera hasn't been positioned yet
    if (!this.hasMovedFromInitialState) {
      // Block ALL collision-based zone detection until enableZoneDetection() is called
      this.logger.log(
        `Ignoring collision detection for "${zoneName}" - waiting for camera animation to position camera`
      );
      return;
    }

    // Only add if not already active (prevents duplicate calls from physics jitter)
    if (this.activeZones.has(zoneName)) {
      return;
    }

    this.activeZones.add(zoneName);
    this.logger.log(
      `[Zone] Added zone "${zoneName}" to active zones: ${Array.from(
        this.activeZones
      ).join(", ")}`
    );

    // Determine which zone to use when multiple are active
    // Priority: use the zone that matches currentZone if it's still active, otherwise prefer more specific zones
    let targetZone = null;
    if (this.currentZone && this.activeZones.has(this.currentZone)) {
      // Current zone is still active - keep it (prevents switching at boundaries)
      targetZone = this.currentZone;
    } else if (this.activeZones.size > 0) {
      // Current zone is not active, but others are - pick most specific zone
      // Zone priority (more specific first): alley zones > intersections > plaza > fourWay
      const priorityOrder = [
        "alleyIntro",
        "alleyNavigable",
        "alleyLongView",
        "threeWay2",
        "threeWay",
        "plaza",
        "fourWay", // Broadest - overlaps with many zones
      ];

      // Find highest priority zone that's active
      for (const priorityZone of priorityOrder) {
        if (this.activeZones.has(priorityZone)) {
          targetZone = priorityZone;
          break;
        }
      }

      // Fallback (shouldn't happen if all zones are in priority list)
      if (!targetZone) {
        targetZone = Array.from(this.activeZones)[0];
      }
    }

    // Only change zone if target is different and valid
    if (targetZone && targetZone !== this.currentZone) {
      // Debounce zone changes to prevent rapid switching
      this.pendingZoneChange = targetZone;

      // Clear existing timeout
      if (this.zoneChangeTimeout) {
        clearTimeout(this.zoneChangeTimeout);
      }

      // Set new timeout
      this.zoneChangeTimeout = setTimeout(() => {
        if (this.pendingZoneChange !== null) {
          const debouncedZone = this.pendingZoneChange;
          this.pendingZoneChange = null;

          // Only set zone if it's still different and still in active zones
          if (
            debouncedZone !== this.currentZone &&
            this.activeZones.has(debouncedZone)
          ) {
            this.setZone(debouncedZone).catch((error) => {
              this.logger.error("Error setting zone:", error);
            });
          }
        }
        this.zoneChangeTimeout = null;
      }, this.zoneChangeDebounceDelay);
    }
  }

  /**
   * Remove a zone from the active zones set (called by ColliderManager when exiting)
   * @param {string} zoneName - Zone name
   */
  removeActiveZone(zoneName) {
    if (!this.isActive) return;

    // Only remove if actually active (prevents duplicate calls from physics jitter)
    if (!this.activeZones.has(zoneName)) {
      return;
    }

    this.activeZones.delete(zoneName);
    this.logger.log(
      `[Zone] Removed zone "${zoneName}" from active zones: ${
        Array.from(this.activeZones).join(", ") || "none"
      }`
    );

    // If we removed the current zone and there are other active zones, switch to one of them using priority
    if (zoneName === this.currentZone && this.activeZones.size > 0) {
      // Use same priority order as addActiveZone
      const priorityOrder = [
        "alleyIntro",
        "alleyNavigable",
        "alleyLongView",
        "threeWay2",
        "threeWay",
        "plaza",
        "fourWay",
      ];

      // Find highest priority zone that's active
      let newZone = null;
      for (const priorityZone of priorityOrder) {
        if (this.activeZones.has(priorityZone)) {
          newZone = priorityZone;
          break;
        }
      }

      // Fallback
      if (!newZone) {
        newZone = Array.from(this.activeZones)[0];
      }

      this.logger.log(
        `[Zone] Current zone "${zoneName}" removed. Selected new zone "${newZone}" from active zones: ${Array.from(
          this.activeZones
        ).join(", ")}`
      );

      if (newZone !== this.currentZone) {
        // Debounce the switch
        this.pendingZoneChange = newZone;

        if (this.zoneChangeTimeout) {
          clearTimeout(this.zoneChangeTimeout);
        }

        this.zoneChangeTimeout = setTimeout(() => {
          if (this.pendingZoneChange !== null) {
            const debouncedZone = this.pendingZoneChange;
            this.pendingZoneChange = null;

            if (
              debouncedZone !== this.currentZone &&
              this.activeZones.has(debouncedZone)
            ) {
              // Zone detection should only be enabled via enableZoneDetection()
              // Don't auto-enable here to prevent false positives during init
              this.setZone(debouncedZone).catch((error) => {
                this.logger.error("Error switching zone after exit:", error);
              });
            }
          }
          this.zoneChangeTimeout = null;
        }, this.zoneChangeDebounceDelay);
      }
    }
    // NOTE: We don't clear zones when activeZones.size === 0 here
    // Zones are only cleared when actually leaving the exterior area (via handleStateChange)
    // This prevents clearing zones during brief boundary transitions
  }
}

export default ZoneManager;
