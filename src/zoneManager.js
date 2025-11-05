import { Logger } from "./utils/logger.js";
import { GAME_STATES } from "./gameData.js";

/**
 * ZoneManager - Manages loading/unloading of exterior splat zones based on player location
 *
 * Zone mapping:
 * - IntroAlley: loads IntroAlley + FourWay
 * - FourWay: loads FourWay + IntroAlley + ThreeWay + Plaza
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

    // Zone to splat mapping
    this.zoneToSplats = {
      introAlley: ["introAlley", "fourWay"],
      fourWay: ["fourWay", "introAlley", "threeWay", "plaza"],
      threeWay: ["threeWay", "fourWay", "threeWay2"],
      threeWay2: ["threeWay2", "threeWay", "plaza"],
      plaza: ["plaza", "threeWay2", "fourWay"],
    };

    // Track currently loaded splats
    this.loadedSplats = new Set();

    // Current zone (null when not in exterior)
    this.currentZone = null;

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
   * Handle game state changes
   * @param {Object} newState - New game state
   * @param {Object} oldState - Previous game state
   */
  handleStateChange(newState, oldState) {
    if (!newState) return;

    const wasActive = this.isActive;
    this.isActive = newState.currentState < GAME_STATES.OFFICE_INTERIOR;

    // If we transitioned out of exterior, unload all exterior splats
    if (wasActive && !this.isActive) {
      this.logger.log("Exiting exterior area - unloading all exterior splats");
      this.unloadAllExteriorSplats();
      this.currentZone = null;
      return;
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
      // Don't set zone here - let collision detection handle it via addActiveZone
      return;
    }

    // Zone changes are now handled entirely by collision detection (addActiveZone/removeActiveZone)
    // We don't react to currentZone state changes here to avoid feedback loops
  }

  /**
   * Set the current zone and load/unload splats accordingly
   * @param {string} zone - Zone name (e.g., "introAlley", "fourWay")
   */
  async setZone(zone) {
    if (!this.isActive) {
      this.logger.warn(
        `Cannot set zone "${zone}" - ZoneManager is not active (must be before OFFICE_INTERIOR)`
      );
      return;
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

    const oldZone = this.currentZone;
    this.currentZone = zone;

    this.logger.log(`Zone changed: ${oldZone || "none"} -> ${zone}`);

    // Determine which splats should be loaded for this zone
    const requiredSplats = this.zoneToSplats[zone] || [];
    this.logger.log(
      `Required splats for zone "${zone}": ${requiredSplats.join(", ")}`
    );
    this.logger.log(
      `Currently loaded splats: ${Array.from(this.loadedSplats).join(", ")}`
    );

    // First, verify which splats are already loaded and in scene - these should NOT be touched
    const alreadyLoadedAndInScene = new Set();
    for (const splatId of requiredSplats) {
      if (this.loadedSplats.has(splatId)) {
        const object = this.sceneManager.getObject(splatId);
        if (object && object.parent === this.sceneManager.scene) {
          // Already loaded and in scene - mark it so we don't touch it
          alreadyLoadedAndInScene.add(splatId);
          this.logger.log(
            `Splat "${splatId}" already loaded and in scene - will not touch`
          );
        } else {
          this.logger.log(
            `Splat "${splatId}" is in loadedSplats but not in scene (object: ${!!object}, parent: ${
              object?.parent?.constructor?.name || "none"
            })`
          );
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
            this.logger.log(`Restored splat "${splatId}" to scene`);
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

    this.logger.log(
      `Zone "${zone}" configured - Loaded: ${Array.from(this.loadedSplats).join(
        ", "
      )}`
    );
  }

  /**
   * Load a splat if it's not already loaded
   * @param {string} splatId - Splat ID (e.g., "introAlley", "fourWay")
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

    // Check if object exists in sceneData
    const { sceneObjects } = await import("./sceneData.js");
    const objectData = sceneObjects[splatId];

    if (!objectData) {
      this.logger.warn(`Splat "${splatId}" not found in sceneData`);
      return;
    }

    // Check if already loaded in sceneManager but not in scene
    if (this.sceneManager.hasObject(splatId)) {
      // Object is loaded but not in scene - add it
      if (this.sceneManager.objectsNotInScene.has(splatId)) {
        this.sceneManager.addObjectToScene(splatId);
        this.loadedSplats.add(splatId);
        this.logger.log(`Added existing splat "${splatId}" to scene`);
        return;
      }
      // Object already in scene but not tracked - add to tracking
      const object = this.sceneManager.getObject(splatId);
      if (object && object.parent === this.sceneManager.scene) {
        this.loadedSplats.add(splatId);
        this.logger.log(`Tracked existing splat "${splatId}" in scene`);
        return;
      }
    }

    // Load the splat
    this.logger.log(`Loading splat "${splatId}"...`);
    try {
      await this.sceneManager.loadObject(objectData, false); // false = add to scene immediately
      this.loadedSplats.add(splatId);
      this.logger.log(`✓ Loaded splat "${splatId}"`);
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
      const requiredSplats = this.zoneToSplats[this.currentZone] || [];
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
      this.loadedSplats.delete(splatId);
      this.logger.log(`Unloaded splat "${splatId}" (removed from scene)`);
    } else {
      this.loadedSplats.delete(splatId);
      this.logger.log(`Unloaded splat "${splatId}" (was not in scene)`);
    }
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

    // Only add if not already active (prevents duplicate calls from physics jitter)
    if (this.activeZones.has(zoneName)) {
      return;
    }

    this.activeZones.add(zoneName);
    // Don't log every add - only log when currentZone actually changes

    // Determine which zone to use when multiple are active
    // Priority: use the zone that matches currentZone if it's still active, otherwise use the first active zone
    let targetZone = null;
    if (this.currentZone && this.activeZones.has(this.currentZone)) {
      // Current zone is still active - keep it (prevents switching at boundaries)
      targetZone = this.currentZone;
    } else if (this.activeZones.size > 0) {
      // Current zone is not active, but others are - switch to first active zone
      targetZone = Array.from(this.activeZones)[0];
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
    // Don't log every remove - only log when currentZone actually changes

    // If we removed the current zone and there are other active zones, switch to one of them
    if (zoneName === this.currentZone && this.activeZones.size > 0) {
      const newZone = Array.from(this.activeZones)[0];
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
