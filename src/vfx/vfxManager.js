import { Logger } from "../utils/logger.js";

/**
 * VFXStateManager - Base class for state-driven VFX systems
 *
 * Provides reusable game state management for VFX effects.
 * VFX systems can extend this class to automatically respond to game state changes.
 * All VFX effects are defined in a single centralized vfxData.js file.
 *
 * Usage:
 *
 * 1. Add your VFX effects to vfxData.js:
 *    export const myVfxEffects = {
 *      effectName: {
 *        id: "effectName",
 *        parameters: { ... }, // Your VFX-specific parameters
 *        criteria: { currentState: { $gte: GAME_STATES.INTRO } },
 *        priority: 10,
 *      },
 *    };
 *
 *    // Add to vfxEffects object:
 *    export const vfxEffects = {
 *      desaturation: desaturationEffects,
 *      myVfx: myVfxEffects, // <-- Add your effects here
 *    };
 *
 * 2. Your VFX class extends VFXStateManager:
 *    class MyVFX extends VFXStateManager {
 *      constructor() {
 *        super("MyVFX"); // Pass logger name
 *        // Your VFX initialization
 *      }
 *
 *      applyEffect(effect, state) {
 *        // Apply the effect parameters to your VFX
 *        const params = effect.parameters || {};
 *        this.opacity = params.opacity;
 *        this.color = params.color;
 *        // etc.
 *      }
 *    }
 *
 * 3. In main.js, connect to game manager with just the VFX type name:
 *    myVfx.setGameManager(gameManager, "myVfx");
 *
 * That's it! The VFX will automatically load effects from vfxData.js and respond to state changes.
 */
export class VFXStateManager {
  /**
   * @param {string} loggerName - Name for the logger (e.g., "MyVFX")
   * @param {boolean} debugMode - Enable debug logging
   */
  constructor(loggerName = "VFXStateManager", debugMode = false) {
    this.gameManager = null;
    this.currentEffectId = null;
    this.logger = new Logger(loggerName, debugMode);
    this._getEffectForState = null;
    this._hasEverBeenEnabled = false; // Track if VFX has ever been activated
  }

  /**
   * Set game manager and register event listeners
   * @param {GameManager} gameManager - The game manager instance
   * @param {string|Function} vfxTypeOrGetter - Either:
   *   - A string VFX type name (e.g., 'desaturation') - will auto-import vfxData.js
   *   - A function that returns the effect for a given state: (gameState) => effect
   */
  setGameManager(gameManager, vfxTypeOrGetter) {
    this.gameManager = gameManager;

    // State change handler setup
    const setupListener = (getEffectFn) => {
      this._getEffectForState = getEffectFn;

      // State change handler
      const handleStateChange = (newState, oldState) => {
        this.updateForState(newState);
      };

      // Listen for state changes
      this.gameManager.on("state:changed", handleStateChange);

      // Handle initial state
      const currentState = this.gameManager.getState();
      handleStateChange(currentState, null);

      this.logger.log("Event listeners registered and initial state handled");
    };

    // If it's a string, auto-import vfxData.js and use the type
    if (typeof vfxTypeOrGetter === "string") {
      const vfxType = vfxTypeOrGetter;
      import("./vfxData.js").then((module) => {
        const getEffectFn = (gameState) =>
          module.getVfxEffectForState(vfxType, gameState);
        setupListener(getEffectFn);
      });
    }
    // If it's a function, use it directly
    else if (typeof vfxTypeOrGetter === "function") {
      setupListener(vfxTypeOrGetter);
    } else {
      this.logger.error(
        "vfxTypeOrGetter must be a string (VFX type) or function (getter)"
      );
    }
  }

  /**
   * Update VFX effect based on current game state
   * Uses criteria to determine which effect should be active
   * @param {Object} state - Current game state
   */
  updateForState(state) {
    if (!state || !this._getEffectForState) return;

    const effect = this._getEffectForState(state);

    // If no effect matches
    if (!effect) {
      // Only call onNoEffect if we've previously been enabled
      if (this._hasEverBeenEnabled) {
        this.logger.log("No effect matches current state - disabling");
        this.onNoEffect(state);
        this.currentEffectId = null;
      } else {
        this.logger.log("No effect matches current state (VFX never enabled)");
      }
      return;
    }

    // Enable VFX if this is the first time we have a matching effect
    if (!this._hasEverBeenEnabled) {
      this.logger.log("First effect match - enabling VFX");
      this._hasEverBeenEnabled = true;
      this.onFirstEnable(effect, state);
    }

    // Skip if this is the same effect already applied (unless you want to re-apply)
    if (
      this.currentEffectId === effect.id &&
      !this.shouldReapplyEffect(effect)
    ) {
      return;
    }

    this.logger.log(`Applying effect: ${effect.id}`, effect);

    // Store current effect ID
    this.currentEffectId = effect.id;

    // Call hook to apply effect
    this.applyEffect(effect, state);
  }

  /**
   * Hook: Called the first time an effect matches (VFX should enable itself)
   * Override this method to initialize/enable your VFX on first use
   * @param {Object} effect - Effect data from your data file
   * @param {Object} state - Current game state
   */
  onFirstEnable(effect, state) {
    // Default: just apply the effect
    this.applyEffect(effect, state);
  }

  /**
   * Hook: Apply effect parameters to your VFX system
   * Override this method in your VFX class
   * @param {Object} effect - Effect data from your data file
   * @param {Object} state - Current game state
   */
  applyEffect(effect, state) {
    this.logger.warn(
      "applyEffect() not implemented. Override this method in your VFX class."
    );
  }

  /**
   * Hook: Called when no effect matches current state (only after VFX has been enabled once)
   * Override this method to disable/hide your VFX when no longer needed
   * @param {Object} state - Current game state
   */
  onNoEffect(state) {
    // Default: do nothing
  }

  /**
   * Hook: Determine if an effect should be re-applied even if it's the same ID
   * Override this method if you want to re-apply effects on state changes
   * @param {Object} effect - Effect data
   * @returns {boolean}
   */
  shouldReapplyEffect(effect) {
    return false; // Default: don't re-apply same effect
  }

  /**
   * Get the current effect ID
   * @returns {string|null}
   */
  getCurrentEffectId() {
    return this.currentEffectId;
  }

  /**
   * Check if a specific effect is currently active
   * @param {string} effectId - Effect ID to check
   * @returns {boolean}
   */
  isEffectActive(effectId) {
    return this.currentEffectId === effectId;
  }
}

/**
 * Helper function to find matching effect from effects object
 * Useful for creating getEffectForState functions in data files
 *
 * @param {Object} effects - Object with effect definitions
 * @param {Object} gameState - Current game state
 * @param {Function} checkCriteria - Criteria checking function
 * @returns {Object|null} Matching effect or null
 */
export function findMatchingEffect(effects, gameState, checkCriteria) {
  // Convert to array and sort by priority (highest first)
  const effectsArray = Object.values(effects).sort(
    (a, b) => (b.priority || 0) - (a.priority || 0)
  );

  // Return first matching effect
  for (const effect of effectsArray) {
    if (effect.criteria && checkCriteria(gameState, effect.criteria)) {
      return effect;
    }
  }

  return null;
}

export default VFXStateManager;
