import { Logger } from "./utils/logger.js";

/**
 * VFXManager - Base class for state-driven VFX systems
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
 *        delay: 2.0, // Optional: delay in seconds before applying effect
 *      },
 *    };
 *
 *    // Add to vfxEffects object:
 *    export const vfxEffects = {
 *      desaturation: desaturationEffects,
 *      myVfx: myVfxEffects, // <-- Add your effects here
 *    };
 *
 * 2. Your VFX class extends VFXManager:
 *    class MyVFX extends VFXManager {
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
export class VFXManager {
  /**
   * @param {string} loggerName - Name for the logger (e.g., "MyVFX")
   * @param {boolean} debugMode - Enable debug logging
   */
  constructor(loggerName = "VFXManager", debugMode = false) {
    this.gameManager = null;
    this.currentEffectId = null;
    this.logger = new Logger(loggerName, debugMode);
    this._getEffectForState = null;
    this._hasEverBeenEnabled = false; // Track if VFX has ever been activated
    this._pendingEffect = null; // Effect waiting for delay
    this._delayTimeout = null; // Timeout for delayed effects
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
      // Cancel any pending delayed effect
      this._cancelDelayedEffect();

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

    // Skip if this is the same effect already applied or pending (unless you want to re-apply)
    const isSameEffect =
      this.currentEffectId === effect.id ||
      (this._pendingEffect && this._pendingEffect.id === effect.id);

    if (isSameEffect && !this.shouldReapplyEffect(effect)) {
      return;
    }

    // If a different effect is now matching, cancel any pending delayed effect
    if (this._pendingEffect && this._pendingEffect.id !== effect.id) {
      this._cancelDelayedEffect();
    }

    // Check if effect has a delay
    const delay = effect.delay || 0;

    if (delay > 0) {
      // Effect should be delayed
      this.logger.log(
        `Effect ${effect.id} will be applied in ${delay} seconds`
      );
      this._pendingEffect = effect;

      // Set timeout to apply effect after delay
      this._delayTimeout = setTimeout(() => {
        this.logger.log(`Applying delayed effect: ${effect.id}`, effect);
        this.currentEffectId = effect.id;
        this._pendingEffect = null;
        this._delayTimeout = null;
        this.applyEffect(effect, state);
      }, delay * 1000);
    } else {
      // No delay, apply immediately
      this.logger.log(`Applying effect: ${effect.id}`, effect);
      this.currentEffectId = effect.id;
      this.applyEffect(effect, state);
    }
  }

  /**
   * Cancel any pending delayed effect
   * @private
   */
  _cancelDelayedEffect() {
    if (this._delayTimeout) {
      clearTimeout(this._delayTimeout);
      this._delayTimeout = null;
      if (this._pendingEffect) {
        this.logger.log(`Cancelled delayed effect: ${this._pendingEffect.id}`);
      }
      this._pendingEffect = null;
    }
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

/**
 * VFXSystemManager - Coordinates all VFX effects in the system
 *
 * Creates and manages all VFX effects as a group.
 * Main.js should create one instance of this instead of manually setting up each effect.
 *
 * Usage in main.js:
 *   const vfxManager = new VFXSystemManager(scene, camera, renderer, loadingScreen);
 *   await vfxManager.initialize(sceneManager); // Pass sceneManager for splatMorph
 *   vfxManager.setGameManager(gameManager);
 *   vfxManager.update(deltaTime);
 *   vfxManager.render(scene, camera);
 */
export class VFXSystemManager {
  constructor(scene, camera, renderer, loadingScreen = null) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.loadingScreen = loadingScreen;
    this.logger = new Logger("VFXSystemManager", true);

    this.effects = {};
    this.gameManager = null;

    this.logger.log("Initializing VFX system");
  }

  /**
   * Initialize all VFX effects
   * @param {Object} sceneManager - SceneManager instance (required for splatMorph)
   */
  async initialize(sceneManager = null) {
    // Register loading task
    if (this.loadingScreen) {
      this.loadingScreen.registerTask("vfx-system", 1);
    }

    this.logger.log("Initializing VFX effects...");

    try {
      // Dynamic imports to avoid circular dependency (VFX extends VFXManager)
      const { DesaturationEffect } = await import(
        "./vfx/desaturationEffect.js"
      );
      const { createCloudParticlesShader } = await import(
        "./vfx/cloudParticlesShader.js"
      );
      const { SplatFractalEffect } = await import(
        "./vfx/splatFractalEffect.js"
      );
      const { SplatMorphEffect } = await import("./vfx/splatMorph.js");

      // Create desaturation post-processing effect
      this.logger.log("Creating DesaturationEffect...");
      this.effects.desaturation = new DesaturationEffect(this.renderer);
      this.logger.log("✅ DesaturationEffect created");

      // Create cloud particles shader
      this.logger.log("Creating CloudParticles...");
      this.effects.cloudParticles = createCloudParticlesShader(
        this.scene,
        this.camera
      );
      this.logger.log("✅ CloudParticles created");

      // Create splat fractal effect (requires sceneManager)
      if (sceneManager) {
        this.logger.log("Creating SplatFractalEffect...");
        this.effects.splatFractal = new SplatFractalEffect(
          this.scene,
          sceneManager
        );
        this.logger.log("✅ SplatFractalEffect created");
      }

      // Create splat morph effect (requires sceneManager)
      if (sceneManager) {
        this.logger.log("Creating SplatMorphEffect...");
        this.effects.splatMorph = new SplatMorphEffect(
          this.scene,
          sceneManager
        );
        this.logger.log("✅ SplatMorphEffect created");
      } else {
        this.logger.warn(
          "⚠️ SceneManager not provided - Splat effects not initialized"
        );
      }

      this.logger.log("VFX effects initialized:", Object.keys(this.effects));

      // Complete loading task
      if (this.loadingScreen) {
        this.loadingScreen.completeTask("vfx-system");
      }
    } catch (error) {
      this.logger.error("❌ Failed to initialize VFX:", error);
      throw error;
    }
  }

  /**
   * Connect all VFX effects to game manager
   * @param {GameManager} gameManager - The game manager instance
   */
  setGameManager(gameManager) {
    this.gameManager = gameManager;

    // Connect each effect to game manager
    if (this.effects.desaturation) {
      this.effects.desaturation.setGameManager(gameManager, "desaturation");
    }

    if (this.effects.cloudParticles) {
      this.effects.cloudParticles.setGameManager(gameManager, "cloudParticles");
    }

    if (this.effects.splatFractal) {
      this.effects.splatFractal.setGameManager(gameManager, "splatFractal");
    }

    if (this.effects.splatMorph) {
      this.effects.splatMorph.setGameManager(gameManager, "splatMorph");
    }

    this.logger.log("All VFX effects connected to game manager");
  }

  /**
   * Update all VFX effects
   * @param {number} deltaTime - Time since last frame in seconds
   */
  update(deltaTime) {
    if (this.effects.desaturation) {
      this.effects.desaturation.update(deltaTime);
    }

    if (this.effects.cloudParticles) {
      this.effects.cloudParticles.update(deltaTime);
    }

    if (this.effects.splatFractal) {
      this.effects.splatFractal.update(deltaTime);
    }

    if (this.effects.splatMorph) {
      this.effects.splatMorph.update(deltaTime);
    }
  }

  /**
   * Render scene with post-processing effects
   * @param {THREE.Scene} scene - The scene to render
   * @param {THREE.Camera} camera - The camera to use
   */
  render(scene, camera) {
    // Desaturation effect handles rendering (includes post-processing)
    if (this.effects.desaturation) {
      this.effects.desaturation.render(scene, camera);
    } else {
      // Fallback to direct rendering if no post-processing
      this.renderer.render(scene, camera);
    }
  }

  /**
   * Handle window resize for all effects
   * @param {number} width - New width
   * @param {number} height - New height
   */
  setSize(width, height) {
    if (this.effects.desaturation) {
      this.effects.desaturation.setSize(width, height);
    }
  }

  /**
   * Get a specific effect by name
   * @param {string} name - Effect name (e.g., 'desaturation', 'cloudParticles')
   * @returns {Object|null} The effect or null
   */
  getEffect(name) {
    return this.effects[name] || null;
  }

  /**
   * Dispose of all VFX resources
   */
  dispose() {
    Object.values(this.effects).forEach((effect) => {
      if (effect.dispose) {
        effect.dispose();
      }
    });
    this.effects = {};
  }
}

export default VFXManager;
