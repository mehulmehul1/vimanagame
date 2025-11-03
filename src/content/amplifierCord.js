import * as THREE from "three";
import { Logger } from "../utils/logger.js";
import { GAME_STATES } from "../gameData.js";
import { checkCriteria } from "../utils/criteriaHelper.js";
import PhoneCord from "./phoneCord.js";

/**
 * AmplifierCord - Manages physics-based cord from amplifier to viewmaster
 *
 * Features:
 * - Physics-based cord with chain segments
 * - Two rigid segments at amplifier end (similar to phonebooth)
 * - Connects amplifier's CordAttach to viewmaster
 * - Only exists after LIGHTS_OUT state
 */
class AmplifierCord {
  constructor(options = {}) {
    this.sceneManager = options.sceneManager;
    this.physicsManager = options.physicsManager;
    this.scene = options.scene;
    this.logger = new Logger("AmplifierCord", false);

    // Cord components
    this.cordAttach = null; // CordAttach mesh from amplifier
    this.viewmaster = null; // Viewmaster object (attachment point)
    this.phoneCord = null; // PhoneCord instance

    // Configuration
    this.config = {
      // Cord configuration (long cord spanning several meters, touching ground)
      cordConfig: {
        cordSegments: 50, // Many segments for a long cord
        cordSegmentLength: 0.08, // Longer segments (50 * 0.08 = 4m total length)
        cordSegmentRadius: 0.002,
        cordMass: 0.002,
        cordDamping: 8.0,
        cordAngularDamping: 8.0,
        cordDroopAmount: 8, // Heavy droop so it hangs down and touches ground
        cordRigidSegments: 2, // Two rigid segments at amplifier end
        cordColor: 0x808080, // Grey color
        cordVisualRadius: 0.008,
        cordMetalness: 0.3,
        cordRoughness: 0.8,
        initMode: "horizontal", // Extends horizontally like phonebooth
        // Collision groups: Belongs to group 2, collides with group 3 (Environment)
        cordCollisionGroup: 0x00040002,
      },

      // Cord existence criteria - cord should only exist after LIGHTS_OUT
      cordCriteria: {
        currentState: { $gte: GAME_STATES.LIGHTS_OUT },
      },
    };
  }

  /**
   * Initialize the amplifier cord
   * Sets up event listeners and creates the cord
   */
  initialize(gameManager = null) {
    if (!this.sceneManager) {
      this.logger.warn("No SceneManager provided");
      return;
    }

    this.gameManager = gameManager;

    // Check if amplifier object is loaded before initializing
    if (!this.sceneManager.hasObject("amplifier")) {
      this.logger.log("Amplifier not loaded, skipping initialization");
      return;
    }

    // Listen for game state changes
    if (this.gameManager) {
      this.gameManager.on("state:changed", (newState, oldState) => {
        // Check if phone cord should be created/destroyed based on criteria
        if (this.config.cordCriteria) {
          const criteriaMet = checkCriteria(newState, this.config.cordCriteria);
          if (criteriaMet && !this.phoneCord) {
            // Create cord if criteria met and doesn't exist
            this.createCord();
          } else if (!criteriaMet && this.phoneCord) {
            // Destroy cord if criteria no longer met
            this.logger.log("🗑️ Cord criteria no longer met, destroying cord");
            this.phoneCord.destroy();
            this.phoneCord = null;
          }
        }
      });
    }

    // Find the CordAttach mesh from amplifier
    this.cordAttach = this.sceneManager.findChildByName(
      "amplifier",
      "CordAttach"
    );

    // Find the viewmaster object
    this.viewmaster = this.sceneManager.getObject("viewmaster");

    if (!this.cordAttach) {
      this.logger.warn(
        "CordAttach mesh not found in amplifier model"
      );
    }

    if (!this.viewmaster) {
      this.logger.warn("Viewmaster object not found");
    }

    // Create the cord if criteria is met and components are present
    if (
      this.gameManager &&
      this.cordAttach &&
      this.viewmaster &&
      this.physicsManager
    ) {
      const currentState = this.gameManager.getState();
      const criteriaMet = checkCriteria(
        currentState,
        this.config.cordCriteria
      );

      if (criteriaMet) {
        this.createCord();
      }
    }

    this.logger.log("Initialized");
  }

  /**
   * Create the phone cord using the PhoneCord module
   */
  createCord() {
    if (!this.cordAttach || !this.viewmaster || !this.physicsManager) {
      this.logger.warn(
        "Cannot create cord - missing CordAttach, Viewmaster, or PhysicsManager"
      );
      return;
    }

    // Create the phone cord using the PhoneCord module
    this.phoneCord = new PhoneCord({
      scene: this.scene,
      physicsManager: this.physicsManager,
      cordAttach: this.cordAttach,
      receiver: this.viewmaster, // Viewmaster acts as the "receiver" attachment point
      loggerName: "AmplifierCord.Cord",
      config: this.config.cordConfig,
    });

    const success = this.phoneCord.createCord();
    if (success) {
      this.logger.log("Amplifier cord created successfully");
    } else {
      this.logger.warn("Failed to create amplifier cord");
    }
  }

  /**
   * Update method - call in animation loop
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    // Update the phone cord (handles kinematic anchor and visual line)
    if (this.phoneCord) {
      this.phoneCord.update();
    }
  }

  /**
   * Get phone cord instance
   * @returns {PhoneCord|null}
   */
  getPhoneCord() {
    return this.phoneCord;
  }

  /**
   * Set visibility of the cord
   * @param {boolean} visible - Whether the cord should be visible
   */
  setCordVisibility(visible) {
    if (!this.phoneCord) return;

    // PhoneCord uses cordLineMesh for rendering
    if (this.phoneCord.cordLineMesh) {
      this.phoneCord.cordLineMesh.visible = visible;
      this.logger.log(`Amplifier cord visibility: ${visible}`);
    }
  }

  /**
   * Clean up resources
   */
  destroy() {
    // Destroy the phone cord
    if (this.phoneCord) {
      this.phoneCord.destroy();
      this.phoneCord = null;
    }

    this.cordAttach = null;
    this.viewmaster = null;
    this.logger.log("Destroyed");
  }
}

export default AmplifierCord;

