import { getSceneObjectsForState } from "./sceneData.js";
import { startScreen, GAME_STATES } from "./gameData.js";
import {
  getDebugSpawnState,
  isDebugSpawnActive,
} from "./utils/debugSpawner.js";
import PhoneBooth from "./content/phonebooth.js";
import CandlestickPhone from "./content/candlestickPhone.js";
import AmplifierCord from "./content/amplifierCord.js";
import VideoManager from "./videoManager.js";
import { Logger } from "./utils/logger.js";

/**
 * GameManager - Central game state and event management
 *
 * Features:
 * - Manage game state
 * - Trigger events
 * - Coordinate between different systems
 */

class GameManager {
  constructor() {
    // Check for debug spawn state first
    const debugState = getDebugSpawnState();
    this.state = debugState ? { ...debugState } : { ...startScreen };
    this.isDebugMode = isDebugSpawnActive();

    // Logger for debug messages
    this.logger = new Logger("GameManager", true);

    if (this.isDebugMode) {
      this.logger.log("Debug mode active", this.state);
    }

    this.eventListeners = {};
    this.dialogManager = null;
    this.musicManager = null;
    this.sfxManager = null;
    this.uiManager = null;
    this.sceneManager = null;
    this.phoneBooth = null;
    this.candlestickPhone = null;
    this.amplifierCord = null;

    // Track loaded scene objects
    this.loadedScenes = new Set();
    this.previousObjectsToLoad = null; // Track previous objects list to avoid spam logging

    // Throttle non-currentState change logs
    this.lastStateLogTime = 0;

    // Parse URL parameters on construction
    this.urlParams = this.parseURLParams();
  }

  /**
   * Parse URL parameters
   * @returns {Object} Object with URL parameters
   */
  parseURLParams() {
    const params = {};
    const searchParams = new URLSearchParams(window.location.search);

    for (const [key, value] of searchParams) {
      params[key] = value;
    }

    this.logger.log("URL params:", params);
    return params;
  }

  /**
   * Get a URL parameter value
   * @param {string} key - Parameter name
   * @returns {string|null} Parameter value or null if not found
   */
  getURLParam(key) {
    return this.urlParams[key] || null;
  }

  /**
   * Get the debug spawn character position if in debug mode
   * @returns {Object|null} Position {x, y, z} or null
   */
  getDebugSpawnPosition() {
    if (!this.isDebugMode || !this.state.playerPosition) {
      return null;
    }
    return { ...this.state.playerPosition };
  }

  /**
   * Get the debug spawn character rotation if in debug mode
   * @returns {Object|null} Rotation {x, y, z} in degrees or null
   */
  getDebugSpawnRotation() {
    if (!this.isDebugMode || !this.state.playerRotation) {
      return null;
    }
    return { ...this.state.playerRotation };
  }

  /**
   * Initialize with managers
   * @param {Object} managers - Object containing manager instances
   */
  async initialize(managers = {}) {
    this.dialogManager = managers.dialogManager;
    this.musicManager = managers.musicManager;
    this.sfxManager = managers.sfxManager;
    this.uiManager = managers.uiManager;
    this.characterController = managers.characterController;
    this.cameraAnimationManager = managers.cameraAnimationManager;
    this.sceneManager = managers.sceneManager;
    this.lightManager = managers.lightManager;
    this.inputManager = managers.inputManager;
    this.physicsManager = managers.physicsManager; // Store physics manager reference
    this.camera = managers.camera; // Store camera reference
    this.scene = managers.scene; // Store scene reference
    // Add other managers as needed

    // Set up internal event handlers
    this.setupEventHandlers();

    // Handle initial state (important for debug spawning with controlEnabled: true)
    if (this.state.controlEnabled === true) {
      this.updateCharacterController();
    }

    // Load initial scene objects based on starting state
    // Only load preload: true objects during initial loading
    if (this.sceneManager) {
      await this.updateSceneForState({ preloadOnly: true });
      // Trigger initial animation check after loading
      this.sceneManager.updateAnimationsForState(this.state);
    }

    // Check if candlestickPhone was already loaded (e.g., debug spawn at POST_DRIVE_BY or later)
    if (
      !this.candlestickPhone &&
      this.state.currentState >= GAME_STATES.POST_DRIVE_BY &&
      this.sceneManager?.hasObject("candlestickPhone")
    ) {
      this.logger.log(
        "Initializing candlestick phone (object loaded at startup)"
      );
      this.candlestickPhone = new CandlestickPhone({
        sceneManager: this.sceneManager,
        physicsManager: this.physicsManager,
        scene: this.scene,
        camera: this.camera,
      });
      this.candlestickPhone.initialize(this);
    }

    // Check if amplifier cord should be initialized at startup (e.g., debug spawn at LIGHTS_OUT or later)
    if (
      !this.amplifierCord &&
      this.state.currentState >= GAME_STATES.LIGHTS_OUT &&
      this.sceneManager?.hasObject("amplifier") &&
      this.sceneManager?.hasObject("viewmaster")
    ) {
      this.logger.log(
        "Initializing amplifier cord (objects loaded at startup)"
      );
      this.amplifierCord = new AmplifierCord({
        sceneManager: this.sceneManager,
        physicsManager: this.physicsManager,
        scene: this.scene,
      });
      this.amplifierCord.initialize(this);
    }

    // Initialize content-specific systems AFTER scene is loaded
    this.phoneBooth = new PhoneBooth({
      sceneManager: this.sceneManager,
      lightManager: this.lightManager,
      sfxManager: this.sfxManager,
      physicsManager: managers.physicsManager,
      scene: managers.scene,
      camera: this.camera,
      characterController: this.characterController,
    });
    this.phoneBooth.initialize(this);

    // Initialize drawing manager
    // DISABLED - using DrawingGame instead
    // this.drawingManager = new DrawingManager({
    //   sceneManager: this.sceneManager,
    //   scene: managers.scene,
    //   camera: this.camera,
    //   renderer: managers.renderer,
    // });
    // this.drawingManager.initialize(this);

    // Note: candlestickPhone will be initialized when its scene object loads (POST_DRIVE_BY state)

    // Initialize video manager with state-based playback
    // Note: loadingScreen will be set after initialization via setManagers
    this.videoManager = new VideoManager({
      scene: managers.scene,
      gameManager: this,
      camera: this.camera,
      loadingScreen: null, // Will be set in main.js after VideoManager is created
    });

    // Note: Music, dialogs, SFX, and videos are now handled by their respective managers via state:changed events
    // They handle initial state when their listeners are set up
  }

  /**
   * Set up internal event handlers for game-level logic
   * Note: Individual managers (CharacterController, CameraAnimationManager,
   * MusicManager, DialogManager, SFXManager) now handle their own events directly
   */
  setupEventHandlers() {
    // Listen for character controller enable/disable to manage input
    this.on("character-controller:enabled", () => {
      if (this.inputManager) {
        this.inputManager.enable();
        this.inputManager.showTouchControls();
      }
    });

    this.on("character-controller:disabled", () => {
      if (this.inputManager) {
        this.inputManager.disable();
        this.inputManager.hideTouchControls();
      }
    });

    // Initialize candlestick phone when state reaches POST_DRIVE_BY
    this.on("state:changed", (newState, oldState) => {
      if (
        !this.candlestickPhone &&
        newState.currentState >= GAME_STATES.POST_DRIVE_BY &&
        this.sceneManager?.hasObject("candlestickPhone")
      ) {
        this.logger.log("Initializing candlestick phone (object loaded)");
        this.candlestickPhone = new CandlestickPhone({
          sceneManager: this.sceneManager,
          physicsManager: this.physicsManager,
          scene: this.scene,
          camera: this.camera,
        });
        this.candlestickPhone.initialize(this);
      }

      // Initialize amplifier cord when state reaches LIGHTS_OUT
      if (
        !this.amplifierCord &&
        newState.currentState >= GAME_STATES.LIGHTS_OUT &&
        this.sceneManager?.hasObject("amplifier") &&
        this.sceneManager?.hasObject("viewmaster")
      ) {
        this.logger.log("Initializing amplifier cord (objects loaded)");
        this.amplifierCord = new AmplifierCord({
          sceneManager: this.sceneManager,
          physicsManager: this.physicsManager,
          scene: this.scene,
        });
        this.amplifierCord.initialize(this);
      }
    });
  }

  /**
   * Set game state
   * @param {Object} newState - State updates to apply
   */
  setState(newState) {
    const oldState = { ...this.state };
    this.state = { ...this.state, ...newState };

    // Log state changes with stack trace
    if (
      newState.currentState !== undefined &&
      newState.currentState !== oldState.currentState
    ) {
      this.logger.log(
        `[GameManager] currentState changed from ${oldState.currentState} to ${newState.currentState}`
      );
    } else if (Object.keys(newState).length > 0) {
      // Throttle this log to once per second
      const now = Date.now();
      if (now - this.lastStateLogTime >= 1000) {
        this.logger.log(
          "[GameManager] setState called with (no currentState change):",
          newState
        );
        this.lastStateLogTime = now;
      }
    }

    this.emit("state:changed", this.state, oldState);

    // Update scene objects based on new state (load new objects if needed)
    if (this.sceneManager && newState.currentState !== oldState.currentState) {
      this.updateSceneForState();
    }

    // Update scene animations based on new state
    if (this.sceneManager) {
      this.sceneManager.updateAnimationsForState(this.state);
    }

    // Update character controller only if controlEnabled changed
    if (
      newState.controlEnabled !== undefined &&
      newState.controlEnabled !== oldState.controlEnabled
    ) {
      this.updateCharacterController();
    }
  }

  /**
   * Get current state
   * @returns {Object}
   */
  getState() {
    return { ...this.state };
  }

  /**
   * Update character controller based on current game state
   */
  updateCharacterController() {
    if (!this.characterController) return;

    // Enable character controller when controlEnabled state is true
    if (this.state.controlEnabled === true) {
      this.logger.log("Enabling character controller");
      this.characterController.headbobEnabled = true;
      this.emit("character-controller:enabled");
    } else if (this.state.controlEnabled === false) {
      this.logger.log("Disabling character controller");
      this.characterController.headbobEnabled = false;
      this.emit("character-controller:disabled");
    }
  }

  /**
   * Update scene objects based on current game state
   * Loads new objects that match current state conditions
   * Unloads objects that no longer match current state conditions
   * @param {Object} options - Options object
   * @param {boolean} options.preloadOnly - If true, only load objects with preload: true
   */
  async updateSceneForState(options = {}) {
    if (!this.sceneManager) return;

    const objectsToLoad = getSceneObjectsForState(this.state, options);
    const objectIdsToLoad = new Set(objectsToLoad.map((obj) => obj.id));

    // Debug: Log which objects are being loaded (only if list changed)
    const currentObjectsList = objectsToLoad
      .map((obj) => obj.id)
      .sort()
      .join(",");
    if (
      objectsToLoad.length > 0 &&
      currentObjectsList !== this.previousObjectsToLoad
    ) {
      this.logger.log(
        `Objects to load for state ${this.state.currentState}: ${objectsToLoad
          .map((obj) => obj.id)
          .join(", ")}`
      );
      this.previousObjectsToLoad = currentObjectsList;
    }

    // Find objects that are loaded but should no longer be
    const objectsToUnload = Array.from(this.loadedScenes).filter(
      (id) => !objectIdsToLoad.has(id)
    );

    // Unload objects that no longer match criteria
    if (objectsToUnload.length > 0) {
      this.logger.log(
        `Unloading ${objectsToUnload.length} scene objects no longer needed`,
        objectsToUnload[0]
      );
      objectsToUnload.forEach((id) => {
        // Clean up script instances before removing object
        if (id === "candlestickPhone" && this.candlestickPhone) {
          this.logger.log("Destroying candlestickPhone script instance");
          this.candlestickPhone.destroy();
          this.candlestickPhone = null;
        }

        this.sceneManager.removeObject(id);
        this.loadedScenes.delete(id);
      });
    }

    // Filter out objects that are already loaded
    const newObjects = objectsToLoad.filter(
      (obj) => !this.loadedScenes.has(obj.id)
    );

    // Filter out objects with preload: false - these should only be loaded by ZoneManager
    // They're already prefetched but shouldn't be fully loaded until needed
    const objectsToActuallyLoad = newObjects.filter((obj) => {
      const objPreload = obj.preload !== undefined ? obj.preload : false;
      if (objPreload === false) {
        // Only skip if it's an exterior splat (managed by ZoneManager)
        // Other deferred objects might still be loaded if criteria match
        const isExteriorSplat =
          obj.type === "splat" &&
          obj.id !== "interior" &&
          obj.id !== "officeHell" &&
          obj.id !== "club";
        if (isExteriorSplat) {
          return false; // Skip - ZoneManager will load these when needed
        }
      }
      return true; // Load preload: true objects or non-exterior deferred objects
    });

    // Load new objects that match criteria (excluding deferred exterior splats)
    if (objectsToActuallyLoad.length > 0) {
      this.logger.log(
        `Loading ${objectsToActuallyLoad.length} new scene objects for state`
      );

      // Track loaded objects BEFORE loading to prevent duplicate loads during async operations
      objectsToActuallyLoad.forEach((obj) => this.loadedScenes.add(obj.id));

      await this.sceneManager.loadObjectsForState(objectsToActuallyLoad);
    }

    // Check for objects that are loaded but not in scene (deferred loading)
    // Add them to scene if criteria match
    for (const obj of objectsToLoad) {
      if (
        this.loadedScenes.has(obj.id) &&
        this.sceneManager.objectsNotInScene.has(obj.id)
      ) {
        this.sceneManager.addObjectToScene(obj.id);
      }
    }
  }

  /**
   * Load deferred scene objects (preload: false) - prefetch files but don't fully initialize
   * Called after loading screen completes
   * Files are fetched but objects aren't fully loaded until criteria match
   */
  async loadDeferredSceneObjects() {
    if (!this.sceneManager) return;

    // Import sceneObjects to get all objects directly
    const { sceneObjects } = await import("./sceneData.js");

    // Find all objects with preload: false that aren't already loaded
    const deferredObjects = [];
    for (const [id, obj] of Object.entries(sceneObjects)) {
      const objPreload = obj.preload !== undefined ? obj.preload : false;
      if (!objPreload && !this.loadedScenes.has(id)) {
        deferredObjects.push(obj);
      }
    }

    if (deferredObjects.length === 0) {
      return;
    }

    this.logger.log(
      `Prefetching ${deferredObjects.length} deferred scene objects`
    );

    // Separate into GLTF (smaller, needed first) and splat (larger) objects
    const gltfObjects = deferredObjects.filter((obj) => obj.type === "gltf");
    const splatObjects = deferredObjects.filter((obj) => obj.type === "splat");

    // Prefetch GLTF files immediately (smaller, needed in first scene)
    gltfObjects.forEach((obj) => {
      // Prefetch GLTF files - browser will cache them
      fetch(obj.path, { method: "HEAD" }).catch(() => {
        // Ignore errors - file might not exist or CORS might block HEAD
      });
      // Also try GET to actually cache it
      fetch(obj.path).catch(() => {
        // Ignore errors
      });
    });

    // Prefetch splat files after a delay (larger, not needed immediately)
    setTimeout(() => {
      splatObjects.forEach((obj) => {
        // Prefetch splat files - browser will cache them
        fetch(obj.path).catch(() => {
          // Ignore errors
        });
      });
    }, 2000); // Wait 2 seconds before prefetching splats

    this.logger.log(
      `Prefetched ${deferredObjects.length} deferred scene objects (files cached, will load when criteria match)`
    );
  }

  /**
   * Check if character controller is enabled
   * @returns {boolean}
   */
  isControlEnabled() {
    return this.state.controlEnabled === true;
  }

  /**
   * Pause the game
   */
  pause() {
    this.setState({ isPaused: true });
    this.emit("game:paused");
  }

  /**
   * Resume the game
   */
  resume() {
    this.setState({ isPaused: false });
    this.emit("game:resumed");
  }

  /**
   * Start the game
   */
  start() {
    this.setState({ isPlaying: true, isPaused: false });
    this.emit("game:started");
  }

  /**
   * Stop the game
   */
  stop() {
    this.setState({ isPlaying: false, isPaused: false });
    this.emit("game:stopped");
  }

  /**
   * Add event listener
   * @param {string} event - Event name
   * @param {function} callback - Callback function
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
   * Update method - call in animation loop if needed
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    // Update content-specific systems
    if (this.phoneBooth) {
      this.phoneBooth.update(dt);
    }

    if (this.candlestickPhone) {
      this.candlestickPhone.update(dt);
    }

    if (this.amplifierCord) {
      this.amplifierCord.update(dt);
    }

    // DISABLED - using DrawingGame instead
    // if (this.drawingManager) {
    //   this.drawingManager.update(dt);
    // }
  }
}

export default GameManager;
