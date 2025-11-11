import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { Logger } from "./logger.js";

/**
 * GizmoManager - Debug tool for positioning assets in 3D space
 *
 * Features:
 * - Auto-creates gizmo for each object with gizmo: true
 * - Multiple simultaneous gizmos supported (multi-gizmo mode)
 * - Drag gizmo arrows/rings to move/rotate/scale any object
 * - Switch between translate/rotate/scale modes (affects all gizmos)
 * - Log position/rotation/scale on release
 * - Works with meshes, splats, video planes, and colliders
 * - Spawn gizmos dynamically with P key (when ?gizmo URL param present)
 *
 * Usage:
 * - Set gizmo: true in object data (videoData.js, sceneData.js, colliderData.js, etc.)
 * - Gizmo appears automatically for each enabled object
 * - Add ?gizmo to URL to enable P key spawning (blocks pointer lock & idle)
 * - Click object to focus it for logging
 * - P = spawn gizmo 5m in front of camera (only with ?gizmo param)
 * - G = translate, R = rotate, X = scale (all gizmos)
 * - Space = toggle world/local space (all gizmos)
 * - H = toggle visibility (all gizmos)
 * - U = cycle through gizmos and teleport character 5m in front of each
 * - Q/E = fly down/up (flight mode enabled automatically with gizmo)
 * - Drag any gizmo to manipulate, release to log position
 */
class GizmoManager {
  constructor(
    scene,
    camera,
    renderer,
    sceneManager = null,
    characterController = null
  ) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.characterController = characterController;
    this.enabled = false;

    // Check if logging should be enabled based on gizmo presence
    const shouldEnableLogging = this.checkIfLoggingShouldBeEnabled();
    this.logger = new Logger("GizmoManager", shouldEnableLogging);

    // Configurable objects - now supports multiple simultaneous gizmos
    this.objects = []; // Objects that can be selected
    this.controls = new Map(); // Map of object -> TransformControls
    this.controlHelpers = new Map(); // Map of object -> helper Object3D
    this.isGizmoDragging = false;
    this.isGizmoHovering = false;
    this.isVisible = true;
    this.hasGizmoInDefinitions = false; // from data definitions (even if not instantiated)
    this.hasGizmoURLParam = false; // if gizmo URL param is present
    this.currentMode = "translate"; // Current gizmo mode (applies to all)
    this.currentSpace = "world"; // Current space (applies to all)
    // Integration targets for standardized global effects
    this.idleHelper = null;
    this.inputManager = null;
    this.activeObject = null; // Most recently interacted object
    this.spawnedGizmoCounter = 0; // Counter for spawned gizmos
    this.currentGizmoIndex = 0; // For cycling through gizmos with F key

    // Gamepad button state tracking for speed adjustment
    this.lastLBState = false;
    this.lastRBState = false;
    this.speedAdjustCooldown = 0;
    this.speedAdjustCooldownDuration = 0.2; // 200ms cooldown between adjustments

    // Gamepad D-pad state tracking for vertical flight movement
    this.lastDPadUpState = false;
    this.lastDPadDownState = false;

    // Gamepad trigger state tracking for rotation speed adjustment
    this.lastLTState = false;
    this.lastRTState = false;
    this.rotationAdjustCooldown = 0;
    this.rotationAdjustCooldownDuration = 0.2; // 200ms cooldown between adjustments

    // Gamepad X/B button state tracking for camera roll adjustment
    this.lastXState = false;
    this.lastBState = false;
    this.lastRollLogTime = 0; // For logging roll adjustments

    // Raycaster for object picking
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Check for gizmo URL parameter
    this.checkGizmoURLParam();

    // Always enable (will only affect objects with gizmo: true)
    this.enable();

    // Note: Scene objects will be registered later by main.js after they're loaded
    // Don't call registerSceneObjects here - sceneManager is empty at construction time
  }

  /**
   * Check if logging should be enabled based on gizmo presence
   * @returns {boolean}
   */
  checkIfLoggingShouldBeEnabled() {
    // Check URL params for gizmo parameter
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const hasGizmoURLParam = urlParams.has("gizmo");
      if (hasGizmoURLParam) {
        return true;
      }
    } catch (e) {
      // Ignore errors in URL param checking
    }

    // Check if any gizmo data exists (will be checked later when data is available)
    // For now, assume logging should be enabled if we're in a development context
    // The actual check will happen in applyGlobalBlocksFromDefinitions
    return false; // Will be updated when data is processed
  }

  /**
   * Update logging state based on gizmo data availability
   */
  updateLoggingBasedOnGizmoData() {
    const hasGizmoRuntime = this.objects && this.objects.length > 0;
    const shouldEnableLogging =
      this.hasGizmoInDefinitions || this.hasGizmoURLParam || hasGizmoRuntime;
    if (this.logger.debug !== shouldEnableLogging) {
      this.logger.setDebug(shouldEnableLogging);
      this.logger.log(
        `Logging ${
          shouldEnableLogging ? "enabled" : "disabled"
        } based on gizmo data availability`
      );
    }
  }

  /**
   * Check if gizmo URL parameter is present
   */
  checkGizmoURLParam() {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      this.hasGizmoURLParam = urlParams.has("gizmo");
      if (this.hasGizmoURLParam) {
        this.updateLoggingBasedOnGizmoData();
      }
    } catch (e) {
      this.logger.warn("Failed to check URL params:", e);
      this.hasGizmoURLParam = false;
    }
  }

  /**
   * Wire integrations so gizmo presence can standardize global effects
   * @param {Object} idleHelper
   * @param {Object} inputManager
   */
  setIntegration(idleHelper, inputManager) {
    this.idleHelper = idleHelper || null;
    this.inputManager = inputManager || null;
    this.updateGlobalBlocks();
  }

  /**
   * Apply or clear global blocks based on whether any gizmo objects are registered
   */
  updateGlobalBlocks() {
    const hasGizmoRuntime = this.objects && this.objects.length > 0;
    const hasGizmo =
      this.hasGizmoInDefinitions || hasGizmoRuntime || this.hasGizmoURLParam;
    if (
      this.idleHelper &&
      typeof this.idleHelper.setGlobalDisable === "function"
    ) {
      this.idleHelper.setGlobalDisable(hasGizmo);
    }
    if (
      this.inputManager &&
      typeof this.inputManager.setPointerLockBlocked === "function"
    ) {
      // Block pointer lock when gizmos are present (gizmo mode)
      // Only ENABLE blocking when gizmos are present, don't DISABLE it
      // (other systems like DrawingManager might need it blocked)
      if (hasGizmo) {
        this.inputManager.setPointerLockBlocked(true);
      }
    }

    // Enable/disable flight mode based on gizmo presence
    if (this.characterController) {
      if (
        hasGizmo &&
        typeof this.characterController.enableFlightMode === "function"
      ) {
        this.characterController.enableFlightMode();
      } else if (
        !hasGizmo &&
        typeof this.characterController.disableFlightMode === "function"
      ) {
        this.characterController.disableFlightMode();
      }

      // Disable headbob (including idle headbob) in gizmo mode
      if (hasGizmo) {
        this.characterController.headbobEnabled = false;
      } else {
        // Restore headbob when exiting gizmo mode (only if not explicitly disabled elsewhere)
        // Check if gameManager has control enabled to avoid overriding other disables
        if (
          window.gameManager &&
          typeof window.gameManager.isControlEnabled === "function" &&
          window.gameManager.isControlEnabled()
        ) {
          this.characterController.headbobEnabled = true;
        }
      }
    }
  }

  /**
   * Evaluate gizmo flags directly from data definitions (scene/video/etc.)
   * and standardize global side-effects regardless of instantiation status
   * @param {Object} options
   * @param {Object|Array} options.sceneDefs - map or array of scene defs
   * @param {Object|Array} options.videoDefs - map or array of video defs
   * @param {Object|Array} options.colliderDefs - map or array of collider defs
   * @param {Object|Array} options.lightDefs - map or array of light defs
   */
  applyGlobalBlocksFromDefinitions({
    sceneDefs = null,
    videoDefs = null,
    colliderDefs = null,
    lightDefs = null,
  } = {}) {
    const collect = (defs) => {
      if (!defs) return [];
      if (Array.isArray(defs)) return defs;
      if (typeof defs === "object") return Object.values(defs);
      return [];
    };
    const all = [
      ...collect(sceneDefs),
      ...collect(videoDefs),
      ...collect(colliderDefs),
      ...collect(lightDefs),
    ];
    this.hasGizmoInDefinitions = all.some((d) => d && d.gizmo === true);

    // Update logging based on gizmo data availability
    this.updateLoggingBasedOnGizmoData();

    // Inform gameManager (if available via window) so all managers share the same flag
    try {
      if (
        window?.gameManager &&
        typeof window.gameManager.setState === "function"
      ) {
        window.gameManager.setState({
          hasGizmoInData: this.hasGizmoInDefinitions,
        });
      }
    } catch {}
    this.updateGlobalBlocks();
  }

  /**
   * Register all gizmo-enabled scene objects from a SceneManager
   * @param {Object} sceneManager - Instance of SceneManager
   */
  registerSceneObjects(sceneManager) {
    if (!sceneManager || !sceneManager.objects || !sceneManager.objectData) {
      return;
    }
    try {
      sceneManager.objects.forEach((obj, id) => {
        const data = sceneManager.objectData.get(id);
        if (data && data.gizmo) {
          this.registerObject(obj, id, data.type || "scene");
        }
      });
    } catch (e) {
      this.logger.warn("registerSceneObjects failed:", e);
    }
  }

  /**
   * Register all gizmo-enabled lights from a LightManager
   * @param {Object} lightManager - Instance of LightManager
   * @param {Object} lightData - Light data definitions
   */
  registerLights(lightManager, lightData) {
    if (!lightManager || !lightData) {
      return;
    }
    try {
      // Iterate through light data definitions
      for (const [key, config] of Object.entries(lightData)) {
        if (config.gizmo) {
          // Get the light object from lightManager
          const lightObj = lightManager.getLight(config.id);
          if (lightObj) {
            this.registerObject(lightObj, config.id, "light");
          }
        }
      }
    } catch (e) {
      this.logger.warn("registerLights failed:", e);
    }
  }

  /**
   * Check whether any scene object is gizmo-enabled
   * @param {Object} sceneManager - Instance of SceneManager
   * @returns {boolean}
   */
  hasAnyGizmoObjects(sceneManager) {
    try {
      const values = Array.from(sceneManager?.objectData?.values() || []);
      return values.some((d) => d && d.gizmo === true);
    } catch {
      return false;
    }
  }

  /**
   * If any gizmo-enabled object exists, disable idle behaviors globally
   * @param {Object} idleHelper - Instance of IdleHelper
   * @param {Object} sceneManager - Instance of SceneManager
   */
  applyIdleBlockIfNeeded(idleHelper, sceneManager) {
    if (!idleHelper) return;
    if (this.hasAnyGizmoObjects(sceneManager)) {
      if (typeof idleHelper.setGlobalDisable === "function") {
        idleHelper.setGlobalDisable(true);
        this.logger.log(
          "IdleHelper: Globally disabled due to gizmo-enabled object(s)"
        );
      }
    }
  }

  /**
   * If any gizmo-enabled object exists, block pointer lock for easier gizmo manipulation
   * @param {Object} inputManager - Instance of InputManager
   * @param {Object} sceneManager - Instance of SceneManager
   */
  applyPointerLockBlockIfNeeded(inputManager, sceneManager) {
    if (
      !inputManager ||
      typeof inputManager.setPointerLockBlocked !== "function"
    )
      return;
    const shouldBlock = this.hasAnyGizmoObjects(sceneManager);
    inputManager.setPointerLockBlocked(shouldBlock);
    if (shouldBlock) {
      this.logger.log(
        "InputManager: Pointer lock blocked due to gizmo-enabled object(s)"
      );
    }
  }

  /**
   * Enable the gizmo system
   */
  enable() {
    if (this.enabled) return;
    this.enabled = true;

    // Setup event listeners (controls created per-object in registerObject)
    this.setupEventListeners();

    this.logger.log("Enabled (multi-gizmo mode)");
  }

  /**
   * Disable the gizmo system
   */
  disable() {
    if (!this.enabled) return;
    this.enabled = false;

    // Dispose all controls
    for (const control of this.controls.values()) {
      if (control) {
        control.dispose();
      }
    }
    this.controls.clear();

    // Remove all helpers
    for (const helper of this.controlHelpers.values()) {
      if (helper && helper.parent) {
        helper.parent.remove(helper);
      }
    }
    this.controlHelpers.clear();

    this.removeEventListeners();
  }

  /**
   * Register an object that can be selected and manipulated
   * Creates and shows a gizmo immediately (multi-gizmo mode)
   * @param {THREE.Object3D} object - The object to register
   * @param {string} id - Optional identifier for logging
   * @param {string} type - Optional type (mesh, splat, video)
   */
  registerObject(object, id = null, type = "object") {
    if (!object) {
      this.logger.warn("Attempted to register null object");
      return;
    }

    const item = {
      object,
      id: id || object.name || "unnamed",
      type,
    };

    this.objects.push(item);

    // Create a TransformControls for this object
    const control = new TransformControls(
      this.camera,
      this.renderer.domElement
    );
    control.setMode(this.currentMode);
    control.setSpace(this.currentSpace);
    control.attach(object);

    // Add the visual helper Object3D
    if (typeof control.getHelper === "function") {
      const helper = control.getHelper();
      if (helper) {
        this.scene.add(helper);
        helper.visible = this.isVisible;
        this.controlHelpers.set(object, helper);
      }
    }

    // Setup event listeners for this control
    control.addEventListener("dragging-changed", (event) => {
      if (event.value) {
        this.isGizmoDragging = true;
        this.activeObject = item;
        this.logger.log(`Dragging "${item.id}"`);
      } else {
        this.isGizmoDragging = false;
        this.logger.log(`Drag ended "${item.id}"`);
        this.logObjectTransform(item);
      }
    });

    control.addEventListener("hoveron", () => {
      this.isGizmoHovering = true;
    });

    control.addEventListener("hoveroff", () => {
      this.isGizmoHovering = false;
    });

    this.controls.set(object, control);

    this.logger.log(`Registered "${item.id}" (${type}) with gizmo`, object);
    this.logger.log(`  Total registered objects: ${this.objects.length}`);

    // Update logging based on runtime gizmo presence
    this.updateLoggingBasedOnGizmoData();

    // Standardize side-effects when any gizmo is present
    this.updateGlobalBlocks();
  }

  /**
   * Select an already-registered object by id (sets it as active for logging)
   * @param {string} id
   */
  selectObjectById(id) {
    if (!id) return;
    const item = this.objects.find((it) => it.id === id);
    if (item) {
      this.activeObject = item;
      this.logger.log(`Set active object to "${id}"`);
    }
  }

  /**
   * Unregister an object and remove its gizmo
   * @param {THREE.Object3D} object - The object to unregister
   */
  unregisterObject(object) {
    const index = this.objects.findIndex((item) => item.object === object);
    if (index !== -1) {
      this.objects.splice(index, 1);
    }

    // Dispose the control
    const control = this.controls.get(object);
    if (control) {
      control.dispose();
      this.controls.delete(object);
    }

    // Remove the helper
    const helper = this.controlHelpers.get(object);
    if (helper && helper.parent) {
      helper.parent.remove(helper);
    }
    this.controlHelpers.delete(object);

    // Update logging based on runtime gizmo presence
    this.updateLoggingBasedOnGizmoData();

    // Update global effects when gizmo set changes
    this.updateGlobalBlocks();
  }

  /**
   * Setup event listeners for gizmo interaction
   */
  setupEventListeners() {
    // Mouse click for object selection/focusing
    this.onMouseDown = this.handleMouseDown.bind(this);
    this.renderer.domElement.addEventListener("mousedown", this.onMouseDown);

    // Mouse up for logging position
    this.onMouseUp = this.handleMouseUp.bind(this);
    this.renderer.domElement.addEventListener("mouseup", this.onMouseUp);

    // Keyboard shortcuts
    this.onKeyDown = this.handleKeyDown.bind(this);
    window.addEventListener("keydown", this.onKeyDown);

    // Note: drag/hover events are set per-control in registerObject()
  }

  /**
   * Returns true if pointer is interacting with the gizmo (hovering or dragging)
   */
  isPointerOverGizmo() {
    return this.isGizmoHovering || this.isGizmoDragging;
  }

  /**
   * Remove event listeners
   */
  removeEventListeners() {
    if (this.onMouseDown) {
      this.renderer.domElement.removeEventListener(
        "mousedown",
        this.onMouseDown
      );
    }
    if (this.onMouseUp) {
      this.renderer.domElement.removeEventListener("mouseup", this.onMouseUp);
    }
    if (this.onKeyDown) {
      window.addEventListener("keydown", this.onKeyDown);
    }
  }

  /**
   * Handle mouse down for object focusing (sets active for keyboard shortcuts)
   */
  handleMouseDown(event) {
    // Ignore if any gizmo is being dragged
    const anyDragging = Array.from(this.controls.values()).some(
      (ctrl) => ctrl.dragging
    );
    if (anyDragging) return;

    // Calculate mouse position in normalized device coordinates
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Raycast to find intersected objects
    this.raycaster.setFromCamera(this.mouse, this.camera);

    // Get all selectable objects
    const selectableObjects = this.objects.map((item) => item.object);
    const intersects = this.raycaster.intersectObjects(selectableObjects, true);

    if (intersects.length > 0) {
      // Find the top-level registered object
      let focusedObj = null;
      for (const item of this.objects) {
        if (
          intersects[0].object === item.object ||
          intersects[0].object.parent === item.object ||
          item.object.children.includes(intersects[0].object)
        ) {
          focusedObj = item;
          break;
        }
      }

      if (focusedObj) {
        this.activeObject = focusedObj;
        this.logger.log(`Focused "${focusedObj.id}" (${focusedObj.type})`);
      }
    }
  }

  /**
   * Handle mouse up
   */
  handleMouseUp(event) {
    // Position logging is handled in per-control dragging-changed events
  }

  /**
   * Handle keyboard shortcuts (applies to all gizmos)
   */
  handleKeyDown(event) {
    if (!this.enabled) {
      this.logger.log("handleKeyDown called but not enabled");
      return;
    }

    // Ignore if typing in an input
    if (
      document.activeElement.tagName === "INPUT" ||
      document.activeElement.tagName === "TEXTAREA"
    ) {
      return;
    }

    switch (event.key) {
      case "p":
      case "P":
        this.logger.log(
          "P key pressed, hasGizmoURLParam:",
          this.hasGizmoURLParam
        );
        // Spawn gizmo if URL param is present
        if (this.hasGizmoURLParam) {
          this.spawnGizmoInFrontOfCamera();
        } else {
          this.logger.log("P key pressed but gizmo URL param not present");
        }
        break;
      case "g":
      case "G":
        if (this.controls.size === 0) return;
        this.currentMode = "translate";
        for (const control of this.controls.values()) {
          control.setMode("translate");
        }
        this.logger.log("Mode = Translate (all gizmos)");
        break;
      case "r":
      case "R":
        if (this.controls.size === 0) return;
        this.currentMode = "rotate";
        for (const control of this.controls.values()) {
          control.setMode("rotate");
        }
        this.logger.log("Mode = Rotate (all gizmos)");
        break;
      case "x":
      case "X":
        if (this.controls.size === 0) return;
        this.currentMode = "scale";
        for (const control of this.controls.values()) {
          control.setMode("scale");
        }
        this.logger.log("Mode = Scale (all gizmos)");
        break;
      case " ":
        if (this.controls.size === 0) return;
        this.currentSpace = this.currentSpace === "world" ? "local" : "world";
        for (const control of this.controls.values()) {
          control.setSpace(this.currentSpace);
        }
        this.logger.log(
          `Space = ${
            this.currentSpace === "world" ? "World" : "Local"
          } (all gizmos)`
        );
        break;
      case "h":
      case "H":
        if (this.controls.size === 0) return;
        this.setVisible(!this.isVisible);
        this.logger.log(`${this.isVisible ? "Shown" : "Hidden"} (all gizmos)`);
        break;
      case "u":
      case "U":
        this.cycleAndTeleportToNextGizmo();
        break;
      case "Escape":
        if (this.activeObject) {
          this.logger.log(`Cleared focus from "${this.activeObject.id}"`);
          this.activeObject = null;
        }
        break;
    }
  }

  /**
   * Spawn a gizmo 5 meters in front of the camera
   */
  spawnGizmoInFrontOfCamera() {
    if (!this.camera) {
      this.logger.warn("Cannot spawn gizmo - no camera reference");
      return;
    }

    // Create a small sphere mesh to attach the gizmo to
    const geometry = new THREE.SphereGeometry(0.1, 16, 16);
    const material = new THREE.MeshBasicMaterial({
      color: 0xff00ff,
      wireframe: true,
    });
    const sphere = new THREE.Mesh(geometry, material);

    // Calculate position 5 meters in front of camera
    const cameraDirection = new THREE.Vector3();
    this.camera.getWorldDirection(cameraDirection);

    const spawnPosition = new THREE.Vector3();
    spawnPosition.copy(this.camera.position);
    spawnPosition.addScaledVector(cameraDirection, 5);

    sphere.position.copy(spawnPosition);
    this.scene.add(sphere);

    // Generate unique ID for this spawned gizmo
    this.spawnedGizmoCounter++;
    const gizmoId = `spawned-gizmo-${this.spawnedGizmoCounter}`;

    // Register the sphere with the gizmo manager
    this.registerObject(sphere, gizmoId, "spawned");

    // Log spawn position in clean text format
    const spawnText = `position: {x: ${spawnPosition.x.toFixed(
      2
    )}, y: ${spawnPosition.y.toFixed(2)}, z: ${spawnPosition.z.toFixed(2)}}`;
    this.logger.logRaw(`Spawned gizmo "${gizmoId}" at ${spawnText}`);
  }

  /**
   * Cycle to next gizmo and teleport character to it
   */
  cycleAndTeleportToNextGizmo() {
    if (this.objects.length === 0) {
      this.logger.log("No gizmo objects available");
      return;
    }

    if (this.objects.length === 1) {
      // Only one gizmo, just teleport to it
      this.activeObject = this.objects[0];
      this.teleportCharacterToObject(this.objects[0]);
      return;
    }

    // Multiple gizmos - cycle through them
    this.currentGizmoIndex = (this.currentGizmoIndex + 1) % this.objects.length;
    const nextGizmo = this.objects[this.currentGizmoIndex];
    this.activeObject = nextGizmo;

    this.logger.log(
      `Cycling to gizmo ${this.currentGizmoIndex + 1}/${
        this.objects.length
      } ("${nextGizmo.id}")`
    );

    this.teleportCharacterToObject(nextGizmo);
  }

  /**
   * Teleport character to 5 meters in front of an object
   * @param {Object} item - Gizmo item with object reference
   */
  teleportCharacterToObject(item) {
    if (!item || !item.object) {
      this.logger.warn("Cannot teleport - no object specified");
      return;
    }

    // Get character controller and physics manager from window globals
    const physicsManager = this.characterController?.physicsManager;

    if (!this.characterController) {
      this.logger.warn("Cannot teleport - character controller not found");
      return;
    }

    const obj = item.object;

    // Calculate the object's forward direction (-Z in local space)
    const forward = new THREE.Vector3(0, 0, -1);
    forward.applyQuaternion(obj.quaternion);
    forward.normalize();

    // Calculate position 5 meters in front of object
    const teleportPosition = new THREE.Vector3();
    teleportPosition.copy(obj.position);
    teleportPosition.addScaledVector(forward, 5);

    // Ensure Y position is at least 0.9 (character controller minimum height)
    teleportPosition.y = Math.max(0.9, teleportPosition.y);

    // Teleport character controller
    if (this.characterController.character) {
      this.characterController.character.setTranslation(
        {
          x: teleportPosition.x,
          y: teleportPosition.y,
          z: teleportPosition.z,
        },
        true
      );

      this.logger.log(`Teleported character to 5m in front of "${item.id}"`);
      this.logger.logRaw(
        `  Object position: {x: ${obj.position.x.toFixed(
          2
        )}, y: ${obj.position.y.toFixed(2)}, z: ${obj.position.z.toFixed(2)}}`
      );
      this.logger.logRaw(
        `  Object forward: {x: ${forward.x.toFixed(3)}, y: ${forward.y.toFixed(
          3
        )}, z: ${forward.z.toFixed(3)}}`
      );
      this.logger.logRaw(
        `  Teleport position: {x: ${teleportPosition.x.toFixed(
          2
        )}, y: ${teleportPosition.y.toFixed(
          2
        )}, z: ${teleportPosition.z.toFixed(2)}}`
      );
    } else {
      this.logger.warn("Cannot teleport - character physics body not found");
    }
  }

  /**
   * Show/hide all gizmo visuals and interaction
   */
  setVisible(visible) {
    this.isVisible = !!visible;

    // Update all helpers
    for (const helper of this.controlHelpers.values()) {
      if (helper) {
        helper.visible = this.isVisible;
      }
    }

    // Update all controls
    for (const control of this.controls.values()) {
      if (control) {
        control.enabled = this.isVisible;
      }
    }
  }

  /**
   * Select an object (sets it as active for logging/interaction)
   * In multi-gizmo mode, all gizmos are always visible
   */
  selectObject(item) {
    this.activeObject = item;
    this.logger.log(`Selected "${item.id}" (${item.type})`);
    this.logObjectTransform(item);
  }

  /**
   * Deselect current object (clears active focus)
   */
  deselectObject() {
    if (this.activeObject) {
      this.logger.log(`Deselected "${this.activeObject.id}"`);
      this.activeObject = null;
    }
  }

  /**
   * Log an object's transform as formatted text
   * @param {Object} item - Optional item to log (defaults to activeObject)
   */
  logObjectTransform(item = null) {
    const target = item || this.activeObject;
    if (!target) return;

    const obj = target.object;
    const pos = obj.position;
    const rot = obj.rotation;
    const scale = obj.scale;

    // Create formatted text with line breaks (single log statement)
    const transformText =
      `position: {x: ${pos.x.toFixed(2)}, y: ${pos.y.toFixed(
        2
      )}, z: ${pos.z.toFixed(2)}},\n` +
      `rotation: {x: ${rot.x.toFixed(4)}, y: ${rot.y.toFixed(
        4
      )}, z: ${rot.z.toFixed(4)}},\n` +
      `scale: {x: ${scale.x.toFixed(2)}, y: ${scale.y.toFixed(
        2
      )}, z: ${scale.z.toFixed(2)}}`;

    // Log the formatted text (without prefix for clean output)
    this.logger.logRaw(transformText);
  }

  /**
   * Update method (call in animation loop if needed)
   */
  update(dt) {
    // TransformControls handles its own updates

    // Handle gamepad speed adjustment in gizmo mode
    this.updateGamepadSpeedAdjustment(dt);

    // Handle gamepad rotation speed adjustment in gizmo mode
    this.updateGamepadRotationAdjustment(dt);

    // Handle gamepad camera roll adjustment in gizmo mode
    this.updateGamepadRollAdjustment(dt);

    // Handle gamepad D-pad vertical flight controls in gizmo mode
    this.updateGamepadFlightControls(dt);
  }

  /**
   * Update gamepad speed adjustment (LB/RB buttons)
   * @param {number} dt - Delta time
   */
  updateGamepadSpeedAdjustment(dt) {
    // Only work in gizmo mode
    const hasGizmoRuntime = this.objects && this.objects.length > 0;
    const hasGizmo =
      this.hasGizmoInDefinitions || hasGizmoRuntime || this.hasGizmoURLParam;
    if (!hasGizmo) return;

    // Need character controller and input manager
    if (!this.characterController || !this.inputManager) return;

    // Update cooldown timer
    if (this.speedAdjustCooldown > 0) {
      this.speedAdjustCooldown -= dt;
    }

    // Get gamepad
    const gamepad = this.inputManager.getGamepad();
    if (!gamepad) return;

    // Check shoulder buttons using InputManager's gamepad mapping
    const buttonLB = this.inputManager.gamepadMapping.BUTTON_LB; // 4
    const buttonRB = this.inputManager.gamepadMapping.BUTTON_RB; // 5
    const lbPressed = gamepad.buttons[buttonLB]?.pressed || false;
    const rbPressed = gamepad.buttons[buttonRB]?.pressed || false;

    // Detect button press (edge trigger, not hold)
    if (this.speedAdjustCooldown <= 0) {
      if (rbPressed && !this.lastRBState) {
        // RB pressed - increase speed
        this.characterController.increaseSpeed();
        this.speedAdjustCooldown = this.speedAdjustCooldownDuration;
        this.logger.log("Speed increased (RB)");
      } else if (lbPressed && !this.lastLBState) {
        // LB pressed - decrease speed
        this.characterController.decreaseSpeed();
        this.speedAdjustCooldown = this.speedAdjustCooldownDuration;
        this.logger.log("Speed decreased (LB)");
      }
    }

    // Store previous state for edge detection
    this.lastLBState = lbPressed;
    this.lastRBState = rbPressed;
  }

  /**
   * Update gamepad rotation speed adjustment (LT/RT triggers)
   * @param {number} dt - Delta time
   */
  updateGamepadRotationAdjustment(dt) {
    // Only work in gizmo mode
    const hasGizmoRuntime = this.objects && this.objects.length > 0;
    const hasGizmo =
      this.hasGizmoInDefinitions || hasGizmoRuntime || this.hasGizmoURLParam;
    if (!hasGizmo) return;

    // Need input manager
    if (!this.inputManager) return;

    // Update cooldown timer
    if (this.rotationAdjustCooldown > 0) {
      this.rotationAdjustCooldown -= dt;
    }

    // Get gamepad
    const gamepad = this.inputManager.getGamepad();
    if (!gamepad) return;

    // Check trigger buttons using InputManager's gamepad mapping
    const buttonLT = this.inputManager.gamepadMapping.BUTTON_LT; // 6
    const buttonRT = this.inputManager.gamepadMapping.BUTTON_RT; // 7
    const ltButton = gamepad.buttons[buttonLT];
    const rtButton = gamepad.buttons[buttonRT];

    // Triggers are analog (value 0-1), but we treat them as digital for adjustment
    // Check if trigger is pressed past threshold
    const triggerThreshold = this.inputManager.triggerThreshold || 0.5;
    const ltPressed = ltButton && ltButton.value > triggerThreshold;
    const rtPressed = rtButton && rtButton.value > triggerThreshold;

    // Detect trigger press (edge trigger, not hold)
    if (this.rotationAdjustCooldown <= 0) {
      if (rtPressed && !this.lastRTState) {
        // RT pressed - increase rotation speed
        this.inputManager.increaseRotationSensitivity();
        this.rotationAdjustCooldown = this.rotationAdjustCooldownDuration;
        this.logger.log("Rotation speed increased (RT)");
      } else if (ltPressed && !this.lastLTState) {
        // LT pressed - decrease rotation speed
        this.inputManager.decreaseRotationSensitivity();
        this.rotationAdjustCooldown = this.rotationAdjustCooldownDuration;
        this.logger.log("Rotation speed decreased (LT)");
      }
    }

    // Store previous state for edge detection
    this.lastLTState = ltPressed;
    this.lastRTState = rtPressed;
  }

  /**
   * Update gamepad camera roll adjustment (X/B buttons)
   * @param {number} dt - Delta time
   */
  updateGamepadRollAdjustment(dt) {
    // Only work in gizmo mode
    const hasGizmoRuntime = this.objects && this.objects.length > 0;
    const hasGizmo =
      this.hasGizmoInDefinitions || hasGizmoRuntime || this.hasGizmoURLParam;
    if (!hasGizmo) return;

    // Need character controller and input manager
    if (!this.characterController || !this.inputManager) return;

    // Get gamepad
    const gamepad = this.inputManager.getGamepad();
    if (!gamepad) return;

    // Check X and B buttons using InputManager's gamepad mapping
    const buttonX = this.inputManager.gamepadMapping.BUTTON_X; // 2
    const buttonB = this.inputManager.gamepadMapping.BUTTON_B; // 1
    const xPressed = gamepad.buttons[buttonX]?.pressed || false;
    const bPressed = gamepad.buttons[buttonB]?.pressed || false;

    // Adjust roll based on button state (continuous adjustment while held)
    // X = roll left (negative), B = roll right (positive)
    // Use slow speed for fine control
    const rollSpeed = this.characterController.gizmoRollSpeed || 0.1;
    let rollDelta = 0;

    if (xPressed && !bPressed) {
      // X pressed - roll left (negative)
      rollDelta = -rollSpeed * dt;
    } else if (bPressed && !xPressed) {
      // B pressed - roll right (positive)
      rollDelta = rollSpeed * dt;
    }

    if (rollDelta !== 0) {
      this.characterController.adjustRoll(rollDelta);
      // Log occasionally to verify it's working (every 0.5 seconds)
      if (!this.lastRollLogTime || Date.now() - this.lastRollLogTime > 500) {
        const rollDeg = THREE.MathUtils.radToDeg(this.characterController.currentRoll);
        this.logger.log(`Camera roll: ${rollDeg.toFixed(2)}° (X=${xPressed}, B=${bPressed})`);
        this.lastRollLogTime = Date.now();
      }
    }

    // Store previous state (for potential future edge detection if needed)
    this.lastXState = xPressed;
    this.lastBState = bPressed;
  }

  /**
   * Update gamepad D-pad vertical flight controls (D-pad up/down)
   * @param {number} dt - Delta time
   */
  updateGamepadFlightControls(dt) {
    // Only work in gizmo mode (flight mode is enabled when gizmos are present)
    const hasGizmoRuntime = this.objects && this.objects.length > 0;
    const hasGizmo =
      this.hasGizmoInDefinitions || hasGizmoRuntime || this.hasGizmoURLParam;
    if (!hasGizmo) return;

    // Need character controller and input manager
    if (!this.characterController || !this.inputManager) return;

    // Only work if flight mode is enabled
    if (!this.characterController.flightMode) return;

    // Get gamepad
    const gamepad = this.inputManager.getGamepad();
    if (!gamepad) return;

    // Check D-pad buttons using InputManager's gamepad mapping
    const buttonDPadUp = this.inputManager.gamepadMapping.BUTTON_DPAD_UP; // 12
    const buttonDPadDown = this.inputManager.gamepadMapping.BUTTON_DPAD_DOWN; // 13
    const dpadUpPressed = gamepad.buttons[buttonDPadUp]?.pressed || false;
    const dpadDownPressed = gamepad.buttons[buttonDPadDown]?.pressed || false;

    // Update vertical input based on D-pad state
    // D-pad up = fly up (positive), D-pad down = fly down (negative)
    if (dpadUpPressed && !dpadDownPressed) {
      this.characterController.verticalInput = 1; // Up
    } else if (dpadDownPressed && !dpadUpPressed) {
      this.characterController.verticalInput = -1; // Down
    } else if (!dpadUpPressed && !dpadDownPressed) {
      // Both released - check if we just released (edge detection)
      // Only reset to 0 if we were previously holding one of them
      if (this.lastDPadUpState || this.lastDPadDownState) {
        this.characterController.verticalInput = 0;
      }
    }
    // If both are pressed, don't change (keep current state)

    // Store previous state for edge detection
    this.lastDPadUpState = dpadUpPressed;
    this.lastDPadDownState = dpadDownPressed;
  }

  /**
   * Clean up
   */
  destroy() {
    this.disable();
    this.objects = [];
  }
}

export default GizmoManager;
