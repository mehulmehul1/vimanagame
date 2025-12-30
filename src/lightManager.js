/**
 * LightManager.js - SCENE LIGHTING AND AUDIO-REACTIVE EFFECTS
 * =============================================================================
 *
 * ROLE: Manages all lighting in the scene including Three.js lights, splat-based
 * lights via SplatEdit, audio-reactive lights, and lens flares.
 *
 * KEY RESPONSIBILITIES:
 * - Load lights from lightData.js definitions
 * - Create Three.js lights (point, spot, directional)
 * - Create splat-based lights via SplatEdit layers
 * - Manage audio-reactive lights that pulse to sound
 * - Add lens flare effects to lights
 * - Criteria-based light visibility
 * - Parent lights to scene objects
 *
 * LIGHT TYPES:
 * - point: Omnidirectional point light
 * - spot: Directed spotlight with cone
 * - directional: Parallel rays (sun-like)
 * - splat: SplatEdit color/brightness modification layer
 *
 * AUDIO-REACTIVE LIGHTS:
 * Lights can pulse intensity/color based on audio playback.
 * Configured via AudioReactiveLight module.
 *
 * SPLAT LIGHTS:
 * Use SplatEdit to modify splat appearance with spherical SDFs.
 * Affects splat rendering without adding traditional 3D lights.
 *
 * =============================================================================
 */

import * as THREE from "three";
import {
  SplatEdit,
  SplatEditSdf,
  SplatEditSdfType,
  SplatEditRgbaBlendMode,
} from "@sparkjsdev/spark";
import AudioReactiveLight from "./vfx/audioReactiveLight.js";
import {
  Lensflare,
  LensflareElement,
} from "three/examples/jsm/objects/Lensflare.js";
import { lights } from "./lightData.js";
import { checkCriteria } from "./utils/criteriaHelper.js";
import { Logger } from "./utils/logger.js";

class LightManager {
  constructor(scene, sceneManager = null, gameManager = null) {
    this.scene = scene;
    this.sceneManager = sceneManager; // Optional, used for parenting lights under scene objects
    this.gameManager = gameManager; // Optional, used for criteria checking
    this.lights = new Map(); // Map of id -> THREE.Light or SplatEdit
    this.reactiveLights = new Map(); // Map of id -> { light, audioReactive }
    this.splatLayers = new Map(); // Map of id -> SplatEdit layer
    this.pendingAttachments = new Map(); // Map of id -> { object3D, config }
    this.gameState = null; // Store current game state for criteria checking
    this.lensFlares = new Map(); // Map of light id -> { lensflare, config }

    // Logger for debug messages
    this.logger = new Logger("LightManager", false);

    // Automatically load lights from data on initialization
    this.gameState = gameManager?.getState();
    this.loadLightsFromData(lights, this.gameState);

    // Initialize lens flare visibility based on initial game state
    if (this.gameState) {
      this.updateLensFlareVisibility(this.gameState);
    }
  }

  /**
   * Resolve the parent Object3D for a light based on config
   * @param {Object} config
   * @returns {THREE.Object3D}
   */
  _resolveParent(config) {
    // Default to scene when no sceneManager or no parent specified
    if (!this.sceneManager) return this.scene;

    // Support either { parentId, childName } or nested attachTo: { objectId, childName }
    const parentId = config?.parentId || config?.attachTo?.objectId;
    const childName = config?.childName || config?.attachTo?.childName;

    if (!parentId) return this.scene;

    // If a child name is provided, try to attach to that specific child
    if (childName && typeof this.sceneManager.findChildByName === "function") {
      const child = this.sceneManager.findChildByName(parentId, childName);
      if (child) return child;
    }

    // Otherwise attach to the parent object root
    if (typeof this.sceneManager.getObject === "function") {
      const parentObj = this.sceneManager.getObject(parentId);
      if (parentObj) return parentObj;
    }

    // Fallback to scene if not found
    return this.scene;
  }

  /**
   * Track objects that requested parenting but whose parent isn't loaded yet
   * @param {string} id
   * @param {THREE.Object3D} object3D
   * @param {Object} config
   */
  _maybeTrackPendingAttachment(id, object3D, config, parent) {
    const wantsParent = !!(config?.parentId || config?.attachTo);
    if (wantsParent && parent === this.scene && id) {
      this.pendingAttachments.set(id, { object3D, config });
      const parentId = config?.parentId || config?.attachTo?.objectId;
      this.logger.log(`⏸️ "${id}" waiting for parent "${parentId}" to load`);
    }
  }

  /**
   * Try to resolve any lights that couldn't find their parent at creation time
   */
  _tryResolvePendingAttachments() {
    if (!this.sceneManager || this.pendingAttachments.size === 0) return;

    let splatLightReattached = false;

    for (const [id, entry] of Array.from(this.pendingAttachments.entries())) {
      const { object3D, config } = entry;
      const parent = this._resolveParent(config);
      if (parent !== this.scene) {
        // Reparent without preserving world transform so config local offsets apply
        if (object3D.parent) {
          object3D.parent.remove(object3D);
        }
        parent.add(object3D);
        this.pendingAttachments.delete(id);
        this.logger.log(
          `✅ Reattached "${id}" under resolved parent (type: ${object3D.constructor.name})`
        );

        // Check if this is a splat light layer (SplatEdit)
        // Check both the constructor name and if it's stored in splatLayers
        const isSplatLight =
          this.splatLayers.has(id) || object3D.constructor.name === "SplatEdit";
        if (isSplatLight) {
          splatLightReattached = true;
          this.logger.log(`  → This is a splat light, will rebuild fog`);
        }
      }
    }

    // Rebuild fog if any splat lights were reattached
    if (splatLightReattached && window.cloudParticles) {
      this.logger.log("🌫️ Rebuilding fog after splat light reattachment...");
      window.cloudParticles.rebuild();
    }

    // Log pending attachments for debugging
    if (this.pendingAttachments.size > 0) {
      const pendingIds = Array.from(this.pendingAttachments.keys());
      this.logger.log(`⏳ Still waiting for parents: ${pendingIds.join(", ")}`);
    }
  }

  /**
   * Load and create all lights from data
   * @param {Object} lightsData - Object containing light definitions
   * @param {Object} gameState - Current game state for criteria checking
   */
  loadLightsFromData(lightsData, gameState = null) {
    this.logger.log("Loading lights from data...");

    for (const [key, config] of Object.entries(lightsData)) {
      // Check criteria if gameState is provided
      if (gameState && config.criteria) {
        if (!checkCriteria(gameState, config.criteria)) {
          this.logger.log(`⏭️ Skipping light "${key}" - criteria not met`);
          continue;
        }
      }

      this.logger.log(`🔦 Processing light "${key}" (type: ${config.type})`);
      try {
        if (config.type === "SplatLight") {
          this.createSplatLight(config);
        } else {
          // Regular Three.js light
          this.createLight(config);
        }
      } catch (error) {
        this.logger.error(`❌ Error creating light "${key}":`, error);
      }
    }

    this.logger.log(`Created ${this.lights.size} light(s)`);
  }

  /**
   * Create a light from config (dispatches to specific creator)
   * @param {Object} config - Light configuration
   * @returns {THREE.Light}
   */
  createLight(config) {
    switch (config.type) {
      case "AmbientLight":
        return this.createAmbientLight(config);
      case "HemisphereLight":
        return this.createHemisphereLight(config);
      case "DirectionalLight":
        return this.createDirectionalLight(config);
      case "PointLight":
        return this.createPointLight(config);
      case "SpotLight":
        return this.createSpotLight(config);
      default:
        this.logger.warn(`Unknown light type "${config.type}"`);
        return null;
    }
  }

  /**
   * Create a splat-based light
   * @param {Object} config - Splat light configuration
   * @returns {SplatEdit}
   */
  createSplatLight(config) {
    // Create lighting layer for the splat light
    const layer = new SplatEdit({
      rgbaBlendMode:
        SplatEditRgbaBlendMode[config.rgbaBlendMode] ||
        SplatEditRgbaBlendMode.ADD_RGBA,
      sdfSmooth: config.sdfSmooth ?? 0.1,
      softEdge: config.softEdge ?? 1.2,
    });

    // Set render order if specified (for controlling which objects are affected)
    if (config.renderOrder !== undefined) {
      layer.renderOrder = config.renderOrder;
    }

    const parent = this._resolveParent(config);
    parent.add(layer);
    this._maybeTrackPendingAttachment(config.id, layer, config, parent);

    // Create the splat light using SplatEditSdf
    const splatType =
      SplatEditSdfType[config.splatType] || SplatEditSdfType.SPHERE;
    const splatLight = new SplatEditSdf({
      type: splatType,
      color: new THREE.Color(config.color.r, config.color.g, config.color.b),
      radius: config.radius ?? 1,
      opacity: config.opacity ?? 0,
    });

    if (config.position) {
      splatLight.position.set(
        config.position.x ?? 0,
        config.position.y ?? 0,
        config.position.z ?? 0
      );
    }

    if (config.rotation) {
      splatLight.rotation.set(
        config.rotation.x ?? 0,
        config.rotation.y ?? 0,
        config.rotation.z ?? 0
      );
    }

    layer.add(splatLight);

    // Store references
    if (config.id) {
      this.lights.set(config.id, splatLight);
      this.splatLayers.set(config.id, layer);
    }

    // Create Three.js light duplicate if requested
    if (config.threeLightDuplicate) {
      this.createThreeLightDuplicate(config);
    }

    this.logger.log(
      `Created splat light "${config.id}" at (${config.position.x}, ${config.position.y}, ${config.position.z})`
    );

    // Rebuild fog to pick up new splat light
    if (window.cloudParticles) {
      window.cloudParticles.rebuild();
    }

    return layer;
  }

  /**
   * Create a Three.js light duplicate for a splat light
   * @param {Object} config - Splat light configuration
   * @returns {THREE.Light|null}
   */
  createThreeLightDuplicate(config) {
    // Get duplicate configuration (can be boolean true or an object with overrides)
    const duplicateConfig =
      typeof config.threeLightDuplicate === "object"
        ? config.threeLightDuplicate
        : {};

    // Convert splat color to hex
    const colorHex = new THREE.Color(
      config.color.r,
      config.color.g,
      config.color.b
    ).getHex();

    // Default to PointLight
    const lightType = duplicateConfig.type || "PointLight";

    // Build Three.js light config
    const threeLightConfig = {
      id: `${config.id}-three-duplicate`,
      type: lightType,
      color: duplicateConfig.color ?? colorHex,
      intensity: duplicateConfig.intensity ?? 1.0,
      position: duplicateConfig.position ?? config.position,
      distance: duplicateConfig.distance ?? 0,
      decay: duplicateConfig.decay ?? 2,
      castShadow: duplicateConfig.castShadow ?? false,
    };

    // Propagate parenting so the duplicate attaches to the same parent
    if (config.parentId) threeLightConfig.parentId = config.parentId;
    if (config.childName) threeLightConfig.childName = config.childName;
    if (config.attachTo) threeLightConfig.attachTo = { ...config.attachTo };

    // Create the light
    const light = this.createLight(threeLightConfig);

    if (light) {
      this.logger.log(`Created Three.js duplicate light for "${config.id}"`);

      // If gizmo is requested on the duplicate, register it
      if (duplicateConfig.gizmo && window.gizmoManager) {
        window.gizmoManager.registerObject(light, threeLightConfig.id, "light");
      }
    }

    return light;
  }

  /**
   * Create lens flare textures (radial glow, halo, soft glow)
   * @private
   * @returns {Object} Texture objects
   */
  _createLensFlareTextures() {
    const applyTextureSettings = (texture) => {
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.premultiplyAlpha = true;
      texture.needsUpdate = true;
      return texture;
    };

    const createRadialTexture = (size, innerAlpha, midAlpha, outerAlpha) => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = size;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, size, size);
      const gradient = ctx.createRadialGradient(
        size / 2,
        size / 2,
        0,
        size / 2,
        size / 2,
        size / 2
      );
      gradient.addColorStop(0.0, `rgba(255, 247, 224, ${innerAlpha})`);
      gradient.addColorStop(0.45, `rgba(255, 239, 205, ${midAlpha})`);
      gradient.addColorStop(1.0, `rgba(255, 232, 191, ${outerAlpha})`);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, size, size);
      return applyTextureSettings(new THREE.CanvasTexture(canvas));
    };

    const createRingTexture = (size, alpha) => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = size;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, size, size);
      const gradient = ctx.createRadialGradient(
        size / 2,
        size / 2,
        (size / 2) * 0.45,
        size / 2,
        size / 2,
        size / 2
      );
      gradient.addColorStop(0.0, "rgba(255, 255, 255, 0)");
      gradient.addColorStop(0.6, `rgba(255, 255, 255, ${alpha * 0.2})`);
      gradient.addColorStop(0.8, `rgba(255, 255, 255, ${alpha})`);
      gradient.addColorStop(1.0, "rgba(255, 255, 255, 0)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, size, size);
      return applyTextureSettings(new THREE.CanvasTexture(canvas));
    };

    return {
      core: createRadialTexture(384, 0.9, 0.35, 0.02),
      glow: createRadialTexture(512, 0.4, 0.15, 0.0),
      softGlow: createRadialTexture(640, 0.15, 0.05, 0.0),
      ring: createRingTexture(384, 0.45),
    };
  }

  /**
   * Attach a lens flare to a light
   * @param {THREE.Light} light - Light to attach flare to
   * @param {Object} config - Lens flare configuration from lightData
   * @returns {Object} { lensflare, baseColors, elements }
   */
  createLensFlareForLight(light, config) {
    if (!config?.enabled) return;

    const lensflare = new Lensflare();
    const textures = this._createLensFlareTextures();
    const flareColor = new THREE.Color(0xffe5c9); // Warm golden flare
    const baseColors = [];
    const elements = [];

    const elementConfigs = config.elements || [
      { size: 180, distance: 0.0 },
      { size: 140, distance: 0.25 },
      { size: 260, distance: 0.55 },
      { size: 320, distance: 0.85 },
    ];

    const textureSequence = [
      textures.core,
      textures.ring,
      textures.glow,
      textures.softGlow,
    ];

    for (let i = 0; i < elementConfigs.length; i++) {
      const elementConfig = elementConfigs[i];
      const texture = textureSequence[i] || textures.softGlow;
      const elementColor = flareColor.clone();
      baseColors.push(elementColor.clone());

      const element = new LensflareElement(
        texture,
        elementConfig.size,
        elementConfig.distance,
        elementColor
      );
      elements.push(element);
      lensflare.addElement(element);
    }

    light.add(lensflare);
    this.logger.log(
      `Created lens flare for light "${config.id || light.name}"`
    );

    // Initialize elements to zero intensity (invisible)
    for (let i = 0; i < elements.length; i++) {
      elements[i].size = 0;
      elements[i].color.set(0, 0, 0);
    }

    return { lensflare, baseColors, elements };
  }

  /**
   * Create an ambient light
   * @param {Object} config - Light configuration
   * @returns {THREE.AmbientLight}
   */
  createAmbientLight(config = {}) {
    const light = new THREE.AmbientLight(
      config.color ?? 0xffffff,
      config.intensity ?? 1.0
    );

    if (config.id) {
      this.lights.set(config.id, light);
    }

    const parent = this._resolveParent(config);
    parent.add(light);
    this._maybeTrackPendingAttachment(config.id, light, config, parent);
    return light;
  }

  /**
   * Create a hemisphere light
   * @param {Object} config - Light configuration
   * @returns {THREE.HemisphereLight}
   */
  createHemisphereLight(config = {}) {
    const light = new THREE.HemisphereLight(
      config.skyColor ?? 0xffffff,
      config.groundColor ?? 0x000000,
      config.intensity ?? 1.0
    );

    if (config.position) {
      light.position.set(
        config.position.x ?? 0,
        config.position.y ?? 0,
        config.position.z ?? 0
      );
    }

    if (config.id) {
      this.lights.set(config.id, light);
    }

    const parent = this._resolveParent(config);
    parent.add(light);
    this._maybeTrackPendingAttachment(config.id, light, config, parent);
    return light;
  }

  /**
   * Create a directional light
   * @param {Object} config - Light configuration
   * @returns {THREE.DirectionalLight}
   */
  createDirectionalLight(config = {}) {
    const light = new THREE.DirectionalLight(
      config.color ?? 0xffffff,
      config.intensity ?? 1.0
    );

    if (config.position) {
      light.position.set(
        config.position.x ?? 0,
        config.position.y ?? 0,
        config.position.z ?? 0
      );
    }

    if (config.castShadow !== undefined) {
      light.castShadow = config.castShadow;
    }

    // Configure shadow properties if shadow config is provided
    if (config.shadow && light.castShadow) {
      if (config.shadow.mapSize) {
        light.shadow.mapSize.width = config.shadow.mapSize.width ?? 512;
        light.shadow.mapSize.height = config.shadow.mapSize.height ?? 512;
      }
      if (config.shadow.camera) {
        const cam = config.shadow.camera;
        if (cam.left !== undefined) light.shadow.camera.left = cam.left;
        if (cam.right !== undefined) light.shadow.camera.right = cam.right;
        if (cam.top !== undefined) light.shadow.camera.top = cam.top;
        if (cam.bottom !== undefined) light.shadow.camera.bottom = cam.bottom;
        if (cam.near !== undefined) light.shadow.camera.near = cam.near;
        if (cam.far !== undefined) light.shadow.camera.far = cam.far;
        light.shadow.camera.updateProjectionMatrix();
      }
      if (config.shadow.bias !== undefined) {
        light.shadow.bias = config.shadow.bias;
      }
      if (config.shadow.normalBias !== undefined) {
        light.shadow.normalBias = config.shadow.normalBias;
      }
      this.logger.log(
        `Configured shadow for "${config.id}" - mapSize: ${light.shadow.mapSize.width}x${light.shadow.mapSize.height}`
      );
    }

    if (config.id) {
      this.lights.set(config.id, light);
    }

    const parent = this._resolveParent(config);
    parent.add(light);
    this._maybeTrackPendingAttachment(config.id, light, config, parent);
    return light;
  }

  /**
   * Create a point light
   * @param {Object} config - Light configuration
   * @returns {THREE.PointLight}
   */
  createPointLight(config = {}) {
    const light = new THREE.PointLight(
      config.color ?? 0xffffff,
      config.intensity ?? 1.0,
      config.distance ?? 0,
      config.decay ?? 2
    );

    if (config.position) {
      light.position.set(
        config.position.x ?? 0,
        config.position.y ?? 0,
        config.position.z ?? 0
      );
    }

    if (config.castShadow !== undefined) {
      light.castShadow = config.castShadow;
    }

    if (config.id) {
      this.lights.set(config.id, light);
    }

    const parent = this._resolveParent(config);
    parent.add(light);
    this._maybeTrackPendingAttachment(config.id, light, config, parent);

    // Create lens flare if enabled in the config
    if (config.lensFlare) {
      // Always create lens flare, we'll manage visibility via updateLensFlareVisibility
      const { lensflare, baseColors, elements } = this.createLensFlareForLight(
        light,
        config.lensFlare
      );
      this.lensFlares.set(config.id, {
        lensflare: lensflare,
        config: config.lensFlare,
        elements: elements,
        currentIntensity: 0, // Start faded out
        targetIntensity: 0,
        startIntensity: 0, // Track where fade started from
        fadeProgress: 0,
        isFading: false,
        baseColors: baseColors, // Store base colors for fading
        delayRemaining: 0, // Track fade delay
      });
    }

    return light;
  }

  /**
   * Create a spot light
   * @param {Object} config - Light configuration
   * @returns {THREE.SpotLight}
   */
  createSpotLight(config = {}) {
    const light = new THREE.SpotLight(
      config.color ?? 0xffffff,
      config.intensity ?? 1.0,
      config.distance ?? 0,
      config.angle ?? Math.PI / 3,
      config.penumbra ?? 0,
      config.decay ?? 2
    );

    this.logger.log(
      `🔦 Creating SpotLight "${config.id}" with intensity ${light.intensity}, distance ${light.distance}, angle ${light.angle}`
    );

    if (config.position) {
      light.position.set(
        config.position.x ?? 0,
        config.position.y ?? 0,
        config.position.z ?? 0
      );
    }

    if (config.castShadow !== undefined) {
      light.castShadow = config.castShadow;
    }

    if (config.id) {
      this.lights.set(config.id, light);
    }

    const parent = this._resolveParent(config);
    parent.add(light);
    this.logger.log(
      `  SpotLight parent: ${parent.type || parent.constructor.name}`
    );

    // Set up target after parenting so it can be relative to the parent
    if (config.target) {
      light.target.position.set(
        config.target.x ?? 0,
        config.target.y ?? 0,
        config.target.z ?? 0
      );
      // Add target to the same parent so it moves with the light
      parent.add(light.target);
      this.logger.log(
        `  SpotLight target at (${light.target.position.x}, ${light.target.position.y}, ${light.target.position.z})`
      );
    } else {
      // Default: point forward along parent's +Z axis
      light.target.position.set(0, 0, 10);
      parent.add(light.target);
      this.logger.log(`  SpotLight target at default (0, 0, 10)`);
    }

    this._maybeTrackPendingAttachment(config.id, light, config, parent);
    return light;
  }

  /**
   * Create an audio-reactive light
   * @param {string} id - Unique identifier for this light
   * @param {Howl} howl - Howl instance to analyze
   * @param {Object} config - Light and reactivity configuration
   * @returns {Object} { light, audioReactive }
   */
  createReactiveLight(id, howl, config) {
    try {
      // Create THREE.js light based on type
      let light;
      switch (config.type) {
        case "PointLight":
          light = this.createPointLight({
            ...config,
            id: null, // Don't double-register
            intensity: config.baseIntensity ?? 1.0,
          });
          break;
        case "SpotLight":
          light = this.createSpotLight({
            ...config,
            id: null,
            intensity: config.baseIntensity ?? 1.0,
          });
          break;
        case "DirectionalLight":
          light = this.createDirectionalLight({
            ...config,
            id: null,
            intensity: config.baseIntensity ?? 1.0,
          });
          break;
        default:
          this.logger.warn(`Unknown light type "${config.type}" for "${id}"`);
          return null;
      }

      // Create audio-reactive controller
      const audioReactive = new AudioReactiveLight(light, howl, {
        baseIntensity: config.baseIntensity,
        reactivityMultiplier: config.reactivityMultiplier,
        smoothing: config.smoothing,
        frequencyRange: config.frequencyRange,
        minIntensity: config.minIntensity,
        maxIntensity: config.maxIntensity,
        noiseFloor: config.noiseFloor,
      });

      // Store references
      this.lights.set(id, light);
      this.reactiveLights.set(id, { light, audioReactive });

      this.logger.log(`Created reactive light "${id}"`);
      return { light, audioReactive };
    } catch (error) {
      this.logger.error(`Error creating reactive light "${id}":`, error);
      return null;
    }
  }

  /**
   * Get a light by ID
   * @param {string} id - Light ID
   * @returns {THREE.Light|null}
   */
  getLight(id) {
    return this.lights.get(id) || null;
  }

  /**
   * Get reactive light data by ID
   * @param {string} id - Light ID
   * @returns {Object|null} { light, audioReactive }
   */
  getReactiveLight(id) {
    return this.reactiveLights.get(id) || null;
  }

  /**
   * Remove a light by ID
   * @param {string} id - Light ID
   */
  removeLight(id) {
    const light = this.lights.get(id);
    if (light) {
      if (light.parent) {
        light.parent.remove(light);
      } else {
        this.scene.remove(light);
      }
      this.lights.delete(id);
    }

    // Remove splat layer if present
    const splatLayer = this.splatLayers.get(id);
    if (splatLayer) {
      if (splatLayer.parent) {
        splatLayer.parent.remove(splatLayer);
      } else {
        this.scene.remove(splatLayer);
      }
      this.splatLayers.delete(id);
    }

    // Also remove from reactive lights if present
    const reactive = this.reactiveLights.get(id);
    if (reactive) {
      reactive.audioReactive.destroy();
      this.reactiveLights.delete(id);
    }

    // Remove lens flare if present
    const lensFlare = this.lensFlares.get(id);
    if (lensFlare) {
      if (lensFlare.lensflare.parent) {
        lensFlare.lensflare.parent.remove(lensFlare.lensflare);
      } else {
        this.scene.remove(lensFlare.lensflare);
      }
      this.lensFlares.delete(id);
    }
  }

  /**
   * Update lights based on game state - creates/removes lights with criteria
   * @param {Object} gameState - Current game state
   */
  updateLightsForState(gameState) {
    if (!gameState) return;

    this.gameState = gameState;

    // Check all lights in the data
    for (const [key, config] of Object.entries(lights)) {
      const lightId = config.id;
      const exists = this.lights.has(lightId) || this.splatLayers.has(lightId);

      // Check if criteria are met
      if (config.criteria) {
        const criteriaMet = checkCriteria(gameState, config.criteria);

        // If criteria met and light doesn't exist, create it
        if (criteriaMet && !exists) {
          this.logger.log(`🔦 Creating light "${lightId}" (criteria now met)`);
          try {
            if (config.type === "SplatLight") {
              this.createSplatLight(config);
            } else {
              this.createLight(config);
            }
          } catch (error) {
            this.logger.error(`❌ Error creating light "${lightId}":`, error);
          }
        }
        // If criteria not met and light exists, remove it
        else if (!criteriaMet && exists) {
          this.logger.log(
            `🔦 Removing light "${lightId}" (criteria no longer met)`
          );
          this.removeLight(lightId);
        }
      }
    }

    // Update lens flare visibility based on criteria
    this.updateLensFlareVisibility(gameState);
  }

  /**
   * Update lens flare visibility based on game state criteria
   * @param {Object} gameState - Current game state
   */
  updateLensFlareVisibility(gameState) {
    for (const [lightId, flareData] of this.lensFlares) {
      if (flareData.config?.criteria) {
        const criteriaMet = checkCriteria(gameState, flareData.config.criteria);
        const fadeDuration = flareData.config.fadeDuration || 1.5;
        const fadeDelay = flareData.config.fadeDelay || 0;

        // Set target intensity based on criteria
        const newTargetIntensity = criteriaMet ? 1.0 : 0.0;

        // If target changed, start fading (with delay)
        if (newTargetIntensity !== flareData.targetIntensity) {
          this.logger.log(
            `Lens flare "${lightId}" fade triggered: ${flareData.currentIntensity} -> ${newTargetIntensity} (delay: ${fadeDelay}s, duration: ${fadeDuration}s)`
          );
          flareData.startIntensity = flareData.currentIntensity; // Remember where we're fading from
          flareData.targetIntensity = newTargetIntensity;
          flareData.fadeProgress = 0;
          flareData.delayRemaining = fadeDelay;
          flareData.isFading = true;
        }
      }
    }
  }

  /**
   * Update lens flare fade effects (call this in animation loop)
   * @param {number} deltaTime - Delta time in seconds
   */
  updateLensFlares(deltaTime) {
    for (const [lightId, flareData] of this.lensFlares) {
      if (!flareData.isFading) continue;

      // Handle delay countdown
      if (flareData.delayRemaining > 0) {
        flareData.delayRemaining -= deltaTime;
        if (flareData.delayRemaining > 0) {
          continue; // Still waiting for delay to elapse
        }
        flareData.delayRemaining = 0; // Ensure exact 0
      }

      const fadeDuration = flareData.config.fadeDuration || 1.5;
      flareData.fadeProgress += deltaTime / fadeDuration;

      if (flareData.fadeProgress >= 1.0) {
        flareData.fadeProgress = 1.0;
        flareData.isFading = false;
        flareData.currentIntensity = flareData.targetIntensity;
        this.logger.log(
          `Lens flare "${lightId}" fade complete. Intensity: ${flareData.currentIntensity}`
        );
      } else {
        // Smooth easing (ease-in-out)
        const t = flareData.fadeProgress;
        const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
        // Lerp from start to target using eased progress
        flareData.currentIntensity = THREE.MathUtils.lerp(
          flareData.startIntensity,
          flareData.targetIntensity,
          eased
        );
      }

      // Update lensflare element sizes and opacity based on intensity
      if (flareData.lensflare && flareData.elements) {
        const elementConfigs = flareData.config.elements || [];
        for (let i = 0; i < flareData.elements.length; i++) {
          const element = flareData.elements[i];
          const baseSize = elementConfigs[i]?.size || 100;
          const baseColor = flareData.baseColors?.[i];

          // Scale size based on intensity
          element.size = baseSize * flareData.currentIntensity;

          // Interpolate color from transparent to base color
          if (baseColor) {
            element.color
              .copy(baseColor)
              .multiplyScalar(flareData.currentIntensity);
          }
        }
      }
    }
  }

  /**
   * Update all audio-reactive lights
   * Call this in your animation loop
   * @param {number} dt - Delta time (not used, but kept for consistency)
   */
  updateReactiveLights(dt) {
    // Attempt to resolve any deferred attachments (parents that loaded later)
    this._tryResolvePendingAttachments();

    for (const { audioReactive } of this.reactiveLights.values()) {
      audioReactive.update();
    }
  }

  /**
   * Enable/disable a reactive light
   * @param {string} id - Light ID
   * @param {boolean} enabled - Enable or disable
   */
  setReactiveLightEnabled(id, enabled) {
    const reactive = this.reactiveLights.get(id);
    if (reactive) {
      if (enabled) {
        reactive.audioReactive.enable();
      } else {
        reactive.audioReactive.disable();
      }
    }
  }

  /**
   * Get all light IDs
   * @returns {Array<string>}
   */
  getLightIds() {
    return Array.from(this.lights.keys());
  }

  /**
   * Get all reactive light IDs
   * @returns {Array<string>}
   */
  getReactiveLightIds() {
    return Array.from(this.reactiveLights.keys());
  }

  /**
   * Clean up all lights
   */
  destroy() {
    // Clean up reactive lights
    for (const [id, { light, audioReactive }] of this.reactiveLights) {
      audioReactive.destroy();
      if (light.parent) {
        light.parent.remove(light);
      } else {
        this.scene.remove(light);
      }
    }
    this.reactiveLights.clear();

    // Clean up splat layers
    for (const [id, layer] of this.splatLayers) {
      if (layer.parent) {
        layer.parent.remove(layer);
      } else {
        this.scene.remove(layer);
      }
    }
    this.splatLayers.clear();

    // Clean up regular lights
    for (const [id, light] of this.lights) {
      if (light.parent) {
        light.parent.remove(light);
      } else {
        this.scene.remove(light);
      }
    }
    this.lights.clear();

    // Clean up lens flares
    for (const [id, lensFlare] of this.lensFlares) {
      if (lensFlare.lensflare.parent) {
        lensFlare.lensflare.parent.remove(lensFlare.lensflare);
      } else {
        this.scene.remove(lensFlare.lensflare);
      }
    }
    this.lensFlares.clear();
  }
}

export default LightManager;
