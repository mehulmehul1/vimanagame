/**
 * main.js - VIMANA GAME ENTRY POINT
 * =============================================================================
 *
 * =============================================================================
 */

import * as THREE from 'three';
import { createSplatRenderer } from './rendering/VisionarySplatRenderer.ts';
import { createOptimalRenderer, logRendererInfo, RendererCapabilities } from './core/renderer.js';
import PhysicsManager from '@engine/physicsManager.js';
import CharacterController from '@engine/characterController.js';
import InputManager from '@engine/inputManager.js';
import LightManager from '@engine/lightManager.js';
import GameManager from '@engine/gameManager.js';
import ColliderManager from '@engine/colliderManager.js';
import SceneManager from '@engine/sceneManager.js';
import { detectPlatform } from '@engine/utils/platformDetection.js';
import { Logger } from '@engine/utils/logger.js';

// Import Vimana game data
import { GAME_STATES } from './gameData.js';
import { sceneObjects } from './sceneData.js';
import { colliders } from './colliderData.js';
import { lights } from './lightData.js';
import { HarpRoom } from './scenes/HarpRoom';

const logger = new Logger('Vimana', true);

// ==============================================================================
// VIMANA GAME CLASS
// ==============================================================================

class VimanaGame {
  constructor() {
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.splatRenderer = null; // Visionary-only renderer (no SparkJS fallback)

    // Managers
    this.gameManager = null;
    this.physicsManager = null;
    this.characterController = null;
    this.inputManager = null;
    this.sceneManager = null;
    this.lightManager = null;
    this.colliderManager = null;
    this.character = null;
    this.harpRoom = null;

    // Debug modes
    this.debugWireframe = false;
    this.debugWireframeMeshes = []; // Store original materials for toggling
    this.debugLightHelpers = false;
    this.debugHitboxes = false;
    this.debugCrosshair = false;

    // Debug menu element
    this.debugMenuElement = null;
    this.crosshairElement = null;
    this._isUpdatingMenu = false; // Flag to prevent recursive menu updates

    this.isInitialized = false;
  }

  async init() {
    if (this.isInitialized) return;

    logger.log('🎮 Vimana: Initializing...');

    // Get viewport size
    const getViewportSize = () => {
      if (window.visualViewport) {
        return { width: window.visualViewport.width, height: window.visualViewport.height };
      }
      return { width: window.innerWidth, height: window.innerHeight };
    };

    const initialSize = getViewportSize();

    // Create Three.js scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x87ceeb); // Sky blue background

    // Create camera
    this.camera = new THREE.PerspectiveCamera(60, initialSize.width / initialSize.height, 0.01, 100);
    this.camera.position.set(0, 1, 0);  // Move back a bit to see the scene
    this.scene.add(this.camera);  // Add camera to scene (required for camera children to render)

    // Add minimal ambient light so unlit areas aren't pitch black
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
    this.scene.add(ambientLight);

    // Create renderer (WebGPU with WebGL2 fallback)
    // Phase 1: Switch to WebGPURenderer when available, fall back to WebGL2
    // Visionary Integration: Using WebGPU for advanced rendering
    this.renderer = await createOptimalRenderer(
      { alpha: true, antialias: false },
      null, // no existing canvas yet
      // { requiresWebGL: true } // Constraint removed to enable WebGPU
    );
    this.renderer.setSize(initialSize.width, initialSize.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Log renderer type for debugging
    logRendererInfo(this.renderer);

    // Initialize global renderer capabilities
    await RendererCapabilities.init(this.renderer);
    window.rendererCapabilities = RendererCapabilities;

    // Check canvas before appending
    const canvas = this.renderer.domElement;
    logger.log(`🖼️ Canvas size: ${canvas.width}x${canvas.height}, CSS size would be: ${initialSize.width}x${initialSize.height}`);

    document.body.appendChild(canvas);

    // Verify canvas is in DOM
    logger.log(`🖼️ Canvas in DOM: ${document.body.contains(canvas)}`);
    logger.log(`🖼️ Canvas display: ${getComputedStyle(canvas).display}, visibility: ${getComputedStyle(canvas).visibility}`);

    // Hide HTML loading screen immediately after canvas is ready
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
      loadingScreen.classList.add('hidden');
      logger.log('✅ Loading screen hidden');
    }

    // Create game manager
    this.gameManager = new GameManager();
    window.gameManager = this.gameManager;

    // Detect platform
    const platformInfo = detectPlatform(this.gameManager);
    logger.log(`✅ Platform: Mobile=${platformInfo?.isMobile}`);

    // Create VisionarySplatRenderer for Gaussian splats
    this.splatRenderer = await createSplatRenderer(this.renderer, this.scene, {
      apertureAngle: 2 * Math.atan(0.005),
      focalDistance: 6.0,
      renderOrder: 9998
    });
    logger.log('✅ VisionarySplatRenderer created');

    // Create physics manager
    this.physicsManager = new PhysicsManager();
    logger.log('✅ PhysicsManager created');

    // Create scene manager (without gameManager to avoid loading shadow's assets)
    this.sceneManager = new SceneManager(this.scene, {
      renderer: this.renderer,
      splatRenderer: this.splatRenderer,
    });
    window.sceneManager = this.sceneManager;

    // Create light manager
    this.lightManager = new LightManager(this.scene, this.sceneManager, this.gameManager);
    logger.log('✅ LightManager created');

    // Clear engine's default lights (alley scene lights) - we'll only use GLB lights
    try {
      if (this.lightManager && this.lightManager.lights) {
        this.lightManager.lights.clear();
      }
      // Only traverse if scene exists and has the method
      if (this.scene?.traverse) {
        this.scene.traverse((child) => {
          if (child.isLight && child.parent) {
            child.parent.remove(child);
          }
        });
      }
    } catch (e) {
      console.warn('Could not clear lights:', e);
    }
    logger.log('✅ Cleared engine default lights');

    // Create character at spawn position
    // Get spawn position from scene data (will be used later for character creation)
    const musicRoom = sceneObjects.find(s => s.id === 'music_room');
    const spawnPos = musicRoom?.spawn?.position || { x: 0, y: 0.0, z: 0 };
    const spawnRot = musicRoom?.spawn?.rotation || { x: 0, y: 0, z: 0 };
    logger.log(`📍 Will spawn at: ${JSON.stringify(spawnPos)}, rotation: ${JSON.stringify(spawnRot)}`);
    this.character = this.physicsManager.createCharacter(spawnPos, spawnRot);

    // Store spawn position for later use (avoiding WASM issues with translation())
    this._spawnPosition = { ...spawnPos };
    this._spawnRotation = { ...spawnRot };

    // Create input manager
    this.inputManager = new InputManager(this.renderer.domElement, this.gameManager);
    logger.log('✅ InputManager created');

    // Create character controller (without sfxManager)
    this.characterController = new CharacterController(
      this.character,
      this.camera,
      this.renderer,
      this.inputManager,
      null,  // sfxManager
      this.splatRenderer,
      null,  // animationManager
      spawnRot
    );
    logger.log('✅ CharacterController created');

    // Explicitly set camera rotation to ensure it's facing forward (not down or corrupted)
    this.camera.rotation.set(0, 0, 0);
    this.camera.quaternion.set(0, 0, 0, 1);
    logger.log(`✅ Camera rotation reset to: (${this.camera.rotation.x.toFixed(2)}, ${this.camera.rotation.y.toFixed(2)}, ${this.camera.rotation.z.toFixed(2)})`);

    // Remove AudioListener from camera to prevent NaN crash in render loop
    // CharacterController adds a THREE.AudioListener but we use Howler for audio
    if (this.characterController.audioListener) {
      this.camera.remove(this.characterController.audioListener);
      // Don't set to null - character controller still needs the reference
      logger.log('✅ AudioListener removed from camera (using Howler instead)');
    }

    // Link physics manager to scene manager
    this.sceneManager.physicsManager = this.physicsManager;

    // Create collider manager
    this.colliderManager = new ColliderManager(
      this.physicsManager,
      this.gameManager,
      colliders,
      this.scene,
      this.sceneManager,
      null
    );
    this.colliderManager.setCamera(this.camera);
    window.colliderManager = this.colliderManager;
    this.sceneManager.setColliderManager(this.colliderManager);
    logger.log('✅ ColliderManager initialized');

    // Set up manager references
    this.characterController.setGameManager(this.gameManager);
    this.characterController.setSceneManager(this.sceneManager);
    this.characterController.setPhysicsManager(this.physicsManager);

    // Set initial vimana game state
    this.gameManager.setState({
      currentState: GAME_STATES.MUSIC_ROOM,
      controlEnabled: true,
      isPlaying: true,

      // Shell collection initial state
      archiveOfVoices: false,
      galleryOfForms: false,
      hydroponicMemory: false,
      engineOfGrowth: false,
      shellsCollected: 0,
      currentChamber: 'archiveOfVoices',

      // Harp puzzle initial state
      harpState: 0, // IDLE
      harpTargetSequence: [],
      harpSequenceProgress: 0,
      harpSequencesCompleted: 0,
      harpRequiredSequences: 3,
      harpPlayedStrings: [],

      // Jelly creature initial state
      jellyVisible: true,
      jellyEmotion: 'neutral',
      jellyDemonstrating: false,

      // Visual effect initial state
      waterRippleIntensity: 0,
      vibratingString: -1,
      vortexActivation: 0,
    });

    // Update blocking colliders now that state is set (creates floor collider)
    this.colliderManager.updateBlockingColliders();
    logger.log('✅ Blocking colliders updated after state set');

    // Load vimana's scene data
    await this.loadVimanaScene();

    // Initialize HarpRoom mechanics (pass gameManager for state integration)
    this.harpRoom = new HarpRoom(this.scene, this.camera, this.renderer, this.gameManager, this.splatRenderer);

    // Find the loaded music room object to pass to initialize
    // GLB root is named "Scene Collection", not "Scene"
    // Pass this.scene directly so HarpRoom can search the full hierarchy
    await this.harpRoom.initialize(this.scene);

    // Create debug menu after harpRoom is ready
    this.createDebugMenu();

    // Debug: Log scene info
    logger.log(`📍 Camera position: ${this.camera.position.x.toFixed(2)}, ${this.camera.position.y.toFixed(2)}, ${this.camera.position.z.toFixed(2)}`);
    logger.log(`📍 Camera rotation: ${this.camera.rotation.x.toFixed(2)}, ${this.camera.rotation.y.toFixed(2)}, ${this.camera.rotation.z.toFixed(2)}`);

    // Count renderable objects
    let meshCount = 0;
    this.scene.traverse((obj) => {
      if (obj.isMesh) {
        meshCount++;
        if (obj.material) {
          logger.log(`  Mesh "${obj.name}": visible=${obj.visible}, material=${obj.material.type || 'none'}, opacity=${obj.material?.opacity ?? 1}`);
        }
      }
    });
    logger.log(`📊 Total meshes in scene: ${meshCount}`);

    // Enable controls
    this.inputManager.enable();

    // Set up event listeners
    this.setupEventListeners();

    // Start handling harp interactions
    this.setupInteractions();

    // Start render loop
    this.startRenderLoop();

    this.isInitialized = true;
    logger.log('✅ Vimana: Initialized - ready to play!');
  }

  async loadVimanaScene() {
    logger.log('🏠 Loading Vimana music room...');

    // Load the music room GLB
    const musicRoom = sceneObjects.find(s => s.id === 'music_room');
    if (musicRoom) {
      try {
        await this.sceneManager.loadObject(musicRoom);
        logger.log('✅ Music room loaded');

        // Create trimesh colliders from all meshes for accurate collision
        this.createTrimeshCollidersFromScene();

        // Apply double-side rendering to all meshes so they're visible from inside
        this.scene.traverse((obj) => {
          if (obj.isMesh && obj.material) {
            // Handle both single materials and material arrays
            if (Array.isArray(obj.material)) {
              obj.material.forEach(mat => {
                mat.side = THREE.DoubleSide;
              });
            } else {
              obj.material.side = THREE.DoubleSide;
            }
          }
        });
        logger.log('✅ Applied double-side rendering to all meshes');

        // Scale down GLB light intensities (Blender exports much higher values than Three.js expects)
        let scaledLightCount = 0;
        this.scene.traverse((obj) => {
          if (obj.isLight) {
            const originalIntensity = obj.intensity;
            // Scale different light types appropriately
            if (obj.type === 'PointLight') {
              obj.intensity = originalIntensity * 0.01; // Scale to 1%
            } else if (obj.type === 'DirectionalLight') {
              obj.intensity = originalIntensity * 0.0001; // Scale to 0.01%
            } else if (obj.type === 'SpotLight') {
              obj.intensity = originalIntensity * 0.01; // Scale to 1%
            }
            scaledLightCount++;
            logger.log(`  💡 Scaled ${obj.type} "${obj.name || '(unnamed)'}": ${originalIntensity.toFixed(2)} → ${obj.intensity.toFixed(2)}`);
          }
        });
        logger.log(`✅ Scaled ${scaledLightCount} GLB lights to Three.js values`);

        // Log what was loaded
        this.scene.traverse((obj) => {
          if (obj.isMesh) {
            logger.log(`  - Mesh: ${obj.name || '(unnamed)'}`);
          }
        });
      } catch (err) {
        logger.log(`⚠️ Could not load music room GLB: ${err.message}`);
      }
    }

    // Load lights
    logger.log(`🔦 Loading ${Object.keys(lights).length} lights from vimana lightData...`);
    this.lightManager.loadLightsFromData(lights);
    logger.log(`✅ Lights loaded. LightManager has ${this.lightManager.lights?.size || 0} lights.`);

    // Debug: Log all lights in scene
    this.debugLights();

    // Check for debug light helpers via URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    this.debugLightHelpers = urlParams.get('debugLights') === 'true';
    if (this.debugLightHelpers) {
      this.toggleLightHelpers();
    }

    // Set camera to spawn position (character was already created at spawn position in init)
    // Use stored spawn position to avoid WASM issues
    this.camera.position.set(
      this._spawnPosition.x,
      this._spawnPosition.y + 0.7,
      this._spawnPosition.z
    );
    logger.log(`✅ Camera spawned at ${JSON.stringify(this._spawnPosition)}`);

    // Set camera rotation from spawn (Y rotation in degrees)
    if (this._spawnRotation) {
      const yaw = THREE.MathUtils.degToRad(this._spawnRotation.y || 0);
      const pitch = THREE.MathUtils.degToRad(this._spawnRotation.x || 0);
      this.camera.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));
      logger.log(`✅ Camera rotation set to: ${JSON.stringify(this._spawnRotation)}`);
    }

    // Visionary renderer handles its own positioning - no need to set position
    // (SparkJS fallback removed - using Visionary-only)

    // Check for debug wireframe mode via URL parameter (reuse urlParams from above)
    this.debugWireframe = urlParams.get('debugWireframe') === 'true';
    if (this.debugWireframe) {
      this.toggleWireframeMode();
      logger.log('🔧 Debug wireframe mode enabled via URL parameter');
    }
  }

  /**
   * Create trimesh colliders from all meshes in the scene for accurate physics collision
   * This replaces the simple box collider with actual mesh-based collision
   */
  createTrimeshCollidersFromScene() {
    logger.log('🔧 Creating trimesh colliders from scene meshes...');

    // Find the "Scene" root which contains all the GLB content
    const sceneRoot = this.scene.children.find(c => c.name === 'Scene');
    if (!sceneRoot) {
      logger.log('⚠️ No "Scene" root found, looking for meshes directly...');
      // Use the main scene instead
      this.createTrimeshFromObject(this.scene);
    } else {
      this.createTrimeshFromObject(sceneRoot);
    }

    logger.log('✅ Trimesh colliders created');
  }

  /**
   * Recursively create trimesh colliders from all meshes in an object
   * @param {THREE.Object3D} object - The object to traverse
   */
  createTrimeshFromObject(object) {
    // Meshes that should NOT have collision (decorative only)
    const noCollisionMeshes = ['Window_1', 'Window_2', 'Window_3', 'Gate_Plug'];

    object.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const meshName = child.name || '(unnamed)';

        // Skip meshes that should not have collision (decorative only)
        if (noCollisionMeshes.includes(meshName)) {
          logger.log(`  ⏭️ Skipping collision for: ${meshName} (decorative)`);
          return;
        }

        // Skip collision meshes if they exist (convention: COLL_ prefix)
        if (meshName.startsWith('COLL_')) {
          logger.log(`  📦 Creating collider from: ${meshName}`);
        }

        // Create trimesh collider from this mesh's geometry
        const geometryData = this.physicsManager.extractGeometryFromObject(child);
        if (geometryData) {
          const colliderId = `trimesh_${meshName}`;
          const result = this.physicsManager.createTrimeshCollider(
            colliderId,
            child,
            { x: 0, y: 0, z: 0 },  // Position is already in world space via extractGeometryFromObject
            { x: 0, y: 0, z: 0, w: 1 }
          );

          if (result) {
            logger.log(`  ✅ Trimesh collider created for: ${meshName} (${geometryData.vertices.length / 3} vertices)`);
          }
        }
      }
    });
  }

  setupEventListeners() {
    // Window resize
    const handleResize = () => {
      const getViewportSize = () => {
        if (window.visualViewport) {
          return { width: window.visualViewport.width, height: window.visualViewport.height };
        }
        return { width: window.innerWidth, height: window.innerHeight };
      };

      const { width, height } = getViewportSize();
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', handleResize);
      window.visualViewport.addEventListener('scroll', handleResize);
    }

    // Keyboard handler for debug mode
    window.addEventListener('keydown', (e) => {
      // Toggle wireframe debug mode with 'C' key
      if (e.key === 'c' || e.key === 'C') {
        this.toggleWireframeMode();
        logger.log(`🔧 Wireframe debug mode: ${this.debugWireframe ? 'ON' : 'OFF'}`);
        this.updateDebugMenu();
      }
      // Toggle light helpers with 'L' key
      if (e.key === 'l' || e.key === 'L') {
        this.toggleLightHelpers();
        this.updateDebugMenu();
      }
      // Toggle hitbox debug with 'H' key
      if (e.key === 'h' || e.key === 'H') {
        this.toggleHitboxDebug();
      }
      // Toggle crosshair with 'X' key
      if (e.key === 'x' || e.key === 'X') {
        this.toggleCrosshair();
      }
      // Toggle debug menu visibility with 'F1' key
      if (e.key === 'F1') {
        e.preventDefault();
        if (this.debugMenuElement) {
          this.toggleDebugMenu();
        } else {
          this.createDebugMenu();
        }
      }
    });
  }

  setupInteractions() {
    // Flag to track if AudioContext has been resumed
    let audioContextResumed = false;

    const resumeAudioContext = async () => {
      if (audioContextResumed) return;
      audioContextResumed = true;

      console.log('[main.js] Resuming AudioContext on user interaction...');

      // Resume HarmonyChord and GentleAudioFeedback through HarpRoom
      if (this.harpRoom && this.harpRoom.getAudioSystems) {
        const audioSystems = this.harpRoom.getAudioSystems();
        if (audioSystems.harmonyChord) {
          await audioSystems.harmonyChord.resume();
          console.log('[main.js] HarmonyChord resumed');
        }
        if (audioSystems.gentleAudioFeedback) {
          await audioSystems.gentleAudioFeedback.resume();
          console.log('[main.js] GentleAudioFeedback resumed');
        }
      }

      console.log('[main.js] AudioContext resumed successfully');
    };

    const handleInteraction = (event) => {
      // Resume AudioContext on first interaction
      resumeAudioContext();

      if (!this.harpRoom) return;

      // DIAGNOSTIC LOG: Log input coordinates
      console.log('[main.js DIAGNOSTIC] Interaction event:', {
        type: event.type,
        clientX: event.clientX,
        clientY: event.clientY,
        pointerLocked: !!document.pointerLockElement
      });

      // Use center of screen if pointer is locked, otherwise use mouse position
      const mouse = new THREE.Vector2();
      if (document.pointerLockElement) {
        mouse.set(0, 0); // Raycast from center
      } else {
        mouse.set(
          (event.clientX / window.innerWidth) * 2 - 1,
          -(event.clientY / window.innerHeight) * 2 + 1
        );
      }

      // DIAGNOSTIC LOG: Log normalized mouse coordinates
      console.log('[main.js DIAGNOSTIC] Normalized mouse:', {
        x: mouse.x.toFixed(3),
        y: mouse.y.toFixed(3)
      });

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, this.camera);

      // DIAGNOSTIC LOG: Log raycaster before passing to harpRoom
      console.log('[main.js DIAGNOSTIC] Raycaster created:', {
        rayOrigin: raycaster.ray.origin,
        rayDirection: raycaster.ray.direction
      });

      this.harpRoom.onClick(raycaster);
    };

    console.log('[main.js] Setting up interaction listeners...');
    window.addEventListener('mousedown', handleInteraction);
    
    // Also allow interaction via Space/Enter if looking at strings
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space' || e.code === 'Enter') {
        handleInteraction({ clientX: window.innerWidth / 2, clientY: window.innerHeight / 2 });
      }
    });
  }

  /**
   * Toggle wireframe debug mode for all meshes in the scene
   * This helps visualize the geometry and diagnose rendering issues
   */
  toggleWireframeMode() {
    this.debugWireframe = !this.debugWireframe;

    // Create wireframe material if needed
    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0x00ff00,
      wireframe: true,
      side: THREE.DoubleSide,
    });

    // Traverse scene and apply/remove wireframe material
    this.scene.traverse((child) => {
      if (child.isMesh) {
        if (this.debugWireframe) {
          // Store original material
          if (!child.userData.originalMaterial) {
            child.userData.originalMaterial = child.material;
          }
          child.material = wireframeMaterial;
          child.renderOrder = 9999; // Render on top
        } else {
          // Restore original material
          if (child.userData.originalMaterial) {
            child.material = child.userData.originalMaterial;
          }
        }
      }
    });

    logger.log(`🔧 Wireframe mode ${this.debugWireframe ? 'enabled' : 'disabled'}`);

    this.updateDebugMenu();
  }

  /**
   * Log all meshes in the scene with their positions and visibility
   * Useful for debugging rendering issues
   */
  logSceneMeshes() {
    logger.log('🔍 Scene mesh dump:');
    const meshInfo = [];

    this.scene.traverse((child) => {
      if (child.isMesh) {
        const info = {
          name: child.name || '(unnamed)',
          position: child.position.toArray().map(v => v.toFixed(2)),
          visible: child.visible,
          materialType: child.material?.type || 'none',
          renderOrder: child.renderOrder,
        };
        meshInfo.push(info);
        logger.log(`  - ${info.name}: pos=[${info.position}], visible=${info.visible}, material=${info.materialType}`);
      }
    });

    logger.log(`📊 Total meshes: ${meshInfo.length}`);
    return meshInfo;
  }

  /**
   * Log all lights in the scene with their positions and properties
   */
  debugLights() {
    logger.log('💡 Scene light dump:');
    const lightInfo = [];

    // Get lights from lightManager's lights map
    if (this.lightManager && this.lightManager.lights) {
      for (const [id, light] of this.lightManager.lights) {
        const info = {
          id: id,
          type: light.type || light.constructor?.name || 'Unknown',
          name: light.name || '(unnamed)',
          position: light.position?.toArray?.().map(v => v.toFixed(2)) || [0, 0, 0],
          color: light.color?.getHexString?.() || 'ffffff',
          intensity: light.intensity?.toFixed?.(2) || 0,
          visible: light.visible ?? true,
        };
        lightInfo.push(info);

        // Add type-specific info
        let typeInfo = '';
        if (info.type === 'PointLight') {
          typeInfo = ` range=${light.distance?.toFixed(2) || 0}, decay=${light.decay?.toFixed(2) || 0}`;
        } else if (info.type === 'SpotLight') {
          typeInfo = ` angle=${light.angle?.toFixed(2) || 0}, penumbra=${light.penumbra?.toFixed(2) || 0}`;
        } else if (info.type === 'DirectionalLight') {
          const target = light.target?.position?.toArray?.().map(v => v.toFixed(2)) || [0, 0, 0];
          typeInfo = ` target=[${target.join(', ')}]`;
        } else if (info.type === 'HemisphereLight') {
          const groundColor = light.groundColor?.getHexString?.() || 'ffffff';
          typeInfo = ` groundColor=#${groundColor}`;
        }

        logger.log(`  - ${info.type} "${info.name}" (id: ${info.id}): pos=[${info.position.join(', ')}], color=#${info.color}, intensity=${info.intensity}${typeInfo ? ',' + typeInfo : ''}`);
      }
    }

    // Also traverse scene for any lights not managed by lightManager
    this.scene.traverse((child) => {
      if (child.isLight) {
        // Check if this light is already in our list (by reference)
        const alreadyListed = lightInfo.some(info => info.position === child.position?.toArray?.().map(v => v.toFixed(2)));
        if (!alreadyListed) {
          const info = {
            type: child.type,
            name: child.name || '(unnamed)',
            position: child.position.toArray().map(v => v.toFixed(2)),
            color: child.color?.getHexString() || 'ffffff',
            intensity: child.intensity?.toFixed(2) || 0,
            visible: child.visible,
          };
          lightInfo.push(info);
          logger.log(`  - ${info.type} "${info.name}" (scene): pos=[${info.position.join(', ')}], color=#${info.color}, intensity=${info.intensity}`);
        }
      }
    });

    logger.log(`📊 Total lights: ${lightInfo.length}`);
    return lightInfo;
  }

  /**
   * Toggle visual helpers for all lights in the scene
   */
  toggleLightHelpers() {
    this.debugLightHelpers = !this.debugLightHelpers;

    if (this.debugLightHelpers) {
      // Create simple helpers for all lights
      this._lightHelpers = [];

      // Get lights from lightManager
      if (this.lightManager && this.lightManager.lights) {
        for (const [id, light] of this.lightManager.lights) {
          const color = light.color?.getHex?.() || 0xffff00;
          const pos = light.position || new THREE.Vector3(0, 0, 0);

          // Create a simple wireframe sphere at light position
          const geometry = new THREE.SphereGeometry(0.3, 8, 8);
          const material = new THREE.MeshBasicMaterial({
            color: color,
            wireframe: true,
          });
          const helper = new THREE.Mesh(geometry, material);
          helper.position.copy(pos);

          // Add text label with light ID
          helper.userData = { lightId: id, lightType: light.type || 'Light' };

          this.scene.add(helper);
          this._lightHelpers.push(helper);
        }
      }

      // Also traverse scene for any lights not managed by lightManager
      this.scene.traverse((child) => {
        if (child.isLight) {
          const color = child.color?.getHex() || 0xffff00;

          const geometry = new THREE.SphereGeometry(0.3, 8, 8);
          const material = new THREE.MeshBasicMaterial({
            color: color,
            wireframe: true,
          });
          const helper = new THREE.Mesh(geometry, material);
          helper.position.copy(child.position);
          helper.userData = { lightType: child.type };

          this.scene.add(helper);
          this._lightHelpers.push(helper);
        }
      });

      logger.log(`💡 Light helpers enabled (${this._lightHelpers.length} helpers)`);
    } else {
      // Remove all light helpers
      if (this._lightHelpers) {
        this._lightHelpers.forEach(helper => {
          this.scene.remove(helper);
          helper.geometry?.dispose();
          helper.material?.dispose();
        });
        this._lightHelpers = [];
      }

      logger.log('💡 Light helpers disabled');
    }

    this.updateDebugMenu();
  }

  /**
   * Toggle hitbox debug visibility
   * Shows/hides magenta wireframe boxes around harp strings
   */
  toggleHitboxDebug() {
    this.debugHitboxes = !this.debugHitboxes;

    if (this.harpRoom && this.harpRoom.toggleHitboxDebug) {
      this.harpRoom.toggleHitboxDebug();
    }

    logger.log(`📦 Hitbox debug: ${this.debugHitboxes ? 'ON' : 'OFF'}`);

    // Update menu checkbox if it exists
    this.updateDebugMenu();
  }

  /**
   * Toggle crosshair visibility
   * Shows a crosshair at screen center for precise clicking
   */
  toggleCrosshair() {
    this.debugCrosshair = !this.debugCrosshair;

    if (this.debugCrosshair) {
      // Create crosshair if it doesn't exist
      if (!this.crosshairElement) {
        this.crosshairElement = document.createElement('div');
        this.crosshairElement.id = 'debug-crosshair';
        this.crosshairElement.style.cssText = `
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 20px;
          height: 20px;
          pointer-events: none;
          z-index: 9999;
        `;
        this.crosshairElement.innerHTML = `
          <div style="position: absolute; top: 9px; left: 0; width: 20px; height: 2px; background: rgba(0, 255, 0, 0.8);"></div>
          <div style="position: absolute; top: 0; left: 9px; width: 2px; height: 20px; background: rgba(0, 255, 0, 0.8);"></div>
          <div style="position: absolute; top: 7px; left: 7px; width: 6px; height: 6px; border: 1px solid rgba(0, 255, 0, 0.8); border-radius: 50%;"></div>
        `;
      }
      this.crosshairElement.style.display = 'block';
      document.body.appendChild(this.crosshairElement);
    } else {
      if (this.crosshairElement) {
        this.crosshairElement.style.display = 'none';
      }
    }

    logger.log(`🎯 Crosshair: ${this.debugCrosshair ? 'ON' : 'OFF'}`);

    // Update menu checkbox if it exists
    this.updateDebugMenu();
  }

  /**
   * Create unified debug menu with checkboxes
   */
  createDebugMenu() {
    if (this.debugMenuElement) return; // Already created

    const menu = document.createElement('div');
    menu.id = 'debug-menu';
    menu.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.8);
      color: #fff;
      padding: 15px;
      border-radius: 8px;
      font-family: monospace;
      font-size: 12px;
      z-index: 10000;
      min-width: 200px;
    `;

    menu.innerHTML = `
      <div style="margin-bottom: 10px; font-weight: bold; color: #00ff00;">🔧 DEBUG MENU</div>
      <label style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" id="debug-wireframe" ${this.debugWireframe ? 'checked' : ''}>
        🕸️ Wireframe Mode (C)
      </label>
      <label style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" id="debug-lights" ${this.debugLightHelpers ? 'checked' : ''}>
        💡 Light Helpers (L)
      </label>
      <label style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" id="debug-hitboxes" ${this.debugHitboxes ? 'checked' : ''}>
        📦 Hitbox Debug (H)
      </label>
      <label style="display: block; margin-bottom: 8px; cursor: pointer;">
        <input type="checkbox" id="debug-crosshair" ${this.debugCrosshair ? 'checked' : ''}>
        🎯 Crosshair (X)
      </label>
      <div style="margin-top: 10px; font-size: 10px; color: #888;">
        Press F1 to toggle menu
      </div>
    `;

    document.body.appendChild(menu);
    this.debugMenuElement = menu;

    // Add event listeners for checkboxes
    document.getElementById('debug-wireframe').addEventListener('change', (e) => {
      if (this._isUpdatingMenu) return;
      if (e.target.checked !== this.debugWireframe) {
        this.toggleWireframeMode();
      }
    });

    document.getElementById('debug-lights').addEventListener('change', (e) => {
      if (this._isUpdatingMenu) return;
      if (e.target.checked !== this.debugLightHelpers) {
        this.toggleLightHelpers();
      }
    });

    document.getElementById('debug-hitboxes').addEventListener('change', (e) => {
      if (this._isUpdatingMenu) return;
      if (e.target.checked !== this.debugHitboxes) {
        this.toggleHitboxDebug();
      }
    });

    document.getElementById('debug-crosshair').addEventListener('change', (e) => {
      if (this._isUpdatingMenu) return;
      if (e.target.checked !== this.debugCrosshair) {
        this.toggleCrosshair();
      }
    });
  }

  /**
   * Update debug menu checkboxes to match current state
   * Flag to prevent recursive updates from checkbox change events
   */
  updateDebugMenu() {
    if (!this.debugMenuElement) return;

    // Set flag to prevent recursive updates
    this._isUpdatingMenu = true;

    const wireframeCb = document.getElementById('debug-wireframe');
    const lightsCb = document.getElementById('debug-lights');
    const hitboxesCb = document.getElementById('debug-hitboxes');
    const crosshairCb = document.getElementById('debug-crosshair');

    if (wireframeCb) wireframeCb.checked = this.debugWireframe;
    if (lightsCb) lightsCb.checked = this.debugLightHelpers;
    if (hitboxesCb) hitboxesCb.checked = this.debugHitboxes;
    if (crosshairCb) crosshairCb.checked = this.debugCrosshair;

    // Clear flag after updates
    this._isUpdatingMenu = false;
  }

  /**
   * Toggle debug menu visibility
   */
  toggleDebugMenu() {
    if (this.debugMenuElement) {
      const isVisible = this.debugMenuElement.style.display !== 'none';
      this.debugMenuElement.style.display = isVisible ? 'none' : 'block';
      logger.log(`🔧 Debug menu: ${isVisible ? 'HIDDEN' : 'SHOWN'}`);
    }
  }

  startRenderLoop() {
    let lastTime;
    let frameCount = 0;

    logger.log('🔄 Starting render loop...');

    this.renderer.setAnimationLoop((time) => {
      try {
        frameCount++;
        if (frameCount <= 3) {
          logger.log(`🔄 Frame ${frameCount} - render loop is running!`);
        }

        const t = time * 0.001;
        const dt = Math.min(0.1, t - (lastTime ?? t));
        lastTime = t;

        // Update physics
        this.physicsManager.step(dt);

        // Update collider manager
        if (this.character) {
          this.colliderManager.update(this.character, false);
        }

        // Update input
        this.inputManager.update(dt);

        // Update character controller
        if (this.gameManager.isControlEnabled()) {
          this.characterController.update(dt);
        }

        // Update scene manager
        this.sceneManager.update(dt);

        // Update game manager
        this.gameManager.update(dt);

        // Update lights
        if (this.lightManager) {
          this.lightManager.updateReactiveLights?.(dt);
          this.lightManager.updateLensFlares?.(dt);
        }

        // Visionary renderer handles its own positioning - no sync needed (SparkJS removed)

        // Update Harp Room logic
        if (this.harpRoom) {
          this.harpRoom.render(); // This now updates logic and renders
        } else {
          // Fallback render if harpRoom isn't ready
          this.renderer.render(this.scene, this.camera);
        }
      } catch (e) {
        if (frameCount <= 5) {
          logger.log(`❌ Render error at frame ${frameCount}: ${e.message}`);
          console.error(e);
        }
      }
    });
  }

  destroy() {
    if (this.physicsManager) this.physicsManager.destroy();
    if (this.inputManager) this.inputManager.destroy();
  }
}

// ==============================================================================
// INITIALIZE GAME
// ==============================================================================

let vimanaGame;

function initVimana() {
  vimanaGame = new VimanaGame();
  vimanaGame.init();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initVimana);
} else {
  initVimana();
}

window.vimanaGame = vimanaGame;

// Expose debug functions to console
window.debugVimana = {
  toggleWireframe: () => vimanaGame?.toggleWireframeMode(),
  logMeshes: () => vimanaGame?.logSceneMeshes(),
  getMeshes: () => vimanaGame?.scene?.children?.filter(c => c.isMesh) || [],
  logLights: () => vimanaGame?.debugLights(),
  toggleLightHelpers: () => vimanaGame?.toggleLightHelpers(),
  getLights: () => {
    const lights = [];
    vimanaGame?.scene?.traverse((child) => {
      if (child.isLight) lights.push(child);
    });
    return lights;
  },
  // Phase 1: Check renderer type and capabilities
  getRendererInfo: () => {
    if (!vimanaGame?.renderer) return null;
    return {
      type: window.rendererType || 'Unknown',
      isWebGPU: window.rendererType === 'WebGPU',
      capabilities: window.rendererCapabilities?.instance || null,
      hasCompute: window.rendererCapabilities?.hasCompute(),
      hasTSL: window.rendererCapabilities?.hasTSL(),
    };
  },
  logRenderer: () => {
    const info = window.debugVimana.getRendererInfo();
    console.log('📊 Renderer Info:', {
      Type: info?.type,
      'WebGPU': info?.isWebGPU ? '✅' : '❌',
      'Compute Shaders': info?.hasCompute ? '✅' : '❌',
      'TSL Support': info?.hasTSL ? '✅' : '❌',
    });
    return info;
  },
};
