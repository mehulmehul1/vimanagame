import * as THREE from "three";
import { SparkRenderer } from "@sparkjsdev/spark";
import { Howler } from "howler";
import PhysicsManager from "./physicsManager.js";
import CharacterController from "./characterController.js";
import InputManager from "./inputManager.js";
import MusicManager from "./musicManager.js";
import SFXManager from "./sfxManager.js";
import LightManager from "./lightManager.js";
import OptionsMenu from "./ui/optionsMenu.js";
import DialogManager from "./dialogManager.js";
import DialogChoiceUI from "./ui/dialogChoiceUI.js";
import GameManager from "./gameManager.js";
import UIManager from "./ui/uiManager.js";
import ColliderManager from "./colliderManager.js";
import SceneManager from "./sceneManager.js";
import colliders from "./colliderData.js";
import {
  sceneObjects,
  getEnvMapWorldCenters,
  captureEnvMapAtScene,
} from "./sceneData.js";
import { videos } from "./videoData.js";
import { lights } from "./lightData.js";
import { StartScreen } from "./ui/startScreen.js";
import { GAME_STATES } from "./gameData.js";
import AnimationManager from "./animationManager.js";
import cameraAnimations from "./animationCameraData.js";
import objectAnimations from "./animationObjectData.js";
import GizmoManager from "./utils/gizmoManager.js";
import { VFXSystemManager } from "./vfxManager.js";
import { LoadingScreen } from "./ui/loadingScreen.js";
import { Logger } from "./utils/logger.js";
import { DrawingRecognitionManager } from "./drawing/drawingRecognitionManager.js";
import { DrawingManager } from "./drawing/drawingManager.js";
import "./styles/optionsMenu.css";
import "./styles/dialog.css";
import "./styles/loadingScreen.css";
import "./styles/fullscreenButton.css";

const logger = new Logger("Main", true);

// Add global error handlers to catch silent failures
window.addEventListener("unhandledrejection", (event) => {
  logger.error("❌ Unhandled promise rejection:", event.reason);
});

window.addEventListener("error", (event) => {
  logger.error("❌ Global error:", event.error || event.message);
});

logger.log("🚀 Main.js starting...");

// Initialize loading screen immediately (before any asset loading)
const loadingScreen = new LoadingScreen();
logger.log("✅ Loading screen created");

// Register loading tasks (scene assets and audio files will register themselves as they load)
loadingScreen.registerTask("initialization", 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.01,
  200
);
camera.position.set(0, 5, 0);
scene.add(camera); // Add camera to scene so its children render

const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap; // Soft shadows
renderer.toneMapping = THREE.CineonToneMapping; // Better HDR tone mapping for bloom effects
renderer.toneMappingExposure = 1.0; // Adjust exposure for bloom
renderer.outputColorSpace = THREE.SRGBColorSpace; // Proper color space
renderer.domElement.style.opacity = "0"; // Hide renderer until loading is complete
document.body.appendChild(renderer.domElement);

// Create a SparkRenderer with depth of field effect
const apertureSize = 0.01; // Very small aperture for subtle DoF
const focalDistance = 6.0;
const apertureAngle = 2 * Math.atan((0.5 * apertureSize) / focalDistance);

logger.log("Creating SparkRenderer...");
const spark = new SparkRenderer({
  renderer,
  apertureAngle: apertureAngle,
  focalDistance: focalDistance,
});
spark.renderOrder = 9998;
scene.add(spark);
logger.log("✅ SparkRenderer created");

// Initialize game manager early to check for debug spawn
const gameManager = new GameManager();

// Expose for debug console access
window.gameManager = gameManager;

// Initialize scene manager (objects will be loaded by gameManager based on state)
// Pass loadingScreen for progress tracking, renderer for contact shadows, gameManager for state updates
const sceneManager = new SceneManager(scene, {
  loadingScreen,
  renderer,
  sparkRenderer: spark,
  gameManager,
});

// Make scene manager globally accessible for mesh lookups
window.sceneManager = sceneManager;

// Expose environment map utilities for console debugging
window.getEnvMapWorldCenters = getEnvMapWorldCenters;
window.captureEnvMapAtScene = captureEnvMapAtScene;

// Note: gizmoManager will be passed to sceneManager after initialization

// Connect loading screen to game manager so it can transition state when done
loadingScreen.setGameManager(gameManager);

// Initialize the physics manager
logger.log("Creating PhysicsManager...");
const physicsManager = new PhysicsManager();
logger.log("✅ PhysicsManager created");

// Initialize light manager BEFORE fog so splat lights can affect fog particles
// Pass sceneManager so lights can be parented under specific scene objects
// Pass gameManager so lights can check criteria before loading
logger.log("Creating LightManager...");
const lightManager = new LightManager(scene, sceneManager, gameManager);
logger.log("✅ LightManager created");

// Initialize VFX system manager AFTER light manager (so splat lights can affect fog)
// Pass loadingScreen for progress tracking
const vfxManager = new VFXSystemManager(scene, camera, renderer, loadingScreen);
try {
  await vfxManager.initialize(sceneManager); // Pass sceneManager for splatMorph effect
  logger.log("✅ VFX Manager initialized");
} catch (error) {
  logger.error("❌ Failed to initialize VFX Manager:", error);
  throw error;
}

// Make VFX manager globally accessible
window.vfxManager = vfxManager;

// Initialize Drawing Recognition Manager
logger.log("Creating DrawingRecognitionManager...");
const drawingRecognitionManager = new DrawingRecognitionManager(gameManager);
try {
  await drawingRecognitionManager.initialize();
  logger.log("✅ DrawingRecognitionManager initialized");
} catch (error) {
  logger.error("❌ Failed to initialize DrawingRecognitionManager:", error);
}

// Initialize Drawing Manager
logger.log("Creating DrawingManager...");
const drawingManager = new DrawingManager(
  scene,
  drawingRecognitionManager,
  gameManager
);
logger.log("✅ DrawingManager created");

// Make drawing managers globally accessible for debugging
window.drawingRecognitionManager = drawingRecognitionManager;
window.drawingManager = drawingManager;

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  vfxManager.setSize(window.innerWidth, window.innerHeight);
});

// Use debug spawn position if available, otherwise default

const defaultSpawnPos = {
  x: 0,
  y: 0.9,
  z: 0,
};

const defaultSpawnRot = {
  x: 0,
  y: 180,
  z: 0,
};

const spawnPos = gameManager.getDebugSpawnPosition() || defaultSpawnPos;
const character = physicsManager.createCharacter(spawnPos, defaultSpawnRot);
// Removed visual mesh for character;

// Sync camera to character spawn position
camera.position.set(
  spawnPos.x,
  spawnPos.y + 0.8, // Character Y + camera height offset
  spawnPos.z
);

// Initialize SFX manager (pass lightManager for audio-reactive lights)
const sfxManager = new SFXManager({
  masterVolume: 0.5,
  lightManager: lightManager,
  loadingScreen: loadingScreen,
});

// Initialize input manager (handles keyboard, mouse, and gamepad)
// Note: Pass gameManager so inputManager can check game state before allowing pointer lock
const inputManager = new InputManager(renderer.domElement, gameManager);

// Disable input initially - will be enabled when game starts
inputManager.disable();

// Pass input manager reference to drawing manager
drawingManager.setInputManager(inputManager);

// Initialize character controller (will be disabled until intro completes)
const characterController = new CharacterController(
  character,
  camera,
  renderer,
  inputManager,
  sfxManager,
  spark, // Pass spark renderer for DoF control
  null, // idleHelper (set later)
  defaultSpawnRot // Initial rotation from spawn data
);

// Register SFX from data
import { sfxSounds } from "./sfxData.js";
sfxManager._data = sfxSounds; // Keep a reference to definitions for state-based autoplay/stop
sfxManager.registerSoundsFromData(sfxSounds);

// Make character controller and input manager globally accessible for options menu
window.characterController = characterController;
window.inputManager = inputManager;
window.camera = camera;

// Initialize animation manager now that all dependencies exist
const cameraAnimationManager = new AnimationManager(
  camera,
  characterController,
  gameManager,
  {
    loadingScreen: loadingScreen,
    physicsManager: physicsManager,
    sceneManager: sceneManager,
  }
);

// Load camera animations from data
cameraAnimationManager.loadAnimationsFromData(cameraAnimations);
// Object animations are loaded directly in animationManager.js

// Make it globally accessible for debugging/scripting
window.cameraAnimationManager = cameraAnimationManager;

// Initialize lighting system
//const lightingSystem = new LightingSystem(scene);

// Initialize UI manager (manages all UI elements and z-index)
const uiManager = new UIManager(gameManager);

// Initialize music manager (automatically loads tracks from musicData)
const musicManager = new MusicManager({
  defaultVolume: 0.6,
  loadingScreen: loadingScreen,
});

// Initialize dialog choice UI
const dialogChoiceUI = new DialogChoiceUI({
  gameManager: gameManager,
  sfxManager: sfxManager,
  inputManager: inputManager,
});

// Initialize dialog manager with HTML captions
const dialogManager = new DialogManager({
  audioVolume: 1.0,
  useSplats: false, // Use HTML instead of text splats
  sfxManager: sfxManager, // Link to SFX manager for volume control
  gameManager: gameManager, // Link to game manager for state updates
  dialogChoiceUI: dialogChoiceUI, // Link to dialog choice UI
  loadingScreen: loadingScreen, // For progress tracking
});

// Set manager references for loading screen (for deferred asset loading)
loadingScreen.setManagers({
  renderer: renderer,
  musicManager: musicManager,
  sfxManager: sfxManager,
  dialogManager: dialogManager,
  cameraAnimationManager: cameraAnimationManager,
});

// Initialize start screen - will be created when state transitions to START_SCREEN
let startScreen = null;

// One-time listener to initialize StartScreen
const initStartScreen = (newState, oldState) => {
  if (newState.currentState === GAME_STATES.START_SCREEN && !startScreen) {
    logger.log("Creating StartScreen");
    // Calculate camera target position based on actual character spawn
    const cameraTargetPos = new THREE.Vector3(
      spawnPos.x,
      spawnPos.y + characterController.cameraHeight, // Character Y + camera offset
      spawnPos.z
    );

    startScreen = new StartScreen(camera, scene, {
      targetPosition: cameraTargetPos,
      targetRotation: {
        yaw: THREE.MathUtils.degToRad(defaultSpawnRot.y),
        pitch: 0,
      },
      transitionDuration: 8.0,
      uiManager: uiManager,
      sceneManager: sceneManager,
      sfxManager: sfxManager,
      dialogManager: dialogManager,
      inputManager: inputManager,
      glbAnimationStartProgress: 0.55, // Optional: Start at 50% through the GLB animation (0-1)
    });

    // Remove this listener after it runs once
    gameManager.off("state:changed", initStartScreen);
  }
};

gameManager.on("state:changed", initStartScreen);

// Initialize options menu
const optionsMenu = new OptionsMenu({
  musicManager: musicManager,
  sfxManager: sfxManager,
  gameManager: gameManager,
  uiManager: uiManager,
  sparkRenderer: spark,
  characterController: characterController,
  startScreen: startScreen,
});

// Connect physics manager to scene manager BEFORE initializing gameManager
// This ensures physics colliders can be created when scene objects load
sceneManager.physicsManager = physicsManager;

// Preload dialog audio files
import { dialogTracks } from "./dialogData.js";
dialogManager.preloadDialogs(dialogTracks);

// Link dialog manager to choice UI
dialogChoiceUI.dialogManager = dialogManager;

// Link dialog manager to start screen for timing
if (startScreen) {
  startScreen.dialogManager = dialogManager;
}

// Register dialog manager with SFX manager
sfxManager.registerDialogManager(dialogManager);

// Register dialog volume control with SFX manager
if (sfxManager && dialogManager) {
  // Create a proxy object that implements the setVolume interface
  const dialogVolumeControl = {
    setVolume: (volume) => {
      dialogManager.setVolume(volume);
    },
  };
  // Boost dialog base volume so it is louder relative to SFX master
  sfxManager.registerSound("dialog", dialogVolumeControl, 2.0);
}

// Initialize gameManager with all managers (async - loads initial scene objects)
try {
  logger.log("Starting gameManager initialization...");
  await gameManager.initialize({
    dialogManager: dialogManager,
    musicManager: musicManager,
    sfxManager: sfxManager,
    uiManager: uiManager,
    characterController: characterController,
    cameraAnimationManager: cameraAnimationManager,
    sceneManager: sceneManager,
    lightManager: lightManager,
    physicsManager: physicsManager,
    inputManager: inputManager,
    scene: scene,
    camera: camera,
    renderer: renderer,
  });
  logger.log("✅ GameManager initialized");
  loadingScreen.completeTask("initialization");
} catch (error) {
  logger.error("❌ Failed to initialize GameManager:", error);
  throw error;
}

// Set up event listeners for managers
characterController.setGameManager(gameManager);
characterController.setSceneManager(sceneManager); // For first-person body attachment
characterController.setPhysicsManager(physicsManager); // For floor height raycasts

// Initialize camera animation manager AFTER CharacterController has registered its listeners
cameraAnimationManager.initialize();

musicManager.setGameManager(gameManager);
sfxManager.setGameManager(gameManager);
vfxManager.setGameManager(gameManager);

// Apply caption settings now that dialogManager is available
optionsMenu.applyCaptions();

// Set up light manager state change listener to dynamically load/unload lights based on criteria
gameManager.on("state:changed", (newState, oldState) => {
  lightManager.updateLightsForState(newState);
});

// Initialize UI components (idleHelper, fullscreenButton)
uiManager.initializeComponents({
  dialogManager,
  cameraAnimationManager,
  dialogChoiceUI,
  inputManager,
  characterController,
  sparkRenderer: spark,
});

// Initialize gizmo manager for debug positioning
const gizmoManager = new GizmoManager(
  scene,
  camera,
  renderer,
  sceneManager,
  characterController
);

// Initialize collider manager with scene and sceneManager references
const colliderManager = new ColliderManager(
  physicsManager,
  gameManager,
  colliders,
  scene,
  sceneManager,
  gizmoManager
);

// Make collider manager globally accessible for debugging
window.colliderManager = colliderManager;

// Pass gizmo manager to scene manager (physics manager already set earlier)
sceneManager.gizmoManager = gizmoManager;
if (gameManager.videoManager) {
  gameManager.videoManager.gizmoManager = gizmoManager;
}

// Register any already-loaded scene objects with gizmo manager
gizmoManager.registerSceneObjects(sceneManager);

// Register lights with gizmo manager
gizmoManager.registerLights(lightManager, lights);

// Make gizmo manager globally accessible for debugging
window.gizmoManager = gizmoManager;
// Make game manager globally accessible for gizmoManager setState integration
window.gameManager = gameManager;

// Standardize global effects via managers (sceneManager/videoManager set game state)

// Force gizmo detection from definitions regardless of state
gizmoManager.applyGlobalBlocksFromDefinitions({
  sceneDefs: sceneObjects,
  videoDefs: videos,
  colliderDefs: colliders,
  lightDefs: lights,
});

// Standardize: let gizmo manager own global side-effects from now on
// IMPORTANT: Set integration BEFORE applyGlobalBlocksFromDefinitions so inputManager is available
gizmoManager.setIntegration(uiManager?.components?.idleHelper, inputManager);

// Allow InputManager to detect gizmo hover/drag to enable drag-to-look when not over gizmo
if (typeof inputManager.setGizmoProbe === "function") {
  inputManager.setGizmoProbe(() => gizmoManager.isPointerOverGizmo());
}

let lastTime;
renderer.setAnimationLoop(function animate(time) {
  const t = time * 0.001;
  const dt = Math.min(0.033, t - (lastTime ?? t));
  lastTime = t;

  // Update start screen (camera animation and transition)
  if (startScreen && startScreen.isActive) {
    startScreen.update(dt);
    startScreen.checkIntroStart(sfxManager, gameManager);
  }

  // Don't update most game logic if options menu is open or start screen is active
  if (!optionsMenu.isOpen && (!startScreen || !startScreen.isActive)) {
    // Update input manager (gamepad state)
    inputManager.update(dt);

    // Update camera animation manager
    cameraAnimationManager.update(dt);

    // Update character controller (handles input, physics, camera, headbob)
    if (gameManager.isControlEnabled() && !cameraAnimationManager.playing) {
      characterController.update(dt);
    }

    // Physics step
    physicsManager.step();

    // Update collider manager (check for trigger intersections)
    if (gameManager.isControlEnabled()) {
      colliderManager.update(character);
    }

    // (moved below) Video manager is updated unconditionally so videos render during START_SCREEN too

    // Update Howler listener position for spatial audio
    Howler.pos(camera.position.x, camera.position.y, camera.position.z);

    // Update Howler listener orientation (forward and up vectors)
    const cameraDirection = new THREE.Vector3();
    camera.getWorldDirection(cameraDirection);
    Howler.orientation(
      cameraDirection.x,
      cameraDirection.y,
      cameraDirection.z,
      camera.up.x,
      camera.up.y,
      camera.up.z
    );
  }

  // Always update video manager (billboarding and texture updates need to run during START_SCREEN)
  if (gameManager.videoManager) {
    gameManager.videoManager.update(dt);
  }

  // Update title sequence (pass dt in seconds)
  const titleSequence = startScreen ? startScreen.getTitleSequence() : null;
  if (titleSequence) {
    titleSequence.update(dt);

    // Enable character controller when the title outro begins
    if (!gameManager.isControlEnabled() && titleSequence.hasOutroStarted()) {
      gameManager.setState({ controlEnabled: true });
    }
  }

  // Always update music manager (handles fades)
  musicManager.update(dt);

  // Always update SFX manager (handles delayed sound playback)
  sfxManager.update(dt);

  // Always update dialog manager (handles caption timing)
  dialogManager.update(dt);

  // Always update dialog choice UI (handles gamepad navigation)
  dialogChoiceUI.update(dt);

  // Always update scene manager (handles GLTF animations)
  sceneManager.update(dt);

  // Update scene animations based on game state
  sceneManager.updateAnimationsForState(gameManager.state);

  // Update contact shadows (must be called before rendering)
  sceneManager.updateContactShadows(dt);

  // Always update game manager (handles receiver lerp, etc.)
  gameManager.update(dt);

  // Always update audio-reactive lights
  lightManager.updateReactiveLights(dt);

  // Update lens flare fade effects
  lightManager.updateLensFlares(dt);

  // Update all VFX effects (fog, desaturation, etc.)
  vfxManager.update(dt);

  // Update drawing recognition manager
  if (drawingRecognitionManager) {
    drawingRecognitionManager.update(dt);
  }

  // Render with VFX post-processing effects
  vfxManager.render(scene, camera);

  // Render text splats on top (separate scene for title sequence)
  if (startScreen && startScreen.getTextRenderInfo) {
    const textInfo = startScreen.getTextRenderInfo();
    if (textInfo && textInfo.scene && textInfo.camera) {
      renderer.autoClear = false;
      renderer.render(textInfo.scene, textInfo.camera);
      renderer.autoClear = true;
    }
  }
});
