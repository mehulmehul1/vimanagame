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
import ZoneManager from "./zoneManager.js";
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
import { TimePassesSequence } from "./ui/timePassesSequence.js";
import { GAME_STATES } from "./gameData.js";
import AnimationManager from "./animationManager.js";
import cameraAnimations from "./animationCameraData.js";
import objectAnimations from "./animationObjectData.js";
import GizmoManager from "./utils/gizmoManager.js";
import { VFXSystemManager } from "./vfxManager.js";
import ViewmasterController from "./content/viewmasterController.js";
import { LoadingScreen } from "./ui/loadingScreen.js";
import { Logger } from "./utils/logger.js";
import { DrawingRecognitionManager } from "./drawing/drawingRecognitionManager.js";
import { DrawingManager } from "./drawing/drawingManager.js";
import { RuneManager } from "./content/runeManager.js";
import { detectPlatform } from "./utils/platformDetection.js";
import { unlockAllAudioContexts } from "./vfx/proceduralAudio.js";
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
window.loadingScreen = loadingScreen; // Make accessible globally for lazy initialization
logger.log("✅ Loading screen created");

// Register loading tasks (scene assets and audio files will register themselves as they load)
loadingScreen.registerTask("initialization", 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.01,
  100
);
camera.position.set(0, 5, 0);
scene.add(camera); // Add camera to scene so its children render

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap; // Soft shadows
renderer.toneMapping = THREE.CineonToneMapping; // Better HDR tone mapping
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
  maxStdDev: Math.sqrt(5),
  minAlpha: 0.8 * (1.0 / 255.0),
});
spark.renderOrder = 9998;
scene.add(spark);
logger.log("✅ SparkRenderer created");

// Initialize game manager early to check for debug spawn
const gameManager = new GameManager();

// Expose for debug console access
window.gameManager = gameManager;

// Detect platform capabilities early so other systems can use the state
const platformInfo = detectPlatform(gameManager);
logger.log(
  `✅ Platform detection complete - Mobile: ${platformInfo?.isMobile}, iOS: ${platformInfo?.isIOS}, Fullscreen supported: ${platformInfo?.isFullscreenSupported}`
);

// Initialize options menu early to set performance profile before scene loading
// This must happen before sceneManager initialization so the correct assets load
const optionsMenu = new OptionsMenu({
  gameManager: gameManager,
});

// Set performance profile based on priority:
// 1. URL parameter (highest priority - for sharing links)
// 2. Saved localStorage value
// 3. Auto-detection based on platform (lowest priority)

// Check URL parameter first
const urlProfile = gameManager.getURLParam("performanceProfile");
let profileToUse = null;
let profileSource = null;

if (urlProfile && ["mobile", "laptop", "desktop", "max"].includes(urlProfile)) {
  // URL parameter takes highest priority
  profileToUse = urlProfile;
  profileSource = "URL parameter";
  // Save URL parameter to localStorage so it persists
  optionsMenu.settings.performanceProfile = urlProfile;
  optionsMenu.saveSettings();
} else {
  // Check localStorage for saved setting
  const savedSettings = optionsMenu.loadSettings();
  const savedProfile = savedSettings?.performanceProfile;

  if (
    savedProfile &&
    ["mobile", "laptop", "desktop", "max"].includes(savedProfile)
  ) {
    // Use saved profile from localStorage
    profileToUse = savedProfile;
    profileSource = "saved settings";
  } else {
    // No saved profile - auto-detect based on platform
    if (platformInfo?.isMobile) {
      profileToUse = "mobile";
      profileSource = "platform detection";
    } else {
      profileToUse = "laptop";
      profileSource = "default";
    }
  }
}

// Apply the determined profile
optionsMenu.setPerformanceProfile(profileToUse);
logger.log(`✅ Performance profile set to ${profileToUse} (${profileSource})`);

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

// Initialize Rune Manager
logger.log("Creating RuneManager...");
const runeManager = new RuneManager(scene, gameManager);
logger.log("✅ RuneManager created");

// Make rune manager globally accessible
window.runeManager = runeManager;

// Initialize Drawing Recognition Manager (lazy initialization - will load TensorFlow when needed)
logger.log("Creating DrawingRecognitionManager...");
const drawingRecognitionManager = new DrawingRecognitionManager(gameManager);
// Note: initialize() will be called lazily when drawing game starts (LIGHTS_OUT/CURSOR states)
// This defers loading ~15MB of TensorFlow.js assets until after loading screen

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

window.captureDrawing = (width, height, filename) => {
  return drawingRecognitionManager.captureDrawing(width, height, filename);
};

window.captureStrokeData = () => {
  return drawingRecognitionManager.captureStrokeData();
};

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  vfxManager.setSize(window.innerWidth, window.innerHeight);

  // Update text camera aspect ratios
  if (startScreen && startScreen.textCamera) {
    startScreen.textCamera.aspect = window.innerWidth / window.innerHeight;
    startScreen.textCamera.updateProjectionMatrix();
  }
  if (timePassesSequence && timePassesSequence.textCamera) {
    timePassesSequence.textCamera.aspect =
      window.innerWidth / window.innerHeight;
    timePassesSequence.textCamera.updateProjectionMatrix();
  }
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
const spawnRot = gameManager.getDebugSpawnRotation() || defaultSpawnRot;
const character = physicsManager.createCharacter(spawnPos, spawnRot);
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
  spawnRot // Initial rotation from spawn data
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

const viewmasterController = new ViewmasterController({
  gameManager,
  animationManager: cameraAnimationManager,
  sceneManager,
  vfxManager,
});
viewmasterController.initialize();
window.viewmasterController = viewmasterController;

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
// Note: videoManager is created inside gameManager.initialize(), so we need to set it after
loadingScreen.setManagers({
  renderer: renderer,
  musicManager: musicManager,
  sfxManager: sfxManager,
  dialogManager: dialogManager,
  cameraAnimationManager: cameraAnimationManager,
  videoManager: null, // Will be set after gameManager initialization
});

// Initialize start screen - will be created when state transitions to START_SCREEN
let startScreen = null;

// Initialize time passes sequence - will be created when state transitions to LIGHTS_OUT
let timePassesSequence = null;

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

// One-time listener to initialize TimePassesSequence
const initTimePassesSequence = (newState, oldState) => {
  if (newState.currentState === GAME_STATES.LIGHTS_OUT && !timePassesSequence) {
    logger.log("Creating TimePassesSequence");
    timePassesSequence = new TimePassesSequence(camera, {
      uiManager: uiManager,
      gameManager: gameManager,
    });
    timePassesSequence.start();
    // Remove this listener after it runs once
    gameManager.off("state:changed", initTimePassesSequence);
  }
};

gameManager.on("state:changed", initTimePassesSequence);

// Options menu was already initialized early (before scene loading) to set performance profile
// Now update it with all the manager references
optionsMenu.musicManager = musicManager;
optionsMenu.sfxManager = sfxManager;
optionsMenu.sparkRenderer = spark;
optionsMenu.characterController = characterController;
optionsMenu.startScreen = startScreen;

// Register options menu with UIManager (must be done after UIManager is created)
optionsMenu.registerWithUIManager(uiManager);

// Connect physics manager to scene manager BEFORE initializing gameManager
// This ensures physics colliders can be created when scene objects load
sceneManager.physicsManager = physicsManager;

// Preload dialog audio files
import { dialogTracks, VIEWMASTER_OVERHEAT_THRESHOLD } from "./dialogData.js";
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

// Pass videoManager to dialogManager for video-synced captions
if (gameManager.videoManager && dialogManager) {
  dialogManager.videoManager = gameManager.videoManager;
}

// Pass loadingScreen to videoManager and update loadingScreen with videoManager reference
if (gameManager.videoManager) {
  gameManager.videoManager.loadingScreen = loadingScreen;
  loadingScreen.setManagers({
    renderer: renderer,
    musicManager: musicManager,
    sfxManager: sfxManager,
    dialogManager: dialogManager,
    cameraAnimationManager: cameraAnimationManager,
    videoManager: gameManager.videoManager,
  });
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

// Set camera on collider manager for camera-based zone detection during START_SCREEN and animations
colliderManager.setCamera(camera);

// Set SparkRenderer on collider manager for updating accumulator origin position when triggers fire
colliderManager.setSparkRenderer(spark);

// Initialize SparkRenderer position to character position
// This prevents float16 quantization artifacts at startup
const charPos = character.translation();
spark.position.set(charPos.x, charPos.y, charPos.z);

// Set collider manager reference on scene manager (for trigger colliders from GLTF meshes)
sceneManager.setColliderManager(colliderManager);

// Initialize zone manager for exterior splat loading/unloading
const zoneManager = new ZoneManager(gameManager, sceneManager);
window.zoneManager = zoneManager; // Make globally accessible for debugging

// Set SparkRenderer on zone manager for updating accumulator origin position when zones change
zoneManager.setSparkRenderer(spark);

// Make zoneManager accessible on gameManager for ColliderManager
if (gameManager) {
  gameManager.zoneManager = zoneManager;
  gameManager.colliderManager = colliderManager; // Set colliderManager reference
}

logger.log("✅ ZoneManager initialized");

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

// Global user interaction handler to unlock all audio contexts (Safari autoplay policy)
// This ensures procedural audio works reliably after any user interaction
let audioUnlocked = false;
const unlockAudioOnInteraction = () => {
  if (!audioUnlocked) {
    unlockAllAudioContexts();
    // Also unlock Howler for SFX/Music
    if (typeof Howler !== "undefined" && typeof Howler.unlock === "function") {
      Howler.unlock();
    }
    audioUnlocked = true;
    logger.log("✅ Audio contexts unlocked via user interaction");
    // Keep listener active in case new audio contexts are created
  } else {
    // Still unlock on subsequent interactions (new contexts might be created)
    unlockAllAudioContexts();
  }
};

// Listen for any user interaction to unlock audio
window.addEventListener("click", unlockAudioOnInteraction, { once: false });
window.addEventListener("touchstart", unlockAudioOnInteraction, {
  once: false,
});
window.addEventListener("keydown", unlockAudioOnInteraction, { once: false });
window.addEventListener("mousedown", unlockAudioOnInteraction, { once: false });

// IMPORTANT: Set integration BEFORE applyGlobalBlocksFromDefinitions so inputManager is available
gizmoManager.setIntegration(uiManager?.components?.idleHelper, inputManager);

// Allow InputManager to detect gizmo hover/drag to enable drag-to-look when not over gizmo
if (typeof inputManager.setGizmoProbe === "function") {
  inputManager.setGizmoProbe(() => gizmoManager.isPointerOverGizmo());
}

let lastTime;
renderer.setAnimationLoop(function animate(time) {
  const t = time * 0.001;
  // Cap dt at 0.1s (10fps) to prevent spiral of death on very slow frames
  // But allow larger values than 0.033s so animations complete in real time on slower machines
  const dt = Math.min(0.1, t - (lastTime ?? t));
  lastTime = t;

  // Update start screen (camera animation and transition)
  if (startScreen && startScreen.isActive) {
    startScreen.update(dt);
    startScreen.checkIntroStart(sfxManager, gameManager);
  }

  // Determine if we should use camera-based zone detection
  // Use camera during START_SCREEN camera animation (after animation loads) and transition to starting position
  // Also use camera when camera animation manager is playing (but not during normal gameplay)
  const currentState = gameManager.getState();
  const isStartScreenActive =
    startScreen && startScreen.isActive && !startScreen.isLoadingAnimation; // Only after animation has loaded and camera is positioned
  const isStartScreenTransition =
    startScreen && startScreen.isFollowingUnifiedPath;
  const shouldUseCamera =
    isStartScreenActive ||
    isStartScreenTransition ||
    (cameraAnimationManager.isPlaying && !gameManager.isControlEnabled());

  // Update camera probe position if using camera (must happen before physics step)
  if (
    shouldUseCamera &&
    colliderManager.camera &&
    colliderManager.cameraProbeBody
  ) {
    colliderManager.cameraProbeBody.setTranslation(
      {
        x: colliderManager.camera.position.x,
        y: colliderManager.camera.position.y,
        z: colliderManager.camera.position.z,
      },
      true // wake up the body
    );
  }

  // Physics step - needed for zone detection even during START_SCREEN
  // Must happen AFTER camera probe position update so collision detection uses current position
  // Pass dt for fixed timestep physics (ensures consistent movement speed regardless of framerate)
  if (!optionsMenu.isOpen) {
    physicsManager.step(dt);
  }

  // Update collider manager for collision detection (always update, even during START_SCREEN)
  // Zone colliders use camera probe when shouldUseCamera=true
  // Regular trigger colliders always use character body (if available)
  // NOTE: This checks intersections AFTER physics step so collision detection uses updated positions
  if (shouldUseCamera) {
    // During camera transitions: check zones with camera, regular triggers with character (if available)
    colliderManager.update(characterController.character || null, true);
  } else if (characterController.character) {
    // Normal gameplay: use character body for all colliders (always check if body exists)
    colliderManager.update(characterController.character, false);
  }

  // Don't update most game logic if options menu is open or start screen is active
  if (!optionsMenu.isOpen && (!startScreen || !startScreen.isActive)) {
    // Update input manager (gamepad state)
    inputManager.update(dt);

    // Check if we need to update character controller before animation manager (for blending)
    const isBlendingAnimation =
      cameraAnimationManager.isPlaying &&
      cameraAnimationManager.blendWithPlayer;

    // If blending, update character controller first so animation manager can blend with latest player input
    if (isBlendingAnimation && gameManager.isControlEnabled()) {
      characterController.update(dt);
    }

    // Update camera animation manager
    cameraAnimationManager.update(dt);

    // Update character controller (handles input, physics, camera, headbob)
    // Skip if we already updated it above for blending
    if (
      gameManager.isControlEnabled() &&
      !isBlendingAnimation &&
      !cameraAnimationManager.playing
    ) {
      characterController.update(dt);
    }
  }

  // (moved below) Video manager is updated unconditionally so videos render during START_SCREEN too

  // Update Howler listener position for spatial audio (only if not in blocked game logic section)
  if (!optionsMenu.isOpen && (!startScreen || !startScreen.isActive)) {
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
  if (titleSequence && !titleSequence.isComplete()) {
    titleSequence.update(dt);

    // Enable character controller when the title outro begins
    if (!gameManager.isControlEnabled() && titleSequence.hasOutroStarted()) {
      gameManager.setState({ controlEnabled: true });
    }
  }

  // Update time passes sequence
  if (timePassesSequence) {
    timePassesSequence.update(dt);
  }

  // Always update music manager (handles fades)
  musicManager.update(dt);

  // Always update SFX manager (handles delayed sound playback)
  sfxManager.update(dt);

  // Update viewmaster insanity intensity in game state (for dialog criteria)
  if (characterController && gameManager) {
    const intensity = characterController.getViewmasterInsanityIntensity();
    const currentState = gameManager.getState();
    const previousIntensity = currentState.viewmasterInsanityIntensity || 0;

    // Detect when intensity crosses threshold (from below to above)
    if (
      previousIntensity < VIEWMASTER_OVERHEAT_THRESHOLD &&
      intensity >= VIEWMASTER_OVERHEAT_THRESHOLD
    ) {
      const currentOverheatCount = currentState.viewmasterOverheatCount || 0;
      const newCount = currentOverheatCount + 1;
      gameManager.setState({
        viewmasterInsanityIntensity: intensity,
        viewmasterOverheatCount: newCount,
        viewmasterOverheatDialogIndex: newCount % 2, // Cycle: 0, 1, 0, 1...
      });
    } else if (intensity !== previousIntensity) {
      // Only update if intensity actually changed
      gameManager.setState({ viewmasterInsanityIntensity: intensity });
    }
  }

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

  // Update drawing manager
  if (drawingManager) {
    drawingManager.update(dt);
  }

  if (viewmasterController) {
    viewmasterController.update(dt);
  }

  // Update rune manager
  if (runeManager) {
    runeManager.update(dt);
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

  // Render time passes text sequence
  if (timePassesSequence) {
    const textInfo = timePassesSequence.getTextRenderInfo();
    if (textInfo && textInfo.scene && textInfo.camera) {
      renderer.autoClear = false;
      renderer.render(textInfo.scene, textInfo.camera);
      renderer.autoClear = true;
    }
  }
});
