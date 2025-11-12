import { GAME_STATES, startScreen } from "../gameData.js";
import { sceneObjects } from "../sceneData.js";
import { Logger } from "./logger.js";

const logger = new Logger("DebugSpawner", false);

/**
 * DebugSpawner - Debug utility for spawning into specific game states
 *
 * Usage:
 * - Add ?gameState=<STATE_NAME> to URL (e.g., ?gameState=DRIVE_BY)
 * - All managers will initialize with correct state (music, SFX, dialogs, scenes)
 * - Any state name from GAME_STATES in gameData.js is automatically supported
 * - Custom overrides can be defined in stateOverrides for specific positioning/settings
 */

/**
 * Custom overrides for specific states that need non-default settings
 */
const stateOverrides = {
  START_SCREEN: {
    controlEnabled: false,
    // No playerPosition - use main.js default
  },

  TITLE_SEQUENCE: {
    controlEnabled: false,
    // No playerPosition - use main.js default
  },

  TITLE_SEQUENCE_COMPLETE: {
    controlEnabled: true,
    // No playerPosition - use main.js default
    playerRotation: { x: 0.0, y: 180, z: 0.0 },
  },

  INTRO_COMPLETE: {
    isPlaying: true,
    controlEnabled: true,
    // No playerPosition - use main.js default
  },

  NEAR_RADIO: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: 0,
      y: 0.8,
      z: 28,
    },
  },

  PHONE_BOOTH_RINGING: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: sceneObjects.phonebooth.position.x + 5,
      y: 0.8,
      z: sceneObjects.phonebooth.position.z - 8,
    },
  },

  ANSWERED_PHONE: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: sceneObjects.phonebooth.position.x,
      y: 0.8,
      z: sceneObjects.phonebooth.position.z - 0.15,
    },
  },

  DIALOG_CHOICE_1: {
    isPlaying: true,
    controlEnabled: true,
  },

  DRIVE_BY_PREAMBLE: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: sceneObjects.phonebooth.position.x,
      y: 1.9,
      z: sceneObjects.phonebooth.position.z - 0.15,
    },
  },

  DRIVE_BY: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: sceneObjects.phonebooth.position.x,
      y: 0.8,
      z: sceneObjects.phonebooth.position.z,
    },
  },

  POST_DRIVE_BY: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: 11.68,
      y: 0.8,
      z: 64.35,
    },
  },

  DOORS_CLOSE: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: 1.94,
      y: 2.13,
      z: 79.85,
    },
    playerRotation: { x: 0.0, y: 110, z: 0.0 }, // Rotation in DEGREES (converted to radians internally)
  },

  ENTERING_OFFICE: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: 1.94,
      y: 2.13,
      z: 79.85,
    },
    playerRotation: { x: 0.0, y: 110, z: 0.0 }, // Rotation in DEGREES (converted to radians internally)
  },

  OFFICE_INTERIOR: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: -2.69,
      y: 1.5,
      z: 84.05,
    },
  },

  OFFICE_PHONE_ANSWERED: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: {
      x: -4.69,
      y: 1.5,
      z: 83.05,
    },
  },

  PRE_VIEWMASTER: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },

  VIEWMASTER: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },

  VIEWMASTER_COLOR: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },
  VIEWMASTER_DISSOLVE: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },

  VIEWMASTER_HELL: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },

  POST_VIEWMASTER: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },

  PRE_EDISON: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },

  EDISON: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },

  DIALOG_CHOICE_2: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },

  CZAR_STRUGGLE: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 84.66 },
  },

  SHOULDER_TAP: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
  },

  LIGHTS_OUT: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
  },

  WAKING_UP: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
    playerRotation: { x: 0.0, y: 0, z: 0.0 }, // Rotation in DEGREES (converted to radians internally)
  },

  CURSOR: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 }, // Inside club scene
  },

  CURSOR_FINAL: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 }, // Inside club scene
  },

  SHADOW_AMPLIFICATIONS: {
    isPlaying: true,
    controlEnabled: true,
    isViewmasterEquipped: true, // Viewmaster should be on in this state
    viewmasterManuallyRemoved: false,
    viewmasterOverheatDialogIndex: null,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
  },

  POST_CURSOR: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
  },

  OUTRO: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
  },

  OUTRO_LECLAIRE: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
  },

  OUTRO_CZAR: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
  },

  OUTRO_CAT: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
  },

  GAME_OVER: {
    isPlaying: true,
    controlEnabled: true,
    playerPosition: { x: -5.14, y: 2.05, z: 83.66 },
    playerRotation: { x: 0.0, y: 0, z: 0.0 }, // Rotation in DEGREES (converted to radians internally)
  },
};

/**
 * Generate a default preset for any game state
 * @param {number} stateValue - The GAME_STATES value
 * @returns {Object} State preset
 */
function createDefaultPreset(stateValue) {
  return {
    ...startScreen, // Includes playerPosition: {x:0, y:0, z:0} from startScreen
    currentState: stateValue,
    controlEnabled: true,
    // playerPosition can be overridden to undefined to use main.js default
  };
}

/**
 * Get state preset - dynamically supports all GAME_STATES
 * @param {string} stateName - Name of the state (e.g., "DRIVE_BY")
 * @returns {Object} State preset
 */
function getStatePreset(stateName) {
  // Check if this is a valid GAME_STATE
  if (!(stateName in GAME_STATES)) {
    return null;
  }

  const stateValue = GAME_STATES[stateName];
  const defaultPreset = createDefaultPreset(stateValue);
  const overrides = stateOverrides[stateName] || {};

  return {
    ...defaultPreset,
    ...overrides,
  };
}

/**
 * Debug state presets - dynamically generated for all GAME_STATES
 */
export const debugStatePresets = new Proxy(
  {},
  {
    get(target, prop) {
      if (typeof prop === "string" && prop in GAME_STATES) {
        return getStatePreset(prop);
      }
      return undefined;
    },
    has(target, prop) {
      return typeof prop === "string" && prop in GAME_STATES;
    },
    ownKeys() {
      return Object.keys(GAME_STATES);
    },
    getOwnPropertyDescriptor(target, prop) {
      if (typeof prop === "string" && prop in GAME_STATES) {
        return {
          enumerable: true,
          configurable: true,
        };
      }
      return undefined;
    },
  }
);

/**
 * Parse URL and get debug state preset
 * @returns {Object|null} State preset if debug spawn is requested, null otherwise
 */
export function getDebugSpawnState() {
  const urlParams = new URLSearchParams(window.location.search);
  const gameStateParam = urlParams.get("gameState");

  if (!gameStateParam) {
    return null;
  }

  // Try to find matching preset
  const preset = debugStatePresets[gameStateParam];

  if (!preset) {
    logger.warn(
      `Unknown gameState "${gameStateParam}". Available states:`,
      Object.keys(debugStatePresets)
    );
    return null;
  }

  logger.log(`Spawning into state "${gameStateParam}"`);
  logger.log("Preset includes playerRotation:", preset.playerRotation);
  const result = { ...preset };
  logger.log("Returning preset with playerRotation:", result.playerRotation);
  return result;
}

/**
 * Check if debug spawn is active
 * @returns {boolean}
 */
export function isDebugSpawnActive() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.has("gameState");
}

/**
 * Get the name of the current debug spawn state
 * @returns {string|null}
 */
export function getDebugSpawnStateName() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get("gameState");
}

/**
 * Apply character position from debug state
 * @param {Object} character - Physics character/rigid body
 * @param {Object} debugState - Debug state preset
 */
export function applyDebugCharacterPosition(character, debugState) {
  if (!character || !debugState || !debugState.playerPosition) {
    return;
  }

  const pos = debugState.playerPosition;
  character.translation.x = pos.x;
  character.translation.y = pos.y;
  character.translation.z = pos.z;

  logger.log(`Set character position to (${pos.x}, ${pos.y}, ${pos.z})`);
}

export default {
  getDebugSpawnState,
  isDebugSpawnActive,
  getDebugSpawnStateName,
  applyDebugCharacterPosition,
  debugStatePresets,
};
