/**
 * gameData.js - VIMANA GAME STATE DEFINITIONS
 * =============================================================================
 *
 * ROLE: Centralized definition of game state enums, shell collection states,
 * and the initial state object used at startup.
 *
 * GAME FLOW:
 * 1. LOADING - Initial loading screen
 * 2. MUSIC_ROOM - Player spawns in music room, can interact with harp
 * 3. HARP_COMPLETE - All duet sequences completed, shell available
 * 4. SHELL_COLLECTED - Shell collected, puzzle complete
 *
 * SHELL COLLECTION:
 * The music room contains 4 chambers, each with a shell to collect:
 * - archiveOfVoices (Harp chamber)
 * - galleryOfForms
 * - hydroponicMemory
 * - engineOfGrowth
 *
 * =============================================================================
 */

/**
 * Vimana game states
 * Note: States must be >= TITLE_SEQUENCE (2) for pointer lock to work
 */
export const GAME_STATES = {
  LOADING: -1,
  TITLE_SEQUENCE: 2,     // Reference for when pointer lock becomes available
  MUSIC_ROOM: 10,        // Player in music room, harp interaction available
  HARP_COMPLETE: 11,     // Duet completed, shell available to collect
  SHELL_COLLECTED: 12,   // Shell collected, white flash ending triggered
  TRANSSCENDENT: 13,     // After white flash, ascended state
};

/**
 * Shell chamber identifiers
 * Maps to the 4 chambers in the vimana
 */
export const CHAMBER_STATES = {
  // Harp chamber (Archive of Voices) - first shell
  ARCHIVE_OF_VOICES: 'archiveOfVoices',

  // Other chambers (to be implemented)
  GALLERY_OF_FORMS: 'galleryOfForms',
  HYDROPONIC_MEMORY: 'hydroponicMemory',
  ENGINE_OF_GROWTH: 'engineOfGrowth',
};

/**
 * Chamber display names for UI
 */
export const CHAMBER_NAMES = {
  archiveOfVoices: 'Archive of Voices',
  galleryOfForms: 'Gallery of Forms',
  hydroponicMemory: 'Hydroponic Memory',
  engineOfGrowth: 'Engine of Growth',
};

/**
 * Harp puzzle states
 */
export const HARP_STATES = {
  IDLE: 0,           // Waiting for player to approach
  TEACHING: 1,       // Jelly teaching sequence
  PLAYER_TURN: 2,    // Player repeating sequence
  CORRECT: 3,        // Player played correct note
  WRONG: 4,          // Player played wrong note
  COMPLETE: 5,       // Full sequence completed
};

/**
 * Initial game state applied at startup
 */
export const initialGameState = {
  // Session/game lifecycle
  isPlaying: false,
  isPaused: false,

  // Current scene
  currentScene: null,

  // High-level state name
  currentState: GAME_STATES.LOADING,

  // Control flow
  controlEnabled: false,

  // ================================================================
  // SHELL COLLECTION STATE
  // ================================================================
  // Each chamber has a boolean indicating if its shell was collected
  [CHAMBER_STATES.ARCHIVE_OF_VOICES]: false,
  [CHAMBER_STATES.GALLERY_OF_FORMS]: false,
  [CHAMBER_STATES.HYDROPONIC_MEMORY]: false,
  [CHAMBER_STATES.ENGINE_OF_GROWTH]: false,

  // Total shells collected (0-4)
  shellsCollected: 0,

  // Which chamber is the player currently in
  currentChamber: CHAMBER_STATES.ARCHIVE_OF_VOICES,

  // ================================================================
  // HARP PUZZLE STATE
  // ================================================================
  // Harp interaction state (see HARP_STATES)
  harpState: HARP_STATES.IDLE,

  // Current note sequence the player needs to play
  harpTargetSequence: [],

  // Player's progress in the current sequence (index)
  harpSequenceProgress: 0,

  // Total sequences completed (out of required amount)
  harpSequencesCompleted: 0,

  // Required sequences to complete the puzzle
  harpRequiredSequences: 3,

  // Which strings have been played in current attempt
  harpPlayedStrings: [],

  // ================================================================
  // JELLY CREATURE STATE
  // ================================================================
  // Is the jelly creature currently visible/active
  jellyVisible: true,

  // Jelly's current emotion (affects appearance)
  jellyEmotion: 'neutral', // neutral, happy, sad, encouraging

  // Is jelly demonstrating a note
  jellyDemonstrating: false,

  // ================================================================
  // VISUAL/EFFECT STATE
  // ================================================================
  // Water ripple intensity (for shader)
  waterRippleIntensity: 0,

  // Which string is currently vibrating (0-5, or -1 for none)
  vibratingString: -1,

  // Vortex activation level (0-1)
  vortexActivation: 0,

  // ================================================================
  // PLATFORM DETECTION
  // ================================================================
  isIOS: false,
  isSafari: false,
  isFullscreenSupported: true,
  isMobile: false,
};

/**
 * Helper to check if all shells are collected
 */
export function isAllShellsCollected(state) {
  return (
    state[CHAMBER_STATES.ARCHIVE_OF_VOICES] &&
    state[CHAMBER_STATES.GALLERY_OF_FORMS] &&
    state[CHAMBER_STATES.HYDROPONIC_MEMORY] &&
    state[CHAMBER_STATES.ENGINE_OF_GROWTH]
  );
}

/**
 * Helper to get number of collected shells
 */
export function getCollectedShellCount(state) {
  let count = 0;
  if (state[CHAMBER_STATES.ARCHIVE_OF_VOICES]) count++;
  if (state[CHAMBER_STATES.GALLERY_OF_FORMS]) count++;
  if (state[CHAMBER_STATES.HYDROPONIC_MEMORY]) count++;
  if (state[CHAMBER_STATES.ENGINE_OF_GROWTH]) count++;
  return count;
}

export default initialGameState;
