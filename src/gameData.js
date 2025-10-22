/**
 * Game Data
 *
 * Centralized definition of game state keys and their default values.
 * Keep this in sync with systems that read state (music, dialog, SFX, colliders).
 */

/**
 * Canonical state names/flags used across data files
 */
export const GAME_STATES = {
  LOADING: -1, // Game is loading
  START_SCREEN: 0, // Game has loaded, START and OPTIONS buttons available, fullscreen available, camera animation plays
  INTRO: 1, // Intro sequence plays, camera animation plays
  TITLE_SEQUENCE: 2, // Title sequence plays, camera animation plays
  TITLE_SEQUENCE_COMPLETE: 3, // Title sequence completes, intro narration starts, player starts
  CAT_DIALOG_CHOICE: 4, // Player chooses reaction to cat
  NEAR_RADIO: 5, // Player approaches the radio (triggers radio audio)
  PHONE_BOOTH_RINGING: 6,
  ANSWERED_PHONE: 7,
  DIALOG_CHOICE_1: 8,
  DRIVE_BY_PREAMBLE: 9,
  DRIVE_BY: 10,
  POST_DRIVE_BY: 11,
  ENTERING_OFFICE: 12,
  OFFICE_INTERIOR: 13,
  SHOULDER_TAP: 14,
  PUNCH_OUT: 15,
  FALLEN: 16,
};

export const DIALOG_RESPONSE_TYPES = {
  EMPATH: 0,
  PSYCHOLOGIST: 1,
  LAWFUL: 2,
  CAT_GOOD_KITTY: 3,
  CAT_DAMN_CATS: 4,
};

/**
 * Initial game state applied at startup (and can be reused for resets).
 */
export const startScreen = {
  // Session/game lifecycle
  isPlaying: false,
  isPaused: false,

  // Scene and world
  currentScene: null,

  // High-level state name
  currentState: GAME_STATES.LOADING,

  // Control flow
  controlEnabled: false, // When true, character controller updates/inputs are enabled

  // Debug/authoring
  hasGizmoInData: false, // True when any data object (scene/video/etc.) declares gizmo: true

  // Display state
  isFullscreen: false, // When true, the app is in fullscreen mode

  // Platform detection (set by UIManager at initialization)
  isIOS: false, // True if running on iOS (iPhone/iPad)
  isFullscreenSupported: true, // True if Fullscreen API is supported (false on iOS)
};

export default startScreen;
