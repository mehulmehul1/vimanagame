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
  LOADING: -1,
  START_SCREEN: 0,
  INTRO: 1,
  TITLE_SEQUENCE: 2,
  TITLE_SEQUENCE_COMPLETE: 3,
  CAT_DIALOG_CHOICE: 4,
  NEAR_RADIO: 5,
  PHONE_BOOTH_RINGING: 6,
  ANSWERED_PHONE: 7,
  DIALOG_CHOICE_1: 8,
  DRIVE_BY_PREAMBLE: 9,
  DRIVE_BY: 10,
  POST_DRIVE_BY: 11,
  ENTERING_OFFICE: 12,
  OFFICE_INTERIOR: 13,
  OFFICE_PHONE_ANSWERED: 14,
  PRE_VIEWMASTER: 15,
  VIEWMASTER: 16,
  VIEWMASTER_COLOR: 17,
  VIEWMASTER_DISSOLVE: 18,
  VIEWMASTER_DIALOG: 19,
  VIEWMASTER_HELL: 20,
  POST_VIEWMASTER: 21,
  CAT_DIALOG_CHOICE_2: 22,
  PRE_EDISON: 23,
  EDISON: 24,
  DIALOG_CHOICE_2: 25,
  CZAR_STRUGGLE: 26,
  SHOULDER_TAP: 27,
  PUNCH_OUT: 28,
  FALLEN: 29,
  LIGHTS_OUT: 30,
  WAKING_UP: 31,
  SHADOW_AMPLIFICATIONS: 32,
  CAT_SAVE: 33,
  CURSOR: 34,
  CURSOR_FINAL: 35,
  POST_CURSOR: 36,
  GAME_OVER: 37,
};

export const DIALOG_RESPONSE_TYPES = {
  EMPATH: 0,
  PSYCHOLOGIST: 1,
  LAWFUL: 2,
  CAT_GOOD_KITTY: 3,
  CAT_DAMN_CATS: 4,
  CAT_MY_FRIEND: 5,
  CAT_GIT: 6,
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

  // Zone management (for exterior splat loading/unloading)
  currentZone: "plaza", // Current zone name - starts in plaza during start screen

  // Control flow
  controlEnabled: false, // When true, character controller updates/inputs are enabled

  // Debug/authoring
  hasGizmoInData: false, // True when any data object (scene/video/etc.) declares gizmo: true

  // Display state
  isFullscreen: false, // When true, the app is in fullscreen mode

  // View-Master state
  isViewmasterEquipped: false, // True when the headset is currently being worn
  viewmasterManuallyRemoved: false, // True when viewmaster was manually removed (not timed out)
  viewmasterInsanityIntensity: 0.0, // Normalized intensity (0.0 to 1.0) from characterController
  viewmasterOverheatCount: 0, // Increments each time intensity crosses 0.5 threshold
  viewmasterOverheatDialogIndex: null, // Dialog index (0 or 1) computed from count % 2, null until first threshold crossing

  // Drawing game state (CURSOR/CURSOR_FINAL)
  drawingSuccessCount: 0, // Number of successful drawings (0-3)
  drawingFailureCount: 0, // Total number of failed attempts (increments each failure)
  lastDrawingSuccess: null, // Boolean: true if last attempt succeeded, false if failed, null if no attempt yet
  currentDrawingTarget: null, // String: current target symbol ("lightning", "star", "circle")

  // Platform detection (set by UIManager at initialization)
  isIOS: false, // True if running on iOS (iPhone/iPad)
  isFullscreenSupported: true, // True if Fullscreen API is supported (false on iOS)
  isMobile: false, // True if device supports touch input
};

export default startScreen;
