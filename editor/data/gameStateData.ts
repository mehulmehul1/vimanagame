/**
 * gameStateData.ts - Game State Definitions for Scene Flow Navigator
 * =============================================================================
 *
 * This file mirrors the GAME_STATES and STATE_CATEGORIES from the main game
 * (src/gameData.js and src/ui/sceneSelectorMenu.js) for use in the editor.
 *
 * The Scene Flow Navigator visualizes the story flow as a node graph.
 */

/**
 * Game State enum - matches src/gameData.js
 */
export const GAME_STATES: Record<string, number> = {
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
  DOORS_CLOSE: 12,
  ENTERING_OFFICE: 13,
  OFFICE_INTERIOR: 14,
  OFFICE_PHONE_ANSWERED: 15,
  PRE_VIEWMASTER: 16,
  VIEWMASTER: 17,
  VIEWMASTER_COLOR: 18,
  VIEWMASTER_DISSOLVE: 19,
  VIEWMASTER_DIALOG: 20,
  VIEWMASTER_HELL: 21,
  POST_VIEWMASTER: 22,
  CAT_DIALOG_CHOICE_2: 23,
  PRE_EDISON: 24,
  EDISON: 25,
  DIALOG_CHOICE_2: 26,
  CZAR_STRUGGLE: 27,
  SHOULDER_TAP: 28,
  PUNCH_OUT: 29,
  FALLEN: 30,
  LIGHTS_OUT: 31,
  WAKING_UP: 32,
  SHADOW_AMPLIFICATIONS: 33,
  CAT_SAVE: 34,
  CURSOR: 35,
  CURSOR_FINAL: 36,
  POST_CURSOR: 37,
  OUTRO: 38,
  OUTRO_LECLAIRE: 39,
  OUTRO_CAT: 40,
  OUTRO_CZAR: 41,
  OUTRO_CREDITS: 42,
  OUTRO_MOVIE: 43,
  GAME_OVER: 44,
};

/**
 * State categories for organization - matches src/ui/sceneSelectorMenu.js
 */
export const STATE_CATEGORIES: Record<string, { states: string[]; color: string; description: string }> = {
  "Intro & Title": {
    states: ["LOADING", "START_SCREEN", "INTRO", "TITLE_SEQUENCE", "TITLE_SEQUENCE_COMPLETE"],
    color: "#3b82f6", // blue
    description: "Game introduction and title sequence",
  },
  "Exterior - Plaza": {
    states: ["CAT_DIALOG_CHOICE", "NEAR_RADIO"],
    color: "#22c55e", // green
    description: "Exterior plaza scenes",
  },
  "Phone Booth Scene": {
    states: ["PHONE_BOOTH_RINGING", "ANSWERED_PHONE", "DIALOG_CHOICE_1"],
    color: "#a855f7", // purple
    description: "Phone booth interaction",
  },
  "Drive By Scene": {
    states: ["DRIVE_BY_PREAMBLE", "DRIVE_BY", "POST_DRIVE_BY"],
    color: "#f59e0b", // amber
    description: "Drive by event",
  },
  "Office Entry": {
    states: ["DOORS_CLOSE", "ENTERING_OFFICE", "OFFICE_INTERIOR"],
    color: "#64748b", // slate
    description: "Entering the office",
  },
  "Office Phone": {
    states: ["OFFICE_PHONE_ANSWERED", "PRE_VIEWMASTER"],
    color: "#8b5cf6", // violet
    description: "Office phone call",
  },
  "ViewMaster Sequence": {
    states: ["VIEWMASTER", "VIEWMASTER_COLOR", "VIEWMASTER_DISSOLVE", "VIEWMASTER_DIALOG", "VIEWMASTER_HELL", "POST_VIEWMASTER"],
    color: "#ef4444", // red
    description: "View-Master hallucination sequence",
  },
  "Cat Dialog 2": {
    states: ["CAT_DIALOG_CHOICE_2"],
    color: "#06b6d4", // cyan
    description: "Second cat dialog",
  },
  "Edison Scene": {
    states: ["PRE_EDISON", "EDISON", "DIALOG_CHOICE_2"],
    color: "#ec4899", // pink
    description: "Edison confrontation",
  },
  "Czar Struggle": {
    states: ["CZAR_STRUGGLE", "SHOULDER_TAP", "PUNCH_OUT", "FALLEN", "LIGHTS_OUT"],
    color: "#f97316", // orange
    description: "Physical confrontation with Czar",
  },
  "Waking Up": {
    states: ["WAKING_UP", "SHADOW_AMPLIFICATIONS", "CAT_SAVE"],
    color: "#14b8a6", // teal
    description: "Waking up sequence",
  },
  "Drawing Minigame": {
    states: ["CURSOR", "CURSOR_FINAL", "POST_CURSOR"],
    color: "#eab308", // yellow
    description: "Drawing/rune minigame",
  },
  "Outro Sequences": {
    states: ["OUTRO", "OUTRO_LECLAIRE", "OUTRO_CAT", "OUTRO_CZAR", "OUTRO_CREDITS", "OUTRO_MOVIE", "GAME_OVER"],
    color: "#6366f1", // indigo
    description: "End game sequences",
  },
};

/**
 * Type definitions for scene nodes
 */
export interface SceneNodeData {
  id: string;
  label: string;
  stateName: string;
  stateValue: number;
  category: string;
  description: string;
  criteria?: any;
}

/**
 * Get category name for a state
 */
export function getStateCategory(stateName: string): string | null {
  for (const [category, data] of Object.entries(STATE_CATEGORIES)) {
    if (data.states.includes(stateName)) {
      return category;
    }
  }
  return null;
}

/**
 * Get category data for a state
 */
export function getStateCategoryData(stateName: string) {
  const categoryName = getStateCategory(stateName);
  if (!categoryName) return null;
  return STATE_CATEGORIES[categoryName] || null;
}

/**
 * Get category color for a state
 */
export function getStateColor(stateName: string): string {
  for (const [, data] of Object.entries(STATE_CATEGORIES)) {
    if (data.states.includes(stateName)) {
      return data.color;
    }
  }
  return "#94a3b8"; // default gray
}

/**
 * Format state name for display
 */
export function formatStateName(stateName: string): string {
  return stateName
    .split("_")
    .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
    .join(" ");
}

/**
 * Get all states sorted by value
 */
export function getStatesSortedByValue(): Array<{ name: string; value: number; category: string | null }> {
  return Object.entries(GAME_STATES)
    .filter(([name]) => name !== 'LOADING' && name !== 'GAME_OVER') // Skip endpoints
    .map(([name, value]) => ({
      name,
      value,
      category: getStateCategory(name),
    }))
    .sort((a, b) => a.value - b.value);
}

export default GAME_STATES;
