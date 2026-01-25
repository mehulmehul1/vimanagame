/**
 * generateGameState.js - Game State Code Generator
 * =============================================================================
 *
 * Reads editor/data/storyData.json and generates src/gameData.js
 * with GAME_STATES enum, DIALOG_RESPONSE_TYPES enum, and startScreen object.
 *
 * Usage: node scripts/generateGameState.js
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const STORY_DATA_PATH = path.join(__dirname, '../editor/data/storyData.json');
const OUTPUT_PATH = path.join(__dirname, '../src/gameData.js');
const BACKUP_PATH = path.join(__dirname, '../src/gameData.backup.js');

/**
 * Read story data from JSON file
 */
function readStoryData() {
  const content = fs.readFileSync(STORY_DATA_PATH, 'utf-8');
  return JSON.parse(content);
}

/**
 * Generate GAME_STATES enum from story data
 */
function generateGameStatesEnum(storyData) {
  const lines = ['export const GAME_STATES = {'];

  // Sort states by value
  const sortedStates = Object.values(storyData.states)
    .filter(s => s.id !== 'LOADING') // Skip LOADING as it's handled specially
    .sort((a, b) => a.value - b.value);

  for (const state of sortedStates) {
    lines.push(`  ${state.id}: ${state.value},`);
  }

  // Add endpoints
  lines.push('  LOADING: -1,');
  lines.push('  GAME_OVER: 44,');
  lines.push('};');

  return lines.join('\n');
}

/**
 * Generate DIALOG_RESPONSE_TYPES enum
 */
function generateDialogResponseTypes() {
  return `export const DIALOG_RESPONSE_TYPES = {
  EMPATH: 0,
  PSYCHOLOGIST: 1,
  LAWFUL: 2,
  CAT_GOOD_KITTY: 3,
  CAT_DAMN_CATS: 4,
  CAT_MY_FRIEND: 5,
  CAT_GIT: 6,
};`;
}

/**
 * Generate startScreen object
 */
function generateStartScreen() {
  return `/**
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
  currentZone: "plaza",

  // Control flow
  controlEnabled: false,

  // Debug/authoring
  hasGizmoInData: false,

  // Display state
  isFullscreen: false,

  // View-Master state
  isViewmasterEquipped: false,
  viewmasterManuallyRemoved: false,
  viewmasterInsanityIntensity: 0.0,
  viewmasterOverheatCount: 0,
  viewmasterOverheatDialogIndex: null,

  // Drawing game state (CURSOR/CURSOR_FINAL)
  drawingSuccessCount: 0,
  drawingFailureCount: 0,
  lastDrawingSuccess: null,
  currentDrawingTarget: null,
  sawRune: false,
  runeSightings: 0,

  // Platform detection (set by UIManager at initialization)
  isIOS: false,
  isSafari: false,
  isFullscreenSupported: true,
  isMobile: false,
};

export default startScreen;`;
}

/**
 * Generate the complete gameData.js file
 */
function generateGameDataFile() {
  const storyData = readStoryData();
  const timestamp = new Date().toISOString();

  const content = `/**
 * gameData.js - GAME STATE DEFINITIONS AND INITIAL VALUES
 * =============================================================================
 *
 * AUTO-GENERATED from editor/data/storyData.json
 * Generation timestamp: ${timestamp}
 *
 * WARNING: This file is auto-generated. Do not edit directly.
 * Modify the story data in the editor and run the generator instead.
 *
 * ROLE: Centralized definition of game state enums, response types, and the
 * initial state object used at startup and for resets.
 *
 * KEY EXPORTS:
 * - GAME_STATES: Enum of all narrative states (LOADING through GAME_OVER)
 * - DIALOG_RESPONSE_TYPES: Enum for multiple-choice dialog responses
 * - startScreen: Initial state object applied at game start
 *
 * STATE PROGRESSION:
 * Game states are numeric and generally increase as the story progresses.
 * Criteria can use comparison operators ($gte, $lt, etc.) against these values.
 *
 * USAGE:
 * All data files (dialog, music, video, etc.) import GAME_STATES for criteria.
 * GameManager initializes with startScreen state.
 *
 * =============================================================================
 */

${generateGameStatesEnum(storyData)}

${generateDialogResponseTypes()}

${generateStartScreen()}
`;

  return content;
}

/**
 * Main execution
 */
function main() {
  console.log('🎮 Game State Code Generator');
  console.log('=============================');
  console.log(`Reading: ${STORY_DATA_PATH}`);
  console.log(`Writing: ${OUTPUT_PATH}`);

  try {
    // Create backup of existing file
    if (fs.existsSync(OUTPUT_PATH)) {
      fs.copyFileSync(OUTPUT_PATH, BACKUP_PATH);
      console.log('✓ Backup created: gameData.backup.js');
    }

    // Generate new file
    const content = generateGameDataFile();
    fs.writeFileSync(OUTPUT_PATH, content, 'utf-8');

    const storyDataObj = JSON.parse(fs.readFileSync(STORY_DATA_PATH, 'utf-8'));
    console.log(`✓ Generated GAME_STATES with ${Object.keys(storyDataObj.states).length} states`);
    console.log('✓ Generated DIALOG_RESPONSE_TYPES enum');
    console.log('✓ Generated startScreen object');
    console.log('');
    console.log('✅ Game state code generation complete!');
    console.log('');
    console.log('To restore backup:');
    console.log(`  mv ${BACKUP_PATH} ${OUTPUT_PATH}`);

  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { main };
