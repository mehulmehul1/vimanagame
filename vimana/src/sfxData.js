/**
 * sfxData.js - VIMANA SOUND EFFECT DEFINITIONS
 * =============================================================================
 *
 * ROLE: Defines all sound effects for the vimana game
 *
 * SOUND ASSETS:
 * Place your audio files in: /assets/sfx/
 *
 * Supported formats: .mp3, .ogg, .wav
 *
 * =============================================================================
 */

import { GAME_STATES } from './gameData.js';

/**
 * Sound effect definitions
 * Each sfx has:
 * - id: Unique identifier for this sound
 * - src: Path to audio file (relative to /assets/)
 * - criteria: When this sfx should be available/loaded
 * - volume: Default volume (0-1)
 * - loop: Whether to loop the sound
 */
export const sfx = {
  // ================================================================
  // HARP STRING SOUNDS
  // ================================================================
  // The 6 strings of the harp, each with a unique tone
  harpString1: {
    id: 'harp-string-1',
    src: '/assets/sfx/harp-string-1.mp3', // C4 note
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.8,
    loop: false,
  },

  harpString2: {
    id: 'harp-string-2',
    src: '/assets/sfx/harp-string-2.mp3', // D4 note
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.8,
    loop: false,
  },

  harpString3: {
    id: 'harp-string-3',
    src: '/assets/sfx/harp-string-3.mp3', // E4 note
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.8,
    loop: false,
  },

  harpString4: {
    id: 'harp-string-4',
    src: '/assets/sfx/harp-string-4.mp3', // F4 note
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.8,
    loop: false,
  },

  harpString5: {
    id: 'harp-string-5',
    src: '/assets/sfx/harp-string-5.mp3', // G4 note
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.8,
    loop: false,
  },

  harpString6: {
    id: 'harp-string-6',
    src: '/assets/sfx/harp-string-6.mp3', // A4 note
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.8,
    loop: false,
  },

  // ================================================================
  // PUZZLE FEEDBACK SOUNDS
  // ================================================================
  correctNote: {
    id: 'correct-note',
    src: '/assets/sfx/correct-note.mp3',
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.5,
    loop: false,
  },

  wrongNote: {
    id: 'wrong-note',
    src: '/assets/sfx/wrong-note.mp3',
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.5,
    loop: false,
  },

  sequenceComplete: {
    id: 'sequence-complete',
    src: '/assets/sfx/sequence-complete.mp3',
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.7,
    loop: false,
  },

  puzzleComplete: {
    id: 'puzzle-complete',
    src: '/assets/sfx/puzzle-complete.mp3',
    criteria: { currentState: { $gte: GAME_STATES.HARP_COMPLETE } },
    volume: 0.8,
    loop: false,
  },

  // ================================================================
  // SHELL COLLECTION SOUNDS
  // ================================================================
  shellAppear: {
    id: 'shell-appear',
    src: '/assets/sfx/shell-appear.mp3',
    criteria: { currentState: GAME_STATES.HARP_COMPLETE },
    volume: 0.8,
    loop: false,
  },

  shellCollect: {
    id: 'shell-collect',
    src: '/assets/sfx/shell-collect.mp3',
    criteria: { currentState: { $gte: GAME_STATES.HARP_COMPLETE } },
    volume: 1.0,
    loop: false,
  },

  // ================================================================
  // JELLY CREATURE SOUNDS
  // ================================================================
  jellyAppear: {
    id: 'jelly-appear',
    src: '/assets/sfx/jelly-appear.mp3',
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.6,
    loop: false,
  },

  jellyEncourage: {
    id: 'jelly-encourage',
    src: '/assets/sfx/jelly-encourage.mp3',
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.6,
    loop: false,
  },

  jellyHappy: {
    id: 'jelly-happy',
    src: '/assets/sfx/jelly-happy.mp3',
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.6,
    loop: false,
  },

  // ================================================================
  // WATER/ENVIRONMENT SOUNDS
  // ================================================================
  waterRipple: {
    id: 'water-ripple',
    src: '/assets/sfx/water-ripple.mp3',
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.4,
    loop: false,
  },

  vortexHum: {
    id: 'vortex-hum',
    src: '/assets/sfx/vortex-hum.mp3',
    criteria: { currentState: { $gte: GAME_STATES.HARP_COMPLETE } },
    volume: 0.3,
    loop: true,
  },

  // ================================================================
  // UI SOUNDS
  // ================================================================
  uiClick: {
    id: 'ui-click',
    src: '/assets/sfx/ui-click.mp3',
    criteria: null, // Always available
    volume: 0.5,
    loop: false,
  },
};

/**
 * Helper to get harp string sound by index (0-5)
 */
export function getHarpStringSound(index) {
  const sounds = ['harpString1', 'harpString2', 'harpString3', 'harpString4', 'harpString5', 'harpString6'];
  return sounds[index] || 'harpString1';
}

/**
 * Helper to get all sfx that match current game state
 */
export function getSfxForState(gameState) {
  const { checkCriteria } = await import('./utils/criteriaHelper.js');
  return Object.values(sfx).filter(sound => {
    if (!sound.criteria) return true;
    return checkCriteria(gameState, sound.criteria);
  });
}

export default sfx;
