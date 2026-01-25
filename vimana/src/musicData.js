/**
 * musicData.js - VIMANA MUSIC TRACK DEFINITIONS
 * =============================================================================
 *
 * ROLE: Defines background music tracks for the vimana game
 *
 * MUSIC ASSETS:
 * Place your audio files in: /assets/music/
 *
 * Supported formats: .mp3, .ogg
 *
 * =============================================================================
 */

import { GAME_STATES } from './gameData.js';

/**
 * Music track definitions
 * Each track has:
 * - id: Unique identifier
 * - src: Path to audio file (relative to /assets/)
 * - criteria: When this track should play
 * - volume: Default volume (0-1)
 * - loop: Whether to loop (usually true for music)
 * - fadeIn: Fade in duration in seconds
 * - fadeOut: Fade out duration in seconds
 */
export const music = {
  // ================================================================
  // MUSIC ROOM AMBIENCE
  // ================================================================
  musicRoomAmbient: {
    id: 'music-room-ambient',
    src: '/assets/music/music-room-ambient.mp3',
    criteria: { currentState: GAME_STATES.MUSIC_ROOM },
    volume: 0.4,
    loop: true,
    fadeIn: 2.0,
    fadeOut: 1.5,
  },

  // ================================================================
  // HARP COMPLETION - BUILD UP
  // ================================================================
  harpComplete: {
    id: 'harp-complete',
    src: '/assets/music/harp-complete.mp3',
    criteria: { currentState: GAME_STATES.HARP_COMPLETE },
    volume: 0.6,
    loop: true,
    fadeIn: 1.0,
    fadeOut: 2.0,
  },

  // ================================================================
  // SHELL COLLECTION - TRANSCENDENT
  // ================================================================
  shellCollect: {
    id: 'shell-collect',
    src: '/assets/music/shell-collect.mp3',
    criteria: { currentState: { $gte: GAME_STATES.SHELL_COLLECTED } },
    volume: 0.8,
    loop: false, // Play once then transition
    fadeIn: 0.5,
    fadeOut: 3.0,
  },

  transcendence: {
    id: 'transcendence',
    src: '/assets/music/transcendence.mp3',
    criteria: { currentState: GAME_STATES.TRANSSCENDENT },
    volume: 0.7,
    loop: true,
    fadeIn: 5.0,
    fadeOut: 3.0,
  },
};

/**
 * Helper to get music track for current game state
 */
export function getMusicForState(gameState) {
  const { checkCriteria } = await import('./utils/criteriaHelper.js');
  return Object.values(music).find(track => {
    if (!track.criteria) return true;
    return checkCriteria(gameState, track.criteria);
  });
}

export default music;
