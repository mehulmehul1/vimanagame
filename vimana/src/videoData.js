/**
 * videoData.js - VIMANA VIDEO DEFINITIONS
 * =============================================================================
 *
 * ROLE: Defines intro video that plays before the game starts
 *
 * VIDEOS:
 * - intro: Opening video that transitions to music room
 *
 * =============================================================================
 */

import { GAME_STATES } from './gameData.js';

/**
 * Export videos object for VideoManager
 */
export const videos = {
  intro: {
    id: 'intro',
    // PLACE YOUR INTRO VIDEO HERE:
    videoPath: '/assets/videos/vimanaload.mp4',

    // Video covers the screen
    position: { x: 0, y: 0, z: -2 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 16, y: 9, z: 1 },

    // Audio settings
    muted: false,
    volume: 1.0,
    loop: false,

    // Load immediately
    preload: true,
    autoPlay: false,  // Will be triggered manually

    // When to show this video
    criteria: {
      currentState: GAME_STATES.VIDEO_INTRO,
    },

    // What happens when video ends
    onComplete: (gameManager) => {
      // Transition to music room
      gameManager.setState({
        currentState: GAME_STATES.MUSIC_ROOM,
        controlEnabled: true,
        introVideoWatched: true,
      });
    },
  },
};
