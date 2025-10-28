/**
 * Video Data Structure
 *
 * Each video contains:
 * - id: Unique identifier for the video
 * - videoPath: Path to the video file (WebM with alpha channel)
 * - preload: If true, load during loading screen; if false, load after (default: false)
 * - position: {x, y, z} position in 3D space OR function(gameManager) => {x, y, z} for dynamic positioning
 * - rotation: {x, y, z} rotation in radians
 * - scale: {x, y, z} scale multipliers
 * - loop: Whether the video should loop
 * - muted: Whether the video should be muted (default: true)
 * - volume: Volume level 0.0-1.0 (default: 1.0)
 * - playbackRate: Playback speed multiplier (default: 1.0, 0.5 = half speed, 2.0 = double speed)
 * - spatialAudio: Enable 3D spatial audio (default: false)
 * - audioPositionOffset: {x, y, z} offset from video position for audio source (default: {x:0, y:0, z:0})
 * - pannerAttr: Web Audio API PannerNode attributes (default: HRTF, inverse distance)
 * - billboard: Whether the video should always face the camera
 * - criteria: Optional object with key-value pairs that must match game state for video to play
 *   - Simple equality: { currentState: GAME_STATES.INTRO }
 *   - Comparison operators: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.DRIVE_BY } }
 *   - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
 *   - If criteria matches → video should play
 *   - If criteria doesn't match → video should stop
 * - spawnCriteria: Optional separate criteria for when to spawn/show the video (mesh appears)
 *   - If provided without playCriteria: video spawns, then plays after `delay` seconds (time-based)
 *   - If provided with playCriteria: video spawns, then plays when playCriteria match (state-based)
 *   - If not provided, uses criteria for both spawning and playing (original behavior)
 * - playCriteria: Optional separate criteria for when to play the video (after it's spawned)
 *   - Only used if spawnCriteria is also provided
 *   - When provided: enables state-based playback (waits for playCriteria state to match)
 *   - When omitted: uses time-based playback (plays after delay seconds from spawn)
 * - autoPlay: If true, automatically play when criteria are met (default: false)
 * - delay: Delay in seconds before playing the video when criteria are met (default: 0)
 * - once: If true, only play once (tracked automatically)
 * - priority: Higher priority videos are checked first (default: 0)
 * - gizmo: If true, enable debug gizmo for positioning visual objects (G=move, R=rotate, S=scale)
 * - onComplete: Optional function called when video ends, receives gameManager
 *
 * Usage:
 * import { videos } from './videoData.js';
 * videoManager.playVideo('drive-by');
 * // or reference directly: videos.driveBy.position
 */

import { GAME_STATES } from "./gameData.js";
import { checkCriteria } from "./utils/criteriaHelper.js";
import { Logger } from "./utils/logger.js";

const logger = new Logger("VideoData", false);

export const videos = {
  shadowGlimpse: {
    id: "shadowGlimpse",
    videoPath: "/video/shadow-glimpse.webm",
    preload: false, // Load after loading screen
    position: { x: -17.95, y: 1.22, z: 39.24 },
    rotation: { x: 0.0, y: 1.4075, z: 0.0 },
    scale: { x: 0.95, y: 0.88, z: 1.0 },
    loop: false,
    muted: true,
    billboard: true,
    criteria: {
      shadowGlimpse: true,
    },
    autoPlay: true,
    once: true,
    priority: 0,
  },
  cat: {
    id: "cat",
    videoPath: "/video/cat.webm",
    preload: false, // Load after loading screen
    position: { x: -24.13, y: -1.48, z: 21.46 },
    rotation: { x: 0.0, y: 1.5708, z: 0.0 },
    scale: { x: 1, y: 1, z: 1 },
    loop: false,
    muted: false,
    billboard: true,
    criteria: {
      heardCat: true,
    },
    autoPlay: true,
    once: true,
    priority: 0,
  },
  cat2: {
    id: "cat2",
    videoPath: "/video/cat.webm",
    preload: false, // Load after loading screen
    position: { x: -1.74, y: 1.88, z: 80.75 },
    rotation: { x: 0.0, y: -0.8856, z: 0.0 },
    scale: { x: 0.27, y: 0.27, z: 1.0 },
    loop: false,
    muted: false,
    billboard: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_VIEWMASTER,
        $lte: GAME_STATES.PUNCH_OUT,
      },
    },
    autoPlay: true,
    once: true,
    priority: 0,
    delay: 3.5,
  },
  punch: {
    id: "punch",
    videoPath: "/video/punch.webm",
    preload: false, // Load after loading screen
    position: (gameManager) => {
      // Calculate position behind player
      if (!gameManager?.characterController) {
        logger.warn("Cannot get player position, using origin");
        return { x: 0, y: 1.8, z: 0 };
      }

      const playerPos = gameManager.characterController.character.translation();
      const yaw = gameManager.characterController.yaw;

      // Calculate behind player (opposite of forward direction)
      // Distance behind player
      const distanceBehind = 1.35;
      const heightOffset = 1.5;

      // Forward vector is: x = -sin(yaw), z = -cos(yaw)
      // Behind is the opposite: x = sin(yaw), z = cos(yaw)
      const behindX = playerPos.x + Math.sin(yaw) * distanceBehind;
      const behindZ = playerPos.z + Math.cos(yaw) * distanceBehind;

      logger.log(
        `Punch video spawning behind player at: { x: ${behindX.toFixed(
          2
        )}, y: ${(playerPos.y + heightOffset).toFixed(2)}, z: ${behindZ.toFixed(
          2
        )} }`
      );

      return {
        x: behindX,
        y: playerPos.y + heightOffset,
        z: behindZ,
      };
    },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 0.79, y: 0.79, z: 0.79 },
    loop: false,
    muted: true,
    billboard: true,
    once: true,
    priority: 0,
    spawnCriteria: {
      currentState: { $gte: GAME_STATES.SHOULDER_TAP }, // Video stays spawned from shoulder tap onward
    },
    autoPlay: true,
    delay: 0.2, // Play immediately when playCriteria match
  },
};

/**
 * Get videos that match current game state
 * @param {Object} gameState - Current game state
 * @returns {Array} Array of matching video configurations
 */
export function getVideosForState(gameState) {
  return Object.values(videos).filter((video) => {
    // Check if video has criteria
    if (!video.criteria) {
      return false;
    }

    // Check if criteria match current state
    const shouldPlay = checkCriteria(gameState, video.criteria);
    logger.log(
      `Checking video "${video.id}" - currentState: ${gameState.currentState}, criteria:`,
      video.criteria,
      `shouldPlay: ${shouldPlay}`
    );

    return shouldPlay;
  });
}
