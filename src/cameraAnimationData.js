/**
 * Camera Animation Data Structure
 *
 * Defines camera animations, lookats, and character movements triggered by game state changes.
 *
 * Common properties:
 * - id: Unique identifier
 * - type: "animation", "lookat", or "moveTo"
 * - description: Human-readable description
 * - criteria: Optional object with key-value pairs that must match game state
 *   - Simple equality: { currentState: GAME_STATES.INTRO }
 *   - Comparison operators: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.PHONE_BOOTH_RINGING } }
 *   - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
 * - priority: Higher priority animations are checked first (default: 0)
 * - playOnce: If true, only plays once per game session (default: false)
 * - delay: Delay in seconds before playing after state conditions are met (default: 0)
 *
 * Type-specific properties:
 *
 * For type "animation" or "jsonAnimation":
 * - path: Path to the animation JSON file
 * - preload: If true, load during loading screen; if false, load after (default: false)
 * - syncController: If true, sync character controller yaw/pitch to final camera pose (default: true)
 * - restoreInput: If true, restore input controls when complete (default: true)
 * - scaleY: Optional Y-axis scale multiplier for animation (default: 1.0)
 *   - Values < 1.0 compress vertical motion, > 1.0 expand it
 *   - Example: 0.8 would reduce vertical movement by 20%
 * - playbackRate: Optional playback speed multiplier (default: 1.0)
 *   - Values < 1.0 play slower, > 1.0 play faster
 *   - Example: 0.5 for half speed, 2.0 for double speed
 *
 * For type "lookat":
 * SINGLE LOOKAT (simple position):
 * - position: {x, y, z} world position to look at
 * - transitionTime: Time for the initial look-at transition in seconds (default: 2.0)
 * - lookAtHoldDuration: How long to hold at target before returning or restoring control (default: 0)
 * - returnToOriginalView: If true, return to original view before restoring control (default: false)
 * - returnTransitionTime: Time for the return transition in seconds (default: same as transitionTime)
 * - enableZoom: If true, enable zoom/DoF effect (default: false)
 * - zoomOptions: Optional zoom configuration (see below)
 *
 * SEQUENCE LOOKAT (array of positions):
 * - positions: Array of {x, y, z} world positions to look at in sequence
 * - transitionTime: Default transition time for all positions (can be overridden per position)
 * - lookAtHoldDuration: Default hold duration for all positions (can be overridden per position)
 * - returnToOriginalView: If true, return to original view after final position (default: false)
 * - returnTransitionTime: Time for the return transition in seconds (default: same as transitionTime)
 * - enableZoom: Default zoom setting for all positions (can be overridden per position)
 * - zoomOptions: Default zoom configuration for all positions (can be overridden per position)
 * - sequenceSettings: Array of per-position overrides (optional)
 *   - Each entry can override: transitionTime, lookAtHoldDuration, enableZoom, zoomOptions
 *   - Example: [null, { transitionTime: 0.5 }, { enableZoom: false }]
 *   - Use null to use defaults for that position
 * - loop: If true, continuously cycle through positions (default: false)
 *
 * COMMON LOOKAT PROPERTIES:
 * - restoreInput: If true, restore input controls when complete (default: true)
 *   - If false, inputs remain disabled after animation - you must manually re-enable them
 * - zoomOptions: Optional zoom configuration
 *   - zoomFactor: Camera zoom multiplier (e.g., 2.0 for 2x zoom)
 *   - minAperture: DoF effect strength at peak
 *   - maxAperture: DoF effect strength at rest
 *   - transitionStart: When to start zoom (0-1, fraction of transitionTime)
 *   - transitionDuration: How long zoom IN and OUT transitions take in seconds
 *   - holdDuration: How long to hold at target zoom (independent of lookAtHoldDuration)
 * - onComplete: Optional callback when lookat completes. Receives gameManager as parameter.
 *   Example: onComplete: (gameManager) => { gameManager.setState({...}); }
 *
 * Input control: Always disabled during lookat. By default, restored when complete (or if zoom is enabled
 *                without returnToOriginalView, after holdDuration + transitionDuration). Set restoreInput
 *                to false to keep inputs disabled after animation completes.
 *
 * For type "moveTo":
 * - position: {x, y, z} world position to move character to
 * - rotation: {yaw, pitch} target rotation in radians (optional)
 * - transitionTime: Time for the movement transition in seconds (default: 2.0)
 * - inputControl: What input to disable during movement
 *   - disableMovement: Disable movement input (default: true)
 *   - disableRotation: Disable rotation input (default: true)
 * - onComplete: Optional callback when movement completes (no parameters passed)
 *
 * For type "fade":
 * - color: THREE.Color or {r, g, b} color to fade to (default: white {r:1, g:1, b:1})
 * - fadeInTime: Time to fade in (go to full opacity) in seconds (default: 0.1)
 * - holdTime: Time to hold at full opacity in seconds (default: 0)
 * - fadeOutTime: Time to fade out (go to zero opacity) in seconds (default: 1.0)
 * - maxOpacity: Maximum opacity to reach (0-1, default: 1.0)
 * - onComplete: Optional callback when fade completes (no parameters passed)
 *
 * Usage:
 * import { cameraAnimations, getCameraAnimationForState } from './cameraAnimationData.js';
 */

import { GAME_STATES } from "./gameData.js";
import { checkCriteria } from "./utils/criteriaHelper.js";
import { videos } from "./videoData.js";
import { sceneObjects } from "./sceneData.js";
import { Logger } from "./utils/logger.js";

// Create module-level logger
const logger = new Logger("CameraAnimationData", false);

export const cameraAnimations = {
  catLookat: {
    id: "catLookat",
    type: "lookat",
    description: "Look at cat video when player hears cat sound",
    position: videos.cat.position,
    transitionTime: 0.75,
    returnToOriginalView: true,
    returnTransitionTime: 1.25,
    enableZoom: true,
    zoomOptions: {
      zoomFactor: 1.8,
      minAperture: 0.15,
      maxAperture: 0.35,
      transitionStart: 0.7,
      transitionDuration: 2.0,
      holdDuration: 2.9,
    },
    criteria: { heardCat: true },
    playOnce: true,
    priority: 100,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.CAT_DIALOG_CHOICE });
    },
  },

  radioLookat: {
    id: "radioLookat",
    type: "lookat",
    description: "Look at radio when player approaches it",
    position: {
      x: sceneObjects.radio.position.x,
      y: sceneObjects.radio.position.y + 0.5,
      z: sceneObjects.radio.position.z,
    },
    transitionTime: 0.75,
    lookAtHoldDuration: 1.5, // Hold at radio for 1.5 seconds before returning
    returnToOriginalView: true,
    returnTransitionTime: 1.0,
    criteria: { currentState: GAME_STATES.NEAR_RADIO },
    playOnce: true,
    priority: 100,
  },

  shadowGlimpseLookat: {
    id: "shadowGlimpseLookat",
    type: "lookat",
    description: "Look at shadow glimpse video when player enters trigger",
    position: videos.shadowGlimpse.position,
    transitionTime: 0.6,
    returnToOriginalView: false,
    enableZoom: true,
    zoomOptions: {
      zoomFactor: 1.5,
      minAperture: 0.2,
      maxAperture: 0.35,
      transitionStart: 0.6,
      transitionDuration: 1.5,
      holdDuration: 2.0,
    },
    criteria: { shadowGlimpse: true },
    playOnce: true,
    priority: 100,
  },

  phoneBoothLookat: {
    id: "phoneBoothLookat",
    type: "lookat",
    description: "Look at phone booth when it starts ringing",
    position: {
      x: sceneObjects.phonebooth.position.x,
      y: 0.9,
      z: sceneObjects.phonebooth.position.z,
    },
    transitionTime: 1,
    enableZoom: true, // Enable dramatic zoom/DoF when looking at phone booth
    zoomOptions: {
      zoomFactor: 2.0, // More dramatic 2x zoom
      minAperture: 0.2, // Stronger DoF effect
      maxAperture: 0.4,
      transitionStart: 0.5, // Start zooming earlier (60% of look-at)
      transitionDuration: 2.5, // Slower, more dramatic transition
      holdDuration: 2.0, // Hold the zoom longer for dramatic effect
    },
    criteria: {
      currentState: {
        $in: [GAME_STATES.PHONE_BOOTH_RINGING],
      },
    },
    priority: 100,
    playOnce: true,
    delay: 0.25, // Wait 0.5 seconds before looking at phone booth
  },

  passageLookat: {
    id: "passageLookat",
    type: "lookat",
    description: "Look at passage after LeClaire tells you to",
    positions: [
      {
        x: sceneObjects.phonebooth.position.x,
        y: 0.9,
        z: sceneObjects.phonebooth.position.z,
      },
      sceneObjects.interior.position,
    ],
    transitionTime: 1,
    lookAtHoldDuration: 2.0,
    criteria: {
      currentState: {
        $in: [GAME_STATES.POST_DRIVE_BY],
      },
    },
    priority: 100,
    playOnce: true,
    delay: 0.25, // Wait 0.5 seconds before looking at phone booth
    sequenceSettings: [
      null, // Position 0: use defaults
      {
        // Position 1: different zoom settings
        enableZoom: true,
        zoomOptions: {
          zoomFactor: 1.5,
          minAperture: 0.25,
          maxAperture: 0.4,
          transitionStart: 0.5,
          transitionDuration: 2.0, // Important: this controls zoom fade timing
          holdDuration: 2.0,
        },
      },
    ],
  },

  phonoAndPhoneLookat: {
    id: "phonoAndPhoneLookat",
    type: "lookat",
    description: "Look at passage after LeClaire tells you to",
    positions: [
      sceneObjects.edison.position,
      sceneObjects.candlestickPhone.position,
    ],
    transitionTime: 1,
    lookAtHoldDuration: 2.0,
    criteria: {
      currentState: {
        $in: [GAME_STATES.OFFICE_INTERIOR],
      },
    },
    priority: 100,
    playOnce: true,
    delay: 0.25, // Wait 0.5 seconds before looking at phone booth
  },

  phoneBoothMoveTo: {
    id: "phoneBoothMoveTo",
    type: "moveTo",
    description: "Move character into phone booth when player enters trigger",
    position: {
      x: sceneObjects.phonebooth.position.x,
      y: 0.4,
      z: sceneObjects.phonebooth.position.z - 0.2,
    },
    rotation: {
      yaw: Math.PI, // Face the phone (90 degrees)
      pitch: 0,
    },
    transitionTime: 1.5,
    inputControl: {
      disableMovement: true, // Disable movement
      disableRotation: false, // Allow rotation (player can look around)
    },
    criteria: { currentState: GAME_STATES.ANSWERED_PHONE },
    priority: 100,
    playOnce: true,
  },

  carLookat: {
    id: "carLookat",
    type: "lookat",
    description: "Look at car from within phone booth",
    position: { x: -5.9, y: 0.76, z: 68.35 },
    transitionTime: 1,
    enableZoom: true, // Enable dramatic zoom/DoF when looking at phone booth
    zoomOptions: {
      zoomFactor: 2.0, // More dramatic 2x zoom
      minAperture: 0.2, // Stronger DoF effect
      maxAperture: 0.4,
      transitionStart: 0.5, // Start zooming earlier (60% of look-at)
      transitionDuration: 2.5, // Slower, more dramatic transition
      holdDuration: 2.0, // Hold the zoom longer for dramatic effect
    },
    criteria: { currentState: GAME_STATES.DRIVE_BY_PREAMBLE },
    restoreInput: false,
    priority: 100,
    playOnce: true,
    delay: 0, // Wait 0.5 seconds before looking at phone booth
  },

  lookAndJump: {
    id: "lookAndJump",
    type: "jsonAnimation",
    path: "/json/look-and-jump.json",
    description: "Camera animation for drive-by sequence",
    criteria: { currentState: GAME_STATES.DRIVE_BY },
    priority: 100,
    playOnce: true,
    syncController: true,
    restoreInput: true,
    delay: 1.5, // Wait 2 seconds after DRIVE_BY state before animation
    scaleY: 0.8, // Optional: reduce vertical motion by 20%
    playbackRate: 1.25, // Optional: adjust playback speed (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.POST_DRIVE_BY });
    },
  },

  shoulderTap: {
    id: "shoulderTap",
    type: "lookat",
    position: videos.punch.position,
    transitionTime: 0.75,
    criteria: { currentState: GAME_STATES.SHOULDER_TAP },
    priority: 100,
    playOnce: true,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.PUNCH_OUT });
    },
  },

  punchOut: {
    id: "punchOut",
    type: "jsonAnimation",
    path: "/json/punchout.json",
    description: "Camera animation for punch-out sequence",
    criteria: { currentState: GAME_STATES.PUNCH_OUT },
    priority: 100,
    playOnce: true,
    syncController: false, // Don't sync controller - leave camera where it lands
    restoreInput: false, // Don't restore input - leave player frozen
    delay: 0.1,
    scaleY: 0.4,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.FALLEN });
    },
  },

  punchWhiteout: {
    id: "punchWhiteout",
    type: "fade",
    description: "Whiteout effect when punch impacts",
    color: { r: 1, g: 1, b: 1 }, // White
    fadeInTime: 0.05, // Very fast flash to white
    holdTime: 0.01, // Brief hold
    fadeOutTime: 1.5, // Slower fade back to normal
    maxOpacity: 0.8, // Full whiteout
    criteria: { currentState: GAME_STATES.PUNCH_OUT },
    priority: 100,
    playOnce: true,
    delay: 0.15, // Sync with video punch impact
  },

  fallenBlackout: {
    id: "fallenBlackout",
    type: "fade",
    description: "Blackout effect after falling",
    color: { r: 0, g: 0, b: 0 }, // Black
    fadeInTime: 3.0, // Slow fade to black
    holdTime: 10, // Long hold in darkness
    fadeOutTime: 1.5, // Fade back to normal
    maxOpacity: 1.0, // Full blackout
    criteria: { currentState: GAME_STATES.FALLEN },
    priority: 100,
    playOnce: true,
    delay: 1.5, // Delay after falling
  },

  // Example of a lookat sequence with per-position settings:
  // sequenceExample: {
  //   id: "sequenceExample",
  //   type: "lookat",
  //   description: "Look at multiple positions in sequence",
  //   positions: [
  //     { x: 5.94, y: 0.9, z: 65.76 },  // Phone booth
  //     { x: 4.04, y: 0.8, z: 35.45 },  // Radio
  //     { x: -5.9, y: 0.76, z: 68.35 }, // Car
  //   ],
  //   // Default settings for all positions
  //   transitionTime: 1.0,
  //   lookAtHoldDuration: 2.0,
  //   enableZoom: true,
  //   zoomOptions: {
  //     zoomFactor: 2.0,
  //     minAperture: 0.2,
  //     maxAperture: 0.4,
  //     transitionStart: 0.5,
  //     transitionDuration: 2.5,
  //     holdDuration: 2.0,
  //   },
  //   // Per-position overrides (optional)
  //   sequenceSettings: [
  //     null, // Position 0: use defaults
  //     { // Position 1: faster transition, longer hold
  //       transitionTime: 0.5,
  //       lookAtHoldDuration: 3.0,
  //     },
  //     { // Position 2: different zoom settings
  //       enableZoom: true,
  //       zoomOptions: {
  //         zoomFactor: 1.5,
  //         minAperture: 0.25,
  //         holdDuration: 3.0,
  //       }
  //     }
  //   ],
  //   loop: false, // Set to true to continuously cycle through positions
  //   returnToOriginalView: true, // Return to original view after last position
  //   restoreInput: true,
  //   criteria: { currentState: GAME_STATES.YOUR_STATE },
  //   priority: 100,
  //   playOnce: true,
  // },
};

/**
 * Get all camera animations that should play for the current game state
 * @param {Object} gameState - Current game state
 * @param {Set} playedAnimations - Set of animation IDs that have already been played (for playOnce check)
 * @returns {Array} Array of matching camera animation data
 */
export function getCameraAnimationsForState(
  gameState,
  playedAnimations = new Set()
) {
  // Convert to array and sort by priority (highest first)
  const animations = Object.values(cameraAnimations).sort(
    (a, b) => (b.priority || 0) - (a.priority || 0)
  );

  logger.log(`Checking ${animations.length} animations for state:`, gameState);

  const matchingAnimations = [];

  // Find all animations matching criteria that haven't been played yet
  for (const animation of animations) {
    if (!animation.criteria) {
      logger.log(`Animation '${animation.id}' has no criteria, skipping`);
      continue;
    }

    const matches = checkCriteria(gameState, animation.criteria);
    logger.log(
      `Animation '${animation.id}' criteria:`,
      animation.criteria,
      `matches:`,
      matches
    );

    if (matches) {
      // Check playOnce - skip if already played
      if (animation.playOnce && playedAnimations.has(animation.id)) {
        logger.log(
          `Animation '${animation.id}' matches but already played (playOnce), skipping...`
        );
        continue;
      }
      matchingAnimations.push(animation);
    }
  }

  logger.log(`Found ${matchingAnimations.length} matching animation(s)`);
  return matchingAnimations;
}

/**
 * Get the camera animation that should play for the current game state
 * @param {Object} gameState - Current game state
 * @param {Set} playedAnimations - Set of animation IDs that have already been played (for playOnce check)
 * @returns {Object|null} Camera animation data or null if none match
 * @deprecated Use getCameraAnimationsForState to get all matching animations
 */
export function getCameraAnimationForState(
  gameState,
  playedAnimations = new Set()
) {
  const animations = getCameraAnimationsForState(gameState, playedAnimations);
  return animations.length > 0 ? animations[0] : null;
}

export default cameraAnimations;
