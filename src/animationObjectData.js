/**
 * Object Animation Data Structure
 *
 * Defines animations for scene objects triggered by game state changes.
 *
 * Common properties:
 * - id: Unique identifier
 * - type: "objectAnimation"
 * - description: Human-readable description
 * - criteria: Optional object with key-value pairs that must match game state
 *   - Simple equality: { currentState: GAME_STATES.INTRO }
 *   - Comparison operators: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.PHONE_BOOTH_RINGING } }
 *   - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
 * - priority: Higher priority animations are checked first (default: 0)
 * - playOnce: If true, only plays once per game session (default: false)
 * - delay: Delay in seconds before playing after state conditions are met (default: 0)
 *
 * Object Animation properties:
 * - targetObjectId: ID of the scene object from sceneData.js to animate
 * - childMeshName: Optional name of child mesh to animate (if not specified, animates root object)
 * - duration: Animation duration in seconds (default: 1.0)
 * - properties: Object containing properties to animate
 *   - position: { from: {x, y, z}, to: {x, y, z} } or { to: [{x, y, z}, {x, y, z}, ...] } - Animate position (optional)
 *   - rotation: { from: {x, y, z}, to: {x, y, z} } or { to: [{x, y, z}, {x, y, z}, ...] } - Animate rotation in radians (optional)
 *   - scale: { from: {x, y, z}, to: {x, y, z} } or { from: number, to: number } - Animate scale (optional)
 *   - opacity: { from: number, to: number } - Animate material opacity 0-1 (optional)
 *   - Note: If "from" is omitted, uses current value.
 *   - Note: "to" can be a single value OR an array for keyframe animation
 *   - Keyframes are evenly distributed across the duration (e.g., 3 keyframes = 0%, 50%, 100%)
 * - easing: Easing function name (default: "linear")
 *   - Options: "linear", "easeInQuad", "easeOutQuad", "easeInOutQuad", "easeInCubic",
 *     "easeOutCubic", "easeInOutCubic", "easeInOutElastic"
 * - loop: If true, continuously loop the animation (default: false)
 * - yoyo: If true, reverse animation on alternate loops (requires loop: true) (default: false)
 * - reparentToCamera: If true, reparent object to camera before animating (default: false)
 * - reverseOnCriteria: Optional criteria object - when matched, animation reverses direction
 *   - Uses same criteria format as main criteria (simple equality or comparison operators)
 *   - Example: { currentState: GAME_STATES.NEXT_STATE } to reverse when entering next state
 * - onComplete: Optional callback when animation completes. Receives gameManager as parameter.
 *   Example: onComplete: (gameManager) => { gameManager.setState({...}); }
 *
 * Usage:
 * import { objectAnimations } from './animationObjectData.js';
 */

import { GAME_STATES } from "./gameData.js";
import { Logger } from "./utils/logger.js";

// Create module-level logger
const logger = new Logger("ObjectAnimationData", false);

export const objectAnimations = {
  viewmasterPeering: {
    id: "viewmasterPeering",
    type: "objectAnimation",
    description:
      "Animate viewmaster up to player's face for first-person viewing",
    targetObjectId: "viewmaster",
    duration: 3.0,
    properties: {
      position: {
        // This will be dynamically calculated to be in front of camera (camera-local space)
        to: [
          { x: 0, y: 0, z: -0.4 },
          { x: 0, y: 0, z: -0.2 },
          { x: 0, y: 0, z: 0.1 },
        ],
      },
      rotation: {
        to: [
          { x: 0, y: 0, z: 0 },
          { x: 0, y: 0, z: 0 },
        ],
      },
    },
    reparentToCamera: true, // Reparent to camera before animating (like phone receiver)
    easing: "easeInOutQuad",
    criteria: { currentState: GAME_STATES.VIEWMASTER },
    priority: 100,
    playOnce: true,
  },

  viewmasterToggleOn: {
    id: "viewmasterToggleOn",
    type: "objectAnimation",
    description: "Equip viewmaster during free-toggle phase",
    targetObjectId: "viewmaster",
    duration: 1.6,
    properties: {
      position: {
        to: [
          { x: 0, y: 0, z: -0.4 },
          { x: 0, y: 0, z: -0.2 },
          { x: 0, y: 0, z: 0.1 },
        ],
      },
      rotation: {
        to: [
          { x: 0, y: 0, z: 0 },
          { x: 0, y: 0, z: 0 },
        ],
      },
    },
    easing: "easeInOutQuad",
    reparentToCamera: true,
    priority: 80,
    playOnce: false,
  },

  viewmasterToggleOff: {
    id: "viewmasterToggleOff",
    type: "objectAnimation",
    description: "Stow viewmaster just out of view when toggled off",
    targetObjectId: "viewmaster",
    duration: 1.4,
    properties: {
      position: {
        to: [
          { x: 0, y: 0, z: -0.2 },
          { x: 0, y: 0.5, z: -0.2 },
          { x: 0, y: 1, z: 0 },
        ],
      },
    },
    easing: "easeInOutQuad",
    reparentToCamera: true,
    priority: 80,
    playOnce: false,
  },

  viewmasterPutDown: {
    id: "viewmasterPutDown",
    type: "objectAnimation",
    description:
      "Remove viewmaster from face with upward movement, synced with vertical wipe effects",
    targetObjectId: "viewmaster",
    duration: 4.0, // Match edisonWipeIn duration for synchronization
    properties: {
      position: {
        // Reverse the peering animation with upward arc, then back to original world position
        to: [
          { x: 0, y: 0, z: -0.2 }, // Move up and further back (lifting motion)
          { x: 0, y: 0.5, z: -0.2 }, // Return to original world position (will un-parent from camera)
          { x: 0, y: 1, z: 0 }, // Return to original world position (will un-parent from camera)
        ],
      },
    },
    easing: "easeInOutQuad",
    criteria: { currentState: GAME_STATES.POST_VIEWMASTER },
    priority: 100,
    playOnce: true,
  },

  viewmasterLightsOutPosition: {
    id: "viewmasterLightsOutPosition",
    type: "objectAnimation",
    description:
      "Position viewmaster 3m in front of player during LIGHTS_OUT and WAKING_UP (unseen behind blackout fade, then correct when fade clears)",
    targetObjectId: "viewmaster",
    duration: 0.01,
    properties: {
      position: {
        // Player position: { x: -5.14, y: 2.05, z: 83.66 } with yaw: 0 (facing -Z)
        // 3m in front means 3m in -Z direction: z: 83.66 - 3 = 80.66
        to: { x: -3.31, y: 0.85, z: 81.3 },
      },
      rotation: {
        // Reset rotation for clean positioning
        to: { x: 0.6079, y: -0.5014, z: 0.2916 },
      },
    },
    easing: "easeInOutQuad",
    criteria: {
      currentState: {
        $in: [
          GAME_STATES.LIGHTS_OUT,
          GAME_STATES.WAKING_UP,
          GAME_STATES.SHADOW_AMPLIFICATIONS,
        ],
      },
    },
    priority: 100,
    playOnce: true, // Will retry automatically if object isn't loaded yet (handled in animationManager)
  },

  doorsOpenLeft: {
    id: "doorsOpenLeft",
    type: "objectAnimation",
    description: "Open left door by rotating 120 degrees on Y axis",
    targetObjectId: "doors",
    childMeshName: "Big_Door_L",
    duration: 3.0,
    properties: {
      rotation: {
        to: { y: (Math.PI * 2) / 3 }, // 120 degrees on Y only (preserves X and Z)
      },
    },
    easing: "easeInOutQuad",
    criteria: { currentState: GAME_STATES.POST_DRIVE_BY },
    priority: 50,
    playOnce: true,
    delay: 6.25,
  },

  doorsOpenRight: {
    id: "doorsOpenRight",
    type: "objectAnimation",
    description: "Open right door by rotating 120 degrees on Y axis",
    targetObjectId: "doors",
    childMeshName: "Big_Door_R",
    duration: 3.0,
    properties: {
      rotation: {
        to: { y: (Math.PI * 2) / 3 }, // 120 degrees on Y only (preserves X and Z)
      },
    },
    easing: "easeInOutQuad",
    criteria: { currentState: GAME_STATES.POST_DRIVE_BY },
    priority: 50,
    playOnce: true,
    delay: 6.25,
  },

  doorsCloseLeft: {
    id: "doorsCloseLeft",
    type: "objectAnimation",
    description: "Close left door by rotating back to 0 degrees on Y axis",
    targetObjectId: "doors",
    childMeshName: "Big_Door_L",
    duration: 3.0,
    properties: {
      rotation: {
        to: { y: 0 }, // Return to original rotation
      },
    },
    easing: "easeInOutQuad",
    criteria: { currentState: GAME_STATES.ENTERING_OFFICE },
    priority: 50,
    playOnce: true,
  },

  doorsCloseRight: {
    id: "doorsCloseRight",
    type: "objectAnimation",
    description: "Close right door by rotating back to 0 degrees on Y axis",
    targetObjectId: "doors",
    childMeshName: "Big_Door_R",
    duration: 3.0,
    properties: {
      rotation: {
        to: { y: 0 }, // Return to original rotation
      },
    },
    easing: "easeInOutQuad",
    criteria: { currentState: GAME_STATES.ENTERING_OFFICE },
    priority: 50,
    playOnce: true,
  },

  // Example of objectAnimation (simple):
  // radioFloat: {
  //   id: "radioFloat",
  //   type: "objectAnimation",
  //   description: "Make radio float up and down",
  //   targetObjectId: "radio", // Must match ID in sceneData.js
  //   duration: 2.0,
  //   properties: {
  //     position: {
  //       from: { x: 4.04, y: -0.28, z: 35.45 },
  //       to: { x: 4.04, y: 0.5, z: 35.45 }
  //     },
  //     rotation: {
  //       to: { x: 0, y: 3.14159, z: 0 } // Omit "from" to use current value
  //     },
  //     scale: {
  //       from: 2.46, // Can use single number for uniform scale
  //       to: 3.0
  //     }
  //   },
  //   easing: "easeInOutQuad",
  //   loop: true,
  //   yoyo: true, // Reverse on alternate loops (float up, then down, repeat)
  //   criteria: { currentState: GAME_STATES.NEAR_RADIO },
  //   priority: 50,
  //   playOnce: false,
  // },
  //
  // Example of objectAnimation (keyframes):
  // radioSpin: {
  //   id: "radioSpin",
  //   type: "objectAnimation",
  //   description: "Make radio move through multiple positions",
  //   targetObjectId: "radio",
  //   duration: 3.0,
  //   properties: {
  //     position: {
  //       // Array of positions - will be evenly distributed across duration
  //       to: [
  //         { x: 4.04, y: 0.5, z: 35.45 },   // 33% of time
  //         { x: 4.5, y: 0.8, z: 35.0 },     // 66% of time
  //         { x: 4.04, y: -0.28, z: 35.45 }  // 100% of time (back to start)
  //       ]
  //     },
  //     rotation: {
  //       // Can also keyframe rotation
  //       to: [
  //         { x: 0, y: Math.PI / 2, z: 0 },
  //         { x: 0, y: Math.PI, z: 0 },
  //         { x: 0, y: Math.PI * 2, z: 0 }
  //       ]
  //     }
  //   },
  //   easing: "easeInOutQuad",
  //   loop: true,
  //   criteria: { currentState: GAME_STATES.NEAR_RADIO },
  //   priority: 50,
  // },
};

export default objectAnimations;
