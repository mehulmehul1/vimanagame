/**
 * Camera Animation Data Structure
 *
 * Defines camera animations, lookats, character movements, and fade effects triggered by game state changes.
 *
 * Common properties:
 * - id: Unique identifier
 * - type: "animation", "lookat", "moveTo", "fade"
 * - description: Human-readable description
 * - criteria: Optional object with key-value pairs that must match game state
 *   - Simple equality: { currentState: GAME_STATES.INTRO }
 *   - Comparison operators: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.PHONE_BOOTH_RINGING } }
 *   - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
 * - priority: Higher priority animations are checked first (default: 0)
 * - playOnce: If true, only plays once per game session (default: false)
 * - delay: Delay in seconds before playing after state conditions are met (default: 0)
 * - fireOnEvent: (optional) Event name to listen for on gameManager. When this event is emitted, the animation will play
 *   - Takes precedence over criteria-based triggering (animation can play on event even if criteria don't match)
 *   - Useful for event-driven animations triggered by dialogs, interactions, or other game events
 *   - Example: fireOnEvent: "shadow:speaks" - animation plays when "shadow:speaks" event is emitted
 *   - Note: If both fireOnEvent and criteria are specified, the animation can trigger from either source
 * - playNext: Chain to another animation after this one completes (supported by all types)
 *   - Can be an animation object (e.g., cameraAnimations.nextAnim) or string ID (e.g., "nextAnim")
 *   - Allows creating animation sequences without requiring game state changes between animations
 *   - The chained animation's delay property is respected if specified
 *   - Example: playNext: "leclaireLookat" or playNext: cameraAnimations.leclaireLookat
 *   - Note: When using playNext, onComplete is called before moving to the next animation
 *
 * Input Control Properties Summary:
 * - restoreInput: Supported by ALL types (jsonAnimation, lookat, moveTo, fade)
 *   - Controls what input is restored after animation completes
 *   - Can be boolean (true/false) for backward compatibility or object { movement: boolean, rotation: boolean }
 *   - Examples:
 *     - restoreInput: true (or { movement: true, rotation: true }) - restore both movement and rotation
 *     - restoreInput: false (or { movement: false, rotation: false }) - restore nothing, camera frozen
 *     - restoreInput: { movement: true, rotation: false } - restore movement only, keep rotation disabled
 * - inputControl: Supported by "moveTo" and "fade" types
 *   - Controls what input to disable DURING animation: { disableMovement: boolean, disableRotation: boolean }
 *   - For moveTo: defaults to { disableMovement: true, disableRotation: true }
 *   - For fade: defaults to { disableMovement: false, disableRotation: false }
 *   - NOT supported by jsonAnimation or lookat (input is always fully disabled during those animations)
 *
 * Type-specific properties:
 *
 * For type "animation" or "jsonAnimation":
 * - path: Path to the animation JSON file
 * - preload: If true, load during loading screen; if false, load after (default: false)
 * - restoreInput: What input to restore after animation completes (default: true or { movement: true, rotation: true })
 *   - Boolean: true restores both, false restores nothing
 *   - Object: { movement: boolean, rotation: boolean } - selectively restore movement and/or rotation
 *   - Examples:
 *     - restoreInput: true - restore both movement and rotation
 *     - restoreInput: { movement: true, rotation: false } - restore movement only
 *     - restoreInput: false - restore nothing, camera freezes at final position
 * - scaleY: Optional Y-axis scale multiplier for animation (default: 1.0)
 *   - Values < 1.0 compress vertical motion, > 1.0 expand it
 *   - Example: 0.8 would reduce vertical movement by 20%
 * - playbackRate: Optional playback speed multiplier (default: 1.0)
 *   - Values < 1.0 play slower, > 1.0 play faster
 *   - Example: 0.5 for half speed, 2.0 for double speed
 *   - Note: If duration is specified, playbackRate is calculated automatically and this property is ignored
 * - duration: Optional target duration in seconds to play the animation (overrides playbackRate if specified)
 *   - The system calculates the required playbackRate automatically to achieve this duration
 *   - Example: duration: 5.0 will play the animation in exactly 5 seconds regardless of its natural length
 *   - If both duration and playbackRate are specified, duration takes precedence
 * - playbackPercentage: Optional percentage of animation to play (0.0 to 1.0, default: 1.0)
 *   - Values < 1.0 play only a portion of the animation from the start
 *   - Example: 0.5 plays the first 50% of the animation and treats it as the full duration
 *   - The animation plays at normal speed but stops early
 * - blendWithPlayer: If true, blend animation with player movement and idle (default: false)
 *   - When enabled, animation mixes with current player-controlled camera transform
 *   - Player can still move and look around, with animation layered on top
 * - blendAmount: Blend strength (0.0 to 1.0, default: 0.5)
 *   - 0.0 = fully player-controlled, 1.0 = fully animated
 *   - Example: 0.3 = 30% animation, 70% player control
 *
 * For type "lookat":
 * SINGLE LOOKAT (simple position):
 * - position: {x, y, z} world position to look at OR function(gameManager) => {x, y, z} for dynamic positioning
 *   - When using characterController.getPosition({x, y, z}), the object is an OFFSET from player position, not world coordinates
 *   - Example: getPosition({ x: 0, y: 1.5, z: -1 }) = 1 meter behind player at eye level (z: -1 is behind, z: 1 is in front)
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
 * - restoreInput: What input to restore after animation completes (default: true or { movement: true, rotation: true })
 *   - Boolean: true restores both, false restores nothing
 *   - Object: { movement: boolean, rotation: boolean } - selectively restore movement and/or rotation
 *   - Examples:
 *     - restoreInput: true - restore both movement and rotation
 *     - restoreInput: { movement: false, rotation: true } - restore rotation only
 *     - restoreInput: false - restore nothing, inputs remain disabled
 *     - restoreInput: { movement: true, rotation: false } - restore movement only, keep rotation disabled
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
 * Input control: Input is always fully disabled (both movement and rotation) during lookat to prevent conflicts.
 *                By default, input is restored when complete (or if zoom is enabled without returnToOriginalView,
 *                after holdDuration + transitionDuration). Use restoreInput to control what gets restored after completion.
 *
 * For type "moveTo":
 * - position: {x, y, z} world position to move character to
 * - rotation: {yaw, pitch} target rotation in radians (optional, ignored if lookat is set)
 * - lookat: {x, y, z} world position to look at during movement (optional, overrides rotation)
 * - transitionTime: Time for the movement transition in seconds (default: 2.0)
 * - autoHeight: If true, automatically calculate Y position based on floor collider at X/Z coordinates (default: false)
 *   - When enabled, the Y value in position is ignored and calculated via raycast
 *   - Prevents character from falling after animation when floor height changes
 * - inputControl: What input to disable during movement
 *   - disableMovement: Disable movement input (default: true)
 *   - disableRotation: Disable rotation input (default: true)
 * - restoreInput: What input to restore after movement completes (default: true or { movement: true, rotation: true })
 *   - Boolean: true restores both, false restores nothing
 *   - Object: { movement: boolean, rotation: boolean } - selectively restore movement and/or rotation
 * - onComplete: Optional callback when movement completes (no parameters passed)
 *
 * For type "fade":
 * - color: THREE.Color or {r, g, b} color to fade to (default: white {r:1, g:1, b:1})
 * - fadeInTime: Time to fade in (go to full opacity) in seconds (default: 0.1)
 * - holdTime: Time to hold at full opacity in seconds (default: 0)
 * - fadeOutTime: Time to fade out (go to zero opacity) in seconds (default: 1.0)
 * - maxOpacity: Maximum opacity to reach (0-1, default: 1.0)
 * - persistWhileCriteria: If true, fade persists at max opacity as long as criteria match (default: false)
 *   - When enabled, fade will stay at maxOpacity even after holdTime expires, until criteria no longer match
 *   - Once criteria stop matching, fade will complete normally or another fade can replace it
 * - inputControl: What input to disable during fade (optional)
 *   - disableMovement: Disable movement input (default: false)
 *   - disableRotation: Disable rotation input (default: false)
 * - restoreInput: What input to restore after fade completes (default: true or { movement: true, rotation: true })
 *   - Boolean: true restores both, false restores nothing
 *   - Object: { movement: boolean, rotation: boolean } - selectively restore movement and/or rotation
 * - onFadeInComplete: Optional callback when fade-in reaches max opacity. Receives gameManager as parameter.
 *   - Useful with persistWhileCriteria to trigger logic once black screen is reached
 * - onComplete: Optional callback when fade completes (returns to 0 opacity). Receives gameManager as parameter.
 *
 * Usage:
 * import { cameraAnimations, getCameraAnimationForState } from './animationCameraData.js';
 */

import { GAME_STATES } from "./gameData.js";
import * as THREE from "three";
import { checkCriteria } from "./utils/criteriaHelper.js";
import { videos } from "./videoData.js";
import { sceneObjects } from "./sceneData.js";
import { Logger } from "./utils/logger.js";
import { VIEWMASTER_OVERHEAT_THRESHOLD } from "./dialogData.js";

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

  cat2Lookat: {
    id: "cat2Lookat",
    type: "lookat",
    description: "Look at cat2 video in POST_VIEWMASTER state",
    position: videos.cat2.position,
    transitionTime: 0.75,
    returnToOriginalView: false,
    enableZoom: true,
    zoomOptions: {
      zoomFactor: 1.8,
      minAperture: 0.15,
      maxAperture: 0.35,
      transitionStart: 0.7,
      transitionDuration: 1,
      holdDuration: 1.8,
    },
    criteria: { currentState: GAME_STATES.POST_VIEWMASTER },
    playOnce: true,
    priority: 100,
    delay: 3.0,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.CAT_DIALOG_CHOICE_2 });
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
      y: 1.5,
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

  phoneBoothMoveTo: {
    id: "phoneBoothMoveTo",
    type: "moveTo",
    description: "Move character into phone booth when player enters trigger",
    position: {
      x: sceneObjects.phonebooth.position.x,
      y: 1.2, // Y will be auto-calculated based on floor collider
      z: sceneObjects.phonebooth.position.z - 0.15,
    },
    rotation: {
      yaw: Math.PI, // Face the phone (90 degrees)
      pitch: 0,
    },
    transitionTime: 1.5,
    autoHeight: true, // Automatically calculate Y based on floor at X/Z
    inputControl: {
      disableMovement: true, // Disable movement
      disableRotation: false, // Allow rotation (player can look around)
    },
    restoreInput: false, // Keep movement disabled after animation completes
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
    delay: 0.3, // Wait 0.5 seconds before looking at phone booth
  },

  lookAndJump: {
    id: "lookAndJump",
    type: "jsonAnimation",
    path: "/json/look-and-jump.json",
    preload: false,
    description: "Camera animation for drive-by sequence",
    criteria: { currentState: GAME_STATES.DRIVE_BY },
    priority: 100,
    playOnce: true,
    restoreInput: true,
    delay: 1.0,
    scaleY: 0.425,
    playbackRate: 1.175,
    onComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.POST_DRIVE_BY });
    },
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
      { x: 4.73, y: 1.29, z: 79.05 },
    ],
    transitionTime: 1,
    lookAtHoldDuration: 4.0,
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

  edisonColorLookat: {
    id: "edisonColorLookat",
    type: "lookat",
    description: "Look at edison phonograph during viewmaster color phase",
    position: {
      x: sceneObjects.edison.position.x,
      y: sceneObjects.edison.position.y + 0.75,
      z: sceneObjects.edison.position.z,
    },
    transitionTime: 1.0,
    criteria: {
      currentState: GAME_STATES.VIEWMASTER_DISSOLVE,
    },
    priority: 100,
    playOnce: true,
  },

  viewmasterMoveTo: {
    id: "viewmasterMoveTo",
    type: "moveTo",
    description: "Move character near Viewmaster and look down at it",
    position: { x: -5.14, y: 2.15, z: 84.66 },
    lookat: sceneObjects.viewmaster.position, // Look at viewmaster (below eye level)
    transitionTime: 1.5,
    autoHeight: true, // Automatically calculate Y based on floor at X/Z
    inputControl: {
      disableMovement: true, // Disable movement
      disableRotation: false, // Allow rotation (player can look around)
    },
    criteria: { currentState: GAME_STATES.PRE_VIEWMASTER },
    priority: 100,
    playOnce: true,
  },

  candlestickPhoneLookat: {
    id: "candlestickPhoneLookat",
    type: "lookat",
    description: "Look at candlestick phone",
    position: sceneObjects.candlestickPhone.position,
    transitionTime: 1.0,
    criteria: { currentState: GAME_STATES.PRE_EDISON },
    priority: 100,
    playOnce: true,
    delay: 1.0,
  },

  candlestickPhoneLookatCzarStruggle: {
    id: "candlestickPhoneLookatCzarStruggle",
    type: "lookat",
    description: "Look at candlestick phone during Czar struggle",
    position: sceneObjects.candlestickPhone.position,
    transitionTime: 1.0,
    criteria: { currentState: GAME_STATES.CZAR_STRUGGLE },
    priority: 100,
    playOnce: true,
  },

  edisonMoveTo: {
    id: "edisonMoveTo",
    type: "moveTo",
    description: "Move character near Edison phonograph and look at it",
    position: { x: -5.14, y: 2.15, z: 84.66 },
    lookat: sceneObjects.edison.position,
    transitionTime: 1.5,
    autoHeight: true,
    inputControl: {
      disableMovement: true,
      disableRotation: false,
    },
    criteria: { currentState: GAME_STATES.EDISON },
    priority: 100,
    playOnce: true,
  },

  shoulderTap: {
    id: "shoulderTap",
    type: "lookat",
    position: (() => {
      let cachedPos = null;
      return (gameManager) => {
        if (cachedPos) return cachedPos;
        if (!gameManager?.characterController) {
          cachedPos = { x: 0, y: 1.5, z: 0 };
          return cachedPos;
        }
        const playerPos = gameManager.characterController.getPosition({
          x: 0,
          y: 0,
          z: 0,
        });
        const backward = new THREE.Vector3(0, 0, 1);
        const camera = gameManager.characterController.camera;
        if (camera) {
          backward.applyQuaternion(camera.quaternion);
          // Rotate slightly left (around up axis) to guarantee left rotation
          const leftRotation = new THREE.Quaternion().setFromAxisAngle(
            new THREE.Vector3(0, 1, 0),
            -Math.PI * 0.01 // ~18 degrees left
          );
          backward.applyQuaternion(leftRotation);
        }
        cachedPos = {
          x: playerPos.x + backward.x * 1.0,
          y: playerPos.y + 1.4,
          z: playerPos.z + backward.z * 1.0,
        };
        return cachedPos;
      };
    })(),
    transitionTime: 1.0,
    lookAtHoldDuration: 0,
    returnToOriginalView: false,
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
    preload: false,
    description: "Camera animation for punch-out sequence",
    criteria: {
      currentState: {
        $gte: GAME_STATES.PUNCH_OUT,
        $lt: GAME_STATES.LIGHTS_OUT,
      },
    },
    priority: 100,
    playOnce: true,
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
    delay: 0.05, // Sync with video punch impact
  },

  postCursorWhiteout: {
    id: "postCursorWhiteout",
    type: "fade",
    description: "Fade to white at POST_CURSOR, hold, then fade back",
    color: { r: 1, g: 1, b: 1 }, // White
    fadeInTime: 0.5, // Fade to white over 0.5 seconds
    holdTime: 1.0, // Hold at white for 0.5 seconds
    fadeOutTime: 2.0, // Fade back in over 2 seconds
    maxOpacity: 1.0, // Full white
    startFrom: "current", // Start from current fade opacity
    criteria: { currentState: GAME_STATES.POST_CURSOR },
    priority: 100,
    playOnce: true,
    onStart: (gameManager) => {
      // Hide shadowTrance videos when whiteout starts
      const videoManager = gameManager?.videoManager;
      if (videoManager) {
        const shadowTrance = videoManager.getVideoPlayer("shadowTrance");
        const shadowTranceSafari =
          videoManager.getVideoPlayer("shadowTranceSafari");
        if (shadowTrance) shadowTrance.setVisible(false);
        if (shadowTranceSafari) shadowTranceSafari.setVisible(false);
      }
    },
    // Note: OUTRO state is now set by postCursorColor desaturation effect onComplete
  },

  fallenBlackout: {
    id: "fallenBlackout",
    type: "fade",
    description:
      "Fade to black after falling, persists through FALLEN and LIGHTS_OUT states",
    color: { r: 0, g: 0, b: 0 },
    fadeInTime: 3.0,
    holdTime: 0,
    fadeOutTime: 0,
    maxOpacity: 1.0,
    persistWhileCriteria: true,
    criteria: {
      currentState: { $in: [GAME_STATES.FALLEN, GAME_STATES.LIGHTS_OUT] },
    },
    priority: 100,
    playOnce: true,
    delay: 1.5,
    onFadeInComplete: (gameManager) => {
      gameManager.setState({ currentState: GAME_STATES.LIGHTS_OUT });

      if (gameManager.characterController) {
        gameManager.characterController.resetToUpright();
      }

      // WAKING_UP state will be triggered by TimePassesSequence when it completes
    },
  },

  lightsOutMoveTo: {
    id: "lightsOutMoveTo",
    type: "moveTo",
    description:
      "Move player to consistent position/rotation during blackout (unseen behind fade)",
    position: { x: -5.14, y: 2.05, z: 83.66 }, // From WAKING_UP debug spawn
    rotation: { yaw: 0, pitch: 0 }, // From WAKING_UP debug spawn
    transitionTime: 0.1,
    autoHeight: false, // Use exact Y position
    inputControl: {
      disableMovement: true,
      disableRotation: true,
    },
    restoreInput: false, // Keep inputs disabled (will be restored by wakingUp animation)
    criteria: { currentState: GAME_STATES.LIGHTS_OUT },
    priority: 100,
    playOnce: true,
    delay: 0.5, // Start after fade begins but before it completes
  },

  wakingUpFadeIn: {
    id: "wakingUpFadeIn",
    type: "fade",
    description: "Fade from black back to vision when waking up",
    color: { r: 0, g: 0, b: 0 },
    fadeInTime: 0,
    holdTime: 0,
    fadeOutTime: 8.0,
    maxOpacity: 1.0,
    startFrom: "current",
    criteria: { currentState: GAME_STATES.WAKING_UP },
    priority: 100,
    playOnce: true,
  },

  wakingUp: {
    id: "wakingUp",
    type: "jsonAnimation",
    path: "/json/waking-up.json",
    preload: false,
    description:
      "Passing out animation blending with player movement during wake up",
    criteria: { currentState: GAME_STATES.WAKING_UP },
    priority: 90,
    playOnce: true,
    blendWithPlayer: true,
    blendAmount: 0.25,
    restoreInput: {
      movement: false, // Keep movement disabled after animation
      rotation: true, // Restore rotation after animationfalse
    },
    playNext: "leclaireLookat",
    duration: 6.0,
  },

  leclaireLookat: {
    id: "leclaireLookat",
    type: "lookat",
    description: "Look at LeClaire",
    position: videos.hesTiedUsUp.position,
    transitionTime: 1.0,
    priority: 110,
    playNext: "shadowUnkindLookat",
    enableZoom: true,
    lookAtHoldDuration: 1.5,
    zoomOptions: {
      zoomFactor: 2.0,
      transitionStart: 0.6,
      transitionDuration: 1.5,
      holdDuration: 1.5,
    },
  },

  shadowUnkindLookat: {
    id: "shadowUnkindLookat",
    type: "lookat",
    description: "Look at shadow unkind video",
    position: videos.soUnkind.position,
    transitionTime: 1.0,
    priority: 100,
    playOnce: true,
    enableZoom: true,
    restoreInput: {
      movement: false,
      rotation: true,
    },
    zoomOptions: {
      zoomFactor: 2.0,
      minAperture: 0.2,
      maxAperture: 0.35,
      transitionStart: 0.6,
      transitionDuration: 1.5,
      holdDuration: 6.0,
    },
  },

  shadowAmplificationsLookat: {
    id: "shadowAmplificationsLookat",
    type: "lookat",
    description: "Look at shadowAmplifications video when it appears",
    position: videos.shadowAmplifications.position,
    transitionTime: 1.0,
    priority: 100,
    playOnce: false,
    fireOnEvent: "video:play:shadowAmplifications",
    enableZoom: true,
    restoreInput: {
      movement: false,
      rotation: true,
    },
    zoomOptions: {
      zoomFactor: 2.0,
      minAperture: 0.2,
      maxAperture: 0.35,
      transitionStart: 0.6,
      transitionDuration: 1.5,
      holdDuration: 7.0,
    },
  },

  amplifierLookat: {
    id: "amplifierLookat",
    type: "lookat",
    description:
      "Look at amplifier, quick glance to shadow, then back to amplifier",
    positions: [
      sceneObjects.amplifier.position, // Amplifier position
      videos.shadowAmplifications.position,
      sceneObjects.amplifier.position, // Back to amplifier (quick glance)
    ],
    transitionTime: 1.0,
    lookAtHoldDuration: 2.0, // Hold at amplifier initially
    returnToOriginalView: false,
    priority: 100,
    playOnce: false, // Allow retry if object not loaded yet
    fireOnEvent: "shadow:amplifications",
    enableZoom: true,
    restoreInput: {
      movement: false,
      rotation: true,
    },
    zoomOptions: {
      zoomFactor: 1.5,
      minAperture: 0.2,
      maxAperture: 0.35,
      transitionStart: 0.6,
      transitionDuration: 0.75,
      holdDuration: 1.5,
    },
    sequenceSettings: [
      null, // Position 0: use defaults (amplifier with zoom)
      {
        // Position 1: quick glance at shadow (zoom, fast transition)
        transitionTime: 1.0,
        lookAtHoldDuration: 4.5,
        zoomOptions: {
          zoomFactor: 2.0,
          minAperture: 0.2,
          maxAperture: 0.35,
          transitionStart: 0.6,
          transitionDuration: 1.5,
          holdDuration: 4.5,
        },
      },
      {
        // Position 2: quick glance back to amplifier (no zoom, fast transition)
        transitionTime: 1.0,
        lookAtHoldDuration: 1.0,
        enableZoom: false,
      },
    ],
  },

  cursorHesTiedUsUpLookat: {
    id: "cursorHesTiedUsUpLookat",
    type: "lookat",
    description:
      "Look at hesTiedUsUp video position when entering CURSOR state",
    position: videos.hesTiedUsUp.position,
    transitionTime: 1.0,
    lookAtHoldDuration: 2.0,
    criteria: { currentState: GAME_STATES.CURSOR },
    priority: 100,
    playOnce: true,
    // playNext: "shadowTranceLookat", // shadowTrance video is commented out
  },

  // shadowTranceLookat: {
  //   id: "shadowTranceLookat",
  //   type: "lookat",
  //   description: "Look at shadow trance video",
  //   position: videos.shadowTrance.position, // shadowTrance video is commented out
  //   transitionTime: 1.0,
  //   priority: 105,
  //   lookAtHoldDuration: 2.0,
  //   enableZoom: false,
  //   restoreInput: {
  //     movement: true,
  //     rotation: true,
  //   },
  // },

  woozy: {
    id: "woozy",
    type: "jsonAnimation",
    path: "/json/woozy.json",
    preload: false,
    description:
      "Woozy camera animation blending with player movement during viewmaster overheat",
    criteria: {
      currentState: { $in: [GAME_STATES.CURSOR, GAME_STATES.CURSOR_FINAL] },
      // Allow woozy effect when intensity is high (works whether mask is equipped or just removed)
      viewmasterInsanityIntensity: { $gte: 0.85 },
    },
    priority: 95,
    playOnce: false,
    restoreInput: true,
    blendWithPlayer: true,
    blendAmount: 0.8,

    playbackPercentage: 0.5,
  },

  runeLookat: {
    id: "runeLookat",
    type: "lookat",
    description: "Look at rune when sighted near screen edge",
    position: (gameManager) => {
      // Get rune position from drawing manager
      const drawingManager = window.drawingManager;
      if (!drawingManager || !drawingManager.currentGoalRune) {
        return { x: 0, y: 0, z: 0 }; // Fallback
      }
      const rune = drawingManager.currentGoalRune;
      if (!rune || !rune.mesh) {
        return { x: 0, y: 0, z: 0 }; // Fallback
      }
      const worldPosition = new THREE.Vector3();
      rune.mesh.getWorldPosition(worldPosition);
      return { x: worldPosition.x, y: worldPosition.y, z: worldPosition.z };
    },
    transitionTime: 1.0,
    lookAtHoldDuration: 2.0, // Hold on rune for 2 seconds
    returnToOriginalView: false,
    enableZoom: true,
    restoreInput: false, // Don't restore yet - will restore after portal lookat
    zoomOptions: {
      zoomFactor: 1.3,
      minAperture: 0.15,
      maxAperture: 0.35,
      transitionStart: 0.0,
      transitionDuration: 1.0,
      holdDuration: 3.0, // Hold zoom for 2 seconds (hold on rune longer)
      disableDoF: true, // Disable DoF effect, only use zoom
    },
    playNext: "portalLookat",
    priority: 100,
    criteria: {
      currentState: { $in: [GAME_STATES.CURSOR, GAME_STATES.CURSOR_FINAL] },
      sawRune: true,
    },
    playOnce: false, // Allow multiple rune sightings
    // Note: onComplete is called before playNext, so we reset sawRune after a delay
    // to allow playNext to trigger without criteria interference
    onComplete: (gameManager) => {
      // Reset sawRune flag after a short delay to allow playNext to trigger
      // The playNext animation doesn't use criteria, so this is safe
      setTimeout(() => {
        gameManager.setState({ sawRune: false });
      }, 100);
    },
  },

  portalLookat: {
    id: "portalLookat",
    type: "lookat",
    description: "Look at particle portal after rune lookat",
    position: (gameManager) => {
      // Get portal (canvas) position from drawing manager
      const drawingManager = window.drawingManager;
      if (!drawingManager) {
        return { x: 0, y: 1.5, z: -2 }; // Default fallback
      }

      // Try to get world position from canvas mesh
      const canvasMesh = drawingManager.canvasMesh;
      if (canvasMesh) {
        const worldPos = new THREE.Vector3();
        canvasMesh.getWorldPosition(worldPos);
        return { x: worldPos.x, y: worldPos.y, z: worldPos.z };
      }

      // Fallback to stored canvas position
      return drawingManager.canvasPosition || { x: 0, y: 1.5, z: -2 };
    },
    transitionTime: 1.0,
    returnToOriginalView: false,
    enableZoom: false,
    restoreInput: true, // Restore input after portal lookat
    delay: 0, // No additional delay - playNextDelay already accounts for hold duration
    priority: 100,
    onStart: (gameManager) => {
      // Unequip viewmaster when portal lookat starts
      const currentState = gameManager?.getState();
      if (currentState?.isViewmasterEquipped) {
        gameManager.setState({ isViewmasterEquipped: false });
      }
    },
  },

  catChewLookat: {
    id: "catChewLookat",
    type: "lookat",
    description: "Look at cat chew video when headset comes off after glitch",
    position: videos.catChew.position,
    transitionTime: 1.0,
    returnToOriginalView: false,
    enableZoom: true,
    restoreInput: {
      movement: false,
      rotation: true,
    },
    zoomOptions: {
      zoomFactor: 2.0,
      minAperture: 0.15,
      maxAperture: 0.35,
      transitionStart: 0.6,
      transitionDuration: 1.5,
      holdDuration: 3.0,
    },
    criteria: { currentState: GAME_STATES.CAT_SAVE },
    playOnce: true,
    priority: 100,
  },

  letterLookat: {
    id: "letterLookat",
    type: "lookat",
    description: "Look at animated letter plane during outro",
    position: (gameManager) => {
      // Get letter object from sceneManager
      const letterObject = gameManager?.sceneManager?.getObject("letter");
      if (!letterObject) {
        // Letter not loaded yet - return a position in front of camera as fallback
        // This will keep the lookat active until the letter loads
        const camera = gameManager?.characterController?.camera;
        if (camera) {
          const forward = new THREE.Vector3(0, 0, -1);
          forward.applyQuaternion(camera.quaternion);
          return {
            x: camera.position.x + forward.x * 4,
            y: camera.position.y + forward.y * 4,
            z: camera.position.z + forward.z * 4,
          };
        }
        return { x: 0, y: 0, z: -4 };
      }

      // Check if letter is reparented to camera - if so, stop tracking (lookat should complete)
      const camera = gameManager?.characterController?.camera;
      if (camera && letterObject.parent === camera) {
        // Letter is reparented to camera - return current camera position to stop tracking
        // This will cause the lookat to complete since the target is at camera position
        return {
          x: camera.position.x,
          y: camera.position.y,
          z: camera.position.z,
        };
      }

      // Letter is not parented to camera - use normal world position calculation
      // Force update world matrix of the container first (cascades to all children)
      // GLTF animations update object matrices, so we need to update world matrix first
      letterObject.updateMatrixWorld(true);

      // Find the "Plane" mesh - that's what's being animated
      let planeMesh = null;
      letterObject.traverse((child) => {
        if (!planeMesh && child.isMesh && child.name === "Plane") {
          planeMesh = child;
        }
      });

      // Use the Plane mesh if found, otherwise fall back to container
      const targetObject = planeMesh || letterObject;

      if (!targetObject) {
        logger.error("No target object found for letter lookat");
        return { x: 0, y: 0, z: -4 };
      }

      // Get world position of the target object
      // World matrix is already updated above
      const worldPos = new THREE.Vector3();
      targetObject.getWorldPosition(worldPos);

      // Validate position
      if (
        !isFinite(worldPos.x) ||
        !isFinite(worldPos.y) ||
        !isFinite(worldPos.z)
      ) {
        logger.warn(
          `Invalid world position for letter: (${worldPos.x}, ${worldPos.y}, ${worldPos.z})`
        );
        // Fallback to container position
        worldPos.copy(letterObject.position);
      }

      return { x: worldPos.x, y: worldPos.y, z: worldPos.z };
    },
    transitionTime: 0.5,
    returnToOriginalView: false,
    restoreInput: true, // Restore input when lookat completes
    criteria: { currentState: GAME_STATES.OUTRO },
    playOnce: true,
    priority: 100,
    delay: 1.0, // Wait for letter to load
  },

  paSpeakerLookat: {
    id: "paSpeakerLookat",
    type: "lookat",
    description: "Look at PA speaker during outro",
    position: sceneObjects.paSpeaker.position,
    transitionTime: 1.0,
    returnToOriginalView: false,
    restoreInput: true,
    enableZoom: true,
    delay: 2.0,
    zoomOptions: {
      zoomFactor: 2.0,
      minAperture: 0.2,
      maxAperture: 0.35,
      transitionStart: 0.6,
      transitionDuration: 1.5,
      holdDuration: 5.0,
    },
    criteria: { currentState: GAME_STATES.OUTRO_CZAR },
    playOnce: true,
    priority: 100,
  },

  catOutroLookat: {
    id: "catOutroLookat",
    type: "lookat",
    description: "Look at cat outro video",
    position: videos.cat2Loop.position,
    transitionTime: 1.0,
    returnToOriginalView: false,
    enableZoom: true,
    restoreInput: {
      movement: false,
      rotation: true,
    },
    zoomOptions: {
      zoomFactor: 2.0,
      minAperture: 0.15,
      maxAperture: 0.35,
      transitionStart: 0.6,
      transitionDuration: 1.5,
      holdDuration: 3.0,
    },
    criteria: { currentState: GAME_STATES.OUTRO_CAT },
    playOnce: true,
    priority: 100,
    onComplete: (gameManager) => {
      // After lookat completes, wait 2s then transition to OUTRO_CZAR
      setTimeout(() => {
        gameManager.setState({ currentState: GAME_STATES.OUTRO_CZAR });
      }, 2000);
    },
  },

  creditsLookat: {
    id: "creditsLookat",
    type: "lookat",
    description: "Look at credits video",
    position: videos.credits.position,
    transitionTime: 1.0,
    returnToOriginalView: false,
    enableZoom: true,
    restoreInput: true,
    zoomOptions: {
      zoomFactor: 2.0,
      minAperture: 0.15,
      maxAperture: 0.35,
      transitionStart: 0.6,
      transitionDuration: 1.5,
      holdDuration: 5.0,
    },
    criteria: { currentState: GAME_STATES.OUTRO_CREDITS },
    playOnce: true,
    priority: 100,
  },
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
