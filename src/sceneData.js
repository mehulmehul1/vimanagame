/**
 * Scene Data Structure
 *
 * Defines scene objects like splat meshes and GLTF models.
 *
 * Each object contains:
 * - id: Unique identifier
 * - type: Type of object ('splat', 'gltf', etc.)
 * - path: Path to the asset file
 * - position: {x, y, z} position in 3D space
 * - rotation: {x, y, z} rotation in radians (Euler angles)
 * - scale: {x, y, z} scale multipliers or uniform number
 * - description: Human-readable description
 * - options: Type-specific options
 *   - useContainer: (GLTF) Wrap model in a container group
 *   - visible: If false, object will be invisible (useful for animation helpers)
 *   - physicsCollider: (GLTF) If true, create a Rapier trimesh collider from mesh geometry
 *   - debugMaterial: (GLTF) If true, apply semi-transparent wireframe material for debugging
 * - criteria: Optional object with key-value pairs that must match game state
 *   - Simple equality: { currentState: GAME_STATES.CHAPTER_2 }
 *   - Comparison operators: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.DRIVE_BY } }
 *   - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
 * - loadByDefault: If true, load regardless of state (default: false)
 * - priority: Higher priority objects are loaded first (default: 0)
 * - gizmo: If true, enable debug gizmo for positioning visual objects (G=move, R=rotate, S=scale)
 * - animations: Array of animation definitions (for GLTF objects with animations)
 *   - id: Unique identifier for this animation
 *   - clipName: Name of animation clip in GLTF (null = use first clip)
 *   - loop: Whether to loop the animation
 *   - criteria: Optional object with key-value pairs that must match game state for animation to play
 *     - Simple equality: { currentState: GAME_STATES.ANSWERED_PHONE }
 *     - Comparison operators: { currentState: { $gte: GAME_STATES.INTRO, $lt: GAME_STATES.DRIVE_BY } }
 *     - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
 *     - If criteria matches and not playing → play it
 *     - If criteria doesn't match and playing → stop it
 *   - autoPlay: If true, automatically play when criteria are met
 *   - playOnce: If true, only play once per game session
 *   - timeScale: Playback speed (1.0 = normal)
 *   - removeObjectOnFinish: If true, remove parent object from scene when animation finishes
 *
 * Usage:
 * import { sceneObjects, getSceneObjectsForState } from './sceneData.js';
 */

import { GAME_STATES } from "./gameData.js";
import { checkCriteria } from "./criteriaHelper.js";
import { Logger } from "./utils/logger.js";

// Create module-level logger
const logger = new Logger("SceneData", false);

const interiorPosition = { x: 4.55, y: -0.22, z: 78.51 };
const interiorRotation = { x: -3.1416, y: 1.0358, z: -3.1416 };

export const sceneObjects = {
  exterior: {
    id: "exterior",
    type: "splat",
    path: "/exterior-edit.sog",
    description: "Main exterior environment splat mesh",
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
    quaternion: { x: 1, y: 0, z: 0, w: 0 },
    priority: 100, // Load first
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
    },
  },

  interior: {
    id: "interior",
    type: "splat",
    path: "/stairs-and-green-room.sog",
    description: "Main exterior environment splat mesh",
    position: interiorPosition,
    rotation: { x: 0.0, y: -1.0385, z: -3.1416 },
    scale: { x: 1, y: 1, z: 1 },
    priority: 100, // Load first
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },

  phonebooth: {
    id: "phonebooth",
    type: "gltf",
    path: "/gltf/phonebooth.glb",
    description:
      "Phone booth GLTF model with CordAttach and Receiver children (uses PhoneCord module)",
    position: { x: 5.94, y: -0.76, z: 65.76 },
    rotation: { x: 0, y: Math.PI / 2, z: 0 }, // 90 degrees around Y axis
    scale: { x: 1.25, y: 1.25, z: 1.25 },
    options: {
      // Create a container group for proper scaling
      useContainer: true,
    },
    priority: 50,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
    },
    animations: [
      {
        id: "phonebooth-ring", // Identifier for this animation
        clipName: null, // null = use first animation clip from GLTF
        loop: false, // Whether the animation should loop
        criteria: {
          currentState: {
            $gte: GAME_STATES.ANSWERED_PHONE,
            $lt: GAME_STATES.DIALOG_CHOICE_1,
          },
        },
        autoPlay: true, // Automatically play when criteria are met
        playOnce: true, // Only play once per session
        timeScale: 1.0, // Playback speed (1.0 = normal)
      },
    ],
  },

  radio: {
    id: "radio",
    type: "gltf",
    path: "/gltf/radio-1.glb",
    description: "Radio GLTF model",
    position: { x: 4.04, y: -0.28, z: 35.45 },
    rotation: { x: 3.1293, y: 1.1503, z: -2.9357 },
    scale: { x: 2.46, y: 2.46, z: 2.46 },
    options: {
      useContainer: true,
    },
    priority: 50,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
    },
  },

  car: {
    id: "car",
    type: "gltf",
    path: "/gltf/Old_Car_01.glb",
    description: "Car GLTF model",
    position: { x: -15.67, y: -0.425, z: 62.5 },
    rotation: { x: 0.0, y: 0.8859, z: 0.0 },
    scale: { x: 0.9, y: 0.9, z: 0.9 },
    options: {
      useContainer: true,
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.ANSWERED_PHONE,
        $lt: GAME_STATES.POST_DRIVE_BY,
      },
    },
    priority: 50,
    animations: [
      {
        id: "drive-by-anim",
        clipName: null,
        loop: false,
        autoPlay: true,
        timeScale: 0.475,
        removeObjectOnFinish: true, // Despawn car after animation completes
        criteria: {
          currentState: {
            $gte: GAME_STATES.DRIVE_BY_PREAMBLE,
            $lt: GAME_STATES.POST_DRIVE_BY,
          },
        },
      },
    ],
  },

  carTestSplat: {
    id: "carTestSplat",
    type: "splat",
    path: "/car-test.sog",
    description: "Car test splat - positioned at origin for testing",
    position: { x: 8.42, y: -0.37, z: 32.46 },
    rotation: { x: -2.6226, y: 1.2478, z: -0.5263 },
    scale: { x: 2.92, y: 2.92, z: 2.92 },
    priority: 100,
    criteria: {
      currentState: {
        $gte: GAME_STATES.LOADING,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
    },
  },

  officeCollider: {
    id: "officeCollider",
    type: "gltf",
    path: "/gltf/colliders/office-collider-v01.glb",
    description: "Office interior physics collider mesh",
    position: interiorPosition,
    rotation: interiorRotation,
    scale: { x: 1, y: 1, z: 1 },
    options: {
      useContainer: true,
      visible: false, // Set to true to see debug material
      physicsCollider: true, // Flag to create physics trimesh collider
      debugMaterial: false, // Apply semi-transparent debug material to visualize collider
    },
    loadByDefault: false,
    priority: 90,
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },

  coneCurve: {
    id: "coneCurve",
    type: "gltf",
    path: "/gltf/ConeCurve.glb",
    description: "Camera curve animation for start screen",
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },
    options: {
      useContainer: true,
      visible: false, // Hide the cone - it's just a helper for camera animation
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.START_SCREEN,
        $lt: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
      },
    },
    priority: 100,
    animations: [
      {
        id: "coneCurve-anim",
        clipName: null, // Use first animation clip
        loop: true,
        autoPlay: true,
        timeScale: 0.15,
        criteria: {
          currentState: {
            $gte: GAME_STATES.START_SCREEN,
            $lt: GAME_STATES.TITLE_SEQUENCE_COMPLETE,
          },
        },
      },
    ],
  },

  viewmaster: {
    id: "viewmaster",
    type: "gltf",
    path: "/gltf/viewmaster.glb",
    preload: false,
    description: "Viewmaster GLTF model",
    position: { x: -7.01, y: 1.37, z: 88.8 },
    rotation: { x: 0.2324, y: 0.0, z: 0.2949 },
    scale: { x: 1, y: 1, z: 1 },
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },

  edison: {
    id: "edison",
    type: "gltf",
    path: "/gltf/edison.glb",
    preload: false,
    description: "edison cylinder player",
    position: { x: -3.66, y: 1.09, z: 89.43 },
    rotation: { x: 3.1416, y: -0.0245, z: 3.1416 },
    scale: { x: 1.32, y: 1.32, z: 1.32 },
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
  },

  candlestickPhone: {
    id: "candlestickPhone",
    type: "gltf",
    path: "/gltf/candlestickPhone.glb",
    preload: false,
    description:
      "Candlestick phone with CordAttach and Receiver children (uses PhoneCord module)",
    position: { x: -4.69, y: 1.1, z: 85.05 },
    rotation: { x: -0.0001, y: 1.5114, z: 0.0001 },
    scale: { x: 1.32, y: 1.32, z: 1.32 },
    gizmo: true,
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
      },
    },
    // Note: This object has a companion script (src/content/candlestickPhone.js)
    // The script manages the CordAttach and Receiver children using the PhoneCord module
    // Initialize in main.js or scene manager after the object is loaded
  },

  //   firstPersonBody: {
  //     id: "firstPersonBody",
  //     type: "gltf",
  //     path: "/gltf/mixamo-test-rig.glb",
  //     description: "First-person body model (will be reparented to camera)",
  //     position: { x: 0, y: 0, z: 0 }, // Will be repositioned when attached to camera
  //     rotation: { x: 0, y: 0, z: 0 }, // Will be reoriented when attached to camera
  //     scale: 1.0,
  //     options: {
  //       useContainer: true,
  //     },
  //     loadByDefault: true,
  //     priority: 90, // Load early, before most objects but after exterior
  //     animations: [
  //       {
  //         id: "firstPersonBody-walk",
  //         clipName: "Walking (1)_3", // The walking animation
  //         loop: true,
  //         autoPlay: false, // CharacterController will manage this manually
  //         timeScale: 1.0,
  //       },
  //       {
  //         id: "firstPersonBody-idle",
  //         clipName: "Idle_1", // The idle animation
  //         loop: true,
  //         autoPlay: false, // CharacterController will manage this manually
  //         timeScale: 1.0,
  //       },
  // //     ],
  //   },
};

/**
 * Get scene objects that should be loaded for the current game state
 * @param {Object} gameState - Current game state
 * @returns {Array<Object>} Array of scene objects that should be loaded
 */
export function getSceneObjectsForState(gameState) {
  // Convert to array and sort by priority (descending)
  const sortedObjects = Object.values(sceneObjects).sort(
    (a, b) => (b.priority || 0) - (a.priority || 0)
  );

  const matchingObjects = [];

  for (const obj of sortedObjects) {
    // Always include objects marked as loadByDefault
    if (obj.loadByDefault === true) {
      matchingObjects.push(obj);
      continue;
    }

    // Check criteria (supports operators like $gte, $lt, etc.)
    if (obj.criteria) {
      const matches = checkCriteria(gameState, obj.criteria);
      if (!matches) {
        logger.log(
          `${obj.id} does NOT match criteria (state=${gameState.currentState})`
        );
        continue;
      }
    }

    // If we get here, all conditions passed
    matchingObjects.push(obj);
  }

  return matchingObjects;
}

/**
 * Get a scene object by ID
 * @param {string} id - Object ID
 * @returns {Object|null} Scene object data or null if not found
 */
export function getSceneObject(id) {
  return sceneObjects[id] || null;
}

/**
 * Get all scene object IDs
 * @returns {Array<string>} Array of all object IDs
 */
export function getAllSceneObjectIds() {
  return Object.keys(sceneObjects);
}

/**
 * Get all objects of a specific type
 * @param {string} type - Object type ('splat', 'gltf', etc.)
 * @returns {Array<Object>} Array of scene objects matching the type
 */
export function getSceneObjectsByType(type) {
  return Object.values(sceneObjects).filter((obj) => obj.type === type);
}

export default sceneObjects;
