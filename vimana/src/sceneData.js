/**
 * sceneData.js - VIMANA SCENE DEFINITIONS
 * =============================================================================
 *
 * ROLE: Defines all scenes/zones for the Vimana game
 *
 * SCENES:
 * - music_room: The main music room environment
 *
 * =============================================================================
 */

import { GAME_STATES } from './gameData.js';

/**
 * Scene objects that can be loaded based on game state
 */
export const sceneObjects = [
  {
    // ================================================================
    // MUSIC ROOM - Main game scene
    // ================================================================
    id: 'music_room',
    name: 'Music Room',
    type: 'gltf',  // GLB/GLTF model

    // File paths - PLACE YOUR FILES HERE:
    path: '/assets/models/musicroom.glb',  // Export from Blender: File > Export > glTF 2.0 (.glb)

    // Position in world
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 },

    // Keep lights from GLB model (don't remove them)
    keepLights: true,

    // Player spawn point
    spawn: {
      position: { x: 0, y: 0, z: 0 },
      rotation: { x: 0, y: 0, z: 0 }
    },

    // State criteria for when this loads
    criteria: {
      currentState: GAME_STATES.MUSIC_ROOM,
    },

    // Optional: Environment map for reflections (DISABLED to debug brightness)
    // envMapCenter: { x: 0, y: 1, z: 0 },
    // envMapRadius: 15,
  },
];

/**
 * Import helper for state checking
 */
export function getGameState() {
  return window.gameManager?.getState() || {};
}
