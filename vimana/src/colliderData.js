/**
 * colliderData.js - VIMANA COLLIDER DEFINITIONS
 * =============================================================================
 *
 * ROLE: Physics colliders for the music room
 *
 * COLLIDERS:
 * - floor: Basic floor for walking
 *
 * =============================================================================
 */

import { GAME_STATES } from './gameData.js';

/**
 * Export colliders array for PhysicsManager
 */
export const colliders = [
  {
    // Floor collider - Half-extents for Rapier cuboid
    // Position at y=-0.5 with half-height 0.5 gives floor surface at y=0
    id: 'music_room_floor',
    type: 'box',
    position: { x: 0, y: -0.5, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    dimensions: { x: 50, y: 0.5, z: 50 },
    blocking: true,

    // Use default collision groups (0xffffffff) - collides with everything

    criteria: {
      currentState: GAME_STATES.MUSIC_ROOM,
    },
  },
];
