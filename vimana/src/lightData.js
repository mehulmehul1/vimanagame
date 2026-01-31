/**
 * lightData.js - VIMANA LIGHTING DEFINITIONS
 * =============================================================================
 *
 * ROLE: Light definitions for the music room
 *
 * NOTE: Lights are now loaded from the GLB model (keepLights: true in sceneData.js)
 * This file is kept empty - no additional lights defined here.
 *
 * =============================================================================
 */

import { GAME_STATES } from './gameData.js';

/**
 * Export lights array for LightManager
 * Empty - all lights come from the GLB model
 */
export const lights = [
    // Fallback Ambient Light REMOVED to debug brightness
    // {
    //     type: 'AmbientLight',
    //     color: 0xffffff,
    //     intensity: 0.5,
    // },
    // Fallback Directional Light RESTORED
    {
        type: 'DirectionalLight',
        color: 0xfff0dd,
        intensity: 1.0,
        position: { x: 5, y: 10, z: 5 },
        castShadow: true,
    },

];
