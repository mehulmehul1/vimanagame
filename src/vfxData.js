/**
 * VFX Data - Centralized VFX effect definitions
 *
 * All VFX systems are configured here with their state-based behaviors.
 * Each VFX type has its own section with effect definitions.
 *
 * Structure:
 * - Each effect has: id, parameters, criteria, priority, delay (optional)
 * - parameters: VFX-specific settings (opacity, color, speed, etc.)
 * - criteria: Game state conditions using the criteria helper
 * - priority: Higher priority effects override lower ones
 * - delay: Optional delay in seconds before applying effect after criteria is met
 *
 * Usage:
 * import { getVfxEffectForState } from './vfxData.js';
 * const effect = getVfxEffectForState('desaturation', gameState);
 */

import { GAME_STATES } from "./gameData.js";
import { checkCriteria } from "./utils/criteriaHelper.js";
import { findMatchingEffect } from "./vfxManager.js";

/**
 * Desaturation Effect
 * Controls color-to-grayscale transitions
 */
export const desaturationEffects = {
  // Default: Color for most of the game
  defaultColor: {
    id: "defaultColor",
    parameters: {
      target: 0.0, // Full color
      duration: 2.0,
      mode: "fade",
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.START_SCREEN,
        $lt: GAME_STATES.POST_DRIVE_BY,
      },
    },
    priority: 0,
  },

  preOfficeGrayscale: {
    id: "preOfficeGrayscale",
    parameters: {
      target: 1.0, // Grayscale
      duration: 0.0,
      mode: "fade",
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.POST_DRIVE_BY,
        $lte: GAME_STATES.VIEWMASTER_COLOR,
      },
    },
    priority: 0,
  },

  officeColor: {
    id: "officeColor",
    parameters: {
      target: 0.0, // Full color
      duration: 5.0,
      mode: "bleed",
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.VIEWMASTER,
      },
    },
    priority: 20,
    delay: 3.0, // Wait 2 seconds after criteria is met before applying
  },
};

/**
 * Cloud Particles Effect
 * Controls fog/particle system behavior
 */
export const cloudParticleEffects = {
  // Example effects - customize based on your needs
  defaultFog: {
    id: "defaultFog",
    parameters: {
      opacity: 0.01,
      windSpeed: -0.5,
      particleCount: 22000,
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.START_SCREEN,
        $lt: GAME_STATES.OFFICE_INTERIOR,
      },
    },
    priority: 0,
  },

  // Add more cloud particle effects here as needed
  // heavyFog: {
  //   id: "heavyFog",
  //   parameters: {
  //     opacity: 0.02,
  //     windSpeed: -0.8,
  //   },
  //   criteria: {
  //     currentState: GAME_STATES.SOME_STATE,
  //   },
  //   priority: 10,
  // },
};

/**
 * Title Sequence Effect
 * Controls title animation behavior
 */
export const titleSequenceEffects = {
  // Add title sequence effects here
  // mainTitle: {
  //   id: "mainTitle",
  //   parameters: {
  //     visible: true,
  //     fadeSpeed: 1.0,
  //   },
  //   criteria: {
  //     currentState: GAME_STATES.TITLE_SEQUENCE,
  //   },
  //   priority: 0,
  // },
};

/**
 * Splat Fractal Effect
 * Applies shader-based fractal effects to splat meshes
 */
export const splatFractalEffects = {
  // Apply waves effect to BOTH interior and office-hell during VIEWMASTER_HELL
  // Ramps from 0 to 0.8 intensity over 4 seconds
  hellWaves: {
    id: "hellWaves",
    parameters: {
      effectType: "waves",
      intensity: 0.8,
      rampDuration: 4.0, // Ramp up over 4 seconds before morph
      targetMeshIds: ["interior", "officeHell"], // Apply to both splats
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.VIEWMASTER_HELL,
      },
    },
    priority: 10,
  },
};

/**
 * Splat Morph Effect
 * Controls morphing transition between two splat scenes
 */
export const splatMorphEffects = {
  // Trigger the morph transition during VIEWMASTER_HELL state
  hellTransition: {
    id: "hellTransition",
    parameters: {
      speedMultiplier: 1.0,
      staySeconds: 4, // How long to stay on current splat before transitioning
      transitionSeconds: 4.0, // Duration of the morph transition
      randomRadius: 8.0, // Radius of scatter cloud (about the size of the office room)
      scatterCenter: { x: -5.14, y: 3.05, z: 84.66 }, // Center point of scatter cloud
      trigger: "start", // Start the transition
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.VIEWMASTER_HELL,
      },
    },
    priority: 10,
  },
};

/**
 * All VFX effects organized by type
 */
export const vfxEffects = {
  desaturation: desaturationEffects,
  cloudParticles: cloudParticleEffects,
  titleSequence: titleSequenceEffects,
  splatFractal: splatFractalEffects,
  splatMorph: splatMorphEffects,
};

/**
 * Get VFX effect for a specific type based on game state
 * @param {string} vfxType - Type of VFX (e.g., 'desaturation', 'cloudParticles')
 * @param {Object} gameState - Current game state
 * @returns {Object|null} Matching effect or null
 */
export function getVfxEffectForState(vfxType, gameState) {
  const effects = vfxEffects[vfxType];
  if (!effects) {
    console.warn(`Unknown VFX type: ${vfxType}`);
    return null;
  }
  return findMatchingEffect(effects, gameState, checkCriteria);
}

/**
 * Convenience functions for specific VFX types
 */
export function getDesaturationForState(gameState) {
  return getVfxEffectForState("desaturation", gameState);
}

export function getCloudParticleEffectForState(gameState) {
  return getVfxEffectForState("cloudParticles", gameState);
}

export function getTitleSequenceEffectForState(gameState) {
  return getVfxEffectForState("titleSequence", gameState);
}

export function getSplatFractalEffectForState(gameState) {
  return getVfxEffectForState("splatFractal", gameState);
}

export function getSplatMorphEffectForState(gameState) {
  return getVfxEffectForState("splatMorph", gameState);
}

export default vfxEffects;
