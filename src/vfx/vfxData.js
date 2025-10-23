/**
 * VFX Data - Centralized VFX effect definitions
 *
 * All VFX systems are configured here with their state-based behaviors.
 * Each VFX type has its own section with effect definitions.
 *
 * Structure:
 * - Each effect has: id, parameters, criteria, priority
 * - parameters: VFX-specific settings (opacity, color, speed, etc.)
 * - criteria: Game state conditions using the criteria helper
 * - priority: Higher priority effects override lower ones
 *
 * Usage:
 * import { getVfxEffectForState } from './vfxData.js';
 * const effect = getVfxEffectForState('desaturation', gameState);
 */

import { GAME_STATES } from "../gameData.js";
import { checkCriteria } from "../utils/criteriaHelper.js";
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
        $lt: GAME_STATES.VIEWMASTER,
      },
    },
    priority: 0,
  },

  officeColor: {
    id: "officeColor",
    parameters: {
      target: 0.0, // Full color
      duration: 2.0,
      mode: "bleed",
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.VIEWMASTER,
      },
    },
    priority: 20,
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
 * All VFX effects organized by type
 */
export const vfxEffects = {
  desaturation: desaturationEffects,
  cloudParticles: cloudParticleEffects,
  titleSequence: titleSequenceEffects,
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

export default vfxEffects;
