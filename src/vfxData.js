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
        $lt: GAME_STATES.VIEWMASTER, // Stop before VIEWMASTER so officeColor can take over
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
        $lt: GAME_STATES.POST_VIEWMASTER, // Stop before POST_VIEWMASTER
      },
    },
    priority: 20,
    delay: 3.0,
  },

  // Wipe to grayscale synchronized with splatMorph reverse transition
  postViewmasterWipe: {
    id: "postViewmasterWipe",
    parameters: {
      target: 1.0, // Full grayscale
      duration: 2.25, // Match splatMorph hellReverseTransition
      mode: "wipe",
      wipeDirection: "bottom-to-top",
      wipeSoftness: 0.15,
      suppressAudio: true, // Don't play audio on reverse transition
    },
    criteria: {
      currentState: {
        $eq: GAME_STATES.POST_VIEWMASTER,
      },
    },
    priority: 30, // Higher than officeColor
  },

  postViewmasterGrayscale: {
    id: "postViewmasterGrayscale",
    parameters: {
      target: 1.0, // Grayscale
      duration: 0.0,
      mode: "fade",
    },
    criteria: {
      currentState: {
        $gt: GAME_STATES.POST_VIEWMASTER, // Stop before VIEWMASTER so officeColor can take over
      },
    },
    priority: 0,
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
      rampOutDuration: 1.0, // Ramp down over 1 second when leaving state
      targetMeshIds: ["interior", "officeHell"], // Apply to both splats
    },
    criteria: {
      currentState: GAME_STATES.VIEWMASTER_HELL,
    },
    priority: 10,
  },
};

/**
 * Splat Morph Effect
 * Controls morphing transition between two splat scenes
 */
export const splatMorphEffects = {
  // Forward transition: scatter effect when entering VIEWMASTER_HELL
  hellTransition: {
    id: "hellTransition",
    parameters: {
      speedMultiplier: 1.0,
      staySeconds: 4, // How long to stay on current splat before transitioning
      transitionSeconds: 4.0, // Duration of the morph transition
      randomRadius: 8.0, // Radius of scatter cloud (about the size of the office room)
      scatterCenter: { x: -5.14, y: 3.05, z: 84.66 }, // Center point of scatter cloud
      mode: "scatter", // Use scatter effect for forward transition
      trigger: "start", // Start the transition
    },
    criteria: {
      currentState: {
        $eq: GAME_STATES.VIEWMASTER_HELL, // Only at VIEWMASTER_HELL state
      },
    },
    priority: 10,
  },

  // Reverse transition: wipe effect when leaving hell (returning to normal)
  // Triggers at POST_VIEWMASTER state
  hellReverseTransition: {
    id: "hellReverseTransition",
    parameters: {
      speedMultiplier: 1.0,
      staySeconds: 0, // No delay, start immediately
      transitionSeconds: 2.75, // Slightly faster wipe back
      mode: "wipe", // Use wipe effect for reverse transition
      wipeDirection: "bottom-to-top", // Wipe from bottom to top
      wipeSoftness: 0.15, // Soft edge for smooth transition
      trigger: "start",
      suppressAudio: true, // Don't play audio on reverse transition
    },
    criteria: {
      currentState: {
        $eq: GAME_STATES.POST_VIEWMASTER, // Only trigger at POST_VIEWMASTER, not states after
      },
    },
    priority: 20, // Higher priority than forward transition
  },
};

/**
 * Dissolve Effect
 * Controls dissolve transitions on GLTF objects with particle emission
 */
export const dissolveEffects = {
  // Dissolve edison phonograph and candlestick phone during VIEWMASTER_COLOR (noise-based with particles)
  edisonDissolve: {
    id: "edisonDissolve",
    parameters: {
      targetObjectIds: ["edison", "candlestickPhone"], // Both office objects
      progress: -14.0, // Start fully visible
      targetProgress: 14.0, // End fully dissolved
      autoAnimate: false, // Don't bounce back and forth - one-way transition
      transitionDuration: 5.0, // Duration in seconds to dissolve
      mode: "noise", // Use noise-based dissolve
      edgeColor1: "#66bbff", // Medium bright cyan
      edgeColor2: "#3388cc", // Deeper blue
      particleColor: "#4499dd", // Medium blue particles
      frequency: 4, // Noise frequency (0-5) - higher = more varied, less banding
      edgeWidth: 0.5, // Subtle edge width
      bloomStrength: 12.0, // Optional bloom strength (0-20)
      particleIntensity: 1.0, // Particle brightness multiplier
      enableAudio: true, // Enable wishy-washy oscillating static audio
      // Particle system parameters
      particleSize: 50.0, // Base particle size (20-100)
      particleDecimation: 5, // 1=all vertices, 10=every 10th (fewer particles)
      particleDispersion: 2.0, // Max travel distance (2-20)
      particleVelocitySpread: 0.1, // Velocity randomness (0.05-0.3)
    },
    criteria: {
      currentState: {
        $gte: GAME_STATES.VIEWMASTER_DISSOLVE,
        $lt: GAME_STATES.POST_VIEWMASTER,
      },
    },
    priority: 10,
  },

  // Reverse transition: wipe both objects back in at POST_VIEWMASTER (synchronized with desaturation/morph wipes)
  edisonWipeIn: {
    id: "edisonWipeIn",
    parameters: {
      targetObjectIds: ["edison", "candlestickPhone"], // Both office objects
      progress: 15.0, // Start fully dissolved (progress > max dissolveValue)
      targetProgress: -15.0, // End fully visible (progress < min dissolveValue)
      autoAnimate: false,
      transitionDuration: 2.5, // Match desaturation and splatMorph wipe duration
      mode: "wipe", // Use wipe mode instead of noise
      wipeDirection: "bottom-to-top", // Match desaturation wipe
      wipeSoftness: 0.15, // Match desaturation wipe softness
      suppressAudio: true, // Don't play audio on reverse transition
      // No particles for wipe mode
      particleSize: 0, // Disable particles
      particleDecimation: 999, // Effectively no particles
      edgeWidth: 0.005, // Subtle edge width
    },
    criteria: {
      currentState: { $eq: GAME_STATES.POST_VIEWMASTER },
    },
    priority: 20, // Higher priority than dissolve
  },
};

/**
 * All VFX effects organized by type
 */
export const vfxEffects = {
  desaturation: desaturationEffects,
  cloudParticles: cloudParticleEffects,
  splatFractal: splatFractalEffects,
  splatMorph: splatMorphEffects,
  dissolve: dissolveEffects,
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

export function getSplatFractalEffectForState(gameState) {
  return getVfxEffectForState("splatFractal", gameState);
}

export function getSplatMorphEffectForState(gameState) {
  return getVfxEffectForState("splatMorph", gameState);
}

export function getDissolveEffectForState(gameState) {
  return getVfxEffectForState("dissolve", gameState);
}

export default vfxEffects;
