/**
 * src/systems/fluid/interaction/index.ts
 * =====================================
 *
 * Harp-to-water interaction system exports.
 * Manages coupling between harp string vibrations and fluid simulation.
 */

export {
    HarpWaterInteraction,
    STRING_POSITIONS,
    STRING_FREQUENCIES,
    DEFAULT_HARP_CONFIG,
    type StringWaterInteraction,
} from './HarpWaterInteraction';

export { default as HarpWaterInteraction } from './HarpWaterInteraction';

export {
    calculateStringForce,
    calculateFalloff,
    getStringFrequency,
    calculatePhase,
    calculateRipplePattern,
    combineForces,
    clampForce,
    StringForceCalculator,
    type ForceCalculationParams,
    type ForceResult,
} from './StringForceCalculator';

export { default as StringForceCalculator } from './StringForceCalculator';

export {
    StringRippleEffect,
    StringRippleRenderer,
    DEFAULT_RIPPLE_CONFIG,
    type Ripple,
    type RippleConfig,
} from './StringRippleEffect';

export { default as StringRippleEffect } from './StringRippleEffect';

/**
 * Create a complete harp-water interaction system
 *
 * Factory function for creating all interaction components.
 *
 * @param device - WebGPU device
 * @param config - Optional configuration overrides
 * @returns Object containing all interaction systems
 */
export function createHarpInteractionSystem(
    device: GPUDevice,
    config?: {
        harp?: Partial<typeof import('./HarpWaterInteraction').DEFAULT_HARP_CONFIG>;
        ripple?: Partial<typeof import('./StringRippleEffect').DEFAULT_RIPPLE_CONFIG>;
    }
) {
    const { HarpWaterInteraction } = require('./HarpWaterInteraction');
    const { StringRippleEffect } = require('./StringRippleEffect');

    const harpInteraction = new HarpWaterInteraction(device, config?.harp);
    const rippleEffect = new StringRippleEffect(config?.ripple);

    return {
        harp: harpInteraction,
        ripples: rippleEffect,

        /**
         * Called when a harp string is plucked
         */
        onStringPlucked(stringIndex: number, intensity: number) {
            harpInteraction.onStringPlucked(stringIndex, intensity);
            rippleEffect.spawnRipple(stringIndex, intensity);
        },

        /**
         * Update all interaction systems
         */
        update(deltaTime: number) {
            harpInteraction.update(deltaTime);
            rippleEffect.update(deltaTime);
        },

        /**
         * Clean up all resources
         */
        destroy() {
            harpInteraction.destroy();
            rippleEffect.destroy();
        },
    };
}
