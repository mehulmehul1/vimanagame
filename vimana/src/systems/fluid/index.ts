/**
 * src/systems/fluid/index.ts
 * ============================
 *
 * WaterBall fluid simulation system exports.
 * MLS-MPM particle physics with WebGPU compute shaders.
 */

export { MLSMPMSimulator, MAX_PARTICLES, MLSMPM_PARTICLE_STRUCT_SIZE } from './MLSMPMSimulator';
export { default as MLSMPMSimulator } from './MLSMPMSimulator';
export { DepthThicknessRenderer } from './render/DepthThicknessRenderer';
export { FluidSurfaceRenderer } from './render/FluidSurfaceRenderer';
export { createProceduralCubemap, createCubemapView } from './render/createCubemap';
export { SphereConstraintAnimator } from './animation/SphereConstraintAnimator';
export { default as SphereConstraintAnimator } from './animation/SphereConstraintAnimator';
export * from './types';

// Harp-water interaction exports
export {
    HarpWaterInteraction,
    StringRippleEffect,
    StringForceCalculator,
    STRING_POSITIONS,
    STRING_FREQUENCIES,
    DEFAULT_HARP_CONFIG,
    DEFAULT_RIPPLE_CONFIG,
    createHarpInteractionSystem,
} from './interaction';
export { default as HarpWaterInteraction } from './interaction/HarpWaterInteraction';
export { default as StringRippleEffect } from './interaction/StringRippleEffect';
export { default as StringForceCalculator } from './interaction/StringForceCalculator';
export * from './interaction';

// Debug interface setup
export function setupDebugViews(
    simulator: import('./MLSMPMSimulator').default,
    depthRenderer?: import('./render/DepthThicknessRenderer').default,
    fluidRenderer?: import('./render/FluidSurfaceRenderer').default,
    sphereAnimator?: import('./animation/SphereConstraintAnimator').default,
    harpInteraction?: import('./interaction/HarpWaterInteraction').default
): void {
    if (!(window as any).debugVimana) {
        (window as any).debugVimana = {};
    }

    const debugAPI: any = {
        // Particle info
        getParticleCount: () => simulator.getParticleCount(),
        getState: () => simulator.getState(),

        // Simulation controls
        reset: (initBoxSize?: [number, number, number], sphereRadius?: number) => {
            simulator.reset(initBoxSize, sphereRadius);
        },
        spawnParticles: (count: number = 100) => {
            simulator.spawnParticles(count);
        },
        changeBoxSize: (realBoxSize: [number, number, number]) => {
            simulator.changeBoxSize(realBoxSize);
        },
        changeSphereRadius: (radius: number) => {
            simulator.changeSphereRadius(radius);
        },

        // Debug controls
        toggleWireframe: () => simulator.toggleWireframe(),
        isWireframe: () => simulator.isWireframe(),

        // Mouse interaction
        updateMouseInfo: (
            mouseCoord: [number, number],
            mouseVel: [number, number],
            mouseRadius: number
        ) => {
            simulator.updateMouseInfo(mouseCoord, mouseVel, mouseRadius);
        },
    };

    // Add depth renderer debug views if available
    if (depthRenderer) {
        debugAPI.showDepthMap = () => depthRenderer.showDepthMap();
        debugAPI.showThicknessMap = () => depthRenderer.showThicknessMap();
        debugAPI.hideDebugView = () => depthRenderer.hideDebugView();
        debugAPI.getDebugMode = () => depthRenderer.getDebugMode();
    }

    // Add fluid surface renderer debug views if available
    if (fluidRenderer) {
        debugAPI.toggleNormals = () => fluidRenderer.toggleNormals();
        debugAPI.isShowingNormals = () => fluidRenderer.isShowingNormals();
    }

    // Add sphere animator debug views if available
    if (sphereAnimator) {
        debugAPI.getBoxRatio = () => sphereAnimator.getBoxRatio();
        debugAPI.setBoxRatio = (ratio: number) => sphereAnimator.setBoxRatio(ratio);
        debugAPI.getTunnelRadius = () => sphereAnimator.getTunnelRadius();
        debugAPI.isTunnelOpen = () => sphereAnimator.isTunnelOpen();
        debugAPI.getAnimatorState = () => sphereAnimator.getState();
    }

    // Add harp-water interaction debug views if available
    if (harpInteraction) {
        debugAPI.getStringForce = (index: number) => harpInteraction.getStringForce(index);
        debugAPI.getStringInteraction = (index: number) => harpInteraction.getStringInteraction(index);
        debugAPI.getAllInteractions = () => harpInteraction.getAllInteractions();
        debugAPI.getForceBuffer = () => harpInteraction.getForceBuffer();
        debugAPI.triggerString = (index: number, intensity: number = 1.0) => {
            harpInteraction.onStringPlucked(index, intensity);
        };
        debugAPI.resetHarpInteraction = () => harpInteraction.reset();
        debugAPI.setHarpDebugMode = (enabled: boolean) => harpInteraction.setDebugMode(enabled);
    }

    (window as any).debugVimana.fluid = debugAPI;

    console.log('[Fluid] Debug views available at window.debugVimana.fluid');
}
