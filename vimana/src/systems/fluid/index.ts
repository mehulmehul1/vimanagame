/**
 * src/systems/fluid/index.ts
 * ============================
 *
 * WaterBall fluid simulation system exports.
 * MLS-MPM particle physics with WebGPU compute shaders.
 *
 * Story 4.7: TSL Compute System Activation
 * - TSL compute shaders for Three.js WebGPU integration
 * - Raw WGSL compute shaders (existing MLSMPMSimulator)
 */

// Raw WGSL compute shaders (existing implementation)
export { MLSMPMSimulator, MAX_PARTICLES, MLSMPM_PARTICLE_STRUCT_SIZE } from './MLSMPMSimulator';

// TSL compute shaders (Story 4.7 - Three.js WebGPU integration)
export {
    FluidComputeNode,
    createFluidCompute,
    createDesktopFluidCompute,
    createMobileFluidCompute,
} from './compute/FluidCompute';
export { ParticleBuffers, createParticleBuffers } from './compute/ParticleBuffers';
export {
    createMLSMPMUniforms,
    createSimpleFluidCompute,
    updateUniforms,
    getUniformValues,
    DEFAULT_MLSPM_UNIFORMS,
} from './compute/MLSMPMStages';
export { DepthThicknessRenderer } from './render/DepthThicknessRenderer';
export { FluidSurfaceRenderer } from './render/FluidSurfaceRenderer';
export { createProceduralCubemap, createCubemapView } from './render/createCubemap';
export { SphereConstraintAnimator } from './animation/SphereConstraintAnimator';

// Hybrid renderer - Three.js Points for fluid particles (avoids WebGPU render conflicts)
export { FluidParticlesRenderer } from './FluidParticlesRenderer';

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

// Player-water interaction exports
export {
    PlayerWaterInteraction,
    PlayerWakeEffect,
    PlayerWakeRenderer,
    DEFAULT_PLAYER_CONFIG,
    DEFAULT_WAKE_CONFIG,
} from './interaction';

// Debug interface setup
export function setupDebugViews(
    simulator: import('./MLSMPMSimulator').default,
    depthRenderer?: import('./render/DepthThicknessRenderer').default,
    fluidRenderer?: import('./render/FluidSurfaceRenderer').default,
    sphereAnimator?: import('./animation/SphereConstraintAnimator').default,
    harpInteraction?: import('./interaction/HarpWaterInteraction').default,
    playerInteraction?: import('./interaction/PlayerWaterInteraction').default,
    wakeEffect?: import('./interaction/PlayerWakeEffect').default
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

    // Add player-water interaction debug views if available
    if (playerInteraction) {
        debugAPI.getPlayerInteraction = () => playerInteraction.getState();
        debugAPI.isPlayerInWater = () => playerInteraction.isInWater();
        debugAPI.getPlayerImmersionDepth = () => playerInteraction.getImmersionDepth();
        debugAPI.getPlayerSpeed = () => playerInteraction.getPlayerSpeed();
        debugAPI.getPlayerMovementDirection = () => playerInteraction.getPlayerMovementDirection();
        debugAPI.getBuoyancyForce = () => playerInteraction.getBuoyancyForce();
        debugAPI.getDragFactor = () => playerInteraction.getDragFactor();
        debugAPI.getWaterSurfaceY = () => playerInteraction.getWaterSurfaceY();
        debugAPI.setWaterSurfaceY = (y: number) => playerInteraction.setWaterSurfaceY(y);
        debugAPI.setPlayerMass = (mass: number) => playerInteraction.setPlayerMass(mass);
        debugAPI.setPlayerPosition = (pos: [number, number, number]) => playerInteraction.setPlayerPosition(pos);
        debugAPI.setPlayerVelocity = (vel: [number, number, number]) => playerInteraction.setPlayerVelocity(vel);
        debugAPI.resetPlayerInteraction = () => playerInteraction.reset();
        debugAPI.setPlayerDebugMode = (enabled: boolean) => playerInteraction.setDebugMode(enabled);
        debugAPI.getPlayerBuffer = () => playerInteraction.getInteractionBuffer();
    }

    // Add wake effect debug views if available
    if (wakeEffect) {
        debugAPI.getWakeParticleCount = () => wakeEffect.getParticleCount();
        debugAPI.getWakeParticles = () => wakeEffect.getWakeParticles();
        debugAPI.clearWake = () => wakeEffect.clear();
        debugAPI.setWakeDebugMode = (enabled: boolean) => wakeEffect.setDebugMode(enabled);
    }

    (window as any).debugVimana.fluid = debugAPI;

    console.log('[Fluid] Debug views available at window.debugVimana.fluid');
}
