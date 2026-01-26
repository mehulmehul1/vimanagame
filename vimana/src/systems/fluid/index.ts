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
export * from './types';

// Debug interface setup
export function setupDebugViews(
    simulator: import('./MLSMPMSimulator').default,
    depthRenderer?: import('./render/DepthThicknessRenderer').default,
    fluidRenderer?: import('./render/FluidSurfaceRenderer').default
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

    (window as any).debugVimana.fluid = debugAPI;

    console.log('[Fluid] Debug views available at window.debugVimana.fluid');
}
