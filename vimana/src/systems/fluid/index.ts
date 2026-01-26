/**
 * src/systems/fluid/index.ts
 * ============================
 *
 * WaterBall fluid simulation system exports.
 * MLS-MPM particle physics with WebGPU compute shaders.
 */

export { MLSMPMSimulator, MAX_PARTICLES, MLSMPM_PARTICLE_STRUCT_SIZE } from './MLSMPMSimulator';
export { default as MLSMPMSimulator } from './MLSMPMSimulator';
export * from './types';

// Debug interface setup
export function setupDebugViews(simulator: import('./MLSMPMSimulator').default): void {
    if (!(window as any).debugVimana) {
        (window as any).debugVimana = {};
    }

    (window as any).debugVimana.fluid = {
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

    console.log('[Fluid] Debug views available at window.debugVimana.fluid');
}
