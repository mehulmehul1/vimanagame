/**
 * MLSMPMSimulator.ts - MLS-MPM Particle Physics Simulation
 * =========================================================
 *
 * Main particle system orchestrator for WaterBall fluid simulation.
 * Uses WebGPU compute shaders to simulate 10,000 fluid particles.
 *
 * Based on: https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/mls-mpm.ts
 *
 * Architecture:
 * 1. Particle-to-Grid (p2g_1, p2g_2): Transfer particle mass/velocity to grid
 * 2. Update Grid: Apply forces, constraints, and mouse interaction
 * 3. Grid-to-Particle (g2p): Update particle velocities from grid
 * 4. Copy Position: Prepare data for rendering
 */

import {
    FLUID_CONFIG,
    Cell,
    MouseInfo,
    PosVel,
    RenderUniforms,
    SimulationState,
    vec3,
} from './types';
import clearGridWGSL from './compute/clearGrid.wgsl';
import spawnParticlesWGSL from './compute/spawnParticles.wgsl';
import p2g_1WGSL from './compute/p2g_1.wgsl';
import p2g_2WGSL from './compute/p2g_2.wgsl';
import updateGridWGSL from './compute/updateGrid.wgsl';
import g2pWGSL from './compute/g2p.wgsl';
import copyPositionWGSL from './compute/copyPosition.wgsl';

export const MLSMPM_PARTICLE_STRUCT_SIZE = 80;
export const MAX_PARTICLES = 10000;

export interface MLSMPMSimulatorOptions {
    device: GPUDevice;
    canvas?: HTMLCanvasElement;
    initBoxSize?: vec3;
    sphereRadius?: number;
    renderDiameter?: number;
    renderUniformBuffer?: GPUBuffer;
    depthTextureView?: GPUTextureView;
}

export class MLSMPMSimulator {
    // Device and context
    public readonly device: GPUDevice;
    private canvas: HTMLCanvasElement | null;

    // Configuration
    public readonly max_x_grids = 80;
    public readonly max_y_grids = 80;
    public readonly max_z_grids = 80;
    public readonly cellStructSize = 16;
    public restDensity = 4.0;

    // Simulation state
    public numParticles = 0;
    public gridCount = 0;
    private frameCount = 0;
    private spawned = false;

    // Box dimensions
    private initBoxSize: vec3;
    private realBoxSize: vec3;
    public sphereRadius = 15.0;

    // Render diameter for particle sizing
    private renderDiameter: number;

    // Mouse interaction state
    private mouseInfoValues: ArrayBuffer;
    private mouseInfoViews: {
        screenSize: Float32Array;
        mouseCoord: Float32Array;
        mouseVel: Float32Array;
        mouseRadius: Float32Array;
    };

    // GPU Buffers
    public particleBuffer: GPUBuffer;
    public posvelBuffer: GPUBuffer;
    public cellBuffer: GPUBuffer;
    public densityBuffer: GPUBuffer;

    // Uniform buffers
    public realBoxSizeBuffer: GPUBuffer;
    public initBoxSizeBuffer: GPUBuffer;
    public numParticlesBuffer: GPUBuffer;
    public mouseInfoUniformBuffer: GPUBuffer;
    public sphereRadiusBuffer: GPUBuffer;

    // Compute pipelines
    private clearGridPipeline: GPUComputePipeline;
    private spawnParticlesPipeline: GPUComputePipeline;
    private p2g1Pipeline: GPUComputePipeline;
    private p2g2Pipeline: GPUComputePipeline;
    private updateGridPipeline: GPUComputePipeline;
    private g2pPipeline: GPUComputePipeline;
    private copyPositionPipeline: GPUComputePipeline;

    // Bind groups
    private clearGridBindGroup: GPUBindGroup;
    private spawnParticlesBindGroup: GPUBindGroup;
    private p2g1BindGroup: GPUBindGroup;
    private p2g2BindGroup: GPUBindGroup;
    private updateGridBindGroup: GPUBindGroup;
    private g2pBindGroup: GPUBindGroup;
    private copyPositionBindGroup: GPUBindGroup;

    // Debug state
    private wireframeMode = false;
    private targetNumParticles = MAX_PARTICLES;

    constructor(options: MLSMPMSimulatorOptions) {
        const {
            device,
            canvas,
            initBoxSize = [52, 52, 52],
            sphereRadius = 15.0,
            renderDiameter = 0.3,
            renderUniformBuffer,
            depthTextureView,
        } = options;

        this.device = device;
        this.canvas = canvas || null;
        this.initBoxSize = initBoxSize;
        this.realBoxSize = [...initBoxSize] as vec3;
        this.sphereRadius = sphereRadius;
        this.renderDiameter = renderDiameter;

        // Calculate grid count
        const maxGridCount = this.max_x_grids * this.max_y_grids * this.max_z_grids;
        this.gridCount = Math.ceil(initBoxSize[0]) * Math.ceil(initBoxSize[1]) * Math.ceil(initBoxSize[2]);

        if (this.gridCount > maxGridCount) {
            throw new Error(`gridCount (${this.gridCount}) exceeds maxGridCount (${maxGridCount})`);
        }

        // Initialize mouse info
        this.mouseInfoValues = new ArrayBuffer(32);
        this.mouseInfoViews = {
            screenSize: new Float32Array(this.mouseInfoValues, 0, 2),
            mouseCoord: new Float32Array(this.mouseInfoValues, 8, 2),
            mouseVel: new Float32Array(this.mouseInfoValues, 16, 2),
            mouseRadius: new Float32Array(this.mouseInfoValues, 24, 1),
        };

        if (canvas) {
            this.mouseInfoViews.screenSize.set([canvas.width, canvas.height]);
        }

        // Create buffers
        this.createBuffers(renderUniformBuffer, depthTextureView);

        // Create compute pipelines and bind groups
        this.createPipelines();
        this.createBindGroups(renderUniformBuffer, depthTextureView);

        console.log('[MLSMPMSimulator] Initialized with', {
            maxParticles: MAX_PARTICLES,
            gridCount: this.gridCount,
            initBoxSize,
            sphereRadius,
        });
    }

    private createBuffers(renderUniformBuffer?: GPUBuffer, depthTextureView?: GPUTextureView): void {
        const maxGridCount = this.max_x_grids * this.max_y_grids * this.max_z_grids;

        // Particle buffers (main simulation data)
        this.particleBuffer = this.device.createBuffer({
            label: 'MLS-MPM Particle Buffer',
            size: MLSMPM_PARTICLE_STRUCT_SIZE * MAX_PARTICLES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Position-velocity buffer for rendering
        this.posvelBuffer = this.device.createBuffer({
            label: 'MLS-MPM PosVel Buffer',
            size: 32 * MAX_PARTICLES, // vec3 + vec3 + f32 = 32 bytes per particle
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Grid cell buffer
        this.cellBuffer = this.device.createBuffer({
            label: 'MLS-MPM Cell Buffer',
            size: this.cellStructSize * maxGridCount,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Density buffer
        this.densityBuffer = this.device.createBuffer({
            label: 'MLS-MPM Density Buffer',
            size: 4 * MAX_PARTICLES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Uniform buffers
        this.realBoxSizeBuffer = this.device.createBuffer({
            label: 'Real Box Size Buffer',
            size: 12, // vec3f
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.initBoxSizeBuffer = this.device.createBuffer({
            label: 'Init Box Size Buffer',
            size: 12, // vec3f
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.numParticlesBuffer = this.device.createBuffer({
            label: 'Num Particles Buffer',
            size: 4, // u32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.mouseInfoUniformBuffer = this.device.createBuffer({
            label: 'Mouse Info Buffer',
            size: this.mouseInfoValues.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.sphereRadiusBuffer = this.device.createBuffer({
            label: 'Sphere Radius Buffer',
            size: 4, // f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Initialize uniform buffers
        this.device.queue.writeBuffer(this.initBoxSizeBuffer, 0, new Float32Array(this.initBoxSize));
        this.device.queue.writeBuffer(this.realBoxSizeBuffer, 0, new Float32Array(this.realBoxSize));
        this.device.queue.writeBuffer(this.numParticlesBuffer, 0, new Uint32Array([0]));
        this.device.queue.writeBuffer(this.sphereRadiusBuffer, 0, new Float32Array([this.sphereRadius]));
        this.device.queue.writeBuffer(this.mouseInfoUniformBuffer, 0, this.mouseInfoValues);
    }

    private createPipelines(): void {
        const constants = {
            fixed_point_multiplier: FLUID_CONFIG.fixedPointMultiplier,
            stiffness: FLUID_CONFIG.stiffness,
            rest_density: this.restDensity,
            dynamic_viscosity: FLUID_CONFIG.dynamicViscosity,
            dt: FLUID_CONFIG.dt,
        };

        // Clear grid pipeline
        const clearGridModule = this.device.createShaderModule({ code: clearGridWGSL });
        this.clearGridPipeline = this.device.createComputePipeline({
            label: 'Clear Grid Pipeline',
            layout: 'auto',
            compute: { module: clearGridModule },
        });

        // Spawn particles pipeline
        const spawnParticlesModule = this.device.createShaderModule({ code: spawnParticlesWGSL });
        this.spawnParticlesPipeline = this.device.createComputePipeline({
            label: 'Spawn Particles Pipeline',
            layout: 'auto',
            compute: { module: spawnParticlesModule },
        });

        // P2G Stage 1 pipeline
        const p2g1Module = this.device.createShaderModule({ code: p2g_1WGSL });
        this.p2g1Pipeline = this.device.createComputePipeline({
            label: 'P2G Stage 1 Pipeline',
            layout: 'auto',
            compute: {
                module: p2g1Module,
                constants: { fixed_point_multiplier: constants.fixed_point_multiplier },
            },
        });

        // P2G Stage 2 pipeline
        const p2g2Module = this.device.createShaderModule({ code: p2g_2WGSL });
        this.p2g2Pipeline = this.device.createComputePipeline({
            label: 'P2G Stage 2 Pipeline',
            layout: 'auto',
            compute: {
                module: p2g2Module,
                constants: {
                    fixed_point_multiplier: constants.fixed_point_multiplier,
                    stiffness: constants.stiffness,
                    rest_density: constants.rest_density,
                    dynamic_viscosity: constants.dynamic_viscosity,
                    dt: constants.dt,
                },
            },
        });

        // Update grid pipeline
        const updateGridModule = this.device.createShaderModule({ code: updateGridWGSL });
        this.updateGridPipeline = this.device.createComputePipeline({
            label: 'Update Grid Pipeline',
            layout: 'auto',
            compute: {
                module: updateGridModule,
                constants: {
                    fixed_point_multiplier: constants.fixed_point_multiplier,
                    dt: constants.dt,
                },
            },
        });

        // G2P pipeline
        const g2pModule = this.device.createShaderModule({ code: g2pWGSL });
        this.g2pPipeline = this.device.createComputePipeline({
            label: 'G2P Pipeline',
            layout: 'auto',
            compute: {
                module: g2pModule,
                constants: {
                    fixed_point_multiplier: constants.fixed_point_multiplier,
                    dt: constants.dt,
                },
            },
        });

        // Copy position pipeline
        const copyPositionModule = this.device.createShaderModule({ code: copyPositionWGSL });
        this.copyPositionPipeline = this.device.createComputePipeline({
            label: 'Copy Position Pipeline',
            layout: 'auto',
            compute: { module: copyPositionModule },
        });
    }

    private createBindGroups(renderUniformBuffer?: GPUBuffer, depthTextureView?: GPUTextureView): void {
        // Clear grid bind group
        this.clearGridBindGroup = this.device.createBindGroup({
            layout: this.clearGridPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cellBuffer } },
            ],
        });

        // Spawn particles bind group
        this.spawnParticlesBindGroup = this.device.createBindGroup({
            layout: this.spawnParticlesPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.initBoxSizeBuffer } },
                { binding: 2, resource: { buffer: this.numParticlesBuffer } },
            ],
        });

        // P2G Stage 1 bind group
        this.p2g1BindGroup = this.device.createBindGroup({
            layout: this.p2g1Pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.cellBuffer } },
                { binding: 2, resource: { buffer: this.initBoxSizeBuffer } },
                { binding: 3, resource: { buffer: this.numParticlesBuffer } },
            ],
        });

        // P2G Stage 2 bind group
        this.p2g2BindGroup = this.device.createBindGroup({
            layout: this.p2g2Pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.cellBuffer } },
                { binding: 2, resource: { buffer: this.initBoxSizeBuffer } },
                { binding: 3, resource: { buffer: this.numParticlesBuffer } },
                { binding: 4, resource: { buffer: this.densityBuffer } },
            ],
        });

        // Update grid bind group
        const updateGridEntries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: this.cellBuffer } },
            { binding: 1, resource: { buffer: this.realBoxSizeBuffer } },
            { binding: 2, resource: { buffer: this.initBoxSizeBuffer } },
            { binding: 5, resource: { buffer: this.mouseInfoUniformBuffer } },
        ];

        // Optional: render uniforms and depth texture for mouse interaction
        if (renderUniformBuffer) {
            updateGridEntries.push({ binding: 3, resource: { buffer: renderUniformBuffer } });
        }
        if (depthTextureView) {
            updateGridEntries.push({ binding: 4, resource: depthTextureView });
        }

        this.updateGridBindGroup = this.device.createBindGroup({
            layout: this.updateGridPipeline.getBindGroupLayout(0),
            entries: updateGridEntries,
        });

        // G2P bind group
        this.g2pBindGroup = this.device.createBindGroup({
            layout: this.g2pPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.cellBuffer } },
                { binding: 2, resource: { buffer: this.realBoxSizeBuffer } },
                { binding: 3, resource: { buffer: this.initBoxSizeBuffer } },
                { binding: 4, resource: { buffer: this.numParticlesBuffer } },
                { binding: 5, resource: { buffer: this.sphereRadiusBuffer } },
            ],
        });

        // Copy position bind group
        this.copyPositionBindGroup = this.device.createBindGroup({
            layout: this.copyPositionPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.posvelBuffer } },
                { binding: 2, resource: { buffer: this.numParticlesBuffer } },
                { binding: 3, resource: { buffer: this.densityBuffer } },
            ],
        });
    }

    /**
     * Reset simulation with new parameters
     */
    public reset(initBoxSize?: vec3, sphereRadius?: number): void {
        const newInitBoxSize = initBoxSize || this.initBoxSize;
        const newSphereRadius = sphereRadius || this.sphereRadius;

        this.initBoxSize = [...newInitBoxSize] as vec3;
        this.realBoxSize = [...newInitBoxSize] as vec3;
        this.sphereRadius = newSphereRadius;

        const maxGridCount = this.max_x_grids * this.max_y_grids * this.max_z_grids;
        this.gridCount = Math.ceil(newInitBoxSize[0]) * Math.ceil(newInitBoxSize[1]) * Math.ceil(newInitBoxSize[2]);

        if (this.gridCount > maxGridCount) {
            throw new Error(`gridCount (${this.gridCount}) exceeds maxGridCount (${maxGridCount})`);
        }

        this.frameCount = 0;
        this.spawned = false;
        this.numParticles = 0;
        this.targetNumParticles = MAX_PARTICLES;

        // Update uniform buffers
        this.device.queue.writeBuffer(this.initBoxSizeBuffer, 0, new Float32Array(newInitBoxSize));
        this.device.queue.writeBuffer(this.realBoxSizeBuffer, 0, new Float32Array(this.realBoxSize));
        this.device.queue.writeBuffer(this.sphereRadiusBuffer, 0, new Float32Array([newSphereRadius]));
        this.changeNumParticles(0);
    }

    /**
     * Change the simulation box size (for sphere tunnel animation)
     */
    public changeBoxSize(realBoxSize: vec3): void {
        this.realBoxSize = [...realBoxSize] as vec3;
        this.device.queue.writeBuffer(this.realBoxSizeBuffer, 0, new Float32Array(realBoxSize));
    }

    /**
     * Change sphere radius for tunnel effect
     */
    public changeSphereRadius(radius: number): void {
        this.sphereRadius = radius;
        this.device.queue.writeBuffer(this.sphereRadiusBuffer, 0, new Float32Array([radius]));
    }

    /**
     * Update the current particle count
     */
    public changeNumParticles(numParticles: number): void {
        this.numParticles = Math.min(numParticles, MAX_PARTICLES);
        this.device.queue.writeBuffer(this.numParticlesBuffer, 0, new Uint32Array([this.numParticles]));
    }

    /**
     * Update mouse interaction info
     */
    public updateMouseInfo(mouseCoord: [number, number], mouseVel: [number, number], mouseRadius: number): void {
        this.mouseInfoViews.mouseCoord.set(mouseCoord);
        this.mouseInfoViews.mouseVel.set(mouseVel);
        this.mouseInfoViews.mouseRadius.set([mouseRadius]);
        this.device.queue.writeBuffer(this.mouseInfoUniformBuffer, 0, this.mouseInfoValues);
    }

    /**
     * Spawn particles in dam-break pattern
     */
    public spawnParticles(count: number = 100): void {
        const spawnCount = Math.min(count, MAX_PARTICLES - this.numParticles);
        if (spawnCount <= 0) return;

        const commandEncoder = this.device.createCommandEncoder({ label: 'Spawn Particles Encoder' });
        const computePass = commandEncoder.beginComputePass();

        computePass.setBindGroup(0, this.spawnParticlesBindGroup);
        computePass.setPipeline(this.spawnParticlesPipeline);
        computePass.dispatchWorkgroups(Math.ceil(spawnCount / 64));

        computePass.end();
        this.device.queue.submit([commandEncoder.finish()]);

        this.numParticles += spawnCount;
        this.changeNumParticles(this.numParticles);
    }

    /**
     * Execute the full MLS-MPM simulation pipeline
     * Call this each frame to update particle physics
     */
    public execute(commandEncoder: GPUCommandEncoder, substeps: number = 2): void {
        // Spawn particles progressively at startup
        if (this.frameCount % 2 === 0 && this.numParticles < this.targetNumParticles) {
            const spawnCount = Math.min(FLUID_CONFIG.spawnRate, this.targetNumParticles - this.numParticles);
            if (spawnCount > 0) {
                const spawnEncoder = this.device.createCommandEncoder({ label: 'Spawn Encoder' });
                const spawnPass = spawnEncoder.beginComputePass();
                spawnPass.setBindGroup(0, this.spawnParticlesBindGroup);
                spawnPass.setPipeline(this.spawnParticlesPipeline);
                spawnPass.dispatchWorkgroups(Math.ceil(spawnCount / 64));
                spawnPass.end();
                this.device.queue.submit([spawnEncoder.finish()]);
                this.numParticles += spawnCount;
                this.changeNumParticles(this.numParticles);
            }
        }

        const computePass = commandEncoder.beginComputePass();

        // Run simulation substeps for stability
        for (let i = 0; i < substeps; i++) {
            // 1. Clear grid
            computePass.setBindGroup(0, this.clearGridBindGroup);
            computePass.setPipeline(this.clearGridPipeline);
            computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64));

            // 2. P2G Stage 1: Transfer mass and velocity
            computePass.setBindGroup(0, this.p2g1BindGroup);
            computePass.setPipeline(this.p2g1Pipeline);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));

            // 3. P2G Stage 2: MLS-MPM update
            computePass.setBindGroup(0, this.p2g2BindGroup);
            computePass.setPipeline(this.p2g2Pipeline);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));

            // 4. Update grid (boundary constraints, mouse interaction)
            computePass.setBindGroup(0, this.updateGridBindGroup);
            computePass.setPipeline(this.updateGridPipeline);
            computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64));

            // 5. G2P: Grid to particle transfer with sphere constraint
            computePass.setBindGroup(0, this.g2pBindGroup);
            computePass.setPipeline(this.g2pPipeline);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
        }

        // 6. Copy positions to render buffer (once per frame)
        computePass.setBindGroup(0, this.copyPositionBindGroup);
        computePass.setPipeline(this.copyPositionPipeline);
        computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));

        computePass.end();

        this.frameCount++;
    }

    /**
     * Simulate one frame of fluid physics
     * Creates a command encoder and executes the compute pipeline
     * This is the main entry point for frame-by-frame simulation
     */
    public simulate(deltaTime: number, substeps: number = 2): void {
        // Create command encoder for this frame's compute work
        const commandEncoder = this.device.createCommandEncoder({
            label: 'MLS-MPM Simulation Encoder'
        });

        // Execute the full compute pipeline
        this.execute(commandEncoder, substeps);

        // Submit compute work to GPU queue
        this.device.queue.submit([commandEncoder.finish()]);
    }

    /**
     * Get current simulation state
     */
    public getState(): SimulationState {
        return {
            numParticles: this.numParticles,
            boxWidthRatio: this.realBoxSize[2] / this.initBoxSize[2],
            isTunnelOpen: this.realBoxSize[2] / this.initBoxSize[2] <= 0.55,
        };
    }

    /**
     * Get particle count
     */
    public getParticleCount(): number {
        return this.numParticles;
    }

    /**
     * Toggle wireframe debug mode
     */
    public toggleWireframe(): void {
        this.wireframeMode = !this.wireframeMode;
        return this.wireframeMode;
    }

    /**
     * Check if wireframe mode is active
     */
    public isWireframe(): boolean {
        return this.wireframeMode;
    }

    /**
     * Destroy all GPU resources
     */
    public destroy(): void {
        this.particleBuffer.destroy();
        this.posvelBuffer.destroy();
        this.cellBuffer.destroy();
        this.densityBuffer.destroy();
        this.realBoxSizeBuffer.destroy();
        this.initBoxSizeBuffer.destroy();
        this.numParticlesBuffer.destroy();
        this.mouseInfoUniformBuffer.destroy();
        this.sphereRadiusBuffer.destroy();
    }
}

export default MLSMPMSimulator;
