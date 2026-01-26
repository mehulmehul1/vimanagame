/**
 * DepthThicknessRenderer.ts - Depth & Thickness Rendering Pipeline
 * =================================================================
 *
 * Multi-pass rendering pipeline for WaterBall fluid simulation.
 * Generates depth maps and thickness maps from particle positions.
 *
 * Pipeline:
 * 1. Depth Pass: Render particle depths (r32float) with depth testing
 * 2. Bilateral Filter: 4 iterations (2x horizontal, 2x vertical) for smooth edges
 * 3. Thickness Pass: Render particle thickness with additive blending (r16float)
 * 4. Gaussian Blur: 1 iteration for thickness smoothing
 *
 * Based on: https://github.com/matsuoka-601/WaterBall/blob/master/render/fluidRender.ts
 */

import depthMapWGSL from './shaders/depthMap.wgsl';
import bilateralWGSL from './shaders/bilateral.wgsl';
import thicknessMapWGSL from './shaders/thicknessMap.wgsl';
import gaussianWGSL from './shaders/gaussian.wgsl';
import { FLUID_CONFIG, type RenderUniforms } from '../types';

export interface DepthThicknessRendererOptions {
    device: GPUDevice;
    canvas: HTMLCanvasElement;
    posvelBuffer: GPUBuffer;
    renderUniformBuffer: GPUBuffer;
    restDensity?: number;
}

export class DepthThicknessRenderer {
    public readonly device: GPUDevice;
    public readonly canvas: HTMLCanvasElement;
    public readonly restDensity: number;

    // Render pipelines
    private depthMapPipeline: GPURenderPipeline;
    private bilateralFilterPipeline: GPURenderPipeline;
    private thicknessMapPipeline: GPURenderPipeline;
    private gaussianBlurPipeline: GPURenderPipeline;

    // Textures
    public depthMapTexture: GPUTexture;
    public depthMapTextureView: GPUTextureView;
    public tmpDepthMapTexture: GPUTexture;
    public tmpDepthMapTextureView: GPUTextureView;
    public thicknessTexture: GPUTexture;
    public thicknessTextureView: GPUTextureView;
    public tmpThicknessTexture: GPUTexture;
    public tmpThicknessTextureView: GPUTextureView;
    public depthTestTexture: GPUTexture;
    public depthTestTextureView: GPUTextureView;

    // Bind groups
    private depthMapBindGroup: GPUBindGroup;
    private bilateralFilterBindGroups: GPUBindGroup[];
    private thicknessMapBindGroup: GPUBindGroup;
    private gaussianBlurBindGroups: GPUBindGroup[];

    // Uniform buffers
    private stretchStrengthBuffer: GPUBuffer;
    private filterXUniformBuffer: GPUBuffer;
    private filterYUniformBuffer: GPUBuffer;

    // Constants
    private readonly maxFilterSize = 100;
    private readonly blurDepthScale = 10;
    private readonly blurFilterSize = 12;
    private readonly diameter: number;
    private readonly fov: number;

    // Debug state
    private debugMode: 'none' | 'depth' | 'thickness' = 'none';

    constructor(options: DepthThicknessRendererOptions) {
        const {
            device,
            canvas,
            posvelBuffer,
            renderUniformBuffer,
            restDensity = FLUID_CONFIG.restDensity,
        } = options;

        this.device = device;
        this.canvas = canvas;
        this.restDensity = restDensity;

        // Calculate derived constants
        this.diameter = 2 * 0.3; // 2 * renderDiameter
        this.fov = Math.PI / 4; // 45 degrees default

        // Create textures
        this.createTextures();

        // Create uniform buffers
        this.createUniformBuffers();

        // Create shader modules and pipelines
        this.createPipelines(renderUniformBuffer, restDensity);

        // Create bind groups
        this.createBindGroups(posvelBuffer, renderUniformBuffer);

        console.log('[DepthThicknessRenderer] Initialized with', {
            width: canvas.width,
            height: canvas.height,
            restDensity,
        });
    }

    private createTextures(): void {
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Depth map textures (r32float for high precision depth)
        this.depthMapTexture = this.device.createTexture({
            label: 'Depth Map Texture',
            size: [width, height, 1],
            format: 'r32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.depthMapTextureView = this.depthMapTexture.createView();

        this.tmpDepthMapTexture = this.device.createTexture({
            label: 'Temporary Depth Map Texture',
            size: [width, height, 1],
            format: 'r32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.tmpDepthMapTextureView = this.tmpDepthMapTexture.createView();

        // Thickness textures (r16float for memory efficiency)
        this.thicknessTexture = this.device.createTexture({
            label: 'Thickness Texture',
            size: [width, height, 1],
            format: 'r16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.thicknessTextureView = this.thicknessTexture.createView();

        this.tmpThicknessTexture = this.device.createTexture({
            label: 'Temporary Thickness Texture',
            size: [width, height, 1],
            format: 'r16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.tmpThicknessTextureView = this.tmpThicknessTexture.createView();

        // Depth test texture for depth testing during depth pass
        this.depthTestTexture = this.device.createTexture({
            label: 'Depth Test Texture',
            size: [width, height, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthTestTextureView = this.depthTestTexture.createView();
    }

    private createUniformBuffers(): void {
        // Stretch strength buffer for velocity-based particle stretching
        this.stretchStrengthBuffer = this.device.createBuffer({
            label: 'Stretch Strength Buffer',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Initialize with default value
        this.device.queue.writeBuffer(this.stretchStrengthBuffer, 0, new Float32Array([0.5]));

        // Filter direction buffers for X and Y passes
        const filterXValues = new Float32Array([1.0, 0.0]);
        const filterYValues = new Float32Array([0.0, 1.0]);

        this.filterXUniformBuffer = this.device.createBuffer({
            label: 'Filter X Uniform Buffer',
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.filterXUniformBuffer, 0, filterXValues);

        this.filterYUniformBuffer = this.device.createBuffer({
            label: 'Filter Y Uniform Buffer',
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.filterYUniformBuffer, 0, filterYValues);
    }

    private createPipelines(renderUniformBuffer: GPUBuffer, restDensity: number): void {
        const screenConstants = {
            screenWidth: this.canvas.width,
            screenHeight: this.canvas.height,
        };

        const filterConstants = {
            depth_threshold: 0.3 * this.blurDepthScale,
            max_filter_size: this.maxFilterSize,
            projected_particle_constant:
                (this.blurFilterSize * this.diameter * 0.05 * (this.canvas.height / 2)) /
                Math.tan(this.fov / 2),
        };

        const renderEffectConstants = {
            restDensity,
            densitySizeScale: 4.0,
        };

        // Create shader modules
        const depthMapModule = this.device.createShaderModule({ code: depthMapWGSL });
        const bilateralModule = this.device.createShaderModule({ code: bilateralWGSL });
        const thicknessMapModule = this.device.createShaderModule({ code: thicknessMapWGSL });
        const gaussianModule = this.device.createShaderModule({ code: gaussianWGSL });

        // Depth map pipeline
        this.depthMapPipeline = this.device.createRenderPipeline({
            label: 'Depth Map Pipeline',
            layout: 'auto',
            vertex: {
                module: depthMapModule,
                constants: renderEffectConstants,
            },
            fragment: {
                module: depthMapModule,
                targets: [
                    {
                        format: 'r32float',
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float',
            },
        });

        // Bilateral filter pipeline
        this.bilateralFilterPipeline = this.device.createRenderPipeline({
            label: 'Bilateral Filter Pipeline',
            layout: 'auto',
            vertex: {
                module: bilateralModule,
                constants: screenConstants,
            },
            fragment: {
                module: bilateralModule,
                constants: filterConstants,
                targets: [
                    {
                        format: 'r32float',
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        // Thickness map pipeline
        this.thicknessMapPipeline = this.device.createRenderPipeline({
            label: 'Thickness Map Pipeline',
            layout: 'auto',
            vertex: {
                module: thicknessMapModule,
                constants: renderEffectConstants,
            },
            fragment: {
                module: thicknessMapModule,
                targets: [
                    {
                        format: 'r16float',
                        writeMask: GPUColorWrite.RED,
                        blend: {
                            color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
                            alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
                        },
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        // Gaussian blur pipeline
        this.gaussianBlurPipeline = this.device.createRenderPipeline({
            label: 'Gaussian Blur Pipeline',
            layout: 'auto',
            vertex: {
                module: gaussianModule,
                constants: screenConstants,
            },
            fragment: {
                module: gaussianModule,
                targets: [
                    {
                        format: 'r16float',
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });
    }

    private createBindGroups(posvelBuffer: GPUBuffer, renderUniformBuffer: GPUBuffer): void {
        // Depth map bind group
        this.depthMapBindGroup = this.device.createBindGroup({
            label: 'Depth Map Bind Group',
            layout: this.depthMapPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: posvelBuffer } },
                { binding: 1, resource: { buffer: renderUniformBuffer } },
                { binding: 2, resource: { buffer: this.stretchStrengthBuffer } },
            ],
        });

        // Bilateral filter bind groups (X and Y)
        this.bilateralFilterBindGroups = [
            this.device.createBindGroup({
                label: 'Bilateral Filter X Bind Group',
                layout: this.bilateralFilterPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 1, resource: this.depthMapTextureView },
                    { binding: 2, resource: { buffer: this.filterXUniformBuffer } },
                ],
            }),
            this.device.createBindGroup({
                label: 'Bilateral Filter Y Bind Group',
                layout: this.bilateralFilterPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 1, resource: this.tmpDepthMapTextureView },
                    { binding: 2, resource: { buffer: this.filterYUniformBuffer } },
                ],
            }),
        ];

        // Thickness map bind group
        this.thicknessMapBindGroup = this.device.createBindGroup({
            label: 'Thickness Map Bind Group',
            layout: this.thicknessMapPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: posvelBuffer } },
                { binding: 1, resource: { buffer: renderUniformBuffer } },
                { binding: 2, resource: { buffer: this.stretchStrengthBuffer } },
            ],
        });

        // Gaussian blur bind groups (X and Y)
        this.gaussianBlurBindGroups = [
            this.device.createBindGroup({
                label: 'Gaussian Blur X Bind Group',
                layout: this.gaussianBlurPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 1, resource: this.thicknessTextureView },
                    { binding: 2, resource: { buffer: this.filterXUniformBuffer } },
                ],
            }),
            this.device.createBindGroup({
                label: 'Gaussian Blur Y Bind Group',
                layout: this.gaussianBlurPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 1, resource: this.tmpThicknessTextureView },
                    { binding: 2, resource: { buffer: this.filterYUniformBuffer } },
                ],
            }),
        ];
    }

    /**
     * Execute the depth and thickness rendering pipeline
     * Call this each frame after particle simulation
     */
    public execute(commandEncoder: GPUCommandEncoder, numParticles: number, stretchStrength: number = 0.5): void {
        // Update stretch strength
        this.device.queue.writeBuffer(this.stretchStrengthBuffer, 0, new Float32Array([stretchStrength]));

        // === DEPTH PASS ===
        const depthMapPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [
                {
                    view: this.depthMapTextureView,
                    clearValue: { r: 1e6, g: 0.0, b: 0.0, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: this.depthTestTextureView,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        };

        const depthMapPassEncoder = commandEncoder.beginRenderPass(depthMapPassDescriptor);
        depthMapPassEncoder.setBindGroup(0, this.depthMapBindGroup);
        depthMapPassEncoder.setPipeline(this.depthMapPipeline);
        depthMapPassEncoder.draw(6, numParticles);
        depthMapPassEncoder.end();

        // === BILATERAL FILTER PASS (4 iterations: 2x horizontal, 2x vertical) ===
        const depthFilterPassDescriptors: GPURenderPassDescriptor[] = [
            {
                colorAttachments: [
                    {
                        view: this.tmpDepthMapTextureView,
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            },
            {
                colorAttachments: [
                    {
                        view: this.depthMapTextureView,
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            },
        ];

        for (let iter = 0; iter < 2; iter++) {
            // Horizontal pass
            const depthFilterPassEncoderX = commandEncoder.beginRenderPass(depthFilterPassDescriptors[0]);
            depthFilterPassEncoderX.setBindGroup(0, this.bilateralFilterBindGroups[0]);
            depthFilterPassEncoderX.setPipeline(this.bilateralFilterPipeline);
            depthFilterPassEncoderX.draw(6);
            depthFilterPassEncoderX.end();

            // Vertical pass
            const depthFilterPassEncoderY = commandEncoder.beginRenderPass(depthFilterPassDescriptors[1]);
            depthFilterPassEncoderY.setBindGroup(0, this.bilateralFilterBindGroups[1]);
            depthFilterPassEncoderY.setPipeline(this.bilateralFilterPipeline);
            depthFilterPassEncoderY.draw(6);
            depthFilterPassEncoderY.end();
        }

        // === THICKNESS PASS ===
        const thicknessMapPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [
                {
                    view: this.thicknessTextureView,
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        };

        const thicknessMapPassEncoder = commandEncoder.beginRenderPass(thicknessMapPassDescriptor);
        thicknessMapPassEncoder.setBindGroup(0, this.thicknessMapBindGroup);
        thicknessMapPassEncoder.setPipeline(this.thicknessMapPipeline);
        thicknessMapPassEncoder.draw(6, numParticles);
        thicknessMapPassEncoder.end();

        // === GAUSSIAN BLUR PASS (1 iteration: X then Y) ===
        const thicknessFilterPassDescriptors: GPURenderPassDescriptor[] = [
            {
                colorAttachments: [
                    {
                        view: this.tmpThicknessTextureView,
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            },
            {
                colorAttachments: [
                    {
                        view: this.thicknessTextureView,
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
            },
        ];

        // Horizontal pass
        const thicknessFilterPassEncoderX = commandEncoder.beginRenderPass(thicknessFilterPassDescriptors[0]);
        thicknessFilterPassEncoderX.setBindGroup(0, this.gaussianBlurBindGroups[0]);
        thicknessFilterPassEncoderX.setPipeline(this.gaussianBlurPipeline);
        thicknessFilterPassEncoderX.draw(6);
        thicknessFilterPassEncoderX.end();

        // Vertical pass
        const thicknessFilterPassEncoderY = commandEncoder.beginRenderPass(thicknessFilterPassDescriptors[1]);
        thicknessFilterPassEncoderY.setBindGroup(0, this.gaussianBlurBindGroups[1]);
        thicknessFilterPassEncoderY.setPipeline(this.gaussianBlurPipeline);
        thicknessFilterPassEncoderY.draw(6);
        thicknessFilterPassEncoderY.end();
    }

    /**
     * Resize all textures when canvas size changes
     */
    public resize(width: number, height: number): void {
        // Destroy old textures
        this.depthMapTexture.destroy();
        this.tmpDepthMapTexture.destroy();
        this.thicknessTexture.destroy();
        this.tmpThicknessTexture.destroy();
        this.depthTestTexture.destroy();

        // Recreate textures with new size
        const createTexture = (label: string, format: GPUTextureFormat) => {
            const texture = this.device.createTexture({
                label,
                size: [width, height, 1],
                format,
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            });
            return texture;
        };

        this.depthMapTexture = createTexture('Depth Map Texture', 'r32float');
        this.depthMapTextureView = this.depthMapTexture.createView();

        this.tmpDepthMapTexture = createTexture('Temporary Depth Map Texture', 'r32float');
        this.tmpDepthMapTextureView = this.tmpDepthMapTexture.createView();

        this.thicknessTexture = createTexture('Thickness Texture', 'r16float');
        this.thicknessTextureView = this.thicknessTexture.createView();

        this.tmpThicknessTexture = createTexture('Temporary Thickness Texture', 'r16float');
        this.tmpThicknessTextureView = this.tmpThicknessTexture.createView();

        this.depthTestTexture = this.device.createTexture({
            label: 'Depth Test Texture',
            size: [width, height, 1],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthTestTextureView = this.depthTestTexture.createView();

        // Recreate bind groups with new texture views
        // Note: This would require storing posvelBuffer and renderUniformBuffer
        // or having a recreateBindGroups() method
    }

    /**
     * Enable debug visualization mode
     */
    public showDepthMap(): void {
        this.debugMode = 'depth';
    }

    /**
     * Enable debug visualization mode
     */
    public showThicknessMap(): void {
        this.debugMode = 'thickness';
    }

    /**
     * Disable debug visualization
     */
    public hideDebugView(): void {
        this.debugMode = 'none';
    }

    /**
     * Get current debug mode
     */
    public getDebugMode(): 'none' | 'depth' | 'thickness' {
        return this.debugMode;
    }

    /**
     * Get texture views for external rendering (e.g., FluidSurfaceRenderer)
     */
    public getDepthTextureView(): GPUTextureView {
        return this.depthMapTextureView;
    }

    public getThicknessTextureView(): GPUTextureView {
        return this.thicknessTextureView;
    }

    /**
     * Destroy all GPU resources
     */
    public destroy(): void {
        this.depthMapTexture.destroy();
        this.tmpDepthMapTexture.destroy();
        this.thicknessTexture.destroy();
        this.tmpThicknessTexture.destroy();
        this.depthTestTexture.destroy();
        this.stretchStrengthBuffer.destroy();
        this.filterXUniformBuffer.destroy();
        this.filterYUniformBuffer.destroy();
    }
}

export default DepthThicknessRenderer;
