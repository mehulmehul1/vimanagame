/**
 * FluidSurfaceRenderer.ts - Fluid Surface Shader Renderer
 * =======================================================
 *
 * Final fluid surface renderer that combines depth maps, thickness maps,
 * and environment reflections to produce the WaterBall visual result.
 *
 * Pipeline:
 * 1. Takes depth texture from DepthThicknessRenderer
 * 2. Takes thickness texture from DepthThicknessRenderer
 * 3. Samples environment cubemap for reflections
 * 4. Renders fullscreen quad with fluid shader
 *
 * Based on: https://github.com/matsuoka-601/WaterBall/blob/master/render/fluidRender.ts
 */

import fluidWGSL from './shaders/fluid.wgsl';
import type { RenderUniforms } from '../types';

export interface FluidSurfaceRendererOptions {
    device: GPUDevice;
    canvas: HTMLCanvasElement;
    renderUniformBuffer: GPUBuffer;
    depthTextureView: GPUTextureView;
    thicknessTextureView: GPUTextureView;
    cubemapTextureView?: GPUTextureView;
}

export class FluidSurfaceRenderer {
    public readonly device: GPUDevice;
    public readonly canvas: HTMLCanvasElement;

    // Render pipeline
    private fluidPipeline: GPURenderPipeline;

    // Texture views (references to textures owned by DepthThicknessRenderer)
    public depthTextureView: GPUTextureView;
    public thicknessTextureView: GPUTextureView;
    private cubemapTextureView: GPUTextureView | null;

    // Bind group
    private fluidBindGroup: GPUBindGroup;

    // Sampler
    private sampler: GPUSampler;

    // Debug uniform buffer
    private debugUniformBuffer: GPUBuffer;

    // Debug state
    private showNormals = false;

    constructor(options: FluidSurfaceRendererOptions) {
        const {
            device,
            canvas,
            renderUniformBuffer,
            depthTextureView,
            thicknessTextureView,
            cubemapTextureView,
        } = options;

        this.device = device;
        this.canvas = canvas;
        this.depthTextureView = depthTextureView;
        this.thicknessTextureView = thicknessTextureView;
        this.cubemapTextureView = cubemapTextureView || null;

        // Create sampler
        this.sampler = device.createSampler({
            label: 'Fluid Sampler',
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
        });

        // Create debug uniform buffer (16 bytes: show_normals f32 + padding vec3f)
        this.debugUniformBuffer = device.createBuffer({
            label: 'Debug Uniform Buffer',
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Initialize with show_normals = 0
        this.device.queue.writeBuffer(this.debugUniformBuffer, 0, new Float32Array([0, 0, 0, 0]));

        // Create pipeline
        this.createPipeline(canvas.width, canvas.height);

        // Create bind group
        this.createBindGroup(renderUniformBuffer, depthTextureView, thicknessTextureView, cubemapTextureView);

        console.log('[FluidSurfaceRenderer] Initialized with', {
            width: canvas.width,
            height: canvas.height,
            hasCubemap: !!cubemapTextureView,
        });
    }

    private createPipeline(width: number, height: number): void {
        const screenConstants = {
            screenWidth: width,
            screenHeight: height,
        };

        const fluidModule = this.device.createShaderModule({ code: fluidWGSL });

        this.fluidPipeline = this.device.createRenderPipeline({
            label: 'Fluid Surface Pipeline',
            layout: 'auto',
            vertex: {
                module: fluidModule,
                constants: screenConstants,
            },
            fragment: {
                module: fluidModule,
                targets: [
                    {
                        format: 'bgra8unorm', // Default presentation format
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });
    }

    private createBindGroup(
        renderUniformBuffer: GPUBuffer,
        depthTextureView: GPUTextureView,
        thicknessTextureView: GPUTextureView,
        cubemapTextureView?: GPUTextureView
    ): void {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: this.sampler },
            { binding: 1, resource: depthTextureView },
            { binding: 2, resource: { buffer: renderUniformBuffer } },
            { binding: 3, resource: thicknessTextureView },
            { binding: 5, resource: { buffer: this.debugUniformBuffer } },
        ];

        if (cubemapTextureView) {
            entries.push({ binding: 4, resource: cubemapTextureView });
        }

        this.fluidBindGroup = this.device.createBindGroup({
            label: 'Fluid Surface Bind Group',
            layout: this.fluidPipeline.getBindGroupLayout(0),
            entries,
        });
    }

    /**
     * Update texture views (call after resize when textures are recreated)
     */
    public updateTextures(
        depthTextureView: GPUTextureView,
        thicknessTextureView: GPUTextureView,
        cubemapTextureView?: GPUTextureView
    ): void {
        this.depthTextureView = depthTextureView;
        this.thicknessTextureView = thicknessTextureView;
        this.cubemapTextureView = cubemapTextureView || null;
        // Note: Bind group needs to be recreated, caller should handle this
    }

    /**
     * Recreate bind group after texture update
     */
    public recreateBindGroup(renderUniformBuffer: GPUBuffer): void {
        this.createBindGroup(
            renderUniformBuffer,
            this.depthTextureView,
            this.thicknessTextureView,
            this.cubemapTextureView || undefined
        );
    }

    /**
     * Execute the fluid surface render pass
     * Call this each frame after depth/thickness rendering
     * This should be the final render pass to the swap chain
     */
    public execute(commandEncoder: GPUCommandEncoder, view: GPUTextureView): void {
        const fluidPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [
                {
                    view: view,
                    clearValue: { r: 0.7, g: 0.7, b: 0.75, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        };

        const fluidPassEncoder = commandEncoder.beginRenderPass(fluidPassDescriptor);
        fluidPassEncoder.setBindGroup(0, this.fluidBindGroup);
        fluidPassEncoder.setPipeline(this.fluidPipeline);
        fluidPassEncoder.draw(6); // Fullscreen triangle (2 triangles, 6 vertices)
        fluidPassEncoder.end();
    }

    /**
     * Resize pipeline when canvas size changes
     */
    public resize(width: number, height: number): void {
        // Recreate pipeline with new screen size constants
        this.createPipeline(width, height);
    }

    /**
     * Toggle normal debug visualization
     */
    public toggleNormals(): boolean {
        this.showNormals = !this.showNormals;
        // Update the debug uniform buffer
        const value = this.showNormals ? 1.0 : 0.0;
        this.device.queue.writeBuffer(this.debugUniformBuffer, 0, new Float32Array([value, 0, 0, 0]));
        console.log('[FluidSurfaceRenderer] Normal debug visualization:', this.showNormals ? 'ON' : 'OFF');
        return this.showNormals;
    }

    /**
     * Check if normal debug mode is active
     */
    public isShowingNormals(): boolean {
        return this.showNormals;
    }

    /**
     * Destroy all GPU resources
     */
    public destroy(): void {
        this.sampler.destroy();
        this.debugUniformBuffer.destroy();
    }
}

export default FluidSurfaceRenderer;
