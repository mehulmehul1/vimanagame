import * as THREE from 'three';
import { MLSMPMSimulator } from './MLSMPMSimulator';

/**
 * FluidParticlesRenderer - Renders fluid particles as Three.js Points
 * ================================================================
 * 
 * Hybrid approach: Uses WebGPU compute for physics simulation,
 * but renders particles using standard Three.js Points for compatibility.
 * 
 * This avoids WebGPU/Three.js render loop conflicts while still
 * showing 3D fluid particle dynamics on top of the backdrop water.
 */

export class FluidParticlesRenderer {
    private simulator: MLSMPMSimulator;
    private geometry: THREE.BufferGeometry;
    private material: THREE.PointsMaterial;
    private points: THREE.Points;
    private positions: Float32Array;
    private maxParticles: number;
    
    // CPU buffer for reading GPU data
    private cpuReadBuffer: GPUBuffer | null = null;
    private device: GPUDevice;
    private isMapping: boolean = false;  // Track if buffer is currently being mapped
    
    // Particle appearance
    private particleSize: number;
    private particleColor: THREE.Color;
    private particleOpacity: number;
    
    constructor(
        simulator: MLSMPMSimulator,
        options: {
            particleSize?: number;
            particleColor?: number | THREE.Color;
            particleOpacity?: number;
        } = {}
    ) {
        this.simulator = simulator;
        this.device = simulator.device;
        this.maxParticles = 10000; // MLS-MPM max
        
        // Appearance settings
        this.particleSize = options.particleSize ?? 0.15;
        this.particleColor = options.particleColor instanceof THREE.Color 
            ? options.particleColor 
            : new THREE.Color(options.particleColor ?? 0x74ccf4); // Light cyan
        this.particleOpacity = options.particleOpacity ?? 0.6;
        
        // Create CPU read buffer for GPU data
        this.cpuReadBuffer = this.device.createBuffer({
            size: this.maxParticles * 4 * 4, // 4 floats per particle (pos + density), 4 bytes each
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        
        // Create Three.js geometry
        this.geometry = new THREE.BufferGeometry();
        this.positions = new Float32Array(this.maxParticles * 3);
        this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
        
        // Create particle material
        this.material = new THREE.PointsMaterial({
            color: this.particleColor,
            size: this.particleSize,
            transparent: true,
            opacity: this.particleOpacity,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            sizeAttenuation: true,
        });
        
        // Create Points mesh
        this.points = new THREE.Points(this.geometry, this.material);
        this.points.frustumCulled = false; // Always render
        
        console.log('[FluidParticlesRenderer] Initialized with', {
            maxParticles: this.maxParticles,
            particleSize: this.particleSize,
            particleColor: this.particleColor.getHexString(),
            opacity: this.particleOpacity,
        });
    }
    
    /**
     * Get the Three.js Points mesh to add to scene
     */
    public getMesh(): THREE.Points {
        return this.points;
    }
    
    /**
     * Update particle positions from WebGPU buffer
     * Call this before each render frame
     */
    public async update(): Promise<void> {
        if (!this.cpuReadBuffer || this.simulator.numParticles === 0) {
            this.points.visible = false;
            return;
        }

        // Skip if already mapping (prevents "outstanding map pending" errors)
        if (this.isMapping) {
            return;
        }

        this.points.visible = true;

        try {
            this.isMapping = true;

            // Copy from GPU buffer to CPU read buffer
            const commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(
                this.simulator.posvelBuffer,
                0,
                this.cpuReadBuffer,
                0,
                this.simulator.numParticles * 4 * 4
            );
            this.device.queue.submit([commandEncoder.finish()]);

            // Map and read the buffer
            await this.cpuReadBuffer.mapAsync(GPUMapMode.READ);
            const mappedData = new Float32Array(this.cpuReadBuffer.getMappedRange());

            // Update Three.js positions
            // Data format: [x, y, z, density] per particle
            const numParticles = Math.min(this.simulator.numParticles, this.maxParticles);
            for (let i = 0; i < numParticles; i++) {
                this.positions[i * 3 + 0] = mappedData[i * 4 + 0];
                this.positions[i * 3 + 1] = mappedData[i * 4 + 1];
                this.positions[i * 3 + 2] = mappedData[i * 4 + 2];
            }

            // Zero out remaining particles
            for (let i = numParticles; i < this.maxParticles; i++) {
                this.positions[i * 3 + 0] = 0;
                this.positions[i * 3 + 1] = -1000; // Hide below floor
                this.positions[i * 3 + 2] = 0;
            }

            this.cpuReadBuffer.unmap();

            // Mark geometry as needing update
            this.geometry.attributes.position.needsUpdate = true;
            this.geometry.setDrawRange(0, numParticles);

        } catch (error) {
            console.error('[FluidParticlesRenderer] Failed to update particles:', error);
        } finally {
            this.isMapping = false;
        }
    }
    
    /**
     * Set particle size
     */
    public setParticleSize(size: number): void {
        this.particleSize = size;
        this.material.size = size;
    }
    
    /**
     * Set particle color
     */
    public setColor(color: number | THREE.Color): void {
        if (color instanceof THREE.Color) {
            this.material.color.copy(color);
        } else {
            this.material.color.setHex(color);
        }
    }
    
    /**
     * Set particle opacity
     */
    public setOpacity(opacity: number): void {
        this.material.opacity = opacity;
    }
    
    /**
     * Toggle visibility
     */
    public setVisible(visible: boolean): void {
        this.points.visible = visible;
    }
    
    /**
     * Check if visible
     */
    public isVisible(): boolean {
        return this.points.visible;
    }
    
    /**
     * Clean up resources
     */
    public dispose(): void {
        if (this.cpuReadBuffer) {
            this.cpuReadBuffer.destroy();
            this.cpuReadBuffer = null;
        }
        
        this.geometry.dispose();
        this.material.dispose();
        
        console.log('[FluidParticlesRenderer] Disposed');
    }
}

export default FluidParticlesRenderer;
