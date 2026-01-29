import * as THREE from 'three';
import { VortexMaterial } from './VortexMaterial';
import { VortexParticles } from './VortexParticles';
import { WaterMaterial } from './WaterMaterial';

/**
 * VortexSystem - Main controller for the vortex visual effect
 *
 * Combines the SDF torus mesh with particle system and coordinates
 * activation based on duet progress.
 */
export class VortexSystem extends THREE.Group {
    private static readonly VORTEX_POSITION = new THREE.Vector3(0, 0.5, 2);

    private vortexMesh: THREE.Mesh;
    private vortexMaterial: VortexMaterial;
    private particles: VortexParticles;
    private time: number = 0;
    private activation: number = 0;

    // Reference to water material for coordinated effects
    private waterMaterial?: WaterMaterial;

    constructor(particleCount: number = 2000) {
        super();

        // Create vortex torus mesh
        const torusGeometry = new THREE.TorusGeometry(2.0, 0.4, 32, 64);
        this.vortexMaterial = new VortexMaterial();
        this.vortexMesh = new THREE.Mesh(torusGeometry, this.vortexMaterial.getMaterial());
        // Position at local origin; Group handles world placement
        this.vortexMesh.position.set(0, 0, 0);

        this.add(this.vortexMesh);

        // Create particle system
        this.particles = new VortexParticles(particleCount);
        this.particles.position.set(0, 0, 0);
        this.add(this.particles);

        // Default world position
        this.position.copy(VortexSystem.VORTEX_POSITION);
    }

    /**
     * Link water material for coordinated harmonic effects
     */
    public setWaterMaterial(waterMaterial: WaterMaterial): void {
        this.waterMaterial = waterMaterial;
    }

    /**
     * Main update loop - call each frame
     */
    public update(deltaTime: number, cameraPosition: THREE.Vector3): void {
        this.time += deltaTime;

        // Update vortex material
        this.vortexMaterial.setTime(this.time);
        this.vortexMaterial.setCameraPosition(cameraPosition);

        // Update particles
        this.particles.update(deltaTime, this.activation);

        // Update water if linked
        if (this.waterMaterial) {
            this.waterMaterial.setCameraPosition(cameraPosition);
            this.waterMaterial.setHarmonicResonance(this.activation);
            this.waterMaterial.decayVelocities(deltaTime);
        }
    }

    /**
     * Update duet progress (0-1)
     * This drives the vortex activation
     */
    public updateDuetProgress(progress: number): void {
        this.activation = Math.max(0, Math.min(1, progress));

        // Update vortex material activation
        this.vortexMaterial.setActivation(this.activation);
        this.vortexMaterial.setDuetProgress(this.activation);

        // Update water duet progress
        if (this.waterMaterial) {
            this.waterMaterial.setDuetProgress(this.activation);
        }
    }

    /**
     * Trigger ripple effect on water when string is played
     */
    public triggerStringRipple(stringIndex: number, intensity: number = 1.0): void {
        if (this.waterMaterial) {
            this.waterMaterial.triggerStringRipple(stringIndex, intensity);
        }
    }

    /**
     * Get current activation level
     */
    public getActivation(): number {
        return this.activation;
    }

    /**
     * Check if vortex is fully activated (completion condition)
     */
    public isFullyActivated(): boolean {
        return this.activation >= 0.99;
    }

    /**
     * Cleanup method for memory management
     * CRITICAL: Must be called before scene transitions
     */
    public destroy(): void {
        // Remove and dispose vortex mesh
        this.remove(this.vortexMesh);
        this.vortexMesh.geometry.dispose();
        this.vortexMaterial.destroy();
        this.vortexMesh = null as any;

        // Remove and dispose particles
        this.remove(this.particles);
        this.particles.destroy();
        this.particles = null as any;

        // Clear water reference
        this.waterMaterial = undefined;
    }
}
