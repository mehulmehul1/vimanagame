/**
 * VortexActivationController - Controls vortex activation based on duet progress
 *
 * Manages the visual intensification of the vortex system as the player
 * progresses through the duet sequences. Connects duet progress to
 * visual effects.
 */

import * as THREE from 'three';
import { VortexSystem } from './VortexSystem';
import { WaterMaterial } from './WaterMaterial';
import { VortexLightingManager } from './VortexLightingManager';
import { PlatformRideAnimator } from './PlatformRideAnimator';

export interface VortexActivationConfig {
    /** Smoothing factor for activation lerp (0-1) */
    smoothing: number;
    /** Base emissive intensity */
    baseEmissive: number;
    /** Maximum emissive intensity at full activation */
    maxEmissive: number;
    /** Enable platform ride on completion */
    enablePlatformRide: boolean;
}

export interface VortexActivationEvents {
    /** Dispatched when platform ride begins */
    'vortex-complete': void;
    /** Dispatched when platform ride completes */
    'platform-arrived': void;
}

export class VortexActivationController extends THREE.EventDispatcher {
    private vortexSystem: VortexSystem;
    private waterMaterial: WaterMaterial;
    private lightingManager: VortexLightingManager;
    private platformAnimator: PlatformRideAnimator | null;

    private activation: number = 0;
    private targetActivation: number = 0;
    private config: VortexActivationConfig;

    constructor(
        vortexSystem: VortexSystem,
        waterMaterial: WaterMaterial,
        platformMesh: THREE.Mesh | null,
        scene: THREE.Scene,
        camera: THREE.Camera | null = null,
        vortexPosition: THREE.Vector3 | null = null,
        config: Partial<VortexActivationConfig> = {}
    ) {
        super();

        this.vortexSystem = vortexSystem;
        this.waterMaterial = waterMaterial;
        this.config = {
            smoothing: 0.1,
            baseEmissive: 0.2,
            maxEmissive: 3.0,
            enablePlatformRide: true,
            ...config
        };

        // Create lighting manager
        this.lightingManager = new VortexLightingManager(scene);

        // Create platform animator if platform mesh provided
        // Use vortex position as target (if provided), otherwise use default
        this.platformAnimator = platformMesh ?
            new PlatformRideAnimator(
                platformMesh,
                camera,
                vortexPosition ? { targetPosition: vortexPosition } : {}
            ) : null;
    }

    /**
     * Set target activation level (0-1)
     * Called by duet progress updates
     */
    public setActivation(progress: number): void {
        this.targetActivation = Math.max(0, Math.min(1, progress));
    }

    /**
     * Get current activation level
     */
    public getActivation(): number {
        return this.activation;
    }

    /**
     * Update activation - call each frame
     */
    public update(deltaTime: number): void {
        // Smooth lerping toward target
        const smoothing = this.config.smoothing;
        this.activation += (this.targetActivation - this.activation) * smoothing;

        // Update all systems
        this.updateVortexSystem();
        this.updateWaterSurface();
        this.updateLighting();
        this.updatePlatformAnimator(deltaTime);

        // Check for full activation
        if (this.activation >= 0.99 && this.platformAnimator) {
            if (!this.platformAnimator.hasStarted() && this.config.enablePlatformRide) {
                this.startPlatformRide();
            }
        }
    }

    /**
     * Update vortex system activation
     */
    private updateVortexSystem(): void {
        // Update vortex activation
        this.vortexSystem.updateDuetProgress(this.activation);
    }

    /**
     * Update water surface with harmonic resonance
     */
    private updateWaterSurface(): void {
        // Update harmonic resonance
        this.waterMaterial.setHarmonicResonance(this.activation);
        this.waterMaterial.setDuetProgress(this.activation);

        // Bioluminescent intensity boost
        const bioBoost = 1.0 + (this.activation * 0.5);
        // This would be handled by the material's internal logic
    }

    /**
     * Update dynamic lighting
     */
    private updateLighting(): void {
        this.lightingManager.setActivation(this.activation);
    }

    /**
     * Update platform ride animation
     */
    private updatePlatformAnimator(deltaTime: number): void {
        if (this.platformAnimator) {
            this.platformAnimator.update(deltaTime);

            if (this.platformAnimator.isComplete() && this.config.enablePlatformRide) {
                this.config.enablePlatformRide = false; // Prevent duplicate events
                this.dispatchEvent({ type: 'platform-arrived' } as any);
            }
        }
    }

    /**
     * Start platform ride sequence
     */
    private startPlatformRide(): void {
        if (this.platformAnimator) {
            this.platformAnimator.startRide();
            this.dispatchEvent({ type: 'vortex-complete' } as VortexActivationEvents['vortex-complete']);
        }
    }

    /**
     * Check if fully activated
     */
    public isFullyActivated(): boolean {
        return this.activation >= 0.99;
    }

    /**
     * Reset activation (for testing/restart)
     */
    public reset(): void {
        this.activation = 0;
        this.targetActivation = 0;
        this.config.enablePlatformRide = true;
        if (this.platformAnimator) {
            this.platformAnimator.reset();
        }
        this.lightingManager.reset();
    }

    /**
     * Update configuration
     */
    public updateConfig(updates: Partial<VortexActivationConfig>): void {
        this.config = { ...this.config, ...updates };
    }

    /**
     * Get platform animator
     */
    public getPlatformAnimator(): PlatformRideAnimator | null {
        return this.platformAnimator;
    }

    /**
     * Get lighting manager
     */
    public getLightingManager(): VortexLightingManager {
        return this.lightingManager;
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.lightingManager.destroy();

        // Clear references
        // Note: vortexSystem and waterMaterial are not owned by this controller
    }
}
