/**
 * VortexLightingManager - Dynamic lighting for vortex activation
 *
 * Manages core light at vortex center and ambient lighting changes
 * as the duet progresses and vortex activates.
 */

import * as THREE from 'three';

export interface VortexLightingConfig {
    /** Core light position */
    coreLightPosition: THREE.Vector3;
    /** Core light maximum intensity */
    coreLightMaxIntensity: number;
    /** Core light range */
    coreLightRange: number;
    /** Ambient intensity boost at full activation */
    ambientBoost: number;
}

export class VortexLightingManager {
    private scene: THREE.Scene;
    private coreLight: THREE.PointLight;
    private originalAmbientIntensity: number = 0;
    private ambientLight?: THREE.AmbientLight;

    private activation: number = 0;
    private config: VortexLightingConfig;

    constructor(scene: THREE.Scene, config: Partial<VortexLightingConfig> = {}) {
        this.scene = scene;

        this.config = {
            coreLightPosition: new THREE.Vector3(0, 0.5, 2),
            coreLightMaxIntensity: 10.0,
            coreLightRange: 5.0,
            ambientBoost: 0.5,
            ...config
        };

        // Create core light
        this.coreLight = new THREE.PointLight(
            0xffffff, // White
            0, // Initial intensity (off)
            this.config.coreLightRange
        );
        this.coreLight.position.copy(this.config.coreLightPosition);
        this.coreLight.castShadow = false; // Ethereal light, no shadows
        this.scene.add(this.coreLight);

        // Find existing ambient light to preserve its intensity
        scene.traverse((object) => {
            if (object instanceof THREE.AmbientLight && !this.ambientLight) {
                this.ambientLight = object;
                this.originalAmbientIntensity = object.intensity;
            }
        });
    }

    /**
     * Set activation level (0-1)
     */
    public setActivation(activation: number): void {
        this.activation = Math.max(0, Math.min(1, activation));
        this.updateLights();
    }

    /**
     * Get current activation
     */
    public getActivation(): number {
        return this.activation;
    }

    /**
     * Update all lights based on current activation
     */
    private updateLights(): void {
        // Core light intensity: 0 → maxIntensity
        this.coreLight.intensity = this.activation * this.config.coreLightMaxIntensity;

        // Core light color shifts slightly with activation
        // Pure white at full activation, slight cyan tint at lower levels
        const t = this.activation;
        this.coreLight.color.setRGB(
            0.8 + t * 0.2, // R: 0.8 → 1.0
            0.9 + t * 0.1, // G: 0.9 → 1.0
            1.0            // B: Always 1.0
        );

        // Ambient light boost
        if (this.ambientLight) {
            const boost = this.activation * this.config.ambientBoost;
            this.ambientLight.intensity = this.originalAmbientIntensity + boost;
        }
    }

    /**
     * Get core light for external access
     */
    public getCoreLight(): THREE.PointLight {
        return this.coreLight;
    }

    /**
     * Reset to initial state
     */
    public reset(): void {
        this.activation = 0;
        this.updateLights();
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.scene.remove(this.coreLight);
        this.coreLight.dispose();

        // Restore ambient light
        if (this.ambientLight) {
            this.ambientLight.intensity = this.originalAmbientIntensity;
        }
    }
}
