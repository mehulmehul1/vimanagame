/**
 * StringRippleEffect.ts - Visual Ripple Effects for Harp Strings
 * ==============================================================
 *
 * Creates expanding ring visualizations at harp string impact points
 * when strings are plucked. Renders as GPU-accelerated instanced quads.
 *
 * Based on WaterBall interaction visual feedback patterns.
 */

import { vec3 } from '../types';
import { STRING_POSITIONS } from './HarpWaterInteraction';

/**
 * Active ripple state
 */
export interface Ripple {
    position: vec3;        // World position of ripple center
    radius: number;        // Current radius
    maxRadius: number;     // Maximum radius before fade out
    alpha: number;         // Current opacity (0-1)
    age: number;           // Time since spawn (seconds)
    maxAge: number;        // Maximum lifetime (seconds)
    stringIndex: number;   // Which string created this ripple
    thickness: number;     // Ring thickness
}

/**
 * Ripple effect configuration
 */
export interface RippleConfig {
    maxRipples: number;        // Maximum concurrent ripples
    defaultMaxRadius: number;  // Default max radius
    defaultMaxAge: number;     // Default lifetime
    defaultThickness: number;  // Default ring thickness
    growthRate: number;        // How fast ripples expand
    color: [number, number, number]; // RGB color
}

/**
 * Default ripple configuration
 */
export const DEFAULT_RIPPLE_CONFIG: RippleConfig = {
    maxRipples: 50,
    defaultMaxRadius: 5.0,
    defaultMaxAge: 3.0,
    defaultThickness: 0.15,
    growthRate: 2.0,
    color: [0.4, 0.7, 1.0], // Light blue
};

/**
 * StringRippleEffect - Visual ripple renderer
 *
 * Manages expanding ring effects at string impact points.
 * Uses Three.js for rendering (can be adapted to pure WebGPU).
 */
export class StringRippleEffect {
    private ripples: Ripple[] = [];
    private config: RippleConfig;

    // Time tracking
    private elapsedTime = 0;

    // Debug state
    private debugMode = false;
    private rippleCount = 0;

    constructor(config: Partial<RippleConfig> = {}) {
        this.config = { ...DEFAULT_RIPPLE_CONFIG, ...config };
        console.log('[StringRippleEffect] Initialized with', this.config);
    }

    /**
     * Spawn a new ripple when a string is plucked
     * @param stringIndex - 0-5, which string was plucked
     * @param intensity - 0-1, pluck intensity
     * @param customPosition - Optional custom position override
     */
    public spawnRipple(
        stringIndex: number,
        intensity: number = 1.0,
        customPosition?: vec3
    ): void {
        if (stringIndex < 0 || stringIndex >= 6) {
            console.warn(`[StringRippleEffect] Invalid string index: ${stringIndex}`);
            return;
        }

        // Remove oldest ripples if at max capacity
        while (this.ripples.length >= this.config.maxRipples) {
            this.ripples.shift();
        }

        // Get position (use custom or string position)
        const position = customPosition
            ? [...customPosition] as vec3
            : [...STRING_POSITIONS[stringIndex]] as vec3;

        // Adjust Y to water level
        position[1] = 0.05; // Slightly above water surface

        // Calculate ripple properties based on intensity
        const maxRadius = this.config.defaultMaxRadius * (0.5 + intensity * 0.5);
        const maxAge = this.config.defaultMaxAge * (0.7 + intensity * 0.3);
        const thickness = this.config.defaultThickness * (0.8 + intensity * 0.4);

        const ripple: Ripple = {
            position,
            radius: 0.1, // Start small
            maxRadius,
            alpha: 1.0,
            age: 0,
            maxAge,
            stringIndex,
            thickness,
        };

        this.ripples.push(ripple);
        this.rippleCount++;

        if (this.debugMode) {
            console.log(`[StringRippleEffect] Spawned ripple for string ${stringIndex}`, {
                position,
                intensity,
                maxRadius,
            });
        }
    }

    /**
     * Spawn multiple ripples (e.g., for chord effects)
     * @param stringIndices - Array of string indices
     * @param intensity - Base intensity
     */
    public spawnMultipleRipples(stringIndices: number[], intensity: number = 1.0): void {
        for (const index of stringIndices) {
            // Stagger slightly for visual interest
            setTimeout(() => {
                this.spawnRipple(index, intensity * 0.8);
            }, Math.random() * 100);
        }
    }

    /**
     * Update all active ripples
     * @param deltaTime - Time since last frame (seconds)
     */
    public update(deltaTime: number): void {
        this.elapsedTime += deltaTime;

        // Update each ripple
        for (let i = this.ripples.length - 1; i >= 0; i--) {
            const ripple = this.ripples[i];

            // Age the ripple
            ripple.age += deltaTime;

            // Expand radius
            ripple.radius += this.config.growthRate * deltaTime;

            // Fade out based on age
            const ageProgress = ripple.age / ripple.maxAge;
            ripple.alpha = 1.0 - ageProgress;

            // Also fade as radius approaches max
            const radiusProgress = ripple.radius / ripple.maxRadius;
            ripple.alpha *= 1.0 - radiusProgress * 0.5;

            // Remove dead ripples
            if (ripple.age >= ripple.maxAge || ripple.alpha <= 0.01) {
                this.ripples.splice(i, 1);
            }
        }
    }

    /**
     * Get all active ripples for rendering
     */
    public getActiveRipples(): ReadonlyArray<Ripple> {
        return this.ripples;
    }

    /**
     * Get ripples for a specific string
     */
    public getRipplesForString(stringIndex: number): Ripple[] {
        return this.ripples.filter(r => r.stringIndex === stringIndex);
    }

    /**
     * Check if a string has active ripples
     */
    public hasActiveRipples(stringIndex: number): boolean {
        return this.ripples.some(r => r.stringIndex === stringIndex && r.alpha > 0.1);
    }

    /**
     * Get total ripple count (active)
     */
    public getActiveCount(): number {
        return this.ripples.length;
    }

    /**
     * Get total ripple count (lifetime)
     */
    public getTotalSpawned(): number {
        return this.rippleCount;
    }

    /**
     * Clear all active ripples
     */
    public clear(): void {
        this.ripples = [];
    }

    /**
     * Set configuration
     */
    public setConfig(config: Partial<RippleConfig>): void {
        this.config = { ...this.config, ...config };
    }

    /**
     * Get configuration
     */
    public getConfig(): RippleConfig {
        return { ...this.config };
    }

    /**
     * Enable/disable debug mode
     */
    public setDebugMode(enabled: boolean): void {
        this.debugMode = enabled;
    }

    /**
     * Get debug info
     */
    public getDebugInfo(): {
        activeRipples: number;
        totalSpawned: number;
        config: RippleConfig;
    } {
        return {
            activeRipples: this.ripples.length,
            totalSpawned: this.rippleCount,
            config: { ...this.config },
        };
    }

    /**
     * Get ripple data for custom rendering
     * Returns flattened array for use in shader
     */
    public getRenderData(): Float32Array {
        // Each ripple needs: position(3), radius(1), alpha(1), thickness(1), stringIndex(1)
        const stride = 7;
        const data = new Float32Array(this.ripples.length * stride);

        for (let i = 0; i < this.ripples.length; i++) {
            const ripple = this.ripples[i];
            const offset = i * stride;
            data[offset + 0] = ripple.position[0];
            data[offset + 1] = ripple.position[1];
            data[offset + 2] = ripple.position[2];
            data[offset + 3] = ripple.radius;
            data[offset + 4] = ripple.alpha;
            data[offset + 5] = ripple.thickness;
            data[offset + 6] = ripple.stringIndex;
        }

        return data;
    }

    /**
     * Destroy and cleanup
     */
    public destroy(): void {
        this.clear();
        this.rippleCount = 0;
    }
}

/**
 * StringRippleRenderer - Three.js renderer for ripple effects
 *
 * Optional class for rendering ripples using Three.js.
 * Can be used if you want visible feedback beyond fluid displacement.
 */
export class StringRippleRenderer {
    private effect: StringRippleEffect;
    private mesh?: THREE.Mesh;

    constructor(
        private scene: THREE.Scene,
        config?: Partial<RippleConfig>
    ) {
        this.effect = new StringRippleEffect(config);
        this.createMesh();
    }

    private createMesh(): void {
        // Create a simple ring geometry for each ripple
        // For production, use instanced rendering for performance
        const geometry = new THREE.RingGeometry(0.1, 0.2, 32);
        const material = new THREE.MeshBasicMaterial({
            color: 0x66b3ff,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide,
            depthWrite: false,
        });

        // Note: This is a simplified version
        // For full implementation, use instanced meshes or custom shader
        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.visible = false;
        this.mesh.rotation.x = -Math.PI / 2; // Lay flat on water
        this.scene.add(this.mesh);
    }

    /**
     * Spawn ripple at string position
     */
    public spawnRipple(stringIndex: number, intensity: number = 1.0): void {
        this.effect.spawnRipple(stringIndex, intensity);
    }

    /**
     * Update ripples
     */
    public update(deltaTime: number): void {
        this.effect.update(deltaTime);

        // Update mesh visualization (simplified - shows single ripple)
        const ripples = this.effect.getActiveRipples();
        if (ripples.length > 0 && this.mesh) {
            const latest = ripples[ripples.length - 1];
            this.mesh.visible = true;
            this.mesh.position.set(latest.position[0], latest.position[1], latest.position[2]);

            // Scale based on radius
            const scale = latest.radius * 2;
            this.mesh.scale.set(scale, scale, 1);

            // Update opacity
            if (this.mesh.material) {
                (this.mesh.material as THREE.MeshBasicMaterial).opacity = latest.alpha;
            }
        } else if (this.mesh) {
            this.mesh.visible = false;
        }
    }

    /**
     * Get the underlying effect for direct access
     */
    public getEffect(): StringRippleEffect {
        return this.effect;
    }

    /**
     * Clean up
     */
    public destroy(): void {
        this.effect.destroy();
        if (this.mesh) {
            this.scene.remove(this.mesh);
            if (this.mesh.geometry) this.mesh.geometry.dispose();
            if (this.mesh.material) (this.mesh.material as THREE.Material).dispose();
        }
    }
}

export default StringRippleEffect;
