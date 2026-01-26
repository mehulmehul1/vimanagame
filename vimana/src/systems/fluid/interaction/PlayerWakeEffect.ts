/**
 * PlayerWakeEffect.ts - Player Wake Visual Effects
 * =================================================
 *
 * Creates visual wake trails behind player movement in water.
 * Renders expanding ring particles at water surface.
 */

import { vec3 } from '../types';

/**
 * Individual wake particle
 */
export interface WakeParticle {
    position: vec3;
    velocity: vec3; // Direction of wake expansion
    radius: number;
    maxRadius: number;
    alpha: number;
    age: number;
    lifetime: number;
    thickness: number;
}

/**
 * Wake effect configuration
 */
export interface WakeConfig {
    maxParticles: number;
    minSpeed: number; // Minimum player speed to create wake
    baseRadius: number;
    radiusGrowthRate: number;
    baseLifetime: number;
    baseAlpha: number;
    spawnInterval: number; // Seconds between spawns
    ringThickness: number;
}

/**
 * Default wake configuration
 */
export const DEFAULT_WAKE_CONFIG: WakeConfig = {
    maxParticles: 50,
    minSpeed: 0.1,
    baseRadius: 0.1,
    radiusGrowthRate: 0.5,
    baseLifetime: 2.0,
    baseAlpha: 0.6,
    spawnInterval: 0.1,
    ringThickness: 0.05,
} as const;

/**
 * PlayerWakeEffect - Visual wake trail system
 *
 * Creates and renders expanding ring particles behind player
 * when moving through water.
 */
export class PlayerWakeEffect {
    private wakeParticles: WakeParticle[] = [];
    private timeSinceLastSpawn = 0;

    // Configuration
    private config: WakeConfig;

    // Debug state
    private debugMode = false;

    constructor(config: Partial<WakeConfig> = {}) {
        this.config = { ...DEFAULT_WAKE_CONFIG, ...config };

        console.log('[PlayerWakeEffect] Initialized with', {
            maxParticles: this.config.maxParticles,
            minSpeed: this.config.minSpeed,
            baseLifetime: this.config.baseLifetime,
        });
    }

    /**
     * Update wake particles
     * @param deltaTime - Time since last frame in seconds
     * @param playerPosition - Current player position
     * @param playerVelocity - Current player velocity
     * @param isInWater - Whether player is in water
     */
    public update(
        deltaTime: number,
        playerPosition: vec3,
        playerVelocity: vec3,
        isInWater: boolean
    ): void {
        // Only create wake if player is in water and moving
        if (isInWater) {
            const speed = Math.sqrt(
                playerVelocity[0] ** 2 +
                playerVelocity[1] ** 2 +
                playerVelocity[2] ** 2
            );

            if (speed >= this.config.minSpeed) {
                this.timeSinceLastSpawn += deltaTime;

                // Spawn new wake particles at interval
                while (this.timeSinceLastSpawn >= this.config.spawnInterval) {
                    this.spawnWakeParticle(playerPosition, playerVelocity, speed);
                    this.timeSinceLastSpawn -= this.config.spawnInterval;
                }
            }
        }

        // Update existing particles
        this.updateParticles(deltaTime);
    }

    /**
     * Spawn a new wake particle
     */
    private spawnWakeParticle(playerPosition: vec3, playerVelocity: vec3, speed: number): void {
        // Calculate position behind player (for wake trail effect)
        const velLength = Math.sqrt(playerVelocity[0] ** 2 + playerVelocity[2] ** 2);
        let behindPos: vec3;

        if (velLength < 0.001) {
            // Player barely moving, spawn at position
            behindPos = [...playerPosition] as vec3;
        } else {
            // Spawn slightly behind player based on movement direction
            const dirX = -playerVelocity[0] / velLength;
            const dirZ = -playerVelocity[2] / velLength;
            behindPos = [
                playerPosition[0] + dirX * 0.3,
                playerPosition[1], // At player's feet level
                playerPosition[2] + dirZ * 0.3,
            ] as vec3;
        }

        // Scale properties based on speed
        const speedFactor = Math.min(speed / 5.0, 1.0);
        const initialRadius = this.config.baseRadius + speedFactor * 0.2;
        const maxRadius = 1.0 + speedFactor * 0.5;
        const lifetime = this.config.baseLifetime + speedFactor * 0.5;

        const particle: WakeParticle = {
            position: behindPos,
            velocity: [playerVelocity[0] * 0.1, 0, playerVelocity[2] * 0.1],
            radius: initialRadius,
            maxRadius,
            alpha: this.config.baseAlpha * speedFactor,
            age: 0,
            lifetime,
            thickness: this.config.ringThickness * (1 + speedFactor),
        };

        this.wakeParticles.push(particle);

        // Cull old particles if over limit
        if (this.wakeParticles.length > this.config.maxParticles) {
            this.wakeParticles.shift();
        }

        if (this.debugMode) {
            console.log(`[PlayerWakeEffect] Spawned wake particle (total: ${this.wakeParticles.length})`);
        }
    }

    /**
     * Update all wake particles
     */
    private updateParticles(deltaTime: number): void {
        this.wakeParticles = this.wakeParticles.filter(particle => {
            particle.age += deltaTime;

            // Expand radius
            particle.radius += this.config.radiusGrowthRate * deltaTime;

            // Fade out based on age
            const ageRatio = particle.age / particle.lifetime;
            particle.alpha = particle.alpha * (1 - ageRatio);

            // Move slightly with initial velocity
            particle.position[0] += particle.velocity[0] * deltaTime;
            particle.position[2] += particle.velocity[2] * deltaTime;

            // Remove dead particles
            return particle.age < particle.lifetime;
        });
    }

    /**
     * Get all wake particles for rendering
     */
    public getWakeParticles(): ReadonlyArray<WakeParticle> {
        return this.wakeParticles;
    }

    /**
     * Get wake particle count
     */
    public getParticleCount(): number {
        return this.wakeParticles.length;
    }

    /**
     * Clear all wake particles
     */
    public clear(): void {
        this.wakeParticles = [];
        this.timeSinceLastSpawn = 0;
    }

    /**
     * Set configuration values
     */
    public setConfig(config: Partial<WakeConfig>): void {
        this.config = { ...this.config, ...config };
    }

    /**
     * Get current configuration
     */
    public getConfig(): WakeConfig {
        return { ...this.config };
    }

    /**
     * Enable debug mode for console logging
     */
    public setDebugMode(enabled: boolean): void {
        this.debugMode = enabled;
    }

    /**
     * Check if debug mode is enabled
     */
    public isDebugMode(): boolean {
        return this.debugMode;
    }

    /**
     * Clean up resources
     */
    public destroy(): void {
        this.wakeParticles = [];
    }
}

/**
 * PlayerWakeRenderer - Three.js rendering for wake effects
 *
 * Renders wake particles as expanding rings on the water surface.
 */
export class PlayerWakeRenderer {
    private geometry: THREE.RingGeometry;
    private material: THREE.MeshBasicMaterial;
    private mesh: THREE.InstancedMesh;
    private dummy = new THREE.Object3D();
    private color = new THREE.Color();

    constructor(maxParticles: number = 50) {
        // Create instanced mesh for wake rings
        this.geometry = new THREE.RingGeometry(0.1, 0.15, 32);
        this.material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide,
            depthWrite: false,
        });

        this.mesh = new THREE.InstancedMesh(this.geometry, this.material, maxParticles);
        this.mesh.count = 0;
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    }

    /**
     * Update instanced mesh with wake particle data
     */
    public render(wakeParticles: ReadonlyArray<WakeParticle>): void {
        this.mesh.count = wakeParticles.length;

        for (let i = 0; i < wakeParticles.length; i++) {
            const particle = wakeParticles[i];

            // Position
            this.dummy.position.set(
                particle.position[0],
                particle.position[1],
                particle.position[2]
            );

            // Rotate to face upward (horizontal ring)
            this.dummy.rotation.set(-Math.PI / 2, 0, 0);

            // Scale based on radius
            this.dummy.scale.set(particle.radius, particle.radius, 1);

            this.dummy.updateMatrix();
            this.mesh.setMatrixAt(i, this.dummy.matrix);

            // Set color with alpha
            this.color.setRGB(1, 1, 1);
            this.mesh.setColorAt(i, this.color);
        }

        this.mesh.instanceMatrix.needsUpdate = true;
        if (this.mesh.instanceColor) {
            this.mesh.instanceColor.needsUpdate = true;
        }
    }

    /**
     * Get the Three.js mesh for adding to scene
     */
    public getMesh(): THREE.InstancedMesh {
        return this.mesh;
    }

    /**
     * Set wake ring color
     */
    public setColor(color: number | string | THREE.Color): void {
        this.material.color.set(color);
    }

    /**
     * Clean up resources
     */
    public dispose(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}

export default PlayerWakeEffect;
