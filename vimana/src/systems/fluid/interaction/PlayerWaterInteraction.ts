/**
 * PlayerWaterInteraction.ts - Player-to-Water Interaction System
 * =============================================================
 *
 * Manages coupling between player character and fluid simulation.
 * Player displaces water particles, experiences buoyancy, and creates wake effects.
 *
 * Based on WaterBall mouse interaction:
 * https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/mls-mpm.ts#L125
 */

import { vec3 } from '../types';

/**
 * Player water interaction state
 */
export interface PlayerWaterInteractionState {
    // Player state
    position: vec3;
    velocity: vec3;
    previousPosition: vec3;

    // Collision geometry
    colliderRadius: number;
    colliderHeight: number;
    buoyancyFactor: number;

    // Water interaction
    isInWater: boolean;
    waterSurfaceY: number;
    immersionDepth: number;

    // Physics state
    wasInWater: boolean;
    entrySpeed: number;
    exitSpeed: number;
}

/**
 * Player interaction force data for GPU
 */
export interface PlayerInteractionData {
    position: [number, number, number];
    velocity: [number, number, number];
    radius: number;
    strength: number;
}

/**
 * Default configuration for player-water interaction
 */
export const DEFAULT_PLAYER_CONFIG = {
    colliderRadius: 0.8,
    colliderHeight: 1.8,
    buoyancyFactor: 0.3,
    waterSurfaceY: 0.0,
    dragCoefficient: 0.5,
    wakeLifetime: 2.0,
    wakeMaxRadius: 1.5,
    forceMultiplier: 2.0,
} as const;

/**
 * PlayerWaterInteraction - Main player-water system
 *
 * Manages coupling between player and the fluid simulation.
 * Forces are uploaded to GPU each frame for the updateGrid shader.
 */
export class PlayerWaterInteraction {
    private state: PlayerWaterInteractionState;

    // Force buffer for GPU (position + velocity + radius + strength = 8 floats)
    private interactionBuffer: GPUBuffer;
    private interactionValues: Float32Array;

    // Configuration
    private config: typeof DEFAULT_PLAYER_CONFIG;

    // Audio callbacks
    private onWaterEntry?: (speed: number, position: vec3) => void;
    private onWaterExit?: (speed: number, position: vec3) => void;
    private onWaterMovement?: (speed: number, position: vec3) => void;

    // Debug state
    private debugMode = false;

    // Player physics state (for buoyancy/drag application)
    private playerMass = 80.0; // kg
    private pendingBuoyancyForce: vec3 = [0, 0, 0];
    private pendingDragFactor = 1.0;

    constructor(
        private device: GPUDevice,
        config: Partial<typeof DEFAULT_PLAYER_CONFIG> = {}
    ) {
        this.config = { ...DEFAULT_PLAYER_CONFIG, ...config };

        // Create interaction buffer: position(3) + velocity(3) + radius(1) + strength(1) = 8 floats
        const bufferSize = 8 * 4;
        this.interactionBuffer = this.device.createBuffer({
            label: 'Player Interaction Buffer',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.interactionValues = new Float32Array(8);

        // Initialize state
        this.state = {
            position: [0, 100, 0], // Start above water
            velocity: [0, 0, 0],
            previousPosition: [0, 100, 0],
            colliderRadius: this.config.colliderRadius,
            colliderHeight: this.config.colliderHeight,
            buoyancyFactor: this.config.buoyancyFactor,
            isInWater: false,
            waterSurfaceY: this.config.waterSurfaceY,
            immersionDepth: 0.0,
            wasInWater: false,
            entrySpeed: 0.0,
            exitSpeed: 0.0,
        };

        console.log('[PlayerWaterInteraction] Initialized with', {
            colliderRadius: this.config.colliderRadius,
            colliderHeight: this.config.colliderHeight,
            buoyancyFactor: this.config.buoyancyFactor,
            waterSurfaceY: this.config.waterSurfaceY,
        });
    }

    /**
     * Get the interaction buffer for use in compute shaders
     */
    public getInteractionBuffer(): GPUBuffer {
        return this.interactionBuffer;
    }

    /**
     * Get current interaction state
     */
    public getState(): PlayerWaterInteractionState {
        return { ...this.state };
    }

    /**
     * Get pending buoyancy force (to be applied to player)
     */
    public getBuoyancyForce(): vec3 {
        return [...this.pendingBuoyancyForce] as vec3;
    }

    /**
     * Get pending drag factor (to be applied to player velocity)
     */
    public getDragFactor(): number {
        return this.pendingDragFactor;
    }

    /**
     * Update player position from character controller
     * Call this each frame before update()
     */
    public setPlayerPosition(position: vec3): void {
        this.state.previousPosition = [...this.state.position] as vec3;
        this.state.position = [...position] as vec3;
    }

    /**
     * Update player velocity from character controller
     * Call this each frame before update()
     */
    public setPlayerVelocity(velocity: vec3): void {
        this.state.velocity = [...velocity] as vec3;
    }

    /**
     * Set audio callbacks for water sounds
     */
    public setAudioCallbacks(callbacks: {
        onWaterEntry?: (speed: number, position: vec3) => void;
        onWaterExit?: (speed: number, position: vec3) => void;
        onWaterMovement?: (speed: number, position: vec3) => void;
    }): void {
        this.onWaterEntry = callbacks.onWaterEntry;
        this.onWaterExit = callbacks.onWaterExit;
        this.onWaterMovement = callbacks.onWaterMovement;
    }

    /**
     * Update interaction state each frame
     * @param deltaTime - Time since last frame in seconds
     */
    public update(deltaTime: number): void {
        // Store previous water state for entry/exit detection
        this.state.wasInWater = this.state.isInWater;

        // Calculate feet position
        const feetY = this.state.position[1] - (this.state.colliderHeight * 0.5);

        // Check if player is in water
        this.state.isInWater = feetY < this.state.waterSurfaceY;

        // Calculate immersion depth (0 = dry, 1 = fully submerged)
        if (this.state.isInWater) {
            const submergedAmount = this.state.waterSurfaceY - feetY;
            this.state.immersionDepth = Math.min(submergedAmount / this.state.colliderHeight, 1.0);
        } else {
            this.state.immersionDepth = 0.0;
        }

        // Detect water entry
        if (!this.state.wasInWater && this.state.isInWater) {
            this.state.entrySpeed = Math.sqrt(
                this.state.velocity[0] ** 2 +
                this.state.velocity[1] ** 2 +
                this.state.velocity[2] ** 2
            );
            if (this.onWaterEntry) {
                this.onWaterEntry(this.state.entrySpeed, this.state.position);
            }
            if (this.debugMode) {
                console.log(`[PlayerWaterInteraction] Entered water at speed ${this.state.entrySpeed.toFixed(2)} m/s`);
            }
        }

        // Detect water exit
        if (this.state.wasInWater && !this.state.isInWater) {
            this.state.exitSpeed = Math.sqrt(
                this.state.velocity[0] ** 2 +
                this.state.velocity[1] ** 2 +
                this.state.velocity[2] ** 2
            );
            if (this.onWaterExit) {
                this.onWaterExit(this.state.exitSpeed, this.state.position);
            }
            if (this.debugMode) {
                console.log(`[PlayerWaterInteraction] Exited water at speed ${this.state.exitSpeed.toFixed(2)} m/s`);
            }
        }

        // Calculate buoyancy and drag
        this.calculateBuoyancyAndDrag();

        // Continuous water movement sound
        if (this.state.isInWater && this.onWaterMovement) {
            const speed = this.getPlayerSpeed();
            if (speed > 0.5) {
                this.onWaterMovement(speed, this.state.position);
            }
        }

        // Upload interaction data to GPU
        this.uploadInteractionData();
    }

    /**
     * Calculate buoyancy force and drag for player
     */
    private calculateBuoyancyAndDrag(): void {
        if (!this.state.isInWater) {
            this.pendingBuoyancyForce = [0, 0, 0];
            this.pendingDragFactor = 1.0;
            return;
        }

        // Buoyancy calculation
        const densityWater = 1.0;
        const densityPlayer = 0.98; // Slightly less than water (human body)
        const gravity = 9.81;

        // Approximate volume of submerged cylinder
        const radius = this.state.colliderRadius;
        const height = this.state.colliderHeight * this.state.immersionDepth;
        const submergedVolume = Math.PI * radius * radius * height;

        // Buoyancy force = weight of displaced fluid
        const buoyancyMagnitude = submergedVolume * densityWater * gravity;
        const playerWeight = this.playerMass * gravity;
        const netBuoyancyForce = Math.max(buoyancyMagnitude - playerWeight, -100);

        // Clamp to prevent shooting out of water
        const clampedForce = Math.min(netBuoyancyForce, 500);

        this.pendingBuoyancyForce = [0, clampedForce * this.state.buoyancyFactor, 0];

        // Water drag (slows movement)
        const dragCoeff = this.config.dragCoefficient * this.state.immersionDepth;
        this.pendingDragFactor = Math.max(1.0 - dragCoeff, 0.5);
    }

    /**
     * Upload interaction data to GPU buffer
     */
    private uploadInteractionData(): void {
        // Calculate strength based on immersion depth
        const strength = this.state.isInWater ? this.state.immersionDepth * this.config.forceMultiplier : 0.0;

        this.interactionValues[0] = this.state.position[0];
        this.interactionValues[1] = this.state.position[1];
        this.interactionValues[2] = this.state.position[2];
        this.interactionValues[3] = this.state.velocity[0];
        this.interactionValues[4] = this.state.velocity[1];
        this.interactionValues[5] = this.state.velocity[2];
        this.interactionValues[6] = this.state.colliderRadius;
        this.interactionValues[7] = strength;

        this.device.queue.writeBuffer(this.interactionBuffer, 0, this.interactionValues);
    }

    /**
     * Get player movement direction (normalized)
     */
    public getPlayerMovementDirection(): vec3 {
        const delta: vec3 = [
            this.state.position[0] - this.state.previousPosition[0],
            this.state.position[1] - this.state.previousPosition[1],
            this.state.position[2] - this.state.previousPosition[2],
        ];

        const length = Math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2);
        if (length < 0.001) {
            return [0, 0, 0];
        }

        return [delta[0] / length, delta[1] / length, delta[2] / length];
    }

    /**
     * Get player speed
     */
    public getPlayerSpeed(): number {
        return Math.sqrt(
            this.state.velocity[0] ** 2 +
            this.state.velocity[1] ** 2 +
            this.state.velocity[2] ** 2
        );
    }

    /**
     * Check if player is in water
     */
    public isInWater(): boolean {
        return this.state.isInWater;
    }

    /**
     * Get immersion depth (0-1)
     */
    public getImmersionDepth(): number {
        return this.state.immersionDepth;
    }

    /**
     * Get water surface Y level
     */
    public getWaterSurfaceY(): number {
        return this.state.waterSurfaceY;
    }

    /**
     * Set water surface Y level
     */
    public setWaterSurfaceY(y: number): void {
        this.state.waterSurfaceY = y;
        if (this.debugMode) {
            console.log(`[PlayerWaterInteraction] Water surface Y set to ${y}`);
        }
    }

    /**
     * Set configuration values
     */
    public setConfig(config: Partial<typeof DEFAULT_PLAYER_CONFIG>): void {
        this.config = { ...this.config, ...config };

        if (config.colliderRadius !== undefined) {
            this.state.colliderRadius = config.colliderRadius;
        }
        if (config.colliderHeight !== undefined) {
            this.state.colliderHeight = config.colliderHeight;
        }
        if (config.buoyancyFactor !== undefined) {
            this.state.buoyancyFactor = config.buoyancyFactor;
        }
        if (config.waterSurfaceY !== undefined) {
            this.state.waterSurfaceY = config.waterSurfaceY;
        }
    }

    /**
     * Get current configuration
     */
    public getConfig(): typeof DEFAULT_PLAYER_CONFIG {
        return { ...this.config };
    }

    /**
     * Set player mass (for buoyancy calculations)
     */
    public setPlayerMass(mass: number): void {
        this.playerMass = Math.max(1, mass);
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
     * Reset interaction state
     */
    public reset(): void {
        this.state.isInWater = false;
        this.state.wasInWater = false;
        this.state.immersionDepth = 0.0;
        this.state.entrySpeed = 0.0;
        this.state.exitSpeed = 0.0;
        this.pendingBuoyancyForce = [0, 0, 0];
        this.pendingDragFactor = 1.0;
        this.interactionValues.fill(0);
        this.device.queue.writeBuffer(this.interactionBuffer, 0, this.interactionValues);
    }

    /**
     * Clean up GPU resources
     */
    public destroy(): void {
        this.interactionBuffer.destroy();
    }
}

export default PlayerWaterInteraction;
