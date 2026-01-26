/**
 * HarpWaterInteraction.ts - Harp-to-Water Interaction System
 * ==========================================================
 *
 * Couples harp string vibrations to the fluid simulation.
 * Each of the 6 harp strings transfers energy to the water
 * through velocity fields that displace particles.
 *
 * Based on WaterBall mouse interaction:
 * https://github.com/matsuoka-601/WaterBall/blob/master/mls-mpm/mls-mpm.ts#L125
 */

import { vec3 } from '../types';

/**
 * String water interaction state
 */
export interface StringWaterInteraction {
    stringIndex: number;        // 0-5
    stringPosition: vec3;       // World-space string position
    waterSurfaceY: number;      // Y-level of water surface

    // Current vibration state
    amplitude: number;          // 0-1, vibrational intensity
    phase: number;              // Oscillation phase
    frequency: number;          // String frequency (Hz)

    // Interaction parameters
    influenceRadius: number;    // How far string affects water
    forceMultiplier: number;    // Strength of interaction
}

/**
 * String configuration from HarpRoom scene
 */
export const STRING_POSITIONS: vec3[] = [
    [49.8, 2.3, -7.7],  // String 1 - C4
    [49.9, 2.3, -7.3],  // String 2 - D4
    [49.9, 2.4, -7.0],  // String 3 - E4
    [49.9, 2.4, -6.7],  // String 4 - F4
    [49.8, 2.3, -6.4],  // String 5 - G4
    [49.8, 2.3, -6.1],  // String 6 - A4
];

/**
 * Musical frequencies for each string
 */
export const STRING_FREQUENCIES = [
    261.63,  // C4
    293.66,  // D4
    329.63,  // E4
    349.23,  // F4
    392.00,  // G4
    440.00,  // A4
];

/**
 * Default configuration for harp-water interaction
 */
export const DEFAULT_HARP_CONFIG = {
    influenceRadius: 8.0,     // How far string affects water
    forceMultiplier: 5.0,     // Strength of force
    decayRate: 0.95,          // Amplitude decay per frame
    waterSurfaceY: 0.0,       // Default water level
};

/**
 * HarpWaterInteraction - Main interaction system
 *
 * Manages coupling between harp strings and the fluid simulation.
 * Forces are uploaded to GPU each frame for the updateGrid shader.
 */
export class HarpWaterInteraction {
    private interactions: Map<number, StringWaterInteraction> = new Map();

    // Force buffer for GPU (6 strings × 3 components = 18 floats)
    private forceBuffer: GPUBuffer;
    private forceValues: Float32Array;

    // Configuration
    private config: typeof DEFAULT_HARP_CONFIG;

    // Debug state
    private debugMode = false;

    constructor(
        private device: GPUDevice,
        config: Partial<typeof DEFAULT_HARP_CONFIG> = {}
    ) {
        this.config = { ...DEFAULT_HARP_CONFIG, ...config };

        // Create force buffer: 6 strings × 3 components (xyz) × 4 bytes
        const bufferSize = 6 * 3 * 4;
        this.forceBuffer = this.device.createBuffer({
            label: 'Harp String Forces Buffer',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.forceValues = new Float32Array(18);

        // Initialize all string interactions
        this.initializeStrings();

        console.log('[HarpWaterInteraction] Initialized with', {
            stringCount: 6,
            influenceRadius: this.config.influenceRadius,
            forceMultiplier: this.config.forceMultiplier,
        });
    }

    /**
     * Initialize string interaction states
     */
    private initializeStrings(): void {
        for (let i = 0; i < 6; i++) {
            this.interactions.set(i, {
                stringIndex: i,
                stringPosition: [...STRING_POSITIONS[i]] as vec3,
                waterSurfaceY: this.config.waterSurfaceY,
                amplitude: 0,
                phase: 0,
                frequency: STRING_FREQUENCIES[i],
                influenceRadius: this.config.influenceRadius,
                forceMultiplier: this.config.forceMultiplier,
            });
        }
    }

    /**
     * Get the force buffer for use in compute shaders
     */
    public getForceBuffer(): GPUBuffer {
        return this.forceBuffer;
    }

    /**
     * Called when a harp string is plucked
     * @param stringIndex - 0-5, which string was plucked
     * @param intensity - 0-1, how hard the string was plucked
     */
    public onStringPlucked(stringIndex: number, intensity: number): void {
        if (stringIndex < 0 || stringIndex >= 6) {
            console.warn(`[HarpWaterInteraction] Invalid string index: ${stringIndex}`);
            return;
        }

        const interaction = this.interactions.get(stringIndex);
        if (!interaction) {
            console.warn(`[HarpWaterInteraction] No interaction for string ${stringIndex}`);
            return;
        }

        // Set amplitude and reset phase
        interaction.amplitude = Math.max(0, Math.min(1, intensity));
        interaction.phase = 0;

        if (this.debugMode) {
            console.log(`[HarpWaterInteraction] String ${stringIndex} plucked with intensity ${intensity.toFixed(2)}`);
        }
    }

    /**
     * Update forces for all strings
     * Called each frame before simulation step
     * @param deltaTime - Time since last frame in seconds
     */
    public update(deltaTime: number): void {
        // Clear force values
        this.forceValues.fill(0);

        for (let i = 0; i < 6; i++) {
            const interaction = this.interactions.get(i);
            if (!interaction) {
                continue;
            }

            // Skip strings with negligible amplitude
            if (interaction.amplitude <= 0.001) {
                continue;
            }

            // Calculate force based on string vibration
            const force = this.calculateStringForce(interaction, deltaTime);

            // Store in force buffer (offset by 3 per string)
            const offset = i * 3;
            this.forceValues[offset + 0] = force[0]; // X
            this.forceValues[offset + 1] = force[1]; // Y
            this.forceValues[offset + 2] = force[2]; // Z

            // Decay amplitude
            interaction.amplitude *= this.config.decayRate;

            // Update phase
            interaction.phase += interaction.frequency * deltaTime * Math.PI * 2;
        }

        // Upload forces to GPU
        this.device.queue.writeBuffer(this.forceBuffer, 0, this.forceValues);
    }

    /**
     * Calculate force vector for a vibrating string
     * @param interaction - String interaction state
     * @param deltaTime - Time step
     * @returns Force vector [x, y, z]
     */
    private calculateStringForce(interaction: StringWaterInteraction, deltaTime: number): vec3 {
        const { amplitude, phase, forceMultiplier, stringPosition } = interaction;

        // Primary force is downward (into water)
        const forceY = -amplitude * forceMultiplier;

        // Slight horizontal spread based on string vibration pattern
        // Creates more natural ripples instead of perfect circles
        const vibrationPattern = Math.sin(phase);
        const spreadX = vibrationPattern * amplitude * 0.3;
        const spreadZ = Math.cos(phase * 0.7) * amplitude * 0.3;

        return [spreadX, forceY, spreadZ];
    }

    /**
     * Get current force for a specific string
     * @param stringIndex - 0-5, which string
     * @returns Current force vector [x, y, z]
     */
    public getStringForce(stringIndex: number): vec3 {
        if (stringIndex < 0 || stringIndex >= 6) {
            return [0, 0, 0];
        }

        const offset = stringIndex * 3;
        return [
            this.forceValues[offset + 0],
            this.forceValues[offset + 1],
            this.forceValues[offset + 2],
        ];
    }

    /**
     * Get interaction state for a specific string
     */
    public getStringInteraction(stringIndex: number): StringWaterInteraction | undefined {
        return this.interactions.get(stringIndex);
    }

    /**
     * Get all interaction states
     */
    public getAllInteractions(): StringWaterInteraction[] {
        return Array.from(this.interactions.values());
    }

    /**
     * Set configuration values
     */
    public setConfig(config: Partial<typeof DEFAULT_HARP_CONFIG>): void {
        this.config = { ...this.config, ...config };

        // Update existing interactions
        for (const interaction of this.interactions.values()) {
            if (config.influenceRadius !== undefined) {
                interaction.influenceRadius = config.influenceRadius;
            }
            if (config.forceMultiplier !== undefined) {
                interaction.forceMultiplier = config.forceMultiplier;
            }
            if (config.waterSurfaceY !== undefined) {
                interaction.waterSurfaceY = config.waterSurfaceY;
            }
        }
    }

    /**
     * Get current configuration
     */
    public getConfig(): typeof DEFAULT_HARP_CONFIG {
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
     * Reset all string amplitudes to zero
     */
    public reset(): void {
        for (const interaction of this.interactions.values()) {
            interaction.amplitude = 0;
            interaction.phase = 0;
        }
        this.forceValues.fill(0);
        this.device.queue.writeBuffer(this.forceBuffer, 0, this.forceValues);
    }

    /**
     * Clean up GPU resources
     */
    public destroy(): void {
        this.forceBuffer.destroy();
        this.interactions.clear();
    }
}

export default HarpWaterInteraction;
