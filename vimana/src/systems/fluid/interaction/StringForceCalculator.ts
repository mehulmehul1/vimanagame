/**
 * StringForceCalculator.ts - String Force Calculation Utilities
 * =============================================================
 *
 * Helper functions for calculating forces from harp string vibrations.
 * Provides mathematical models for converting string parameters to
 * fluid simulation forces.
 */

import { vec3 } from '../types';
import { STRING_FREQUENCIES } from './HarpWaterInteraction';

/**
 * Force calculation parameters
 */
export interface ForceCalculationParams {
    amplitude: number;        // Current vibration amplitude (0-1)
    phase: number;            // Oscillation phase (radians)
    frequency: number;        // String frequency (Hz)
    stringPosition: vec3;     // World-space string position
    targetPosition: vec3;     // Target point (e.g., water surface point)
    forceMultiplier: number;  // Overall force strength
}

/**
 * Force calculation result
 */
export interface ForceResult {
    force: vec3;              // Force vector [x, y, z]
    magnitude: number;        // Force magnitude
    direction: vec3;          // Normalized direction
    falloff: number;          // Distance-based falloff (0-1)
}

/**
 * Calculate force from string vibration
 * @param params - Force calculation parameters
 * @returns Force result with vector, magnitude, direction, and falloff
 */
export function calculateStringForce(params: ForceCalculationParams): ForceResult {
    const {
        amplitude,
        phase,
        stringPosition,
        targetPosition,
        forceMultiplier,
    } = params;

    // Direction from string to target
    const direction = [
        targetPosition[0] - stringPosition[0],
        targetPosition[1] - stringPosition[1],
        targetPosition[2] - stringPosition[2],
    ] as vec3;

    // Calculate distance
    const distance = Math.sqrt(
        direction[0] * direction[0] +
        direction[1] * direction[1] +
        direction[2] * direction[2]
    );

    // Normalize direction
    let normalizedDirection: vec3;
    if (distance > 0.0001) {
        normalizedDirection = [
            direction[0] / distance,
            direction[1] / distance,
            direction[2] / distance,
        ];
    } else {
        normalizedDirection = [0, -1, 0]; // Default downward
    }

    // Calculate falloff based on distance (inverse square law approximation)
    const influenceRadius = 8.0;
    const falloff = Math.max(0, 1 - distance / influenceRadius);
    const falloffSquared = falloff * falloff; // Sharper falloff

    // Calculate base force magnitude
    const baseMagnitude = amplitude * forceMultiplier * falloffSquared;

    // Add vibration pattern modulation
    // Higher frequency strings have more rapid oscillation
    const vibrationModulation = Math.sin(phase) * 0.2 + 0.8; // 0.6 to 1.0
    const magnitude = baseMagnitude * vibrationModulation;

    // Calculate force vector
    const force: vec3 = [
        normalizedDirection[0] * magnitude * 0.3, // X - slight horizontal
        normalizedDirection[1] * magnitude,        // Y - primary direction (downward)
        normalizedDirection[2] * magnitude * 0.3, // Z - slight horizontal
    ];

    return {
        force,
        magnitude,
        direction: normalizedDirection,
        falloff: falloffSquared,
    };
}

/**
 * Calculate distance-based falloff
 * @param distance - Distance from string
 * @param radius - Influence radius
 * @returns Falloff factor (0-1)
 */
export function calculateFalloff(distance: number, radius: number = 8.0): number {
    if (distance >= radius) return 0;
    const normalizedDistance = distance / radius;
    // Use smooth cubic falloff: (1 - x)^3
    const falloff = 1 - normalizedDistance;
    return falloff * falloff * falloff;
}

/**
 * Get string frequency by index
 * @param stringIndex - 0-5
 * @returns Frequency in Hz
 */
export function getStringFrequency(stringIndex: number): number {
    if (stringIndex >= 0 && stringIndex < STRING_FREQUENCIES.length) {
        return STRING_FREQUENCIES[stringIndex];
    }
    return 440; // Default to A4
}

/**
 * Calculate phase from time and frequency
 * @param time - Time in seconds
 * @param frequency - Frequency in Hz
 * @returns Phase in radians
 */
export function calculatePhase(time: number, frequency: number): number {
    return (time * frequency * Math.PI * 2) % (Math.PI * 2);
}

/**
 * Calculate vibration pattern for visual ripple
 * Creates a sinusoidal pattern based on distance and time
 * @param distance - Distance from center
 * @param time - Time in seconds
 * @param frequency - Wave frequency
 * @param wavelength - Wave wavelength
 * @returns Vibration value (-1 to 1)
 */
export function calculateRipplePattern(
    distance: number,
    time: number,
    frequency: number = 2.0,
    wavelength: number = 2.0
): number {
    const k = (Math.PI * 2) / wavelength; // Wave number
    const omega = Math.PI * 2 * frequency; // Angular frequency
    const phase = k * distance - omega * time;
    return Math.sin(phase) * Math.exp(-distance * 0.3); // Decay with distance
}

/**
 * Compute force for multiple strings (superposition)
 * @param forces - Array of force vectors
 * @returns Combined force vector
 */
export function combineForces(forces: vec3[]): vec3 {
    const result: vec3 = [0, 0, 0];
    for (const force of forces) {
        result[0] += force[0];
        result[1] += force[1];
        result[2] += force[2];
    }
    return result;
}

/**
 * Clamp force magnitude to maximum value
 * @param force - Force vector
 * @param maxMagnitude - Maximum allowed magnitude
 * @returns Clamped force vector
 */
export function clampForce(force: vec3, maxMagnitude: number): vec3 {
    const magnitude = Math.sqrt(force[0] ** 2 + force[1] ** 2 + force[2] ** 2);
    if (magnitude <= maxMagnitude) return force;

    const scale = maxMagnitude / magnitude;
    return [force[0] * scale, force[1] * scale, force[2] * scale];
}

/**
 * StringForceCalculator - Main calculator class
 *
 * Provides a convenient interface for calculating string forces
 * with caching and optimization.
 */
export class StringForceCalculator {
    private cache = new Map<string, ForceResult>();
    private cacheHits = 0;
    private cacheMisses = 0;

    /**
     * Calculate force with caching
     * @param params - Force calculation parameters
     * @param cacheKey - Optional cache key
     */
    public calculate(params: ForceCalculationParams, cacheKey?: string): ForceResult {
        if (cacheKey && this.cache.has(cacheKey)) {
            this.cacheHits++;
            return this.cache.get(cacheKey)!;
        }

        this.cacheMisses++;
        const result = calculateStringForce(params);

        if (cacheKey) {
            this.cache.set(cacheKey, result);

            // Limit cache size
            if (this.cache.size > 1000) {
                const firstKey = this.cache.keys().next().value;
                this.cache.delete(firstKey);
            }
        }

        return result;
    }

    /**
     * Clear the calculation cache
     */
    public clearCache(): void {
        this.cache.clear();
        this.cacheHits = 0;
        this.cacheMisses = 0;
    }

    /**
     * Get cache statistics
     */
    public getCacheStats(): { hits: number; misses: number; size: number } {
        return {
            hits: this.cacheHits,
            misses: this.cacheMisses,
            size: this.cache.size,
        };
    }
}

export default StringForceCalculator;
