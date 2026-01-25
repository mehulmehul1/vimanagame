/**
 * GentleFeedback - Camera shake system for non-punitive feedback
 *
 * Provides subtle camera shake when player plays wrong note.
 * Uses sinusoidal patterns for organic, non-jarring motion.
 */

import * as THREE from 'three';

export interface ShakeConfig {
    /** Decay rate for shake intensity (higher = faster) */
    decayRate: number;
    /** Maximum shake offset in world units */
    maxOffset: number;
    /** Shake frequency multipliers for each axis */
    frequencies: {
        x: number;
        y: number;
        z: number;
    };
    /** Amplitude multipliers for each axis */
    amplitudes: {
        x: number;
        y: number;
        z: number;
    };
}

export class GentleFeedback {
    private camera: THREE.Camera;
    private isShaking: boolean = false;
    private shakeIntensity: number = 0;
    private shakeOffset: THREE.Vector3;
    private shakeEndTime: number = 0;
    private config: ShakeConfig;
    private enabled: boolean = true;

    /**
     * Shake intensity presets
     */
    public static readonly INTENSITY = {
        /** Subtle reminder - gentle nudge */
        SUBTLE: 0.1,
        /** Premature play - light hint */
        PREMATURE: 0.3,
        /** Wrong note - noticeable but not jarring */
        WRONG_NOTE: 0.5
    } as const;

    constructor(camera: THREE.Camera, config: Partial<ShakeConfig> = {}) {
        this.camera = camera;
        this.shakeOffset = new THREE.Vector3();

        this.config = {
            decayRate: 8.0,
            maxOffset: 0.5,
            frequencies: { x: 17, y: 23, z: 29 },
            amplitudes: { x: 0.5, y: 0.3, z: 0.2 },
            ...config
        };
    }

    /**
     * Start camera shake with given intensity
     *
     * @param intensity Shake intensity (0-1)
     * @param duration Duration in seconds (default: 0.5)
     */
    public shake(intensity: number, duration: number = 0.5): void {
        if (!this.enabled) return;

        // Clamp intensity
        this.shakeIntensity = Math.max(0, Math.min(1, intensity));

        this.isShaking = true;
        this.shakeEndTime = performance.now() + duration * 1000;
    }

    /**
     * Start subtle reminder shake
     */
    public shakeSubtle(): void {
        this.shake(GentleFeedback.INTENSITY.SUBTLE, 0.3);
    }

    /**
     * Start premature play shake
     */
    public shakePremature(): void {
        this.shake(GentleFeedback.INTENSITY.PREMATURE, 0.3);
    }

    /**
     * Start wrong note shake
     */
    public shakeWrongNote(): void {
        this.shake(GentleFeedback.INTENSITY.WRONG_NOTE, 0.5);
    }

    /**
     * Update shake - call every frame
     *
     * @param deltaTime Frame time in seconds
     */
    public update(deltaTime: number): void {
        if (!this.isShaking) return;

        const now = performance.now();

        // Check if shake should end
        if (now >= this.shakeEndTime || this.shakeIntensity < 0.01) {
            this.stopShake();
            return;
        }

        // Decay intensity: intensity *= exp(-dt * decayRate)
        this.shakeIntensity *= Math.exp(-deltaTime * this.config.decayRate);

        // Generate shake offset with different frequencies per axis
        const time = now / 1000;
        this.shakeOffset.set(
            Math.sin(time * this.config.frequencies.x) * this.shakeIntensity * this.config.amplitudes.x,
            Math.sin(time * this.config.frequencies.y) * this.shakeIntensity * this.config.amplitudes.y,
            Math.sin(time * this.config.frequencies.z) * this.shakeIntensity * this.config.amplitudes.z
        );

        // Clamp to maximum offset
        this.shakeOffset.clampLength(0, this.config.maxOffset * this.shakeIntensity);

        // Apply offset to camera (additive)
        if (this.camera) {
            this.camera.position.add(this.shakeOffset);
        }
    }

    /**
     * Stop shaking and reset offset
     */
    private stopShake(): void {
        this.isShaking = false;
        this.shakeIntensity = 0;
        this.shakeOffset.set(0, 0, 0);
    }

    /**
     * Immediately stop all shake and reset
     */
    public stop(): void {
        this.stopShake();
    }

    /**
     * Enable or disable shake feedback
     */
    public setEnabled(enabled: boolean): void {
        this.enabled = enabled;
        if (!enabled) {
            this.stop();
        }
    }

    /**
     * Check if currently shaking
     */
    public isActive(): boolean {
        return this.isShaking;
    }

    /**
     * Get current shake intensity
     */
    public getIntensity(): number {
        return this.shakeIntensity;
    }

    /**
     * Get current shake offset
     */
    public getOffset(): THREE.Vector3 {
        return this.shakeOffset;
    }

    /**
     * Update configuration
     */
    public updateConfig(updates: Partial<ShakeConfig>): void {
        this.config = { ...this.config, ...updates };
    }

    /**
     * Get current configuration
     */
    public getConfig(): Readonly<ShakeConfig> {
        return { ...this.config };
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.stop();
        // Camera is external reference, don't dispose it
    }
}
