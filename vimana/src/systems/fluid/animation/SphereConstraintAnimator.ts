/**
 * SphereConstraintAnimator.ts - Plane to Sphere Tunnel Animation
 * ==============================================================
 *
 * Animates the water simulation bounds from a flat plane to a hollow
 * sphere tunnel as duetProgress goes from 0→1.
 *
 * Based on: https://github.com/matsuoka-601/WaterBall/blob/master/main.ts
 *
 * Animation States:
 * - uDuetProgress = 0.0: Flat plane water (ArenaFloor)
 * - uDuetProgress = 0.5: Beginning to curve upward
 * - uDuetProgress = 1.0: Complete sphere tunnel (hollow center)
 */

import { vec3 } from '../types';
import MLSMPMSimulator from '../MLSMPMSimulator';

export interface SphereAnimatorConfig {
    simulator: MLSMPMSimulator;
    initBoxSize: vec3;
    minClosingSpeed?: number;
    maxOpeningSpeed?: number;
    tunnelOpenThreshold?: number;
}

export interface SphereAnimationState {
    boxWidthRatio: number;
    targetBoxWidthRatio: number;
    isTunnelOpen: boolean;
    tunnelRadius: number;
}

/**
 * SphereConstraintAnimator - Handles plane→sphere animation
 *
 * The animation works by shrinking the Z dimension of the simulation bounds.
 * As the box shrinks, the sphere constraint in g2p.wgsl pushes particles
 * toward the sphere surface, creating a hollow tunnel center.
 */
export class SphereConstraintAnimator {
    // Reference to the simulator
    private simulator: MLSMPMSimulator;

    // Initial box dimensions (constant)
    private readonly initBoxSize: vec3;

    // Animation state
    private boxWidthRatio: number = 0.5;
    private targetBoxWidthRatio: number = 0.5;

    // Animation parameters (speed limits for smooth transitions)
    private readonly minClosingSpeed: number;  // Maximum speed when shrinking
    private readonly maxOpeningSpeed: number;  // Maximum speed when expanding
    private readonly tunnelOpenThreshold: number; // Ratio at which tunnel is considered "open"

    constructor(config: SphereAnimatorConfig) {
        this.simulator = config.simulator;
        this.initBoxSize = config.initBoxSize;

        // Animation speed limits (from WaterBall)
        this.minClosingSpeed = config.minClosingSpeed ?? -0.01;
        this.maxOpeningSpeed = config.maxOpeningSpeed ?? 0.04;
        this.tunnelOpenThreshold = config.tunnelOpenThreshold ?? 0.95;

        // Initialize simulator with starting box size
        const startBoxSize = this.calculateBoxSize(0.5); // Start at half ratio
        this.simulator.changeBoxSize(startBoxSize);
    }

    /**
     * Calculate the real box size based on the box width ratio
     * The Z dimension shrinks to create the sphere/tunnel shape
     */
    private calculateBoxSize(ratio: number): vec3 {
        return [
            this.initBoxSize[0],  // X remains constant
            this.initBoxSize[1],  // Y remains constant
            this.initBoxSize[2] * Math.max(0.3, Math.min(1.0, ratio)), // Z shrinks (min 30%)
        ] as vec3;
    }

    /**
     * Main update method - call each frame
     *
     * @param duetProgress - 0.0 to 1.0 (puzzle completion progress)
     * @param deltaTime - Frame time in seconds
     */
    public update(duetProgress: number, deltaTime: number): void {
        // Map duetProgress to target box ratio
        // duetProgress 0.0 → box ratio 0.5 (starting state)
        // duetProgress 1.0 → box ratio 0.3 (fully formed tunnel)
        this.targetBoxWidthRatio = 0.5 - (duetProgress * 0.2);

        // Calculate the change needed
        const dVal = this.targetBoxWidthRatio - this.boxWidthRatio;

        // Clamp the change to respect speed limits
        // This creates smooth, natural-feeling animation
        let adjusted = dVal;
        if (adjusted < this.minClosingSpeed) {
            adjusted = this.minClosingSpeed;
        } else if (adjusted > this.maxOpeningSpeed) {
            adjusted = this.maxOpeningSpeed;
        }

        // Apply the change
        this.boxWidthRatio += adjusted;

        // Clamp to valid range
        this.boxWidthRatio = Math.max(0.3, Math.min(1.0, this.boxWidthRatio));

        // Update the simulator's box size
        const realBoxSize = this.calculateBoxSize(this.boxWidthRatio);
        this.simulator.changeBoxSize(realBoxSize);
    }

    /**
     * Get the current animation state
     */
    public getState(): SphereAnimationState {
        return {
            boxWidthRatio: this.boxWidthRatio,
            targetBoxWidthRatio: this.targetBoxWidthRatio,
            isTunnelOpen: this.isTunnelOpen(),
            tunnelRadius: this.getTunnelRadius(),
        };
    }

    /**
     * Get the current box width ratio
     * Useful for debug visualization
     */
    public getBoxRatio(): number {
        return this.boxWidthRatio;
    }

    /**
     * Get the tunnel radius
     * This is the sphere radius from the simulator
     */
    public getTunnelRadius(): number {
        return this.simulator.sphereRadius;
    }

    /**
     * Check if the tunnel is fully open (player can enter)
     * Tunnel is considered open when box ratio reaches threshold
     */
    public isTunnelOpen(): boolean {
        return this.boxWidthRatio <= (1.0 - this.tunnelOpenThreshold * 0.5);
    }

    /**
     * Reset the animation to initial state
     */
    public reset(): void {
        this.boxWidthRatio = 0.5;
        this.targetBoxWidthRatio = 0.5;
        const startBoxSize = this.calculateBoxSize(0.5);
        this.simulator.changeBoxSize(startBoxSize);
    }

    /**
     * Set the box ratio directly (for testing/debug)
     */
    public setBoxRatio(ratio: number): void {
        this.boxWidthRatio = Math.max(0.3, Math.min(1.0, ratio));
        this.targetBoxWidthRatio = this.boxWidthRatio;
        const realBoxSize = this.calculateBoxSize(this.boxWidthRatio);
        this.simulator.changeBoxSize(realBoxSize);
    }
}

export default SphereConstraintAnimator;
