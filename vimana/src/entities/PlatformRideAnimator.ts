/**
 * PlatformRideAnimator - Platform detachment and ride animation
 *
 * Animates the player platform detaching from the floor and riding
 * toward the vortex when duet is complete.
 */

import * as THREE from 'three';

export interface PlatformRideConfig {
    /** Target position (in front of vortex) */
    targetPosition: THREE.Vector3;
    /** Ride duration in seconds */
    rideDuration: number;
    /** Detachment duration in seconds */
    detachDuration: number;
    /** Arc height during ride */
    arcHeight: number;
}

export class PlatformRideAnimator {
    private platform: THREE.Mesh;
    private startPosition: THREE.Vector3;
    private detachPosition: THREE.Vector3;
    private targetPosition: THREE.Vector3;

    private rideProgress: number = 0;
    private detachProgress: number = 0;
    private rideStarted: boolean = false;
    private isDetaching: boolean = false;
    private rideComplete: boolean = false;

    private originalColor: THREE.Color = new THREE.Color();
    private config: PlatformRideConfig;

    // Easing functions
    private static readonly easeInOutCubic = (t: number): number => {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    };

    constructor(platform: THREE.Mesh, config: Partial<PlatformRideConfig> = {}) {
        this.platform = platform;
        this.startPosition = platform.position.clone();

        // Detach position is slightly lower
        this.detachPosition = this.startPosition.clone();
        this.detachPosition.y -= 0.1;

        this.config = {
            targetPosition: new THREE.Vector3(0, 0.5, 1.0),
            rideDuration: 5.0,
            detachDuration: 1.0,
            arcHeight: 0.3,
            ...config
        };

        // Clone target position
        this.targetPosition = this.config.targetPosition.clone();

        // Store original color
        if (platform.material) {
            const mat = platform.material as THREE.MaterialStandardMaterial;
            if (mat.color) {
                this.originalColor.copy(mat.color);
            }
        }
    }

    /**
     * Start the platform ride sequence
     */
    public startRide(): void {
        this.rideStarted = true;
        this.isDetaching = true;
    }

    /**
     * Check if ride has started
     */
    public hasStarted(): boolean {
        return this.rideStarted;
    }

    /**
     * Check if ride is complete
     */
    public isComplete(): boolean {
        return this.rideComplete;
    }

    /**
     * Update animation - call each frame
     */
    public update(deltaTime: number): void {
        if (!this.rideStarted || this.rideComplete) return;

        // Phase 1: Detachment
        if (this.isDetaching) {
            this.detachProgress += deltaTime / this.config.detachDuration;

            if (this.detachProgress >= 1) {
                this.detachProgress = 1;
                this.isDetaching = false;
            }

            // Eased detachment
            const eased = PlatformRideAnimator.easeInOutCubic(this.detachProgress);
            this.platform.position.lerpVectors(
                this.startPosition,
                this.detachPosition,
                eased
            );

            // Color shift: gray → warm amber
            this.updatePlatformColor(eased, 0xffaa44);

            return; // Wait for detachment to complete
        }

        // Phase 2: Ride to vortex
        this.rideProgress += deltaTime / this.config.rideDuration;

        if (this.rideProgress >= 1) {
            this.rideProgress = 1;
            this.rideComplete = true;
        }

        // Apply easing
        const eased = PlatformRideAnimator.easeInOutCubic(this.rideProgress);

        // Calculate arc height (sine curve peaking at midpoint)
        const arcOffset = Math.sin(eased * Math.PI) * this.config.arcHeight;

        // Interpolate base position
        this.platform.position.lerpVectors(
            this.detachPosition,
            this.targetPosition,
            eased
        );

        // Add arc height
        this.platform.position.y += arcOffset;
    }

    /**
     * Update platform color during detachment
     */
    private updatePlatformColor(progress: number, targetHex: number): void {
        if (!this.platform.material) return;

        const mat = this.platform.material as THREE.MaterialStandardMaterial;
        if (!mat.color) return;

        const targetColor = new THREE.Color(targetHex);
        mat.color.lerpColors(this.originalColor, targetColor, progress);
    }

    /**
     * Get current ride progress (0-1)
     */
    public getRideProgress(): number {
        if (this.isDetaching) {
            return this.detachProgress * 0.2; // First 20% is detachment
        }
        return 0.2 + (this.rideProgress * 0.8); // Remaining 80% is ride
    }

    /**
     * Get current position
     */
    public getCurrentPosition(): THREE.Vector3 {
        return this.platform.position.clone();
    }

    /**
     * Get target position
     */
    public getTargetPosition(): THREE.Vector3 {
        return this.targetPosition.clone();
    }

    /**
     * Reset animation
     */
    public reset(): void {
        this.rideProgress = 0;
        this.detachProgress = 0;
        this.rideStarted = false;
        this.isDetaching = false;
        this.rideComplete = false;

        // Reset position
        this.platform.position.copy(this.startPosition);

        // Reset color
        if (this.platform.material) {
            const mat = this.platform.material as THREE.MaterialStandardMaterial;
            if (mat.color) {
                mat.color.copy(this.originalColor);
            }
        }
    }

    /**
     * Set target position
     */
    public setTargetPosition(position: THREE.Vector3): void {
        this.targetPosition.copy(position);
    }

    /**
     * Get platform mesh
     */
    public getPlatform(): THREE.Mesh {
        return this.platform;
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        // Platform is external reference, don't dispose
    }
}
