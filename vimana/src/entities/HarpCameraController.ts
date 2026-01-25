import * as THREE from 'three';

/**
 * Harp interaction states
 */
export enum HarpInteractionState {
    FREE_ROAM = 'FREE_ROAM',       // Normal exploration
    APPROACHING = 'APPROACHING',   // Walking toward harp
    LOCKING = 'LOCKING',           // Camera transitioning to locked view
    LOCKED = 'LOCKED',             // Camera locked, ready to play
    PLAYING = 'PLAYING',           // Actively playing strings
    UNLOCKING = 'UNLOCKING'        // Returning to free roam
}

/**
 * Camera lock-on configuration
 */
export interface HarpCameraConfig {
    lockDuration: number;          // Time to transition to locked view (seconds)
    unlockDuration: number;        // Time to return to free roam (seconds)
    lockedFieldOfView: number;     // FOV when locked
    freeFieldOfView: number;       // FOV when free roaming
    interactionDistance: number;   // Max distance to trigger harp interaction
}

/**
 * HarpCameraController - Manages camera lock-on for harp interaction
 *
 * When player approaches the harp and presses E, the camera smoothly
 * transitions to a focused view of the harp strings for gameplay.
 */
export class HarpCameraController {
    private state: HarpInteractionState = HarpInteractionState.FREE_ROAM;
    private transitionTime: number = 0;

    // Camera positions
    private freeRoamPosition: THREE.Vector3;
    private freeRoamLookAt: THREE.Vector3;
    private lockedPosition: THREE.Vector3 = new THREE.Vector3(0, 1.8, 3.5);
    private lockedLookAt: THREE.Vector3 = new THREE.Vector3(0, 1.2, 0);

    // Current interpolated values
    private currentLookAt: THREE.Vector3 = new THREE.Vector3();

    // Camera reference
    private camera: THREE.PerspectiveCamera;
    private originalFOV: number;

    // Harp position (for lock-on target)
    private harpPosition: THREE.Vector3 = new THREE.Vector3(0, 0, 0);

    // Player control state
    private playerControlsEnabled: boolean = true;

    // Configuration
    private config: HarpCameraConfig = {
        lockDuration: 1.0,
        unlockDuration: 0.8,
        lockedFieldOfView: 50,
        freeFieldOfView: 75,
        interactionDistance: 4
    };

    // Callbacks
    private onStateChange?: (state: HarpInteractionState) => void;
    private onInteractionStart?: () => void;
    private onInteractionEnd?: () => void;

    constructor(camera: THREE.PerspectiveCamera) {
        this.camera = camera;
        this.originalFOV = camera.fov;

        // Store initial camera state as free roam
        this.freeRoamPosition = camera.position.clone();
        this.freeRoamLookAt = new THREE.Vector3(0, 1, -1); // Default look direction

        this.currentLookAt.copy(this.freeRoamLookAt);
    }

    /**
     * Set the harp position for lock-on calculations
     */
    public setHarpPosition(position: THREE.Vector3): void {
        this.harpPosition.copy(position);

        // Update locked position to be relative to harp
        this.lockedPosition.set(
            position.x,
            position.y + 1.8,
            position.z + 3.5
        );
        this.lockedLookAt.set(
            position.x,
            position.y + 1.2,
            position.z
        );
    }

    /**
     * Update stored free roam position (call each frame when free roaming)
     */
    public updateFreeRoamPosition(position: THREE.Vector3, lookAt: THREE.Vector3): void {
        if (this.state === HarpInteractionState.FREE_ROAM) {
            this.freeRoamPosition.copy(position);
            this.freeRoamLookAt.copy(lookAt);
            this.currentLookAt.copy(lookAt);
        }
    }

    /**
     * Check if player is close enough to interact with harp
     */
    public canInteract(playerPosition: THREE.Vector3): boolean {
        if (this.state !== HarpInteractionState.FREE_ROAM) return false;
        const distance = playerPosition.distanceTo(this.harpPosition);
        return distance <= this.config.interactionDistance;
    }

    /**
     * Start the lock-on sequence
     */
    public engageLockOn(): void {
        if (this.state !== HarpInteractionState.FREE_ROAM) return;

        this.state = HarpInteractionState.LOCKING;
        this.transitionTime = 0;
        this.playerControlsEnabled = false;

        this.notifyStateChange();
        if (this.onInteractionStart) {
            this.onInteractionStart();
        }
    }

    /**
     * Disengage lock-on and return to free roam
     */
    public disengageLockOn(): void {
        if (this.state !== HarpInteractionState.LOCKED &&
            this.state !== HarpInteractionState.PLAYING) return;

        this.state = HarpInteractionState.UNLOCKING;
        this.transitionTime = 0;

        this.notifyStateChange();
    }

    /**
     * Update camera controller
     */
    public update(deltaTime: number): void {
        switch (this.state) {
            case HarpInteractionState.LOCKING:
                this.updateLocking(deltaTime);
                break;

            case HarpInteractionState.LOCKED:
            case HarpInteractionState.PLAYING:
                // Maintain locked position
                this.camera.position.copy(this.lockedPosition);
                this.currentLookAt.copy(this.lockedLookAt);
                this.camera.lookAt(this.lockedLookAt);
                break;

            case HarpInteractionState.UNLOCKING:
                this.updateUnlocking(deltaTime);
                break;
        }
    }

    /**
     * Update locking transition
     */
    private updateLocking(deltaTime: number): void {
        this.transitionTime += deltaTime;
        const t = Math.min(this.transitionTime / this.config.lockDuration, 1.0);

        // Smooth easing
        const easedT = this.smoothStep(t);

        // Interpolate position
        this.camera.position.lerpVectors(
            this.freeRoamPosition,
            this.lockedPosition,
            easedT
        );

        // Interpolate look-at
        this.currentLookAt.lerpVectors(
            this.freeRoamLookAt,
            this.lockedLookAt,
            easedT
        );
        this.camera.lookAt(this.currentLookAt);

        // Interpolate FOV
        this.camera.fov = THREE.MathUtils.lerp(
            this.config.freeFieldOfView,
            this.config.lockedFieldOfView,
            easedT
        );
        this.camera.updateProjectionMatrix();

        // Check completion
        if (t >= 1.0) {
            this.state = HarpInteractionState.LOCKED;
            this.playerControlsEnabled = false;
            this.notifyStateChange();
        }
    }

    /**
     * Update unlocking transition
     */
    private updateUnlocking(deltaTime: number): void {
        this.transitionTime += deltaTime;
        const t = Math.min(this.transitionTime / this.config.unlockDuration, 1.0);

        // Smooth easing
        const easedT = this.smoothStep(t);

        // Interpolate position back to free roam
        this.camera.position.lerpVectors(
            this.lockedPosition,
            this.freeRoamPosition,
            easedT
        );

        // Interpolate look-at
        this.currentLookAt.lerpVectors(
            this.lockedLookAt,
            this.freeRoamLookAt,
            easedT
        );
        this.camera.lookAt(this.currentLookAt);

        // Interpolate FOV
        this.camera.fov = THREE.MathUtils.lerp(
            this.config.lockedFieldOfView,
            this.config.freeFieldOfView,
            easedT
        );
        this.camera.updateProjectionMatrix();

        // Check completion
        if (t >= 1.0) {
            this.state = HarpInteractionState.FREE_ROAM;
            this.playerControlsEnabled = true;
            this.notifyStateChange();
            if (this.onInteractionEnd) {
                this.onInteractionEnd();
            }
        }
    }

    /**
     * Smooth step easing function
     */
    private smoothStep(t: number): number {
        return t * t * (3 - 2 * t);
    }

    /**
     * Notify state change listeners
     */
    private notifyStateChange(): void {
        if (this.onStateChange) {
            this.onStateChange(this.state);
        }
    }

    /**
     * Get current state
     */
    public getState(): HarpInteractionState {
        return this.state;
    }

    /**
     * Check if player controls are enabled
     */
    public areControlsEnabled(): boolean {
        return this.playerControlsEnabled;
    }

    /**
     * Check if camera is locked onto harp
     */
    public isLocked(): boolean {
        return this.state === HarpInteractionState.LOCKED ||
               this.state === HarpInteractionState.PLAYING;
    }

    /**
     * Enter playing state (when player starts interacting with strings)
     */
    public enterPlayingState(): void {
        if (this.state === HarpInteractionState.LOCKED) {
            this.state = HarpInteractionState.PLAYING;
            this.notifyStateChange();
        }
    }

    /**
     * Set state change callback
     */
    public setOnStateChange(callback: (state: HarpInteractionState) => void): void {
        this.onStateChange = callback;
    }

    /**
     * Set interaction start callback
     */
    public setOnInteractionStart(callback: () => void): void {
        this.onInteractionStart = callback;
    }

    /**
     * Set interaction end callback
     */
    public setOnInteractionEnd(callback: () => void): void {
        this.onInteractionEnd = callback;
    }

    /**
     * Update configuration
     */
    public setConfig(config: Partial<HarpCameraConfig>): void {
        this.config = { ...this.config, ...config };
    }

    /**
     * Get configuration
     */
    public getConfig(): HarpCameraConfig {
        return { ...this.config };
    }

    /**
     * Force reset to free roam (for cleanup)
     */
    public reset(): void {
        this.state = HarpInteractionState.FREE_ROAM;
        this.transitionTime = 0;
        this.playerControlsEnabled = true;
        this.camera.fov = this.originalFOV;
        this.camera.updateProjectionMatrix();
        this.notifyStateChange();
    }
}
