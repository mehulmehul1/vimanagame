/**
 * WhiteFlashManager - Orchestrates the transcendence ending sequence
 *
 * Coordinates white flash, shell collection, camera ascent,
 * audio crescendo, and final fade.
 */

import * as THREE from 'three';
import { WhiteFlashEnding, FlashState } from './WhiteFlashEnding';
import { WhiteFlashAudio } from '../audio/WhiteFlashAudio';
import { ShellManager } from './ShellManager';

export interface EndingCallbacks {
    /** Called when ending sequence starts */
    onStart?: () => void;
    /** Called when ascent phase begins */
    onAscent?: () => void;
    /** Called when ending sequence completes */
    onComplete?: () => void;
}

export interface EndingConfig {
    /** Camera ascent height in world units */
    ascentHeight: number;
    /** Camera ascent duration in seconds */
    ascentDuration: number;
    /** Auto-trigger on shell collect */
    autoTrigger: boolean;
}

export class WhiteFlashManager {
    private whiteFlash: WhiteFlashEnding;
    private audio: WhiteFlashAudio;
    private shellManager: ShellManager;
    private camera: THREE.Camera;
    private scene: THREE.Scene;

    private config: EndingConfig;
    private callbacks: EndingCallbacks;

    private active: boolean = false;
    private ascentProgress: number = 0;
    private cameraStartY: number = 0;

    constructor(
        scene: THREE.Scene,
        camera: THREE.Camera,
        callbacks: EndingCallbacks = {},
        config: Partial<EndingConfig> = {}
    ) {
        this.scene = scene;
        this.camera = camera;
        this.callbacks = callbacks;

        this.config = {
            ascentHeight: 10,
            ascentDuration: 4.0,
            autoTrigger: true,
            ...config
        };

        this.whiteFlash = new WhiteFlashEnding();
        this.whiteFlash.visible = false;
        scene.add(this.whiteFlash);

        this.audio = new WhiteFlashAudio();
        this.shellManager = ShellManager.getInstance();

        this.setupEventListeners();
    }

    /**
     * Setup event listeners
     */
    private setupEventListeners(): void {
        // Auto-trigger on final shell collection
        if (this.config.autoTrigger) {
            window.addEventListener('shell-collected', this.handleShellCollected);
        }
    }

    /**
     * Handle shell collected event
     */
    private handleShellCollected = (): void => {
        // Check if this was the final shell (4/4)
        if (this.shellManager.isComplete() && !this.active) {
            // Small delay before triggering ending
            setTimeout(() => this.start(), 1500);
        }
    };

    /**
     * Start the ending sequence
     */
    public start(): void {
        if (this.active) return;

        this.active = true;
        this.ascentProgress = 0;
        this.cameraStartY = this.camera.position.y;

        // Position flash in front of camera
        this.whiteFlash.positionInFrontOfCamera(this.camera);

        // Start the sequence
        this.whiteFlash.trigger(() => {
            this.onFlashComplete();
        });

        // Start audio
        this.audio.startCrescendo();

        // Callback
        if (this.callbacks.onStart) {
            this.callbacks.onStart();
        }
    }

    /**
     * Update the ending sequence
     */
    public update(deltaTime: number, time: number): void {
        if (!this.active) return;

        // Update white flash
        this.whiteFlash.update(deltaTime, time);

        // Update audio
        this.audio.update(deltaTime);

        // Handle camera ascent during active flash
        if (this.whiteFlash.isActive()) {
            this.updateCameraAscent(deltaTime);
        }

        // Sync flash position with camera
        if (this.whiteFlash.visible) {
            this.whiteFlash.positionInFrontOfCamera(this.camera);
        }
    }

    /**
     * Update camera ascent animation
     */
    private updateCameraAscent(deltaTime: number): void {
        const state = this.whiteFlash.getState();

        if (state === 'ascending') {
            const ascentSpeed = this.config.ascentHeight / this.config.ascentDuration;
            this.ascentProgress += deltaTime * ascentSpeed;

            // Smooth step easing
            const t = Math.min(this.ascentProgress / this.config.ascentHeight, 1);
            const eased = t * t * (3 - 2 * t);

            this.camera.position.y = this.cameraStartY + eased * this.config.ascentHeight;

            // Trigger ascent callback on first frame
            if (t > 0 && this.ascentProgress - deltaTime * ascentSpeed <= 0) {
                if (this.callbacks.onAscent) {
                    this.callbacks.onAscent();
                }
            }
        }
    }

    /**
     * Handle flash complete
     */
    private onFlashComplete(): void {
        this.active = false;

        // Callback
        if (this.callbacks.onComplete) {
            this.callbacks.onComplete();
        }
    }

    /**
     * Check if ending is active
     */
    public isActive(): boolean {
        return this.active;
    }

    /**
     * Check if ending has completed
     */
    public isComplete(): boolean {
        return this.whiteFlash.isComplete();
    }

    /**
     * Get current flash state
     */
    public getFlashState(): FlashState {
        return this.whiteFlash.getState();
    }

    /**
     * Get current camera exposure multiplier (for renderer)
     */
    public getExposureMultiplier(): number {
        if (!this.active) return 1.0;
        return 1.0 + this.whiteFlash.getCurrentIntensity() * 0.5;
    }

    /**
     * Manually trigger ending (for testing)
     */
    public trigger(): void {
        this.start();
    }

    /**
     * Update configuration
     */
    public updateConfig(updates: Partial<EndingConfig>): void {
        this.config = { ...this.config, ...updates };
    }

    /**
     * Update callbacks
     */
    public setCallbacks(callbacks: Partial<EndingCallbacks>): void {
        this.callbacks = { ...this.callbacks, ...callbacks };
    }

    /**
     * Reset ending state
     */
    public reset(): void {
        this.active = false;
        this.ascentProgress = 0;
        this.whiteFlash.reset();
        this.audio.reset();
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        window.removeEventListener('shell-collected', this.handleShellCollected);

        this.whiteFlash.destroy();
        this.audio.destroy();
    }
}
