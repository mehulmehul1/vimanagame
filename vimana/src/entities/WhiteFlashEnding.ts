/**
 * WhiteFlashEnding - Transcendent white flash transition effect
 *
 * Creates a luminous white flash that engulfs the camera
 * when the shell is collected, symbolizing transcendence.
 */

import * as THREE from 'three';
import { whiteFlashVertexShader, whiteFlashFragmentShader } from '../shaders';

export type FlashState = 'idle' | 'triggering' | 'ascending' | 'fading' | 'complete';

export interface FlashConfig {
    /** Duration of initial white flash in seconds */
    flashDuration: number;
    /** Duration of ascent phase in seconds */
    ascentDuration: number;
    /** Duration of fade out in seconds */
    fadeDuration: number;
    /** Maximum brightness (0-1, typically >1 for HDR effect) */
    maxBrightness: number;
}

export class WhiteFlashEnding extends THREE.Mesh {
    private flashMaterial: THREE.ShaderMaterial;
    private state: FlashState;
    private progress: number;
    private uniforms: {
        uTime: { value: number };
        uProgress: { value: number };
        uIntensity: { value: number };
        uColor: { value: THREE.Color };
    };

    private config: FlashConfig;
    private startTime: number;
    private onCompleteCallback?: () => void;

    constructor(config: Partial<FlashConfig> = {}) {
        // Create local variables FIRST (before accessing 'this')
        const uniforms = {
            uTime: { value: 0 },
            uProgress: { value: 0 },
            uIntensity: { value: 0 },
            uColor: { value: new THREE.Color(1, 1, 1) }
        };

        const material = new THREE.ShaderMaterial({
            vertexShader: whiteFlashVertexShader,
            fragmentShader: whiteFlashFragmentShader,
            uniforms: uniforms,
            transparent: true,
            depthWrite: false,
            depthTest: false
        });

        // Full-screen quad
        const geometry = new THREE.PlaneGeometry(20, 20);

        // Call super FIRST before accessing 'this'
        super(geometry, material);

        // NOW we can assign to 'this'
        this.config = {
            flashDuration: 2.0,
            ascentDuration: 4.0,
            fadeDuration: 3.0,
            maxBrightness: 2.5,
            ...config
        };

        this.uniforms = uniforms;
        this.flashMaterial = material;
        this.state = 'idle';
        this.progress = 0;
        this.startTime = 0;

        // Position in front of camera
        this.renderOrder = 999;

        // Hide by default - only show when triggered
        this.visible = false;
    }

    /**
     * Trigger the white flash sequence
     */
    public trigger(onComplete?: () => void): void {
        if (this.state !== 'idle') return;

        this.state = 'triggering';
        this.progress = 0;
        this.startTime = performance.now();
        this.onCompleteCallback = onComplete;
        this.visible = true;
    }

    /**
     * Update the flash animation
     */
    public update(deltaTime: number, time: number): void {
        if (this.state === 'idle' || this.state === 'complete') {
            return;
        }

        this.uniforms.uTime.value = time;
        this.progress += deltaTime;

        const elapsed = (performance.now() - this.startTime) / 1000;

        switch (this.state) {
            case 'triggering':
                this.updateTriggering(elapsed);
                break;
            case 'ascending':
                this.updateAscending(elapsed);
                break;
            case 'fading':
                this.updateFading(elapsed);
                break;
        }
    }

    /**
     * Update triggering phase - rapid white flash
     */
    private updateTriggering(elapsed: number): void {
        const t = Math.min(elapsed / this.config.flashDuration, 1);

        // Exponential ease-in for dramatic effect
        const eased = t * t;

        this.uniforms.uIntensity.value = eased * this.config.maxBrightness;
        this.uniforms.uProgress.value = eased * 0.3;

        // Pure white with subtle warmth
        this.uniforms.uColor.value.setRGB(
            1.0,
            0.98 + eased * 0.02,
            0.95 + eased * 0.05
        );

        if (t >= 1) {
            this.state = 'ascending';
        }
    }

    /**
     * Update ascending phase - transcendence ascent
     */
    private updateAscending(elapsed: number): void {
        const phaseTime = elapsed - this.config.flashDuration;
        const t = Math.min(phaseTime / this.config.ascentDuration, 1);

        // Smooth step for serene ascent
        const eased = t * t * (3 - 2 * t);

        this.uniforms.uProgress.value = 0.3 + eased * 0.5;

        // Color shifts toward golden during ascent
        const hue = 0.08; // Golden hue
        const sat = eased * 0.3;
        this.uniforms.uColor.value.setHSL(hue, sat, 1.0);

        // Subtle pulsing
        const pulse = Math.sin(elapsed * 3) * 0.1 + 0.9;
        this.uniforms.uIntensity.value = this.config.maxBrightness * pulse;

        if (t >= 1) {
            this.state = 'fading';
        }
    }

    /**
     * Update fading phase - gentle fade to scene
     */
    private updateFading(elapsed: number): void {
        const phaseTime = elapsed - this.config.flashDuration - this.config.ascentDuration;
        const t = Math.min(phaseTime / this.config.fadeDuration, 1);

        // Exponential ease-out
        const eased = 1 - Math.pow(1 - t, 3);

        this.uniforms.uIntensity.value = this.config.maxBrightness * (1 - eased);
        this.uniforms.uProgress.value = 0.8 - eased * 0.8;

        // Return to pure white before fading
        this.uniforms.uColor.value.setRGB(1, 1, 1);

        if (t >= 1) {
            this.state = 'complete';
            this.visible = false;

            if (this.onCompleteCallback) {
                this.onCompleteCallback();
            }

            // Dispatch completion event
            window.dispatchEvent(new CustomEvent('white-flash-complete'));
        }
    }

    /**
     * Get current state
     */
    public getState(): FlashState {
        return this.state;
    }

    /**
     * Check if flash is active
     */
    public isActive(): boolean {
        return this.state !== 'idle' && this.state !== 'complete';
    }

    /**
     * Check if flash is complete
     */
    public isComplete(): boolean {
        return this.state === 'complete';
    }

    /**
     * Reset to idle state
     */
    public reset(): void {
        this.state = 'idle';
        this.progress = 0;
        this.uniforms.uIntensity.value = 0;
        this.uniforms.uProgress.value = 0;
        this.visible = false;
    }

    /**
     * Get current intensity (for camera exposure adjustment)
     */
    public getCurrentIntensity(): number {
        return this.uniforms.uIntensity.value;
    }

    /**
     * Set in front of camera
     */
    public positionInFrontOfCamera(camera: THREE.Camera): void {
        const distance = 5;
        const direction = new THREE.Vector3();
        camera.getWorldDirection(direction);
        this.position.copy(camera.position).add(direction.multiplyScalar(distance));
        this.lookAt(camera.position);
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        const scene = this.parent;
        if (scene) {
            scene.remove(this);
        }
        this.geometry.dispose();
        this.flashMaterial.dispose();
    }
}
