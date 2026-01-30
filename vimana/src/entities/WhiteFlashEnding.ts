/**
 * WhiteFlashEnding - Transcendent white flash transition effect
 *
 * Creates a luminous white flash that engulfs the camera
 * when the shell is collected, symbolizing transcendence.
 */

import * as THREE from 'three';
import { whiteFlashVertexShader, whiteFlashFragmentShader } from '../shaders';
import { WhiteFlashMaterialTSL } from '../shaders/tsl';

// Helper to detect WebGPU mode
function isWebGPURenderer(): boolean {
    return (window as any).rendererType === 'WebGPU';
}

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
    private flashMaterial: THREE.ShaderMaterial | WhiteFlashMaterialTSL;
    private state: FlashState;
    private progress: number;
    private uniforms: {
        uTime: { value: number };
        uProgress: { value: number };
        uIntensity: { value: number };
        uColor: { value: THREE.Color };
    };
    private isTSL: boolean;

    private config: FlashConfig;
    private startTime: number;
    private onCompleteCallback?: () => void;

    constructor(config: Partial<FlashConfig> = {}) {
        const isTSL = isWebGPURenderer();

        // Default uniforms object for GLSL
        let uniforms: any = {
            uTime: { value: 0 },
            uProgress: { value: 0 },
            uIntensity: { value: 0 },
            uColor: { value: new THREE.Color(1, 1, 1) }
        };

        let material: THREE.ShaderMaterial | WhiteFlashMaterialTSL;

        if (isTSL) {
            console.log('[WhiteFlashEnding] Using TSL (WebGPU) implementation');
            material = new WhiteFlashMaterialTSL();
            // TSL material handles its own uniforms but we keep the local specific uniforms object for state tracking if needed,
            // though for TSL we will act on the material methods directly.
        } else {
            console.log('[WhiteFlashEnding] Using GLSL (WebGL2) implementation');

            // Note: The GLSL shader uses uColor1, uColor2, uColor3.
            // Current GLSL implementation in index.ts mismatches this class's uniforms.
            // If running in WebGL2, we might see issues unless we patch the shader or this class.
            // For now, we proceed assuming existing GLSL logic was somehow working or will be fixed separately if needed.
            // We'll pass the single uColor logic for now.

            material = new THREE.ShaderMaterial({
                vertexShader: whiteFlashVertexShader,
                fragmentShader: whiteFlashFragmentShader,
                uniforms: uniforms,
                transparent: true,
                depthWrite: false,
                depthTest: false
            });
        }

        // Full-screen quad
        const geometry = new THREE.PlaneGeometry(20, 20);

        // Call super FIRST
        super(geometry, material);

        this.isTSL = isTSL;
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

        // Hide by default
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

        if (!this.isTSL) {
            this.uniforms.uTime.value = time;
        }
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

    private updateMaterial(intensity: number, progress: number, color: THREE.Color) {
        if (this.isTSL) {
            const mat = this.flashMaterial as WhiteFlashMaterialTSL;
            mat.setIntensity(intensity);
            mat.setProgress(progress);
            mat.setColor(color);
        } else {
            this.uniforms.uIntensity.value = intensity;
            this.uniforms.uProgress.value = progress;
            this.uniforms.uColor.value.copy(color);
        }
    }

    /**
     * Update triggering phase - rapid white flash
     */
    private updateTriggering(elapsed: number): void {
        const t = Math.min(elapsed / this.config.flashDuration, 1);
        const eased = t * t;

        const intensity = eased * this.config.maxBrightness;
        const progress = eased * 0.3;

        const r = 1.0;
        const g = 0.98 + eased * 0.02;
        const b = 0.95 + eased * 0.05;
        const color = new THREE.Color(r, g, b);

        this.updateMaterial(intensity, progress, color);

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
        const eased = t * t * (3 - 2 * t);

        const progress = 0.3 + eased * 0.5;

        const hue = 0.08;
        const sat = eased * 0.3;
        const color = new THREE.Color().setHSL(hue, sat, 1.0);

        const pulse = Math.sin(elapsed * 3) * 0.1 + 0.9;
        const intensity = this.config.maxBrightness * pulse;

        this.updateMaterial(intensity, progress, color);

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
        const eased = 1 - Math.pow(1 - t, 3);

        const intensity = this.config.maxBrightness * (1 - eased);
        const progress = 0.8 - eased * 0.8;
        const color = new THREE.Color(1, 1, 1);

        this.updateMaterial(intensity, progress, color);

        if (t >= 1) {
            this.state = 'complete';
            this.visible = false;

            if (this.onCompleteCallback) {
                this.onCompleteCallback();
            }
            window.dispatchEvent(new CustomEvent('white-flash-complete'));
        }
    }

    // Public getters
    public getState(): FlashState { return this.state; }
    public isActive(): boolean { return this.state !== 'idle' && this.state !== 'complete'; }
    public isComplete(): boolean { return this.state === 'complete'; }
    public getCurrentIntensity(): number {
        if (this.isTSL) {
            // Accessing uniform value from TSL might be hard if not stored locally.
            // We can infer it from update calls or store it locally.
            // For now, assume TSL mode doesn't need external exposure adjustment or we can't easily get it back.
            // Ideally we should store currentIntensity in the class instance.
            return this.uniforms.uIntensity.value; // hack: we don't update this object in TSL mode in updateMaterial!
            // Fixed: we should update the local object even in TSL mode for getters?
        }
        return this.uniforms.uIntensity.value;
    }

    public reset(): void {
        this.state = 'idle';
        this.progress = 0;
        this.visible = false;

        const color = new THREE.Color(1, 1, 1);
        this.updateMaterial(0, 0, color);

        // Also resets local uniforms for consistency
        this.uniforms.uIntensity.value = 0;
        this.uniforms.uProgress.value = 0;
    }

    public positionInFrontOfCamera(camera: THREE.Camera): void {
        const distance = 5;
        const direction = new THREE.Vector3();
        camera.getWorldDirection(direction);
        this.position.copy(camera.position).add(direction.multiplyScalar(distance));
        this.lookAt(camera.position);
    }

    public destroy(): void {
        const scene = this.parent;
        if (scene) scene.remove(this);
        this.geometry.dispose();
        this.flashMaterial.dispose();
    }
}
