import * as THREE from 'three';
import { summonRingVertexShader, summonRingFragmentShader } from '../shaders';
import { SummonRingMaterialTSL } from '../shaders/tsl';

/**
 * Detect if WebGPU/TSL is available
 */
function isWebGPURenderer(): boolean {
    return (window as any).rendererType === 'WebGPU';
}

/**
 * SummonRing - Visual effect that appears on water before jelly emerges
 *
 * Creates an expanding ring of light that draws attention to spawn location.
 * Helps players notice where jellies will appear.
 *
 * Auto-selects TSL (WebGPU) or GLSL (WebGL2) implementation.
 */
export class SummonRing extends THREE.Mesh {
    private material: THREE.ShaderMaterial | SummonRingMaterialTSL;
    private isTSL: boolean;
    private animTime: number = 0;
    private duration: number = 0.6;
    private state: 'playing' | 'complete' = 'playing';
    private onCompleteCallback?: () => void;

    // Uniforms object for API compatibility (GLSL mode only)
    public uniforms: {
        uTime: { value: number };
        uDuration: { value: number };
        uColor: { value: THREE.Color };
    };

    constructor() {
        // Detect renderer type BEFORE accessing 'this'
        const isTSL = isWebGPURenderer();
        const duration = 0.6;

        // Create geometry
        const geometry = new THREE.RingGeometry(0.1, 0.3, 64);

        // Initialize uniforms for GLSL mode
        const uniforms = {
            uTime: { value: 0 },
            uDuration: { value: duration },
            uColor: { value: new THREE.Color(0x00ffff) }
        };

        let material: THREE.ShaderMaterial | SummonRingMaterialTSL;

        if (isTSL) {
            // Use TSL material for WebGPU
            material = new SummonRingMaterialTSL();
            console.log('[SummonRing] Using TSL (WebGPU) implementation');
        } else {
            // Use GLSL shader material for WebGL2 fallback
            material = new THREE.ShaderMaterial({
                vertexShader: summonRingVertexShader,
                fragmentShader: summonRingFragmentShader,
                uniforms: uniforms,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: false,
                blending: THREE.AdditiveBlending
            });
            console.log('[SummonRing] Using GLSL (WebGL2) implementation');
        }

        // MUST call super() before accessing 'this'
        super(geometry, material);

        this.material = material;
        this.isTSL = isTSL;
        this.uniforms = uniforms;
        this.duration = duration;

        // Lay flat on water surface
        this.rotation.x = -Math.PI / 2;
        this.position.y = 0.05;
    }

    /**
     * Start the summon ring animation
     */
    public play(onComplete?: () => void): void {
        this.state = 'playing';
        this.animTime = 0;
        this.onCompleteCallback = onComplete;
        this.visible = true;
        this.setTime(0);
    }

    /**
     * Update the animation
     */
    public update(deltaTime: number): void {
        if (this.state !== 'playing') return;

        this.animTime += deltaTime;
        this.setTime(this.animTime);

        // Scale up as ring expands
        const scale = 1 + this.animTime / this.duration * 2;
        this.scale.set(scale, scale, 1);

        // Check completion
        if (this.animTime >= this.duration) {
            this.state = 'complete';
            this.visible = false;
            if (this.onCompleteCallback) {
                this.onCompleteCallback();
                this.onCompleteCallback = undefined;
            }
        }
    }

    /**
     * Check if animation is complete
     */
    public isComplete(): boolean {
        return this.state === 'complete';
    }

    /**
     * Reset for reuse
     */
    public reset(): void {
        this.state = 'playing';
        this.animTime = 0;
        this.visible = false;
        this.scale.set(1, 1, 1);
        this.setTime(0);
    }

    /**
     * Set the ring color
     */
    public setColor(color: THREE.Color): void {
        if (this.isTSL) {
            (this.material as SummonRingMaterialTSL).setColor(color);
        } else {
            this.uniforms.uColor.value.copy(color);
        }
    }

    /**
     * Set animation duration
     */
    public setDuration(seconds: number): void {
        this.duration = seconds;
        if (this.isTSL) {
            (this.material as SummonRingMaterialTSL).setDuration(seconds);
        } else {
            this.uniforms.uDuration.value = seconds;
        }
    }

    /**
     * Set time (internal method)
     */
    private setTime(time: number): void {
        if (this.isTSL) {
            (this.material as SummonRingMaterialTSL).setTime(time);
        } else {
            this.uniforms.uTime.value = time;
        }
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}
