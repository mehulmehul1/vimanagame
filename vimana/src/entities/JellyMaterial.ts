import * as THREE from 'three';
import { jellyVertexShader, jellyFragmentShader } from '../shaders';
import { JellyMaterialTSL } from '../shaders/tsl';

/**
 * Detect if WebGPU/TSL is available
 *
 * Checks for global renderer type flag set by createWebGPURenderer
 */
function isWebGPURenderer(): boolean {
    return (window as any).rendererType === 'WebGPU';
}

/**
 * JellyMaterial - Bioluminescent jelly creature shader
 *
 * Creates the organic pulsing jelly creatures that teach the player the song.
 * Features organic pulsing animation, teaching state enhancement, fresnel-based
 * rim lighting, and simplex noise displacement for organic movement.
 *
 * Automatically selects TSL (WebGPU) or GLSL (WebGL2) implementation
 * based on the current renderer type.
 *
 * Story: 4.6 - Jelly Shader TSL Migration
 */
export class JellyMaterial {
    private material: THREE.ShaderMaterial | InstanceType<typeof JellyMaterialTSL>;
    private isTSL: boolean;

    // Color constants
    public static readonly BIOLUMINESCENT_COLOR = new THREE.Color(0x88ccff);  // Soft cyan-blue
    public static readonly BASE_JELLY_COLOR = new THREE.Color(0.4, 0.8, 0.7);
    public static readonly INTERNAL_COLOR = new THREE.Color(0.8, 0.9, 0.85);

    // Uniforms object for API compatibility (GLSL mode only)
    public uniforms: {
        uTime: { value: number };
        uPulseRate: { value: number };
        uIsTeaching: { value: number };
        uTeachingIntensity: { value: number };
        uBioluminescentColor: { value: THREE.Color };
        uCameraPosition: { value: THREE.Vector3 };
    };

    constructor() {
        this.isTSL = isWebGPURenderer();

        if (this.isTSL) {
            // Use TSL material for WebGPU
            this.material = new JellyMaterialTSL();
            console.log('[JellyMaterial] Using TSL (WebGPU) implementation');
        } else {
            // Use GLSL shader material for WebGL2 fallback
            const uniforms = {
                uTime: { value: 0 },
                uPulseRate: { value: 2.0 },
                uIsTeaching: { value: 0.0 },
                uTeachingIntensity: { value: 0.0 },
                uBioluminescentColor: { value: JellyMaterial.BIOLUMINESCENT_COLOR.clone() },
                uCameraPosition: { value: new THREE.Vector3() }
            };

            this.material = new THREE.ShaderMaterial({
                vertexShader: jellyVertexShader,
                fragmentShader: jellyFragmentShader,
                uniforms: uniforms,
                transparent: true,
                side: THREE.DoubleSide
            });

            this.uniforms = uniforms;
            console.log('[JellyMaterial] Using GLSL (WebGL2) implementation');
        }
    }

    // ========================================================================
    // THREE.JS MATERIAL FORWARDING (makes JellyMaterial behave like Material)
    // ========================================================================

    /** Forward to material's transparent property */
    get transparent(): boolean {
        return this.material.transparent;
    }
    set transparent(value: boolean) {
        this.material.transparent = value;
    }

    /** Forward to material's side property */
    get side(): THREE.Side {
        return this.material.side;
    }
    set side(value: THREE.Side) {
        (this.material as THREE.ShaderMaterial).side = value;
    }

    /** Forward to material's depthWrite property */
    get depthWrite(): boolean {
        return this.material.depthWrite;
    }
    set depthWrite(value: boolean) {
        this.material.depthWrite = value;
    }

    /** Forward to material's needsUpdate property */
    get needsUpdate(): boolean {
        return (this.material as THREE.ShaderMaterial).needsUpdate;
    }
    set needsUpdate(value: boolean) {
        (this.material as THREE.ShaderMaterial).needsUpdate = value;
    }

    // ========================================================================
    // PUBLIC API (same for both TSL and GLSL versions)
    // ========================================================================

    /**
     * Update time for animation
     */
    public setTime(time: number): void {
        if (this.isTSL) {
            (this.material as JellyMaterialTSL).setTime(time);
        } else {
            this.uniforms.uTime.value = time;
        }
    }

    /**
     * Update pulse rate (frequency of pulsing animation)
     */
    public setPulseRate(rate: number): void {
        if (this.isTSL) {
            (this.material as JellyMaterialTSL).setPulseRate(rate);
        } else {
            this.uniforms.uPulseRate.value = rate;
        }
    }

    /**
     * Set teaching state
     * When true, enhances glow and adds warm color
     */
    public setTeaching(isTeaching: boolean): void {
        if (this.isTSL) {
            (this.material as JellyMaterialTSL).setTeaching(isTeaching);
        } else {
            this.uniforms.uIsTeaching.value = isTeaching ? 1.0 : 0.0;
        }
    }

    /**
     * Set teaching intensity (0-1)
     * Controls how much the teaching state affects appearance
     */
    public setTeachingIntensity(intensity: number): void {
        const clamped = Math.max(0, Math.min(1, intensity));
        if (this.isTSL) {
            (this.material as JellyMaterialTSL).setTeachingIntensity(clamped);
        } else {
            this.uniforms.uTeachingIntensity.value = clamped;
        }
    }

    /**
     * Set bioluminescent color
     */
    public setColor(color: THREE.Color): void {
        if (this.isTSL) {
            (this.material as JellyMaterialTSL).setColor(color);
        } else {
            this.uniforms.uBioluminescentColor.value.copy(color);
        }
    }

    /**
     * Update camera position for fresnel calculations
     */
    public setCameraPosition(position: THREE.Vector3): void {
        if (this.isTSL) {
            (this.material as JellyMaterialTSL).setCameraPosition(position);
        } else {
            this.uniforms.uCameraPosition.value.copy(position);
        }
    }

    /**
     * Get current pulse rate
     */
    public getPulseRate(): number {
        if (this.isTSL) {
            return (this.material as JellyMaterialTSL).getPulseRate();
        } else {
            return this.uniforms.uPulseRate.value;
        }
    }

    /**
     * Get current teaching state
     */
    public getTeaching(): boolean {
        if (this.isTSL) {
            return (this.material as JellyMaterialTSL).getTeaching();
        } else {
            return this.uniforms.uIsTeaching.value > 0.5;
        }
    }

    /**
     * Get current teaching intensity
     */
    public getTeachingIntensity(): number {
        if (this.isTSL) {
            return (this.material as JellyMaterialTSL).getTeachingIntensity();
        } else {
            return this.uniforms.uTeachingIntensity.value;
        }
    }

    /**
     * Get the underlying Three.js material
     * For internal use with Three.js Mesh objects
     */
    public getMaterial(): THREE.Material {
        return this.material;
    }

    /**
     * Cleanup method for memory management
     */
    public destroy(): void {
        this.material.dispose();
    }

    /**
     * Dispose method (alias for destroy)
     */
    public dispose(): void {
        this.material.dispose();
    }
}

/**
 * Export the TSL class directly for advanced use cases
 * Allows consumers to explicitly use TSL implementation
 */
export { JellyMaterialTSL } from '../shaders/tsl';
