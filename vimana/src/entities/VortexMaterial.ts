import * as THREE from 'three';
import { vortexVertexShader, vortexFragmentShader } from '../shaders';
import { VortexMaterialTSL } from '../shaders/tsl';

/**
 * Detect if WebGPU/TSL is available
 *
 * Checks for global renderer type flag set by createWebGPURenderer
 */
function isWebGPURenderer(): boolean {
    return (window as any).rendererType === 'WebGPU';
}

/**
 * VortexMaterial - SDF torus shader with activation-based animation
 *
 * Creates the glowing vortex portal that intensifies based on duet progress.
 * Features spin animation, breathing displacement, and edge glow.
 *
 * Automatically selects TSL (WebGPU) or GLSL (WebGL2) implementation
 * based on the current renderer type.
 *
 * Story: 4.3 - Vortex Shader TSL Migration
 */
export class VortexMaterial {
    private material: THREE.ShaderMaterial | InstanceType<typeof VortexMaterialTSL>;
    private isTSL: boolean;

    // Color constants
    public static readonly INNER_COLOR = new THREE.Color(0x00ffff);  // Cyan
    public static readonly OUTER_COLOR = new THREE.Color(0x8800ff);  // Purple
    public static readonly CORE_COLOR = new THREE.Color(0xffffff);   // White

    // Uniforms object for API compatibility (GLSL mode only)
    public uniforms: {
        uTime: { value: number };
        uVortexActivation: { value: number };
        uDuetProgress: { value: number };
        uInnerColor: { value: THREE.Color };
        uOuterColor: { value: THREE.Color };
        uCoreColor: { value: THREE.Color };
        uCameraPosition: { value: THREE.Vector3 };
    };

    constructor() {
        this.isTSL = isWebGPURenderer();

        if (this.isTSL) {
            // Use TSL material for WebGPU
            this.material = new VortexMaterialTSL();
            console.log('[VortexMaterial] Using TSL (WebGPU) implementation');
        } else {
            // Use GLSL shader material for WebGL2 fallback
            const uniforms = {
                uTime: { value: 0 },
                uVortexActivation: { value: 0 },
                uDuetProgress: { value: 0 },
                uInnerColor: { value: VortexMaterial.INNER_COLOR.clone() },
                uOuterColor: { value: VortexMaterial.OUTER_COLOR.clone() },
                uCoreColor: { value: VortexMaterial.CORE_COLOR.clone() },
                uCameraPosition: { value: new THREE.Vector3() }
            };

            this.material = new THREE.ShaderMaterial({
                vertexShader: vortexVertexShader,
                fragmentShader: vortexFragmentShader,
                uniforms: uniforms,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: false
            });

            this.uniforms = uniforms;
            console.log('[VortexMaterial] Using GLSL (WebGL2) implementation');
        }
    }

    // ========================================================================
    // THREE.JS MATERIAL FORWARDING (makes VortexMaterial behave like Material)
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
            (this.material as VortexMaterialTSL).setTime(time);
        } else {
            this.uniforms.uTime.value = time;
        }
    }

    /**
     * Update vortex activation level (0-1)
     * Controls intensity, spin speed, and color transition
     */
    public setActivation(activation: number): void {
        const clamped = Math.max(0, Math.min(1, activation));
        if (this.isTSL) {
            (this.material as VortexMaterialTSL).setActivation(clamped);
        } else {
            this.uniforms.uVortexActivation.value = clamped;
        }
    }

    /**
     * Update duet progress (0-1)
     */
    public setDuetProgress(progress: number): void {
        const clamped = Math.max(0, Math.min(1, progress));
        if (this.isTSL) {
            (this.material as VortexMaterialTSL).setDuetProgress(clamped);
        } else {
            this.uniforms.uDuetProgress.value = clamped;
        }
    }

    /**
     * Update camera position for fresnel calculations
     */
    public setCameraPosition(position: THREE.Vector3): void {
        if (this.isTSL) {
            (this.material as VortexMaterialTSL).setCameraPosition(position);
        } else {
            this.uniforms.uCameraPosition.value.copy(position);
        }
    }

    /**
     * Get current activation level
     */
    public getActivation(): number {
        if (this.isTSL) {
            return (this.material as VortexMaterialTSL).getActivation();
        } else {
            return this.uniforms.uVortexActivation.value;
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
export { VortexMaterialTSL } from '../shaders/tsl';
