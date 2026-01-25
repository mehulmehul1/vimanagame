import * as THREE from 'three';
import { vortexVertexShader, vortexFragmentShader } from '../shaders';

/**
 * VortexMaterial - SDF torus shader with activation-based animation
 *
 * Creates the glowing vortex portal that intensifies based on duet progress.
 * Features spin animation, breathing displacement, and edge glow.
 */
export class VortexMaterial extends THREE.ShaderMaterial {
    private static readonly INNER_COLOR = new THREE.Color(0x00ffff);  // Cyan
    private static readonly OUTER_COLOR = new THREE.Color(0x8800ff);  // Purple
    private static readonly CORE_COLOR = new THREE.Color(0xffffff);   // White

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
        const uniforms = {
            uTime: { value: 0 },
            uVortexActivation: { value: 0 },
            uDuetProgress: { value: 0 },
            uInnerColor: { value: VortexMaterial.INNER_COLOR.clone() },
            uOuterColor: { value: VortexMaterial.OUTER_COLOR.clone() },
            uCoreColor: { value: VortexMaterial.CORE_COLOR.clone() },
            uCameraPosition: { value: new THREE.Vector3() }
        };

        super({
            vertexShader: vortexVertexShader,
            fragmentShader: vortexFragmentShader,
            uniforms: uniforms,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });

        this.uniforms = uniforms;
    }

    /**
     * Update time for animation
     */
    public setTime(time: number): void {
        this.uniforms.uTime.value = time;
    }

    /**
     * Update vortex activation level (0-1)
     * Controls intensity, spin speed, and color transition
     */
    public setActivation(activation: number): void {
        this.uniforms.uVortexActivation.value = Math.max(0, Math.min(1, activation));
    }

    /**
     * Update duet progress (0-1)
     */
    public setDuetProgress(progress: number): void {
        this.uniforms.uDuetProgress.value = Math.max(0, Math.min(1, progress));
    }

    /**
     * Update camera position for fresnel calculations
     */
    public setCameraPosition(position: THREE.Vector3): void {
        this.uniforms.uCameraPosition.value.copy(position);
    }

    /**
     * Get current activation level
     */
    public getActivation(): number {
        return this.uniforms.uVortexActivation.value;
    }

    /**
     * Cleanup method for memory management
     */
    public destroy(): void {
        this.dispose();
    }
}
