import * as THREE from 'three';
import { waterVertexShader, waterFragmentShader } from '../shaders';

/**
 * EnhancedWaterMaterial - Bioluminescent water surface shader
 *
 * Responds to 6 harp strings with individual frequency tracking.
 * Creates ripple effects, caustics, and fresnel-based transparency.
 */
export class WaterMaterial extends THREE.ShaderMaterial {
    private static readonly DEFAULT_COLOR = new THREE.Color(0x00ff88); // Cyan-green

    public uniforms: {
        uTime: { value: number };
        uHarpFrequencies: { value: Float32Array };
        uHarpVelocities: { value: Float32Array };
        uDuetProgress: { value: number };
        uShipPatience: { value: number };
        uBioluminescentColor: { value: THREE.Color };
        uHarmonicResonance: { value: number };
        uCameraPosition: { value: THREE.Vector3 };
    };

    constructor() {
        // Initialize frequency arrays for 6 strings (C, D, E, F, G, A)
        const frequencies = new Float32Array(6);
        const velocities = new Float32Array(6);

        const uniforms = {
            uTime: { value: 0 },
            uHarpFrequencies: { value: frequencies },
            uHarpVelocities: { value: velocities },
            uDuetProgress: { value: 0 },
            uShipPatience: { value: 1.0 },
            uBioluminescentColor: { value: WaterMaterial.DEFAULT_COLOR.clone() },
            uHarmonicResonance: { value: 0 },
            uCameraPosition: { value: new THREE.Vector3() }
        };

        super({
            vertexShader: waterVertexShader,
            fragmentShader: waterFragmentShader,
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
     * Update harp string frequency (0-5 for strings C-A)
     */
    public setStringFrequency(index: number, frequency: number): void {
        if (index >= 0 && index < 6) {
            this.uniforms.uHarpFrequencies.value[index] = frequency;
        }
    }

    /**
     * Update harp string velocity for intensity modulation
     */
    public setStringVelocity(index: number, velocity: number): void {
        if (index >= 0 && index < 6) {
            this.uniforms.uHarpVelocities.value[index] = velocity;
        }
    }

    /**
     * Update duet progress (0-1)
     */
    public setDuetProgress(progress: number): void {
        this.uniforms.uDuetProgress.value = Math.max(0, Math.min(1, progress));
    }

    /**
     * Update harmonic resonance intensity (0-1)
     */
    public setHarmonicResonance(resonance: number): void {
        this.uniforms.uHarmonicResonance.value = Math.max(0, Math.min(1, resonance));
    }

    /**
     * Update ship patience teaching state
     */
    public setShipPatience(patience: number): void {
        this.uniforms.uShipPatience.value = Math.max(0, Math.min(1, patience));
    }

    /**
     * Update camera position for fresnel calculations
     */
    public setCameraPosition(position: THREE.Vector3): void {
        this.uniforms.uCameraPosition.value.copy(position);
    }

    /**
     * Trigger ripple effect on a specific string
     */
    public triggerStringRipple(stringIndex: number, intensity: number = 1.0): void {
        if (stringIndex >= 0 && stringIndex < 6) {
            this.uniforms.uHarpVelocities.value[stringIndex] = intensity;
        }
    }

    /**
     * Decay all velocities toward zero (call each frame)
     */
    public decayVelocities(deltaTime: number, decayRate: number = 2.0): void {
        const velocities = this.uniforms.uHarpVelocities.value;
        for (let i = 0; i < 6; i++) {
            if (velocities[i] > 0) {
                velocities[i] = Math.max(0, velocities[i] - decayRate * deltaTime);
            }
        }
    }

    /**
     * Cleanup method for memory management
     */
    public destroy(): void {
        this.dispose();
    }
}
