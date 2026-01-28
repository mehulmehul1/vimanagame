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
        // WaterBall-inspired uniforms
        uSphereRadius: { value: number };
        uSphereCenter: { value: THREE.Vector3 };
        uEnvMap: { value: THREE.Texture | null };
        uThickness: { value: number };
        uDensity: { value: number };
        uF0: { value: number }; // Fresnel reflectance at normal incidence
        // Jelly creature interaction uniforms
        uJellyPositions: { value: Float32Array };   // 6 jellies × 3 (xyz)
        uJellyVelocities: { value: Float32Array }; // 6 jelly movement intensities
        uJellyActive: { value: Float32Array };     // 6 jelly active states
    };

    constructor() {
        // Initialize frequency arrays for 6 strings (C, D, E, F, G, A)
        const frequencies = new Float32Array(6);
        const velocities = new Float32Array(6);

        // Initialize jelly interaction arrays
        const jellyPositions = new Float32Array(6 * 3);  // 6 jellies × 3 (xyz)
        const jellyVelocities = new Float32Array(6);     // Movement intensity
        const jellyActive = new Float32Array(6);         // Active state (0 or 1)

        const uniforms = {
            uTime: { value: 0 },
            uHarpFrequencies: { value: frequencies },
            uHarpVelocities: { value: velocities },
            uDuetProgress: { value: 0 },
            uShipPatience: { value: 1.0 },
            uBioluminescentColor: { value: WaterMaterial.DEFAULT_COLOR.clone() },
            uHarmonicResonance: { value: 0 },
            uCameraPosition: { value: new THREE.Vector3() },
            // WaterBall-inspired defaults
            uSphereRadius: { value: 8.0 }, // Radius of sphere transformation
            uSphereCenter: { value: new THREE.Vector3(0, 0, -5) }, // Center of sphere
            uEnvMap: { value: null }, // Optional environment map for reflections
            uThickness: { value: 1.0 }, // Water thickness for transmittance
            uDensity: { value: 0.7 }, // Density for absorption (matches WaterBall)
            uF0: { value: 0.02 }, // Fresnel base reflectance (water)
            // Jelly creature interaction uniforms
            uJellyPositions: { value: jellyPositions },
            uJellyVelocities: { value: jellyVelocities },
            uJellyActive: { value: jellyActive }
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

    // ========== WaterBall-inspired Control Methods ==========

    /**
     * Set sphere radius for plane-to-sphere transformation
     * Higher values create larger sphere as duetProgress increases
     */
    public setSphereRadius(radius: number): void {
        this.uniforms.uSphereRadius.value = Math.max(0.1, radius);
    }

    /**
     * Set center point for sphere transformation
     * This is the focal point around which the water forms a tunnel
     */
    public setSphereCenter(center: THREE.Vector3): void {
        this.uniforms.uSphereCenter.value.copy(center);
    }

    /**
     * Set environment map for reflections
     * Pass null to use procedural gradient fallback
     */
    public setEnvMap(texture: THREE.Texture | null): void {
        this.uniforms.uEnvMap.value = texture;
        this.needsUpdate = true;
    }

    /**
     * Set water thickness for transmittance calculation
     * Thicker water = more light absorption = deeper color
     */
    public setThickness(thickness: number): void {
        this.uniforms.uThickness.value = Math.max(0.01, thickness);
    }

    /**
     * Set absorption density
     * Higher density = more color absorption = more opaque water
     * WaterBall uses 0.7 as default
     */
    public setDensity(density: number): void {
        this.uniforms.uDensity.value = Math.max(0, density);
    }

    /**
     * Set Fresnel F0 (reflectance at normal incidence)
     * Water: 0.02, Ice: 0.02-0.09, Glass: 0.04
     * Higher = more reflective at glancing angles
     */
    public setF0(f0: number): void {
        this.uniforms.uF0.value = Math.max(0, Math.min(1, f0));
    }

    /**
     * Animate sphere transformation
     * Call this with duetProgress to smoothly transform flat water → sphere tunnel
     */
    public setSphereTransformation(progress: number, radius: number): void {
        this.uniforms.uDuetProgress.value = Math.max(0, Math.min(1, progress));
        this.uniforms.uSphereRadius.value = Math.max(0.1, radius);
    }

    // ========== Jelly Creature Interaction Methods ==========

    /**
     * Update jelly position for water interaction
     * @param index Jelly index (0-5)
     * @param position World position of the jelly
     */
    public setJellyPosition(index: number, position: THREE.Vector3): void {
        if (index >= 0 && index < 6) {
            const arr = this.uniforms.uJellyPositions.value;
            arr[index * 3 + 0] = position.x;
            arr[index * 3 + 1] = position.y;
            arr[index * 3 + 2] = position.z;
        }
    }

    /**
     * Update jelly velocity for ripple intensity
     * @param index Jelly index (0-5)
     * @param velocity Movement intensity (0-1 recommended)
     */
    public setJellyVelocity(index: number, velocity: number): void {
        if (index >= 0 && index < 6) {
            this.uniforms.uJellyVelocities.value[index] = velocity;
        }
    }

    /**
     * Set jelly active state (visible/hidden)
     * @param index Jelly index (0-5)
     * @param active True if jelly is visible and should create ripples
     */
    public setJellyActive(index: number, active: boolean): void {
        if (index >= 0 && index < 6) {
            this.uniforms.uJellyActive.value[index] = active ? 1.0 : 0.0;
        }
    }

    /**
     * Update all jelly positions from JellyManager
     * Call this each frame to sync jelly positions with water shader
     */
    public updateJellyPositions(jellyManager: any): void {
        if (!jellyManager) return;

        for (let i = 0; i < 6; i++) {
            const jelly = jellyManager.getJelly?.(i);
            if (jelly && !jelly.isHidden?.()) {
                // Update position
                this.setJellyPosition(i, jelly.position);

                // Calculate velocity based on state
                const state = jelly.getState?.();
                const isMoving = state === 'spawning' || state === 'submerging';
                this.setJellyVelocity(i, isMoving ? 0.5 : 0.1);

                // Mark as active
                this.setJellyActive(i, true);
            } else {
                // Jelly is hidden
                this.setJellyActive(i, false);
            }
        }
    }
}
