import * as THREE from 'three';
import { waterVertexShader, waterFragmentShader } from '../shaders';
import { WaterMaterialTSL } from '../shaders/tsl';

/**
 * Detect if WebGPU/TSL is available
 *
 * Checks for global renderer type flag set by createWebGPURenderer
 */
function isWebGPURenderer(): boolean {
    return (window as any).rendererType === 'WebGPU';
}

/**
 * EnhancedWaterMaterial - Bioluminescent water surface shader
 *
 * Responds to 6 harp strings with individual frequency tracking.
 * Creates ripple effects, caustics, and fresnel-based transparency.
 *
 * Automatically selects TSL (WebGPU) or GLSL (WebGL2) implementation
 * based on the current renderer type.
 *
 * Story: 4.4 - Water Material TSL Migration
 */
export class WaterMaterial {
    private material: THREE.ShaderMaterial | InstanceType<typeof WaterMaterialTSL>;
    private isTSL: boolean;

    private static readonly DEFAULT_COLOR = new THREE.Color(0x00ff88); // Cyan-green

    // Uniforms object for API compatibility (GLSL mode only)
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
        this.isTSL = isWebGPURenderer();

        // Initialize frequency arrays for 6 strings (C, D, E, F, G, A)
        const frequencies = new Float32Array(6);
        const velocities = new Float32Array(6);

        // Initialize jelly interaction arrays
        const jellyPositions = new Float32Array(6 * 3);  // 6 jellies × 3 (xyz)
        const jellyVelocities = new Float32Array(6);     // Movement intensity
        const jellyActive = new Float32Array(6);         // Active state (0 or 1)

        if (this.isTSL) {
            // Use TSL material for WebGPU
            this.material = new WaterMaterialTSL();
            console.log('[WaterMaterial] Using TSL (WebGPU) implementation');

            // Initialize uniforms object for API compatibility
            this.uniforms = {
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
        } else {
            // Use GLSL shader material for WebGL2 fallback
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

            this.material = new THREE.ShaderMaterial({
                vertexShader: waterVertexShader,
                fragmentShader: waterFragmentShader,
                uniforms: uniforms,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: false
            });

            this.uniforms = uniforms;
            console.log('[WaterMaterial] Using GLSL (WebGL2) implementation');
        }
    }

    // ========================================================================
    // THREE.JS MATERIAL FORWARDING (makes WaterMaterial behave like Material)
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
            (this.material as WaterMaterialTSL).setTime(time);
        } else {
            this.uniforms.uTime.value = time;
        }
    }

    /**
     * Update harp string frequency (0-5 for strings C-A)
     */
    public setStringFrequency(index: number, frequency: number): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setStringFrequency(index, frequency);
        } else {
            if (index >= 0 && index < 6) {
                this.uniforms.uHarpFrequencies.value[index] = frequency;
            }
        }
    }

    /**
     * Update harp string velocity for intensity modulation
     */
    public setStringVelocity(index: number, velocity: number): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setStringVelocity(index, velocity);
        } else {
            if (index >= 0 && index < 6) {
                this.uniforms.uHarpVelocities.value[index] = velocity;
            }
        }
    }

    /**
     * Update duet progress (0-1)
     */
    public setDuetProgress(progress: number): void {
        const clamped = Math.max(0, Math.min(1, progress));
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setDuetProgress(clamped);
        } else {
            this.uniforms.uDuetProgress.value = clamped;
        }
    }

    /**
     * Update harmonic resonance intensity (0-1)
     */
    public setHarmonicResonance(resonance: number): void {
        const clamped = Math.max(0, Math.min(1, resonance));
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setHarmonicResonance(clamped);
        } else {
            this.uniforms.uHarmonicResonance.value = clamped;
        }
    }

    /**
     * Update ship patience teaching state
     */
    public setShipPatience(patience: number): void {
        const clamped = Math.max(0, Math.min(1, patience));
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setShipPatience(clamped);
        } else {
            this.uniforms.uShipPatience.value = clamped;
        }
    }

    /**
     * Update camera position for fresnel calculations
     */
    public setCameraPosition(position: THREE.Vector3): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setCameraPosition(position);
        } else {
            this.uniforms.uCameraPosition.value.copy(position);
        }
    }

    /**
     * Trigger ripple effect on a specific string
     */
    public triggerStringRipple(stringIndex: number, intensity: number = 1.0): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).triggerStringRipple(stringIndex, intensity);
        } else {
            if (stringIndex >= 0 && stringIndex < 6) {
                this.uniforms.uHarpVelocities.value[stringIndex] = intensity;
            }
        }
    }

    /**
     * Decay all velocities toward zero (call each frame)
     */
    public decayVelocities(deltaTime: number, decayRate: number = 2.0): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).decayVelocities(deltaTime, decayRate);
        } else {
            const velocities = this.uniforms.uHarpVelocities.value;
            for (let i = 0; i < 6; i++) {
                if (velocities[i] > 0) {
                    velocities[i] = Math.max(0, velocities[i] - decayRate * deltaTime);
                }
            }
        }
    }

    /**
     * Cleanup method for memory management
     */
    public destroy(): void {
        this.material.dispose();
    }

    // ========== WaterBall-inspired Control Methods ==========

    /**
     * Set sphere radius for plane-to-sphere transformation
     * Higher values create larger sphere as duetProgress increases
     */
    public setSphereRadius(radius: number): void {
        const clamped = Math.max(0.1, radius);
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setSphereRadius(clamped);
        } else {
            this.uniforms.uSphereRadius.value = clamped;
        }
    }

    /**
     * Set center point for sphere transformation
     * This is the focal point around which the water forms a tunnel
     */
    public setSphereCenter(center: THREE.Vector3): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setSphereCenter(center);
        } else {
            this.uniforms.uSphereCenter.value.copy(center);
        }
    }

    /**
     * Set environment map for reflections
     * Pass null to use procedural gradient fallback
     */
    public setEnvMap(texture: THREE.Texture | null): void {
        if (this.isTSL) {
            // TSL uses MeshPhysicalNodeMaterial which handles envMap differently
            // For now, we keep this for API compatibility
            console.warn('[WaterMaterial] TSL envMap handling not yet implemented');
        } else {
            this.uniforms.uEnvMap.value = texture;
            this.needsUpdate = true;
        }
    }

    /**
     * Set water thickness for transmittance calculation
     * Thicker water = more light absorption = deeper color
     */
    public setThickness(thickness: number): void {
        const clamped = Math.max(0.01, thickness);
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setThickness(clamped);
        } else {
            this.uniforms.uThickness.value = clamped;
        }
    }

    /**
     * Set absorption density
     * Higher density = more color absorption = more opaque water
     * WaterBall uses 0.7 as default
     */
    public setDensity(density: number): void {
        const clamped = Math.max(0, density);
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setDensity(clamped);
        } else {
            this.uniforms.uDensity.value = clamped;
        }
    }

    /**
     * Set Fresnel F0 (reflectance at normal incidence)
     * Water: 0.02, Ice: 0.02-0.09, Glass: 0.04
     * Higher = more reflective at glancing angles
     */
    public setF0(f0: number): void {
        const clamped = Math.max(0, Math.min(1, f0));
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setF0(clamped);
        } else {
            this.uniforms.uF0.value = clamped;
        }
    }

    /**
     * Animate sphere transformation
     * Call this with duetProgress to smoothly transform flat water → sphere tunnel
     */
    public setSphereTransformation(progress: number, radius: number): void {
        const clampedProgress = Math.max(0, Math.min(1, progress));
        const clampedRadius = Math.max(0.1, radius);
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setSphereTransformation(clampedProgress, clampedRadius);
        } else {
            this.uniforms.uDuetProgress.value = clampedProgress;
            this.uniforms.uSphereRadius.value = clampedRadius;
        }
    }

    // ========== Jelly Creature Interaction Methods ==========

    /**
     * Update jelly position for water interaction
     * @param index Jelly index (0-5)
     * @param position World position of the jelly
     */
    public setJellyPosition(index: number, position: THREE.Vector3): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setJellyPosition(index, position);
        } else {
            if (index >= 0 && index < 6) {
                const arr = this.uniforms.uJellyPositions.value;
                arr[index * 3 + 0] = position.x;
                arr[index * 3 + 1] = position.y;
                arr[index * 3 + 2] = position.z;
            }
        }
    }

    /**
     * Update jelly velocity for ripple intensity
     * @param index Jelly index (0-5)
     * @param velocity Movement intensity (0-1 recommended)
     */
    public setJellyVelocity(index: number, velocity: number): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setJellyVelocity(index, velocity);
        } else {
            if (index >= 0 && index < 6) {
                this.uniforms.uJellyVelocities.value[index] = velocity;
            }
        }
    }

    /**
     * Set jelly active state (visible/hidden)
     * @param index Jelly index (0-5)
     * @param active True if jelly is visible and should create ripples
     */
    public setJellyActive(index: number, active: boolean): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).setJellyActive(index, active);
        } else {
            if (index >= 0 && index < 6) {
                this.uniforms.uJellyActive.value[index] = active ? 1.0 : 0.0;
            }
        }
    }

    /**
     * Update all jelly positions from JellyManager
     * Call this each frame to sync jelly positions with water shader
     */
    public updateJellyPositions(jellyManager: any): void {
        if (this.isTSL) {
            (this.material as WaterMaterialTSL).updateJellyPositions(jellyManager);
        } else {
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

    /**
     * Get the underlying Three.js material
     * For internal use with Three.js Mesh objects
     */
    public getMaterial(): THREE.Material {
        return this.material;
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
export { WaterMaterialTSL } from '../shaders/tsl';
