import * as THREE from 'three';
import { teachingBeamVertexShader, teachingBeamFragmentShader } from '../shaders';
import { TeachingBeamMaterialTSL } from '../shaders/tsl';

/**
 * Detect if WebGPU/TSL is available
 */
function isWebGPURenderer(): boolean {
    return (window as any).rendererType === 'WebGPU';
}

/**
 * TeachingBeam - Visual beam connecting jelly to target string
 *
 * Creates a glowing energy beam that shows players which string
 * the jelly is demonstrating. Uses a scrolling texture for
 * animated flow effect.
 *
 * Auto-selects TSL (WebGPU) or GLSL (WebGL2) implementation.
 */
export class TeachingBeam extends THREE.Mesh {
    private material: THREE.ShaderMaterial | TeachingBeamMaterialTSL;
    private isTSL: boolean;
    private animTime: number = 0;
    private intensity: number = 0;
    private targetIntensity: number = 0;
    private jellyPosition: THREE.Vector3;
    private stringPosition: THREE.Vector3;
    private isActive: boolean = false;

    // Uniforms object for API compatibility (GLSL mode only)
    public uniforms: {
        uTime: { value: number };
        uIntensity: { value: number };
        uColor: { value: THREE.Color };
        uCameraPosition: { value: THREE.Vector3 };
    };

    constructor() {
        // Cylinder geometry (will be stretched and oriented)
        const geometry = new THREE.CylinderGeometry(0.08, 0.08, 1, 16, 1, true);

        const isTSL = isWebGPURenderer();

        // Initialize uniforms for GLSL mode
        const uniforms = {
            uTime: { value: 0 },
            uIntensity: { value: 0 },
            uColor: { value: new THREE.Color(0x00ffff) },
            uCameraPosition: { value: new THREE.Vector3() }
        };

        let material: THREE.ShaderMaterial | TeachingBeamMaterialTSL;

        if (isTSL) {
            material = new TeachingBeamMaterialTSL();
            console.log('[TeachingBeam] Using TSL (WebGPU) implementation');
        } else {
            material = new THREE.ShaderMaterial({
                vertexShader: teachingBeamVertexShader,
                fragmentShader: teachingBeamFragmentShader,
                uniforms: uniforms,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: false,
                blending: THREE.AdditiveBlending
            });
            console.log('[TeachingBeam] Using GLSL (WebGL2) implementation');
        }

        // MUST call super() before accessing 'this'
        super(geometry, material);

        this.material = material;
        this.isTSL = isTSL;
        this.uniforms = uniforms;
        this.jellyPosition = new THREE.Vector3();
        this.stringPosition = new THREE.Vector3();
        this.visible = false;
    }

    /**
     * Activate the beam with start and end positions
     */
    public activate(jellyPos: THREE.Vector3, stringPos: THREE.Vector3): void {
        this.jellyPosition.copy(jellyPos);
        this.stringPosition.copy(stringPos);
        this.isActive = true;
        this.targetIntensity = 1.0;
        this.visible = true;
        this.updateBeamGeometry();
    }

    /**
     * Deactivate the beam
     */
    public deactivate(): void {
        this.targetIntensity = 0;
        this.isActive = false;
    }

    /**
     * Update animation
     */
    public update(deltaTime: number, time: number, cameraPos: THREE.Vector3): void {
        // Update time uniform for scrolling effect
        this.setTime(time);
        this.setCameraPosition(cameraPos);

        // Smooth intensity transition
        this.intensity += (this.targetIntensity - this.intensity) * deltaTime * 5;
        this.setIntensity(this.intensity);

        // Hide when intensity is near zero
        if (this.intensity < 0.01 && !this.isActive) {
            this.visible = false;
        }

        // Update beam orientation if active
        if (this.isActive) {
            this.updateBeamGeometry();
        }
    }

    /**
     * Update beam geometry to connect points
     */
    private updateBeamGeometry(): void {
        // Calculate midpoint
        const midPoint = new THREE.Vector3()
            .addVectors(this.jellyPosition, this.stringPosition)
            .multiplyScalar(0.5);

        this.position.copy(midPoint);

        // Calculate distance (beam length)
        const distance = this.jellyPosition.distanceTo(this.stringPosition);
        this.scale.set(1, distance, 1);

        // Orient cylinder to point from string to jelly
        this.lookAt(this.jellyPosition);
        this.rotateX(Math.PI / 2); // Cylinder defaults to Y-up
    }

    /**
     * Set beam color
     */
    public setColor(color: THREE.Color | number): void {
        const colorObj = typeof color === 'number' ? new THREE.Color(color) : color;
        if (this.isTSL) {
            (this.material as TeachingBeamMaterialTSL).setColor(colorObj);
        } else {
            this.uniforms.uColor.value.copy(colorObj);
        }
    }

    /**
     * Check if beam is currently visible
     */
    public isVisible(): boolean {
        return this.intensity > 0.1;
    }

    private setTime(value: number): void {
        if (this.isTSL) {
            (this.material as TeachingBeamMaterialTSL).setTime(value);
        } else {
            this.uniforms.uTime.value = value;
        }
    }

    private setIntensity(value: number): void {
        if (this.isTSL) {
            (this.material as TeachingBeamMaterialTSL).setIntensity(value);
        } else {
            this.uniforms.uIntensity.value = value;
        }
    }

    private setCameraPosition(position: THREE.Vector3): void {
        if (this.isTSL) {
            (this.material as TeachingBeamMaterialTSL).setCameraPosition(position);
        } else {
            this.uniforms.uCameraPosition.value.copy(position);
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
