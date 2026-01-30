import * as THREE from 'three';
import { stringHighlightVertexShader, stringHighlightFragmentShader } from '../shaders';
import { StringHighlightMaterialTSL } from '../shaders/tsl';

/**
 * Detect if WebGPU/TSL is available
 */
function isWebGPURenderer(): boolean {
    return (window as any).rendererType === 'WebGPU';
}

/**
 * StringHighlight - Manages glow effects on harp strings
 *
 * Adds glowing outlines and animations to strings during demonstration
 * to make it obvious which string the player should play.
 *
 * Auto-selects TSL (WebGPU) or GLSL (WebGL2) implementation.
 */
export class StringHighlight extends THREE.Group {
    private highlights: Map<number, StringGlow> = new Map();
    private scene: THREE.Scene;

    constructor(scene: THREE.Scene) {
        super();
        this.scene = scene;
    }

    /**
     * Highlight a specific string
     */
    public highlightString(stringIndex: number, position: THREE.Vector3): void {
        if (!this.highlights.has(stringIndex)) {
            const glow = new StringGlow(position);
            this.highlights.set(stringIndex, glow);
            this.scene.add(glow);
        } else {
            const glow = this.highlights.get(stringIndex)!;
            glow.updatePosition(position);
        }

        const glow = this.highlights.get(stringIndex)!;
        glow.activate();
    }

    /**
     * Remove highlight from a string
     */
    public unhighlightString(stringIndex: number): void {
        const glow = this.highlights.get(stringIndex);
        if (glow) {
            glow.deactivate();
        }
    }

    /**
     * Clear all highlights
     */
    public clearAll(): void {
        this.highlights.forEach(glow => glow.deactivate());
    }

    /**
     * Update all highlights
     */
    public update(deltaTime: number, time: number): void {
        this.highlights.forEach(glow => glow.update(deltaTime, time));

        // Clean up fully faded highlights
        for (const [index, glow] of this.highlights) {
            if (glow.isFinished()) {
                glow.destroy();
                this.scene.remove(glow);
                this.highlights.delete(index);
            }
        }
    }

    /**
     * Get active highlight count
     */
    public getActiveCount(): number {
        let count = 0;
        this.highlights.forEach(glow => {
            if (glow.isActive()) count++;
        });
        return count;
    }

    /**
     * Cleanup all highlights
     */
    public destroy(): void {
        this.highlights.forEach(glow => {
            glow.destroy();
            this.scene.remove(glow);
        });
        this.highlights.clear();
    }
}

/**
 * StringGlow - Individual string glow effect
 *
 * Auto-selects TSL (WebGPU) or GLSL (WebGL2) implementation.
 */
class StringGlow extends THREE.Mesh {
    private material: THREE.ShaderMaterial | StringHighlightMaterialTSL;
    private isTSL: boolean;
    private state: 'activating' | 'active' | 'deactivating' | 'inactive' = 'inactive';
    private animTime: number = 0;
    private intensity: number = 0;

    // Uniforms object for API compatibility (GLSL mode only)
    public uniforms: {
        uTime: { value: number };
        uIntensity: { value: number };
        uColor: { value: THREE.Color };
        uCameraPosition: { value: THREE.Vector3 };
    };

    constructor(position: THREE.Vector3) {
        // Vertical capsule/cylinder for string glow
        const geometry = new THREE.CapsuleGeometry(0.08, 2, 8, 16);

        const isTSL = isWebGPURenderer();

        // Initialize uniforms for GLSL mode
        const uniforms = {
            uTime: { value: 0 },
            uIntensity: { value: 0 },
            uColor: { value: new THREE.Color(0x00ffff) },
            uCameraPosition: { value: new THREE.Vector3() }
        };

        let material: THREE.ShaderMaterial | StringHighlightMaterialTSL;

        if (isTSL) {
            material = new StringHighlightMaterialTSL();
            console.log('[StringHighlight] Using TSL (WebGPU) implementation');
        } else {
            material = new THREE.ShaderMaterial({
                vertexShader: stringHighlightVertexShader,
                fragmentShader: stringHighlightFragmentShader,
                uniforms: uniforms,
                transparent: true,
                depthWrite: false,
                blending: THREE.AdditiveBlending,
                side: THREE.DoubleSide
            });
            console.log('[StringHighlight] Using GLSL (WebGL2) implementation');
        }

        // MUST call super() before accessing 'this'
        super(geometry, material);
        this.material = material;
        this.isTSL = isTSL;
        this.uniforms = uniforms;

        this.position.copy(position);
        this.visible = false;
    }

    public activate(): void {
        this.state = 'activating';
        this.animTime = 0;
        this.visible = true;
    }

    public deactivate(): void {
        if (this.state !== 'inactive') {
            this.state = 'deactivating';
            this.animTime = 0;
        }
    }

    public updatePosition(position: THREE.Vector3): void {
        this.position.copy(position);
    }

    public update(deltaTime: number, time: number): void {
        this.setTime(time);

        const fadeSpeed = 4.0;

        switch (this.state) {
            case 'activating':
                this.intensity = Math.min(1, this.intensity + deltaTime * fadeSpeed);
                if (this.intensity >= 1) {
                    this.state = 'active';
                }
                break;

            case 'active':
                this.intensity = 0.8 + Math.sin(time * 3) * 0.2;
                break;

            case 'deactivating':
                this.intensity = Math.max(0, this.intensity - deltaTime * fadeSpeed);
                if (this.intensity <= 0) {
                    this.state = 'inactive';
                    this.visible = false;
                }
                break;
        }

        this.setIntensity(this.intensity);
    }

    public isActive(): boolean {
        return this.state === 'active' || this.state === 'activating';
    }

    public isFinished(): boolean {
        return this.state === 'inactive' && this.intensity <= 0;
    }

    public setCameraPosition(position: THREE.Vector3): void {
        if (this.isTSL) {
            (this.material as StringHighlightMaterialTSL).setCameraPosition(position);
        } else {
            this.uniforms.uCameraPosition.value.copy(position);
        }
    }

    private setTime(time: number): void {
        if (this.isTSL) {
            (this.material as StringHighlightMaterialTSL).setTime(time);
        } else {
            this.uniforms.uTime.value = time;
        }
    }

    private setIntensity(value: number): void {
        if (this.isTSL) {
            (this.material as StringHighlightMaterialTSL).setIntensity(value);
        } else {
            this.uniforms.uIntensity.value = value;
        }
    }

    public destroy(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}
