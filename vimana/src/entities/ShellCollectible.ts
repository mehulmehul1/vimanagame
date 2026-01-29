/**
 * ShellCollectible - Nautilus shell collectible with procedural SDF shader
 *
 * Beautiful iridescent shell that materializes when the duet is complete.
 * Players can collect it to add to their UI inventory.
 *
 * Auto-selects TSL (WebGPU) or GLSL (WebGL2) implementation.
 */

import * as THREE from 'three';
import { shellVertexShader, shellFragmentShader } from '../shaders';
import { ShellMaterialTSL } from '../shaders/tsl';

/**
 * Detect if WebGPU/TSL is available
 */
function isWebGPURenderer(): boolean {
    return (window as any).rendererType === 'WebGPU';
}

export type ShellState = 'materializing' | 'idle' | 'collecting' | 'collected';

export interface ShellCollectibleConfig {
    spawnPosition: THREE.Vector3;
    scale: number;
    materializeDuration: number;
    collectDuration: number;
}

/**
 * ShellMaterial - Wrapper that auto-selects TSL or GLSL implementation
 */
class ShellMaterial {
    private material: THREE.ShaderMaterial | InstanceType<typeof ShellMaterialTSL>;
    private isTSL: boolean;

    // Uniforms object for API compatibility (GLSL mode only)
    public uniforms: {
        uTime: { value: number };
        uAppearProgress: { value: number };
        uDissolveAmount: { value: number };
        uCameraPosition: { value: THREE.Vector3 };
    };

    constructor() {
        this.isTSL = isWebGPURenderer();

        if (this.isTSL) {
            // Use TSL material for WebGPU
            this.material = new ShellMaterialTSL();
            console.log('[ShellMaterial] Using TSL (WebGPU) implementation');

            // Create uniforms object for API compatibility
            this.uniforms = {
                uTime: { value: 0 },
                uAppearProgress: { value: 0 },
                uDissolveAmount: { value: 1.0 },
                uCameraPosition: { value: new THREE.Vector3() }
            };
        } else {
            // Use GLSL shader material for WebGL2 fallback
            const uniforms = {
                uTime: { value: 0 },
                uAppearProgress: { value: 0 },
                uDissolveAmount: { value: 1.0 },
                uCameraPosition: { value: new THREE.Vector3() }
            };

            this.material = new THREE.ShaderMaterial({
                vertexShader: shellVertexShader,
                fragmentShader: shellFragmentShader,
                uniforms: uniforms,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: false
            });

            this.uniforms = uniforms;
            console.log('[ShellMaterial] Using GLSL (WebGL2) implementation');
        }
    }

    // THREE.JS MATERIAL FORWARDING
    get transparent(): boolean {
        return this.material.transparent;
    }
    set transparent(value: boolean) {
        this.material.transparent = value;
    }

    get side(): THREE.Side {
        return this.material.side;
    }
    set side(value: THREE.Side) {
        (this.material as THREE.ShaderMaterial).side = value;
    }

    get depthWrite(): boolean {
        return this.material.depthWrite;
    }
    set depthWrite(value: boolean) {
        this.material.depthWrite = value;
    }

    // PUBLIC API
    public setTime(time: number): void {
        if (this.isTSL) {
            (this.material as ShellMaterialTSL).setTime(time);
        } else {
            this.uniforms.uTime.value = time;
        }
    }

    public setAppearProgress(progress: number): void {
        if (this.isTSL) {
            (this.material as ShellMaterialTSL).setAppearProgress(progress);
        } else {
            this.uniforms.uAppearProgress.value = progress;
        }
    }

    public setDissolveAmount(amount: number): void {
        if (this.isTSL) {
            (this.material as ShellMaterialTSL).setDissolveAmount(amount);
        } else {
            this.uniforms.uDissolveAmount.value = amount;
        }
    }

    public setCameraPosition(position: THREE.Vector3): void {
        if (this.isTSL) {
            (this.material as ShellMaterialTSL).setCameraPosition(position);
        } else {
            this.uniforms.uCameraPosition.value.copy(position);
        }
    }

    public getMaterial(): THREE.Material {
        return this.material;
    }

    public dispose(): void {
        this.material.dispose();
    }
}

export class ShellCollectible extends THREE.Mesh {
    private shellMaterial: ShellMaterial;
    private state: ShellState;
    private materializeProgress: number = 0;
    private collectProgress: number = 0;
    private startPosition: THREE.Vector3;
    private uiTargetPosition: THREE.Vector3;
    private startRotation: THREE.Euler;

    // Uniforms reference for quick access (backward compatibility)
    public uniforms: {
        uTime: { value: number };
        uAppearProgress: { value: number };
        uDissolveAmount: { value: number };
        uCameraPosition: { value: THREE.Vector3 };
    };

    private config: ShellCollectibleConfig;

    constructor(scene: THREE.Scene, camera: THREE.Camera, config: Partial<ShellCollectibleConfig> = {}) {
        // Create shell material wrapper (auto-selects TSL/GLSL)
        const shellMaterial = new ShellMaterial();

        // High-poly sphere for smooth SDF displacement
        const geometry = new THREE.SphereGeometry(1, 128, 128);

        super(geometry, shellMaterial.getMaterial());

        this.shellMaterial = shellMaterial;
        this.uniforms = shellMaterial.uniforms;
        this.uiTargetPosition = new THREE.Vector3();

        this.config = {
            spawnPosition: new THREE.Vector3(0, 1.0, 1.0),
            scale: 0.15,
            materializeDuration: 3.0,
            collectDuration: 1.5,
            ...config
        };

        this.position.copy(this.config.spawnPosition);
        this.scale.setScalar(this.config.scale);
        this.startPosition = this.config.spawnPosition.clone();
        this.startRotation = this.rotation.clone();
        this.state = 'materializing';

        scene.add(this);
    }

    /**
     * Update shell animation
     */
    public update(deltaTime: number, time: number, cameraPosition: THREE.Vector3): void {
        this.shellMaterial.setTime(time);
        this.shellMaterial.setCameraPosition(cameraPosition);

        switch (this.state) {
            case 'materializing':
                this.updateMaterializing(deltaTime);
                break;
            case 'idle':
                this.updateIdle(deltaTime, time);
                break;
            case 'collecting':
                this.updateCollecting(deltaTime);
                break;
        }
    }

    /**
     * Update materializing animation
     */
    private updateMaterializing(deltaTime: number): void {
        this.materializeProgress += deltaTime / this.config.materializeDuration;

        if (this.materializeProgress >= 1.0) {
            this.materializeProgress = 1.0;
            this.shellMaterial.setAppearProgress(1.0);
            this.shellMaterial.setDissolveAmount(0.0);
            this.state = 'idle';
            return;
        }

        this.shellMaterial.setAppearProgress(this.materializeProgress);
        // Dissolve from 1.0 to 0.0 during materialize
        this.shellMaterial.setDissolveAmount(1.0 - this.materializeProgress);
    }

    /**
     * Update idle animation (bobbing and slow rotation)
     */
    private updateIdle(deltaTime: number, time: number): void {
        // Bobbing animation
        const bob = Math.sin(time * 2.0) * 0.05;
        this.position.y = this.startPosition.y + bob;

        // Slow rotation
        this.rotation.y += THREE.MathUtils.degToRad(15) * deltaTime;
    }

    /**
     * Update collection animation
     */
    private updateCollecting(deltaTime: number): void {
        this.collectProgress += deltaTime / this.config.collectDuration;

        if (this.collectProgress >= 1.0) {
            this.collectProgress = 1.0;
            this.state = 'collected';
            this.onCollectComplete();
            return;
        }

        // Smoothstep easing for natural motion
        const t = this.collectProgress;
        const eased = t * t * (3 - 2 * t);

        // Interpolate toward UI position
        this.position.lerpVectors(this.startPosition, this.uiTargetPosition, eased);

        // Scale down during flight
        const scale = this.config.scale * (1 - eased * 0.8);
        this.scale.setScalar(scale);

        // Spin during flight
        this.rotation.y += eased * Math.PI * 4 * deltaTime;
    }

    /**
     * Start collection animation
     */
    public collect(uiPosition: THREE.Vector3): void {
        if (this.state !== 'idle') return;
        this.state = 'collecting';
        this.uiTargetPosition = uiPosition.clone();
    }

    /**
     * Check if shell is hovered by raycast
     */
    public isHovered(raycaster: THREE.Raycaster): boolean {
        if (this.state !== 'idle') return false;
        const intersects = raycaster.intersectObject(this);
        return intersects.length > 0;
    }

    /**
     * Get current state
     */
    public getState(): ShellState {
        return this.state;
    }

    /**
     * Check if ready to collect
     */
    public canCollect(): boolean {
        return this.state === 'idle';
    }

    /**
     * Handle collection complete
     */
    private onCollectComplete(): void {
        // Remove from scene
        const scene = this.parent;
        if (scene) {
            scene.remove(this);
        }

        // Dispatch event
        window.dispatchEvent(new CustomEvent('shell-collected', {
            detail: { chamber: 'archive-of-voices' }
        }));

        // Cleanup resources
        this.geometry.dispose();
        this.shellMaterial.dispose();
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        const scene = this.parent;
        if (scene) {
            scene.remove(this);
        }
        this.geometry.dispose();
        this.shellMaterial.dispose();
    }
}

// Export TSL class for advanced use cases
export { ShellMaterialTSL } from '../shaders/tsl';
