/**
 * ShellCollectible - Nautilus shell collectible with procedural SDF shader
 *
 * Beautiful iridescent shell that materializes when the duet is complete.
 * Players can collect it to add to their UI inventory.
 */

import * as THREE from 'three';
import { shellVertexShader, shellFragmentShader } from '../shaders';

export type ShellState = 'materializing' | 'idle' | 'collecting' | 'collected';

export interface ShellCollectibleConfig {
    spawnPosition: THREE.Vector3;
    scale: number;
    materializeDuration: number;
    collectDuration: number;
}

export class ShellCollectible extends THREE.Mesh {
    private shellMaterial: THREE.ShaderMaterial;
    private state: ShellState;
    private materializeProgress: number = 0;
    private collectProgress: number = 0;
    private startPosition: THREE.Vector3;
    private uiTargetPosition: THREE.Vector3;
    private startRotation: THREE.Euler;

    // Uniforms reference for quick access
    public uniforms: {
        uTime: { value: number };
        uAppearProgress: { value: number };
        uDissolveAmount: { value: number };
        uCameraPosition: { value: THREE.Vector3 };
    };

    private config: ShellCollectibleConfig;

    constructor(scene: THREE.Scene, camera: THREE.Camera, config: Partial<ShellCollectibleConfig> = {}) {
        // Create shader uniforms
        const uniforms = {
            uTime: { value: 0 },
            uAppearProgress: { value: 0 },
            uDissolveAmount: { value: 1.0 },
            uCameraPosition: { value: new THREE.Vector3() }
        };

        // Shell material with procedural SDF shader
        const material = new THREE.ShaderMaterial({
            vertexShader: shellVertexShader,
            fragmentShader: shellFragmentShader,
            uniforms: uniforms,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });

        // High-poly sphere for smooth SDF displacement
        const geometry = new THREE.SphereGeometry(1, 128, 128);

        super(geometry, material);

        this.uniforms = uniforms;
        this.shellMaterial = material;
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
        this.uniforms.uTime.value = time;
        this.uniforms.uCameraPosition.value.copy(cameraPosition);

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
            this.uniforms.uAppearProgress.value = 1.0;
            this.uniforms.uDissolveAmount.value = 0.0;
            this.state = 'idle';
            return;
        }

        this.uniforms.uAppearProgress.value = this.materializeProgress;
        // Dissolve from 1.0 to 0.0 during materialize
        this.uniforms.uDissolveAmount.value = 1.0 - this.materializeProgress;
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
