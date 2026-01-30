/**
 * SynchronizedSplashEffect - Visual effect for synchronized jellyfish splash
 *
 * STORY-HARP-102: Creates expanding ripple rings from all jelly positions
 * simultaneously, emphasizing the "unified" nature of the turn signal.
 *
 * Design Philosophy:
 * - The splash is NOT part of the musical sequence
 * - It is a clear SIGNAL, not gameplay content
 * - All movement synchronized emphasizes "togetherness"
 */

import * as THREE from 'three';

interface SplashRipple {
    position: THREE.Vector3;
    radius: number;
    maxRadius: number;
    alpha: number;
    age: number;
}

export class SynchronizedSplashEffect extends THREE.Group {
    private ripples: SplashRipple[] = [];
    private active: boolean = false;
    private maxAge: number = 0.8; // Duration matches audio

    // Reusable geometry and materials for performance
    private ringGeometry: THREE.RingGeometry;
    private ringMaterial: THREE.MeshBasicMaterial;

    // Mesh pool for rendering ripples
    private rippleMeshes: THREE.Mesh[] = [];

    constructor() {
        super();

        // Create ring geometry (will be scaled per ripple)
        this.ringGeometry = new THREE.RingGeometry(0.1, 0.15, 32);

        // Create material with cyan/white bioluminescent color
        this.ringMaterial = new THREE.MeshBasicMaterial({
            color: 0x88ffff,
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });

        // Pre-create mesh pool (max 6 jellies)
        for (let i = 0; i < 6; i++) {
            const mesh = new THREE.Mesh(this.ringGeometry, this.ringMaterial.clone());
            mesh.visible = false;
            mesh.rotation.x = -Math.PI / 2; // Lay flat on water surface
            this.add(mesh);
            this.rippleMeshes.push(mesh);
        }

        console.log('[SynchronizedSplashEffect] Initialized');
    }

    /**
     * Trigger synchronized splash effect
     *
     * @param positions - World positions of all jellies splashing
     */
    public trigger(positions: THREE.Vector3[]): void {
        this.active = true;
        this.ripples = [];

        // Hide all meshes first
        for (const mesh of this.rippleMeshes) {
            mesh.visible = false;
        }

        // Create ripple at each jelly position
        for (let i = 0; i < positions.length && i < this.rippleMeshes.length; i++) {
            const pos = positions[i];
            this.ripples.push({
                position: pos.clone(),
                radius: 0.1,
                maxRadius: 2.5,
                alpha: 1.0,
                age: 0
            });

            // Position and show mesh
            const mesh = this.rippleMeshes[i];
            mesh.position.copy(pos);
            mesh.visible = true;
            mesh.scale.set(1, 1, 1);
            (mesh.material as THREE.MeshBasicMaterial).opacity = 0.8;
        }

        console.log(`[SynchronizedSplashEffect] Triggered with ${positions.length} ripples`);
    }

    /**
     * Update all active ripples
     *
     * @param deltaTime - Time since last frame in seconds
     */
    public update(deltaTime: number): void {
        if (!this.active) return;

        for (let i = 0; i < this.ripples.length; i++) {
            const ripple = this.ripples[i];
            const mesh = this.rippleMeshes[i];

            if (!mesh) continue;

            // Age the ripple
            ripple.age += deltaTime;

            // Update radius (expand from 0.1 to maxRadius)
            const progress = ripple.age / this.maxAge;
            ripple.radius = THREE.MathUtils.lerp(0.1, ripple.maxRadius, progress);

            // Update alpha (fade out)
            ripple.alpha = 1.0 - progress;

            // Apply to mesh
            mesh.scale.set(ripple.radius, ripple.radius, 1);
            (mesh.material as THREE.MeshBasicMaterial).opacity = ripple.alpha * 0.8;
        }

        // Remove finished ripples
        this.ripples = this.ripples.filter(r => r.age < this.maxAge);

        // Hide meshes for finished ripples
        for (let i = this.ripples.length; i < this.rippleMeshes.length; i++) {
            this.rippleMeshes[i].visible = false;
        }

        if (this.ripples.length === 0) {
            this.active = false;
        }
    }

    /**
     * Check if effect is currently active
     */
    public isActive(): boolean {
        return this.active;
    }

    /**
     * Force stop all effects
     */
    public stop(): void {
        this.active = false;
        this.ripples = [];
        for (const mesh of this.rippleMeshes) {
            mesh.visible = false;
        }
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.stop();

        for (const mesh of this.rippleMeshes) {
            this.remove(mesh);
            if (mesh.material instanceof THREE.Material) {
                mesh.material.dispose();
            }
        }

        this.ringGeometry.dispose();
        this.ringMaterial.dispose();

        this.rippleMeshes = [];
    }
}
