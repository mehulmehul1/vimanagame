import * as THREE from 'three';

/**
 * VortexParticles - Particle system flowing through the torus vortex
 *
 * Features:
 * - 2000 particles in spiral torus distribution
 * - LOD system with skip rates based on activation
 * - Spin animation with proper trigonometry
 * - CRITICAL: Z-coordinate uses dist, NOT sinAngle (bug fix)
 * - CRITICAL: tubeOffset randomized on each reset
 */
export class VortexParticles extends THREE.Points {
    private static readonly MAX_PARTICLES = 2000;
    private static readonly TORUS_MAJOR_RADIUS = 2.0;
    private static readonly TORUS_TUBE_RADIUS = 0.4;

    private geometry: THREE.BufferGeometry;
    private material: THREE.PointsMaterial;
    private positions: Float32Array;
    private colors: Float32Array;
    private angles: Float32Array;
    private tubeOffsets: Float32Array;
    private speeds: Float32Array;

    private time: number = 0;
    private activation: number = 0;

    constructor(particleCount: number = VortexParticles.MAX_PARTICLES) {
        // Initialize arrays
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const angles = new Float32Array(particleCount);
        const tubeOffsets = new Float32Array(particleCount);
        const speeds = new Float32Array(particleCount);

        // Create geometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Create material with additive blending
        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });

        super(geometry, material);

        this.geometry = geometry;
        this.material = material;
        this.positions = positions;
        this.colors = colors;
        this.angles = angles;
        this.tubeOffsets = tubeOffsets;
        this.speeds = speeds;

        // Initialize particles
        this.initParticles();
    }

    /**
     * Initialize all particles in torus distribution
     */
    private initParticles(): void {
        const count = this.positions.length / 3;

        for (let i = 0; i < count; i++) {
            this.resetParticle(i, true);
        }

        this.geometry.attributes.position.needsUpdate = true;
        this.geometry.attributes.color.needsUpdate = true;
    }

    /**
     * Reset a single particle to initial position
     * CRITICAL BUG FIX: Randomize tubeOffset INSIDE this function
     */
    private resetParticle(index: number, initial: boolean = false): void {
        const R = VortexParticles.TORUS_MAJOR_RADIUS;
        const r = VortexParticles.TORUS_TUBE_RADIUS;

        // Angle around torus (0 to 2PI)
        const angle = Math.random() * Math.PI * 2;
        this.angles[index] = angle;

        // CRITICAL BUG FIX: Randomize tubeOffset on each reset
        // Original bug had this in init only, causing all particles to same offset
        const tubeOffset = (Math.random() - 0.5) * 0.8;
        this.tubeOffsets[index] = tubeOffset;

        // Speed variation
        this.speeds[index] = 0.5 + Math.random() * 0.5;

        // Initial position calculation
        this.updateParticlePosition(index);

        // Color based on position (cyan to purple gradient)
        const t = (angle / (Math.PI * 2) + 0.5) % 1.0;
        this.colors[index * 3] = 0;         // R
        this.colors[index * 3 + 1] = 1;     // G
        this.colors[index * 3 + 2] = t;     // B
    }

    /**
     * Update particle position based on current angle
     * CRITICAL BUG FIX: Z-coordinate MUST use dist, NOT sinAngle
     */
    private updateParticlePosition(index: number): void {
        const R = VortexParticles.TORUS_MAJOR_RADIUS;
        const r = VortexParticles.TORUS_TUBE_RADIUS;
        const angle = this.angles[index];
        const tubeOffset = this.tubeOffsets[index];

        // Torus position calculation
        const cosAngle = Math.cos(angle);
        const sinAngle = Math.sin(angle);
        const dist = R + r * cosAngle;

        // CRITICAL BUG FIX: Use dist for Z, not sinAngle
        // WRONG: this.positions[i * 3 + 2] = this.positions[i * 3 + 1];
        // CORRECT:
        this.positions[index * 3] = dist * Math.sin(angle);     // X
        this.positions[index * 3 + 1] = r * sinAngle + tubeOffset;  // Y
        this.positions[index * 3 + 2] = dist * Math.cos(angle); // Z - Uses dist!
    }

    /**
     * Update particle animation
     * Implements LOD skip logic based on activation
     */
    public update(deltaTime: number, activationLevel: number): void {
        this.time += deltaTime;
        this.activation = activationLevel;

        const count = this.positions.length / 3;
        let needsUpdate = false;

        for (let i = 0; i < count; i++) {
            // CRITICAL: LOD skip logic
            // Skip particles based on activation for performance
            const lodSkip = activationLevel < 0.3 ? (i % 2 === 0) :
                            activationLevel < 0.6 ? (i % 4 === 0) : false;
            if (lodSkip) continue;

            // Spin speed increases with activation
            const spinSpeed = (1.0 + activationLevel * 3.0) * this.speeds[i];
            this.angles[i] += spinSpeed * deltaTime;

            // Wrap angle
            if (this.angles[i] > Math.PI * 2) {
                this.angles[i] -= Math.PI * 2;
            }

            this.updateParticlePosition(i);
            needsUpdate = true;
        }

        if (needsUpdate) {
            this.geometry.attributes.position.needsUpdate = true;
        }

        // Update material opacity based on activation
        this.material.opacity = 0.3 + activationLevel * 0.5;
    }

    /**
     * Set particle count for LOD
     */
    public setParticleCount(count: number): void {
        // For mobile fallback, we'd create a smaller buffer
        // This is a placeholder for dynamic LOD adjustment
    }

    /**
     * Cleanup method for memory management
     */
    public destroy(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}
