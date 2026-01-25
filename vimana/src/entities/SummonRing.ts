import * as THREE from 'three';

/**
 * SummonRing - Visual effect that appears on water before jelly emerges
 *
 * Creates an expanding ring of light that draws attention to spawn location.
 * Helps players notice where jellies will appear.
 */
export class SummonRing extends THREE.Mesh {
    private material: THREE.ShaderMaterial;
    private animTime: number = 0;
    private duration: number = 0.6; // Duration before jelly spawns
    private state: 'playing' | 'complete' = 'playing';
    private onCompleteCallback?: () => void;

    constructor() {
        // Ring geometry (flat circle on water surface)
        const geometry = new THREE.RingGeometry(0.1, 0.3, 64);

        // Shader for animated expanding ring effect
        const animDuration = 0.6; // Duration before jelly spawns

        const uniforms = {
            uTime: { value: 0 },
            uDuration: { value: animDuration },
            uColor: { value: new THREE.Color(0x00ffff) }
        };

        const vertexShader = `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            uniform float uTime;
            uniform float uDuration;
            uniform vec3 uColor;
            varying vec2 vUv;

            void main() {
                // Distance from center (0.5, 0.5 is center)
                vec2 center = vec2(0.5, 0.5);
                float dist = distance(vUv, center);

                // Progress (0 to 1)
                float progress = uTime / uDuration;

                // Create expanding ring
                float ringWidth = 0.15;
                float ringCenter = progress * 0.8;
                float ring = 1.0 - smoothstep(ringCenter, ringCenter + ringWidth, dist);
                ring *= smoothstep(ringCenter - ringWidth, ringCenter, dist);

                // Fade out at edges
                float alpha = ring * (1.0 - progress);

                // Inner glow
                float innerGlow = 1.0 - smoothstep(0.0, 0.3, dist);
                innerGlow *= progress * 0.5;

                vec3 finalColor = uColor * (ring + innerGlow);
                float finalAlpha = alpha * 0.8 + innerGlow * 0.3;

                gl_FragColor = vec4(finalColor, finalAlpha);
            }
        `;

        const material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        super(geometry, material);
        this.material = material;
        this.duration = animDuration;

        // Lay flat on water surface
        this.rotation.x = -Math.PI / 2;
        this.position.y = 0.05; // Slightly above water to prevent z-fighting
    }

    /**
     * Start the summon ring animation
     */
    public play(onComplete?: () => void): void {
        this.state = 'playing';
        this.animTime = 0;
        this.onCompleteCallback = onComplete;
        this.visible = true;
        this.material.uniforms.uTime.value = 0;
    }

    /**
     * Update the animation
     */
    public update(deltaTime: number): void {
        if (this.state !== 'playing') return;

        this.animTime += deltaTime;
        this.material.uniforms.uTime.value = this.animTime;

        // Scale up as ring expands
        const scale = 1 + this.animTime / this.duration * 2;
        this.scale.set(scale, scale, 1);

        // Check completion
        if (this.animTime >= this.duration) {
            this.state = 'complete';
            this.visible = false;
            if (this.onCompleteCallback) {
                this.onCompleteCallback();
                this.onCompleteCallback = undefined;
            }
        }
    }

    /**
     * Check if animation is complete
     */
    public isComplete(): boolean {
        return this.state === 'complete';
    }

    /**
     * Reset for reuse
     */
    public reset(): void {
        this.state = 'playing';
        this.animTime = 0;
        this.visible = false;
        this.scale.set(1, 1, 1);
        this.material.uniforms.uTime.value = 0;
    }

    /**
     * Set the ring color
     */
    public setColor(color: THREE.Color): void {
        this.material.uniforms.uColor.value.copy(color);
    }

    /**
     * Set animation duration
     */
    public setDuration(seconds: number): void {
        this.duration = seconds;
        this.material.uniforms.uDuration.value = seconds;
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}
