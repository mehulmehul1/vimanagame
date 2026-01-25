import * as THREE from 'three';

/**
 * StringHighlight - Manages glow effects on harp strings
 *
 * Adds glowing outlines and animations to strings during demonstration
 * to make it obvious which string the player should play.
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
        // Update all active highlights
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
 */
class StringGlow extends THREE.Mesh {
    private material: THREE.ShaderMaterial;
    private state: 'activating' | 'active' | 'deactivating' | 'inactive' = 'inactive';
    private animTime: number = 0;
    private intensity: number = 0;

    constructor(position: THREE.Vector3) {
        // Vertical capsule/cylinder for string glow
        const geometry = new THREE.CapsuleGeometry(0.08, 2, 8, 16);

        const uniforms = {
            uTime: { value: 0 },
            uIntensity: { value: 0 },
            uColor: { value: new THREE.Color(0x00ffff) },
            uCameraPosition: { value: new THREE.Vector3() }
        };

        const vertexShader = `
            varying vec3 vNormal;
            varying vec3 vWorldPosition;
            varying vec2 vUv;

            void main() {
                vNormal = normalize(normalMatrix * normal);
                vec4 worldPos = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPos.xyz;
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            uniform float uIntensity;
            uniform vec3 uColor;
            uniform vec3 uCameraPosition;
            uniform float uTime;

            varying vec3 vNormal;
            varying vec3 vWorldPosition;
            varying vec2 vUv;

            void main() {
                // Fresnel rim lighting
                vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
                float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 3.0);

                // Animated pulse traveling up the string
                float pulse = sin(vUv.y * 10.0 - uTime * 5.0) * 0.5 + 0.5;
                pulse = pow(pulse, 2.0);

                // Combine effects
                float glow = fresnel * 0.7 + pulse * 0.3;

                // Apply intensity
                vec3 color = uColor * glow * uIntensity;

                // Alpha based on intensity and fresnel
                float alpha = (fresnel * 0.5 + 0.3) * uIntensity;

                gl_FragColor = vec4(color, alpha);
            }
        `;

        const material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide
        });

        super(geometry, material);
        this.material = material;

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
        this.material.uniforms.uTime.value = time;
        // Camera position would be set by parent

        const fadeSpeed = 4.0;

        switch (this.state) {
            case 'activating':
                this.intensity = Math.min(1, this.intensity + deltaTime * fadeSpeed);
                if (this.intensity >= 1) {
                    this.state = 'active';
                }
                break;

            case 'active':
                // Pulse intensity slightly while active
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

        this.material.uniforms.uIntensity.value = this.intensity;
    }

    public isActive(): boolean {
        return this.state === 'active' || this.state === 'activating';
    }

    public isFinished(): boolean {
        return this.state === 'inactive' && this.intensity <= 0;
    }

    public setCameraPosition(position: THREE.Vector3): void {
        this.material.uniforms.uCameraPosition.value.copy(position);
    }

    public destroy(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}
