import * as THREE from 'three';

/**
 * TeachingBeam - Visual beam connecting jelly to target string
 *
 * Creates a glowing energy beam that shows players which string
 * the jelly is demonstrating. Uses a scrolling texture for
 * animated flow effect.
 */
export class TeachingBeam extends THREE.Mesh {
    private material: THREE.ShaderMaterial;
    private animTime: number = 0;
    private intensity: number = 0;
    private targetIntensity: number = 0;
    private jellyPosition: THREE.Vector3;
    private stringPosition: THREE.Vector3;
    private isActive: boolean = false;

    constructor() {
        // Cylinder geometry (will be stretched and oriented)
        const geometry = new THREE.CylinderGeometry(0.08, 0.08, 1, 16, 1, true);

        const vertexShader = `
            varying vec2 vUv;
            varying vec3 vWorldPosition;
            varying vec3 vNormal;

            void main() {
                vUv = uv;
                vNormal = normalize(normalMatrix * normal);
                vec4 worldPos = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPos.xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            uniform float uTime;
            uniform float uIntensity;
            uniform vec3 uColor;
            uniform vec3 uCameraPosition;

            varying vec2 vUv;
            varying vec3 vWorldPosition;
            varying vec3 vNormal;

            void main() {
                // Scrolling effect for energy flow
                float flow = mod(vUv.y * 4.0 - uTime * 3.0, 1.0);
                float pulse = sin(flow * 6.28) * 0.5 + 0.5;

                // Fresnel edge glow
                vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
                float fresnel = pow(1.0 - abs(dot(viewDir, vNormal)), 2.0);

                // Core beam
                float core = 1.0 - abs(vUv.x - 0.5) * 2.0;
                core = pow(core, 3.0);

                // Combine effects
                float beam = core * 0.6 + fresnel * 0.4;
                beam += pulse * 0.3;

                // Color with intensity
                vec3 color = uColor * beam * uIntensity;

                // Alpha fades at ends
                float endFade = smoothstep(0.0, 0.1, vUv.y) * smoothstep(1.0, 0.9, vUv.y);
                float alpha = beam * endFade * uIntensity;

                gl_FragColor = vec4(color, alpha);
            }
        `;

        const uniforms = {
            uTime: { value: 0 },
            uIntensity: { value: 0 },
            uColor: { value: new THREE.Color(0x00ffff) },
            uCameraPosition: { value: new THREE.Vector3() }
        };

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
        this.material.uniforms.uTime.value = time;
        this.material.uniforms.uCameraPosition.value.copy(cameraPos);

        // Smooth intensity transition
        this.intensity += (this.targetIntensity - this.intensity) * deltaTime * 5;
        this.material.uniforms.uIntensity.value = this.intensity;

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
        this.material.uniforms.uColor.value.copy(colorObj);
    }

    /**
     * Check if beam is currently visible
     */
    public isVisible(): boolean {
        return this.intensity > 0.1;
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}
