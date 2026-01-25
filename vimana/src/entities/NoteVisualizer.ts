import * as THREE from 'three';

/**
 * NoteVisualizer - Visual representation of notes being played
 *
 * Shows vertical bars above each harp string when notes are demonstrated
 * or played. Height represents frequency, color represents note.
 */
export class NoteVisualizer extends THREE.Group {
    private bars: VisualizerBar[] = [];
    private stringPositions: THREE.Vector3[] = [];

    // Note colors matching rainbow spectrum
    private readonly noteColors = [
        0xff4444, // C - Red
        0xff8844, // D - Orange
        0xffcc44, // E - Yellow
        0x44ff44, // F - Green
        0x4488ff, // G - Blue
        0xcc44ff  // A - Purple
    ];

    // Note frequencies (relative height)
    private readonly noteHeights = [1.0, 1.12, 1.26, 1.33, 1.5, 1.68];

    constructor() {
        super();
        this.createBars();
    }

    private createBars(): void {
        for (let i = 0; i < 6; i++) {
            const bar = new VisualizerBar(this.noteColors[i]);
            this.bars.push(bar);
            this.add(bar);
        }
    }

    /**
     * Set string positions for bar placement
     */
    public setStringPositions(positions: THREE.Vector3[]): void {
        this.stringPositions = positions.map(p => p.clone());
        this.updateBarPositions();
    }

    private updateBarPositions(): void {
        for (let i = 0; i < this.bars.length && i < this.stringPositions.length; i++) {
            const pos = this.stringPositions[i];
            // Position bar above string
            this.bars[i].updatePosition(
                new THREE.Vector3(pos.x, pos.y + 1.5, pos.z)
            );
        }
    }

    /**
     * Show a note visualization
     */
    public showNote(noteIndex: number, duration: number = 1.0): void {
        if (noteIndex >= 0 && noteIndex < this.bars.length) {
            const height = this.noteHeights[noteIndex];
            this.bars[noteIndex].activate(height, duration);
        }
    }

    /**
     * Show multiple notes (for chord)
     */
    public showChord(noteIndices: number[], duration: number = 1.0): void {
        noteIndices.forEach(index => {
            if (index >= 0 && index < this.bars.length) {
                const height = this.noteHeights[index];
                this.bars[index].activate(height, duration);
            }
        });
    }

    /**
     * Update all bars
     */
    public update(deltaTime: number, time: number): void {
        this.bars.forEach(bar => bar.update(deltaTime, time));
    }

    /**
     * Hide all bars immediately
     */
    public hideAll(): void {
        this.bars.forEach(bar => bar.deactivate());
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.bars.forEach(bar => bar.destroy());
        this.bars = [];
    }
}

/**
 * VisualizerBar - Individual note visualization bar
 */
class VisualizerBar extends THREE.Mesh {
    private material: THREE.ShaderMaterial;
    private baseHeight: number = 1.0;
    private currentHeight: number = 0;
    private state: 'idle' | 'rising' | 'holding' | 'falling' = 'idle';
    private animTime: number = 0;
    private holdDuration: number = 0.5;
    private color: THREE.Color;

    constructor(color: number) {
        // Box geometry for bar
        const geometry = new THREE.BoxGeometry(0.15, 1, 0.15);
        geometry.translate(0, 0.5, 0); // Pivot at bottom

        const noteColor = new THREE.Color(color);

        const uniforms = {
            uTime: { value: 0 },
            uColor: { value: noteColor },
            uIntensity: { value: 0 },
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

            varying vec3 vNormal;
            varying vec3 vWorldPosition;
            varying vec2 vUv;

            void main() {
                // Fresnel edge glow
                vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
                float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.0);

                // Vertical gradient (brighter at top)
                float verticalGrad = vUv.y;

                // Combine
                vec3 color = uColor;
                color += vec3(1.0) * fresnel * 0.5;
                color *= uIntensity;

                float alpha = (0.3 + verticalGrad * 0.5 + fresnel * 0.2) * uIntensity;

                gl_FragColor = vec4(color, alpha);
            }
        `;

        const material = new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        super(geometry, material);
        this.material = material;
        this.color = noteColor;

        this.scale.y = 0;
        this.visible = false;
    }

    public activate(height: number, duration: number): void {
        this.baseHeight = height;
        this.holdDuration = duration;
        this.state = 'rising';
        this.animTime = 0;
        this.visible = true;
    }

    public deactivate(): void {
        this.state = 'falling';
        this.animTime = 0;
    }

    public updatePosition(position: THREE.Vector3): void {
        this.position.copy(position);
    }

    public update(deltaTime: number, time: number): void {
        this.material.uniforms.uTime.value = time;

        const riseSpeed = 4.0;
        const fallSpeed = 3.0;

        switch (this.state) {
            case 'rising':
                this.currentHeight = Math.min(
                    this.baseHeight,
                    this.currentHeight + deltaTime * riseSpeed * 2
                );
                this.material.uniforms.uIntensity.value = Math.min(1, this.currentHeight / this.baseHeight);

                if (this.currentHeight >= this.baseHeight) {
                    this.state = 'holding';
                    this.animTime = 0;
                }
                break;

            case 'holding':
                this.animTime += deltaTime;
                // Subtle pulse while holding
                const pulse = Math.sin(time * 8) * 0.1 + 0.9;
                this.material.uniforms.uIntensity.value = pulse;

                if (this.animTime >= this.holdDuration) {
                    this.state = 'falling';
                    this.animTime = 0;
                }
                break;

            case 'falling':
                this.currentHeight = Math.max(0, this.currentHeight - deltaTime * fallSpeed);
                this.material.uniforms.uIntensity.value = this.currentHeight / this.baseHeight;

                if (this.currentHeight <= 0) {
                    this.state = 'idle';
                    this.visible = false;
                    this.scale.y = 0;
                }
                break;
        }

        this.scale.y = this.currentHeight;
    }

    public setCameraPosition(position: THREE.Vector3): void {
        this.material.uniforms.uCameraPosition.value.copy(position);
    }

    public destroy(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}
