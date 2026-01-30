import * as THREE from 'three';
import { noteVisualizerVertexShader, noteVisualizerFragmentShader } from '../shaders';
import { NoteVisualizerMaterialTSL } from '../shaders/tsl';

/**
 * Detect if WebGPU/TSL is available
 */
function isWebGPURenderer(): boolean {
    return (window as any).rendererType === 'WebGPU';
}

/**
 * NoteVisualizer - Visual representation of notes being played
 *
 * Shows vertical bars above each harp string when notes are demonstrated
 * or played. Height represents frequency, color represents note.
 *
 * Auto-selects TSL (WebGPU) or GLSL (WebGL2) implementation.
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
 *
 * Auto-selects TSL (WebGPU) or GLSL (WebGL2) implementation.
 */
class VisualizerBar extends THREE.Mesh {
    private material: THREE.ShaderMaterial | NoteVisualizerMaterialTSL;
    private isTSL: boolean;
    private baseHeight: number = 1.0;
    private currentHeight: number = 0;
    private state: 'idle' | 'rising' | 'holding' | 'falling' = 'idle';
    private animTime: number = 0;
    private holdDuration: number = 0.5;
    private color: THREE.Color;

    // Uniforms object for API compatibility (GLSL mode only)
    public uniforms: {
        uTime: { value: number };
        uColor: { value: THREE.Color };
        uIntensity: { value: number };
        uCameraPosition: { value: THREE.Vector3 };
    };

    constructor(color: number) {
        // Box geometry for bar
        const geometry = new THREE.BoxGeometry(0.15, 1, 0.15);
        geometry.translate(0, 0.5, 0); // Pivot at bottom

        const noteColor = new THREE.Color(color);
        const isTSL = isWebGPURenderer();

        // Initialize uniforms for GLSL mode
        const uniforms = {
            uTime: { value: 0 },
            uColor: { value: noteColor },
            uIntensity: { value: 0 },
            uCameraPosition: { value: new THREE.Vector3() }
        };

        let material: THREE.ShaderMaterial | NoteVisualizerMaterialTSL;

        if (isTSL) {
            material = new NoteVisualizerMaterialTSL();
            (material as NoteVisualizerMaterialTSL).setColor(noteColor);
            console.log('[NoteVisualizer] Using TSL (WebGPU) implementation');
        } else {
            material = new THREE.ShaderMaterial({
                vertexShader: noteVisualizerVertexShader,
                fragmentShader: noteVisualizerFragmentShader,
                uniforms: uniforms,
                transparent: true,
                depthWrite: false,
                blending: THREE.AdditiveBlending
            });
            console.log('[NoteVisualizer] Using GLSL (WebGL2) implementation');
        }

        // MUST call super() before accessing 'this'
        super(geometry, material);
        this.material = material;
        this.isTSL = isTSL;
        this.uniforms = uniforms;
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
        this.setTime(time);

        const riseSpeed = 4.0;
        const fallSpeed = 3.0;

        switch (this.state) {
            case 'rising':
                this.currentHeight = Math.min(
                    this.baseHeight,
                    this.currentHeight + deltaTime * riseSpeed * 2
                );
                this.setIntensity(Math.min(1, this.currentHeight / this.baseHeight));

                if (this.currentHeight >= this.baseHeight) {
                    this.state = 'holding';
                    this.animTime = 0;
                }
                break;

            case 'holding':
                this.animTime += deltaTime;
                const pulse = Math.sin(time * 8) * 0.1 + 0.9;
                this.setIntensity(pulse);

                if (this.animTime >= this.holdDuration) {
                    this.state = 'falling';
                    this.animTime = 0;
                }
                break;

            case 'falling':
                this.currentHeight = Math.max(0, this.currentHeight - deltaTime * fallSpeed);
                this.setIntensity(this.currentHeight / this.baseHeight);

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
        if (this.isTSL) {
            (this.material as NoteVisualizerMaterialTSL).setCameraPosition(position);
        } else {
            this.uniforms.uCameraPosition.value.copy(position);
        }
    }

    private setTime(time: number): void {
        if (this.isTSL) {
            (this.material as NoteVisualizerMaterialTSL).setTime(time);
        } else {
            this.uniforms.uTime.value = time;
        }
    }

    private setIntensity(value: number): void {
        if (this.isTSL) {
            (this.material as NoteVisualizerMaterialTSL).setIntensity(value);
        } else {
            this.uniforms.uIntensity.value = value;
        }
    }

    public destroy(): void {
        this.geometry.dispose();
        this.material.dispose();
    }
}
