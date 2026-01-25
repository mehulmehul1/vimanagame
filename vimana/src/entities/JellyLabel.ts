import * as THREE from 'three';

/**
 * JellyLabelManager - Manages note labels above jelly creatures
 *
 * Shows note names (C, D, E, F, G, A) above jellies when they are teaching.
 * Uses billboarding sprites that always face the camera.
 */
export class JellyLabelManager extends THREE.Group {
    private labels: Map<number, JellyLabel> = new Map();
    private noteNames = ['C', 'D', 'E', 'F', 'G', 'A'];
    private noteColors = [
        0xff6b6b, // C - Red
        0xffa06b, // D - Orange
        0xffd93d, // E - Yellow
        0x6bcb77, // F - Green
        0x4d96ff, // G - Blue
        0x9b59b6  // A - Purple
    ];

    constructor() {
        super();
    }

    /**
     * Show label above a jelly
     */
    public showLabel(noteIndex: number, jellyPosition: THREE.Vector3): void {
        if (!this.labels.has(noteIndex)) {
            const label = new JellyLabel(
                this.noteNames[noteIndex],
                this.noteColors[noteIndex]
            );
            this.labels.set(noteIndex, label);
            this.add(label);
        }

        const label = this.labels.get(noteIndex)!;
        label.show();
        label.updatePosition(jellyPosition);
    }

    /**
     * Hide a specific label
     */
    public hideLabel(noteIndex: number): void {
        const label = this.labels.get(noteIndex);
        if (label) {
            label.hide();
        }
    }

    /**
     * Hide all labels
     */
    public hideAll(): void {
        this.labels.forEach(label => label.hide());
    }

    /**
     * Update label positions
     */
    public update(deltaTime: number, cameraPosition: THREE.Vector3): void {
        this.labels.forEach(label => {
            label.update(deltaTime);
            label.lookAt(cameraPosition);
        });
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.labels.forEach(label => label.destroy());
        this.labels.clear();
    }
}

/**
 * JellyLabel - Individual note label sprite
 */
class JellyLabel extends THREE.Sprite {
    private material: THREE.SpriteMaterial;
    private targetPosition: THREE.Vector3;
    private offsetPosition: THREE.Vector3;
    private state: 'showing' | 'visible' | 'hiding' | 'hidden' = 'hidden';
    private animTime: number = 0;
    private bobOffset: number = 0;

    constructor(noteName: string, color: number) {
        // Create label texture
        const texture = JellyLabel.createLabelTexture(noteName, color);

        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            opacity: 0,
            depthTest: true,
            depthWrite: false,
            blending: THREE.NormalBlending
        });

        super(material);
        this.material = material;

        this.targetPosition = new THREE.Vector3();
        this.offsetPosition = new THREE.Vector3(0, 1.5, 0); // Above jelly
        this.scale.set(0.8, 0.4, 1);
        this.visible = false;
    }

    /**
     * Create a label texture with note name
     */
    private static createLabelTexture(text: string, color: number): THREE.CanvasTexture {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 64;
        const ctx = canvas.getContext('2d')!;

        // Background glow (rounded rect)
        const gradient = ctx.createRadialGradient(64, 32, 0, 64, 32, 60);
        const colorHex = '#' + color.toString(16).padStart(6, '0');
        gradient.addColorStop(0, colorHex + 'cc');
        gradient.addColorStop(0.7, colorHex + '66');
        gradient.addColorStop(1, colorHex + '00');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 128, 64);

        // Text
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = colorHex;
        ctx.shadowBlur = 10;
        ctx.fillText(text, 64, 32);

        return new THREE.CanvasTexture(canvas);
    }

    public show(): void {
        if (this.state === 'hidden' || this.state === 'hiding') {
            this.state = 'showing';
            this.animTime = 0;
            this.visible = true;
        }
    }

    public hide(): void {
        if (this.state === 'visible' || this.state === 'showing') {
            this.state = 'hiding';
            this.animTime = 0;
        }
    }

    public updatePosition(jellyPosition: THREE.Vector3): void {
        this.targetPosition.copy(jellyPosition).add(this.offsetPosition);
        this.position.copy(this.targetPosition);
    }

    public update(deltaTime: number): void {
        this.animTime += deltaTime;

        // Bob animation
        this.bobOffset = Math.sin(this.animTime * 2) * 0.1;
        this.position.y = this.targetPosition.y + this.bobOffset;

        // Fade animations
        const fadeSpeed = 3.0;
        let targetOpacity = 0;

        switch (this.state) {
            case 'showing':
                targetOpacity = 1;
                this.material.opacity = Math.min(1, this.material.opacity + deltaTime * fadeSpeed);
                if (this.material.opacity >= 1) {
                    this.state = 'visible';
                }
                break;

            case 'visible':
                this.material.opacity = 1;
                break;

            case 'hiding':
                this.material.opacity = Math.max(0, this.material.opacity - deltaTime * fadeSpeed);
                if (this.material.opacity <= 0) {
                    this.state = 'hidden';
                    this.visible = false;
                }
                break;
        }
    }

    public isVisible(): boolean {
        return this.state === 'visible' || this.state === 'showing';
    }

    public destroy(): void {
        this.material.map?.dispose();
        this.material.dispose();
    }
}
