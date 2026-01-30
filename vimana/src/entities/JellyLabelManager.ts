import * as THREE from 'three';

/**
 * SequenceLabel - Individual sequence indicator sprite
 */
interface SequenceLabel {
    sprite: THREE.Sprite;
    sequenceNumber: number;
    visible: boolean;
}

/**
 * JellyLabelManager - Manages visual indicators for jellyfish
 *
 * Shows sequence numbers (①, ②, ③) above jellies during phrase demonstration.
 * Helps the player remember the order of notes.
 */
export class JellyLabelManager extends THREE.Group {
    private labels: Map<number, SequenceLabel> = new Map();

    // Cache for number materials to avoid re-generating textures
    private materialCache: Map<number, THREE.SpriteMaterial> = new Map();

    constructor() {
        super();
    }

    /**
     * Show sequence number above jellyfish
     *
     * @param stringIndex - Which string/jelly (0-5)
     * @param sequenceNumber - Position in sequence (0, 1, 2...)
     * @param worldPosition - Where jellyfish is located
     */
    public showSequenceLabel(stringIndex: number, sequenceNumber: number, worldPosition: THREE.Vector3): void {
        let label = this.labels.get(stringIndex);

        if (!label) {
            label = this.createLabel(stringIndex);
            this.labels.set(stringIndex, label);
            this.add(label.sprite);
        }

        // Position above jellyfish
        const labelHeight = 2.0;  // ABOVE jellyfish
        label.sprite.position.set(
            worldPosition.x,
            worldPosition.y + labelHeight,
            worldPosition.z
        );

        // Set number material (uses cache)
        label.sprite.material = this.getOrCreateNumberMaterial(sequenceNumber + 1);
        (label.sprite.material as THREE.SpriteMaterial).opacity = 0.9;

        // Animate in
        label.sprite.scale.set(0, 0, 0);
        this.animateLabelIn(label);

        // Store sequence info
        label.sequenceNumber = sequenceNumber;
        label.visible = true;
        label.sprite.visible = true;
    }

    /**
     * Create a new label container
     */
    private createLabel(stringIndex: number): SequenceLabel {
        // Initial material (will be replaced in showSequenceLabel)
        const material = new THREE.SpriteMaterial({
            transparent: true,
            opacity: 0,
            depthTest: false,
            depthWrite: false
        });

        const sprite = new THREE.Sprite(material);
        sprite.renderOrder = 100; // Render on top

        return {
            sprite,
            sequenceNumber: -1,
            visible: false
        };
    }

    /**
     * Get or create material for a specific sequence number
     */
    private getOrCreateNumberMaterial(num: number): THREE.SpriteMaterial {
        if (this.materialCache.has(num)) {
            return this.materialCache.get(num)!;
        }

        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d')!;

        // Draw glowing circle background
        const centerX = 64;
        const centerY = 64;
        const radius = 50;

        // Outer glow
        const gradient = ctx.createRadialGradient(centerX, centerY, radius * 0.5, centerX, centerY, radius);
        gradient.addColorStop(0, 'rgba(0, 255, 255, 0.4)');
        gradient.addColorStop(0.8, 'rgba(0, 255, 255, 0.2)');
        gradient.addColorStop(1, 'rgba(0, 255, 255, 0.0)');

        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Outer ring
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 4;
        ctx.stroke();

        // Draw number text (circled unicode looks better but harder to control font-wise on all systems)
        // We'll draw a nice number inside the circle
        ctx.font = 'bold 72px Arial, sans-serif';
        ctx.fillStyle = '#ffffff'; // White for contrast
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Add shadow for bioluminescent look
        ctx.shadowColor = '#00ffff';
        ctx.shadowBlur = 15;

        const circledNumber = this.toCircledNumber(num);
        ctx.fillText(circledNumber, centerX, centerY);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            opacity: 0.9,
            depthTest: false,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        this.materialCache.set(num, material);
        return material;
    }

    /**
     * Convert number to circled unicode character
     */
    private toCircledNumber(n: number): string {
        const circled = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩'];
        return circled[n - 1] || n.toString();
    }

    /**
     * Animate label appearance (scale up with bounce)
     */
    private animateLabelIn(label: SequenceLabel): void {
        const startTime = performance.now();
        const duration = 500; // ms

        const animate = (time: number) => {
            const elapsed = time - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Bounce out effect
            const scale = this.easeOutBack(progress);
            label.sprite.scale.set(scale, scale, 1);

            if (progress < 1 && label.visible) {
                requestAnimationFrame(animate);
            }
        };
        requestAnimationFrame(animate);
    }

    /**
     * Animate label disappearance
     */
    private animateLabelOut(label: SequenceLabel): void {
        const startTime = performance.now();
        const duration = 300; // ms
        const startScale = label.sprite.scale.x;

        const animate = (time: number) => {
            const elapsed = time - startTime;
            const progress = Math.min(elapsed / duration, 1);

            const scale = startScale * (1 - progress);
            label.sprite.scale.set(scale, scale, 1);

            const mat = label.sprite.material as THREE.SpriteMaterial;
            mat.opacity = 0.9 * (1 - progress);

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                label.visible = false;
                label.sprite.visible = false;
            }
        };
        requestAnimationFrame(animate);
    }

    /**
     * Ease out back (overshoot)
     */
    private easeOutBack(x: number): number {
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
    }

    /**
     * Hide specific label
     */
    public hideLabel(stringIndex: number): void {
        const label = this.labels.get(stringIndex);
        if (label && label.visible) {
            this.animateLabelOut(label);
        }
    }

    /**
     * Hide all labels
     */
    public hideAll(): void {
        this.labels.forEach((label) => {
            if (label.visible) {
                this.animateLabelOut(label);
            }
        });
    }

    /**
     * Update labels
     */
    public update(deltaTime: number, cameraPosition: THREE.Vector3): void {
        this.labels.forEach((label) => {
            if (!label.visible) return;
            // Sprites handle billboarding automatically in Three.js standard rendering,
            // but we can ensure they oriented correctly if needed.
        });
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        this.materialCache.forEach(mat => {
            mat.map?.dispose();
            mat.dispose();
        });
        this.materialCache.clear();
        this.labels.clear();
    }
}
