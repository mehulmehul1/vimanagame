/**
 * ShellUIOverlay - HUD overlay for shell collection progress
 *
 * Displays 4 chamber slots with drawn nautilus icons.
 * Animates collection and persists visual state.
 */

import { ShellManager, ShellCollectionState, CHAMBER_NAMES } from '../entities/ShellManager';

export interface UISlot {
    chamber: keyof ShellCollectionState;
    element: HTMLElement;
    canvas: HTMLCanvasElement;
    collected: boolean;
}

export interface UIOverlayConfig {
    position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
    scale: number;
    opacity: number;
    showLabels: boolean;
}

const DEFAULT_CONFIG: UIOverlayConfig = {
    position: 'bottom-left',
    scale: 0.5,
    opacity: 0.9,
    showLabels: false  // Hide labels for cleaner look
};

export class ShellUIOverlay {
    private container: HTMLElement | null = null;
    private slots: UISlot[] = [];
    private shellManager: ShellManager;
    private config: UIOverlayConfig;

    constructor(config: Partial<UIOverlayConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.shellManager = ShellManager.getInstance();

        // Create on initialization
        this.createOverlay();

        // Listen for collection events
        window.addEventListener('shell-collected', this.handleShellCollected);

        // Sync initial state
        this.syncState();
    }

    /**
     * Create the overlay DOM structure
     */
    private createOverlay(): void {
        this.container = document.createElement('div');
        this.container.id = 'shell-ui-overlay';
        this.container.className = `shell-ui-overlay position-${this.config.position}`;

        // Create 4 slots for each chamber
        const chambers: (keyof ShellCollectionState)[] = [
            'archiveOfVoices',
            'galleryOfForms',
            'hydroponicMemory',
            'engineOfGrowth'
        ];

        chambers.forEach((chamber, index) => {
            const slot = this.createSlot(chamber, index);
            this.slots.push(slot);
            this.container?.appendChild(slot.element);
        });

        document.body.appendChild(this.container);
    }

    /**
     * Create a single shell slot
     */
    private createSlot(chamber: keyof ShellCollectionState, index: number): UISlot {
        const element = document.createElement('div');
        element.className = 'shell-slot';
        element.dataset.chamber = chamber;
        element.style.animationDelay = `${index * 100}ms`;

        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        canvas.className = 'shell-icon';

        const label = document.createElement('span');
        label.className = 'shell-label';
        label.textContent = CHAMBER_NAMES[chamber];

        if (!this.config.showLabels) {
            label.style.display = 'none';
        }

        element.appendChild(canvas);
        element.appendChild(label);

        // Draw initial (empty) state
        this.drawEmptyIcon(canvas);

        return {
            chamber,
            element,
            canvas,
            collected: false
        };
    }

    /**
     * Draw empty slot icon (outline only)
     */
    private drawEmptyIcon(canvas: HTMLCanvasElement): void {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;
        const cx = w / 2;
        const cy = h / 2;

        ctx.clearRect(0, 0, w, h);

        // Draw dashed outline
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);

        ctx.beginPath();
        this.drawNautilusPath(ctx, cx, cy, 24, 0);
        ctx.stroke();

        ctx.setLineDash([]);
    }

    /**
     * Draw collected shell icon (filled with glow)
     */
    private drawCollectedIcon(canvas: HTMLCanvasElement, animated: boolean = false): void {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const w = canvas.width;
        const h = canvas.height;
        const cx = w / 2;
        const cy = h / 2;

        ctx.clearRect(0, 0, w, h);

        // Outer glow
        const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, 30);
        gradient.addColorStop(0, 'rgba(255, 200, 100, 0.4)');
        gradient.addColorStop(0.7, 'rgba(255, 150, 50, 0.1)');
        gradient.addColorStop(1, 'rgba(255, 150, 50, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, w, h);

        // Shell body with gradient
        const shellGradient = ctx.createRadialGradient(cx - 5, cy - 5, 0, cx, cy, 20);
        shellGradient.addColorStop(0, '#ffd700');
        shellGradient.addColorStop(0.5, '#ffaa00');
        shellGradient.addColorStop(1, '#ff6600');

        ctx.fillStyle = shellGradient;
        ctx.beginPath();
        this.drawNautilusPath(ctx, cx, cy, 20, 0);
        ctx.fill();

        // Inner spiral detail
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        this.drawNautilusPath(ctx, cx, cy, 12, 1);
        ctx.stroke();

        // Highlight
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.beginPath();
        ctx.arc(cx - 6, cy - 6, 4, 0, Math.PI * 2);
        ctx.fill();
    }

    /**
     * Draw nautilus spiral path
     * Uses golden ratio approximation for organic shell shape
     */
    private drawNautilusPath(
        ctx: CanvasRenderingContext2D,
        cx: number,
        cy: number,
        radius: number,
        offset: number
    ): void {
        const goldenRatio = 1.618033988749;
        const turns = 2.5;
        const segments = 30;

        ctx.moveTo(
            cx + Math.cos(0) * radius * 0.2,
            cy + Math.sin(0) * radius * 0.2
        );

        for (let i = 0; i <= segments; i++) {
            const t = i / segments;
            const angle = t * Math.PI * 2 * turns;
            const r = radius * (0.2 + 0.8 * Math.pow(t, 0.7));

            const x = cx + Math.cos(angle) * r;
            const y = cy + Math.sin(angle) * r;

            ctx.lineTo(x, y);
        }

        ctx.closePath();
    }

    /**
     * Handle shell collected event
     */
    private handleShellCollected = (e: Event): void => {
        const customEvent = e as CustomEvent<{ chamber: string }>;
        const chamberSlug = customEvent.detail.chamber;

        // Find matching slot
        const slot = this.slots.find(s => {
            const slugMap: Record<string, keyof ShellCollectionState> = {
                'archive-of-voices': 'archiveOfVoices',
                'gallery-of-forms': 'galleryOfForms',
                'hydroponic-memory': 'hydroponicMemory',
                'engine-of-growth': 'engineOfGrowth'
            };
            return s.chamber === slugMap[chamberSlug];
        });

        if (slot && !slot.collected) {
            slot.collected = true;
            this.animateCollection(slot);
        }
    };

    /**
     * Animate slot collection
     */
    private animateCollection(slot: UISlot): void {
        slot.element.classList.add('collected');
        slot.element.classList.add('pulse');

        // Animate drawing
        let frame = 0;
        const animate = () => {
            frame++;
            const progress = Math.min(frame / 20, 1);

            this.drawCollectedIcon(slot.canvas, progress < 1);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        animate();

        // Remove pulse after animation
        setTimeout(() => {
            slot.element.classList.remove('pulse');
        }, 1000);
    }

    /**
     * Sync UI with current shell state
     */
    private syncState(): void {
        const state = this.shellManager.getState();

        this.slots.forEach(slot => {
            if (state[slot.chamber] && !slot.collected) {
                slot.collected = true;
                slot.element.classList.add('collected');
                this.drawCollectedIcon(slot.canvas);
            }
        });
    }

    /**
     * Get screen position for shell fly-to animation
     */
    public getEmptySlotPosition(): { x: number; y: number } | null {
        const emptySlot = this.slots.find(s => !s.collected);
        if (!emptySlot) return null;

        const rect = emptySlot.element.getBoundingClientRect();
        return {
            x: rect.left + rect.width / 2,
            y: rect.top + rect.height / 2
        };
    }

    /**
     * Update overlay visibility
     */
    public setVisible(visible: boolean): void {
        if (this.container) {
            this.container.style.display = visible ? 'flex' : 'none';
        }
    }

    /**
     * Update overlay opacity
     */
    public setOpacity(opacity: number): void {
        this.config.opacity = Math.max(0, Math.min(1, opacity));
        if (this.container) {
            this.container.style.opacity = `${this.config.opacity}`;
        }
    }

    /**
     * Get number of collected shells
     */
    public getCollectedCount(): number {
        return this.slots.filter(s => s.collected).length;
    }

    /**
     * Check if all shells collected
     */
    public isComplete(): boolean {
        return this.slots.every(s => s.collected);
    }

    /**
     * Cleanup
     */
    public destroy(): void {
        window.removeEventListener('shell-collected', this.handleShellCollected);

        if (this.container && this.container.parentNode) {
            this.container.parentNode.removeChild(this.container);
        }

        this.slots = [];
        this.container = null;
    }
}
