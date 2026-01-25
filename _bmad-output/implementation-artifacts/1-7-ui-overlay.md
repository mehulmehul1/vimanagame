# Story 1.7: UI Overlay System

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **player collecting shells from each chamber of the Vimana**,
I want **a subtle UI overlay showing my collected shells**,
so that **I can track my progress through the four chambers and feel the satisfaction of completing the collection**.

## Acceptance Criteria

1. [ ] ShellUIOverlay with 4 slots at top-left of screen
2. [ ] Canvas-drawn nautilus spiral icons for each slot
3. [ ] Slot fill animation with glow effect on collection
4. [ ] Gentle pulse animation on filled slots
5. [ ] Shell count tracking (0-4) across all chambers
6. [ ] Global window.shellUI registration for cross-scene access
7. [ ] Event listener for 'shell-collected' events
8. [ ] localStorage persistence for shell collection state

## Tasks / Subtasks

- [ ] Create ShellUIOverlay HTML/CSS structure (AC: #1)
  - [ ] Fixed position overlay at top-left (16px padding)
  - [ ] Z-index: 1000 (above all 3D content)
  - [ ] Flex container with 4 slot divs
  - [ ] Each slot: 48×48px, circular, semi-transparent background
  - [ ] Empty slot: 20% opacity white, thin border
  - [ ] Filled slot: 100% opacity, glow effect
  - [ ] Responsive scaling on smaller screens
  - [ ] CSS transitions for all state changes (300ms ease-out)
- [ ] Create Canvas-based shell icon rendering (AC: #2)
  - [ ] HTML5 Canvas for drawing nautilus spiral icons
  - [ ] **Spiral drawing function:**
    - Start from center, spiral outward
    - Logarithmic spiral: `r = a * e^(b*θ)`
    - 3-4 complete rotations for full shell
    - Line width: 2px
  - [ ] Fill with gradient (white → gold → pink)
  - [ ] Cache rendered icon as data URL for reuse
  - [ ] Draw on canvas context within each slot
  - [ ] 32×32px draw size within 48×48px slot
- [ ] Implement slot fill animation (AC: #3, #4)
  - [ ] On collection: Trigger animation sequence
  - [ ] **Phase 1 (0-0.3s):** Flash white (scale 1.0 → 1.3)
  - [ ] **Phase 2 (0.3-0.6s):** Settle to filled state (scale 1.3 → 1.0)
  - [ ] **Phase 3 (0.6s+):** Gentle pulse animation
  - [ ] Box-shadow glow: `0 0 10px rgba(255, 215, 0, 0.8)`
  - [ ] Glow color: Gold for most recent, white for older
  - [ ] CSS keyframe animation for continuous pulse
- [ ] Create pulse animation for filled slots (AC: #4)
  - [ ] Subtle scale pulse: 1.0 ↔ 1.05
  - [ ] Duration: 2 seconds per pulse cycle
  - [ ] Ease-in-out for smooth breathing
  - [ ] Opacity pulse: 1.0 ↔ 0.9
  - [ ] Glow intensity pulse: 10px ↔ 15px shadow blur
  - [ ] Apply to all filled slots, staggered by 0.2s each
- [ ] Implement shell count tracking (AC: #5)
  - [ ] Map of 4 chambers to slot indices:
    - 0: Archive of Voices (Culture)
    - 1: Gallery of Forms (Art)
    - 2: Hydroponic Memory (Nature)
    - 3: Engine of Growth (Technology)
  - [ ] collectedShells: Set<chamberId> for tracking
  - [ ] getCollectedCount(): number (0-4)
  - [ ] isCollected(chamberId): boolean
  - [ ] Update UI state when collection changes
  - [ ] Display count number below slots (optional)
- [ ] Create global shellUI registration (AC: #6)
  - [ ] Register on window object: `window.shellUI`
  - [ ] Expose methods: collectShell(chamberId), getProgress(), reset()
  - [ ] Available from any scene/script
  - [ ] Singleton pattern (only one instance exists)
  - [ ] Initialize on scene load
- [ ] Add event listener integration (AC: #7)
  - [ ] Listen for 'shell-collected' custom events
  - [ ] Event detail includes chamber identifier
  - [ ] On receive: Update slot, save state, trigger animation
  - [ ] Debounce rapid events (100ms minimum between)
  - [ ] Log collection for debugging
- [ ] Implement localStorage persistence (AC: #8)
  - [ ] Save key: 'vimana-shell-collection'
  - [ ] Data format: JSON object with chamber boolean flags
  - [ ] Save on every collection event
  - [ ] Load on UI initialization
  - [ ] Handle missing/invalid data gracefully
  - [ ] Provide reset method for testing
- [ ] Create ShellUIManager TypeScript class (AC: #1-8)
  - [ ] Encapsulate all UI logic
  - [ ] Constructor: Create DOM elements, bind events
  - [ ] Methods: initialize(), collect(), update(), destroy()
  - [ ] Private: renderSlots(), animateSlot(), saveState(), loadState()
  - [ ] Expose public API for global registration
- [ ] Add accessibility features
  - [ ] ARIA labels on slots ("Shell 1 of 4: Archive of Voices")
  - [ ] Role="group" on container
  - [ ] aria-live for collection announcements
  - [ ] Keyboard navigation support (optional)
  - [ ] High contrast mode support
- [ ] Performance and polish
  - [ ] Ensure animations maintain 60 FPS
  - [ ] Test on mobile devices (touch targets)
  - [ ] Verify z-index layering with other UI
  - [ ] CSS will-change for GPU acceleration

## Dev Notes

### Project Structure Notes

**Primary Framework:** TypeScript + HTML/CSS
**Rendering:** Canvas 2D API for icons
**Storage:** localStorage for persistence

**File Organization:**
```
vimana/
├── src/
│   ├── ui/
│   │   ├── ShellUIOverlay.ts
│   │   ├── ShellUISlots.ts
│   │   └── shell-ui.css
│   └── types/
│       └── shell-types.ts
```

### HTML Structure

**DOM Layout:**
```html
<div id="shell-ui-overlay" class="shell-ui-overlay" role="group" aria-label="Collected Shells">
    <div class="shell-slots-container">
        <div class="shell-slot" data-chamber="archive-of-voices" data-index="0" aria-label="Shell 1 of 4: Archive of Voices">
            <canvas class="shell-icon" width="32" height="32"></canvas>
        </div>
        <div class="shell-slot" data-chamber="gallery-of-forms" data-index="1" aria-label="Shell 2 of 4: Gallery of Forms">
            <canvas class="shell-icon" width="32" height="32"></canvas>
        </div>
        <div class="shell-slot" data-chamber="hydroponic-memory" data-index="2" aria-label="Shell 3 of 4: Hydroponic Memory">
            <canvas class="shell-icon" width="32" height="32"></canvas>
        </div>
        <div class="shell-slot" data-chamber="engine-of-growth" data-index="3" aria-label="Shell 4 of 4: Engine of Growth">
            <canvas class="shell-icon" width="32" height="32"></canvas>
        </div>
    </div>
    <div class="shell-count" aria-live="polite">
        <span id="shell-current">0</span> / <span id="shell-total">4</span>
    </div>
</div>
```

### CSS Styles

**Complete Stylesheet:**
```css
/* shell-ui.css */
.shell-ui-overlay {
    position: fixed;
    top: 16px;
    left: 16px;
    z-index: 1000;
    pointer-events: none; /* Let clicks pass through */
    font-family: 'Georgia', serif;
}

.shell-slots-container {
    display: flex;
    gap: 12px;
    margin-bottom: 8px;
}

.shell-slot {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 300ms ease-out;
    opacity: 0.2;
}

.shell-slot.filled {
    opacity: 1.0;
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 215, 0, 0.8);
    box-shadow: 0 0 10px rgba(255, 215, 0, 0.6);
    animation: shellPulse 2s ease-in-out infinite;
}

.shell-slot.filled.latest {
    border-color: rgba(255, 215, 0, 1);
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.9);
}

.shell-slot.collecting {
    animation: shellCollect 600ms ease-out forwards;
}

.shell-icon {
    width: 32px;
    height: 32px;
    opacity: 0;
    transition: opacity 300ms ease-out;
}

.shell-slot.filled .shell-icon {
    opacity: 1;
}

.shell-count {
    color: rgba(255, 255, 255, 0.7);
    font-size: 14px;
    text-align: center;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}

/* Animations */
@keyframes shellCollect {
    0% {
        transform: scale(1.0);
        opacity: 0.2;
    }
    30% {
        transform: scale(1.3);
        opacity: 1;
        background: rgba(255, 255, 255, 0.8);
    }
    60% {
        transform: scale(1.1);
        opacity: 1;
    }
    100% {
        transform: scale(1.0);
        opacity: 1;
    }
}

@keyframes shellPulse {
    0%, 100% {
        transform: scale(1.0);
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.6);
    }
    50% {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.8);
    }
}

/* Responsive */
@media (max-width: 768px) {
    .shell-ui-overlay {
        top: 8px;
        left: 8px;
    }
    .shell-slot {
        width: 40px;
        height: 40px;
    }
    .shell-icon {
        width: 24px;
        height: 24px;
    }
}

/* High contrast mode */
@media (prefers-contrast: high) {
    .shell-slot {
        border-width: 3px;
    }
    .shell-slot.filled {
        border-color: #ffff00;
        background: rgba(255, 255, 0, 0.3);
    }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    .shell-slot,
    .shell-icon {
        transition: none;
        animation: none;
    }
}
```

### Canvas Icon Rendering

**Nautilus Spiral Drawing:**
```typescript
class ShellIconRenderer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
    }

    draw(filled: boolean = false): void {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const cx = width / 2;
        const cy = height / 2;

        ctx.clearRect(0, 0, width, height);

        if (!filled) {
            this.drawEmpty(ctx, cx, cy);
        } else {
            this.drawFilled(ctx, cx, cy);
        }
    }

    private drawEmpty(ctx: CanvasRenderingContext2D, cx: number, cy: number): void {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, Math.PI * 2);
        ctx.stroke();
    }

    private drawFilled(ctx: CanvasRenderingContext2D, cx: number, cy: number): void {
        // Nautilus spiral parameters
        const rotations = 3.5;
        const points = 100;
        const a = 2;
        const b = 0.15;

        // Create gradient
        const gradient = ctx.createRadialGradient(cx, cy, 2, cx, cy, 14);
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(0.5, '#ffd700');
        gradient.addColorStop(1, '#ffc0cb');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';

        ctx.beginPath();
        for (let i = 0; i <= points; i++) {
            const angle = (i / points) * Math.PI * 2 * rotations;
            const radius = a * Math.exp(b * angle);

            // Scale to fit
            const maxRadius = 14;
            const scale = maxRadius / (a * Math.exp(b * Math.PI * 2 * rotations));

            const x = cx + Math.cos(angle) * radius * scale;
            const y = cy + Math.sin(angle) * radius * scale;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Add subtle glow
        ctx.shadowColor = '#ffd700';
        ctx.shadowBlur = 5;
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset
    }
}
```

### ShellUIOverlay Class

**Main Implementation:**
```typescript
interface ShellCollectionState {
    archiveOfVoices: boolean;
    galleryOfForms: boolean;
    hydroponicMemory: boolean;
    engineOfGrowth: boolean;
}

const CHAMBER_INDEX_MAP: Record<string, keyof ShellCollectionState> = {
    'archive-of-voices': 'archiveOfVoices',
    'gallery-of-forms': 'galleryOfForms',
    'hydroponic-memory': 'hydroponicMemory',
    'engine-of-growth': 'engineOfGrowth'
};

const INDEX_TO_CHAMBER = ['archive-of-voices', 'gallery-of-forms', 'hydroponic-memory', 'engine-of-growth'];

class ShellUIOverlay {
    private container: HTMLElement;
    private slots: HTMLElement[];
    private iconRenderers: ShellIconRenderer[];
    private state: ShellCollectionState;
    private STORAGE_KEY = 'vimana-shell-collection';

    constructor() {
        this.state = this.loadState();
        this.createDOM();
        this.bindEvents();
        this.updateUI();
        this.registerGlobal();
    }

    private createDOM(): void {
        // Create main container
        this.container = document.createElement('div');
        this.container.id = 'shell-ui-overlay';
        this.container.className = 'shell-ui-overlay';
        this.container.setAttribute('role', 'group');
        this.container.setAttribute('aria-label', 'Collected Shells');

        // Create slots container
        const slotsContainer = document.createElement('div');
        slotsContainer.className = 'shell-slots-container';
        this.container.appendChild(slotsContainer);

        this.slots = [];
        this.iconRenderers = [];

        // Create 4 slots
        for (let i = 0; i < 4; i++) {
            const slot = document.createElement('div');
            slot.className = 'shell-slot';
            slot.dataset.chamber = INDEX_TO_CHAMBER[i];
            slot.dataset.index = i.toString();
            slot.setAttribute('aria-label', `Shell ${i + 1} of 4: ${this.formatChamberName(INDEX_TO_CHAMBER[i])}`);

            const canvas = document.createElement('canvas');
            canvas.className = 'shell-icon';
            canvas.width = 32;
            canvas.height = 32;

            slot.appendChild(canvas);
            slotsContainer.appendChild(slot);

            this.slots.push(slot);
            this.iconRenderers.push(new ShellIconRenderer(canvas));
        }

        // Create count display
        const countDiv = document.createElement('div');
        countDiv.className = 'shell-count';
        countDiv.innerHTML = `
            <span id="shell-current">0</span> / <span id="shell-total">4</span>
        `;
        countDiv.setAttribute('aria-live', 'polite');
        this.container.appendChild(countDiv);

        // Append to body
        document.body.appendChild(this.container);
    }

    private bindEvents(): void {
        // Listen for shell collection events
        window.addEventListener('shell-collected', this.handleShellCollected.bind(this));
    }

    private handleShellCollected(event: CustomEvent): void {
        const chamberId = event.detail.chamber;
        this.collectShell(chamberId);
    }

    collectShell(chamberId: string): void {
        const stateKey = CHAMBER_INDEX_MAP[chamberId];
        if (!stateKey || this.state[stateKey]) return; // Already collected

        // Update state
        this.state[stateKey] = true;

        // Save to localStorage
        this.saveState();

        // Update UI
        this.updateUI();

        // Animate the specific slot
        const slotIndex = INDEX_TO_CHAMBER.indexOf(chamberId);
        this.animateSlot(slotIndex);

        // Announce for accessibility
        this.announceCollection(chamberId);
    }

    private updateUI(): void {
        const count = this.getCollectedCount();

        // Update count display
        const currentEl = document.getElementById('shell-current');
        if (currentEl) {
            currentEl.textContent = count.toString();
        }

        // Update each slot
        for (let i = 0; i < 4; i++) {
            const slot = this.slots[i];
            const chamberId = INDEX_TO_CHAMBER[i];
            const stateKey = CHAMBER_INDEX_MAP[chamberId];
            const isCollected = this.state[stateKey];

            // Update slot classes
            slot.classList.remove('filled', 'latest');
            if (isCollected) {
                slot.classList.add('filled');
            }

            // Redraw icon
            this.iconRenderers[i].draw(isCollected);
        }

        // Mark latest collected
        if (count > 0) {
            const latestIndex = this.getLatestCollectedIndex();
            this.slots[latestIndex].classList.add('latest');
        }
    }

    private animateSlot(index: number): void {
        const slot = this.slots[index];
        slot.classList.remove('collecting');
        void slot.offsetWidth; // Trigger reflow
        slot.classList.add('collecting');
    }

    private announceCollection(chamberId: string): void {
        const chamberName = this.formatChamberName(chamberId);
        const count = this.getCollectedCount();
        const message = `Shell collected from ${chamberName}. ${count} of 4 shells collected.`;

        // Announce to screen readers
        const countDiv = this.container.querySelector('.shell-count');
        if (countDiv) {
            countDiv.setAttribute('aria-label', message);
        }
    }

    private formatChamberName(chamberId: string): string {
        const names: Record<string, string> = {
            'archive-of-voices': 'Archive of Voices',
            'gallery-of-forms': 'Gallery of Forms',
            'hydroponic-memory': 'Hydroponic Memory',
            'engine-of-growth': 'Engine of Growth'
        };
        return names[chamberId] || chamberId;
    }

    getCollectedCount(): number {
        return Object.values(this.state).filter(Boolean).length;
    }

    isCollected(chamberId: string): boolean {
        const stateKey = CHAMBER_INDEX_MAP[chamberId];
        return stateKey ? this.state[stateKey] : false;
    }

    getProgress(): number {
        return this.getCollectedCount() / 4;
    }

    private getLatestCollectedIndex(): number {
        // Find the highest index that's collected
        for (let i = 3; i >= 0; i--) {
            if (this.state[CHAMBER_INDEX_MAP[INDEX_TO_CHAMBER[i]]]) {
                return i;
            }
        }
        return 0;
    }

    private saveState(): void {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.state));
        } catch (e) {
            console.warn('Failed to save shell collection state:', e);
        }
    }

    private loadState(): ShellCollectionState {
        try {
            const saved = localStorage.getItem(this.STORAGE_KEY);
            if (saved) {
                return JSON.parse(saved);
            }
        } catch (e) {
            console.warn('Failed to load shell collection state:', e);
        }

        // Default state
        return {
            archiveOfVoices: false,
            galleryOfForms: false,
            hydroponicMemory: false,
            engineOfGrowth: false
        };
    }

    reset(): void {
        this.state = {
            archiveOfVoices: false,
            galleryOfForms: false,
            hydroponicMemory: false,
            engineOfGrowth: false
        };
        this.saveState();
        this.updateUI();
    }

    private registerGlobal(): void {
        // Register as global singleton
        (window as any).shellUI = {
            collect: (chamberId: string) => this.collectShell(chamberId),
            isCollected: (chamberId: string) => this.isCollected(chamberId),
            getProgress: () => this.getProgress(),
            getCount: () => this.getCollectedCount(),
            reset: () => this.reset()
        };
    }

    destroy(): void {
        this.container.remove();
        window.removeEventListener('shell-collected', this.handleShellCollected.bind(this));
        delete (window as any).shellUI;
    }
}
```

### Initialization

**On Scene Load:**
```typescript
class HarpRoom {
    private shellUI: ShellUIOverlay;

    initialize() {
        this.shellUI = new ShellUIOverlay();
    }
}
```

### Dependencies

**Previous Story:** 1.6 Shell Collection (shell collect events trigger this)

**Next Story:** 1.8 White Flash Ending (UI shows progress before ending)

**External Dependencies:**
- DOM API
- Canvas 2D API
- localStorage API
- Custom event system

### Storage Schema

**localStorage Format:**
```json
{
    "archiveOfVoices": false,
    "galleryOfForms": false,
    "hydroponicMemory": false,
    "engineOfGrowth": false
}
```

### References

- [Source: music-room-proto-epic.md#Story 1.7]
- [Source: gdd.md#UI overlay requirements]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

### File List

- `src/ui/ShellUIOverlay.ts` (create)
- `src/ui/ShellUISlots.ts` (create)
- `src/ui/shell-ui.css` (create)
