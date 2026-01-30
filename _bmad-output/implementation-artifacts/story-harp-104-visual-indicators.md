# STORY-HARP-104: Visual Sequence Indicators

**Epic**: `HARP-ENHANCEMENT` - Harp Minigame Design Alignment
**Story ID**: `STORY-HARP-104`
**Points**: `2`
**Status**: `Ready for Dev`
**Owner**: `TBD`
**Related:** Sprint Change Proposal 2026-01-29
**Depends On:** STORY-HARP-101

---

## User Story

As a **player**, I want **visual indicators showing the order of notes in the phrase**, so that **I can understand and remember the sequence the Vimana is teaching me**.

---

## Overview

Add visual number indicators above each jellyfish during phrase demonstration. These indicators (1, 2, 3...) show the player the order of notes in the sequence, helping them remember which strings to play during their response.

**Reference:** `vimana_harp_minigame_design.md` - "Jellyfish position shows WHICH string to play. The ORDER they appear shows the SEQUENCE to remember."

---

## Background

**Design Intent:**
> "Each jellyfish corresponds to a harp string based on horizontal alignment"
> "The sequence of jellyfish jumps = the note sequence"
> "Jellyfish do NOT encode: Rhythm accuracy, Timing precision, Note duration"
> "They teach only one thing: The ORDER of notes the player must play"

**Current State:**
- Jellyfish emerge but no explicit order indicators
- Player must count mentally or rely on position alone
- Less clear for players with lower visual/spatial skills

**Desired State:**
- Number labels (①, ②, ③...) appear above each jellyfish
- Clear indication of sequence order
- Labels fade during player's turn
- Supports accessibility and clarity

---

## Technical Specification

### Enhanced JellyLabelManager

Modify `src/entities/JellyLabelManager.ts`:

```typescript
/**
 * Enhanced label manager for jellyfish with sequence indicators
 */
export class JellyLabelManager extends THREE.Group {
    private labels: Map<number, SequenceLabel> = new Map();
    private labelMaterial: THREE.Material;
    private font: THREE.Font;

    constructor() {
        super();

        // Create glowing label material
        this.labelMaterial = new THREE.SpriteMaterial({
            map: this.createNumberTexture(),
            color: 0x00ffff,  // Cyan bioluminescence
            transparent: true,
            opacity: 0.9,
            depthTest: false,  // Always visible
            depthWrite: false
        });
    }

    /**
     * Show sequence number above jellyfish
     *
     * @param stringIndex - Which string/jelly (0-5)
     * @param sequenceNumber - Position in sequence (0, 1, 2...)
     * @param worldPosition - Where jellyfish is located
     */
    showSequenceLabel(stringIndex: number, sequenceNumber: number, worldPosition: THREE.Vector3): void {
        let label = this.labels.get(stringIndex);

        if (!label) {
            label = this.createLabel(stringIndex);
            this.labels.set(stringIndex, label);
            this.add(label.sprite);
        }

        // Position above jellyfish
        const labelHeight = 2.0;  // Above jellyfish
        label.sprite.position.set(
            worldPosition.x,
            worldPosition.y + labelHeight,
            worldPosition.z
        );

        // Set number texture
        label.sprite.material = this.createNumberMaterial(sequenceNumber + 1);

        // Animate in
        label.sprite.scale.set(0, 0, 0);
        this.animateLabelIn(label);

        // Store sequence info
        label.sequenceNumber = sequenceNumber;
        label.visible = true;
    }

    /**
     * Create number texture for a digit
     */
    private createNumberMaterial(number: number): THREE.SpriteMaterial {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');

        // Draw circle background (glowing)
        ctx.beginPath();
        ctx.arc(64, 64, 50, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 255, 255, 0.3)';
        ctx.fill();

        // Draw outer ring
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)';
        ctx.lineWidth = 4;
        ctx.stroke();

        // Draw number
        ctx.font = 'bold 64px Arial';
        ctx.fillStyle = '#00ffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.toCircledNumber(number), 64, 64);

        const texture = new THREE.CanvasTexture(canvas);
        return new THREE.SpriteMaterial({
            map: texture,
            color: 0xffffff,
            transparent: true,
            opacity: 0.9,
            depthTest: false
        });
    }

    /**
     * Convert number to circled unicode character
     * ① → ①, 2 → ②, etc.
     */
    private toCircledNumber(n: number): string {
        const circled = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩'];
        return circled[n - 1] || n.toString();
    }

    /**
     * Animate label appearance (scale up with bounce)
     */
    private animateLabelIn(label: SequenceLabel): void {
        const startTime = Date.now();
        const duration = 300;

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out bounce
            const scale = this.easeOutBounce(progress);
            label.sprite.scale.setScalar(scale);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        animate();
    }

    /**
     * Hide specific label
     */
    hideLabel(stringIndex: number): void {
        const label = this.labels.get(stringIndex);
        if (label) {
            this.animateLabelOut(label);
        }
    }

    /**
     * Hide all labels (e.g., after splash)
     */
    hideAll(): void {
        for (const label of this.labels.values()) {
            if (label.visible) {
                this.animateLabelOut(label);
            }
        }
    }

    /**
     * Animate label disappearance
     */
    private animateLabelOut(label: SequenceLabel): void {
        const startTime = Date.now();
        const duration = 200;
        const startOpacity = (label.sprite.material as THREE.SpriteMaterial).opacity;

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            label.sprite.scale.setScalar(1 - progress);
            (label.sprite.material as THREE.SpriteMaterial).opacity = startOpacity * (1 - progress);

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                label.visible = false;
            }
        };
        animate();
    }

    /**
     * Easing function for bounce effect
     */
    private easeOutBounce(t: number): number {
        if (t < 1/2.75) {
            return 7.5625 * t * t;
        } else if (t < 2/2.75) {
            return 7.5625 * (t -= 1.5/2.75) * t + 0.75;
        } else if (t < 2.5/2.75) {
            return 7.5625 * (t -= 2.25/2.75) * t + 0.9375;
        } else {
            return 7.5625 * (t -= 2.625/2.75) * t + 0.984375;
        }
    }

    /**
     * Update labels (billboard to camera)
     */
    update(deltaTime: number, cameraPosition: THREE.Vector3): void {
        for (const label of this.labels.values()) {
            if (!label.visible) continue;

            // Make labels always face camera
            label.sprite.lookAt(cameraPosition);
        }
    }
}

interface SequenceLabel {
    sprite: THREE.Sprite;
    sequenceNumber: number;
    visible: boolean;
}
```

---

## Implementation Tasks

1. **[CLASS]** Enhance JellyLabelManager with sequence label support
2. **[METHOD]** Add `showSequenceLabel(stringIndex, sequenceNumber, position)` method
3. **[VISUAL]** Create circled number textures (①, ②, ③...)
4. **[ANIMATION]** Add bounce-in animation for labels
5. **[ANIMATION]** Add fade-out animation for labels
6. **[METHOD]** Add `hideAll()` method for synchronized hiding
7. **[INTEGRATION]** Connect to PatientJellyManager callbacks
8. **[BILLBOARD]** Ensure labels always face camera

---

## File Structure

```
src/entities/
├── JellyLabelManager.ts (enhanced)
└── SequenceLabel interface (new)

src/scenes/
└── HarpRoom.ts (integration updates)
```

---

## Integration Points

**PatientJellyManager.ts:**
```typescript
// During phrase demonstration
private demonstrateCurrentNote(): void {
    const targetNote = TEACHING_SEQUENCES[this.currentSequence][this.currentNoteIndex];

    // Spawn jelly
    this.jellyManager.spawnJelly(targetNote);

    // Show sequence indicator
    if (this.callbacks.onNoteDemonstrated) {
        this.callbacks.onNoteDemonstrated(targetNote, this.currentNoteIndex);
    }
}
```

**HarpRoom.ts:**
```typescript
private setupManagers(): void {
    this.jellyLabels = new JellyLabelManager();
    this.scene.add(this.jellyLabels);

    this.patientJellyManager.setCallbacks({
        // ... existing callbacks ...

        onNoteDemonstrated: (noteIndex: number, sequenceIndex: number) => {
            // Show sequence number above jellyfish
            const jelly = this.jellyManager.getJelly(noteIndex);
            if (jelly) {
                this.jellyLabels.showSequenceLabel(
                    noteIndex,
                    sequenceIndex,  // 0, 1, 2...
                    jelly.position
                );
            }
        },

        onDemonstrationEnd: () => {
            // Hide all labels when player's turn begins
            this.jellyLabels.hideAll();
        }
    });
}

public render(): void {
    // ... existing code ...

    // Update jelly labels (billboard to camera)
    if (this.jellyLabels) {
        this.jellyLabels.update(deltaTime, this.camera.position);
    }
}
```

---

## Visual Specification

| Element | Specification |
|---------|---------------|
| **Numbers** | Circled unicode: ①, ②, ③, ④, ⑤, ⑥ |
| **Color** | Cyan (#00ffff) bioluminescent |
| **Background** | Semi-transparent cyan circle |
| **Border** | Cyan ring, 4px stroke |
| **Font** | Bold, 64px, centered |
| **Size** | 128x128 texture, ~1 unit world scale |
| **Position** | 2 units above jellyfish |
| **Animation In** | Bounce scale, 300ms |
| **Animation Out** | Fade + shrink, 200ms |
| **Behavior** | Always faces camera (billboard) |

---

## Acceptance Criteria

- [ ] Circled numbers (①, ②, ③...) appear above jellies
- [ ] Numbers correspond to sequence order (1st, 2nd, 3rd)
- [ ] Labels bounce in when jellyfish emerges
- [ ] Labels fade out when player's turn begins
- [ ] Labels always face camera (billboarding)
- [ ] Labels are clearly visible against all backgrounds
- [ ] Multiple labels can be visible simultaneously
- [ ] `hideAll()` clears all labels cleanly

---

## Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| STORY-HARP-101 | Phrase-First Mode | Required |
| JellyManager.ts | Existing | ✅ Ready |
| JellyLabelManager.ts | Existing | ✅ Ready (enhancement) |
| THREE.js | External | ✅ Available |

---

## Testing

**Manual Test Cases:**
1. Start phrase-first sequence → labels ①, ②, ③ appear
2. Check label positions → above correct jellyfish
3. Rotate camera → labels always face player
4. Splash occurs → all labels fade out
5. Wrong note → redemonstration → labels reappear

**Visual Verification:**
- [ ] Numbers are legible
- [ ] Contrast is sufficient
- [ ] Animation feels organic (bounce, not stiff)
- [ ] Fade out is smooth
- [ ] No z-fighting with other elements

---

## Accessibility

| Feature | Benefit |
|---------|---------|
| **Number labels** | Helps players with working memory challenges |
| **Circled numbers** | Clear unicode symbols, easier to recognize |
| **Color choice** | Cyan provides good contrast |
| **Billboarding** | Always readable from any angle |
| **Fade duration** | 200ms is not too jarring |

---

## Performance

| Aspect | Impact |
|--------|--------|
| **Texture generation** | 6 textures max, cached after creation |
| **Sprite count** | 3-6 sprites during demo |
| **Draw calls** | 1 per sprite, negligible |
| **Update cost** | Billboard calculation per frame |
| **Memory** | <1MB for all textures |

---

## Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `labelHeight` | 2.0 | 1.5-3.0 | Units above jellyfish |
| `labelScale` | 1.0 | 0.5-2.0 | Size multiplier |
| `animationInMs` | 300 | 100-500 | Bounce-in duration |
| `animationOutMs` | 200 | 100-500 | Fade-out duration |
| `labelColor` | 0x00ffff | hex | Bioluminescent cyan |

---

## Notes

**Design Philosophy:**
- Labels should feel organic, not UI-like
- Circled numbers match the bioluminescent aesthetic
- Animations should be gentle (bounce, not pop)
- Labels are TEACHING AIDS, not gameplay elements

**Future Enhancements:**
- Color coding by sequence (different colors for each phrase)
- Glowing effect on currently-expected note during player's turn
- Smaller indicators during player's turn to remind sequence

---

**Sources:**
- `vimana_harp_minigame_design.md` (Jellyfish teaching section)
- Sprint Change Proposal 2026-01-29
- Existing `JellyLabelManager.ts` implementation
