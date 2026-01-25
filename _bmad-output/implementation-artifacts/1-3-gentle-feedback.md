# Story 1.3: Gentle Feedback System

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **player learning the harp in the Archive of Voices**,
I want **gentle, non-punitive feedback when I play the wrong note**,
so that **I feel guided rather than failed—the ship teaches me, it doesn't test me**.

## Acceptance Criteria

1. [ ] GentleFeedback class with camera shake (no jarring, just noticeable)
2. [ ] GentleAudioFeedback with discordant tone for wrong notes
3. [ ] Shake intensity: 0.5 units for wrong note, 0.3 units for premature play
4. [ ] Discordant tone uses minor second dissonance (playerFreq × 16/15 ratio)
5. [ ] Patient reminder tone plays after wrong note (gentle E4 at 329.63 Hz)
6. [ ] No state reset—player stays on same note, never loses progress
7. [ ] Visual cue on the correct string after wrong attempt (glow the correct string)

## Tasks / Subtasks

- [ ] Create GentleFeedback class with camera shake system (AC: #1, #3)
  - [ ] Initialize with camera reference and shake parameters
  - [ ] shake(intensity, duration) method
  - [ ] Intensity levels: 0.5 (wrong note), 0.3 (premature), 0.1 (subtle reminder)
  - [ ] Duration: 500ms for shake, with exponential decay
  - [ ] Perlin noise or sinusoidal shake pattern (not random)
  - [ ] Apply offset to camera.position, not rotation (maintains look direction)
  - [ ] Smooth return to original position after shake
- [ ] Implement shake decay algorithm (AC: #1, #3)
  - [ ] Decay function: `intensity *= exp(-deltaTime * decayRate)`
  - [ ] Decay rate: 8.0 (for ~500ms total duration)
  - [ ] Update loop runs each frame while shaking
  - [ ] Clamp maximum offset to prevent camera leaving scene bounds
  - [ ] X/Y/Z independent shake with different frequencies for organic feel
- [ ] Create GentleAudioFeedback with Web Audio API (AC: #2, #4, #5)
  - [ ] AudioContext singleton (resume on first user interaction)
  - [ ] Oscillator-based sound synthesis (no external files)
  - [ ] Discordant tone: `frequency = targetFreq * (16/15)` (minor second interval)
  - [ ] Duration: 300ms for discordant tone
  - [ ] Gain envelope: attack 50ms, decay 250ms (smooth fade)
  - [ ] Patient reminder: pure E4 sine wave (329.63 Hz), 200ms duration
  - [ ] Master gain for overall volume control
- [ ] Implement dissonance calculation (AC: #4)
  - [ ] Note frequency mapping:
    - C4: 261.63 Hz
    - D4: 293.66 Hz
    - E4: 329.63 Hz
    - F4: 349.23 Hz
    - G4: 392.00 Hz
    - A4: 440.00 Hz
  - [ ] Discordant frequency formula:
    ```javascript
    const dissonantFreq = correctFreq * (16 / 15); // Minor second (semitone) up
    // Alternative: minor second down
    const dissonantFreq = correctFreq * (15 / 16);
    ```
  - [ ] Use slight detune (±5 cents) for more organic dissonance
  - [ ] Add gentle vibrato (3-5 Hz) to reduce harshness
- [ ] Create reminder tone system (AC: #5)
  - [ ] Play gentle E4 tone after discordant sound
  - [ ] 100ms gap between discordant and reminder
  - [ ] Reminder indicates "try again, listen to the note"
  - [ ] Use soft sine wave (not square/sawtooth) for gentleness
  - [ ] Volume: -12 dB relative to main harp sounds
- [ ] Implement no-reset state management (AC: #6)
  - [ ] Player remains on current note after wrong attempt
  - [ ] No progress loss or score penalty
  - [ ] Same note can be attempted immediately
  - [ ] No "strike" counter or failure tracking
  - [ ] State machine: AWAITING_INPUT → WRONG_PLAYED → AWAITING_INPUT (not back to start)
- [ ] Add visual cue for correct string (AC: #7)
  - [ ] Glow effect on target harp string after wrong note
  - [ ] Emissive intensity: 0.2 (subtle, not overwhelming)
  - [ ] Pulse animation: 0.2 → 0.5 → 0.2 over 1 second
  - [ ] Color: Warm amber (#ffaa44) to contrast with cyan jellies
  - [ ] Lasts 2 seconds then fades
  - [ ] Uses existing HarpString material emissive property
- [ ] Create FeedbackManager to coordinate feedback systems (AC: #1, #2, #6, #7)
  - [ ] triggerWrongNote(targetNoteIndex) method
  - [ ] triggerPrematurePlay() method
  - [ ] triggerReminder(targetNoteIndex) method
  - [ ] Coordinates camera, audio, and visual feedback
  - [ ] Configurable intensity settings
  - [ ] Logging for debugging (can be disabled)
- [ ] Performance and polish
  - [ ] Verify audio plays smoothly without pops/clicks
  - [ ] Test shake doesn't cause motion sickness
  - [ ] Validate Web Audio API resumes correctly
  - [ ] Test rapid wrong-note handling (no audio overlap issues)

## Dev Notes

### Project Structure Notes

**Primary Framework:** Three.js r160+ (WebGPU/WebGL2)
**Audio:** Web Audio API (native browser API)
**Scene Format:** GLB with Gaussian Splatting via Shadow Engine

**File Organization:**
```
vimana/
├── src/
│   ├── audio/
│   │   ├── GentleAudioFeedback.ts
│   │   └── NoteFrequencies.ts (constants)
│   ├── entities/
│   │   ├── GentleFeedback.ts
│   │   └── FeedbackManager.ts
│   └── scenes/
│       └── HarpRoom.ts (main scene controller)
```

### Technical Requirements

**Camera Shake Specifications:**

| Property | Value | Notes |
|----------|-------|-------|
| Wrong note intensity | 0.5 units | Noticeable but not jarring |
| Premature intensity | 0.3 units | Subtle hint |
| Shake duration | 500ms | Exponential decay |
| Decay rate | 8.0 | `exp(-dt * 8)` |
| Shake pattern | Sinusoidal + Perlin | Organic, not random |
| Max offset | ±0.5 units | Clamped to prevent issues |

**Audio Specifications:**

| Sound Type | Frequency | Duration | Wave Type | Volume |
|------------|-----------|----------|-----------|--------|
| Discordant | correct × 16/15 | 300ms | Sine + slight vibrato | -6 dB |
| Reminder | 329.63 Hz (E4) | 200ms | Pure sine | -12 dB |
| Gap | - | 100ms | Silence | -∞ |

**Note Frequency Map:**
```javascript
export const NOTE_FREQUENCIES = {
    C: 261.63, // Middle C
    D: 293.66,
    E: 329.63,
    F: 349.23,
    G: 392.00,
    A: 440.00
};
```

### Camera Shake Implementation

**Shake Algorithm:**
```javascript
class GentleFeedback {
    private isShaking = false;
    private shakeIntensity = 0;
    private shakeOffset = new Vector3();
    private originalPosition = new Vector3();

    shake(intensity: number, duration: number = 0.5) {
        this.shakeIntensity = intensity;
        this.isShaking = true;
        this.originalPosition.copy(this.camera.position);

        // Schedule end of shake
        setTimeout(() => {
            this.isShaking = false;
        }, duration * 1000);
    }

    update(deltaTime: number) {
        if (!this.isShaking) return;

        // Decay intensity
        this.shakeIntensity *= Math.exp(-deltaTime * 8);

        if (this.shakeIntensity < 0.01) {
            this.shakeIntensity = 0;
            this.isShaking = false;
            this.camera.position.copy(this.originalPosition);
            return;
        }

        // Generate shake offset with different frequencies per axis
        const time = performance.now() / 1000;
        this.shakeOffset.set(
            Math.sin(time * 17) * this.shakeIntensity * 0.5,
            Math.sin(time * 23) * this.shakeIntensity * 0.3,
            Math.sin(time * 29) * this.shakeIntensity * 0.2
        );

        // Apply offset
        this.camera.position.copy(this.originalPosition).add(this.shakeOffset);
    }
}
```

### Audio Feedback Implementation

**Web Audio API Setup:**
```javascript
class GentleAudioFeedback {
    private audioContext: AudioContext;
    private masterGain: GainNode;

    constructor() {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        this.masterGain = this.audioContext.createGain();
        this.masterGain.gain.value = 0.3; // Master volume
        this.masterGain.connect(this.audioContext.destination);
    }

    async ensureResumed() {
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    playDiscordantTone(correctFrequency: number, duration: number = 0.3) {
        this.ensureResumed();

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        // Minor second dissonance (semitone up)
        const dissonantFreq = correctFrequency * (16 / 15);

        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(dissonantFreq, this.audioContext.currentTime);

        // Add slight vibrato for organic feel
        oscillator.detune.setValueAtTime(5, this.audioContext.currentTime); // +5 cents

        // Gain envelope
        const now = this.audioContext.currentTime;
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.3, now + 0.05); // Attack
        gainNode.gain.linearRampToValueAtTime(0, now + duration); // Decay

        oscillator.connect(gainNode);
        gainNode.connect(this.masterGain);

        oscillator.start(now);
        oscillator.stop(now + duration);
    }

    playReminderTone(correctFrequency: number, duration: number = 0.2) {
        this.ensureResumed();

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        // Gentle E4 for reminder (or target note)
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(329.63, this.audioContext.currentTime); // E4

        const now = this.audioContext.currentTime;
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.15, now + 0.03);
        gainNode.gain.linearRampToValueAtTime(0, now + duration);

        oscillator.connect(gainNode);
        gainNode.connect(this.masterGain);

        oscillator.start(now);
        oscillator.stop(now + duration);
    }
}
```

### Feedback Coordination

**FeedbackManager:**
```javascript
class FeedbackManager {
    private cameraFeedback: GentleFeedback;
    private audioFeedback: GentleAudioFeedback;

    triggerWrongNote(targetNoteIndex: number) {
        const correctFreq = NOTE_FREQUENCIES[String.fromCharCode(67 + targetNoteIndex)]; // C, D, E...

        // Camera shake
        this.cameraFeedback.shake(0.5, 0.5);

        // Discordant tone
        this.audioFeedback.playDiscordantTone(correctFreq, 0.3);

        // Reminder after gap
        setTimeout(() => {
            this.audioFeedback.playReminderTone(correctFreq, 0.2);
        }, 400); // 300ms tone + 100ms gap

        // Visual cue on correct string
        this.highlightTargetString(targetNoteIndex);
    }

    triggerPrematurePlay() {
        this.cameraFeedback.shake(0.3, 0.3);
        // No audio for premature, just subtle visual hint
    }

    private highlightTargetString(noteIndex: number) {
        // Trigger glow animation on target harp string
        // Uses existing HarpString emissive property
        // Pulse from 0.2 to 0.5 over 1 second, fade over 1 second
    }
}
```

### Memory Management

```javascript
class GentleFeedback {
    destroy() {
        this.isShaking = false;
        this.shakeIntensity = 0;
        // Camera is external reference, don't dispose
    }
}

class GentleAudioFeedback {
    destroy() {
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }
    }
}
```

### Dependencies

**Previous Story:** 1.2 Jelly Creatures (jellies demonstrate correct notes)

**Next Story:** 1.4 Duet Mechanics (feedback integrates with duet state machine)

**External Dependencies:**
- Three.js core: Camera access
- Web Audio API: Sound synthesis
- HarpString entities: For visual cue highlighting

### Philosophy Notes

**Key Design Principle:** "The ship doesn't test you. It teaches you."

This means:
- No "wrong answer" judgment—only guidance
- Shake should feel like a gentle nudge, not a punishment
- Discordant tone says "that's not quite it" not "you failed"
- Immediate reminder keeps player in flow state
- No reset means no loss of progress—always moving forward

**Intensity Calibration:**
- The shake should be subtle enough that a non-gamer doesn't feel discouraged
- The discordant tone should be noticeable but not harsh
- Combined feedback should say "try again" not "you failed"

### References

- [Source: music-room-proto-epic.md#Story 1.3]
- [Source: gdd.md#Contemplative Awakening pillar]
- [Source: narrative-design.md#Patient teaching philosophy]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

### File List

- `src/audio/GentleAudioFeedback.ts` (create)
- `src/audio/NoteFrequencies.ts` (create)
- `src/entities/GentleFeedback.ts` (create)
- `src/entities/FeedbackManager.ts` (create)
